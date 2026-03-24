"""
bgm_xbox.py — FlatOut 2 OG Xbox BGM import support.
Part of FlatOut Blender Tools — https://github.com/RavenDS/flatout-blender-tools

Covers:
  • _detect_xbox_bgm
  • XboxSurface, XboxBGMParser
  • _xbox_parse_ib_blob
  • _xbox_decode_main_vert, _xbox_decode_shadow_vert
  • extract_crash_vertices_xbox
  • build_blender_meshes_xbox
"""
import bpy
import struct
import os
from mathutils import Matrix, Vector
from dataclasses import dataclass, field
from .bgm_common import (
    BinaryReader, BGMMaterial, Model, BGMMesh, BGMObject,
    CrashSurface, ParsedVertex,
    parse_crash_dat, fo2_matrix_to_blender, create_blender_material,
)

# ─────────────────────────── XBOX BGM SUPPORT ───────────────────────────────
#
# FlatOut 2 OG Xbox BGM files share version 0x20000 with PC but replace the
# two-stream (VB + IB) layout with an NV2A push-buffer geometry store.
#
# Detection: version == 0x20000 AND at least one stream has dt=4 or dt=5.
#   dt=4 = small/medium IB (stream2) — NV2A push-buffer
#   dt=5 = large IB (stream3)        — NV2A push-buffer, surfaces with niu≥1200
#   dt=1 vs=12 = shadow VB (float xyz, body type only)
#   dt=1 vs=16 = main VB  (NORMPACKED3 int16 format)
#
# File layout:
#   version → materials → streams → seek_table → mystery_block
#   → surface_table → models → meshes → objects
#
# Surface record: 52 bytes = 7×sint32 + 6×uint32
#   [0]  is_vegetation   [1] material_id  [2] vertex_count
#   [3]  flags           [4] poly_count   [5] poly_mode    [6] num_indices
#   [7]  e0  VB base vertex (absolute index into shadow or main VB)
#   [8]  e1  = 2 (always)
#   [9]  e2  = 1 for body-type non-shadow, 0 for shadow or menucar-type
#   [10] e3  = 0
#   [11] e4  = 0 (stream2/small IB), 1 (stream3/large IB)
#   [12] e5  position of this surface in its IB stream's surface list
#
# Vertex format — main VB (vs=16):
#   int16[3]  pos    × 1/1024
#   uint16    pad
#   uint32    NORMPACKED3:  bits[10:0]=nx×1023, bits[21:11]=ny×1023, bits[31:22]=nz×511
#             (all fields 2's-complement sign-extended from their bit widths)
#   int16[2]  uv     × 1/2048
#
# Vertex format — shadow VB (vs=12):
#   float[3]  pos    (raw, no scaling)
#
# IB push-buffer per surface (NV2A):
#   0x000417FC  N      BEGIN (N=5=list, N=6=strip; N=0=null/end header)
#   0x40000000 | (cnt<<18) | 0x1800   DRAW_INLINE_ARRAY (cnt dwords of index pairs)
#   [cnt×4 bytes: pairs of uint16 absolute VB indices]
#   0x00041808  + uint32(odd_idx, 0)  one extra odd index (if niu is odd)
#
# Seek table (after streams):
#   uint32 count, uint32 unk=0, count×(uint32 size, uint32 off)
#   The LAST entry's off field = number of stream3 surfaces (n_s3).
#
# Mystery block size:
#   n_s3 == 0 → 16 bytes;  n_s3 > 0 → 8×n_s3 bytes
#
# Winding: same Y↔Z reflection as PC/PS2/PSP → same reversal:
#   mode 4 triple (i0,i1,i2) → emit (i2,i1,i0)
#   mode 5 strip  → alternating flip (same as PC extract_indices)


@dataclass
class XboxSurface:
    is_vegetation:    int = 0
    material_id:      int = 0
    vertex_count:     int = 0
    flags:            int = 0
    poly_count:       int = 0
    poly_mode:        int = 0
    num_indices_used: int = 0
    e0: int = 0   # VB base vertex (absolute)
    e1: int = 2
    e2: int = 0   # 1=main body-type surface, 0=shadow or menucar
    e3: int = 0
    e4: int = 0   # 0=stream2, 1=stream3
    e5: int = 0   # position in its IB stream's surface list


def _detect_xbox_bgm(filepath: str) -> bool:
    """Return True when the file has at least one dt=4 or dt=5 stream (Xbox NV2A IB)."""
    try:
        with open(filepath, 'rb') as f:
            raw = f.read()
        r = BinaryReader(raw)
        version = r.u32()
        if version != 0x20000:
            return False

        # skip materials (identical layout to PC BGM at version 0x20000)
        nm = r.u32()
        for _ in range(nm):
            r.u32()          # ident MATC
            r.read_string()  # name
            r.i32()          # nAlpha
            # version 0x20000 always has the extended material fields
            r.read(20 + 12 + 12)       # v92..v74 + v108 + v109
            r.read(16 + 16 + 16 + 16 + 4)  # v98..v101 + v102
            for _ in range(3):
                r.read_string()

        ns = r.u32()
        for _ in range(ns):
            dt = r.i32()
            if dt in (4, 5):
                return True   # Xbox-specific stream type found
            elif dt == 1:
                r.i32(); vc = r.u32(); vs = r.u32(); r.u32()
                r.read(vc * vs)
            elif dt == 2:
                r.i32(); ic = r.u32()
                r.read(ic * 2)
            elif dt == 3:
                r.i32(); vc = r.u32(); vs = r.u32()
                r.read(vc * vs)
            else:
                return False  # unknown dt — bail out
        return False
    except Exception:
        return False


class XboxBGMParser:
    """Parser for FlatOut 2 OG Xbox BGM files.

    Material, model, mesh, and object sections are binary-identical to PC BGM.
    Streams include dt=1 VBs and dt=4/dt=5 NV2A push-buffer IBs.
    Between the last stream and the surface table: a seek table and mystery block,
    whose sizes are derivable from the seek table's last entry.
    """

    def __init__(self, filepath: str):
        self.filepath  = filepath
        self.version   = 0
        self.materials: list = []
        # VB raw bytes — decoded on demand
        self.main_vb:   bytes = b''   # vs=16 NORMPACKED3 format
        self.shadow_vb: bytes = b''   # vs=12 float xyz (may be empty for menucar)
        # IB blobs — parsed lazily
        self.s2_blob:   bytes = b''   # dt=4 small/medium IB
        self.s3_blob:   bytes = b''   # dt=5 large IB (may be empty)
        self.xbox_surfaces: list = []  # list[XboxSurface]
        self.models:    list = []
        self.meshes:    list = []
        self.objects:   list = []

    def parse(self) -> bool:
        with open(self.filepath, 'rb') as f:
            raw = f.read()
        r = BinaryReader(raw)
        self.version = r.u32()

        # ── materials (identical to BGMParser at version 0x20000) ────────────
        nm = r.u32()
        for i in range(nm):
            mat   = BGMMaterial()
            ident = r.u32()
            if ident != 0x4354414D:
                print(f'[Xbox BGM] ERROR: Expected MATC at material {i}')
                return False
            mat.name         = r.read_string()
            mat.nAlpha       = r.i32()
            mat.v92          = r.i32()
            mat.nNumTextures = r.i32()
            mat.nShaderId    = r.i32()
            mat.nUseColormap = r.i32()
            mat.v74          = r.i32()
            mat.v108 = struct.unpack_from('<3i', r.read(12))
            mat.v109 = struct.unpack_from('<3i', r.read(12))
            mat.v98  = struct.unpack_from('<4i', r.read(16))
            mat.v99  = struct.unpack_from('<4i', r.read(16))
            mat.v100 = struct.unpack_from('<4i', r.read(16))
            mat.v101 = struct.unpack_from('<4i', r.read(16))
            mat.v102 = r.i32()
            mat.texture_names = [r.read_string() for _ in range(3)]
            self.materials.append(mat)

        # ── streams ──────────────────────────────────────────────────────────
        # dt=1 vs=12  → shadow VB (float xyz)
        # dt=1 vs=16  → main VB  (NORMPACKED3)
        # dt=4        → small/medium IB blob (NV2A push-buffer)
        # dt=5        → large IB blob (NV2A push-buffer)
        # Stream header for dt=4/5: dt, 0, len(blob), 1  then len bytes
        ns = r.u32()
        for _ in range(ns):
            dt = r.i32()
            if dt == 1:
                r.i32()                        # fc/fouc_extra = 0
                vc = r.u32(); vs = r.u32(); r.u32()   # vc, vs, flags
                data = r.read(vc * vs)
                if vs == 12:
                    self.shadow_vb = data
                else:
                    self.main_vb = data
            elif dt == 2:
                r.i32(); ic = r.u32()          # fc, ic
                r.read(ic * 2)                 # PC index buffer (not used on Xbox)
            elif dt == 3:
                r.i32(); vc = r.u32(); vs = r.u32()
                r.read(vc * vs)
            elif dt in (4, 5):
                r.i32()                        # always 0
                blob_len = r.u32(); r.u32()    # len, 1
                blob = r.read(blob_len)
                if dt == 4:
                    self.s2_blob = blob
                else:
                    self.s3_blob = blob
            else:
                print(f'[Xbox BGM] WARNING: Unknown stream dt={dt}, skipping')
                break

        # ── seek table ───────────────────────────────────────────────────────
        # Use the last entry's offset field to determine n_s3, then compute
        # mystery block size so we can skip over it to reach the surface table.
        seek_count = r.u32()
        r.u32()   # unk = 0
        last_off = 0
        for _ in range(seek_count):
            r.u32()               # size
            last_off = r.u32()    # off; the LAST entry's off = n_s3
        n_s3 = last_off if seek_count > 0 else 0

        # ── mystery block ────────────────────────────────────────────────────
        mystery_size = 16 if n_s3 == 0 else 8 * n_s3
        r.read(mystery_size)

        # ── surface table — 52 bytes (13×uint32) per surface ─────────────────
        nsf = r.u32()
        for _ in range(nsf):
            isveg, mid, vc, flags, pc, pm, niu = struct.unpack_from('<7i', raw, r.pos)
            e0, e1, e2, e3, e4, e5            = struct.unpack_from('<6I', raw, r.pos + 28)
            r.read(52)
            self.xbox_surfaces.append(XboxSurface(
                is_vegetation=isveg, material_id=mid, vertex_count=vc,
                flags=flags, poly_count=pc, poly_mode=pm, num_indices_used=niu,
                e0=e0, e1=e1, e2=e2, e3=e3, e4=e4, e5=e5,
            ))

        # ── models (identical to BGMParser) ─────────────────────────────────
        nmod = r.u32()
        for i in range(nmod):
            m     = Model()
            ident = r.u32()
            if ident != 0x444F4D42:
                print(f'[Xbox BGM] ERROR: Expected BMOD at model {i}')
                return False
            m.nUnk   = r.i32()
            m.name   = r.read_string()
            m.center = r.vec3f()
            m.radius = r.vec3f()
            m.fRadius = r.f32()
            for _ in range(r.u32()):
                m.surface_ids.append(r.i32())
            self.models.append(m)

        # ── meshes (identical to BGMParser) ─────────────────────────────────
        nmesh = r.u32()
        for i in range(nmesh):
            mesh  = BGMMesh()
            ident = r.u32()
            if ident != 0x4853454D:
                print(f'[Xbox BGM] ERROR: Expected MESH at mesh {i}')
                return False
            mesh.name1   = r.read_string()
            mesh.name2   = r.read_string()
            mesh.flags   = r.u32()
            mesh.group   = r.i32()
            mesh.matrix  = list(struct.unpack_from('<16f', r.read(64)))
            for _ in range(r.i32()):
                mesh.model_ids.append(r.i32())
            self.meshes.append(mesh)

        # ── objects (identical to BGMParser) ────────────────────────────────
        nobj = r.u32()
        for i in range(nobj):
            obj   = BGMObject()
            ident = r.u32()
            if ident != 0x434A424F:
                print(f'[Xbox BGM] ERROR: Expected OBJC at object {i}')
                return False
            obj.name1  = r.read_string()
            obj.name2  = r.read_string()
            obj.flags  = r.u32()
            obj.matrix = list(struct.unpack_from('<16f', r.read(64)))
            self.objects.append(obj)

        print(f'[Xbox BGM] Parsed {self.filepath}: version=0x{self.version:X}, '
              f'{len(self.materials)} mats, {len(self.meshes)} meshes, '
              f'{len(self.xbox_surfaces)} surfaces, {len(self.objects)} objects')
        return True


def _xbox_parse_ib_blob(blob: bytes) -> list:
    """Parse an Xbox NV2A push-buffer blob into a list of (poly_mode, indices) per surface.

    Each surface's push-buffer block:
        0x000417FC  N     BEGIN  (N=5=triangle-list, N=6=strip; N=0=null/end)
        DRAW commands until the next 0x000417FC word.

    DRAW_INLINE_ARRAY command:  bits[31:30]=01, cnt=(bits[28:18]), method=0x1800
        Followed by cnt×4 bytes (pairs of uint16 absolute VB indices).
    Odd-index command: 0x00041808 followed by uint16 index + uint16 pad.

    Returns list in surface-encounter order matching the e5 position values.
    """
    surfaces = []
    off  = 0
    n    = len(blob)

    while off + 8 <= n:
        word0 = struct.unpack_from('<I', blob, off)[0]
        if word0 != 0x000417FC:
            off += 4
            continue

        off  += 4
        word1 = struct.unpack_from('<I', blob, off)[0]
        off  += 4

        if word1 == 0:
            # Null / trail header — everything until next 0x000417FC is padding zeros
            # Just continue scanning; zero dwords will be skipped by outer loop.
            continue

        # Real BEGIN: word1 = N (5=list, 6=strip)
        pm      = 5 if word1 == 6 else 4
        indices = []

        while off + 4 <= n:
            cmd = struct.unpack_from('<I', blob, off)[0]
            if cmd == 0x000417FC:
                break  # next header found — do not consume
            off += 4

            if (cmd >> 30) & 3 == 1:
                # DRAW_INLINE_ARRAY: cnt dwords of index pairs
                cnt = (cmd >> 18) & 0x7FF
                for _ in range(cnt):
                    if off + 4 > n:
                        break
                    a, b = struct.unpack_from('<2H', blob, off)
                    off += 4
                    indices.append(a)
                    indices.append(b)
            elif cmd == 0x00041808:
                # Odd trailing index
                if off + 4 <= n:
                    idx_odd = struct.unpack_from('<H', blob, off)[0]
                    off += 4
                    indices.append(idx_odd)
            # All other commands (zeros from null padding) are silently skipped

        if indices:
            surfaces.append((pm, indices))

    return surfaces


def _xbox_decode_main_vert(vb_data: bytes, abs_idx: int):
    """Decode one vertex from the Xbox main VB (vs=16, NORMPACKED3).

    Returns (pos, uv, norm) where each is a float tuple.
    NORMPACKED3 decode:
        nx = sign_extend_11(bits[10:0])  / 1023.0
        ny = sign_extend_11(bits[21:11]) / 1023.0
        nz = sign_extend_10(bits[31:22]) / 511.0
    """
    off = abs_idx * 16
    if off + 16 > len(vb_data):
        return (0.0, 0.0, 0.0), (0.0, 0.0), (0.0, 0.0, 1.0)

    px_i, py_i, pz_i = struct.unpack_from('<3h', vb_data, off)
    norm_w            = struct.unpack_from('<I',  vb_data, off + 8)[0]
    u_i,  v_i         = struct.unpack_from('<2h', vb_data, off + 12)

    pos = (px_i / 1024.0, py_i / 1024.0, pz_i / 1024.0)

    x_r = norm_w & 0x7FF;         nx = (x_r - 2048 if x_r >= 1024 else x_r) / 1023.0
    y_r = (norm_w >> 11) & 0x7FF; ny = (y_r - 2048 if y_r >= 1024 else y_r) / 1023.0
    z_r = (norm_w >> 22) & 0x3FF; nz = (z_r - 1024 if z_r >= 512  else z_r) / 511.0

    return pos, (u_i / 2048.0, v_i / 2048.0), (nx, ny, nz)


def _xbox_decode_shadow_vert(vb_data: bytes, abs_idx: int):
    """Decode one vertex from the Xbox shadow VB (vs=12, float xyz)."""
    off = abs_idx * 12
    if off + 12 > len(vb_data):
        return (0.0, 0.0, 0.0), (0.0, 0.0), (0.0, 0.0, 1.0)
    pos = struct.unpack_from('<3f', vb_data, off)
    return pos, (0.0, 0.0), (0.0, 0.0, 1.0)


def extract_crash_vertices_xbox(crash_surf: 'CrashSurface') -> list:
    """Extract crash vertices from an Xbox crash surface.

    Positions and normals come from the 48-byte weight block (crash_pos / crash_normal).
    UVs are decoded from the Xbox vs=16 NORMPACKED3 vertex buffer via
    _xbox_decode_main_vert, matching the body mesh UV layout exactly.
    Shadow surfaces (vertex_size != 16) get no UV.
    """
    vertices = []
    for i in range(crash_surf.vertex_count):
        v = ParsedVertex()
        w = crash_surf.weights[i]
        v.x, v.y, v.z   = w.crash_pos
        v.nx, v.ny, v.nz = w.crash_normal
        v.has_normal = True
        if crash_surf.vertex_size == 16:
            _, uv, _ = _xbox_decode_main_vert(crash_surf.vertex_data, i)
            v.u, v.v = uv
            v.has_uv = True
        vertices.append(v)
    return vertices


def build_blender_meshes_xbox(context, parser: 'XboxBGMParser', options: dict):
    """Build Blender mesh objects from a parsed Xbox BGM file.

    Creates the same FO2 Body / FO2 Body Dummies / FO2 Body Crash collection
    structure and uses the same FO2→Blender axis transform (Y↔Z swap) as the
    other platform imports.  Geometry is decoded from NV2A push-buffer IB blobs
    and NORMPACKED3 VB data.  Crash meshes are imported from an auto-detected
    Xbox crash.dat when present (same binary structure as PC crash.dat, vs=16).
    """

    bgm_dir      = os.path.dirname(parser.filepath)
    shared_dir   = options.get('shared_texture_dir', '')
    use_alpha    = options.get('use_alpha', True)
    alpha_mode   = options.get('alpha_mode', 'BLEND')
    transparency_overlap = options.get('transparency_overlap', False)
    max_lod      = options.get('max_lod', 0)
    global_scale = options.get('global_scale', 1.0)
    clamp_size   = options.get('clamp_size', 0.0)
    use_origins  = options.get('use_origins', True)
    validate_meshes      = options.get('validate_meshes', False)
    convert_dds          = options.get('convert_dds', False)
    use_backface_culling = options.get('use_backface_culling', True)
    import_body    = options.get('import_body',    True)
    import_crash   = options.get('import_crash',   True)
    import_dummies = options.get('import_dummies', True)

    auto_shared_dir = os.path.join(os.path.dirname(bgm_dir), 'shared')
    if not os.path.isdir(auto_shared_dir):
        auto_shared_dir = ''

    # fixed coordinate transform: FO2(x,y,z) → Blender(x,z,y)  [Y↔Z swap]
    axis_matrix = Matrix((
        (1, 0, 0, 0),
        (0, 0, 1, 0),
        (0, 1, 0, 0),
        (0, 0, 0, 1),
    ))

    # ── parse IB blobs once up-front ─────────────────────────────────────────
    # s2_surfaces[e5] / s3_surfaces[e5] give (poly_mode, [abs_indices])
    s2_surfaces = _xbox_parse_ib_blob(parser.s2_blob)
    s3_surfaces = _xbox_parse_ib_blob(parser.s3_blob)

    # ── crash.dat auto-detection and loading ─────────────────────────────────
    # Xbox crash.dat is structurally identical to PC crash.dat (nv + nvb + vbuf
    # + nv×48 weights), with vs=16 NORMPACKED3 vertices instead of float.
    # parse_crash_dat() handles the binary layout without caring about vs.
    crash_dat_path = options.get('crash_dat_path', '')
    if not crash_dat_path or not os.path.isfile(crash_dat_path):
        base_no_ext = os.path.splitext(parser.filepath)[0]
        for candidate in (
            base_no_ext + '_crash.dat',
            base_no_ext + '-crash.dat',
            os.path.join(bgm_dir, 'crash.dat'),
        ):
            if os.path.isfile(candidate):
                crash_dat_path = candidate
                break
        else:
            crash_dat_path = ''

    # crash_by_model: model_name -> CrashNode
    crash_by_model = {}
    if import_crash and crash_dat_path:
        crash_nodes = parse_crash_dat(crash_dat_path, is_fouc=False)
        print(f'[Xbox BGM] Loaded crash data from: {crash_dat_path}')
        for cn in crash_nodes:
            if cn.name.endswith('_crash'):
                crash_by_model[cn.name[:-6]] = cn

    # ── materials ────────────────────────────────────────────────────────────
    blender_materials = {}
    for i, bgm_mat in enumerate(parser.materials):
        bl_mat = create_blender_material(
            bgm_mat, bgm_dir, shared_dir, use_alpha,
            alpha_mode, transparency_overlap,
            auto_shared_dir, convert_dds,
            use_backface_culling,
            is_fouc=False,
        )
        blender_materials[i] = bl_mat

    # ── FO2 Body collection + root empty ────────────────────────────────────
    fo2_body_coll = bpy.data.collections.get('FO2 Body')
    if fo2_body_coll is None:
        fo2_body_coll = bpy.data.collections.new('FO2 Body')
    if fo2_body_coll.name not in context.scene.collection.children:
        context.scene.collection.children.link(fo2_body_coll)

    root_empty = bpy.data.objects.new('fo2_body', None)
    root_empty.empty_display_type = 'PLAIN_AXES'
    root_empty.empty_display_size = 0.5
    root_empty['bgm_is_fouc']  = False
    root_empty['bgm_is_fo1']   = False
    root_empty['bgm_is_xbox']  = True
    root_empty['bgm_version']  = parser.version
    fo2_body_coll.objects.link(root_empty)

    # ── FO2 Body Crash collection + crash root empty (if crash data found) ────
    fo2_crash_coll   = None
    crash_root_empty = None
    if crash_by_model:
        fo2_crash_coll = bpy.data.collections.get('FO2 Body Crash')
        if fo2_crash_coll is None:
            fo2_crash_coll = bpy.data.collections.new('FO2 Body Crash')
        if fo2_crash_coll.name not in context.scene.collection.children:
            context.scene.collection.children.link(fo2_crash_coll)
        crash_root_empty = bpy.data.objects.new('fo2_body_crash', None)
        crash_root_empty.empty_display_type = 'PLAIN_AXES'
        crash_root_empty.empty_display_size = 0.5
        fo2_crash_coll.objects.link(crash_root_empty)

    # ── FO2 Body Dummies collection + container empty ────────────────────────
    fo2_dummies_coll = bpy.data.collections.get('FO2 Body Dummies')
    if fo2_dummies_coll is None:
        fo2_dummies_coll = bpy.data.collections.new('FO2 Body Dummies')
    if fo2_dummies_coll.name not in context.scene.collection.children:
        context.scene.collection.children.link(fo2_dummies_coll)

    dummies_empty = bpy.data.objects.new('fo2_body_dummies', None)
    dummies_empty.empty_display_type = 'PLAIN_AXES'
    dummies_empty.empty_display_size = 0.5
    fo2_dummies_coll.objects.link(dummies_empty)
    dummies_empty.parent = root_empty

    # ── object empties (dummies) — identical transform logic to other imports ─
    object_empties = {}
    if not import_dummies:
        parser.objects = []
    for bgm_obj in parser.objects:
        obj_empty = bpy.data.objects.new(bgm_obj.name1, None)
        obj_empty.empty_display_type = 'PLAIN_AXES'
        obj_empty.empty_display_size = 0.3
        M = fo2_matrix_to_blender(bgm_obj.matrix)
        obj_mat = Matrix((
            (M[0][0], M[0][2], M[0][1], M[0][3]),
            (M[2][0], M[2][2], M[2][1], M[2][3]),
            (M[1][0], M[1][2], M[1][1], M[1][3]),
            (M[3][0], M[3][2], M[3][1], M[3][3]),
        ))
        obj_mat[0][3] *= global_scale
        obj_mat[1][3] *= global_scale
        obj_mat[2][3] *= global_scale
        obj_empty.matrix_world = obj_mat
        fo2_dummies_coll.objects.link(obj_empty)
        obj_empty.parent = dummies_empty
        obj_empty['bgm_obj_flags'] = bgm_obj.flags
        object_empties[bgm_obj.name1] = obj_empty

    created_objects = []

    if not import_body:
        bpy.ops.object.select_all(action='DESELECT')
        print('[Xbox BGM] Import complete: 0 mesh objects created (import_body=False)')
        return created_objects

    # ── geometry ─────────────────────────────────────────────────────────────
    for bgm_mesh in parser.meshes:
        if not bgm_mesh.model_ids:
            continue
        lod_idx  = min(max_lod, len(bgm_mesh.model_ids) - 1)
        model_id = bgm_mesh.model_ids[lod_idx]
        if model_id < 0 or model_id >= len(parser.models):
            continue
        model = parser.models[model_id]

        mesh_matrix          = fo2_matrix_to_blender(bgm_mesh.matrix)
        crash_node           = crash_by_model.get(model.name)

        all_verts            = []
        all_normals          = []
        all_face_uvs         = []
        all_faces            = []
        all_face_mat_indices = []
        mat_index_map        = {}
        mesh_materials       = []

        # crash mesh accumulators — built in parallel with the body mesh
        crash_surf_pairs     = []  # (XboxSurface, CrashSurface, surf_local_idx)

        for surf_local_idx, surf_id in enumerate(model.surface_ids):
            if surf_id < 0 or surf_id >= len(parser.xbox_surfaces):
                continue
            surf = parser.xbox_surfaces[surf_id]
            if surf.num_indices_used == 0:
                continue

            # Determine IB source (stream2 or stream3) and look up index list
            is_shadow = (surf.flags == 0x0002)
            ib_list   = s3_surfaces if surf.e4 == 1 else s2_surfaces
            if surf.e5 >= len(ib_list):
                continue
            pm, indices = ib_list[surf.e5]

            mat_id = surf.material_id
            if mat_id not in mat_index_map:
                mat_index_map[mat_id] = len(mesh_materials)
                mesh_materials.append(blender_materials.get(mat_id))
            local_mat_idx = mat_index_map[mat_id]

            # Build triangles with winding reversal (Y↔Z reflection compensation)
            # Matches PC extract_indices: mode4=(i2,i1,i0), mode5=alternating flip
            tris = []
            if pm == 4:  # triangle list
                for j in range(0, len(indices) - 2, 3):
                    i0, i1, i2 = indices[j], indices[j + 1], indices[j + 2]
                    tris.append((i2, i1, i0))
            else:         # triangle strip (pm == 5)
                flip = False
                for j in range(len(indices) - 2):
                    i0, i1, i2 = indices[j], indices[j + 1], indices[j + 2]
                    tris.append((i0, i1, i2) if flip else (i2, i1, i0))
                    flip = not flip

            # Decode vertices per triangle loop corner (flat/per-loop like PS2/PSP)
            decode = (_xbox_decode_shadow_vert if is_shadow else _xbox_decode_main_vert)
            vb     = parser.shadow_vb if is_shadow else parser.main_vb

            for i0, i1, i2 in tris:
                # skip degenerate (same as build_blender_meshes filter)
                if i0 == i1 or i1 == i2 or i0 == i2:
                    continue
                base       = len(all_verts)
                face_uvs_t = []
                for abs_idx in (i0, i1, i2):
                    v_pos, v_uv, v_norm = decode(vb, abs_idx)
                    pos = Vector(v_pos)
                    nrm = Vector(v_norm)
                    if not use_origins:
                        pos = mesh_matrix @ pos
                        nrm = mesh_matrix.to_3x3() @ nrm
                    pos  = axis_matrix @ pos
                    pos *= global_scale
                    nrm  = axis_matrix.to_3x3() @ nrm
                    if nrm.length > 0:
                        nrm.normalize()
                    all_verts.append(pos)
                    all_normals.append(nrm)
                    face_uvs_t.append((v_uv[0], 1.0 - v_uv[1]))   # flip V
                all_faces.append((base, base + 1, base + 2))
                all_face_mat_indices.append(local_mat_idx)
                all_face_uvs.append(tuple(face_uvs_t))

            # collect crash surface pair (skip shadow — no crash geometry for those)
            if (not is_shadow and crash_node and
                    surf_local_idx < len(crash_node.surfaces)):
                cs = crash_node.surfaces[surf_local_idx]
                cs.flags = surf.flags   # copy flags for UV format detection
                crash_surf_pairs.append((surf, cs, tris))

        if not all_faces:
            continue

        mesh_name = bgm_mesh.name1 if bgm_mesh.name1 else 'bgm_xbox_mesh'
        bl_mesh   = bpy.data.meshes.new(mesh_name)

        for bl_mat in mesh_materials:
            if bl_mat:
                bl_mesh.materials.append(bl_mat)

        bl_mesh.vertices.add(len(all_verts))
        bl_mesh.loops.add(len(all_faces) * 3)
        bl_mesh.polygons.add(len(all_faces))

        flat_co = []
        for v in all_verts:
            flat_co.extend((v.x, v.y, v.z))
        bl_mesh.vertices.foreach_set('co', flat_co)

        loop_verts = []
        for f in all_faces:
            loop_verts.extend(f)
        bl_mesh.loops.foreach_set('vertex_index', loop_verts)

        bl_mesh.polygons.foreach_set('loop_start', [i * 3 for i in range(len(all_faces))])
        bl_mesh.polygons.foreach_set('loop_total',  [3]   * len(all_faces))

        if all_face_mat_indices:
            bl_mesh.polygons.foreach_set('material_index', all_face_mat_indices)

        bl_mesh.polygons.foreach_set('use_smooth', [True] * len(all_faces))

        if all_face_uvs:
            uv_layer = bl_mesh.uv_layers.new(name='UVMap')
            uv_data  = []
            for face_uvs in all_face_uvs:
                uv_data.extend(face_uvs)
            for i, uv in enumerate(uv_data):
                uv_layer.data[i].uv = uv

        bl_mesh.update()
        bl_mesh.validate()

        if all_normals:
            loop_normals = []
            for f in all_faces:
                for vi in f:
                    loop_normals.append(all_normals[vi])
            try:
                bl_mesh.normals_split_custom_set(loop_normals)
            except Exception:
                try:
                    bl_mesh.use_auto_smooth = True
                    bl_mesh.normals_split_custom_set(loop_normals)
                except Exception:
                    pass

        if validate_meshes:
            bl_mesh.validate(verbose=True)

        if clamp_size > 0:
            max_dim = max(bl_mesh.dimensions) if bl_mesh.dimensions else 0
            if max_dim > clamp_size:
                sf = clamp_size / max_dim
                for vert in bl_mesh.vertices:
                    vert.co *= sf

        bl_obj = bpy.data.objects.new(mesh_name, bl_mesh)
        fo2_body_coll.objects.link(bl_obj)
        created_objects.append(bl_obj)

        if use_origins:
            M  = mesh_matrix
            om = Matrix((
                (M[0][0], M[0][2], M[0][1], M[0][3]),
                (M[2][0], M[2][2], M[2][1], M[2][3]),
                (M[1][0], M[1][2], M[1][1], M[1][3]),
                (M[3][0], M[3][2], M[3][1], M[3][3]),
            ))
            om[0][3] *= global_scale
            om[1][3] *= global_scale
            om[2][3] *= global_scale
            bl_obj.matrix_world = om

        bl_obj.parent         = root_empty
        bl_obj['bgm_flags']   = bgm_mesh.flags
        bl_obj['bgm_group']   = bgm_mesh.group
        bl_obj['bgm_name2']   = bgm_mesh.name2
        bl_obj['bgm_is_xbox'] = True

        # ── crash mesh ────────────────────────────────────────────────────────
        if crash_surf_pairs and fo2_crash_coll:
            crash_all_verts            = []
            crash_all_normals          = []
            crash_all_uvs              = []
            crash_all_faces            = []
            crash_all_face_mat_indices = []
            crash_mat_index_map        = {}
            crash_mesh_materials       = []
            crash_vert_offset          = 0

            for surf, crash_surf, tris in crash_surf_pairs:
                if crash_surf.vertex_count != surf.vertex_count:
                    print(f'[Xbox crash.dat] WARNING: vertex count mismatch for {mesh_name}')
                    continue

                verts = extract_crash_vertices_xbox(crash_surf)
                if not verts:
                    continue

                mat_id = surf.material_id
                if mat_id not in crash_mat_index_map:
                    crash_mat_index_map[mat_id] = len(crash_mesh_materials)
                    crash_mesh_materials.append(blender_materials.get(mat_id))
                crash_local_mat = crash_mat_index_map[mat_id]

                # Xbox IB indices are absolute into the main VB.
                # surf.e0 is the base vertex for this surface, so:
                #   local crash vertex index = abs_idx - surf.e0
                e0 = surf.e0
                nv = len(verts)

                for v in verts:
                    pos = Vector((v.x, v.y, v.z))
                    nrm = Vector((v.nx, v.ny, v.nz))
                    if not use_origins:
                        pos = mesh_matrix @ pos
                        nrm = mesh_matrix.to_3x3() @ nrm
                    pos  = axis_matrix @ pos
                    pos *= global_scale
                    nrm  = axis_matrix.to_3x3() @ nrm
                    if nrm.length > 0:
                        nrm.normalize()
                    crash_all_verts.append(pos)
                    crash_all_normals.append(nrm)
                    crash_all_uvs.append((v.u, 1.0 - v.v) if v.has_uv else (0.0, 0.0))

                for i0, i1, i2 in tris:
                    fi0 = i0 - e0
                    fi1 = i1 - e0
                    fi2 = i2 - e0
                    if not (0 <= fi0 < nv and 0 <= fi1 < nv and 0 <= fi2 < nv):
                        continue
                    if fi0 == fi1 or fi1 == fi2 or fi0 == fi2:
                        continue
                    crash_all_faces.append((
                        crash_vert_offset + fi0,
                        crash_vert_offset + fi1,
                        crash_vert_offset + fi2,
                    ))
                    crash_all_face_mat_indices.append(crash_local_mat)

                crash_vert_offset += nv

            if crash_all_faces:
                crash_mesh_name  = mesh_name + '_crash'
                bl_crash_mesh    = bpy.data.meshes.new(crash_mesh_name)
                for bm in crash_mesh_materials:
                    if bm:
                        bl_crash_mesh.materials.append(bm)

                bl_crash_mesh.vertices.add(len(crash_all_verts))
                bl_crash_mesh.loops.add(len(crash_all_faces) * 3)
                bl_crash_mesh.polygons.add(len(crash_all_faces))

                flat_co = []
                for v in crash_all_verts:
                    flat_co.extend((v.x, v.y, v.z))
                bl_crash_mesh.vertices.foreach_set('co', flat_co)

                lv = []
                for f in crash_all_faces:
                    lv.extend(f)
                bl_crash_mesh.loops.foreach_set('vertex_index', lv)
                bl_crash_mesh.polygons.foreach_set('loop_start',
                    [i * 3 for i in range(len(crash_all_faces))])
                bl_crash_mesh.polygons.foreach_set('loop_total',
                    [3] * len(crash_all_faces))
                if crash_all_face_mat_indices:
                    bl_crash_mesh.polygons.foreach_set('material_index',
                        crash_all_face_mat_indices)
                bl_crash_mesh.polygons.foreach_set('use_smooth',
                    [True] * len(crash_all_faces))

                if crash_all_uvs:
                    uv_layer = bl_crash_mesh.uv_layers.new(name='UVMap')
                    uv_data  = []
                    for f in crash_all_faces:
                        for vi in f:
                            uv_data.append(crash_all_uvs[vi])
                    for i, uv in enumerate(uv_data):
                        uv_layer.data[i].uv = uv

                bl_crash_mesh.update()
                bl_crash_mesh.validate()

                if crash_all_normals:
                    loop_normals = []
                    for f in crash_all_faces:
                        for vi in f:
                            loop_normals.append(crash_all_normals[vi])
                    try:
                        bl_crash_mesh.normals_split_custom_set(loop_normals)
                    except Exception:
                        try:
                            bl_crash_mesh.use_auto_smooth = True
                            bl_crash_mesh.normals_split_custom_set(loop_normals)
                        except Exception:
                            pass

                if validate_meshes:
                    bl_crash_mesh.validate(verbose=True)

                crash_obj = bpy.data.objects.new(crash_mesh_name, bl_crash_mesh)
                fo2_crash_coll.objects.link(crash_obj)
                created_objects.append(crash_obj)
                crash_obj.parent = crash_root_empty
                if use_origins:
                    M  = mesh_matrix
                    cm = Matrix((
                        (M[0][0], M[0][2], M[0][1], M[0][3]),
                        (M[2][0], M[2][2], M[2][1], M[2][3]),
                        (M[1][0], M[1][2], M[1][1], M[1][3]),
                        (M[3][0], M[3][2], M[3][1], M[3][3]),
                    ))
                    cm[0][3] *= global_scale
                    cm[1][3] *= global_scale
                    cm[2][3] *= global_scale
                    crash_obj.matrix_world = cm
                crash_obj['bgm_flags']    = bgm_mesh.flags
                crash_obj['bgm_group']    = bgm_mesh.group
                crash_obj['bgm_is_crash'] = True
                crash_obj['bgm_is_xbox']  = True
                print(f'[Xbox crash.dat] Created crash mesh: {crash_mesh_name} '
                      f'({len(crash_all_verts)} verts, {len(crash_all_faces)} faces)')

    bpy.ops.object.select_all(action='DESELECT')
    for obj in created_objects:
        obj.select_set(True)
    if created_objects:
        context.view_layer.objects.active = created_objects[0]

    print(f'[Xbox BGM] Import complete: {len(created_objects)} mesh objects created')
    return created_objects


