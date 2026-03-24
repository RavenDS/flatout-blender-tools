"""
bgm_ps2.py — FlatOut 2 PS2 BGM import support.
Part of FlatOut Blender Tools — https://github.com/RavenDS/flatout-blender-tools

Also contains the shared type-3 stream detector used for both PS2 and PSP
(_detect_type3_bgm / _detect_ps2_bgm / _detect_psp_bgm), since the two
formats are distinguished only by their surface record layout.

Covers:
  • _detect_type3_bgm / _detect_ps2_bgm / _detect_psp_bgm
  • PS2Surface, PS2BGMParser
  • _ps2_extract_batches, _ps2_triangles_from_adc
  • parse_ps2_crash_dat
  • build_blender_meshes_ps2
"""
import bpy
import struct
import os
from mathutils import Matrix, Vector
from dataclasses import dataclass, field
from .bgm_common import (
    BinaryReader, BGMMaterial, Model, BGMMesh, BGMObject,
    fo2_matrix_to_blender, create_blender_material,
)

# ─────────────────────────── PS2 BGM SUPPORT ────────────────────────────────
#
# FlatOut 2 PS2 BGM files share the same material / model / mesh / object
# binary layout as PC BGM files, but geometry is stored as a single VIF-packet
# blob (stream type 3) rather than separate vertex + index buffers (types 1/2).
# Surfaces are a fixed 10×uint32 record pointing into that blob.
#
# Detection: if the file's stream section contains only type-3 entries (no
# type 1 or 2), it is a PS2 BGM.  Detection is done with a lightweight peek
# that mirrors the existing BGMParser stream loop but stops early.
#
# VIF decoding is ported from bgm2obj_ps2.py / bgm2fbx-ascii_ps2.py.


@dataclass
class PS2Surface:
    unk0:        int = 0
    unk1:        int = 0
    material_id: int = 0
    total_verts: int = 0
    num_batches: int = 0
    unk5:        int = 0
    unk6:        int = 0
    unk7:        int = 0
    blob_offset: int = 0
    blob_size:   int = 0


def _detect_ps2_bgm(filepath: str) -> bool:
    """Return True when the file is a PS2 BGM (type-3-only streams, 40-byte surface records).

    Both PS2 and PSP use a single type-3 stream and can share the same version,
    so version alone cannot distinguish them.  The surface record layout is the
    reliable marker:
      PS2: 10 × uint32 (40 B) — word[0]=0,      word[1]=0x1000 (flags)
      PSP:  8 × uint32 (32 B) — word[0]=0x1000  (flags at start)
    """
    return _detect_type3_bgm(filepath) == 'ps2'


def _detect_psp_bgm(filepath: str) -> bool:
    """Return True when the file is a PSP BGM (type-3-only streams, 32-byte surface records).

    See _detect_ps2_bgm for the disambiguation logic.
    """
    return _detect_type3_bgm(filepath) == 'psp'


def _detect_type3_bgm(filepath: str) -> str:
    """Classify a type-3-only BGM file as 'ps2', 'psp', or '' (not type-3-only).

    After confirming all streams are type 3, reads the first surface record and
    checks word[0]:
      word[0] == 0 and word[1] == 0x1000  →  'ps2'  (40-byte record, unk0 first)
      word[0] == 0x1000                   →  'psp'  (32-byte record, flags first)
    """
    try:
        with open(filepath, 'rb') as f:
            raw = f.read()
        r = BinaryReader(raw)
        version = r.u32()

        # skip materials
        nm = r.u32()
        for _ in range(nm):
            r.u32()          # ident MATC
            r.read_string()  # name
            r.i32()          # nAlpha
            if version >= 0x10004 or version == 0x10002:
                r.read(20 + 12 + 12)  # v92..v74 + v108 + v109
            r.read(16 + 16 + 16 + 16 + 4)  # v98..v101 + v102
            for _ in range(3):
                r.read_string()

        # scan streams — bail out immediately if any non-type-3 stream is found
        ns = r.u32()
        has_type3 = False
        for _ in range(ns):
            dt = r.i32()
            if dt == 1:
                r.i32(); vc = r.u32(); vs = r.u32(); r.u32()
                r.read(vc * vs)
                return ''   # has type-1 → not PS2/PSP
            elif dt == 2:
                r.i32(); ic = r.u32()
                r.read(ic * 2)
                return ''   # has type-2 → not PS2/PSP
            elif dt in (4, 5):
                return ''   # has Xbox IB → not PS2/PSP
            elif dt == 3:
                has_type3 = True
                r.i32(); vc = r.u32(); vs = r.u32()
                r.read(vc * vs)

        if not has_type3:
            return ''

        # peek at first surface record — word[0] distinguishes PS2 from PSP
        nsf = r.u32()
        if nsf == 0:
            return ''
        w0 = r.u32()
        w1 = r.u32()
        if w0 == 0 and w1 == 0x1000:
            return 'ps2'   # 40-byte record: unk0=0, flags=0x1000
        if w0 == 0x1000:
            return 'psp'   # 32-byte record: flags=0x1000
        return ''
    except Exception:
        return ''


class PS2BGMParser:
    """Parser for FlatOut 2 PS2 BGM files.

    Material, model, mesh, and object sections are binary-identical to PC BGM
    and are parsed with the same logic.  Only the stream and surface sections
    differ (VIF blob + fixed 40-byte surface records).
    """

    def __init__(self, filepath: str):
        self.filepath   = filepath
        self.version    = 0
        self.materials: list = []
        self.blob:      bytes = b''
        self.ps2_surfaces: list = []   # list[PS2Surface]
        self.models:    list = []      # list[Model]  (same dataclass as PC)
        self.meshes:    list = []      # list[BGMMesh]
        self.objects:   list = []      # list[BGMObject]

    def parse(self) -> bool:
        with open(self.filepath, 'rb') as f:
            raw = f.read()
        r = BinaryReader(raw)
        self.version = r.u32()

        # ── materials (identical to BGMParser) ──────────────────────────────
        nm = r.u32()
        for i in range(nm):
            mat  = BGMMaterial()
            ident = r.u32()
            if ident != 0x4354414D:
                print(f"[PS2 BGM] ERROR: Expected MATC at material {i}")
                return False
            mat.name = r.read_string()
            mat.nAlpha = r.i32()
            if self.version >= 0x10004 or self.version == 0x10002:
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

        # ── streams — only type 3 expected for PS2 ──────────────────────────
        ns = r.u32()
        for _ in range(ns):
            dt = r.i32()
            if dt == 3:
                r.i32()                   # fouc_extra_format (always 0 for PS2)
                vc = r.u32(); vs = r.u32()
                self.blob = r.read(vc * vs)
            elif dt == 1:
                r.i32(); vc = r.u32(); vs = r.u32(); r.u32()
                r.read(vc * vs)
            elif dt == 2:
                r.i32(); ic = r.u32()
                r.read(ic * 2)

        # ── surfaces — PS2 format: exactly 10 × uint32 = 40 bytes each ──────
        # Layout (from bgm_tool_ps2.py write_ps2_bgm):
        #   [0]=0, [1]=0x1000, [2]=material_id, [3]=total_verts, [4]=num_batches,
        #   [5]=1,  [6]=0x0E,  [7]=0,           [8]=blob_offset, [9]=blob_size
        nsf = r.u32()
        for _ in range(nsf):
            vals = struct.unpack_from('<10I', raw, r.pos)
            r.read(40)
            self.ps2_surfaces.append(PS2Surface(
                unk0=vals[0],        unk1=vals[1],
                material_id=vals[2], total_verts=vals[3],
                num_batches=vals[4], unk5=vals[5],
                unk6=vals[6],        unk7=vals[7],
                blob_offset=vals[8], blob_size=vals[9],
            ))

        # ── models (identical to BGMParser) ─────────────────────────────────
        nmod = r.u32()
        for i in range(nmod):
            m     = Model()
            ident = r.u32()
            if ident != 0x444F4D42:
                print(f"[PS2 BGM] ERROR: Expected BMOD at model {i}")
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
                print(f"[PS2 BGM] ERROR: Expected MESH at mesh {i}")
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
                print(f"[PS2 BGM] ERROR: Expected OBJC at object {i}")
                return False
            obj.name1  = r.read_string()
            obj.name2  = r.read_string()
            obj.flags  = r.u32()
            obj.matrix = list(struct.unpack_from('<16f', r.read(64)))
            self.objects.append(obj)

        print(f"[PS2 BGM] Parsed {self.filepath}: version=0x{self.version:X}, "
              f"{len(self.materials)} mats, {len(self.meshes)} meshes, "
              f"{len(self.ps2_surfaces)} surfaces, {len(self.objects)} objects")
        return True


def _ps2_extract_batches(blob: bytes, surf: PS2Surface) -> list:
    """Extract per-batch vertex data from PS2 VIF packets.

    Each batch dict has keys 'pos', 'uv', 'norm', 'adc' (lists of decoded values).
    Ported from bgm2obj_ps2.py / bgm2fbx-ascii_ps2.py.

    VIF format key:
      V3-16 @ addr 7  → positions (standard / colored)
      V4-16 @ addr 7  → positions (colored, w component discarded)
      V4-16 @ addr 5  → shadow positions + ADC in 4th component
      V2-16 @ addr 8  → UV coordinates
      V4-8  @ addr 9  → normals (xyz) + ADC flag (4th byte)
    """
    off  = surf.blob_offset
    end  = off + surf.blob_size
    batches   = []
    cur_pos   = []; cur_uv = []; cur_norm = []; cur_adc = []

    while off < end:
        if off + 4 > len(blob):
            break
        w   = struct.unpack_from('<I', blob, off)[0]
        cmd = (w >> 24) & 0x7F

        if cmd in (0x00, 0x01, 0x04):   # NOP / STCYCL / ITOP — skip
            off += 4
            continue

        if cmd == 0x17:                  # MSCNT — end of batch, flush
            if cur_pos:
                batches.append({'pos': cur_pos, 'uv': cur_uv,
                                'norm': cur_norm, 'adc': cur_adc})
            cur_pos = []; cur_uv = []; cur_norm = []; cur_adc = []
            off += 4
            continue

        if cmd >= 0x60:                  # UNPACK commands
            vn   = (cmd >> 2) & 3        # vector components (0=1, 1=2, 2=3, 3=4)
            vl   = cmd & 3               # element size   (0=32b, 1=16b, 2=8b, 3=5b)
            num  = (w >> 16) & 0xFF      # number of vectors
            addr = w & 0x3FF             # VU1 destination address
            eb   = [4, 2, 1, 0][vl]     # bytes per element
            comp = [1, 2, 3, 4][vn]     # components per vector
            db   = (num * comp * eb + 3) & ~3   # data bytes (4-byte aligned)
            doff = off + 4               # start of inline data

            if   vn == 2 and vl == 1 and addr == 7:   # V3-16 positions
                for vi in range(num):
                    x, y, z = struct.unpack_from('<3h', blob, doff + vi * 6)
                    cur_pos.append((x / 1024.0, y / 1024.0, z / 1024.0))

            elif vn == 3 and vl == 1 and addr == 7:   # V4-16 positions (colored)
                for vi in range(num):
                    x, y, z, _ = struct.unpack_from('<4h', blob, doff + vi * 8)
                    cur_pos.append((x / 1024.0, y / 1024.0, z / 1024.0))

            elif vn == 3 and vl == 1 and addr == 5:   # V4-16 shadow pos + ADC
                for vi in range(num):
                    x, y, z, adc = struct.unpack_from('<4h', blob, doff + vi * 8)
                    cur_pos.append((x / 1024.0, y / 1024.0, z / 1024.0))
                    cur_adc.append(adc)

            elif vn == 1 and vl == 1 and addr == 8:   # V2-16 UVs
                for vi in range(num):
                    u, v = struct.unpack_from('<2h', blob, doff + vi * 4)
                    cur_uv.append((u / 4096.0, v / 4096.0))

            elif vn == 3 and vl == 2 and addr == 9:   # V4-8 normals + ADC
                for vi in range(num):
                    nx, ny, nz, adc = struct.unpack_from('<4b', blob, doff + vi * 4)
                    cur_norm.append((nx / 127.0, ny / 127.0, nz / 127.0))
                    cur_adc.append(adc)

            off += 4 + db
        else:
            off += 4

    # flush any trailing batch not terminated by MSCNT
    if cur_pos:
        batches.append({'pos': cur_pos, 'uv': cur_uv,
                        'norm': cur_norm, 'adc': cur_adc})
    return batches


def _ps2_triangles_from_adc(positions, uvs, normals, adcs):
    """Decode an ADC-flagged PS2 triangle strip into a list of triangles.

    Each triangle is ((pos,uv,norm), (pos,uv,norm), (pos,uv,norm)).

    ADC flag encoding (4th byte of norm V4-8, or 4th component of shadow V4-16):
      bit 7 (0x80) = SKIP — don't emit a triangle for this vertex
      bit 5 (0x20) = odd winding — swap first two vertices of the triangle

    All vertices accumulate into a sliding window; the window never resets
    between batches (within one surface the strip is continuous).

    Ported from bgm2obj_ps2.py / bgm2fbx-ascii_ps2.py.
    """
    tris  = []
    verts = []   # running window of (pos, uv, norm)

    for i in range(len(positions)):
        p  = positions[i]
        uv = uvs[i]     if i < len(uvs)     else (0.0, 0.0)
        n  = normals[i] if i < len(normals)  else (0.0, 0.0, 1.0)
        verts.append((p, uv, n))

        adc   = adcs[i] if i < len(adcs) else 0
        adc_u = adc & 0xFF
        skip  = (adc_u >> 7) & 1
        odd   = (adc_u >> 5) & 1

        if not skip and i >= 2:
            v0, v1, v2 = verts[i - 2], verts[i - 1], verts[i]
            tris.append((v0, v1, v2) if odd else (v1, v0, v2))

    return tris


def parse_ps2_crash_dat(filepath: str) -> dict:
    """Parse a FlatOut 2 PS2 crash.dat file.

    Returns a dict mapping node_name (str) to a list of surface dicts, one per
    BGM model surface.  Each surface dict has:
        'batch_sizes'  : list[int]  — vertex count per VIF batch
        'total_verts'  : int        — sum of batch_sizes
        'verts'        : list of (base_pos, crash_pos, base_norm, crash_norm)
                         tuples, in flat batch order (matches VIF blob order)

    The crash positions / normals are in FO2 world space (before the Y↔Z axis
    swap); the caller applies the same coordinate transform used for regular
    mesh vertices.

    Format (written by bgm_tool_ps2.py generate_ps2_crash_dat):
        uint32   num_nodes
        per node:
            cstring  name  (e.g. 'body_crash')
            uint32   num_surfaces
            per surface:
                uint32          num_batches
                uint32[nb]      batch_vertex_counts
                uint32[nb]      pos_vif_offsets     (not used at import)
                uint32[nb]      adc_vif_offsets     (not used at import)
                uint32          total_verts
                float32[12*tv]  vertex_data
                    each vertex: base_px,py,pz, crash_px,py,pz,
                                 base_nx,ny,nz, crash_nx,ny,nz
    """
    result = {}
    try:
        data = open(filepath, 'rb').read()
    except (OSError, IOError):
        return result

    r = BinaryReader(data)
    num_nodes = r.u32()

    for _ in range(num_nodes):
        name     = r.read_string()
        num_surfs = r.u32()
        surfs    = []
        for _ in range(num_surfs):
            num_batches  = r.u32()
            batch_sizes  = [r.u32() for _ in range(num_batches)]
            # pos_vif_offsets and adc_vif_offsets — read and discard
            for _ in range(num_batches): r.u32()
            for _ in range(num_batches): r.u32()
            total_verts  = r.u32()
            verts = []
            for _ in range(total_verts):
                raw = struct.unpack_from('<12f', r.read(48))
                verts.append((
                    raw[0:3],   # base_pos
                    raw[3:6],   # crash_pos
                    raw[6:9],   # base_norm
                    raw[9:12],  # crash_norm
                ))
            surfs.append({
                'batch_sizes': batch_sizes,
                'total_verts': total_verts,
                'verts':       verts,
            })
        result[name] = surfs

    print(f'[PS2 crash.dat] Parsed {len(result)} nodes from {os.path.basename(filepath)}')
    return result


def build_blender_meshes_ps2(context, parser: 'PS2BGMParser', options: dict):
    """Build Blender mesh objects from a parsed PS2 BGM file.

    Creates the same FO2 Body / FO2 Body Dummies / FO2 Body Crash collection
    structure and uses the same FO2→Blender axis transform (Y↔Z swap) as
    build_blender_meshes().  Geometry is decoded from PS2 VIF packets.
    Crash meshes are imported from an auto-detected PS2 crash.dat when present.
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

    # ── crash.dat auto-detection (identical patterns to PC import) ───────────
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

    # crash_by_model: model_name -> list[surf_dict]
    crash_by_model = {}
    if import_crash and crash_dat_path:
        crash_nodes = parse_ps2_crash_dat(crash_dat_path)
        print(f'[PS2 BGM] Loaded crash data from: {crash_dat_path}')
        for node_name, surfs in crash_nodes.items():
            if node_name.endswith('_crash'):
                crash_by_model[node_name[:-6]] = surfs  # strip '_crash'

    # ── materials ────────────────────────────────────────────────────────────
    blender_materials = {}
    for i, bgm_mat in enumerate(parser.materials):
        bl_mat = create_blender_material(
            bgm_mat, bgm_dir, shared_dir, use_alpha,
            alpha_mode, transparency_overlap,
            auto_shared_dir, convert_dds,
            use_backface_culling,
            is_fouc=False,
            native_tex_ext='.tm2',
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
    root_empty['bgm_is_fouc'] = False
    root_empty['bgm_is_fo1']  = False
    root_empty['bgm_is_ps2']  = True
    root_empty['bgm_version'] = parser.version
    fo2_body_coll.objects.link(root_empty)

    # ── FO2 Body Crash collection + crash root empty (if crash data found) ──
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

    # ── object empties (dummies) — identical transform logic to PC import ────
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
        print('[PS2 BGM] Import complete: 0 mesh objects created (import_body=False)')
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

        mesh_matrix  = fo2_matrix_to_blender(bgm_mesh.matrix)
        crash_surfs  = crash_by_model.get(model.name)  # list of surf dicts, or None

        # ── regular body mesh ─────────────────────────────────────────────
        all_verts            = []
        all_normals          = []
        all_face_uvs         = []
        all_faces            = []
        all_face_mat_indices = []
        mat_index_map        = {}
        mesh_materials       = []

        # ── crash mesh accumulators ───────────────────────────────────────
        crash_all_verts            = []
        crash_all_normals          = []
        crash_all_face_uvs         = []
        crash_all_faces            = []
        crash_all_face_mat_indices = []
        crash_mat_index_map        = {}
        crash_mesh_materials       = []
        have_crash_data            = False

        for surf_local_idx, surf_id in enumerate(model.surface_ids):
            if surf_id < 0 or surf_id >= len(parser.ps2_surfaces):
                continue
            surf = parser.ps2_surfaces[surf_id]
            if surf.blob_size == 0:
                continue

            mat_id = surf.material_id
            if mat_id not in mat_index_map:
                mat_index_map[mat_id] = len(mesh_materials)
                mesh_materials.append(blender_materials.get(mat_id))
            local_mat_idx = mat_index_map[mat_id]

            # decode VIF batches once — used for both body and crash meshes
            batches = _ps2_extract_batches(parser.blob, surf)

            # ── body triangles ────────────────────────────────────────────
            for batch in batches:
                tris = _ps2_triangles_from_adc(
                    batch['pos'], batch['uv'], batch['norm'], batch['adc']
                )
                for tri in tris:
                    base       = len(all_verts)
                    face_uvs_t = []
                    for v_pos, v_uv, v_norm in tri:
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
                        face_uvs_t.append((v_uv[0], 1.0 - v_uv[1]))
                    all_faces.append((base, base + 1, base + 2))
                    all_face_mat_indices.append(local_mat_idx)
                    all_face_uvs.append(tuple(face_uvs_t))

            # ── crash triangles (if crash data exists for this surface) ───
            if crash_surfs is None or surf_local_idx >= len(crash_surfs):
                continue
            crash_surf = crash_surfs[surf_local_idx]

            # Verify batch sizes match so the vertex mapping is valid
            vif_batch_sizes   = [len(b['pos']) for b in batches]
            crash_batch_sizes = crash_surf['batch_sizes']
            if vif_batch_sizes != crash_batch_sizes:
                print(f'[PS2 crash.dat] WARNING: batch size mismatch for '
                      f'{model.name} surf {surf_local_idx} — skipping crash surface')
                continue

            crash_verts_flat = crash_surf['verts']  # (base_pos, crash_pos, base_norm, crash_norm)

            if mat_id not in crash_mat_index_map:
                crash_mat_index_map[mat_id] = len(crash_mesh_materials)
                crash_mesh_materials.append(blender_materials.get(mat_id))
            crash_local_mat_idx = crash_mat_index_map[mat_id]

            # walk batches, consuming crash_verts_flat in lockstep
            flat_offset = 0
            for batch in batches:
                n = len(batch['pos'])

                # build per-vertex crash pos/norm substituted into the batch
                crash_pos_batch  = []
                crash_norm_batch = []
                for vi in range(n):
                    _, c_pos, _, c_norm = crash_verts_flat[flat_offset + vi]
                    crash_pos_batch.append(c_pos)
                    crash_norm_batch.append(c_norm)
                flat_offset += n

                tris = _ps2_triangles_from_adc(
                    crash_pos_batch, batch['uv'], crash_norm_batch, batch['adc']
                )
                for tri in tris:
                    base       = len(crash_all_verts)
                    face_uvs_t = []
                    for v_pos, v_uv, v_norm in tri:
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
                        crash_all_verts.append(pos)
                        crash_all_normals.append(nrm)
                        face_uvs_t.append((v_uv[0], 1.0 - v_uv[1]))
                    crash_all_faces.append((base, base + 1, base + 2))
                    crash_all_face_mat_indices.append(crash_local_mat_idx)
                    crash_all_face_uvs.append(tuple(face_uvs_t))
                    have_crash_data = True

        # ── build Blender mesh helper (used for both body and crash) ──────
        def _build_bl_mesh(name, verts, normals, faces, face_mat_indices,
                           face_uvs, materials):
            bl_m = bpy.data.meshes.new(name)
            for bmat in materials:
                if bmat:
                    bl_m.materials.append(bmat)
            bl_m.vertices.add(len(verts))
            bl_m.loops.add(len(faces) * 3)
            bl_m.polygons.add(len(faces))
            flat_co = []
            for v in verts:
                flat_co.extend((v.x, v.y, v.z))
            bl_m.vertices.foreach_set('co', flat_co)
            lv = []
            for f in faces:
                lv.extend(f)
            bl_m.loops.foreach_set('vertex_index', lv)
            bl_m.polygons.foreach_set('loop_start', [i * 3 for i in range(len(faces))])
            bl_m.polygons.foreach_set('loop_total',  [3] * len(faces))
            if face_mat_indices:
                bl_m.polygons.foreach_set('material_index', face_mat_indices)
            bl_m.polygons.foreach_set('use_smooth', [True] * len(faces))
            if face_uvs:
                uv_layer = bl_m.uv_layers.new(name='UVMap')
                uv_data  = []
                for fu in face_uvs:
                    uv_data.extend(fu)
                for i, uv in enumerate(uv_data):
                    uv_layer.data[i].uv = uv
            bl_m.update()
            bl_m.validate()
            if normals:
                loop_norms = []
                for f in faces:
                    for vi in f:
                        loop_norms.append(normals[vi])
                try:
                    bl_m.normals_split_custom_set(loop_norms)
                except Exception:
                    try:
                        bl_m.use_auto_smooth = True
                        bl_m.normals_split_custom_set(loop_norms)
                    except Exception:
                        pass
            if validate_meshes:
                bl_m.validate(verbose=True)
            if clamp_size > 0:
                max_dim = max(bl_m.dimensions) if bl_m.dimensions else 0
                if max_dim > clamp_size:
                    sf = clamp_size / max_dim
                    for vert in bl_m.vertices:
                        vert.co *= sf
            return bl_m

        def _apply_obj_matrix(bl_obj, mesh_mat):
            """Apply FO2→Blender matrix to object (same row/col swap as PC import)."""
            M = mesh_mat
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

        # ── body mesh object ──────────────────────────────────────────────
        if not all_faces:
            continue

        mesh_name = bgm_mesh.name1 if bgm_mesh.name1 else 'bgm_ps2_mesh'
        bl_mesh   = _build_bl_mesh(mesh_name, all_verts, all_normals,
                                    all_faces, all_face_mat_indices,
                                    all_face_uvs, mesh_materials)
        bl_obj = bpy.data.objects.new(mesh_name, bl_mesh)
        fo2_body_coll.objects.link(bl_obj)
        created_objects.append(bl_obj)
        if use_origins:
            _apply_obj_matrix(bl_obj, mesh_matrix)
        bl_obj.parent        = root_empty
        bl_obj['bgm_flags']  = bgm_mesh.flags
        bl_obj['bgm_group']  = bgm_mesh.group
        bl_obj['bgm_name2']  = bgm_mesh.name2
        bl_obj['bgm_is_ps2'] = True

        # ── crash mesh object ─────────────────────────────────────────────
        if have_crash_data and crash_all_faces and fo2_crash_coll:
            crash_name    = mesh_name + '_crash'
            bl_crash_mesh = _build_bl_mesh(crash_name,
                                            crash_all_verts, crash_all_normals,
                                            crash_all_faces, crash_all_face_mat_indices,
                                            crash_all_face_uvs, crash_mesh_materials)
            crash_obj = bpy.data.objects.new(crash_name, bl_crash_mesh)
            fo2_crash_coll.objects.link(crash_obj)
            created_objects.append(crash_obj)
            crash_obj.parent = crash_root_empty
            if use_origins:
                _apply_obj_matrix(crash_obj, mesh_matrix)
            crash_obj['bgm_flags']    = bgm_mesh.flags
            crash_obj['bgm_group']    = bgm_mesh.group
            crash_obj['bgm_is_ps2']   = True
            crash_obj['bgm_is_crash'] = True
            print(f'[PS2 crash.dat] Created crash mesh: {crash_name} '
                  f'({len(crash_all_verts)} verts, {len(crash_all_faces)} faces)')

    bpy.ops.object.select_all(action='DESELECT')
    for obj in created_objects:
        obj.select_set(True)
    if created_objects:
        context.view_layer.objects.active = created_objects[0]

    print(f'[PS2 BGM] Import complete: {len(created_objects)} mesh objects created')
    return created_objects


