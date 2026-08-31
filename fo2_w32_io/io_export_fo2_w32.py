bl_info = {
    "name": "FlatOut 2 W32 Track Export",
    "author": "ravenDS",
    "version": (2, 0, 0),
    "blender": (3, 6, 0),
    "location": "File > Export > FlatOut 2 Track (.w32)",
    "description": "Export FlatOut 2 track geometry from Blender scene to W32",
    "category": "Import-Export",
}

import bpy
import bmesh
import struct
import os
import re
import math
import mathutils
from bpy_extras.io_utils import ExportHelper
from bpy.props import StringProperty, BoolProperty

# ── Vertex buffer flags (must match importer) ──
VERTEX_POSITION = 0x2
VERTEX_UV       = 0x100
VERTEX_UV2      = 0x200
VERTEX_NORMAL   = 0x10
VERTEX_COLOR    = 0x40


# ═══════════════════════════════════════════════════════════════════════════
# Binary helpers
# ═══════════════════════════════════════════════════════════════════════════
class BinaryWriter:
    def __init__(self):
        self._parts = []
    def write_raw(self, data): self._parts.append(data)
    def u32(self, v): self._parts.append(struct.pack('<I', v & 0xFFFFFFFF))
    def i32(self, v): self._parts.append(struct.pack('<i', v))
    def f32(self, v): self._parts.append(struct.pack('<f', v))
    def vec3f(self, v): self._parts.append(struct.pack('<3f', *v))
    def vec2f(self, v): self._parts.append(struct.pack('<2f', *v))
    def string(self, s): self._parts.append(s.encode('ascii', errors='replace') + b'\x00')
    def pack(self, fmt, *args): self._parts.append(struct.pack(fmt, *args))
    def write_to_file(self, filepath):
        with open(filepath, 'wb') as f:
            for p in self._parts:
                f.write(p)


class BinaryReader:
    def __init__(self, filepath):
        with open(filepath, 'rb') as f:
            self._data = f.read()
        self._pos = 0
    def tell(self):   return self._pos
    def seek(self, p): self._pos = p
    def u32(self):
        v = struct.unpack_from('<I', self._data, self._pos)[0]; self._pos += 4; return v
    def i32(self):
        v = struct.unpack_from('<i', self._data, self._pos)[0]; self._pos += 4; return v
    def f32(self):
        v = struct.unpack_from('<f', self._data, self._pos)[0]; self._pos += 4; return v
    def vec3f(self):
        v = struct.unpack_from('<3f', self._data, self._pos); self._pos += 12; return v
    def vec2f(self):
        v = struct.unpack_from('<2f', self._data, self._pos); self._pos += 8; return v
    def string(self):
        end = self._data.index(b'\x00', self._pos)
        s = self._data[self._pos:end].decode('ascii', errors='replace')
        self._pos = end + 1; return s
    def read(self, n):
        d = self._data[self._pos:self._pos + n]; self._pos += n; return d
    def raw(self, n):
        d = self._data[self._pos:self._pos + n]; self._pos += n; return d


# ═══════════════════════════════════════════════════════════════════════════
# Coordinate transforms: Blender (Z-up) <-> FO2 (Y-up)
# ═══════════════════════════════════════════════════════════════════════════
def blender_to_fo2_pos(co):
    """Blender (X, Y, Z) -> FO2 (X, Z, Y)"""
    return (co[0], co[2], co[1])

def blender_to_fo2_normal(n):
    return (n[0], n[2], n[1])

def _synthesize_veg_raw(mesh, preserved_raw):
    """Rebuild a vegetation surface's 28-byte-per-billboard buffer from the
    editable quad geometry so billboard placement/size is authorable.

    Billboards are 4 vertices each in the importer's layout (pos = sprite
    CENTER, sizes are HALF-extents: quad spans pos +/- (sw, sh), matching the
    engine's tree-LOD billboard convention):
        v0=(px-sw, py-sh, pz)  v1=(px+sw, py-sh, pz)
        v2=(px+sw, py+sh, pz)  v3=(px-sw, py+sh, pz)   (FO2 space)
    Reconstruction of center/width/height from those verts is exact. The
    per-billboard u32 at offset 20 is the atlas UV rect (u0,v0,u1,v1) packed as
    four bytes in sixteenths of the texture, and the u32 at offset 24 is the
    color LUT index. Both are carried verbatim from the preserved entry i (we
    don't re-encode the atlas rect from the Blender UVs, so the sub-sprite
    selection round-trips exactly); for billboards beyond the preserved count
    (newly added) we fall back to the last preserved entry's values, or zero.

    Returns the new raw bytes, or None if the mesh has no usable quad geometry
    (in which case the caller keeps preserved_raw).
    """
    verts = getattr(mesh, 'vertices', None)
    if not verts or len(verts) < 4 or (len(verts) % 4) != 0:
        return None
    n_bb = len(verts) // 4
    n_pre = len(preserved_raw) // 28
    out = bytearray()
    for i in range(n_bb):
        v0 = blender_to_fo2_pos(verts[i*4 + 0].co)
        v1 = blender_to_fo2_pos(verts[i*4 + 1].co)
        v3 = blender_to_fo2_pos(verts[i*4 + 3].co)
        hw_x0 = v0[0]; hw_x1 = v1[0]
        px = (hw_x0 + hw_x1) * 0.5
        py = (v0[1] + v3[1]) * 0.5      # entry pos is the quad CENTER
        pz = v0[2]
        sw = (hw_x1 - hw_x0) * 0.5      # half-width
        sh = (v3[1] - v0[1]) * 0.5      # half-height
        # atlas UV rect (offset 20) + color LUT index (offset 24) carried
        # verbatim from the preserved entry - the rect round-trips exactly
        src = i if i < n_pre else (n_pre - 1 if n_pre > 0 else -1)
        if src >= 0:
            orient = struct.unpack_from('<I', preserved_raw, src*28 + 20)[0]
            color = struct.unpack_from('<I', preserved_raw, src*28 + 24)[0]
        else:
            orient = 0
            color = 0
        out += struct.pack('<3f2fII', px, py, pz, sw, sh, orient, color)
    return bytes(out)


def _encode_plant_split(plane, axis, lmax, rmin):
    """plant_vdb kd-node: float32 plane with low 2 bits = axis (0=X,1=Y,2=Z;
    3 marks a leaf). Nudge inside [lmax, rmin] so masking keeps the invariant."""
    u = struct.unpack('<I', struct.pack('<f', float(plane)))[0]
    u = (u & 0xFFFFFFFC) | (axis & 3)
    dec = struct.unpack('<f', struct.pack('<I', u))[0]
    if lmax <= dec <= rmin:
        return u
    for step in range(1, 64):
        for cand in (u + 4 * step, u - 4 * step):
            dec = struct.unpack('<f', struct.pack('<I', cand & 0xFFFFFFFF))[0]
            if lmax <= dec <= rmin:
                return cand & 0xFFFFFFFF
    return u

def _build_plant_tree(centers):
    """Culling kd-tree over cluster (x, z) centers for plant_vdb's tail: heap
    array (children 2i+1/2i+2), leaves (idx<<5)|3, X/Z splits partitioning
    centers (verified invariant of every vanilla tree). Any invariant-
    satisfying tree is traversable; the vanilla bake heuristic isn't needed."""
    heap = {}
    AXCODE = (0, 2)
    def rec(idxs, pos):
        if len(idxs) == 1:
            heap[pos] = (idxs[0] << 5) | 3
            return
        best = None
        spreads = []
        for col in (0, 1):
            cs = sorted((centers[i][col], i) for i in idxs)
            spreads.append((cs[-1][0] - cs[0][0], col, cs))
        spreads.sort(reverse=True)
        for _, col, cs in spreads:
            k = len(cs) // 2
            for d in range(0, len(cs)):
                for kk in (k - d, k + d):
                    if 1 <= kk <= len(cs) - 1 and cs[kk - 1][0] < cs[kk][0]:
                        best = (col, cs, kk); break
                if best: break
            if best: break
        if best is None:
            col, cs, kk = 0, spreads[0][2], max(1, len(idxs) // 2)
            lmax = rmin = cs[0][0]
        else:
            col, cs, kk = best
            lmax, rmin = cs[kk - 1][0], cs[kk][0]
        heap[pos] = _encode_plant_split((lmax + rmin) / 2.0, AXCODE[col], lmax, rmin)
        rec([i for _, i in cs[:kk]], 2 * pos + 1)
        rec([i for _, i in cs[kk:]], 2 * pos + 2)
    rec(list(range(len(centers))), 0)
    if not heap:
        return []
    depth = max(p.bit_length() for p in (k + 1 for k in heap))
    arr = [0] * ((1 << depth) - 1)
    for k, v in heap.items():
        arr[k] = v
    return arr


def _rebuild_plant_geom(preserved_raw, plants_col):
    """Rebuild plant_geom.w32's instance array (B) and cluster table (C) from
    the editable plant-cluster meshes, keeping the header verbatim except that
    the global bounding box (offset 72) is EXPANDED if a plant was moved
    outside it. Positions are GLOBAL fractions of that bbox (NOT the per-cluster
    plant_vdb box, which is only a cull volume), so moving a plant just
    re-encodes its global fraction. Byte-identical to preserved_raw when
    nothing moved. Bit layout (a[0]/a[12]/a[28:32] stay 0):
        X = xmin + Fx/2047 *(xmax-xmin)   Fx = a[1:12]  (11-bit)
        Z = zmin + Fz/2047 *(zmax-zmin)   Fz = a[13:24] (11-bit)
        Y = ymin + Fy/65535*(ymax-ymin)   Fy = b[0:16]  (16-bit)
        4-bit variant = a[24:28] ; rotation/wind-phase = b[16:32]  (preserved)"""
    b = preserved_raw
    if len(b) < 4 or struct.unpack_from('<I', b, 0)[0] != 0x62647370:
        return preserved_raw
    # header = magic(4) + someCount(4) + d1[8] + d2[8] + bbox[6] = 4 + 4 + 88 = 96
    header = bytearray(b[:96])
    gbbox = list(struct.unpack_from('<6f', header, 72))   # xmin,xmax,ymin,ymax,zmin,zmax

    by_type = {}
    for obj in plants_col.objects:
        if obj.type != 'MESH':
            continue
        m = obj.data
        if 'fo2_plant_type_index' not in m:
            continue
        by_type[int(m['fo2_plant_type_index'])] = obj
    if not by_type:
        return preserved_raw
    ntypes = max(by_type) + 1

    # ---- pass 1: gather world positions + preserved meta per cluster ----
    clusters = []                                   # [(world[], meta[])]
    gxmn = gxmx = gymn = gymx = gzmn = gzmx = None
    for ti in range(ntypes):
        obj = by_type.get(ti)
        if obj is None:
            clusters.append(([], []))
            continue
        m = obj.data
        orig_ab = []
        if 'fo2_plant_ab_hex' in m:
            ob = bytes.fromhex(str(m['fo2_plant_ab_hex']))
            u = struct.unpack(f'<{len(ob)//4}I', ob)
            orig_ab = [(u[i], u[i + 1]) for i in range(0, len(u), 2)]
        mw = obj.matrix_world
        if m.get('fo2_plant_quads') and len(m.vertices) % 4 == 0:
            # textured-quad layout: 4 verts per instance (v0/v1 = bottom edge,
            # v2/v3 = top). The instance anchor is the bottom-edge midpoint.
            # Midpoint is taken in Blender space (the transforms are linear,
            # so this equals the midpoint of the transformed corners) and
            # reconstructs the imported anchor exactly: quantization to the
            # 11/16-bit grids absorbs any last-ulp float noise.
            world = []
            vs = m.vertices
            for i in range(0, len(vs), 4):
                c0 = vs[i].co; c1 = vs[i + 1].co
                mid = ((c0[0] + c1[0]) * 0.5, (c0[1] + c1[1]) * 0.5,
                       (c0[2] + c1[2]) * 0.5)
                world.append(blender_to_fo2_pos(mw @ mathutils.Vector(mid)))
        else:
            # legacy point-cloud layout: one vertex per instance
            world = [blender_to_fo2_pos(mw @ v.co) for v in m.vertices]
        meta = []
        for i in range(len(world)):
            a0, b0 = orig_ab[i] if i < len(orig_ab) else (0, 0)
            meta.append(((a0 >> 24) & 0xF, (b0 >> 16) & 0xFFFF))   # (variant4, rot)
        clusters.append((world, meta))
        for (x, y, z) in world:
            gxmn = x if gxmn is None else min(gxmn, x); gxmx = x if gxmx is None else max(gxmx, x)
            gymn = y if gymn is None else min(gymn, y); gymx = y if gymx is None else max(gymx, y)
            gzmn = z if gzmn is None else min(gzmn, z); gzmx = z if gzmx is None else max(gzmx, z)

    # ---- expand the global bbox only if a plant moved outside it ----
    if gxmn is not None:
        new = [min(gbbox[0], gxmn), max(gbbox[1], gxmx),
               min(gbbox[2], gymn), max(gbbox[3], gymx),
               min(gbbox[4], gzmn), max(gbbox[5], gzmx)]
        if new != gbbox:
            gbbox = new
            struct.pack_into('<6f', header, 72, *gbbox)
            print("[W32 Export] Scatter plants: global bbox expanded to bound "
                  "moved instances (all positions re-encoded)")
    xmin, xmax, ymin, ymax, zmin, zmax = gbbox

    # ---- pass 2: encode instances as global fractions; set per-cluster cull box ----
    def q(v, lo, hi, scale):
        if hi <= lo:
            return 0
        f = (v - lo) / (hi - lo)
        f = 0.0 if f < 0 else (1.0 if f > 1 else f)
        return int(round(f * scale))
    B = bytearray()
    C = []
    start = 0
    for ti in range(ntypes):
        world, meta = clusters[ti]
        cnt = len(world)
        for i, (x, y, z) in enumerate(world):
            v4, rot = meta[i]
            Fx = q(x, xmin, xmax, 2047.0) & 0x7FF
            Fz = q(z, zmin, zmax, 2047.0) & 0x7FF
            Fy = q(y, ymin, ymax, 65535.0) & 0xFFFF
            a = (Fx << 1) | (Fz << 13) | ((v4 & 0xF) << 24)
            bb = (Fy & 0xFFFF) | ((rot & 0xFFFF) << 16)
            B += struct.pack('<II', a, bb)
        C.append((cnt, start))
        start += cnt
        # per-cluster cull box (plant_vdb) = tight bbox of this cluster's plants.
        # Positions do NOT depend on it; it only keeps culling correct after a
        # move. Kept equal to the imported box when nothing moved (byte-exact).
        obj = by_type.get(ti)
        if obj is not None and cnt:
            xs=[p[0] for p in world]; ys=[p[1] for p in world]; zs=[p[2] for p in world]
            cx=(min(xs)+max(xs))/2; cy=(min(ys)+max(ys))/2; cz=(min(zs)+max(zs))/2
            ex=max((max(xs)-min(xs))/2,1e-3); ey=max((max(ys)-min(ys))/2,1e-3); ez=max((max(zs)-min(zs))/2,1e-3)
            orig = obj.data.get('fo2_plant_center'), obj.data.get('fo2_plant_extent')
            if orig[0] is not None and orig[1] is not None:
                oc=[float(v) for v in orig[0]]; oe=[float(v) for v in orig[1]]
                # if the imported box already bounds the (unmoved) plants, keep it
                if (min(xs)>=oc[0]-oe[0]-1e-4 and max(xs)<=oc[0]+oe[0]+1e-4 and
                    min(ys)>=oc[1]-oe[1]-1e-4 and max(ys)<=oc[1]+oe[1]+1e-4 and
                    min(zs)>=oc[2]-oe[2]-1e-4 and max(zs)<=oc[2]+oe[2]+1e-4):
                    cx,cy,cz=oc; ex,ey,ez=oe
            obj.data['fo2_plant_box_out'] = [cx, cy, cz, ex, ey, ez]

    out = bytearray(header)
    out += struct.pack('<I', start)          # cB
    out += bytes(B)
    out += struct.pack('<I', len(C))         # cC
    for cnt, st in C:
        out += struct.pack('<II', cnt, st)
    return bytes(out)


def blender_matrix_to_fo2(bl_matrix):
    """Blender Matrix -> FO2 row-major float[16]."""
    swap = mathutils.Matrix((
        (1, 0, 0, 0),
        (0, 0, 1, 0),
        (0, 1, 0, 0),
        (0, 0, 0, 1),
    ))
    raw_t = swap.inverted() @ bl_matrix @ swap
    m = [0.0] * 16
    for col in range(4):
        for row in range(4):
            m[row * 4 + col] = raw_t[col][row]
    return m


# ═══════════════════════════════════════════════════════════════════════════
# Vertex format helpers
# ═══════════════════════════════════════════════════════════════════════════
def compute_vertex_size(flags):
    """Compute vertex stride in bytes from flags."""
    size = 12  # position always
    if flags & VERTEX_NORMAL:  size += 12
    if flags & VERTEX_COLOR:   size += 4
    if (flags & VERTEX_UV) or (flags & VERTEX_UV2): size += 8
    if flags & VERTEX_UV2:     size += 8
    return size

def determine_flags_for_mesh(mesh):
    """Determine appropriate vertex flags for a Blender mesh.

    IMPORTANT (matches base.h): the game encodes UV set count in a SINGLE bit,
    not two independent bits:
      * exactly one UV set  -> VERTEX_UV  (0x100)      e.g. 0x142, 0x112
      * two UV sets         -> VERTEX_UV2 (0x200) ONLY e.g. 0x202, 0x212
    Two UV sets do NOT set VERTEX_UV as well. A surface with both 0x100 and
    0x200 set (0x3xx) is invalid: the renderer treats 0x200 as the lightmap/
    colormap marker, so a spurious 0x200 on an ordinary surface makes the game
    bind lightmap1_w2.dds in place of the diffuse texture. Emit exactly one of
    the two UV bits."""
    flags = VERTEX_POSITION | VERTEX_NORMAL
    n_uv = len(mesh.uv_layers) if mesh.uv_layers else 0
    has_color = (hasattr(mesh, 'color_attributes') and len(mesh.color_attributes) > 0) or \
                (hasattr(mesh, 'vertex_colors') and len(mesh.vertex_colors) > 0)
    if n_uv >= 2:
        flags |= VERTEX_UV2          # 0x200 only (two UV sets)
    elif n_uv == 1:
        flags |= VERTEX_UV           # 0x100 (one UV set)
    if has_color:
        flags |= VERTEX_COLOR
    return flags


# FO2 shaders expect an exact vertex layout
#   0x142 = pos + color + uv           (24 bytes)
#   0x202 = pos + uv1 + uv2            (28 bytes)
#   0x212 = pos + normal + uv1 + uv2   (40 bytes)
#   0x112 = pos + normal + uv          (32 bytes)
#   0x152 = pos + normal + color + uv  (36 bytes)
#   0x102 = pos + uv                   (20 bytes)
SHADER_VERTEX_FLAGS = {
    0:  0x142,  # static prelit
    1:  0x202,  # terrain (double UV)
    2:  0x212,  # terrain specular
    3:  0x112,  # dynamic diffuse
    4:  0x112,  # dynamic specular
    19: 0x142,  # tree trunk
    20: 0x142,  # tree branch
    34: 0x152,  # reflecting window (static)
    35: 0x112,  # reflecting window (dynamic)
    36: 0x152,  # static specular
    39: 0x102,  # static nonlit
    40: 0x102,  # static nonlit
}

def flags_for_shader(shader_id):
    """Vertex flags the game expects for a given shader id (fallback: static prelit)."""
    return SHADER_VERTEX_FLAGS.get(shader_id, 0x142)


_BLENDER_SUFFIX_RE = re.compile(r'\.\d{3}$')

def strip_blender_suffix(name):
    """Strip a trailing Blender duplicate suffix (.001, .002, ...) only.
    Unlike name.split('.')[0], this keeps legitimate dots in names."""
    return _BLENDER_SUFFIX_RE.sub('', name)


# ═══════════════════════════════════════════════════════════════════════════
# Ensure fo2_ custom properties exist (set defaults for user-created meshes)
# ═══════════════════════════════════════════════════════════════════════════
def ensure_surface_properties(obj, surface_index):
    """Assign default fo2_ properties to a mesh object if missing (user-created)."""
    mesh = obj.data
    if "fo2_surface_index" not in mesh:
        mesh["fo2_surface_index"] = surface_index
    # NOTE: fo2_flags is intentionally NOT defaulted here. Imported meshes carry
    # their original flags; user-created meshes get the layout the game expects
    # for their material's shader (see flags_for_shader) at extraction time.
    if "fo2_poly_mode" not in mesh:
        mesh["fo2_poly_mode"] = 4
    # Set batch index on the object
    if "fo2_batch_index" not in obj:
        # Determine index within parent collection
        parent_col = None
        for col in bpy.data.collections:
            if obj.name in col.objects:
                parent_col = col
                break
        if parent_col:
            mesh_objs = [o for o in parent_col.objects if o.type == 'MESH']
            try:
                obj["fo2_batch_index"] = mesh_objs.index(obj)
            except ValueError:
                obj["fo2_batch_index"] = 0
        else:
            obj["fo2_batch_index"] = 0

#  Material custom-property names follow the FlatOut Shader panel convention
#  (fo2_bgm_import addon): the panel edits bgm_shader_id / bgm_alpha /
#  bgm_use_colormap / bgm_v92 / bgm_v74 / bgm_v102 / bgm_texture(_0..2) ID
#  properties, so the exporter reads exactly those.
def ensure_material_properties(bl_mat):
    """Assign the FULL default panel (bgm_*) material property set if missing
    (user-created materials get every property an imported material carries,
    so the FlatOut Shader panel and the exporter see an identical shape)."""
    if not bl_mat:
        return
    if "bgm_shader_id" not in bl_mat:
        bl_mat["bgm_shader_id"] = 0
    if "bgm_alpha" not in bl_mat:
        bl_mat["bgm_alpha"] = 0
    if "bgm_use_colormap" not in bl_mat:
        bl_mat["bgm_use_colormap"] = 0
    # v92/v74/v102 are the only extra material fields vanilla data ever
    # populates (everything else is zero-filled); the shader panel edits them
    if "bgm_v92" not in bl_mat:
        bl_mat["bgm_v92"] = 0
    if "bgm_v74" not in bl_mat:
        bl_mat["bgm_v74"] = 0
    if "bgm_v102" not in bl_mat:
        bl_mat["bgm_v102"] = 0
    # Auto-detect texture name from node tree -> .tga extension
    if "bgm_texture_0" not in bl_mat:
        if bl_mat.use_nodes:
            for node in bl_mat.node_tree.nodes:
                if node.type == 'TEX_IMAGE' and node.image:
                    base = os.path.splitext(os.path.basename(node.image.filepath))[0]
                    if not base and node.image.name:
                        base = os.path.splitext(node.image.name)[0]
                    if base:
                        bl_mat["bgm_texture_0"] = base + ".tga"
                        break
        # If still no texture found, use material name as fallback
        if "bgm_texture_0" not in bl_mat:
            bl_mat["bgm_texture_0"] = ""
    for ti in (1, 2):
        if f"bgm_texture_{ti}" not in bl_mat:
            bl_mat[f"bgm_texture_{ti}"] = ""
    if "bgm_num_textures" not in bl_mat:
        bl_mat["bgm_num_textures"] = sum(
            1 for ti in range(3) if bl_mat.get(f"bgm_texture_{ti}"))
    if "bgm_texture" not in bl_mat:
        t0 = str(bl_mat.get("bgm_texture_0", ""))
        t1 = str(bl_mat.get("bgm_texture_1", ""))
        bl_mat["bgm_texture"] = t1 if (t0.lower() in ("colormap.tga", "colormap.dds")
                                       and t1) else t0


# ═══════════════════════════════════════════════════════════════════════════
# Geometry extraction: Blender mesh -> binary VB + IB
# ═══════════════════════════════════════════════════════════════════════════
def _mat_is_identity(m, eps=1e-7):
    """True if a 4x4 matrix (Blender or mock; indexable m[i][j]) is identity.
    Used to keep unedited exports byte-exact: only apply the object transform
    when it is actually non-trivial."""
    if m is None:
        return True
    try:
        for i in range(4):
            for j in range(4):
                if abs(m[i][j] - (1.0 if i == j else 0.0)) > eps:
                    return False
    except Exception:
        return True
    return True

def _xform_dir(m, n):
    """Transform a direction (normal) by the linear 3x3 part of a 4x4 matrix
    (no translation), then normalize. Correct for the rigid/uniform-scale
    transforms object moves use; good enough otherwise."""
    x, y, z = n[0], n[1], n[2]
    rx = m[0][0]*x + m[0][1]*y + m[0][2]*z
    ry = m[1][0]*x + m[1][1]*y + m[1][2]*z
    rz = m[2][0]*x + m[2][1]*y + m[2][2]*z
    l = (rx*rx + ry*ry + rz*rz) ** 0.5
    if l > 1e-12:
        rx, ry, rz = rx/l, ry/l, rz/l
    return (rx, ry, rz)


def extract_mesh_geometry(obj, apply_transform=True):
    """Extract triangulated geometry from a Blender mesh object.
    Returns (vb_data, ib_data, vertex_count, tri_count, flags, center, radius)
    or None on failure."""
    mesh = obj.data

    # Object transform: apply matrix_world so a grabbed/moved/rotated object
    # actually relocates its geometry on export (imported meshes carry
    # identity, so unedited scenes stay byte-exact). Gated on identity to
    # avoid perturbing preserved positions/normals when nothing moved.
    # apply_transform=False for COMPACT-MESH models: their geometry is a
    # shared model placed by a separate per-instance matrix (cm.matrix), so
    # baking matrix_world here would double-transform every instance.
    _mw = getattr(obj, 'matrix_world', None)
    _ident = (not apply_transform) or _mat_is_identity(_mw)

    # Parse flags: explicit fo2_flags override (imported meshes) takes priority,
    # otherwise use the layout the game requires for the material's shader.
    if "fo2_flags" in mesh:
        flags_str = mesh["fo2_flags"]
        if isinstance(flags_str, str):
            flags = int(flags_str, 0)
        else:
            flags = int(flags_str)
    else:
        bl_mat = mesh.materials[0] if mesh.materials else None
        if bl_mat is not None and "bgm_shader_id" in bl_mat:
            flags = flags_for_shader(int(bl_mat["bgm_shader_id"]))
        elif bl_mat is not None:
            # Material has no fo2 properties yet: defaults will assign shader 0
            flags = flags_for_shader(0)
        else:
            flags = determine_flags_for_mesh(mesh)

    # Normalize invalid UV bit combos. The game encodes UV-set count in ONE bit:
    # 0x100 = one UV set, 0x200 = two UV sets (and 0x200 doubles as the
    # lightmap/colormap marker). A surface with BOTH bits (0x3xx) is malformed -
    # the renderer sees the 0x200 marker and binds the lightmap over the diffuse.
    # This can slip in via a stale fo2_flags from an older import or a manual
    # edit, so clear 0x100 whenever 0x200 is present.
    if (flags & VERTEX_UV) and (flags & VERTEX_UV2):
        flags &= ~VERTEX_UV

    has_normal = bool(flags & VERTEX_NORMAL)
    has_color  = bool(flags & VERTEX_COLOR)
    has_uv     = bool(flags & VERTEX_UV) or bool(flags & VERTEX_UV2)
    has_uv2    = bool(flags & VERTEX_UV2)
    vertex_size = compute_vertex_size(flags)

    # Triangulate via bmesh
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    if hasattr(mesh, 'calc_normals_split'):
        mesh.calc_normals_split()
    if hasattr(mesh, 'calc_loop_triangles'):
        mesh.calc_loop_triangles()

    # Build corner normal lookup for Blender 4.1+/5.0
    corner_normals = None
    if hasattr(mesh, 'corner_normals') and len(mesh.corner_normals) > 0:
        corner_normals = mesh.corner_normals

    # Gather UV layers
    uv_layer = mesh.uv_layers[0] if (has_uv and mesh.uv_layers) else None
    uv2_layer = mesh.uv_layers[1] if (has_uv2 and len(mesh.uv_layers) >= 2) else None

    # Gather vertex color layer
    color_data = None
    if has_color:
        if hasattr(mesh, 'color_attributes') and mesh.color_attributes:
            color_data = mesh.color_attributes[0]
        elif hasattr(mesh, 'vertex_colors') and mesh.vertex_colors:
            color_data = mesh.vertex_colors[0]
    # A color layer that is (near-)uniformly white is NOT real prelit paint -
    # it's Blender's / an importer's default fill, and writing it verbatim
    # doubles scene brightness (prelit shaders modulate texture*vcolor*2, so
    # white = 2x). A mesh property fo2_ignore_vertex_colors=1
    # forces the neutral fallback regardless; =0 forces verbatim.
    ignore_colors = False
    force_flag = mesh.get('fo2_ignore_vertex_colors')
    if force_flag is not None:
        ignore_colors = bool(force_flag)
    elif color_data is not None and hasattr(color_data, 'data'):
        try:
            data = color_data.data
            n = len(data)
            if n:
                # sample up to 512 entries; treat as "no real paint" when
                # the layer is dominantly white (>=90%). Justified by vanilla:
                # across 6031 real prelit surfaces the near-white fraction is
                # 0.000 - genuine baked lighting never approaches white, so a
                # majority-white layer is always a default/import fill.
                step = max(1, n // 512)
                seen = 0
                white = 0
                for idx in range(0, n, step):
                    c = data[idx].color
                    seen += 1
                    if c[0] >= 0.98 and c[1] >= 0.98 and c[2] >= 0.98:
                        white += 1
                if seen and white / seen >= 0.90:
                    ignore_colors = True
        except Exception:
            ignore_colors = False
    if ignore_colors:
        color_data = None

    # Build per-loop vertices (unique combos of pos/normal/uv/color)
    # This is necessary because FO2 uses per-face-corner data
    loop_verts = []      # list of packed vertex bytes
    loop_to_idx = {}     # (vi, li) dedup key -> index
    indices = []         # triangle indices into loop_verts
    unique_positions = []

    for tri in mesh.loop_triangles:
        tri_indices = []
        for li in tri.loops:
            loop = mesh.loops[li]
            vi = loop.vertex_index
            co = mesh.vertices[vi].co
            if not _ident:
                co = _mw @ co

            # Build vertex data in FO2 format
            buf = bytearray()

            # Position (Blender -> FO2 coord swap)
            fx, fy, fz = blender_to_fo2_pos(co)
            buf += struct.pack('<3f', fx, fy, fz)

            # Normal
            if has_normal:
                n = None
                # Blender 4.1+/5.0: corner_normals
                if corner_normals is not None:
                    n = corner_normals[li].vector
                # Older Blender: loop.normal from calc_normals_split
                elif hasattr(loop, 'normal') and loop.normal.length_squared > 0:
                    n = loop.normal
                # Smooth: vertex normal
                elif hasattr(tri, 'use_smooth') and tri.use_smooth:
                    n = mesh.vertices[vi].normal
                # Flat: face normal
                if n is None:
                    n = tri.normal
                if not _ident:
                    n = _xform_dir(_mw, n)
                nx, ny, nz = blender_to_fo2_normal(n)
                buf += struct.pack('<3f', nx, ny, nz)

            # Vertex color, packed as D3DCOLOR (0xAARRGGBB) with alpha 0xFF -
            # the layout of every vanilla vertexcolors LUT entry. These
            # literals are rewritten into LUT indices after consolidation.
            if has_color:
                # Meshes with no painted colors get NEUTRAL GRAY 0x80, not
                # white: prelit shaders modulate the texture by vcolor*2
                # (128 = 1.0). Evidence: vanilla LUT stats (forest/city/
                # desert) - mean RGB 62-94, medians 49-92, maximum 155-204,
                # ZERO near-white vertices anywhere; white would render
                # everything at double brightness.
                r, g, b = 128, 128, 128
                if color_data:
                    if hasattr(color_data, 'data'):
                        if color_data.domain == 'POINT':
                            c = color_data.data[vi].color
                        else:
                            c = color_data.data[li].color
                        r = round(max(0, min(1, c[0])) * 255)
                        g = round(max(0, min(1, c[1])) * 255)
                        b = round(max(0, min(1, c[2])) * 255)
                buf += struct.pack('<I', b | (g << 8) | (r << 16) | (0xFF << 24))

            # UV (flip V back: Blender V -> FO2 V = 1-v)
            if has_uv:
                if uv_layer:
                    u, v = uv_layer.data[li].uv
                    buf += struct.pack('<2f', u, 1.0 - v)
                else:
                    buf += struct.pack('<2f', 0.0, 0.0)

            # UV2
            if has_uv2:
                if uv2_layer:
                    u2, v2 = uv2_layer.data[li].uv
                    buf += struct.pack('<2f', u2, 1.0 - v2)
                else:
                    buf += struct.pack('<2f', 0.0, 0.0)

            # Dedup by raw bytes
            key = bytes(buf)
            if key in loop_to_idx:
                idx = loop_to_idx[key]
            else:
                idx = len(loop_verts)
                loop_to_idx[key] = idx
                loop_verts.append(buf)
                unique_positions.append((fx, fy, fz))

            tri_indices.append(idx)

        # Reverse winding to match FO2 convention (importer does i2,i1,i0)
        indices.extend([tri_indices[2], tri_indices[1], tri_indices[0]])

    vertex_count = len(loop_verts)
    tri_count = len(mesh.loop_triangles)

    if vertex_count == 0 or tri_count == 0:
        return None

    # FO2 uses uint16 indices — must not exceed 65535
    if vertex_count > 65535:
        print(f"[W32 Export] WARNING: {obj.name} has {vertex_count} vertices "
              f"(max 65535 for FO2). Skipping mesh.")
        return None

    # Pack vertex buffer
    vb_data = b''.join(bytes(v) for v in loop_verts)

    # Pack index buffer
    ib_data = struct.pack(f'<{len(indices)}H', *indices)

    # Compute bounding box in FO2 space
    xs = [p[0] for p in unique_positions]
    ys = [p[1] for p in unique_positions]
    zs = [p[2] for p in unique_positions]
    bbox_min = (min(xs), min(ys), min(zs))
    bbox_max = (max(xs), max(ys), max(zs))
    center = tuple((bbox_min[i] + bbox_max[i]) * 0.5 for i in range(3))
    radius = tuple((bbox_max[i] - bbox_min[i]) * 0.5 for i in range(3))

    return {
        'vb_data': vb_data,
        'ib_data': ib_data,
        'vertex_count': vertex_count,
        'vertex_size': vertex_size,
        'tri_count': tri_count,
        'index_count': len(indices),
        'flags': flags,
        'center': center,
        'radius': radius,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Find root collection
# ═══════════════════════════════════════════════════════════════════════════
def find_root_collection(context):
    """Find the FO2 root collection in the scene."""
    for col in context.scene.collection.children:
        child_names = {c.name for c in col.children}
        if child_names & {'StaticBatch', 'TreeMesh', 'Objects', 'CompactMesh'}:
            return col
    # Fallback: use the first collection with children
    for col in context.scene.collection.children:
        if col.children:
            return col
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Material gathering
# ═══════════════════════════════════════════════════════════════════════════
def gather_materials(root_col):
    """Gather all unique Blender materials from meshes in the FO2 collections.
    Returns (material_list, name_to_index_map). Dedup is by material NAME -
    Blender hands back a fresh Python wrapper on every access, so wrapper
    objects are unreliable dict keys."""
    mat_set = {}  # name -> index
    mat_list = []

    def register_mat(bl_mat):
        if bl_mat is None:
            return -1
        # display-only helper materials (scatter-plant atlas, atmosphere sky/
        # cloud/skybox) must never enter the track's material table
        if bl_mat.get('fo2_plants_atlas') or bl_mat.get('fo2_display_only'):
            return -1
        if bl_mat.name in mat_set:
            return mat_set[bl_mat.name]
        idx = len(mat_list)
        mat_set[bl_mat.name] = idx
        mat_list.append(bl_mat)
        ensure_material_properties(bl_mat)
        return idx

    # Walk all mesh objects in all sub-collections
    def walk(col):
        for obj in col.objects:
            if obj.type == 'MESH' and obj.data.materials:
                for mat in obj.data.materials:
                    register_mat(mat)
        for child in col.children:
            walk(child)

    walk(root_col)
    return mat_list, mat_set


def build_ex_material_from_raw(raw_dict, idx):
    """Build an export material from a raw dict decoded from the root
    collection's fo2_all_materials_raw blob. Used for materials that no
    Blender mesh references but tree_meshes still depend on (LOD billboards).
    """
    m = ExMaterial()
    m.identifier = 0x4354414D  # "MATC"
    m.name = raw_dict['name'] or f"Material_{idx}"
    m.alpha = raw_dict['alpha']
    m.v92 = raw_dict['v92']
    m.num_textures = raw_dict['num_textures']
    m.shader_id = raw_dict['shader_id']
    m.use_colormap = raw_dict['use_colormap']
    m.v74 = raw_dict['v74']
    m.v108 = b'\x00' * 12
    m.v109 = b'\x00' * 12
    m.v98  = b'\x00' * 16
    m.v99  = b'\x00' * 16
    m.v100 = b'\x00' * 16
    m.v101 = b'\x00' * 16
    m.v102 = 0
    m.texture_names = list(raw_dict['texture_names'])
    return m


def decode_all_materials_raw(raw_bytes):
    """Decode the fo2_all_materials_raw blob into a list of dicts.

    DEFENSIVE: the blob comes from a Blender custom property and can be
    corrupt, truncated, or written by an older/newer importer with a
    different layout. Any inconsistency returns None (caller falls back to
    reconstructing materials from the Blender scene) - it must NEVER raise.
    """
    try:
        out = []
        off = 0
        n = len(raw_bytes)
        while off + 28 <= n:
            shader_id, alpha, use_colormap, v92, v74, num_textures, name_len = \
                struct.unpack_from('<7I', raw_bytes, off)
            # Sanity limits: any violation means the blob layout is not what
            # we expect (older importer version / corrupt property).
            if (shader_id > 10000 or alpha > 1000 or use_colormap > 1000 or
                    num_textures > 16 or name_len > 512):
                return None
            off += 28
            if off + name_len > n:
                return None
            name = raw_bytes[off:off+name_len].decode('utf-8', errors='replace')
            off += name_len
            texture_names = []
            for _ in range(3):
                if off + 4 > n:
                    return None
                tex_len, = struct.unpack_from('<I', raw_bytes, off)
                off += 4
                if tex_len > 512 or off + tex_len > n:
                    return None
                texture_names.append(
                    raw_bytes[off:off+tex_len].decode('utf-8', errors='replace'))
                off += tex_len
            out.append(dict(
                shader_id=shader_id, alpha=alpha, use_colormap=use_colormap,
                v92=v92, v74=v74, num_textures=num_textures,
                name=name, texture_names=texture_names,
            ))
        if off != n:
            # Trailing garbage: layout mismatch.
            return None
        return out if out else None
    except Exception:
        return None


def _load_sibling_module(name):
    """Import a converter shipped beside this plugin (tga2dds / dds2tga).
    Works both when packaged (``from . import name``) and as a loose script."""
    try:
        import importlib
        if __package__:
            return importlib.import_module('.' + name, __package__)
    except Exception:
        pass
    try:
        import importlib.util
        here = os.path.dirname(os.path.abspath(__file__))
        p = os.path.join(here, name + '.py')
        if os.path.isfile(p):
            spec = importlib.util.spec_from_file_location(name, p)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m
    except Exception as e:
        print(f"[W32 Export] could not load {name}: {e}")
    return None


def _texture_name_from_nodes(bl_mat):
    """Return the primary texture filename from a material's node tree, as a
    .tga name (mirrors fo2_bgm_export.get_texture_name_from_material). Prefers
    the image feeding Base Color; falls back to the first image node. Handles
    Windows paths, Blender's // relative prefix, and packed images."""
    if not getattr(bl_mat, 'use_nodes', False) or not bl_mat.node_tree:
        return ""
    nodes = bl_mat.node_tree.nodes

    def _name_from_image(img):
        if img is None:
            return ""
        fp = getattr(img, 'filepath', "") or ""
        if fp.startswith('//'):
            fp = fp[2:]
        base = os.path.basename(fp.replace('\\', '/')) if fp else ""
        if not base:
            base = getattr(img, 'name', "") or ""
        if not base:
            return ""
        stem, ext = os.path.splitext(base)
        # strip Blender's .001 duplicate suffix from image names
        if len(stem) > 4 and stem[-4] == '.' and stem[-3:].isdigit():
            stem = stem[:-4]
        return stem + ".tga"

    # 1) image feeding the Principled BSDF Base Color
    try:
        for n in nodes:
            if getattr(n, 'type', '') == 'BSDF_PRINCIPLED':
                bc = n.inputs.get('Base Color') if hasattr(n.inputs, 'get') else None
                if bc is not None and getattr(bc, 'links', None):
                    src = bc.links[0].from_node
                    # walk back through a mix/multiply if present
                    seen = 0
                    while src is not None and getattr(src, 'type', '') != 'TEX_IMAGE' and seen < 4:
                        nxt = None
                        for inp in getattr(src, 'inputs', []):
                            if getattr(inp, 'links', None):
                                nxt = inp.links[0].from_node; break
                        src = nxt; seen += 1
                    if src is not None and getattr(src, 'type', '') == 'TEX_IMAGE':
                        nm = _name_from_image(getattr(src, 'image', None))
                        if nm:
                            return nm
    except Exception:
        pass

    # 2) first image node anywhere
    for n in nodes:
        if getattr(n, 'type', '') == 'TEX_IMAGE' and getattr(n, 'image', None):
            nm = _name_from_image(n.image)
            if nm:
                return nm
    return ""


def build_ex_material(bl_mat, idx):
    """Build an export material from a Blender material."""
    m = ExMaterial()
    m.identifier = 0x4354414D  # "MATC"
    # Strip Blender's duplicate-name suffix (.001 etc) from the WRITTEN name
    # only; mat_map lookups use the unique Blender name.
    m.name = strip_blender_suffix(bl_mat.name) if bl_mat.name else f"Material_{idx}"
    m.alpha = int(bl_mat.get("bgm_alpha", 0))
    # v92/v74/v102 are the only fields besides the obvious ones that vanilla
    # tracks ever populate (checked across city1a/b/c + desert1a/b; all of
    # v98..v101/v108/v109 are zero everywhere). Round-trip them; they are
    # editable via the FlatOut Shader panel.
    m.v92 = int(bl_mat.get("bgm_v92", 0))
    m.num_textures = 0
    m.shader_id = int(bl_mat.get("bgm_shader_id", 0))
    m.use_colormap = int(bl_mat.get("bgm_use_colormap", 0))
    m.v74 = int(bl_mat.get("bgm_v74", 0))
    m.v108 = b'\x00' * 12
    m.v109 = b'\x00' * 12
    m.v98  = b'\x00' * 16
    m.v99  = b'\x00' * 16
    m.v100 = b'\x00' * 16
    m.v101 = b'\x00' * 16
    m.v102 = int(bl_mat.get("bgm_v102", 0))
    m.texture_names = ["", "", ""]
    for ti in range(3):
        key = f"bgm_texture_{ti}"
        if key in bl_mat:
            # the value stored INSIDE the w32 must carry the .tga extension
            # regardless of what image format the artist actually assigned
            m.texture_names[ti] = _force_tga(str(bl_mat[key]))
    # Colormap materials must keep the "colormap.tga" placeholder in slot 0.
    # The shader panel's texture field writes bgm_texture_0 on edit, so a
    # panel texture change on a terrain material lands in slot 0 - reroute it
    # to the detail slot (1) and restore the placeholder.
    if m.use_colormap and m.texture_names[0] and \
            m.texture_names[0].lower() not in ("colormap.tga", "colormap.dds"):
        m.texture_names[1] = m.texture_names[0]
        m.texture_names[0] = "colormap.tga"
    # Texture-swap detection (matches fo2_bgm_export): if the material has a
    # single image node, its image name is the source of truth for the primary
    # texture - so swapping the image in Blender swaps the exported texture.
    # Only override slot 0 for an UNAMBIGUOUS single-texture material; multi-
    # texture setups (e.g. terrain lightmap x detail) keep their stored slots.
    # Colormap materials are excluded entirely: their slot 0 is always the
    # "colormap.tga" placeholder and their node tree contains the resolved
    # lightmap image (lightmap*_w2), which must never leak into the file;
    # their texture edits flow through the panel reroute above instead.
    node_tex = _texture_name_from_nodes(bl_mat)
    if node_tex and not m.use_colormap and \
            not os.path.basename(node_tex).lower().startswith("lightmap"):
        img_nodes = 0
        if getattr(bl_mat, 'use_nodes', False) and bl_mat.node_tree:
            img_nodes = sum(1 for n in bl_mat.node_tree.nodes
                            if getattr(n, 'type', '') == 'TEX_IMAGE' and getattr(n, 'image', None))
        cur0 = os.path.splitext(os.path.basename(m.texture_names[0]))[0].lower()
        new0 = os.path.splitext(os.path.basename(node_tex))[0].lower()
        if not m.texture_names[0] or (img_nodes <= 1 and new0 != cur0):
            m.texture_names[0] = node_tex
    # Imported materials carry the original texture count; fall back to
    # counting non-empty names for user-created materials.
    if "bgm_num_textures" in bl_mat:
        m.num_textures = int(bl_mat["bgm_num_textures"])
    else:
        m.num_textures = sum(1 for t in m.texture_names if t)
    return m


# ═══════════════════════════════════════════════════════════════════════════
# Data structure classes
# ═══════════════════════════════════════════════════════════════════════════
class ExMaterial:
    __slots__ = (
        'identifier', 'name', 'alpha', 'v92', 'num_textures', 'shader_id',
        'use_colormap', 'v74', 'v108', 'v109',
        'v98', 'v99', 'v100', 'v101', 'v102',
        'texture_names',
    )

class ExVertexBuffer:
    __slots__ = ('id', 'is_vegetation', 'fouc_extra', 'vertex_count',
                 'vertex_size', 'flags', 'data')

class ExIndexBuffer:
    __slots__ = ('id', 'fouc_extra', 'index_count', 'data')

class ExSurface:
    __slots__ = (
        'is_vegetation', 'material_id', 'vertex_count', 'flags',
        'poly_count', 'poly_mode', 'num_indices_used',
        'center', 'radius',
        'num_streams', 'stream_ids', 'stream_offsets',
    )

class ExStaticBatch:
    __slots__ = ('id1', 'bvh_id1', 'bvh_id2', 'center', 'radius', 'unk_v1')

class ExTreeMesh:
    __slots__ = (
        'is_bush', 'unk2', 'bvh_id1', 'bvh_id2', 'matrix', 'scale',
        'trunk_surface_id', 'branch_surface_id', 'leaf_surface_id',
        'color_id', 'lod_id', 'material_id',
    )

class ExTreeLOD:
    __slots__ = ('pos', 'scale', 'values')

class ExModel:
    __slots__ = ('identifier', 'unk', 'name', 'center', 'radius', 'f_radius', 'surfaces')

class ExObject:
    __slots__ = ('identifier', 'name1', 'name2', 'flags', 'matrix')

class ExCollidableModel:
    __slots__ = ('models', 'center', 'radius')

class ExMeshDamageAssoc:
    __slots__ = ('name', 'ids')

class ExCompactMesh:
    __slots__ = ('identifier', 'name1', 'name2', 'flags', 'group', 'matrix',
                 'unk1', 'damage_assoc_id', 'models')

class ExBVHPrimitive:
    __slots__ = ('pos', 'radius', 'id1', 'id2')

class ExBVHNode:
    __slots__ = ('pos', 'radius', 'unk1', 'unk2')

class ExPlantEntry:
    __slots__ = ('pos', 'extent', 'surface_id', 'plant_id')

class ExW32Data:
    def __init__(self):
        self.version = 0x20001
        self.some_map_value = 1
        self.materials = []
        self.streams_order = []
        self.vertex_buffers = []
        self.index_buffers = []
        self.surfaces = []
        self.static_batches = []
        self.tree_colors = []
        self.tree_lods = []
        self.tree_meshes = []
        self.collision_offset_matrix = [
            1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1,
        ]
        self.models = []
        self.objects = []
        self.collidable_models = []
        self.mesh_damage_assoc = []
        self.compact_meshes = []
        self.compact_mesh_group_count = 0
        self.bvh_primitives = []
        self.bvh_nodes = []
        self.plants = []
        self.plant_vdb_header = b''
        self.plant_vdb_footer = b''
        self.effectmap_data = None
        self.resetmap_data = None
        self.vertex_colors_data = None
        self.vertex_colors_lut = []


# ═══════════════════════════════════════════════════════════════════════════
# Buffer consolidation: pack many surfaces into shared VB/IB by format
# ═══════════════════════════════════════════════════════════════════════════
class PendingSurface:
    """Geometry extracted from one Blender mesh, waiting for consolidation."""
    __slots__ = ('vb_data', 'ib_data', 'vertex_count', 'vertex_size',
                 'tri_count', 'index_count', 'flags', 'center', 'radius',
                 'material_id', 'poly_mode', 'is_vegetation')

def extract_pending_surface(obj, mat_map, apply_transform=True):
    """Extract geometry from a Blender mesh and return a PendingSurface."""
    geo = extract_mesh_geometry(obj, apply_transform)
    if geo is None:
        return None
    ps = PendingSurface()
    ps.vb_data = geo['vb_data']
    ps.ib_data = geo['ib_data']
    ps.vertex_count = geo['vertex_count']
    ps.vertex_size = geo['vertex_size']
    ps.tri_count = geo['tri_count']
    ps.index_count = geo['index_count']
    ps.flags = geo['flags']
    ps.center = geo['center']
    ps.radius = geo['radius']
    ps.is_vegetation = False

    bl_mat = obj.data.materials[0] if obj.data.materials else None
    if bl_mat is None:
        ps.material_id = 0
    else:
        # mat_map is keyed by material NAME (Blender material objects are not
        # reliable dict keys - a fresh wrapper per access made every lookup
        # miss, collapsing all surfaces onto material 0).
        mid = mat_map.get(bl_mat.name)
        if mid is None and 'fo2_material_index' in bl_mat:
            mid = int(bl_mat['fo2_material_index'])
        if mid is None:
            print(f"[W32 Export] WARNING: material '{bl_mat.name}' on object "
                  f"'{obj.name}' is not in the export material table - "
                  f"falling back to material 0")
            mid = 0
        ps.material_id = mid

    # Always 4 (triangle list). Writing pm=5 with a synthesized strip made
    # surfaces invisible in game, so the game does not consume strips built
    # from an arbitrary tri list even when they decode correctly on our side.
    # Keep pm=4 across the board until the pm=5 read path is understood.
    ps.poly_mode = 4
    return ps


def consolidate_buffers(pending_list, w):
    """Pack a list of PendingSurface into shared VBs/IBs grouped by vertex format.
    Creates ExVertexBuffer, ExIndexBuffer, ExSurface entries in w.

    IMPORTANT: surface records are created in pending order (surface index ==
    pending index). Vanilla tracks and guarantee that
    static batch i draws surface i; grouping by format must not reorder the
    surface array, only the buffer packing.

    Returns list of (surface_index, center, radius) in same order as pending_list."""

    # ── Pass 1: create one surface per pending, in order ──
    results = []
    for ps in pending_list:
        s = ExSurface()
        s.is_vegetation = 1 if ps.is_vegetation else 0
        s.material_id = ps.material_id
        s.vertex_count = ps.vertex_count
        s.flags = ps.flags
        s.poly_count = ps.tri_count
        s.poly_mode = ps.poly_mode
        s.num_indices_used = ps.index_count
        s.center = ps.center
        s.radius = ps.radius
        s.num_streams = 2
        s.stream_ids = [0, 0]        # filled in pass 2
        s.stream_offsets = [0, 0]    # filled in pass 2
        surf_idx = len(w.surfaces)
        w.surfaces.append(s)
        results.append((surf_idx, ps.center, ps.radius))

    # ── Pass 2: group by format key and pack shared VB/IB streams ──
    groups = {}  # key -> list of (surface_index, PendingSurface)
    for i, ps in enumerate(pending_list):
        key = (ps.vertex_size, ps.flags, ps.is_vegetation)
        if key not in groups:
            groups[key] = []
        groups[key].append((results[i][0], ps))

    # Respect the vanilla vertex-buffer cap. Every VB in every reference
    # track (city1a/b/c, desert1a/b, nascar1a/b) stays at or below 32,767
    # vertices; going higher makes the game misindex the surface — indices
    # >= 32,768 must be read as signed int16 somewhere in the pipeline and
    # end up negative, so the affected surfaces render garbage or vanish.
    # 65,535 (uint16 max) is unsafe.
    MAX_VERTS = 32767

    # Packs are collected first and stream ids assigned afterwards: every
    # vanilla track orders its streams as ALL vertex buffers, then the
    # vegetation buffer, then ALL index buffers - never interleaved.
    packs = []  # list of (vb, ib, [(surf_idx, vb_byte_offset, ib_byte_offset)])

    # Two caps observed across every vanilla track (city1a/b/c, desert1a/b,
    # nascar1a/b): each shared VB holds at most 32,767 vertices and each
    # shared IB holds at most 65,534 indices. Exceeding either produced
    # invisible / mislocated geometry in game. Split sub-groups by whichever
    # cap trips first.
    MAX_INDICES = 65534

    for (vs, flags, is_veg), entries in groups.items():
        sub_groups = []
        current_group = []
        current_verts = 0
        current_indices = 0
        for surf_idx, ps in entries:
            if current_group and (
                current_verts + ps.vertex_count > MAX_VERTS or
                current_indices + ps.index_count > MAX_INDICES
            ):
                sub_groups.append(current_group)
                current_group = []
                current_verts = 0
                current_indices = 0
            current_group.append((surf_idx, ps))
            current_verts += ps.vertex_count
            current_indices += ps.index_count
        if current_group:
            sub_groups.append(current_group)

        for sub in sub_groups:
            vb = ExVertexBuffer()
            vb.is_vegetation = is_veg
            vb.fouc_extra = 0
            vb.vertex_size = vs
            vb.flags = flags if not is_veg else 0

            ib = ExIndexBuffer()
            ib.fouc_extra = 0

            vb_parts = []
            ib_parts = []
            vb_byte_offset = 0
            ib_byte_offset = 0
            total_verts = 0
            total_indices = 0
            members = []

            for surf_idx, ps in sub:
                base_vertex = vb_byte_offset // vs

                # Adjust indices: add base_vertex offset (indices are absolute
                # into the shared VB)
                raw_indices = struct.unpack(f'<{ps.index_count}H', ps.ib_data)
                adjusted = struct.pack(f'<{ps.index_count}H',
                                       *(idx + base_vertex for idx in raw_indices))

                members.append((surf_idx, vb_byte_offset, ib_byte_offset))

                vb_parts.append(ps.vb_data)
                ib_parts.append(adjusted)
                vb_byte_offset += len(ps.vb_data)
                ib_byte_offset += len(adjusted)
                total_verts += ps.vertex_count
                total_indices += ps.index_count

            vb.data = b''.join(vb_parts)
            vb.vertex_count = total_verts
            ib.data = b''.join(ib_parts)
            ib.index_count = total_indices
            packs.append((vb, ib, members))

    # Assign stream ids: VBs occupy ids [0, npacks), IBs [npacks, 2*npacks),
    # matching the vanilla V...V I...I stream layout.
    npacks = len(packs)
    for pi, (vb, ib, members) in enumerate(packs):
        vb_stream_id = pi
        ib_stream_id = npacks + pi
        w.vertex_buffers.append(vb)
        for surf_idx, vb_off, ib_off in members:
            s = w.surfaces[surf_idx]
            s.stream_ids = [vb_stream_id, ib_stream_id]
            s.stream_offsets = [vb_off, ib_off]
    for pi, (vb, ib, members) in enumerate(packs):
        w.index_buffers.append(ib)
    w.streams_order = ([('vb', i) for i in range(npacks)] +
                       [('ib', i) for i in range(npacks)])

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Build W32 from Blender scene (FROM SCRATCH)
# ═══════════════════════════════════════════════════════════════════════════
def build_w32_from_scene(context, options):
    """Build a complete ExW32Data from the current Blender scene."""
    w = ExW32Data()

    root = find_root_collection(context)
    if not root:
        print("[W32 Export] ERROR: No FO2 root collection found")
        return None

    print(f"[W32 Export] Building from scene root: '{root.name}'")

    # Detect real tree metadata EARLY (before materials): when present we emit
    # real TreeMesh records, and tree_mesh.material_id references the vanilla-
    # only LOD materials (alpha_treelod / bushlod_*), so those materials must
    # be KEPT in the material table. When absent, they are unreferenced orphans
    # and get dropped (see material section below).
    have_tree_metadata = False
    _tree_col_early = None
    for c in root.children:
        if c.name == "TreeMesh":
            _tree_col_early = c; break
    if _tree_col_early is not None and (
            'fo2_tree_colors_raw_hex' in root or 'fo2_tree_lods_raw_hex' in root or
            'fo2_tree_colors_raw' in root or 'fo2_tree_lods_raw' in root):
        for subcol in _tree_col_early.children:
            for obj in subcol.objects:
                if obj.type == 'EMPTY' and 'fo2_tree_index' in obj:
                    have_tree_metadata = True
                    break
            if have_tree_metadata:
                break

    # Track collision offset matrix: every vanilla track carries a track-
    # specific Y-axis rotation (city1a/b/c ~31 deg, desert ~54 deg,
    # nascar ~87 deg). Round-trip the imported value verbatim from
    # fo2_collision_matrix_raw on the root collection. Only fall back to
    # from-scratch constant if no imported metadata exists (empty scene) or the property is malformed
    cm_raw = b''
    if 'fo2_collision_matrix_raw_hex' in root:
        try:
            cm_raw = bytes.fromhex(str(root['fo2_collision_matrix_raw_hex']))
        except Exception:
            cm_raw = b''
    elif 'fo2_collision_matrix_raw' in root:
        try:
            cm_raw = bytes(bytearray(int(v) & 0xFF for v in root['fo2_collision_matrix_raw']))
        except Exception:
            cm_raw = b''
    if len(cm_raw) == 64:
        w.collision_offset_matrix = list(struct.unpack('<16f', cm_raw))
    else:
        if len(cm_raw) > 0:
            print(f"[W32 Export] WARNING: preserved collision matrix is "
                  f"{len(cm_raw)} bytes (expected 64) - falling back to "
                  f"constant. Re-import the vanilla "
                  f"track to pick up the correct value.")
        w.collision_offset_matrix = [
            0.560662, 0.0, 0.828045, 0.0,
            0.0,      1.0, 0.0,      0.0,
            -0.828045, 0.0, 0.560662, 0.0,
            0.0,      0.0, 0.0,      1.0,
        ]

    # ── 1. Gather materials ──
    # Vanilla ordering must be preserved because tree_mesh.material_id (round-
    # tripped verbatim from the imported .w32) references the ORIGINAL index -
    # in city1b that means slot 185 == 'alpha_treelod', slot 188 ==
    # 'bushlod_city', and these materials have no mesh geometry (only trees
    # use them for LOD billboards). If we just gather materials from meshes,
    # those slots disappear, tree material_ids point at random other materials,
    # and the game crashes on load.
    #
    # Strategy: reconstruct the ORIGINAL material list index-for-index using
    # the fo2_all_materials_raw blob the importer saved on the root, then
    # overlay Blender materials at their fo2_material_index positions, then
    # append any brand-new Blender materials.
    mat_list_bl, _ = gather_materials(root)  # [bl_mat, ...] in walk order

    original_material_dicts = None
    raw = b''
    if 'fo2_all_materials_raw_hex' in root:
        try:
            raw = bytes.fromhex(str(root['fo2_all_materials_raw_hex']))
        except Exception:
            raw = b''
    elif 'fo2_all_materials_raw' in root:
        # Legacy int-array storage from earlier importer versions.
        try:
            raw = bytes(bytearray(int(v) & 0xFF for v in root['fo2_all_materials_raw']))
        except Exception:
            raw = b''
    if len(raw) > 0:
        original_material_dicts = decode_all_materials_raw(raw)
        if original_material_dicts is None:
            print(f"[W32 Export] WARNING: preserved material blob is corrupt or "
                  f"in an unknown layout ({len(raw)} bytes) - falling back to "
                  f"materials reconstructed from the Blender scene. Re-import "
                  f"with the current importer to restore exact vanilla ordering.")
        else:
            expected_count = int(root.get('fo2_all_materials_count',
                                          len(original_material_dicts)))
            if len(original_material_dicts) != expected_count:
                print(f"[W32 Export] WARNING: material blob decoded to "
                      f"{len(original_material_dicts)} entries, expected {expected_count}")
    elif 'fo2_all_materials_raw' in root or 'fo2_all_materials_raw_hex' in root:
        print(f"[W32 Export] WARNING: preserved material blob is empty - "
              f"materials will be gathered from meshes only; "
              f"other LOD materials may be lost.")

    # mat_map maps material NAME -> final index. Blender material objects are
    # unreliable dict keys across bpy accesses (each access can hand back a
    # fresh Python wrapper), which made every lookup miss and silently fall
    # back to material 0 - i.e. every surface in the exported file pointed at
    # the same material. Names are unique in bpy.data.materials, so they are
    # the correct key.
    mat_map = {}

    if original_material_dicts is not None:
        # Slots: at each original vanilla index, either the Blender material
        # (identified by fo2_material_index) or the raw dict fallback.
        n_orig = len(original_material_dicts)
        slot_bl_mat = [None] * n_orig
        for bl_mat in mat_list_bl:
            if 'fo2_material_index' in bl_mat:
                mi = int(bl_mat['fo2_material_index'])
                if 0 <= mi < n_orig:
                    slot_bl_mat[mi] = bl_mat

        # New Blender materials (no fo2_material_index) go at the tail.
        appended_new = [bl for bl in mat_list_bl
                        if 'fo2_material_index' not in bl]

        # Vanilla-only slots (slot_bl_mat[i] is None) are materials that no
        # imported mesh carries. In practice these are exactly the tree/bush
        # LOD billboard materials (shader 21, v74=2: alpha_treelod, bushlod_*)
        # that vanilla references ONLY from tree_mesh.material_id. This
        # workaround export emits no tree_mesh records, so emitting these
        # materials leaves them as orphans referenced by nothing - and the
        # game's vegetation/LOD init crashes on load when a tree-LOD material
        # has no matching tree_lods/tree_meshes instance data to build the
        # billboard buffer from. So we do NOT emit vanilla-only materials; we
        # emit only the Blender-carried ones (this reproduces the previous,
        # in-game-working exporter's material set) at DENSE indices. mat_map
        # is keyed by name and no exported surface references the dropped
        # materials, so compacting the indices is safe.
        #
        # NOTE: when real tree_mesh emission is implemented, this branch must
        # re-add the vanilla-only materials that the emitted trees reference
        # (via build_ex_material_from_raw) and point tree_mesh.material_id at
        # their new dense indices.
        if have_tree_metadata:
            # Real trees are emitted: tree_mesh.material_id references the
            # vanilla-only LOD materials by their ORIGINAL index, so emit the
            # full list index-for-index, filling empty slots from the raw blob.
            for i in range(n_orig):
                bl = slot_bl_mat[i]
                if bl is not None:
                    mat_map[bl.name] = i
                    w.materials.append(build_ex_material(bl, i))
                else:
                    w.materials.append(build_ex_material_from_raw(
                        original_material_dicts[i], i))
            for j, bl in enumerate(appended_new):
                mat_map[bl.name] = n_orig + j
                w.materials.append(build_ex_material(bl, n_orig + j))
            n_van = sum(1 for s in slot_bl_mat if s is None)
            print(f"[W32 Export] {len(w.materials)} materials "
                  f"({n_orig - n_van} from Blender + {n_van} preserved "
                  f"vanilla-only + {len(appended_new)} new; tree LOD materials "
                  f"kept for tree_mesh.material_id)")
        else:
            # No real trees: the vanilla-only slots are unreferenced orphans
            # (the tree/bush LOD materials). Drop them and index the survivors
            # densely - this reproduces the previous in-game-working material
            # set. mat_map is keyed by name and no exported surface references
            # the dropped materials, so compacting is safe.
            for i in range(n_orig):
                bl = slot_bl_mat[i]
                if bl is not None:
                    mat_map[bl.name] = len(w.materials)
                    w.materials.append(build_ex_material(bl, len(w.materials)))
            for bl in appended_new:
                mat_map[bl.name] = len(w.materials)
                w.materials.append(build_ex_material(bl, len(w.materials)))
            n_kept = sum(1 for s in slot_bl_mat if s is not None)
            n_dropped = sum(1 for s in slot_bl_mat if s is None)
            print(f"[W32 Export] {len(w.materials)} materials "
                  f"({n_kept} from Blender + {len(appended_new)} new; "
                  f"{n_dropped} vanilla-only LOD/orphan materials dropped - "
                  f"no tree_mesh records reference them)")
    else:
        # No (valid) preserved material blob. Reconstruct ordering from the
        # Blender materials' own fo2_material_index where available: slot each
        # indexed material at its original position, then append the rest.
        indexed = [bl for bl in mat_list_bl if 'fo2_material_index' in bl]
        unindexed = [bl for bl in mat_list_bl if 'fo2_material_index' not in bl]
        if indexed:
            n_orig = max(int(bl['fo2_material_index']) for bl in indexed) + 1
            n_orig = max(n_orig, int(root.get('fo2_all_materials_count', 0)))
            slot_bl_mat = [None] * n_orig
            for bl in indexed:
                mi = int(bl['fo2_material_index'])
                if 0 <= mi < n_orig:
                    slot_bl_mat[mi] = bl
            tail = list(unindexed)
            for i in range(n_orig):
                bl = slot_bl_mat[i]
                if bl is not None:
                    mat_map[bl.name] = i
                    w.materials.append(build_ex_material(bl, i))
                else:
                    # Gap: material existed in the vanilla file but no Blender
                    # mesh references it and the raw blob is unavailable.
                    # Emit a harmless placeholder so indices stay aligned.
                    m = ExMaterial()
                    m.identifier = 0x4354414D
                    m.name = f"Material_{i}"
                    m.alpha = 0; m.v92 = 0; m.num_textures = 0
                    m.shader_id = 0; m.use_colormap = 0; m.v74 = 0
                    m.v108 = b'\x00'*12; m.v109 = b'\x00'*12
                    m.v98 = b'\x00'*16; m.v99 = b'\x00'*16
                    m.v100 = b'\x00'*16; m.v101 = b'\x00'*16
                    m.v102 = 0
                    m.texture_names = ['', '', '']
                    w.materials.append(m)
            for j, bl in enumerate(tail):
                mat_map[bl.name] = n_orig + j
                w.materials.append(build_ex_material(bl, n_orig + j))
            print(f"[W32 Export] {len(w.materials)} materials (reconstructed "
                  f"from fo2_material_index; {sum(1 for s in slot_bl_mat if s is None)} "
                  f"placeholder gaps)")
        else:
            # Nothing to go on: emit in gather order.
            for i, bl_mat in enumerate(mat_list_bl):
                mat_map[bl_mat.name] = i
                w.materials.append(build_ex_material(bl_mat, i))
            print(f"[W32 Export] {len(w.materials)} materials (no preserved ordering)")

    # ── Collect ALL pending surfaces from every section ──
    # We'll consolidate them into shared VBs/IBs at the end.
    # Track which pending surface belongs to which section.

    all_pending = []          # list of PendingSurface
    pending_meta = []         # parallel list of metadata dicts

    def add_pending(obj, meta_dict, apply_transform=True):
        """Extract geometry and add to pending list. Returns pending index or -1."""
        # Capture the ORIGINAL surface index (imported meshes carry it) BEFORE
        # ensure_surface_properties defaults it - used to remap scatter-plant
        # surfaceIds to the new numbering on export.
        old_si = obj.data.get("fo2_surface_index", None)
        ensure_surface_properties(obj, len(all_pending))
        ps = extract_pending_surface(obj, mat_map, apply_transform)
        if ps is None:
            return -1
        idx = len(all_pending)
        all_pending.append(ps)
        meta_dict['old_surface_index'] = old_si
        pending_meta.append(meta_dict)
        return idx

    # ── 2. Static Batches ──
    static_col = None
    for c in root.children:
        if c.name == "StaticBatch":
            static_col = c; break

    batch_pending = []  # list of pending indices for static batches
    if static_col:
        mesh_objs = [o for o in static_col.all_objects if o.type == 'MESH']
        for obj in mesh_objs:
            # Vegetation billboards (imported leaf quads) are stored in
            # special 28-byte type-3 streams the game generates; they cannot
            # be written back as regular surfaces.
            if obj.data.get("fo2_vegetation", False):
                print(f"[W32 Export] Skipping vegetation mesh '{obj.name}' "
                      f"(vegetation buffers are not exportable)")
                continue
            pi = add_pending(obj, {'section': 'static_batch'})
            batch_pending.append(pi)

    # ── 3. Tree Meshes -> static batches ──
    # PREVIOUSLY-WORKING BEHAVIOR (restored): trunk and branch geometry is
    # exported as ordinary static batches; leaves (vegetation billboards) are
    # skipped. tree_meshes / tree_colors / tree_lods stay empty. This is the
    # version that produced working in-game exports (everything visible except
    # tree leaves). The real-TreeMesh + vegetation-VB emission that was added
    # later crashed the game and has been removed until it can be made correct.
    tree_col = None
    for c in root.children:
        if c.name == "TreeMesh":
            tree_col = c; break

    # have_tree_metadata was detected early (before the material section) so
    # the vanilla-only LOD materials it needs are kept. The real tree path
    # emits trunk/branch as tree surfaces (NOT static batches) and rebuilds
    # leaves from fo2_veg_raw; the fallback path (no metadata) converts
    # trunk/branch to static batches and skips leaves.
    pending_tree_records = []
    unique_leaf_meshes = []
    leaf_mesh_indices = {}

    tree_batch_count = 0
    if tree_col and not have_tree_metadata:
        # Fallback (no metadata, e.g. hand-built trees): keep the old behavior
        # of exporting trunk/branch as static batches, leaves skipped.
        for subcol in tree_col.children:
            for obj in subcol.objects:
                if obj.type != 'MESH':
                    continue
                if obj.data.get("fo2_vegetation", False) or 'leaf' in obj.name.lower():
                    continue  # vegetation billboards are not exportable
                pi = add_pending(obj, {'section': 'static_batch'})
                if pi >= 0:
                    batch_pending.append(pi)
                    tree_batch_count += 1
        if tree_batch_count:
            print(f"[W32 Export] No tree metadata: converted {tree_batch_count} "
                  f"tree trunk/branch meshes to static batches (leaves skipped)")

    if have_tree_metadata:  # REAL tree-metadata path
        print(f"[W32 Export] Tree metadata found - emitting real TreeMesh "
              f"records + vegetation buffer")
        for subcol in tree_col.children:
                tree_empty = None
                trunk_obj = None
                branch_obj = None
                leaf_mesh = None
                for obj in subcol.objects:
                    if obj.type == 'EMPTY' and 'fo2_tree_index' in obj:
                        tree_empty = obj
                    elif obj.type == 'MESH':
                        nm = obj.name.lower()
                        # Prefer explicit vegetation flag; fall back on name suffix.
                        is_leaf = obj.data.get('fo2_vegetation', False) or '_leaf' in nm
                        if is_leaf:
                            leaf_mesh = obj.data
                        elif '_trunk' in nm:
                            trunk_obj = obj
                        elif '_branch' in nm:
                            branch_obj = obj
                        else:
                            # Unlabelled mesh: guess by heuristic (has fo2_veg_raw = leaf,
                            # otherwise treat as trunk if we haven't seen one, else branch)
                            if 'fo2_veg_raw_hex' in obj.data or 'fo2_veg_raw' in obj.data:
                                leaf_mesh = obj.data
                            elif trunk_obj is None:
                                trunk_obj = obj
                            else:
                                branch_obj = obj

                if tree_empty is None:
                    continue  # skip malformed tree subcollection

                trunk_pi = add_pending(trunk_obj, {'section': 'tree'}) if trunk_obj else -1
                branch_pi = add_pending(branch_obj, {'section': 'tree'}) if branch_obj else -1

                if leaf_mesh is not None and ('fo2_veg_raw_hex' in leaf_mesh or 'fo2_veg_raw' in leaf_mesh):
                    key = id(leaf_mesh)
                    if key not in leaf_mesh_indices:
                        leaf_mesh_indices[key] = len(unique_leaf_meshes)
                        unique_leaf_meshes.append(leaf_mesh)
                pending_tree_records.append((tree_empty, trunk_pi, branch_pi, leaf_mesh))

    # ── 4. Compact Meshes ──
    props_col = None
    for c in root.children:
        if c.name == "CompactMesh":
            props_col = c; break

    compact_data = []      # list of (name, prop_empty, model_key, rep_child)
    model_geometry = {}    # model_key -> list of pending indices
    if props_col:
        # The collection is now flat (no per-prop sub-collections): prop empties
        # and their parented child meshes live directly under CompactMesh. Group
        # each empty with the meshes parented to it. Meshes with no prop-empty
        # parent (hand-added) are emitted as their own placeless instances so
        # they are not silently dropped.
        empties = [o for o in props_col.all_objects
                   if o.type == 'EMPTY' and ("fo2_type" in o or "fo2_flags" in o
                                             or "fo2_is_prop_empty" in o)]
        empty_set = set(empties)
        children_of = {}
        orphans = []
        for o in props_col.all_objects:
            if o.type != 'MESH':
                continue
            par = getattr(o, 'parent', None)
            if par is not None and par in empty_set:
                children_of.setdefault(par.name, []).append(o)
            else:
                orphans.append(o)

        instances = [(e.name, e, children_of.get(e.name, [])) for e in empties]
        for o in orphans:
            instances.append((o.name, None, [o]))

        for inst_name, prop_empty, mesh_objs in instances:
            if not mesh_objs:
                continue
            # Props placed multiple times share mesh datablocks (the importer
            # instances them). Extract each unique model ONCE - in LOCAL/model
            # space (apply_transform=False), because a compact mesh is a shared
            # model placed by its per-instance cm.matrix. Baking matrix_world
            # into the shared geometry would double-transform (and corrupt)
            # every instance that references the model.
            model_key = tuple(o.data.name for o in mesh_objs)
            if model_key not in model_geometry:
                mesh_pis = []
                for obj in mesh_objs:
                    pi = add_pending(obj, {'section': 'compact_mesh'},
                                     apply_transform=False)
                    mesh_pis.append(pi)
                model_geometry[model_key] = mesh_pis

            # Representative child carries this instance's placement: in Blender
            # a child's matrix_world = prop_empty.matrix_world @ its local
            # transform, so it reflects moving EITHER the prop empty OR the
            # mesh itself. cm.matrix is taken from it below.
            compact_data.append((inst_name, prop_empty, model_key, mesh_objs[0]))

    # ═══════════════════════════════════════════════════════════
    # CONSOLIDATE all pending surfaces into shared VBs/IBs
    # ═══════════════════════════════════════════════════════════
    print(f"[W32 Export] Consolidating {len(all_pending)} surfaces into shared buffers...")
    consolidation_results = consolidate_buffers(all_pending, w)

    print(f"[W32 Export] {len(w.vertex_buffers)} VBs, {len(w.index_buffers)} IBs, "
          f"{len(w.surfaces)} surfaces")

    # ── Now build higher-level structures using consolidated surface indices ──

    # Static Batches
    # Vanilla invariant (verified on city1a/b/c): batch.id1 == batch.bvh_id1 ==
    # batch index == surface index. bvh_id2 is the batch's BVH primitive slot;
    # slots must be unique across batches AND trees (trees continue after
    # batches below)
    for pi in batch_pending:
        if pi < 0 or consolidation_results[pi] is None:
            continue
        surf_idx, center, radius = consolidation_results[pi]
        b = ExStaticBatch()
        b.id1 = len(w.static_batches)
        b.bvh_id1 = surf_idx
        b.bvh_id2 = len(w.static_batches)
        b.center = center
        b.radius = radius
        b.unk_v1 = 0
        if b.bvh_id1 != b.id1:
            print(f"[W32 Export] WARNING: batch {b.id1} maps to surface "
                  f"{b.bvh_id1} (expected batch index == surface index)")
        w.static_batches.append(b)
    print(f"[W32 Export] {len(w.static_batches)} static batches")

    # ─────────────── Tree meshes + vegetation buffer ───────────────
    #
    # If the imported track had tree metadata we emit real TreeMesh records
    # + the vegetation VB, giving billboards back. Otherwise this block is
    # a no-op (trunk/branch were converted to static batches above; leaves
    # were skipped).
    if pending_tree_records and have_tree_metadata:
        # Restore tree_colors and tree_lods verbatim from the root collection.
        # Prefer the hex-string form; fall back to the legacy int-array form
        # for scenes imported before the hex migration (those may already be
        # corrupted by Blender, but we try anyway).
        def _blob(container, name):
            if name + "_hex" in container:
                return bytes.fromhex(str(container[name + "_hex"]))
            if name in container:
                return bytes(bytearray(int(v) & 0xFF for v in container[name]))
            return b''
        tc_raw = _blob(root, "fo2_tree_colors_raw")
        n_tc = len(tc_raw) // 4
        w.tree_colors = list(struct.unpack(f'<{n_tc}I', tc_raw))
        tl_raw = _blob(root, "fo2_tree_lods_raw")
        n_tl = len(tl_raw) // 28
        for i in range(n_tl):
            base = i * 28
            l = ExTreeLOD()
            l.pos = struct.unpack_from('<3f', tl_raw, base)
            l.scale = struct.unpack_from('<2f', tl_raw, base + 12)
            l.values = struct.unpack_from('<2I', tl_raw, base + 20)
            w.tree_lods.append(l)

        # Sort the unique leaf meshes by their original surface index so the
        # vegetation VB byte layout matches vanilla exactly (each surface's
        # entries appear at the same relative byte offset within the VB).
        # Meshes without a preserved surface index sort to the end.
        unique_leaf_meshes.sort(key=lambda m: int(m.get('fo2_surface_index', 1 << 30)))

        # Build the vegetation VB by concatenating each unique leaf mesh's
        # raw 28-byte-per-entry bytes. Assign one veg-surface index per
        # unique leaf mesh datablock; multiple tree_meshes referencing the
        # same datablock all share this surface (same as vanilla).
        veg_data = bytearray()
        veg_total_entries = 0
        veg_surface_index_per_mesh = {}  # id(mesh) -> surface_idx
        veg_surface_records = []  # (mesh, byte_offset, poly_count, material_id, flags)
        for mesh in unique_leaf_meshes:
            if "fo2_veg_raw_hex" in mesh:
                raw = bytes.fromhex(str(mesh["fo2_veg_raw_hex"]))
            elif "fo2_veg_raw" in mesh:
                # legacy int-array (may be corrupted by Blender - see importer)
                raw = bytes(bytearray(int(v) & 0xFF for v in mesh["fo2_veg_raw"]))
            else:
                raw = b''
            # Vegetation entries are exactly 28 bytes.
            if len(raw) % 28 != 0:
                raw = raw[:len(raw) - (len(raw) % 28)]

            # CUSTOM-GEOMETRY PATH: rebuild each billboard's center/width/height
            # DIRECTLY from the editable quad geometry so moving/resizing/adding/
            # deleting billboards in Blender is reflected in the export. Verified
            # to reconstruct vanilla pos/size with zero error. The per-billboard
            # orient/flags u32 (offset 20, still opaque) and color LUT index are
            # carried over from the preserved entry when present, else defaulted;
            # this keeps unedited round-trips byte-identical while making the
            # geometry authorable.
            synth = _synthesize_veg_raw(mesh, raw)
            if synth is not None:
                raw = synth

            poly_count = len(raw) // 28
            material_id = int(mesh.get("fo2_veg_material_id", 0))
            flags_str = mesh.get("fo2_veg_flags", "0x142")
            flags = int(flags_str, 0) if isinstance(flags_str, str) else int(flags_str)
            veg_surface_records.append((mesh, len(veg_data), poly_count, material_id, flags))
            veg_data.extend(raw)
            veg_total_entries += poly_count

        # Insert the veg VB into the streams. Regular VBs already occupy
        # stream ids [0, n_normal_vb); IBs occupy [n_normal_vb, n_normal_vb +
        # n_normal_ib). The veg VB slots between them at id n_normal_vb, so
        # every existing surface's IB stream_id must be shifted up by 1.
        n_normal_vb = len(w.vertex_buffers)
        n_normal_ib = len(w.index_buffers)
        for s in w.surfaces:
            if s.num_streams >= 2 and s.stream_ids[1] >= n_normal_vb:
                s.stream_ids[1] += 1

        veg_vb = ExVertexBuffer()
        veg_vb.is_vegetation = True  # emits as stream type 3 (no flags word)
        veg_vb.fouc_extra = 0
        veg_vb.vertex_size = 28
        veg_vb.flags = 0
        veg_vb.vertex_count = veg_total_entries
        veg_vb.data = bytes(veg_data)
        veg_vb_stream_id = n_normal_vb
        w.vertex_buffers.append(veg_vb)

        w.streams_order = (
            [('vb', i) for i in range(n_normal_vb)] +
            [('vb', n_normal_vb)] +
            [('ib', i) for i in range(n_normal_ib)]
        )

        # ── VANILLA SURFACE LAYOUT ORDER ──
        # Every vanilla track lays surfaces out as:
        #   [static-batch surfaces (0..n_batches-1)]
        #   [vegetation surfaces (n_batches..n_batches+n_veg-1)]
        #   [tree trunk/branch + prop-LOD surfaces (n_batches+n_veg..end)]
        # We must match this because:
        #   (a) batch.bvh_id1 == batch index == surface index for batches 0..n_batches-1
        #       stays valid (batches don't move),
        #   (b) tree.leaf_surface_id in vanilla lands in [n_batches, n_batches+n_veg),
        #       and although any in-range index would technically work, the game
        #       appears to rely on this layout (game crashes instantly with veg
        #       surfaces placed at the END of the array).
        # Insert veg surfaces at position n_batches and shift every existing
        # reference to surfaces at index >= n_batches up by n_veg.
        n_batches = len(w.static_batches)
        n_veg = len(veg_surface_records)

        # Emit veg surfaces as a separate list first
        veg_surfaces = []
        for mesh, byte_off, poly_count, material_id, flags in veg_surface_records:
            s = ExSurface()
            s.is_vegetation = 1
            s.material_id = material_id
            s.vertex_count = poly_count * 4  # nverts == 4 corners per billboard
            s.flags = flags
            s.poly_count = poly_count
            s.poly_mode = 0
            s.num_indices_used = 0
            s.center = (0.0, 0.0, 0.0)
            s.radius = (0.0, 0.0, 0.0)
            s.num_streams = 1
            s.stream_ids = [veg_vb_stream_id, 0]
            s.stream_offsets = [byte_off, 0]
            veg_surface_index_per_mesh[id(mesh)] = n_batches + len(veg_surfaces)
            veg_surfaces.append(s)

        # Splice veg surfaces into position [n_batches..n_batches+n_veg)
        w.surfaces = w.surfaces[:n_batches] + veg_surfaces + w.surfaces[n_batches:]

        # Shift every surface reference at index >= n_batches up by n_veg.
        # batch.bvh_id1 stays because all batches have bvh_id1 < n_batches.
        def _shift(idx):
            return idx + n_veg if idx >= n_batches else idx

        # Existing static batches: their bvh_id1 is always < n_batches (invariant).
        # We don't touch them - but we do touch consolidation_results, which
        # holds the (surface_index, material_id) that trunk/branch trees below
        # use to fill in trunk_surface_id / branch_surface_id.
        for i, r in enumerate(consolidation_results):
            if r is not None:
                sidx, center, radius = r
                consolidation_results[i] = (_shift(sidx), center, radius)

        # Model.surfaces reference surfaces too
        for mdl in w.models:
            mdl.surfaces = [_shift(s) for s in mdl.surfaces]

        # Anything already in w.tree_meshes (nothing yet - we emit below) or in
        # per-batch fields would need shifting; batch.bvh_id1 is unchanged and
        # nothing else in ExW32Data references surface indices at this point.

        # Emit TreeMesh records in the imported order (fo2_tree_index).
        # We sort by that so trees stay in the exact order they came from
        # the source file - relevant because tree_colors/tree_lods indices
        # were assigned against that order.
        pending_tree_records.sort(key=lambda t: int(t[0].get('fo2_tree_index', 0)))
        for tree_empty, trunk_pi, branch_pi, leaf_mesh in pending_tree_records:
            tm = ExTreeMesh()
            tm.is_bush = int(tree_empty.get('fo2_is_bush', 0))
            tm.unk2 = int(tree_empty.get('fo2_unk2', 0))
            # Preserve BVH id refs from import when present. The BVH generator
            # below will only override these if the imported id1 is out of
            # range (e.g. surface it referenced no longer exists). Vanilla's
            # tree.bvh_id1 often references a DIFFERENT tree's branch surface;
            # regenerating it from this tree's own branch produces a
            # semantically different value and the game may key on the vanilla
            # reference for LOD/culling proxies.
            tm.bvh_id1 = int(tree_empty.get('fo2_bvh_id1', 0))
            tm.bvh_id2 = int(tree_empty.get('fo2_bvh_id2', 0))
            tm.matrix = blender_matrix_to_fo2(tree_empty.matrix_world)
            sc = tree_empty.get('fo2_scale', [1.0, 1.0, 1.0])
            tm.scale = (float(sc[0]), float(sc[1]), float(sc[2]))
            tm.trunk_surface_id = (consolidation_results[trunk_pi][0]
                                   if trunk_pi >= 0 and consolidation_results[trunk_pi]
                                   else -1)
            tm.branch_surface_id = (consolidation_results[branch_pi][0]
                                    if branch_pi >= 0 and consolidation_results[branch_pi]
                                    else -1)
            if leaf_mesh is not None and id(leaf_mesh) in veg_surface_index_per_mesh:
                tm.leaf_surface_id = veg_surface_index_per_mesh[id(leaf_mesh)]
            else:
                tm.leaf_surface_id = -1
            tm.color_id = int(tree_empty.get('fo2_color_id', -1))
            tm.lod_id = int(tree_empty.get('fo2_lod_id', -1))
            tm.material_id = int(tree_empty.get('fo2_material_id', -1))
            w.tree_meshes.append(tm)
        print(f"[W32 Export] Emitted {len(w.tree_meshes)} tree meshes, "
              f"{len(veg_surface_records)} leaf veg surfaces "
              f"({veg_total_entries} billboards), "
              f"{len(w.tree_colors)} tree colors, {len(w.tree_lods)} tree LODs")

    # Vertex colors -> lighting LUT.
    # Every vanilla track stores LUT indices in the vertex color slot (high
    # byte 0x00/0x02/0x05 - across 450k+ sampled vertices in 5 vanilla tracks
    # there is not a single literal 0xFFxxxxxx color). The bank-0 indices
    # resolve against the per-lighting vertexcolors_w2.w32. Rewrite our
    # literal colors into sequential bank-0 indices and collect the LUT,
    # which export_w32 writes out alongside the track.
    #
    # Known limitation: vanilla surfaces with tinting from bank 2/5 (tree/veg
    # tail region of the LUT) get remapped to bank 0 here, losing their
    # tint. Full round-trip of bank-2/5 references requires the importer to
    # preserve the raw color dwords through Blender's mesh model - not done
    # yet. Affects a small number of surfaces per track (e.g. street lamps
    # in city1b) and produces visibly wrong tint, not missing geometry.
    lut = []
    for vb in w.vertex_buffers:
        if vb.is_vegetation or not (vb.flags & VERTEX_COLOR):
            continue
        stride = vb.vertex_size
        off = 12 + (12 if vb.flags & VERTEX_NORMAL else 0)
        data = bytearray(vb.data)
        for i in range(vb.vertex_count):
            p = i * stride + off
            color = struct.unpack_from('<I', data, p)[0]
            struct.pack_into('<I', data, p, len(lut))
            lut.append(color)
        vb.data = bytes(data)

    # Preserve the vanilla LUT tail region.
    # tree_colors, tree_lods.values[1], vegetation entries, and select
    # static-geometry tints all carry bank-2 (and occasionally bank-5)
    # references into indices FAR PAST our bank-0 slots (up to ~277k in
    # city1b). If we truncate the LUT at our bank-0 count, every bank-2/5
    # reference becomes out-of-range and the game crashes on load.
    #
    # Import saved the full imported LUT on the root collection as
    # fo2_vertex_colors_full_raw. If present, pad our LUT up to its length
    # by copying the tail region VERBATIM. If not present, fall back to the
    # overlay-onto-existing-file path in export_w32() (which requires the
    # vanilla vertexcolors_w2.w32 to be in the output folder).
    full_raw = b''
    if root is not None:
        if 'fo2_vertex_colors_full_raw_hex' in root:
            try:
                full_raw = bytes.fromhex(str(root['fo2_vertex_colors_full_raw_hex']))
            except Exception:
                full_raw = b''
        elif 'fo2_vertex_colors_full_raw' in root:
            try:
                full_raw = bytes(bytearray(
                    int(v) & 0xFF for v in root['fo2_vertex_colors_full_raw']))
            except Exception:
                full_raw = b''
    if len(full_raw) >= 4 and len(full_raw) % 4 == 0:
        full_lut = struct.unpack(f'<{len(full_raw)//4}I', full_raw)
        if len(full_lut) > len(lut):
            tail = list(full_lut[len(lut):])
            lut.extend(tail)
            print(f"[W32 Export] Extended LUT with {len(tail)} vanilla tail "
                  f"entries (indices {len(lut) - len(tail)}..{len(lut) - 1}) "
                  f"for bank-2/5 references from trees/vegetation")
    elif len(full_raw) > 0:
        print(f"[W32 Export] WARNING: preserved vertex-color LUT has "
              f"{len(full_raw)} bytes (not a multiple of 4) - LUT tail "
              f"NOT extended; bank-2/5 references may crash the game.")

    w.vertex_colors_lut = lut
    if lut:
        print(f"[W32 Export] LUT total: {len(lut)} entries "
              f"(bank-0 rewrite + preserved tail)")

    # ─────────────── Scatter plants ───────────────
    # Regenerate the three scatter-plant companion files from the preserved
    # import blobs. plant_geom.w32 and plantcolors_w2.w32 carry no track-surface
    # indices, so they ship verbatim. plant_vdb.gen's per-type surfaceId points
    # at a trunk/branch surface whose index changed under from-scratch export,
    # so we remap it: build old->new from every pending surface that carried an
    # original index (imported trunk/branch do), then patch each surfaceId in
    # place (offset 16 + i*32 + 24) leaving the rest of the file byte-exact.
    w.plant_geom_out = None
    w.plantcolors_out = None
    w.plant_vdb_out = None
    if 'fo2_plant_geom_raw_hex' in root:
        try:
            w.plant_geom_out = bytes.fromhex(str(root['fo2_plant_geom_raw_hex']))
        except Exception:
            w.plant_geom_out = None

    # If editable plant meshes are present, rebuild plant_geom's instance arrays
    # (B) and per-type table (C) from them, re-encoding each vertex position via
    # its type box. The header (someCount, d1, d2, bbox) is kept from the
    # preserved blob. Unedited scenes reproduce plant_geom byte-for-byte (decode
    # ->encode is exact); moving a vertex relocates that plant.
    # Rebuild plant_geom from the editable PlantBillboards meshes so plant
    # placement is fully authorable. Decode/encode of the B-entry is being
    # actively reverse-engineered against the exe instancer; unedited scenes
    # reproduce plant_geom byte-for-byte.
    plants_col = None
    for c in root.children:
        if c.name == "PlantBillboards":
            plants_col = c; break
    if plants_col is not None and w.plant_geom_out:
        w.plant_geom_out = _rebuild_plant_geom(w.plant_geom_out, plants_col)

    if 'fo2_plantcolors_raw_hex' in root:
        try:
            w.plantcolors_out = bytes.fromhex(str(root['fo2_plantcolors_raw_hex']))
        except Exception:
            w.plantcolors_out = None
    if 'fo2_plant_vdb_raw_hex' in root:
        try:
            pvdb = bytearray(bytes.fromhex(str(root['fo2_plant_vdb_raw_hex'])))
        except Exception:
            pvdb = None
        if pvdb and len(pvdb) >= 16 and struct.unpack_from('<I', pvdb, 0)[0] == 0x62647370:
            old_to_new = {}
            for i, meta in enumerate(pending_meta):
                old = meta.get('old_surface_index')
                if old is None:
                    continue
                if i < len(consolidation_results) and consolidation_results[i]:
                    old_to_new[int(old)] = consolidation_results[i][0]
            count = struct.unpack_from('<I', pvdb, 12)[0]
            remapped = missing = 0
            need = 16 + count * 32
            # per-type output box (center+extent) from the rebuilt plant meshes,
            # so plant_vdb agrees with how plant_geom positions were encoded.
            box_by_type = {}
            if plants_col is not None:
                for o in plants_col.objects:
                    if o.type == 'MESH' and 'fo2_plant_type_index' in o.data \
                            and 'fo2_plant_box_out' in o.data:
                        box_by_type[int(o.data['fo2_plant_type_index'])] = \
                            [float(v) for v in o.data['fo2_plant_box_out']]
            if len(pvdb) >= need:
                boxes_changed = False
                for i in range(count):
                    off = 16 + i * 32
                    old = struct.unpack_from('<I', pvdb, off + 24)[0]
                    if old in old_to_new:
                        struct.pack_into('<I', pvdb, off + 24, old_to_new[old])
                        remapped += 1
                    else:
                        missing += 1
                    if i in box_by_type:
                        before = bytes(pvdb[off:off + 24])
                        struct.pack_into('<6f', pvdb, off, *box_by_type[i])
                        if bytes(pvdb[off:off + 24]) != before:
                            boxes_changed = True
                if missing == 0 and boxes_changed:
                    # A cluster box moved, so the culling kd-tree in the tail
                    # (built over cluster centers) is stale - the game looks
                    # clusters up through it and a moved cluster would simply
                    # never render. Rebuild the tail: version, world min/max
                    # over all cluster boxes, and a fresh tree.
                    boxes = [struct.unpack_from('<6f', pvdb, 16 + i * 32)
                             for i in range(count)]
                    bmin = [min(d[k] - d[3 + k] for d in boxes) for k in range(3)]
                    bmax = [max(d[k] + d[3 + k] for d in boxes) for k in range(3)]
                    nodes = _build_plant_tree([(d[0], d[2]) for d in boxes])
                    tail = struct.pack('<I3f3fI', 1, *bmin, *bmax, len(nodes))
                    tail += struct.pack(f'<{len(nodes)}I', *nodes)
                    pvdb = pvdb[:16 + count * 32] + tail
                    print(f"[W32 Export] Scatter plants: cluster boxes changed, "
                          f"culling tree rebuilt ({len(nodes)} nodes)")
                if missing == 0:
                    w.plant_vdb_out = bytes(pvdb)
                    print(f"[W32 Export] Scatter plants: {count} types, "
                          f"all {remapped} surfaceIds remapped")
                else:
                    # Some plant types reference a surface our old->new map does
                    # not cover (not an imported trunk/branch/prop surface).
                    # Emitting them with stale surfaceIds would instance wrong
                    # geometry, so we fall back to NO plants for this track
                    # rather than risk a regression. plant_geom/plantcolors are
                    # also suppressed so nothing dangles.
                    w.plant_geom_out = None
                    w.plantcolors_out = None
                    w.plant_vdb_out = None
                    print(f"[W32 Export] Scatter plants: {remapped}/{count} "
                          f"surfaceIds remapped but {missing} could not be "
                          f"resolved - suppressing plants for this track to "
                          f"avoid instancing wrong geometry. (These types "
                          f"reference a surface not captured on import; needs "
                          f"investigation before plants can ship here.)")


    # Compact Meshes -> Models + CollidableModels + MeshDamageAssoc
    # One model (+ collidable + damage assoc) per UNIQUE model; every compact
    # mesh placement of the same prop references the shared damage assoc.
    built_models = {}  # model_key -> mda_idx (or None if unusable)
    for subcol_name, prop_empty, model_key, rep_child in compact_data:
        subcol_name = strip_blender_suffix(subcol_name)

        if model_key not in built_models:
            built_models[model_key] = None
            model = ExModel()
            model.identifier = 0x444F4D42  # "BMOD"
            model.unk = 4  # always 4 in vanilla tracks;
            model.name = subcol_name
            model.surfaces = []
            all_centers = []

            for pi in model_geometry[model_key]:
                if pi < 0 or consolidation_results[pi] is None:
                    continue
                surf_idx, center, radius = consolidation_results[pi]
                model.surfaces.append(surf_idx)
                all_centers.append((center, radius))

            if model.surfaces:
                all_min = [min(c[0][i] - c[1][i] for c in all_centers) for i in range(3)]
                all_max = [max(c[0][i] + c[1][i] for c in all_centers) for i in range(3)]
                model.center = tuple((all_min[i] + all_max[i]) * 0.5 for i in range(3))
                model.radius = tuple((all_max[i] - all_min[i]) * 0.5 for i in range(3))
                model.f_radius = math.sqrt(sum(r*r for r in model.radius))

                model_idx = len(w.models)
                w.models.append(model)

                col_model = ExCollidableModel()
                col_model.models = [model_idx]
                col_model.center = model.center
                col_model.radius = model.radius
                col_idx = len(w.collidable_models)
                w.collidable_models.append(col_model)

                mda = ExMeshDamageAssoc()
                # Vanilla mda name = compact_mesh name with any trailing
                # instance-number digits stripped. 'dyn_ad_huge_c_01' -> mda
                # 'dyn_ad_huge_c_'; 'dyn_lightpole_00' -> 'dyn_lightpole_';
                # names without trailing digits ('dyn_townsign') stay as-is.
                # Getting this wrong makes the mda section a few hundred bytes
                # bigger than vanilla and shifts every downstream offset - the
                # game may parse the file OK but references into later
                # sections (spvs, footer) end up at the wrong byte, causing
                # crashes during systems init.
                mda_name = subcol_name
                while mda_name and mda_name[-1].isdigit():
                    mda_name = mda_name[:-1]
                mda.name = mda_name
                mda.ids = (col_idx, -1)
                built_models[model_key] = len(w.mesh_damage_assoc)
                w.mesh_damage_assoc.append(mda)

        mda_idx = built_models[model_key]
        if mda_idx is None:
            continue

        cm = ExCompactMesh()
        cm.identifier = 0x4853454D  # "MESH"
        cm.name1 = subcol_name
        cm.name2 = prop_empty.get("fo2_type", subcol_name) if prop_empty else subcol_name
        # 0xE000 = visible in all three track variants
        # Imported props keep their original flags via fo2_flags.
        flags_str = prop_empty.get("fo2_flags", "0xE000") if prop_empty else "0xE000"
        cm.flags = int(flags_str, 0) if isinstance(flags_str, str) else int(flags_str)
        cm.group = int(prop_empty.get("fo2_group", -1)) if prop_empty else -1
        # Placement matrix. Prefer the representative child's world matrix,
        # which in Blender equals prop_empty.matrix_world @ (child's own
        # transform) - so it captures the move whether the user grabbed the
        # prop's placement empty OR the mesh piece itself. Fall back to the
        # empty (e.g. in headless harnesses where child world matrices are not
        # composited and stay identity).
        _cmat = getattr(rep_child, 'matrix_world', None) if rep_child else None
        if _cmat is not None and not _mat_is_identity(_cmat):
            cm.matrix = blender_matrix_to_fo2(_cmat)
        elif prop_empty is not None:
            cm.matrix = blender_matrix_to_fo2(prop_empty.matrix_world)
        else:
            cm.matrix = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
        cm.unk1 = 1  # always 1 in vanilla tracks
        cm.damage_assoc_id = mda_idx
        # cm.models is derived at load time via assoc -> collidable -> models
        # (not serialized for v0x20000+); resolve it the same way here.
        cm.models = list(w.collidable_models[w.mesh_damage_assoc[mda_idx].ids[0]].models)
        w.compact_meshes.append(cm)
        w.compact_mesh_group_count = max(w.compact_mesh_group_count, cm.group + 1)

    if w.compact_mesh_group_count <= 0:
        w.compact_mesh_group_count = 0
    print(f"[W32 Export] {len(w.compact_meshes)} compact meshes, {len(w.models)} models")

    # ── 5. Objects ──
    obj_col = None
    for c in root.children:
        if c.name == "Objects":
            obj_col = c; break

    if obj_col:
        seen_object_indices = set()
        for obj in obj_col.objects:
            # The importer instances prop geometry as several Blender objects
            # per FO2 object ("name_inst0", "name_inst1", ...), all sharing
            # fo2_object_index. Emit exactly one OBJC per original.
            if "fo2_object_index" in obj:
                oi = int(obj["fo2_object_index"])
                if oi in seen_object_indices:
                    continue
                seen_object_indices.add(oi)
            o = ExObject()
            o.identifier = 0x434A424F  # "OBJC"
            name1 = strip_blender_suffix(obj.name)
            name1 = re.sub(r'_inst\d+$', '', name1)
            o.name1 = name1
            # name2 is empty on every vanilla object; only round-trip an
            # explicitly imported value.
            o.name2 = obj.get("fo2_name2", "")
            # 0xE0F9 assign to newly created objects.
            flags_str = obj.get("fo2_flags", "0xE0F9")
            o.flags = int(flags_str, 0) if isinstance(flags_str, str) else int(flags_str)
            o.matrix = blender_matrix_to_fo2(obj.matrix_world)
            w.objects.append(o)
        print(f"[W32 Export] {len(w.objects)} objects")

    # ── 6. Plants ──
    plant_col = None
    for c in root.children:
        if c.name == "Plants":
            plant_col = c; break

    if plant_col:
        stale_ids = 0
        for obj in plant_col.objects:
            if obj.type != 'EMPTY': continue
            p = ExPlantEntry()
            loc = obj.location
            p.pos = blender_to_fo2_pos(loc)
            sc = obj.scale
            p.extent = (sc[0], sc[2], sc[1])
            p.surface_id = int(obj.get("fo2_surface_id", 0))
            p.plant_id = int(obj.get("fo2_plant_id", 0))
            # fo2_surface_id references the ORIGINAL file's surface order;
            # a from-scratch rebuild renumbers surfaces, so out-of-range ids
            # must not be written (they would make the game read OOB).
            if p.surface_id >= len(w.surfaces):
                p.surface_id = 0
                stale_ids += 1
            w.plants.append(p)
        if stale_ids:
            print(f"[W32 Export] WARNING: {stale_ids}/{len(w.plants)} plant "
                  f"clusters had stale surface ids (clamped to 0). Plant data "
                  f"only round-trips reliably in overlay mode.")
        print(f"[W32 Export] {len(w.plants)} plant clusters")

    # ── 7. BVH (always regenerated) ──
    # Vanilla track_bvh.gen structure (decoded from city1a/b/c, desert1a/b):
    #   - primitives: 32-byte AABBs (center + half-extents + id1 + id2),
    #     stored in leaf order; id2 == the primitive's own index; id1 == the
    #     surface/batch id. Each static batch stores its primitive slot in
    #     bvh_id2.
    #   - nodes: strict binary tree of 32-byte AABB records. Internal nodes
    #     have u2 == 0 and u1 == byte offset of the FIRST of two consecutive
    #     child nodes; leaves have u2 == prim count and u1 == byte offset
    #     into the prim array. Children are allocated as the next two free
    #     slots, left subtree serialized fully before the right (verified:
    #     traversal covers every vanilla node exactly once and leaf ranges
    #     partition the prim array contiguously in order).
    #   - vanilla leaves hold 1-3 prims (max seen: 17). A single giant leaf
    #     is far outside anything the renderer was built for.
    prim_in = []  # (owner, center, radius, "batch" | "tree")
    for b in w.static_batches:
        prim_in.append((b, b.center, b.radius, "batch"))
    # Every vanilla tree also occupies one BVH primitive slot. Use the branch
    # surface's bounds (its geometry envelope) if available, else the tree
    # matrix's translation with a small default radius. The bvh_id1 of a
    # tree primitive points at the branch or trunk surface id.
    for tm in w.tree_meshes:
        # Prefer branch bounds; fall back to matrix position.
        source_pi = None
        for pi in (tm.branch_surface_id, tm.trunk_surface_id, tm.leaf_surface_id):
            if 0 <= pi < len(w.surfaces):
                source_pi = pi; break
        if source_pi is not None:
            # Batches with matching bvh_id1 carry the bounds; if this surface
            # is a batch's target, reuse. Otherwise use bounds of (0,0,0) - only
            # applies to leaf surfaces which have no bounds themselves.
            src_center = (tm.matrix[12], tm.matrix[13], tm.matrix[14])
            src_radius = (10.0, 10.0, 10.0)
            for b in w.static_batches:
                if b.bvh_id1 == source_pi:
                    src_center = b.center
                    src_radius = b.radius
                    break
        else:
            src_center = (tm.matrix[12], tm.matrix[13], tm.matrix[14])
            src_radius = (10.0, 10.0, 10.0)
        prim_in.append((tm, src_center, src_radius, "tree"))

    nodes = []      # ExBVHNode, vanilla layout
    prims_out = []  # ExBVHPrimitive in leaf order

    def _aabb_of(items):
        mins = [min(c[i] - r[i] for _, c, r, _k in items) for i in range(3)]
        maxs = [max(c[i] + r[i] for _, c, r, _k in items) for i in range(3)]
        center = tuple((mins[i] + maxs[i]) * 0.5 for i in range(3))
        half = tuple((maxs[i] - mins[i]) * 0.5 for i in range(3))
        return center, half

    def _emit(items, my_idx):
        node = ExBVHNode()
        node.pos, node.radius = _aabb_of(items)
        # Leaf threshold. Vanilla BVHs stop splitting on spatial criteria and
        # routinely pack 8-46 primitives per leaf; no vanilla track_bvh.gen
        # exceeds ~2389 total nodes. The game probably has a fixed node budget around
        # 4096. Packing up to 8 prims/leaf keeps the tree balanced and well
        # under that budget while staying within the leaf sizes vanilla itself uses.
        if len(items) <= 8:
            node.unk1 = len(prims_out) * 32
            node.unk2 = len(items)
            for owner, c, r, kind in items:
                p = ExBVHPrimitive()
                # For a batch, id1 = batch.bvh_id1 (== surface index). For a
                # tree, prefer the imported tm.bvh_id1 if it's still a valid
                # surface index - vanilla often points bvh_id1 at ANOTHER
                # tree's branch surface (semantics unclear but consistent).
                # Only fall back to this tree's own branch/trunk/leaf when
                # the imported value would be out of range.
                if kind == "batch":
                    p.id1 = owner.bvh_id1
                else:
                    imported_id1 = owner.bvh_id1
                    if 0 <= imported_id1 < len(w.surfaces):
                        p.id1 = imported_id1
                    else:
                        for sid in (owner.branch_surface_id,
                                    owner.trunk_surface_id,
                                    owner.leaf_surface_id):
                            if 0 <= sid < len(w.surfaces):
                                p.id1 = sid
                                break
                        else:
                            p.id1 = 0
                p.id2 = len(prims_out)
                p.pos = c
                p.radius = r
                if kind == "batch":
                    owner.bvh_id2 = p.id2
                else:
                    owner.bvh_id1 = p.id1
                    owner.bvh_id2 = p.id2
                prims_out.append(p)
            nodes[my_idx] = node
            return
        # split at the median of the longest axis of the centroid bounds
        cmins = [min(c[i] for _, c, r, _k in items) for i in range(3)]
        cmaxs = [max(c[i] for _, c, r, _k in items) for i in range(3)]
        axis = max(range(3), key=lambda i: cmaxs[i] - cmins[i])
        items = sorted(items, key=lambda it: it[1][axis])
        mid = len(items) // 2
        li = len(nodes)
        nodes.append(None)
        nodes.append(None)
        node.unk1 = li * 32
        node.unk2 = 0
        nodes[my_idx] = node
        _emit(items[:mid], li)
        _emit(items[mid:], li + 1)

    if prim_in:
        import sys as _sys
        _old_limit = _sys.getrecursionlimit()
        _sys.setrecursionlimit(max(_old_limit, 10000))
        nodes.append(None)
        _emit(prim_in, 0)
        _sys.setrecursionlimit(_old_limit)

    w.bvh_primitives = prims_out
    w.bvh_nodes = nodes
    leaf_sizes = [n.unk2 for n in nodes if n.unk2 > 0]
    print(f"[W32 Export] Generated BVH: {len(prims_out)} primitives "
          f"({len(w.static_batches)} batches + {len(w.tree_meshes)} trees), "
          f"{len(nodes)} nodes, max leaf {max(leaf_sizes) if leaf_sizes else 0}")

    print(f"[W32 Export] FINAL: {len(w.surfaces)} surfaces, "
          f"{len(w.vertex_buffers)} VBs, {len(w.index_buffers)} IBs")
    return w


# ═══════════════════════════════════════════════════════════════════════════
# Re-parse original W32 (for overlay mode)
# ═══════════════════════════════════════════════════════════════════════════
def reparse_w32(filepath):
    """Re-parse a W32 file preserving ALL binary fields."""
    r = BinaryReader(filepath)
    w = ExW32Data()
    w.version = r.u32()
    if w.version == 0x20002:
        raise ValueError("FOUC W32 files (0x20002) are not supported: they use "
                         "int16 vertex buffers and per-surface multipliers this "
                         "parser does not handle")
    if w.version < 0x10004 or w.version > 0x20001:
        raise ValueError(f"Unsupported W32 version: 0x{w.version:X}")
    if w.version > 0x20000:
        w.some_map_value = r.u32()
        for _ in range(w.some_map_value - 1): r.u32()

    for i in range(r.u32()):
        m = ExMaterial()
        m.identifier = r.u32(); m.name = r.string()
        m.alpha = r.i32(); m.v92 = r.i32(); m.num_textures = r.i32()
        m.shader_id = r.i32(); m.use_colormap = r.i32(); m.v74 = r.i32()
        m.v108 = r.read(12); m.v109 = r.read(12)
        m.v98 = r.read(16); m.v99 = r.read(16); m.v100 = r.read(16)
        m.v101 = r.read(16); m.v102 = r.i32()
        m.texture_names = [r.string(), r.string(), r.string()]
        w.materials.append(m)

    num_streams = r.u32()
    for i in range(num_streams):
        dt = r.u32()
        if dt in (1, 3):
            vb = ExVertexBuffer(); vb.id = i; vb.is_vegetation = (dt == 3)
            vb.fouc_extra = r.u32(); vb.vertex_count = r.u32()
            vb.vertex_size = r.u32()
            vb.flags = 0 if dt == 3 else r.u32()
            vb.data = r.raw(vb.vertex_count * vb.vertex_size)
            w.vertex_buffers.append(vb)
            w.streams_order.append(('vb', len(w.vertex_buffers) - 1))
        elif dt == 2:
            ib = ExIndexBuffer(); ib.id = i; ib.fouc_extra = r.u32()
            ib.index_count = r.u32(); ib.data = r.raw(ib.index_count * 2)
            w.index_buffers.append(ib)
            w.streams_order.append(('ib', len(w.index_buffers) - 1))

    for i in range(r.u32()):
        s = ExSurface()
        s.is_vegetation = r.i32(); s.material_id = r.i32()
        s.vertex_count = r.i32(); s.flags = r.i32()
        s.poly_count = r.i32(); s.poly_mode = r.i32()
        s.num_indices_used = r.i32()
        s.center = r.vec3f() if w.version < 0x20000 else (0,0,0)
        s.radius = r.vec3f() if w.version < 0x20000 else (0,0,0)
        if w.version >= 0x20000 and i == 0: pass  # already set defaults
        s.num_streams = r.i32()
        s.stream_ids = [0, 0]; s.stream_offsets = [0, 0]
        for j in range(s.num_streams):
            s.stream_ids[j] = r.u32(); s.stream_offsets[j] = r.u32()
        w.surfaces.append(s)

    for i in range(r.u32()):
        b = ExStaticBatch(); b.id1 = r.u32(); b.bvh_id1 = r.u32(); b.bvh_id2 = r.u32()
        if w.version >= 0x20000:
            b.center = r.vec3f(); b.radius = r.vec3f(); b.unk_v1 = 0
        else:
            b.unk_v1 = r.u32(); b.center = (0,0,0); b.radius = (0,0,0)
        w.static_batches.append(b)

    tc = r.u32(); w.tree_colors = [r.u32() for _ in range(tc)]
    for _ in range(r.u32()):
        l = ExTreeLOD(); l.pos = r.vec3f(); l.scale = r.vec2f()
        l.values = (r.u32(), r.u32()); w.tree_lods.append(l)

    for _ in range(r.u32()):
        tm = ExTreeMesh(); tm.is_bush = r.i32(); tm.unk2 = r.i32()
        tm.bvh_id1 = r.i32(); tm.bvh_id2 = r.i32()
        tm.matrix = list(struct.unpack('<16f', r.read(64)))
        tm.scale = r.vec3f()
        tm.trunk_surface_id = r.i32(); tm.branch_surface_id = r.i32()
        tm.leaf_surface_id = r.i32(); tm.color_id = r.i32()
        tm.lod_id = r.i32(); tm.material_id = r.i32()
        w.tree_meshes.append(tm)

    if w.version >= 0x10004:
        w.collision_offset_matrix = list(struct.unpack('<16f', r.read(64)))

    for _ in range(r.u32()):
        m = ExModel(); m.identifier = r.u32(); m.unk = r.u32(); m.name = r.string()
        m.center = r.vec3f(); m.radius = r.vec3f(); m.f_radius = r.f32()
        ns = r.u32(); m.surfaces = [r.i32() for _ in range(ns)]
        w.models.append(m)

    for _ in range(r.u32()):
        o = ExObject(); o.identifier = r.u32(); o.name1 = r.string(); o.name2 = r.string()
        o.flags = r.u32(); o.matrix = list(struct.unpack('<16f', r.read(64)))
        w.objects.append(o)

    if w.version >= 0x20000:
        for _ in range(r.u32()):
            cm = ExCollidableModel(); mc2 = r.u32()
            cm.models = [r.u32() for _ in range(mc2)]
            cm.center = r.vec3f(); cm.radius = r.vec3f()
            w.collidable_models.append(cm)
        for _ in range(r.u32()):
            mda = ExMeshDamageAssoc(); mda.name = r.string()
            mda.ids = (r.i32(), r.i32())
            w.mesh_damage_assoc.append(mda)

    w.compact_mesh_group_count = r.u32()
    for _ in range(r.u32()):
        cm = ExCompactMesh(); cm.identifier = r.u32()
        cm.name1 = r.string(); cm.name2 = r.string()
        cm.flags = r.u32(); cm.group = r.i32()
        cm.matrix = list(struct.unpack('<16f', r.read(64)))
        if w.version >= 0x20000:
            cm.unk1 = r.u32(); cm.damage_assoc_id = r.u32(); cm.models = []
            if cm.damage_assoc_id < len(w.mesh_damage_assoc):
                assoc = w.mesh_damage_assoc[cm.damage_assoc_id]
                if assoc.ids[0] < len(w.collidable_models):
                    cm.models = list(w.collidable_models[assoc.ids[0]].models)
        else:
            cm.unk1 = 0; cm.damage_assoc_id = 0
            lc = r.u32(); cm.models = [r.u32() for _ in range(lc)]
        w.compact_meshes.append(cm)
    return w


# ═══════════════════════════════════════════════════════════════════════════
# W32 section writers
# ═══════════════════════════════════════════════════════════════════════════
def write_w32(w, filepath):
    bw = BinaryWriter()
    bw.u32(w.version)
    if w.version > 0x20000:
        # The parser reads ONE count word (== some_map_value) then skips
        # (some_map_value - 1) further words, so the file always carries at
        # least one header word here - even when some_map_value == 0 (derby4/
        # derby5/derby6 store a single 0 word). The old range(some_map_value)
        # loop wrote ZERO words when some_map_value == 0, dropping that
        # required word and shifting the entire file by 4 bytes -> guaranteed
        # crash on load. Write max(1, some_map_value) words to match.
        for _ in range(max(1, w.some_map_value)): bw.u32(w.some_map_value)

    bw.u32(len(w.materials))
    for m in w.materials:
        bw.u32(m.identifier); bw.string(m.name)
        bw.i32(m.alpha); bw.i32(m.v92); bw.i32(m.num_textures)
        bw.i32(m.shader_id); bw.i32(m.use_colormap); bw.i32(m.v74)
        bw.write_raw(m.v108); bw.write_raw(m.v109)
        bw.write_raw(m.v98); bw.write_raw(m.v99); bw.write_raw(m.v100)
        bw.write_raw(m.v101); bw.i32(m.v102)
        for i in range(3): bw.string(m.texture_names[i])

    bw.u32(len(w.vertex_buffers) + len(w.index_buffers))
    for st, idx in w.streams_order:
        if st == 'vb':
            vb = w.vertex_buffers[idx]
            bw.u32(3 if vb.is_vegetation else 1)
            bw.u32(vb.fouc_extra); bw.u32(vb.vertex_count); bw.u32(vb.vertex_size)
            if not vb.is_vegetation: bw.u32(vb.flags)
            bw.write_raw(vb.data)
        else:
            ib = w.index_buffers[idx]
            bw.u32(2); bw.u32(ib.fouc_extra); bw.u32(ib.index_count)
            bw.write_raw(ib.data)

    bw.u32(len(w.surfaces))
    for s in w.surfaces:
        bw.i32(s.is_vegetation); bw.i32(s.material_id); bw.i32(s.vertex_count)
        bw.i32(s.flags); bw.i32(s.poly_count); bw.i32(s.poly_mode)
        bw.i32(s.num_indices_used)
        if w.version < 0x20000:
            bw.vec3f(s.center); bw.vec3f(s.radius)
        bw.i32(s.num_streams)
        for j in range(s.num_streams):
            bw.u32(s.stream_ids[j]); bw.u32(s.stream_offsets[j])

    bw.u32(len(w.static_batches))
    for b in w.static_batches:
        bw.u32(b.id1); bw.u32(b.bvh_id1); bw.u32(b.bvh_id2)
        if w.version >= 0x20000:
            bw.vec3f(b.center); bw.vec3f(b.radius)
        else: bw.u32(b.unk_v1)

    bw.u32(len(w.tree_colors))
    for tc in w.tree_colors: bw.u32(tc)

    bw.u32(len(w.tree_lods))
    for l in w.tree_lods:
        bw.vec3f(l.pos); bw.vec2f(l.scale); bw.u32(l.values[0]); bw.u32(l.values[1])

    bw.u32(len(w.tree_meshes))
    for tm in w.tree_meshes:
        bw.i32(tm.is_bush); bw.i32(tm.unk2); bw.i32(tm.bvh_id1); bw.i32(tm.bvh_id2)
        bw.pack('<16f', *tm.matrix); bw.vec3f(tm.scale)
        bw.i32(tm.trunk_surface_id); bw.i32(tm.branch_surface_id)
        bw.i32(tm.leaf_surface_id); bw.i32(tm.color_id)
        bw.i32(tm.lod_id); bw.i32(tm.material_id)

    if w.version >= 0x10004:
        bw.pack('<16f', *w.collision_offset_matrix)

    bw.u32(len(w.models))
    for m in w.models:
        bw.u32(m.identifier); bw.u32(m.unk); bw.string(m.name)
        bw.vec3f(m.center); bw.vec3f(m.radius); bw.f32(m.f_radius)
        bw.u32(len(m.surfaces))
        for sid in m.surfaces: bw.i32(sid)

    bw.u32(len(w.objects))
    for o in w.objects:
        bw.u32(o.identifier); bw.string(o.name1); bw.string(o.name2)
        bw.u32(o.flags); bw.pack('<16f', *o.matrix)

    if w.version >= 0x20000:
        bw.u32(len(w.collidable_models))
        for cm in w.collidable_models:
            bw.u32(len(cm.models))
            for mid in cm.models: bw.u32(mid)
            bw.vec3f(cm.center); bw.vec3f(cm.radius)
        bw.u32(len(w.mesh_damage_assoc))
        for mda in w.mesh_damage_assoc:
            bw.string(mda.name); bw.i32(mda.ids[0]); bw.i32(mda.ids[1])

    bw.u32(w.compact_mesh_group_count)
    bw.u32(len(w.compact_meshes))
    for cm in w.compact_meshes:
        bw.u32(cm.identifier); bw.string(cm.name1); bw.string(cm.name2)
        bw.u32(cm.flags); bw.i32(cm.group); bw.pack('<16f', *cm.matrix)
        if w.version >= 0x20000:
            bw.u32(cm.unk1); bw.u32(cm.damage_assoc_id)
        else:
            bw.u32(len(cm.models))
            for mid in cm.models: bw.u32(mid)

    bw.write_to_file(filepath)
    print(f"[W32 Export] Wrote {filepath} ({os.path.getsize(filepath):,} bytes)")


# ═══════════════════════════════════════════════════════════════════════════
# Companion file writers
# ═══════════════════════════════════════════════════════════════════════════
def write_track_bvh(w, filepath):
    if not w.bvh_primitives and not w.bvh_nodes: return
    bw = BinaryWriter()
    bw.u32(0xDEADC0DE); bw.u32(1)
    bw.u32(len(w.bvh_primitives))
    for p in w.bvh_primitives:
        bw.vec3f(p.pos); bw.vec3f(p.radius); bw.i32(p.id1); bw.i32(p.id2)
    bw.u32(len(w.bvh_nodes))
    for n in w.bvh_nodes:
        bw.vec3f(n.pos); bw.vec3f(n.radius); bw.i32(n.unk1); bw.i32(n.unk2)
    bw.write_to_file(filepath)
    print(f"[W32 Export] Wrote {os.path.basename(filepath)}")

def write_plant_vdb(w, filepath, write_if_empty=False):
    # A from-scratch track needs a (possibly empty) plant_vdb.gen next to the w32
    if not w.plants and not write_if_empty: return
    bw = BinaryWriter()
    if w.plant_vdb_header:
        bw.write_raw(w.plant_vdb_header)
    else:
        bw.u32(0x62647370); bw.u32(1); bw.u32(0); bw.i32(len(w.plants))
    for p in w.plants:
        bw.vec3f(p.pos); bw.vec3f(p.extent); bw.u32(p.surface_id); bw.u32(p.plant_id)
    if w.plant_vdb_footer:
        bw.write_raw(w.plant_vdb_footer)
    else:
        # Minimal footer per plants.h: someData u32, float arrays[3],
        # float arrays2[3], trailing count u32 (0 entries follow).
        bw.u32(0)
        bw.vec3f((0, 0, 0)); bw.vec3f((0, 0, 0))
        bw.u32(0)
    bw.write_to_file(filepath)
    print(f"[W32 Export] Wrote {os.path.basename(filepath)}")


# ═══════════════════════════════════════════════════════════════════════════
# Overlay mode: collect transforms from Blender to update reparsed W32
# ═══════════════════════════════════════════════════════════════════════════
def overlay_transforms(context, w):
    """Update transforms in a reparsed W32 from the Blender scene."""
    root = find_root_collection(context)
    if not root: return

    for col in root.children:
        if col.name == "Objects":
            for obj in col.objects:
                if "fo2_object_index" in obj:
                    oi = obj["fo2_object_index"]
                    if 0 <= oi < len(w.objects):
                        w.objects[oi].matrix = blender_matrix_to_fo2(obj.matrix_world)
                elif "fo2_name2" in obj and obj.type == 'EMPTY':
                    for eo in w.objects:
                        if eo.name1 == obj.name.split('.')[0]:
                            eo.matrix = blender_matrix_to_fo2(obj.matrix_world); break

        elif col.name == "TreeMesh":
            for subcol in col.children:
                for obj in subcol.objects:
                    if obj.type == 'EMPTY' and obj.name.startswith("TreeMesh"):
                        try:
                            ti = int(obj.name.replace("TreeMesh", "").split('.')[0])
                            if 0 <= ti < len(w.tree_meshes):
                                w.tree_meshes[ti].matrix = blender_matrix_to_fo2(obj.matrix_world)
                        except ValueError: pass

        elif col.name == "CompactMesh":
            for subcol in col.children:
                for obj in subcol.objects:
                    if obj.type == 'EMPTY' and "fo2_type" in obj:
                        name = obj.name.split('.')[0]
                        for cm in w.compact_meshes:
                            if cm.name1 == name:
                                cm.matrix = blender_matrix_to_fo2(obj.matrix_world); break

        elif col.name == "Plants":
            for obj in col.objects:
                if obj.type != 'EMPTY': continue
                try:
                    pi = int(obj.name.replace("Plant", "").split('.')[0])
                except ValueError: continue
                if 0 <= pi < len(w.plants):
                    w.plants[pi].pos = blender_to_fo2_pos(obj.location)
                    sc = obj.scale
                    w.plants[pi].extent = (sc[0], sc[2], sc[1])

        elif col.name == "TrackBVH":
            for subcol in col.children:
                items = w.bvh_primitives if subcol.name == "BVH_Primitives" else \
                        w.bvh_nodes if subcol.name == "BVH_Nodes" else []
                prefix = "BVHPrim" if subcol.name == "BVH_Primitives" else "BVHNode"
                for obj in subcol.objects:
                    if not obj.name.startswith(prefix): continue
                    try: idx = int(obj.name.replace(prefix, "").split('.')[0])
                    except ValueError: continue
                    if 0 <= idx < len(items):
                        items[idx].pos = blender_to_fo2_pos(obj.location)
                        items[idx].radius = (obj.scale[0], obj.scale[2], obj.scale[1])


def reparse_plant_vdb(filepath, w):
    if not os.path.isfile(filepath): return False
    r = BinaryReader(filepath)
    magic = r.u32()
    if magic != 0x62647370: return False
    unk1 = r.u32(); unk2 = r.u32(); count = r.i32()
    w.plant_vdb_header = struct.pack('<4I', magic, unk1, unk2, count)
    for _ in range(count):
        p = ExPlantEntry()
        p.pos = r.vec3f(); p.extent = r.vec3f()
        p.surface_id = r.u32(); p.plant_id = r.u32()
        w.plants.append(p)
    remaining = len(r._data) - r._pos
    w.plant_vdb_footer = r.raw(remaining)
    return True

def reparse_track_bvh(filepath, w):
    if not os.path.isfile(filepath): return False
    r = BinaryReader(filepath)
    if r.u32() != 0xDEADC0DE: return False
    if r.u32() != 1: return False
    for _ in range(r.u32()):
        p = ExBVHPrimitive()
        p.pos = r.vec3f(); p.radius = r.vec3f()
        p.id1 = r.i32(); p.id2 = r.i32()
        w.bvh_primitives.append(p)
    for _ in range(r.u32()):
        n = ExBVHNode()
        n.pos = r.vec3f(); n.radius = r.vec3f()
        n.unk1 = r.i32(); n.unk2 = r.i32()
        w.bvh_nodes.append(n)
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Main export
# ═══════════════════════════════════════════════════════════════════════════
def export_w32(context, filepath, options):
    original_path = options.get('original_filepath', '')
    use_original = original_path and os.path.isfile(original_path)
    out_dir = os.path.dirname(filepath)

    if use_original:
        # ── OVERLAY MODE: re-parse original, overlay Blender transforms ──
        print(f"[W32 Export] Overlay mode: {original_path}")
        w = reparse_w32(original_path)
        orig_dir = os.path.dirname(original_path)
        orig_base = os.path.splitext(os.path.basename(original_path))[0]

        # Re-parse companion files
        for bp in [os.path.join(orig_dir, "track_bvh.gen"),
                    os.path.join(orig_dir, orig_base + "_bvh.gen")]:
            if reparse_track_bvh(bp, w): break
        for vp in [os.path.join(orig_dir, "plant_vdb.gen"),
                    os.path.join(orig_dir, orig_base + "_plant_vdb.gen")]:
            if reparse_plant_vdb(vp, w): break

        # Raw companion passthrough
        eff_path = os.path.join(orig_dir, "effectmap.4b")
        if os.path.isfile(eff_path):
            with open(eff_path, 'rb') as f: w.effectmap_data = f.read()

        rst_path = os.path.join(orig_dir, "resetmap.4b")
        if os.path.isfile(rst_path):
            with open(rst_path, 'rb') as f: w.resetmap_data = f.read()

        for vcp in [os.path.join(orig_dir, orig_base + "_vertexcolors.w32"),
                     os.path.join(orig_dir, orig_base + "_vertexcolors_w2.w32"),
                     os.path.join(orig_dir, "vertexcolors_w2.w32")]:
            if os.path.isfile(vcp):
                with open(vcp, 'rb') as f: w.vertex_colors_data = f.read()
                break

        overlay_transforms(context, w)

    else:
        # ── FROM-SCRATCH MODE: build W32 from Blender scene ──
        print("[W32 Export] From-scratch mode")
        w = build_w32_from_scene(context, options)
        if w is None:
            return {'CANCELLED'}

    # ── Write output files ──
    write_w32(w, filepath)

    if w.bvh_primitives or w.bvh_nodes:
        _root = find_root_collection(context)
        if _root is None or bool(_root.get("fo2_write_bvh", True)):
            write_track_bvh(w, os.path.join(out_dir, "track_bvh.gen"))
        else:
            print("[W32 Export] BVH generation disabled (fo2_write_bvh=0)")

    # From-scratch tracks always get a plant_vdb.gen (empty if no plants);
    # overlay mode only writes one when the original had one. When we have a
    # remapped plant_vdb from the preserved import, write that instead of empty.
    if getattr(w, 'plant_vdb_out', None):
        with open(os.path.join(out_dir, "plant_vdb.gen"), 'wb') as f:
            f.write(w.plant_vdb_out)
    else:
        write_plant_vdb(w, os.path.join(out_dir, "plant_vdb.gen"),
                        write_if_empty=not use_original)

    # Scatter-plant geometry + colors ship verbatim from import (no surface
    # indices inside them). Without these, plant_vdb types have nothing to
    # place, so plants never render.
    if getattr(w, 'plant_geom_out', None):
        with open(os.path.join(out_dir, "plant_geom.w32"), 'wb') as f:
            f.write(w.plant_geom_out)
    if getattr(w, 'plantcolors_out', None):
        with open(os.path.join(out_dir, "plantcolors_w2.w32"), 'wb') as f:
            f.write(w.plantcolors_out)

    # From-scratch mode: the w32's vertex colors are LUT indices, so the LUT
    # itself must ship with the track. The vanilla LUT also contains a tail
    # region (indices ~259k+ on city1b) referenced by tree colors, vegetation
    # and other lighting consumers - replacing it with a shorter file crashes
    # the game. If a vertexcolors_w2.w32 already exists next to the export
    # target (put the vanilla one there!), our colors are overlaid into its
    # first N entries and the tail is preserved; otherwise a standalone file
    # is written with a warning.
    lut = getattr(w, 'vertex_colors_lut', None)
    if not use_original and lut:
        vc_path = os.path.join(out_dir, "vertexcolors_w2.w32")
        packed = struct.pack(f'<{len(lut)}I', *lut)
        if os.path.isfile(vc_path):
            with open(vc_path, 'rb') as f:
                existing = bytearray(f.read())
            if len(existing) < len(packed):
                print(f"[W32 Export] WARNING: existing vertexcolors_w2.w32 has "
                      f"{len(existing)//4} entries but {len(lut)} are needed; "
                      f"extending it (tail consumers may break)")
                existing.extend(b'\x00' * (len(packed) - len(existing)))
            existing[:len(packed)] = packed
            with open(vc_path, 'wb') as f:
                f.write(existing)
            print(f"[W32 Export] Overlaid {len(lut)} vertex colors into existing "
                  f"vertexcolors_w2.w32 ({len(existing)//4} entries, tail preserved). "
                  f"Copy it over vertexcolors_w2.w32 in every lighting folder of "
                  f"the target track.")
        else:
            with open(vc_path, 'wb') as f:
                f.write(packed)
            print(f"[W32 Export] WARNING: wrote standalone vertexcolors_w2.w32 "
                  f"({len(lut)} entries). The vanilla LUT's tail region (tree/"
                  f"vegetation colors referenced by other track files) is NOT "
                  f"included - place the vanilla vertexcolors_w2.w32 in the "
                  f"export folder before exporting so it can be preserved.")

    if use_original:
        if w.effectmap_data:
            with open(os.path.join(out_dir, "effectmap.4b"), 'wb') as f:
                f.write(w.effectmap_data)
        if w.resetmap_data:
            with open(os.path.join(out_dir, "resetmap.4b"), 'wb') as f:
                f.write(w.resetmap_data)
        bed_src = os.path.join(os.path.dirname(original_path), "resetmap.bed")
        bed_dst = os.path.join(out_dir, "resetmap.bed")
        if os.path.isfile(bed_src) and bed_src != bed_dst:
            import shutil; shutil.copy2(bed_src, bed_dst)
        if w.vertex_colors_data:
            with open(os.path.join(out_dir, "vertexcolors_w2.w32"), 'wb') as f:
                f.write(w.vertex_colors_data)

    # Optionally convert referenced TGA/PNG textures to game-ready DDS
    # (same detection as the W32 organizer: DXT3 for alpha, DXT1 otherwise).
    if options.get('convert_textures_to_dds', False):
        mats = []
        seenm = set()
        try:
            for obj in context.scene.collection.all_objects:
                if obj.type == 'MESH':
                    for m in obj.data.materials:
                        if m is not None and m.name not in seenm:
                            seenm.add(m.name)
                            mats.append(m)
        except Exception:
            pass
        done, skip, missing, failed = _convert_material_textures_to_dds(mats)
        print(f"[W32 Export] Texture conversion: {done} TGA/PNG -> DDS "
              f"({skip} already had DDS, {len(missing)} not found, "
              f"{len(failed)} failed)")

    print("[W32 Export] Export complete!")
    return {'FINISHED'}


# ═══════════════════════════════════════════════════════════════════════════
# Blender operator
# ═══════════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════════════
# W32: Organize & set properties (Object menu operator for custom geometry)
# ═════════════════════════════════════════════════════════════════════════════
def _force_tga(name):
    """Force a texture name to carry the .tga extension the game expects.
    Strips Blender .001 suffixes and swaps any image extension for .tga;
    a bare name gets .tga appended. Empty stays empty."""
    if not name:
        return name
    name = strip_blender_suffix(os.path.basename(name))
    base, ext = os.path.splitext(name)
    if ext.lower() in ('.tga', '.dds', '.png', '.bmp', '.jpg', '.jpeg', '.tif'):
        return base + '.tga'
    return name + '.tga'


def _apply_material_conventions(bl_mat):
    """Apply the standard custom-track material conventions:
      - default shader 0 (static prelit) for anything untagged
      - material/texture names starting with 'alpha' turn alpha on
      - texture slot 0 == 'colormap.tga' marks a lightmapped terrain
        material: use_colormap=1, shader 0 -> 1, and an empty detail slot 1
        is filled with '<material name>.tga'
      - an empty slot 0 is filled with '<material name>.tga'
      - tree shaders (19/20/21) don't draw outside TreeMesh - remap to 0
      - every stored texture name gets the .tga extension
    Returns a short description of what changed (or '')."""
    changed = []
    ensure_material_properties(bl_mat)
    mat_name = strip_blender_suffix(bl_mat.name)
    sid = int(bl_mat.get("bgm_shader_id", 0))
    if sid in (19, 20, 21):
        bl_mat["bgm_shader_id"] = 0
        sid = 0
        changed.append("tree shader -> 0")
    tex = [str(bl_mat.get(f"bgm_texture_{i}", "")) for i in range(3)]
    if not tex[0]:
        # material names may themselves carry an image extension - _force_tga
        # strips it 
        tex[0] = _force_tga(mat_name)
        changed.append("tex0 from name")
    tex = [_force_tga(t) for t in tex]
    if tex[0].lower() == "colormap.tga":
        if not bl_mat.get("bgm_use_colormap"):
            bl_mat["bgm_use_colormap"] = 1
            changed.append("use_colormap")
        if sid == 0:
            bl_mat["bgm_shader_id"] = 1
            changed.append("shader -> terrain")
        if not tex[1]:
            tex[1] = _force_tga(mat_name)
            changed.append("detail from name")
        if int(bl_mat.get("bgm_num_textures", 0)) < 2:
            bl_mat["bgm_num_textures"] = 2
    lname = mat_name.lower()
    if (lname.startswith("alpha") or tex[0].lower().startswith("alpha")) \
            and not bl_mat.get("bgm_alpha"):
        bl_mat["bgm_alpha"] = 1
        changed.append("alpha on")
    for i in range(3):
        if str(bl_mat.get(f"bgm_texture_{i}", "")) != tex[i]:
            bl_mat[f"bgm_texture_{i}"] = tex[i]
            if "tga ext" not in changed and i < 2:
                changed.append("tga ext")
    bl_mat["bgm_num_textures"] = sum(1 for t in tex if t)
    bl_mat["bgm_texture"] = tex[1] if tex[0].lower() == "colormap.tga" and tex[1] else tex[0]
    # Sync the FlatOut Shader panel's RNA properties (registered by the
    # fo2_bgm_import addon) so the panel displays the material correctly
    # right away. Order matters: the enum's update callback rewrites
    # bgm_shader_id / bgm_texture_0 and may force bgm_alpha, so the ID
    # properties set above must win - rewrite them after the sync.
    _sid = int(bl_mat.get("bgm_shader_id", 0))
    _alpha = int(bl_mat.get("bgm_alpha", 0))
    _disp = str(bl_mat.get("bgm_texture", ""))
    try:
        bl_mat.fo2_shader_id = str(_sid)
    except Exception:
        pass
    try:
        bl_mat.fo2_texture = _disp
    except Exception:
        pass
    bl_mat["bgm_shader_id"] = _sid
    bl_mat["bgm_alpha"] = _alpha
    for i in range(3):
        bl_mat[f"bgm_texture_{i}"] = tex[i]
    bl_mat["bgm_texture"] = _disp
    return ", ".join(changed)


# ── BVH preview (organize-time visualization) ──────────────────────────────
# Mirrors the export-time generator in build_w32_from_scene (median-split
# binary tree, <=8 prims/leaf, internal nodes u2==0/u1=child byte offset,
# leaves u1=prim byte offset/u2=count). KEEP THE TWO IN SYNC: the exporter's
# copy is the in-game-validated authority; this one only draws the preview.
def _build_bvh_preview_tree(items):
    """items: list of (center_fo2, radius_fo2, id1). Returns
    (prims, nodes) where prims are (pos, radius, id1, id2) and nodes are
    (pos, radius, unk1, unk2)."""
    prims = []
    nodes = []

    def _aabb(its):
        mins = [min(c[i] - r[i] for c, r, _ in its) for i in range(3)]
        maxs = [max(c[i] + r[i] for c, r, _ in its) for i in range(3)]
        return (tuple((mins[i] + maxs[i]) * 0.5 for i in range(3)),
                tuple((maxs[i] - mins[i]) * 0.5 for i in range(3)))

    def _emit(its, my_idx):
        pos, radius = _aabb(its)
        if len(its) <= 8:
            nodes[my_idx] = (pos, radius, len(prims) * 32, len(its))
            for c, r, id1 in its:
                prims.append((c, r, id1, len(prims)))
            return
        cmins = [min(c[i] for c, r, _ in its) for i in range(3)]
        cmaxs = [max(c[i] for c, r, _ in its) for i in range(3)]
        axis = max(range(3), key=lambda i: cmaxs[i] - cmins[i])
        its = sorted(its, key=lambda it: it[0][axis])
        mid = len(its) // 2
        li = len(nodes)
        nodes.append(None)
        nodes.append(None)
        nodes[my_idx] = (pos, radius, li * 32, 0)
        _emit(its[:mid], li)
        _emit(its[mid:], li + 1)

    if items:
        import sys as _sys
        _old = _sys.getrecursionlimit()
        _sys.setrecursionlimit(max(_old, 10000))
        nodes.append(None)
        _emit(list(items), 0)
        _sys.setrecursionlimit(_old)
    return prims, nodes


def _bvh_preview_box_mesh():
    """Shared unit wireframe cube for the BVH preview (same datablock the
    importer uses, so re-imported and organized scenes share it)."""
    try:
        if "fo2_bvh_cube" in bpy.data.meshes:
            return bpy.data.meshes["fo2_bvh_cube"]
    except Exception:
        pass
    verts = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
             (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    mesh = bpy.data.meshes.new("fo2_bvh_cube")
    mesh.from_pydata(verts, edges, [])
    mesh.update()
    return mesh


def _resolve_material_texture_key(bl_mat):
    """Resolve the texture name a material will export with (used as the
    grouping key when splitting surfaces per texture). Mirrors the logic in
    ensure_material_properties/_apply_material_conventions: stored
    bgm_texture_0 first, then the image node, then the material name; for
    colormap materials the meaningful key is the detail texture."""
    if bl_mat is None:
        return ""
    t0 = str(bl_mat.get("bgm_texture_0", "") or "")
    if not t0 and getattr(bl_mat, 'use_nodes', False):
        try:
            for node in bl_mat.node_tree.nodes:
                if node.type == 'TEX_IMAGE' and node.image:
                    base = os.path.splitext(os.path.basename(node.image.filepath))[0]
                    if not base and node.image.name:
                        base = os.path.splitext(node.image.name)[0]
                    if base:
                        t0 = base + ".tga"
                        break
        except Exception:
            pass
    if not t0:
        t0 = strip_blender_suffix(bl_mat.name)
    t0 = _force_tga(t0).lower()
    if t0 in ("colormap.tga", "colormap.dds"):
        t1 = str(bl_mat.get("bgm_texture_1", "") or "")
        if not t1:
            t1 = strip_blender_suffix(bl_mat.name)
        return _force_tga(t1).lower()
    return t0


def _split_mesh_per_texture(obj, static_col):
    """Split a multi-material mesh object into one object per distinct
    TEXTURE NAME used by its faces (the w32 format allows one material per
    surface, and materials sharing a texture merge into one surface).
    Material slots resolving to the same texture are grouped; each part gets
    the group's first material. Copies geometry, the first UV layer and any
    corner-domain color attribute; world transform is preserved. Returns the
    list of new objects, or None when no split is needed."""
    me = obj.data
    if len(me.materials) <= 1:
        return None
    # material slot -> texture key; group slots per key (insertion order)
    slot_key = [_resolve_material_texture_key(m) for m in me.materials]
    groups = {}
    for si, key in enumerate(slot_key):
        groups.setdefault(key, []).append(si)
    used_keys = []
    key_of_poly = []
    for p in me.polygons:
        mi = getattr(p, 'material_index', 0)
        key = slot_key[mi] if mi < len(slot_key) else (slot_key[0] if slot_key else "")
        key_of_poly.append(key)
        if key not in used_keys:
            used_keys.append(key)
    if len(used_keys) <= 1:
        return None
    uv_src = None
    try:
        if me.uv_layers:
            uv_src = me.uv_layers[0].data
    except Exception:
        pass
    col_src = None
    try:
        for ca in me.color_attributes:
            if getattr(ca, 'domain', '') == 'CORNER':
                col_src = ca.data
                break
    except Exception:
        pass
    new_objs = []
    for key in used_keys:
        vmap = {}
        verts = []
        faces = []
        luvs = []
        lcols = []
        for pi, p in enumerate(me.polygons):
            if key_of_poly[pi] != key:
                continue
            f = []
            for li in p.loop_indices:
                vi = me.loops[li].vertex_index
                if vi not in vmap:
                    vmap[vi] = len(verts)
                    verts.append(tuple(me.vertices[vi].co))
                f.append(vmap[vi])
                if uv_src is not None:
                    try:
                        luvs.append(tuple(uv_src[li].uv))
                    except Exception:
                        uv_src = None
                if col_src is not None:
                    try:
                        lcols.append(tuple(col_src[li].color))
                    except Exception:
                        col_src = None
            faces.append(tuple(f))
        if not faces:
            continue
        stem = os.path.splitext(os.path.basename(key))[0] or "tex"
        new_me = bpy.data.meshes.new(f"{obj.name}_{stem}")
        new_me.from_pydata(verts, [], faces)
        # the group's first material represents every slot sharing the texture
        rep = None
        for si in groups.get(key, []):
            if si < len(me.materials) and me.materials[si] is not None:
                rep = me.materials[si]
                break
        if rep is not None:
            new_me.materials.append(rep)
        # loop order after from_pydata equals the face/corner traversal order
        # above, so sequential per-loop assignment lines up
        if luvs:
            try:
                uvl = new_me.uv_layers.new(name="UVMap")
                for i, uv in enumerate(luvs):
                    uvl.data[i].uv = uv
            except Exception:
                pass
        if lcols:
            try:
                ca = new_me.color_attributes.new(name="Color",
                                                 type='BYTE_COLOR',
                                                 domain='CORNER')
                for i, c in enumerate(lcols):
                    ca.data[i].color = c
            except Exception:
                pass
        new_me.update()
        new_obj = bpy.data.objects.new(f"{obj.name}_{stem}", new_me)
        try:
            new_obj.matrix_world = obj.matrix_world.copy()
        except Exception:
            new_obj.matrix_world = obj.matrix_world
        static_col.objects.link(new_obj)
        new_objs.append(new_obj)
    return new_objs if new_objs else None


# ═════════════════════════════════════════════════════════════════════════════
# Collision mesh generation (for the fo2_collision_io exporter)
# ═════════════════════════════════════════════════════════════════════════════
# Produces the exact structure fo2_collision_io's CDB2 exporter expects: a
# collection named "collision" containing mesh objects that carry
# fo2_surface_id / fo2_lo_flags / fo2_bitmask object properties.
# fo2_hi_flags is deliberately NOT set: collision_io then computes it
# per-triangle from geometry (shadow participation for upward faces), and
# fo2_has_shadow is likewise derived from the bitmask (bit 3 -> auto-computed
# planar shadow UVs), which is its designed path for new geometry.
#
# Surface ids follow collision_io's _surface_name convention (sid-1 indexes
# its FO2_SURFACE_NAMES list). Bitmasks follow vanilla statistics (288k
# triangles over 8 tracks): drivable ground 0xB (shadowed), objects/walls
# 0x3, trees 0x1, water 0x6. lo_flags varies per-triangle in vanilla with 4
# as the universal common value - collision_io's own default for new
# geometry - so 4 is used uniformly.
_COL_SURFACES = {
    #  sid, name,           bitmask
    2:  ("Tarmac",        0xB),
    4:  ("Asphalt",       0xB),
    16: ("Snow",          0xB),
    18: ("Dirt",          0xB),
    24: ("Grass",         0xB),
    25: ("Forest",        0xB),
    26: ("Sand",          0xB),
    27: ("Rock_Terrain",  0x3),
    30: ("Field",         0xB),
    32: ("Concrete",      0x3),
    33: ("Rock_Object",   0x3),
    34: ("Metal",         0x3),
    35: ("Wood",          0x3),
    36: ("Tree",          0x1),
    38: ("Rubber",        0x3),
    39: ("Water",         0x6),
}

# keyword -> (ground_sid, object_sid); None = group not applicable. First
# match (in order) wins; ground/object choice resolved by geometry.
_COL_KEYWORDS = [
    (("tarmac", "asphalt", "road", "street", "highway", "track_"), 2, 32),
    (("grass", "lawn", "turf", "moss"),                            24, 24),
    (("forest", "leaves", "foliage"),                              25, 25),
    (("sand", "beach", "dune"),                                    26, 26),
    (("snow", "ice"),                                              16, 16),
    (("dirt", "mud", "soil", "gravel", "ground", "terrain"),       18, 18),
    (("field", "crop", "wheat"),                                   30, 30),
    (("rock", "stone", "cliff", "boulder", "mountain"),            27, 33),
    (("metal", "steel", "iron", "girder", "pipe", "fence", "rail"), 34, 34),
    (("wood", "plank", "log", "timber", "bark"),                   35, 35),
    (("tree", "trunk"),                                            36, 36),
    (("rubber", "tire", "tyre"),                                   38, 38),
    (("water", "river", "lake", "sea", "ocean"),                   39, 39),
    (("concrete", "cement", "brick", "wall", "bunker", "base"),    32, 32),
]


def _mesh_up_fraction(obj):
    """Fraction of face area with an upward-facing world normal (Blender +Z)
    - the ground-vs-object discriminator."""
    me = obj.data
    mw = obj.matrix_world
    up_area = 0.0
    tot_area = 0.0
    try:
        for p in me.polygons:
            vs = [mw @ me.vertices[vi].co for vi in p.vertices]
            if len(vs) < 3:
                continue
            # polygon area + normal via Newell
            nx = ny = nz = 0.0
            for i in range(len(vs)):
                a = vs[i]
                b = vs[(i + 1) % len(vs)]
                nx += (a[1] - b[1]) * (a[2] + b[2])
                ny += (a[2] - b[2]) * (a[0] + b[0])
                nz += (a[0] - b[0]) * (a[1] + b[1])
            area = 0.5 * (nx * nx + ny * ny + nz * nz) ** 0.5
            if area <= 0:
                continue
            tot_area += area
            if nz / (2 * area) > 0.5:      # normal z-component > 0.5
                up_area += area
    except Exception:
        return 0.0
    return up_area / tot_area if tot_area > 0 else 0.0


def _classify_collision_surface(obj):
    """Pick a collision surface id for an organized render mesh from its
    material/texture names, refined by geometry (up-facing area fraction)."""
    names = []
    for m in obj.data.materials:
        if m is None:
            continue
        names.append(strip_blender_suffix(m.name).lower())
        for ti in range(3):
            t = str(m.get(f"bgm_texture_{ti}", "") or "")
            if t and t.lower() not in ("colormap.tga",):
                names.append(os.path.splitext(t)[0].lower())
    blob = " ".join(names)
    is_groundish = None
    for words, ground_sid, object_sid in _COL_KEYWORDS:
        if any(w in blob for w in words):
            if ground_sid == object_sid:
                return ground_sid
            if is_groundish is None:
                is_groundish = _mesh_up_fraction(obj) >= 0.6
            return ground_sid if is_groundish else object_sid
    # no keyword match: terrain-shader materials are ground; otherwise use
    # geometry - mostly-upward meshes read as drivable ground, the rest as
    # solid objects
    for m in obj.data.materials:
        if m is not None and (int(m.get("bgm_use_colormap", 0))
                              or int(m.get("bgm_shader_id", 0)) in (1, 2)):
            return 24    # Grass: drivable-but-slow, the safe unknown ground
    return 24 if _mesh_up_fraction(obj) >= 0.6 else 32   # Concrete otherwise


def _triangulate_polygon(vs_idx, vs_co):
    """Triangulate one polygon (indices + coords). Quads split along the
    diagonal that minimizes the dihedral angle between the two resulting
    triangles - on sloped terrain grids this follows the surface curvature
    instead of cutting across it. N-gons fall back to a fan."""
    n = len(vs_idx)
    if n == 3:
        return [tuple(vs_idx)]
    if n == 4:
        def _nrm(a, b, c):
            ux, uy, uz = (b[0]-a[0], b[1]-a[1], b[2]-a[2])
            vx, vy, vz = (c[0]-a[0], c[1]-a[1], c[2]-a[2])
            x, y, z = (uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx)
            l = (x*x+y*y+z*z) ** 0.5
            return (x/l, y/l, z/l) if l > 1e-12 else None
        def _split_score(t1, t2):
            n1 = _nrm(*(vs_co[i] for i in t1))
            n2 = _nrm(*(vs_co[i] for i in t2))
            if n1 is None or n2 is None:
                return -2.0
            return n1[0]*n2[0] + n1[1]*n2[1] + n1[2]*n2[2]
        sa = _split_score((0, 1, 2), (0, 2, 3))
        sb = _split_score((0, 1, 3), (1, 2, 3))
        if sb > sa:
            return [(vs_idx[0], vs_idx[1], vs_idx[3]),
                    (vs_idx[1], vs_idx[2], vs_idx[3])]
        return [(vs_idx[0], vs_idx[1], vs_idx[2]),
                (vs_idx[0], vs_idx[2], vs_idx[3])]
    return [(vs_idx[0], vs_idx[i], vs_idx[i+1]) for i in range(1, n-1)]


def _qualifies_right_angle(edge_dir, signed_angle_deg, angle_tol=15.0,
                           max_tilt_deg=5.0):
    """True for an axis-aligned (HORIZONTAL or VERTICAL) CONVEX ~90-degree
    crease.

    edge_dir       : the crease edge's direction vector (blender coords)
    signed_angle_deg: angle between the two adjacent faces, POSITIVE for
                      convex and negative for concave (bmesh's
                      BMEdge.calc_face_angle_signed convention); 0 = flat.

    Two crease orientations qualify, both within max_tilt_deg:
      HORIZONTAL - the crease lies in the XY plane: ground-meets-wall and
                   top-of-step creases the car drives into.
      VERTICAL   - the crease runs along Z: wall-meets-wall corner posts,
                   the vertical edges of blocks/buildings the car sideswipes.
    Creases at an arbitrary tilt are left alone, as are concave creases and
    anything outside 90 +/- angle_tol."""
    import math
    l = math.sqrt(edge_dir[0]**2 + edge_dir[1]**2 + edge_dir[2]**2)
    if l < 1e-9:
        return False
    vert_component = abs(edge_dir[2] / l)
    tilt = math.radians(max_tilt_deg)
    is_horizontal = vert_component <= math.sin(tilt)
    is_vertical = vert_component >= math.cos(tilt)
    if not (is_horizontal or is_vertical):
        return False                      # crease at an arbitrary tilt
    if signed_angle_deg <= 0.0:
        return False                      # concave (or flat) - leave alone
    return abs(signed_angle_deg - 90.0) <= angle_tol


def _bevel_collision_right_angles(me, quantum=0.0, offset=0.12,
                                  angle_tol=15.0, max_tilt_deg=5.0):
    """Bevel convex ~90-degree creases (horizontal and vertical) on a
    collision mesh.

    Uses bmesh.ops.bevel - Blender's own topological bevel - so the mesh
    stays watertight. (A hand-rolled edge recede cannot: moving a vertex for
    one crease breaks every other edge incident to that vertex, which is what
    opened pass-through holes in the earlier attempt.)

    The offset is raised to at least a few int16 grid quanta so the new bevel
    faces survive CDB quantization instead of collapsing into slivers that
    the exporter would drop. Returns the number of creases beveled."""
    try:
        import bmesh
    except Exception:
        return 0        # real-Blender only; clean no-op in the harness
    import math
    off = offset
    if quantum > 0.0:
        off = max(off, 4.0 * quantum)
    try:
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.normal_update()
        targets = []
        for e in bm.edges:
            if len(e.link_faces) != 2:
                continue
            a, b = e.verts[0].co, e.verts[1].co
            d = (b[0]-a[0], b[1]-a[1], b[2]-a[2])
            try:
                ang = math.degrees(e.calc_face_angle_signed())
            except Exception:
                continue
            if _qualifies_right_angle(d, ang, angle_tol, max_tilt_deg):
                targets.append(e)
        if not targets:
            bm.free()
            return 0
        try:
            bmesh.ops.bevel(bm, geom=targets, offset=off,
                            offset_type='OFFSET', segments=1, profile=0.5,
                            affect='EDGES')
        except TypeError:
            # older bmesh without the 'affect' argument
            bmesh.ops.bevel(bm, geom=targets, offset=off,
                            offset_type='OFFSET', segments=1, profile=0.5)
        bm.to_mesh(me)
        bm.free()
        return len(targets)
    except Exception as exc:
        print(f"[W32 Collision] right-angle bevel skipped: {exc}")
        try:
            bm.free()
        except Exception:
            pass
        return 0


def _bevel_right_angles_lists(verts, tris, quantum=0.0, **kw):
    """Bevel convex right angles (horizontal and vertical creases) on raw
    (verts, tris) lists.

    Runs the bmesh bevel through a temporary mesh and hands plain lists back,
    so the degenerate/slicing cleanup pass (_cleanup_degenerate_tris) can run
    AFTERWARDS and repair anything the chamfer introduces on the int16 grid.
    bmesh's chamfer faces come back as quads/ngons, so they are fan
    triangulated here. Returns (verts, tris, n_beveled)."""
    try:
        import bmesh                                    # noqa: F401
    except Exception:
        return verts, tris, 0                           # harness / no bmesh
    tmp = None
    try:
        tmp = bpy.data.meshes.new("__fo2_bevel_tmp")
        tmp.from_pydata([tuple(v) for v in verts], [],
                        [tuple(t) for t in tris])
        n = _bevel_collision_right_angles(tmp, quantum=quantum, **kw)
        if n:
            new_verts = [tuple(v.co) for v in tmp.vertices]
            new_tris = []
            for p in tmp.polygons:
                idx = list(p.vertices)
                if len(idx) == 3:
                    new_tris.append(tuple(idx))
                elif len(idx) > 3:
                    for k in range(1, len(idx) - 1):
                        new_tris.append((idx[0], idx[k], idx[k+1]))
            verts, tris = new_verts, new_tris
        return verts, tris, n
    except Exception as exc:
        print(f"[W32 Collision] right-angle bevel skipped: {exc}")
        return verts, tris, 0
    finally:
        if tmp is not None:
            try:
                bpy.data.meshes.remove(tmp)
            except Exception:
                pass


def _compute_lo_flags(verts, tris_blender, eps=1e-6):
    """Per-triangle lo_flags, reverse-engineered from vanilla collision data.

    Semantics (measured over 4 tracks, tens of thousands of edges, using
    OUTWARD normals - note the CDB file stores reversed winding, so vanilla
    file-order normals point inward/down and the sign flips):
        bit k SET   <-> file edge k is CONCAVE (inside corner)
        bit k CLEAR <-> file edge k is CONVEX  (cube edge / ridge)
    Vanilla convex edges: bit set 0.0-0.2%.  Concave: 95-100%.  Flat edges
    vanilla sets 61-79% of the time, so near-flat is treated as concave here
    (harmless - coplanar faces need no edge contact) which lifts agreement
    with vanilla to 85-96%.

    Edge index mapping: the exporter writes triangles reversed, so
    blender (b0,b1,b2) -> file (b2,b1,b0), giving
        file edge 0 <- blender edge 1
        file edge 1 <- blender edge 0
        file edge 2 <- blender edge 2
    (Getting this mapping wrong is what made an earlier attempt worse than a
    constant: it put the bits on the wrong edges AND inverted the meaning.)

    Bits 3-5 are the matching per-vertex flags, set when both incident file
    edges are set (the 97.5% implication measured in vanilla).
    Inputs are in BLENDER winding. Returns one int per triangle."""
    import math

    def nrm(t):
        a, b, c = verts[t[0]], verts[t[1]], verts[t[2]]
        u = [b[i]-a[i] for i in range(3)]
        v = [c[i]-a[i] for i in range(3)]
        x, y, z = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
        l = math.sqrt(x*x+y*y+z*z)
        return (x/l, y/l, z/l) if l > 1e-12 else None

    ns = [nrm(t) for t in tris_blender]
    em = {}
    for ti, t in enumerate(tris_blender):
        for k in range(3):
            em.setdefault(frozenset((t[k], t[(k+1) % 3])), []).append((ti, k))
    concave = [[False]*3 for _ in tris_blender]
    for e, lst in em.items():
        if len(lst) != 2:
            continue                      # boundary edge -> leave clear
        ev = set(e)
        for (ti, k), (tj, _kj) in ((lst[0], lst[1]), (lst[1], lst[0])):
            if ns[ti] is None or ns[tj] is None:
                continue
            ov = [v for v in tris_blender[tj] if v not in ev]
            if not ov:
                continue
            a = verts[tris_blender[ti][0]]
            h = sum(ns[ti][i]*(verts[ov[0]][i]-a[i]) for i in range(3))
            # neighbour above this face's outward plane -> concave; near-flat
            # counts as concave, only a genuine convex ridge stays clear
            concave[ti][k] = h > -eps
    B2F = {1: 0, 0: 1, 2: 2}
    out = []
    for ti in range(len(tris_blender)):
        lo = 0
        for bk in range(3):
            if concave[ti][bk]:
                lo |= (1 << B2F[bk])
        for j in range(3):
            if ((lo >> j) & 1) and ((lo >> ((j+2) % 3)) & 1):
                lo |= (1 << (3+j))
        out.append(lo)
    return out


def _decimate_collision(verts, tris, err=0.05):
    """Simplify a collision soup by edge collapse, preserving the surface.

    Polygon re-triangulation (ear clipping, constrained Delaunay, monotone
    sweep) was worse on EVERY axis: it discards the render mesh's authored
    tessellation and rebuilds long thin triangles from simplified boundary
    loops, doubling needles, multiplying caps 40x and introducing hundreds of
    grid-degenerate triangles.  Edge collapse keeps the existing tessellation
    and only removes redundant detail, so it reaches vanilla's triangle count
    with better triangle quality than vanilla itself and no degeneracy.

    Three guards make it safe:
      planarity  - the merged vertex must stay within `err` of every plane it
                   touched, so the surface does not drift,
      link condition - a and b may only share the vertices opposite their
                   common edge, otherwise the collapse welds separate sheets
                   and breeds non-manifold edges,
      fold check - no adjacent triangle may reverse its normal.
    Returns (verts, tris, collapsed_count)."""
    import math
    verts = [list(v) for v in verts]
    T = [tuple(t) for t in tris]
    vt = {}
    for ti, t in enumerate(T):
        for i in t:
            vt.setdefault(i, set()).add(ti)
    alive = [True]*len(T)
    remap = {}

    def res(i):
        while i in remap:
            i = remap[i]
        return i

    def nrm(a, b, c):
        u = (b[0]-a[0], b[1]-a[1], b[2]-a[2])
        v = (c[0]-a[0], c[1]-a[1], c[2]-a[2])
        x, y, z = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
        l = math.sqrt(x*x+y*y+z*z)
        return (x/l, y/l, z/l) if l > 1e-12 else None

    def planes_of(vi):
        out = []
        for ti in vt.get(vi, ()):
            if not alive[ti]:
                continue
            pts = [verts[res(x)] for x in T[ti]]
            n = nrm(*pts)
            if n:
                out.append((n, pts[0]))
        return out

    edges = set()
    ecount = {}
    for t in T:
        for k in range(3):
            a, b = t[k], t[(k+1) % 3]
            e = (min(a, b), max(a, b))
            edges.add(e)
            ecount[e] = ecount.get(e, 0) + 1
    # BOUNDARY PRESERVATION: a planarity guard alone lets an open sheet
    # collapse to nothing - every collapse stays in the plane, so a flat road
    # disappears entirely (measured: a 1250-triangle flat grid went to 0).
    # Boundary vertices therefore never move, interior vertices collapse into
    # them, and two boundary vertices may only merge along a boundary edge.
    boundary_v = set()
    boundary_e = set()
    bnbr = {}
    for e, c in ecount.items():
        if c == 1:
            boundary_e.add(e)
            boundary_v.add(e[0])
            boundary_v.add(e[1])
            bnbr.setdefault(e[0], set()).add(e[1])
            bnbr.setdefault(e[1], set()).add(e[0])

    def boundary_ok(a, b, mid):
        """Merging two boundary vertices must not reshape the outline: the
        removed corners have to lie within err of the straightened boundary.
        Without this a flat sheet keeps collapsing along its own perimeter
        until nothing is left (a 1250-triangle grid went to 0 triangles)."""
        ends = []
        for v, other in ((a, b), (b, a)):
            for nb in bnbr.get(v, ()):
                if res(nb) not in (a, b):
                    ends.append(verts[res(nb)])
        if len(ends) < 2:
            return False
        p, q = ends[0], ends[-1]
        ux, uy, uz = q[0]-p[0], q[1]-p[1], q[2]-p[2]
        L = math.sqrt(ux*ux+uy*uy+uz*uz)
        if L < 1e-9:
            return False
        for w in (verts[a], verts[b], mid):
            wx, wy, wz = w[0]-p[0], w[1]-p[1], w[2]-p[2]
            cx = uy*wz-uz*wy
            cy = uz*wx-ux*wz
            cz = ux*wy-uy*wx
            if math.sqrt(cx*cx+cy*cy+cz*cz)/L > err:
                return False
        return True
    order = sorted(edges, key=lambda e: sum((verts[e[0]][i]-verts[e[1]][i])**2
                                            for i in range(3)))
    collapsed = 0
    for a0, b0 in order:
        a, b = res(a0), res(b0)
        if a == b:
            continue
        ba = a in boundary_v
        bb = b in boundary_v
        if ba and bb and (min(a0, b0), max(a0, b0)) not in boundary_e:
            continue                      # would pinch the sheet shut
        if bb and not ba:
            a, b = b, a                   # always keep the boundary vertex
            ba, bb = bb, ba
        pa = planes_of(a)
        pb = planes_of(b)
        if not pa or not pb:
            continue
        if ba and not bb:
            mid = list(verts[a])          # interior vertex folds into boundary
        else:
            mid = [(verts[a][i]+verts[b][i])*0.5 for i in range(3)]
        if ba and bb and not boundary_ok(a, b, mid):
            continue
        drift = False
        for n, p in pa+pb:
            if abs(sum(n[i]*(mid[i]-p[i]) for i in range(3))) > err:
                drift = True
                break
        if drift:
            continue
        na = set()
        nb = set()
        for ti in vt.get(a, ()):
            if alive[ti]:
                na.update(res(x) for x in T[ti])
        for ti in vt.get(b, ()):
            if alive[ti]:
                nb.update(res(x) for x in T[ti])
        na.discard(a); na.discard(b); nb.discard(a); nb.discard(b)
        opp = set()
        for ti in vt.get(a, set()) & vt.get(b, set()):
            if alive[ti]:
                for x in T[ti]:
                    r = res(x)
                    if r not in (a, b):
                        opp.add(r)
        if (na & nb) != opp:
            continue
        flip = False
        for ti in vt.get(a, set()) | vt.get(b, set()):
            if not alive[ti]:
                continue
            r = [res(x) for x in T[ti]]
            if a in r and b in r:
                continue
            on = nrm(*[verts[x] for x in r])
            if on is None:
                continue
            nn = nrm(*[mid if x in (a, b) else verts[x] for x in r])
            if nn is None or sum(on[i]*nn[i] for i in range(3)) < 0.2:
                flip = True
                break
        if flip:
            continue
        verts[a] = mid
        remap[b] = a
        vt.setdefault(a, set()).update(vt.get(b, ()))
        collapsed += 1
        for ti in list(vt[a]):
            if not alive[ti]:
                continue
            r = [res(x) for x in T[ti]]
            if r[0] == r[1] or r[1] == r[2] or r[0] == r[2]:
                alive[ti] = False
    out_t = []
    for ti, t in enumerate(T):
        if not alive[ti]:
            continue
        r = tuple(res(x) for x in t)
        if r[0] != r[1] and r[1] != r[2] and r[0] != r[2]:
            out_t.append(r)
    used = {}
    fv = []
    ft = []
    for t in out_t:
        nt = []
        for i in t:
            j = used.get(i)
            if j is None:
                j = len(fv)
                used[i] = j
                fv.append(tuple(verts[i]))
            nt.append(j)
        ft.append(tuple(nt))
    return fv, ft, collapsed


def _double_side_open_walls(verts, tris, vertical_max_nz=0.7):
    """Give single-sided wall geometry collision on BOTH faces.

    FlatOut collision is single-sided: a triangle blocks only from the side
    its normal points at.  Custom maps very often model walls as flat quads
    with no thickness - grove's Surface13 is 1031 faces in 86 connected
    components, every one an OPEN sheet (zero closed solids), 38 of them
    single panels of four faces or fewer.  Such a wall lets the car through
    from its back face and then blocks it from the inside, which is exactly
    the "I can drive into the house but not back out" symptom.

    For every triangle that belongs to an open component (one with boundary
    edges, i.e. a sheet rather than a solid) and is wall-like rather than
    floor-like, a reverse-wound duplicate is emitted so the surface blocks
    from either side.  Closed solids are left alone - they already present
    an outward face everywhere, and duplicating them would bury faces inside
    the volume.  Near-horizontal faces are left alone too, so the drivable
    ground keeps a single upward face and does not gain an opposing face a
    few millimetres beneath it.

    This can only ADD blocking, never remove it, so unlike a re-orientation
    pass it cannot introduce fall-through. Returns (tris, n_added)."""
    import math
    em = {}
    for ti, t in enumerate(tris):
        for k in range(3):
            em.setdefault(frozenset((t[k], t[(k+1) % 3])), []).append(ti)
    # connected components
    seen = [False]*len(tris)
    open_tri = [False]*len(tris)
    for s in range(len(tris)):
        if seen[s]:
            continue
        seen[s] = True
        stack = [s]
        grp = [s]
        while stack:
            ti = stack.pop()
            for k in range(3):
                for nb in em[frozenset((tris[ti][k], tris[ti][(k+1) % 3]))]:
                    if not seen[nb]:
                        seen[nb] = True
                        stack.append(nb)
                        grp.append(nb)
        # a component with any boundary edge is a sheet, not a solid
        is_open = False
        for ti in grp:
            for k in range(3):
                if len(em[frozenset((tris[ti][k], tris[ti][(k+1) % 3]))]) == 1:
                    is_open = True
                    break
            if is_open:
                break
        if is_open:
            for ti in grp:
                open_tri[ti] = True

    def nz_of(t):
        a, b, c = verts[t[0]], verts[t[1]], verts[t[2]]
        u = (b[0]-a[0], b[1]-a[1], b[2]-a[2])
        v = (c[0]-a[0], c[1]-a[1], c[2]-a[2])
        x = u[1]*v[2]-u[2]*v[1]
        y = u[2]*v[0]-u[0]*v[2]
        z = u[0]*v[1]-u[1]*v[0]
        l = math.sqrt(x*x+y*y+z*z)
        return abs(z)/l if l > 1e-12 else 1.0

    existing = set(frozenset(t) for t in tris)
    out = list(tris)
    added = 0
    for ti, t in enumerate(tris):
        if not open_tri[ti]:
            continue
        if nz_of(t) > vertical_max_nz:
            continue                       # floor/ceiling-like, leave alone
        rev = (t[2], t[1], t[0])
        if frozenset(rev) in existing and tris.count(t) > 1:
            continue                       # already double-sided
        out.append(rev)
        added += 1
    return out, added


def _generate_collision(context, organized, bevel_right_angles=False,
                        compute_lo=False, simplify=False, simplify_err=0.10,
                        double_side=True):
    """Build (or rebuild) the 'collision' collection from the organized
    render meshes: one merged, world-space-baked, slope-aware-triangulated
    mesh object per collision surface id, carrying the object properties
    fo2_collision_io's exporter reads. Returns a summary string."""
    existing = bpy.data.collections.get("collision") if hasattr(bpy.data.collections, 'get') else None
    if existing is None:
        for c in bpy.data.collections:
            if getattr(c, 'name', '') == "collision":
                existing = c
                break
    if existing is not None and not existing.get("fo2_generated_collision"):
        return "existing (imported) collision collection kept as-is"
    if existing is not None:
        for o in list(existing.objects):
            try:
                bpy.data.objects.remove(o, do_unlink=True)
            except Exception:
                try:
                    existing.objects.unlink(o)
                except Exception:
                    pass
        col = existing
    else:
        col = bpy.data.collections.new("collision")
        context.scene.collection.children.link(col)
    col["fo2_generated_collision"] = True

    # bucket world-space triangles per surface id
    buckets = {}   # sid -> (vmap, verts, tris)
    tri_total = 0
    gmax = [0.001, 0.001, 0.001]   # per-GAME-axis max |coord| for inv_mult
    for obj in organized:
        me = obj.data
        sid = _classify_collision_surface(obj)
        if sid not in buckets:
            buckets[sid] = ({}, [], [])
        vmap, verts, tris = buckets[sid]
        mw = obj.matrix_world
        try:
            wco_cache = [tuple(mw @ v.co) for v in me.vertices]
        except Exception:
            continue
        # A mirrored object (negative-scale / negative-determinant world
        # matrix) has its triangle handedness flipped when the transform is
        # baked into world space. Blender hides this when rendering - it
        # flips normals for negative-scaled objects - so the object looks
        # correct in the viewport while the baked collision faces INWARD.
        # FlatOut collision is single-sided, so such a wall lets the car in
        # from outside and traps it inside. Reversing the winding for these
        # objects restores outward orientation.
        try:
            neg_scale = mw.determinant() < 0.0
        except Exception:
            neg_scale = False
        for w in wco_cache:
            g = (abs(w[0]), abs(w[2]), abs(w[1]))
            for i in range(3):
                if g[i] > gmax[i]:
                    gmax[i] = g[i]
        # PRIMARY: use Blender's loop_triangles - the SAME tessellation the
        # w32 exporter bakes into the render geometry, so the collision
        # surface matches what the player sees exactly (no invisible seams
        # or bumps from a different quad-diagonal choice on slopes).
        tri_list = []
        try:
            lts = me.loop_triangles
            for lt in lts:
                t = tuple(lt.vertices)
                if len(t) != 3:      # defensive: real Blender always gives 3
                    tri_list = []
                    break
                tri_list.append(t)
        except Exception:
            tri_list = []
        if tri_list:
            for t in tri_list:
                tri = []
                for vi in t:
                    key = wco_cache[vi]
                    gi = vmap.get(key)
                    if gi is None:
                        gi = len(verts)
                        vmap[key] = gi
                        verts.append(key)
                    tri.append(gi)
                if tri[0] != tri[1] != tri[2] != tri[0]:
                    tris.append(tuple(reversed(tri)) if neg_scale
                                else tuple(tri))
                    tri_total += 1
            continue
        # FALLBACK (no loop_triangles): slope-aware manual triangulation
        for p in me.polygons:
            pv = list(p.vertices)
            if len(pv) < 3:
                continue
            co = [wco_cache[vi] for vi in pv]
            local = list(range(len(pv)))
            for t in _triangulate_polygon(local, co):
                tri = []
                for k in t:
                    key = co[k]
                    gi = vmap.get(key)
                    if gi is None:
                        gi = len(verts)
                        vmap[key] = gi
                        verts.append(key)
                    tri.append(gi)
                if tri[0] != tri[1] != tri[2] != tri[0]:
                    tris.append(tuple(tri))
                    tri_total += 1

    # same inverse multipliers collision_io will compute from these bounds,
    # so degeneracy is detected on the exact int16 grid the CDB uses
    inv_mult = tuple(32767.0 / m for m in gmax)
    made = 0
    fix_stats = {"collapsed": 0, "cap_split": 0, "dropped": 0}
    bev_total = 0
    simp_total = 0
    dbl_total = 0
    for sid in sorted(buckets):
        vmap, verts, tris = buckets[sid]
        if not tris:
            continue
        # SIMPLIFY FIRST: edge collapse is the biggest structural change, so
        # the bevel and the degenerate/slicing cleanup both run on its output
        if simplify and simplify_err > 0.0:
            try:
                verts, tris, nc = _decimate_collision(verts, tris, simplify_err)
                simp_total += nc
            except Exception as exc:
                print(f"[W32 Collision] simplify skipped: {exc}")

        # BEVEL: the chamfer changes topology, so the degenerate /
        # slicing cleanup below must run on its output (otherwise the new
        # sliver faces survive to export and get dropped, leaving holes)
        if bevel_right_angles:
            q = max(1.0 / m for m in inv_mult)
            verts, tris, nb = _bevel_right_angles_lists(verts, tris, quantum=q)
            bev_total += nb
        verts, tris, st = _cleanup_degenerate_tris(verts, tris, inv_mult)
        for k in fix_stats:
            fix_stats[k] += st[k]
        if not tris:
            continue
        name_, bitmask = _COL_SURFACES.get(sid, (f"Unknown_{sid}", 0x3))
        name = f"col_{sid}_{name_}"
        if double_side:
            try:
                tris, nadd = _double_side_open_walls(verts, tris)
                dbl_total += nadd
            except Exception as exc:
                print(f"[W32 Collision] double-siding skipped: {exc}")

        lo_default = 0
        lo_list = None
        if compute_lo:
            try:
                lo_list = _compute_lo_flags(verts, tris)
            except Exception as exc:
                print(f"[W32 Collision] lo_flags computation skipped: {exc}")
                lo_list = None
        me = bpy.data.meshes.new(name)
        me.from_pydata(verts, [], tris)
        if lo_list:
            try:
                attr = me.attributes.new(name="fo2_lo_flags", type='INT',
                                         domain='FACE')
                for i, lo in enumerate(lo_list):
                    attr.data[i].value = int(lo) & 0x3F
            except Exception as exc:
                print(f"[W32 Collision] lo_flags attribute skipped: {exc}")
        # display-only for the w32 pipeline: never organized into StaticBatch
        # nor exported as render geometry (collision_io reads it instead)
        me["fo2_display_only"] = True
        bl_mat = bpy.data.materials.new(name)
        bl_mat["fo2_display_only"] = True     # never a w32 track material
        me.materials.append(bl_mat)
        me.update()
        o = bpy.data.objects.new(name, me)
        o["fo2_surface_id"] = sid
        # lo_flags bit k marks FILE edge k as CONCAVE (verified across vanilla:
        # convex edges have the bit clear in 99.8-100% of cases, n>21000;
        # concave edges have it set in 95-100%). The old constant 4 set bit 2
        # on every triangle, wrongly flagging 1 in 3 convex 90-degree cube
        # edges as concave - which suppresses the edge contact there and lets
        # the car slip through exactly at box corners. 0 = "no concave edges",
        # which is always correct for convex geometry and conservative
        # elsewhere (an extra edge contact at an inside corner is harmless).
        o["fo2_lo_flags"] = int(lo_default)
        # human-readable flags only (no raw bitmask - the booleans ARE the
        # bitmask; semantics confirmed by cross-track survey: bit0=car,
        # bit1=camera, bit2=water, bit3=shadow)
        o["fo2_collide_car"] = bool(bitmask & 1)
        o["fo2_collide_camera"] = bool(bitmask & 2)
        o["fo2_is_water"] = bool(bitmask & 4)
        o["fo2_has_shadow"] = bool(bitmask & 8)
        # fo2_hi_flags intentionally absent -> collision_io computes it
        # per-triangle from geometry
        col.objects.link(o)
        made += 1
    fx = ""
    if any(fix_stats.values()):
        fx = (f" (degenerate fixes: {fix_stats['collapsed']} collapsed, "
              f"{fix_stats['cap_split']} cap-split, {fix_stats['dropped']} dropped)")
    if dbl_total:
        fx += f" (double-sided {dbl_total} thin-wall faces)"
    if simp_total:
        fx += f" (simplified: {simp_total} edges collapsed)"
    if bev_total:
        fx += f" (beveled {bev_total} right-angle creases)"
    return f"collision: {made} surfaces / {tri_total} tris{fx}"


def _f32(x):
    """Snap a Python float to the nearest float32 value.

    Blender stores mesh vertex coordinates as 32-bit floats, so any
    double-precision coordinate we compute (a collapse midpoint, a cap-split
    point) is silently rounded when written into the mesh. The CDB exporter
    then quantizes THOSE rounded numbers. Validating degeneracy in float64
    while the exporter sees float32 makes the two disagree on triangles
    sitting near a grid boundary (measured: ~0.04% of near-degenerate
    triangles flip verdict - a handful per track, which is exactly the
    "a few degenerate triangles dropped" symptom). Snapping every coordinate
    we write means the cleanup certifies the same numbers the exporter
    quantizes."""
    return struct.unpack('<f', struct.pack('<f', x))[0]


def _cleanup_degenerate_tris(verts, tris, inv_mult, max_passes=8):
    """Eliminate triangles that would degenerate under CDB2's int16
    quantization, adapted from Botsch & Kobbelt, "A Robust Procedure to
    Eliminate Degenerate Faces from Triangle Meshes" (VMV 2001).

    The CDB exporter rounds every coordinate to a 32767-step grid and DROPS
    triangles whose quantized cross product is zero (they crash the game),
    which punches collision holes wherever the render mesh has slivers.
    Following the paper's taxonomy:
      - NEEDLES (an edge collapsing to zero on the grid) are removed by
        collapsing that edge to its midpoint - a sub-quantum, invisible
        move; neighbours follow automatically through the shared vertex
        pool, exactly the paper's needle treatment.
      - CAPS (near-collinear triangles whose edges are all long) cannot be
        collapsed without damage; per the paper they must be split at the
        apex. We apply the slice LOCALLY: the long edge is split at the
        apex's orthogonal projection on this triangle AND on every
        neighbour sharing that edge (keeping edges crack-free), turning
        the cap into two needles the next pass collapses. Global plane
        slicing - the paper's fully robust variant - is deliberately NOT
        used: it multiplies the triangle count (their stated drawback),
        and the local form suffices for a collision soup while keeping
        the poly count within one triangle of the input per cap.
    verts: [(x,y,z) blender], tris: [(i,j,k)], inv_mult: game-axis inverse
    multipliers. Returns (verts, tris, stats dict). Geometry moves at most
    half a grid quantum."""
    # work in float32 space: exactly the values Blender will store and the
    # CDB exporter will quantize (see _f32)
    verts = [[_f32(c) for c in v] for v in verts]
    tris = [list(t) for t in tris]
    stats = {"collapsed": 0, "cap_split": 0, "dropped": 0}

    def q(vi):
        v = verts[vi]
        g = (v[0], v[2], v[1])          # blender -> game axes
        return tuple(max(-32767, min(32767, round(g[i] * inv_mult[i])))
                     for i in range(3))

    def degen(t):
        a, b, c = q(t[0]), q(t[1]), q(t[2])
        e1 = (b[0]-a[0], b[1]-a[1], b[2]-a[2])
        e2 = (c[0]-a[0], c[1]-a[1], c[2]-a[2])
        return (e1[1]*e2[2] == e1[2]*e2[1] and
                e1[2]*e2[0] == e1[0]*e2[2] and
                e1[0]*e2[1] == e1[1]*e2[0])

    remap = {}

    def res(i):
        while i in remap:
            i = remap[i]
        return i

    for _pass in range(max_passes):
        tris = [[res(i) for i in t] for t in tris]
        tris = [t for t in tris if t[0] != t[1] and t[1] != t[2] and t[0] != t[2]]
        bad = [ti for ti, t in enumerate(tris) if degen(t)]
        if not bad:
            break
        edge_map = {}
        for ti, t in enumerate(tris):
            for k in range(3):
                e = frozenset((t[k], t[(k+1) % 3]))
                edge_map.setdefault(e, []).append(ti)
        changed = False
        split_edges = {}
        extra = []
        for ti in bad:
            t = tris[ti]
            if t[0] == t[1] or t[1] == t[2] or t[0] == t[2]:
                continue
            qs = [q(i) for i in t]
            el = []
            for k in range(3):
                a, b = qs[k], qs[(k+1) % 3]
                el.append(((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2, k))
            el.sort()
            if el[0][0] <= 2:
                # NEEDLE: collapse the on-grid-zero (or ~1-quantum) edge
                k = el[0][1]
                ia, ib = res(t[k]), res(t[(k+1) % 3])
                if ia == ib:
                    continue
                mid = [_f32((verts[ia][i] + verts[ib][i]) * 0.5)
                       for i in range(3)]
                verts[ia] = mid
                remap[ib] = ia
                stats["collapsed"] += 1
                changed = True
            else:
                # CAP: split the longest edge at the apex projection, on
                # this triangle and every neighbour sharing that edge
                k = el[-1][1]
                ia, ib = t[k], t[(k+1) % 3]
                ekey = frozenset((ia, ib))
                if ekey in split_edges:
                    continue
                A = verts[ia]
                B = verts[ib]
                P = verts[t[(k+2) % 3]]
                ab = [B[i]-A[i] for i in range(3)]
                l2 = ab[0]*ab[0] + ab[1]*ab[1] + ab[2]*ab[2]
                if l2 <= 0:
                    continue
                s = sum((P[i]-A[i]) * ab[i] for i in range(3)) / l2
                s = max(0.05, min(0.95, s))
                newco = [_f32(A[i] + ab[i]*s) for i in range(3)]
                pi = len(verts)
                verts.append(newco)
                split_edges[ekey] = pi
                for nti in edge_map.get(ekey, []):
                    nt = tris[nti]
                    t1 = [pi if x == ib else x for x in nt]
                    t2 = [pi if x == ia else x for x in nt]
                    tris[nti] = t1
                    extra.append(t2)
                stats["cap_split"] += 1
                changed = True
        tris.extend(extra)
        if not changed:
            break
    tris = [[res(i) for i in t] for t in tris]
    tris = [t for t in tris if t[0] != t[1] and t[1] != t[2] and t[0] != t[2]]
    keep = []
    for t in tris:
        if degen(t):
            stats["dropped"] += 1
        else:
            keep.append(tuple(t))
    used = {}
    out_v = []
    out_t = []
    for t in keep:
        nt = []
        for i in t:
            j = used.get(i)
            if j is None:
                j = len(out_v)
                used[i] = j
                out_v.append(tuple(verts[i]))
            nt.append(j)
        out_t.append(tuple(nt))
    return out_v, out_t, stats


# ═════════════════════════════════════════════════════════════════════════════
# ShadowMap painting + generation (shadowmap_w2.dat, 512x512, 255=lit 0=shadow)
# ═════════════════════════════════════════════════════════════════════════════
# Alignment contract (matches fo2_collision_io exactly): the shadowmap spans
# the SYMMETRIC extent [-ext,+ext] per game axis, ext = max absolute
# coordinate of the collision geometry (= 32767 * axis_multiplier). Column =
# game X (blender X), row = game Z (blender Y). File row 0 = north edge;
# blender images store rows bottom-up, so image_row = 511 - file_row (the
# same flip collision_io's export_shadowmap applies). Vanilla values:
# 255 = fully lit, 0 = full shadow, 1-6/248-254 = blur penumbra.
SHADOWMAP_N = 512
# The shadowmap has a FIXED world placement, hardcoded in FlatOut2.exe (the
# 0x8018-byte shadow grid struct built at 0x4B3BC9 is initialized with these
# literals, NOT with any per-track header value):
#   [+0x00] min_x = -1000.0   [+0x04] max_x = +1000.0
#   [+0x0C] min_z = -1000.0   [+0x08] max_z = +1000.0
# So shadowmap_w2.dat ALWAYS covers world X,Z in [-1000, +1000] m, centered
# on the origin, for every track regardless of geometry size. (The per-track
# shadow UVs stored in the CDB are a separate, geometry-normalized thing and
# do NOT govern the .dat's world placement - that was the mistaken
# assumption.) game X = blender X, game Z = blender Y.
SHADOWMAP_WORLD_MIN = -1000.0
SHADOWMAP_WORLD_MAX = 1000.0
SHADOWMAP_WORLD_SPAN = SHADOWMAP_WORLD_MAX - SHADOWMAP_WORLD_MIN  # 2000 m


class _TriRaycaster:
    """Möller-Trumbore ray caster over a triangle soup. Pure-python
    fallback with the same call shape as mathutils.bvhtree; real Blender
    uses the C BVH via _make_raycaster()."""
    def __init__(self, tris):
        self.tris = tris    # [(a,b,c)] world-space tuples

    def ray_cast(self, origin, direction, distance=1e30):
        best_t = distance
        hit = None
        ox, oy, oz = origin
        dx, dy, dz = direction
        for a, b, c in self.tris:
            e1 = (b[0]-a[0], b[1]-a[1], b[2]-a[2])
            e2 = (c[0]-a[0], c[1]-a[1], c[2]-a[2])
            px = dy*e2[2]-dz*e2[1]
            py = dz*e2[0]-dx*e2[2]
            pz = dx*e2[1]-dy*e2[0]
            det = e1[0]*px+e1[1]*py+e1[2]*pz
            if -1e-12 < det < 1e-12:
                continue
            inv = 1.0/det
            tx, ty, tz = ox-a[0], oy-a[1], oz-a[2]
            u = (tx*px+ty*py+tz*pz)*inv
            if u < 0 or u > 1:
                continue
            qx = ty*e1[2]-tz*e1[1]
            qy = tz*e1[0]-tx*e1[2]
            qz = tx*e1[1]-ty*e1[0]
            v = (dx*qx+dy*qy+dz*qz)*inv
            if v < 0 or u+v > 1:
                continue
            t = (e2[0]*qx+e2[1]*qy+e2[2]*qz)*inv
            if 1e-6 < t < best_t:
                best_t = t
                hit = (ox+dx*t, oy+dy*t, oz+dz*t)
        if hit is None:
            return (None, None, -1, None)
        return (hit, None, 0, best_t)


def _make_raycaster(tris):
    """BVH-accelerated caster in real Blender, brute-force fallback in the
    test harness."""
    try:
        from mathutils.bvhtree import BVHTree
        verts = []
        polys = []
        for a, b, c in tris:
            base = len(verts)
            verts.extend((a, b, c))
            polys.append((base, base+1, base+2))
        return BVHTree.FromPolygons(verts, polys)
    except Exception:
        return _TriRaycaster(tris)


def _trace_shadowmap(caster_tris, receiver_tris, light_dir,
                     shadow_value=0, max_length=0.0, blur_passes=1,
                     supersample=2):
    """Trace the 512x512 shadowmap over the FIXED world window
    [-1000,+1000] x [-1000,+1000] (blender X, blender Y). Inputs in BLENDER
    coordinates; light_dir is the direction light TRAVELS (sun -> ground).
    Texel (col,row): world_x = MIN + (col+0.5)/512*SPAN, and the game Z axis
    (blender Y) descends with row - file row 0 is the +Y (north) edge, the
    convention validated against vanilla (Tarmac bright, forest floor dark).
    Returns a bytearray in FILE row order; texels outside the geometry, or
    with no shadow, stay 255 (lit)."""
    N = SHADOWMAP_N
    out = bytearray(b"\xff" * (N * N))
    if not caster_tris:
        return out
    MIN = SHADOWMAP_WORLD_MIN
    SPAN = SHADOWMAP_WORLD_SPAN
    casters = _make_raycaster(caster_tris)
    receivers = _make_raycaster(receiver_tris) if receiver_tris else None
    ld = light_dir
    l = (ld[0]*ld[0]+ld[1]*ld[1]+ld[2]*ld[2]) ** 0.5 or 1.0
    ld = (ld[0]/l, ld[1]/l, ld[2]/l)
    to_light = (-ld[0], -ld[1], -ld[2])
    limit = max_length if max_length and max_length > 0 else 1e30

    # footprint prefilter: only texels under the casters' projected AABB
    cxs = [v[0] for t in caster_tris for v in t]
    cys = [v[1] for t in caster_tris for v in t]
    czs = [v[2] for t in caster_tris for v in t]
    proj = min(limit, 100000.0)
    if receiver_tris:
        floor_z = min(v[2] for t in receiver_tris for v in t)
    else:
        floor_z = min(czs)
    span_z = (max(czs) - floor_z)
    tmax = proj if ld[2] >= -1e-6 else min(proj, span_z / (-ld[2]) + 1.0)
    fx0 = min(min(cxs), min(cxs) + ld[0]*tmax) - 1.0
    fx1 = max(max(cxs), max(cxs) + ld[0]*tmax) + 1.0
    fy0 = min(min(cys), min(cys) + ld[1]*tmax) - 1.0
    fy1 = max(max(cys), max(cys) + ld[1]*tmax) + 1.0

    top_z = max(czs) + 10.0
    # texel <-> world index bounds so we only scan the caster footprint
    col_lo = max(0, int((fx0 - MIN) / SPAN * N))
    col_hi = min(N, int((fx1 - MIN) / SPAN * N) + 1)
    # row runs opposite to +Y: row 0 -> world_y = MAX
    row_for_y = lambda wy: (SHADOWMAP_WORLD_MAX - wy) / SPAN * N
    row_lo = max(0, int(row_for_y(fy1)))
    row_hi = min(N, int(row_for_y(fy0)) + 1)
    # supersampling: SS x SS sub-samples per texel, averaged into a coverage
    # value. In a fixed 2000 m grid each texel is ~3.9 m, so a small map's
    # shadows would otherwise be blocky; sub-sampling recovers soft, accurate
    # edges without changing the (correct) placement. SS=1 -> single ray.
    SS = max(1, int(supersample))
    texel = SPAN / N
    sub = texel / SS
    sub0 = -0.5 * texel + 0.5 * sub          # first sub-sample offset
    lit, shd = 255, shadow_value
    inv_ss2 = 1.0 / (SS * SS)
    for row in range(row_lo, row_hi):
        cy = SHADOWMAP_WORLD_MAX - (row + 0.5) / N * SPAN
        base = row * N
        for col in range(col_lo, col_hi):
            cx = MIN + (col + 0.5) / N * SPAN
            hits = 0
            covered = 0
            for sj in range(SS):
                wy = cy + sub0 + sj * sub
                for si in range(SS):
                    wx = cx + sub0 + si * sub
                    gz = floor_z
                    if receivers is not None:
                        loc, _n, _i, _d = receivers.ray_cast(
                            (wx, wy, top_z), (0.0, 0.0, -1.0),
                            top_z - floor_z + 20.0)
                        if loc is None:
                            continue              # off-geometry sub-sample = lit
                        gz = loc[2]
                    covered += 1
                    loc, _n, _i, _d = casters.ray_cast(
                        (wx, wy, gz + 0.05), to_light, limit)
                    if loc is not None:
                        hits += 1
            if hits:
                # coverage fraction -> value between lit and shadow_value
                frac = hits * inv_ss2
                out[base + col] = int(round(lit + (shd - lit) * frac))
    for _ in range(max(0, blur_passes)):
        srcb = bytes(out)
        for row in range(1, N-1):
            b = row * N
            for col in range(1, N-1):
                s = (srcb[b+col-N-1]+srcb[b+col-N]+srcb[b+col-N+1]
                     + srcb[b+col-1]+srcb[b+col]+srcb[b+col+1]
                     + srcb[b+col+N-1]+srcb[b+col+N]+srcb[b+col+N+1])
                out[b+col] = s // 9
    return out


def _light_direction(light_obj):
    """World direction the light travels (sun points along local -Z)."""
    m = light_obj.matrix_world
    try:
        rows = m.rows
    except AttributeError:
        rows = [list(m[i]) for i in range(3)]
    d = (-rows[0][2], -rows[1][2], -rows[2][2])
    l = (d[0]*d[0]+d[1]*d[1]+d[2]*d[2]) ** 0.5 or 1.0
    return (d[0]/l, d[1]/l, d[2]/l)


def _shadow_scene_meshes(context, painted_only=False):
    """Mesh objects relevant for shadow tracing (same skip rules as the
    organizer). painted_only -> only faces with fo2_shadow_cast set."""
    out = []
    for obj in context.scene.collection.all_objects:
        if obj.type != 'MESH':
            continue
        if obj.data.get('fo2_plant_quads') or obj.data.get('fo2_display_only'):
            continue
        if obj.name.startswith(('BVHPrim', 'BVHNode', 'col_', 'shadowmap')):
            continue
        out.append(obj)
    return out


def _gather_world_tris(obj, painted_only=False):
    me = obj.data
    mw = obj.matrix_world
    try:
        wco = [tuple(mw @ v.co) for v in me.vertices]
    except Exception:
        return []
    paint = None
    if painted_only:
        try:
            a = me.attributes.get("fo2_shadow_cast")
            if a is not None:
                paint = a.data
        except Exception:
            paint = None
        if paint is None:
            return []
    tris = []
    def _emit(pv):
        co = [wco[vi] for vi in pv]
        n = len(co)
        if n == 3:
            tris.append(tuple(co))
        elif n > 3:
            for k in range(1, n-1):
                tris.append((co[0], co[k], co[k+1]))
    for pi, p in enumerate(me.polygons):
        if paint is not None:
            try:
                if not int(paint[pi].value):
                    continue
            except Exception:
                continue
        _emit(list(p.vertices))
    return tris


def _lights_in_scene(context):
    return [o for o in context.scene.collection.all_objects
            if getattr(o, 'type', '') == 'LIGHT']


class FO2_OT_W32PaintShadow(bpy.types.Operator):
    """Paint which faces cast shadows onto the shadowmap.
Pick the light source, then paint faces across any meshes directly in the
3D viewport: Left-drag paints, Ctrl+Left-drag erases, [ and ] change the
brush radius, Esc/Enter/Space finishes. Painted faces show red (the
viewport switches to attribute colors while painting)"""
    bl_idname = "object.fo2_w32_paint_shadow"
    bl_label = "W32: Paint ShadowMap (experimental)"
    bl_options = {'REGISTER', 'UNDO'}

    def _light_items(self, context):
        items = [(o.name, o.name, "") for o in _lights_in_scene(context)]
        return items or [("NONE", "<no lights>", "")]

    light_name: bpy.props.EnumProperty(name="Light source",
                                       items=_light_items)
    brush_radius: bpy.props.FloatProperty(
        name="Brush radius (m)", default=4.0, min=0.0, soft_max=50.0,
        description="Faces whose center lies within this distance of the "
                    "hit point are painted too (0 = single face)")

    def invoke(self, context, event):
        if not _lights_in_scene(context):
            self.report({'ERROR'}, "No light in the scene - add a Sun light "
                                   "to define the shadow direction first")
            return {'CANCELLED'}
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        lights = _lights_in_scene(context)
        if not lights:
            self.report({'ERROR'}, "No light in the scene")
            return {'CANCELLED'}
        name = self.light_name if self.light_name != "NONE" else lights[0].name
        context.scene["fo2_shadow_light"] = name
        self._painting = False
        self._erase = False
        self._centers = {}
        # show attribute colors while painting
        self._shading_backup = None
        try:
            shading = context.space_data.shading
            self._shading_backup = shading.color_type
            shading.color_type = 'VERTEX'
        except Exception:
            pass
        try:
            context.window_manager.modal_handler_add(self)
        except Exception:
            self.report({'INFO'}, "Shadow light set to '%s'" % name)
            return {'FINISHED'}
        self._set_header(context)
        return {'RUNNING_MODAL'}

    def _set_header(self, context):
        try:
            context.area.header_text_set(
                "Paint ShadowMap  |  LMB paint - Ctrl erase - [ ] radius "
                f"({self.brush_radius:.1f}m) - Esc/Enter finish")
        except Exception:
            pass

    def _finish(self, context):
        try:
            if self._shading_backup is not None:
                context.space_data.shading.color_type = self._shading_backup
        except Exception:
            pass
        try:
            context.area.header_text_set(None)
        except Exception:
            pass

    def _ensure_attrs(self, me):
        cast = None
        col = None
        try:
            cast = me.attributes.get("fo2_shadow_cast")
            if cast is None:
                cast = me.attributes.new(name="fo2_shadow_cast", type='INT',
                                         domain='FACE')
        except Exception:
            return None, None
        try:
            col = me.color_attributes.get("fo2_shadow_paint")
            if col is None:
                col = me.color_attributes.new(name="fo2_shadow_paint",
                                              type='BYTE_COLOR',
                                              domain='CORNER')
                for d in col.data:
                    d.color = (1.0, 1.0, 1.0, 1.0)
            try:
                me.color_attributes.active_color = col
            except Exception:
                pass
        except Exception:
            col = None
        return cast, col

    def _paint_at(self, context, event):
        try:
            from bpy_extras import view3d_utils
            region = context.region
            rv3d = context.region_data
            coord = (event.mouse_region_x, event.mouse_region_y)
            origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
            direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
            depsgraph = context.evaluated_depsgraph_get()
            hit, loc, _n, face_index, obj, _m = context.scene.ray_cast(
                depsgraph, origin, direction)
        except Exception:
            return
        if not hit or obj is None or obj.type != 'MESH':
            return
        if obj.data.get('fo2_display_only') or obj.data.get('fo2_plant_quads'):
            return
        me = obj.data
        cast, col = self._ensure_attrs(me)
        if cast is None:
            return
        value = 0 if self._erase else 1
        color = ((1.0, 1.0, 1.0, 1.0) if self._erase
                 else (1.0, 0.15, 0.1, 1.0))
        targets = [face_index] if 0 <= face_index < len(me.polygons) else []
        if self.brush_radius > 0.0:
            # faces of the same object whose center lies within the brush
            key = obj.name
            if key not in self._centers:
                mw = obj.matrix_world
                cs = []
                for p in me.polygons:
                    try:
                        c = mw @ p.center
                        cs.append((c[0], c[1], c[2]))
                    except Exception:
                        cs.append(None)
                self._centers[key] = cs
            cs = self._centers[key]
            r2 = self.brush_radius * self.brush_radius
            lx, ly, lz = loc[0], loc[1], loc[2]
            targets = [i for i, c in enumerate(cs) if c is not None and
                       (c[0]-lx)**2 + (c[1]-ly)**2 + (c[2]-lz)**2 <= r2]
            if 0 <= face_index < len(me.polygons) and face_index not in targets:
                targets.append(face_index)
        for fi in targets:
            try:
                cast.data[fi].value = value
                if col is not None:
                    for li in me.polygons[fi].loop_indices:
                        col.data[li].color = color
            except Exception:
                pass
        try:
            me.update()
        except Exception:
            pass

    def modal(self, context, event):
        if event.type in ('ESC', 'RET', 'SPACE'):
            self._finish(context)
            self.report({'INFO'}, "Shadow painting finished - run "
                                  "'W32: Apply ShadowMap' to trace")
            return {'FINISHED'}

        if event.type == 'LEFT_BRACKET' and event.value == 'PRESS':
            self.brush_radius = max(0.0, self.brush_radius - 1.0)
            self._set_header(context)
            return {'RUNNING_MODAL'}
        if event.type == 'RIGHT_BRACKET' and event.value == 'PRESS':
            self.brush_radius += 1.0
            self._set_header(context)
            return {'RUNNING_MODAL'}
        if event.type == 'LEFTMOUSE':
            self._painting = event.value == 'PRESS'
            self._erase = bool(event.ctrl)
            if self._painting:
                self._paint_at(context, event)
            return {'RUNNING_MODAL'}
        if event.type == 'MOUSEMOVE':
            if self._painting:
                self._erase = bool(event.ctrl)
                self._paint_at(context, event)
                return {'RUNNING_MODAL'}
            return {'PASS_THROUGH'}
        # everything else (MMB orbit, wheel/trackpad zoom & pan, numpad
        # views, shading toggles...) stays fully functional while painting
        return {'PASS_THROUGH'}


class FO2_OT_W32ApplyShadow(bpy.types.Operator):
    """Trace the 512x512 shadowmap from the painted faces and the chosen
light source, and place an aligned preview plane over the map.
Export it via the collision exporter's 'Export shadowmap_w2.dat' option"""
    bl_idname = "object.fo2_w32_apply_shadow"
    bl_label = "W32: Apply ShadowMap (experimental)"
    bl_options = {'REGISTER', 'UNDO'}

    shadow_value: bpy.props.IntProperty(
        name="Shadow value", default=0, min=0, max=254,
        description="Byte written inside shadows (vanilla uses 0; 255 is "
                    "fully lit)")
    max_length: bpy.props.FloatProperty(
        name="Max shadow length (m)", default=0.0, min=0.0, soft_max=2000.0,
        description="Cut shadows past this distance from the caster "
                    "(0 = unlimited)")
    blur_passes: bpy.props.IntProperty(
        name="Softness (blur passes)", default=1, min=0, max=4,
        description="3x3 blur passes producing the soft penumbra fringe "
                    "seen in vanilla files")
    quality: bpy.props.IntProperty(
        name="Edge quality (samples/texel)", default=3, min=1, max=6,
        description="Rays per texel per axis (NxN). The shadowmap is a "
                    "fixed 2000 m / 512 grid (~3.9 m per texel), so higher "
                    "values give smoother, more accurate shadow edges at the "
                    "cost of trace time. 1 = fast/blocky, 3 = balanced")

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        # light
        lights = _lights_in_scene(context)
        if not lights:
            self.report({'ERROR'}, "No light in the scene")
            return {'CANCELLED'}
        lname = str(context.scene.get("fo2_shadow_light", "") or "")
        light = None
        for o in lights:
            if o.name == lname:
                light = o
                break
        if light is None:
            light = lights[0]
        ld = _light_direction(light)
        if ld[2] >= -0.02:
            self.report({'ERROR'}, f"Light '{light.name}' does not point "
                                   "downwards - aim it at the ground")
            return {'CANCELLED'}
        # geometry
        casters = []
        receivers = []
        for obj in _shadow_scene_meshes(context):
            receivers.extend(_gather_world_tris(obj))
            casters.extend(_gather_world_tris(obj, painted_only=True))
        if not casters:
            self.report({'ERROR'}, "No painted shadow casters - run "
                                   "'W32: Paint ShadowMap' first")
            return {'CANCELLED'}
        data = _trace_shadowmap(casters, receivers, ld,
                                self.shadow_value, self.max_length,
                                self.blur_passes, self.quality)
        n_shadow = sum(1 for b in data if b < 250)
        # image (blender row order = flipped file rows, matching
        # collision_io's export_shadowmap round-trip)
        N = SHADOWMAP_N
        img = None
        try:
            img = bpy.data.images.get("fo2_shadowmap")
            if img is not None and tuple(img.size) != (N, N):
                bpy.data.images.remove(img)
                img = None
            if img is None:
                img = bpy.data.images.new("fo2_shadowmap", N, N)
            px = [0.0] * (N * N * 4)
            for irow in range(N):
                frow = N - 1 - irow
                for col in range(N):
                    v = data[frow * N + col] / 255.0
                    o = (irow * N + col) * 4
                    px[o] = px[o+1] = px[o+2] = v
                    px[o+3] = 1.0
            img.pixels[:] = px
            try:
                img.pack()
            except Exception:
                pass
            img.update()
        except Exception as e:
            print(f"[W32 Shadow] image creation unavailable: {e}")
        # keep raw bytes on the scene as the authoritative export source
        context.scene["fo2_shadowmap_hex"] = bytes(data).hex()
        # preview plane, aligned with the symmetric extents
        try:
            old = bpy.data.objects.get("shadowmap_preview")
            if old is not None:
                bpy.data.objects.remove(old, do_unlink=True)
        except Exception:
            pass
        try:
            floor_z = 0.0
            zs = [v[2] for t in receivers for v in t]
            if zs:
                floor_z = min(zs)
            mn, mx = SHADOWMAP_WORLD_MIN, SHADOWMAP_WORLD_MAX
            me = bpy.data.meshes.new("shadowmap_preview")
            # fixed ±1000 m square, centered on origin (game world placement)
            me.from_pydata([(mn, mn, 0), (mx, mn, 0),
                            (mx, mx, 0), (mn, mx, 0)],
                           [], [(0, 1, 2, 3)])
            me["fo2_display_only"] = True
            try:
                uvl = me.uv_layers.new(name="UVMap")
                for i, uv in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):
                    uvl.data[i].uv = uv
            except Exception:
                pass
            try:
                mat = bpy.data.materials.new("shadowmap_preview")
                mat["fo2_display_only"] = True
                mat.use_nodes = True
                nodes = mat.node_tree.nodes
                links = mat.node_tree.links
                nodes.clear()
                out_node = nodes.new("ShaderNodeOutputMaterial")
                bsdf = nodes.new("ShaderNodeBsdfPrincipled")
                links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])
                tex = nodes.new("ShaderNodeTexImage")
                tex.image = img
                tex.interpolation = 'Closest'
                links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
                me.materials.append(mat)
            except Exception:
                pass
            ob = bpy.data.objects.new("shadowmap_preview", me)
            try:
                ob.location.z = floor_z + 0.15
            except Exception:
                pass
            context.scene.collection.objects.link(ob)
        except Exception as e:
            print(f"[W32 Shadow] preview plane unavailable: {e}")
        pct = 100.0 * n_shadow / (N * N)
        self.report({'INFO'}, f"ShadowMap traced: {n_shadow} shadow texels "
                              f"({pct:.1f}%), fixed 2000x2000 m window "
                              f"(+/-1000), light '{light.name}'")
        return {'FINISHED'}


def _rebuild_as_principled(bl_mat):
    """Rebuild the material's node tree as a plain Principled BSDF with the
    texture wired to Base Color - the layout the exporter expects (Mix
    Shader and other special setups break texture/vertex-color handling).
    Keeps the first image found in the old tree. Returns True if rebuilt."""
    try:
        if not getattr(bl_mat, 'use_nodes', False):
            bl_mat.use_nodes = True
        tree = bl_mat.node_tree
        img = None
        for n in tree.nodes:
            if getattr(n, 'type', '') == 'TEX_IMAGE' and getattr(n, 'image', None):
                img = n.image
                break
        types = sorted(getattr(n, 'type', '') for n in tree.nodes)
        if types in (['BSDF_PRINCIPLED', 'OUTPUT_MATERIAL'],
                     ['BSDF_PRINCIPLED', 'OUTPUT_MATERIAL', 'TEX_IMAGE']):
            return False
        tree.nodes.clear()
        out = tree.nodes.new("ShaderNodeOutputMaterial")
        out.location = (300, 0)
        bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (0, 0)
        tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        if img is not None:
            tex = tree.nodes.new("ShaderNodeTexImage")
            tex.location = (-300, 0)
            tex.image = img
            tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        return True
    except Exception as e:
        print("[W32 Organize] principled rebuild failed on %r: %s"
              % (getattr(bl_mat, 'name', '?'), e))
        return False


def _resolve_image_path_generic(img):
    """Existing on-disk path for a Blender image, trying abspath, raw,
    //-stripped, blend-dir variants and packed-image unpacking."""
    raw = (getattr(img, 'filepath', '') or getattr(img, 'filepath_raw', '') or '')
    cands = []
    try:
        cands.append(bpy.path.abspath(raw))
    except Exception:
        pass
    cands.append(raw)
    if raw.startswith('//'):
        cands.append(raw[2:])
    blend_dir = ''
    try:
        blend_dir = os.path.dirname(bpy.data.filepath)
    except Exception:
        pass
    base = os.path.basename(raw.replace('\\', '/'))
    if blend_dir:
        rel = raw[2:] if raw.startswith('//') else raw
        cands.append(os.path.join(blend_dir, rel))
        cands.append(os.path.join(blend_dir, base))
        cands.append(os.path.join(blend_dir, 'Textures', base))
    for c in cands:
        if c and os.path.isfile(c):
            return os.path.normpath(c)
    if getattr(img, 'packed_file', None) is not None and blend_dir:
        try:
            out = os.path.join(blend_dir, base or (img.name + '.png'))
            if not os.path.isfile(out):
                with open(out, 'wb') as f:
                    f.write(img.packed_file.data)
            return out
        except Exception:
            pass
    return None


def _convert_material_textures_to_dds(materials, report=None):
    """Convert every .png/.tga image referenced by the given materials to a
    game-ready .dds beside the source (DXT1, or DXT3 when the material has
    alpha or the image contains non-opaque pixels - the vanilla convention;
    mipmaps on). The .w32 keeps .tga names. This is the SAME logic the W32
    organizer uses. Returns (done, skipped, missing, failed)."""
    png2dds = _load_sibling_module('png2dds')
    tga2dds = _load_sibling_module('tga2dds')
    if png2dds is None and tga2dds is None:
        if report:
            report({'WARNING'}, "png2dds.py/tga2dds.py not found beside the "
                                "plugin - texture conversion skipped")
        return 0, 0, [], []
    done = skip = 0
    missing = []
    failed = []
    seen = set()
    for bl_mat in materials:
        if bl_mat is None or not getattr(bl_mat, 'use_nodes', False):
            continue
        try:
            nodes_it = bl_mat.node_tree.nodes
        except Exception:
            continue
        for n in nodes_it:
            img = getattr(n, 'image', None)
            if img is None:
                continue
            raw = (getattr(img, 'filepath', '') or
                   getattr(img, 'filepath_raw', '') or '')
            ext = os.path.splitext(raw)[1].lower()
            if ext not in ('.png', '.tga'):
                continue
            key = raw.lower()
            if key in seen:
                continue
            seen.add(key)
            fp = _resolve_image_path_generic(img)
            if fp is None:
                missing.append(os.path.basename(raw) or img.name)
                continue
            dds_path = os.path.splitext(fp)[0] + '.dds'
            if os.path.isfile(dds_path):
                skip += 1
                continue
            is_tga = fp.lower().endswith('.tga')
            conv_mod = tga2dds if is_tga else png2dds
            if conv_mod is None:
                failed.append(os.path.basename(fp))
                continue
            fmt = 'DXT1'
            if int(bl_mat.get('bgm_alpha', 0)):
                fmt = 'DXT3'
            else:
                try:
                    if is_tga:
                        _, _, px = tga2dds.read_tga(fp)
                    else:
                        _, _, px = png2dds.read_png(fp)
                    if any(p[3] < 255 for p in px):
                        fmt = 'DXT3'
                except Exception:
                    pass
            try:
                if is_tga:
                    tga2dds.convert_tga_to_dds(fp, dds_path, fmt, mipmaps=True)
                    ok = os.path.isfile(dds_path)
                else:
                    ok = png2dds.process_png(fp, None, fmt, [], True)
                if ok:
                    done += 1
                else:
                    failed.append(os.path.basename(fp))
            except Exception as e:
                failed.append(os.path.basename(fp))
                print(f"[W32] texture convert failed ({os.path.basename(fp)}): {e}")
    return done, skip, missing, failed


def _find_surface_output_node(tree):
    """The node feeding the material output's Surface socket."""
    out = None
    for n in tree.nodes:
        if getattr(n, 'type', '') == 'OUTPUT_MATERIAL':
            if getattr(n, 'is_active_output', True):
                out = n
                break
            out = out or n
    if out is None:
        return None
    try:
        surf = out.inputs.get('Surface')
        if surf and surf.is_linked:
            return surf.links[0].from_node
    except Exception:
        pass
    return None


def _material_has_transparency(tree):
    """True if the surface graph routes through a Transparent BSDF or a Mix
    Shader (alpha-blended multitexture) - the bake then needs an alpha
    channel, and the exported material needs bgm_alpha=1."""
    for n in tree.nodes:
        if getattr(n, 'type', '') in ('BSDF_TRANSPARENT', 'MIX_SHADER',
                                      'BSDF_GLASS'):
            return True
    return False


def _material_is_bakeable(bl_mat):
    """A material worth baking: nodes in use and either more than one image
    texture, or a non-trivial surface node (Mix Shader, Diffuse, etc. - not
    a bare Principled/Image)."""
    if bl_mat is None or not getattr(bl_mat, 'use_nodes', False):
        return False
    try:
        tree = bl_mat.node_tree
    except Exception:
        return False
    imgs = [n for n in tree.nodes
            if getattr(n, 'type', '') == 'TEX_IMAGE' and getattr(n, 'image', None)]
    surf = _find_surface_output_node(tree)
    stype = getattr(surf, 'type', '') if surf is not None else ''
    return len(imgs) > 1 or stype in ('MIX_SHADER', 'BSDF_DIFFUSE',
                                      'BSDF_TRANSPARENT', 'BSDF_GLASS',
                                      'EMISSION', 'ADD_SHADER')


def _bake_materials_to_textures(context, objects, report=None, resolution=1024):
    """Bake each complex material's node graph down to ONE flat texture and
    rewire the material to a plain Principled BSDF sampling that texture,
    so it renders correctly in-game (the .w32 stores one texture per
    material). The baked texture reuses the material's OWN base-texture file
    name (no prefix) so nothing breaks in Blender or in the game - the
    original file is overwritten with the flattened result under a
    baked_textures/ copy, and bgm_texture_0 keeps pointing at the real name.

    Colour is captured with a COMBINED bake using a flat, shadeless setup:
    the world is set to neutral and Cycles' EMIT pass renders the material's
    own emission, so we first wire the flattened surface COLOUR into an
    Emission node. Crucially we bake the colour that feeds the BSDF, never a
    shader socket (wiring a shader into Emission.Color is what produced the
    earlier black-and-white result). Alpha (from a Mix Shader / Transparent
    BSDF) is baked into the image's A channel.

    Cycles-only; a clean no-op elsewhere. Returns the number baked."""
    try:
        import bpy as _bpy
    except Exception:
        return 0
    scene = context.scene
    # verify a real bake op exists (guards the test harness)
    try:
        has_bake = hasattr(bpy.ops.object, 'bake')
    except Exception:
        has_bake = False
    if not has_bake:
        return 0
    prev_engine = None
    try:
        prev_engine = scene.render.engine
        scene.render.engine = 'CYCLES'
        try:
            scene.cycles.samples = 4
            scene.cycles.use_denoising = False
        except Exception:
            pass
    except Exception:
        if report:
            report({'WARNING'}, "Baking needs Cycles - skipped")
        return 0

    blend_dir = ''
    try:
        blend_dir = os.path.dirname(bpy.data.filepath)
    except Exception:
        pass
    out_dir = os.path.join(blend_dir or os.getcwd(), "baked_textures")
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        pass

    def _color_socket_for(tree):
        """Find the COLOUR socket to bake: the Color input of the first
        Diffuse/Principled/Emission BSDF in the graph (the flattened
        _mul/_add/_mask result), NOT a shader socket."""
        surf = _find_surface_output_node(tree)
        # walk: if surf is a Mix Shader, descend into its shader inputs to
        # find a BSDF whose Color input we can read
        candidates = []
        if surf is not None:
            candidates.append(surf)
        for n in tree.nodes:
            if getattr(n, 'type', '') in ('BSDF_DIFFUSE', 'BSDF_PRINCIPLED',
                                          'EMISSION'):
                candidates.append(n)
        for node in candidates:
            for cname in ('Base Color', 'Color'):
                try:
                    sock = node.inputs.get(cname)
                except Exception:
                    sock = None
                if sock is not None and sock.is_linked:
                    return sock.links[0].from_socket
        return None

    def _base_texture_name(bl_mat, tree):
        """The real base texture file name to reuse (prefer bgm_texture_0,
        else the largest/base image node)."""
        t0 = bl_mat.get('bgm_texture_0')
        if t0:
            return os.path.splitext(os.path.basename(str(t0)))[0]
        best = None
        for n in tree.nodes:
            img = getattr(n, 'type', '') == 'TEX_IMAGE' and getattr(n, 'image', None)
            if img:
                nm = os.path.basename(getattr(img, 'filepath', '') or img.name)
                nm = os.path.splitext(nm)[0]
                # prefer a _base/_mul texture as the canonical name
                if best is None or nm.endswith(('_base', '_mul')):
                    best = nm
        return best or strip_blender_suffix(bl_mat.name)

    baked = 0
    done_mats = set()
    for obj in objects:
        if obj.type != 'MESH' or not obj.data.materials:
            continue
        if not obj.data.uv_layers:
            if report:
                report({'WARNING'}, f"{obj.name}: no UV map - skipped for bake")
            continue
        for bl_mat in obj.data.materials:
            if bl_mat is None or bl_mat.name in done_mats:
                continue
            if not _material_is_bakeable(bl_mat):
                continue
            done_mats.add(bl_mat.name)
            tree = bl_mat.node_tree
            nodes = tree.nodes
            links = tree.links
            has_alpha = _material_has_transparency(tree)
            base_name = _base_texture_name(bl_mat, tree)

            # the image DATABLOCK name must carry the .png extension so
            # Blender resolves it in the viewport (bgm_texture_0 below always
            # stays .tga - that is the name the game loads, independent of
            # the on-disk .png the baker writes)
            bake_img = bpy.data.images.new(
                base_name + '.png', resolution, resolution, alpha=has_alpha)
            bake_img.colorspace_settings.name = 'sRGB'

            out_node = None
            for n in nodes:
                if getattr(n, 'type', '') == 'OUTPUT_MATERIAL':
                    out_node = n
                    break
            if out_node is None:
                continue
            surf_in = out_node.inputs.get('Surface')
            orig_from = None
            if surf_in and surf_in.is_linked:
                orig_from = surf_in.links[0].from_socket

            # temporary Emission fed by the flattened COLOUR chain
            emit = nodes.new("ShaderNodeEmission")
            col_src = _color_socket_for(tree)
            if col_src is not None:
                links.new(col_src, emit.inputs['Color'])
            if surf_in is not None:
                links.new(emit.outputs['Emission'], surf_in)

            target = nodes.new("ShaderNodeTexImage")
            target.image = bake_img
            target.select = True
            nodes.active = target

            try:
                bpy.ops.object.select_all(action='DESELECT')
            except Exception:
                pass
            try:
                obj.select_set(True)
                context.view_layer.objects.active = obj
                bpy.ops.object.bake(type='EMIT', use_clear=True, margin=4)
                png_path = os.path.join(out_dir, base_name + '.png')
                bake_img.filepath_raw = png_path
                bake_img.file_format = 'PNG'
                bake_img.save()
                ok = True
            except Exception as e:
                if report:
                    report({'WARNING'}, f"Bake failed for {bl_mat.name}: {e}")
                ok = False

            # restore graph (remove temp emission, relink original surface)
            try:
                nodes.remove(emit)
                if orig_from is not None and surf_in is not None:
                    links.new(orig_from, surf_in)
                nodes.remove(target)
            except Exception:
                pass
            if not ok:
                bpy.data.images.remove(bake_img)
                continue

            # rewire to plain Principled sampling the baked image, keep the
            # REAL texture name (no baked_ prefix) so refs resolve
            _rebuild_as_principled(bl_mat)
            for n in bl_mat.node_tree.nodes:
                if getattr(n, 'type', '') == 'TEX_IMAGE':
                    n.image = bake_img
            bl_mat['bgm_shader_id'] = 0          # Static Prelit: the standard
            bl_mat['bgm_num_textures'] = 1        # single-texture static shader
            bl_mat['bgm_texture_0'] = base_name + '.tga'
            bl_mat['bgm_alpha'] = 1 if has_alpha else 0
            baked += 1
    try:
        if prev_engine is not None:
            scene.render.engine = prev_engine
    except Exception:
        pass
    return baked


class FO2_OT_W32OrganizeScene(bpy.types.Operator):
    """Organize the scene for FlatOut 2 track (.w32) export.
Creates the FO2Track/StaticBatch hierarchy, moves every mesh into
StaticBatch, renames surfaces, applies material conventions and sets the
custom properties the exporter needs. Run this before exporting fully
custom geometry"""
    bl_idname = "object.fo2_w32_organize"
    bl_label = "W32: Organize & set properties"
    bl_options = {'REGISTER', 'UNDO'}

    generate_bvh: bpy.props.BoolProperty(
        name="Generate BVH (track_bvh.gen)",
        description="Write culling data on export. The exporter builds a "
                    "real median-split BVH from the batch bounds; disable "
                    "only if you manage track_bvh.gen yourself",
        default=True,
    )
    apply_conventions: bpy.props.BoolProperty(
        name="Apply material conventions",
        description="Default shader 0, 'alpha' name prefix enables alpha, "
                    "'colormap.tga' in slot 0 marks lightmapped terrain "
                    "(detail texture from the material name), all texture "
                    "names get the .tga extension",
        default=True,
    )
    rename_surfaces: bpy.props.BoolProperty(
        name="Rename meshes to Surface<N>",
        description="Rename organized mesh objects (and their mesh data) to "
                    "the SurfaceN convention used by the importer/exporter",
        default=True,
    )
    split_per_material: bpy.props.BoolProperty(
        name="Split Surface per texture",
        description="Split every multi-material mesh into one mesh per "
                    "distinct TEXTURE before organizing (the w32 format "
                    "allows one material per surface; materials sharing a "
                    "texture are merged into one surface). Use when the map "
                    "is a single mesh with several materials assigned",
        default=True,
    )
    generate_collision: bpy.props.BoolProperty(
        name="Generate collision mesh",
        description="Build a 'collision' collection from the organized "
                    "meshes (slope-aware triangulation, surface types "
                    "classified from texture names and geometry) with the "
                    "properties the fo2_collision_io plugin needs to export "
                    "track_cdb2.gen",
        default=True,
    )
    double_side_walls: bpy.props.BoolProperty(
        name="Double-sided collision for thin walls (experimental)",
        description="This emits a reverse-wound copy of every "
                    "wall-like face that belongs to an open sheet (solids and "
                    "near-horizontal ground are left alone), so those walls "
                    "block from both sides. It only ever ADDS collision.",
        default=False,
    )
    simplify_collision: bpy.props.BoolProperty(
        name="Simplify collision mesh (experimental)",
        description="Merge away tiny and sliver triangles to give the "
                    "collision mesh a lower, cleaner triangle count. Raise "
                    "the tolerance for more reduction",
        default=False,
    )
    simplify_error: bpy.props.FloatProperty(
        name="Simplify tolerance (m)",
        description="How far the collision surface may move. 0.05 is subtle "
                    "(~4 cm mean drift), 0.10 halves the triangle count, 0.30 "
                    "matches vanilla density at ~10 cm drift",
        default=0.10, min=0.0, max=1.0, soft_max=0.5,
    )
    compute_lo_flags: bpy.props.BoolProperty(
        name="Compute collision edge flags (lo_flags)",
        description="Write per-triangle lo_flags marking which edges are "
                    "concave, instead of leaving them all convex. Vanilla "
                    "marks convex edges clear and concave edges set; leaving "
                    "everything clear (the default) is already safe for boxes "
                    "and ramps, this adds the concave markings vanilla uses "
                    "at inside corners. Requires the matching fo2_collision_io",
        default=False,
    )
    bevel_right_angles: bpy.props.BoolProperty(
        name="Bevel 90 degree collision corners (experimental)",
        description="Bevel every convex ~90-degree crease in the generated "
                    "collision mesh into a small chamfer, using Blender's own "
                    "bevel so the mesh stays watertight. Covers both "
                    "horizontal creases (ground-meets-wall, top-of-step) and "
                    "vertical ones (wall corners, block edges). Concave "
                    "inside corners and arbitrarily tilted creases are left "
                    "alone. Runs before the degenerate-face cleanup so the "
                    "chamfer keeps a sane topology",
        default=False,
    )
    bake_materials: bpy.props.BoolProperty(
        name="Bake materials to textures",
        description="Bake each complex material (Mix Shader, _add/_mul/_mask "
                    "multitexture, transparency) down to ONE flat texture and "
                    "rewire it to a plain Static-Prelit Principled BSDF, so "
                    "it renders in-game as authored (the .w32 stores one "
                    "texture per material). The baked result reuses the "
                    "material's real base-texture name and appears on the "
                    "mesh immediately, so you can VERIFY it in the viewport "
                    "before exporting. Requires Cycles; PNGs go to "
                    "baked_textures/ next to the .blend",
        default=False,
    )
    convert_png_textures: bpy.props.BoolProperty(
        name="Convert PNG/TGA textures to DDS",
        description="Convert .png and .tga immediately. "
                    "Can be done on export otherwise.",
        default=False,
    )

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=430)

    def _row(self, layout, prop, icon=None):
        """One property row. The icon is optional and guarded, so if a Blender
        version ever renames it the row still draws without it."""
        if icon:
            try:
                layout.prop(self, prop, icon=icon)
                return
            except Exception:
                pass
        layout.prop(self, prop)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = False
        layout.use_property_decorate = False

        # ── scene hierarchy ────────────────────────────────────────────────
        box = layout.box()
        box.label(text="Scene hierarchy", icon='OUTLINER_OB_GROUP_INSTANCE')
        col = box.column(align=True)
        self._row(col, "rename_surfaces", 'SORTALPHA')
        self._row(col, "split_per_material", 'MOD_EXPLODE')
        col.separator()
        self._row(col, "generate_bvh", 'MOD_MESHDEFORM')

        # ── materials and textures ─────────────────────────────────────────
        box = layout.box()
        box.label(text="Materials & textures", icon='MATERIAL')
        col = box.column(align=True)
        self._row(col, "apply_conventions", 'CHECKMARK')
        col.separator()
        self._row(col, "bake_materials", 'RENDER_STILL')
        self._row(col, "convert_png_textures", 'IMAGE_DATA')

        # ── collision ──────────────────────────────────────────────────────
        box = layout.box()
        box.label(text="Collision mesh", icon='PHYSICS')
        self._row(box, "generate_collision", 'MOD_PHYSICS')

        sub = box.column(align=True)
        sub.enabled = self.generate_collision      # greyed out when off
        self._row(sub, "double_side_walls", 'MOD_SOLIDIFY')
        self._row(sub, "bevel_right_angles", 'MOD_BEVEL')
        sub.separator()
        self._row(sub, "simplify_collision", 'MOD_DECIM')
        tol = sub.row(align=True)
        tol.enabled = self.generate_collision and self.simplify_collision
        tol.prop(self, "simplify_error")
        sub.separator()
        self._row(sub, "compute_lo_flags", 'MOD_EDGESPLIT')


    def execute(self, context):
        # 1. find-or-create the root + StaticBatch hierarchy
        root = None
        for c in context.scene.collection.children:
            if any(ch.name == "StaticBatch" for ch in c.children):
                root = c
                break
        if root is None:
            root = bpy.data.collections.new("FO2Track")
            context.scene.collection.children.link(root)
        static_col = None
        for ch in root.children:
            if ch.name == "StaticBatch":
                static_col = ch
                break
        if static_col is None:
            static_col = bpy.data.collections.new("StaticBatch")
            root.children.link(static_col)

        # 2. gather meshes to organize: every MESH object not already inside
        #    the root hierarchy (and not display-only helpers)
        def _in_tree(col, obj):
            if obj.name in {o.name for o in col.objects}:
                return True
            return any(_in_tree(ch, obj) for ch in col.children)

        moved = 0
        unparented = 0
        organized = []
        for obj in list(context.scene.collection.all_objects):
            if obj.type != 'MESH':
                continue
            if obj.data.get('fo2_plant_quads') or obj.data.get('fo2_display_only'):
                continue
            if obj.name.startswith(('BVHPrim', 'BVHNode', 'col_')):
                continue
            # Collada/FBX imports often parent meshes to empties. Bake the
            # parent transform into the object (clear parent, KEEP transform)
            # so the empties can be deleted safely afterwards and the
            # exporter sees plain world-space meshes.
            if getattr(obj, 'parent', None) is not None:
                try:
                    wm = obj.matrix_world.copy()
                except Exception:
                    wm = obj.matrix_world
                obj.parent = None
                obj.matrix_world = wm
                unparented += 1
            if _in_tree(root, obj):
                if _in_tree(static_col, obj):
                    organized.append(obj)
                continue
            for uc in list(obj.users_collection):
                try:
                    uc.objects.unlink(obj)
                except Exception:
                    pass
            static_col.objects.link(obj)
            organized.append(obj)
            moved += 1

        # 2.5 split multi-material meshes into one surface per TEXTURE
        #     (materials that resolve to the same texture name are merged)
        split_from = 0
        split_to = 0
        if self.split_per_material:
            for obj in list(organized):
                parts = _split_mesh_per_texture(obj, static_col)
                if not parts:
                    continue
                idx = organized.index(obj)
                organized[idx:idx + 1] = parts
                split_from += 1
                split_to += len(parts)
                # unlink from every collection first (guaranteed), then drop
                # the datablock (best effort - real Blender frees it)
                for uc in list(getattr(obj, 'users_collection', []) or []):
                    try:
                        uc.objects.unlink(obj)
                    except Exception:
                        pass
                try:
                    static_col.objects.unlink(obj)
                except Exception:
                    pass
                try:
                    bpy.data.objects.remove(obj, do_unlink=True)
                except Exception:
                    pass

        # 3. rename + per-mesh properties
        existing = len([o for o in static_col.objects if o.type == 'MESH'])
        renamed = 0
        no_uv = 0
        # rename in collection order so SurfaceN matches the surface index
        # the exporter will assign (it iterates the collection in order)
        order = [o for o in static_col.objects if o in organized] or organized
        for i, obj in enumerate(order):
            mesh = obj.data
            if self.rename_surfaces:
                new_name = f"Surface{i}"
                if obj.name != new_name:
                    obj.name = new_name
                    renamed += 1
                mesh.name = new_name
            mesh["fo2_poly_mode"] = 4          # triangle list
            if not mesh.materials:
                bl_mat = bpy.data.materials.new(f"{obj.name}_mat")
                mesh.materials.append(bl_mat)
            try:
                if not mesh.uv_layers:
                    no_uv += 1
            except Exception:
                pass

        # 3.5 optionally bake complex materials to flat textures FIRST, so the
        # conventions/property pass below sees the simplified Principled result
        baked = 0
        if getattr(self, 'bake_materials', False):
            try:
                baked = _bake_materials_to_textures(context, organized,
                                                    report=self.report)
            except Exception as e:
                self.report({'WARNING'}, f"Bake step failed: {e}")
                baked = 0

        # 4. material conventions + full property set
        conv = 0
        principled = 0
        seen = set()
        mat_order = []
        for obj in organized:
            for bl_mat in obj.data.materials:
                if bl_mat is None or bl_mat.name in seen:
                    continue
                seen.add(bl_mat.name)
                mat_order.append(bl_mat)
                if self.apply_conventions:
                    if _apply_material_conventions(bl_mat):
                        conv += 1
                else:
                    ensure_material_properties(bl_mat)
        # fo2_material_index: the material's slot in the export table.
        # gather_materials registers materials in the same mesh/slot order, so
        # sequential assignment here matches the exported indices. Imported
        # materials keep their original index (it keys the raw-overlay path).
        next_idx = 0
        used_idx = {int(m["fo2_material_index"]) for m in mat_order
                    if "fo2_material_index" in m}
        for bl_mat in mat_order:
            if "fo2_material_index" not in bl_mat:
                while next_idx in used_idx:
                    next_idx += 1
                bl_mat["fo2_material_index"] = next_idx
                used_idx.add(next_idx)

        # 5. PNG/TGA -> DDS conversion for referenced textures. The .dds is
        #    written NEXT TO the source .png. Path resolution covers Blender-
        #    relative ('//...') paths, absolute paths, paths relative to the
        #    .blend, and packed images (unpacked beside the .blend first).
        #    Every skipped image is reported so nothing fails silently.
        png_done = 0
        png_skip = 0
        png_missing = []
        png_failed = []
        if self.convert_png_textures:
            png2dds = _load_sibling_module('png2dds')
            tga2dds = _load_sibling_module('tga2dds')
            if png2dds is None and tga2dds is None:
                self.report({'WARNING'}, "png2dds.py/tga2dds.py not found "
                                         "beside the plugin - texture "
                                         "conversion skipped")
            else:
                def _resolve_image_path(img):
                    """Return an existing on-disk path for the image, trying
                    every sensible interpretation of its filepath."""
                    raw = (getattr(img, 'filepath', '') or
                           getattr(img, 'filepath_raw', '') or '')
                    cands = []
                    try:
                        cands.append(bpy.path.abspath(raw))
                    except Exception:
                        pass
                    cands.append(raw)
                    if raw.startswith('//'):
                        cands.append(raw[2:])
                    blend_dir = ''
                    try:
                        blend_dir = os.path.dirname(bpy.data.filepath)
                    except Exception:
                        pass
                    base = os.path.basename(raw.replace('\\', '/'))
                    if blend_dir:
                        rel = raw[2:] if raw.startswith('//') else raw
                        cands.append(os.path.join(blend_dir, rel))
                        cands.append(os.path.join(blend_dir, base))
                        cands.append(os.path.join(blend_dir, 'Textures', base))
                    for c in cands:
                        if c and os.path.isfile(c):
                            return os.path.normpath(c)
                    # packed image: unpack a copy beside the .blend
                    if getattr(img, 'packed_file', None) is not None and blend_dir:
                        try:
                            out = os.path.join(blend_dir, base or (img.name + '.png'))
                            if not os.path.isfile(out):
                                with open(out, 'wb') as f:
                                    f.write(img.packed_file.data)
                                print(f"[W32 Organize] unpacked '{img.name}' "
                                      f"-> {out}")
                            return out
                        except Exception as e:
                            print(f"[W32 Organize] could not unpack "
                                  f"'{img.name}': {e}")
                    return None

                seen_png = set()
                for obj in organized:
                    for bl_mat in obj.data.materials:
                        if bl_mat is None or not getattr(bl_mat, 'use_nodes', False):
                            continue
                        try:
                            nodes_it = bl_mat.node_tree.nodes
                        except Exception:
                            continue
                        for n in nodes_it:
                            img = getattr(n, 'image', None)
                            if img is None:
                                continue
                            raw = (getattr(img, 'filepath', '') or
                                   getattr(img, 'filepath_raw', '') or '')
                            ext = os.path.splitext(raw)[1].lower()
                            if ext not in ('.png', '.tga'):
                                continue
                            key = raw.lower()
                            if key in seen_png:
                                continue
                            seen_png.add(key)
                            fp = _resolve_image_path(img)
                            if fp is None:
                                png_missing.append(os.path.basename(raw) or img.name)
                                print(f"[W32 Organize] texture not found on disk: "
                                      f"{raw!r} (image '{img.name}')")
                                continue
                            dds_path = os.path.splitext(fp)[0] + '.dds'
                            if os.path.isfile(dds_path):
                                png_skip += 1
                                continue
                            is_tga = fp.lower().endswith('.tga')
                            conv_mod = tga2dds if is_tga else png2dds
                            if conv_mod is None:
                                png_failed.append(os.path.basename(fp))
                                continue
                            # vanilla convention: DXT3 for textures that carry
                            # alpha, DXT1 otherwise (survey: alpha_* 253xDXT3,
                            # others 1830xDXT1)
                            fmt = 'DXT1'
                            if int(bl_mat.get('bgm_alpha', 0)):
                                fmt = 'DXT3'
                            else:
                                try:
                                    if is_tga:
                                        _, _, px = tga2dds.read_tga(fp)
                                    else:
                                        _, _, px = png2dds.read_png(fp)
                                    if any(p[3] < 255 for p in px):
                                        fmt = 'DXT3'
                                except Exception:
                                    pass
                            try:
                                if is_tga:
                                    # direct TGA -> DDS (mipmaps on, matching
                                    # the png2dds fallback path)
                                    tga2dds.convert_tga_to_dds(fp, dds_path,
                                                               fmt, mipmaps=True)
                                    ok = os.path.isfile(dds_path)
                                else:
                                    ok = png2dds.process_png(fp, None, fmt, [], True)
                                if ok:
                                    png_done += 1
                                    print(f"[W32 Organize] {os.path.basename(fp)}"
                                          f" -> {os.path.basename(dds_path)} ({fmt})")
                                else:
                                    png_failed.append(os.path.basename(fp))
                            except Exception as e:
                                png_failed.append(os.path.basename(fp))
                                print(f"[W32 Organize] convert failed "
                                      f"({os.path.basename(fp)}): {e}")

        # 6. BVH preview collection (mirrors what the exporter will write).
        #    An existing TrackBVH from a vanilla import is left untouched;
        #    only our own tagged preview is rebuilt.
        bvh_note = ""
        if self.generate_bvh:
            existing = None
            for ch in root.children:
                if ch.name.startswith("TrackBVH"):
                    existing = ch
                    break
            if existing is not None and not existing.get("fo2_bvh_preview"):
                bvh_note = " (imported TrackBVH kept as-is)"
            else:
                if existing is not None:
                    for sub in list(existing.children):
                        for o in list(sub.objects):
                            try:
                                bpy.data.objects.remove(o, do_unlink=True)
                            except Exception:
                                try:
                                    sub.objects.unlink(o)
                                except Exception:
                                    pass
                        try:
                            existing.children.unlink(sub)
                        except Exception:
                            try:
                                existing.children.remove(sub)
                            except Exception:
                                pass
                    bvh_col = existing
                else:
                    bvh_col = bpy.data.collections.new("TrackBVH")
                    root.children.link(bvh_col)
                bvh_col["fo2_bvh_preview"] = True
                items = []
                for i, obj in enumerate(organized):
                    mesh = obj.data
                    try:
                        mw = obj.matrix_world
                        xs = []; ys = []; zs = []
                        for v in mesh.vertices:
                            wv = mw @ v.co
                            xs.append(wv[0]); ys.append(wv[1]); zs.append(wv[2])
                        if not xs:
                            continue
                        # blender AABB -> FO2 (x, y-up, z): swap Y/Z
                        c = ((min(xs)+max(xs))*0.5, (min(zs)+max(zs))*0.5,
                             (min(ys)+max(ys))*0.5)
                        r = ((max(xs)-min(xs))*0.5, (max(zs)-min(zs))*0.5,
                             (max(ys)-min(ys))*0.5)
                        items.append((c, r, i))
                    except Exception:
                        continue
                prims, nodes_ = _build_bvh_preview_tree(items)
                box = _bvh_preview_box_mesh()
                pc = bpy.data.collections.new("BVH_Primitives")
                bvh_col.children.link(pc)
                for pi, (pos, rad, id1, id2) in enumerate(prims):
                    o = bpy.data.objects.new(f"BVHPrim{pi}", box)
                    o.location = (pos[0], pos[2], pos[1])
                    o.scale = (max(rad[0], 0.1), max(rad[2], 0.1), max(rad[1], 0.1))
                    try:
                        o.display_type = 'WIRE'
                    except Exception:
                        pass
                    o["fo2_bvh_id1"] = id1
                    o["fo2_bvh_id2"] = id2
                    pc.objects.link(o)
                nc = bpy.data.collections.new("BVH_Nodes")
                bvh_col.children.link(nc)
                for ni, (pos, rad, u1, u2) in enumerate(nodes_):
                    o = bpy.data.objects.new(f"BVHNode{ni}", box)
                    o.location = (pos[0], pos[2], pos[1])
                    o.scale = (max(rad[0], 0.1), max(rad[2], 0.1), max(rad[1], 0.1))
                    try:
                        o.display_type = 'WIRE'
                    except Exception:
                        pass
                    o["fo2_bvh_unk1"] = u1
                    o["fo2_bvh_unk2"] = u2
                    nc.objects.link(o)
                bvh_note = f", BVH preview: {len(prims)} prims/{len(nodes_)} nodes"
                if not prims:
                    print("[W32 Organize] WARNING: BVH preview empty - no "
                          "organized meshes had vertices")

        # 6.5 collision mesh generation
        col_note = ""
        if self.generate_collision:
            try:
                col_note = ", " + _generate_collision(context, organized,
                    getattr(self, "bevel_right_angles", False),
                    getattr(self, "compute_lo_flags", False),
                    getattr(self, "simplify_collision", False),
                    getattr(self, "simplify_error", 0.10),
                    getattr(self, "double_side_walls", True))
            except Exception as e:
                col_note = f", collision generation FAILED: {e}"
                print(f"[W32 Organize] collision generation failed: {e}")

        # 7. export-time options carried on the root collection
        root["fo2_write_bvh"] = bool(self.generate_bvh)

        msg = (f"Organized {len(organized)} meshes ({moved} moved, "
               f"{unparented} unparented, {renamed} renamed, "
               f"{conv} materials updated")
        if getattr(self, 'bake_materials', False) and baked:
            msg += ", %d baked to flat textures" % baked
        if self.split_per_material and split_from:
            msg += f", {split_from} meshes split into {split_to}"
        if self.convert_png_textures:
            msg += f", {png_done} textures->DDS"
            if png_skip:
                msg += f" ({png_skip} already had DDS)"
            if png_missing:
                msg += f", {len(png_missing)} textures not found on disk"
            if png_failed:
                msg += f", {len(png_failed)} conversions FAILED"
        msg += ")" + bvh_note + col_note
        if no_uv:
            msg += f" - WARNING: {no_uv} meshes have no UV layer"
        warn = bool(no_uv or png_missing or png_failed)
        self.report({'WARNING' if warn else 'INFO'}, msg)
        if png_missing:
            print(f"[W32 Organize] textures not found: {', '.join(png_missing[:10])}"
                  + (" ..." if len(png_missing) > 10 else ""))
        print(f"[W32 Organize] {msg}; BVH generation "
              f"{'ON' if self.generate_bvh else 'OFF'}")
        return {'FINISHED'}


def menu_func_object_w32(self, context):
    self.layout.separator()
    self.layout.operator(FO2_OT_W32OrganizeScene.bl_idname)
    self.layout.operator(FO2_OT_W32PaintShadow.bl_idname)
    self.layout.operator(FO2_OT_W32ApplyShadow.bl_idname)


class EXPORT_OT_fo2_w32(bpy.types.Operator, ExportHelper):
    """Export FlatOut 2 W32 Track Geometry"""
    bl_idname = "export_scene.fo2_w32"
    bl_label = "Export FlatOut 2 Track (.w32)"
    bl_options = {'REGISTER', 'UNDO', 'PRESET'}

    filename_ext = ".w32"
    filter_glob: StringProperty(default="*.w32", options={'HIDDEN'})

    original_filepath: StringProperty(
        name="Original W32 (optional)",
        description="If set, overlay mode: re-parse this file and update transforms. "
                    "If empty, generate W32 from scratch from the Blender scene",
        default="",
        subtype='FILE_PATH',
    )
    convert_textures_to_dds: BoolProperty(
        name="Convert TGA/PNG to DDS",
        description="After export, convert the .tga/.png textures the "
                    "materials reference to game-ready .dds beside the "
                    "sources (DXT3 for alpha textures, DXT1 otherwise - the "
                    "vanilla convention, same as the W32 organizer). The "
                    ".w32 keeps .tga names; the game maps them to the .dds",
        default=False,
    )

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        box = layout.box()
        box.label(text="Mode", icon='FILE')
        box.prop(self, "original_filepath")
        if self.original_filepath:
            box.label(text="Overlay mode: updates transforms only", icon='INFO')
        else:
            box.label(text="From-scratch: generates full W32", icon='INFO')

        box = layout.box()
        box.label(text="Textures", icon='TEXTURE')
        box.prop(self, "convert_textures_to_dds")

    def execute(self, context):
        options = {
            'original_filepath': self.original_filepath,
            'convert_textures_to_dds': self.convert_textures_to_dds,
        }
        return export_w32(context, self.filepath, options)


def menu_func_export(self, context):
    self.layout.operator(EXPORT_OT_fo2_w32.bl_idname, text="FlatOut 2 Track (.w32)")

def register():
    bpy.utils.register_class(EXPORT_OT_fo2_w32)
    bpy.utils.register_class(FO2_OT_W32OrganizeScene)
    bpy.utils.register_class(FO2_OT_W32PaintShadow)
    bpy.utils.register_class(FO2_OT_W32ApplyShadow)
    bpy.types.VIEW3D_MT_object.append(menu_func_object_w32)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)

def unregister():
    bpy.types.VIEW3D_MT_object.remove(menu_func_object_w32)
    bpy.utils.unregister_class(FO2_OT_W32ApplyShadow)
    bpy.utils.unregister_class(FO2_OT_W32PaintShadow)
    bpy.utils.unregister_class(FO2_OT_W32OrganizeScene)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    bpy.utils.unregister_class(EXPORT_OT_fo2_w32)

if __name__ == "__main__":
    register()
