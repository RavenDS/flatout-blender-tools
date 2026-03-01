"""
FlatOut 2 Track Collision (track_cdb2.gen) Import/Export with Shadowmap Support.

Note: The CDB2 file header contains axis_multipliers that define the coordinate mapping
between world space and the int16 vertex range. If the
multipliers change during re-export, the shadowmap breaks.

This plugin preserves the original multipliers through import->edit->export by
storing them as custom properties on the Blender collection. If geometry exceeds
the original bounds (requiring new multipliers), the shadowmap is auto-remapped.

Coordinate systems:
  FO2 game:  X right, Y up, Z forward    (Y-up)
  Blender:   X right, Y forward, Z up    (Z-up)
  Conversion: game(x,y,z) -> blender(x, z, y)

CDB2 structure based on work by @mrwonko: https://github.com/mrwonko/flatout-open-level-editor
"""

import bpy
import os
import struct
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union, cast
from mathutils import Vector


# CONSTANTS

HEADER_SIZE = 64
NODE_SIZE = 8
FILE_MAGIC = b'\x21\x76\x71\x98'
COORDINATE_BITS = 16
MAX_ABS_COORDINATE = (1 << (COORDINATE_BITS - 1)) - 1  # 32767

AXIS_BITS = 2
LEAF_AXIS = 3
INNER_MASK_BITS = 6
LEAF_MASK_BITS = 4
LEAF_KIND_BITS = 3
LEAF_TRIANGLE_OFS_BITS = 23
LEAF_TRIANGLE_COUNT_BITS = 7
LEAF_FLAGS_BITS = 6
LEAF_VERTEX_OFS_BITS = 19

MAX_LEAF_TRIANGLES = (1 << LEAF_TRIANGLE_COUNT_BITS) - 1  # 127 (format max)
TARGET_LEAF_TRIANGLES = 10  # (leaves up to 10 tris, could be edited)

SHADOWMAP_SIZE = 512

FO2_SURFACE_NAMES = [
    "NoCollision", "Tarmac", "Tarmac_Mark", "Asphalt", "Asphalt_Mark",
    "Cement", "Cement_Mark", "Hard", "Hard_Mark", "Medium", "Medium_Mark",
    "Soft", "Soft_Mark", "Derby_Gravel", "Derby_Tarmac", "Snow", "Snow_Mark",
    "Dirt", "Dirt_Mark", "Bridge_Metal", "Bridge_Wooden", "Curb", "Bank_Sand",
    "Grass", "Forest", "Sand", "Rock_Terrain", "Mould", "Snow_Terrain",
    "Field", "Wet", "Concrete", "Rock_Object", "Metal", "Wood", "Tree",
    "Bush", "Rubber", "Water", "River", "Puddle", "No_Camera_Col",
    "Camera_Only_Col", "Reset", "Stunt_Conveyer", "Stunt_Bouncer",
    "Stunt_Curling", "Stunt_Bowling", "Stunt_Tarmac", "Oil",
]

SURFACE_COLORS = {
    "road":    (0.30, 0.30, 0.30, 1.0),
    "terrain": (0.25, 0.50, 0.15, 1.0),
    "object":  (0.55, 0.40, 0.25, 1.0),
    "water":   (0.15, 0.30, 0.70, 1.0),
    "special": (0.80, 0.20, 0.60, 1.0),
    "nocol":   (1.00, 0.00, 1.00, 0.5),
}


def _surface_category(sid):
    if sid == 1: return "nocol"
    if 2 <= sid <= 21: return "road"
    if 22 <= sid <= 31: return "terrain"
    if 32 <= sid <= 38: return "object"
    if 39 <= sid <= 41: return "water"
    return "special"


def _surface_name(sid):
    i = sid - 1
    return FO2_SURFACE_NAMES[i] if 0 <= i < len(FO2_SURFACE_NAMES) else f"Unknown_{sid}"


def _ones(n):
    return (1 << n) - 1


# CDB2 PARSING (IMPORT)

@dataclass
class CDB2Header:
    mins: tuple
    maxs: tuple
    axis_multipliers: tuple
    inverse_axis_multipliers: tuple
    ofs_triangles: int
    ofs_vertices: int
    file_size: int

    @staticmethod
    def read(f, file_size):
        magic = f.read(4)
        if magic != FILE_MAGIC:
            raise ValueError(f"Bad CDB2 magic: {magic.hex()}")
        version = struct.unpack("<I", f.read(4))[0]
        if version != 0:
            raise ValueError(f"Unexpected CDB2 version: {version}")
        mins = struct.unpack("<3i", f.read(12))
        maxs = struct.unpack("<3i", f.read(12))
        axis_mult = struct.unpack("<3f", f.read(12))
        inv_axis_mult = struct.unpack("<3f", f.read(12))
        rel_tri = struct.unpack("<I", f.read(4))[0]
        rel_vert = struct.unpack("<I", f.read(4))[0]
        return CDB2Header(
            mins=mins, maxs=maxs,
            axis_multipliers=axis_mult,
            inverse_axis_multipliers=inv_axis_mult,
            ofs_triangles=HEADER_SIZE + rel_tri,
            ofs_vertices=HEADER_SIZE + rel_vert,
            file_size=file_size)

    @property
    def symmetric_extent(self):
        return tuple(MAX_ABS_COORDINATE * self.axis_multipliers[i] for i in range(3))


@dataclass
class TreeNode:
    axis: int
    bitmask: int
    leaf_kind: int = 0
    leaf_flags: int = 0
    triangle_offset: int = 0
    num_triangles: int = 0
    vert_offset: int = 0
    child0_index: int = 0
    split_max: int = 0
    split_min: int = 0

    @property
    def is_leaf(self):
        return self.axis == 3

    @staticmethod
    def read(data, index):
        lo = struct.unpack_from("<I", data)[0]
        axis = lo & 3; lo >>= 2
        if axis == 3:
            bitmask = lo & _ones(4); lo >>= 4
            kind = lo & _ones(3); lo >>= 3
            tri_ofs = lo
            hi = struct.unpack_from("<I", data, 4)[0]
            num_tris = hi & _ones(7); hi >>= 7
            flags = hi & _ones(6); hi >>= 6
            vert_ofs = hi
            return TreeNode(axis=3, bitmask=bitmask, leaf_kind=kind,
                            leaf_flags=flags, triangle_offset=tri_ofs,
                            num_triangles=num_tris, vert_offset=vert_ofs)
        else:
            bitmask = lo & _ones(6); lo >>= 6
            child_offset = lo
            split_max, split_min = struct.unpack_from("<2h", data, 4)
            return TreeNode(axis=axis, bitmask=bitmask,
                            child0_index=child_offset // NODE_SIZE,
                            split_max=split_max, split_min=split_min)


@dataclass
class ImportTriangle:
    surface_id: int
    lo_flags: int
    hi_flags: int
    bitmask: int
    vert_indices: tuple


def _decode_triangles(node, tri_data):
    triangles = []
    vo = node.vert_offset
    pos = node.triangle_offset

    def g(i=0):
        return tri_data[pos + i]

    def _tri(surface_0based, lo, hi, v0, v1, v2):
        triangles.append(ImportTriangle(
            surface_id=(surface_0based & _ones(6)) + 1,
            lo_flags=lo & _ones(6), hi_flags=hi & _ones(2),
            bitmask=node.bitmask, vert_indices=(v0, v1, v2)))

    kind = node.leaf_kind
    lf = node.leaf_flags

    if kind == 0:
        _tri(lf, g(0) & _ones(6), g(0) >> 6, vo,
             g(1) | g(2) << 8 | (g(3) & _ones(3)) << 16,
             g(3) >> 3 | g(4) << 5 | g(5) << 13)
        pos += 6
        for _ in range(1, node.num_triangles):
            _tri(g(1) & _ones(6), g(0) & _ones(6), g(0) >> 6,
                 g(1) >> 7 | g(2) << 1 | g(3) << 9 | (g(4) & _ones(2)) << 17,
                 g(4) >> 2 | g(5) << 6 | (g(6) & _ones(5)) << 14,
                 g(6) >> 5 | g(7) << 3 | g(8) << 11)
            pos += 9
    elif kind == 1:
        _tri(lf, g(0) & _ones(6), g(0) >> 6, vo,
             g(1) | g(2) << 8 | (g(3) & _ones(3)) << 16,
             g(3) >> 3 | g(4) << 5 | g(5) << 13)
        pos += 6
        for _ in range(1, node.num_triangles):
            _tri(lf, g(0) & _ones(6), (g(0) >> 6) & _ones(1),
                 g(0) >> 7 | g(1) << 1 | g(2) << 9 | (g(3) & _ones(2)) << 17,
                 g(3) >> 2 | g(4) << 6 | (g(5) & _ones(5)) << 14,
                 g(5) >> 5 | g(6) << 3 | g(7) << 11)
            pos += 8
    elif kind == 2:
        for _ in range(node.num_triangles):
            _tri(g(1) & _ones(6), g(0) & _ones(6), g(0) >> 6,
                 vo + g(2), vo + g(3), vo + g(4))
            pos += 5
    elif kind == 3:
        for _ in range(node.num_triangles):
            _tri(lf, g(0) & _ones(6), g(0) >> 6,
                 vo + g(1), vo + g(2), vo + g(3))
            pos += 4
    elif kind == 4:
        for _ in range(node.num_triangles):
            _tri(lf, g(0) & _ones(6), (g(0) >> 6) & _ones(1),
                 vo + (g(0) >> 7 | g(1) << 1 | (g(2) & _ones(2)) << 9),
                 vo + (g(2) >> 2 | (g(3) & _ones(5)) << 6),
                 vo + (g(3) >> 5 | g(4) << 3))
            pos += 5
    elif kind == 5:
        for _ in range(node.num_triangles):
            _tri(lf, g(0) & _ones(6), g(0) >> 6,
                 vo + (g(1) & _ones(5)),
                 vo + (g(1) >> 5 | (g(2) & _ones(2)) << 3),
                 vo + (g(2) >> 2))
            pos += 3
    return triangles


@dataclass
class CDB2Data:
    header: CDB2Header
    triangles: list
    vertex_coords: list  # flat int16 array (indexed by int16 offset)
    vertex_shadow_uvs: dict  # int16_idx -> uint32 (only for stride-5 verts)

    def vertex_to_blender(self, idx):
        m = self.header.axis_multipliers
        x = self.vertex_coords[idx] * m[0]
        y = self.vertex_coords[idx + 1] * m[1]
        z = self.vertex_coords[idx + 2] * m[2]
        return (x, z, y)


def _compute_vertex_shadow_uvs(all_tris, vert_data_raw):
    """Detect stride-5 vertices (4 extra shadow UV bytes) from triangle indices.

    Vertices are packed as either 3 int16s (xyz, stride 3) or 5 int16s
    (xyz + 2 uint16 shadow UVs, stride 5). The gap between consecutive
    sorted vertex indices reveals the stride.

    Returns dict mapping int16_idx -> uint32 for stride-5 vertices.
    """
    # collect all unique vertex int16 indices
    all_vidx = set()
    for tri in all_tris:
        for v in tri.vert_indices:
            all_vidx.add(v)

    if not all_vidx:
        return {}

    sorted_idx = sorted(all_vidx)

    # compute stride from gaps between consecutive vertex indices
    shadow_uvs = {}
    for i in range(len(sorted_idx) - 1):
        gap = sorted_idx[i + 1] - sorted_idx[i]
        if gap == 5:
            # this vertex has 4 extra bytes (2 uint16s) after xyz
            byte_ofs = sorted_idx[i] * 2 + 6
            if byte_ofs + 4 <= len(vert_data_raw):
                uv_u32 = struct.unpack_from("<I", vert_data_raw, byte_ofs)[0]
                shadow_uvs[sorted_idx[i]] = uv_u32

    n_s5 = len(shadow_uvs)
    n_s3 = len(sorted_idx) - n_s5
    print(f"CDB2 vertices: {len(sorted_idx)} unique ({n_s5} with shadow UV, {n_s3} without)")
    return shadow_uvs


def parse_cdb2(filepath):
    file_size = os.path.getsize(filepath)
    with open(filepath, "rb") as f:
        header = CDB2Header.read(f, file_size)
        tree_size = header.ofs_triangles - HEADER_SIZE
        tree_data = f.read(tree_size)
        num_nodes = tree_size // NODE_SIZE
        nodes = [TreeNode.read(tree_data[i*8:(i+1)*8], i) for i in range(num_nodes)]
        tri_data = f.read(header.ofs_vertices - header.ofs_triangles)
        vert_data_size = file_size - header.ofs_vertices
        vert_data_raw = f.read(vert_data_size)
        vert_coords = list(struct.unpack(f"<{vert_data_size//2}h", vert_data_raw))

    all_tris = []
    for node in nodes:
        if node.is_leaf and node.num_triangles > 0:
            all_tris.extend(_decode_triangles(node, tri_data))

    shadow_uvs = _compute_vertex_shadow_uvs(all_tris, vert_data_raw)

    print(f"CDB2: {num_nodes} nodes, {len(all_tris)} tris")
    return CDB2Data(header=header, triangles=all_tris,
                    vertex_coords=vert_coords, vertex_shadow_uvs=shadow_uvs)


# HEADER STORAGE ON COLLECTION

def _store_header_on_collection(col, header):
    """Store original CDB2 header as custom properties for export preservation."""
    col["fo2_cdb2_axis_mult_x"] = header.axis_multipliers[0]
    col["fo2_cdb2_axis_mult_y"] = header.axis_multipliers[1]
    col["fo2_cdb2_axis_mult_z"] = header.axis_multipliers[2]
    col["fo2_cdb2_inv_mult_x"] = header.inverse_axis_multipliers[0]
    col["fo2_cdb2_inv_mult_y"] = header.inverse_axis_multipliers[1]
    col["fo2_cdb2_inv_mult_z"] = header.inverse_axis_multipliers[2]


def _read_header_from_collection(col):
    """Read stored original CDB2 header from collection."""
    try:
        return CDB2Header(
            mins=(0, 0, 0), maxs=(0, 0, 0),
            axis_multipliers=(
                col["fo2_cdb2_axis_mult_x"],
                col["fo2_cdb2_axis_mult_y"],
                col["fo2_cdb2_axis_mult_z"]),
            inverse_axis_multipliers=(
                col["fo2_cdb2_inv_mult_x"],
                col["fo2_cdb2_inv_mult_y"],
                col["fo2_cdb2_inv_mult_z"]),
            ofs_triangles=0, ofs_vertices=0, file_size=0)
    except KeyError:
        return None


# BLENDER MESH BUILDING (IMPORT)

def _create_collision_material(name, color):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    return mat


def build_collision_mesh(cdb, collection, group_by_material=True):
    # group by full flag combo to preserve all per-tri data
    groups = {}
    for tri in cdb.triangles:
        key = (tri.surface_id, tri.lo_flags, tri.hi_flags, tri.bitmask)
        groups.setdefault(key, []).append(tri)

    # pre-count how many flag combos each surface has
    surface_counts = {}
    for (surface_id, lo_flags, hi_flags, bitmask) in groups:
        surface_counts[surface_id] = surface_counts.get(surface_id, 0) + 1

    # shared materials: one per surface_id (visual only, no collision flags)
    surface_materials = {}

    # track sub-collections per surface
    sub_collections = {}
    variant_idx = {}

    for (surface_id, lo_flags, hi_flags, bitmask) in sorted(groups.keys()):
        tris = groups[(surface_id, lo_flags, hi_flags, bitmask)]
        sname = _surface_name(surface_id)
        base_name = f"col_{surface_id}_{sname}"
        needs_subcol = surface_counts[surface_id] > 1

        # pick a unique object name
        if needs_subcol:
            idx = variant_idx.get(surface_id, 0)
            variant_idx[surface_id] = idx + 1
            name = f"{base_name}.{idx:03d}" if idx > 0 else base_name
        else:
            name = base_name

        # get or create shared material for this surface type
        if surface_id not in surface_materials:
            color = SURFACE_COLORS.get(_surface_category(surface_id), (0.5, 0.5, 0.5, 1.0))
            surface_materials[surface_id] = _create_collision_material(
                f"col_{surface_id}_{sname}", color)
        mat = surface_materials[surface_id]

        vert_map = {}
        verts = []
        faces = []

        for tri in tris:
            local_indices = []
            for ci in tri.vert_indices:
                if ci not in vert_map:
                    vert_map[ci] = len(verts)
                    verts.append(cdb.vertex_to_blender(ci))
                local_indices.append(vert_map[ci])
            faces.append(list(reversed(local_indices)))

        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(verts, [], faces)
        mesh.update()
        mesh.materials.append(mat)

        # store per-vertex shadow UV data as a UV map (from original 4 extra bytes)
        has_any_shadow = any(ci in cdb.vertex_shadow_uvs for ci in vert_map)
        if has_any_shadow:
            uv_layer = mesh.uv_layers.new(name="fo2_shadow_uv")
            # build per-vertex UV lookup (int16_idx -> (u_float, v_float))
            vert_uvs = {}
            for ci, local_idx in vert_map.items():
                if ci in cdb.vertex_shadow_uvs:
                    uv_u32 = cdb.vertex_shadow_uvs[ci]
                    u = (uv_u32 & 0xFFFF) / 65535.0
                    v = ((uv_u32 >> 16) & 0xFFFF) / 65535.0
                    vert_uvs[local_idx] = (u, v)
            # write per-loop UVs
            for poly in mesh.polygons:
                for li in poly.loop_indices:
                    vi = mesh.loops[li].vertex_index
                    if vi in vert_uvs:
                        uv_layer.data[li].uv = vert_uvs[vi]
                    else:
                        uv_layer.data[li].uv = (-1.0, -1.0)

        obj = bpy.data.objects.new(name, mesh)
        obj["fo2_has_shadow"] = has_any_shadow

        # collision properties stored on the object (not the material)
        obj["fo2_surface_id"] = surface_id
        obj["fo2_lo_flags"] = lo_flags
        obj["fo2_hi_flags"] = hi_flags
        obj["fo2_bitmask"] = bitmask

        if needs_subcol:
            if surface_id not in sub_collections:
                subcol = bpy.data.collections.new(base_name)
                collection.children.link(subcol)
                sub_collections[surface_id] = subcol
            sub_collections[surface_id].objects.link(obj)
        else:
            collection.objects.link(obj)

    print(f"Created {len(groups)} collision mesh objects ({len(surface_materials)} shared materials) "
          f"in {len(sub_collections)} sub-collections")


# SHADOWMAP

def load_shadowmap(filepath, header, collection):
    if not os.path.isfile(filepath):
        return
    data = open(filepath, "rb").read()
    expected = SHADOWMAP_SIZE * SHADOWMAP_SIZE
    if len(data) != expected:
        return

    img_name = os.path.basename(filepath)
    img = bpy.data.images.new(img_name, SHADOWMAP_SIZE, SHADOWMAP_SIZE)
    pixels = []
    for row in range(SHADOWMAP_SIZE - 1, -1, -1):
        for col in range(SHADOWMAP_SIZE):
            v = data[row * SHADOWMAP_SIZE + col] / 255.0
            pixels.extend((v, v, v, 1.0))
    img.pixels[:] = pixels
    img.pack()
    img.update()

    ext = header.symmetric_extent
    half_x, half_z = ext[0], ext[2]
    verts = [(-half_x, -half_z, 0), (half_x, -half_z, 0),
             (half_x, half_z, 0), (-half_x, half_z, 0)]
    faces = [(0, 1, 2, 3)]

    mesh = bpy.data.meshes.new("shadowmap_plane")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    uv_layer = mesh.uv_layers.new(name="UVMap")
    for i, uv in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):
        uv_layer.data[i].uv = uv

    mat = bpy.data.materials.new("shadowmap_material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out_node = nodes.new("ShaderNodeOutputMaterial"); out_node.location = (300, 0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location = (0, 0)
    links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])
    tex_node = nodes.new("ShaderNodeTexImage"); tex_node.location = (-300, 0)
    tex_node.image = img; tex_node.interpolation = 'Closest'
    links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
    mesh.materials.append(mat)

    obj = bpy.data.objects.new("shadowmap", mesh)
    obj.location.z = header.mins[1] * header.axis_multipliers[1]
    collection.objects.link(obj)
    # store shadowmap source path
    collection["fo2_shadowmap_path"] = filepath


def export_shadowmap(filepath, img):
    """Export a Blender image back to raw shadowmap .dat format."""
    if img.size[0] != SHADOWMAP_SIZE or img.size[1] != SHADOWMAP_SIZE:
        print(f"Shadowmap wrong size: {img.size}")
        return False
    pixels = list(img.pixels)
    out = bytearray(SHADOWMAP_SIZE * SHADOWMAP_SIZE)
    for row in range(SHADOWMAP_SIZE):
        file_row = SHADOWMAP_SIZE - 1 - row
        for col in range(SHADOWMAP_SIZE):
            val = int(round(pixels[(row * SHADOWMAP_SIZE + col) * 4] * 255))
            out[file_row * SHADOWMAP_SIZE + col] = max(0, min(255, val))
    with open(filepath, "wb") as f:
        f.write(out)
    return True


def remap_shadowmap(old_path, old_mult, new_mult, out_path):
    """Resample shadowmap from old coordinate space to new one."""
    old_data = open(old_path, "rb").read()
    if len(old_data) != SHADOWMAP_SIZE * SHADOWMAP_SIZE:
        return False

    old_ext_x = MAX_ABS_COORDINATE * old_mult[0]
    old_ext_z = MAX_ABS_COORDINATE * old_mult[2]
    new_ext_x = MAX_ABS_COORDINATE * new_mult[0]
    new_ext_z = MAX_ABS_COORDINATE * new_mult[2]

    new_data = bytearray(SHADOWMAP_SIZE * SHADOWMAP_SIZE)
    S = SHADOWMAP_SIZE

    for nr in range(S):
        for nc in range(S):
            wx = ((nc + 0.5) / S * 2.0 - 1.0) * new_ext_x
            wz = ((nr + 0.5) / S * 2.0 - 1.0) * new_ext_z
            ou = (wx / old_ext_x + 1.0) * 0.5
            ov = (wz / old_ext_z + 1.0) * 0.5
            if 0.0 <= ou <= 1.0 and 0.0 <= ov <= 1.0:
                oc = min(int(ou * S), S - 1)
                orr = min(int(ov * S), S - 1)
                new_data[nr * S + nc] = old_data[orr * S + oc]
            else:
                new_data[nr * S + nc] = 255

    with open(out_path, "wb") as f:
        f.write(new_data)
    print(f"Shadowmap remapped to {out_path}")
    return True


# CDB2 EXPORT

def _z_up_to_y_up(v):
    return (v[0], v[2], v[1])


def _find_collision_meshes():
    try:
        col = bpy.data.collections['collision']
    except KeyError:
        raise RuntimeError('No "collision" collection found')
    meshes = [o for o in col.all_objects.values()
              if isinstance(o.data, bpy.types.Mesh) and "shadowmap" not in o.name.lower()]
    return col, meshes


# default bitmask per surface_id, from cross-track analysis (125K triangles).
# surfaces with mixed bitmasks use the dominant one (>90% occurrence).
_DEFAULT_BITMASK = {
    # bm=11 (0xB) — shadow terrain / road
    2: 0xB, 3: 0xB, 4: 0xB, 5: 0xB, 6: 0xB, 7: 0xB,    # tarmac/Asphalt/Cement
    8: 0xB, 9: 0xB, 10: 0xB, 11: 0xB, 12: 0xB, 13: 0xB,  # hard/Medium/Soft
    14: 0xB, 15: 0xB,                                       # derby surfaces
    16: 0xB, 17: 0xB, 18: 0xB, 19: 0xB,                    # snow/Dirt
    23: 0xB, 24: 0xB,                                       # grass/forest (dominant)
    # bm=3 (0x3) — non-shadow objects/walls
    1: 0x3,                                                  # nocollision
    20: 0x3, 21: 0x3, 22: 0x3,                              # bridge/curb/bank
    25: 0x3, 26: 0x3, 27: 0x3,                              # sand/rock/mould
    31: 0x3, 32: 0x3, 33: 0x3, 34: 0x3,                    # concrete/rock/metal/wood
    37: 0x3,                                                 # rubber
    # bm=1 (0x1) — reset/fence
    35: 0x1,                                                 # tree
    # bm=6 (0x6) — special
    38: 0x6,                                                 # water
}


def _get_collision_flags(obj, mat):
    """Read collision flags from a Blender object.

    Returns (surface_id, lo_flags, hi_flags, bitmask).
    hi_flags is None when it must be computed per-triangle from geometry.
    """
    if obj is None:
        return 27, 4, None, 0x3

    sid = obj.get("fo2_surface_id")
    lo = obj.get("fo2_lo_flags")
    hi = obj.get("fo2_hi_flags")
    bm = obj.get("fo2_bitmask")

    if sid is not None and lo is not None and hi is not None and bm is not None:
        return int(sid), int(lo), int(hi), int(bm)

    # Derive missing bitmask from fo2_has_shadow or surface_id
    if bm is None:
        has_shadow = obj.get("fo2_has_shadow")
        if has_shadow is not None:
            bm = 0xB if has_shadow else 0x3
        elif sid is not None:
            bm = _DEFAULT_BITMASK.get(int(sid), 0x3)
        else:
            bm = 0x3

    return (
        int(sid) if sid is not None else 27,
        int(lo) if lo is not None else 4,
        int(hi) if hi is not None else None,
        int(bm),
    )


@dataclass
class ExportTriangle:
    vert_indices: tuple   # coord indices (ci*3)
    surface_id: int
    lo_flags: int
    hi_flags: int
    bitmask: int
    aabb_min: tuple
    aabb_max: tuple


class VertexEncoder:
    """Encodes game-space vertices to int16 coordinates with optional shadow UV data.

    Vertices with shadow UV data are stride 5 (xyz + 2 uint16), others stride 3.
    The int16 offset returned accounts for mixed strides.
    """
    def __init__(self, inv_mult):
        self._inv = inv_mult
        self._map = {}             # scaled_xyz -> vertex sequential index
        self._verts = []           # list of (x, y, z) int16 tuples
        self._shadow_uvs = {}      # vertex sequential index -> (u16, v16)
        self._shadow_skip = set()  # vertex indices imported without UV (don't auto-compute)
        self._int16_offsets = []   # computed after finalize

    def encode(self, game_pos, shadow_uv=None):
        """Encode a vertex. shadow_uv is (u16, v16) tuple, "skip" to mark
        imported-without-UV, or None for new geometry (eligible for auto-compute)."""
        scaled = tuple(
            max(-MAX_ABS_COORDINATE, min(MAX_ABS_COORDINATE,
                round(game_pos[i] * self._inv[i])))
            for i in range(3))
        if scaled in self._map:
            idx = self._map[scaled]
            if shadow_uv == "skip":
                self._shadow_skip.add(idx)
            elif shadow_uv is not None and idx not in self._shadow_uvs:
                self._shadow_uvs[idx] = shadow_uv
            return idx
        idx = len(self._verts)
        self._verts.append(scaled)
        self._map[scaled] = idx
        if shadow_uv == "skip":
            self._shadow_skip.add(idx)
        elif shadow_uv is not None:
            self._shadow_uvs[idx] = shadow_uv
        return idx

    def finalize(self):
        """Compute int16 offsets for all vertices. Must be called before encoding triangles."""
        self._int16_offsets = []
        offset = 0
        for i in range(len(self._verts)):
            self._int16_offsets.append(offset)
            offset += 5 if i in self._shadow_uvs else 3

    def get_int16_offset(self, seq_idx):
        """Get the int16 offset for a vertex by sequential index."""
        return self._int16_offsets[seq_idx]

    @property
    def vertices(self):
        return self._verts

    @property
    def shadow_uvs(self):
        return self._shadow_uvs

    def write_vertex_data(self, f):
        """Write all vertex data to file with correct mixed strides."""
        for i, v in enumerate(self._verts):
            f.write(struct.pack("<3h", *v))
            if i in self._shadow_uvs:
                u, v16 = self._shadow_uvs[i]
                f.write(struct.pack("<2H", u, v16))


def _collect_export_triangles(meshes, inv_mult):
    venc = VertexEncoder(inv_mult)
    tris = []
    degens = 0

    for obj in meshes:
        mdata = cast(bpy.types.Mesh, obj.data)

        # Triangulate only if mesh has non-triangle faces
        needs_triangulate = any(len(p.vertices) != 3 for p in mdata.polygons)
        if needs_triangulate:
            import bmesh as _bmesh
            bm = _bmesh.new()
            bm.from_mesh(mdata)
            _bmesh.ops.triangulate(bm, faces=bm.faces)
            tri_mesh = bpy.data.meshes.new(".fo2_tmp_triangulate")
            bm.to_mesh(tri_mesh)
            bm.free()
        else:
            tri_mesh = mdata

        # check fo2_has_shadow property to determine shadow UV handling:
        #   true  + UV layer  = read imported UVs (preserve)
        #   true  + no layer  = auto-compute UVs on export
        #   false / missing   = no shadow UVs (stride 3)
        wants_shadow = obj.get("fo2_has_shadow", False)
        uv_layer = tri_mesh.uv_layers.get("fo2_shadow_uv") if wants_shadow else None

        # build per-vertex shadow UV lookup from loop data
        vert_shadow = {}  # blender vert index -> (u16, v16) or "skip"
        if wants_shadow and uv_layer:
            for poly in tri_mesh.polygons:
                for li in poly.loop_indices:
                    vi = tri_mesh.loops[li].vertex_index
                    if vi in vert_shadow:
                        continue
                    uv = uv_layer.data[li].uv
                    if uv[0] < 0 or uv[1] < 0:
                        # sentinel: imported vertex without UV, keep stride 3
                        vert_shadow[vi] = "skip"
                    else:
                        u16 = max(0, min(65535, round(uv[0] * 65535)))
                        v16 = max(0, min(65535, round(uv[1] * 65535)))
                        vert_shadow[vi] = (u16, v16)
        elif not wants_shadow:
            # mark all vertices as skip (no shadow for this mesh)
            for poly in tri_mesh.polygons:
                for vi in poly.vertices:
                    if vi not in vert_shadow:
                        vert_shadow[vi] = "skip"

        for poly in tri_mesh.polygons:
            if len(poly.vertices) != 3:
                continue
            mat = tri_mesh.materials[poly.material_index] if tri_mesh.materials else None
            sid, lo_flags, hi_flags, bitmask = _get_collision_flags(obj, mat)

            sverts = []
            seq_indices = []
            for vi in poly.vertices:
                wco = obj.matrix_world @ tri_mesh.vertices[vi].co
                gco = _z_up_to_y_up(wco)

                # get shadow UV for this vertex
                shadow_uv = vert_shadow.get(vi)
                # shadow_uv is (u16,v16), "skip", or None (auto-compute eligible)

                seq_idx = venc.encode(gco, shadow_uv)
                # get scaled coords for degenerate/AABB checks
                scaled = venc.vertices[seq_idx]
                sverts.append(scaled)
                seq_indices.append(seq_idx)

            e1 = tuple(sverts[1][i] - sverts[0][i] for i in range(3))
            e2 = tuple(sverts[2][i] - sverts[0][i] for i in range(3))
            cx = e1[1]*e2[2] - e1[2]*e2[1]
            cy = e1[2]*e2[0] - e1[0]*e2[2]
            cz = e1[0]*e2[1] - e1[1]*e2[0]
            if cx == 0 and cy == 0 and cz == 0:
                degens += 1
                continue

            # Reverse winding for game
            seq_indices = list(reversed(seq_indices))
            sverts = list(reversed(sverts))

            # compute hi_flags per-triangle when material didn't provide them.
            #
            # flags supposed byte layout (8 bits) (COULD BE INCORRECT!):
            #   bits 0-3: lo_flags lower nibble   (collision physics)
            #   bits 4-5: lo_flags upper bits     (promotion bits, gated by lower)
            #   bit  6:   shadow participation    = (bitmask & 8) AND (ny > 0)
            #   bit  7:   secondary lighting flag (on some surfaces, not deterministic)
            #
            # plugin stores bits 0-5 as lo_flags (6-bit) and bits 6-7 as hi_flags (2-bit)
            # for roundtrip, imported lo_flags already contains bits 4-5 (promotion)
            # for new geometry: lo_flags defaults to 4 (bits 4-5 = 0, no promotion)
            #
            # cy is the Y component of (e1 × e2) in Blender winding; the game
            # normal Y after winding reversal is −cy.
            if hi_flags is None:
                game_ny = -cy        # positive = upward-facing in game space
                hi_flags = 0
                if (bitmask & 8) and game_ny > 0:
                    hi_flags = 1     # byte bit 6 = shadow participation
                # byte bit 7 = 0 (secondary lighting, not deterministic)

            tris.append(ExportTriangle(
                vert_indices=tuple(seq_indices),  # sequential indices, resolved later
                surface_id=sid, lo_flags=lo_flags, hi_flags=hi_flags, bitmask=bitmask,
                aabb_min=tuple(min(sv[i] for sv in sverts) for i in range(3)),
                aabb_max=tuple(max(sv[i] for sv in sverts) for i in range(3))))

        if needs_triangulate:
            bpy.data.meshes.remove(tri_mesh)

        # Write all computed properties back to object
        obj["fo2_surface_id"] = sid
        obj["fo2_lo_flags"] = lo_flags
        obj["fo2_bitmask"] = bitmask
        if obj.get("fo2_has_shadow") is None:
            obj["fo2_has_shadow"] = bool(bitmask & 8)

    # auto-compute shadow UVs for vertices from fo2_has_shadow=True meshes that don't already have UVs
    # vertices marked "skip" (from fo2_has_shadow=False or imported sentinel -1,-1 UVs) are never touched
    auto_count = 0
    shadow_candidates = set()
    for tri in tris:
        if tri.bitmask & 8:  # bit 3 = shadow-mapped surface
            for si in tri.vert_indices:
                if si not in venc.shadow_uvs and si not in venc._shadow_skip:
                    shadow_candidates.add(si)

    if shadow_candidates:
        all_v = venc.vertices
        bb_min_x = min(v[0] for v in all_v)
        bb_max_x = max(v[0] for v in all_v)
        bb_min_z = min(v[2] for v in all_v)
        bb_max_z = max(v[2] for v in all_v)
        range_x = max(bb_max_x - bb_min_x, 1)
        range_z = max(bb_max_z - bb_min_z, 1)

        for si in shadow_candidates:
            v = all_v[si]
            u = max(0, min(65535, round((v[0] - bb_min_x) / range_x * 65535)))
            sv = max(0, min(65535, round((v[2] - bb_min_z) / range_z * 65535)))
            venc._shadow_uvs[si] = (u, sv)
            auto_count += 1

    # finalize vertex encoder (computes int16 offsets with mixed strides)
    venc.finalize()

    # resolve sequential vertex indices to int16 offsets in all triangles
    resolved_tris = []
    for tri in tris:
        resolved = ExportTriangle(
            vert_indices=tuple(venc.get_int16_offset(si) for si in tri.vert_indices),
            surface_id=tri.surface_id, lo_flags=tri.lo_flags,
            hi_flags=tri.hi_flags, bitmask=tri.bitmask,
            aabb_min=tri.aabb_min, aabb_max=tri.aabb_max)
        resolved_tris.append(resolved)

    n_shadow = len(venc.shadow_uvs)
    if n_shadow:
        msg = f"Export: {n_shadow}/{len(venc.vertices)} vertices have shadow UV"
        if auto_count:
            msg += f" ({n_shadow - auto_count} preserved, {auto_count} auto-computed)"
        print(msg)

    return venc, resolved_tris, degens


# AABB tree

@dataclass
class Leaf:
    index: int
    tris: list
    bitmask: int

@dataclass
class Inner:
    index: int
    axis: int
    split_min: int
    split_max: int
    child0: object = None
    child1: object = None


def _build_tree(my_idx, tris, next_idx):
    if len(tris) <= TARGET_LEAF_TRIANGLES:
        bm = 0
        for t in tris:
            bm |= t.bitmask
        return Leaf(index=my_idx, tris=tris, bitmask=bm & _ones(4))

    # Longest axis split
    amins = [min(t.aabb_min[i] for t in tris) for i in range(3)]
    amaxs = [max(t.aabb_max[i] for t in tris) for i in range(3)]
    exts = [amaxs[i] - amins[i] for i in range(3)]
    axis = exts.index(max(exts))

    tris.sort(key=lambda t: (t.aabb_min[axis] + t.aabb_max[axis]) * 0.5)
    mid = max(1, len(tris) // 2)
    left, right = tris[:mid], tris[mid:]

    smax = max(int(t.aabb_max[axis]) for t in left)
    smin = min(int(t.aabb_min[axis]) for t in right)
    smax = max(-MAX_ABS_COORDINATE, min(MAX_ABS_COORDINATE, smax))
    smin = max(-MAX_ABS_COORDINATE, min(MAX_ABS_COORDINATE, smin))

    # pre-allocate consecutive child indices BEFORE recursing.
    # the format only stores child0's index; child1 is implicitly child0+1.
    child0_idx = next_idx()
    child1_idx = next_idx()
    assert child1_idx == child0_idx + 1

    node = Inner(index=my_idx, axis=axis, split_min=smin, split_max=smax)
    node.child0 = _build_tree(child0_idx, left, next_idx)
    node.child1 = _build_tree(child1_idx, right, next_idx)
    return node


def _iter_tree(node):
    yield node
    if isinstance(node, Inner):
        yield from _iter_tree(node.child0)
        yield from _iter_tree(node.child1)


def _node_bitmask(node):
    """Get bitmask for a node. For leaves, it's stored directly.
    For inner nodes, it's the OR of both children's bitmasks."""
    if isinstance(node, Leaf):
        return node.bitmask & _ones(LEAF_MASK_BITS)
    return (_node_bitmask(node.child0) | _node_bitmask(node.child1)) & _ones(INNER_MASK_BITS)


def _choose_leaf_kind(tris, vert_ofs):
    """Select the most compact triangle encoding kind for a leaf.
    Returns (kind, flags_surface) where flags_surface is the surface stored
    in the leaf node header (used by kinds 1, 3, 4, 5)."""
    first_surf = (tris[0].surface_id - 1) & _ones(6)
    same_surface = all(((t.surface_id - 1) & _ones(6)) == first_surf for t in tris)
    same_lo = all(t.lo_flags == tris[0].lo_flags for t in tris)
    same_hi = all(t.hi_flags == tris[0].hi_flags for t in tris)

    # collect all vertex indices relative to vert_ofs
    rels = []
    for t in tris:
        for vi in t.vert_indices:
            rels.append(vi - vert_ofs)

    all_rel_nonneg = all(r >= 0 for r in rels)
    max_rel = max(rels) if rels else 0

    # kind 5: same surface+flags, 5-bit (v0,v1) / 6-bit (v2) relative offsets
    # v0,v1 max 31, v2 max 63
    if same_surface and same_lo and same_hi and all_rel_nonneg:
        fits_k5 = True
        for t in tris:
            r0 = t.vert_indices[0] - vert_ofs
            r1 = t.vert_indices[1] - vert_ofs
            r2 = t.vert_indices[2] - vert_ofs
            if r0 > 31 or r1 > 31 or r2 > 63:
                fits_k5 = False
                break
        if fits_k5:
            return 5, first_surf

    # kind 3: same surface, 8-bit relative offsets (max 255)
    if same_surface and all_rel_nonneg and max_rel <= 255:
        return 3, first_surf

    # kind 4: same surface+flags, 11-bit relative offsets (max 2047), 1-bit hi_flags
    if same_surface and same_lo and all_rel_nonneg and max_rel <= 2047:
        # Kind 4 only stores 1 bit of hi_flags — check all fit
        if all(t.hi_flags <= 1 for t in tris):
            return 4, first_surf

    # kind 2: per-tri surface, 8-bit relative offsets (max 255)
    if all_rel_nonneg and max_rel <= 255:
        return 2, first_surf

    # kind 1: same surface, absolute 19-bit vertex indices
    # subsequent tris only store 1 bit of hi_flags
    if same_surface and len(tris) > 1:
        if all(t.hi_flags <= 1 for t in tris[1:]):
            return 1, first_surf

    # also use kind 1 for single-tri leaves with same surface (first tri format is same as kind 0)
    if same_surface and len(tris) == 1:
        return 1, first_surf

    # kind 0: fully general (fallback)
    return 0, first_surf


def _encode_leaf_tris(kind, tris, vert_ofs, tri_data):
    """Encode triangle data for the chosen kind. Appends to tri_data."""
    if kind == 0:
        _encode_tris_kind0(tris, vert_ofs, tri_data)
    elif kind == 1:
        _encode_tris_kind1(tris, vert_ofs, tri_data)
    elif kind == 2:
        _encode_tris_kind2(tris, vert_ofs, tri_data)
    elif kind == 3:
        _encode_tris_kind3(tris, vert_ofs, tri_data)
    elif kind == 4:
        _encode_tris_kind4(tris, vert_ofs, tri_data)
    elif kind == 5:
        _encode_tris_kind5(tris, vert_ofs, tri_data)


def _encode_first_tri_6bytes(tri, vert_ofs, tri_data):
    """Encode the first triangle in 6 bytes (shared by kinds 0 and 1).
    v0 = vert_ofs (stored in node), v1/v2 as absolute 19-bit indices."""
    b = bytearray(6)
    b[0] = (tri.lo_flags & _ones(6)) | ((tri.hi_flags & 3) << 6)
    v1 = tri.vert_indices[1]
    v2 = tri.vert_indices[2]
    b[1] = v1 & 0xFF
    b[2] = (v1 >> 8) & 0xFF
    b[3] = ((v1 >> 16) & 0x07) | ((v2 & 0x1F) << 3)
    b[4] = (v2 >> 5) & 0xFF
    b[5] = (v2 >> 13) & 0xFF
    tri_data.extend(b)


def _encode_tris_kind0(tris, vert_ofs, tri_data):
    """Kind 0: first tri 6 bytes, subsequent 9 bytes each.
    Each subsequent tri stores its own surface, lo_flags, hi_flags, and 3x 19-bit absolute vertex indices."""
    _encode_first_tri_6bytes(tris[0], vert_ofs, tri_data)
    for tri in tris[1:]:
        surf = (tri.surface_id - 1) & _ones(6)
        # bitstream: lo_flags(6) | hi_flags(2) | surface(6) | pad(1) | v0(19) | v1(19) | v2(19) = 72 bits
        val = (tri.lo_flags & _ones(6)) | ((tri.hi_flags & 3) << 6)
        val |= surf << 8
        # bit 14 is padding (0)
        val |= tri.vert_indices[0] << 15
        val |= tri.vert_indices[1] << 34
        val |= tri.vert_indices[2] << 53
        b = bytearray(9)
        for i in range(9):
            b[i] = (val >> (i * 8)) & 0xFF
        tri_data.extend(b)


def _encode_tris_kind1(tris, vert_ofs, tri_data):
    """Kind 1: same surface (from leaf_flags). First tri 6 bytes, subsequent 8 bytes each.
    Subsequent: lo_flags(6) | hi_flags(1) | v0(19) | v1(19) | v2(19) = 64 bits."""
    _encode_first_tri_6bytes(tris[0], vert_ofs, tri_data)
    for tri in tris[1:]:
        # bitstream: lo_flags(6) | hi_flags(1) | v0(19) | v1(19) | v2(19) = 64 bits = 8 bytes
        val = (tri.lo_flags & _ones(6))
        val |= (tri.hi_flags & 1) << 6
        val |= tri.vert_indices[0] << 7
        val |= tri.vert_indices[1] << 26
        val |= tri.vert_indices[2] << 45
        b = bytearray(8)
        for i in range(8):
            b[i] = (val >> (i * 8)) & 0xFF
        tri_data.extend(b)


def _encode_tris_kind2(tris, vert_ofs, tri_data):
    """Kind 2: per-tri surface, 8-bit relative vertex offsets. 5 bytes per tri."""
    for tri in tris:
        surf = (tri.surface_id - 1) & _ones(6)
        b = bytearray(5)
        b[0] = (tri.lo_flags & _ones(6)) | ((tri.hi_flags & 3) << 6)
        b[1] = surf & _ones(6)  # only low 6 bits used by decoder
        b[2] = (tri.vert_indices[0] - vert_ofs) & 0xFF
        b[3] = (tri.vert_indices[1] - vert_ofs) & 0xFF
        b[4] = (tri.vert_indices[2] - vert_ofs) & 0xFF
        tri_data.extend(b)


def _encode_tris_kind3(tris, vert_ofs, tri_data):
    """Kind 3: same surface (leaf_flags), 8-bit relative vertex offsets. 4 bytes per tri."""
    for tri in tris:
        b = bytearray(4)
        b[0] = (tri.lo_flags & _ones(6)) | ((tri.hi_flags & 3) << 6)
        b[1] = (tri.vert_indices[0] - vert_ofs) & 0xFF
        b[2] = (tri.vert_indices[1] - vert_ofs) & 0xFF
        b[3] = (tri.vert_indices[2] - vert_ofs) & 0xFF
        tri_data.extend(b)


def _encode_tris_kind4(tris, vert_ofs, tri_data):
    """Kind 4: same surface+flags (leaf_flags), 11-bit relative offsets, 1-bit hi_flags. 5 bytes per tri."""
    for tri in tris:
        r0 = tri.vert_indices[0] - vert_ofs
        r1 = tri.vert_indices[1] - vert_ofs
        r2 = tri.vert_indices[2] - vert_ofs
        # bitstream: lo_flags(6) | hi_flags(1) | v0_rel(11) | v1_rel(11) | v2_rel(11) = 40 bits = 5 bytes
        val = (tri.lo_flags & _ones(6))
        val |= (tri.hi_flags & 1) << 6
        val |= (r0 & _ones(11)) << 7
        val |= (r1 & _ones(11)) << 18
        val |= (r2 & _ones(11)) << 29
        b = bytearray(5)
        for i in range(5):
            b[i] = (val >> (i * 8)) & 0xFF
        tri_data.extend(b)


def _encode_tris_kind5(tris, vert_ofs, tri_data):
    """Kind 5: same surface+flags (leaf_flags), 5/5/6-bit relative offsets. 3 bytes per tri."""
    for tri in tris:
        r0 = tri.vert_indices[0] - vert_ofs
        r1 = tri.vert_indices[1] - vert_ofs
        r2 = tri.vert_indices[2] - vert_ofs
        # bitstream: lo_flags(6) | hi_flags(2) | v0_rel(5) | v1_rel(5) | v2_rel(6) = 24 bits = 3 bytes
        b = bytearray(3)
        b[0] = (tri.lo_flags & _ones(6)) | ((tri.hi_flags & 3) << 6)
        b[1] = (r0 & _ones(5)) | ((r1 & 0x07) << 5)
        b[2] = ((r1 >> 3) & 0x03) | ((r2 & _ones(6)) << 2)
        tri_data.extend(b)


def _encode_leaf(leaf, tri_data):
    """Encode leaf node + triangles using the most compact encoding kind. Returns 8-byte node."""
    count = len(leaf.tris)

    if count == 0:
        # empty leaf node
        lo = LEAF_AXIS  # axis=3, everything else 0
        hi = 0
        return struct.pack("<2I", lo, hi)

    first = leaf.tris[0]
    vert_ofs = first.vert_indices[0]

    kind, flags_surface = _choose_leaf_kind(leaf.tris, vert_ofs)
    tri_ofs = len(tri_data)

    _encode_leaf_tris(kind, leaf.tris, vert_ofs, tri_data)

    # lo dword: axis(2) | bitmask(4) | kind(3) | tri_offset(23)
    lo = tri_ofs & _ones(LEAF_TRIANGLE_OFS_BITS)
    lo <<= LEAF_KIND_BITS
    lo |= kind
    lo <<= LEAF_MASK_BITS
    lo |= leaf.bitmask & _ones(LEAF_MASK_BITS)
    lo <<= AXIS_BITS
    lo |= LEAF_AXIS

    # hi dword: count(7) | flags_surface(6) | vert_ofs(19)
    hi = vert_ofs & _ones(LEAF_VERTEX_OFS_BITS)
    hi <<= LEAF_FLAGS_BITS
    hi |= flags_surface & _ones(LEAF_FLAGS_BITS)
    hi <<= LEAF_TRIANGLE_COUNT_BITS
    hi |= count & _ones(LEAF_TRIANGLE_COUNT_BITS)

    return struct.pack("<2I", lo, hi)


def _encode_inner(node):
    child0_byte_ofs = node.child0.index * NODE_SIZE
    bm = _node_bitmask(node)

    lo = child0_byte_ofs
    lo <<= INNER_MASK_BITS
    lo |= bm & _ones(INNER_MASK_BITS)
    lo <<= AXIS_BITS
    lo |= node.axis & _ones(AXIS_BITS)

    return struct.pack("<I", lo) + struct.pack("<2h", node.split_max, node.split_min)


def export_cdb2(report, filepath, preserve_multipliers=True, shadowmap_path=""):
    """Export collision to CDB2, preserving axis multipliers for shadowmap safety."""
    print(f'CDB2 export: "{filepath}"')

    col, meshes = _find_collision_meshes()
    if not meshes:
        report({'ERROR'}, "No meshes in collision collection")
        return False

    original_header = _read_header_from_collection(col) if preserve_multipliers else None

    if original_header:
        inv_mult = original_header.inverse_axis_multipliers
        axis_mult = original_header.axis_multipliers
        print(f"Using preserved axis multipliers (shadowmap-safe)")
    else:
        # compute from geometry bounds
        gmin = [float('inf')] * 3
        gmax = [float('-inf')] * 3
        for obj in meshes:
            md = cast(bpy.types.Mesh, obj.data)
            for poly in md.polygons:
                for vi in poly.vertices:
                    wco = obj.matrix_world @ md.vertices[vi].co
                    gco = _z_up_to_y_up(wco)
                    for i in range(3):
                        gmin[i] = min(gmin[i], gco[i])
                        gmax[i] = max(gmax[i], gco[i])
        max_abs = [max(abs(gmin[i]), abs(gmax[i]), 0.001) for i in range(3)]
        axis_mult = tuple(ma / MAX_ABS_COORDINATE for ma in max_abs)
        inv_mult = tuple(MAX_ABS_COORDINATE / ma for ma in max_abs)
        print("Computed new axis multipliers (shadowmap may need remapping)")

    # encode geometry
    venc, tris, degens = _collect_export_triangles(meshes, inv_mult)
    if degens > 0:
        report({'WARNING'}, f"{degens} degenerate triangles dropped")
    if not tris:
        report({'ERROR'}, "No valid triangles")
        return False

    # check bounds overflow with preserved multipliers
    if original_header:
        overflow = any(abs(c) > MAX_ABS_COORDINATE
                       for v in venc.vertices for c in v)
        if overflow:
            report({'WARNING'},
                   "Geometry exceeds original bounds! Recomputing multipliers. "
                   "Shadowmap will be auto-remapped if found.")
            # recompute
            max_abs = [max(abs(v[i]) * axis_mult[i]
                          for v in venc.vertices)
                       for i in range(3)]
            for i in range(3):
                max_abs[i] = max(max_abs[i], 0.001)
            new_mult = tuple(ma / MAX_ABS_COORDINATE for ma in max_abs)
            new_inv = tuple(MAX_ABS_COORDINATE / ma for ma in max_abs)

            # remap shadowmap
            sm = shadowmap_path or col.get("fo2_shadowmap_path", "")
            if sm and os.path.isfile(sm):
                remap_shadowmap(sm, axis_mult, new_mult, sm)
                report({'INFO'}, "Shadowmap auto-remapped to new bounds")

            axis_mult = new_mult
            inv_mult = new_inv
            venc, tris, _ = _collect_export_triangles(meshes, inv_mult)

    print(f"Triangles: {len(tris)}, Vertices: {len(venc.vertices)}")

    # build AABB tree
    counter = [0]
    def next_idx():
        v = counter[0]; counter[0] += 1; return v

    t0 = time.monotonic_ns()
    root = _build_tree(next_idx(), list(tris), next_idx)
    dt = (time.monotonic_ns() - t0) / 1e9
    print(f"Tree: {counter[0]} nodes in {dt:.3f}s")

    # flatten and encode
    all_nodes = sorted(_iter_tree(root), key=lambda n: n.index)
    encoded_nodes = bytearray()
    tri_data = bytearray()

    for node in all_nodes:
        if isinstance(node, Leaf):
            encoded_nodes.extend(_encode_leaf(node, tri_data))
        else:
            encoded_nodes.extend(_encode_inner(node))

    # bounding box
    av = venc.vertices
    bb_min = [min(v[i] for v in av) for i in range(3)] if av else [0]*3
    bb_max = [max(v[i] for v in av) for i in range(3)] if av else [0]*3

    # write file
    with open(filepath, "wb") as f:
        f.write(FILE_MAGIC)
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<3i", *bb_min))
        f.write(struct.pack("<3i", *bb_max))
        f.write(struct.pack("<3f", *axis_mult))
        f.write(struct.pack("<3f", *inv_mult))
        nlen = len(encoded_nodes)
        tlen = len(tri_data)
        f.write(struct.pack("<I", nlen))
        f.write(struct.pack("<I", nlen + tlen))
        assert f.tell() == HEADER_SIZE
        f.write(encoded_nodes)
        f.write(tri_data)
        venc.write_vertex_data(f)

    # update stored header
    _store_header_on_collection(col, CDB2Header(
        mins=tuple(bb_min), maxs=tuple(bb_max),
        axis_multipliers=axis_mult,
        inverse_axis_multipliers=inv_mult,
        ofs_triangles=0, ofs_vertices=0, file_size=0))

    print("CDB2 export complete")
    report({'INFO'}, f"Exported {len(tris)} tris, {len(av)} verts")
    return True


# OPERATORS

class FO2_OT_ImportCollision(bpy.types.Operator):
    bl_idname = "import_scene.fo2_track_cdb2"
    bl_label = "Import FlatOut 2 Track Collision"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype='FILE_PATH')
    filter_glob: bpy.props.StringProperty(default="*.gen", options={'HIDDEN'})
    group_by_material: bpy.props.BoolProperty(name="Separate by Surface", default=True)
    import_shadowmap: bpy.props.BoolProperty(name="Import Shadowmap", default=True)
    shadowmap_path: bpy.props.StringProperty(
        name="Shadowmap Path",
        description="Path to shadowmap .dat file. Leave empty to auto-detect "
                    "(searches same folder as .gen, then ../lighting/)",
        subtype='FILE_PATH',
        default="")

    def execute(self, context):
        try:
            cdb = parse_cdb2(self.filepath)
        except (ValueError, struct.error) as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        col = bpy.data.collections.get("collision")
        if col is None:
            col = bpy.data.collections.new("collision")
            context.scene.collection.children.link(col)

        _store_header_on_collection(col, cdb.header)
        build_collision_mesh(cdb, col, self.group_by_material)

        if self.import_shadowmap:
            sm_path = self._find_shadowmap()
            if sm_path:
                load_shadowmap(sm_path, cdb.header, col)
                self.report({'INFO'}, f"Shadowmap loaded from {sm_path}")
            else:
                self.report({'WARNING'}, "Shadowmap not found")

        self.report({'INFO'}, f"Imported {len(cdb.triangles)} triangles")
        return {'FINISHED'}

    def _find_shadowmap(self):
        """Find shadowmap: user path → same dir as .gen → ../lighting/"""
        # user-specified path
        if self.shadowmap_path and os.path.isfile(self.shadowmap_path):
            return self.shadowmap_path

        cdb_dir = os.path.dirname(self.filepath)

        # same directory as the .gen file
        for fn in os.listdir(cdb_dir):
            if fn.endswith("shadowmap_w2.dat"):
                return os.path.join(cdb_dir, fn)

        # ../lighting/ relative to .gen file
        lighting_dir = os.path.join(cdb_dir, "..", "lighting")
        if os.path.isdir(lighting_dir):
            for fn in os.listdir(lighting_dir):
                if fn.endswith("shadowmap_w2.dat"):
                    return os.path.join(lighting_dir, fn)

        return None

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class FO2_OT_ExportCollision(bpy.types.Operator):
    bl_idname = "export_scene.fo2_track_cdb2"
    bl_label = "Export FlatOut 2 Track Collision"
    bl_options = {'REGISTER'}

    filepath: bpy.props.StringProperty(subtype='FILE_PATH')
    filter_glob: bpy.props.StringProperty(default="*.gen", options={'HIDDEN'})

    preserve_multipliers: bpy.props.BoolProperty(
        name="Preserve Axis Multipliers (Shadowmap-safe)",
        description="Keep original coordinate mapping so the shadowmap stays aligned. "
                    "Disable only for completely new collision meshes without a shadowmap",
        default=True)

    def execute(self, context):
        try:
            sm_path = ""
            cdb_dir = os.path.dirname(self.filepath)
            lighting_dir = os.path.join(cdb_dir, "..", "lighting")
            if os.path.isdir(lighting_dir):
                for fn in os.listdir(lighting_dir):
                    if fn.endswith("shadowmap_w2.dat"):
                        sm_path = os.path.join(lighting_dir, fn)
                        break

            ok = export_cdb2(self.report, self.filepath,
                             self.preserve_multipliers, sm_path)
            if not ok:
                return {'CANCELLED'}

        except RuntimeError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


# REGISTRATION

def _menu_import(self, context):
    self.layout.operator(FO2_OT_ImportCollision.bl_idname,
                         text="FlatOut 2 Track Collision (.gen)")

def _menu_export(self, context):
    self.layout.operator(FO2_OT_ExportCollision.bl_idname,
                         text="FlatOut 2 Track Collision (.gen)")

def register():
    bpy.utils.register_class(FO2_OT_ImportCollision)
    bpy.utils.register_class(FO2_OT_ExportCollision)
    bpy.types.TOPBAR_MT_file_import.append(_menu_import)
    bpy.types.TOPBAR_MT_file_export.append(_menu_export)

def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(_menu_export)
    bpy.types.TOPBAR_MT_file_import.remove(_menu_import)
    bpy.utils.unregister_class(FO2_OT_ExportCollision)
    bpy.utils.unregister_class(FO2_OT_ImportCollision)
