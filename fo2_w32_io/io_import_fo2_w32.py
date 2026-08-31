bl_info = {
    "name": "FlatOut 2 W32 Track Import",
    "author": "ravenDS",
    "version": (2, 0, 0),
    "blender": (3, 6, 0),
    "location": "File > Import > FlatOut 2 Track (.w32)",
    "description": "Import FlatOut 2 track geometry, textures, plants, props and BVH",
    "category": "Import-Export",
}

import bpy
import bmesh
import struct
import os
import math
import mathutils
from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty, BoolProperty, EnumProperty, FloatProperty
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Vertex buffer flags
# ─────────────────────────────────────────────────────────────────────────────
VERTEX_POSITION = 0x2
VERTEX_UV       = 0x100
VERTEX_UV2      = 0x200
VERTEX_NORMAL   = 0x10
VERTEX_COLOR    = 0x40
VERTEX_INT16    = 0x2000

FO2_SHADER_NAMES = {
    0: "static prelit", 1: "terrain", 2: "terrain specular",
    3: "dynamic diffuse", 4: "dynamic specular", 5: "car body",
    6: "car window", 7: "car diffuse", 8: "car metal",
    9: "car tire", 10: "car lights", 11: "car shear",
    12: "car scale", 13: "shadow project", 14: "car lights unlit",
    15: "default", 16: "vertex color", 17: "shadow sampler",
    18: "grass", 19: "tree trunk", 20: "tree branch",
    21: "tree leaf", 22: "particle", 23: "sunflare",
    24: "intensitymap", 25: "water", 26: "skinning",
    27: "tree lod", 28: "DUMMY", 29: "clouds",
    30: "car bodylod", 31: "vertex color static", 32: "car window damaged",
    33: "skin shadow", 34: "reflecting window (static)",
    35: "reflecting window (dynamic)", 36: "deprecated static specular",
    37: "skybox", 38: "ghost body", 39: "static nonlit",
    40: "dynamic nonlit", 41: "racemap",
}


# ═════════════════════════════════════════════════════════════════════════════
# Data structures
# ═════════════════════════════════════════════════════════════════════════════
class Material:
    __slots__ = (
        'name', 'alpha', 'v92', 'num_textures', 'shader_id',
        'use_colormap', 'v74', 'v108', 'v109',
        'v98', 'v99', 'v100', 'v101', 'v102',
        'texture_names',
    )
    def __init__(self):
        self.name = ""
        self.alpha = 0
        self.v92 = 0
        self.num_textures = 0
        self.shader_id = 0
        self.use_colormap = 0
        self.v74 = 0
        self.v108 = (0, 0, 0)
        self.v109 = (0, 0, 0)
        self.v98 = (0, 0, 0, 0)
        self.v99 = (0, 0, 0, 0)
        self.v100 = (0, 0, 0, 0)
        self.v101 = (0, 0, 0, 0)
        self.v102 = 0
        self.texture_names = ["", "", ""]


class VertexBuffer:
    __slots__ = ('id', 'is_vegetation', 'vertex_count', 'vertex_size', 'flags', 'data')
    def __init__(self):
        self.id = 0
        self.is_vegetation = False
        self.vertex_count = 0
        self.vertex_size = 0
        self.flags = 0
        self.data = b''


class IndexBuffer:
    __slots__ = ('id', 'index_count', 'data')
    def __init__(self):
        self.id = 0
        self.index_count = 0
        self.data = b''


class Surface:
    __slots__ = (
        'is_vegetation', 'material_id', 'vertex_count', 'flags',
        'poly_count', 'poly_mode', 'num_indices_used',
        'center', 'radius',
        'num_streams', 'stream_ids', 'stream_offsets',
        'num_references',
    )
    def __init__(self):
        self.is_vegetation = 0
        self.material_id = 0
        self.vertex_count = 0
        self.flags = 0
        self.poly_count = 0
        self.poly_mode = 0
        self.num_indices_used = 0
        self.center = (0.0, 0.0, 0.0)
        self.radius = (0.0, 0.0, 0.0)
        self.num_streams = 0
        self.stream_ids = [0, 0]
        self.stream_offsets = [0, 0]
        self.num_references = 0


class StaticBatch:
    __slots__ = ('id1', 'bvh_id1', 'bvh_id2', 'center', 'radius')


class TreeMesh:
    __slots__ = (
        'is_bush', 'unk2', 'bvh_id1', 'bvh_id2', 'matrix', 'scale',
        'trunk_surface_id', 'branch_surface_id', 'leaf_surface_id',
        'color_id', 'lod_id', 'material_id',
    )


class TreeLOD:
    __slots__ = ('pos', 'scale', 'values')


class Model:
    __slots__ = ('name', 'unk', 'center', 'radius', 'f_radius', 'surfaces')


class ObjectDummy:
    __slots__ = ('name1', 'name2', 'flags', 'matrix')


class CollidableModel:
    __slots__ = ('models', 'center', 'radius')


class MeshDamageAssoc:
    __slots__ = ('name', 'ids')


class CompactMesh:
    __slots__ = ('name1', 'name2', 'flags', 'group', 'matrix',
                 'unk1', 'damage_assoc_id', 'models')


class BVHPrimitive:
    __slots__ = ('pos', 'radius', 'id1', 'id2')


class BVHNode:
    __slots__ = ('pos', 'radius', 'unk1', 'unk2')


class PlantEntry:
    __slots__ = ('pos', 'extent', 'surface_id', 'plant_id')


class VegetationEntry:
    """Single entry from the vegetation buffer: a billboard leaf/grass quad."""
    __slots__ = ('pos', 'scale_w', 'scale_h', 'color_idx')


# ═════════════════════════════════════════════════════════════════════════════
# Binary reader
# ═════════════════════════════════════════════════════════════════════════════
class BinaryReader:
    def __init__(self, filepath):
        with open(filepath, 'rb') as f:
            self._data = f.read()
        self._pos = 0

    def tell(self):
        return self._pos

    def seek(self, pos):
        self._pos = pos

    def skip(self, n):
        self._pos += n

    def read(self, n):
        d = self._data[self._pos:self._pos + n]
        self._pos += n
        return d

    def u32(self):
        v = struct.unpack_from('<I', self._data, self._pos)[0]
        self._pos += 4
        return v

    def i32(self):
        v = struct.unpack_from('<i', self._data, self._pos)[0]
        self._pos += 4
        return v

    def f32(self):
        v = struct.unpack_from('<f', self._data, self._pos)[0]
        self._pos += 4
        return v

    def vec3f(self):
        v = struct.unpack_from('<3f', self._data, self._pos)
        self._pos += 12
        return v

    def vec2f(self):
        v = struct.unpack_from('<2f', self._data, self._pos)
        self._pos += 8
        return v

    def string(self):
        end = self._data.index(b'\x00', self._pos)
        s = self._data[self._pos:end].decode('ascii', errors='replace')
        self._pos = end + 1
        return s

    def raw_bytes(self, n):
        d = self._data[self._pos:self._pos + n]
        self._pos += n
        return d


# ═════════════════════════════════════════════════════════════════════════════
# W32 Data container
# ═════════════════════════════════════════════════════════════════════════════
class W32Data:
    def __init__(self):
        self.version = 0
        self.some_map_value = 1
        self.materials = []
        self.vertex_buffers = []
        self.index_buffers = []
        self.surfaces = []
        self.static_batches = []
        self.tree_colors = []
        self.tree_lods = []
        self.tree_meshes = []
        self.collision_offset_matrix = [0.0] * 16
        self.models = []
        self.objects = []
        self.collidable_models = []
        self.mesh_damage_assoc = []
        self.compact_meshes = []
        self.compact_mesh_group_count = 0
        self.vertex_colors = []
        # Companion data
        self.bvh_primitives = []
        self.bvh_nodes = []
        self.plants = []
        self.plant_vdb_bounds_min = (0, 0, 0)
        self.plant_vdb_bounds_max = (0, 0, 0)
        self.effectmap = None   # MapOverlay
        self.resetmap = None    # MapOverlay

    def find_vertex_buffer(self, stream_id):
        for vb in self.vertex_buffers:
            if vb.id == stream_id:
                return vb
        return None

    def find_index_buffer(self, stream_id):
        for ib in self.index_buffers:
            if ib.id == stream_id:
                return ib
        return None


# ═════════════════════════════════════════════════════════════════════════════
# Companion file parsers
# ═════════════════════════════════════════════════════════════════════════════
def parse_vertex_colors(filepath):
    colors = []
    if not os.path.isfile(filepath):
        return colors
    with open(filepath, 'rb') as f:
        data = f.read()
    for i in range(len(data) // 4):
        colors.append(struct.unpack_from('<I', data, i * 4)[0])
    return colors


# ═════════════════════════════════════════════════════════════════════════════
# 4B Map Parsing (effectmap, resetmap)
# ═════════════════════════════════════════════════════════════════════════════
class MapOverlay:
    __slots__ = ('name', 'width', 'height', 'data', 'bounds')
    def __init__(self, name=""):
        self.name = name
        self.width = 256
        self.height = 128
        self.data = None       # raw bytes
        self.bounds = None     # (top_left_x, top_left_z, bot_right_x, bot_right_z) in FO2 space


def parse_4b_file(filepath):
    """Parse a .4b file (256×128 grayscale map)."""
    if not os.path.isfile(filepath):
        return None
    with open(filepath, 'rb') as f:
        data = f.read()
    if len(data) != 32768:  # 256 * 128
        print(f"[W32] Warning: {os.path.basename(filepath)} unexpected size {len(data)}, expected 32768")
        return None
    return data


def parse_bed_file(filepath):
    """Parse a .bed file for map bounds.
    Returns (top_left_x, top_left_z, bot_right_x, bot_right_z) in FO2 game space."""
    if not os.path.isfile(filepath):
        return None
    import re
    with open(filepath, 'r', errors='ignore') as f:
        text = f.read()
    tl = re.search(r'TopLeft\s*=\s*\{\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\}', text)
    br = re.search(r'BottomRight\s*=\s*\{\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\}', text)
    if tl and br:
        return (float(tl.group(1)), float(tl.group(2)),
                float(br.group(1)), float(br.group(2)))
    return None


def parse_track_bvh(filepath, w32):
    """Parse track_bvh.gen -> BVH primitives and nodes."""
    if not os.path.isfile(filepath):
        return False
    r = BinaryReader(filepath)
    if r.u32() != 0xDEADC0DE:
        return False
    if r.u32() != 1:
        return False

    prim_count = r.u32()
    for _ in range(prim_count):
        p = BVHPrimitive()
        p.pos = r.vec3f()
        p.radius = r.vec3f()
        p.id1 = r.i32()
        p.id2 = r.i32()
        w32.bvh_primitives.append(p)

    node_count = r.u32()
    for _ in range(node_count):
        n = BVHNode()
        n.pos = r.vec3f()
        n.radius = r.vec3f()
        n.unk1 = r.i32()
        n.unk2 = r.i32()
        w32.bvh_nodes.append(n)

    print(f"[W32] Loaded track_bvh.gen: {prim_count} primitives, {node_count} nodes")
    return True


def parse_plant_vdb(filepath, w32):
    """Parse plant_vdb.gen -> plant cluster positions."""
    if not os.path.isfile(filepath):
        return False
    r = BinaryReader(filepath)
    if r.u32() != 0x62647370:  # "psdb"
        return False
    r.u32()  # ignored
    r.u32()  # ignored
    count = r.i32()

    for _ in range(count):
        p = PlantEntry()
        p.pos = r.vec3f()
        p.extent = r.vec3f()
        p.surface_id = r.u32()
        p.plant_id = r.u32()
        w32.plants.append(p)

    r.u32()  # someData
    w32.plant_vdb_bounds_min = r.vec3f()
    w32.plant_vdb_bounds_max = r.vec3f()

    print(f"[W32] Loaded plant_vdb.gen: {count} plant clusters")
    return True


# ═════════════════════════════════════════════════════════════════════════════
# W32 Parser
# ═════════════════════════════════════════════════════════════════════════════
def parse_w32(filepath, options):
    """Parse a FlatOut 2 W32 track file and companion files."""
    r = BinaryReader(filepath)
    w = W32Data()

    w.version = r.u32()
    if w.version == 0x20002:
        raise ValueError("FOUC W32 files (0x20002) are not supported: they use "
                         "int16 vertex buffers and per-surface multipliers this "
                         "parser does not handle")
    if w.version < 0x10004 or w.version > 0x20001:
        raise ValueError(f"Unsupported W32 version: 0x{w.version:X}")

    if w.version > 0x20000:
        w.some_map_value = r.u32()
        for _ in range(w.some_map_value - 1):
            r.u32()

    base_dir = os.path.dirname(filepath)
    base_name = os.path.splitext(os.path.basename(filepath))[0]

    # Default track folder layout (Bugbear's release structure):
    #
    #   <track>/data/       (atmosphere.ini, camera.ini, resetmap.bed, ...)
    #   <track>/geometry/   (track_geom.w32, track_bvh.gen, plant_vdb.gen, ...)
    #   <track>/lighting/   (vertexcolors_w2.w32, plantcolors_w2.w32,
    #                        shadowmap_w2.dat, sh_w2.ini, ...)
    #
    # When the user imports track_geom.w32 from geometry/, companion files
    # live in the sibling data/ and lighting/ folders. Search all of them.
    parent_dir = os.path.dirname(base_dir) if base_dir else base_dir
    search_dirs = [base_dir]
    if parent_dir and parent_dir != base_dir:
        for sib in ('lighting', 'data', 'geometry'):
            p = os.path.join(parent_dir, sib)
            if os.path.isdir(p) and p not in search_dirs:
                search_dirs.append(p)
        if parent_dir not in search_dirs:
            search_dirs.append(parent_dir)

    def find_companion(names):
        """Return the first existing path where any of `names` is found in
        the search directories, or None if nothing matches."""
        for d in search_dirs:
            for n in names:
                p = os.path.join(d, n)
                if os.path.isfile(p):
                    return p
        return None

    # -- Load vertex colors --
    # Every prelit vertex color in the w32 is an index into this LUT; without
    # it, colors import as white placeholders (and export bakes white).
    prefix = base_name
    for suffix in ('_track_geom', '_geom'):
        if prefix.endswith(suffix):
            prefix = prefix[:-len(suffix)]
            break
    vc_path = find_companion([
        base_name + "_vertexcolors.w32",
        base_name + "_vertexcolors_w2.w32",
        prefix + "_vertexcolors_w2.w32",
        "vertexcolors_w2.w32",
    ])
    if vc_path is not None:
        w.vertex_colors = parse_vertex_colors(vc_path)
        print(f"[W32] Loaded {len(w.vertex_colors)} vertex colors from {vc_path}")
    else:
        print("[W32] WARNING: no vertexcolors_w2.w32 found in the track's "
              "folder or its lighting/ sibling - prelit colors will import "
              "as white.")

    # -- Load track BVH --
    if options.get('import_bvh', False):
        bvh_path = find_companion([
            base_name + "_bvh.gen",
            base_name + "_track_bvh.gen",
            "track_bvh.gen",
        ])
        if bvh_path is not None:
            parse_track_bvh(bvh_path, w)

    # -- Load plant VDB --
    if options.get('import_plants', False):
        vdb_path = find_companion([
            base_name + "_plant_vdb.gen",
            "plant_vdb.gen",
        ])
        if vdb_path is not None:
            parse_plant_vdb(vdb_path, w)

    # -- Preserve the scatter-plant companion files verbatim --
    # plant_geom.w32 (instance positions), plantcolors_w2.w32 (per-instance
    # colors) and plant_vdb.gen (plant TYPE prototypes) drive the game's
    # scatter plants. plant_geom/plantcolors contain no track-surface indices,
    # so they round-trip byte-for-byte; plant_vdb.surfaceId references trunk/
    # branch surfaces and gets remapped on export. Store as hex (int arrays
    # corrupt in Blender). Attached to the root collection below.
    w._plant_geom_raw = b''
    w._plantcolors_raw = b''
    w._plant_vdb_raw = b''
    pg_path = find_companion([base_name + "_plant_geom.w32", "plant_geom.w32"])
    if pg_path:
        with open(pg_path, 'rb') as f:
            w._plant_geom_raw = f.read()
    pc_path = find_companion([base_name + "_plantcolors_w2.w32",
                              "plantcolors_w2.w32"])
    if pc_path:
        with open(pc_path, 'rb') as f:
            w._plantcolors_raw = f.read()
    pvdb_path = find_companion([base_name + "_plant_vdb.gen", "plant_vdb.gen"])
    if pvdb_path:
        with open(pvdb_path, 'rb') as f:
            w._plant_vdb_raw = f.read()

    # -- Load effectmap and resetmap 4B overlays --
    if options.get('import_maps', False):
        # Parse resetmap.bed for bounds (used by both maps)
        bed_path = find_companion(["resetmap.bed"])
        map_bounds = parse_bed_file(bed_path) if bed_path else None

        # Effectmap
        eff_path = find_companion(["effectmap.4b"])
        eff_data = parse_4b_file(eff_path) if eff_path else None
        if eff_data is not None:
            w.effectmap = MapOverlay("effectmap")
            w.effectmap.data = eff_data
            w.effectmap.bounds = map_bounds
            print(f"[W32] Loaded {eff_path}")

        # Resetmap
        rst_path = find_companion(["resetmap.4b"])
        rst_data = parse_4b_file(rst_path) if rst_path else None
        if rst_data is not None:
            w.resetmap = MapOverlay("resetmap")
            w.resetmap.data = rst_data
            w.resetmap.bounds = map_bounds
            print(f"[W32] Loaded {rst_path}")

    # == Materials ==
    num_materials = r.u32()
    for i in range(num_materials):
        mat = Material()
        ident = r.u32()
        if ident != 0x4354414D:
            raise ValueError(f"Invalid material identifier at material {i}")
        mat.name = r.string()
        mat.alpha = r.i32()
        mat.v92 = r.i32()
        mat.num_textures = r.i32()
        mat.shader_id = r.i32()
        mat.use_colormap = r.i32()
        mat.v74 = r.i32()
        mat.v108 = struct.unpack('<3i', r.read(12))
        mat.v109 = struct.unpack('<3i', r.read(12))
        mat.v98 = struct.unpack('<4i', r.read(16))
        mat.v99 = struct.unpack('<4i', r.read(16))
        mat.v100 = struct.unpack('<4i', r.read(16))
        mat.v101 = struct.unpack('<4i', r.read(16))
        mat.v102 = r.i32()
        mat.texture_names = [r.string(), r.string(), r.string()]
        w.materials.append(mat)

    # == Streams ==
    num_streams = r.u32()
    for i in range(num_streams):
        data_type = r.u32()
        if data_type == 1:
            vb = VertexBuffer()
            vb.id = i
            _fouc = r.u32()
            vb.vertex_count = r.u32()
            vb.vertex_size = r.u32()
            vb.flags = r.u32()
            vb.data = r.raw_bytes(vb.vertex_count * vb.vertex_size)
            w.vertex_buffers.append(vb)
        elif data_type == 2:
            ib = IndexBuffer()
            ib.id = i
            _fouc = r.u32()
            ib.index_count = r.u32()
            ib.data = r.raw_bytes(ib.index_count * 2)
            w.index_buffers.append(ib)
        elif data_type == 3:
            vb = VertexBuffer()
            vb.id = i
            vb.is_vegetation = True
            _fouc = r.u32()
            vb.vertex_count = r.u32()
            vb.vertex_size = r.u32()
            vb.flags = 0
            vb.data = r.raw_bytes(vb.vertex_count * vb.vertex_size)
            w.vertex_buffers.append(vb)
        else:
            raise ValueError(f"Unknown stream type {data_type} at stream {i}")

    # == Surfaces ==
    num_surfaces = r.u32()
    for i in range(num_surfaces):
        s = Surface()
        s.is_vegetation = r.i32()
        s.material_id = r.i32()
        s.vertex_count = r.i32()
        s.flags = r.i32()
        s.poly_count = r.i32()
        s.poly_mode = r.i32()
        s.num_indices_used = r.i32()
        if w.version < 0x20000:
            s.center = r.vec3f()
            s.radius = r.vec3f()
        s.num_streams = r.i32()
        s.stream_ids = [0, 0]
        s.stream_offsets = [0, 0]
        for j in range(s.num_streams):
            s.stream_ids[j] = r.u32()
            s.stream_offsets[j] = r.u32()
        w.surfaces.append(s)

    # == Static Batches ==
    num_batches = r.u32()
    for i in range(num_batches):
        b = StaticBatch()
        b.id1 = r.u32()
        b.bvh_id1 = r.u32()
        b.bvh_id2 = r.u32()
        if w.version >= 0x20000:
            b.center = r.vec3f()
            b.radius = r.vec3f()
            if b.bvh_id1 < len(w.surfaces):
                w.surfaces[b.bvh_id1].center = b.center
                w.surfaces[b.bvh_id1].radius = b.radius
        else:
            _unk = r.u32()
            b.center = (0, 0, 0)
            b.radius = (0, 0, 0)
        w.static_batches.append(b)

    # == Tree Colors ==
    tc_count = r.u32()
    w.tree_colors = [r.u32() for _ in range(tc_count)]

    # == Tree LODs ==
    tl_count = r.u32()
    for _ in range(tl_count):
        lod = TreeLOD()
        lod.pos = r.vec3f()
        lod.scale = r.vec2f()
        lod.values = (r.u32(), r.u32())
        w.tree_lods.append(lod)

    # == Tree Meshes ==
    tm_count = r.u32()
    for _ in range(tm_count):
        tm = TreeMesh()
        tm.is_bush = r.i32()
        tm.unk2 = r.i32()
        tm.bvh_id1 = r.i32()
        tm.bvh_id2 = r.i32()
        tm.matrix = list(struct.unpack('<16f', r.read(64)))
        tm.scale = r.vec3f()
        tm.trunk_surface_id = r.i32()
        tm.branch_surface_id = r.i32()
        tm.leaf_surface_id = r.i32()
        tm.color_id = r.i32()
        tm.lod_id = r.i32()
        tm.material_id = r.i32()
        w.tree_meshes.append(tm)

    # == Collision Offset Matrix ==
    if w.version >= 0x10004:
        w.collision_offset_matrix = list(struct.unpack('<16f', r.read(64)))

    # == Models ==
    mc = r.u32()
    for _ in range(mc):
        m = Model()
        if r.u32() != 0x444F4D42:
            raise ValueError("Invalid model identifier")
        m.unk = r.u32()
        m.name = r.string()
        m.center = r.vec3f()
        m.radius = r.vec3f()
        m.f_radius = r.f32()
        ns = r.u32()
        m.surfaces = [r.i32() for _ in range(ns)]
        w.models.append(m)

    # == Objects ==
    oc = r.u32()
    for _ in range(oc):
        o = ObjectDummy()
        if r.u32() != 0x434A424F:
            raise ValueError("Invalid object identifier")
        o.name1 = r.string()
        o.name2 = r.string()
        o.flags = r.u32()
        o.matrix = list(struct.unpack('<16f', r.read(64)))
        w.objects.append(o)

    # == Collidable Models (v >= 0x20000) ==
    if w.version >= 0x20000:
        cc = r.u32()
        for _ in range(cc):
            cm = CollidableModel()
            mc2 = r.u32()
            cm.models = [r.u32() for _ in range(mc2)]
            cm.center = r.vec3f()
            cm.radius = r.vec3f()
            w.collidable_models.append(cm)

        ac = r.u32()
        for _ in range(ac):
            mda = MeshDamageAssoc()
            mda.name = r.string()
            mda.ids = (r.i32(), r.i32())
            w.mesh_damage_assoc.append(mda)

    # == Compact Meshes ==
    w.compact_mesh_group_count = r.u32()
    cmc = r.u32()
    for _ in range(cmc):
        cm = CompactMesh()
        if r.u32() != 0x4853454D:
            raise ValueError("Invalid compact mesh identifier")
        cm.name1 = r.string()
        cm.name2 = r.string()
        cm.flags = r.u32()
        cm.group = r.i32()
        cm.matrix = list(struct.unpack('<16f', r.read(64)))
        if w.version >= 0x20000:
            cm.unk1 = r.u32()
            cm.damage_assoc_id = r.u32()
            if cm.damage_assoc_id < len(w.mesh_damage_assoc):
                assoc = w.mesh_damage_assoc[cm.damage_assoc_id]
                if assoc.ids[0] < len(w.collidable_models):
                    cm.models = list(w.collidable_models[assoc.ids[0]].models)
                else:
                    cm.models = []
            else:
                cm.models = []
        else:
            lc = r.u32()
            cm.models = [r.u32() for _ in range(lc)]
        w.compact_meshes.append(cm)

    print(f"[W32] Parsed: {len(w.materials)} materials, {len(w.surfaces)} surfaces, "
          f"{len(w.static_batches)} batches, {len(w.tree_meshes)} trees, "
          f"{len(w.models)} models, {len(w.objects)} objects, "
          f"{len(w.compact_meshes)} compact meshes")
    return w


# ═════════════════════════════════════════════════════════════════════════════
# Coordinate system: FO2 (Y-up) -> Blender (Z-up)
# ═════════════════════════════════════════════════════════════════════════════
def fo2_to_blender_pos(x, y, z):
    return (x, z, y)


def fo2_to_blender_normal(x, y, z):
    return (x, z, y)


def fo2_matrix_to_blender(m):
    """FO2 4x4 -> Blender Matrix.
    FO2 uses row-vector convention (v*M) with translation in last row (m[12..14]).
    Blender uses column-vector convention (M*v) with translation in last column.
    So we transpose, then apply the Y/Z coordinate swap.
    Swap: (x,y,z) -> (x, z, y) matching fo2_to_blender_pos."""
    raw = mathutils.Matrix((
        (m[0],  m[4],  m[8],  m[12]),
        (m[1],  m[5],  m[9],  m[13]),
        (m[2],  m[6],  m[10], m[14]),
        (m[3],  m[7],  m[11], m[15]),
    ))
    swap = mathutils.Matrix((
        (1, 0, 0, 0),
        (0, 0, 1, 0),
        (0, 1, 0, 0),
        (0, 0, 0, 1),
    ))
    return swap @ raw @ swap.inverted()


# ═════════════════════════════════════════════════════════════════════════════
# Vertex data extraction
# ═════════════════════════════════════════════════════════════════════════════
def resolve_vertex_color(color_val, vertex_colors_lut):
    """Resolve a vertex color value to (r,g,b,a) floats.
    Values are (bank << 24) | index into the vertexcolors LUT (banks 0/2/5 in
    vanilla data); LUT entries are D3DCOLOR dwords (0xAARRGGBB)."""
    high_byte = (color_val >> 24) & 0xFF
    if high_byte >= 0xFF:
        return (((color_val >> 16) & 0xFF) / 255.0,
                ((color_val >> 8) & 0xFF) / 255.0,
                (color_val & 0xFF) / 255.0,
                1.0)
    elif vertex_colors_lut:
        idx = color_val & 0xFFFFFF
        if idx < len(vertex_colors_lut):
            c = vertex_colors_lut[idx]
            return (((c >> 16) & 0xFF) / 255.0,
                    ((c >> 8) & 0xFF) / 255.0,
                    (c & 0xFF) / 255.0,
                    1.0)
    return (1.0, 1.0, 1.0, 1.0)


def extract_vertices(vb, surface, vertex_colors_lut):
    """Extract vertex data from a regular (non-vegetation) vertex buffer."""
    flags = vb.flags
    stride = vb.vertex_size
    offset = surface.stream_offsets[0]
    count = surface.vertex_count

    positions = []
    normals = []
    uvs = []
    uv2s = []
    colors = []

    has_normal = (flags & VERTEX_NORMAL) != 0
    has_color = (flags & VERTEX_COLOR) != 0
    has_uv = (flags & VERTEX_UV) != 0 or (flags & VERTEX_UV2) != 0
    has_uv2 = (flags & VERTEX_UV2) != 0

    for i in range(count):
        ptr = offset + i * stride

        x, y, z = struct.unpack_from('<3f', vb.data, ptr)
        positions.append(fo2_to_blender_pos(x, y, z))
        ptr += 12

        if has_normal:
            nx, ny, nz = struct.unpack_from('<3f', vb.data, ptr)
            normals.append(fo2_to_blender_normal(nx, ny, nz))
            ptr += 12

        if has_color:
            color_val = struct.unpack_from('<I', vb.data, ptr)[0]
            colors.append(resolve_vertex_color(color_val, vertex_colors_lut))
            ptr += 4

        if has_uv:
            u, v = struct.unpack_from('<2f', vb.data, ptr)
            uvs.append((u, 1.0 - v))
            ptr += 8

        if has_uv2:
            u2, v2 = struct.unpack_from('<2f', vb.data, ptr)
            uv2s.append((u2, 1.0 - v2))
            ptr += 8

    return positions, normals, uvs, uv2s, colors


def extract_vegetation_quads(vb, surface, vertex_colors_lut):
    """Extract billboard quads from a vegetation buffer surface.
    Each vegetation entry (28 bytes):
      float pos[3], float scale_w, float scale_h, float pad, uint32 color_idx
    The surface poly_count = number of entries; vertex_count = poly_count * 4."""
    veg_stride = vb.vertex_size  # 28
    byte_offset = surface.stream_offsets[0]
    num_entries = surface.poly_count

    positions = []
    colors = []
    uvs = []

    for i in range(num_entries):
        ptr = byte_offset + i * veg_stride

        px, py, pz = struct.unpack_from('<3f', vb.data, ptr)
        sw, sh = struct.unpack_from('<2f', vb.data, ptr + 12)
        # bytes 20-23 are the atlas UV rect (u0, v0, u1, v1) in sixteenths of
        # the texture (0..16 -> 0.0..1.0). Vegetation textures are sprite
        # atlases; each billboard samples one sub-rectangle. The importer
        # previously assumed the whole texture (0,0)-(1,1), which is wrong for
        # any surface whose billboards use different atlas cells (e.g. bush
        # sprites). Verified on 4 tracks: bytes are always <=16 and ordered.
        au0, av0, au1, av1 = vb.data[ptr + 20:ptr + 24]
        color_idx = struct.unpack_from('<I', vb.data, ptr + 24)[0]

        col = (1.0, 1.0, 1.0, 1.0)
        if vertex_colors_lut and color_idx < len(vertex_colors_lut):
            c = vertex_colors_lut[color_idx]
            col = ((c & 0xFF) / 255.0, ((c >> 8) & 0xFF) / 255.0,
                   ((c >> 16) & 0xFF) / 255.0, 1.0)

        # Billboard quad, FO2 space -> Blender. Final semantics, calibrated
        # against the engine's own tree-LOD billboards (same record family:
        # every tree_lod satisfies pos.y - scale_h == tree base EXACTLY, i.e.
        # pos is the billboard CENTER and the scales are HALF-extents):
        #   - pos = sprite CENTER; quad spans pos +/- (sw, sh)
        #   - quad is 2*sw wide, 2*sh tall (at 1x the composited foliage is
        #     sparse confetti; 2x forms the dense canopy the game renders)
        # The earlier top-anchor reading scored well only against a biased
        # proxy (branch-plane bounds); sprites legitimately puff above the
        # branch geometry, and top-anchoring sank every bush half a sprite
        # into the ground.
        v0 = fo2_to_blender_pos(px - sw, py - sh, pz)
        v1 = fo2_to_blender_pos(px + sw, py - sh, pz)
        v2 = fo2_to_blender_pos(px + sw, py + sh, pz)
        v3 = fo2_to_blender_pos(px - sw, py + sh, pz)

        # Atlas sub-rect UVs. Unlike regular surface UVs (top-down, imported
        # as 1-v), the rect's V bytes measure from the image BOTTOM - verified
        # against the sprite art itself: rect borders land on fully
        # transparent pixels only under the bottom-up reading (mean border
        # alpha 0.0 vs up to 28 for top-down cuts through opaque sprites).
        # So Blender V (0 = image bottom) uses the bytes directly, no flip.
        # Full-atlas rects (0,0,16,16) still map to (0,0)-(1,1) unchanged.
        u_l, u_r = au0 / 16.0, au1 / 16.0
        v_bot, v_top = av0 / 16.0, av1 / 16.0

        positions.extend([v0, v1, v2, v3])
        colors.extend([col, col, col, col])
        uvs.extend([(u_l, v_bot), (u_r, v_bot), (u_r, v_top), (u_l, v_top)])

    faces = []
    for i in range(num_entries):
        b = i * 4
        faces.append((b, b + 1, b + 2))
        faces.append((b, b + 2, b + 3))

    return positions, uvs, colors, faces


def extract_faces(ib, surface, vb_stride):
    """Extract triangle faces from an index buffer."""
    offset = surface.stream_offsets[1]
    base_vertex = surface.stream_offsets[0] // vb_stride
    faces = []

    if surface.poly_mode == 5:
        # Triangle strip. Standard GPU decode:
        #   even j -> (i0, i1, i2)   odd j -> (i1, i0, i2)
        # We flip winding on import (matching what the pm=4 branch does with
        # (i2, i1, i0)) so the exporter's own winding flip lands the game's
        # actual GPU triangles back as GPU triangles. Fixed:
        #   even j -> (i2, i1, i0)   odd j -> (i2, i0, i1)
        # Previously the odd-j branch emitted (i0, i1, i2), which is the
        # unreversed strip winding - every other pm=5 triangle came out with
        # inverted normals and vanished under back-face culling on re-export.
        flip = False
        for j in range(surface.poly_count):
            idx_off = offset + j * 2
            i0 = struct.unpack_from('<H', ib.data, idx_off)[0] - base_vertex
            i1 = struct.unpack_from('<H', ib.data, idx_off + 2)[0] - base_vertex
            i2 = struct.unpack_from('<H', ib.data, idx_off + 4)[0] - base_vertex
            vc = surface.vertex_count
            if i0 < 0 or i1 < 0 or i2 < 0 or i0 >= vc or i1 >= vc or i2 >= vc:
                flip = not flip
                continue
            if i0 == i1 or i1 == i2 or i0 == i2:
                flip = not flip
                continue
            faces.append((i2, i0, i1) if flip else (i2, i1, i0))
            flip = not flip
    elif surface.poly_mode == 4:
        for j in range(surface.poly_count):
            idx_off = offset + j * 6
            i0 = struct.unpack_from('<H', ib.data, idx_off)[0] - base_vertex
            i1 = struct.unpack_from('<H', ib.data, idx_off + 2)[0] - base_vertex
            i2 = struct.unpack_from('<H', ib.data, idx_off + 4)[0] - base_vertex
            vc = surface.vertex_count
            if i0 < 0 or i1 < 0 or i2 < 0 or i0 >= vc or i1 >= vc or i2 >= vc:
                continue
            faces.append((i2, i1, i0))
    return faces


# ═════════════════════════════════════════════════════════════════════════════
# Blender material creation
# ═════════════════════════════════════════════════════════════════════════════
def _try_set(obj, attr, val):
    """Safely set an attribute, swallowing exceptions from removed APIs."""
    if hasattr(obj, attr):
        try:
            setattr(obj, attr, val)
        except (TypeError, AttributeError):
            pass


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
        print(f"[W32] could not load {name}: {e}")
    return None

_DDS2TGA = None
def _dds_to_tga(dds_path, out_dir):
    """Convert a DDS to TGA in out_dir; return the TGA path or '' on failure."""
    global _DDS2TGA
    if _DDS2TGA is None:
        _DDS2TGA = _load_sibling_module('dds2tga') or False
    if not _DDS2TGA:
        return ''
    base = os.path.splitext(os.path.basename(dds_path))[0]
    out_tga = os.path.join(out_dir, base + '.tga')
    try:
        if not os.path.isfile(out_tga):
            _DDS2TGA.convert_dds_to_tga(dds_path, out_tga)
            print(f"[W32] Converted: {os.path.basename(dds_path)} -> {base}.tga")
        return out_tga
    except Exception as exc:
        print(f"[W32] DDS->TGA failed for {os.path.basename(dds_path)}: {exc}")
        return ''


def _find_texture_image(tex_name, search_dirs, convert_dds=False):
    """Find and load a texture image. TGA has absolute priority in every
    directory; if only a DDS is found and convert_dds is set, it is converted
    to an editable TGA (placed beside the DDS's search dir) and that is loaded.
    Case-insensitive. Follows the fo2_bgm_import resolution convention."""
    if not tex_name:
        return None
    base_tex = os.path.splitext(tex_name)[0]
    base_lower = base_tex.lower()

    # Reuse an already-loaded image
    for existing in bpy.data.images:
        existing_base = os.path.splitext(os.path.basename(existing.filepath or existing.name))[0]
        if existing_base.lower() == base_lower:
            return existing

    found_dds = ""
    for sdir in search_dirs:
        if not sdir or not os.path.isdir(sdir):
            continue
        try:
            entries = os.listdir(sdir)
        except OSError:
            continue
        lower = {e.lower(): e for e in entries}
        # TGA first (absolute priority)
        for ext in ('.tga', '.png'):
            hit = lower.get(base_lower + ext)
            if hit:
                try:
                    return bpy.data.images.load(os.path.join(sdir, hit))
                except Exception:
                    pass
        # remember first DDS for fallback / conversion
        if not found_dds:
            hit = lower.get(base_lower + '.dds')
            if hit:
                found_dds = os.path.join(sdir, hit)

    if found_dds:
        if convert_dds:
            out_dir = next((d for d in search_dirs if d and os.path.isdir(d)),
                           os.path.dirname(found_dds))
            tga = _dds_to_tga(found_dds, out_dir)
            if tga:
                try:
                    return bpy.data.images.load(tga)
                except Exception:
                    pass
        try:
            return bpy.data.images.load(found_dds)   # let Blender try the DDS
        except Exception:
            pass
    return None


def _create_multiply_node(nodes, links, color_a, color_b, location=(0, 0)):
    """Create a Multiply mix node for Blender 4.x/5.x.
    color_a, color_b: (node, output_name) tuples.
    Returns (node, output_name) for the result."""
    mix = nodes.new('ShaderNodeMix')
    mix.location = location
    mix.data_type = 'RGBA'
    mix.blend_type = 'MULTIPLY'
    # Factor is input 0
    mix.inputs[0].default_value = 1.0
    # Color A = input 6, Color B = input 7
    links.new(color_a[0].outputs[color_a[1]], mix.inputs[6])
    links.new(color_b[0].outputs[color_b[1]], mix.inputs[7])
    # Color Result = output 2
    return mix, 2


_VANILLA_DIRS_CACHE = {}
def _ci_subdirs(parent, name):
    """Return case-insensitive matches of a sub-folder `name` under parent."""
    out = []
    try:
        for e in os.listdir(parent):
            if e.lower() == name.lower() and os.path.isdir(os.path.join(parent, e)):
                out.append(os.path.join(parent, e))
    except OSError:
        pass
    return out


def _vanilla_texture_dirs(w32_dir):
    """Resolve the game's vanilla texture folders from a track's import path.
    Tracks live at <flatout>/tracks/<theme>/<track>/<variant>/geometry/; their
    diffuse textures live at <flatout>/tracks/<theme>/textures/, and shared art
    lives at <flatout>/global/<category>/. Returns the theme textures folder
    first (track-specific), then global/ plus its category sub-folders. []
    if the path is not under a tracks/ tree. Cached per import directory."""
    if not w32_dir:
        return []
    key = os.path.abspath(w32_dir)
    if key in _VANILLA_DIRS_CACHE:
        return _VANILLA_DIRS_CACHE[key]

    dirs = []
    parts = key.replace('\\', '/').split('/')
    idx = None
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == 'tracks':
            idx = i
            break
    if idx is not None:
        # <flatout>/tracks/<theme>/textures  (primary vanilla location)
        if idx + 1 < len(parts):
            theme_root = '/'.join(parts[:idx + 2])   # <root>/tracks/<theme>
            dirs += _ci_subdirs(theme_root, 'textures')
        # <flatout>/global + its category sub-folders (shared art)
        root = '/'.join(parts[:idx]) if idx >= 1 else '/'.join(parts[:1])
        for gdir in _ci_subdirs(root, 'global'):
            dirs.append(gdir)
            try:
                for sub in sorted(os.listdir(gdir)):
                    sp = os.path.join(gdir, sub)
                    if os.path.isdir(sp):
                        dirs.append(sp)
            except OSError:
                pass

    _VANILLA_DIRS_CACHE[key] = dirs
    return dirs


def _set_image_alpha_mode(img, keep_alpha):
    """Set the image datablock's alpha handling: materials with alpha=0 get
    their texture's Alpha set to 'None' (ignore the channel) instead of
    'Straight'. Images are SHARED datablocks, so once any alpha-enabled
    material (or the colormap path, which always keeps its alpha) claims an
    image, alpha-disabled users of the same image can no longer turn it off."""
    if img is None:
        return
    try:
        if keep_alpha:
            img["fo2_keep_alpha"] = True
            img.alpha_mode = 'STRAIGHT'
        elif not img.get("fo2_keep_alpha"):
            img.alpha_mode = 'NONE'
    except Exception:
        pass


def create_blender_material(mat, tex_dir, mat_index, extra_tex_dir="", convert_dds=False):
    bl_name = mat.name if mat.name else f"Material_{mat_index}"
    bl_mat = bpy.data.materials.new(name=bl_name)
    bl_mat.use_nodes = True
    # ── FlatOut Shader panel wiring (fo2_bgm_import addon) ──
    # The panel displays via the RNA properties fo2_shader_id (enum) and
    # fo2_texture (string) and stores its edits in bgm_* ID properties; the
    # exporter reads the bgm_* props back. Property names and update order
    # deliberately match fo2_bgm_import/bgm_common.py so one panel serves
    # BGM and W32 materials identically.
    bl_mat["bgm_shader_id"] = mat.shader_id
    # The panel's "BGM Texture" field should show the meaningful diffuse.
    # Colormap materials (terrain) carry the "colormap.tga" placeholder in
    # slot 0 and their real detail texture in slot 1 - display that instead.
    _is_cmap0 = mat.texture_names[0].lower() in ("colormap.tga", "colormap.dds")
    display_tex = mat.texture_names[1] if (_is_cmap0 and mat.texture_names[1]) \
        else mat.texture_names[0]
    # Sync the RNA props FIRST: their update callbacks (_shader_update /
    # _texture_update) rewrite bgm_shader_id / bgm_texture_0 and may force
    # bgm_alpha for window/light/tire shaders, so the file's own values below
    # must come after to win (bgm_texture_0 in particular must stay the
    # colormap placeholder for terrain materials). Guarded: the RNA props
    # only exist when the shader-panel addon is registered.
    try:
        bl_mat.fo2_shader_id = str(mat.shader_id)
    except Exception:
        pass
    try:
        bl_mat.fo2_texture = display_tex
    except Exception:
        pass
    bl_mat["bgm_alpha"] = mat.alpha
    bl_mat["bgm_use_colormap"] = mat.use_colormap
    bl_mat["bgm_v92"] = mat.v92
    bl_mat["bgm_v74"] = mat.v74
    bl_mat["bgm_v102"] = mat.v102
    bl_mat["bgm_num_textures"] = mat.num_textures
    bl_mat["bgm_texture"] = display_tex
    for ti in range(3):
        bl_mat[f"bgm_texture_{ti}"] = mat.texture_names[ti]

    nodes = bl_mat.node_tree.nodes
    links = bl_mat.node_tree.links
    for n in nodes:
        nodes.remove(n)

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (600, 0)
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (200, 0)
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    # Alpha blend setup
    if mat.alpha:
        for enum_val in ('CLIP', 'ALPHA_CLIP'):
            try:
                _try_set(bl_mat, 'blend_method', enum_val)
                break
            except:
                pass
        _try_set(bl_mat, 'use_backface_culling', False)
    else:
        _try_set(bl_mat, 'use_backface_culling', True)

    # Texture search directories, in priority order:
    #   1. user-specified Textures Folder (extra_tex_dir)
    #   2. the .w32's own folder (where the track was imported from)
    #   3. the game's vanilla global/ folder (derived from the .w32 path)
    search_dirs = [
        os.path.join(tex_dir, "textures"),
        os.path.join(tex_dir, "Textures"),
        tex_dir,
        # sibling lighting/ folder holds the per-variant lightmap*_w2.dds that
        # the terrain shaders substitute for the "colormap.tga" placeholder
        os.path.join(os.path.dirname(tex_dir), "lighting"),
    ]
    if extra_tex_dir:
        search_dirs = [
            extra_tex_dir,
            os.path.join(extra_tex_dir, "textures"),
            os.path.join(extra_tex_dir, "Textures"),
        ] + search_dirs
    # vanilla textures: <flatout>/tracks/<theme>/textures/ (track diffuse maps)
    # plus <flatout>/global/<category>/ (shared art) - resolved from the path.
    search_dirs = search_dirs + _vanilla_texture_dirs(tex_dir)

    sid = mat.shader_id

    # ── Terrain shaders (1, 2): lightmap(UV1) × detail(UV2) ──
    # The shader (emu_lightmapped.sha) binds:
    #   Tex0 = lightmap/colormap on texcoord0 (UV1)
    #   Tex1 = detail texture on texcoord1 (UV2)
    # Stage 0: SelectArg1 (Tex0), Stage 1: Modulate Tex1 × Current
    # Result: lightmap × detail
    # In the W32, tex_names[0]="colormap.tga" is a placeholder - the engine
    # substitutes the real lightmap at runtime (e.g. lightmap1_w2.dds)
    if sid in (1, 2):
        bsdf.inputs['Roughness'].default_value = 0.8 if sid == 1 else 0.5

        # Detail texture on UV2 (texcoord1)
        detail_img = _find_texture_image(mat.texture_names[1], search_dirs, convert_dds)
        # Lightmap: resolve "colormap.tga" placeholder to actual lightmap file
        cmap_name = mat.texture_names[0]
        cmap_img = None
        if cmap_name.lower() in ("colormap.tga", "colormap.dds"):
            # Search for lightmap*.dds in the track directory
            for sdir in search_dirs:
                if not os.path.isdir(sdir):
                    continue
                try:
                    for fn in sorted(os.listdir(sdir)):
                        if fn.lower().startswith("lightmap") and fn.lower().endswith(('.tga', '.dds', '.png')):
                            cmap_img = _find_texture_image(fn, search_dirs, convert_dds)
                            if cmap_img:
                                break
                except:
                    pass
                if cmap_img:
                    break
        else:
            cmap_img = _find_texture_image(cmap_name, search_dirs, convert_dds)

        # colormap/lightmap ALWAYS keeps its alpha; the detail texture follows
        # the material's alpha flag (alpha=0 -> image Alpha set to 'None')
        _set_image_alpha_mode(cmap_img, True)
        _set_image_alpha_mode(detail_img, bool(mat.alpha))

        if detail_img and cmap_img:
            # Both textures: lightmap × detail
            cmap_node = nodes.new('ShaderNodeTexImage')
            cmap_node.location = (-600, 200)
            cmap_node.label = "lightmap"
            cmap_node.image = cmap_img

            uv1 = nodes.new('ShaderNodeUVMap')
            uv1.location = (-850, 200)
            uv1.uv_map = "UVMap"
            links.new(uv1.outputs['UV'], cmap_node.inputs['Vector'])

            detail_node = nodes.new('ShaderNodeTexImage')
            detail_node.location = (-600, -150)
            detail_node.label = mat.texture_names[1]
            detail_node.image = detail_img

            uv2 = nodes.new('ShaderNodeUVMap')
            uv2.location = (-850, -150)
            uv2.uv_map = "UVMap2"
            links.new(uv2.outputs['UV'], detail_node.inputs['Vector'])

            mix, mix_out = _create_multiply_node(
                nodes, links,
                (cmap_node, 'Color'), (detail_node, 'Color'),
                location=(-200, 100))
            links.new(mix.outputs[mix_out], bsdf.inputs['Base Color'])

        elif detail_img:
            # Only detail found: use it directly
            detail_node = nodes.new('ShaderNodeTexImage')
            detail_node.location = (-400, 0)
            detail_node.label = mat.texture_names[1]
            detail_node.image = detail_img
            links.new(detail_node.outputs['Color'], bsdf.inputs['Base Color'])

        elif cmap_img:
            # Only lightmap found: use it directly
            cmap_node = nodes.new('ShaderNodeTexImage')
            cmap_node.location = (-400, 0)
            cmap_node.label = "lightmap"
            cmap_node.image = cmap_img
            links.new(cmap_node.outputs['Color'], bsdf.inputs['Base Color'])

    # ── Static prelit (0): texture × vertex colors ──
    elif sid == 0:
        tex_name = mat.texture_names[0]
        tex_img = _find_texture_image(tex_name, search_dirs, convert_dds)
        _set_image_alpha_mode(tex_img, bool(mat.alpha))

        if tex_img:
            tex_node = nodes.new('ShaderNodeTexImage')
            tex_node.location = (-600, 200)
            tex_node.label = tex_name
            tex_node.image = tex_img

            # Vertex color attribute
            vc_node = nodes.new('ShaderNodeAttribute')
            vc_node.location = (-600, -100)
            vc_node.attribute_name = "Color"
            vc_node.attribute_type = 'GEOMETRY'

            mix, mix_out = _create_multiply_node(
                nodes, links,
                (tex_node, 'Color'), (vc_node, 'Color'),
                location=(-200, 100))
            links.new(mix.outputs[mix_out], bsdf.inputs['Base Color'])

            if mat.alpha:
                links.new(tex_node.outputs['Alpha'], bsdf.inputs['Alpha'])
        else:
            # No texture: just use vertex colors
            vc_node = nodes.new('ShaderNodeAttribute')
            vc_node.location = (-400, 0)
            vc_node.attribute_name = "Color"
            vc_node.attribute_type = 'GEOMETRY'
            links.new(vc_node.outputs['Color'], bsdf.inputs['Base Color'])

    # ── Tree branch/leaf (20, 21): texture with alpha, double-sided ──
    elif sid in (20, 21):
        _try_set(bl_mat, 'use_backface_culling', False)
        if mat.alpha:
            for enum_val in ('CLIP', 'ALPHA_CLIP'):
                try:
                    _try_set(bl_mat, 'blend_method', enum_val)
                    break
                except:
                    pass

        tex_name = mat.texture_names[0]
        tex_img = _find_texture_image(tex_name, search_dirs, convert_dds)
        _set_image_alpha_mode(tex_img, bool(mat.alpha))
        tex_node = nodes.new('ShaderNodeTexImage')
        tex_node.location = (-400, 0)
        tex_node.label = tex_name or "texture"
        if tex_img:
            tex_node.image = tex_img
            links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
            # alpha-disabled materials get NO alpha link (BSDF Alpha stays 1.0)
            if mat.alpha:
                links.new(tex_node.outputs['Alpha'], bsdf.inputs['Alpha'])

    # ── Water (25) ──
    elif sid == 25:
        bsdf.inputs['Roughness'].default_value = 0.05
        bsdf.inputs['IOR'].default_value = 1.33
        bsdf.inputs['Base Color'].default_value = (0.1, 0.2, 0.35, 1.0)

    # ── Default: single texture on UV1 ──
    else:
        # Use tex1 if available (some materials store diffuse there), else tex0
        tex_name = mat.texture_names[1] if mat.texture_names[1] else mat.texture_names[0]
        tex_img = _find_texture_image(tex_name, search_dirs, convert_dds)
        _set_image_alpha_mode(tex_img, bool(mat.alpha))

        if tex_img or tex_name:
            tex_node = nodes.new('ShaderNodeTexImage')
            tex_node.location = (-400, 0)
            tex_node.label = tex_name or "texture"
            if tex_img:
                tex_node.image = tex_img
                links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
                if mat.alpha:
                    links.new(tex_node.outputs['Alpha'], bsdf.inputs['Alpha'])

    return bl_mat


# ═════════════════════════════════════════════════════════════════════════════
# Mesh creation helpers
# ═════════════════════════════════════════════════════════════════════════════
def apply_custom_normals(mesh, normals):
    if not normals:
        return
    mesh.update()
    # Blender 4.2+: custom split normals only display on smooth-shaded
    # polygons (same requirement we hit in the BGM importer).
    try:
        mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    except Exception:
        pass
    try:
        mesh.normals_split_custom_set_from_vertices(normals)
    except (AttributeError, RuntimeError):
        try:
            loop_normals = [normals[l.vertex_index] for l in mesh.loops]
            mesh.normals_split_custom_set(loop_normals)
        except:
            pass


def apply_vertex_colors(mesh, colors, domain='POINT'):
    if not colors:
        return
    if hasattr(mesh, 'color_attributes'):
        attr = mesh.color_attributes.new(name="Color", type='FLOAT_COLOR', domain=domain)
        if domain == 'POINT':
            for vi, col in enumerate(colors):
                if vi < len(attr.data):
                    attr.data[vi].color = col
        else:
            for li in range(len(mesh.loops)):
                vi = mesh.loops[li].vertex_index
                if vi < len(colors):
                    attr.data[li].color = colors[vi]
    else:
        vc = mesh.vertex_colors.new(name="Color")
        for loop in mesh.loops:
            vi = loop.vertex_index
            if vi < len(colors):
                vc.data[loop.index].color = colors[vi]


def apply_uvs(mesh, uvs, name="UVMap"):
    if not uvs:
        return
    uv_layer = mesh.uv_layers.new(name=name)
    for loop in mesh.loops:
        vi = loop.vertex_index
        if vi < len(uvs):
            uv_layer.data[loop.index].uv = uvs[vi]


def create_mesh_from_surface(name, w32, surface, bl_materials, surface_idx):
    """Create a Blender mesh from a regular (non-vegetation) surface."""
    vb = w32.find_vertex_buffer(surface.stream_ids[0])
    if not vb:
        return None
    if surface.num_streams < 2:
        return None
    ib = w32.find_index_buffer(surface.stream_ids[1])
    if not ib:
        return None
    if surface.poly_count <= 0:
        return None

    positions, normals, uvs, uv2s, colors = extract_vertices(vb, surface, w32.vertex_colors)
    faces = extract_faces(ib, surface, vb.vertex_size)
    if not positions or not faces:
        return None

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(positions, [], faces)

    if 0 <= surface.material_id < len(bl_materials) and bl_materials[surface.material_id]:
        mesh.materials.append(bl_materials[surface.material_id])

    apply_uvs(mesh, uvs, "UVMap")
    apply_uvs(mesh, uv2s, "UVMap2")
    apply_vertex_colors(mesh, colors)
    apply_custom_normals(mesh, normals)

    mesh["fo2_surface_index"] = surface_idx
    mesh["fo2_poly_mode"] = surface.poly_mode
    mesh["fo2_flags"] = f"0x{surface.flags:X}"
    mesh.update()
    mesh.validate()
    return bpy.data.objects.new(name, mesh)


def create_vegetation_mesh(name, w32, surface, bl_materials, surface_idx):
    """Create billboard quad geometry from a vegetation surface.

    Stores the raw 28-byte-per-entry buffer verbatim on the mesh datablock so
    export can round-trip the exact billboard positions/sizes/atlas UVs/color
    indices - even though the visual Blender geometry (4 corners per quad) is
    reconstructed for editing convenience, we do NOT round-trip through it."""
    vb = w32.find_vertex_buffer(surface.stream_ids[0])
    if not vb or not vb.is_vegetation:
        return None
    if surface.poly_count <= 0:
        return None

    positions, uvs, colors, faces = extract_vegetation_quads(vb, surface, w32.vertex_colors)
    if not positions or not faces:
        return None

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(positions, [], faces)

    if 0 <= surface.material_id < len(bl_materials) and bl_materials[surface.material_id]:
        mesh.materials.append(bl_materials[surface.material_id])

    apply_uvs(mesh, uvs, "UVMap")
    apply_vertex_colors(mesh, colors)

    # Preserve the exact raw bytes of every 28-byte entry that belongs to this
    # surface. Stored as a HEX STRING: Blender's custom-property int arrays
    # do NOT round-trip losslessly (large arrays get corrupted - length and
    # values change), which silently breaks the vegetation VB on export. Hex
    # strings round-trip cleanly, same as the other preserved blobs.
    raw = bytes(vb.data[surface.stream_offsets[0]:
                        surface.stream_offsets[0] + surface.poly_count * 28])
    mesh["fo2_veg_raw_hex"] = raw.hex()
    mesh["fo2_veg_material_id"] = surface.material_id
    mesh["fo2_veg_poly_count"] = surface.poly_count
    mesh["fo2_veg_flags"] = f"0x{surface.flags:X}"

    mesh["fo2_surface_index"] = surface_idx
    mesh["fo2_vegetation"] = True
    mesh.update()
    mesh.validate()
    return bpy.data.objects.new(name, mesh)


# ═════════════════════════════════════════════════════════════════════════════
# BVH visualization
# ═════════════════════════════════════════════════════════════════════════════
def create_bvh_box_mesh():
    """Create a shared unit cube wireframe mesh for BVH visualization."""
    if "fo2_bvh_cube" in bpy.data.meshes:
        return bpy.data.meshes["fo2_bvh_cube"]

    verts = [
        (-1, -1, -1), ( 1, -1, -1), ( 1,  1, -1), (-1,  1, -1),
        (-1, -1,  1), ( 1, -1,  1), ( 1,  1,  1), (-1,  1,  1),
    ]
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7),
    ]
    mesh = bpy.data.meshes.new("fo2_bvh_cube")
    mesh.from_pydata(verts, edges, [])
    mesh.update()
    return mesh


# ═════════════════════════════════════════════════════════════════════════════
# Main import
# ═════════════════════════════════════════════════════════════════════════════
def _get_plants_material(tex_dir, options):
    """Get-or-create the shared scatter-plant material ("fo2_plants"), textured
    with the track's plants atlas (plants.tga/plants.dds, a 4x4 grid: 4 plant
    types as rows x 4 variations as columns). Uses the same texture search
    order as regular materials. Node setup is guarded so the sim harness
    (node_tree may be mocked) still works."""
    existing = bpy.data.materials.get("fo2_plants") if hasattr(bpy.data.materials, 'get') else None
    if existing:
        return existing
    bl_mat = bpy.data.materials.new(name="fo2_plants")
    bl_mat["fo2_plants_atlas"] = True
    _try_set(bl_mat, 'use_nodes', True)
    for enum_val in ('CLIP', 'ALPHA_CLIP'):
        try:
            _try_set(bl_mat, 'blend_method', enum_val)
            break
        except Exception:
            pass
    _try_set(bl_mat, 'use_backface_culling', False)

    search_dirs = []
    user_dir = options.get('texture_dir', '') if options else ''
    if user_dir:
        search_dirs += [user_dir, os.path.join(user_dir, 'textures')]
    search_dirs += [tex_dir, os.path.join(tex_dir, 'textures')]
    search_dirs += _vanilla_texture_dirs(tex_dir)
    img = _find_texture_image("plants.tga", search_dirs,
                              convert_dds=bool(options.get('convert_textures_to_tga')) if options else False)
    if img:
        try:
            nodes = bl_mat.node_tree.nodes
            links = bl_mat.node_tree.links
            for n in list(nodes):
                nodes.remove(n)
            output = nodes.new('ShaderNodeOutputMaterial')
            output.location = (600, 0)
            bsdf = nodes.new('ShaderNodeBsdfPrincipled')
            bsdf.location = (200, 0)
            links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
            tex_node = nodes.new('ShaderNodeTexImage')
            tex_node.location = (-200, 0)
            tex_node.image = img
            links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
            links.new(tex_node.outputs['Alpha'], bsdf.inputs['Alpha'])
        except Exception as e:
            print(f"[W32] plants material node setup skipped: {e}")
    else:
        print("[W32] plants atlas texture not found in search dirs")
    return bl_mat


def _create_plant_meshes(root_col, w32, tex_dir=None, options=None):
    """One mesh per plant cluster; one textured billboard QUAD per instance,
    bottom-anchored at its decoded world position (vertex order per instance:
    v0=bottom-left, v1=bottom-right, v2=top-right, v3=top-left). Moving all 4
    vertices of a quad relocates that plant; the exporter reads the bottom
    edge midpoint as the instance position. Positions are GLOBAL fractions of
    the plant_geom header bounding box (offset 72), NOT the per-cluster
    plant_vdb box - the vdb box is only a cull volume. Verified on every
    vanilla track: all instances land inside their cluster's vdb box, and
    per-cluster field means correlate 1.000 with the box centres. Stores the
    original (a,b) per instance so export re-encodes only moved positions and
    preserves the 4-bit variant field + rotation verbatim. Bit layout
    (a[0]/a[12]/a[28:32] always 0):
        X = xmin + (a>>1 &0x7FF)/2047 *(xmax-xmin)   a[1:12]  11-bit
        Z = zmin + (a>>13&0x7FF)/2047 *(zmax-zmin)   a[13:24] 11-bit
        Y = ymin + (b     &0xFFFF)/65535*(ymax-ymin) b[0:16]  16-bit
        variant = a[24:28] (0-15: the 4x4 plants-atlas cell, row=variant//4
        counted from the image TOP, col=variant%4 left-to-right)
        b[16:24] = per-instance yaw (byte/256 * 2pi about the vertical axis;
        plants are static rotated quads, not camera-facing billboards)
        b[24:32] = per-instance size fraction (lerps the per-type size
        ranges below)
    The header's 8-float array at offset 40 is the per-TYPE (min,max) SIZE
    range of the square billboard (type t -> side in [d2[2t], d2[2t+1]],
    1.2-1.4m on forest; derby tracks halve the table): plants render at the
    natural aspect of their square atlas cells. The offset-8 array
    (0.40/0.45 pairs) is not display geometry; its role is not yet
    identified (sway amplitude or cull radius are the likely candidates).
    All pairs are ordered min<=max on every vanilla track. Display quads
    lerp within the size range using the instance's preserved random
    byte."""
    raw = w32._plant_geom_raw
    if len(raw) < 4 or struct.unpack_from('<I', raw, 0)[0] != 0x62647370:
        return
    gbbox = struct.unpack_from('<6f', raw, 72)      # xmin,xmax,ymin,ymax,zmin,zmax
    p = 4
    def u32():
        nonlocal p; v = struct.unpack_from('<I', raw, p)[0]; p += 4; return v
    u32()                                   # someCount
    p += 8 * 4 + 8 * 4 + 6 * 4              # d1[8], d2[8], bbox[6]
    cB = u32()
    B = [(u32(), u32()) for _ in range(cB)]
    cC = u32()
    C = [(u32(), u32()) for _ in range(cC)]
    if cC != len(w32.plants):
        print(f"[W32] plant_geom cluster count {cC} != plant_vdb {len(w32.plants)}; "
              f"skipping editable plant meshes")
        return

    plants_col = bpy.data.collections.new("PlantBillboards")
    root_col.children.link(plants_col)
    root_col["fo2_plant_gbbox"] = [float(v) for v in gbbox]
    xmin, xmax, ymin, ymax, zmin, zmax = gbbox
    # per-type display sizes from the header (min,max) tables -> midpoints
    d1 = struct.unpack_from('<8f', raw, 8)
    d2 = struct.unpack_from('<8f', raw, 40)
    plants_mat = _get_plants_material(tex_dir or "", options or {})
    total = 0
    for ti, (cnt, start) in enumerate(C):
        pl = w32.plants[ti]
        verts = []
        faces = []
        uvs = []
        ab = []
        for j in range(cnt):
            a, b = B[start + j]
            x = xmin + ((a >> 1) & 0x7FF) / 2047.0 * (xmax - xmin)
            z = zmin + ((a >> 13) & 0x7FF) / 2047.0 * (zmax - zmin)
            y = ymin + (b & 0xFFFF) / 65535.0 * (ymax - ymin)
            variant = (a >> 24) & 0xF
            t = variant >> 2
            # per-instance size: SQUARE quad (matching the square atlas
            # cells - plants render at their natural texture aspect), side
            # lerped within the per-type range from the offset-40 header
            # table (1.2-1.4m on forest; derby halves it) using the
            # instance's preserved random byte b[24:32] (deterministic -
            # same plant always gets the same size). The offset-8 table
            # (0.40/0.45) is NOT display geometry (role not yet identified;
            # sway amplitude or cull radius are the likely candidates).
            f = ((b >> 24) & 0xFF) / 255.0
            s_ = d2[2*t] + (d2[2*t+1] - d2[2*t]) * f
            h = s_
            hw = s_ * 0.5
            # per-instance yaw: b[16:24] is the rotation byte (the remaining
            # uniform-random field of the record) - plants are STATIC quads
            # with a baked heading, not camera-facing billboards, which is
            # why the game shows them at varied/diagonal orientations.
            # yaw = byte/256 * 2*pi around the vertical (FO2 Y) axis.
            ang = ((b >> 16) & 0xFF) / 256.0 * 6.283185307179586
            rx = math.cos(ang) * hw
            rz = math.sin(ang) * hw
            # bottom-anchored upright quad, rotated about the anchor
            # (FO2 space -> Blender). The bottom-edge midpoint stays exactly
            # (x,z), so the exporter's anchor reconstruction is unaffected.
            verts.append(fo2_to_blender_pos(x - rx, y,     z - rz))
            verts.append(fo2_to_blender_pos(x + rx, y,     z + rz))
            verts.append(fo2_to_blender_pos(x + rx, y + h, z + rz))
            verts.append(fo2_to_blender_pos(x - rx, y + h, z - rz))
            bidx = j * 4
            faces.append((bidx, bidx + 1, bidx + 2))
            faces.append((bidx, bidx + 2, bidx + 3))
            # 4x4 atlas cell: col = variant%4 (left to right), row = variant//4
            # counted from the image TOP (grid-index reading order, unlike the
            # tree-sprite atlas rects whose v0/v1 are coordinates from the
            # bottom). Cross-track evidence: derby arenas use only variants
            # 12-15 and their weeds are the dry brown sprites in the atlas's
            # bottom row; forest's dominant 8-11 (60% of instances) is the
            # dense dark-green forest-floor row.
            u0 = (variant & 3) * 0.25
            v0 = (3 - (variant >> 2)) * 0.25    # Blender V=0 at image bottom
            u1, v1 = u0 + 0.25, v0 + 0.25
            uvs.extend([(u0, v0), (u1, v0), (u1, v1), (u0, v1)])
            ab.append(a); ab.append(b)
        mesh = bpy.data.meshes.new(f"PlantType{ti}")
        mesh.from_pydata(verts, [], faces)
        if plants_mat:
            mesh.materials.append(plants_mat)
        apply_uvs(mesh, uvs, "UVMap")
        mesh["fo2_plant_type_index"] = ti
        mesh["fo2_plant_surface_id"] = int(pl.surface_id)
        mesh["fo2_plant_id"] = int(pl.plant_id)
        # marker: 4 verts per instance, bottom edge = instance anchor
        mesh["fo2_plant_quads"] = True
        # cull box (plant_vdb) kept for reference/culling; NOT the position ref
        mesh["fo2_plant_center"] = [float(v) for v in pl.pos]
        mesh["fo2_plant_extent"] = [float(v) for v in pl.extent]
        # original (a,b) per instance so export preserves variant/rotation and
        # reproduces unmoved plants byte-for-byte
        mesh["fo2_plant_ab_hex"] = struct.pack(f'<{len(ab)}I', *ab).hex()
        mesh.update()
        obj = bpy.data.objects.new(f"PlantType{ti}", mesh)
        plants_col.objects.link(obj)
        total += cnt
    print(f"[W32] Created {len(C)} plant-cluster meshes ({total} textured "
          f"billboard quads) - move a quad's 4 vertices to relocate that plant")


def import_w32(context, filepath, options):
    print(f"[W32] Importing: {filepath}")
    tex_dir = os.path.dirname(filepath)
    w32 = parse_w32(filepath, options)

    # -- Create materials --
    bl_materials = []
    tex_loaded = 0
    tex_missing = 0
    for i, mat in enumerate(w32.materials):
        bl_mat = create_blender_material(mat, tex_dir, i, options.get('texture_dir',''), options.get('convert_textures_to_tga', False))
        bl_materials.append(bl_mat)
        if bl_mat:
            # Preserve vanilla material index; export uses this to place each
            # Blender material back at its original slot so tree_mesh's
            # material_id (which we round-trip verbatim from the .w32) still
            # points at the correct material - even for materials like
            # 'alpha_treelod' / 'bushlod_city' that have no mesh geometry but
            # are referenced by trees for LOD billboard rendering.
            bl_mat["fo2_material_index"] = i
            # Fake user prevents Blender from purging materials that no mesh
            # references. Without this, alpha_treelod / bushlod_city vanish on
            # save/reload.
            try: bl_mat.use_fake_user = True
            except Exception: pass
        # Count texture loading results
        if bl_mat and bl_mat.use_nodes:
            for node in bl_mat.node_tree.nodes:
                if node.type == 'TEX_IMAGE':
                    if node.image:
                        tex_loaded += 1
                    elif node.label:
                        tex_missing += 1
    if tex_loaded or tex_missing:
        print(f"[W32] Textures: {tex_loaded} loaded, {tex_missing} missing")

    # -- Root collection --
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    root_col = bpy.data.collections.new(base_name)
    context.scene.collection.children.link(root_col)

    # Store the full ORIGINAL material list on the root collection so export
    # can recreate materials that Blender dropped (materials referenced by
    # tree_mesh.material_id but not attached to any mesh, most notably
    # alpha_treelod / bushlod_city / alpha_bushlod - if these vanish, every
    # tree_mesh referring to them gets an out-of-range material_id and the
    # game crashes on load).
    #
    # Store as a compact binary block: for each material we pack
    #   u32 shader_id, u32 alpha, u32 use_colormap,
    #   u32 v92, u32 v74, u32 num_textures,
    #   u32 name_len, <name bytes>,
    #   for tex in 3: u32 tex_len, <tex bytes>
    import struct as _s
    mat_blob = bytearray()
    for m in w32.materials:
        name_b = (m.name or "").encode('utf-8', errors='replace')
        mat_blob.extend(_s.pack('<7I',
            m.shader_id, m.alpha, m.use_colormap,
            m.v92, m.v74, m.num_textures,
            len(name_b)))
        mat_blob.extend(name_b)
        for ti in range(3):
            tn = (m.texture_names[ti] or "").encode('utf-8', errors='replace')
            mat_blob.extend(_s.pack('<I', len(tn)))
            mat_blob.extend(tn)
    # Stored as a HEX STRING: Blender string properties round-trip verbatim,
    # whereas int-array custom properties proved fragile (a corrupt array
    # crashed the exporter's decoder).
    root_col["fo2_all_materials_raw_hex"] = bytes(mat_blob).hex()
    root_col["fo2_all_materials_count"] = len(w32.materials)

    # Every vanilla track has its OWN track-collision matrix (Y-axis rotation
    # at a track-specific angle - city1b ~31 degrees, desert ~54, nascar ~87).
    # Hardcoding from-scratch constant here is wrong for
    # imported tracks and misaligns collision by ~25 degrees on city1b - the
    # game crashes because the collision volume no longer matches visible
    # geometry. Round-trip the vanilla value verbatim.
    root_col["fo2_collision_matrix_raw_hex"] = _s.pack(
        '<16f', *w32.collision_offset_matrix).hex()

    # Preserve the FULL vertex colors LUT so bank-2 / bank-5 references from
    # tree_colors, tree_lods.values[1], vegetation entries, and static-geometry
    # tint slots still resolve on export. Without this, exporting into a
    # folder that doesn't already contain the vanilla vertexcolors_w2.w32
    # leaves the tail region (indices ~len(bank_0_colors)..end) missing, and
    # every bank-2/5 reference becomes out-of-range - the game crashes on load.
    if w32.vertex_colors:
        lut_bytes = _s.pack(f'<{len(w32.vertex_colors)}I', *w32.vertex_colors)
        root_col["fo2_vertex_colors_full_raw_hex"] = lut_bytes.hex()

    # ══════════════════════════════════════════════════════════════
    # STATIC BATCH GEOMETRY
    # ══════════════════════════════════════════════════════════════
    if options.get('import_static', True):
        static_col = bpy.data.collections.new("StaticBatch")
        root_col.children.link(static_col)
        created = 0
        for bi, batch in enumerate(w32.static_batches):
            sid = batch.bvh_id1
            if sid >= len(w32.surfaces):
                continue
            surface = w32.surfaces[sid]
            if surface.poly_mode == 0:
                continue  # vegetation - handled separately
            obj = create_mesh_from_surface(f"Surface{sid}", w32, surface, bl_materials, sid)
            if obj:
                static_col.objects.link(obj)
                obj["fo2_batch_index"] = bi
                created += 1
        print(f"[W32] Created {created} static batch meshes")

    # ══════════════════════════════════════════════════════════════
    # TREE MESHES (trunk, branch, leaf/vegetation)
    # ══════════════════════════════════════════════════════════════
    if options.get('import_trees', True):
        tree_col = bpy.data.collections.new("TreeMesh")
        root_col.children.link(tree_col)
        created_geo = 0
        created_veg = 0

        # Preserve tree_colors and tree_lods VERBATIM on the root collection.
        # tree_colors are (bank<<24)|LUT-index tuples that reference the tail
        # region of vertexcolors_w2.w32; tree_lods carry positions/scales plus
        # two opaque packed dwords per LOD (vals[0]/vals[1] - not yet decoded).
        # Neither can be synthesized in Blender, so we round-trip the raw
        # bytes on the root collection and reuse them verbatim on export.
        import struct as _s
        tc_bytes = _s.pack(f'<{len(w32.tree_colors)}I', *w32.tree_colors)
        root_col["fo2_tree_colors_raw_hex"] = tc_bytes.hex()
        lod_parts = []
        for lod in w32.tree_lods:
            lod_parts.append(_s.pack('<3f2f2I', lod.pos[0], lod.pos[1], lod.pos[2],
                                     lod.scale[0], lod.scale[1],
                                     lod.values[0], lod.values[1]))
        tl_bytes = b''.join(lod_parts)
        root_col["fo2_tree_lods_raw_hex"] = tl_bytes.hex()

        # Cache already-created vegetation mesh data to share across trees
        veg_mesh_cache = {}  # surface_id -> mesh_data

        for ti, tm in enumerate(w32.tree_meshes):
            tree_node_col = bpy.data.collections.new(f"TreeMesh{ti}")
            tree_col.children.link(tree_node_col)

            # Tree surfaces are already in world space (matrix is NOT applied to geometry).
            # Store the matrix as a metadata empty only.
            tree_empty = bpy.data.objects.new(f"TreeMesh{ti}", None)
            tree_empty.matrix_world = fo2_matrix_to_blender(tm.matrix)
            tree_empty.empty_display_type = 'ARROWS'
            tree_empty.empty_display_size = 0.5
            tree_empty["fo2_is_bush"] = tm.is_bush
            tree_empty["fo2_color_id"] = tm.color_id
            tree_empty["fo2_lod_id"] = tm.lod_id
            tree_empty["fo2_material_id"] = tm.material_id
            # Extra fields (always 0/1 in vanilla but round-trip them anyway)
            tree_empty["fo2_unk2"] = tm.unk2
            tree_empty["fo2_scale"] = list(tm.scale)
            # BVH slot references. In vanilla, tm.bvh_id1 often points at a
            # DIFFERENT tree's branch/proxy surface (semantics unclear -
            # possibly an LOD/culling proxy). Round-trip so export can match
            # vanilla exactly; if the value isn't preserved, our export
            # regenerates it from this tree's own branch/trunk/leaf which is
            # semantically different from vanilla and may cause the game to
            # look up wrong bounding data.
            tree_empty["fo2_bvh_id1"] = tm.bvh_id1
            tree_empty["fo2_bvh_id2"] = tm.bvh_id2
            # Original tree_mesh index; export uses this to preserve order.
            tree_empty["fo2_tree_index"] = ti
            tree_node_col.objects.link(tree_empty)

            # Trunk & Branch (world-space geometry, NOT parented to matrix)
            for label, sid in [("trunk", tm.trunk_surface_id),
                               ("branch", tm.branch_surface_id)]:
                if sid < 0 or sid >= len(w32.surfaces):
                    continue
                surface = w32.surfaces[sid]
                obj = create_mesh_from_surface(f"Surface{sid}_{label}", w32, surface, bl_materials, sid)
                if obj:
                    tree_node_col.objects.link(obj)
                    # Do NOT parent to tree_empty - vertices are in world space
                    created_geo += 1

            # Leaf (vegetation billboard quads, also world-space)
            if options.get('import_vegetation', True):
                sid = tm.leaf_surface_id
                if 0 <= sid < len(w32.surfaces):
                    surface = w32.surfaces[sid]
                    if surface.poly_mode == 0:
                        # Reuse mesh data if this vegetation surface was already built
                        if sid in veg_mesh_cache:
                            obj = bpy.data.objects.new(f"Surface{sid}_leaf", veg_mesh_cache[sid])
                        else:
                            obj = create_vegetation_mesh(
                                f"Surface{sid}_leaf", w32, surface, bl_materials, sid)
                            if obj:
                                veg_mesh_cache[sid] = obj.data
                        if obj:
                            tree_node_col.objects.link(obj)
                            # Vegetation IS in local space (shared across trees),
                            # so it DOES need the tree matrix
                            obj.parent = tree_empty
                            created_veg += 1

        print(f"[W32] Created {created_geo} tree surfaces + {created_veg} vegetation billboards "
              f"({len(veg_mesh_cache)} unique veg meshes)")

    # Attach preserved scatter-plant companion files (see plant-load section),
    # then build editable per-type plant meshes (one vertex per instance at its
    # decoded world position; move a vertex to relocate that plant).
    if getattr(w32, '_plant_geom_raw', b''):
        root_col["fo2_plant_geom_raw_hex"] = w32._plant_geom_raw.hex()
    if getattr(w32, '_plantcolors_raw', b''):
        root_col["fo2_plantcolors_raw_hex"] = w32._plantcolors_raw.hex()
    if getattr(w32, '_plant_vdb_raw', b''):
        root_col["fo2_plant_vdb_raw_hex"] = w32._plant_vdb_raw.hex()
    if (options.get('import_plants', False) and getattr(w32, '_plant_geom_raw', b'')
            and w32.plants):
        _create_plant_meshes(root_col, w32, tex_dir, options)

    if options.get('import_atmosphere', False):
        try:
            _import_atmosphere(root_col, filepath, options)
        except Exception as e:
            print(f"[W32] atmosphere import failed: {e}")

    # ══════════════════════════════════════════════════════════════
    # COMPACT MESHES (dynamic breakable props) - build geometry library
    # ══════════════════════════════════════════════════════════════
    # Cache: model_index -> list of mesh_data objects
    model_mesh_cache = {}
    # Cache: compact_mesh_name -> model_index (first/LOD0)
    prop_model_lookup = {}

    if options.get('import_props', True):
        props_col = bpy.data.collections.new("CompactMesh")
        root_col.children.link(props_col)
        created = 0

        for ci, cm in enumerate(w32.compact_meshes):
            # Prop placement node. Child meshes are parented to it (not put in
            # a per-prop sub-collection - those held no data and cluttered the
            # outliner). Export re-groups instances by this parent empty.
            prop_empty = bpy.data.objects.new(cm.name1, None)
            prop_empty.matrix_world = fo2_matrix_to_blender(cm.matrix)
            prop_empty.empty_display_type = 'PLAIN_AXES'
            prop_empty.empty_display_size = 0.3
            prop_empty["fo2_type"] = cm.name2
            prop_empty["fo2_flags"] = f"0x{cm.flags:X}"
            prop_empty["fo2_group"] = cm.group
            prop_empty["fo2_is_prop_empty"] = True
            props_col.objects.link(prop_empty)

            if cm.models:
                model_idx = cm.models[0]  # LOD0
                prop_model_lookup[cm.name1] = model_idx

                if model_idx not in model_mesh_cache:
                    # Build meshes for this model
                    mesh_list = []
                    if model_idx < len(w32.models):
                        model = w32.models[model_idx]
                        for sid in model.surfaces:
                            if sid < 0 or sid >= len(w32.surfaces):
                                continue
                            surface = w32.surfaces[sid]
                            obj = create_mesh_from_surface(
                                f"Surface{sid}", w32, surface, bl_materials, sid)
                            if obj:
                                mesh_list.append(obj.data)
                                props_col.objects.link(obj)
                                obj.parent = prop_empty
                                created += 1
                    model_mesh_cache[model_idx] = mesh_list
                else:
                    # Reuse already-built mesh data
                    for mi, mesh_data in enumerate(model_mesh_cache[model_idx]):
                        obj = bpy.data.objects.new(f"{cm.name1}_m{mi}", mesh_data)
                        props_col.objects.link(obj)
                        obj.parent = prop_empty
                        created += 1

        print(f"[W32] Created {created} compact mesh surfaces "
              f"({len(model_mesh_cache)} unique models)")

    # ══════════════════════════════════════════════════════════════
    # OBJECTS (placement empties + prop instancing)
    # ══════════════════════════════════════════════════════════════
    if options.get('import_objects', True):
        obj_col = bpy.data.collections.new("Objects")
        root_col.children.link(obj_col)

        # Build name lookup: object name -> compact mesh name
        cm_name_lookup = {}
        for cm in w32.compact_meshes:
            cm_name_lookup[cm.name1] = cm.name1
            cm_name_lookup[cm.name2] = cm.name1
            # Strip common prefixes for fuzzy matching
            for prefix in ('dyn_', 'sta_', 'static_', 'Dynamic_', 'Static_'):
                if cm.name1.startswith(prefix):
                    short = cm.name1[len(prefix):]
                    if short not in cm_name_lookup:
                        cm_name_lookup[short] = cm.name1

        instanced = 0
        plain = 0

        for oi, ob in enumerate(w32.objects):
            mat = fo2_matrix_to_blender(ob.matrix)

            # Try to instance prop geometry at object position
            cm_name = None
            if options.get('instance_props', True):
                for candidate in [ob.name1, ob.name2]:
                    if candidate in cm_name_lookup:
                        cm_name = cm_name_lookup[candidate]
                        break
                    # Try stripping dummy_ prefix
                    for prefix in ('dummy_', 'Dummy_', 'DUMMY_'):
                        if candidate.startswith(prefix):
                            stripped = candidate[len(prefix):]
                            if stripped in cm_name_lookup:
                                cm_name = cm_name_lookup[stripped]
                                break
                    if cm_name:
                        break

            if cm_name and cm_name in prop_model_lookup:
                model_idx = prop_model_lookup[cm_name]
                if model_idx in model_mesh_cache and model_mesh_cache[model_idx]:
                    for mi, mesh_data in enumerate(model_mesh_cache[model_idx]):
                        inst = bpy.data.objects.new(f"{ob.name1}_inst{mi}", mesh_data)
                        inst.matrix_world = mat
                        inst["fo2_object_index"] = oi
                        inst["fo2_flags"] = f"0x{ob.flags:X}"
                        inst["fo2_source_prop"] = cm_name
                        obj_col.objects.link(inst)
                    instanced += 1
                    continue

            # Plain empty for unmatched objects
            empty = bpy.data.objects.new(ob.name1, None)
            empty.empty_display_type = 'ARROWS'
            empty.empty_display_size = 1.0
            empty.matrix_world = mat
            empty["fo2_name2"] = ob.name2
            empty["fo2_flags"] = f"0x{ob.flags:X}"
            obj_col.objects.link(empty)
            plain += 1

        print(f"[W32] Objects: {instanced} instanced, {plain} empties")

    # ══════════════════════════════════════════════════════════════
    # TRACK BVH VISUALIZATION
    # ══════════════════════════════════════════════════════════════
    if options.get('import_bvh', False) and (w32.bvh_primitives or w32.bvh_nodes):
        bvh_col = bpy.data.collections.new("TrackBVH")
        root_col.children.link(bvh_col)
        cube_mesh = create_bvh_box_mesh()

        if w32.bvh_primitives:
            prim_col = bpy.data.collections.new("BVH_Primitives")
            bvh_col.children.link(prim_col)
            for pi, prim in enumerate(w32.bvh_primitives):
                obj = bpy.data.objects.new(f"BVHPrim{pi}", cube_mesh)
                px, py, pz = fo2_to_blender_pos(*prim.pos)
                # Radius: swap Y/Z to match position swap
                rx, ry, rz = abs(prim.radius[0]), abs(prim.radius[2]), abs(prim.radius[1])
                obj.location = (px, py, pz)
                obj.scale = (rx if rx > 0 else 0.1,
                             ry if ry > 0 else 0.1,
                             rz if rz > 0 else 0.1)
                obj.display_type = 'WIRE'
                obj["fo2_bvh_id1"] = prim.id1
                obj["fo2_bvh_id2"] = prim.id2
                prim_col.objects.link(obj)

        if w32.bvh_nodes:
            node_col = bpy.data.collections.new("BVH_Nodes")
            bvh_col.children.link(node_col)
            for ni, node in enumerate(w32.bvh_nodes):
                obj = bpy.data.objects.new(f"BVHNode{ni}", cube_mesh)
                px, py, pz = fo2_to_blender_pos(*node.pos)
                rx, ry, rz = abs(node.radius[0]), abs(node.radius[2]), abs(node.radius[1])
                obj.location = (px, py, pz)
                obj.scale = (rx if rx > 0 else 0.1,
                             ry if ry > 0 else 0.1,
                             rz if rz > 0 else 0.1)
                obj.display_type = 'WIRE'
                obj["fo2_bvh_unk1"] = node.unk1
                obj["fo2_bvh_unk2"] = node.unk2
                node_col.objects.link(obj)

        print(f"[W32] Created {len(w32.bvh_primitives)} BVH primitives, "
              f"{len(w32.bvh_nodes)} BVH nodes")

    # ══════════════════════════════════════════════════════════════
    # PLANT VDB (plant cluster positions)
    # ══════════════════════════════════════════════════════════════
    if options.get('import_plants', False) and w32.plants:
        plant_col = bpy.data.collections.new("Plants")
        root_col.children.link(plant_col)

        for pi, plant in enumerate(w32.plants):
            empty = bpy.data.objects.new(f"Plant{pi}", None)
            px, py, pz = fo2_to_blender_pos(*plant.pos)
            empty.location = (px, py, pz)

            ex, ey, ez = plant.extent
            empty.empty_display_type = 'CUBE'
            empty.empty_display_size = 1.0
            # Swap Y/Z to match position swap
            empty.scale = (ex if ex > 0 else 0.5,
                           ez if ez > 0 else 0.5,
                           ey if ey > 0 else 0.5)
            empty["fo2_surface_id"] = plant.surface_id
            empty["fo2_plant_id"] = plant.plant_id
            plant_col.objects.link(empty)

        print(f"[W32] Created {len(w32.plants)} plant cluster empties")

    # ══════════════════════════════════════════════════════════════
    # EFFECTMAP / RESETMAP (4B overlays)
    # ══════════════════════════════════════════════════════════════
    if options.get('import_maps', False):
        for overlay in [w32.effectmap, w32.resetmap]:
            if overlay is None or overlay.data is None:
                continue
            if overlay.bounds is None:
                print(f"[W32] Warning: No .bed bounds for {overlay.name}, skipping")
                continue

            tl_x, tl_z, br_x, br_z = overlay.bounds
            W, H = overlay.width, overlay.height  # 256×128

            # Create Blender image (256×128) with color-coded pixels
            img_name = f"fo2_{overlay.name}"
            img = bpy.data.images.new(img_name, width=W, height=H, alpha=True)

            # Build RGBA pixel array - color code by value
            # Each unique value gets a distinct color for easy identification
            color_lut = {}
            pixels = [0.0] * (W * H * 4)
            for y in range(H):
                for x in range(W):
                    # 4B row 0 = top of map in game, but Blender images have row 0 at bottom
                    src_row = (H - 1 - y)
                    val = overlay.data[src_row * W + x]

                    if val not in color_lut:
                        # Generate a deterministic color from value
                        # Use the nibbles to create hue variation
                        hi = (val >> 4) & 0xF
                        lo = val & 0xF
                        r = ((hi * 37) % 256) / 255.0
                        g = ((lo * 67 + 80) % 256) / 255.0
                        b = ((val * 13 + 40) % 256) / 255.0
                        color_lut[val] = (r, g, b)

                    r, g, b = color_lut[val]
                    idx = (y * W + x) * 4
                    pixels[idx + 0] = r
                    pixels[idx + 1] = g
                    pixels[idx + 2] = b
                    pixels[idx + 3] = 0.7  # semi-transparent

            img.pixels[:] = pixels
            img.pack()

            # Create material for the overlay plane
            mat = bpy.data.materials.new(name=img_name)
            mat.use_nodes = True
            # Enable transparency for overlay
            try:
                mat.surface_render_method = 'BLENDED'
            except:
                pass
            try:
                mat.blend_method = 'BLEND'
            except:
                pass
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            for n in nodes:
                nodes.remove(n)
            output = nodes.new('ShaderNodeOutputMaterial')
            output.location = (400, 0)
            bsdf = nodes.new('ShaderNodeBsdfPrincipled')
            bsdf.location = (0, 0)
            links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
            tex_node = nodes.new('ShaderNodeTexImage')
            tex_node.location = (-400, 0)
            tex_node.image = img
            tex_node.interpolation = 'Closest'
            links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
            links.new(tex_node.outputs['Alpha'], bsdf.inputs['Alpha'])

            # Create plane mesh with correct world-space bounds
            # .bed: TopLeft(X, Z), BottomRight(X, Z) in FO2 game space
            # FO2 (X_right, Y_up, Z_forward) -> Blender (X_right, Y_forward, Z_up)
            # So FO2-X → Blender-X, FO2-Z → Blender-Y
            x0, x1 = tl_x, br_x       # Blender X range
            y0, y1 = br_z, tl_z        # Blender Y range (Z in game)
            z_height = 0.5             # slight elevation above ground

            verts = [
                (x0, y0, z_height),  # bottom-left
                (x1, y0, z_height),  # bottom-right
                (x1, y1, z_height),  # top-right
                (x0, y1, z_height),  # top-left
            ]
            faces = [(0, 1, 2, 3)]
            uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]

            mesh = bpy.data.meshes.new(img_name)
            mesh.from_pydata(verts, [], faces)
            mesh.materials.append(mat)

            # Apply UVs
            uv_layer = mesh.uv_layers.new(name="UVMap")
            for li, loop in enumerate(mesh.loops):
                uv_layer.data[li].uv = uvs[loop.vertex_index]

            mesh.update()

            # Create object in its own collection
            if not any(c.name == "TrackMaps" for c in root_col.children):
                maps_col = bpy.data.collections.new("TrackMaps")
                root_col.children.link(maps_col)
            else:
                maps_col = next(c for c in root_col.children if c.name == "TrackMaps")

            obj = bpy.data.objects.new(overlay.name, mesh)
            maps_col.objects.link(obj)

            # Store metadata
            obj["fo2_map_type"] = overlay.name
            obj["fo2_bounds_tl"] = f"{tl_x}, {tl_z}"
            obj["fo2_bounds_br"] = f"{br_x}, {br_z}"

            # Log value legend
            val_counts = {}
            for b in overlay.data:
                val_counts[b] = val_counts.get(b, 0) + 1
            legend = ", ".join(f"0x{v:02X}:{c}" for v, c in sorted(val_counts.items(), key=lambda x: -x[1]))
            print(f"[W32] Created {overlay.name} overlay ({W}×{H}): {legend}")

    # -- Summary --
    print(f"[W32] Import complete: {base_name}")
    print(f"[W32]   Version: 0x{w32.version:X}")
    print(f"[W32]   Materials: {len(w32.materials)}, Surfaces: {len(w32.surfaces)}")
    print(f"[W32]   Static batches: {len(w32.static_batches)}, Trees: {len(w32.tree_meshes)}")
    print(f"[W32]   Models: {len(w32.models)}, Objects: {len(w32.objects)}")
    print(f"[W32]   Compact meshes: {len(w32.compact_meshes)}")
    if w32.bvh_primitives:
        print(f"[W32]   BVH: {len(w32.bvh_primitives)} primitives, {len(w32.bvh_nodes)} nodes")
    if w32.plants:
        print(f"[W32]   Plants: {len(w32.plants)} clusters")
    if w32.effectmap:
        print(f"[W32]   Effectmap: loaded")
    if w32.resetmap:
        print(f"[W32]   Resetmap: loaded")

    return {'FINISHED'}


# ═════════════════════════════════════════════════════════════════════════════
# Operator & Registration
# ═════════════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════════════
# Atmosphere import (data/atmosphere.ini): sky dome, cloud layers, skybox, sun
# ═════════════════════════════════════════════════════════════════════════════
def _parse_atmosphere_ini(path):
    """Parse the Lua-style atmosphere.ini into a dict. Handles:
        Key = 123 / 1.5            (numbers)
        Key = "some/path.tga"      (strings)
        Key = {a, b, c}            (tuples of numbers)
        SkyGradient = { [1] = {r,g,b}, [2] = {...}, ... }   (color table)
    Comment lines start with '--'."""
    vals = {}
    try:
        text = open(path, 'r', errors='replace').read()
    except OSError:
        return vals
    import re as _re
    # multi-line gradient tables: Name = { [i] = {r,g,b}, ... }
    for mt in _re.finditer(r'(\w+)\s*=\s*\{\s*(\[\s*\d+\s*\].*?)\}\s*(?:\r?\n\s*\r?\n|\Z)',
                           text, _re.S):
        rows = _re.findall(r'\{\s*([^{}]*?)\s*\}', mt.group(2))
        table = []
        for row in rows:
            try:
                table.append(tuple(float(x) for x in row.split(',') if x.strip()))
            except ValueError:
                pass
        if table:
            vals[mt.group(1)] = table
    for line in text.splitlines():
        line = line.split('--')[0].strip()
        m = _re.match(r'(\w+)\s*=\s*(.+)', line)
        if not m:
            continue
        key, raw = m.group(1), m.group(2).strip()
        if key in vals:            # gradient tables already captured
            continue
        if raw.startswith('"'):
            vals[key] = raw.strip('"')
        elif raw.startswith('{'):
            try:
                vals[key] = tuple(float(x) for x in raw.strip('{}').split(',') if x.strip())
            except ValueError:
                pass
        else:
            try:
                vals[key] = float(raw) if '.' in raw else int(raw)
            except ValueError:
                vals[key] = raw
    return vals


def _atmo_material(name, img, use_alpha):
    """Simple display-only material for atmosphere elements. Tagged so the
    exporter never adds it to the track's material table."""
    bl_mat = bpy.data.materials.new(name=name)
    bl_mat["fo2_display_only"] = True
    _try_set(bl_mat, 'use_nodes', True)
    _try_set(bl_mat, 'use_backface_culling', False)
    if use_alpha:
        for enum_val in ('BLEND', 'CLIP', 'ALPHA_CLIP'):
            try:
                _try_set(bl_mat, 'blend_method', enum_val)
                break
            except Exception:
                pass
    try:
        nodes = bl_mat.node_tree.nodes
        links = bl_mat.node_tree.links
        for n in list(nodes):
            nodes.remove(n)
        output = nodes.new('ShaderNodeOutputMaterial')
        output.location = (400, 0)
        # Emission-style: sky/cloud art is prelit, shouldn't receive shading
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.location = (100, 0)
        try:
            bsdf.inputs['Emission Strength'].default_value = 1.0
            bsdf.inputs['Roughness'].default_value = 1.0
        except Exception:
            pass
        links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        if img is not None:
            tex = nodes.new('ShaderNodeTexImage')
            tex.location = (-250, 0)
            tex.image = img
            links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
            try:
                links.new(tex.outputs['Color'], bsdf.inputs['Emission Color'])
            except Exception:
                pass
            if use_alpha:
                links.new(tex.outputs['Alpha'], bsdf.inputs['Alpha'])
    except Exception as e:
        print(f"[W32] atmosphere material '{name}' node setup skipped: {e}")
    return bl_mat


def _import_atmosphere(root_col, w32_path, options):
    """Import <track>/data/atmosphere.ini as an 'Atmosphere' collection:
    sun light (direction/intensity from the ini), procedural sky dome,
    curved cloud layers (top+bottom textures), and the theme skybox cube
    (global/skybox/<theme>_day faces - the specific set is chosen by the
    game's track table, which is not in the track folder, so the theme's
    _day set is the default heuristic). Everything is display-only: nothing
    in this collection exports."""
    geom_dir = os.path.dirname(w32_path)
    track_root = os.path.dirname(geom_dir)
    ini = None
    for sub in ('data', 'Data'):
        cand = os.path.join(track_root, sub, 'atmosphere.ini')
        if os.path.isfile(cand):
            ini = cand
            break
    if not ini:
        print("[W32] atmosphere.ini not found - skipping atmosphere")
        return
    ai = _parse_atmosphere_ini(ini)
    convert_dds = bool(options.get('convert_textures_to_tga'))
    search_dirs = [os.path.join(track_root, 'data'), geom_dir]
    user_dir = options.get('texture_dir', '')
    if user_dir:
        search_dirs = [user_dir, os.path.join(user_dir, 'textures')] + search_dirs
    search_dirs += _vanilla_texture_dirs(geom_dir)

    col = bpy.data.collections.new("Atmosphere")
    root_col.children.link(col)
    for k, v in ai.items():
        try:
            col[f"fo2_atmo_{k}"] = list(v) if isinstance(v, (tuple, list)) else v
        except Exception:
            pass

    # ── Sun light ─────────────────────────────────────────────────────────
    sun_dir = ai.get('Sun_Direction', (0.0, 1.0, 0.0))
    sun_int = float(ai.get('Sun_Intensity', 1.0))
    try:
        light = bpy.data.lights.new(name="Sun", type='SUN')
        _try_set(light, 'energy', sun_int)
        sun_obj = bpy.data.objects.new("Sun", light)
        col.objects.link(sun_obj)
        # FO2 (x, y-up, z) -> Blender (x, z, y-up); Sun_Direction points
        # TOWARD the sun. A Blender sun shines along its local -Z, so aim
        # local +Z at the sun direction: euler XYZ =
        #   (-acos(vz), 0, atan2(vy, vx) - pi/2)
        vx, vz, vy_up = float(sun_dir[0]), float(sun_dir[2]), float(sun_dir[1])
        v = (vx, vz, vy_up)   # blender (x, y, z)
        l = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) or 1.0
        v = (v[0]/l, v[1]/l, v[2]/l)
        rx = -math.acos(max(-1.0, min(1.0, v[2])))
        rz = math.atan2(v[1], v[0]) - math.pi/2 if (abs(v[0]) > 1e-9 or abs(v[1]) > 1e-9) else 0.0
        _try_set(sun_obj, 'rotation_euler', (rx, 0.0, rz))
        sun_obj.location = (v[0]*100.0, v[1]*100.0, v[2]*100.0)
        sun_obj["fo2_sun_direction"] = [float(x) for x in sun_dir]
        sun_obj["fo2_sun_intensity"] = sun_int
    except Exception as e:
        print(f"[W32] sun light skipped: {e}")

    # ── Sky dome ──────────────────────────────────────────────────────────
    sides = int(ai.get('SkyDome_Sides', 64))
    slices = int(ai.get('SkyDome_Slices', 16))
    radius = float(ai.get('SkyDome_Radius', 30000))
    sides_v = max(8, min(sides, 128))
    slices_v = max(4, min(slices, 64))
    verts = []
    faces = []
    for s in range(slices_v):            # rings: horizon -> just below apex
        phi = (s / slices_v) * math.pi / 2
        ry = radius * math.sin(phi)
        rr = radius * math.cos(phi)
        for i in range(sides_v):
            th = 2 * math.pi * i / sides_v
            verts.append(fo2_to_blender_pos(rr * math.cos(th), ry, rr * math.sin(th)))
    apex = len(verts)
    verts.append(fo2_to_blender_pos(0.0, radius, 0.0))
    for s in range(slices_v - 1):
        for i in range(sides_v):
            a = s * sides_v + i
            b_ = s * sides_v + (i + 1) % sides_v
            faces.append((a, b_, b_ + sides_v, a + sides_v))
    top = (slices_v - 1) * sides_v
    for i in range(sides_v):
        faces.append((top + i, top + (i + 1) % sides_v, apex))
    dome = bpy.data.meshes.new("SkyDome")
    dome.from_pydata(verts, [], faces)
    # SkyGradient (when present): vertical color ramp, row 1 = horizon
    grad = ai.get('SkyGradient')
    if grad:
        cols = []
        for s in range(slices_v):
            gi = min(len(grad) - 1, int(s / max(1, slices_v - 1) * (len(grad) - 1)))
            g = grad[gi]
            cols.extend([(g[0], g[1], g[2], 1.0)] * sides_v)
        cols.append((*grad[-1][:3], 1.0))
        apply_vertex_colors(dome, cols)
    dome.materials.append(_atmo_material("atmo_skydome", None, False))
    dome.update()
    dome_obj = bpy.data.objects.new("SkyDome", dome)
    col.objects.link(dome_obj)

    # ── Cloud layers (bottom + top) ──────────────────────────────────────
    grid = max(2, min(int(ai.get('CloudLayer_VtxGridSize', 32)), 128))
    alt = float(ai.get('CloudLayer_Altitude', 500))
    size = float(ai.get('CloudLayer_Size', 4000))
    tiling = float(ai.get('CloudLayer_Tiling', 1))
    curv = float(ai.get('CloudLayer_Curvature', 0))
    volume = float(ai.get('CloudLayer_Volume', 0))

    def _cloud_tex(key, fallback):
        name = os.path.basename(str(ai.get(key, ''))) or fallback
        img = _find_texture_image(name, search_dirs, convert_dds)
        if img is None and name.lower() != fallback.lower():
            # retail path names (atmos_cloud*.tga) are often absent from
            # extracted data; the global/atmosphere sets carry the art
            img = _find_texture_image(fallback, search_dirs, convert_dds)
        return img

    tex_top = _cloud_tex('CloudLayerTopTexture', 'default_clouds_top.tga')
    tex_bot = _cloud_tex('CloudLayerBottomTexture', 'default_clouds_bottom.tga')

    def _cloud_mesh(name, y_base, img):
        vs = []
        uvs = []
        fs = []
        half = size / 2
        for gy in range(grid):
            for gx in range(grid):
                fx = gx / (grid - 1)
                fz = gy / (grid - 1)
                x = -half + size * fx
                z = -half + size * fz
                r2 = (x*x + z*z) / (half*half)
                y = y_base - curv * min(1.0, r2)   # bowl: edges drop by curvature
                vs.append(fo2_to_blender_pos(x, y, z))
        for gy in range(grid - 1):
            for gx in range(grid - 1):
                a = gy * grid + gx
                fs.append((a, a + 1, a + grid + 1, a + grid))
        m = bpy.data.meshes.new(name)
        m.from_pydata(vs, [], fs)
        # tiled UVs (per-loop)
        luv = []
        for (a, b_, c, d) in fs:
            for vi in (a, b_, c, d):
                gx = vi % grid
                gy = vi // grid
                luv.append((gx / (grid - 1) * tiling, gy / (grid - 1) * tiling))
        try:
            uv_layer = m.uv_layers.new(name="UVMap")
            for li, uv in enumerate(luv):
                uv_layer.data[li].uv = uv
        except Exception:
            pass
        m.materials.append(_atmo_material(name + "_mat", img, True))
        m.update()
        o = bpy.data.objects.new(name, m)
        col.objects.link(o)
        return o

    _cloud_mesh("CloudLayerBottom", alt, tex_bot)
    if volume > 0:
        _cloud_mesh("CloudLayerTop", alt + volume, tex_top)

    # ── Skybox cube (theme heuristic: <theme>_day) ───────────────────────
    theme = ""
    parts = geom_dir.replace('\\', '/').split('/')
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == 'tracks' and i + 1 < len(parts):
            theme = parts[i + 1].lower()
            break
    skyset = f"{theme}_day" if theme else ""
    face_imgs = {}
    if skyset:
        for face in ('f', 'b', 'l', 'r', 'u'):
            face_imgs[face] = _find_texture_image(f"{skyset}_{face}.tga", search_dirs, convert_dds)
    if any(face_imgs.values()):
        half = radius * 0.6
        # FO2 forward +Z = Blender +Y. Inward-facing quads.
        Fz = fo2_to_blender_pos
        FACES = {
            'f': [Fz(-half, -half,  half), Fz( half, -half,  half), Fz( half,  half,  half), Fz(-half,  half,  half)],
            'b': [Fz( half, -half, -half), Fz(-half, -half, -half), Fz(-half,  half, -half), Fz( half,  half, -half)],
            'l': [Fz(-half, -half, -half), Fz(-half, -half,  half), Fz(-half,  half,  half), Fz(-half,  half, -half)],
            'r': [Fz( half, -half,  half), Fz( half, -half, -half), Fz( half,  half, -half), Fz( half,  half,  half)],
            'u': [Fz(-half,  half,  half), Fz( half,  half,  half), Fz( half,  half, -half), Fz(-half,  half, -half)],
        }
        for face, corners in FACES.items():
            img = face_imgs.get(face)
            if img is None:
                continue
            m = bpy.data.meshes.new(f"Skybox_{face}")
            m.from_pydata(corners, [], [(0, 1, 2, 3)])
            try:
                uv_layer = m.uv_layers.new(name="UVMap")
                for li, uv in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):
                    uv_layer.data[li].uv = uv
            except Exception:
                pass
            m.materials.append(_atmo_material(f"atmo_skybox_{face}", img, False))
            m.update()
            o = bpy.data.objects.new(f"Skybox_{face}", m)
            col.objects.link(o)
        print(f"[W32] Atmosphere: skybox set '{skyset}' "
              f"({sum(1 for v in face_imgs.values() if v)}/5 faces)")
    else:
        print(f"[W32] Atmosphere: no skybox set found for theme '{theme}'")
    print(f"[W32] Atmosphere imported from {os.path.basename(ini)} "
          f"(sun, sky dome, {'2' if volume > 0 else '1'} cloud layers)")


class IMPORT_OT_fo2_w32(bpy.types.Operator, ImportHelper):
    """Import FlatOut 2 W32 Track Geometry"""
    bl_idname = "import_scene.fo2_w32"
    bl_label = "Import FlatOut 2 Track (.w32)"
    bl_options = {'REGISTER', 'UNDO', 'PRESET'}

    filename_ext = ".w32"
    filter_glob: StringProperty(default="*.w32", options={'HIDDEN'})

    import_static: BoolProperty(
        name="Static Geometry",
        description="Import static batch meshes (terrain, roads, buildings)",
        default=True,
    )
    import_trees: BoolProperty(
        name="Tree Meshes",
        description="Import tree/vegetation meshes (trunk + branch surfaces)",
        default=True,
    )
    import_vegetation: BoolProperty(
        name="Vegetation Billboards",
        description="Import leaf/grass billboard quads from vegetation buffers",
        default=True,
    )
    import_props: BoolProperty(
        name="Dynamic Props",
        description="Import compact meshes (breakable props, dynamic objects)",
        default=True,
    )
    import_objects: BoolProperty(
        name="Object Instances",
        description="Import the object-placement section and instance the "
                    "matching prop geometry at each placement (dummies + "
                    "instancing are one step - dummies alone do nothing)",
        default=True,
    )
    import_bvh: BoolProperty(
        name="Track BVH",
        description="Import track_bvh.gen as wireframe bounding volumes",
        default=True,
    )
    import_plants: BoolProperty(
        name="Plant Clusters",
        description="Import plant_geom scatter as editable billboard meshes",
        default=True,
    )
    import_maps: BoolProperty(
        name="Track Maps (effectmap/resetmap)",
        description="Import effectmap.4b and resetmap.4b as color-coded overlays",
        default=False,
    )
    import_atmosphere: BoolProperty(
        name="Atmosphere (sky, clouds, sun)",
        description="Import data/atmosphere.ini: sun light, sky dome, cloud "
                    "layers and the theme skybox (display-only, never exported)",
        default=True,
    )
    convert_textures_to_tga: BoolProperty(
        name="Convert Textures to TGA",
        description="Load every material's textures; when a texture is only "
                    "found as DDS, convert it to an editable TGA (beside the "
                    ".w32 or in the textures folder) so Blender can display it",
        default=True,
    )
    texture_dir: StringProperty(
        name="Textures Folder",
        description="Extra folder to search for shared track textures "
                    "(diffuse maps usually live outside the track folder). "
                    "Leave blank to search only next to the .w32",
        default="",
        subtype='DIR_PATH',
    )

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        box = layout.box()
        box.label(text="Geometry", icon='MESH_DATA')
        box.prop(self, "import_static")
        box.prop(self, "import_trees")
        box.prop(self, "import_vegetation")

        box = layout.box()
        box.label(text="Props & Objects", icon='OBJECT_DATA')
        box.prop(self, "import_props")
        box.prop(self, "import_objects")

        box = layout.box()
        box.label(text="Auxiliary Data", icon='OUTLINER_DATA_EMPTY')
        box.prop(self, "import_bvh")
        box.prop(self, "import_plants")
        box.prop(self, "import_maps")
        box.prop(self, "import_atmosphere")

        box = layout.box()
        box.label(text="Textures", icon='TEXTURE')
        box.prop(self, "convert_textures_to_tga")
        box.prop(self, "texture_dir")

    def execute(self, context):
        options = {
            'import_static':     self.import_static,
            'import_trees':      self.import_trees,
            'import_vegetation': self.import_vegetation,
            'import_props':      self.import_props,
            'import_objects':    self.import_objects,
            'instance_props':    self.import_objects,
            'import_bvh':        self.import_bvh,
            'import_plants':     self.import_plants,
            'import_maps':       self.import_maps,
            'import_atmosphere': self.import_atmosphere,
            'convert_textures_to_tga': self.convert_textures_to_tga,
            'texture_dir':       self.texture_dir,
        }
        return import_w32(context, self.filepath, options)


def menu_func_import(self, context):
    self.layout.operator(IMPORT_OT_fo2_w32.bl_idname, text="FlatOut 2 Track (.w32)")


def register():
    bpy.utils.register_class(IMPORT_OT_fo2_w32)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.utils.unregister_class(IMPORT_OT_fo2_w32)


if __name__ == "__main__":
    register()
