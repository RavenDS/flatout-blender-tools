bl_info = {
    "name": "FlatOut BGM Import (Car)",
    "author": "ravenDS",
    "version": (1, 5, 5),
    "blender": (3, 6, 0),
    "location": "File > Import > FlatOut Car BGM (.bgm)",
    "description": "Import FlatOut 1/2/UC BGM car model files",
    "category": "Import-Export",
    "doc_url":     "https://github.com/RavenDS",
    "tracker_url": "https://github.com/RavenDS/flatout-blender-tools/issues",
}

import bpy
import bmesh
import struct
import os
import math
from bpy.props import (
    StringProperty,
    BoolProperty,
    FloatProperty,
    IntProperty,
    EnumProperty,
)
from bpy_extras.io_utils import ImportHelper
from mathutils import Matrix, Vector
from dataclasses import dataclass, field
from typing import Optional
from . import dds2tga as _dds2tga
from . import dds_normal as _dds_normal


# BGM PARSER (standalone, no Blender dependency)

# vertex stream flags
VERTEX_POSITION = 0x002
VERTEX_NORMAL   = 0x010
VERTEX_COLOR    = 0x040
VERTEX_UV       = 0x100
VERTEX_UV2      = 0x200
VERTEX_INT16    = 0x2000

FOUC_VERTEX_FLAGS = 0x2242
FOUC_VERTEX_SCALE = 1.0 / 1024.0  # int16 * this = metres


@dataclass
class BGMMaterial:
    name: str = ""
    nAlpha: int = 0
    v92: int = 0
    nNumTextures: int = 0
    nShaderId: int = 0
    nUseColormap: int = 0
    v74: int = 0
    v108: tuple = (0, 0, 0)
    v109: tuple = (0, 0, 0)
    v98: tuple = (0, 0, 0, 0)
    v99: tuple = (0, 0, 0, 0)
    v100: tuple = (0, 0, 0, 0)
    v101: tuple = (0, 0, 0, 0)
    v102: int = 0
    texture_names: list = field(default_factory=lambda: ["", "", ""])


@dataclass
class VertexBuffer:
    buf_id: int = 0
    is_vegetation: bool = False
    fouc_extra_format: int = 0
    vertex_count: int = 0
    vertex_size: int = 0
    flags: int = 0
    data: bytes = b""


@dataclass
class IndexBuffer:
    buf_id: int = 0
    fouc_extra_format: int = 0
    index_count: int = 0
    data: bytes = b""


@dataclass
class Surface:
    is_vegetation: int = 0
    material_id: int = 0
    vertex_count: int = 0
    flags: int = 0
    poly_count: int = 0
    poly_mode: int = 0
    num_indices_used: int = 0
    center: tuple = (0.0, 0.0, 0.0)
    radius: tuple = (0.0, 0.0, 0.0)
    num_streams_used: int = 0
    stream_id: list = field(default_factory=lambda: [0, 0])
    stream_offset: list = field(default_factory=lambda: [0, 0])
    fouc_vertex_multiplier: list = field(default_factory=lambda: [0.0, 0.0, 0.0, FOUC_VERTEX_SCALE])


@dataclass
class Model:
    nUnk: int = 0
    name: str = ""
    center: tuple = (0.0, 0.0, 0.0)
    radius: tuple = (0.0, 0.0, 0.0)
    fRadius: float = 0.0
    surface_ids: list = field(default_factory=list)


@dataclass
class BGMMesh:
    name1: str = ""
    name2: str = ""
    flags: int = 0
    group: int = -1
    matrix: list = field(default_factory=lambda: [0.0] * 16)
    model_ids: list = field(default_factory=list)


@dataclass
class BGMObject:
    name1: str = ""
    name2: str = ""
    flags: int = 0
    matrix: list = field(default_factory=lambda: [0.0] * 16)


# CRASH.DAT PARSER

@dataclass
class CrashWeight:
    """Per-vertex base and crash positions/normals (FO2 format)."""
    base_pos: tuple = (0.0, 0.0, 0.0)
    crash_pos: tuple = (0.0, 0.0, 0.0)
    base_normal: tuple = (0.0, 0.0, 0.0)
    crash_normal: tuple = (0.0, 0.0, 0.0)


@dataclass
class CrashSurface:
    """One surface within a crash node — mirrors a BGM model surface."""
    vertex_count: int = 0
    vertex_size: int = 0
    vertex_data: bytes = b""      # raw vertex buffer (same format as BGM)
    flags: int = 0                # copied from BGM surface flags
    weights: list = field(default_factory=list)  # list[CrashWeight]


@dataclass
class CrashNode:
    """One crash node, corresponds to a BGM model (name = model_name + '_crash')."""
    name: str = ""
    surfaces: list = field(default_factory=list)  # list[CrashSurface]


def parse_crash_dat(filepath: str, is_fouc: bool = False) -> list:
    """Parse a FO2 or FOUC crash.dat file. Returns list of CrashNode."""
    nodes = []
    try:
        with open(filepath, 'rb') as f:
            data = f.read()
    except (OSError, IOError):
        return nodes

    r = BinaryReader(data)
    node_count = struct.unpack_from('<I', r.read(4), 0)[0]

    for i in range(node_count):
        node = CrashNode()
        node.name = r.read_string()
        num_surfaces = struct.unpack_from('<I', r.read(4), 0)[0]

        for j in range(num_surfaces):
            surf = CrashSurface()
            num_verts = struct.unpack_from('<I', r.read(4), 0)[0]
            surf.vertex_count = num_verts

            if is_fouc:
                # FOUC: no vbuffer, weights are tCrashDataWeightsFOUC (40 bytes each)
                # int16[3] basePos, int16[3] crashPos,
                # uint8[4] baseUnkBump1, uint8[4] crashUnkBump1,
                # uint8[4] baseUnkBump2, uint8[4] crashUnkBump2,
                # uint8[4] baseNormals, uint8[4] crashNormals,
                # uint16[2] baseUV
                surf.vertex_size = 0
                surf.vertex_data = b''
                surf.weights = []
                SCALE = FOUC_VERTEX_SCALE
                for k in range(num_verts):
                    raw = r.read(40)
                    bp = struct.unpack_from('<3h', raw, 0)
                    cp = struct.unpack_from('<3h', raw, 6)
                    bn = struct.unpack_from('<4B', raw, 28)
                    cn = struct.unpack_from('<4B', raw, 32)
                    w = CrashWeight(
                        base_pos=(bp[0]*SCALE, bp[1]*SCALE, bp[2]*SCALE),
                        crash_pos=(cp[0]*SCALE, cp[1]*SCALE, cp[2]*SCALE),
                        # tCrashDataWeightsFOUC normal encoding matches tVertexDataFOUC:
                        # buffer[0]=FO2.z, buffer[1]=FO2.y, buffer[2]=FO2.x
                        # formula: (uint8 / 127.0) - 1.0
                        base_normal=((bn[2]/127.0)-1.0, (bn[1]/127.0)-1.0, (bn[0]/127.0)-1.0),
                        crash_normal=((cn[2]/127.0)-1.0, (cn[1]/127.0)-1.0, (cn[0]/127.0)-1.0),
                    )
                    surf.weights.append(w)
            else:
                # FO2: vcount, vbytes, vbuffer, then 48-byte weights
                num_verts_bytes = struct.unpack_from('<I', r.read(4), 0)[0]
                surf.vertex_size = num_verts_bytes // num_verts if num_verts > 0 else 0
                surf.vertex_data = r.read(num_verts_bytes)
                surf.weights = []
                for k in range(num_verts):
                    raw = struct.unpack_from('<12f', r.read(48), 0)
                    w = CrashWeight(
                        base_pos=raw[0:3],
                        crash_pos=raw[3:6],
                        base_normal=raw[6:9],
                        crash_normal=raw[9:12],
                    )
                    surf.weights.append(w)
            node.surfaces.append(surf)
        nodes.append(node)

    print(f"[crash.dat] Parsed {len(nodes)} crash nodes ({'FOUC' if is_fouc else 'FO2'})")
    return nodes


@dataclass
class ParsedVertex:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    nx: float = 0.0
    ny: float = 0.0
    nz: float = 0.0
    u: float = 0.0
    v: float = 0.0
    r: float = 1.0
    g: float = 1.0
    b: float = 1.0
    a: float = 1.0
    has_normal: bool = False
    has_uv: bool = False
    has_color: bool = False


class BinaryReader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def read(self, n: int) -> bytes:
        result = self.data[self.pos:self.pos + n]
        self.pos += n
        return result

    def u8(self) -> int:
        return struct.unpack_from('<B', self.data, self._adv(1))[0]

    def u16(self) -> int:
        return struct.unpack_from('<H', self.data, self._adv(2))[0]

    def u32(self) -> int:
        return struct.unpack_from('<I', self.data, self._adv(4))[0]

    def i32(self) -> int:
        return struct.unpack_from('<i', self.data, self._adv(4))[0]

    def f32(self) -> float:
        return struct.unpack_from('<f', self.data, self._adv(4))[0]

    def vec3f(self) -> tuple:
        return (self.f32(), self.f32(), self.f32())

    def read_string(self) -> str:
        start = self.pos
        while self.data[self.pos] != 0:
            self.pos += 1
        s = self.data[start:self.pos].decode('ascii', errors='replace')
        self.pos += 1
        return s

    def _adv(self, n: int) -> int:
        old = self.pos
        self.pos += n
        return old


class BGMParser:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.version = 0
        self.is_fouc = False
        self.materials: list[BGMMaterial] = []
        self.vertex_buffers: dict[int, VertexBuffer] = {}
        self.index_buffers: dict[int, IndexBuffer] = {}
        self.surfaces: list[Surface] = []
        self.models: list[Model] = []
        self.meshes: list[BGMMesh] = []
        self.objects: list[BGMObject] = []

    def parse(self) -> bool:
        with open(self.filepath, 'rb') as f:
            data = f.read()
        r = BinaryReader(data)

        self.version = r.u32()

        if self.version not in (0x20000, 0x10004, 0x10002):
            print(f"[BGM] WARNING: Unexpected version 0x{self.version:X}")

        # materials
        num_materials = r.u32()
        for i in range(num_materials):
            mat = BGMMaterial()
            ident = r.u32()
            if ident != 0x4354414D:  # MATC
                print(f"[BGM] ERROR: Expected MATC at material {i}")
                return False
            mat.name = r.read_string()
            mat.nAlpha = r.i32()
            if self.version >= 0x10004 or self.version == 0x10002:
                mat.v92 = r.i32()
                mat.nNumTextures = r.i32()
                mat.nShaderId = r.i32()
                mat.nUseColormap = r.i32()
                mat.v74 = r.i32()
                mat.v108 = struct.unpack_from('<3i', r.read(12))
                mat.v109 = struct.unpack_from('<3i', r.read(12))
            mat.v98 = struct.unpack_from('<4i', r.read(16))
            mat.v99 = struct.unpack_from('<4i', r.read(16))
            mat.v100 = struct.unpack_from('<4i', r.read(16))
            mat.v101 = struct.unpack_from('<4i', r.read(16))
            mat.v102 = r.i32()
            mat.texture_names = [r.read_string() for _ in range(3)]
            self.materials.append(mat)

        # streams
        num_streams = r.u32()
        for i in range(num_streams):
            data_type = r.i32()
            if data_type == 1:
                vb = VertexBuffer(buf_id=i)
                vb.fouc_extra_format = r.i32()
                if vb.fouc_extra_format > 0:
                    self.is_fouc = True
                vb.vertex_count = r.u32()
                vb.vertex_size = r.u32()
                vb.flags = r.u32()
                vb.data = r.read(vb.vertex_count * vb.vertex_size)
                self.vertex_buffers[i] = vb
            elif data_type == 2:
                ib = IndexBuffer(buf_id=i)
                ib.fouc_extra_format = r.i32()
                ib.index_count = r.u32()
                ib.data = r.read(ib.index_count * 2)
                self.index_buffers[i] = ib
            elif data_type == 3:
                vb = VertexBuffer(buf_id=i, is_vegetation=True)
                vb.fouc_extra_format = r.i32()
                vb.vertex_count = r.u32()
                vb.vertex_size = r.u32()
                vb.flags = 0
                vb.data = r.read(vb.vertex_count * vb.vertex_size)
                self.vertex_buffers[i] = vb
            else:
                print(f"[BGM] ERROR: Unknown stream type {data_type}")
                return False

        # surfaces
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
            if self.version < 0x20000:
                s.center = r.vec3f()
                s.radius = r.vec3f()
            if self.is_fouc:
                s.fouc_vertex_multiplier = [r.f32() for _ in range(4)]
            s.num_streams_used = r.i32()
            if s.num_streams_used < 1 or s.num_streams_used > 2:
                print(f"[BGM] ERROR: Invalid stream count {s.num_streams_used} for surface {i}")
                return False
            s.stream_id = [0, 0]
            s.stream_offset = [0, 0]
            for j in range(s.num_streams_used):
                s.stream_id[j] = r.u32()
                s.stream_offset[j] = r.u32()
            self.surfaces.append(s)

        # models
        num_models = r.u32()
        for i in range(num_models):
            m = Model()
            ident = r.u32()
            if ident != 0x444F4D42:  # BMOD
                print(f"[BGM] ERROR: Expected BMOD at model {i}")
                return False
            m.nUnk = r.i32()
            m.name = r.read_string()
            m.center = r.vec3f()
            m.radius = r.vec3f()
            m.fRadius = r.f32()
            num_surf = r.u32()
            for _ in range(num_surf):
                m.surface_ids.append(r.i32())
            self.models.append(m)

        # BGM meshes
        num_meshes = r.u32()
        for i in range(num_meshes):
            mesh = BGMMesh()
            ident = r.u32()
            if ident != 0x4853454D:  # MESH
                print(f"[BGM] ERROR: Expected MESH at mesh {i}")
                return False
            mesh.name1 = r.read_string()
            mesh.name2 = r.read_string()
            mesh.flags = r.u32()
            mesh.group = r.i32()
            mesh.matrix = list(struct.unpack_from('<16f', r.read(64)))
            num_m = r.i32()
            for _ in range(num_m):
                mesh.model_ids.append(r.i32())
            self.meshes.append(mesh)

        # objects
        num_objects = r.u32()
        for i in range(num_objects):
            obj = BGMObject()
            ident = r.u32()
            if ident != 0x434A424F:  # OBJC
                print(f"[BGM] ERROR: Expected OBJC at object {i}")
                return False
            obj.name1 = r.read_string()
            obj.name2 = r.read_string()
            obj.flags = r.u32()
            obj.matrix = list(struct.unpack_from('<16f', r.read(64)))
            self.objects.append(obj)

        print(f"[BGM] Parsed {self.filepath}: version=0x{self.version:X}, "
              f"{len(self.materials)} mats, {len(self.meshes)} meshes, "
              f"{len(self.surfaces)} surfaces, {len(self.objects)} objects")
        return True


# VERTEX / INDEX EXTRACTION

def extract_vertices(parser: BGMParser, surface: Surface) -> list:
    vb = parser.vertex_buffers.get(surface.stream_id[0])
    if vb is None:
        return []
    flags = vb.flags if not vb.is_vegetation else surface.flags
    stride = vb.vertex_size
    base_offset = surface.stream_offset[0]
    is_fouc = parser.is_fouc or (vb.fouc_extra_format > 0)
    vertices = []

    for i in range(surface.vertex_count):
        v = ParsedVertex()
        offset = base_offset + i * stride

        if is_fouc:
            # tVertexDataFOUC layout (32 bytes):
            #   offset  0: int16[3]  vPos
            #   offset  6: uint16    pad
            #   offset  8: uint8[4]  vTangents
            #   offset 12: uint8[4]  vBitangents
            #   offset 16: uint8[4]  vNormals   ← [0]=FO2.z, [1]=FO2.y, [2]=FO2.x
            #   offset 20: uint8[4]  vVertexColors
            #   offset 24: uint16[2] vUV1
            #   offset 28: uint16[2] vUV2
            # Position decode per C++ reference (w32fbxexport.h):
            #   raw = int16 value
            #   FO2.x = (raw_x + mult[0]) * mult[3]
            #   FO2.y = (raw_y + mult[1]) * mult[3]
            #   FO2.z = (raw_z + mult[2]) * mult[3]
            # mult[0,1,2] are per-surface int16-space offsets (non-zero on shadow/special surfaces)
            # mult[3] is the scale (default 0.000977 = 1/1024, but can differ per surface)
            scale  = surface.fouc_vertex_multiplier[3] if surface.fouc_vertex_multiplier[3] != 0 else FOUC_VERTEX_SCALE
            off_x  = surface.fouc_vertex_multiplier[0]
            off_y  = surface.fouc_vertex_multiplier[1]
            off_z  = surface.fouc_vertex_multiplier[2]
            px, py, pz = struct.unpack_from('<3h', vb.data, offset)
            v.x = (px + off_x) * scale
            v.y = (py + off_y) * scale
            v.z = (pz + off_z) * scale
            nrm = struct.unpack_from('<4B', vb.data, offset + 16)
            v.nx = (nrm[2] / 127.0) - 1.0  # FO2 X  (buffer byte 18)
            v.ny = (nrm[1] / 127.0) - 1.0  # FO2 Y  (buffer byte 17)
            v.nz = (nrm[0] / 127.0) - 1.0  # FO2 Z  (buffer byte 16)
            v.has_normal = True
            col = struct.unpack_from('<4B', vb.data, offset + 20)
            v.r, v.g, v.b, v.a = col[0]/255.0, col[1]/255.0, col[2]/255.0, col[3]/255.0
            v.has_color = True
            uv = struct.unpack_from('<2h', vb.data, offset + 24)
            v.u, v.v = uv[0] / 2048.0, uv[1] / 2048.0
            v.has_uv = True
        else:
            flt_offset = 0
            v.x, v.y, v.z = struct.unpack_from('<3f', vb.data, offset)
            flt_offset += 3
            if flags & VERTEX_NORMAL:
                bp = offset + flt_offset * 4
                v.nx, v.ny, v.nz = struct.unpack_from('<3f', vb.data, bp)
                v.has_normal = True
                flt_offset += 3
            if flags & VERTEX_COLOR:
                bp = offset + flt_offset * 4
                c = struct.unpack_from('<I', vb.data, bp)[0]
                v.r = (c & 0xFF) / 255.0
                v.g = ((c >> 8) & 0xFF) / 255.0
                v.b = ((c >> 16) & 0xFF) / 255.0
                v.a = ((c >> 24) & 0xFF) / 255.0
                v.has_color = True
                flt_offset += 1
            if (flags & VERTEX_UV) or (flags & VERTEX_UV2):
                bp = offset + flt_offset * 4
                v.u, v.v = struct.unpack_from('<2f', vb.data, bp)
                v.has_uv = True
                flt_offset += 2

        vertices.append(v)
    return vertices


def extract_indices(parser: BGMParser, surface: Surface) -> list:
    if surface.num_streams_used < 2:
        return []
    ib = parser.index_buffers.get(surface.stream_id[1])
    if ib is None:
        return []
    base_offset = surface.stream_offset[1]
    indices = []
    if surface.poly_mode == 4:
        for j in range(surface.poly_count):
            off = base_offset + j * 6
            i0, i1, i2 = struct.unpack_from('<3H', ib.data, off)
            # reverse winding
            indices.append((i2, i1, i0))
    elif surface.poly_mode == 5:
        flip = False
        for j in range(surface.poly_count):
            off = base_offset + j * 2
            i0, i1, i2 = struct.unpack_from('<3H', ib.data, off)
            if flip:
                indices.append((i0, i1, i2))
            else:
                indices.append((i2, i1, i0))
            flip = not flip
    return indices


# MATRIX / AXIS HELPERS

def fo2_matrix_to_blender(m: list) -> Matrix:
    """Convert a FO2 column-major 4x4 matrix to a Blender Matrix.
    FO2 stores column-major, Blender Matrix() takes row-major."""
    return Matrix((
        (m[0], m[4], m[8],  m[12]),
        (m[1], m[5], m[9],  m[13]),
        (m[2], m[6], m[10], m[14]),
        (m[3], m[7], m[11], m[15]),
    ))


AXIS_MAP = {
    'X':  Vector((1, 0, 0)),
    '-X': Vector((-1, 0, 0)),
    'Y':  Vector((0, 1, 0)),
    '-Y': Vector((0, -1, 0)),
    'Z':  Vector((0, 0, 1)),
    '-Z': Vector((0, 0, -1)),
}

AXIS_ITEMS = [
    ('X', "X", ""),
    ('-X', "-X", ""),
    ('Y', "Y", ""),
    ('-Y', "-Y", ""),
    ('Z', "Z", ""),
    ('-Z', "-Z", ""),
]


def build_axis_matrix(forward: str, up: str) -> Matrix:
    """Build a rotation matrix that maps FO2 axes (Y-up, -Z forward) to the forward/up convention."""
    fwd = AXIS_MAP[forward]
    upv = AXIS_MAP[up]
    right = upv.cross(fwd)
    if right.length < 0.001:
        # degenerate, forward and up are parallel, pick fallback
        right = Vector((1, 0, 0))
    right.normalize()
    # re-orthogonalize
    actual_up = fwd.cross(right)
    actual_up.normalize()
    return Matrix((
        (right.x, fwd.x, actual_up.x, 0),
        (right.y, fwd.y, actual_up.y, 0),
        (right.z, fwd.z, actual_up.z, 0),
        (0, 0, 0, 1),
    ))


# TEXTURE RESOLUTION

def tga_to_dds(name: str) -> str:
    if not name:
        return name
    base, ext = os.path.splitext(name)
    if ext.lower() == '.tga':
        return base + '.dds'
    return name


def find_texture_file(tex_name: str, bgm_dir: str, shared_dir: str,
                      auto_shared_dir: str = "", convert_dds: bool = False,
                      use_normal_converter: bool = False) -> str:
    """Resolve a texture filename to a full path on disk.

    Search order per directory: TGA first, then DDS.
    Directories searched: bgm_dir -> auto_shared_dir -> shared_dir.

    If convert_dds=True and the texture is only found as a DDS, it is
    converted to TGA (placed in bgm_dir) and the TGA path is returned.
    Does case-insensitive matching on all platforms."""
    if not tex_name:
        return ""

    base = os.path.splitext(tex_name)[0]
    tga_name = base + '.tga'
    dds_name = base + '.dds'

    # build ordered search list, avoid duplicates
    search_dirs = [bgm_dir]
    if auto_shared_dir and auto_shared_dir != bgm_dir:
        search_dirs.append(auto_shared_dir)
    if shared_dir and shared_dir not in search_dirs:
        search_dirs.append(shared_dir)

    found_dds = ""  # best DDS path found (first dir wins)

    for search_dir in search_dirs:
        if not search_dir or not os.path.isdir(search_dir):
            continue
        try:
            entries = os.listdir(search_dir)
        except OSError:
            continue
        entries_lower = {e.lower(): e for e in entries}

        # TGA has absolute priority in every directory
        tga_match = entries_lower.get(tga_name.lower())
        if tga_match:
            return os.path.join(search_dir, tga_match)

        # remember the first DDS found for fallback
        if not found_dds:
            dds_match = entries_lower.get(dds_name.lower())
            if dds_match:
                found_dds = os.path.join(search_dir, dds_match)

    # no TGA found anywhere, deal with DDS
    if found_dds:
        if convert_dds:
            out_tga = os.path.join(bgm_dir, tga_name)
            try:
                if use_normal_converter: # if texture is _normal
                    _dds_normal.convert_normalmap(found_dds, out_tga, to_fouc=False, tga_out=True)
                else:
                    _dds2tga.convert_dds_to_tga(found_dds, out_tga)
                print(f"[BGM Import] Converted: {os.path.basename(found_dds)} → {tga_name}")
                return out_tga
            except Exception as exc:
                print(f"[BGM Import] DDS→TGA conversion failed for "
                      f"{os.path.basename(found_dds)}: {exc}")
        return found_dds  # fall back to DDS so Blender can still attempt loading

    return ""


# BLENDER MATERIAL CREATION

def _find_sibling_texture(base_tex_name: str, suffix: str, bgm_dir: str,
                           shared_dir: str, auto_shared_dir: str,
                           convert_dds: bool) -> str:
    """Find a sidecar texture like skin1_normal.dds / skin1_specular.dds.
    base_tex_name is e.g. 'skin1.tga'. Returns resolved path or ''."""
    import os as _os
    stem = _os.path.splitext(base_tex_name)[0]
    for ext in ('.dds', '.tga', '.png'):
        candidate = stem + suffix + ext
        path = find_texture_file(candidate, bgm_dir, shared_dir,
                                  auto_shared_dir, convert_dds,
                                  use_normal_converter=(suffix == '_normal'))
        if path:
            return path
    return ""


def _load_or_find_image(tex_path: str) -> 'bpy.types.Image | None':
    """Load an image, reusing existing if already in bpy.data.images."""
    img_basename = os.path.basename(tex_path)
    for existing in bpy.data.images:
        if existing.filepath and os.path.basename(existing.filepath) == img_basename:
            return existing
    try:
        return bpy.data.images.load(tex_path)
    except RuntimeError:
        print(f"[BGM] WARNING: Could not load texture: {tex_path}")
        return None


def create_blender_material(bgm_mat: BGMMaterial, bgm_dir: str, shared_dir: str,
                            use_alpha: bool, alpha_mode: str = 'BLEND',
                            transparency_overlap: bool = False,
                            auto_shared_dir: str = "",
                            convert_dds: bool = False,
                            use_backface_culling: bool = True,
                            is_fouc: bool = False,
                            import_normal_maps: bool = True,
                            import_specular_maps: bool = False) -> bpy.types.Material:
    """Create a Blender material with Principled BSDF from a BGM material."""
    mat_name = bgm_mat.name if bgm_mat.name else "bgm_unnamed"
    bl_mat = bpy.data.materials.new(name=mat_name)
    bl_mat.use_nodes = True
    nodes = bl_mat.node_tree.nodes
    links = bl_mat.node_tree.links

    # clear defaults
    nodes.clear()

    # create output + principled BSDF
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    # set some defaults
    bsdf.inputs['Specular IOR Level'].default_value = 0.0
    bsdf.inputs['Roughness'].default_value = 0.9

    # find diffuse texture (slot 0 is primary for BGM)
    tex_name = ""
    for idx in (0, 1, 2):
        if bgm_mat.texture_names[idx]:
            tex_name = bgm_mat.texture_names[idx]
            break

    if tex_name:
        # FO1 stores "body.tga" in the BGM but the actual file on disk is skin1.tga
        tex_lookup_name = "skin1.tga" if tex_name.lower() == "body.tga" else tex_name
        tex_path = find_texture_file(tex_lookup_name, bgm_dir, shared_dir,
                                      auto_shared_dir, convert_dds)

        if tex_path:
            img = _load_or_find_image(tex_path)

            if img:
                img.alpha_mode = 'STRAIGHT'

                tex_node = nodes.new('ShaderNodeTexImage')
                tex_node.image = img
                tex_node.location = (-400, 0)
                links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])

                mat_has_alpha = use_alpha and (bgm_mat.nAlpha != 0)
                if mat_has_alpha:
                    links.new(tex_node.outputs['Alpha'], bsdf.inputs['Alpha'])

                # FOUC: look for _normal and _specular sidecar textures
                if is_fouc:
                    if import_normal_maps:
                        nrm_path = _find_sibling_texture(
                            tex_name, '_normal', bgm_dir, shared_dir, auto_shared_dir, convert_dds)
                        if nrm_path:
                            nrm_img = _load_or_find_image(nrm_path)
                            if nrm_img:
                                nrm_img.colorspace_settings.name = 'Non-Color'
                                nrm_node = nodes.new('ShaderNodeTexImage')
                                nrm_node.image = nrm_img
                                nrm_node.location = (-700, -200)
                                nrm_map = nodes.new('ShaderNodeNormalMap')
                                nrm_map.location = (-400, -200)
                                links.new(nrm_node.outputs['Color'], nrm_map.inputs['Color'])
                                links.new(nrm_map.outputs['Normal'], bsdf.inputs['Normal'])

                    if import_specular_maps:
                        spec_path = _find_sibling_texture(
                            tex_name, '_specular', bgm_dir, shared_dir, auto_shared_dir, convert_dds)
                        if spec_path:
                            spec_img = _load_or_find_image(spec_path)
                            if spec_img:
                                spec_img.colorspace_settings.name = 'Non-Color'
                                spec_node = nodes.new('ShaderNodeTexImage')
                                spec_node.image = spec_img
                                spec_node.location = (-700, -500)
                                links.new(spec_node.outputs['Color'], bsdf.inputs['Specular IOR Level'])
        else:
            # Texture not found — leave a placeholder
            print(f"[BGM] WARNING: Texture not found: {tex_name}")
            tex_node = nodes.new('ShaderNodeTexImage')
            tex_node.label = tex_name
            tex_node.location = (-400, 0)
            links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])

    # store shader metadata as custom properties
    bl_mat["bgm_shader_id"] = bgm_mat.nShaderId
    # sync the enum property so the panel shows the correct shader immediately
    try:
        bl_mat.fo2_shader_id = str(bgm_mat.nShaderId)
    except Exception:
        pass
    try:
        bl_mat.fo2_texture = tex_name
    except Exception:
        pass
    bl_mat["bgm_alpha"] = bgm_mat.nAlpha
    bl_mat["bgm_v92"] = bgm_mat.v92
    bl_mat["bgm_num_textures"] = bgm_mat.nNumTextures
    bl_mat["bgm_use_colormap"] = bgm_mat.nUseColormap
    bl_mat["bgm_v74"] = bgm_mat.v74
    bl_mat["bgm_v102"] = bgm_mat.v102
    bl_mat["bgm_texture"] = tex_name  # e.g. "windows.tga"
    # store all 3 texture slot names
    for ti in range(3):
        bl_mat[f"bgm_texture_{ti}"] = bgm_mat.texture_names[ti]
    
    # set alpha blend mode per-material based on the game's own alpha flag
    # mat_has_alpha may not be set if no texture was found, so derive it here
    mat_has_alpha = use_alpha and (bgm_mat.nAlpha != 0)
    if mat_has_alpha:
        try:
            bl_mat.blend_method = alpha_mode
        except AttributeError:
            pass
        try:
            bl_mat.shadow_method = 'CLIP' if alpha_mode == 'BLEND' else 'HASHED'
        except AttributeError:
            pass
        try:
            bl_mat.show_transparent_back = transparency_overlap
        except AttributeError:
            pass
    else:
        try:
            bl_mat.blend_method = 'OPAQUE'
        except AttributeError:
            pass

    bl_mat.use_backface_culling = use_backface_culling

    return bl_mat


def extract_crash_vertices(crash_surf: 'CrashSurface', bgm_surf_flags: int) -> list:
    """Extract vertices from a crash surface using crash positions/normals and UVs from the crash vertex buffer."""
    vertices = []
    flags = bgm_surf_flags if bgm_surf_flags else crash_surf.flags
    stride = crash_surf.vertex_size

    for i in range(crash_surf.vertex_count):
        v = ParsedVertex()
        w = crash_surf.weights[i]

        # positions and normals from crash weights
        v.x, v.y, v.z = w.crash_pos
        v.nx, v.ny, v.nz = w.crash_normal
        v.has_normal = True

        # UVs from the crash vertex buffer (same layout as BGM vertex buffer)
        offset = i * stride
        flt_offset = 3  # skip position (3 floats)
        if flags & VERTEX_NORMAL:
            flt_offset += 3
        if flags & VERTEX_COLOR:
            bp = offset + flt_offset * 4
            if bp + 4 <= len(crash_surf.vertex_data):
                c = struct.unpack_from('<I', crash_surf.vertex_data, bp)[0]
                v.r = (c & 0xFF) / 255.0
                v.g = ((c >> 8) & 0xFF) / 255.0
                v.b = ((c >> 16) & 0xFF) / 255.0
                v.a = ((c >> 24) & 0xFF) / 255.0
                v.has_color = True
            flt_offset += 1
        if (flags & VERTEX_UV) or (flags & VERTEX_UV2):
            bp = offset + flt_offset * 4
            if bp + 8 <= len(crash_surf.vertex_data):
                v.u, v.v = struct.unpack_from('<2f', crash_surf.vertex_data, bp)
                v.has_uv = True

        vertices.append(v)
    return vertices


# BLENDER MESH BUILDER

def build_blender_meshes(context, parser: BGMParser, options: dict):
    """Build Blender mesh objects from parsed BGM data."""

    bgm_dir = os.path.dirname(parser.filepath)
    shared_dir = options.get('shared_texture_dir', '')
    use_alpha = options.get('use_alpha', True)
    alpha_mode = options.get('alpha_mode', 'BLEND')
    transparency_overlap = options.get('transparency_overlap', False)
    max_lod = options.get('max_lod', 0)
    global_scale = options.get('global_scale', 1.0)
    clamp_size = options.get('clamp_size', 0.0)
    use_origins = options.get('use_origins', True)
    split_by_object = options.get('split_by_object', True)
    split_by_group = options.get('split_by_group', False)
    validate_meshes = options.get('validate_meshes', False)
    convert_dds = options.get('convert_dds', False)
    import_normal_maps  = options.get('import_normal_maps', True)
    import_specular_maps = options.get('import_specular_maps', False)
    use_backface_culling = options.get('use_backface_culling', True)

    # auto-detect a shared texture directory one level up (e.g. data/cars/shared)
    auto_shared_dir = os.path.join(os.path.dirname(bgm_dir), 'shared')
    if not os.path.isdir(auto_shared_dir):
        auto_shared_dir = ""

    # fixed coordinate transform: FO2 (x,y,z) → Blender (x, z, y)
    # car front faces Blender +Y, implemented as Y<->Z swap (self-inverse).
    axis_matrix = Matrix((
        (1, 0, 0, 0),
        (0, 0, 1, 0),
        (0, 1, 0, 0),
        (0, 0, 0, 1),
    ))

    # load crash.dat, auto-detect if user didn't specify a path
    crash_dat_path = options.get('crash_dat_path', '')
    if not crash_dat_path or not os.path.isfile(crash_dat_path):
        # try <input_name>_crash.dat first then crash.dat in same directory
        base_no_ext = os.path.splitext(parser.filepath)[0]
        candidate1 = base_no_ext + "_crash.dat"
        candidate2 = os.path.join(bgm_dir, "crash.dat")
        # also try hyphen variant for compatibility
        candidate3 = base_no_ext + "-crash.dat"
        if os.path.isfile(candidate1):
            crash_dat_path = candidate1
        elif os.path.isfile(candidate2):
            crash_dat_path = candidate2
        elif os.path.isfile(candidate3):
            crash_dat_path = candidate3
        else:
            crash_dat_path = ''
    import_crash   = options.get('import_crash',   True)
    import_body    = options.get('import_body',    True)
    import_dummies = options.get('import_dummies', True)

    crash_nodes = []
    if import_crash and crash_dat_path and os.path.isfile(crash_dat_path):
        crash_nodes = parse_crash_dat(crash_dat_path, is_fouc=parser.is_fouc)
        print(f"[BGM Import] Loaded crash data from: {crash_dat_path}")
    # build lookup: model_name -> CrashNode (crash node name = model_name + "_crash")
    crash_by_model = {}
    for cn in crash_nodes:
        if cn.name.endswith("_crash"):
            model_name = cn.name[:-6]  # strip "_crash"
            crash_by_model[model_name] = cn

    # create materials
    blender_materials = {}
    for i, bgm_mat in enumerate(parser.materials):
        bl_mat = create_blender_material(bgm_mat, bgm_dir, shared_dir, use_alpha,
                                            alpha_mode, transparency_overlap,
                                            auto_shared_dir, convert_dds,
                                            use_backface_culling,
                                            is_fouc=parser.is_fouc,
                                            import_normal_maps=import_normal_maps,
                                            import_specular_maps=import_specular_maps)
        blender_materials[i] = bl_mat

    # collect surfaces per mesh (and crash surfaces if crash.dat exists)
    mesh_exports = []
    crash_exports = []  # parallel to mesh_exports: list of (bgm_mesh, [(surf, crash_surf, flags), ...])
    for mesh in parser.meshes:
        surfaces = []
        crash_surfaces = []  # list of (bgm_surface, crash_surface, bgm_flags)
        num_lods = min(len(mesh.model_ids), max_lod + 1)
        for lod_idx in range(num_lods):
            model_id = mesh.model_ids[lod_idx]
            if model_id < 0 or model_id >= len(parser.models):
                continue
            model = parser.models[model_id]
            crash_node = crash_by_model.get(model.name)
            for surf_idx, surf_id in enumerate(model.surface_ids):
                if surf_id < 0 or surf_id >= len(parser.surfaces):
                    continue
                surf = parser.surfaces[surf_id]
                if surf.num_streams_used < 2:
                    continue
                if surf.stream_id[0] not in parser.vertex_buffers:
                    continue
                if surf.stream_id[1] not in parser.index_buffers:
                    continue
                if surf.poly_count <= 0:
                    continue
                surfaces.append(surf)
                # match crash surface if available
                if crash_node and surf_idx < len(crash_node.surfaces):
                    cs = crash_node.surfaces[surf_idx]
                    # copy flags from BGM surface for vertex format detection
                    cs.flags = surf.flags
                    crash_surfaces.append((surf, cs))
        if surfaces:
            mesh_exports.append((mesh, surfaces))
            crash_exports.append((mesh, crash_surfaces))

    # create "FO2 Body" collection (reuse if it already exists)
    fo2_body_coll = bpy.data.collections.get("FO2 Body")
    if fo2_body_coll is None:
        fo2_body_coll = bpy.data.collections.new("FO2 Body")
    if fo2_body_coll.name not in context.scene.collection.children:
        context.scene.collection.children.link(fo2_body_coll)

    # create root empty
    root_empty = bpy.data.objects.new("fo2_body", None)
    root_empty.empty_display_type = 'PLAIN_AXES'
    root_empty.empty_display_size = 0.5
    root_empty["bgm_is_fouc"] = parser.is_fouc
    root_empty["bgm_is_fo1"]  = (parser.version < 0x20000 and not parser.is_fouc)
    root_empty["bgm_version"] = parser.version
    fo2_body_coll.objects.link(root_empty)

    # create crash root empty and "FO2 Body Crash" collection (if crash.dat exists)
    fo2_crash_coll = None
    crash_root_empty = None
    if crash_by_model:
        fo2_crash_coll = bpy.data.collections.get("FO2 Body Crash")
        if fo2_crash_coll is None:
            fo2_crash_coll = bpy.data.collections.new("FO2 Body Crash")
        if fo2_crash_coll.name not in context.scene.collection.children:
            context.scene.collection.children.link(fo2_crash_coll)

        crash_root_empty = bpy.data.objects.new("fo2_body_crash", None)
        crash_root_empty.empty_display_type = 'PLAIN_AXES'
        crash_root_empty.empty_display_size = 0.5
        fo2_crash_coll.objects.link(crash_root_empty)

    # create "FO2 Body Dummies" collection
    fo2_dummies_coll = bpy.data.collections.get("FO2 Body Dummies")
    if fo2_dummies_coll is None:
        fo2_dummies_coll = bpy.data.collections.new("FO2 Body Dummies")
    if fo2_dummies_coll.name not in context.scene.collection.children:
        context.scene.collection.children.link(fo2_dummies_coll)

    dummies_empty = bpy.data.objects.new("fo2_body_dummies", None)
    dummies_empty.empty_display_type = 'PLAIN_AXES'
    dummies_empty.empty_display_size = 0.5
    fo2_dummies_coll.objects.link(dummies_empty)
    dummies_empty.parent = root_empty

    # build group empties for objects (dummies)
    object_empties = {}
    if not import_dummies:
        parser.objects = []
    for bgm_obj in parser.objects:
        obj_empty = bpy.data.objects.new(bgm_obj.name1, None)
        obj_empty.empty_display_type = 'PLAIN_AXES'
        obj_empty.empty_display_size = 0.3

        # convert FO2 matrix to Blender space.
        # new mapping: FO2(x,y,z) -> Blender(x,z,y) — swap rows/cols 1<->2, no sign flips.
        M = fo2_matrix_to_blender(bgm_obj.matrix)
        obj_mat = Matrix((
            (M[0][0], M[0][2], M[0][1], M[0][3]),
            (M[2][0], M[2][2], M[2][1], M[2][3]),
            (M[1][0], M[1][2], M[1][1], M[1][3]),
            (M[3][0], M[3][2], M[3][1], M[3][3]),
        ))
        # apply global scale to translation
        obj_mat[0][3] *= global_scale
        obj_mat[1][3] *= global_scale
        obj_mat[2][3] *= global_scale
        obj_empty.matrix_world = obj_mat

        fo2_dummies_coll.objects.link(obj_empty)
        obj_empty.parent = dummies_empty
        obj_empty["bgm_obj_flags"] = bgm_obj.flags
        object_empties[bgm_obj.name1] = obj_empty

    created_objects = []

    if not import_body:
        mesh_exports = []
        crash_exports = []
    for (bgm_mesh, surfaces), (_, crash_surface_pairs) in zip(mesh_exports, crash_exports):
        mesh_matrix = fo2_matrix_to_blender(bgm_mesh.matrix)

        # merge all surfaces of this mesh into one Blender mesh
        all_verts = []
        all_normals = []
        all_uvs = []        # per-vertex UV (used only for newly created verts)
        all_face_uvs = []   # per-loop UV (one entry per face-corner, always correct)
        all_colors = []
        all_faces = []
        all_face_mat_indices = []
        mat_index_map = {}  # bgm mat id -> local mesh mat index
        mesh_materials = []  # ordered list of blender materials for this mesh

        # two level vertex deduplication to eliminate seam creases when merging surfaces:
        #
        # 1 — by (stream_id, absolute_vb_index): surfaces that share the same
        #   physical VB data (same stream, same index) always get the same Blender vertex.
        #
        # 2 — by (decoded_position, decoded_normal): surfaces that DON'T share VB
        #   indices but have vertices at the same 3D position with the same normal (i.e.
        #   seam boundary vertices duplicated into separate VB regions) are also merged.
        #   Vertices at the same position with DIFFERENT normals are kept separate — those
        #   represent intentional hard edges.

        abs_stream_idx_to_vert = {}   # (stream_id, abs_vb_idx) -> bl_vi
        pos_nrm_to_vert       = {}    # (pos_key, nrm_key)       -> bl_vi

        for surf in surfaces:
            vb = parser.vertex_buffers[surf.stream_id[0]]
            verts = extract_vertices(parser, surf)
            if not verts:
                continue
            faces = extract_indices(parser, surf)
            if not faces:
                continue

            base_vertex_offset = surf.stream_offset[0] // vb.vertex_size

            # material index
            mat_id = surf.material_id
            if mat_id not in mat_index_map:
                mat_index_map[mat_id] = len(mesh_materials)
                if mat_id in blender_materials:
                    mesh_materials.append(blender_materials[mat_id])
                else:
                    mesh_materials.append(None)
            local_mat_idx = mat_index_map[mat_id]

            has_normals = any(v.has_normal for v in verts)
            has_uvs = any(v.has_uv for v in verts)
            has_colors = any(v.has_color for v in verts)

            local_to_blender = {}  # local surface vert index -> index in all_verts
            for vi, v in enumerate(verts):
                # level 1: exact VB index match
                abs_key = (surf.stream_id[0], base_vertex_offset + vi)
                if abs_key in abs_stream_idx_to_vert:
                    local_to_blender[vi] = abs_stream_idx_to_vert[abs_key]
                    continue

                # level 2: same decoded position + same decoded normal
                # use raw float values rounded to avoid fp noise; 
                # normal as 3-tuple of rounded floats so quantization doesn't prevent matching.
                nrm_key = (round(v.nx, 4), round(v.ny, 4), round(v.nz, 4))                           if has_normals else None
                pos_key = (round(v.x, 6), round(v.y, 6), round(v.z, 6))
                pn_key  = (pos_key, nrm_key)
                if pn_key in pos_nrm_to_vert:
                    bl_vi = pos_nrm_to_vert[pn_key]
                    abs_stream_idx_to_vert[abs_key] = bl_vi
                    local_to_blender[vi] = bl_vi
                    continue

                # New vertex
                bl_vi = len(all_verts)
                abs_stream_idx_to_vert[abs_key] = bl_vi
                pos_nrm_to_vert[pn_key]         = bl_vi
                local_to_blender[vi]            = bl_vi

                pos = Vector((v.x, v.y, v.z))
                if not use_origins:
                    pos = mesh_matrix @ pos
                pos = axis_matrix @ pos
                pos *= global_scale
                all_verts.append(pos)

                if has_normals:
                    nrm = Vector((v.nx, v.ny, v.nz))
                    if not use_origins:
                        nrm = mesh_matrix.to_3x3() @ nrm
                    nrm = axis_matrix.to_3x3() @ nrm
                    if nrm.length > 0:
                        nrm.normalize()
                    all_normals.append(nrm)
                else:
                    all_normals.append(Vector((0, 0, 1)))

                if has_uvs:
                    all_uvs.append((v.u, 1.0 - v.v))
                else:
                    all_uvs.append((0.0, 0.0))

                if has_colors:
                    all_colors.append((v.r, v.g, v.b, v.a))

            # add faces (reversed winding), using deduplicated vertex indices
            for i0, i1, i2 in faces:
                fi0 = i0 - base_vertex_offset
                fi1 = i1 - base_vertex_offset
                fi2 = i2 - base_vertex_offset
                if not (0 <= fi0 < len(verts) and 0 <= fi1 < len(verts) and 0 <= fi2 < len(verts)):
                    continue
                bl0 = local_to_blender[fi0]
                bl1 = local_to_blender[fi1]
                bl2 = local_to_blender[fi2]
                if bl0 == bl1 or bl1 == bl2 or bl0 == bl2:
                    continue
                all_faces.append((bl0, bl1, bl2))
                all_face_mat_indices.append(local_mat_idx)
                # store per-loop UVs from the actual source vertices so that
                # merged verts (level-2 dedup) keep their correct per-corner UV
                if has_uvs:
                    all_face_uvs.append((
                        (verts[fi0].u, 1.0 - verts[fi0].v),
                        (verts[fi1].u, 1.0 - verts[fi1].v),
                        (verts[fi2].u, 1.0 - verts[fi2].v),
                    ))
                else:
                    all_face_uvs.append(((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)))

        if not all_faces:
            continue

        # create Blender mesh
        mesh_name = bgm_mesh.name1 if bgm_mesh.name1 else "bgm_mesh"
        bl_mesh = bpy.data.meshes.new(mesh_name)

        # assign materials
        for bl_mat in mesh_materials:
            if bl_mat:
                bl_mesh.materials.append(bl_mat)

        # build geometry
        bl_mesh.vertices.add(len(all_verts))
        bl_mesh.loops.add(len(all_faces) * 3)
        bl_mesh.polygons.add(len(all_faces))

        # flat vertex positions
        flat_co = []
        for v in all_verts:
            flat_co.extend((v.x, v.y, v.z))
        bl_mesh.vertices.foreach_set("co", flat_co)

        # loop vertex indices
        loop_verts = []
        for f in all_faces:
            loop_verts.extend(f)
        bl_mesh.loops.foreach_set("vertex_index", loop_verts)

        # polygon loop starts and sizes
        loop_starts = [i * 3 for i in range(len(all_faces))]
        loop_totals = [3] * len(all_faces)
        bl_mesh.polygons.foreach_set("loop_start", loop_starts)
        bl_mesh.polygons.foreach_set("loop_total", loop_totals)

        # material indices
        if all_face_mat_indices:
            bl_mesh.polygons.foreach_set("material_index", all_face_mat_indices)

        # smooth shading must be enabled on every polygon or Blender ignores
        # custom split normals entirely (the pre-4.1 use_auto_smooth mechanism
        # is gone in 4.2+; per-polygon use_smooth is the replacement)
        bl_mesh.polygons.foreach_set("use_smooth", [True] * len(all_faces))

        # UV layer must be set BEFORE update/validate which may remove degenerate faces
        if all_face_uvs:
            uv_layer = bl_mesh.uv_layers.new(name="UVMap")
            uv_data = []
            for face_uvs in all_face_uvs:
                uv_data.extend(face_uvs)
            for i, uv in enumerate(uv_data):
                uv_layer.data[i].uv = uv

        # vertex color layer also before update/validate
        if all_colors:
            try:
                vcol = bl_mesh.color_attributes.new(
                    name="Color", type='BYTE_COLOR', domain='CORNER'
                )
                color_data = []
                for f in all_faces:
                    for vi in f:
                        if vi < len(all_colors):
                            color_data.append(all_colors[vi])
                        else:
                            color_data.append((1.0, 1.0, 1.0, 1.0))
                for i, c in enumerate(color_data):
                    vcol.data[i].color = c
            except Exception:
                # older Blender fallback
                try:
                    vcol = bl_mesh.vertex_colors.new(name="Color")
                    for i, f in enumerate(all_faces):
                        for j, vi in enumerate(f):
                            loop_idx = i * 3 + j
                            if vi < len(all_colors):
                                vcol.data[loop_idx].color = all_colors[vi]
                except Exception:
                    pass

        # update + validate geometry before setting custom normals
        bl_mesh.update()
        bl_mesh.validate()

        # custom split normals — must be set LAST, no update() after or Blender
        # recalculates and overwrites them
        if all_normals:
            loop_normals = []
            for f in all_faces:
                for vi in f:
                    loop_normals.append(all_normals[vi])

            try:
                bl_mesh.normals_split_custom_set(loop_normals)
            except Exception:
                try:
                    # older Blender (pre-4.1) needed use_auto_smooth=True first
                    bl_mesh.use_auto_smooth = True
                    bl_mesh.normals_split_custom_set(loop_normals)
                except Exception:
                    pass
        # intentionally no bl_mesh.update() here — it would overwrite custom normals

        if validate_meshes:
            bl_mesh.validate(verbose=True)

        # clamp
        if clamp_size > 0:
            max_dim = max(bl_mesh.dimensions) if bl_mesh.dimensions else 0
            if max_dim > clamp_size:
                scale_factor = clamp_size / max_dim
                for vert in bl_mesh.vertices:
                    vert.co *= scale_factor

        # create object
        bl_obj = bpy.data.objects.new(mesh_name, bl_mesh)
        fo2_body_coll.objects.link(bl_obj)
        created_objects.append(bl_obj)

        if use_origins:
            # vertices are in local space (mesh matrix NOT baked).
            # convert FO2 matrix to Blender space: swap rows/cols 1↔2, no sign flips.
            M = mesh_matrix  # already row-major from fo2_matrix_to_blender
            obj_mat = Matrix((
                (M[0][0], M[0][2], M[0][1], M[0][3]),
                (M[2][0], M[2][2], M[2][1], M[2][3]),
                (M[1][0], M[1][2], M[1][1], M[1][3]),
                (M[3][0], M[3][2], M[3][1], M[3][3]),
            ))
            obj_mat[0][3] *= global_scale
            obj_mat[1][3] *= global_scale
            obj_mat[2][3] *= global_scale
            bl_obj.matrix_world = obj_mat

            bl_obj.parent = root_empty
        else:
            # groups and dummies mode — parent to matching object empty
            bl_obj.parent = root_empty
            # try to find a parent object/dummy by group index or name
            if bgm_mesh.group >= 0 and bgm_mesh.group < len(parser.objects):
                parent_obj = parser.objects[bgm_mesh.group]
                if parent_obj.name1 in object_empties:
                    bl_obj.parent = object_empties[parent_obj.name1]

        # store BGM metadata
        bl_obj["bgm_flags"] = bgm_mesh.flags
        bl_obj["bgm_group"] = bgm_mesh.group
        bl_obj["bgm_name2"] = bgm_mesh.name2

        # create crash mesh if crash.dat data exists
        if crash_surface_pairs:
            crash_all_verts = []
            crash_all_normals = []
            crash_all_uvs = []
            crash_all_faces = []
            crash_all_face_mat_indices = []
            crash_mat_index_map = {}
            crash_mesh_materials = []
            crash_vert_offset = 0

            for bgm_surf, crash_surf in crash_surface_pairs:
                # validate vertex counts match
                if crash_surf.vertex_count != bgm_surf.vertex_count:
                    print(f"[crash.dat] WARNING: vertex count mismatch for {mesh_name}")
                    continue

                # extract crash vertices (crash positions/normals + UVs from buffer)
                verts = extract_crash_vertices(crash_surf, bgm_surf.flags)
                if not verts:
                    continue
                faces = extract_indices(parser, bgm_surf)
                if not faces:
                    continue

                vb = parser.vertex_buffers[bgm_surf.stream_id[0]]
                base_vertex_offset = bgm_surf.stream_offset[0] // vb.vertex_size

                # material
                mat_id = bgm_surf.material_id
                if mat_id not in crash_mat_index_map:
                    crash_mat_index_map[mat_id] = len(crash_mesh_materials)
                    crash_mesh_materials.append(blender_materials.get(mat_id))
                local_mat_idx = crash_mat_index_map[mat_id]

                # transform crash vertices (same coordinate conversion as base)
                for v in verts:
                    pos = Vector((v.x, v.y, v.z))
                    if not use_origins:
                        pos = mesh_matrix @ pos
                    pos = axis_matrix @ pos
                    pos *= global_scale
                    crash_all_verts.append(pos)

                    nrm = Vector((v.nx, v.ny, v.nz))
                    if not use_origins:
                        nrm = mesh_matrix.to_3x3() @ nrm
                    nrm = axis_matrix.to_3x3() @ nrm
                    if nrm.length > 0:
                        nrm.normalize()
                    crash_all_normals.append(nrm)

                    if v.has_uv:
                        crash_all_uvs.append((v.u, 1.0 - v.v))
                    else:
                        crash_all_uvs.append((0.0, 0.0))

                # faces (same indices as base surface)
                for i0, i1, i2 in faces:
                    fi0 = i0 - base_vertex_offset
                    fi1 = i1 - base_vertex_offset
                    fi2 = i2 - base_vertex_offset
                    if not (0 <= fi0 < len(verts) and 0 <= fi1 < len(verts) and 0 <= fi2 < len(verts)):
                        continue
                    if fi0 == fi1 or fi1 == fi2 or fi0 == fi2:
                        continue
                    crash_all_faces.append((
                        crash_vert_offset + fi0,
                        crash_vert_offset + fi1,
                        crash_vert_offset + fi2,
                    ))
                    crash_all_face_mat_indices.append(local_mat_idx)

                crash_vert_offset += len(verts)

            if crash_all_faces:
                crash_mesh_name = mesh_name + "_crash"
                bl_crash_mesh = bpy.data.meshes.new(crash_mesh_name)
                for bl_mat in crash_mesh_materials:
                    if bl_mat:
                        bl_crash_mesh.materials.append(bl_mat)

                bl_crash_mesh.vertices.add(len(crash_all_verts))
                bl_crash_mesh.loops.add(len(crash_all_faces) * 3)
                bl_crash_mesh.polygons.add(len(crash_all_faces))

                flat_co = []
                for v in crash_all_verts:
                    flat_co.extend((v.x, v.y, v.z))
                bl_crash_mesh.vertices.foreach_set("co", flat_co)

                loop_verts = []
                for f in crash_all_faces:
                    loop_verts.extend(f)
                bl_crash_mesh.loops.foreach_set("vertex_index", loop_verts)

                loop_starts = [i * 3 for i in range(len(crash_all_faces))]
                loop_totals = [3] * len(crash_all_faces)
                bl_crash_mesh.polygons.foreach_set("loop_start", loop_starts)
                bl_crash_mesh.polygons.foreach_set("loop_total", loop_totals)

                if crash_all_face_mat_indices:
                    bl_crash_mesh.polygons.foreach_set("material_index", crash_all_face_mat_indices)

                bl_crash_mesh.polygons.foreach_set("use_smooth", [True] * len(crash_all_faces))

                # UV layer before update/validate
                if crash_all_uvs:
                    uv_layer = bl_crash_mesh.uv_layers.new(name="UVMap")
                    uv_data = []
                    for f in crash_all_faces:
                        for vi in f:
                            uv_data.append(crash_all_uvs[vi])
                    for i, uv in enumerate(uv_data):
                        uv_layer.data[i].uv = uv

                # update + validate geometry, then set custom normals last (no update after)
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
                # intentionally no bl_crash_mesh.update() here

                crash_obj = bpy.data.objects.new(crash_mesh_name, bl_crash_mesh)
                fo2_crash_coll.objects.link(crash_obj)
                created_objects.append(crash_obj)

                # parent to crash root empty
                crash_obj.parent = crash_root_empty
                if use_origins:
                    # same transform as base mesh object
                    M = mesh_matrix
                    crash_mat = Matrix((
                        (M[0][0], M[0][2], M[0][1], M[0][3]),
                        (M[2][0], M[2][2], M[2][1], M[2][3]),
                        (M[1][0], M[1][2], M[1][1], M[1][3]),
                        (M[3][0], M[3][2], M[3][1], M[3][3]),
                    ))
                    crash_mat[0][3] *= global_scale
                    crash_mat[1][3] *= global_scale
                    crash_mat[2][3] *= global_scale
                    crash_obj.matrix_world = crash_mat
                crash_obj["bgm_flags"] = bgm_mesh.flags
                crash_obj["bgm_group"] = bgm_mesh.group
                crash_obj["bgm_is_crash"] = True

                print(f"[crash.dat] Created crash mesh: {crash_mesh_name} "
                      f"({len(crash_all_verts)} verts, {len(crash_all_faces)} faces)")

    # select all created objects
    bpy.ops.object.select_all(action='DESELECT')
    for obj in created_objects:
        obj.select_set(True)
    if created_objects:
        context.view_layer.objects.active = created_objects[0]

    print(f"[BGM] Import complete: {len(created_objects)} mesh objects created")
    return created_objects


# BODY.INI PARSER

def parse_body_ini(filepath: str) -> dict:
    """Parse a FlatOut 2 body.ini file.
    Returns a dict with keys:
      'full_min', 'full_max',
      'bottom_min', 'bottom_max',
      'top_min', 'top_max'
    Each value is a (x, y, z) tuple, or None if not found.
    """
    result = {
        'full_min':   None, 'full_max':   None,
        'bottom_min': None, 'bottom_max': None,
        'top_min':    None, 'top_max':    None,
    }
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
    except (OSError, IOError):
        return result

    key_map = {
        'CollisionFullMin':   'full_min',
        'CollisionFullMax':   'full_max',
        'CollisionBottomMin': 'bottom_min',
        'CollisionBottomMax': 'bottom_max',
        'CollisionTopMin':    'top_min',
        'CollisionTopMax':    'top_max',
    }

    for ini_key, dict_key in key_map.items():
        import re as _re
        m = _re.search(ini_key + r'\s*=\s*\{([^}]*)\}', text)
        if m:
            nums = _re.findall(r'[-+]?\d*\.?\d+', m.group(1))
            if len(nums) >= 3:
                result[dict_key] = (float(nums[0]), float(nums[1]), float(nums[2]))

    print(f"[body.ini] Parsed collision boxes from {filepath}")
    return result


def build_collision_boxes(context, body_data: dict, root_empty, global_scale: float):
    """Create wire-frame cube empties for each collision AABB in body.ini.
    FO2 coords (x, y, z) → Blender (x, z, y) to match the import transform."""

    # FO2 -> Blender axis swap: x stays, y<->z swap
    def fo2_to_bl(v):
        return (v[0] * global_scale, v[2] * global_scale, v[1] * global_scale)

    boxes = [
        ('fo2_collision_full',   'full_min',   'full_max',   (0.8, 0.1, 0.1, 0.5)),
        ('fo2_collision_bottom', 'bottom_min', 'bottom_max', (0.1, 0.6, 0.1, 0.5)),
        ('fo2_collision_top',    'top_min',    'top_max',    (0.1, 0.3, 0.9, 0.5)),
    ]

    coll = bpy.data.collections.get("FO2 Body Collision")
    if coll is None:
        coll = bpy.data.collections.new("FO2 Body Collision")
    if coll.name not in context.scene.collection.children:
        context.scene.collection.children.link(coll)

    for obj_name, min_key, max_key, color in boxes:
        v_min = body_data.get(min_key)
        v_max = body_data.get(max_key)
        if v_min is None or v_max is None:
            continue

        bl_min = fo2_to_bl(v_min)
        bl_max = fo2_to_bl(v_max)

        # centre and dimensions in Blender space
        cx = (bl_min[0] + bl_max[0]) * 0.5
        cy = (bl_min[1] + bl_max[1]) * 0.5
        cz = (bl_min[2] + bl_max[2]) * 0.5
        sx = abs(bl_max[0] - bl_min[0])
        sy = abs(bl_max[1] - bl_min[1])
        sz = abs(bl_max[2] - bl_min[2])

        empty = bpy.data.objects.new(obj_name, None)
        empty.empty_display_type = 'CUBE'
        # blender cube empty has half-size = display_size on each axis so we set display_size = 0.5 and bake the real extents into scale
        empty.empty_display_size = 0.5
        empty.scale = (sx, sy, sz)
        coll.objects.link(empty)
        empty.parent = root_empty
        empty.location = (cx, cy, cz)

        # color the empty for easy identification in the viewport
        empty.color = color

        # store raw values as custom properties
        empty["fo2_min"] = list(v_min)
        empty["fo2_max"] = list(v_max)

        print(f"[body.ini] Created {obj_name}: min={v_min} max={v_max}")

    return


# CAMERA.INI PARSER

@dataclass
class CameraEntry:
    index: int = 0
    animation_type: int = 1
    position_type: int = 2
    target_type: int = -1
    zoom_type: int = 1
    tracker_type: int = 1
    near_clipping: float = 0.5
    far_clipping: float = 1000.0
    min_display_time: float = 4.0
    max_display_time: float = 9.0
    lod_level: int = 1
    position_offset: tuple = (0.0, 0.0, 0.0)
    target_offset: tuple = (0.0, 0.0, 0.0)
    fov: float = 90.0
    tracker_stiffness: tuple = (0.0, 0.0, 0.0)
    tracker_min_ground: float = 1.0
    tracker_clamp_ground: float = 0.3


def parse_camera_ini(filepath: str) -> list:
    """Parse a FlatOut 2 camera.ini. Returns list[CameraEntry]."""
    import re as _re
    entries = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
    except (OSError, IOError):
        return entries

    # strip comments
    text = _re.sub(r'--[^\n]*', '', text)

    def _f(s):
        try: return float(s.strip())
        except ValueError: return 0.0

    def _i(s):
        try: return int(s.strip())
        except ValueError: return 0

    def _vec3(s):
        nums = _re.findall(r'[-+]?\d*\.?\d+', s)
        return tuple(_f(n) for n in nums[:3]) if len(nums) >= 3 else (0.0, 0.0, 0.0)

    def extract_top_level_blocks(src):
        """Yield (index, block_content) for each top-level [N]= { ... } in src,
        using brace counting so nested blocks are not mistaken for new entries."""
        pattern = _re.compile(r'\[(\d+)\]\s*=\s*\{')
        pos = 0
        while pos < len(src):
            m = pattern.search(src, pos)
            if not m:
                break
            idx = int(m.group(1))
            # walk forward counting braces to find matching closing brace
            depth = 0
            start = m.end() - 1   # points at the opening '{'
            i = start
            while i < len(src):
                if src[i] == '{':
                    depth += 1
                elif src[i] == '}':
                    depth -= 1
                    if depth == 0:
                        yield idx, src[start + 1 : i]
                        pos = i + 1
                        break
                i += 1
            else:
                break

    def first_vec3_in(block, key):
        m = _re.search(key + r'\s*=\s*\{([^}]*)\}', block)
        return _vec3(m.group(1)) if m else None

    def first_float_in(block, key):
        m = _re.search(key + r'\s*=\s*([-+]?\d*\.?\d+)', block)
        return _f(m.group(1)) if m else None

    def first_int_in(block, key):
        m = _re.search(key + r'\s*=\s*([-+]?\d+)', block)
        return _i(m.group(1)) if m else None

    # find the outer Cameras = { ... } block first
    outer = _re.search(r'Cameras\s*=\s*\{', text)
    if not outer:
        return entries
    # extract content between the outer braces using depth counting
    depth = 0
    cam_body = ""
    for i in range(outer.end() - 1, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                cam_body = text[outer.end() : i]
                break

    # now iterate only the direct children of Cameras= {}
    for idx, block in extract_top_level_blocks(cam_body):
        cam = CameraEntry(index=idx)

        for attr, key in [
            ('animation_type', 'AnimationType'),
            ('position_type',  'PositionType'),
            ('target_type',    'TargetType'),
            ('zoom_type',      'ZoomType'),
            ('tracker_type',   'TrackerType'),
            ('lod_level',      'LodLevel'),
        ]:
            v = first_int_in(block, key)
            if v is not None:
                setattr(cam, attr, v)

        for attr, key in [
            ('near_clipping',    'NearClipping'),
            ('far_clipping',     'FarClipping'),
            ('min_display_time', 'MinDisplayTime'),
            ('max_display_time', 'MaxDisplayTime'),
        ]:
            v = first_float_in(block, key)
            if v is not None:
                setattr(cam, attr, v)

        # PositionFrames: find [1]= block inside, then Offset
        m = _re.search(r'PositionFrames\s*=\s*\{', block)
        if m:
            for _, fb in extract_top_level_blocks(block[m.end() - 1:]):
                v = first_vec3_in(fb, 'Offset')
                if v:
                    cam.position_offset = v
                break  # only first frame

        # TargetFrames
        m = _re.search(r'TargetFrames\s*=\s*\{', block)
        if m:
            for _, fb in extract_top_level_blocks(block[m.end() - 1:]):
                v = first_vec3_in(fb, 'Offset')
                if v:
                    cam.target_offset = v
                break

        # ZoomFrames FOV
        m = _re.search(r'ZoomFrames\s*=\s*\{', block)
        if m:
            for _, fb in extract_top_level_blocks(block[m.end() - 1:]):
                v = first_float_in(fb, 'FOV')
                if v is not None:
                    cam.fov = v
                break

        # TrackerData — manually extract the block content using brace counting
        # (can't use [^}] because Stiffness contains inner braces)
        m = _re.search(r'TrackerData\s*=\s*\{', block)
        if m:
            depth = 0
            start = m.end() - 1
            i = start
            while i < len(block):
                if block[i] == '{':
                    depth += 1
                elif block[i] == '}':
                    depth -= 1
                    if depth == 0:
                        td = block[start + 1 : i]
                        break
                i += 1
            else:
                td = ""
            if td:
                sv = first_vec3_in(td, 'Stiffness')
                if sv:
                    cam.tracker_stiffness = sv
                gv = first_float_in(td, 'MinGround')
                if gv is not None:
                    cam.tracker_min_ground = gv
                cv = first_float_in(td, 'ClampGround')
                if cv is not None:
                    cam.tracker_clamp_ground = cv

        entries.append(cam)

    print(f"[camera.ini] Parsed {len(entries)} cameras from {filepath}")
    return entries


def build_camera_objects(context, cam_entries: list, root_empty, global_scale: float):
    """Create Blender camera objects for each CameraEntry.
    FO2 offset (x, y, z) → Blender (x, z, y) — same y↔z swap as rest of import.
    Cameras with TargetType != -1 get a Track To constraint aimed at a target empty."""

    coll = bpy.data.collections.get("FO2 Body Cameras")
    if coll is None:
        coll = bpy.data.collections.new("FO2 Body Cameras")
    if coll.name not in context.scene.collection.children:
        context.scene.collection.children.link(coll)

    def fo2_to_bl(v, scale=1.0):
        return (-v[0] * scale, v[2] * scale, v[1] * scale)

    for cam in cam_entries:
        cam_name = f"fo2_camera_{cam.index}"

        # camera data
        cam_data = bpy.data.cameras.new(cam_name)
        cam_data.clip_start = cam.near_clipping
        cam_data.clip_end   = cam.far_clipping
        # convert FO2 horizontal FOV (degrees) to Blender lens angle
        import math
        cam_data.angle = math.radians(cam.fov)
        cam_data.type  = 'PERSP'

        # camera object
        cam_obj = bpy.data.objects.new(cam_name, cam_data)
        coll.objects.link(cam_obj)
        cam_obj.parent = root_empty
        pos_bl = fo2_to_bl(cam.position_offset, global_scale)
        cam_obj.location = pos_bl

        # cameras with no target face forward (+Y). Blender cameras point -Z by default, so rotate 90° around X to align with +Y
        # cameras with a target get their orientation from the Track To constraint
        if cam.target_type == -1:
            import math
            cam_obj.rotation_euler = (math.pi / 2, 0.0, 0.0)

        # custom properties
        cam_obj["fo2_cam_index"]          = cam.index
        cam_obj["fo2_animation_type"]     = cam.animation_type
        cam_obj["fo2_position_type"]      = cam.position_type
        cam_obj["fo2_target_type"]        = cam.target_type
        cam_obj["fo2_zoom_type"]          = cam.zoom_type
        cam_obj["fo2_tracker_type"]       = cam.tracker_type
        cam_obj["fo2_lod_level"]          = cam.lod_level
        cam_obj["fo2_min_display_time"]   = cam.min_display_time
        cam_obj["fo2_max_display_time"]   = cam.max_display_time
        cam_obj["fo2_fov"]                = cam.fov
        cam_obj["fo2_position_offset"]    = list(cam.position_offset)
        if cam.target_type != -1:
            cam_obj["fo2_target_offset"]  = list(cam.target_offset)
        if cam.tracker_type == 2:
            cam_obj["fo2_tracker_stiffness"]    = list(cam.tracker_stiffness)
            cam_obj["fo2_tracker_min_ground"]   = cam.tracker_min_ground
            cam_obj["fo2_tracker_clamp_ground"] = cam.tracker_clamp_ground

        # target empty + Track To constraint
        if cam.target_type != -1:
            tgt_name = f"fo2_camera_{cam.index}_target"
            tgt_obj  = bpy.data.objects.new(tgt_name, None)
            tgt_obj.empty_display_type = 'SPHERE'
            tgt_obj.empty_display_size = 0.05
            coll.objects.link(tgt_obj)
            tgt_obj.parent   = root_empty
            tgt_obj.location = fo2_to_bl(cam.target_offset, global_scale)

            con = cam_obj.constraints.new('TRACK_TO')
            con.target    = tgt_obj
            con.track_axis = 'TRACK_NEGATIVE_Z'
            con.up_axis    = 'UP_Y'

    print(f"[camera.ini] Created {len(cam_entries)} cameras under fo2_body_cameras")



# BLENDER OPERATOR

class ImportBGM(bpy.types.Operator, ImportHelper):
    """Import a FlatOut BGM car model file"""
    bl_idname = "import_scene.bgm"
    bl_label = "Import FlatOut BGM"
    bl_options = {'REGISTER', 'UNDO', 'PRESET'}

    filename_ext = ".bgm"
    filter_glob: StringProperty(default="*.bgm", options={'HIDDEN'})

    # transform
    global_scale: FloatProperty(
        name="Scale",
        min=0.001, max=1000.0,
        default=1.0,
        description="Global import scale",
    )
    clamp_size: FloatProperty(
        name="Clamp Bounding Box",
        min=0.0, max=10000.0,
        default=0.0,
        description="Clamp object dimensions to this size (0 = disabled)",
    )

    # mesh

    validate_meshes: BoolProperty(
        name="Validate Meshes",
        default=False,
        description="Run Blender mesh validation after import (slower but catches errors)",
    )

    # textures & materials

    shared_texture_dir: StringProperty(
        name="Shared Textures",
        subtype='DIR_PATH',
        default="",
        description="Folder with shared textures (common.dds, windows.dds, etc.) "
                    "used when textures aren't found next to the BGM file or in "
                    "the auto-detected ../shared/ directory",
    )
    convert_dds: BoolProperty(
        name="Convert DDS to TGA",
        default=True,
        description="When a texture is only available as DDS, convert it to TGA "
                    "and save it next to the BGM file. The material will reference "
                    "the converted TGA",
    )
    import_normal_maps: BoolProperty(
        name="Import Normal Maps (FOUC)",
        default=False,
        description="For FlatOut UC models, detect and wire <texture>_normal sidecar "
                    "textures into the material's Normal input",
    )
    import_specular_maps: BoolProperty(
        name="Import Specular Maps (FOUC)",
        default=False,
        description="For FlatOut UC models, detect and wire <texture>_specular sidecar "
                    "textures into the material's Specular input. Disable for a cleaner "
                    "viewport look",
    )
    crash_dat_path: StringProperty(
        name="Crash Data (.dat)",
        subtype='FILE_PATH',
        default="",
        description="Path to crash.dat file for importing deformed crash meshes. "
                    "Leave empty to auto-detect (<name>-crash.dat or crash.dat)",
    )
    use_alpha: BoolProperty(
        name="Import Alpha",
        default=True,
        description="Link DDS alpha channel to material transparency. "
                    "Disable for fully opaque import",
    )
    use_backface_culling: BoolProperty(
        name="Backface Culling",
        default=True,
        description="Enable backface culling in the viewport for all imported materials",
    )
    alpha_mode: EnumProperty(
        name="Alpha Mode",
        items=[
            ('BLEND', "Blended", "True transparency with alpha compositing. "
                                 "Smooth but may have sorting artifacts"),
            ('HASHED', "Dithered", "Noise-based transparency. "
                                   "No sorting issues but grainy look"),
        ],
        default='BLEND',
        description="How transparent surfaces are rendered in EEVEE",
    )
    transparency_overlap: BoolProperty(
        name="Transparency Overlap",
        default=False,
        description="Render backfaces of transparent surfaces. "
                    "Disable to avoid doubling artifacts on windows",
    )

    # LOD

    max_lod: IntProperty(
        name="Max LOD Level",
        min=0, max=10,
        default=0,
        description="Maximum LOD level to import (0 = highest detail only)",
    )

    # collision

    import_body: BoolProperty(
        name="Import Body",
        default=True,
        description="Import the car body meshes (FO2 Body collection)",
    )
    import_crash: BoolProperty(
        name="Import Crash",
        default=True,
        description="Import crash deform meshes from crash.dat (FO2 Body Crash collection)",
    )
    import_dummies: BoolProperty(
        name="Import Dummies",
        default=True,
        description="Import dummy/object empties (FO2 Body Dummies collection)",
    )
    import_body_ini: BoolProperty(
        name="Import Collision Boxes (body.ini)",
        default=True,
        description="Parse body.ini (auto-detected next to the BGM file) and "
                    "create wire-frame cube empties for the full, bottom and top "
                    "collision bounding boxes in a FO2 Body Collision collection",
    )
    import_camera_ini: BoolProperty(
        name="Import Cameras (camera.ini)",
        default=True,
        description="Parse camera.ini (auto-detected next to the BGM file) and "
                    "create Blender camera objects in a FO2 Body Cameras collection. "
                    "Cameras with a target get a Track To constraint",
    )

    def draw(self, context):
        layout = self.layout

        # transform
        box = layout.box()
        box.label(text="Transform", icon='ORIENTATION_GLOBAL')
        box.prop(self, "global_scale")
        box.prop(self, "clamp_size")

        # mesh
        box = layout.box()
        box.label(text="Mesh", icon='MESH_DATA')
        box.prop(self, "validate_meshes")
        box.prop(self, "max_lod")

        # textures
        box = layout.box()
        box.label(text="Textures & Materials", icon='MATERIAL')
        box.prop(self, "shared_texture_dir")
        box.prop(self, "crash_dat_path")
        box.prop(self, "convert_dds")
        box.prop(self, "use_alpha")
        box.prop(self, "use_backface_culling")
        row = box.row()
        row.enabled = self.use_alpha
        row.prop(self, "alpha_mode")
        row = box.row()
        row.enabled = (self.use_alpha and self.alpha_mode == 'BLEND')
        row.prop(self, "transparency_overlap")

        # bgm data
        box = layout.box()
        box.label(text="BGM Data", icon='MESH_DATA')
        box.prop(self, "import_body")
        box.prop(self, "import_crash")
        box.prop(self, "import_dummies")
        box.prop(self, "import_body_ini")
        box.prop(self, "import_camera_ini")

        # FOUC debug
        box = layout.box()
        box.label(text="FOUC Debug", icon='TOOL_SETTINGS')
        box.prop(self, "import_normal_maps")
        box.prop(self, "import_specular_maps")

    def execute(self, context):
        filepath = self.filepath

        # parse BGM
        parser = BGMParser(filepath)
        if not parser.parse():
            self.report({'ERROR'}, f"Failed to parse BGM file: {filepath}")
            return {'CANCELLED'}

        options = {
            'shared_texture_dir': bpy.path.abspath(self.shared_texture_dir) if self.shared_texture_dir else "",
            'crash_dat_path': bpy.path.abspath(self.crash_dat_path) if self.crash_dat_path else "",
            'use_alpha': self.use_alpha,
            'alpha_mode': self.alpha_mode,
            'transparency_overlap': self.transparency_overlap,
            'max_lod': self.max_lod,
            'global_scale': self.global_scale,
            'clamp_size': self.clamp_size,
            'validate_meshes': self.validate_meshes,
            'convert_dds': self.convert_dds,
            'use_backface_culling': self.use_backface_culling,
            'import_normal_maps': self.import_normal_maps,
            'import_specular_maps': self.import_specular_maps,
            'import_body': self.import_body,
            'import_crash': self.import_crash,
            'import_dummies': self.import_dummies,
        }

        objects = build_blender_meshes(context, parser, options)

        if not objects:
            self.report({'WARNING'}, "No meshes were imported")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Imported {len(objects)} objects from {os.path.basename(filepath)}")

        # Set scene game mode to match the imported file
        try:
            if parser.is_fouc:
                context.scene.fo2_game_mode = 'FOUC'
            elif parser.version < 0x20000:
                context.scene.fo2_game_mode = 'FO1'
            else:
                context.scene.fo2_game_mode = 'FO2'
        except Exception:
            pass

        # body.ini
        if self.import_body_ini:
            bgm_dir  = os.path.dirname(filepath)
            ini_path = os.path.join(bgm_dir, "body.ini")
            if os.path.isfile(ini_path):
                body_data = parse_body_ini(ini_path)
                root_empty = context.scene.objects.get("fo2_body")
                if root_empty:
                    build_collision_boxes(context, body_data,
                                          root_empty, self.global_scale)
                    self.report({'INFO'}, "Imported collision boxes from body.ini")
            else:
                print(f"[BGM Import] body.ini not found at {ini_path}, skipping")

        # camera.ini
        if self.import_camera_ini:
            bgm_dir  = os.path.dirname(filepath)
            ini_path = os.path.join(bgm_dir, "camera.ini")
            if os.path.isfile(ini_path):
                cam_entries = parse_camera_ini(ini_path)
                if cam_entries:
                    root_empty = context.scene.objects.get("fo2_body")
                    if root_empty:
                        build_camera_objects(context, cam_entries,
                                             root_empty, self.global_scale)
                        self.report({'INFO'}, f"Imported {len(cam_entries)} cameras from camera.ini")
            else:
                print(f"[BGM Import] camera.ini not found at {ini_path}, skipping")

        return {'FINISHED'}


# SHADER ID PANEL

FO2_SHADER_NAMES = {
    0:  "Static Prelit",
    1:  "Terrain",
    2:  "Terrain Specular",
    3:  "Dynamic Diffuse",
    4:  "Dynamic Specular",
    5:  "Car Body",
    6:  "Car Window",
    7:  "Car Diffuse",
    8:  "Car Metal",
    9:  "Car Tire",
    10: "Car Lights",
    11: "Car Shear",
    12: "Car Scale",
    13: "Shadow Project",
    14: "Car Lights Unlit",
    15: "Default",
    16: "Vertex Color",
    17: "Shadow Sampler",
    18: "Grass",
    19: "Tree Trunk",
    20: "Tree Branch",
    21: "Tree Leaf",
    22: "Particle",
    23: "Sunflare",
    24: "Intensitymap",
    25: "Water",
    26: "Skinning",
    27: "Tree LOD (Default)",
    28: "Dummy (PS2 Streak)",
    29: "Clouds (UV Scroll)",
    30: "Car Body LOD",
    31: "Vertex Color Static",
    32: "Car Window Damaged",
    33: "Skin Shadow",
    34: "Reflecting Window (Static)",
    35: "Reflecting Window (Dynamic)",
    36: "Deprecated Static Window",
    37: "Skybox",
    38: "Ghost Body",
    39: "Static Nonlit",
    40: "Dynamic Nonlit",
    41: "Racemap",
}

FOUC_SHADER_NAMES = {
    0:  "Static Prelit",
    1:  "Terrain",
    2:  "Terrain Specular",
    3:  "Dynamic Diffuse",
    4:  "Dynamic Specular",
    5:  "Car Body",
    6:  "Car Window",
    7:  "Car Diffuse",
    8:  "Car Metal",
    9:  "Car Tire Rim",
    10: "Car Lights",
    11: "Car Shear",
    12: "Car Scale",
    13: "Shadow Project",
    14: "Car Lights Unlit",
    15: "Default",
    16: "Vertex Color",
    17: "Shadow Sampler",
    18: "Grass",
    19: "Tree Trunk",
    20: "Tree Branch",
    21: "Tree Leaf",
    22: "Particle",
    23: "Sunflare",
    24: "Intensitymap",
    25: "Water",
    26: "Skinning",
    27: "Tree LOD (Default)",
    28: "Deprecated (PS2 Streak)",
    29: "Clouds (UV Scroll)",
    30: "Car Body LOD",
    31: "Deprecated Vertex Color Static",
    32: "Car Window Damaged",
    33: "Skin Shadow (Deprecated)",
    34: "Reflecting Window (Static)",
    35: "Reflecting Window (Dynamic)",
    36: "Deprecated Static Window",
    37: "Skybox",
    38: "Horizon",
    39: "Ghost Body",
    40: "Static Nonlit",
    41: "Dynamic Nonlit",
    42: "Skid Marks",
    43: "Car Interior",
    44: "Car Tire",
    45: "Puddle",
    46: "Ambient Shadow",
    47: "Local Water",
    48: "Static Specular/Hilight",
    49: "Lightmapped Planar Reflection",
    50: "Racemap",
    51: "HDR Default (Runtime)",
    52: "Ambient Particle",
    53: "Videoscreen (Dynamic)",
    54: "Videoscreen (Static)",
}

FO2_SHADER_ITEMS = [
    (str(k), f"{k} \u2013 {v}", "")
    for k, v in sorted(FO2_SHADER_NAMES.items())
]

FOUC_SHADER_ITEMS = [
    (str(k), f"{k} \u2013 {v}", "")
    for k, v in sorted(FOUC_SHADER_NAMES.items())
]


# shaders that explicitly force alpha on or off regardless of material name
# none = leave alpha untouched when this shader is selected
FO2_SHADER_FORCED_ALPHA = {
    6:  1,    # car window — always alpha
    9:  1,    # car tire/rim — alpha=1 (rim rule)
    10: 1,    # car lights — always alpha
    12: 0,    # car scale — FORCENOALPHA (scaleshock/shearhock)
    14: 1,    # car lights unlit — same family as lights
    32: 1,    # car window damaged — same family as window
    34: 1,    # reflecting window (static)
    35: 1,    # reflecting window (dynamic)
    36: 1,    # deprecated static window
}


def _shader_update(self, context):
    """Write enum selection back to bgm_shader_id.
    Only update alpha when the shader explicitly forces a value."""
    sid = int(self.fo2_shader_id)
    self["bgm_shader_id"] = sid
    forced = FO2_SHADER_FORCED_ALPHA.get(sid)
    if forced is not None:
        self["bgm_alpha"] = forced


class FO2_OT_ToggleMatProp(bpy.types.Operator):
    """Toggle a 0/1 integer custom property on the active material"""
    bl_idname  = "fo2.toggle_mat_prop"
    bl_label   = "Toggle FO2 Material Property"
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    prop_name: bpy.props.StringProperty()

    def execute(self, context):
        mat = context.material
        if mat is None:
            return {'CANCELLED'}
        mat[self.prop_name] = 0 if mat.get(self.prop_name, 0) else 1
        return {'FINISHED'}


class FO2_OT_EditMatInt(bpy.types.Operator):
    """Edit an integer custom property on the active material"""
    bl_idname  = "fo2.edit_mat_int"
    bl_label   = "Edit FO2 Material Int Property"
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    prop_name: bpy.props.StringProperty()
    value: bpy.props.IntProperty(name="Value")

    def invoke(self, context, event):
        mat = context.material
        if mat is None:
            return {'CANCELLED'}
        self.value = int(mat.get(self.prop_name, 0))
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        self.layout.prop(self, "value", text=self.prop_name)

    def execute(self, context):
        mat = context.material
        if mat is None:
            return {'CANCELLED'}
        mat[self.prop_name] = self.value
        return {'FINISHED'}


def _texture_update(self, context):
    self["bgm_texture"]   = self.fo2_texture
    self["bgm_texture_0"] = self.fo2_texture


def _get_shader_items(self, context):
    """Dynamic shader items based on game mode scene property."""
    scene = context.scene if context else None
    if scene and getattr(scene, "fo2_game_mode", "FO2") == "FOUC":
        return FOUC_SHADER_ITEMS
    return FO2_SHADER_ITEMS


class FO2_PT_ShaderPanel(bpy.types.Panel):
    """FlatOut shader ID panel in Material Properties"""
    bl_label       = "FlatOut Shader"
    bl_idname      = "MATERIAL_PT_fo2_shader"
    bl_space_type  = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context     = "material"

    @classmethod
    def poll(cls, context):
        return context.material is not None

    def draw(self, context):
        mat    = context.material
        layout = self.layout

        # Game mode toggle at top of panel
        scene = context.scene
        row = layout.row(align=True)
        row.prop(scene, "fo2_game_mode", expand=True)

        layout.prop(mat, "fo2_shader_id", text="Shader")
        layout.prop(mat, "fo2_texture", text="BGM Texture")

        layout.separator()

        # alpha stored as int 0/1, show as checkbox
        row = layout.row()
        row.label(text="Alpha:")
        alpha_val = bool(mat.get("bgm_alpha", 0))
        op = row.operator("fo2.toggle_mat_prop", text="", icon='CHECKBOX_HLT' if alpha_val else 'CHECKBOX_DEHLT', emboss=False)
        op.prop_name = "bgm_alpha"

        # Use Colormap
        row = layout.row()
        row.label(text="Use Colormap:")
        cm_val = bool(mat.get("bgm_use_colormap", 0))
        op = row.operator("fo2.toggle_mat_prop", text="", icon='CHECKBOX_HLT' if cm_val else 'CHECKBOX_DEHLT', emboss=False)
        op.prop_name = "bgm_use_colormap"

        layout.separator()

        # v92, v74, v102 — integer fields
        col = layout.column(align=True)
        col.label(text="v92:")
        col.operator("fo2.edit_mat_int", text=str(mat.get("bgm_v92", 0))).prop_name = "bgm_v92"
        col.label(text="v74:")
        col.operator("fo2.edit_mat_int", text=str(mat.get("bgm_v74", 0))).prop_name = "bgm_v74"
        col.label(text="v102:")
        col.operator("fo2.edit_mat_int", text=str(mat.get("bgm_v102", 0))).prop_name = "bgm_v102"


# REGISTRATION

def menu_func_import(self, context):
    self.layout.operator(ImportBGM.bl_idname, text="FlatOut Car BGM (.bgm)")


def register():
    bpy.types.Scene.fo2_game_mode = bpy.props.EnumProperty(
        name="Game",
        items=[('FO1', "FlatOut 1", ""), ('FO2', "FlatOut 2", ""), ('FOUC', "FlatOut UC", "")],
        default='FO2',
    )
    bpy.types.Material.fo2_shader_id = bpy.props.EnumProperty(
        name="Shader",
        items=_get_shader_items,
        default=None,
        update=_shader_update,
    )
    bpy.types.Material.fo2_texture = bpy.props.StringProperty(
        name="FlatOut Texture",
        default="",
        update=_texture_update,
    )
    bpy.utils.register_class(FO2_OT_ToggleMatProp)
    bpy.utils.register_class(FO2_OT_EditMatInt)
    bpy.utils.register_class(FO2_PT_ShaderPanel)
    bpy.utils.register_class(ImportBGM)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.utils.unregister_class(ImportBGM)
    bpy.utils.unregister_class(FO2_PT_ShaderPanel)
    bpy.utils.unregister_class(FO2_OT_EditMatInt)
    bpy.utils.unregister_class(FO2_OT_ToggleMatProp)
    del bpy.types.Material.fo2_texture
    del bpy.types.Material.fo2_shader_id
    del bpy.types.Scene.fo2_game_mode
    del bpy.types.Material.fo2_texture
    del bpy.types.Material.fo2_shader_id


if __name__ == "__main__":
    register()
