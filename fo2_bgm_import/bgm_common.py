"""
bgm_common.py — Shared data structures, parsers and Blender helpers
used by every platform import path.
Part of FlatOut Blender Tools — https://github.com/RavenDS/flatout-blender-tools

Covers:
  • Dataclasses:  BGMMaterial, VertexBuffer, IndexBuffer, Surface, Model,
                  BGMMesh, BGMObject, CrashWeight, CrashSurface, CrashNode,
                  ParsedVertex
  • BinaryReader
  • BGMParser  (PC / FlatOut 1 / FOUC)
  • parse_crash_dat  (PC / FOUC crash.dat)
  • Vertex/index extraction helpers
  • Matrix / axis helpers
  • Texture resolution + Blender material creation
  • extract_crash_vertices  (PC / FOUC)
  • build_blender_meshes    (PC / FOUC)
"""

import bpy
import struct
import os
import math
from mathutils import Matrix, Vector
from dataclasses import dataclass, field
from typing import Optional
from . import dds2tga as _dds2tga
from . import dds_normal as _dds_normal
from . import tm2tga as _tm2tga

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
                      use_normal_converter: bool = False,
                      native_ext: str = "") -> str:
    """Resolve a texture filename to a full path on disk.

    Search order per directory: TGA first, then native (tm2/tex if given), then DDS.
    Directories searched: bgm_dir -> auto_shared_dir -> shared_dir.

    native_ext: platform-specific fallback extension before DDS:
      '.tm2' for PS2,  '.tex' for PSP.  When found, tm2tga.convert_to_tga()
      produces a TGA beside the BGM file and that TGA path is returned.

    If convert_dds=True and the texture is only found as DDS, it is
    converted to TGA (placed in bgm_dir) and the TGA path is returned.
    Does case-insensitive matching on all platforms."""
    if not tex_name:
        return ""

    base = os.path.splitext(tex_name)[0]
    tga_name    = base + '.tga'
    dds_name    = base + '.dds'
    native_name = (base + native_ext) if native_ext else ""

    # build ordered search list, avoid duplicates
    search_dirs = [bgm_dir]
    if auto_shared_dir and auto_shared_dir != bgm_dir:
        search_dirs.append(auto_shared_dir)
    if shared_dir and shared_dir not in search_dirs:
        search_dirs.append(shared_dir)

    found_native = ""  # best native (.tm2/.tex) path found
    found_dds    = ""  # best DDS path found

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

        # remember the first native (.tm2/.tex) found
        if native_name and not found_native:
            nat_match = entries_lower.get(native_name.lower())
            if nat_match:
                found_native = os.path.join(search_dir, nat_match)

        # remember the first DDS found for fallback
        if not found_dds:
            dds_match = entries_lower.get(dds_name.lower())
            if dds_match:
                found_dds = os.path.join(search_dir, dds_match)

    # no TGA found — try native format (.tm2 / .tex) before DDS
    if found_native:
        out_tga = os.path.join(bgm_dir, tga_name)
        try:
            _tm2tga.convert_to_tga(found_native, out_tga)
            print(f"[BGM Import] Converted: {os.path.basename(found_native)} → {tga_name}")
            return out_tga
        except Exception as exc:
            print(f"[BGM Import] {native_ext.upper()[1:]}→TGA conversion failed for "
                  f"{os.path.basename(found_native)}: {exc}")
        # fall through to DDS if conversion failed

    # no TGA or native found, deal with DDS
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
                            import_specular_maps: bool = False,
                            native_tex_ext: str = "") -> bpy.types.Material:
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
                                      auto_shared_dir, convert_dds,
                                      native_ext=native_tex_ext)

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


