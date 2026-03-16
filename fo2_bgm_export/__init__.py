bl_info = {
    "name":        "FlatOut 2 BGM Export (Car)",
    "author":      "ravenDS",
    "version":     (1, 4, 2),
    "blender":     (3, 6, 0),
    "location":    "File > Export > FlatOut 2 BGM Car (.bgm)",
    "description": "Export FlatOut 2 car model (BGM) files. Based on reverse-egineering work by Chloe (FlatOutW32BGMTool)",
    "category":    "Import-Export",
    "doc_url":     "https://github.com/RavenDS",
    "tracker_url": "https://github.com/RavenDS/flatout-blender-tools/issues",
}

import bpy
import bmesh
import os
import struct
import re
import math
from bpy.props import (StringProperty, FloatProperty, BoolProperty,
                       EnumProperty)
from bpy_extras.io_utils import ExportHelper
from mathutils import Matrix, Vector
from . import tga2dds as _tga2dds


# CONSTANTS

BGM_VERSION_FO2 = 0x20000

# identifiers
MATC_ID = 0x4354414D  # "MATC"
BMOD_ID = 0x444F4D42  # "BMOD"
MESH_ID = 0x4853454D  # "MESH"
OBJC_ID = 0x434A424F  # "OBJC"

# vertex stream flags
VERTEX_POSITION = 0x002
VERTEX_NORMAL   = 0x010
VERTEX_COLOR    = 0x040
VERTEX_UV       = 0x100

# shader IDs (FO2)
SHADER_STATIC_PRELIT    = 0
SHADER_DYNAMIC_DIFFUSE  = 3
SHADER_CAR_BODY         = 5
SHADER_CAR_WINDOW       = 6
SHADER_CAR_DIFFUSE      = 7
SHADER_CAR_METAL        = 8
SHADER_CAR_TIRE         = 9
SHADER_CAR_LIGHTS       = 10
SHADER_CAR_SHEAR        = 11
SHADER_CAR_SCALE        = 12
SHADER_SHADOW_PROJECT   = 13
SHADER_SKINNING         = 26


# MATERIAL PRIORITY TABLE  (from FlatOutW32BGMTool_gcp.toml)

MATERIAL_PRIORITIES = {
    "shearshock": 1, "shearspring": 2,
}
# Populate light priorities
for _side in ("", "_l", "_r", "_l_2", "_r_2", "_2"):
    for _type in ("light_brake", "light_reverse", "light_front"):
        name_b = f"{_type}{_side}_b"
        name_b2 = f"{_type}{_side}_b_2"
        name_2b = f"{_type}{_side[:-2] + '_2' if _side.endswith('_2') else _side}_b"
        MATERIAL_PRIORITIES[f"{_type}_b{_side}"] = 1
        MATERIAL_PRIORITIES[f"{_type}{_side}_b"] = 1
        MATERIAL_PRIORITIES[f"{_type}_b{_side}_2"] = 1
        MATERIAL_PRIORITIES[f"{_type}{_side}_b_2"] = 1
        # non-b variants at priority 2
        MATERIAL_PRIORITIES[f"{_type}{_side}"] = 2


def get_material_priority(name: str) -> int:
    """Get sort priority for a material name (higher = drawn later)."""
    return MATERIAL_PRIORITIES.get(name.lower(), 0)


# SHADER CONFIG  (from shaders.txt / FixupFBXCarMaterial)

def get_shader_for_material(mat_name: str, tex_name: str) -> tuple:
    """Determine shader_id, alpha, v92, texture_override from material name.

    Returns (shader_id, alpha, v92, texture_name_override_or_None).
    """
    name = mat_name.lower()
    shader = SHADER_CAR_METAL  # default
    alpha = 0
    v92 = 0
    tex_override = None

    # name-prefix rules (order matters, first match wins)
    if name.startswith("shadow") or name.endswith("shadow"):
        shader = SHADER_SHADOW_PROJECT
    elif name.startswith("body"):
        shader = SHADER_CAR_BODY
        tex_override = "skin1.tga"
    elif name.startswith("interior"):
        shader = SHADER_CAR_DIFFUSE
    elif name.startswith("grille"):
        shader = SHADER_CAR_DIFFUSE
        alpha = 1
    elif name.startswith("window"):
        shader = SHADER_CAR_WINDOW
    elif name.startswith("shear"):
        shader = SHADER_CAR_SHEAR
    elif name.startswith("scaleshock") or name.startswith("shearhock"):
        shader = SHADER_CAR_SCALE
        alpha = 0  # forced no alpha (FORCENOALPHA)
    elif name.startswith("shock") or name.startswith("spring") or name.startswith("scale"):
        shader = SHADER_CAR_SCALE
    elif name.startswith("tire"):
        shader = SHADER_CAR_DIFFUSE  # TIRE -> shader 7 in FO2
    elif name.startswith("rim"):
        shader = SHADER_CAR_TIRE  # RIM -> shader 9
        alpha = 1
    elif name.startswith("light"):
        shader = SHADER_CAR_LIGHTS
        v92 = 2
    elif name.startswith("terrain") or name.startswith("groundplane"):
        shader = SHADER_CAR_DIFFUSE
        alpha = 1
    elif name.startswith("male") or name.startswith("female"):
        shader = SHADER_SKINNING

    # texture-based rules
    tex_lower = tex_name.lower() if tex_name else ""
    if tex_lower in ("lights.tga", "windows.tga", "shock.tga"):
        alpha = 1

    # name-suffix overrides (highest priority)
    if name.endswith("_alpha"):
        alpha = 1
    if name.endswith("_noalpha"):
        alpha = 0

    return shader, alpha, v92, tex_override


# VERTEX FORMAT PER SHADER

def get_vertex_format(shader_id: int) -> tuple:
    """Return (flags, vertex_size_bytes) for a given shader.

    FO2 BGM vertex layouts:
      body/skinning:  pos(12) + normal(12) + color(4) + uv(8)  = 36  flags 0x152
      shadow:         pos(12)                                  = 12  flags 0x002
      everything else: pos(12) + normal(12) + uv(8)            = 32  flags 0x112
    """
    if shader_id in (SHADER_CAR_BODY, SHADER_SKINNING):
        flags = VERTEX_POSITION | VERTEX_NORMAL | VERTEX_COLOR | VERTEX_UV
        return flags, 36
    elif shader_id == SHADER_SHADOW_PROJECT:
        return VERTEX_POSITION, 12
    else:
        flags = VERTEX_POSITION | VERTEX_NORMAL | VERTEX_UV
        return flags, 32


# COORDINATE TRANSFORMS
#
# Blender -> FO2:  fo2 = (bl_x, bl_z, bl_y)
# (car front faces Blender +Y. FO2 Y<->Z swap, no negations.)

def blender_to_fo2_pos(co, inv_scale: float = 1.0):
    """Convert a Blender world position to FO2 vertex position.

    Mapping: fo2 = (bl_x, bl_z, bl_y) / scale
    """
    return (co.x * inv_scale, co.z * inv_scale, co.y * inv_scale)


def blender_to_fo2_normal(nrm):
    """Convert a Blender normal to FO2 normal.

    Same transform as positions (without scale):
    fo2 = (bl_x, bl_z, bl_y)
    """
    return (nrm.x, nrm.z, nrm.y)


def blender_to_fo2_matrix_flat(bl_mat) -> list:
    """Convert a Blender 4×4 matrix to FO2 column-major float[16].

    Export position mapping: fo2 = (bl_x, bl_z, bl_y)
    M_fo2 = A * M_bl * A  where A = [[1,0,0],[0,0,1],[0,1,0]] (self-inverse Y↔Z swap)
    Result: swap rows/cols 1↔2, no sign changes.
    """
    M = bl_mat
    intermediate = Matrix((
        (M[0][0], M[0][2], M[0][1], M[0][3]),
        (M[2][0], M[2][2], M[2][1], M[2][3]),
        (M[1][0], M[1][2], M[1][1], M[1][3]),
        (M[3][0], M[3][2], M[3][1], M[3][3]),
    ))

    # flatten as column-major
    flat = [0.0] * 16
    for col in range(4):
        for row in range(4):
            flat[col * 4 + row] = intermediate[row][col]
    return flat


# TEXTURE EXTRACTION FROM BLENDER MATERIAL

def get_texture_name_from_material(bl_mat) -> str:
    """Extract the texture filename from a Blender material.

    Walks the node tree looking for an Image Texture node.
    Handles Windows paths, Blender relative paths (//), and packed images.
    """
    if not bl_mat or not bl_mat.use_nodes:
        return ""

    for node in bl_mat.node_tree.nodes:
        if node.type == 'TEX_IMAGE' and node.image:
            name = ""
            # try filepath first, normalise separators for cross-platform
            fp = node.image.filepath or ""
            fp = fp.replace('\\', '/')   # windows -> POSIX
            fp = fp.lstrip('/')          # strip Blender relative '//'
            if fp:
                name = fp.rsplit('/', 1)[-1]  # basename
            # fallback to Blender's internal image name
            if not name and node.image.name:
                name = node.image.name
            if name:
                # strip any duplicate suffix from image name
                name = re.sub(r'\.\d{3}$', '', name)
                # convert extension to .tga
                base, ext = os.path.splitext(name)
                if ext:
                    name = base + ".tga"
                elif not name.endswith(".tga"):
                    name += ".tga"
                return name
    return ""


# FO2 DATA STRUCTURES

class FO2Material:
    def __init__(self):
        self.identifier = MATC_ID
        self.name = ""
        self.alpha = 0
        self.v92 = 0
        self.num_textures = 0
        self.shader_id = SHADER_CAR_METAL
        self.use_colormap = 0
        self.v74 = 0
        self.v108 = [0, 0, 0]
        self.v109 = [0, 0, 0]
        self.v98 = [0, 0, 0, 0]
        self.v99 = [0, 0, 0, 0]
        self.v100 = [0, 0, 0, 0]
        self.v101 = [0, 0, 0, 0]
        self.v102 = 0
        self.texture_names = ["", "", ""]

    def write(self, f):
        f.write(struct.pack('<I', self.identifier))
        f.write(self.name.encode('ascii') + b'\x00')
        f.write(struct.pack('<i', self.alpha))
        f.write(struct.pack('<i', self.v92))
        f.write(struct.pack('<i', self.num_textures))
        f.write(struct.pack('<i', self.shader_id))
        f.write(struct.pack('<i', self.use_colormap))
        f.write(struct.pack('<i', self.v74))
        f.write(struct.pack('<3i', *self.v108))
        f.write(struct.pack('<3i', *self.v109))
        f.write(struct.pack('<4i', *self.v98))
        f.write(struct.pack('<4i', *self.v99))
        f.write(struct.pack('<4i', *self.v100))
        f.write(struct.pack('<4i', *self.v101))
        f.write(struct.pack('<i', self.v102))
        for i in range(3):
            f.write(self.texture_names[i].encode('ascii') + b'\x00')


class FO2VertexBuffer:
    def __init__(self, buf_id, flags, vertex_size, vertex_count, data_bytes):
        self.id = buf_id
        self.flags = flags
        self.vertex_size = vertex_size
        self.vertex_count = vertex_count
        self.data = data_bytes  # raw bytes

    def write(self, f):
        f.write(struct.pack('<i', 1))             # type = vertex buffer
        f.write(struct.pack('<i', 0))             # foucExtraFormat = 0 (FO2)
        f.write(struct.pack('<I', self.vertex_count))
        f.write(struct.pack('<I', self.vertex_size))
        f.write(struct.pack('<I', self.flags))
        f.write(self.data)


class FO2IndexBuffer:
    def __init__(self, buf_id, index_count, data_bytes):
        self.id = buf_id
        self.index_count = index_count
        self.data = data_bytes  # raw bytes (uint16)

    def write(self, f):
        f.write(struct.pack('<i', 2))             # type = index buffer
        f.write(struct.pack('<i', 0))             # foucExtraFormat = 0
        f.write(struct.pack('<I', self.index_count))
        f.write(self.data)


class FO2Surface:
    def __init__(self):
        self.is_vegetation = 0
        self.material_id = 0
        self.vertex_count = 0
        self.flags = 0
        self.poly_count = 0
        self.poly_mode = 4   # triangles
        self.num_indices_used = 0
        self.num_streams_used = 2
        self.stream_id = [0, 0]
        self.stream_offset = [0, 0]

    def write(self, f):
        f.write(struct.pack('<i', self.is_vegetation))
        f.write(struct.pack('<i', self.material_id))
        f.write(struct.pack('<i', self.vertex_count))
        f.write(struct.pack('<i', self.flags))
        f.write(struct.pack('<i', self.poly_count))
        f.write(struct.pack('<i', self.poly_mode))
        f.write(struct.pack('<i', self.num_indices_used))
        # FO2 does NOT write vCenter/vRadius for surfaces (only FO1 does)
        f.write(struct.pack('<i', self.num_streams_used))
        for j in range(self.num_streams_used):
            f.write(struct.pack('<I', self.stream_id[j]))
            f.write(struct.pack('<I', self.stream_offset[j]))


class FO2Model:
    def __init__(self):
        self.identifier = BMOD_ID
        self.unk = 0
        self.name = ""
        self.center = [0.0, 0.0, 0.0]
        self.radius = [0.0, 0.0, 0.0]
        self.f_radius = 0.0
        self.surface_ids = []

    def write(self, f):
        f.write(struct.pack('<I', self.identifier))
        f.write(struct.pack('<i', self.unk))
        f.write(self.name.encode('ascii') + b'\x00')
        f.write(struct.pack('<3f', *self.center))
        f.write(struct.pack('<3f', *self.radius))
        f.write(struct.pack('<f', self.f_radius))
        f.write(struct.pack('<i', len(self.surface_ids)))
        for sid in self.surface_ids:
            f.write(struct.pack('<i', sid))


class FO2CompactMesh:
    """BGM mesh entry (car body part)."""
    def __init__(self):
        self.identifier = MESH_ID
        self.name1 = ""
        self.name2 = ""
        self.flags = 0x0
        self.group = -1
        self.matrix = [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ]
        self.model_ids = []

    def write(self, f):
        f.write(struct.pack('<I', self.identifier))
        f.write(self.name1.encode('ascii') + b'\x00')
        f.write(self.name2.encode('ascii') + b'\x00')
        f.write(struct.pack('<I', self.flags))
        f.write(struct.pack('<i', self.group))
        f.write(struct.pack('<16f', *self.matrix))
        f.write(struct.pack('<i', len(self.model_ids)))
        for mid in self.model_ids:
            f.write(struct.pack('<i', mid))


class FO2Object:
    """Dummy / object point."""
    def __init__(self):
        self.identifier = OBJC_ID
        self.name1 = ""
        self.name2 = ""
        self.flags = 0xE0F9
        self.matrix = [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ]

    def write(self, f):
        f.write(struct.pack('<I', self.identifier))
        f.write(self.name1.encode('ascii') + b'\x00')
        f.write(self.name2.encode('ascii') + b'\x00')
        f.write(struct.pack('<I', self.flags))
        f.write(struct.pack('<16f', *self.matrix))


# MESH -> VERTEX / INDEX BUFFER BUILDING

def fo2_colmajor_to_rowmajor(flat):
    """Convert FO2 column-major float[16] to row-major 4x4 list-of-lists."""
    return [
        [flat[0], flat[4], flat[8],  flat[12]],
        [flat[1], flat[5], flat[9],  flat[13]],
        [flat[2], flat[6], flat[10], flat[14]],
        [flat[3], flat[7], flat[11], flat[15]],
    ]


def invert_4x4(m):
    """Invert a 4x4 matrix (list-of-lists). Returns None if singular."""
    # Use mathutils for robustness
    mat = Matrix((m[0], m[1], m[2], m[3]))
    try:
        inv = mat.inverted()
        return [[inv[r][c] for c in range(4)] for r in range(4)]
    except ValueError:
        return None


def build_buffers_for_material(obj, mat_index, flags, vertex_size,
                               inv_scale, buf_id_start,
                               fo2_mesh_matrix_inv=None,
                               mesh_override=None):
    """Build a vertex buffer and index buffer for all faces of `mat_index` on the given Blender mesh object.

    If fo2_mesh_matrix_inv is provided (4x4 list-of-lists), positions and 
    normals are transformed from FO2 world space to FO2 local space.

    Returns (FO2VertexBuffer, FO2IndexBuffer, vertex_count, poly_count) or None if no faces.
    """
    mesh = mesh_override if mesh_override is not None else obj.data
    mat_world = obj.matrix_world

    has_normal = (flags & VERTEX_NORMAL) != 0
    has_color  = (flags & VERTEX_COLOR)  != 0
    has_uv     = (flags & VERTEX_UV)     != 0

    # get UV layer
    uv_layer = mesh.uv_layers.active if has_uv else None

    # get vertex color layer
    color_attr = None
    if has_color:
        if hasattr(mesh, 'color_attributes') and len(mesh.color_attributes) > 0:
            color_attr = mesh.color_attributes[0]
        elif hasattr(mesh, 'vertex_colors') and len(mesh.vertex_colors) > 0:
            color_attr = mesh.vertex_colors[0]

    # collect faces for this material
    faces = [p for p in mesh.polygons if p.material_index == mat_index]
    if not faces:
        return None

    # precompute normals
    if hasattr(mesh, 'calc_normals_split'):
        mesh.calc_normals_split()

    # build unique vertices and index list
    # key: (pos_tuple, nrm_tuple, uv_tuple, color_tuple) -> vert index
    vert_map = {}
    unique_verts = []
    indices = []

    for poly in faces:
        if len(poly.loop_indices) != 3:
            continue  # skip non-tris (shouldn't happen if triangulated)

        face_indices = []
        for loop_idx in poly.loop_indices:
            loop = mesh.loops[loop_idx]
            vi = loop.vertex_index

            # world-space position
            world_co = mat_world @ mesh.vertices[vi].co
            fo2_pos = blender_to_fo2_pos(world_co, inv_scale)

            # transform to local space if mesh matrix inverse provided
            if fo2_mesh_matrix_inv:
                M = fo2_mesh_matrix_inv
                px, py, pz = fo2_pos
                fo2_pos = (
                    M[0][0]*px + M[0][1]*py + M[0][2]*pz + M[0][3],
                    M[1][0]*px + M[1][1]*py + M[1][2]*pz + M[1][3],
                    M[2][0]*px + M[2][1]*py + M[2][2]*pz + M[2][3],
                )

            # normal (use split normal for correct shading)
            fo2_nrm = (0.0, 0.0, 0.0)
            if has_normal:
                nrm = Vector(loop.normal)
                # transform normal to world space (rotation only)
                nrm = (mat_world.to_3x3() @ nrm).normalized()
                fo2_nrm = blender_to_fo2_normal(nrm)
                # transform normal to local space (rotation only)
                if fo2_mesh_matrix_inv:
                    M = fo2_mesh_matrix_inv
                    nx, ny, nz = fo2_nrm
                    fo2_nrm = (
                        M[0][0]*nx + M[0][1]*ny + M[0][2]*nz,
                        M[1][0]*nx + M[1][1]*ny + M[1][2]*nz,
                        M[2][0]*nx + M[2][1]*ny + M[2][2]*nz,
                    )
                # clamp
                fo2_nrm = tuple(max(-1.0, min(1.0, c)) for c in fo2_nrm)

            # UV
            fo2_uv = (0.0, 0.0)
            if has_uv and uv_layer:
                uv = uv_layer.data[loop_idx].uv
                fo2_uv = (uv[0], 1.0 - uv[1])  # V flip

            # vertex color (RGBA packed as uint32)
            fo2_color = (255, 255, 255, 255)
            if has_color and color_attr:
                try:
                    cd = color_attr.data[loop_idx]
                    c = cd.color
                    fo2_color = (
                        max(0, min(255, int(c[0] * 255 + 0.5))),
                        max(0, min(255, int(c[1] * 255 + 0.5))),
                        max(0, min(255, int(c[2] * 255 + 0.5))),
                        max(0, min(255, int(c[3] * 255 + 0.5))),
                    )
                except (IndexError, AttributeError):
                    pass

            # round for dedup key (avoid float precision issues)
            def _r(v, d=6): return tuple(round(x, d) for x in v)

            key = (_r(fo2_pos), _r(fo2_nrm), _r(fo2_uv), fo2_color)

            if key not in vert_map:
                vert_map[key] = len(unique_verts)
                unique_verts.append((fo2_pos, fo2_nrm, fo2_uv, fo2_color, vi))

            face_indices.append(vert_map[key])

        # original winding order reversed to match FO2 convention
        # {face[2], face[1], face[0]}
        indices.extend([face_indices[2], face_indices[1], face_indices[0]])

    if not unique_verts or not indices:
        return None

    vertex_count = len(unique_verts)
    poly_count = len(indices) // 3

    if vertex_count > 65535:
        print(f"[BGM Export] WARNING: {obj.name} material slot {mat_index} "
              f"has {vertex_count} vertices (max 65535)!")

    # pack vertex data
    vdata = bytearray()
    blender_vis = []  # blender vertex index for each unique vertex
    for pos, nrm, uv, color, bvi in unique_verts:
        blender_vis.append(bvi)
        # position (3 floats, always present)
        vdata += struct.pack('<3f', *pos)
        # normal (3 floats)
        if has_normal:
            vdata += struct.pack('<3f', *nrm)
        # vertex color (1 uint32, RGBA little-endian)
        if has_color:
            vdata += struct.pack('<4B', color[0], color[1], color[2], color[3])
        # UV (2 floats)
        if has_uv:
            vdata += struct.pack('<2f', *uv)

    # pack index data (uint16)
    idata = bytearray()
    for idx in indices:
        idata += struct.pack('<H', idx)

    vbuf = FO2VertexBuffer(buf_id_start, flags, vertex_size,
                           vertex_count, bytes(vdata))
    ibuf = FO2IndexBuffer(buf_id_start + 1, len(indices), bytes(idata))

    return vbuf, ibuf, vertex_count, poly_count, blender_vis


# BUILD FO2 MATERIAL FROM BLENDER MATERIAL

def build_fo2_material(bl_mat) -> FO2Material:
    """Convert a Blender material to an FO2 material struct."""
    mat = FO2Material()
    mat.name = bl_mat.name

    # strip Blender duplicate suffix (.001, .004, etc..)
    clean_name = re.sub(r'\.\d{3}$', '', mat.name)
    mat.name = clean_name

    # get texture name from the node tree
    tex_name = bl_mat.get("bgm_texture", "") or get_texture_name_from_material(bl_mat)

    # check for stored custom properties from the importer first
    # only use them if bgm_shader_id is actually set to a non-empty value
    has_stored_shader = (
        "bgm_shader_id" in bl_mat
        and bl_mat["bgm_shader_id"] is not None
        and str(bl_mat["bgm_shader_id"]).strip() != ""
    )
    if has_stored_shader and "bgm_alpha" in bl_mat:
        mat.shader_id = int(bl_mat["bgm_shader_id"])
        mat.alpha = int(bl_mat["bgm_alpha"])
        # Preserve v92 if stored
        if "bgm_v92" in bl_mat:
            mat.v92 = int(bl_mat["bgm_v92"])
        # Preserve other stored properties
        if "bgm_use_colormap" in bl_mat:
            mat.use_colormap = int(bl_mat["bgm_use_colormap"])
        if "bgm_v74" in bl_mat:
            mat.v74 = int(bl_mat["bgm_v74"])
        if "bgm_v102" in bl_mat:
            mat.v102 = int(bl_mat["bgm_v102"])
        # restore all 3 texture slot names if stored
        if "bgm_texture_0" in bl_mat:
            for ti in range(3):
                key = f"bgm_texture_{ti}"
                if key in bl_mat:
                    mat.texture_names[ti] = str(bl_mat[key])
            tex_name = mat.texture_names[0]  # primary texture for further logic
        # still apply texture override rules for special shaders
        if mat.shader_id == SHADER_CAR_BODY:
            tex_name = "skin1.tga"
        elif mat.shader_id == SHADER_CAR_LIGHTS:
            mat.v92 = 2
    else:
        # infer everything from name
        shader_id, alpha, v92, tex_override = get_shader_for_material(
            mat.name, tex_name)
        mat.shader_id = shader_id
        mat.alpha = alpha
        mat.v92 = v92
        if tex_override is not None:
            tex_name = tex_override

    # if still no texture, try deriving from material name
    if not tex_name:
        clean = mat.name
        for ext in (".tga", ".png", ".dds", ".bmp"):
            if clean.lower().endswith(ext):
                clean = clean[:-4]
                break
        tex_name = clean + ".tga"

    # only set slot 0 if we didn't already restore all slots from stored data
    if not mat.texture_names[0]:
        mat.texture_names[0] = tex_name
    mat.num_textures = sum(1 for t in mat.texture_names if t)

    return mat


# SCENE COLLECTION & EXPORT

def find_root_empty(context):
    """Find the fo2_body empty (importer root). Returns None if not found."""
    return context.scene.objects.get("fo2_body")


def find_crash_root_empty(context):
    """Find the fo2_body_crash empty. Returns None if not found."""
    return context.scene.objects.get("fo2_body_crash")


def collect_objects_under(root, context):
    """Collect mesh objects and empties parented to *root*.

    Mesh objects come from direct children of root.
    Dummy empties come from the fo2_body_dummies child of root (if present),
    falling back to direct empty children of root for backwards compatibility.

    Returns (mesh_objects, empty_objects).
    """
    mesh_objects = []
    empty_objects = []

    if root is None:
        return mesh_objects, empty_objects

    # find fo2_body_dummies child for dummy collection
    dummies_root = next(
        (c for c in root.children if c.type == 'EMPTY' and c.name == 'fo2_body_dummies'),
        None
    )

    for obj in root.children:
        if obj.type == 'MESH' and obj.data:
            mesh_objects.append(obj)
        elif obj.type == 'EMPTY' and obj != dummies_root:
            # intermediate empties that parent mesh objects (not dummy points)
            has_mesh_child = any(c.type == 'MESH' for c in obj.children)
            if has_mesh_child:
                for child in obj.children:
                    if child.type == 'MESH' and child.data:
                        mesh_objects.append(child)
            elif dummies_root is None:
                # fallback: no fo2_body_dummies — treat direct empty children as dummies
                empty_objects.append(obj)

    # collect dummies from fo2_body_dummies
    if dummies_root is not None:
        for obj in dummies_root.children:
            if obj.type == 'EMPTY':
                empty_objects.append(obj)

    return mesh_objects, empty_objects


def triangulate_mesh(obj):
    """Ensure mesh is triangulated (non-destructive via bmesh copy)."""
    mesh = obj.data
    needs_tri = any(len(p.vertices) != 3 for p in mesh.polygons)
    if not needs_tri:
        return mesh, False

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    new_mesh = bpy.data.meshes.new(mesh.name + "_tri_export")
    bm.to_mesh(new_mesh)
    bm.free()
    if hasattr(new_mesh, 'calc_normals_split'):
        new_mesh.calc_normals_split()
    return new_mesh, True


def write_crash_dat(filepath: str, crash_data: list):
    """Write a FO2 crash.dat file.

    crash_data: list of (model_name, surfaces_list)
      surfaces_list: list of (base_vbuf_bytes, crash_vbuf_bytes, vertex_size, vertex_count)
    """
    if not crash_data:
        print("[BGM Export] No crash data found, skipping crash.dat")
        return

    print(f"[BGM Export] Writing crash.dat: {filepath}")
    with open(filepath, 'wb') as f:
        f.write(struct.pack('<I', len(crash_data)))

        for model_name, surfaces in crash_data:
            # null-terminated name: model_name + "_crash"
            name = model_name + "_crash"
            f.write(name.encode('ascii') + b'\x00')

            f.write(struct.pack('<I', len(surfaces)))
            for base_vdata, crash_vdata, vsize, vcount in surfaces:
                # vertex count
                f.write(struct.pack('<I', vcount))
                # vertex data size in bytes
                f.write(struct.pack('<I', vcount * vsize))
                # BASE vertex buffer (not crash)
                f.write(base_vdata)

                # crash weights: 48 bytes per vertex
                # (basePos[3], crashPos[3], baseNorm[3], crashNorm[3])
                for i in range(vcount):
                    off = i * vsize
                    # positions are first 3 floats (offset 0)
                    base_pos  = base_vdata[off:off + 12]
                    crash_pos = crash_vdata[off:off + 12]
                    # normals are next 3 floats (offset 12)
                    base_nrm  = base_vdata[off + 12:off + 24]
                    crash_nrm = crash_vdata[off + 12:off + 24]
                    f.write(base_pos)
                    f.write(crash_pos)
                    f.write(base_nrm)
                    f.write(crash_nrm)

    print(f"[BGM Export] crash.dat done: {len(crash_data)} nodes")


def write_bgm(filepath: str, context, options: dict):
    """Main export function."""
    global_scale = options.get('global_scale', 1.0)
    inv_scale = 1.0 / global_scale if global_scale != 0 else 1.0
    use_priorities = options.get('use_priorities', True)
    auto_triangulate = options.get('auto_triangulate', True)

    root = find_root_empty(context)
    mesh_objects, empty_objects = collect_objects_under(root, context)

    # collect crash mesh objects parented to fo2_body_crash
    crash_root = find_crash_root_empty(context)
    crash_mesh_map = {}  # base_model_name -> crash Blender object
    if crash_root:
        for obj in crash_root.children:
            if obj.type != 'MESH' or not obj.data:
                continue
            name = re.sub(r'\.\d{3}$', '', obj.name)
            if name.endswith('_crash'):
                base_name = name[:-6]
                crash_mesh_map[base_name] = obj

    if not mesh_objects and not empty_objects:
        print("[BGM Export] ERROR: No objects found under fo2_body!")
        return {'CANCELLED'}

    # sanitize mesh custom properties
    for obj in mesh_objects:
        changed = False
        if "bgm_flags" not in obj or obj["bgm_flags"] is None:
            obj["bgm_flags"] = 0
            changed = True
        if "bgm_group" not in obj or obj["bgm_group"] is None:
            obj["bgm_group"] = -1
            changed = True
        if "bgm_name2" not in obj:
            obj["bgm_name2"] = ""
            changed = True
        if changed:
            obj.update_tag()
            print(f"[BGM Export] Initialized missing BGM properties on mesh: {obj.name}")

    # sanitize material custom properties
    # Collect all unique shader IDs already in use, so we can assign a fresh one if bgm_shader_id is missing from a material
    used_shader_ids = set()
    for obj in mesh_objects:
        for slot in obj.material_slots:
            if slot.material and "bgm_shader_id" in slot.material:
                try:
                    used_shader_ids.add(int(slot.material["bgm_shader_id"]))
                except (TypeError, ValueError):
                    pass

    def _next_unused_shader_id():
        sid = 0
        while sid in used_shader_ids:
            sid += 1
        used_shader_ids.add(sid)
        return sid

    seen_materials = set()
    for obj in mesh_objects:
        for slot in obj.material_slots:
            bl_mat = slot.material
            if not bl_mat or id(bl_mat) in seen_materials:
                continue
            seen_materials.add(id(bl_mat))

            mat_changed = False

            # derive texture name from node tree as fallback
            tex_name = bl_mat.get("bgm_texture", "") or get_texture_name_from_material(bl_mat)
            if tex_name:
                base, ext = os.path.splitext(tex_name)
                if ext.lower() != '.tga':
                    tex_name = base + '.tga'
            else:
                tex_name = re.sub(r'\.\d{3}$', '', bl_mat.name) + '.tga'

            if "bgm_alpha" not in bl_mat:
                bl_mat["bgm_alpha"] = 0
                mat_changed = True
            if "bgm_num_textures" not in bl_mat:
                bl_mat["bgm_num_textures"] = 1
                mat_changed = True
            if "bgm_shader_id" not in bl_mat:
                bl_mat["bgm_shader_id"] = _next_unused_shader_id()
                mat_changed = True
            else:
                # still register the existing one so _next_unused_shader_id stays correct
                try:
                    used_shader_ids.add(int(bl_mat["bgm_shader_id"]))
                except (TypeError, ValueError):
                    pass
            if "bgm_texture" not in bl_mat:
                bl_mat["bgm_texture"] = tex_name
                mat_changed = True
            if "bgm_texture_0" not in bl_mat:
                bl_mat["bgm_texture_0"] = tex_name
                mat_changed = True
            if "bgm_texture_1" not in bl_mat:
                bl_mat["bgm_texture_1"] = ""
                mat_changed = True
            if "bgm_texture_2" not in bl_mat:
                bl_mat["bgm_texture_2"] = ""
                mat_changed = True
            if "bgm_use_colormap" not in bl_mat:
                bl_mat["bgm_use_colormap"] = 0
                mat_changed = True
            if "bgm_v102" not in bl_mat:
                bl_mat["bgm_v102"] = 0
                mat_changed = True
            if "bgm_v74" not in bl_mat:
                bl_mat["bgm_v74"] = 0
                mat_changed = True
            if "bgm_v92" not in bl_mat:
                bl_mat["bgm_v92"] = 0
                mat_changed = True

            if mat_changed:
                print(f"[BGM Export] Initialized missing BGM properties on material: {bl_mat.name}")

    # collect & deduplicate materials
    # we need a deterministic ordering with priority support
    bl_mat_to_fo2_id = {}  # blender material -> FO2 material index
    fo2_materials = []     # ordered list

    # gather all unique materials
    all_mat_slots = []
    for obj in mesh_objects:
        for slot in obj.material_slots:
            if slot.material and slot.material not in bl_mat_to_fo2_id:
                bl_mat_to_fo2_id[slot.material] = None  # placeholder
                all_mat_slots.append(slot.material)

    # sort by priority then assign IDs
    if use_priorities:
        all_mat_slots.sort(key=lambda m: get_material_priority(m.name))

    for i, bl_mat in enumerate(all_mat_slots):
        fo2_mat = build_fo2_material(bl_mat)
        bl_mat_to_fo2_id[bl_mat] = i
        fo2_materials.append(fo2_mat)

    # build streams, surfaces, models, compact meshes
    fo2_vbufs = []
    fo2_ibufs = []
    fo2_surfaces = []
    fo2_models = []
    fo2_compact_meshes = []

    stream_id_counter = 0

    # crash data: list of (model_name, [(base_vdata, crash_vdata, vsize, vcount), ...])
    all_crash_data = []

    # sort mesh objects by priority (process low-priority materials first)
    if use_priorities:
        def mesh_sort_key(obj):
            prios = []
            for slot in obj.material_slots:
                if slot.material:
                    prios.append(get_material_priority(slot.material.name))
            return min(prios) if prios else 0
        mesh_objects.sort(key=mesh_sort_key)

    temp_meshes = []  # track temp meshes for cleanup

    for obj in mesh_objects:
        # triangulate if needed — produce a temp mesh, never swap obj.data
        if auto_triangulate:
            work_mesh, is_temp = triangulate_mesh(obj)
        else:
            work_mesh, is_temp = obj.data, False
        if is_temp:
            temp_meshes.append(work_mesh)
        orig_mesh = None  # unused now, kept for cleanup compat

        # ensure split normals (removed in Blender 4.1+)
        if hasattr(work_mesh, 'calc_normals_split'):
            work_mesh.calc_normals_split()

        # gather material slots and sort by priority
        mat_slots = list(range(len(obj.material_slots)))
        if use_priorities:
            mat_slots.sort(key=lambda mi: get_material_priority(
                obj.material_slots[mi].material.name
            ) if obj.material_slots[mi].material else 0)

        # build model for this mesh object
        model = FO2Model()
        model.name = re.sub(r'\.\d{3}$', '', obj.name)

        # check for stored FO2 mesh matrix (from import)
        # derive FO2 mesh matrix from the Blender object's world transform
        # same conjugation as dummy objects, works for both imported and new models
        stored_matrix = None
        fo2_mesh_matrix_inv = None
        mat = obj.matrix_world.copy()
        # check if object has a non-identity transform
        is_identity = True
        for r in range(4):
            for c in range(4):
                expected = 1.0 if r == c else 0.0
                if abs(mat[r][c] - expected) > 1e-5:
                    is_identity = False
                    break
            if not is_identity:
                break

        if not is_identity:
            # convert Blender matrix_world to FO2 column-major flat[16].
            # new mapping fo2=(bl_x,bl_z,bl_y): swap rows/cols 1<->2, no sign changes.
            M = mat.copy()
            M[0][3] *= inv_scale
            M[1][3] *= inv_scale
            M[2][3] *= inv_scale
            fo2_rows = (
                (M[0][0], M[0][2], M[0][1], M[0][3]),
                (M[2][0], M[2][2], M[2][1], M[2][3]),
                (M[1][0], M[1][2], M[1][1], M[1][3]),
                (M[3][0], M[3][2], M[3][1], M[3][3]),
            )
            stored_matrix = [0.0] * 16
            for col in range(4):
                for row in range(4):
                    stored_matrix[col * 4 + row] = fo2_rows[row][col]

            row_major = fo2_colmajor_to_rowmajor(stored_matrix)
            fo2_mesh_matrix_inv = invert_4x4(row_major)
            if fo2_mesh_matrix_inv is None:
                print(f"[BGM Export] WARNING: singular mesh matrix for {obj.name}, using identity")
                stored_matrix = None

        aabb_min = [1e9, 1e9, 1e9]
        aabb_max = [-1e9, -1e9, -1e9]

        # track per-surface info for crash matching:
        # list of (mat_idx, flags, vsize, base_vbuf_data, vert_count)
        surface_build_info = []

        for mat_idx in mat_slots:
            slot = obj.material_slots[mat_idx]
            if not slot.material:
                continue

            fo2_mat_id = bl_mat_to_fo2_id.get(slot.material, 0)
            fo2_mat = fo2_materials[fo2_mat_id]
            flags, vsize = get_vertex_format(fo2_mat.shader_id)

            result = build_buffers_for_material(
                obj, mat_idx, flags, vsize, inv_scale, stream_id_counter,
                fo2_mesh_matrix_inv=fo2_mesh_matrix_inv,
                mesh_override=work_mesh)
            if not result:
                continue

            vbuf, ibuf, vert_count, poly_count, blender_vis = result

            # create surface
            surface = FO2Surface()
            surface.material_id = fo2_mat_id
            surface.vertex_count = vert_count
            surface.flags = flags
            surface.poly_count = poly_count
            surface.num_indices_used = poly_count * 3
            surface.stream_id = [stream_id_counter, stream_id_counter + 1]
            surface.stream_offset = [0, 0]

            # update AABB from vertex buffer positions
            vdata = vbuf.data
            for vi in range(vert_count):
                off = vi * vsize
                px, py, pz = struct.unpack_from('<3f', vdata, off)
                aabb_min[0] = min(aabb_min[0], px)
                aabb_min[1] = min(aabb_min[1], py)
                aabb_min[2] = min(aabb_min[2], pz)
                aabb_max[0] = max(aabb_max[0], px)
                aabb_max[1] = max(aabb_max[1], py)
                aabb_max[2] = max(aabb_max[2], pz)

            model.surface_ids.append(len(fo2_surfaces))
            fo2_surfaces.append(surface)
            fo2_vbufs.append(vbuf)
            fo2_ibufs.append(ibuf)
            surface_build_info.append((mat_idx, flags, vsize, vbuf.data, vert_count, blender_vis))
            stream_id_counter += 2

        # finalize model AABB
        if model.surface_ids:
            model.center = [
                (aabb_max[0] + aabb_min[0]) * 0.5,
                (aabb_max[1] + aabb_min[1]) * 0.5,
                (aabb_max[2] + aabb_min[2]) * 0.5,
            ]
            model.radius = [
                abs(aabb_max[0] - aabb_min[0]) * 0.5,
                abs(aabb_max[1] - aabb_min[1]) * 0.5,
                abs(aabb_max[2] - aabb_min[2]) * 0.5,
            ]
            model.f_radius = math.sqrt(sum(r * r for r in model.radius))

            # build crash data if matching crash mesh exists
            model_name = re.sub(r'\.\d{3}$', '', obj.name)
            crash_obj = crash_mesh_map.get(model_name)
            if crash_obj and surface_build_info:
                if auto_triangulate:
                    crash_work_mesh, crash_is_temp = triangulate_mesh(crash_obj)
                else:
                    crash_work_mesh, crash_is_temp = crash_obj.data, False
                if crash_is_temp:
                    temp_meshes.append(crash_work_mesh)

                crash_mat_world = crash_obj.matrix_world
                crash_verts = crash_work_mesh.vertices

                crash_surfaces = []
                for base_mat_idx, flags, vsize, base_vdata, base_vcount, blender_vis in surface_build_info:
                    has_normal = bool(flags & VERTEX_NORMAL)
                    crash_vdata = bytearray(base_vdata)  # start with base (preserves UVs/colors)

                    for buf_idx, bvi in enumerate(blender_vis):
                        off = buf_idx * vsize
                        # position from crash mesh vertex
                        if bvi < len(crash_verts):
                            world_co = crash_mat_world @ crash_verts[bvi].co
                        else:
                            # topology mismatch — keep base position
                            world_co = obj.matrix_world @ work_mesh.vertices[bvi].co
                        crash_pos = blender_to_fo2_pos(world_co, inv_scale)
                        if fo2_mesh_matrix_inv:
                            M = fo2_mesh_matrix_inv
                            px, py, pz = crash_pos
                            crash_pos = (
                                M[0][0]*px + M[0][1]*py + M[0][2]*pz + M[0][3],
                                M[1][0]*px + M[1][1]*py + M[1][2]*pz + M[1][3],
                                M[2][0]*px + M[2][1]*py + M[2][2]*pz + M[2][3],
                            )
                        struct.pack_into('<3f', crash_vdata, off, *crash_pos)
                        # normal: keep base normal (crash normals aren't used by the engine)

                    crash_surfaces.append((base_vdata, bytes(crash_vdata), vsize, base_vcount))

                if crash_surfaces:
                    all_crash_data.append((model_name, crash_surfaces))

            # build compact mesh
            cm = FO2CompactMesh()
            cm.name1 = re.sub(r'\.\d{3}$', '', obj.name)
            cm.name2 = obj.get("bgm_name2", "")
            cm.flags = obj.get("bgm_flags", 0x0)
            cm.group = obj.get("bgm_group", -1)
            # use stored FO2 matrix if available, else identity
            if stored_matrix:
                cm.matrix = list(stored_matrix)
            else:
                cm.matrix = [
                    1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1,
                ]
            cm.model_ids.append(len(fo2_models))

            fo2_models.append(model)
            fo2_compact_meshes.append(cm)

    # cleanup temp meshes
    for tm in temp_meshes:
        bpy.data.meshes.remove(tm)

    # build dummies
    fo2_objects = []
    for empty in empty_objects:
        fo2_obj = FO2Object()
        fo2_obj.name1 = re.sub(r'\.\d{3}$', '', empty.name)
        fo2_obj.name2 = ""
        fo2_obj.flags = 0xE0F9

        # convert Blender matrix to FO2 column-major flat[16].
        # new mapping fo2=(bl_x,bl_z,bl_y): swap rows/cols 1<->2, no sign changes.
        mat = empty.matrix_world.copy()
        # undo global scale on translation
        mat[0][3] *= inv_scale
        mat[1][3] *= inv_scale
        mat[2][3] *= inv_scale
        M = mat
        fo2_rows = (
            (M[0][0], M[0][2], M[0][1], M[0][3]),
            (M[2][0], M[2][2], M[2][1], M[2][3]),
            (M[1][0], M[1][2], M[1][1], M[1][3]),
            (M[3][0], M[3][2], M[3][1], M[3][3]),
        )
        flat = [0.0] * 16
        for col in range(4):
            for row in range(4):
                flat[col * 4 + row] = fo2_rows[row][col]
        fo2_obj.matrix = flat

        fo2_objects.append(fo2_obj)

    # write BGM binary
    print(f"[BGM Export] Writing {filepath}")
    print(f"  Materials: {len(fo2_materials)}")
    print(f"  Streams:   {stream_id_counter}")
    print(f"  Surfaces:  {len(fo2_surfaces)}")
    print(f"  Models:    {len(fo2_models)}")
    print(f"  Meshes:    {len(fo2_compact_meshes)}")
    print(f"  Objects:   {len(fo2_objects)}")
    if all_crash_data:
        total_crash_surfs = sum(len(surfs) for _, surfs in all_crash_data)
        print(f"  Crash nodes: {len(all_crash_data)} ({total_crash_surfs} surfaces)")

    with open(filepath, 'wb') as f:
        # file version
        f.write(struct.pack('<I', BGM_VERSION_FO2))

        # materials
        f.write(struct.pack('<I', len(fo2_materials)))
        for mat in fo2_materials:
            mat.write(f)

        # streams (interleaved by ID: vbuf0, ibuf0, vbuf1, ibuf1, ...)
        f.write(struct.pack('<I', stream_id_counter))
        for i in range(stream_id_counter):
            for vb in fo2_vbufs:
                if vb.id == i:
                    vb.write(f)
            for ib in fo2_ibufs:
                if ib.id == i:
                    ib.write(f)

        # surfaces
        f.write(struct.pack('<I', len(fo2_surfaces)))
        for surf in fo2_surfaces:
            surf.write(f)

        # models
        f.write(struct.pack('<I', len(fo2_models)))
        for model in fo2_models:
            model.write(f)

        # compact meshes (BGM meshes)
        f.write(struct.pack('<I', len(fo2_compact_meshes)))
        for cm in fo2_compact_meshes:
            cm.write(f)

        # objects
        f.write(struct.pack('<I', len(fo2_objects)))
        for obj in fo2_objects:
            obj.write(f)

    print("[BGM Export] Done!")

    # write crash.dat if crash data was collected
    if all_crash_data:
        bgm_base = os.path.splitext(filepath)[0]
        if options.get('overwrite_crash_dat', False):
            crash_dat_path = os.path.join(os.path.dirname(filepath), "crash.dat")
        else:
            crash_dat_path = bgm_base + "_crash.dat"
        write_crash_dat(crash_dat_path, all_crash_data)

    # convert TGAs to DDS if requested
    if options.get('convert_tga_to_dds', False):
        dds_fmt     = options.get('dds_format', 'DXT5')
        delete_tgas = options.get('delete_tgas', False)
        export_dir  = os.path.dirname(os.path.abspath(filepath))

        seen_tgas   = set()
        converted   = 0

        for fo2_mat in fo2_materials:
            for tex_name in fo2_mat.texture_names:
                if not tex_name or not tex_name.lower().endswith('.tga'):
                    continue
                tga_path = os.path.join(export_dir, tex_name)
                if not os.path.isfile(tga_path) or tga_path in seen_tgas:
                    continue
                seen_tgas.add(tga_path)

                # find a non-colliding DDS filename
                base     = os.path.splitext(tga_path)[0]
                dds_path = base + '.dds'
                if os.path.exists(dds_path):
                    i = 1
                    while os.path.exists(f"{base}({i}).dds"):
                        i += 1
                    dds_path = f"{base}({i}).dds"

                try:
                    _tga2dds.convert_tga_to_dds(tga_path, dds_path, dds_fmt)
                    converted += 1
                    if delete_tgas:
                        os.remove(tga_path)
                        print(f"[BGM Export] Deleted TGA: {os.path.basename(tga_path)}")
                except Exception as exc:
                    print(f"[BGM Export] TGA→DDS failed for "
                          f"{os.path.basename(tga_path)}: {exc}")

        print(f"[BGM Export] Texture conversion done: "
              f"{converted} TGA(s) → DDS ({dds_fmt})")

    # write body.ini if collision empties exist
    if options.get('export_body_ini', True):
        body_ini_path = os.path.splitext(filepath)[0].rsplit(os.sep, 1)
        body_ini_path = os.path.join(os.path.dirname(filepath), "body.ini")
        write_body_ini(body_ini_path, context, inv_scale)

    # write camera.ini if camera objects exist
    if options.get('export_camera_ini', True):
        cam_ini_path = os.path.join(os.path.dirname(filepath), "camera.ini")
        write_camera_ini(cam_ini_path, context, inv_scale)

    return {'FINISHED'}


def write_body_ini(filepath: str, context, inv_scale: float):
    """Write body.ini collision boxes from the fo2_collision_* empties in the scene.

    Prefers the raw fo2_min/fo2_max custom properties stored on import.
    Falls back to deriving from the empty's world location and scale if those
    properties are absent (i.e. the user created the boxes manually).
    Blender -> FO2 axis: bl(x,y,z) -> fo2(x,z,y)  (y<->z swap, inverse of import).
    """
    box_defs = [
        ('fo2_collision_full',   'CollisionFullMin',   'CollisionFullMax'),
        ('fo2_collision_bottom', 'CollisionBottomMin', 'CollisionBottomMax'),
        ('fo2_collision_top',    'CollisionTopMin',    'CollisionTopMax'),
    ]

    lines = []
    found_any = False

    for obj_name, key_min, key_max in box_defs:
        empty = context.scene.objects.get(obj_name)
        if empty is None:
            print(f"[BGM Export] body.ini: {obj_name} not found, skipping")
            continue

        # prefer stored raw FO2 values (exact round-trip)
        if "fo2_min" in empty and "fo2_max" in empty:
            v_min = tuple(empty["fo2_min"])
            v_max = tuple(empty["fo2_max"])
        else:
            # derive from Blender world location + scale
            # empty_display_size=0.5, scale=dimensions -> half-extents = scale*0.5
            loc   = empty.matrix_world.translation
            s     = empty.matrix_world.to_scale()
            hx, hy, hz = s.x * 0.5, s.y * 0.5, s.z * 0.5
            # blender -> FO2: x stays, y<->z swap, undo global scale
            cx = loc.x * inv_scale
            cy = loc.z * inv_scale   # bl_y in FO2 = bl_z in Blender
            cz = loc.y * inv_scale   # bl_z in FO2 = bl_y in Blender
            ex = hx * inv_scale
            ey = hz * inv_scale
            ez = hy * inv_scale
            v_min = (cx - ex, cy - ey, cz - ez)
            v_max = (cx + ex, cy + ey, cz + ez)

        def fmt(v):
            return f"{{{v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}}}"

        lines.append(f"{key_min}\t = {fmt(v_min)}")
        lines.append(f"{key_max}\t = {fmt(v_max)}")
        lines.append("")
        found_any = True

    if not found_any:
        print("[BGM Export] body.ini: no collision empties found, skipping")
        return

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"[BGM Export] Written body.ini: {filepath}")


def write_camera_ini(filepath: str, context, inv_scale: float):
    """Write camera.ini from fo2_camera_* objects in the scene.

    For every camera object whose name starts with 'fo2_camera_' (and doesn't
    end with '_target'), all required custom properties are validated first.
    Any missing property that can be derived from Blender data is recomputed
    and written back to the object so Blender reflects the state.
    Blender → FO2 position: bl(x,y,z) → fo2(x,z,y), undo global scale.
    """
    import math, re as _re

    def bl_to_fo2_offset(loc, inv_s):
        return (-loc.x * inv_s, loc.z * inv_s, loc.y * inv_s)

    # collect camera objects sorted by index
    cam_objs = []
    for obj in context.scene.objects:
        if obj.type != 'CAMERA':
            continue
        if not obj.name.startswith('fo2_camera_'):
            continue
        if obj.name.endswith('_target'):
            continue
        cam_objs.append(obj)

    if not cam_objs:
        print("[BGM Export] camera.ini: no fo2_camera_* objects found, skipping")
        return

    # sort by index extracted from name
    def _cam_index(obj):
        m = _re.search(r'fo2_camera_(\d+)', obj.name)
        return int(m.group(1)) if m else 9999
    cam_objs.sort(key=_cam_index)

    # sanitize + recompute missing properties
    for obj in cam_objs:
        cam_data = obj.data
        changed  = False

        # determine index from name
        idx = _cam_index(obj)

        # detect target from TrackTo constraint
        track_con  = next((c for c in obj.constraints if c.type == 'TRACK_TO'), None)
        has_target = track_con is not None and track_con.target is not None

        # position offset: derive from world location, undo scale
        pos_fo2 = bl_to_fo2_offset(obj.matrix_world.translation, inv_scale)

        # target offset: derive from target empty's world location
        tgt_fo2 = (0.0, 0.0, 0.0)
        if has_target:
            tgt_fo2 = bl_to_fo2_offset(
                track_con.target.matrix_world.translation, inv_scale)

        # FOV: Blender stores lens angle in radians
        fov_deg = math.degrees(cam_data.angle)

        defaults = {
            'fo2_cam_index':          idx,
            'fo2_animation_type':     1,
            'fo2_position_type':      2,
            'fo2_target_type':        2 if has_target else -1,
            'fo2_zoom_type':          1,
            'fo2_tracker_type':       2 if has_target else 1,
            'fo2_lod_level':          1,
            'fo2_min_display_time':   4.0,
            'fo2_max_display_time':   9.0,
            'fo2_fov':                round(fov_deg, 6),
            'fo2_position_offset':    list(pos_fo2),
            'fo2_target_offset':      list(tgt_fo2),
            'fo2_near_clipping':      cam_data.clip_start,
            'fo2_far_clipping':       cam_data.clip_end,
            'fo2_tracker_stiffness':  [0.25, 0.115, 0.0],
            'fo2_tracker_min_ground': 1.0,
            'fo2_tracker_clamp_ground': 0.3,
        }

        for key, val in defaults.items():
            if key not in obj:
                obj[key] = val
                changed  = True

        # always refresh the derivable spatial properties from current Blender state
        for key, val in [
            ('fo2_cam_index',      idx),
            ('fo2_position_offset', list(pos_fo2)),
            ('fo2_target_offset',   list(tgt_fo2)),
            ('fo2_target_type',     2 if has_target else -1),
            ('fo2_tracker_type',    2 if has_target else 1),
            ('fo2_fov',             round(fov_deg, 6)),
            ('fo2_near_clipping',   cam_data.clip_start),
            ('fo2_far_clipping',    cam_data.clip_end),
        ]:
            if obj[key] != val:
                obj[key] = val
                changed  = True

        if changed:
            obj.update_tag()
            print(f"[BGM Export] camera.ini: updated properties on {obj.name}")

        # ensure viewport rotation is correct regardless of how the object was created.
        # no-target cameras face +Y (90° around X), targeted cameras are driven by the TrackTo constraint so their rotation_euler is ignored
        import math
        if not has_target:
            desired_rot = (math.pi / 2, 0.0, 0.0)
            if tuple(obj.rotation_euler) != desired_rot:
                obj.rotation_euler = desired_rot
                obj.update_tag()

    # write camera.ini
    def fv(v):   return f"{v:.6f}"
    def fvec(v): return "{" + ", ".join(fv(x) for x in v) + "}"

    lines = ["Cameras=", "{"]

    for obj in cam_objs:
        idx = int(obj["fo2_cam_index"])
        pos = list(obj["fo2_position_offset"])
        tgt = list(obj["fo2_target_offset"])
        has_target = int(obj["fo2_target_type"]) != -1
        has_tracker = int(obj["fo2_tracker_type"]) == 2

        lines += [
            f"\t[{idx}]=",
            "\t{",
            f"\t\tAnimationType={int(obj['fo2_animation_type'])},",
            f"\t\tPositionType={int(obj['fo2_position_type'])},",
            f"\t\tTargetType={int(obj['fo2_target_type'])},",
            f"\t\tZoomType={int(obj['fo2_zoom_type'])},",
            f"\t\tTrackerType={int(obj['fo2_tracker_type'])},",
            f"\t\tNearClipping={fv(float(obj['fo2_near_clipping']))},",
            f"\t\tFarClipping={fv(float(obj['fo2_far_clipping']))},",
            f"\t\tMinDisplayTime={fv(float(obj['fo2_min_display_time']))},",
            f"\t\tMaxDisplayTime={fv(float(obj['fo2_max_display_time']))},",
            f"\t\tLodLevel={int(obj['fo2_lod_level'])},",
            "\t\tPositionFrames=",
            "\t\t{",
            "\t\t\t[1]=",
            "\t\t\t{",
            f"\t\t\t\tOffset={fvec(pos)},",
            "\t\t\t},",
            "\t\t},",
        ]

        if has_target:
            lines += [
                "\t\tTargetFrames=",
                "\t\t{",
                "\t\t\t[1]=",
                "\t\t\t{",
                f"\t\t\t\tOffset={fvec(tgt)},",
                "\t\t\t},",
                "\t\t},",
            ]

        lines += [
            "\t\tZoomFrames=",
            "\t\t{",
            "\t\t\t[1]=",
            "\t\t\t{",
            f"\t\t\t\tFOV={fv(float(obj['fo2_fov']))},",
            "\t\t\t},",
            "\t\t},",
        ]

        if has_tracker:
            stiff = list(obj.get("fo2_tracker_stiffness", [0.25, 0.115, 0.0]))
            lines += [
                "\t\tTrackerData=",
                "\t\t{",
                f"\t\t\tStiffness={fvec(stiff)},",
                f"\t\t\tMinGround={fv(float(obj.get('fo2_tracker_min_ground', 1.0)))},",
                f"\t\t\tClampGround={fv(float(obj.get('fo2_tracker_clamp_ground', 0.3)))},",
                "\t\t},",
            ]

        lines.append("\t},")

    lines.append("}")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"[BGM Export] Written camera.ini: {filepath} ({len(cam_objs)} cameras)")


# BLENDER OPERATOR

class ExportBGM(bpy.types.Operator, ExportHelper):
    """Export FlatOut 2 BGM car model"""
    bl_idname = "export_scene.fo2_bgm"
    bl_label = "Export FO2 BGM"
    bl_options = {'PRESET', 'UNDO'}

    filename_ext = ".bgm"
    filter_glob: StringProperty(default="*.bgm", options={'HIDDEN'})

    # transform
    global_scale: FloatProperty(
        name="Scale",
        description="Global scale factor (inverse of import scale)",
        default=1.0,
        min=0.001,
        max=1000.0,
    )

    use_priorities: BoolProperty(
        name="Sort by Material Priority",
        description="Sort materials and surfaces by draw priority "
                    "(lights drawn after body, etc.)",
        default=True,
    )
    auto_triangulate: BoolProperty(
        name="Auto-Triangulate Meshes",
        description="Automatically triangulate any mesh with quads or n-gons before "
                    "export. The original mesh is not modified",
        default=True,
    )

    # texture conversion
    convert_tga_to_dds: BoolProperty(
        name="Convert TGA to DDS",
        description="Convert TGA textures referenced by the exported model back to DDS",
        default=False,
    )
    dds_format: EnumProperty(
        name="DDS Format",
        description="DDS compression format to use",
        items=[
            ('DXT1', "DXT1", "No alpha / 1-bit alpha (smallest file size)"),
            ('DXT3', "DXT3", "Sharp / binary alpha (explicit 4-bit alpha per pixel)"),
            ('DXT5', "DXT5", "Smooth alpha gradients (interpolated, best quality)"),
        ],
        default='DXT3',
    )
    delete_tgas: BoolProperty(
        name="Delete Converted TGAs",
        description="Delete the source TGA files after successful DDS conversion",
        default=False,
    )

    # collision
    export_body_ini: BoolProperty(
        name="Export Collision Boxes (body.ini)",
        default=True,
        description="Write body.ini next to the exported BGM file using the "
                    "fo2_collision_full / bottom / top empties in the scene",
    )

    # crash dat
    overwrite_crash_dat: BoolProperty(
        name="Overwrite crash.dat",
        default=False,
        description="Write crash data to crash.dat in the same folder, "
                    "overwriting any existing file",
    )

    # cameras
    export_camera_ini: BoolProperty(
        name="Export Cameras (camera.ini)",
        default=True,
        description="Write camera.ini next to the exported BGM file from "
                    "fo2_camera_* objects. Missing properties are recomputed "
                    "from Blender camera data and written back to the objects",
    )

    def draw(self, context):
        layout = self.layout

        # transform
        box = layout.box()
        box.label(text="Transform", icon='ORIENTATION_GLOBAL')
        box.prop(self, "global_scale")

        # options
        box = layout.box()
        box.label(text="Options", icon='PREFERENCES')
        box.prop(self, "use_priorities")
        box.prop(self, "auto_triangulate")

        # texture conversion
        box = layout.box()
        box.label(text="Texture Conversion", icon='IMAGE_DATA')
        box.prop(self, "convert_tga_to_dds")
        row = box.row()
        row.enabled = self.convert_tga_to_dds
        row.prop(self, "dds_format")
        row = box.row()
        row.enabled = self.convert_tga_to_dds
        row.prop(self, "delete_tgas")

        # collision
        box = layout.box()
        box.label(text="Collision", icon='MOD_WIREFRAME')
        box.prop(self, "export_body_ini")
        box.prop(self, "overwrite_crash_dat")

        # cameras
        box = layout.box()
        box.label(text="Cameras", icon='CAMERA_DATA')
        box.prop(self, "export_camera_ini")

    def execute(self, context):
        options = {
            'global_scale': self.global_scale,
            'use_priorities': self.use_priorities,
            'auto_triangulate': self.auto_triangulate,
            'convert_tga_to_dds': self.convert_tga_to_dds,
            'dds_format': self.dds_format,
            'delete_tgas': self.delete_tgas,
            'export_body_ini': self.export_body_ini,
            'overwrite_crash_dat': self.overwrite_crash_dat,
            'export_camera_ini': self.export_camera_ini,
        }
        result = write_bgm(self.filepath, context, options)
        if result == {'CANCELLED'}:
            self.report({'ERROR'}, "No objects found to export")
        else:
            self.report({'INFO'},
                        f"Exported BGM to {os.path.basename(self.filepath)}")
        return result


# REGISTRATION

def menu_func_export(self, context):
    self.layout.operator(ExportBGM.bl_idname, text="FlatOut 2 Car BGM (.bgm)")


def register():
    bpy.utils.register_class(ExportBGM)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    bpy.utils.unregister_class(ExportBGM)


if __name__ == "__main__":
    register()
