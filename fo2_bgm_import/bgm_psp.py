"""
bgm_psp.py — FlatOut Head On (PSP) BGM import support.
Part of FlatOut Blender Tools — https://github.com/RavenDS/flatout-blender-tools

Covers:
  • PSPSurface, PSPBGMParser
  • _psp_infer_stride, _psp_extract_surface_tris
  • build_blender_meshes_psp
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

# ─────────────────────────── PSP BGM SUPPORT ────────────────────────────────
#
# FlatOut Head On (PSP) BGM files share the same material / model / mesh /
# object layout as PC FO2, but geometry is stored as PSP GE display list
# commands rather than indexed vertex / index buffers.
#
# Detection: version == 0x20000 AND streams are type-3-only (no type 1 or 2).
# (PC FO2 also has version 0x20000 but uses type-1 + type-2 streams.)
#
# Surface record: 8 × uint32 = 32 bytes
#   [0] flags    = 0x1000
#   [1] mid      = material index
#   [2] nv       = vertex count
#   [3] nb       = 1
#   [4] f4       = 0x27
#   [5] f5       = 0
#   [6] boff     = byte offset into blob
#   [7] bsz      = byte size of GE chunk
#
# GE chunk layout
#   offset  0: ORIGIN   0x14000000
#   offset  4: BASE(0)  0x10000000
#   offset  8: VADR(20) 0x01000014
#   offset 12: PRIM     0x04_pp_00_nn  pp=prim_type  nn=vertex_count
#   offset 16: RET      0x0b000000
#   offset 20: vertex data (nv × stride bytes)
#   padding to 16-byte alignment
#
# Vertex formats
#   Standard  14 bytes: uint16 u,v | int8 nx,ny,nz,pad | int16 px,py,pz
#   Shadow     6 bytes: int16 px,py,pz
#
# UV decoding: float_uv = (raw_uint16 − 16384) / 2048.0
#   (inverse of encoder: raw = round(PC_UV × 2048) + 16384)
#
# Primitive types (bits 23:16 of PRIM command word)
#   3 = GU_TRIANGLES      (from PC poly_mode 4, NvTriStrip converted)
#   4 = GU_TRIANGLE_STRIP (from PC poly_mode 5, passed through)
#
# Winding: same Y↔Z reflection issue as PS2 — compensated by swapping v0↔v1
# on even strip positions (same rule as _ps2_triangles_from_adc after fix).


@dataclass
class PSPSurface:
    flags:       int = 0
    material_id: int = 0
    nv:          int = 0
    nb:          int = 1
    f4:          int = 0
    f5:          int = 0
    blob_offset: int = 0
    blob_size:   int = 0





class PSPBGMParser:
    """Parser for FlatOut Head On (PSP) BGM files.

    Material, model, mesh, and object sections are binary-identical to PC BGM.
    The stream section holds a single type-3 entry carrying the raw GE display
    list blob (fc=0, vc=blob_size, vs=1).  Surface records are 8 × uint32.
    """

    def __init__(self, filepath: str):
        self.filepath     = filepath
        self.version      = 0
        self.materials: list = []
        self.blob:      bytes = b''
        self.psp_surfaces: list = []   # list[PSPSurface]
        self.models:    list = []      # list[Model]
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
            mat   = BGMMaterial()
            ident = r.u32()
            if ident != 0x4354414D:
                print(f'[PSP BGM] ERROR: Expected MATC at material {i}')
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

        # ── stream: exactly 1 type-3 entry, fc=0, vc=blob_size, vs=1 ────────
        ns = r.u32()
        for _ in range(ns):
            dt = r.i32()
            if dt == 3:
                r.i32()                   # fc = 0
                vc = r.u32(); vs = r.u32()
                self.blob = r.read(vc * vs)
            elif dt == 1:
                r.i32(); vc = r.u32(); vs = r.u32(); r.u32()
                r.read(vc * vs)
            elif dt == 2:
                r.i32(); ic = r.u32()
                r.read(ic * 2)

        # ── surfaces — PSP format: exactly 8 × uint32 = 32 bytes each ───────
        nsf = r.u32()
        for _ in range(nsf):
            vals = struct.unpack_from('<8I', raw, r.pos)
            r.read(32)
            self.psp_surfaces.append(PSPSurface(
                flags       = vals[0],
                material_id = vals[1],
                nv          = vals[2],
                nb          = vals[3],
                f4          = vals[4],
                f5          = vals[5],
                blob_offset = vals[6],
                blob_size   = vals[7],
            ))

        # ── models (identical to BGMParser) ─────────────────────────────────
        nmod = r.u32()
        for i in range(nmod):
            m     = Model()
            ident = r.u32()
            if ident != 0x444F4D42:
                print(f'[PSP BGM] ERROR: Expected BMOD at model {i}')
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
                print(f'[PSP BGM] ERROR: Expected MESH at mesh {i}')
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
                print(f'[PSP BGM] ERROR: Expected OBJC at object {i}')
                return False
            obj.name1  = r.read_string()
            obj.name2  = r.read_string()
            obj.flags  = r.u32()
            obj.matrix = list(struct.unpack_from('<16f', r.read(64)))
            self.objects.append(obj)

        print(f'[PSP BGM] Parsed {self.filepath}: version=0x{self.version:X}, '
              f'{len(self.materials)} mats, {len(self.meshes)} meshes, '
              f'{len(self.psp_surfaces)} surfaces, {len(self.objects)} objects')
        return True


def _psp_infer_stride(nv: int, bsz: int) -> int:
    """Infer vertex stride (14 = standard, 6 = shadow) from blob_size and nv.

    GE chunk = 20-byte header + nv*stride + padding-to-16B.
    Try both known strides and return whichever matches bsz exactly.
    Falls back to 14 (standard) if neither matches.
    """
    for stride in (14, 6):
        raw = 20 + nv * stride
        pad = (16 - raw % 16) % 16
        if raw + pad == bsz:
            return stride
    return 14  # fallback


def _psp_extract_surface_tris(blob: bytes, surf: PSPSurface) -> list:
    """Decode one PSP GE surface chunk into a list of triangles.

    Each triangle is ((pos,uv,norm), (pos,uv,norm), (pos,uv,norm)).
    Winding is reversed to compensate for the Y↔Z axis-swap reflection
    (same correction as _ps2_triangles_from_adc):
      even strip position i → emit (v[i+1], v[i], v[i+2])
      odd  strip position i → emit (v[i],   v[i+1], v[i+2])
    Triangle-list surfaces use (v[i+1], v[i], v[i+2]) for every triplet.

    UV decode: float_uv = (raw_uint16 − 16384) / 2048.0
    """
    if surf.nv == 0 or surf.blob_size < 20:
        return []

    off = surf.blob_offset

    # Read prim type from PRIM command at header offset 12
    # Word = 0x04_pp_00_nn:  pp = bits[23:16], nn = bits[15:0]
    prim_word = struct.unpack_from('<I', blob, off + 12)[0]
    prim_type = (prim_word >> 16) & 0xFF   # 3=GU_TRIANGLES, 4=GU_TRIANGLE_STRIP

    stride    = _psp_infer_stride(surf.nv, surf.blob_size)
    is_shadow = (stride == 6)

    # Decode vertex data (starts at header offset 20)
    vdata_start = off + 20
    verts = []
    for i in range(surf.nv):
        voff = vdata_start + i * stride
        if voff + stride > len(blob):
            break
        if is_shadow:
            px, py, pz = struct.unpack_from('<3h', blob, voff)
            verts.append(((px / 1024.0, py / 1024.0, pz / 1024.0),
                          (0.0, 0.0),
                          (0.0, 0.0, 1.0)))
        else:
            # uint16 u, v | int8 nx,ny,nz,pad | int16 px,py,pz
            u_raw, v_raw            = struct.unpack_from('<2H', blob, voff)
            nx, ny, nz, _pad        = struct.unpack_from('<4b', blob, voff + 4)
            px, py, pz              = struct.unpack_from('<3h', blob, voff + 8)
            verts.append(((px / 1024.0,  py / 1024.0,  pz / 1024.0),
                          ((u_raw - 16384) / 2048.0, (v_raw - 16384) / 2048.0),
                          (nx / 127.0, ny / 127.0, nz / 127.0)))

    # Decode triangles — reverse winding for Y↔Z reflection compensation
    tris = []
    if prim_type == 3:   # GU_TRIANGLES: fixed groups of 3
        for i in range(0, len(verts) - 2, 3):
            v0, v1, v2 = verts[i], verts[i + 1], verts[i + 2]
            if v0[0] == v1[0] or v1[0] == v2[0] or v0[0] == v2[0]:
                continue  # degenerate
            tris.append((v1, v0, v2))  # reversed winding
    elif prim_type == 4:  # GU_TRIANGLE_STRIP
        for i in range(len(verts) - 2):
            v0, v1, v2 = verts[i], verts[i + 1], verts[i + 2]
            if v0[0] == v1[0] or v1[0] == v2[0] or v0[0] == v2[0]:
                continue  # degenerate (strip join vertex)
            # even i → swap v0↔v1; odd i → keep order (mirrors PS2 winding fix)
            if i % 2 == 0:
                tris.append((v1, v0, v2))
            else:
                tris.append((v0, v1, v2))
    return tris


def build_blender_meshes_psp(context, parser: 'PSPBGMParser', options: dict):
    """Build Blender mesh objects from a parsed PSP (FlatOut Head On) BGM file.

    Creates the same FO2 Body / FO2 Body Dummies collection structure and uses
    the same FO2→Blender axis transform (Y↔Z swap) as the PC and PS2 imports.
    Geometry is decoded from PSP GE display list chunks.
    No crash.dat exists for PSP files.
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
    root_empty['bgm_is_fouc'] = False
    root_empty['bgm_is_fo1']  = False
    root_empty['bgm_is_psp']  = True
    root_empty['bgm_version'] = parser.version
    fo2_body_coll.objects.link(root_empty)

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

    # ── object empties (dummies) — identical transform logic to PC/PS2 import
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
        print('[PSP BGM] Import complete: 0 mesh objects created (import_body=False)')
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
        all_verts            = []
        all_normals          = []
        all_face_uvs         = []
        all_faces            = []
        all_face_mat_indices = []
        mat_index_map        = {}
        mesh_materials       = []

        for surf_id in model.surface_ids:
            if surf_id < 0 or surf_id >= len(parser.psp_surfaces):
                continue
            surf = parser.psp_surfaces[surf_id]
            if surf.blob_size == 0 or surf.nv == 0:
                continue

            mat_id = surf.material_id
            if mat_id not in mat_index_map:
                mat_index_map[mat_id] = len(mesh_materials)
                mesh_materials.append(blender_materials.get(mat_id))
            local_mat_idx = mat_index_map[mat_id]

            tris = _psp_extract_surface_tris(parser.blob, surf)
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
                    face_uvs_t.append((v_uv[0], 1.0 - v_uv[1]))  # flip V
                all_faces.append((base, base + 1, base + 2))
                all_face_mat_indices.append(local_mat_idx)
                all_face_uvs.append(tuple(face_uvs_t))

        if not all_faces:
            continue

        mesh_name = bgm_mesh.name1 if bgm_mesh.name1 else 'bgm_psp_mesh'
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

        bl_obj.parent        = root_empty
        bl_obj['bgm_flags']  = bgm_mesh.flags
        bl_obj['bgm_group']  = bgm_mesh.group
        bl_obj['bgm_name2']  = bgm_mesh.name2
        bl_obj['bgm_is_psp'] = True

    bpy.ops.object.select_all(action='DESELECT')
    for obj in created_objects:
        obj.select_set(True)
    if created_objects:
        context.view_layer.objects.active = created_objects[0]

    print(f'[PSP BGM] Import complete: {len(created_objects)} mesh objects created')
    return created_objects


