# FlatOut Blender Tools — https://github.com/RavenDS/flatout-blender-tools
# __init__.py — Blender add-on entry point: imports all submodules,
# registers the ImportBGM operator and the FlatOut Shader panel.

bl_info = {
    "name": "FlatOut BGM Import (Car)",
    "author": "ravenDS",
    "version": (1, 6, 1),
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

from .bgm_common import (
    # dataclasses
    BGMMaterial, VertexBuffer, IndexBuffer, Surface, Model,
    BGMMesh, BGMObject, CrashWeight, CrashSurface, CrashNode, ParsedVertex,
    # parsers / helpers
    BinaryReader, BGMParser, parse_crash_dat,
    extract_vertices, extract_indices,
    fo2_matrix_to_blender, build_axis_matrix,
    find_texture_file, create_blender_material,
    extract_crash_vertices, build_blender_meshes,
    # texture helpers used by submodules (re-exported for convenience)
)
from .bgm_ps2 import (
    _detect_ps2_bgm, _detect_psp_bgm,
    PS2BGMParser, build_blender_meshes_ps2,
)
from .bgm_psp import (
    PSPBGMParser, build_blender_meshes_psp,
)
from .bgm_xbox import (
    _detect_xbox_bgm,
    XboxBGMParser, build_blender_meshes_xbox,
)
from . import dds2tga as _dds2tga
from . import dds_normal as _dds_normal


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

        # parse BGM — detect PS2 / PSP / Xbox vs PC format before choosing the parser
        if _detect_ps2_bgm(filepath):
            parser = PS2BGMParser(filepath)
            if not parser.parse():
                self.report({'ERROR'}, f"Failed to parse PS2 BGM file: {filepath}")
                return {'CANCELLED'}
        elif _detect_psp_bgm(filepath):
            parser = PSPBGMParser(filepath)
            if not parser.parse():
                self.report({'ERROR'}, f"Failed to parse PSP BGM file: {filepath}")
                return {'CANCELLED'}
        elif _detect_xbox_bgm(filepath):
            parser = XboxBGMParser(filepath)
            if not parser.parse():
                self.report({'ERROR'}, f"Failed to parse Xbox BGM file: {filepath}")
                return {'CANCELLED'}
        else:
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

        if isinstance(parser, PS2BGMParser):
            objects = build_blender_meshes_ps2(context, parser, options)
        elif isinstance(parser, PSPBGMParser):
            objects = build_blender_meshes_psp(context, parser, options)
        elif isinstance(parser, XboxBGMParser):
            objects = build_blender_meshes_xbox(context, parser, options)
        else:
            objects = build_blender_meshes(context, parser, options)

        if not objects:
            self.report({'WARNING'}, "No meshes were imported")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Imported {len(objects)} objects from {os.path.basename(filepath)}")

        # Set scene game mode to match the imported file
        try:
            if getattr(parser, 'is_fouc', False):
                context.scene.fo2_game_mode = 'FOUC'
            elif isinstance(parser, (PS2BGMParser, PSPBGMParser, XboxBGMParser)):
                context.scene.fo2_game_mode = 'FO2'
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
