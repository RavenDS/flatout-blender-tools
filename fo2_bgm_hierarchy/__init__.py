"""
FlatOut BGM – Hierarchy Reorganiser
Converts any existing scene hierarchy into the flat layout the BGM exporter expects,
and stamps the appropriate game-mode metadata so the shader panel and exporter both
auto-select the correct target format.

Resulting hierarchy:
  FO2 Body collection
    fo2_body  (EMPTY, world origin)
      <mesh objects — one per car part>
      fo2_body_dummies  (EMPTY)
        <dummy empties — OBJC entries>

  FO2 Body Crash collection  (only if crash meshes exist)
    fo2_body_crash  (EMPTY, world origin)
      <mesh objects named *_crash>

Three reorganise operators are available in View3D > Object:
  • FO2: Reorganise for FlatOut 1
  • FO2: Reorganise for FlatOut 2
  • FO2: Reorganise for FlatOut UC
"""

bl_info = {
    "name":        "FlatOut BGM Hierarchy Reorganiser",
    "author":      "ravenDS",
    "version":     (2, 0, 1),
    "blender":     (3, 6, 0),
    "location":    "View3D > Object > FO2: Reorganise",
    "description": "Flatten any scene hierarchy into the layout the BGM exporter expects",
    "category":    "Import-Export",
}

import bpy
import re
import os
import math


# Shader / material property helpers

SHADER_CAR_METAL      = 8
SHADER_CAR_BODY       = 5
SHADER_CAR_WINDOW     = 6
SHADER_CAR_DIFFUSE    = 7
SHADER_CAR_TIRE       = 9
SHADER_CAR_LIGHTS     = 10
SHADER_CAR_SHEAR      = 11
SHADER_CAR_SCALE      = 12
SHADER_SHADOW_PROJECT = 13
SHADER_SKINNING       = 26
# FOUC-only shaders (FO1/FO2 use SHADER_CAR_DIFFUSE for these material names)
SHADER_FOUC_INTERIOR  = 43
SHADER_FOUC_TIRE      = 44


def _get_texture_name_from_material(bl_mat) -> str:
    """Extract diffuse texture filename from the node tree."""
    if not bl_mat or not bl_mat.use_nodes:
        return ""
    for node in bl_mat.node_tree.nodes:
        if node.type == 'TEX_IMAGE' and node.image:
            fp   = (node.image.filepath or "").replace('\\', '/').lstrip('/')
            name = fp.rsplit('/', 1)[-1] if fp else node.image.name
            if name:
                name = re.sub(r'\.\d{3}$', '', name)
                base, ext = os.path.splitext(name)
                return base + '.tga'
    return ""


def _get_shader_for_material(mat_name: str, tex_name: str,
                              game_mode: str = 'FO2') -> tuple:
    """Return (shader_id, alpha, v92, tex_override).

    Fallback for materials with no shader set.
    """
    name  = mat_name.lower()
    shader, alpha, v92 = SHADER_CAR_METAL, 0, 0
    tex_override = None
    is_fouc = (game_mode == 'FOUC')

    if name.startswith("shadow") or name.endswith("shadow"):
        shader       = SHADER_SHADOW_PROJECT
        tex_override = "shadow.tga"
    elif name.startswith("body"):
        shader       = SHADER_CAR_BODY
        tex_override = "skin1.tga"
    elif name.startswith("interior"):
        shader = SHADER_FOUC_INTERIOR if is_fouc else SHADER_CAR_DIFFUSE
    elif name.startswith("grille"):
        shader, alpha = SHADER_CAR_DIFFUSE, 1
    elif name.startswith("window"):
        shader = SHADER_CAR_WINDOW
    elif name.startswith("shear"):
        # also catches shearhock / shearshock — vanilla puts all of them on 11
        shader = SHADER_CAR_SHEAR
    elif name.startswith("scaleshock") or name.startswith("scalespring"):
        shader, alpha = SHADER_CAR_SCALE, 0
    elif name.startswith("scale"):
        shader = SHADER_CAR_SCALE
    elif name.startswith("shock") or name.startswith("spring"):
        # bare shock/spring are shear (11) in vanilla, not scale (12)
        shader = SHADER_CAR_SHEAR
    elif name.startswith("tire"):
        shader = (SHADER_FOUC_TIRE if is_fouc else
                  (SHADER_CAR_TIRE if game_mode == 'FO1' else SHADER_CAR_DIFFUSE))
    elif name.startswith("rim"):
        shader, alpha = SHADER_CAR_TIRE, 1
    elif name.startswith("light"):
        shader = SHADER_CAR_LIGHTS
        v92    = 0 if game_mode == 'FO1' else 2
    elif name.startswith("terrain") or name.startswith("groundplane"):
        shader, alpha = SHADER_CAR_DIFFUSE, 1
    elif name.startswith("male") or name.startswith("female"):
        shader = SHADER_SKINNING

    tex_lower = tex_name.lower() if tex_name else ""
    if tex_lower in ("lights.tga", "windows.tga", "shock.tga"):
        alpha = 1
    if name.endswith("_alpha"):
        alpha = 1
    if name.endswith("_noalpha"):
        alpha = 0

    return shader, alpha, v92, tex_override


def _sanitize_mesh_and_material_props(mesh_objects, game_mode: str = 'FO2'):
    """Ensure all BGM custom properties exist on meshes and their materials.

    game_mode controls version-specific defaults (e.g. v92 on light materials).
    """
    for obj in mesh_objects:
        changed = False
        if "bgm_flags" not in obj or obj["bgm_flags"] is None:
            obj["bgm_flags"] = 0;  changed = True
        if "bgm_group" not in obj or obj["bgm_group"] is None:
            obj["bgm_group"] = -1; changed = True
        if "bgm_name2" not in obj:
            if not obj.name.endswith("_crash"):
                obj["bgm_name2"] = ""; changed = True
        elif obj.name.endswith("_crash"):
            del obj["bgm_name2"]
            changed = True
        obj["bgm_is_crash"] = obj.name.endswith("_crash")
        if changed:
            obj.update_tag()

    seen: set = set()
    for obj in mesh_objects:
        for slot in obj.material_slots:
            bl_mat = slot.material
            if not bl_mat or id(bl_mat) in seen:
                continue
            seen.add(id(bl_mat))

            tex_name = (bl_mat.get("bgm_texture", "")
                        or _get_texture_name_from_material(bl_mat))
            if tex_name:
                base, ext = os.path.splitext(tex_name)
                if ext.lower() != '.tga':
                    tex_name = base + '.tga'
            else:
                tex_name = re.sub(r'\.\d{3}$', '', bl_mat.name) + '.tga'

            changed = False

            if "bgm_alpha" not in bl_mat:
                bl_mat["bgm_alpha"] = 0; changed = True
            if "bgm_num_textures" not in bl_mat:
                bl_mat["bgm_num_textures"] = 1; changed = True

            if "bgm_shader_id" not in bl_mat:
                # No stored shader — infer from name with game-mode-correct defaults
                clean = re.sub(r'\.\d{3}$', '', bl_mat.name)
                shader_id, alpha, v92, tex_override = _get_shader_for_material(
                    clean, tex_name, game_mode=game_mode)
                if tex_override:
                    tex_name = tex_override
                bl_mat["bgm_shader_id"] = shader_id
                bl_mat["bgm_alpha"]     = alpha
                bl_mat["bgm_v92"]       = v92
                changed = True
            else:
                # Already has a shader — update v92 on lights if it was set to the wrong game-mode default (0 vs 2).
                try:
                    sid = int(bl_mat["bgm_shader_id"])
                    if sid == SHADER_CAR_LIGHTS:
                        correct_v92 = 0 if game_mode == 'FO1' else 2
                        if bl_mat.get("bgm_v92", correct_v92) != correct_v92:
                            bl_mat["bgm_v92"] = correct_v92
                            changed = True
                except (TypeError, ValueError):
                    pass

            if "bgm_texture"    not in bl_mat:
                bl_mat["bgm_texture"]   = tex_name; changed = True
            if "bgm_texture_0"  not in bl_mat:
                bl_mat["bgm_texture_0"] = tex_name; changed = True
            if "bgm_texture_1"  not in bl_mat:
                bl_mat["bgm_texture_1"] = "";       changed = True
            if "bgm_texture_2"  not in bl_mat:
                bl_mat["bgm_texture_2"] = "";       changed = True
            if "bgm_use_colormap" not in bl_mat:
                bl_mat["bgm_use_colormap"] = 0;     changed = True
            if "bgm_v102" not in bl_mat:
                bl_mat["bgm_v102"] = 0;             changed = True
            if "bgm_v74"  not in bl_mat:
                bl_mat["bgm_v74"]  = 0;             changed = True
            if "bgm_v92"  not in bl_mat:
                bl_mat["bgm_v92"]  = 0;             changed = True

            if changed:
                print(f"[FO2 Reorganise] Initialised BGM props on material: "
                      f"{bl_mat.name} (game_mode={game_mode})")

            # Sync RNA properties after all custom props are written
            try:
                bl_mat.fo2_shader_id = str(int(bl_mat.get("bgm_shader_id", 8)))
            except Exception:
                pass
            try:
                bl_mat.fo2_texture = str(bl_mat.get("bgm_texture", ""))
            except Exception:
                pass


# Hierarchy helpers

def depth_of(obj):
    d, p = 0, obj.parent
    while p:
        d += 1; p = p.parent
    return d


def collect_all_descendants(obj):
    result = []
    for child in obj.children:
        result.append(child)
        result.extend(collect_all_descendants(child))
    return result


def collect_leaf_meshes(obj):
    meshes = []
    for child in obj.children:
        if child.type == 'MESH' and child.data and len(child.data.vertices) > 0:
            meshes.append(child)
        meshes.extend(collect_leaf_meshes(child))
    return meshes


def is_crash(obj):
    cur = obj
    while cur:
        if '_crash' in cur.name:
            return True
        cur = cur.parent
    return False


def base_name(name):
    return re.sub(r'\.\d{3}$', '', name)


def ensure_collection(scene, name):
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
    if coll.name not in scene.collection.children:
        scene.collection.children.link(coll)
    return coll


def link_to_collection(obj, coll):
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    coll.objects.link(obj)


# Core reorganise

def do_reorganise_scene(game_mode: str = 'FO2'):
    """Flatten the scene hierarchy and stamp game-mode metadata.

    game_mode: 'FO1' | 'FO2' | 'FOUC'

    Metadata written to fo2_body:
      bgm_is_fo1   – True when game_mode == 'FO1' (read by exporter invoke)
      bgm_is_fouc  – True when game_mode == 'FOUC' (read by exporter invoke)
      bgm_version  – header version constant (informational)

    scene.fo2_game_mode is also set so the Shader panel shows the right list.
    """
    scene   = bpy.context.scene
    context = bpy.context

    is_fo1  = (game_mode == 'FO1')
    is_fouc = (game_mode == 'FOUC')
    # OBJC flags: FO1 originals use 0x0, FO2/FOUC use 0xE0F9
    obj_flags = 0x0 if is_fo1 else 0xE0F9

    if context.active_object and context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    # build / reuse empties + collections
    fo2_body_coll    = ensure_collection(scene, "FO2 Body")
    fo2_crash_coll   = ensure_collection(scene, "FO2 Body Crash")
    fo2_dummies_coll = ensure_collection(scene, "FO2 Body Dummies")

    fo2_body = scene.objects.get("fo2_body")
    if fo2_body is None:
        fo2_body = bpy.data.objects.new("fo2_body", None)
        fo2_body.empty_display_type = 'PLAIN_AXES'
        fo2_body.empty_display_size = 0.5
    link_to_collection(fo2_body, fo2_body_coll)
    fo2_body.parent = None

    # stamp game-mode metadata on root empty
    fo2_body["bgm_is_fo1"]  = is_fo1
    fo2_body["bgm_is_fouc"] = is_fouc
    fo2_body["bgm_version"] = (0x00010004 if is_fo1 else 0x20000)

    fo2_crash = scene.objects.get("fo2_body_crash")
    if fo2_crash is None:
        fo2_crash = bpy.data.objects.new("fo2_body_crash", None)
        fo2_crash.empty_display_type = 'PLAIN_AXES'
        fo2_crash.empty_display_size = 0.5
    link_to_collection(fo2_crash, fo2_crash_coll)
    fo2_crash.parent = None

    fo2_dummies = scene.objects.get("fo2_body_dummies")
    if fo2_dummies is None:
        fo2_dummies = bpy.data.objects.new("fo2_body_dummies", None)
        fo2_dummies.empty_display_type = 'PLAIN_AXES'
        fo2_dummies.empty_display_size = 0.5
    link_to_collection(fo2_dummies, fo2_dummies_coll)
    fo2_dummies.parent = fo2_body

    skip = {fo2_body, fo2_crash, fo2_dummies}
    skip_prefixes = ('fo2_collision_', 'fo2_camera_', 'fo2_body_lights',
                     'fo2_body_cameras', 'fo2_body_collision')

    # handle scene-level Objects empty (dummies from imported hierarchies)
    for obj in list(scene.objects):
        if obj in skip:
            continue
        if obj.type == 'EMPTY' and re.sub(r'\.\d{3}$', '', obj.name) == 'Objects':
            print(f"[FO2 Reorganise] Moving {len(list(obj.children))} dummies "
                  f"from scene-level 'Objects'")
            for child in list(obj.children):
                if child.type == 'EMPTY':
                    world = child.matrix_world.copy()
                    child.parent = fo2_dummies
                    child.matrix_world = world
                    child.name = base_name(child.name)
                    child["bgm_obj_flags"] = obj_flags
                    link_to_collection(child, fo2_dummies_coll)
            bpy.data.objects.remove(obj, do_unlink=True)
            break

    # find source root or re-flatten in place
    source_root = None
    best_children = 0
    for obj in scene.objects:
        if obj in skip:
            continue
        if any(obj.name.startswith(p) for p in skip_prefixes):
            continue
        if obj.parent is not None:
            continue
        n = len(list(obj.children))
        if n > best_children:
            source_root = obj
            best_children = n

    if source_root is not None:
        groups = list(source_root.children)
        print(f"[FO2 Reorganise] Source root: '{source_root.name}' "
              f"with {best_children} direct children ({game_mode})")
    else:
        groups = list(fo2_body.children)
        source_root = None
        print(f"[FO2 Reorganise] No external root — "
              f"re-flattening {len(groups)} children of fo2_body ({game_mode})")

    renamed = 0
    removed = 0

    for group in groups:
        if group in skip:
            continue
        if any(group.name.startswith(p) for p in skip_prefixes):
            continue

        group_base  = base_name(group.name)
        all_desc    = collect_all_descendants(group)
        leaf_meshes = collect_leaf_meshes(group)

        group_is_mesh = (group.type == 'MESH' and group.data
                         and len(group.data.vertices) > 0)

        if (group_is_mesh and group.parent == fo2_body
                and group.name == group_base):
            continue

        inner_objects_empty = next(
            (o for o in all_desc
             if o.type == 'EMPTY'
             and re.sub(r'\.\d{3}$', '', o.name) == 'Objects'), None)
        inner_dummies = []
        if inner_objects_empty:
            inner_dummies = [c for c in inner_objects_empty.children
                             if c.type == 'EMPTY']
        promoted_dummy_ids = set(id(o) for o in inner_dummies)
        standalone_dummies = []
        for o in all_desc:
            if o.type != 'EMPTY' or id(o) in promoted_dummy_ids:
                continue
            if o == inner_objects_empty:
                continue
            if (len(collect_leaf_meshes(o)) == 0
                    and not any(c.type == 'MESH' for c in o.children)):
                standalone_dummies.append(o)
        all_dummies = inner_dummies + standalone_dummies

        if not leaf_meshes and not group_is_mesh:
            print(f"[FO2 Reorganise] '{group_base}': no geometry, skipping")
            continue

        regular_meshes = [m for m in leaf_meshes if not is_crash(m)]
        crash_meshes   = [m for m in leaf_meshes if is_crash(m)]
        if group_is_mesh:
            (crash_meshes if is_crash(group) else regular_meshes).insert(0, group)

        print(f"[FO2 Reorganise] '{group_base}': "
              f"{len(regular_meshes)} regular, {len(crash_meshes)} crash, "
              f"{len(all_dummies)} dummies")

        for mesh_obj in regular_meshes:
            world = mesh_obj.matrix_world.copy()
            mesh_obj.parent = fo2_body
            mesh_obj.matrix_world = world
            mesh_obj.name = group_base
            link_to_collection(mesh_obj, fo2_body_coll)
            renamed += 1

        for mesh_obj in crash_meshes:
            world = mesh_obj.matrix_world.copy()
            mesh_obj.parent = fo2_crash
            mesh_obj.matrix_world = world
            n = group_base if group_base.endswith('_crash') else group_base + '_crash'
            mesh_obj.name = n
            link_to_collection(mesh_obj, fo2_crash_coll)
            renamed += 1

        for dummy in all_dummies:
            world = dummy.matrix_world.copy()
            dummy.parent = fo2_dummies
            dummy.matrix_world = world
            dummy.name = base_name(dummy.name)
            dummy["bgm_obj_flags"] = obj_flags
            link_to_collection(dummy, fo2_dummies_coll)

        promoted = set(id(o) for o in regular_meshes + crash_meshes + all_dummies)
        to_remove = [o for o in all_desc
                     if id(o) not in promoted and o not in skip]
        if not group_is_mesh and group not in skip:
            to_remove.append(group)
        to_remove.sort(key=depth_of, reverse=True)
        for obj in to_remove:
            mesh_data = obj.data if obj.type == 'MESH' else None
            try:
                bpy.data.objects.remove(obj, do_unlink=True)
                removed += 1
            except ReferenceError:
                pass
            if mesh_data and mesh_data.users == 0:
                try:
                    bpy.data.meshes.remove(mesh_data)
                except ReferenceError:
                    pass

    if source_root is not None and source_root not in skip:
        try:
            bpy.data.objects.remove(source_root, do_unlink=True)
            removed += 1
        except ReferenceError:
            pass

    # stray crash meshes under fo2_body → fo2_body_crash
    for child in list(fo2_body.children):
        if child.type == 'MESH' and '_crash' in child.name:
            world = child.matrix_world.copy()
            child.parent = fo2_crash
            child.matrix_world = world
            link_to_collection(child, fo2_crash_coll)

    # remove fo2_body_crash if empty
    if not list(fo2_crash.children):
        bpy.data.objects.remove(fo2_crash, do_unlink=True)
        print("[FO2 Reorganise] No crash meshes — removed fo2_body_crash")

    # merge same-name children
    _merge_same_name_children(fo2_body, fo2_body_coll)
    if scene.objects.get("fo2_body_crash"):
        _merge_same_name_children(
            scene.objects["fo2_body_crash"], fo2_crash_coll)

    # strip .001/.002/etc — collision-safe rename
    for root_obj in [fo2_body, scene.objects.get("fo2_body_crash")]:
        if root_obj is None:
            continue
        children = [c for c in root_obj.children if c not in skip]
        targets = []
        for child in children:
            clean_obj  = re.sub(r'\.\d{3}$', '', child.name)
            clean_data = (re.sub(r'\.\d{3}$', '', child.data.name)
                          if child.type == 'MESH' and child.data else None)
            targets.append((child, clean_obj, clean_data))
        for i, (child, _, _) in enumerate(targets):
            child.name = f"__fo2tmp_{i}__"
            if child.type == 'MESH' and child.data:
                child.data.name = f"__fo2tmpd_{i}__"
        for child, clean_obj, clean_data in targets:
            child.name = clean_obj
            if child.type == 'MESH' and child.data and clean_data:
                child.data.name = clean_data

    # rename mesh data to match object name
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.parent and obj.parent.type == 'EMPTY':
            obj.data.name = obj.name

    # sanitize all BGM custom properties with game-mode-correct defaults
    all_mesh_objs = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    _sanitize_mesh_and_material_props(all_mesh_objs, game_mode=game_mode)

    # stamp game mode on scene so the Shader panel shows the right shader list
    try:
        scene.fo2_game_mode = game_mode
    except Exception:
        pass

    # update bgm_obj_flags on all existing dummies (catches pre-existing dummies)
    for obj in bpy.data.objects:
        if (obj.type == 'EMPTY'
                and obj.parent
                and obj.parent.name == 'fo2_body_dummies'):
            if "bgm_obj_flags" not in obj:
                obj["bgm_obj_flags"] = obj_flags

    print(f"[FO2 Reorganise] Done ({game_mode}): {renamed} promoted, "
          f"{removed} containers removed")
    return renamed, removed


# Merge same-name children

def _merge_same_name_children(parent_obj, coll):
    if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    groups = {}
    for child in list(parent_obj.children):
        if child.type != 'MESH' or not child.data:
            continue
        key = base_name(child.name)
        groups.setdefault(key, []).append(child)

    for bname, objects in groups.items():
        if len(objects) < 2:
            continue
        print(f"[FO2 Reorganise] Merging {len(objects)} meshes as '{bname}'")
        for obj in bpy.context.view_layer.objects:
            obj.select_set(False)
        for obj in objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = objects[0]
        try:
            with bpy.context.temp_override(
                active_object=objects[0],
                selected_objects=objects,
                selected_editable_objects=objects,
            ):
                bpy.ops.object.join()
        except (RuntimeError, TypeError):
            try:
                bpy.ops.object.join()
            except RuntimeError as e:
                print(f"[FO2 Reorganise] WARNING: Could not merge '{bname}': {e}")
                continue
        merged = bpy.context.active_object
        if merged:
            merged.name = bname
            link_to_collection(merged, coll)


# Operators

class FO2_OT_ReorganiseForFO1(bpy.types.Operator):
    """Reorganise the current scene for FlatOut 1 BGM export.
Sets version 0x00010004, object flags 0x0, v92=0 on light materials"""
    bl_idname  = "object.fo2_reorganise_fo1"
    bl_label   = "FO2: Reorganise for FlatOut 1"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        r, rem = do_reorganise_scene(game_mode='FO1')
        if r == 0 and rem == 0:
            self.report({'WARNING'},
                        "Nothing to reorganise — check the console for details")
        else:
            self.report({'INFO'},
                        f"Reorganised for FO1: {r} promoted, {rem} removed")
        return {'FINISHED'}


class FO2_OT_ReorganiseForFO2(bpy.types.Operator):
    """Reorganise the current scene for FlatOut 2 BGM export.
Sets version 0x00020000, object flags 0xE0F9, v92=2 on light materials"""
    bl_idname  = "object.fo2_reorganise_fo2"
    bl_label   = "FO2: Reorganise for FlatOut 2"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        r, rem = do_reorganise_scene(game_mode='FO2')
        if r == 0 and rem == 0:
            self.report({'WARNING'},
                        "Nothing to reorganise — check the console for details")
        else:
            self.report({'INFO'},
                        f"Reorganised for FO2: {r} promoted, {rem} removed")
        return {'FINISHED'}


class FO2_OT_ReorganiseForFOUC(bpy.types.Operator):
    """Reorganise the current scene for FlatOut Ultimate Carnage BGM export.
Sets version 0x00020000 + FOUC vertex format, object flags 0xE0F9, v92=2 on light materials"""
    bl_idname  = "object.fo2_reorganise_fouc"
    bl_label   = "FO2: Reorganise for FlatOut UC"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        r, rem = do_reorganise_scene(game_mode='FOUC')
        if r == 0 and rem == 0:
            self.report({'WARNING'},
                        "Nothing to reorganise — check the console for details")
        else:
            self.report({'INFO'},
                        f"Reorganised for FOUC: {r} promoted, {rem} removed")
        return {'FINISHED'}


class FO2_OT_ViewDummiesAsCubes(bpy.types.Operator):
    """Set all fo2_body_dummies empties to display as 0.03 m cubes with name and in-front enabled"""
    bl_idname  = "object.fo2_view_dummies_as_cubes"
    bl_label   = "FO2: View Dummies as Cubes"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        count = 0
        parent_empty = bpy.data.objects.get('fo2_body_dummies')
        if parent_empty and parent_empty.type == 'EMPTY':
            parent_empty.empty_display_type = 'CUBE'
            parent_empty.empty_display_size = 0.16
            parent_empty.show_name          = True
            parent_empty.show_in_front      = True
        for obj in bpy.data.objects:
            if (obj.type == 'EMPTY'
                    and obj.parent
                    and obj.parent.name == 'fo2_body_dummies'):
                obj.empty_display_type = 'CUBE'
                obj.empty_display_size = 0.03
                obj.show_name          = True
                obj.show_in_front      = True
                count += 1
        self.report({'INFO'}, f"Set {count} dummies to cube display")
        return {'FINISHED'}


class FO2_OT_ViewDummiesAsAxes(bpy.types.Operator):
    """Set all fo2_body_dummies empties to display as 0.3 m axes with name and in-front disabled"""
    bl_idname  = "object.fo2_view_dummies_as_axes"
    bl_label   = "FO2: View Dummies as Axes"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        count = 0
        parent_empty = bpy.data.objects.get('fo2_body_dummies')
        if parent_empty and parent_empty.type == 'EMPTY':
            parent_empty.empty_display_type = 'PLAIN_AXES'
            parent_empty.empty_display_size = 0.3
            parent_empty.show_name          = False
            parent_empty.show_in_front      = False
        for obj in bpy.data.objects:
            if (obj.type == 'EMPTY'
                    and obj.parent
                    and obj.parent.name == 'fo2_body_dummies'):
                obj.empty_display_type = 'PLAIN_AXES'
                obj.empty_display_size = 0.3
                obj.show_name          = False
                obj.show_in_front      = False
                count += 1
        self.report({'INFO'}, f"Set {count} dummies to axes display")
        return {'FINISHED'}


# Collision / segment box display  (CUBE empty  <->  real mesh cube)

def _is_collision_box(obj):
    """True if obj is a car-body collision box (fo2_collision_*) or a driver ragdoll
    segment (fo2_segment_* / carries the fo2_driver_segment property)."""
    if obj.name.startswith("fo2_collision_"):
        return True
    if obj.name.startswith("fo2_segment_"):
        return True
    if obj.get("fo2_driver_segment") is not None:
        return True
    return False


def _box_cube_mesh(half):
    """Shared cube mesh datablock spanning +/-half on each local axis. A box's real
    size comes from the object's scale, so the local cube just mirrors the empty's
    display (verts at +/-empty_display_size), keeping the box identical in size."""
    name = "fo2_box_cube_%.4f" % half
    me = bpy.data.meshes.get(name)
    if me is not None and len(me.vertices) == 8:
        return me
    if me is None:
        me = bpy.data.meshes.new(name)
    h = half
    verts = [(-h, -h, -h), (h, -h, -h), (h, h, -h), (-h, h, -h),
             (-h, -h,  h), (h, -h,  h), (h, h,  h), (-h, h,  h)]
    faces = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
             (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    me.from_pydata(verts, [], faces)
    me.update()
    return me


def _transfer_box(src, dst):
    """Copy transform, parenting, collections, custom properties, colour and display
    flags between two box objects. Preserves the world transform and, crucially, the
    rotation_quaternion / rotation_mode the bones.ini exporter reads from segments."""
    dst.rotation_mode = src.rotation_mode
    dst.location = src.location.copy()
    dst.rotation_quaternion = src.rotation_quaternion.copy()
    dst.rotation_euler = src.rotation_euler.copy()
    dst.scale = src.scale.copy()
    dst.color = src.color
    dst.show_name = src.show_name
    dst.show_in_front = src.show_in_front
    for k in src.keys():
        try:
            dst[k] = src[k]
        except Exception:  # noqa: BLE001
            pass
    for c in src.users_collection:
        if dst.name not in c.objects:
            c.objects.link(dst)
    dst.parent = src.parent
    dst.matrix_parent_inverse = src.matrix_parent_inverse.copy()


class FO2_OT_ViewCollisionsAsCubes(bpy.types.Operator):
    """Replace car-body collision boxes & driver ragdoll segment empties with real
    mesh cubes of identical size, transform and metadata. Fully reversible via
    'View Collisions as Empties'; does not affect the BGM/bones.ini export"""
    bl_idname  = "object.fo2_view_collisions_as_cubes"
    bl_label   = "FO2: View Collisions as Cubes"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        targets = [o for o in list(bpy.data.objects)
                   if o.type == 'EMPTY' and _is_collision_box(o)]
        count = 0
        for e in targets:
            half = e.empty_display_size or 0.5
            mesh = _box_cube_mesh(round(half, 4))
            m = bpy.data.objects.new("__fo2_box_tmp__", mesh)
            _transfer_box(e, m)
            m.show_wire = True            # edges visible over the solid box
            name = e.name
            bpy.data.objects.remove(e, do_unlink=True)
            m.name = name                 # reclaim the exact name (no .001 suffix)
            count += 1
        self.report({'INFO'},
                    f"Converted {count} collision/segment box(es) to mesh cubes")
        return {'FINISHED'}


class FO2_OT_ViewCollisionsAsEmpties(bpy.types.Operator):
    """Convert car-body collision box & driver segment mesh cubes back to
    CUBE-display empties (reverse of 'View Collisions as Cubes')"""
    bl_idname  = "object.fo2_view_collisions_as_empties"
    bl_label   = "FO2: View Collisions as Empties"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        targets = [o for o in list(bpy.data.objects)
                   if o.type == 'MESH' and _is_collision_box(o)]
        count = 0
        for m in targets:
            half = 0.5
            if m.data and len(m.data.vertices):
                half = max((max(abs(c) for c in v.co) for v in m.data.vertices),
                           default=0.5) or 0.5
            e = bpy.data.objects.new("__fo2_box_tmp__", None)
            e.empty_display_type = 'CUBE'
            e.empty_display_size = half
            _transfer_box(m, e)
            name = m.name
            mesh = m.data
            bpy.data.objects.remove(m, do_unlink=True)
            if (mesh is not None and mesh.users == 0
                    and mesh.name.startswith("fo2_box_cube")):
                bpy.data.meshes.remove(mesh)
            e.name = name
            count += 1
        self.report({'INFO'},
                    f"Converted {count} collision/segment box(es) back to empties")
        return {'FINISHED'}


# Standard-startpoints template
#
# Extracted verbatim from FO2 vanilla `city1b_startpoints.bed` (8 startpoints
# in a 4×2 staggered grid), then translated so the cluster centroid sits at
# (0, 0, 0) in FO2 space with all Y forced to 0 (Blender Z = 0 = ground plane).
# Every inter-point distance is preserved exactly (translation is
# distance-preserving); individual rotations are kept verbatim so the grid
# reproduces the mild per-point rotation variation seen in vanilla.
#
# Format: ((fo2_x, fo2_y=0, fo2_z), (rot_x[3], rot_y[3], rot_z[3]))
_STARTPOINTS_TEMPLATE = [
    (( 9.317154, 0.0,  5.890960),
     (0.244631, 0.0, -0.969616,  0.0, 1.0, 0.0,  0.969616, 0.0, 0.244631)),
    ((11.027939, 0.0, -0.380494),
     (0.269929, 0.0, -0.962880,  0.0, 1.0, 0.0,  0.962880, 0.0, 0.269929)),
    (( 4.489945, 0.0, -1.836701),
     (0.253083, 0.0, -0.967444,  0.0, 1.0, 0.0,  0.967444, 0.0, 0.253083)),
    (( 2.548997, 0.0,  3.927917),
     (0.244631, 0.0, -0.969616,  0.0, 1.0, 0.0,  0.969616, 0.0, 0.244631)),
    ((-2.429321, 0.0, -3.988037),
     (0.261516, 0.0, -0.965199,  0.0, 1.0, 0.0,  0.965199, 0.0, 0.261516)),
    ((-4.473190, 0.0,  1.924255),
     (0.261516, 0.0, -0.965199,  0.0, 1.0, 0.0,  0.965199, 0.0, 0.261516)),
    ((-9.231529, 0.0, -5.668457),
     (0.244631, 0.0, -0.969616,  0.0, 1.0, 0.0,  0.969616, 0.0, 0.244631)),
    ((-11.249992, 0.0,  0.130554),
     (0.261516, 0.0, -0.965199,  0.0, 1.0, 0.0,  0.965199, 0.0, 0.261516)),
]


def _fo2_startpoint_rot_to_blender_matrix(rot):
    """Convert FO2 startpoint rotation (9 floats, rows = [right, up, fwd] in
    FO2 axes) into a Blender 3x3 rotation matrix.

    FO2 uses Y-up, so each direction (x, y, z) → Blender (x, z, y). This is
    the same mapping the fo2_trackai_import addon uses when importing existing
    startpoints, so freshly-generated empties look identical to imported ones."""
    from mathutils import Matrix
    r = rot
    return Matrix((
        (r[0], r[6], r[3]),
        (r[2], r[8], r[5]),
        (r[1], r[7], r[4]),
    ))


class FO2_OT_AddStandardStartpoints(bpy.types.Operator):
    """Add a set of 8 standard FO2 racing startpoints (grid pattern taken
    from vanilla city1b, centered at world origin on Z=0). Empties get all
    fo2_startpoint_* custom properties populated so they roundtrip cleanly
    through the TrackAI exporter. The whole cluster can be freely rotated
    afterward — its collection origin is at (0, 0, 0)."""
    bl_idname  = "object.fo2_add_standard_startpoints"
    bl_label   = "TrackAI: Add Standard Startpoints"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from mathutils import Matrix, Vector

        # Locate a TrackAI root (any TrackAI_* collection with an AI subsection
        # or one it started as a section itself). If nothing found, create a
        # bare `TrackAI_Custom` root so the exporter can pick it up.
        root_col = None
        SECTION_HINTS = ("AISplines", "Splitpoints", "Startpoints", "Path")
        for col in bpy.data.collections:
            if not col.name.startswith("TrackAI_"):
                continue
            has_ai_child = any(
                any(child.name.startswith(f"TrackAI_{h}") for h in SECTION_HINTS)
                for child in col.children
            )
            if has_ai_child:
                root_col = col; break
        if root_col is None:
            # Loose fallback: first TrackAI_* collection at scene root
            for col in context.scene.collection.children:
                if col.name.startswith("TrackAI_"):
                    root_col = col; break
        if root_col is None:
            root_col = bpy.data.collections.new("TrackAI_Custom")
            context.scene.collection.children.link(root_col)
            self.report({'INFO'}, "Created new TrackAI_Custom root collection")

        # Locate or create the TrackAI_Startpoints sub-collection under root.
        sp_col = None
        for child in root_col.children:
            if child.name == "TrackAI_Startpoints" or child.name.startswith("TrackAI_Startpoints"):
                sp_col = child; break
        if sp_col is None:
            sp_col = bpy.data.collections.new("TrackAI_Startpoints")
            root_col.children.link(sp_col)

        # Refuse to clobber existing startpoints — safer than silent overwrite.
        existing = [o for o in sp_col.objects
                    if o.type == 'EMPTY' and o.name.startswith("Startpoint")]
        if existing:
            self.report({'WARNING'},
                        f"TrackAI_Startpoints already contains {len(existing)} "
                        f"startpoint empties — refusing to overwrite. Delete "
                        f"them first if you want to reset the grid.")
            return {'CANCELLED'}

        # Materialise the 8 empties. The importer stores per-point:
        #   fo2_startpoint_position/rotation  — from the binary blob
        #   fo2_bed_startpoint_position/rotation  — from the .bed text file
        #   fo2_import_rot_matrix  — for delta-detecting rotation edits
        # We populate all of them from the same template so the exporter
        # treats them like freshly-imported empties.
        for i, (pos_fo2, rot9) in enumerate(_STARTPOINTS_TEMPLATE):
            # FO2 (x, y, z) → Blender (x, z, y). y is already 0.
            loc_bl = Vector((pos_fo2[0], pos_fo2[2], pos_fo2[1]))

            empty = bpy.data.objects.new(f"Startpoint{i+1}", None)
            empty.empty_display_type = 'ARROWS'
            empty.empty_display_size = 3.0
            empty.location = loc_bl

            rot_mat = _fo2_startpoint_rot_to_blender_matrix(rot9)
            empty.matrix_world = Matrix.Translation(loc_bl) @ rot_mat.to_4x4()

            # Snapshot current rotation for future delta-detection on export.
            m = empty.matrix_world.to_3x3()
            empty['fo2_import_rot_matrix'] = [
                m[0][0], m[0][1], m[0][2],
                m[1][0], m[1][1], m[1][2],
                m[2][0], m[2][1], m[2][2],
            ]
            empty['fo2_startpoint_index']       = i
            empty['fo2_startpoint_position']    = list(pos_fo2)
            empty['fo2_startpoint_rotation']    = list(rot9)
            # .bed values default to the same as the binary blob — the
            # exporter tracks any rotation delta and re-derives .bed from it.
            empty['fo2_bed_startpoint_position'] = list(pos_fo2)
            empty['fo2_bed_startpoint_rotation'] = list(rot9)

            sp_col.objects.link(empty)

        self.report({'INFO'},
                    f"Added {len(_STARTPOINTS_TEMPLATE)} standard startpoints "
                    f"to '{sp_col.name}' (rotate the collection to face them "
                    f"any direction).")
        return {'FINISHED'}


class FO2_OT_SnapStartpointsToRibbon(bpy.types.Operator):
    """Snap each TrackAI startpoint's Z position to the closest ribbon mesh
    surface. Preserves X/Y position and rotation. Handles ribbon slopes by
    querying each startpoint independently — every empty is snapped to the
    closest point on the closest ribbon at its own X/Y location.

    Usage: move startpoints around in top view without worrying about Z,
    then run this to put them all cleanly on the track surface."""
    bl_idname  = "object.fo2_snap_startpoints_to_ribbon"
    bl_label   = "TrackAI: Snap Startpoints To Ribbon"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from mathutils import Vector

        # Ribbon meshes get named `{sec_name}_Ribbon` by the trackai import
        # addon (e.g. Path0_Ribbon). Match on that suffix pattern; any MESH
        # object whose name contains `_Ribbon` qualifies. Skip zero-vertex
        # meshes so a stale placeholder doesn't crash closest_point_on_mesh.
        ribbons = [o for o in bpy.data.objects
                   if o.type == 'MESH'
                   and '_Ribbon' in o.name
                   and o.data is not None
                   and len(o.data.vertices) > 0]
        if not ribbons:
            self.report({'WARNING'},
                        "No ribbon meshes found in the scene (expected "
                        "objects named like 'Path0_Ribbon')")
            return {'CANCELLED'}

        # Gather startpoints from every TrackAI_Startpoints collection. Use
        # the fo2_startpoint_index custom property as the definitive marker —
        # matches how the exporter identifies them, and skips unrelated
        # empties that a user might have parked in the same collection.
        startpoints = []
        for col in bpy.data.collections:
            if not col.name.startswith("TrackAI_Startpoints"):
                continue
            for obj in col.objects:
                if (obj.type == 'EMPTY'
                        and obj.get('fo2_startpoint_index', -1) >= 0):
                    startpoints.append(obj)
        if not startpoints:
            self.report({'WARNING'},
                        "No startpoints found in any TrackAI_Startpoints "
                        "collection (need fo2_startpoint_index property)")
            return {'CANCELLED'}

        snapped = 0
        for sp in startpoints:
            # Startpoint world position. Empties in a Blender collection have
            # no collection-level transform, so location == world coords unless
            # the empty has been parented — we handle both by going through
            # matrix_world.
            world_pt = sp.matrix_world.translation.copy()

            best_dist = float('inf')
            best_world_z = None
            for ribbon in ribbons:
                # closest_point_on_mesh works in the mesh's local space, so
                # translate the query point into ribbon coords first, then
                # translate the result back to world.
                try:
                    inv_mw = ribbon.matrix_world.inverted()
                except (ValueError, ZeroDivisionError):
                    continue  # singular / non-invertible transform
                local_pt = inv_mw @ world_pt
                try:
                    result = ribbon.closest_point_on_mesh(local_pt)
                except Exception:
                    continue
                # API returns (success, location, normal, index)
                if not result[0]:
                    continue
                closest_world = ribbon.matrix_world @ result[1]
                dist = (closest_world - world_pt).length
                if dist < best_dist:
                    best_dist = dist
                    best_world_z = closest_world.z

            if best_world_z is None:
                continue

            # Update only the world Z — preserve X, Y, and rotation.
            # For a parented empty we'd have to convert world Z back to local,
            # but startpoints imported/created by the addon aren't parented,
            # so direct assignment works.
            if sp.parent is None:
                sp.location.z = best_world_z
            else:
                # General case: recompute local coords from the parent transform
                current_world = sp.matrix_world.translation.copy()
                current_world.z = best_world_z
                try:
                    local = sp.parent.matrix_world.inverted() @ current_world
                    sp.location = local
                except (ValueError, ZeroDivisionError):
                    continue
            snapped += 1

        self.report({'INFO'},
                    f"Snapped {snapped}/{len(startpoints)} startpoints "
                    f"to closest ribbon surface")
        return {'FINISHED'}


# Reverse-track helpers

_TRACKAI_SECTION_RE = re.compile(r'^TrackAI_Path(\d+)$')


def _find_trackai_root_col():
    """Locate the TrackAI_* collection acting as the track's root."""
    SECTION_HINTS = ("AISplines", "Splitpoints", "Startpoints", "Path")
    # Prefer a collection with an AI-flavoured child
    for col in bpy.data.collections:
        if not col.name.startswith("TrackAI_"):
            continue
        for child in col.children:
            if any(child.name.startswith(f"TrackAI_{h}") for h in SECTION_HINTS):
                return col
    # Fallback: first TrackAI_* found
    for col in bpy.data.collections:
        if col.name.startswith("TrackAI_"):
            return col
    return None


def _reverse_nurbs_points_inplace(curve_obj):
    """Reverse the point order of every NURBS spline on a curve object."""
    if curve_obj is None or curve_obj.type != 'CURVE':
        return
    for spline in curve_obj.data.splines:
        if spline.type != 'NURBS':
            continue
        pts = [(p.co.x, p.co.y, p.co.z, p.co.w) for p in spline.points]
        for i, coords in enumerate(reversed(pts)):
            spline.points[i].co = coords


def _swap_and_reverse_curves(curve_a, curve_b):
    """Move (reversed) point data from curve_a into curve_b and vice versa.
    Used for Left↔Right boundary swap when reversing track direction: the
    old left edge becomes the new right edge (with points in reverse order),
    and the old right edge becomes the new left edge."""
    if (curve_a is None or curve_b is None
            or curve_a.type != 'CURVE' or curve_b.type != 'CURVE'):
        return

    def _read(curve):
        splines = []
        for sp in curve.data.splines:
            if sp.type != 'NURBS':
                continue
            splines.append([(p.co.x, p.co.y, p.co.z, p.co.w) for p in sp.points])
        return splines

    def _write(curve, splines_data):
        for i, sp in enumerate(curve.data.splines):
            if sp.type != 'NURBS' or i >= len(splines_data):
                continue
            for j, coords in enumerate(reversed(splines_data[i])):
                if j < len(sp.points):
                    sp.points[j].co = coords

    a_data = _read(curve_a)
    b_data = _read(curve_b)
    _write(curve_a, b_data)  # curve_a receives reversed b (new left = old right reversed)
    _write(curve_b, a_data)  # curve_b receives reversed a (new right = old left reversed)


def _delete_node_empties_in_section(sec_col):
    """Remove every `{sec_name}_Node*` empty from a TrackAI section
    collection. Nodes will be regenerated from curves on the next export."""
    prefix_re = re.compile(r'.*_Node\d+$')
    to_remove = []
    for obj in list(sec_col.objects):
        if obj.type == 'EMPTY' and prefix_re.match(obj.name):
            to_remove.append(obj)
    for obj in to_remove:
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except Exception:
            pass
    return len(to_remove)


def _reindex_splitpoints(splitpoints_col):
    """Reverse the fo2_splitpoint_index values (keeping the highest index
    as-is — it's the start/finish line in vanilla convention).

    For N splitpoints indexed 0..N-1, the last (N-1) stays; others get
    remapped: new = (N-2) - old.

    Also updates object names via a two-pass rename to avoid Blender's
    duplicate-name auto-suffix."""
    items = []
    for obj in splitpoints_col.objects:
        idx = obj.get('fo2_splitpoint_index', -1)
        if idx is None or int(idx) < 0:
            continue
        items.append((int(idx), obj))
    if len(items) < 2:
        return 0

    n = len(items)
    last_idx = n - 1

    # Two-pass rename: first give every reindexed object a unique temp name,
    # then set the final name. Prevents "Splitpoint9_Gate.001" collisions.
    tmp_names = {}
    for old_idx, obj in items:
        if old_idx == last_idx:
            new_idx = last_idx
        else:
            new_idx = (n - 2) - old_idx
        obj['fo2_splitpoint_index'] = new_idx
        tmp = f"__fo2_reverse_tmp__Splitpoint{new_idx + 1}_Gate"
        tmp_names[obj.name_full] = (obj, new_idx, obj.name)
        obj.name = tmp

    for _, (obj, new_idx, original_name) in tmp_names.items():
        obj.name = f"Splitpoint{new_idx + 1}_Gate"

    return n


def _mirror_position_across_line(P, L, axis_dir):
    """Reflect a 3-tuple position across the vertical plane through L with
    horizontal direction `axis_dir`. Y (vertical) is preserved so the point
    stays at the same elevation."""
    ax = axis_dir[0]; az = axis_dir[2]
    if ax*ax + az*az < 1e-12:
        return tuple(P)
    # Normal to reflection plane: perpendicular to axis in the XZ plane.
    # For axis_dir=(ax, _, az), plane normal ∝ (-az, 0, ax).
    nx, nz = -az, ax
    nlen = math.sqrt(nx*nx + nz*nz)
    nx /= nlen; nz /= nlen
    dx = P[0] - L[0]
    dy = P[1] - L[1]
    dz = P[2] - L[2]
    dot = dx * nx + dz * nz
    rx = dx - 2.0 * dot * nx
    rz = dz - 2.0 * dot * nz
    return (L[0] + rx, L[1] + dy, L[2] + rz)


def _mirror_vector_across_line(V, axis_dir):
    """Reflect a 3-tuple vector (direction, not position) across the vertical
    plane whose horizontal direction is `axis_dir`. Y is preserved."""
    ax = axis_dir[0]; az = axis_dir[2]
    if ax*ax + az*az < 1e-12:
        return tuple(V)
    nx, nz = -az, ax
    nlen = math.sqrt(nx*nx + nz*nz)
    nx /= nlen; nz /= nlen
    dot = V[0] * nx + V[2] * nz
    return (V[0] - 2.0 * dot * nx, V[1], V[2] - 2.0 * dot * nz)


def _fo2_cross(a, b):
    """Cross product of two 3-tuples."""
    return (a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0])


class FO2_OT_ReverseTrack(bpy.types.Operator):
    """Reverse the entire TrackAI direction in-place.

    Executes, in order:
      1. Reverse each section's curves — CenterLine and TargetLine get
         point-order reversed; Left/Right boundaries swap contents and
         reverse (old left becomes new right, reversed, and vice versa).
      2. Delete every node empty (they will regenerate from the reversed
         curves on the next export, with newly-computed forwards and
         geometry-derived speed hints — matches your item 2 goal).
      3. Reverse splitpoint indices — the highest-indexed splitpoint (the
         start/finish line in vanilla convention) stays at its index; all
         others get remapped so 0↔N-2, 1↔N-3, etc.
      4. Mirror startpoint positions and rotations across the finish-line
         axis (defined by the last splitpoint's Left→Right direction).
      5. Snap startpoints to the ribbon surface for correct Z on slopes.

    Note: this operator assumes the file's *curves* determine node order
    on export (which they do — `build_section_nodes` iterates the sampled
    curve points). Reversing splitpoint indices alone would not reverse
    node direction; the curves have to be reversed too. Both are done."""
    bl_idname = "object.fo2_reverse_track"
    bl_label = "TrackAI: Reverse Track"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from mathutils import Matrix, Vector

        root_col = _find_trackai_root_col()
        if root_col is None:
            self.report({'ERROR'},
                        "No TrackAI_* collection found in scene")
            return {'CANCELLED'}

        # 1. Reverse curves per section
        sections = []
        for child in root_col.children:
            if _TRACKAI_SECTION_RE.match(child.name):
                sections.append(child)
        if not sections:
            self.report({'WARNING'},
                        "No TrackAI_Path{N} sub-collections found — "
                        "nothing to reverse for the racing line itself")

        curves_reversed = 0
        nodes_deleted = 0
        for sec_col in sections:
            center = None; target = None; left = None; right = None
            for obj in sec_col.objects:
                if obj.type != 'CURVE':
                    continue
                if "_CenterLine" in obj.name:    center = obj
                elif "_TargetLine" in obj.name:  target = obj
                elif "_LeftBoundary" in obj.name: left = obj
                elif "_RightBoundary" in obj.name: right = obj

            if center is not None:
                try:
                    bpy.data.objects.remove(center, do_unlink=True)
                except Exception:
                    pass
                curves_reversed += 1
            if target is not None:
                _reverse_nurbs_points_inplace(target)
                curves_reversed += 1
            # Boundary swap+reverse only if BOTH exist (otherwise partial
            # state would be worse than doing nothing on that pair).
            if left is not None and right is not None:
                _swap_and_reverse_curves(left, right)
                # Swap the object names too. Two-pass via a temp prefix so
                # Blender doesn't auto-suffix with .001 when the second
                # rename would collide.
                left_original = left.name
                right_original = right.name
                left.name = "__fo2_reverse_tmp__" + left_original
                right.name = left_original.replace("LeftBoundary", "RightBoundary")
                left.name = right_original.replace("RightBoundary", "LeftBoundary")
                curves_reversed += 2

            # 2. Delete node empties in this section
            nodes_deleted += _delete_node_empties_in_section(sec_col)

        # 3. Reverse splitpoint indices
        splitpoints_col = None
        for child in root_col.children:
            if child.name == "TrackAI_Splitpoints":
                splitpoints_col = child; break
        splits_reindexed = 0
        finish_line = None
        if splitpoints_col is not None:
            # Capture the finish-line splitpoint (highest index) BEFORE the
            # reindex — it stays highest, but we grab the object handle now
            # to avoid re-searching after names change.
            best_idx = -1
            for obj in splitpoints_col.objects:
                idx = obj.get('fo2_splitpoint_index', -1)
                if idx is not None and int(idx) > best_idx:
                    best_idx = int(idx); finish_line = obj
            splits_reindexed = _reindex_splitpoints(splitpoints_col)

        # 4. Mirror startpoints across finish-line axis
        startpoints_col = None
        for child in root_col.children:
            if child.name.startswith("TrackAI_Startpoints"):
                startpoints_col = child; break
        startpoints_mirrored = 0
        if startpoints_col is not None and finish_line is not None:
            # Read L and R in world space from the finish-line splitpoint's
            # mesh (vertices [0]=left, [1]=position, [2]=right — set by the
            # trackai importer). Accounts for any user-applied movement of
            # the splitpoint object itself.
            try:
                if len(finish_line.data.vertices) >= 3:
                    v_left  = finish_line.matrix_world @ Vector(finish_line.data.vertices[0].co)
                    v_right = finish_line.matrix_world @ Vector(finish_line.data.vertices[2].co)
                else:
                    v_left = v_right = None
            except Exception:
                v_left = v_right = None

            if v_left is not None and v_right is not None:
                # Convert Blender L/R → FO2 space for the mirror math (the
                # startpoint's stored properties live in FO2 space, and the
                # mirror is a linear operation that works in either frame).
                # Blender (x, y, z) → FO2 (x, z, y).
                L_fo2 = (v_left.x, v_left.z, v_left.y)
                R_fo2 = (v_right.x, v_right.z, v_right.y)
                axis_dir_fo2 = (R_fo2[0] - L_fo2[0],
                                R_fo2[1] - L_fo2[1],
                                R_fo2[2] - L_fo2[2])

                for sp_obj in list(startpoints_col.objects):
                    if sp_obj.type != 'EMPTY':
                        continue
                    if sp_obj.get('fo2_startpoint_index', -1) < 0:
                        continue

                    # Position in FO2 space, derived from current Blender loc
                    # to respect any user edits. Blender (x, y, z) → FO2 (x, z, y).
                    old_pos_fo2 = (sp_obj.location.x,
                                   sp_obj.location.z,
                                   sp_obj.location.y)
                    new_pos_fo2 = _mirror_position_across_line(
                        old_pos_fo2, L_fo2, axis_dir_fo2)

                    # Rotation: mirror forward vector, keep up, recompute
                    # right = up × forward (FO2's convention — verified via
                    # roundtrip). This yields a valid rotation matrix for a
                    # car facing the opposite direction.
                    old_rot = sp_obj.get('fo2_startpoint_rotation')
                    if old_rot and len(old_rot) == 9:
                        old_fwd = (float(old_rot[6]), float(old_rot[7]), float(old_rot[8]))
                        old_up  = (float(old_rot[3]), float(old_rot[4]), float(old_rot[5]))
                    else:
                        # Fallback: derive from current matrix_world
                        m = sp_obj.matrix_world.to_3x3()
                        # Blender local Y = FO2 forward (in Blender coords);
                        # convert back via Blender (x, y, z) → FO2 (x, z, y).
                        by = (m[0][1], m[1][1], m[2][1])
                        bz = (m[0][2], m[1][2], m[2][2])
                        old_fwd = (by[0], by[2], by[1])
                        old_up  = (bz[0], bz[2], bz[1])

                    new_fwd = _mirror_vector_across_line(old_fwd, axis_dir_fo2)
                    new_up = old_up  # vertical, unaffected by horizontal mirror
                    new_right = _fo2_cross(new_up, new_fwd)

                    new_rot9 = (new_right[0], new_right[1], new_right[2],
                                new_up[0],    new_up[1],    new_up[2],
                                new_fwd[0],   new_fwd[1],   new_fwd[2])

                    # Push new position back into Blender location
                    # (FO2 (x, y, z) → Blender (x, z, y)).
                    sp_obj.location = Vector((new_pos_fo2[0],
                                              new_pos_fo2[2],
                                              new_pos_fo2[1]))

                    # Push new rotation via matrix_world. Use the same
                    # FO2→Blender matrix helper the "add startpoints"
                    # operator uses so the result matches importer output.
                    rot_mat = _fo2_startpoint_rot_to_blender_matrix(new_rot9)
                    sp_obj.matrix_world = (Matrix.Translation(sp_obj.location)
                                           @ rot_mat.to_4x4())

                    # Update custom properties
                    sp_obj['fo2_startpoint_position'] = list(new_pos_fo2)
                    sp_obj['fo2_startpoint_rotation'] = list(new_rot9)
                    sp_obj['fo2_bed_startpoint_position'] = list(new_pos_fo2)
                    sp_obj['fo2_bed_startpoint_rotation'] = list(new_rot9)

                    # Snapshot new import matrix so the exporter's delta
                    # detection treats this as the new "resting" state.
                    m = sp_obj.matrix_world.to_3x3()
                    sp_obj['fo2_import_rot_matrix'] = [
                        m[0][0], m[0][1], m[0][2],
                        m[1][0], m[1][1], m[1][2],
                        m[2][0], m[2][1], m[2][2],
                    ]

                    startpoints_mirrored += 1

        # 5. Snap startpoints to ribbon Z. Reuse the exact snap logic — call
        # the operator directly so any future improvements to snapping are
        # picked up for free.
        snap_result = None
        try:
            snap_result = bpy.ops.object.fo2_snap_startpoints_to_ribbon()
        except Exception:
            pass

        self.report({'INFO'},
                    f"Reversed track: {curves_reversed} curves reversed, "
                    f"{nodes_deleted} node empties deleted, "
                    f"{splits_reindexed} splitpoints reindexed, "
                    f"{startpoints_mirrored} startpoints mirrored"
                    + (" & snapped to ribbon"
                       if snap_result and 'FINISHED' in snap_result else ""))
        return {'FINISHED'}


# TrackAI Preview / Generation Operator
#
# Exposes the fo2_trackai_export generation pipeline as a modal properties
# dialog — same tunable parameters, no file save. Users can preview the
# generated CenterLines, TargetLines, node empties, and speed-hint values in
# Blender, adjust, and only actually export when satisfied.
#
# Depends on fo2_trackai_export being installed & enabled. Uses dry_run=True
# on the exporter's own function so there's zero logic duplication.


def _find_export_trackai():
    """Locate the export_trackai function from the sibling plugin, regardless
    of whether it was installed as a legacy addon or a Blender 4.2+ extension
    (bl_ext.user_default.*, bl_ext.blender_org.*, etc.)."""
    import sys
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if mod_name.endswith('fo2_trackai_export') and hasattr(mod, 'export_trackai'):
            return mod.export_trackai
    return None


class FO2_OT_PreviewTrackAI(bpy.types.Operator):
    """Run the TrackAI generation pipeline in-scene without writing any files.

    Opens a dialog with every knob the export operator exposes (CenterLine
    offset, TargetLine method + LERP + smoothing, speed_hint lookahead +
    radius). On OK, curves and node empties are created/updated in Blender
    as if you had exported, but nothing hits disk. Inspect the result in the
    Outliner, tweak manually if needed, then run the real export when ready.

    Requires the fo2_trackai_export plugin to be installed and enabled."""
    bl_idname = "object.fo2_preview_trackai"
    bl_label = "TrackAI: Preview / Generate"
    bl_options = {'REGISTER', 'UNDO'}

    # --- Properties (mirror fo2_trackai_export operator, minus file toggles)

    auto_generate_center: bpy.props.BoolProperty(
        name="Auto-generate CenterLine",
        description="If a section has no _CenterLine curve, create one by "
                    "offsetting RightBoundary perpendicular toward the track "
                    "interior. Requires both boundaries.",
        default=True,
    )
    center_offset: bpy.props.FloatProperty(
        name="Offset",
        description="Perpendicular distance from RightBoundary to the "
                    "generated CenterLine (FO2 units). 3.40 matches the "
                    "empirical mean across vanilla tracks",
        default=3.40, min=0.0, max=50.0, step=10, precision=2,
    )

    auto_generate_target: bpy.props.BoolProperty(
        name="Auto-generate TargetLine",
        description="If a section has no _TargetLine curve, create one from "
                    "the boundaries. Runs after CenterLine generation.",
        default=True,
    )
    target_method: bpy.props.EnumProperty(
        name="Method",
        description="How to synthesise the TargetLine when auto-generating",
        items=[
            ('SMOOTH', "Smoothed racing line",
             "Corridor-clamped Chaikin smoothing. Straights sit near t; turns "
             "pull the curve toward the corridor edge."),
            ('DUPLICATE', "Duplicate boundary",
             "Copy one boundary verbatim (nascar-style AI)."),
        ],
        default='SMOOTH',
    )
    target_lerp: bpy.props.FloatProperty(
        name="Base position",
        description="Initial LERP position inside the ribbon. 0 = "
                    "RightBoundary (inner), 0.5 = center, 1 = LeftBoundary "
                    "(outer). 0.30 = vanilla mean",
        default=0.30, min=0.0, max=1.0, step=5, precision=2,
    )
    target_smooth_iters: bpy.props.IntProperty(
        name="Smoothing passes",
        description="Chaikin iterations. 0 = plain LERP with no smoothing",
        default=10, min=0, max=50,
    )
    target_source: bpy.props.EnumProperty(
        name="Duplicate from",
        description="Which boundary to duplicate when method is Duplicate",
        items=[
            ('RIGHT', "RightBoundary", "Duplicate the inner boundary"),
            ('LEFT',  "LeftBoundary",  "Duplicate the outer boundary"),
        ],
        default='RIGHT',
    )

    generate_speed_hints: bpy.props.BoolProperty(
        name="Generate speed hints from geometry",
        description="Compute per-node fo2_speed_hint from curvature when "
                    "generating nodes from scratch. Unchecked = all "
                    "generated nodes get MAX (no limit). Existing empties "
                    "are always preserved verbatim",
        default=True,
    )

    speed_lookahead: bpy.props.IntProperty(
        name="Lookahead",
        description="How many nodes ahead the speed_hint algorithm scans "
                    "for the tightest upcoming turn",
        default=3, min=1, max=15,
    )
    speed_radius_threshold: bpy.props.FloatProperty(
        name="Radius",
        description="Corner radius above which speed uncaps to MAX. Lower = "
                    "less sensitive (only tight turns slow the AI). Higher = "
                    "more sensitive (mild curves also trigger slowdown)",
        default=7071.0, min=100.0, max=30000.0, step=100, precision=1,
    )

    flip_boundaries: bpy.props.BoolProperty(
        name="Flip boundaries",
        description="Swap Left and Right boundaries when they get derived "
                    "from a Ribbon mesh. If CenterLine ends up on the "
                    "outer edge of the track instead of down the middle, "
                    "toggle this and re-run. Applied before every "
                    "downstream generator (CenterLine, TargetLine, "
                    "nodes, speed hints)",
        default=False,
    )

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=420)

    def draw(self, context):
        layout = self.layout

        info = layout.box()
        info.label(text="Runs the export generation pipeline without saving.",
                   icon='INFO')
        info.label(text="Curves & node empties appear in the Outliner.")

        # CenterLine
        box = layout.box()
        box.label(text="Auto-generation", icon='NODETREE')
        row = box.row(); row.prop(self, "auto_generate_center")
        sub = box.column(); sub.enabled = self.auto_generate_center
        sub.prop(self, "center_offset")

        box.separator()

        # TargetLine
        row = box.row(); row.prop(self, "auto_generate_target")
        sub = box.column(); sub.enabled = self.auto_generate_target
        sub.prop(self, "target_method")
        if self.target_method == 'SMOOTH':
            sub.prop(self, "target_lerp", slider=True)
            sub.prop(self, "target_smooth_iters")
        else:
            sub.prop(self, "target_source", expand=True)

        # Speed hint
        box = layout.box()
        box.label(text="Speed hint (AI cornering)", icon='AUTO')
        box.prop(self, "generate_speed_hints")
        sub = box.column()
        sub.enabled = self.generate_speed_hints
        sub.prop(self, "speed_lookahead")
        sub.prop(self, "speed_radius_threshold")

        # Boundary side override — only matters when boundaries actually
        # get derived from a Ribbon this run.
        box = layout.box()
        box.label(text="Boundaries", icon='MOD_MIRROR')
        box.prop(self, "flip_boundaries")

    def execute(self, context):
        export_fn = _find_export_trackai()
        if export_fn is None:
            self.report({'ERROR'},
                        "fo2_trackai_export plugin not found. Install and "
                        "enable it before running Preview.")
            return {'CANCELLED'}

        # Match the export operator's "disable auto-gen when nothing is
        # missing" UX so a stale True doesn't try to clobber existing curves.
        any_target_missing = _any_section_missing_targetline_for_preview()
        any_center_missing = _any_section_missing_centerline_for_preview()

        options = {
            'dry_run': True,
            # Companion-file toggles — irrelevant in dry-run (skipped entirely),
            # but pass explicit False so the exporter never tries to touch disk.
            'export_splines_ai': False,
            'export_startpoints_bed': False,
            'export_splitpoints_bed': False,
            'auto_generate_center': self.auto_generate_center and any_center_missing,
            'center_offset': float(self.center_offset),
            'auto_generate_target': self.auto_generate_target and any_target_missing,
            'target_method': self.target_method,
            'target_source': self.target_source,
            'target_lerp': float(self.target_lerp),
            'target_smooth_iters': int(self.target_smooth_iters),
            'speed_lookahead': int(self.speed_lookahead),
            'speed_radius_threshold': float(self.speed_radius_threshold),
            'generate_speed_hints': self.generate_speed_hints,
            'flip_boundaries': self.flip_boundaries,
        }

        try:
            # filepath is unused in dry-run mode (null-writer), but pass an
            # empty string rather than None so any accidental os.path.* calls
            # inside the exporter don't blow up on typing.
            result = export_fn("", context, options)
        except Exception as e:
            self.report({'ERROR'}, f"Preview failed: {e}")
            import traceback; traceback.print_exc()
            return {'CANCELLED'}

        # Force the Outliner / 3D viewport to redraw the newly-generated
        # curves & node empties without needing a click-away.
        try:
            for area in context.screen.areas:
                area.tag_redraw()
        except Exception:
            pass

        self.report({'INFO'},
                    "Preview complete — inspect the generated curves & "
                    "node empties in the Outliner. Run the real export "
                    "when you're happy with the result.")
        return {'FINISHED'}


def _any_section_missing_centerline_for_preview():
    """Same check as fo2_trackai_export._any_section_missing_centerline but
    inlined here so bgm_hierarchy doesn't have a hard import dependency on
    the exporter's private helpers. Skips empty Path collections."""
    for col in bpy.data.collections:
        if not col.name.startswith("TrackAI_"):
            continue
        for child in col.children:
            if not _TRACKAI_SECTION_RE.match(child.name):
                continue
            if len(child.objects) == 0:
                continue  # empty placeholder — nothing to generate
            has_center = any(
                obj.type == 'CURVE' and "_CenterLine" in obj.name
                for obj in child.objects
            )
            if not has_center:
                return True
    return False


def _any_section_missing_targetline_for_preview():
    """Same check as fo2_trackai_export._any_section_missing_targetline.
    Skips empty Path collections."""
    for col in bpy.data.collections:
        if not col.name.startswith("TrackAI_"):
            continue
        for child in col.children:
            if not _TRACKAI_SECTION_RE.match(child.name):
                continue
            if len(child.objects) == 0:
                continue  # empty placeholder — nothing to generate
            has_target = any(
                obj.type == 'CURVE' and "_TargetLine" in obj.name
                for obj in child.objects
            )
            if not has_target:
                return True
    return False


# Make Hierarchy Operator
#
# Takes user-created ribbon meshes and builds the TrackAI collection tree
# around them, so a rough scene ("I made one Ribbon plane") turns into the
# structured hierarchy the exporter expects. Doesn't generate curves or
# nodes — that's the job of Preview / real Export.


_RIBBON_NAME_RE = re.compile(
    r'^(?:TrackAI[_-])?(?:Path)?(\d+)?[_-]?[Rr]ibbon(\d+)?(?:\.\d+)?$'
)


def _classify_ribbon_meshes():
    """Find every MESH object in the scene whose name looks like a ribbon
    (variously: 'Ribbon', 'Ribbon0', 'Path0_Ribbon', 'TrackAI_Ribbon', etc.).
    Returns a list of (mesh_obj, section_index) tuples with section indices
    assigned as follows:
      - Explicit index in the name (Ribbon0, Path2_Ribbon, TrackAI_Ribbon3) → use it
      - No index (Ribbon, TrackAI_Ribbon) → assign next available starting from 0
    Any collisions are resolved by bumping later ribbons to unused indices."""
    candidates = []
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        m = _RIBBON_NAME_RE.match(obj.name)
        if not m:
            continue
        # The regex has two capture groups for a leading and trailing digit;
        # prefer whichever is present (Path2_Ribbon → group 1, Ribbon2 → group 2)
        idx = None
        if m.group(1) is not None:
            idx = int(m.group(1))
        elif m.group(2) is not None:
            idx = int(m.group(2))
        candidates.append([obj, idx])

    # Assign indices to unindexed ribbons using the lowest slot not already claimed
    claimed = {c[1] for c in candidates if c[1] is not None}
    next_free = 0
    for entry in candidates:
        if entry[1] is not None:
            continue
        while next_free in claimed:
            next_free += 1
        entry[1] = next_free
        claimed.add(next_free)
        next_free += 1

    # Resolve collisions between two ribbons that both explicitly claimed the
    # same index (unlikely but possible if user has both 'Ribbon0' and
    # 'Path0_Ribbon'). First one wins, others get bumped to the next free slot.
    seen = set()
    for entry in candidates:
        while entry[1] in seen:
            entry[1] += 1
        seen.add(entry[1])

    candidates.sort(key=lambda c: c[1])
    return [(obj, idx) for obj, idx in candidates]


class FO2_OT_MakeTrackAIHierarchy(bpy.types.Operator):
    """Build the TrackAI collection tree from ribbon meshes already present
    in the scene. Auto-detects names like 'Ribbon', 'Ribbon0', 'Path0_Ribbon',
    'TrackAI_Ribbon' etc.; unindexed ribbons get numbered 0, 1, 2, … in
    Blender's object order.

    Creates 'TrackAI_Custom' (or reuses any existing TrackAI_* root) and one
    'TrackAI_Path{N}' sub-collection per detected ribbon, then moves and
    renames each ribbon into its section. Nothing else is generated —
    boundaries, centerlines, nodes, and everything else comes later from
    the Preview / Export operators."""
    bl_idname  = "object.fo2_make_trackai_hierarchy"
    bl_label   = "TrackAI: Make Hierarchy"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        ribbons = _classify_ribbon_meshes()
        if not ribbons:
            self.report({'WARNING'},
                        "No ribbon meshes found. Create a MESH named "
                        "'Ribbon' (or 'Ribbon0', 'Path0_Ribbon', "
                        "'TrackAI_Ribbon' …) first.")
            return {'CANCELLED'}

        # Reuse existing TrackAI_* root if present; otherwise create a fresh
        # 'TrackAI_Custom' root at scene root.
        root_col = _find_trackai_root_col()
        if root_col is None:
            root_col = bpy.data.collections.new("TrackAI_Custom")
            context.scene.collection.children.link(root_col)
            self.report({'INFO'},
                        "Created new TrackAI_Custom root collection")

        # Ensure a TrackAI_Path{N} sub-collection exists for each ribbon,
        # move+rename the ribbon into it.
        placed = 0
        for obj, sec_idx in ribbons:
            sec_name = f"TrackAI_Path{sec_idx}"
            sec_col = None
            for child in root_col.children:
                if child.name == sec_name:
                    sec_col = child; break
            if sec_col is None:
                sec_col = bpy.data.collections.new(sec_name)
                root_col.children.link(sec_col)

            # Unlink from every existing collection, then link into the
            # section — moves the ribbon rather than duplicating it.
            for col in list(obj.users_collection):
                try:
                    col.objects.unlink(obj)
                except Exception:
                    pass
            sec_col.objects.link(obj)

            # Rename to canonical form (Blender will auto-suffix if there's
            # already a collision, but with distinct section indices there
            # shouldn't be one).
            target_name = f"Path{sec_idx}_Ribbon"
            if obj.name != target_name:
                obj.name = target_name
            placed += 1

        self.report({'INFO'},
                    f"Placed {placed} ribbon(s) into "
                    f"'{root_col.name}' as Path0..Path{ribbons[-1][1]}")
        return {'FINISHED'}


# Rebuild Ribbon Operator
#
# Turns any track mesh (triangulated, over-detailed, self-crossing, closed
# ring or open stem) into a clean L/R-alternating quad strip the exporter
# can parse. The goal is vanilla-style ribbons, not geometric fidelity.
#
# Two shapes, auto-detected from border-loop topology:
#
#   RING (closed circuit): the mesh has two significant border loops —
#   outer perimeter and inner hole. Strategy: resample the outer loop
#   uniformly, generate the inner boundary by offsetting perpendicular-
#   inward by the track width. Ignores the inner loop's messy topology
#   (self-crossings etc.) entirely.
#
#   STEM (open branch): the mesh has ONE significant border loop that runs
#   down one side, around the end cap, back up the other side. Strategy:
#   "fold detection" — find the offset c such that arc positions s and
#   (c - s) on the loop pair up across the track (minimising median pair
#   distance). The fold's two fixed points are the track ends. Split the
#   loop there into two sides, resample each side uniformly, then pair
#   with a monotone closest-point pass (handles curved stems where the
#   outer side is longer than the inner).


def _list_mesh_objects_for_ribbon(self, context):
    """Enum-items callback: every MESH object currently in the scene."""
    items = []
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            items.append((obj.name, obj.name, f"Rebuild from '{obj.name}'"))
    if not items:
        items = [('_NONE_', "(no meshes in scene)", "")]
    return items


def _extract_border_loops(mesh_obj):
    """Group border edges of a mesh into ordered vertex loops (each a list
    of vertex indices, in the order traversed by edge-walking).

    Border edges are edges belonging to exactly one face — the mesh's
    outer perimeter plus any interior holes. Loops shorter than 3 verts
    are dropped."""
    from collections import defaultdict
    mesh = mesh_obj.data
    edge_faces = defaultdict(list)
    for fi, poly in enumerate(mesh.polygons):
        vs = list(poly.vertices)
        for i in range(len(vs)):
            v0 = vs[i]; v1 = vs[(i + 1) % len(vs)]
            edge_faces[(min(v0, v1), max(v0, v1))].append(fi)
    border = [e for e, fs in edge_faces.items() if len(fs) == 1]
    vert_edges = defaultdict(list)
    for e in border:
        vert_edges[e[0]].append(e)
        vert_edges[e[1]].append(e)
    visited = set(); loops = []
    for start_edge in border:
        if start_edge in visited:
            continue
        loop = [start_edge[0]]; cur = start_edge[0]
        while True:
            cand = [e for e in vert_edges[cur] if e not in visited]
            if not cand:
                break
            ne = cand[0]; visited.add(ne)
            nxt = ne[1] if ne[0] == cur else ne[0]
            loop.append(nxt); cur = nxt
            if cur == loop[0]:
                loop.pop(); break
        if len(loop) >= 3:
            loops.append(loop)
    return loops


# --- arc-length helpers for closed polyline loops (world-space points) ---

def _rr_dist(a, b):
    import math
    return math.sqrt((a[0] - b[0]) ** 2
                     + (a[1] - b[1]) ** 2
                     + (a[2] - b[2]) ** 2)


def _rr_cumlens(pts):
    """Cumulative arc lengths, including the closing wrap segment."""
    cum = [0.0]
    for i in range(len(pts)):
        cum.append(cum[-1] + _rr_dist(pts[i], pts[(i + 1) % len(pts)]))
    return cum


def _rr_point_at(pts, cum, s):
    """Point at arc position s (mod total) on the closed polyline."""
    T = cum[-1]
    s = s % T
    lo, hi = 0, len(pts)
    while lo < hi:
        mid = (lo + hi) // 2
        if cum[mid + 1] < s:
            lo = mid + 1
        else:
            hi = mid
    j = lo
    seg = cum[j + 1] - cum[j]
    t = (s - cum[j]) / seg if seg > 1e-9 else 0.0
    p0 = pts[j]; p1 = pts[(j + 1) % len(pts)]
    return [p0[k] + t * (p1[k] - p0[k]) for k in range(3)]


def _rr_sample_range(pts, cum, s0, s1, N):
    """N points from arc position s0 forward to s1 (wrapping if needed)."""
    T = cum[-1]
    span = (s1 - s0) % T
    if span == 0:
        span = T
    if N == 1:
        return [_rr_point_at(pts, cum, s0)]
    return [_rr_point_at(pts, cum, s0 + span * (i / (N - 1)))
            for i in range(N)]


def _rr_find_stem_fold(pts, K=512, step=4):
    """Find the fold offset of an open-stem border loop.

    A stem's single border loop runs down one side of the track, around an
    end cap, back up the other side, and around the second cap. So there
    exists an offset c where arc positions s and (c - s) sit across the
    track from each other. Searching c for the minimum median pair
    distance finds it; the fold's fixed points (s = c/2 and c/2 + T/2)
    are the two track ends.

    Returns (e1, e2, median_width, T): the two end arc-positions, the
    median across-track distance at the best fold, and the total loop
    arc length."""
    cum = _rr_cumlens(pts)
    T = cum[-1]
    P = [_rr_point_at(pts, cum, T * i / K) for i in range(K)]
    idxs = list(range(0, K, step))
    best_ci = 0
    best_err = float('inf')
    for ci in range(K):
        ds = sorted(_rr_dist(P[i], P[(ci - i) % K]) for i in idxs)
        med = ds[len(ds) // 2]
        if med < best_err:
            best_err = med
            best_ci = ci
    c = T * best_ci / K
    e1 = (c / 2) % T
    e2 = (c / 2 + T / 2) % T
    return e1, e2, best_err, T


class FO2_OT_RebuildRibbon(bpy.types.Operator):
    """Rebuild any track mesh into a clean L/R-alternating quad strip ribbon.

    Auto-detects closed circuits (ring: outer boundary + inward offset)
    vs open stems (fold detection to find the track ends, then pair the
    two sides). Aims for vanilla-style ribbons, not geometric fidelity."""
    bl_idname  = "object.fo2_rebuild_ribbon"
    bl_label   = "TrackAI: Rebuild Ribbon from Mesh"
    bl_options = {'REGISTER', 'UNDO'}

    source_mesh: bpy.props.EnumProperty(
        name="Source mesh",
        description="Which mesh in the scene to convert into a ribbon",
        items=_list_mesh_objects_for_ribbon,
    )
    shape_mode: bpy.props.EnumProperty(
        name="Track shape",
        description="Closed circuit vs open stem branch",
        items=[
            ('AUTO', "Auto-detect",
             "Ring if the mesh has a significant second border loop "
             "(inner hole), stem otherwise"),
            ('RING', "Closed loop",
             "Force ring handling: outer boundary + inward offset, "
             "closed ribbon"),
            ('STEM', "Open stem",
             "Force stem handling: fold detection + two-side pairing, "
             "open ribbon"),
        ],
        default='AUTO',
    )
    num_pairs: bpy.props.IntProperty(
        name="Resolution (L/R pairs)",
        description="Number of L/R vertex pairs in the new ribbon. Total "
                    "vertex count = 2× this. Vanilla ribbons range 30–500 "
                    "pairs depending on track length",
        default=100, min=4, max=2000,
    )
    width_mode: bpy.props.EnumProperty(
        name="Track width",
        description="How to determine the ribbon's width",
        items=[
            ('AUTO',   "Auto-detect from mesh",
             "Ring: median outer-to-inner-loop distance. Stem: actual "
             "across-track pairing from the mesh geometry (follows "
             "narrowing/widening roads)"),
            ('MANUAL', "Manual",
             "Force a fixed width everywhere (direction still derived "
             "from the mesh)"),
        ],
        default='AUTO',
    )
    manual_width: bpy.props.FloatProperty(
        name="Width",
        description="Track width in Blender units. Used only when Track "
                    "width is set to Manual",
        default=20.0, min=0.1, max=1000.0, precision=2,
    )
    end_trim: bpy.props.FloatProperty(
        name="End trim (× width)",
        description="Stems only: how much arc to trim off each track end, "
                    "as a multiple of the track width. Trimming skips the "
                    "end caps so the ribbon starts at (near-)full width "
                    "instead of converging to a point. ~0.5 for squared "
                    "caps, ~0.8 for rounded caps",
        default=0.75, min=0.0, max=3.0, precision=2,
    )
    flip_inward: bpy.props.BoolProperty(
        name="Flip inward / swap sides",
        description="Ring: reverse the inward-offset direction (use if the "
                    "generated boundary lands outside the track). Stem: "
                    "swap which side is Left vs Right",
        default=False,
    )
    output_name: bpy.props.StringProperty(
        name="Output name",
        description="Name for the new ribbon object. Use 'Ribbon' or "
                    "'Path0_Ribbon' so 'Make Hierarchy' picks it up later",
        default="Ribbon",
    )
    keep_source: bpy.props.BoolProperty(
        name="Keep source mesh",
        description="Add the ribbon alongside the source; uncheck to delete "
                    "the source after rebuilding",
        default=True,
    )

    def invoke(self, context, event):
        if (context.active_object is not None
                and context.active_object.type == 'MESH'):
            self.source_mesh = context.active_object.name
        return context.window_manager.invoke_props_dialog(self, width=420)

    def draw(self, context):
        layout = self.layout
        info = layout.box()
        info.label(text="Rebuilds a clean vanilla-style ribbon from any mesh.",
                   icon='MESH_GRID')
        info.label(text="Handles closed circuits AND open stem branches.")
        col = layout.column()
        col.prop(self, "source_mesh")
        col.prop(self, "shape_mode")
        col.prop(self, "num_pairs")
        col.prop(self, "width_mode")
        if self.width_mode == 'MANUAL':
            col.prop(self, "manual_width")
        col.prop(self, "end_trim")
        col.prop(self, "flip_inward")
        col.prop(self, "output_name")
        col.prop(self, "keep_source")

    # --- shape-specific builders -----------------------------------------

    def _build_ring(self, loops_pts, N):
        """Outer loop + perpendicular inward offset. Returns (L, R, width,
        closed=True)."""
        import math
        outer = loops_pts[0]
        xs = [p[0] for p in outer]
        ys = [p[1] for p in outer]
        zs = [p[2] for p in outer]
        ext = [max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)]
        vax = ext.index(min(ext))
        h1, h2 = [a for a in range(3) if a != vax]

        if self.width_mode == 'AUTO':
            if len(loops_pts) < 2:
                self.report({'WARNING'},
                            "Only one border loop; using Manual width "
                            f"{self.manual_width}")
                width = float(self.manual_width)
            else:
                inner = loops_pts[1]
                stride = max(1, len(outer) // 30)
                ds = sorted(min(_rr_dist(outer[i], q) for q in inner)
                            for i in range(0, len(outer), stride))
                width = ds[len(ds) // 2]
                if width < 0.01:
                    self.report({'WARNING'},
                                f"Auto width tiny ({width:.3f}); using "
                                f"Manual {self.manual_width}")
                    width = float(self.manual_width)
        else:
            width = float(self.manual_width)

        cum = _rr_cumlens(outer)
        sA = _rr_sample_range(outer, cum, 0.0, cum[-1], N + 1)[:-1]
        cx = sum(p[h1] for p in sA) / N
        cy = sum(p[h2] for p in sA) / N

        def build_R(sign):
            import math as _m
            out = []
            for i in range(N):
                pp = sA[(i - 1) % N]; pn = sA[(i + 1) % N]
                tx = pn[h1] - pp[h1]; ty = pn[h2] - pp[h2]
                tl = _m.sqrt(tx * tx + ty * ty)
                if tl < 1e-9:
                    out.append(list(sA[i])); continue
                tx /= tl; ty /= tl
                nx, ny = (-ty, tx) if sign > 0 else (ty, -tx)
                r = list(sA[i])
                r[h1] += nx * width
                r[h2] += ny * width
                out.append(r)
            return out

        R_plus = build_R(+1)
        R_minus = build_R(-1)
        d_plus = sum(math.sqrt((r[h1] - cx) ** 2 + (r[h2] - cy) ** 2)
                     for r in R_plus)
        d_minus = sum(math.sqrt((r[h1] - cx) ** 2 + (r[h2] - cy) ** 2)
                      for r in R_minus)
        R = R_plus if d_plus < d_minus else R_minus
        if self.flip_inward:
            R = R_minus if R is R_plus else R_plus
        return sA, R, width, True

    def _build_stem(self, loops_pts, N):
        """Fold-detect the track ends on the single border loop, split into
        two sides, resample each, pair with a monotone closest-point pass.
        Returns (L, R, median_width, closed=False)."""
        import math
        loop = loops_pts[0]
        cum = _rr_cumlens(loop)
        e1, e2, fold_w, T = _rr_find_stem_fold(loop)

        # Misclassification guard: on a closed ring forced (or auto-mis-
        # detected) into stem mode, folding pairs diametrically-opposite
        # points and the "width" comes out as a huge fraction of the mesh.
        xs = [p[0] for p in loop]; ys = [p[1] for p in loop]
        zs = [p[2] for p in loop]
        diag = math.sqrt((max(xs) - min(xs)) ** 2
                         + (max(ys) - min(ys)) ** 2
                         + (max(zs) - min(zs)) ** 2)
        if diag > 1e-6 and fold_w > 0.15 * diag:
            self.report({'WARNING'},
                        f"Fold width {fold_w:.1f} is {100 * fold_w / diag:.0f}% "
                        f"of the mesh size — this mesh may actually be a "
                        f"closed loop. Try Track shape = Closed loop.")

        margin = float(self.end_trim) * max(fold_w, 1e-3)
        side_len = (e2 - e1) % T
        other_len = T - side_len
        margin = min(margin, 0.25 * side_len, 0.25 * other_len)

        A = _rr_sample_range(loop, cum, e1 + margin, e2 - margin, N)
        M = max(4 * N, 400)
        wrap = T if (e1 - margin) % T < (e2 + margin) % T else 0
        B_dense = _rr_sample_range(loop, cum, e2 + margin,
                                   e1 - margin + wrap, M)
        B_dense = list(reversed(B_dense))

        # Monotone closest-point pairing: handles curved stems where the
        # two sides have different arc lengths without letting pairs
        # cross each other.
        B = []
        j_prev = 0
        win = max(3, (M // N) * 3)
        for k in range(N):
            j_hi = min(M - 1, j_prev + win)
            best_j = j_prev
            best_d = float('inf')
            for j in range(j_prev, j_hi + 1):
                d = _rr_dist(A[k], B_dense[j])
                if d < best_d:
                    best_d = d
                    best_j = j
            B.append(B_dense[best_j])
            j_prev = best_j

        if self.width_mode == 'MANUAL':
            manual = float(self.manual_width)
            R = []
            prev_dir = None
            for k in range(N):
                dx = [B[k][i] - A[k][i] for i in range(3)]
                dl = math.sqrt(sum(d * d for d in dx))
                if dl < 1e-9:
                    dirv = prev_dir if prev_dir else [0.0, 0.0, 1.0]
                else:
                    dirv = [d / dl for d in dx]
                    prev_dir = dirv
                R.append([A[k][i] + dirv[i] * manual for i in range(3)])
            B = R

        if self.flip_inward:
            A, B = B, A

        widths = sorted(_rr_dist(A[i], B[i]) for i in range(N))
        return A, B, widths[N // 2], False

    # ----------------------------------------------------------------------

    def execute(self, context):
        # Resolve source
        if self.source_mesh in ('', '_NONE_'):
            self.report({'ERROR'}, "No source mesh selected")
            return {'CANCELLED'}
        source = bpy.data.objects.get(self.source_mesh)
        if source is None or source.type != 'MESH':
            self.report({'ERROR'},
                        f"'{self.source_mesh}' is not a mesh object")
            return {'CANCELLED'}
        if len(source.data.vertices) < 3:
            self.report({'ERROR'},
                        f"'{source.name}' has fewer than 3 vertices")
            return {'CANCELLED'}

        loops_idx = _extract_border_loops(source)
        if not loops_idx:
            self.report({'ERROR'},
                        f"'{source.name}' has no border edges — mesh is "
                        f"fully closed (no boundary). Cannot rebuild.")
            return {'CANCELLED'}

        # World-space loops, sorted by arc length (not vertex count — a
        # dense small loop must not outrank a sparse big one)
        mw = source.matrix_world
        loops_pts = [[tuple(mw @ source.data.vertices[i].co) for i in loop]
                     for loop in loops_idx]
        loops_pts.sort(key=lambda L: _rr_cumlens(L)[-1], reverse=True)

        # Shape classification
        if self.shape_mode == 'AUTO':
            if (len(loops_pts) >= 2
                    and _rr_cumlens(loops_pts[1])[-1]
                        > 0.3 * _rr_cumlens(loops_pts[0])[-1]):
                shape = 'RING'
            else:
                shape = 'STEM'
        else:
            shape = self.shape_mode

        N = int(self.num_pairs)
        if shape == 'RING':
            L, R, width, closed = self._build_ring(loops_pts, N)
        else:
            L, R, width, closed = self._build_stem(loops_pts, N)

        # Assemble mesh: alternating L, R, L, R, ... in file order
        new_verts = []
        for i in range(N):
            new_verts.append(tuple(L[i]))
            new_verts.append(tuple(R[i]))
        new_faces = []
        quad_count = N if closed else N - 1
        for i in range(quad_count):
            L0 = 2 * i
            R0 = 2 * i + 1
            L1 = 2 * ((i + 1) % N)
            R1 = 2 * ((i + 1) % N) + 1
            new_faces.append((L0, R0, R1, L1))

        me = bpy.data.meshes.new(self.output_name)
        me.from_pydata(new_verts, [], new_faces)
        me.update()
        new_obj = bpy.data.objects.new(self.output_name, me)

        if source.users_collection:
            source.users_collection[0].objects.link(new_obj)
        else:
            context.scene.collection.objects.link(new_obj)

        if not self.keep_source and source != new_obj:
            source_name = source.name
            bpy.data.objects.remove(source, do_unlink=True)
            note = f" (deleted source '{source_name}')"
        else:
            note = ""

        self.report({'INFO'},
                    f"Built {'closed' if closed else 'open'} "
                    f"{shape.lower()} ribbon '{new_obj.name}': {N} L/R "
                    f"pairs, width~{width:.2f}u, {len(new_faces)} quads"
                    + note)
        return {'FINISHED'}


# Reverse Node Indexes Operator (experimental)
#
# Standalone counterpart to Reverse Track's step 2: only touches the
# fo2_node_index values on node empties inside user-selected sections,
# leaving curves, positions, and everything else alone. Marked experimental
# because reversing only the index numbers (without also flipping forwards,
# swapping boundaries, or re-ordering the sequence links) may or may not
# produce an in-game correct reversal — that's for the user to test.


def _sections_with_nodes_enum_items(self, context):
    """Dynamic items callback for the section-selection ENUM_FLAG. Returns
    every TrackAI_Path{N} collection that contains at least one node empty,
    labelled with its current node count."""
    node_name_re = re.compile(r'.*_Node\d+$')
    items = []
    slot = 0
    matched_sections = []
    for col in bpy.data.collections:
        m = _TRACKAI_SECTION_RE.match(col.name)
        if not m:
            continue
        n_nodes = sum(
            1 for obj in col.objects
            if obj.type == 'EMPTY' and node_name_re.match(obj.name)
        )
        if n_nodes > 0:
            matched_sections.append((int(m.group(1)), col.name, n_nodes))

    matched_sections.sort(key=lambda t: t[0])
    for sec_num, name, n_nodes in matched_sections:
        items.append((
            name,
            f"{name}  ({n_nodes} nodes)",
            f"Reverse fo2_node_index across the {n_nodes} nodes of {name}",
            1 << slot,
        ))
        slot += 1

    if not items:
        # Callback must return at least one item to keep Blender happy —
        # invoke() has already refused entry when this list would be empty,
        # so this sentinel is a safety net only.
        return [('_NONE_', "(no sections with node empties)", "", 0)]
    return items


class FO2_OT_ReverseNodeIndexes(bpy.types.Operator):
    """Reverse the fo2_node_index values on node empties inside selected
    TrackAI_Path{N} sections. Node positions, forwards, boundaries, and
    everything else are left untouched — only the sequence order changes.

    Experimental: for a fully-consistent reversal (including forwards,
    boundary swap, splitpoint reindex, startpoint mirror), use the
    'TrackAI: Reverse Track' operator instead."""
    bl_idname  = "object.fo2_reverse_node_indexes"
    bl_label   = "TrackAI: Reverse Node Indexes (experimental)"
    bl_options = {'REGISTER', 'UNDO'}

    selected_sections: bpy.props.EnumProperty(
        name="Sections",
        description="Which TrackAI_Path{N} sections should have their node "
                    "indexes reversed. Multi-select supported",
        items=_sections_with_nodes_enum_items,
        options={'ENUM_FLAG'},
    )

    def invoke(self, context, event):
        # Refuse to open the dialog when nothing is reversible — matches the
        # 'throw error if there are no nodes' behaviour the task asked for.
        node_name_re = re.compile(r'.*_Node\d+$')
        has_any = False
        for col in bpy.data.collections:
            if not _TRACKAI_SECTION_RE.match(col.name):
                continue
            for obj in col.objects:
                if obj.type == 'EMPTY' and node_name_re.match(obj.name):
                    has_any = True; break
            if has_any:
                break
        if not has_any:
            self.report({'ERROR'},
                        "No node empties found in any TrackAI_Path{N} "
                        "section. Nothing to reverse.")
            return {'CANCELLED'}
        return context.window_manager.invoke_props_dialog(self, width=380)

    def draw(self, context):
        layout = self.layout
        info = layout.box()
        info.label(text="Experimental: only reverses fo2_node_index values.",
                   icon='ERROR')
        info.label(text="Positions and other node data are unchanged.")
        layout.separator()
        layout.label(text="Sections to reverse:", icon='SORTBYEXT')
        layout.prop(self, "selected_sections", expand=True)

    def execute(self, context):
        selection = self.selected_sections
        if not selection or selection == {'_NONE_'}:
            self.report({'WARNING'}, "No sections selected — nothing to do")
            return {'CANCELLED'}

        node_name_re = re.compile(r'.*_Node\d+$')
        total_sections = 0
        total_nodes = 0
        for sec_name in selection:
            if sec_name == '_NONE_':
                continue
            col = bpy.data.collections.get(sec_name)
            if col is None:
                continue
            # Collect node empties, sort by current fo2_node_index
            nodes = [obj for obj in col.objects
                     if obj.type == 'EMPTY' and node_name_re.match(obj.name)]
            nodes.sort(key=lambda o: int(o.get('fo2_node_index', 0)))
            N = len(nodes)
            if N == 0:
                continue
            # Assign reversed indices. The node whose index was 0 becomes N-1,
            # and so on. Nothing else on the empty is touched.
            for i, obj in enumerate(nodes):
                obj['fo2_node_index'] = N - 1 - i
            total_sections += 1
            total_nodes += N

        self.report({'INFO'},
                    f"Reversed {total_nodes} node index(es) across "
                    f"{total_sections} section(s)")
        return {'FINISHED'}


# Reverse Splitpoint Indexes Operator
#
# Standalone counterpart to Reverse Track's step 3. Uses the exact same
# reindex logic (highest index stays highest — that's the start/finish line
# in vanilla convention), just without touching curves, nodes, or startpoints.


class FO2_OT_ReverseSplitpointIndexes(bpy.types.Operator):
    """Reverse the fo2_splitpoint_index values of every splitpoint in the
    TrackAI_Splitpoints collection. The highest-indexed splitpoint stays at
    its index (start/finish line — matches vanilla convention). All others
    are remapped so 0 ↔ N-2, 1 ↔ N-3, etc.

    Same reindex used by the full Reverse Track operator — just isolated to
    splitpoints only."""
    bl_idname  = "object.fo2_reverse_splitpoint_indexes"
    bl_label   = "TrackAI: Reverse Splitpoint Indexes"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        # Find the TrackAI_Splitpoints collection wherever it lives — usually
        # under a TrackAI_* root, but tolerate loose placement too.
        split_col = bpy.data.collections.get("TrackAI_Splitpoints")
        if split_col is None:
            root = _find_trackai_root_col()
            if root is not None:
                for child in root.children:
                    if child.name == "TrackAI_Splitpoints":
                        split_col = child; break
        if split_col is None:
            self.report({'ERROR'},
                        "No TrackAI_Splitpoints collection found")
            return {'CANCELLED'}

        # _reindex_splitpoints returns the count of items reindexed (0 or 1
        # if there was nothing meaningful to do).
        count = _reindex_splitpoints(split_col)
        if count < 2:
            self.report({'WARNING'},
                        "Fewer than 2 splitpoints found — nothing to reverse")
            return {'CANCELLED'}

        self.report({'INFO'},
                    f"Reversed {count} splitpoint index(es) (highest stayed "
                    f"as start/finish line)")
        return {'FINISHED'}


# Registration

def menu_func_object(self, context):
    self.layout.separator()
    self.layout.operator(FO2_OT_ReorganiseForFO1.bl_idname)
    self.layout.operator(FO2_OT_ReorganiseForFO2.bl_idname)
    self.layout.operator(FO2_OT_ReorganiseForFOUC.bl_idname)
    self.layout.separator()
    self.layout.operator(FO2_OT_ViewDummiesAsCubes.bl_idname)
    self.layout.operator(FO2_OT_ViewDummiesAsAxes.bl_idname)
    self.layout.separator()
    self.layout.operator(FO2_OT_ViewCollisionsAsCubes.bl_idname)
    self.layout.operator(FO2_OT_ViewCollisionsAsEmpties.bl_idname)
    self.layout.separator()
    self.layout.operator(FO2_OT_MakeTrackAIHierarchy.bl_idname)
    self.layout.operator(FO2_OT_RebuildRibbon.bl_idname)
    self.layout.operator(FO2_OT_AddStandardStartpoints.bl_idname)
    self.layout.operator(FO2_OT_SnapStartpointsToRibbon.bl_idname)
    self.layout.operator(FO2_OT_ReverseTrack.bl_idname)
    self.layout.operator(FO2_OT_ReverseNodeIndexes.bl_idname)
    self.layout.operator(FO2_OT_ReverseSplitpointIndexes.bl_idname)
    self.layout.operator(FO2_OT_PreviewTrackAI.bl_idname)


def register():
    bpy.utils.register_class(FO2_OT_ReorganiseForFO1)
    bpy.utils.register_class(FO2_OT_ReorganiseForFO2)
    bpy.utils.register_class(FO2_OT_ReorganiseForFOUC)
    bpy.utils.register_class(FO2_OT_ViewDummiesAsCubes)
    bpy.utils.register_class(FO2_OT_ViewDummiesAsAxes)
    bpy.utils.register_class(FO2_OT_ViewCollisionsAsCubes)
    bpy.utils.register_class(FO2_OT_ViewCollisionsAsEmpties)
    bpy.utils.register_class(FO2_OT_MakeTrackAIHierarchy)
    bpy.utils.register_class(FO2_OT_RebuildRibbon)
    bpy.utils.register_class(FO2_OT_AddStandardStartpoints)
    bpy.utils.register_class(FO2_OT_SnapStartpointsToRibbon)
    bpy.utils.register_class(FO2_OT_ReverseTrack)
    bpy.utils.register_class(FO2_OT_ReverseNodeIndexes)
    bpy.utils.register_class(FO2_OT_ReverseSplitpointIndexes)
    bpy.utils.register_class(FO2_OT_PreviewTrackAI)
    bpy.types.VIEW3D_MT_object.append(menu_func_object)


def unregister():
    bpy.types.VIEW3D_MT_object.remove(menu_func_object)
    bpy.utils.unregister_class(FO2_OT_PreviewTrackAI)
    bpy.utils.unregister_class(FO2_OT_ReverseSplitpointIndexes)
    bpy.utils.unregister_class(FO2_OT_ReverseNodeIndexes)
    bpy.utils.unregister_class(FO2_OT_ReverseTrack)
    bpy.utils.unregister_class(FO2_OT_SnapStartpointsToRibbon)
    bpy.utils.unregister_class(FO2_OT_AddStandardStartpoints)
    bpy.utils.unregister_class(FO2_OT_RebuildRibbon)
    bpy.utils.unregister_class(FO2_OT_MakeTrackAIHierarchy)
    bpy.utils.unregister_class(FO2_OT_ViewCollisionsAsEmpties)
    bpy.utils.unregister_class(FO2_OT_ViewCollisionsAsCubes)
    bpy.utils.unregister_class(FO2_OT_ViewDummiesAsAxes)
    bpy.utils.unregister_class(FO2_OT_ViewDummiesAsCubes)
    bpy.utils.unregister_class(FO2_OT_ReorganiseForFOUC)
    bpy.utils.unregister_class(FO2_OT_ReorganiseForFO2)
    bpy.utils.unregister_class(FO2_OT_ReorganiseForFO1)


if __name__ == "__main__":
    register()
