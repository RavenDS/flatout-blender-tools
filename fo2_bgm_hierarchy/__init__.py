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
    "version":     (1, 3, 2),
    "blender":     (3, 6, 0),
    "location":    "View3D > Object > FO2: Reorganise",
    "description": "Flatten any scene hierarchy into the layout the BGM exporter expects",
    "category":    "Import-Export",
}

import bpy
import re
import os


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

    game_mode affects v92 for light materials:
      FO1        -> v92 = 0  (original FO1 files have v92=0 on all materials)
      FO2 / FOUC -> v92 = 2
    """
    name  = mat_name.lower()
    shader, alpha, v92 = SHADER_CAR_METAL, 0, 0
    tex_override = None

    if name.startswith("shadow") or name.endswith("shadow"):
        shader       = SHADER_SHADOW_PROJECT
        tex_override = "shadow.tga"
    elif name.startswith("body"):
        shader       = SHADER_CAR_BODY
        tex_override = "skin1.tga"
    elif name.startswith("interior"):
        shader = SHADER_CAR_DIFFUSE
    elif name.startswith("grille"):
        shader, alpha = SHADER_CAR_DIFFUSE, 1
    elif name.startswith("window"):
        shader = SHADER_CAR_WINDOW
    elif name.startswith("shear"):
        shader = SHADER_CAR_SHEAR
    elif name.startswith("scaleshock") or name.startswith("shearhock") or name.startswith("shearshock"):
        shader, alpha = SHADER_CAR_SCALE, 0
    elif name.startswith("shock") or name.startswith("spring") or name.startswith("scale"):
        shader = SHADER_CAR_SCALE
    elif name.startswith("tire"):
        shader = SHADER_CAR_DIFFUSE
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

    used_shader_ids: set = set()
    for obj in mesh_objects:
        for slot in obj.material_slots:
            if slot.material and "bgm_shader_id" in slot.material:
                try:
                    used_shader_ids.add(int(slot.material["bgm_shader_id"]))
                except (TypeError, ValueError):
                    pass

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
                used_shader_ids.add(shader_id)
                changed = True
            else:
                # Already has a shader — update v92 on lights if it was set to the wrong game-mode default (0 vs 2).
                try:
                    sid = int(bl_mat["bgm_shader_id"])
                    used_shader_ids.add(sid)
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


# Registration

def menu_func_object(self, context):
    self.layout.separator()
    self.layout.operator(FO2_OT_ReorganiseForFO1.bl_idname)
    self.layout.operator(FO2_OT_ReorganiseForFO2.bl_idname)
    self.layout.operator(FO2_OT_ReorganiseForFOUC.bl_idname)


def register():
    bpy.utils.register_class(FO2_OT_ReorganiseForFO1)
    bpy.utils.register_class(FO2_OT_ReorganiseForFO2)
    bpy.utils.register_class(FO2_OT_ReorganiseForFOUC)
    bpy.types.VIEW3D_MT_object.append(menu_func_object)


def unregister():
    bpy.types.VIEW3D_MT_object.remove(menu_func_object)
    bpy.utils.unregister_class(FO2_OT_ReorganiseForFOUC)
    bpy.utils.unregister_class(FO2_OT_ReorganiseForFO2)
    bpy.utils.unregister_class(FO2_OT_ReorganiseForFO1)


if __name__ == "__main__":
    register()
