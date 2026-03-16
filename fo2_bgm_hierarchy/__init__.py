"""
FlatOut 2 BGM – Hierarchy Reorganiser
Converts any existing scene hierarchy into the flat layout the BGM exporter expects:

  FO2 Body collection
    fo2_body  (EMPTY, world origin)
      <mesh objects — one per car part>
      fo2_body_dummies  (EMPTY)
        <dummy empties — OBJC entries>

  FO2 Body Crash collection
    fo2_body_crash  (EMPTY, world origin)
      <mesh objects named *_crash>

All other collections / objects (collision, cameras, lights) are left untouched.

Two ways to use:
  - File > Import > FO2 Reorganise BGM Hierarchy (.blend)
  - View3D > Object > FO2: Reorganise Current Scene
"""

bl_info = {
    "name": "FlatOut 2 BGM Hierarchy Reorganiser",
    "author": "ravenDS",
    "version": (2, 0, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Object > FO2: Reorganise",
    "description": "Flatten any scene hierarchy into the layout the BGM exporter expects",
    "category": "Import-Export",
}

import bpy
import re
import os
import tempfile
from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty


_FLAG_FILE = os.path.join(tempfile.gettempdir(), "fo2_reorganise_pending")


# helpers

def depth_of(obj):
    d, p = 0, obj.parent
    while p:
        d += 1
        p = p.parent
    return d


def collect_all_descendants(obj):
    result = []
    for child in obj.children:
        result.append(child)
        result.extend(collect_all_descendants(child))
    return result


def collect_leaf_meshes(obj):
    """Recursively collect all mesh objects with actual geometry."""
    meshes = []
    for child in obj.children:
        if child.type == 'MESH' and child.data and len(child.data.vertices) > 0:
            meshes.append(child)
        meshes.extend(collect_leaf_meshes(child))
    return meshes


def is_crash(obj):
    """True if this object or any ancestor has '_crash' in its name."""
    cur = obj
    while cur:
        if '_crash' in cur.name:
            return True
        cur = cur.parent
    return False


def base_name(name):
    """Strip Blender duplicate suffix (.001 etc.) and _crash suffix."""
    n = re.sub(r'\.\d{3}$', '', name)
    return n


def ensure_collection(scene, name):
    """Get or create a scene-level collection by name."""
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
    if coll.name not in scene.collection.children:
        scene.collection.children.link(coll)
    return coll


def link_to_collection(obj, coll):
    """Ensure obj is linked to coll and no other collection."""
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    coll.objects.link(obj)


# core reorganise

def do_reorganise_scene():
    scene   = bpy.context.scene
    context = bpy.context

    # step 1: ensure Object mode
    if context.active_object and context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    # step 2: build / reuse canonical empties + collections
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

    # step 3: handle scene-level Objects empty (dummies)
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
                    link_to_collection(child, fo2_dummies_coll)
            bpy.data.objects.remove(obj, do_unlink=True)
            break

    # step 4: find groups to process
    # first try a scene-level source root (external file, first run)
    # If none exists, the scene was already partially reorganised -> use fo2_body's own children as the groups to flatten.
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
              f"with {best_children} direct children")
    else:
        # Already under fo2_body — re-flatten in place
        groups = list(fo2_body.children)
        source_root = None
        print(f"[FO2 Reorganise] No external root found — "
              f"re-flattening {len(groups)} children of fo2_body")

    renamed = 0
    removed = 0

    # step 5: process each group
    # target: fo2_body -> mesh (named group_base)  [completely flat]
    #         fo2_body_crash -> mesh (named group_base_crash)  [flat]

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

        # already a correctly named direct mesh child of fo2_body — skip
        if (group_is_mesh and group.parent == fo2_body
                and group.name == group_base):
            continue

        # dummies: inner Objects empty + standalone empties with no geometry
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
            if len(collect_leaf_meshes(o)) == 0 and not any(
                    c.type == 'MESH' for c in o.children):
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

        # regular meshes -> directly under fo2_body, named group_base
        for mesh_obj in regular_meshes:
            world = mesh_obj.matrix_world.copy()
            mesh_obj.parent = fo2_body
            mesh_obj.matrix_world = world
            mesh_obj.name = group_base
            link_to_collection(mesh_obj, fo2_body_coll)
            renamed += 1

        # crash meshes -> directly under fo2_body_crash
        for mesh_obj in crash_meshes:
            world = mesh_obj.matrix_world.copy()
            mesh_obj.parent = fo2_crash
            mesh_obj.matrix_world = world
            n = group_base if group_base.endswith('_crash') else group_base + '_crash'
            mesh_obj.name = n
            link_to_collection(mesh_obj, fo2_crash_coll)
            renamed += 1

        # dummies -> fo2_body_dummies
        for dummy in all_dummies:
            world = dummy.matrix_world.copy()
            dummy.parent = fo2_dummies
            dummy.matrix_world = world
            dummy.name = base_name(dummy.name)
            link_to_collection(dummy, fo2_dummies_coll)

        # remove ALL intermediates including the group container itself
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

    # step 6: remove external source root if used
    if source_root is not None and source_root not in skip:
        try:
            bpy.data.objects.remove(source_root, do_unlink=True)
            removed += 1
        except ReferenceError:
            pass

    # step 7: stray crash meshes under fo2_body -> fo2_body_crash
    for child in list(fo2_body.children):
        if child.type == 'MESH' and '_crash' in child.name:
            world = child.matrix_world.copy()
            child.parent = fo2_crash
            child.matrix_world = world
            link_to_collection(child, fo2_crash_coll)

    # step 8: remove fo2_body_crash if empty
    if not list(fo2_crash.children):
        bpy.data.objects.remove(fo2_crash, do_unlink=True)
        print("[FO2 Reorganise] No crash meshes — removed fo2_body_crash")

    # step 9: merge same-name meshes directly under fo2_body
    _merge_same_name_children(fo2_body, fo2_body_coll)
    if scene.objects.get("fo2_body_crash"):
        _merge_same_name_children(
            scene.objects["fo2_body_crash"], fo2_crash_coll)

    # step 10: strip .001/.002/etc — collision-safe rename
    # we must avoid Blender auto-appending suffixes when the clean name is already taken
    # strategy: record (object, clean_name) pairs, rename everything to guaranteed-unique temps, then apply clean names
    for root_obj in [fo2_body, scene.objects.get("fo2_body_crash")]:
        if root_obj is None:
            continue
        children = [c for c in root_obj.children if c not in skip]

        # record clean target name for each child BEFORE any renaming
        targets = []
        for child in children:
            clean_obj  = re.sub(r'\.\d{3}$', '', child.name)
            clean_data = (re.sub(r'\.\d{3}$', '', child.data.name)
                          if child.type == 'MESH' and child.data else None)
            targets.append((child, clean_obj, clean_data))

        # pass A: unique temp names so nothing collides
        for i, (child, _, _) in enumerate(targets):
            child.name = f"__fo2tmp_{i}__"
            if child.type == 'MESH' and child.data:
                child.data.name = f"__fo2tmpd_{i}__"

        # pass B: apply clean names
        for child, clean_obj, clean_data in targets:
            child.name = clean_obj
            if child.type == 'MESH' and child.data and clean_data:
                child.data.name = clean_data

    # step 11: rename mesh data to match their object name
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.parent and obj.parent.type == 'EMPTY':
            obj.data.name = obj.name

    print(f"[FO2 Reorganise] Done: {renamed} objects promoted, "
          f"{removed} containers removed")
    return renamed, removed


# Merge same-name children

def _merge_same_name_children(parent_obj, coll):
    """Join child meshes that share the same base name into one."""
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



# Persistent load_post handler

@bpy.app.handlers.persistent
def _on_file_loaded(dummy):
    if not os.path.exists(_FLAG_FILE):
        return
    try:
        os.remove(_FLAG_FILE)
    except OSError:
        pass

    total_renamed, total_removed = do_reorganise_scene()
    msg = (f"Reorganised: {total_renamed} objects promoted, "
           f"{total_removed} containers removed")
    print(f"[FO2 Reorganise] {msg}")

    def draw_popup(self, context):
        self.layout.label(text=msg)
    bpy.context.window_manager.popup_menu(
        draw_popup, title="FO2 Reorganise Complete", icon='INFO')


# ─────────────────────────────────────────────────────────────────────
# Operators
# ─────────────────────────────────────────────────────────────────────

class FO2_OT_ReorganiseBGM(bpy.types.Operator, ImportHelper):
    """Open a .blend file and reorganise its hierarchy for BGM export"""
    bl_idname  = "import_scene.fo2_reorganise_bgm"
    bl_label   = "FO2 Reorganise BGM Hierarchy"
    bl_options = {'REGISTER'}

    filename_ext = ".blend"
    filter_glob: StringProperty(default="*.blend", options={'HIDDEN'})

    def execute(self, context):
        with open(_FLAG_FILE, 'w') as f:
            f.write("pending")
        bpy.ops.wm.open_mainfile(filepath=self.filepath)
        return {'FINISHED'}


class FO2_OT_ReorganiseCurrentScene(bpy.types.Operator):
    """Reorganise the current scene's hierarchy for BGM export"""
    bl_idname  = "object.fo2_reorganise_current"
    bl_label   = "FO2: Reorganise Current Scene"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        total_renamed, total_removed = do_reorganise_scene()
        if total_renamed == 0 and total_removed == 0:
            self.report({'WARNING'},
                        "Nothing to reorganise — check the console for details")
        else:
            self.report({'INFO'},
                        f"Reorganised: {total_renamed} objects promoted, "
                        f"{total_removed} containers removed")
        return {'FINISHED'}


# Registration

def menu_func_import(self, context):
    self.layout.operator(FO2_OT_ReorganiseBGM.bl_idname,
                         text="FO2 Reorganise BGM Hierarchy (.blend)")

def menu_func_object(self, context):
    self.layout.operator(FO2_OT_ReorganiseCurrentScene.bl_idname)

def register():
    bpy.utils.register_class(FO2_OT_ReorganiseBGM)
    bpy.utils.register_class(FO2_OT_ReorganiseCurrentScene)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    bpy.types.VIEW3D_MT_object.append(menu_func_object)
    if _on_file_loaded not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_file_loaded)

def unregister():
    if _on_file_loaded in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_file_loaded)
    bpy.types.VIEW3D_MT_object.remove(menu_func_object)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.utils.unregister_class(FO2_OT_ReorganiseCurrentScene)
    bpy.utils.unregister_class(FO2_OT_ReorganiseBGM)

if __name__ == "__main__":
    register()
