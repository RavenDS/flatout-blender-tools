bl_info = {
    "name":        "FlatOut 2 TrackAI Exporter",
    "author":      "ravenDS (github.com/ravenDS)",
    "version":     (2, 2, 0),
    "blender":     (3, 6, 0),
    "location":    "File > Export > FlatOut 2 TrackAI (.bin)",
    "description": "Export FlatOut 2 AI path data (trackai.bin + .bed)",
    "category":    "Import-Export",
    "doc_url":     "https://github.com/RavenDS",
    "tracker_url": "https://github.com/RavenDS/flatout-blender-tools/issues",
}

import bpy
import struct
import os
import math
import base64
from bpy.props import (StringProperty, BoolProperty, FloatProperty)
from bpy_extras.io_utils import ExportHelper
from mathutils import Vector


# CONSTANTS

TAG_FILE_HEADER    = 0x00270276
TAG_SPLINE_SECTION = 0x00290276
TAG_NODE_START     = 0x00230276
TAG_NODE_END       = 0x00240276
TAG_SECTION_SEP    = 0x00260276
TAG_FILE_END       = 0x00280276
TAG_EXTRA_START    = 0x00280876
TAG_STARTPOINTS    = 0x00300876
TAG_SPLITPOINTS    = 0x00310876
TAG_SPLITPOINT_SUB = 0x00010976
TAG_AI_BVH_1       = 0x00020976
TAG_AI_BVH_2       = 0x00290876
TAG_AI_BVH_HDR4    = 0x00020376
TAG_AI_BVH_LEAVES_END = 0x00030376
TAG_AI_BVH_END1    = 0x00050376
TAG_AI_BVH_END2    = 0x00010376

SECTION_NAMES = ("AISplines", "Splitpoints", "Startpoints")

# HELPERS

def blender_to_fo2(vec):
    """Blender (x,y,z) to FO2 (x,z,y)"""
    return (vec[0], vec[2], vec[1])

def blender_dir_to_fo2(vec):
    return (vec[0], vec[2], vec[1])

def write_u32(f, v):
    f.write(struct.pack('<I', v))

def write_i32(f, v):
    f.write(struct.pack('<i', v))

def write_f32(f, v):
    f.write(struct.pack('<f', v))

def write_vec3(f, v):
    f.write(struct.pack('<3f', v[0], v[1], v[2]))

def normalize(v):
    x, y, z = v
    length = math.sqrt(x*x + y*y + z*z)
    if length < 1e-12:
        return (0.0, 0.0, 0.0)
    return (x/length, y/length, z/length)

def cross(a, b):
    return (
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    )

def vec_sub(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

def vec_add(a, b):
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

def vec_scale(a, s):
    return (a[0]*s, a[1]*s, a[2]*s)

def vec_dist(a, b):
    d = vec_sub(a, b)
    return math.sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2])

def vec_len(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])


# FIND TRACKAI ROOT COLLECTION

def find_trackai_root(context):
    """Find the TrackAI root collection in the scene."""
    for col in bpy.data.collections:
        if col.name.startswith("TrackAI_"):
            # Check it's a root (has section sub-collections)
            for child in col.children:
                if any(child.name.startswith(f"TrackAI_{sn}") for sn in SECTION_NAMES):
                    return col
                if child.get('fo2_section_index', -1) >= 0:
                    return col
    return None


# READ CURVES FROM BLENDER

def sample_curve_points(obj):
    """Get control points from a NURBS curve object, converting back to FO2 coords."""
    if obj is None or obj.type != 'CURVE':
        return []
    points = []
    for spline in obj.data.splines:
        for pt in spline.points:
            world = obj.matrix_world @ Vector((pt.co.x, pt.co.y, pt.co.z))
            points.append(blender_to_fo2(world))
    return points


def find_object_in_collection(col, suffix):
    """Find an object in a collection by name suffix."""
    for obj in col.objects:
        if obj.name.endswith(suffix):
            return obj
    return None


def find_object_containing(col, substring):
    """Find an object in collection whose name contains substring."""
    for obj in col.objects:
        if substring in obj.name:
            return obj
    return None


# READ NODE EMPTIES

def gather_empties(col, section_name):
    """Collect all node empties sorted by file position (order in binary file)."""
    empties = []
    prefix = f"{section_name}_Node"
    for obj in col.objects:
        if obj.name.startswith(prefix) and obj.type == 'EMPTY':
            # fo2_file_position is the sequential position in the file (0..n-1)
            # fo2_node_index is the linked-list pointer (NOT sequential!)
            file_pos = obj.get('fo2_file_position', -1)
            if file_pos >= 0:
                empties.append((file_pos, obj))
            else:
                # Fallback for old imports without fo2_file_position:
                # use seq_index which is usually correct
                seq = obj.get('fo2_seq_index', -1)
                if seq >= 0:
                    empties.append((seq, obj))
    empties.sort(key=lambda x: x[0])
    return [e[1] for e in empties]


# BUILD NODES FROM CURVES + EMPTIES

def compute_forward(centers, i, n, is_closed):
    """Compute forward direction from prev/next center positions"""
    if n < 2:
        return (1.0, 0.0, 0.0)
    if is_closed:
        prev = centers[(i - 1) % n]
        nxt = centers[(i + 1) % n]
    else:
        if i == 0:
            prev = centers[0]
            nxt = centers[1]
        elif i == n - 1:
            prev = centers[n - 2]
            nxt = centers[n - 1]
        else:
            prev = centers[i - 1]
            nxt = centers[i + 1]
    d = vec_sub(nxt, prev)
    return normalize(d)


def _read_vec3_prop(e, key, fallback):
    """Read a vec3 custom property from an empty, return as tuple"""
    v = e.get(key)
    if v and len(v) == 3:
        return (float(v[0]), float(v[1]), float(v[2]))
    return fallback


def build_section_nodes(centers, lefts, rights, targets, n, is_closed,
                        empties, section_index):
    """Build binary node data for one section
    
    If empties are present, ALL fields are read from them (positions included).
    Curves are only used as fallback when there are no empties.
    """
    if n == 0:
        return b''

    has_empties = len(empties) == n
    buf = bytearray()
    _u = lambda v: int(v) & 0xFFFFFFFF

    for i in range(n):
        # defaults from curves
        center = centers[i]
        left = lefts[i] if i < len(lefts) else center
        right = rights[i] if i < len(rights) else center
        mid = vec_scale(vec_add(left, right), 0.5)
        target = targets[i] if i < len(targets) else mid

        forward = compute_forward(centers, i, n, is_closed)
        right_dir_vec = vec_sub(right, left)
        right_dir = normalize(right_dir_vec)
        up = normalize(cross(forward, right_dir))

        rotation = (
            right_dir[0], up[0], forward[0],
            right_dir[1], up[1], forward[1],
            right_dir[2], up[2], forward[2],
        )

        width_left = vec_dist(center, left)
        width_right = vec_dist(center, right)

        if i == 0:
            cumul = 0.0
        else:
            cumul = 0.0
            for j in range(1, i + 1):
                cumul += vec_dist(centers[j-1], centers[j])

        interp_weights = (1.0, 0.0, 0.0)
        unk_neg1 = -1.0
        speed_hint = 1000000.0
        unk3 = 0
        sentinel1 = -1
        speed_hint2 = 1000000.0
        flag = 1
        seq_index = i
        unk4 = 0
        sentinel2 = -1
        unk5 = 0
        unk1 = 0
        unk2 = 0

        if is_closed:
            prev_index = (n - 1) if i == 0 else (i - 1)
        else:
            prev_index = 0 if i == 0 else (i - 1)

        node_index = i + 1  # 1-based

        # override fields from empties when available
        if has_empties:
            e = empties[i]
            node_index = int(e.get('fo2_node_index', node_index))
            unk1 = int(e.get('fo2_unk1', unk1))
            prev_index = int(e.get('fo2_prev_index', prev_index))
            unk2 = int(e.get('fo2_unk2', unk2))

            # position
            center = _read_vec3_prop(e, 'fo2_center', center)
            left = _read_vec3_prop(e, 'fo2_left', left)
            right = _read_vec3_prop(e, 'fo2_right', right)
            mid = _read_vec3_prop(e, 'fo2_mid', mid)
            target = _read_vec3_prop(e, 'fo2_target', target)

            # apply movement delta: derive import location from fo2_center,
            # compare to current obj.location, apply delta to all positions
            import_bl = (center[0], center[2], center[1])  # fo2_to_blender
            delta_bl = (e.location[0] - import_bl[0],
                        e.location[1] - import_bl[1],
                        e.location[2] - import_bl[2])
            delta_fo2 = (delta_bl[0], delta_bl[2], delta_bl[1])  # blender_to_fo2
            if abs(delta_fo2[0]) > 1e-6 or abs(delta_fo2[1]) > 1e-6 or abs(delta_fo2[2]) > 1e-6:
                center = (center[0] + delta_fo2[0], center[1] + delta_fo2[1], center[2] + delta_fo2[2])
                left = (left[0] + delta_fo2[0], left[1] + delta_fo2[1], left[2] + delta_fo2[2])
                right = (right[0] + delta_fo2[0], right[1] + delta_fo2[1], right[2] + delta_fo2[2])
                mid = (mid[0] + delta_fo2[0], mid[1] + delta_fo2[1], mid[2] + delta_fo2[2])
                target = (target[0] + delta_fo2[0], target[1] + delta_fo2[1], target[2] + delta_fo2[2])

            # direction
            forward = _read_vec3_prop(e, 'fo2_forward', forward)
            right_dir = _read_vec3_prop(e, 'fo2_right_dir', right_dir)

            # rotation matrix
            stored_rot = e.get('fo2_rotation')
            if stored_rot and len(stored_rot) == 9:
                rotation = tuple(float(v) for v in stored_rot)

            # interpolation
            stored_iw = e.get('fo2_interp_weights')
            if stored_iw and len(stored_iw) == 3:
                interp_weights = tuple(float(v) for v in stored_iw)

            # scalar fields
            width_left = float(e.get('fo2_width_left', width_left))
            width_right = float(e.get('fo2_width_right', width_right))
            cumul = float(e.get('fo2_cumul_distance', cumul))
            unk_neg1 = float(e.get('fo2_unk_neg1', unk_neg1))
            speed_hint = float(e.get('fo2_speed_hint', speed_hint))
            unk3 = int(e.get('fo2_unk3', unk3))
            sentinel1 = int(e.get('fo2_sentinel1', sentinel1))
            speed_hint2 = float(e.get('fo2_speed_hint2', speed_hint2))
            flag = int(e.get('fo2_flag', flag))
            seq_index = int(e.get('fo2_seq_index', seq_index))
            unk4 = int(e.get('fo2_unk4', unk4))
            sentinel2 = int(e.get('fo2_sentinel2', sentinel2))
            unk5 = int(e.get('fo2_unk5', unk5))

        # write node (208 bytes)
        buf += struct.pack('<I', TAG_NODE_START)
        buf += struct.pack('<I', _u(node_index))
        buf += struct.pack('<I', _u(unk1))
        buf += struct.pack('<I', _u(prev_index))
        buf += struct.pack('<I', _u(unk2))
        buf += struct.pack('<9f', *[float(v) for v in rotation])
        buf += struct.pack('<3f', *[float(v) for v in center])
        buf += struct.pack('<3f', *[float(v) for v in left])
        buf += struct.pack('<3f', *[float(v) for v in right])
        buf += struct.pack('<3f', *[float(v) for v in mid])
        buf += struct.pack('<3f', *[float(v) for v in target])
        buf += struct.pack('<3f', *[float(v) for v in forward])
        buf += struct.pack('<3f', *[float(v) for v in right_dir])
        buf += struct.pack('<3f', *[float(v) for v in interp_weights])
        buf += struct.pack('<f', float(width_left))
        buf += struct.pack('<f', float(width_right))
        buf += struct.pack('<f', float(cumul))
        buf += struct.pack('<f', float(unk_neg1))
        buf += struct.pack('<f', float(speed_hint))
        buf += struct.pack('<i', int(unk3))
        buf += struct.pack('<i', int(sentinel1))
        buf += struct.pack('<f', float(speed_hint2))
        buf += struct.pack('<I', _u(flag))
        buf += struct.pack('<I', _u(seq_index))
        buf += struct.pack('<I', _u(unk4))
        buf += struct.pack('<i', int(sentinel2))
        buf += struct.pack('<I', _u(unk5))
        buf += struct.pack('<I', TAG_NODE_END)

    assert len(buf) == n * 208, f"Node data size mismatch: {len(buf)} != {n*208}"
    return bytes(buf)


# MAIN EXPORT

def export_trackai(filepath, context, options):
    root_col = find_trackai_root(context)
    if root_col is None:
        raise ValueError("No TrackAI collection found in scene")

    print(f"[TrackAI Export] Found root: {root_col.name}")

    # gather section collections by fo2_section_index 
    section_cols = []
    for child in root_col.children:
        sec_idx = child.get('fo2_section_index', -1)
        if sec_idx >= 0:
            sec_name = child.name.replace("TrackAI_", "", 1)
            section_cols.append((sec_idx, sec_name, child))
    section_cols.sort(key=lambda x: x[0])
    section_cols = [(name, col) for _, name, col in section_cols]

    if not section_cols:
        raise ValueError("No spline section collections found")

    # write trackai.bin 
    with open(filepath, 'wb') as f:
        write_u32(f, TAG_FILE_HEADER)
        write_u32(f, len(section_cols))

        for sec_i, (sec_name, sec_col) in enumerate(section_cols):
            is_closed = sec_col.get('fo2_is_closed', True)
            footer_b64 = sec_col.get('fo2_footer', '')

            # Find curves
            center_obj = find_object_containing(sec_col, "_CenterLine")
            left_obj = find_object_containing(sec_col, "_LeftBoundary")
            right_obj = find_object_containing(sec_col, "_RightBoundary")
            target_obj = find_object_containing(sec_col, "_TargetLine")

            centers = sample_curve_points(center_obj)
            lefts = sample_curve_points(left_obj)
            rights = sample_curve_points(right_obj)
            targets = sample_curve_points(target_obj)

            if not centers:
                centers = lefts or rights
                if not centers:
                    # Empty section (0 nodes) — still write header + footer
                    if footer_b64:
                        footer_bytes = base64.b64decode(footer_b64)
                    else:
                        footer_bytes = struct.pack('<IfIII', sec_i if sec_i > 0 else 0, 0.5, 0, 0, 2)
                    write_u32(f, TAG_SPLINE_SECTION)
                    write_u32(f, 0)
                    f.write(footer_bytes)
                    write_u32(f, TAG_SECTION_SEP)
                    print(f"[TrackAI Export] Section '{sec_name}': 0 nodes (empty)")
                    continue

            # match boundary count to center count
            n = len(centers)
            while len(lefts) < n:
                lefts.append(lefts[-1] if lefts else centers[0])
            while len(rights) < n:
                rights.append(rights[-1] if rights else centers[0])
            while len(targets) < n:
                targets.append(targets[-1] if targets else centers[0])
            lefts = lefts[:n]
            rights = rights[:n]
            targets = targets[:n]

            # gather empties (if present all fields are read from them)
            empties = gather_empties(sec_col, sec_name)

            # footer
            if footer_b64:
                footer_bytes = base64.b64decode(footer_b64)
            else:
                footer_bytes = struct.pack('<IfIII', sec_i if sec_i > 0 else 0, 0.5, 0, 0, 2)

            # build and write section
            write_u32(f, TAG_SPLINE_SECTION)
            write_u32(f, n)

            node_data = build_section_nodes(
                centers, lefts, rights, targets, n, is_closed,
                empties, sec_i)
            f.write(node_data)

            # Footer
            f.write(footer_bytes)

            # separator after every section (including last, required before extra data)
            write_u32(f, TAG_SECTION_SEP)

            print(f"[TrackAI Export] Section '{sec_name}': {n} nodes, "
                  f"closed={is_closed}, empties={'yes' if len(empties)==n else 'no'}")

        # extra data block (startpoints, splitpoints, BV)
        _write_extra_data(f, root_col, section_cols)

    base_dir = os.path.dirname(filepath)

    # splines.ai
    if options.get('export_splines_ai', True):
        if not _export_splines_from_empties(root_col, base_dir):
            splines_raw = root_col.get('fo2_splines_ai', '')
            if splines_raw:
                out_path = os.path.join(base_dir, "splines.ai")
                with open(out_path, 'w', newline='\n') as f:
                    f.write(splines_raw)
                print(f"[TrackAI Export] Wrote splines.ai (verbatim, {len(splines_raw)} chars)")

    # splitpoints.bed
    if options.get('export_splitpoints_bed', True):
        if not _export_splitpoints_from_objects(root_col, base_dir):
            splitpoints_raw = root_col.get('fo2_splitpoints_bed', '')
            if splitpoints_raw:
                out_path = os.path.join(base_dir, "splitpoints.bed")
                with open(out_path, 'w', newline='\n') as f:
                    f.write(splitpoints_raw)
                print(f"[TrackAI Export] Wrote splitpoints.bed (verbatim)")

    # startpoints.bed
    if options.get('export_startpoints_bed', True):
        if not _export_startpoints_from_objects(root_col, base_dir):
            startpoints_raw = root_col.get('fo2_startpoints_bed', '')
            if startpoints_raw:
                out_path = os.path.join(base_dir, "startpoints.bed")
                with open(out_path, 'w', newline='\n') as f:
                    f.write(startpoints_raw)
                print(f"[TrackAI Export] Wrote startpoints.bed (verbatim)")

    print(f"[TrackAI Export] Complete: {filepath}")
    return {'FINISHED'}


# EXTRA DATA BLOCK (startpoints + splitpoints + AI BVH)

def _gather_startpoint_empties(root_col):
    """Collect startpoint empties sorted by index."""
    sp_col = None
    for child in root_col.children:
        if child.name == "TrackAI_Startpoints":
            sp_col = child
            break
    if not sp_col:
        return []

    items = []
    for obj in sp_col.objects:
        idx = obj.get('fo2_startpoint_index', -1)
        rot = obj.get('fo2_startpoint_rotation')
        if idx >= 0 and rot and len(rot) == 9:
            # Use current Blender location converted to FO2 space
            pos = blender_to_fo2(obj.location)
            items.append((idx, pos, tuple(float(v) for v in rot)))
    items.sort(key=lambda x: x[0])
    return items


def _gather_splitpoint_objects(root_col):
    """Collect splitpoint objects sorted by index."""
    sp_col = None
    for child in root_col.children:
        if child.name == "TrackAI_Splitpoints":
            sp_col = child
            break
    if not sp_col:
        return []

    items = []
    for obj in sp_col.objects:
        idx = obj.get('fo2_splitpoint_index', -1)
        pos_orig = obj.get('fo2_splitpoint_position')
        left_orig = obj.get('fo2_splitpoint_left')
        right_orig = obj.get('fo2_splitpoint_right')
        if idx >= 0 and pos_orig and left_orig and right_orig:
            # Mesh origin is at world origin; obj.location is the movement delta
            delta = blender_to_fo2(obj.location)
            pos = (float(pos_orig[0]) + delta[0],
                   float(pos_orig[1]) + delta[1],
                   float(pos_orig[2]) + delta[2])
            left = (float(left_orig[0]) + delta[0],
                    float(left_orig[1]) + delta[1],
                    float(left_orig[2]) + delta[2])
            right = (float(right_orig[0]) + delta[0],
                     float(right_orig[1]) + delta[1],
                     float(right_orig[2]) + delta[2])
            items.append((idx, pos, left, right))
    items.sort(key=lambda x: x[0])
    return items


def _gather_section_node_data(section_cols):
    """Gather FO2-space node positions from all sections for AI BVH generation.

    Returns list of (is_closed, nodes) tuples, where nodes is a list of dicts:
        {center, left, right, index, seq_index, sec_idx}
    """
    all_sections = []
    for sec_i, (sec_name, sec_col) in enumerate(section_cols):
        is_closed = sec_col.get('fo2_is_closed', True)
        empties = gather_empties(sec_col, sec_name)
        nodes = []
        for e in empties:
            center = _read_vec3_prop(e, 'fo2_center', None)
            left = _read_vec3_prop(e, 'fo2_left', None)
            right = _read_vec3_prop(e, 'fo2_right', None)
            idx = int(e.get('fo2_node_index', 0))
            seq = int(e.get('fo2_seq_index', 0))
            if center and left and right:
                # apply movement delta from empty location
                import_bl = (center[0], center[2], center[1])
                delta_bl = (e.location[0] - import_bl[0],
                            e.location[1] - import_bl[1],
                            e.location[2] - import_bl[2])
                delta_fo2 = (delta_bl[0], delta_bl[2], delta_bl[1])
                if abs(delta_fo2[0]) > 1e-6 or abs(delta_fo2[1]) > 1e-6 or abs(delta_fo2[2]) > 1e-6:
                    center = (center[0] + delta_fo2[0], center[1] + delta_fo2[1], center[2] + delta_fo2[2])
                    left = (left[0] + delta_fo2[0], left[1] + delta_fo2[1], left[2] + delta_fo2[2])
                    right = (right[0] + delta_fo2[0], right[1] + delta_fo2[1], right[2] + delta_fo2[2])
                nodes.append({
                    'center': center, 'left': left, 'right': right,
                    'index': idx, 'seq_index': seq, 'sec_idx': sec_i,
                })
        all_sections.append((is_closed, nodes))
    return all_sections


def _compute_segment_aabb(node_a, node_b):
    """Compute 2D (XZ) AABB covering the segment between two nodes.
    Returns (min_x, 0, min_z, max_x, 0, max_z)."""
    xs = [node_a['center'][0], node_a['left'][0], node_a['right'][0],
          node_b['center'][0], node_b['left'][0], node_b['right'][0]]
    zs = [node_a['center'][2], node_a['left'][2], node_a['right'][2],
          node_b['center'][2], node_b['left'][2], node_b['right'][2]]
    return (min(xs), 0.0, min(zs), max(xs), 0.0, max(zs))


def _generate_ai_bvh(all_sections):
    """Generate AI BVH leaf entries from section node data.

    Each leaf covers the segment from a node to its linked-list successor.
    For closed loops, the last node wraps to the first.
    For open paths, the last node's successor is in another section
    (found by matching the node's index value to seq_index in all sections).

    all_sections: list of (is_closed, nodes_list) tuples.

    Returns list of (node_ref, aabb) tuples, where:
        node_ref = (sec_idx << 24) | seq_index
        aabb = (min_x, 0, min_z, max_x, 0, max_z)
    """
    # Build global index for cross-section lookups:
    # For each section, map seq_index -> node dict
    sec_by_seq = []
    for is_closed, sec_nodes in all_sections:
        by_seq = {}
        for n in sec_nodes:
            by_seq[n['seq_index']] = n
        sec_by_seq.append(by_seq)

    leaves = []
    for sec_i, (is_closed, sec_nodes) in enumerate(all_sections):
        n_nodes = len(sec_nodes)
        if n_nodes == 0:
            continue

        for i, node in enumerate(sec_nodes):
            # Find successor node
            next_node = None
            if i < n_nodes - 1:
                # Not the last node: next is simply the next in file order
                next_node = sec_nodes[i + 1]
            elif is_closed:
                # Last node of closed loop: wraps to first
                next_node = sec_nodes[0]
            else:
                # Last node of open path: cross-section reference
                target_idx = node['index']
                # Search other sections for a node with this seq_index
                for other_sec_i, other_by_seq in enumerate(sec_by_seq):
                    if other_sec_i == sec_i:
                        continue
                    if target_idx in other_by_seq:
                        next_node = other_by_seq[target_idx]
                        break
                if next_node is None:
                    # Fallback: use current node's own AABB
                    next_node = node

            aabb = _compute_segment_aabb(node, next_node)
            node_ref = (sec_i << 24) | node['seq_index']
            leaves.append((node_ref, aabb))

    return leaves


def _build_bvh_tree(num_leaves):
    """Build a balanced binary tree over num_leaves leaf entries.

    Returns a list of (index, type) pairs where:
      type 0 = internal node reference (index points into this array)
      type 2 = leaf entry reference (index is the leaf entry index)

    The root's two children are always at pairs[0] and pairs[1].
    An internal ref to index N means its children are at pairs[N] and pairs[N+1].
    """
    if num_leaves == 0:
        return []
    if num_leaves == 1:
        return [(0, 2), (0, 2)]  # degenerate: both children point to leaf 0

    pairs = [None] * (num_leaves * 2)
    next_idx = [2]  # reserve 0,1 for root; use list for closure mutation

    def build(start, end):
        count = end - start
        if count == 1:
            return (start, 2)  # leaf reference

        mid = start + count // 2

        # Allocate pair for this internal node (pre-order)
        my_idx = next_idx[0]
        next_idx[0] += 2
        pairs[my_idx] = None      # placeholder
        pairs[my_idx + 1] = None

        left = build(start, mid)
        right = build(mid, end)

        pairs[my_idx] = left
        pairs[my_idx + 1] = right
        return (my_idx, 0)  # ref: children at pairs[my_idx] and pairs[my_idx+1]

    mid = num_leaves // 2
    left = build(0, mid)
    right = build(mid, num_leaves)
    pairs[0] = left
    pairs[1] = right

    return pairs[:next_idx[0]]


def _write_extra_data(f, root_col, section_cols):
    """Write the complete extra data block: startpoints + splitpoints + AI BVH."""
    # TAG_EXTRA_START
    write_u32(f, TAG_EXTRA_START)

    # startpoints
    startpoints = _gather_startpoint_empties(root_col)
    write_u32(f, TAG_STARTPOINTS)
    write_u32(f, len(startpoints))
    for idx, pos, rot in startpoints:
        write_vec3(f, pos)
        f.write(struct.pack('<9f', *rot))
    print(f"[TrackAI Export] Wrote {len(startpoints)} startpoints")

    # splitpoints
    splitpoints = _gather_splitpoint_objects(root_col)
    write_u32(f, TAG_SPLITPOINTS)
    write_u32(f, TAG_SPLITPOINT_SUB)
    write_u32(f, len(splitpoints))
    for idx, pos, left, right in splitpoints:
        write_vec3(f, pos)
        write_vec3(f, left)
        write_vec3(f, right)
    print(f"[TrackAI Export] Wrote {len(splitpoints)} splitpoints")

    # AI BV
    all_sections = _gather_section_node_data(section_cols)
    total_nodes = sum(len(nodes) for _, nodes in all_sections)

    if total_nodes > 0:
        leaves = _generate_ai_bvh(all_sections)

        # header (24 bytes)
        write_u32(f, TAG_AI_BVH_1)
        write_u32(f, TAG_AI_BVH_2)
        write_u32(f, TAG_SPLINE_SECTION)  # reused tag
        write_u32(f, total_nodes)
        write_u32(f, TAG_AI_BVH_HDR4)
        write_u32(f, 0)  # reserved

        # leaf entries (total_nodes × 32 bytes)
        # try to match original leaf ordering for byte-perfect round-trip
        stored_tree_b64 = root_col.get('fo2_ai_bvh_tree', '')
        stored_leaf_count = root_col.get('fo2_ai_bvh_leaf_count', -1)
        stored_leaf_order = root_col.get('fo2_ai_bvh_leaf_order', None)

        use_stored_order = (stored_tree_b64
                            and stored_leaf_count == total_nodes
                            and stored_leaf_order
                            and len(stored_leaf_order) == total_nodes)

        if use_stored_order:
            # build lookup: node_ref -> aabb
            leaf_by_ref = {ref: aabb for ref, aabb in leaves}
            # reorder leaves to match original
            ordered_leaves = []
            for node_ref in stored_leaf_order:
                node_ref = int(node_ref)
                if node_ref in leaf_by_ref:
                    ordered_leaves.append((node_ref, leaf_by_ref[node_ref]))
                else:
                    # fallback: zero AABB (shouldn't happen if count matches)
                    ordered_leaves.append((node_ref, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
            leaves = ordered_leaves

        for i, (node_ref, aabb) in enumerate(leaves):
            write_u32(f, node_ref)
            write_f32(f, aabb[0])  # min_x
            write_f32(f, aabb[1])  # 0
            write_f32(f, aabb[2])  # min_z
            write_f32(f, aabb[3])  # max_x
            write_f32(f, aabb[4])  # 0
            write_f32(f, aabb[5])  # max_z
            # tree_index: sequential for all but last, which gets LEAVES_END tag
            if i < len(leaves) - 1:
                write_u32(f, i + 1)
            else:
                write_u32(f, TAG_AI_BVH_LEAVES_END)

        print(f"[TrackAI Export] Wrote AI BVH: {len(leaves)} leaf entries"
              f"{' (original order)' if use_stored_order else ''}")

        # internal tree structure
        if use_stored_order:
            # reuse original tree structure (node count unchanged)
            tree_data = base64.b64decode(stored_tree_b64)
            f.write(tree_data)
            print(f"[TrackAI Export] Reused stored AI BVH tree ({len(tree_data)} bytes)")
        else:
            # generate balanced binary tree
            tree_pairs = _build_bvh_tree(len(leaves))
            pair_count = len(tree_pairs)

            write_u32(f, pair_count)
            write_u32(f, 0x00040376)  # tree section tag
            for ref, typ in tree_pairs:
                write_u32(f, ref)
                write_u32(f, typ)
            write_u32(f, TAG_AI_BVH_END1)
            write_u32(f, TAG_AI_BVH_END2)

            if stored_leaf_count >= 0 and stored_leaf_count != total_nodes:
                print(f"[TrackAI Export] Generated new AI BVH tree "
                      f"(node count changed: {stored_leaf_count} -> {total_nodes}, "
                      f"{pair_count} tree pairs)")
            else:
                print(f"[TrackAI Export] Generated AI BVH tree "
                      f"({pair_count} tree pairs)")

    # FILE_END
    write_u32(f, TAG_FILE_END)




# COMPANION FILE GENERATION (when no stored raw data)

def _export_splines_from_empties(root_col, base_dir):
    """Generate splines.ai from AISpline empties. Returns True if written."""
    spline_col = None
    for child in root_col.children:
        if child.name == "TrackAI_AISplines":
            spline_col = child
            break
    if not spline_col:
        return False

    # Group empties by spline name
    splines = {}
    for obj in spline_col.objects:
        name = obj.get('fo2_spline_name', '')
        if not name:
            continue
        idx = obj.get('fo2_spline_index', 0)

        # Delta approach: stored game coords + movement delta
        orig_pos = obj.get('fo2_spline_position')
        if orig_pos and len(orig_pos) == 3:
            # Derive import-time Blender location from stored FO2 position
            import_loc = (orig_pos[0], orig_pos[2], orig_pos[1])  # fo2_to_blender
            # Compute how much the user moved this empty in Blender
            delta_bl = (obj.location[0] - import_loc[0],
                        obj.location[1] - import_loc[1],
                        obj.location[2] - import_loc[2])
            # Convert delta to game space (swap Y/Z)
            delta_game = (delta_bl[0], delta_bl[2], delta_bl[1])
            fo2_pos = (orig_pos[0] + delta_game[0],
                       orig_pos[1] + delta_game[1],
                       orig_pos[2] + delta_game[2])
        else:
            # Fallback: direct conversion from Blender location
            fo2_pos = blender_to_fo2(obj.location)

        if name not in splines:
            splines[name] = []
        splines[name].append((idx, fo2_pos))

    if not splines:
        return False

    out_path = os.path.join(base_dir, "splines.ai")
    with open(out_path, 'w', newline='\n') as f:
        f.write(f"Count = {len(splines)}\n\nSplines = {{")
        for name, pts in splines.items():
            pts.sort(key=lambda x: x[0])
            f.write(f'\n\t["{name}"] = {{\n')
            f.write(f"\t\tCount = {len(pts)},\n")
            f.write(f"\t\tControlPoints = {{")
            for i, (idx, pos) in enumerate(pts):
                f.write(f"\n\t\t\t[{i+1}] = {{ {pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f} }},")
            f.write(f"\n\t\t}},\n\t}},\n")
        f.write("\n}\n")
    print(f"[TrackAI Export] Generated splines.ai from empties")
    return True


def _export_splitpoints_from_objects(root_col, base_dir):
    """Generate splitpoints.bed from splitpoint objects using delta approach. Returns True if written."""
    split_col = None
    for child in root_col.children:
        if child.name == "TrackAI_Splitpoints":
            split_col = child
            break
    if not split_col:
        return False

    splitpoints = []
    for obj in split_col.objects:
        idx = obj.get('fo2_splitpoint_index', -1)
        if idx < 0:
            continue

        # Try delta approach using .bed coords
        bed_pos = obj.get('fo2_bed_splitpoint_position')
        bed_left = obj.get('fo2_bed_splitpoint_left')
        bed_right = obj.get('fo2_bed_splitpoint_right')

        if bed_pos and bed_left and bed_right:
            # Splitpoint mesh origin is at world origin, so delta = obj.location
            delta_bl = (obj.location[0], obj.location[1], obj.location[2])
            # Convert delta to game space (swap Y/Z)
            delta_game = (delta_bl[0], delta_bl[2], delta_bl[1])
            # Apply delta to all three .bed points
            pos = (bed_pos[0] + delta_game[0],
                   bed_pos[1] + delta_game[1],
                   bed_pos[2] + delta_game[2])
            left = (bed_left[0] + delta_game[0],
                    bed_left[1] + delta_game[1],
                    bed_left[2] + delta_game[2])
            right = (bed_right[0] + delta_game[0],
                     bed_right[1] + delta_game[1],
                     bed_right[2] + delta_game[2])
        else:
            # Fallback: read binary coords from custom properties
            bin_pos = obj.get('fo2_splitpoint_position')
            bin_left = obj.get('fo2_splitpoint_left')
            bin_right = obj.get('fo2_splitpoint_right')
            if bin_pos and bin_left and bin_right:
                pos = tuple(float(v) for v in bin_pos)
                left = tuple(float(v) for v in bin_left)
                right = tuple(float(v) for v in bin_right)
            else:
                continue

        splitpoints.append((idx, pos, left, right))

    if not splitpoints:
        return False

    # Sort by index
    splitpoints.sort(key=lambda x: x[0])

    out_path = os.path.join(base_dir, "splitpoints.bed")
    with open(out_path, 'w', newline='\n') as f:
        f.write(f"Count = {len(splitpoints)}\n\nSplitpoints = {{")
        for i, (idx, pos, left, right) in enumerate(splitpoints):
            f.write(f"\n\t[{i+1}] = {{")
            f.write(f"\n\t\tPosition = {{ {pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f} }},")
            f.write(f"\n\t\tLeft = {{ {left[0]:.6f}, {left[1]:.6f}, {left[2]:.6f} }},")
            f.write(f"\n\t\tRight = {{ {right[0]:.6f}, {right[1]:.6f}, {right[2]:.6f} }},")
            f.write(f"\n\n\t}},")
        f.write("\n}\n")
    print(f"[TrackAI Export] Generated splitpoints.bed ({len(splitpoints)} entries)")
    return True


def _export_startpoints_from_objects(root_col, base_dir):
    """Generate startpoints.bed from startpoint empties using delta approach. Returns True if written."""
    start_col = None
    for child in root_col.children:
        if child.name == "TrackAI_Startpoints":
            start_col = child
            break
    if not start_col:
        return False

    startpoints = []
    for obj in start_col.objects:
        idx = obj.get('fo2_startpoint_index', -1)
        if idx < 0:
            continue

        # Try delta approach using .bed coords
        bed_pos = obj.get('fo2_bed_startpoint_position')
        bed_rot = obj.get('fo2_bed_startpoint_rotation')
        bin_pos = obj.get('fo2_startpoint_position')

        if bed_pos and bed_rot and bin_pos and len(bed_pos) == 3 and len(bed_rot) == 9 and len(bin_pos) == 3:
            # Derive import-time Blender location from stored binary position
            import_loc = (bin_pos[0], bin_pos[2], bin_pos[1])  # fo2_to_blender
            # Compute movement delta in Blender space
            delta_bl = (obj.location[0] - import_loc[0],
                        obj.location[1] - import_loc[1],
                        obj.location[2] - import_loc[2])
            # Convert delta to game space (swap Y/Z)
            delta_game = (delta_bl[0], delta_bl[2], delta_bl[1])
            pos = (bed_pos[0] + delta_game[0],
                   bed_pos[1] + delta_game[1],
                   bed_pos[2] + delta_game[2])
            rot = tuple(float(v) for v in bed_rot)
        else:
            # Fallback: no .bed data, use binary rotation and Blender location
            rot_raw = obj.get('fo2_startpoint_rotation')
            if rot_raw and len(rot_raw) == 9:
                pos = blender_to_fo2(obj.location)
                rot = tuple(float(v) for v in rot_raw)
            else:
                continue

        startpoints.append((idx, pos, rot))

    if not startpoints:
        return False

    # Sort by index
    startpoints.sort(key=lambda x: x[0])

    out_path = os.path.join(base_dir, "startpoints.bed")
    with open(out_path, 'w', newline='\n') as f:
        f.write(f"Count = {len(startpoints)}\n\nStartpoints = {{")
        for i, (idx, pos, rot) in enumerate(startpoints):
            # Clamp near-zero values like the C++ tool does
            rot_c = tuple(0.0 if abs(v) < 0.001 else v for v in rot)
            f.write(f"\n\t[{i+1}] = {{")
            f.write(f"\n\t\tPosition = {{ {pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f} }},")
            f.write(f"\n\t\tOrientation = {{")
            f.write(f"\n\t\t\t[\"x\"]={{{rot_c[0]:.6f},{rot_c[1]:.6f},{rot_c[2]:.6f}}},")
            f.write(f"\n\t\t\t[\"y\"]={{{rot_c[3]:.6f},{rot_c[4]:.6f},{rot_c[5]:.6f}}},")
            f.write(f"\n\t\t\t[\"z\"]={{{rot_c[6]:.6f},{rot_c[7]:.6f},{rot_c[8]:.6f}}},")
            f.write(f"\n\t\t}},")
            f.write(f"\n\n\t}},")
        f.write("\n}\n")
    print(f"[TrackAI Export] Generated startpoints.bed ({len(startpoints)} entries)")
    return True


# OPERATOR

class ExportTrackAI(bpy.types.Operator, ExportHelper):
    """Export FlatOut 2 Track AI path data"""
    bl_idname = "export_scene.fo2_trackai"
    bl_label = "Export FO2 Track AI"
    bl_options = {'PRESET'}

    filename_ext = ".bin"
    filter_glob: StringProperty(default="*.bin", options={'HIDDEN'})

    export_splines_ai: BoolProperty(
        name="splines.ai",
        description="Export AI border splines companion file",
        default=True,
    )

    export_startpoints_bed: BoolProperty(
        name="startpoints.bed",
        description="Export startpoints companion file",
        default=True,
    )

    export_splitpoints_bed: BoolProperty(
        name="splitpoints.bed",
        description="Export splitpoints companion file",
        default=True,
    )

    def draw(self, context):
        layout = self.layout

        box = layout.box()
        box.label(text="Companion Files", icon='FILE_FOLDER')
        box.prop(self, "export_splines_ai")
        box.prop(self, "export_startpoints_bed")
        box.prop(self, "export_splitpoints_bed")

    def execute(self, context):
        options = {
            'export_splines_ai': self.export_splines_ai,
            'export_startpoints_bed': self.export_startpoints_bed,
            'export_splitpoints_bed': self.export_splitpoints_bed,
        }
        try:
            result = export_trackai(self.filepath, context, options)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            import traceback; traceback.print_exc()
            return {'CANCELLED'}
        self.report({'INFO'}, f"Exported Track AI to {os.path.basename(self.filepath)}")
        return result


def menu_func_export(self, context):
    self.layout.operator(ExportTrackAI.bl_idname, text="FlatOut 2 TrackAI (.bin)")


def register():
    bpy.utils.register_class(ExportTrackAI)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    bpy.utils.unregister_class(ExportTrackAI)


if __name__ == "__main__":
    register()