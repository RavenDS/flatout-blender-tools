bl_info = {
    "name":        "FlatOut 2 TrackAI Importer",
    "author":      "ravenDS",
    "version":     (2, 0, 1),
    "blender":     (3, 6, 0),
    "location":    "File > Import > FlatOut 2 TrackAI (.bin)",
    "description": "Import FlatOut 2 AI path data (trackai.bin +.bed)",
    "category":    "Import-Export",
    "doc_url":     "https://github.com/RavenDS",
    "tracker_url": "https://github.com/RavenDS/flatout-blender-tools/issues",
}

import bpy
import struct
import os
import math
import base64
from bpy.props import (StringProperty, BoolProperty, FloatProperty, EnumProperty)
from bpy_extras.io_utils import ImportHelper
from mathutils import Vector, Matrix

# CONSTANTS

TAG_FILE_HEADER      = 0x00270276  # file start
TAG_SPLINE_SECTION   = 0x00290276  # spline section header (count follows)
TAG_NODE_START       = 0x00230276  # AI node begin
TAG_NODE_END         = 0x00240276  # AI node end
TAG_SECTION_SEP      = 0x00260276  # separator between sections
TAG_FILE_END         = 0x00280276  # file end
TAG_EXTRA_START      = 0x00280876  # extra data wrapper
TAG_STARTPOINTS      = 0x00300876  # embedded startpoints
TAG_SPLITPOINTS      = 0x00310876  # embedded splitpoints
TAG_SPLITPOINT_SUB   = 0x00010976  # splitpoints subtag (always)
TAG_AI_BVH_1         = 0x00020976  # AI BVH sub-block start
TAG_AI_BVH_2         = 0x00290876  # AI BVH secondary tag
TAG_AI_BVH_LEAVES_END = 0x00030376  # marks end of leaf entries (in last leaf's tree_index)
TAG_AI_BVH_END1      = 0x00050376  # AI BVH footer tag 1
TAG_AI_BVH_END2      = 0x00010376  # AI BVH footer tag 2

NODE_SIZE = 208  # bytes per AI node (including start+end tags)


# DATA STRUCTURES

class AINode:
    """Single AI waypoint node on the track."""
    __slots__ = (
        'index', 'unk1', 'prev_index', 'unk2',
        'rotation',           # 3x3 rotation matrix (9 floats)
        'center',             # center of track at this point
        'left',               # left boundary position
        'right',              # right boundary position
        'mid',                # midpoint / AI target
        'target',             # adjusted target (differs on curves)
        'forward',            # forward direction (normalized)
        'right_dir',          # right/lateral direction (normalized)
        'interp_weights',     # 3 interpolation weights
        'width_left',         # track half-width to left
        'width_right',        # track half-width to right
        'cumul_distance',     # cumulative distance along spline
        'unk_neg1',           # always -1.0
        'speed_hint',         # speed / priority hint
        'unk3', 'sentinel1',
        'speed_hint2',        # repeat of speed_hint
        'flag', 'seq_index',
        'unk4', 'sentinel2', 'unk5',
    )

    def __repr__(self):
        return (f"AINode(idx={self.index}, prev={self.prev_index}, "
                f"center=({self.center[0]:.1f}, {self.center[1]:.1f}, {self.center[2]:.1f}), "
                f"dist={self.cumul_distance:.1f})")


class AISplineSection:
    """One closed-loop AI spline (main racing line, pit lane, shortcuts, etc.)."""
    __slots__ = ('nodes', 'footer', 'section_index')

    def __init__(self):
        self.nodes = []
        self.footer = None
        self.section_index = 0


class TrackAIData:
    """Complete parsed trackai.bin data."""
    __slots__ = ('sections', 'num_sections', 'extra_data',
                 'startpoints', 'splitpoints',
                 'ai_splines', 'splines_ai_raw',
                 'bed_splitpoints_raw', 'bed_startpoints_raw',
                 'ai_bvh_tree_data', 'ai_bvh_leaf_order')

    def __init__(self):
        self.sections = []
        self.num_sections = 0
        self.extra_data = b''  # unparsed data after main splines
        self.startpoints = []
        self.splitpoints = []
        self.ai_splines = []
        self.splines_ai_raw = ''
        self.bed_splitpoints_raw = ''
        self.bed_startpoints_raw = ''
        self.ai_bvh_tree_data = b''  # internal tree nodes (after leaf entries, before FILE_END)
        self.ai_bvh_leaf_order = []  # list of node_ref values for leaf ordering


class Startpoint:
    __slots__ = ('position', 'rotation')

class Splitpoint:
    __slots__ = ('position', 'left', 'right')

class AIBorderSpline:
    __slots__ = ('name', 'points')
    def __init__(self, name=''):
        self.name = name
        self.points = []


# PARSER

def read_u32(data, offset):
    return struct.unpack_from('<I', data, offset)[0], offset + 4

def read_i32(data, offset):
    return struct.unpack_from('<i', data, offset)[0], offset + 4

def read_f32(data, offset):
    return struct.unpack_from('<f', data, offset)[0], offset + 4

def read_vec3(data, offset):
    x, y, z = struct.unpack_from('<3f', data, offset)
    return (x, y, z), offset + 12


def parse_node(data, offset):
    """Parse a single 208-byte AI node starting at offset."""
    node = AINode()
    start = offset

    tag, offset = read_u32(data, offset)
    assert tag == TAG_NODE_START, f"Expected node start tag at {start}, got 0x{tag:08x}"

    node.index, offset = read_u32(data, offset)
    node.unk1, offset = read_u32(data, offset)
    node.prev_index, offset = read_u32(data, offset)
    node.unk2, offset = read_u32(data, offset)

    # 3x3 rotation matrix
    node.rotation = struct.unpack_from('<9f', data, offset)
    offset += 36

    # positions
    node.center, offset = read_vec3(data, offset)
    node.left, offset = read_vec3(data, offset)
    node.right, offset = read_vec3(data, offset)
    node.mid, offset = read_vec3(data, offset)
    node.target, offset = read_vec3(data, offset)

    # directions
    node.forward, offset = read_vec3(data, offset)
    node.right_dir, offset = read_vec3(data, offset)

    # weights and widths
    node.interp_weights, offset = read_vec3(data, offset)
    node.width_left, offset = read_f32(data, offset)
    node.width_right, offset = read_f32(data, offset)

    # distance and tail data
    node.cumul_distance, offset = read_f32(data, offset)
    node.unk_neg1, offset = read_f32(data, offset)
    node.speed_hint, offset = read_f32(data, offset)
    node.unk3, offset = read_i32(data, offset)
    node.sentinel1, offset = read_i32(data, offset)
    node.speed_hint2, offset = read_f32(data, offset)
    node.flag, offset = read_u32(data, offset)
    node.seq_index, offset = read_u32(data, offset)
    node.unk4, offset = read_u32(data, offset)
    node.sentinel2, offset = read_i32(data, offset)
    node.unk5, offset = read_u32(data, offset)

    end_tag, offset = read_u32(data, offset)
    assert end_tag == TAG_NODE_END, f"Expected node end tag, got 0x{end_tag:08x}"
    assert offset - start == NODE_SIZE, f"Node size mismatch: {offset - start} != {NODE_SIZE}"

    return node, offset


def parse_trackai(filepath):
    """Parse a trackai.bin file and return TrackAIData."""
    with open(filepath, 'rb') as f:
        data = f.read()

    result = TrackAIData()
    offset = 0

    # file header
    tag, offset = read_u32(data, offset)
    if tag != TAG_FILE_HEADER:
        raise ValueError(f"Not a valid trackai.bin: expected 0x{TAG_FILE_HEADER:08x}, got 0x{tag:08x}")

    result.num_sections, offset = read_u32(data, offset)
    print(f"[TrackAI] File size: {len(data)} bytes, sections: {result.num_sections}")

    for sec_idx in range(result.num_sections):
        section = AISplineSection()
        section.section_index = sec_idx

        # Section header
        tag, offset = read_u32(data, offset)
        if tag != TAG_SPLINE_SECTION:
            raise ValueError(f"Expected spline section tag at offset {offset-4}, got 0x{tag:08x}")

        num_nodes, offset = read_u32(data, offset)
        print(f"[TrackAI]   Section {sec_idx}: {num_nodes} nodes")

        # Parse nodes
        for _ in range(num_nodes):
            node, offset = parse_node(data, offset)
            section.nodes.append(node)

        # section footer (20 bytes before the separator tag or end)
        # footer: section_id(u32), speed_factor(f32), 0(u32), 0(u32), next_section_id(u32)
        section.footer = data[offset:offset+20]
        offset += 20

        # section separator (except possibly last)
        if offset < len(data):
            peek, _ = read_u32(data, offset)
            if peek == TAG_SECTION_SEP:
                if sec_idx < result.num_sections - 1:
                    offset += 4  # consume inter-section separator
                # else: trailing separator stays in extra_data

        result.sections.append(section)

    # everything after is extra data (including any trailing separator, startpoints, splitpoints, bvtree)
    result.extra_data = data[offset:]
    print(f"[TrackAI]   Extra data after splines: {len(result.extra_data)} bytes")

    # parse structured fields out of the extra data block
    _parse_extra_data(result)

    return result


def _parse_extra_data(result):
    """Extract startpoints, splitpoints, and AI BVH from the extra data block.
    Components are stored individually on result for proper reconstruction on export."""
    extra = result.extra_data
    if len(extra) < 8:
        return
    offset = 0
    tag, offset = read_u32(extra, offset)

    # skip trailing section separator if present
    if tag == TAG_SECTION_SEP:
        if offset >= len(extra):
            return
        tag, offset = read_u32(extra, offset)

    if tag != TAG_EXTRA_START:
        return

    # startpoints (0x00300876)
    if offset >= len(extra):
        return
    tag, offset = read_u32(extra, offset)
    if tag == TAG_STARTPOINTS:
        count, offset = read_u32(extra, offset)
        for i in range(count):
            sp = Startpoint()
            sp.position, offset = read_vec3(extra, offset)
            sp.rotation = struct.unpack_from('<9f', extra, offset)
            offset += 36
            result.startpoints.append(sp)
        print(f"[TrackAI]   Parsed {count} embedded startpoints")

    # splitpoints (0x00310876)
    if offset >= len(extra):
        return
    tag, offset = read_u32(extra, offset)
    if tag == TAG_SPLITPOINTS:
        subtag, offset = read_u32(extra, offset)  # 0x00010976
        count, offset = read_u32(extra, offset)
        for i in range(count):
            sp = Splitpoint()
            sp.position, offset = read_vec3(extra, offset)
            sp.left, offset = read_vec3(extra, offset)
            sp.right, offset = read_vec3(extra, offset)
            result.splitpoints.append(sp)
        print(f"[TrackAI]   Parsed {count} embedded splitpoints")

    # AI BVH (0x00020976)
    if offset >= len(extra):
        return
    tag, offset = read_u32(extra, offset)
    if tag == TAG_AI_BVH_1:
        tag2, offset = read_u32(extra, offset)   # 0x00290876
        tag3, offset = read_u32(extra, offset)   # 0x00290276 (TAG_SPLINE_SECTION reused)
        total_nodes, offset = read_u32(extra, offset)
        tag4, offset = read_u32(extra, offset)   # 0x00020376
        reserved, offset = read_u32(extra, offset)  # 0

        # Skip leaf entries (total_nodes × 32 bytes)
        leaf_end = offset + total_nodes * 32
        if leaf_end > len(extra):
            print(f"[TrackAI]   WARNING: AI BVH leaf data truncated")
            return

        # Extract leaf ordering (list of node_ref values) for round-trip
        leaf_order = []
        for i in range(total_nodes):
            node_ref = struct.unpack_from('<I', extra, offset + i * 32)[0]
            leaf_order.append(node_ref)
        result.ai_bvh_leaf_order = leaf_order

        print(f"[TrackAI]   Parsed AI BVH: {total_nodes} leaf entries")
        offset = leaf_end

        # Everything from here until TAG_FILE_END is internal tree data
        # Find TAG_FILE_END from the end
        file_end_offset = len(extra) - 4
        end_tag = struct.unpack_from('<I', extra, file_end_offset)[0]
        if end_tag == TAG_FILE_END:
            # Store internal tree data (between leaves and FILE_END)
            result.ai_bvh_tree_data = extra[offset:file_end_offset]
            print(f"[TrackAI]   AI BVH internal tree: {len(result.ai_bvh_tree_data)} bytes")
        else:
            # No FILE_END tag found, store everything remaining
            result.ai_bvh_tree_data = extra[offset:]
            print(f"[TrackAI]   AI BVH tree data (no FILE_END): {len(result.ai_bvh_tree_data)} bytes")


def _parse_splines_ai(filepath, result):
    """Parse the splines.ai companion file."""
    if not os.path.isfile(filepath):
        return
    with open(filepath, 'r') as f:
        result.splines_ai_raw = f.read()
    current_spline = None
    for line in result.splines_ai_raw.splitlines():
        stripped = line.strip()
        if stripped.startswith('["') and '"]' in stripped:
            name = stripped.split('"')[1]
            current_spline = AIBorderSpline(name)
            result.ai_splines.append(current_spline)
        elif current_spline and stripped.startswith('[') and '= {' in stripped:
            try:
                parts = stripped.split('{')[1].split('}')[0]
                coords = [float(v.strip()) for v in parts.split(',')]
                if len(coords) == 3:
                    current_spline.points.append(tuple(coords))
            except (ValueError, IndexError):
                pass
    for sp in result.ai_splines:
        print(f"[TrackAI]   AI Spline '{sp.name}': {len(sp.points)} points")


def _parse_companion_files(filepath, result, custom_paths=None):
    """Load companion text files. Uses custom paths if provided, otherwise
    looks in the same directory as the .bin file."""
    base_dir = os.path.dirname(filepath)
    cp = custom_paths or {}

    # splines.ai
    splines_path = cp.get('splines_ai_path', '')
    if not splines_path:
        splines_path = os.path.join(base_dir, "splines.ai")
    _parse_splines_ai(splines_path, result)

    # splitpoints.bed
    split_path = cp.get('splitpoints_bed_path', '')
    if not split_path:
        split_path = os.path.join(base_dir, "splitpoints.bed")
    if os.path.isfile(split_path):
        with open(split_path, 'r') as f:
            result.bed_splitpoints_raw = f.read()
        print(f"[TrackAI]   Loaded splitpoints.bed from {split_path}")

    # startpoints.bed
    start_path = cp.get('startpoints_bed_path', '')
    if not start_path:
        start_path = os.path.join(base_dir, "startpoints.bed")
    if os.path.isfile(start_path):
        with open(start_path, 'r') as f:
            result.bed_startpoints_raw = f.read()
        print(f"[TrackAI]   Loaded startpoints.bed from {start_path}")


# COORDINATE TRANSFORMS
#
# FlatOut 2 uses (X_right, Y_up, Z_forward)
# Blender   uses (X_right, Y_forward, Z_up)
#
# Mapping: bl = (fo2_x, fo2_z, fo2_y)
# Or equivalently: swap Y and Z

def fo2_to_blender(pos, scale=1.0):
    """Convert FO2 position (x, y, z) to Blender (x, z, y)."""
    return Vector((pos[0] * scale, pos[2] * scale, pos[1] * scale))


def fo2_dir_to_blender(d):
    """Convert FO2 direction to Blender."""
    return Vector((d[0], d[2], d[1]))


# BLENDER SCENE CREATION

# color palette for spline sections
SECTION_COLORS = [
    (0.2, 0.8, 0.2, 1.0),   # Green  - main racing line
    (0.8, 0.6, 0.1, 1.0),   # Orange - pit lane / shortcut
    (0.2, 0.5, 0.9, 1.0),   # Blue   - alternate path
    (0.8, 0.2, 0.2, 1.0),   # Red    - shortcut / hazard
    (0.7, 0.2, 0.8, 1.0),   # Purple
    (0.2, 0.8, 0.8, 1.0),   # Cyan
    (0.9, 0.9, 0.2, 1.0),   # Yellow
    (0.6, 0.6, 0.6, 1.0),   # Gray
]

BOUNDARY_COLORS = [
    (0.15, 0.55, 0.15, 0.7),
    (0.6, 0.4, 0.05, 0.7),
    (0.15, 0.35, 0.65, 0.7),
    (0.6, 0.15, 0.15, 0.7),
    (0.5, 0.15, 0.6, 0.7),
    (0.15, 0.6, 0.6, 0.7),
    (0.65, 0.65, 0.15, 0.7),
    (0.4, 0.4, 0.4, 0.7),
]


def get_or_create_material(name, color):
    """Get or create a simple emission material."""
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        output = nodes.new('ShaderNodeOutputMaterial')
        emission = nodes.new('ShaderNodeEmission')
        emission.inputs['Color'].default_value = color
        emission.inputs['Strength'].default_value = 1.0
        links.new(emission.outputs['Emission'], output.inputs['Surface'])
    return mat


def create_curve_from_positions(name, positions, closed, material, width=0.3):
    """Create a Blender curve object from a list of Vector positions."""
    curve_data = bpy.data.curves.new(name, type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.resolution_u = 12
    curve_data.bevel_depth = width
    curve_data.bevel_resolution = 2

    spline = curve_data.splines.new('NURBS')
    spline.points.add(len(positions) - 1)
    for i, pos in enumerate(positions):
        spline.points[i].co = (pos.x, pos.y, pos.z, 1.0)
    spline.use_cyclic_u = closed
    spline.order_u = 3

    obj = bpy.data.objects.new(name, curve_data)
    obj.data.materials.append(material)
    return obj


def create_mesh_ribbon(name, left_positions, right_positions, closed, material):
    """Create a mesh ribbon between left and right boundary positions."""
    n = len(left_positions)
    if n < 2:
        return None

    verts = []
    faces = []

    for i in range(n):
        verts.append(left_positions[i])
        verts.append(right_positions[i])

    for i in range(n - 1):
        li = i * 2
        ri = li + 1
        li_next = (i + 1) * 2
        ri_next = li_next + 1
        faces.append((li, ri, ri_next, li_next))

    if closed and n > 2:
        li = (n - 1) * 2
        ri = li + 1
        faces.append((li, ri, 1, 0))

    import bmesh
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([v[:] for v in verts], [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    obj.data.materials.append(material)
    return obj


def create_direction_arrows(name, nodes, scale, material, arrow_scale=2.0):
    """Create small arrow meshes showing forward direction at each node."""
    import bmesh

    verts = []
    faces = []
    vert_idx = 0

    for node in nodes:
        center = fo2_to_blender(node.center, scale)
        fwd = fo2_dir_to_blender(node.forward) * arrow_scale
        right = fo2_dir_to_blender(node.right_dir) * (arrow_scale * 0.3)

        # Simple triangle arrow
        tip = center + fwd
        base_l = center - right
        base_r = center + right

        verts.extend([center[:], tip[:], base_l[:], base_r[:]])
        faces.append((vert_idx, vert_idx + 2, vert_idx + 1))
        faces.append((vert_idx, vert_idx + 1, vert_idx + 3))
        vert_idx += 4

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    obj.data.materials.append(material)
    return obj


def create_node_empties(name, nodes, scale, collection):
    """Create empty objects at each node with custom properties."""
    for file_pos, node in enumerate(nodes):
        pos = fo2_to_blender(node.center, scale)
        empty = bpy.data.objects.new(
            f"{name}_Node{node.index}", None)
        empty.empty_display_type = 'ARROWS'
        empty.empty_display_size = 1.0
        empty.location = pos

        # Build rotation matrix for the empty
        fwd = fo2_dir_to_blender(node.forward)
        right = fo2_dir_to_blender(node.right_dir)
        try:
            if fwd.length > 1e-6 and right.length > 1e-6:
                fwd = fwd.normalized()
                right = right.normalized()
                up = fwd.cross(right)
                if up.length > 1e-6:
                    up = up.normalized()
                    rot_mat = Matrix((
                        (right.x, fwd.x, up.x),
                        (right.y, fwd.y, up.y),
                        (right.z, fwd.z, up.z),
                    )).to_4x4()
                    empty.matrix_world = Matrix.Translation(pos) @ rot_mat
        except Exception:
            pass  # keep default orientation

        # store metadata as custom properties.
        # Blender custom properties use signed int32 internally, so u32 values above 0x7FFFFFFF (like 0xFFFFFFFF sentinel) must be reinterpreted as signed to avoid OverflowError.
        def _i(v):
            """Reinterpret u32 as i32 for Blender storage."""
            return v if v <= 0x7FFFFFFF else v - 0x100000000

        empty['fo2_node_index'] = _i(node.index)
        empty['fo2_unk1'] = _i(node.unk1)
        empty['fo2_prev_index'] = _i(node.prev_index)
        empty['fo2_unk2'] = _i(node.unk2)
        empty['fo2_rotation'] = list(node.rotation)
        empty['fo2_center'] = list(node.center)
        empty['fo2_left'] = list(node.left)
        empty['fo2_right'] = list(node.right)
        empty['fo2_mid'] = list(node.mid)
        empty['fo2_target'] = list(node.target)
        empty['fo2_forward'] = list(node.forward)
        empty['fo2_right_dir'] = list(node.right_dir)
        empty['fo2_interp_weights'] = list(node.interp_weights)
        empty['fo2_width_left'] = node.width_left
        empty['fo2_width_right'] = node.width_right
        empty['fo2_cumul_distance'] = node.cumul_distance
        empty['fo2_unk_neg1'] = node.unk_neg1
        empty['fo2_speed_hint'] = node.speed_hint
        empty['fo2_unk3'] = _i(node.unk3)
        empty['fo2_sentinel1'] = node.sentinel1
        empty['fo2_speed_hint2'] = node.speed_hint2
        empty['fo2_flag'] = _i(node.flag)
        empty['fo2_seq_index'] = _i(node.seq_index)
        empty['fo2_unk4'] = _i(node.unk4)
        empty['fo2_sentinel2'] = node.sentinel2
        empty['fo2_unk5'] = _i(node.unk5)
        empty['fo2_file_position'] = file_pos

        collection.objects.link(empty)


def create_startpoints(startpoints, scale, collection):
    """Create empties for start grid positions."""
    for i, sp in enumerate(startpoints):
        pos = fo2_to_blender(sp.position, scale)
        empty = bpy.data.objects.new(f"Startpoint{i+1}", None)
        empty.empty_display_type = 'SINGLE_ARROW'
        empty.empty_display_size = 3.0
        empty.location = pos
        empty['fo2_startpoint_index'] = i
        empty['fo2_startpoint_position'] = list(sp.position)
        empty['fo2_startpoint_rotation'] = list(sp.rotation)
        collection.objects.link(empty)


def create_splitpoints(splitpoints, scale, collection):
    """Create gate-line meshes for checkpoint positions."""
    mat = get_or_create_material("TrackAI_Splitpoint", (1.0, 0.3, 0.0, 1.0))
    for i, sp in enumerate(splitpoints):
        pos = fo2_to_blender(sp.position, scale)
        left = fo2_to_blender(sp.left, scale)
        right = fo2_to_blender(sp.right, scale)
        mesh = bpy.data.meshes.new(f"Splitpoint{i+1}_Gate")
        mesh.from_pydata([left[:], pos[:], right[:]], [(0, 1), (1, 2)], [])
        mesh.update()
        obj = bpy.data.objects.new(f"Splitpoint{i+1}_Gate", mesh)
        obj.data.materials.append(mat)
        obj['fo2_splitpoint_index'] = i
        obj['fo2_splitpoint_position'] = list(sp.position)
        obj['fo2_splitpoint_left'] = list(sp.left)
        obj['fo2_splitpoint_right'] = list(sp.right)
        collection.objects.link(obj)


def create_ai_splines(ai_splines, scale, collection):
    """Create curves + empties for AI border splines from splines.ai."""
    colors = [
        (0.9, 0.3, 0.3, 1.0),
        (0.3, 0.3, 0.9, 1.0),
        (0.9, 0.9, 0.3, 1.0),
    ]
    for idx, spline in enumerate(ai_splines):
        if not spline.points:
            continue
        color = colors[idx % len(colors)]
        mat = get_or_create_material(f"TrackAI_AISpline_{spline.name}", color)
        points = [fo2_to_blender(p, scale) for p in spline.points]
        curve = create_curve_from_positions(
            f"AISpline_{spline.name}", points, True, mat, 0.25)
        collection.objects.link(curve)
        for j, pt in enumerate(points):
            empty = bpy.data.objects.new(
                f"AISpline_{spline.name}_CP{j+1}", None)
            empty.empty_display_type = 'SPHERE'
            empty.empty_display_size = 0.5
            empty.location = pt
            empty['fo2_spline_name'] = spline.name
            empty['fo2_spline_index'] = j
            empty['fo2_spline_position'] = list(spline.points[j])
            collection.objects.link(empty)


def import_trackai(filepath, context, options):
    """Main import function."""
    scale = options.get('global_scale', 1.0)
    import_boundaries = options.get('import_boundaries', True)
    import_ribbon = options.get('import_ribbon', True)
    import_arrows = options.get('import_arrows', True)
    import_empties = options.get('import_empties', False)
    import_center_curve = options.get('import_center_curve', True)
    curve_width = options.get('curve_width', 0.15)
    import_startpoints = options.get('import_startpoints', True)
    import_splitpoints = options.get('import_splitpoints', True)
    import_ai_splines = options.get('import_ai_splines', True)

    # parse the binary file
    ai_data = parse_trackai(filepath)

    # parse companion files (splines.ai, splitpoints.bed, startpoints.bed)
    custom_paths = {
        'startpoints_bed_path': options.get('startpoints_bed_path', ''),
        'splitpoints_bed_path': options.get('splitpoints_bed_path', ''),
        'splines_ai_path': options.get('splines_ai_path', ''),
    }
    _parse_companion_files(filepath, ai_data, custom_paths)

    # create a parent collection
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    root_col = bpy.data.collections.new(f"TrackAI_{base_name}")
    context.scene.collection.children.link(root_col)

    # store parsed data on the root collection for round-trip export
    if ai_data.splines_ai_raw:
        root_col['fo2_splines_ai'] = ai_data.splines_ai_raw
    if ai_data.bed_splitpoints_raw:
        root_col['fo2_splitpoints_bed'] = ai_data.bed_splitpoints_raw
    if ai_data.bed_startpoints_raw:
        root_col['fo2_startpoints_bed'] = ai_data.bed_startpoints_raw
    # store AI BVH internal tree for round-trip (reused when node count unchanged)
    if ai_data.ai_bvh_tree_data:
        root_col['fo2_ai_bvh_tree'] = base64.b64encode(ai_data.ai_bvh_tree_data).decode('ascii')
        total_nodes = sum(len(s.nodes) for s in ai_data.sections)
        root_col['fo2_ai_bvh_leaf_count'] = total_nodes
    if ai_data.ai_bvh_leaf_order:
        root_col['fo2_ai_bvh_leaf_order'] = ai_data.ai_bvh_leaf_order

    for sec_idx, section in enumerate(ai_data.sections):
        if not section.nodes:
            continue

        color_idx = sec_idx % len(SECTION_COLORS)
        color = SECTION_COLORS[color_idx]
        bnd_color = BOUNDARY_COLORS[color_idx]

        sec_name = f"Path{sec_idx}"

        sec_col = bpy.data.collections.new(f"TrackAI_{sec_name}")
        root_col.children.link(sec_col)

        # store section metadata for round-trip export
        sec_col['fo2_section_index'] = sec_idx
        if section.footer:
            sec_col['fo2_footer'] = base64.b64encode(section.footer).decode('ascii')

        # determine if this is a closed loop
        # closed: last node's linked-list index wraps back to the first node
        # open: last node's index cross-references a node in another section
        is_closed = len(section.nodes) > 2
        if is_closed:
            first = section.nodes[0]
            last = section.nodes[-1]
            is_closed = (last.index == first.seq_index)

        sec_col['fo2_is_closed'] = is_closed

        nodes = section.nodes

        # gather positions
        centers = [fo2_to_blender(n.center, scale) for n in nodes]
        lefts = [fo2_to_blender(n.left, scale) for n in nodes]
        rights = [fo2_to_blender(n.right, scale) for n in nodes]
        mids = [fo2_to_blender(n.mid, scale) for n in nodes]
        targets = [fo2_to_blender(n.target, scale) for n in nodes]

        # center line curve
        if import_center_curve:
            mat_center = get_or_create_material(f"TrackAI_{sec_name}_Center", color)
            center_curve = create_curve_from_positions(
                f"{sec_name}_CenterLine", centers, is_closed, mat_center, curve_width)
            sec_col.objects.link(center_curve)

        # left/right boundary curves
        if import_boundaries:
            mat_bnd = get_or_create_material(f"TrackAI_{sec_name}_Boundary", bnd_color)

            left_curve = create_curve_from_positions(
                f"{sec_name}_LeftBoundary", lefts, is_closed, mat_bnd, curve_width * 0.5)
            sec_col.objects.link(left_curve)

            right_curve = create_curve_from_positions(
                f"{sec_name}_RightBoundary", rights, is_closed, mat_bnd, curve_width * 0.5)
            sec_col.objects.link(right_curve)

        # track ribbon (filled area between left and right)
        if import_ribbon:
            ribbon_color = (color[0], color[1], color[2], 0.25)
            mat_ribbon = get_or_create_material(f"TrackAI_{sec_name}_Ribbon", ribbon_color)
            mat_ribbon.blend_method = 'BLEND' if hasattr(mat_ribbon, 'blend_method') else 'OPAQUE'
            ribbon = create_mesh_ribbon(
                f"{sec_name}_Ribbon", lefts, rights, is_closed, mat_ribbon)
            if ribbon:
                sec_col.objects.link(ribbon)

        # direction arrows
        if import_arrows:
            mat_arrow = get_or_create_material(f"TrackAI_{sec_name}_Arrow",
                                                (1.0, 1.0, 0.0, 1.0))
            arrows = create_direction_arrows(
                f"{sec_name}_Arrows", nodes, scale, mat_arrow, arrow_scale=1.5)
            sec_col.objects.link(arrows)

        # node empties with metadata
        if import_empties:
            create_node_empties(sec_name, nodes, scale, sec_col)

        print(f"[TrackAI] Created section '{sec_name}': {len(nodes)} nodes, "
              f"closed={is_closed}")

    # add target line (AI preferred position)
    # the target positions show where the AI actually aims on the track
    for sec_idx, section in enumerate(ai_data.sections):
        if not section.nodes:
            continue
        color_idx = sec_idx % len(SECTION_COLORS)
        sec_name = f"Path{sec_idx}"

        targets = [fo2_to_blender(n.target, scale) for n in section.nodes]
        is_closed = len(section.nodes) > 2
        if is_closed:
            first = section.nodes[0]
            last = section.nodes[-1]
            is_closed = (last.index == first.seq_index)

        target_color = (1.0, 0.95, 0.0, 1.0)
        mat_target = get_or_create_material(f"TrackAI_{sec_name}_Target", target_color)
        target_curve = create_curve_from_positions(
            f"{sec_name}_TargetLine", targets, is_closed, mat_target, curve_width * 0.7)

        # find the section collection
        for col in root_col.children:
            if sec_name in col.name:
                col.objects.link(target_curve)
                break

    print(f"[TrackAI] Import complete: {len(ai_data.sections)} sections, "
          f"total {sum(len(s.nodes) for s in ai_data.sections)} nodes")

    # startpoints (embedded in trackai.bin extra data)
    if import_startpoints and ai_data.startpoints:
        sp_col = bpy.data.collections.new("TrackAI_Startpoints")
        root_col.children.link(sp_col)
        create_startpoints(ai_data.startpoints, scale, sp_col)
        print(f"[TrackAI]   Created {len(ai_data.startpoints)} startpoints")

    # splitpoints (embedded in trackai.bin extra data)
    if import_splitpoints and ai_data.splitpoints:
        sp_col = bpy.data.collections.new("TrackAI_Splitpoints")
        root_col.children.link(sp_col)
        create_splitpoints(ai_data.splitpoints, scale, sp_col)
        print(f"[TrackAI]   Created {len(ai_data.splitpoints)} splitpoints")

    # AI Border Splines (from splines.ai)
    if import_ai_splines and ai_data.ai_splines:
        sp_col = bpy.data.collections.new("TrackAI_AISplines")
        root_col.children.link(sp_col)
        create_ai_splines(ai_data.ai_splines, scale, sp_col)
        print(f"[TrackAI]   Created {len(ai_data.ai_splines)} AI border splines")

    return {'FINISHED'}


# BLENDER OPERATOR

class ImportTrackAI(bpy.types.Operator, ImportHelper):
    """Import FlatOut 2 Track AI path data"""
    bl_idname = "import_scene.fo2_trackai"
    bl_label = "Import FO2 Track AI"
    bl_options = {'PRESET', 'UNDO'}

    filename_ext = ".bin"
    filter_glob: StringProperty(default="*.bin", options={'HIDDEN'})

    global_scale: FloatProperty(
        name="Scale",
        description="Global scale factor",
        default=1.0,
        min=0.001,
        max=1000.0,
    )

    import_center_curve: BoolProperty(
        name="Center Line",
        description="Import the AI center line curve",
        default=True,
    )

    import_boundaries: BoolProperty(
        name="Left/Right Boundaries",
        description="Import left and right track boundary curves",
        default=True,
    )

    import_ribbon: BoolProperty(
        name="Track Ribbon",
        description="Import filled ribbon mesh between boundaries",
        default=True,
    )

    import_arrows: BoolProperty(
        name="Direction Arrows",
        description="Import direction indicator arrows at each node",
        default=True,
    )

    import_empties: BoolProperty(
        name="Node Empties (with metadata)",
        description="Import empties at each node with FO2 properties",
        default=True,
    )

    import_startpoints: BoolProperty(
        name="Startpoints (grid)",
        description="Start grid positions embedded in trackai.bin",
        default=True,
    )

    import_splitpoints: BoolProperty(
        name="Splitpoints (checkpoints)",
        description="Checkpoint gates embedded in trackai.bin",
        default=True,
    )

    import_ai_splines: BoolProperty(
        name="AI Splines (splines.ai)",
        description="AI border lines from companion splines.ai file",
        default=True,
    )

    curve_width: FloatProperty(
        name="Curve Width",
        description="Bevel width for curves",
        default=0.15,
        min=0.01,
        max=5.0,
    )

    startpoints_bed_path: StringProperty(
        name="startpoints.bed",
        description="Custom path to startpoints.bed (leave empty for auto-detect next to .bin)",
        default="",
        subtype='FILE_PATH',
    )

    splitpoints_bed_path: StringProperty(
        name="splitpoints.bed",
        description="Custom path to splitpoints.bed (leave empty for auto-detect next to .bin)",
        default="",
        subtype='FILE_PATH',
    )

    splines_ai_path: StringProperty(
        name="splines.ai",
        description="Custom path to splines.ai (leave empty for auto-detect next to .bin)",
        default="",
        subtype='FILE_PATH',
    )

    def draw(self, context):
        layout = self.layout

        box = layout.box()
        box.label(text="Transform", icon='ORIENTATION_GLOBAL')
        box.prop(self, "global_scale")
        box.prop(self, "curve_width")

        box = layout.box()
        box.label(text="Import Options", icon='IMPORT')
        box.prop(self, "import_center_curve")
        box.prop(self, "import_boundaries")
        box.prop(self, "import_ribbon")
        box.prop(self, "import_arrows")
        box.prop(self, "import_empties")

        box = layout.box()
        box.label(text="Track Data", icon='OUTLINER_DATA_EMPTY')
        box.prop(self, "import_startpoints")
        box.prop(self, "import_splitpoints")
        box.prop(self, "import_ai_splines")

        box = layout.box()
        box.label(text="Companion Files (optional)", icon='FILE_FOLDER')
        box.label(text="Leave empty to auto-detect next to .bin")
        box.prop(self, "startpoints_bed_path")
        box.prop(self, "splitpoints_bed_path")
        box.prop(self, "splines_ai_path")

    def execute(self, context):
        options = {
            'global_scale': self.global_scale,
            'import_boundaries': self.import_boundaries,
            'import_ribbon': self.import_ribbon,
            'import_arrows': self.import_arrows,
            'import_empties': self.import_empties,
            'import_center_curve': self.import_center_curve,
            'import_startpoints': self.import_startpoints,
            'import_splitpoints': self.import_splitpoints,
            'import_ai_splines': self.import_ai_splines,
            'curve_width': self.curve_width,
            'startpoints_bed_path': self.startpoints_bed_path,
            'splitpoints_bed_path': self.splitpoints_bed_path,
            'splines_ai_path': self.splines_ai_path,
        }
        try:
            result = import_trackai(self.filepath, context, options)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}

        self.report({'INFO'},
                    f"Imported Track AI from {os.path.basename(self.filepath)}")
        return result


# REGISTRATION

def menu_func_import(self, context):
    self.layout.operator(ImportTrackAI.bl_idname, text="FlatOut 2 TrackAI (.bin)")


def register():
    bpy.utils.register_class(ImportTrackAI)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.utils.unregister_class(ImportTrackAI)


if __name__ == "__main__":
    register()
