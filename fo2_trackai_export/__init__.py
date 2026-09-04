bl_info = {
    "name":        "FlatOut 2 TrackAI Exporter",
    "author":      "ravenDS, additional edits by Cryptid",
    "version":     (2, 3, 2),
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
import re
import base64
from bpy.props import (StringProperty, BoolProperty, FloatProperty, IntProperty, EnumProperty)
from bpy_extras.io_utils import ExportHelper
from mathutils import Vector, Matrix


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


class _NullBinaryFile:
    """No-op stand-in for a real file, used when export_trackai runs with
    dry_run=True. Accepts all write() / tell() / close() calls silently so
    the rest of the export path — including every side effect on Blender
    state (curve creation, node-empty sync, custom-property updates, UI
    refresh) — runs exactly as it would for a real export, minus the actual
    disk writes. Lets the "Preview" operator reuse the exporter's generation
    logic verbatim."""
    def write(self, data):
        return len(data) if data else 0
    def tell(self):
        return 0
    def close(self):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


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


def _blender_3x3_to_fo2_node_rot(mat):
    """Extract FO2 node rotation, forward, right_dir from Blender 3x3 matrix.

    Import builds Blender matrix with columns: [right_bl, fwd_bl, up_bl]
    where each FO2 dir (x,y,z) becomes Blender (x,z,y).

    Returns (rotation_9f, forward_3f, right_dir_3f) in FO2 space.
    """
    m = mat
    # Blender columns
    right_bl = (m[0][0], m[1][0], m[2][0])
    fwd_bl   = (m[0][1], m[1][1], m[2][1])
    up_bl    = (m[0][2], m[1][2], m[2][2])
    # Convert to FO2: bl(x,y,z) → fo2(x,z,y)
    right_fo2 = (right_bl[0], right_bl[2], right_bl[1])
    fwd_fo2   = (fwd_bl[0], fwd_bl[2], fwd_bl[1])
    up_fo2    = (up_bl[0], up_bl[2], up_bl[1])
    # Node binary rotation: columns = [right, up, fwd]
    rotation = (
        right_fo2[0], up_fo2[0], fwd_fo2[0],
        right_fo2[1], up_fo2[1], fwd_fo2[1],
        right_fo2[2], up_fo2[2], fwd_fo2[2],
    )
    return rotation, fwd_fo2, right_fo2


def _blender_3x3_to_fo2_startpoint_rot(mat):
    """Extract FO2 startpoint rotation from Blender 3x3 matrix.

    Import builds Blender matrix with columns: [right_bl, fwd_bl, up_bl]
    FO2 startpoint rotation: rows = [right_fo2, up_fo2, fwd_fo2]

    Returns tuple of 9 floats.
    """
    m = mat
    right_fo2 = (m[0][0], m[2][0], m[1][0])
    fwd_fo2   = (m[0][1], m[2][1], m[1][1])
    up_fo2    = (m[0][2], m[2][2], m[1][2])
    return (
        right_fo2[0], right_fo2[1], right_fo2[2],
        up_fo2[0],    up_fo2[1],    up_fo2[2],
        fwd_fo2[0],   fwd_fo2[1],   fwd_fo2[2],
    )


def _fo2_startpoint_rot_to_blender_3x3(rot):
    """Convert FO2 startpoint rotation (9 floats) to Blender 3x3 Matrix."""
    r = rot
    return Matrix((
        (r[0], r[6], r[3]),
        (r[2], r[8], r[5]),
        (r[1], r[7], r[4]),
    ))


def _rot_matrix_changed(obj):
    """Check if Blender rotation changed vs import time.

    Compares current matrix_world.to_3x3() against the stored
    fo2_import_rot_matrix (set at import time from the same Blender value).
    Returns True if changed or if no import matrix stored (from-scratch data).
    """
    stored = obj.get('fo2_import_rot_matrix')
    if not stored or len(stored) != 9:
        return True  # no reference → treat as new/changed
    current = obj.matrix_world.to_3x3()
    s = [float(v) for v in stored]
    for i in range(3):
        for j in range(3):
            if abs(current[i][j] - s[i * 3 + j]) > 1e-6:
                return True
    return False


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


def sample_target_curve_points(obj):
    """Read TargetLine points, including uncommitted edit-mode changes."""
    if obj is None or obj.type != 'CURVE':
        return []
    if obj.mode == 'EDIT':
        try:
            # Curve control points edited in Edit Mode can remain in Blender's
            # edit buffer until the mode changes. Flush only the TargetLine so
            # exporting directly from Edit Mode sees the current coordinates.
            obj.update_from_editmode()
        except (AttributeError, RuntimeError):
            pass
    return sample_curve_points(obj)


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


# SECTION DISCOVERY & RIBBON <-> BOUNDARY CONVERSION

# Section collections created by the importer are named TrackAI_Path{N}.
# The exporter accepts this name pattern so a user who creates sections manually
# (or whose custom properties got wiped) doesn't hit the "no sections found" trap.
_SECTION_NAME_RE = re.compile(r'^TrackAI_Path(\d+)$')


def _discover_section_collections(root_col):
    """Discover section collections under root_col.

    Detection order:
      1. Name pattern `TrackAI_Path{N}` (works even when custom props are missing).
      2. Legacy fallback: any child with `fo2_section_index >= 0`.

    Missing default properties are filled in on discovered collections so
    subsequent exports are deterministic:
      * fo2_section_index -> derived from name (or existing prop)
      * fo2_is_closed     -> True (matches every vanilla sec 0 and closed branch)
    fo2_footer is left absent; the export path already synthesises a valid
    footer when it's missing.

    Returns `[(name, col), ...]` sorted by section index.
    """
    found = []
    for child in root_col.children:
        m = _SECTION_NAME_RE.match(child.name)
        if m:
            idx = int(child.get('fo2_section_index', int(m.group(1))))
            found.append((idx, child))
        elif child.get('fo2_section_index', -1) >= 0:
            found.append((int(child['fo2_section_index']), child))

    found.sort(key=lambda x: x[0])

    for idx, child in found:
        if 'fo2_section_index' not in child:
            child['fo2_section_index'] = idx
        if 'fo2_is_closed' not in child:
            child['fo2_is_closed'] = True

    return [(child.name.replace("TrackAI_", "", 1), child) for _, child in found]


def _ordered_open_ribbon_rows(ribbon_obj):
    """Return ordered (left_vertex, right_vertex) rows for an open quad strip.

    Blender mesh edits do not preserve the importer's alternating vertex-index
    convention. Follow shared quad edges instead, then keep each side
    continuous by choosing the orientation with the shortest rail-to-rail
    movement. Non-strip or closed topology returns None so callers can retain
    the legacy index-pair fallback.
    """
    mesh = ribbon_obj.data
    polygons = list(mesh.polygons)
    if len(polygons) < 2 or any(len(poly.vertices) != 4
                                for poly in polygons):
        return None

    face_edges = []
    edge_faces = {}
    for face_index, poly in enumerate(polygons):
        vertices = list(poly.vertices)
        edges = []
        for i, vertex in enumerate(vertices):
            edge = tuple(sorted((vertex, vertices[(i + 1) % 4])))
            edges.append(edge)
            edge_faces.setdefault(edge, []).append(face_index)
        face_edges.append(edges)

    adjacency = [set() for _ in polygons]
    for users in edge_faces.values():
        if len(users) == 2:
            a, b = users
            adjacency[a].add(b)
            adjacency[b].add(a)
        elif len(users) > 2:
            return None

    endpoints = [i for i, neighbours in enumerate(adjacency)
                 if len(neighbours) == 1]
    if len(endpoints) != 2 or any(len(neighbours) not in (1, 2)
                                  for neighbours in adjacency):
        return None

    def shared_edge(face_a, face_b):
        common = set(face_edges[face_a]).intersection(face_edges[face_b])
        return next(iter(common)) if len(common) == 1 else None

    def terminal_edge(face_index, inner_edge):
        opposite = [edge for edge in face_edges[face_index]
                    if not set(edge).intersection(inner_edge)]
        return opposite[0] if len(opposite) == 1 else None

    def endpoint_key(face_index):
        neighbour = next(iter(adjacency[face_index]))
        inner = shared_edge(face_index, neighbour)
        terminal = terminal_edge(face_index, inner) if inner else None
        return min(terminal) if terminal else len(mesh.vertices)

    current = min(endpoints, key=endpoint_key)
    ordered_faces = []
    previous = None
    while current is not None:
        ordered_faces.append(current)
        following = [index for index in adjacency[current]
                     if index != previous]
        if len(following) > 1:
            return None
        previous, current = current, (following[0] if following else None)
    if len(ordered_faces) != len(polygons):
        return None

    first_shared = shared_edge(ordered_faces[0], ordered_faces[1])
    last_shared = shared_edge(ordered_faces[-2], ordered_faces[-1])
    first_row = terminal_edge(ordered_faces[0], first_shared)
    last_row = terminal_edge(ordered_faces[-1], last_shared)
    if first_row is None or last_row is None:
        return None

    unordered_rows = [first_row]
    for i in range(len(ordered_faces) - 1):
        row = shared_edge(ordered_faces[i], ordered_faces[i + 1])
        if row is None:
            return None
        unordered_rows.append(row)
    unordered_rows.append(last_row)

    world = [ribbon_obj.matrix_world @ vertex.co
             for vertex in mesh.vertices]
    first = unordered_rows[0]
    left = min(first)
    right = first[1] if first[0] == left else first[0]
    rows = [(left, right)]
    for a, b in unordered_rows[1:]:
        direct = ((world[left] - world[a]).length_squared
                  + (world[right] - world[b]).length_squared)
        crossed = ((world[left] - world[b]).length_squared
                   + (world[right] - world[a]).length_squared)
        if crossed < direct:
            a, b = b, a
        rows.append((a, b))
        left, right = a, b
    return rows


def _extract_boundaries_from_ribbon(sec_col):
    """Derive left/right boundary point lists from a section's Ribbon mesh.

    Open quad ribbons are read from their face topology, which remains valid
    after extrusion/subdivision even when Blender changes vertex numbering.
    Other topology falls back to the importer's alternating convention:
    v[0]=left[0], v[1]=right[0], v[2]=left[1], v[3]=right[1], ...

    Handles odd-length vertex lists by trimming the trailing unpaired vertex,
    which is the common failure mode when a user post-edits the ribbon
    (Merge by Distance, manual delete) and knocks the count off-parity.

    Returns (lefts, rights) in FO2 coordinates, or (None, None) if no valid
    ribbon is present. Every failure path prints a diagnostic to the system
    console — silent no-ops were causing "Preview generates nothing and
    there's no error" reports.
    """
    ribbon_obj = find_object_containing(sec_col, "_Ribbon")
    if ribbon_obj is None:
        print(f"[TrackAI Export] '{sec_col.name}': no _Ribbon mesh found in "
              f"section — nothing to fall back on")
        return None, None
    if ribbon_obj.type != 'MESH':
        print(f"[TrackAI Export] '{sec_col.name}': '{ribbon_obj.name}' is "
              f"not a MESH (type={ribbon_obj.type}); cannot use as ribbon")
        return None, None
    if ribbon_obj.mode == 'EDIT':
        ribbon_obj.update_from_editmode()
    verts = ribbon_obj.data.vertices
    n_verts = len(verts)
    if n_verts < 4:
        print(f"[TrackAI Export] '{sec_col.name}': ribbon "
              f"'{ribbon_obj.name}' has only {n_verts} vertices, need >=4")
        return None, None
    # Trim odd counts to the even prefix. Otherwise range(0, n_verts, 2)
    # would try to read verts[n_verts] on the final iteration and crash;
    # rejecting outright caused silent export failures whenever a user
    # nudged the mesh count off-parity (Merge by Distance is a common
    # culprit — collapsing one L/R pair into a single vertex takes the
    # total down to 2N-1).
    if n_verts % 2 != 0:
        print(f"[TrackAI Export] '{sec_col.name}': ribbon "
              f"'{ribbon_obj.name}' has odd vertex count {n_verts}; "
              f"trimming last vertex and proceeding with {n_verts - 1} "
              f"(pairing may be slightly off — inspect the mesh if the "
              f"generated boundaries look wrong)")
        n_verts -= 1
    rows = _ordered_open_ribbon_rows(ribbon_obj)
    if rows is not None:
        print(f"[TrackAI Export] '{sec_col.name}': reading Ribbon as an "
              f"open quad strip ({len(rows)} rows)")
    else:
        rows = [(i, i + 1) for i in range(0, n_verts, 2)]
        print(f"[TrackAI Export] '{sec_col.name}': Ribbon is not an open "
              f"quad strip; using alternating vertex indices")

    lefts = []
    rights = []
    mw = ribbon_obj.matrix_world
    for left_index, right_index in rows:
        left_world = mw @ verts[left_index].co
        right_world = mw @ verts[right_index].co
        lefts.append(blender_to_fo2(left_world))
        rights.append(blender_to_fo2(right_world))
    return lefts, rights


def _detect_ribbon_closure(lefts_fo2, rights_fo2):
    """Auto-detect whether a ribbon extracted from a mesh is closed (race
    circuit) or open (stem branch, pit spur, dead-end).

    Rationale: relying on the section's `fo2_is_closed` custom property is
    unsafe — it defaults to True at section-discovery time, which forces
    stems into cyclic NURBS boundaries and corrupts every downstream
    generator (CenterLine, TargetLine, node graph, speed hints). The
    ribbon mesh's own geometry is the source of truth: a closed loop has
    its first and last L/R pairs adjacent (the ribbon wraps around), a
    stem has them at opposite ends of the track.

    Threshold: closed if `dist(first_pair, last_pair) < 1.5× average step
    size between consecutive pairs`. Requires at least 3 pairs to make a
    meaningful judgement — shorter ribbons return False (open) since a
    2-pair ribbon has no interior for the notion of "closure" to be
    distinct from the whole thing.
    """
    import math
    if lefts_fo2 is None or rights_fo2 is None:
        return False
    n = len(lefts_fo2)
    if n < 3 or len(rights_fo2) < 3:
        return False

    def d(a, b):
        return math.sqrt((a[0] - b[0]) ** 2
                         + (a[1] - b[1]) ** 2
                         + (a[2] - b[2]) ** 2)

    step_sum = sum(d(lefts_fo2[i], lefts_fo2[i + 1]) for i in range(n - 1))
    if step_sum < 1e-6:
        return False
    avg_step = step_sum / (n - 1)
    gap_l = d(lefts_fo2[0], lefts_fo2[-1])
    gap_r = d(rights_fo2[0], rights_fo2[-1])
    threshold = 1.5 * avg_step
    return gap_l < threshold and gap_r < threshold


def _create_track_curve(sec_col, name, fo2_points, is_closed):
    """Create a Blender NURBS curve object from FO2-space points and link it to
    sec_col. Used both by the Ribbon->Boundaries fallback and the TargetLine
    auto-generator so their output looks identical to importer-created curves.
    Returns the newly-created object."""
    curve_data = bpy.data.curves.new(name, type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.resolution_u = 12
    curve_data.bevel_depth = 0.15
    curve_data.bevel_resolution = 2
    spline = curve_data.splines.new('NURBS')
    spline.points.add(len(fo2_points) - 1)
    for i, p in enumerate(fo2_points):
        # FO2 (x, y, z) -> Blender (x, z, y)
        spline.points[i].co = (p[0], p[2], p[1], 1.0)
    spline.use_cyclic_u = bool(is_closed)
    spline.order_u = 3
    obj = bpy.data.objects.new(name, curve_data)
    sec_col.objects.link(obj)
    return obj


def _smoothed_racing_line(lefts, rights, t_base, iterations, is_closed, alpha=0.5):
    """Produce a per-node racing line inside the (right, left) corridor.

    Method: initialize each point at LERP(right, left, t_base), then run a
    Chaikin-style smoothing pass N times — each pass blends every point 50%
    toward the midpoint of its two neighbours, then reprojects and clamps to
    stay on the (right, left) segment at that node.

    Empirical fit across 15 vanilla tracks:
      * Best universal defaults are t_base ≈ 0.30, iterations ≈ 10, alpha ≈ 0.5,
        giving RMS ≈ 2.2u vs. vanilla targets (vs. ~5.5u for the old "always
        inner boundary" approach).
      * Straight sections stay near a constant t_base offset from the inner
        boundary; turns naturally pull the smoothed curve toward the corridor
        limits, producing a wider-entry / apex-cut / wider-exit shape.

    Returns a list of FO2-space points (one per input node)."""
    n = min(len(lefts), len(rights))
    if n == 0:
        return []

    def lerp_point(i, t):
        r = rights[i]; l = lefts[i]
        return (r[0] + t * (l[0] - r[0]),
                r[1] + t * (l[1] - r[1]),
                r[2] + t * (l[2] - r[2]))

    pts = [lerp_point(i, t_base) for i in range(n)]

    if n < 3 or iterations <= 0:
        return pts  # no smoothing possible / requested

    for _ in range(int(iterations)):
        new_pts = list(pts)
        for i in range(n):
            if is_closed:
                p_prev = pts[(i - 1) % n]
                p_next = pts[(i + 1) % n]
            else:
                # Open path: endpoints stay put (no neighbours to average with)
                if i == 0 or i == n - 1:
                    continue
                p_prev = pts[i - 1]
                p_next = pts[i + 1]

            # Blend current toward the neighbour midpoint
            mid = (0.5 * (p_prev[0] + p_next[0]),
                   0.5 * (p_prev[1] + p_next[1]),
                   0.5 * (p_prev[2] + p_next[2]))
            blended = ((1.0 - alpha) * pts[i][0] + alpha * mid[0],
                       (1.0 - alpha) * pts[i][1] + alpha * mid[1],
                       (1.0 - alpha) * pts[i][2] + alpha * mid[2])

            # Project blended onto the (right, left) segment; clamp t ∈ [0, 1]
            r = rights[i]; l = lefts[i]
            lr = (l[0] - r[0], l[1] - r[1], l[2] - r[2])
            d_lr = lr[0]*lr[0] + lr[1]*lr[1] + lr[2]*lr[2]
            if d_lr < 1e-9:
                new_pts[i] = r
                continue
            br = (blended[0] - r[0], blended[1] - r[1], blended[2] - r[2])
            t_proj = (br[0]*lr[0] + br[1]*lr[1] + br[2]*lr[2]) / d_lr
            t_proj = max(0.0, min(1.0, t_proj))
            new_pts[i] = (r[0] + t_proj * lr[0],
                          r[1] + t_proj * lr[1],
                          r[2] + t_proj * lr[2])
        pts = new_pts

    return pts


def _compute_default_speed_hints(centers, n, is_closed,
                                  span=2, lookahead=3,
                                  radius_threshold=7071.0, cap=1000000.0):
    """Per-node default value for fo2_speed_hint / fo2_speed_hint2 derived from
    local track curvature.

    Formula: sh[i] = min(cap, C · R_min²) where R_min is the smallest
    circumradius over the lookahead window [i, i+lookahead]. Circumradius at
    each node is computed from three centers spaced `span` nodes apart (wider
    span reduces noise on smooth curves). On straight sections R → ∞ so sh
    caps at 1,000,000, matching the vanilla sentinel. At tight turns R is
    small so sh drops accordingly.

    Two knobs are user-exposed via the export operator so tuning is possible
    without editing code:

      * `lookahead` (default 3): how many nodes ahead the algorithm scans for
        the tightest upcoming turn. Larger = the AI starts slowing earlier.
      * `radius_threshold` (default 7071): the corner radius above which
        speed uncaps. Any R > radius_threshold produces sh=cap. Smaller
        values make the detection LESS sensitive (only tighter curves slow
        the AI); larger values make it MORE sensitive (mild curves also
        trigger slowdown).

    Internally, C = cap / radius_threshold² so that R = radius_threshold
    yields sh = cap exactly. The default (7071) reproduces the original
    C ≈ 0.02 constant fitted from vanilla data.

    Used only as the from-scratch default. When empties supply fo2_speed_hint
    the empty's value overrides (existing behaviour, preserves roundtrip
    byte-identity)."""
    if n < 3:
        return [cap] * max(n, 0)

    # Derive the curvature coefficient from the user-friendly radius threshold.
    # sh = C · R²; hitting the cap exactly at R = radius_threshold means
    # C = cap / radius_threshold². Guard against a zero threshold.
    if radius_threshold > 1e-6:
        C = cap / (radius_threshold * radius_threshold)
    else:
        C = 0.02  # fallback to the empirical default

    # Per-node circumradius. Cross product taken in the XZ plane (Y=up in FO2).
    radii = []
    for i in range(n):
        if is_closed:
            p1 = centers[(i - span) % n]
            p3 = centers[(i + span) % n]
        else:
            p1 = centers[max(0, i - span)]
            p3 = centers[min(n - 1, i + span)]
        p2 = centers[i]
        v1x = p2[0] - p1[0]; v1z = p2[2] - p1[2]
        v2x = p3[0] - p2[0]; v2z = p3[2] - p2[2]
        cross_y = v1x * v2z - v1z * v2x
        if abs(cross_y) < 1e-9:
            radii.append(float('inf'))
            continue
        a = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 + (p2[2]-p1[2])**2)
        b = math.sqrt((p3[0]-p2[0])**2 + (p3[1]-p2[1])**2 + (p3[2]-p2[2])**2)
        c = math.sqrt((p3[0]-p1[0])**2 + (p3[1]-p1[1])**2 + (p3[2]-p1[2])**2)
        area = 0.5 * abs(cross_y)
        if area < 1e-9:
            radii.append(float('inf'))
        else:
            radii.append((a * b * c) / (4 * area))

    result = []
    for i in range(n):
        r_min = float('inf')
        for k in range(lookahead + 1):
            j = (i + k) % n if is_closed else min(i + k, n - 1)
            if radii[j] < r_min:
                r_min = radii[j]
        if math.isinf(r_min):
            result.append(cap)
        else:
            result.append(min(cap, C * r_min * r_min))
    return result


def _u32_to_signed(v):
    """Blender IDProperty ints are signed 32-bit — reinterpret u32 sentinels
    like 0xFFFFFFFF as -1 for storage. Matches the importer's convention so
    subsequent exports read the same value back."""
    return v if v <= 0x7FFFFFFF else v - 0x100000000


def _sync_node_empties(sec_col, sec_name, node_bytes):
    """After build_section_nodes has emitted the 208-byte-per-node blob for a
    section, parse it back and create/update per-node empties in sec_col so
    the Blender scene reflects the exported state immediately — no re-import
    needed.

    Field offsets and property set exactly match what fo2_trackai_import's
    create_node_empties writes, so a subsequent re-export from the same scene
    round-trips through the empties path (byte-identical).

    Existing empties (matched by fo2_file_position) are updated in place; new
    ones are created; stale ones beyond the current node count are removed.
    Naming follows the importer convention: {sec_name}_Node{node_index}.
    Assumes global_scale = 1.0 (the exporter's implicit convention throughout;
    fixing the scale-!=-1 case is a separate concern flagged elsewhere)."""
    if not node_bytes:
        return

    n_nodes = len(node_bytes) // 208
    prefix = f"{sec_name}_Node"

    # Index existing node empties by their stored file position so we can
    # update-in-place instead of destroy-and-recreate (preserves object refs,
    # parenting, drivers, etc.).
    existing_by_pos = {}
    for obj in list(sec_col.objects):
        if obj.type != 'EMPTY' or not obj.name.startswith(prefix):
            continue
        fp = obj.get('fo2_file_position', -1)
        if fp is not None and int(fp) >= 0:
            existing_by_pos[int(fp)] = obj

    _i = _u32_to_signed

    for ni in range(n_nodes):
        off = ni * 208
        # Field layout must mirror fo2_trackai_import's parse_node.
        idx, unk1, prev_idx, unk2 = struct.unpack_from('<4I', node_bytes, off + 4)
        rotation = struct.unpack_from('<9f', node_bytes, off + 20)
        center   = struct.unpack_from('<3f', node_bytes, off + 56)
        left     = struct.unpack_from('<3f', node_bytes, off + 68)
        right    = struct.unpack_from('<3f', node_bytes, off + 80)
        mid      = struct.unpack_from('<3f', node_bytes, off + 92)
        target   = struct.unpack_from('<3f', node_bytes, off + 104)
        fwd      = struct.unpack_from('<3f', node_bytes, off + 116)
        rd       = struct.unpack_from('<3f', node_bytes, off + 128)
        iw       = struct.unpack_from('<3f', node_bytes, off + 140)
        wl       = struct.unpack_from('<f',  node_bytes, off + 152)[0]
        wr       = struct.unpack_from('<f',  node_bytes, off + 156)[0]
        cumul    = struct.unpack_from('<f',  node_bytes, off + 160)[0]
        neg1     = struct.unpack_from('<f',  node_bytes, off + 164)[0]
        sh       = struct.unpack_from('<f',  node_bytes, off + 168)[0]
        unk3, sent1 = struct.unpack_from('<2i', node_bytes, off + 172)
        sh2      = struct.unpack_from('<f',  node_bytes, off + 180)[0]
        flag, seq, unk4 = struct.unpack_from('<3I', node_bytes, off + 184)
        sent2    = struct.unpack_from('<i',  node_bytes, off + 196)[0]
        unk5     = struct.unpack_from('<I',  node_bytes, off + 200)[0]

        # Match existing empty by file position; create a new one otherwise.
        empty = existing_by_pos.pop(ni, None)
        if empty is None:
            empty_name = f"{prefix}{_i(idx)}"
            empty = bpy.data.objects.new(empty_name, None)
            empty.empty_display_type = 'ARROWS'
            empty.empty_display_size = 1.0
            sec_col.objects.link(empty)

        # Location + rotation matrix from forward/right_dir (matches importer)
        pos_bl   = Vector((center[0], center[2], center[1]))
        fwd_bl   = Vector((fwd[0], fwd[2], fwd[1]))
        right_bl = Vector((rd[0],  rd[2],  rd[1]))
        empty.location = pos_bl
        try:
            if fwd_bl.length > 1e-6 and right_bl.length > 1e-6:
                fwd_n   = fwd_bl.normalized()
                right_n = right_bl.normalized()
                up_n    = fwd_n.cross(right_n)
                if up_n.length > 1e-6:
                    up_n = up_n.normalized()
                    rot_mat = Matrix((
                        (right_n.x, fwd_n.x, up_n.x),
                        (right_n.y, fwd_n.y, up_n.y),
                        (right_n.z, fwd_n.z, up_n.z),
                    )).to_4x4()
                    empty.matrix_world = Matrix.Translation(pos_bl) @ rot_mat
        except Exception:
            pass  # keep default orientation on math failure

        # Snapshot current Blender rotation for future delta-detection on export.
        m = empty.matrix_world.to_3x3()
        empty['fo2_import_rot_matrix'] = [
            m[0][0], m[0][1], m[0][2],
            m[1][0], m[1][1], m[1][2],
            m[2][0], m[2][1], m[2][2],
        ]

        # Custom-property set — must match fo2_trackai_import's create_node_empties.
        empty['fo2_node_index']      = _i(idx)
        empty['fo2_unk1']            = _i(unk1)
        empty['fo2_prev_index']      = _i(prev_idx)
        empty['fo2_unk2']            = _i(unk2)
        empty['fo2_rotation']        = list(rotation)
        empty['fo2_center']          = list(center)
        empty['fo2_left']            = list(left)
        empty['fo2_right']           = list(right)
        empty['fo2_mid']             = list(mid)
        empty['fo2_target']          = list(target)
        empty['fo2_forward']         = list(fwd)
        empty['fo2_right_dir']       = list(rd)
        empty['fo2_interp_weights']  = list(iw)
        empty['fo2_width_left']      = wl
        empty['fo2_width_right']     = wr
        empty['fo2_cumul_distance']  = cumul
        empty['fo2_unk_neg1']        = neg1
        empty['fo2_speed_hint']      = sh
        empty['fo2_unk3']            = _i(unk3)
        empty['fo2_sentinel1']       = sent1
        empty['fo2_speed_hint2']     = sh2
        empty['fo2_flag']            = _i(flag)
        empty['fo2_seq_index']       = _i(seq)
        empty['fo2_unk4']            = _i(unk4)
        empty['fo2_sentinel2']       = sent2
        empty['fo2_unk5']            = _i(unk5)
        empty['fo2_file_position']   = ni

    # Prune stale empties whose file positions no longer exist (e.g. node
    # count decreased between exports).
    for stale in existing_by_pos.values():
        try:
            bpy.data.objects.remove(stale, do_unlink=True)
        except Exception:
            pass


def _any_section_missing_targetline():
    """Scene inspection used by the export operator's draw() to decide whether
    the 'Auto-generate TargetLine' checkbox should be enabled. Returns True if
    any TrackAI_Path{N} collection anywhere in the file lacks a _TargetLine
    curve object.

    Completely empty Path collections (no ribbon, no boundaries, no curves,
    no empties — literally nothing) are treated as inactive and skipped;
    they don't count as 'missing a target', because there's no geometry to
    generate one from."""
    for col in bpy.data.collections:
        if not col.name.startswith("TrackAI_"):
            continue
        for child in col.children:
            if not _SECTION_NAME_RE.match(child.name):
                continue
            if len(child.objects) == 0:
                continue  # empty placeholder collection — nothing to generate
            has_target = any(
                obj.type == 'CURVE' and "_TargetLine" in obj.name
                for obj in child.objects
            )
            if not has_target:
                return True
    return False


def _any_section_missing_centerline():
    """Scene inspection used by the export operator's draw() to decide whether
    the 'Auto-generate CenterLine' checkbox should be enabled. Returns True if
    any TrackAI_Path{N} collection anywhere in the file lacks a _CenterLine
    curve object.

    Same empty-collection skip as _any_section_missing_targetline."""
    for col in bpy.data.collections:
        if not col.name.startswith("TrackAI_"):
            continue
        for child in col.children:
            if not _SECTION_NAME_RE.match(child.name):
                continue
            if len(child.objects) == 0:
                continue  # empty placeholder collection — nothing to generate
            has_center = any(
                obj.type == 'CURVE' and "_CenterLine" in obj.name
                for obj in child.objects
            )
            if not has_center:
                return True
    return False


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


def _nearest_route_segment(point, route, is_closed):
    """Return the nearest route segment index and interpolation factor."""
    n = len(route)
    segment_count = n if is_closed else n - 1
    best = None
    for i in range(max(0, segment_count)):
        a = route[i]
        b = route[(i + 1) % n]
        ab = vec_sub(b, a)
        length_sq = ab[0]*ab[0] + ab[1]*ab[1] + ab[2]*ab[2]
        if length_sq < 1e-12:
            continue
        ap = vec_sub(point, a)
        t = (ap[0]*ab[0] + ap[1]*ab[1] + ap[2]*ab[2]) / length_sq
        t = max(0.0, min(1.0, t))
        projected = vec_add(a, vec_scale(ab, t))
        distance = vec_dist(point, projected)
        candidate = (distance, i, t)
        if best is None or candidate[0] < best[0] - 1e-7:
            best = candidate
    return None if best is None else (best[1], best[2], best[0])


def _infer_path0_branch_refs(main_centers, branch_centers,
                             main_is_closed=True):
    """Infer an open branch's incoming/outgoing Path0 sequence links.

    FlatOut treats these as departure/continuation links: the branch starts
    after its nearest Path0 segment and rejoins at the node following its
    nearest final Path0 segment.
    """
    if len(main_centers) < 2 or len(branch_centers) < 2:
        return None, None
    start_hit = _nearest_route_segment(
        branch_centers[0], main_centers, main_is_closed)
    end_hit = _nearest_route_segment(
        branch_centers[-1], main_centers, main_is_closed)
    if start_hit is None or end_hit is None:
        return None, None

    prev_seq = int(start_hit[0])
    end_segment = int(end_hit[0])
    if main_is_closed:
        next_seq = (end_segment + 1) % len(main_centers)
    else:
        next_seq = min(end_segment + 1, len(main_centers) - 1)
    return (0, prev_seq), (0, next_seq)


def _effective_node_centers(curve_centers, empties):
    """Apply the same empty position overrides used by node serialization."""
    if len(empties) != len(curve_centers):
        return list(curve_centers)
    result = []
    for i, empty in enumerate(empties):
        center = _read_vec3_prop(empty, 'fo2_center', curve_centers[i])
        import_bl = (center[0], center[2], center[1])
        delta_fo2 = (empty.location[0] - import_bl[0],
                     empty.location[2] - import_bl[2],
                     empty.location[1] - import_bl[1])
        result.append((center[0] + delta_fo2[0],
                       center[1] + delta_fo2[1],
                       center[2] + delta_fo2[2]))
    return result


def _read_vec3_prop(e, key, fallback):
    """Read a vec3 custom property from an empty, return as tuple"""
    v = e.get(key)
    if v and len(v) == 3:
        return (float(v[0]), float(v[1]), float(v[2]))
    return fallback


def build_section_nodes(centers, lefts, rights, targets, n, is_closed,
                        empties, section_index,
                        branch_prev_ref=None, branch_next_ref=None,
                        speed_lookahead=3, speed_radius_threshold=7071.0,
                        generate_speed_hints=True,
                        prefer_curve_target=False):
    """Build binary node data for one section
    
    If empties are present, their fields are read as the roundtrip source.
    When prefer_curve_target is true, TargetLine coordinates stay authoritative
    so edits to that curve are exported even when imported node empties exist.

    From-scratch defaults follow the vanilla patterns verified across 10 tracks
    / 33 sections / 1728 nodes:
      * Closed loops: idx[i]=(i+1)%N, prev[i]=(i-1)%N (in-section wrap)
      * Open paths:   interior idx=i+1, prev=i-1; endpoints are cross-section
      * Section 0:    unk1=unk2=unk4=0, sentinel2=-1 uniformly
      * Section >0:   unk4=sec_i uniformly,
                      unk1=sec_i except last=0,
                      unk2=sec_i except first=0,
                      first.sent2 = first.prev + 1
      * interp_weights: iw[0]+iw[2]==1 always (per-node lateral weights)

    branch_prev_ref / branch_next_ref: optional (parent_sec_idx, parent_seq_idx)
    tuples that supply cross-section connections for sec >0 endpoints.

    speed_lookahead / speed_radius_threshold: forwarded to
    _compute_default_speed_hints when computing from-scratch speed_hint
    defaults; ignored when empties are present. Exposed through the export
    operator so users can tune AI cornering behaviour without editing code.
    """
    if n == 0:
        return b''

    has_empties = len(empties) == n
    buf = bytearray()
    _u = lambda v: int(v) & 0xFFFFFFFF

    # Cross-section endpoints derived from user-supplied branch refs.
    # prev[0] and idx[N-1] point into another section's seq space;
    # sent2[0] = prev[0] + 1 (empirical rule, 23/23 vanilla branches).
    if section_index > 0 and branch_prev_ref is not None:
        branch_prev_seq = int(branch_prev_ref[1])
    else:
        branch_prev_seq = None
    if section_index > 0 and branch_next_ref is not None:
        branch_next_seq = int(branch_next_ref[1])
    else:
        branch_next_seq = None

    # Precompute the per-node default speed hint from geometry — only when
    # there are no empties to source values from AND the user has enabled
    # geometry-based speed_hint generation. Existing empties are never
    # touched (roundtrip preserved byte-for-byte). When speed-hint generation
    # is disabled, default_speed_hints stays None so the per-node fallback
    # below emits the 1,000,000 sentinel (equivalent to "no limit").
    if has_empties or not generate_speed_hints:
        default_speed_hints = None
    else:
        default_speed_hints = _compute_default_speed_hints(
            centers, n, is_closed,
            lookahead=int(speed_lookahead),
            radius_threshold=float(speed_radius_threshold))

    for i in range(n):
        is_first = (i == 0)
        is_last = (i == n - 1)

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

        # iw[0]+iw[2]==1 is an exact vanilla invariant; (0.13, 0.6, 0.87)
        # is the empirical average and matches within observed ranges.
        interp_weights = (0.13, 0.6, 0.87)
        unk_neg1 = -1.0
        # Geometry-derived default; empties override further down. Falls back
        # to 1M (sentinel = "no limit") when empties supply values anyway.
        _sh_default = default_speed_hints[i] if default_speed_hints else 1000000.0
        speed_hint = _sh_default
        unk3 = 0
        sentinel1 = -1
        speed_hint2 = _sh_default
        flag = 1
        seq_index = i
        unk5 = 0

        # Section-index-based unk pattern:
        if section_index == 0:
            unk1 = 0
            unk2 = 0
            unk4 = 0
        else:
            unk4 = section_index
            unk1 = 0 if is_last else section_index
            unk2 = 0 if is_first else section_index

        # Linked-list forward pointer:
        #   closed: (i+1) % N (last wraps back to seq 0)
        #   open:   i+1 for interior; last = branch_next_seq or self-safe (N-1)
        if is_closed:
            node_index = (i + 1) % n
        else:
            if is_last:
                node_index = branch_next_seq if branch_next_seq is not None else (n - 1)
            else:
                node_index = i + 1

        # Linked-list back pointer:
        #   sec 0 closed: (i-1) % N (first wraps to N-1)
        #   sec >0:       first = branch_prev_seq (or safe 0), then (i-1)
        if is_first:
            if section_index == 0 and is_closed:
                prev_index = n - 1
            elif branch_prev_seq is not None:
                prev_index = branch_prev_seq
            else:
                prev_index = 0  # self-safe fallback (in-range, won't crash)
        else:
            prev_index = i - 1

        # sentinel2:
        #   sec 0: -1 uniformly
        #   sec >0 first node: prev+1 (cross-section connect in parent)
        #   sec >0 other nodes: -1
        if section_index > 0 and is_first:
            sentinel2 = prev_index + 1
        else:
            sentinel2 = -1

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
            if not prefer_curve_target:
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
                if not prefer_curve_target:
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

        # Geometry-derived branch endpoint links remain authoritative even
        # when imported/generated empties contain stale endpoint self-links.
        if section_index > 0 and not is_closed:
            if is_first and branch_prev_seq is not None:
                prev_index = branch_prev_seq
                sentinel2 = prev_index + 1
            if is_last and branch_next_seq is not None:
                node_index = branch_next_seq

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

def _sync_transforms_to_props(root_col):
    """Update stored custom properties from current Blender transforms.

    Only updates when the user actually changed the rotation (compared
    against fo2_import_rot_matrix stored at import time). The existing
    write routines read from custom props and stay untouched.

    - Node empties: updates fo2_rotation, fo2_forward, fo2_right_dir
    - Startpoint empties: updates fo2_startpoint_rotation, and applies
      a rotation delta to fo2_bed_startpoint_rotation if present
    """
    # --- node empties ---
    for child in root_col.children:
        sec_idx = child.get('fo2_section_index', -1)
        if sec_idx < 0:
            continue
        for obj in child.objects:
            if obj.get('fo2_forward') is None or obj.get('fo2_right_dir') is None:
                continue  # not a node empty

            # Position: update fo2_center from Blender location,
            # apply same offset to left/right/mid/target
            old_center = _read_vec3_prop(obj, 'fo2_center', None)
            if old_center:
                new_center = blender_to_fo2(obj.location)
                dx = new_center[0] - old_center[0]
                dy = new_center[1] - old_center[1]
                dz = new_center[2] - old_center[2]
                obj['fo2_center'] = list(new_center)
                if abs(dx) > 1e-6 or abs(dy) > 1e-6 or abs(dz) > 1e-6:
                    for key in ('fo2_left', 'fo2_right', 'fo2_mid', 'fo2_target'):
                        old = _read_vec3_prop(obj, key, None)
                        if old:
                            obj[key] = [old[0] + dx, old[1] + dy, old[2] + dz]

            # Rotation: only update if user actually changed it
            if not _rot_matrix_changed(obj):
                continue
            mat = obj.matrix_world.to_3x3()
            rotation, forward, right_dir = _blender_3x3_to_fo2_node_rot(mat)
            obj['fo2_rotation'] = list(rotation)
            obj['fo2_forward'] = list(forward)
            obj['fo2_right_dir'] = list(right_dir)

    # --- startpoint empties ---
    sp_col = None
    for child in root_col.children:
        if child.name == "TrackAI_Startpoints":
            sp_col = child
            break
    if not sp_col:
        return

    for obj in sp_col.objects:
        if obj.get('fo2_startpoint_index', -1) < 0:
            continue

        # Position: update from Blender location
        obj['fo2_startpoint_position'] = list(blender_to_fo2(obj.location))
        if not _rot_matrix_changed(obj):
            continue  # unchanged

        current_bl = obj.matrix_world.to_3x3()
        new_rot = _blender_3x3_to_fo2_startpoint_rot(current_bl)

        # .bed rotation delta
        old_rot_raw = obj.get('fo2_startpoint_rotation')
        bed_rot_raw = obj.get('fo2_bed_startpoint_rotation')
        if old_rot_raw and bed_rot_raw and len(old_rot_raw) == 9 and len(bed_rot_raw) == 9:
            old_rot = tuple(float(v) for v in old_rot_raw)
            import_bl = _fo2_startpoint_rot_to_blender_3x3(old_rot)
            delta_bl = current_bl @ import_bl.transposed()
            bed_bl = _fo2_startpoint_rot_to_blender_3x3(tuple(float(v) for v in bed_rot_raw))
            new_bed_bl = delta_bl @ bed_bl
            obj['fo2_bed_startpoint_rotation'] = list(_blender_3x3_to_fo2_startpoint_rot(new_bed_bl))

        # Update binary rotation prop (AFTER reading old value for delta above)
        obj['fo2_startpoint_rotation'] = list(new_rot)

    print("[TrackAI Export] Synced Blender transforms → custom properties")


def export_trackai(filepath, context, options):
    root_col = find_trackai_root(context)
    if root_col is None:
        raise ValueError("No TrackAI collection found in scene")

    print(f"[TrackAI Export] Found root: {root_col.name}")

    # Sync current Blender transforms into custom properties
    _sync_transforms_to_props(root_col)

    # gather section collections (name pattern OR legacy custom property);
    # also fills in missing default properties so subsequent exports are stable.
    section_cols = _discover_section_collections(root_col)

    if not section_cols:
        raise ValueError("No spline section collections found "
                         "(expected TrackAI_Path0, TrackAI_Path1, ...)")

    # Dry-run mode: skip disk writes but keep every Blender-side side effect
    # (curve gen, node empty sync, custom-prop updates). Used by the Preview
    # operator in fo2_bgm_hierarchy so users can inspect+tweak generated
    # geometry before committing to a real export.
    dry_run = bool(options.get('dry_run', False))
    if dry_run:
        print("[TrackAI Export] DRY RUN — generating in-scene artifacts only, "
              "no files will be written")

    # write trackai.bin 
    _bin_file_cm = _NullBinaryFile() if dry_run else open(filepath, 'wb')
    with _bin_file_cm as f:
        write_u32(f, TAG_FILE_HEADER)
        write_u32(f, len(section_cols))

        auto_gen_target = bool(options.get('auto_generate_target', True))
        target_method = str(options.get('target_method', 'SMOOTH'))
        target_source = str(options.get('target_source', 'RIGHT'))
        target_lerp = float(options.get('target_lerp', 0.30))
        target_smooth_iters = int(options.get('target_smooth_iters', 10))
        auto_gen_center = bool(options.get('auto_generate_center', True))
        center_offset = float(options.get('center_offset', 3.40))
        speed_lookahead = int(options.get('speed_lookahead', 3))
        speed_radius_threshold = float(options.get('speed_radius_threshold', 7071.0))
        generate_speed_hints = bool(options.get('generate_speed_hints', True))
        flip_boundaries = bool(options.get('flip_boundaries', False))
        main_route_centers = None
        main_route_is_closed = True

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
            targets = sample_target_curve_points(target_obj)
            prefer_curve_target = bool(targets)

            # Ribbon fallback: when Left/Right curves are missing (e.g. the user
            # only has a Ribbon mesh for the section), derive boundaries from
            # the ribbon's alternating vertices AND materialize them as Blender
            # curve objects so the user sees them immediately in the Outliner.
            if not lefts or not rights:
                r_lefts, r_rights = _extract_boundaries_from_ribbon(sec_col)
                if r_lefts is not None:
                    # Flip Boundaries: on some ribbon geometries the L/R
                    # sides come out inverted (auto-detection can't always
                    # know which side of the strip is "outside" of the
                    # eventual racing line). Swap here — before anything
                    # else uses r_lefts/r_rights — so CenterLine,
                    # TargetLine, nodes, and speed hints all cascade off
                    # the corrected assignment.
                    if flip_boundaries:
                        print(f"[TrackAI Export] Section '{sec_name}': "
                              f"flip_boundaries=True, swapping L/R "
                              f"before generation")
                        r_lefts, r_rights = r_rights, r_lefts

                    # Auto-detect closure from the ribbon's actual geometry.
                    # The section's `fo2_is_closed` defaults to True (line
                    # 275-276) — safe for a race circuit's main loop, but
                    # catastrophically wrong for stem branches: forcing a stem
                    # into cyclic NURBS wraps the last boundary point back to
                    # the first, corrupting CenterLine, TargetLine, node graph
                    # and speed hints. Overriding here (and updating the prop
                    # so downstream reads see the correct value) is the fix.
                    detected_closed = _detect_ribbon_closure(r_lefts, r_rights)
                    if detected_closed != is_closed:
                        print(f"[TrackAI Export] Section '{sec_name}': "
                              f"ribbon geometry indicates "
                              f"{'closed' if detected_closed else 'open'}, "
                              f"overriding fo2_is_closed={is_closed}")
                        is_closed = detected_closed
                        sec_col['fo2_is_closed'] = detected_closed

                    derived = []
                    if not lefts:
                        lefts = r_lefts
                        left_obj = _create_track_curve(
                            sec_col, f"{sec_name}_LeftBoundary",
                            r_lefts, is_closed)
                        derived.append('left')
                    if not rights:
                        rights = r_rights
                        right_obj = _create_track_curve(
                            sec_col, f"{sec_name}_RightBoundary",
                            r_rights, is_closed)
                        derived.append('right')
                    if derived:
                        print(f"[TrackAI Export] Section '{sec_name}': "
                              f"created {'/'.join(derived)} boundary curve(s) "
                              f"from Ribbon mesh")

            # CenterLine auto-generation: after boundaries are settled (either
            # user-drawn or ribbon-derived), offset RightBoundary perpendicular
            # to LeftBoundary by a small distance (vanilla mean ~3.40 units).
            # This places CenterLine just outside RightBoundary on the interior
            # side of the track — matching the empirical vanilla layout where
            # the ribbon (Left↔Right) covers the outer portion of the drivable
            # surface and CenterLine sits ~3.4u further inside.
            # Requires BOTH boundaries; if only one is available we can't
            # disambiguate the interior side and skip with a warning.
            if center_obj is None and auto_gen_center:
                if lefts and rights:
                    n_c = min(len(lefts), len(rights))
                    gen_centers = []
                    for i in range(n_c):
                        l = lefts[i]; r = rights[i]
                        dx = r[0] - l[0]; dy = r[1] - l[1]; dz = r[2] - l[2]
                        d_len = math.sqrt(dx*dx + dy*dy + dz*dz)
                        if d_len < 1e-6:
                            # Degenerate: left ≈ right at this node — use r as-is
                            gen_centers.append(r)
                        else:
                            gen_centers.append((r[0] + center_offset * dx / d_len,
                                                r[1] + center_offset * dy / d_len,
                                                r[2] + center_offset * dz / d_len))
                    center_obj = _create_track_curve(
                        sec_col, f"{sec_name}_CenterLine",
                        gen_centers, is_closed)
                    centers = gen_centers
                    print(f"[TrackAI Export] Section '{sec_name}': "
                          f"generated CenterLine (offset {center_offset:.2f}u "
                          f"from RightBoundary toward track interior)")
                else:
                    print(f"[TrackAI Export] Section '{sec_name}': "
                          f"cannot auto-generate CenterLine — needs both "
                          f"LeftBoundary and RightBoundary")

            # TargetLine auto-generation: after boundaries are settled (either
            # user-drawn or ribbon-derived), synthesise a target line the AI
            # will follow. Two methods:
            #
            #   SMOOTH   — corridor-clamped racing line: LERP at t_base within
            #              the (right, left) ribbon, then Chaikin-smooth for
            #              N iterations. Straights sit near t_base; turns pull
            #              the curve toward the corridor edge (wider entry / cut
            #              apex / wider exit). Empirically best across 14/15
            #              vanilla tracks vs. duplication (avg RMS 2.2u vs 5.5u).
            #   DUPLICATE — copy one boundary verbatim (nascar-style behaviour).
            #
            # The generated curve becomes immediately visible in Blender via
            # the UI refresh at the end of export.
            if target_obj is None and auto_gen_target:
                target_points = None
                gen_desc = None

                if target_method == 'SMOOTH':
                    if lefts and rights:
                        target_points = _smoothed_racing_line(
                            lefts, rights,
                            t_base=target_lerp,
                            iterations=target_smooth_iters,
                            is_closed=bool(is_closed))
                        gen_desc = (f"smoothed racing line "
                                    f"(t={target_lerp:.2f}, "
                                    f"{target_smooth_iters} iters)")
                else:  # DUPLICATE
                    source_points = rights if target_source == 'RIGHT' else lefts
                    source_name = ("RightBoundary" if target_source == 'RIGHT'
                                   else "LeftBoundary")
                    if source_points:
                        target_points = list(source_points)
                        gen_desc = f"duplicated {source_name}"

                if target_points:
                    target_obj = _create_track_curve(
                        sec_col, f"{sec_name}_TargetLine",
                        target_points, is_closed)
                    targets = list(target_points)
                    prefer_curve_target = True
                    print(f"[TrackAI Export] Section '{sec_name}': "
                          f"generated TargetLine ({gen_desc})")

            # Warn about missing curves so the user knows if boundaries/target
            # will collapse to degenerate positions (all-lefts-equal-center[0]
            # etc.). Not fatal — writing valid but degenerate positions is
            # preferable to failing the export.
            missing = []
            if not centers and center_obj is None: missing.append("_CenterLine")
            if not lefts and left_obj is None:     missing.append("_LeftBoundary")
            if not rights and right_obj is None:   missing.append("_RightBoundary")
            if not targets and target_obj is None: missing.append("_TargetLine")
            if missing:
                empties_present = bool(gather_empties(sec_col, sec_name))
                if not empties_present:
                    print(f"[TrackAI Export] WARNING: Section '{sec_name}' is missing "
                          f"{', '.join(missing)} curve(s). Missing sides will collapse "
                          f"to center[0] on export, producing a zero-width track edge. "
                          f"The AI may misbehave near those nodes.")

            if not centers:
                centers = lefts or rights
                if not centers:
                    # Empty section (0 nodes) — still write header + footer
                    if footer_b64:
                        footer_bytes = base64.b64decode(footer_b64)
                    else:
                        footer_bytes = struct.pack('<IfIII', 1 if sec_i > 0 else 0, 0.5, 0, 0, 2)
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
            effective_centers = _effective_node_centers(centers, empties)

            if sec_i == 0:
                main_route_centers = effective_centers
                main_route_is_closed = bool(is_closed)

            # Infer open alternate-route endpoints from Path0 geometry even
            # when node empties exist; generated empties commonly retain
            # endpoint self-links which isolate the branch from the game graph.
            branch_prev_ref = None
            branch_next_ref = None
            if (sec_i > 0 and not is_closed and main_route_centers):
                branch_prev_ref, branch_next_ref = _infer_path0_branch_refs(
                    main_route_centers, effective_centers,
                    main_route_is_closed)
                if branch_prev_ref and branch_next_ref:
                    sec_col['fo2_branch_prev_ref'] = list(branch_prev_ref)
                    sec_col['fo2_branch_next_ref'] = list(branch_next_ref)
                    print(f"[TrackAI Export] Section '{sec_name}': "
                          f"connected to Path0 "
                          f"prev={branch_prev_ref[1]}, "
                          f"next={branch_next_ref[1]}")

            # Stored refs remain the fallback when geometry inference is not
            # possible, for example with an incomplete main route.
            if branch_prev_ref is None:
                _bp = sec_col.get('fo2_branch_prev_ref')
                if _bp and len(_bp) == 2:
                    branch_prev_ref = (int(_bp[0]), int(_bp[1]))
            if branch_next_ref is None:
                _bn = sec_col.get('fo2_branch_next_ref')
                if _bn and len(_bn) == 2:
                    branch_next_ref = (int(_bn[0]), int(_bn[1]))

            # footer
            if footer_b64:
                footer_bytes = base64.b64decode(footer_b64)
            else:
                footer_bytes = struct.pack('<IfIII', 1 if sec_i > 0 else 0, 0.5, 0, 0, 2)

            # build and write section
            write_u32(f, TAG_SPLINE_SECTION)
            write_u32(f, n)

            node_data = build_section_nodes(
                centers, lefts, rights, targets, n, is_closed,
                empties, sec_i,
                branch_prev_ref=branch_prev_ref,
                branch_next_ref=branch_next_ref,
                speed_lookahead=speed_lookahead,
                speed_radius_threshold=speed_radius_threshold,
                generate_speed_hints=generate_speed_hints,
                prefer_curve_target=prefer_curve_target)
            f.write(node_data)

            # Mirror the just-written node records back into Blender as per-node
            # empties (creating them for from-scratch exports, updating existing
            # ones otherwise). Users no longer need to re-import to see the
            # nodes appear in the Outliner with all fo2_* properties populated.
            _sync_node_empties(sec_col, sec_name, node_data)

            # Footer
            f.write(footer_bytes)

            # separator after every section (including last, required before extra data)
            write_u32(f, TAG_SECTION_SEP)

            print(f"[TrackAI Export] Section '{sec_name}': {n} nodes, "
                  f"closed={is_closed}, empties={'yes' if len(empties)==n else 'no'}")

        # extra data block (startpoints, splitpoints, BV)
        _write_extra_data(f, root_col, section_cols)

    if dry_run:
        print("[TrackAI Export] DRY RUN complete: curves/nodes/props updated "
              "in Blender, no files written")
        return {'FINISHED'}

    base_dir = os.path.dirname(filepath)

    def _persist_companion(name, prop_key):
        """Read the just-written companion file from disk and store its text
        back on root_col, so the Properties panel shows the current state."""
        try:
            with open(os.path.join(base_dir, name), 'r') as fh:
                root_col[prop_key] = fh.read()
        except Exception:
            pass  # non-fatal; the file itself is fine

    # splines.ai
    if options.get('export_splines_ai', True):
        if _export_splines_from_empties(root_col, base_dir):
            _persist_companion("splines.ai", 'fo2_splines_ai')
        else:
            splines_raw = root_col.get('fo2_splines_ai', '')
            if splines_raw:
                out_path = os.path.join(base_dir, "splines.ai")
                with open(out_path, 'w', newline='\n') as f:
                    f.write(splines_raw)
                print(f"[TrackAI Export] Wrote splines.ai (verbatim, {len(splines_raw)} chars)")

    # splitpoints.bed
    if options.get('export_splitpoints_bed', True):
        if _export_splitpoints_from_objects(root_col, base_dir):
            _persist_companion("splitpoints.bed", 'fo2_splitpoints_bed')
        else:
            splitpoints_raw = root_col.get('fo2_splitpoints_bed', '')
            if splitpoints_raw:
                out_path = os.path.join(base_dir, "splitpoints.bed")
                with open(out_path, 'w', newline='\n') as f:
                    f.write(splitpoints_raw)
                print(f"[TrackAI Export] Wrote splitpoints.bed (verbatim)")

    # startpoints.bed
    if options.get('export_startpoints_bed', True):
        if _export_startpoints_from_objects(root_col, base_dir):
            _persist_companion("startpoints.bed", 'fo2_startpoints_bed')
        else:
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


def _splitpoint_world_positions(obj):
    """Read (pos, left, right) FO2-space positions of a splitpoint gate by
    transforming its three mesh vertices through the object's world matrix.

    The importer builds each gate mesh with:
        vertex 0 = left, vertex 1 = pos, vertex 2 = right
    all placed in Blender space at import time (create_splitpoints in the
    importer, line ~975 of fo2_trackai_import). Reading world positions
    honours every transform the user may apply — translate, rotate, scale,
    edit-mode vertex moves, parenting — where the old obj.location-only
    "delta" read silently dropped rotations, scale, and per-vertex edits.

    Returns (pos_fo2, left_fo2, right_fo2) or None if the object is not a
    mesh or has fewer than 3 vertices.
    """
    if obj is None or obj.type != 'MESH':
        return None
    verts = obj.data.vertices
    if len(verts) < 3:
        return None
    mw = obj.matrix_world
    left_w  = mw @ verts[0].co
    pos_w   = mw @ verts[1].co
    right_w = mw @ verts[2].co
    return (
        blender_to_fo2(pos_w),
        blender_to_fo2(left_w),
        blender_to_fo2(right_w),
    )


def _gather_splitpoint_objects(root_col):
    """Collect (idx, pos, left, right) tuples in FO2 space for every
    splitpoint gate under TrackAI_Splitpoints. Sorted by index.

    Reads world-space vertex positions rather than adding an obj.location
    "delta" to a stored original — this is what makes rotation and
    per-vertex edits actually export instead of being silently dropped.

    Side effect: refreshes the fo2_splitpoint_* custom properties (and
    the fo2_bed_splitpoint_* mirrors if present) so they reflect the
    current effective state after any user edits. The .bed mirrors get
    the same values because the importer places both from a single
    game-space triplet — if a vanilla file ever splits them, the mesh
    was built from the embedded triplet, so that's the authoritative
    source anyway.
    """
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
        if idx < 0:
            continue
        world = _splitpoint_world_positions(obj)
        if world is None:
            # Object was somehow stripped of its mesh — fall back to the
            # stored embedded coords so we don't drop the splitpoint
            # entirely, but nothing user-edited will be captured.
            pos = obj.get('fo2_splitpoint_position')
            left = obj.get('fo2_splitpoint_left')
            right = obj.get('fo2_splitpoint_right')
            if pos and left and right:
                items.append((idx,
                              tuple(float(v) for v in pos),
                              tuple(float(v) for v in left),
                              tuple(float(v) for v in right)))
            continue
        pos, left, right = world
        items.append((idx, pos, left, right))
        # Refresh props to reflect the current effective state.
        obj['fo2_splitpoint_position'] = list(pos)
        obj['fo2_splitpoint_left']     = list(left)
        obj['fo2_splitpoint_right']    = list(right)
        if 'fo2_bed_splitpoint_position' in obj:
            obj['fo2_bed_splitpoint_position'] = list(pos)
            obj['fo2_bed_splitpoint_left']     = list(left)
            obj['fo2_bed_splitpoint_right']    = list(right)
    items.sort(key=lambda x: x[0])
    return items


def _gather_section_node_data(section_cols):
    """Gather FO2-space node positions from all sections for AI BVH generation.

    Returns list of (is_closed, nodes) tuples, where nodes is a list of dicts:
        {center, left, right, index, seq_index, sec_idx}

    Prefers reading from node empties (round-trip fidelity). Falls back to
    sampling the section's centre/left/right curves when no empties are present,
    which is what makes from-scratch tracks produce a valid AI BVH block —
    vanilla files always carry exactly one BVH leaf per node, and skipping the
    block (as the old empties-only path did) crashes the game.
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

        # Curves fallback: no empties in this section -> synthesize node dicts
        # from the section's centre/left/right curves so the BVH generator has
        # what it needs. Uses the same idx pattern build_section_nodes writes:
        #   closed -> (i+1) % N   ;   open -> i+1 (last self-clamped to N-1)
        if not nodes:
            center_obj = find_object_containing(sec_col, "_CenterLine")
            left_obj = find_object_containing(sec_col, "_LeftBoundary")
            right_obj = find_object_containing(sec_col, "_RightBoundary")
            centers_c = sample_curve_points(center_obj)
            lefts_c = sample_curve_points(left_obj)
            rights_c = sample_curve_points(right_obj)
            if not centers_c:
                centers_c = lefts_c or rights_c
            nc = len(centers_c)
            if nc > 0:
                while len(lefts_c) < nc:
                    lefts_c.append(lefts_c[-1] if lefts_c else centers_c[0])
                while len(rights_c) < nc:
                    rights_c.append(rights_c[-1] if rights_c else centers_c[0])
                for i in range(nc):
                    if is_closed:
                        node_idx = (i + 1) % nc
                    else:
                        node_idx = (i + 1) if i < nc - 1 else (nc - 1)
                    nodes.append({
                        'center': centers_c[i], 'left': lefts_c[i], 'right': rights_c[i],
                        'index': node_idx, 'seq_index': i, 'sec_idx': sec_i,
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
    """Build a balanced binary AI-BVH tree, matching the vanilla encoding.

    Vanilla format (verified across 10 tracks, 33 sections):
      * Root is stored at pair[0] (NOT implicit — the array itself starts with
        the root descriptor).
      * type=0: internal node. `ref` = index of the first of two consecutive
                child descriptors at pairs[ref] and pairs[ref+1].
      * type=1: leaf singleton. `ref` = leaf index (covers exactly 1 leaf).
      * type=2: leaf pair. `ref` = first leaf index; the second leaf is
                implicitly at ref+1 (covers 2 consecutive leaves). MUST have
                ref+1 < num_leaves.

    Total pair count = 2*m - 1 where m = number of leaf descriptors
    (verified formula match across all 10 vanilla tracks).

    The old (broken) generator used type=2 for every SINGLE leaf, which the
    game reads as a 2-leaf group — over-reading past the last leaf on the
    from-scratch path and crashing.
    """
    if num_leaves == 0:
        return []
    if num_leaves == 1:
        return [(0, 1)]  # degenerate: root is a lone singleton

    # Group leaves into descriptors: consecutive pairs (type=2) plus a
    # trailing singleton (type=1) if num_leaves is odd. Sequential grouping
    # matches spatial adjacency because the leaves are ordered along the
    # track (adjacent leaves = adjacent track segments).
    leaf_descs = []
    i = 0
    while i + 1 < num_leaves:
        leaf_descs.append((i, 2))
        i += 2
    if i < num_leaves:
        leaf_descs.append((i, 1))

    m = len(leaf_descs)
    if m == 1:
        return [leaf_descs[0]]

    # Build a balanced binary tree over the leaf descriptors. Root sits at
    # pairs[0]; internal-node descriptors reserve two consecutive child slots
    # before recursing into their halves.
    pairs = [None]  # placeholder for root; filled at the end

    def build(descs):
        if len(descs) == 1:
            return descs[0]
        mid = len(descs) // 2
        left_descs = descs[:mid]
        right_descs = descs[mid:]
        my_children_start = len(pairs)
        pairs.append(None)
        pairs.append(None)
        left_desc = build(left_descs)
        right_desc = build(right_descs)
        pairs[my_children_start] = left_desc
        pairs[my_children_start + 1] = right_desc
        return (my_children_start, 0)

    root_desc = build(leaf_descs)
    pairs[0] = root_desc
    return pairs


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

        # internal tree structure — build the bytes once, then both write them
        # to the file and store them back on root_col so the Properties panel
        # picks up the up-to-date values immediately after export.
        if use_stored_order:
            # reuse original tree structure (node count unchanged)
            tree_data = base64.b64decode(stored_tree_b64)
            print(f"[TrackAI Export] Reused stored AI BVH tree ({len(tree_data)} bytes)")
        else:
            tree_pairs = _build_bvh_tree(len(leaves))
            pair_count = len(tree_pairs)
            parts = [
                struct.pack('<I', pair_count),
                struct.pack('<I', 0x00040376),  # tree section tag
            ]
            for ref, typ in tree_pairs:
                parts.append(struct.pack('<II', ref, typ))
            parts.append(struct.pack('<II', TAG_AI_BVH_END1, TAG_AI_BVH_END2))
            tree_data = b''.join(parts)

            if stored_leaf_count >= 0 and stored_leaf_count != total_nodes:
                print(f"[TrackAI Export] Generated new AI BVH tree "
                      f"(node count changed: {stored_leaf_count} -> {total_nodes}, "
                      f"{pair_count} tree pairs)")
            else:
                print(f"[TrackAI Export] Generated AI BVH tree "
                      f"({pair_count} tree pairs)")

        f.write(tree_data)

        # persist the tree + leaf metadata onto root_col so the Blender UI
        # reflects the state that was just written, and so a subsequent export
        # from the same session round-trips through the "stored" path.
        root_col['fo2_ai_bvh_leaf_count'] = total_nodes
        root_col['fo2_ai_bvh_leaf_order'] = [int(nr) for nr, _ in leaves]
        root_col['fo2_ai_bvh_tree'] = base64.b64encode(tree_data).decode('ascii')

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
    """Generate splitpoints.bed from splitpoint objects. Returns True if
    written. Delegates gathering to _gather_splitpoint_objects so the .bed
    output picks up rotation/scale/vertex edits and prop refreshes for
    free — no more delta-only path that quietly ignored rotations."""
    splitpoints = _gather_splitpoint_objects(root_col)
    if not splitpoints:
        return False

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


# OPERATORS

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
# Uses this plugin's own export_trackai() function with dry_run=True
# so there's zero logic duplication or dependency on a separate addon.

def _find_export_trackai():
    """Return this plugin's own TrackAI export pipeline."""
    return export_trackai

class FO2_OT_PreviewTrackAI(bpy.types.Operator):
    """Run the TrackAI generation pipeline in-scene without writing any files.

    Opens a dialog with every knob the export operator exposes (CenterLine
    offset, TargetLine method + LERP + smoothing, speed_hint lookahead +
    radius). On OK, curves and node empties are created/updated in Blender
    as if you had exported, but nothing hits disk. Inspect the result in the
    Outliner, tweak manually if needed, then run the real export when ready.

    Uses the export pipeline from this plugin directly."""
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


def _project_to_route_xy(point, route_points, is_closed):
    """Return (distance_along_route, horizontal_tangent) nearest to point."""
    n = len(route_points)
    segment_count = n if is_closed else n - 1
    best = None
    distance_along = 0.0
    for i in range(segment_count):
        a = route_points[i]
        b = route_points[(i + 1) % n]
        dx = b.x - a.x
        dy = b.y - a.y
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-12:
            continue
        seg_len = math.sqrt(seg_len_sq)
        t = ((point.x - a.x) * dx + (point.y - a.y) * dy) / seg_len_sq
        t = max(0.0, min(1.0, t))
        px = a.x + t * dx
        py = a.y + t * dy
        dist_sq = (point.x - px) ** 2 + (point.y - py) ** 2
        candidate = (dist_sq, distance_along + t * seg_len,
                     Vector((dx / seg_len, dy / seg_len, 0.0)))
        if best is None or candidate[0] < best[0]:
            best = candidate
        distance_along += seg_len
    if best is None:
        return None
    return best[1], best[2], distance_along


class TRACKAI_OT_rotate_splitpoints_to_route(bpy.types.Operator):
    """Align and number splitpoint gates in Path0 CenterLine direction."""
    bl_idname = "trackai.rotate_splitpoints_to_route_direction"
    bl_label = "TrackAI: Rotate Splitpoints to Route Direction"
    bl_description = (
        "Rotate splitpoint gates perpendicular to Path0_CenterLine, enforce "
        "vertex 0=Left and vertex 2=Right, and renumber them in route order"
    )
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if context.mode != 'OBJECT':
            return False
        root = find_trackai_root(context)
        if root is None:
            return False
        path0 = next((c for c in root.children
                      if c.name == "TrackAI_Path0"), None)
        split_col = next((c for c in root.children
                          if c.name == "TrackAI_Splitpoints"), None)
        return path0 is not None and split_col is not None

    def execute(self, context):
        root = find_trackai_root(context)
        path0 = next((c for c in root.children
                      if c.name == "TrackAI_Path0"), None)
        split_col = next((c for c in root.children
                          if c.name == "TrackAI_Splitpoints"), None)
        center_obj = find_object_containing(path0, "_CenterLine")
        if center_obj is None or center_obj.type != 'CURVE':
            self.report({'ERROR'}, "TrackAI_Path0 needs a _CenterLine curve")
            return {'CANCELLED'}

        route_points = []
        is_closed = bool(path0.get('fo2_is_closed', True))
        for spline in center_obj.data.splines:
            if len(spline.points) >= 2:
                route_points.extend(center_obj.matrix_world @ p.co.xyz
                                    for p in spline.points)
                is_closed = bool(spline.use_cyclic_u)
                break
            if len(spline.bezier_points) >= 2:
                route_points.extend(center_obj.matrix_world @ p.co
                                    for p in spline.bezier_points)
                is_closed = bool(spline.use_cyclic_u)
                break
        if len(route_points) < 2:
            self.report({'ERROR'}, "Path0_CenterLine has fewer than two points")
            return {'CANCELLED'}

        gates = []
        for obj in split_col.objects:
            if obj.type != 'MESH' or len(obj.data.vertices) < 3:
                continue
            center_world = obj.matrix_world @ obj.data.vertices[1].co
            projected = _project_to_route_xy(
                center_world, route_points, is_closed)
            if projected is None:
                continue
            along, tangent, route_length = projected
            gates.append({
                'obj': obj,
                'old_index': int(obj.get('fo2_splitpoint_index', 0)),
                'along': along,
                'tangent': tangent,
                'route_length': route_length,
            })
        if not gates:
            self.report({'ERROR'}, "No valid splitpoint gate meshes found")
            return {'CANCELLED'}

        anchor = next((g for g in gates if g['old_index'] == 0),
                      min(gates, key=lambda g: g['along']))
        route_length = anchor['route_length']
        gates.sort(key=lambda g: ((g['along'] - anchor['along']) % route_length,
                                  g['old_index']))
        for new_index, gate in enumerate(gates):
            obj = gate['obj']
            verts = obj.data.vertices
            mw = obj.matrix_world
            inv = mw.inverted_safe()
            left_old = mw @ verts[0].co
            center = mw @ verts[1].co
            right_old = mw @ verts[2].co
            left_len = math.hypot(left_old.x - center.x,
                                  left_old.y - center.y)
            right_len = math.hypot(right_old.x - center.x,
                                   right_old.y - center.y)
            if left_len < 1e-6 or right_len < 1e-6:
                half = 0.5 * math.hypot(right_old.x - left_old.x,
                                        right_old.y - left_old.y)
                left_len = right_len = max(half, 1.0)

            forward = gate['tangent']
            driver_left = Vector((-forward.y, forward.x, 0.0))
            driver_right = -driver_left
            left_new = center + driver_left * left_len
            right_new = center + driver_right * right_len
            left_new.z = left_old.z
            right_new.z = right_old.z
            verts[0].co = inv @ left_new
            verts[1].co = inv @ center
            verts[2].co = inv @ right_new
            obj.data.update()
            obj['fo2_splitpoint_index'] = new_index

            pos_fo2 = blender_to_fo2(center)
            left_fo2 = blender_to_fo2(left_new)
            right_fo2 = blender_to_fo2(right_new)
            obj['fo2_splitpoint_position'] = list(pos_fo2)
            obj['fo2_splitpoint_left'] = list(left_fo2)
            obj['fo2_splitpoint_right'] = list(right_fo2)
            if 'fo2_bed_splitpoint_position' in obj:
                obj['fo2_bed_splitpoint_position'] = list(pos_fo2)
                obj['fo2_bed_splitpoint_left'] = list(left_fo2)
                obj['fo2_bed_splitpoint_right'] = list(right_fo2)

        for i, gate in enumerate(gates):
            gate['obj'].name = f"__TrackAI_SplitpointTemp_{i}__"
        for i, gate in enumerate(gates):
            gate['obj'].name = f"Splitpoint{i + 1}_Gate"
        context.view_layer.update()
        _refresh_ui(context)
        self.report({'INFO'},
                    f"Aligned and renumbered {len(gates)} splitpoints")
        return {'FINISHED'}


class TRACKAI_OT_ribbon_from_boundaries(bpy.types.Operator):
    """Create a Ribbon mesh from the active section's Left/Right boundary curves"""
    bl_idname = "trackai.ribbon_from_boundaries"
    bl_label = "TrackAI: Ribbon from Boundaries"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        col = context.collection
        return col is not None and _SECTION_NAME_RE.match(col.name) is not None

    def execute(self, context):
        sec_col = context.collection
        left_obj = find_object_containing(sec_col, "_LeftBoundary")
        right_obj = find_object_containing(sec_col, "_RightBoundary")
        if left_obj is None or right_obj is None:
            self.report({'ERROR'},
                        "Section needs both _LeftBoundary and _RightBoundary curves")
            return {'CANCELLED'}

        lefts_fo2 = sample_curve_points(left_obj)
        rights_fo2 = sample_curve_points(right_obj)
        if not lefts_fo2 or not rights_fo2:
            self.report({'ERROR'}, "Left/Right boundary curves have no points")
            return {'CANCELLED'}

        # Trim to matching length
        n = min(len(lefts_fo2), len(rights_fo2))
        lefts_fo2 = lefts_fo2[:n]
        rights_fo2 = rights_fo2[:n]

        sec_name = sec_col.name.replace("TrackAI_", "", 1)
        is_closed = bool(sec_col.get('fo2_is_closed', True))

        # Remove any existing ribbon to avoid duplicates
        existing = find_object_containing(sec_col, "_Ribbon")
        if existing is not None and existing.type == 'MESH':
            mesh = existing.data
            bpy.data.objects.remove(existing, do_unlink=True)
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)

        # Build alternating left/right verts in Blender space
        verts = []
        for i in range(n):
            # FO2 (x, y, z) -> Blender (x, z, y)
            verts.append((lefts_fo2[i][0], lefts_fo2[i][2], lefts_fo2[i][1]))
            verts.append((rights_fo2[i][0], rights_fo2[i][2], rights_fo2[i][1]))
        faces = []
        for i in range(n - 1):
            li = i * 2
            faces.append((li, li + 1, li + 3, li + 2))
        if is_closed and n > 2:
            li = (n - 1) * 2
            faces.append((li, li + 1, 1, 0))

        mesh = bpy.data.meshes.new(f"{sec_name}_Ribbon")
        mesh.from_pydata(verts, [], faces)
        mesh.update()
        obj = bpy.data.objects.new(f"{sec_name}_Ribbon", mesh)
        sec_col.objects.link(obj)

        self.report({'INFO'},
                    f"Created Ribbon: {len(verts)} verts, {len(faces)} faces")
        return {'FINISHED'}


class TRACKAI_OT_boundaries_from_ribbon(bpy.types.Operator):
    """Create _LeftBoundary and _RightBoundary curves from the active section's Ribbon mesh"""
    bl_idname = "trackai.boundaries_from_ribbon"
    bl_label = "TrackAI: Boundaries from Ribbon"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        col = context.collection
        return col is not None and _SECTION_NAME_RE.match(col.name) is not None

    def execute(self, context):
        sec_col = context.collection
        lefts_fo2, rights_fo2 = _extract_boundaries_from_ribbon(sec_col)
        if lefts_fo2 is None:
            self.report({'ERROR'},
                        "No valid Ribbon mesh in section (need alternating "
                        "left/right vertices in file order)")
            return {'CANCELLED'}

        sec_name = sec_col.name.replace("TrackAI_", "", 1)

        # Auto-detect closure from the ribbon's geometry rather than trusting
        # the section's `fo2_is_closed` prop (which defaults to True and
        # would force stem branches into cyclic NURBS, breaking every
        # downstream generator). Update the prop so the exporter sees the
        # corrected value on the next run.
        detected_closed = _detect_ribbon_closure(lefts_fo2, rights_fo2)
        prev_closed = bool(sec_col.get('fo2_is_closed', True))
        if detected_closed != prev_closed:
            print(f"[TrackAI] Section '{sec_name}': ribbon geometry "
                  f"indicates {'closed' if detected_closed else 'open'}, "
                  f"updating fo2_is_closed accordingly")
            sec_col['fo2_is_closed'] = detected_closed
        is_closed = detected_closed

        # Remove existing boundary curves to avoid duplicates
        for suffix in ("_LeftBoundary", "_RightBoundary"):
            existing = find_object_containing(sec_col, suffix)
            if existing is not None and existing.type == 'CURVE':
                data = existing.data
                bpy.data.objects.remove(existing, do_unlink=True)
                if data.users == 0:
                    bpy.data.curves.remove(data)

        def _make_curve(name, fo2_points):
            curve_data = bpy.data.curves.new(name, type='CURVE')
            curve_data.dimensions = '3D'
            curve_data.resolution_u = 12
            curve_data.bevel_depth = 0.15
            curve_data.bevel_resolution = 2
            spline = curve_data.splines.new('NURBS')
            spline.points.add(len(fo2_points) - 1)
            for i, p in enumerate(fo2_points):
                # FO2 (x, y, z) -> Blender (x, z, y)
                spline.points[i].co = (p[0], p[2], p[1], 1.0)
            spline.use_cyclic_u = is_closed
            spline.order_u = 3
            obj = bpy.data.objects.new(name, curve_data)
            sec_col.objects.link(obj)
            return obj

        _make_curve(f"{sec_name}_LeftBoundary", lefts_fo2)
        _make_curve(f"{sec_name}_RightBoundary", rights_fo2)

        self.report({'INFO'},
                    f"Created LeftBoundary + RightBoundary ({len(lefts_fo2)} points each)")
        return {'FINISHED'}


# EXPORT OPERATOR

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

    auto_generate_target: BoolProperty(
        name="Auto-generate TargetLine",
        description="If a section has no _TargetLine curve, create one from "
                    "the boundaries. Happens after Ribbon->Boundary derivation "
                    "and after CenterLine generation, so ribbon-only sections "
                    "still get a target.",
        default=True,
    )

    target_method: EnumProperty(
        name="Method",
        description="How to synthesise the TargetLine when auto-generating",
        items=[
            ('SMOOTH',    "Smoothed racing line",
             "Corridor-clamped Chaikin smoothing: LERP at t within the ribbon, "
             "then iteratively smooth. Straight sections stay near t; turns "
             "naturally pull the curve toward the corridor edge (wider entry, "
             "cut apex, wider exit). Best fit across most vanilla tracks."),
            ('DUPLICATE', "Duplicate boundary",
             "Copy one boundary verbatim (nascar-style AI that hugs a wall)."),
        ],
        default='SMOOTH',
    )

    target_lerp: FloatProperty(
        name="Base position",
        description="Initial LERP position inside the ribbon. 0 = inner "
                    "(RightBoundary), 0.5 = ribbon center, 1 = outer "
                    "(LeftBoundary). Default 0.30 matches the empirical mean "
                    "across vanilla tracks that don't hug a boundary uniformly",
        default=0.30,
        min=0.0, max=1.0, step=5, precision=2,
    )

    target_smooth_iters: IntProperty(
        name="Smoothing passes",
        description="Number of Chaikin smoothing iterations applied to the "
                    "initial LERP line. 0 = plain LERP (no smoothing). Grid "
                    "search on 15 vanilla tracks converged around 10; more "
                    "passes produce smoother turns but at some point stop "
                    "improving",
        default=10,
        min=0, max=50,
    )

    target_source: EnumProperty(
        name="Duplicate from",
        description="Which boundary to duplicate when method is Duplicate",
        items=[
            ('RIGHT', "RightBoundary", "Duplicate the inner boundary (RightBoundary in our plugin's convention)"),
            ('LEFT',  "LeftBoundary",  "Duplicate the outer boundary (LeftBoundary in our plugin's convention)"),
        ],
        default='RIGHT',
    )

    auto_generate_center: BoolProperty(
        name="Auto-generate CenterLine",
        description="If a section has no _CenterLine curve, create one by "
                    "offsetting RightBoundary perpendicular toward the track "
                    "interior. Requires both boundaries (uses Ribbon-derived "
                    "ones when applicable). Runs before TargetLine.",
        default=True,
    )

    center_offset: FloatProperty(
        name="Offset",
        description="Perpendicular distance from RightBoundary to the "
                    "generated CenterLine, in FO2 units. Default 3.40 matches "
                    "the empirical mean across vanilla tracks (nascar is "
                    "exactly 3.00; other tracks range 3.0-4.0)",
        default=3.40,
        min=0.0, max=50.0, step=10, precision=2,
    )

    generate_speed_hints: BoolProperty(
        name="Generate speed hints from geometry",
        description="Compute per-node fo2_speed_hint from local track "
                    "curvature when nodes are being generated from scratch. "
                    "When unchecked, newly-generated nodes get the MAX "
                    "sentinel (1,000,000) meaning 'no limit', matching the "
                    "previous behaviour. Only affects nodes being generated "
                    "from curves — existing node empties are always "
                    "preserved verbatim regardless of this setting",
        default=True,
    )

    flip_boundaries: BoolProperty(
        name="Flip boundaries",
        description="Swap Left and Right boundaries when they get derived "
                    "from a Ribbon mesh. Ribbon geometry doesn't always "
                    "self-identify which side is which — if CenterLine "
                    "ends up on the outer edge of the track instead of "
                    "down the middle (because RightBoundary was really "
                    "the outer one), toggle this and re-export. Applied "
                    "before every downstream generator (CenterLine, "
                    "TargetLine, nodes, speed hints), so the swap "
                    "cascades cleanly",
        default=False,
    )

    speed_lookahead: IntProperty(
        name="Lookahead",
        description="How many nodes ahead the speed_hint algorithm scans for "
                    "the tightest upcoming turn. Larger values make the AI "
                    "start slowing earlier before turns. Default 3 nodes",
        default=3,
        min=1, max=15,
    )

    speed_radius_threshold: FloatProperty(
        name="Radius",
        description="Corner radius (FO2 units) above which speed uncaps to "
                    "MAX. Any turn with radius > this value produces full "
                    "speed. Lower values = less sensitive detection (only "
                    "tight turns slow the AI). Higher = more sensitive "
                    "(mild curves also trigger slowdown). Default 7071 "
                    "matches the empirical curvature-coefficient fit "
                    "(C ≈ 0.02) across 15 vanilla tracks",
        default=7071.0,
        min=100.0, max=30000.0, step=100, precision=1,
    )

    def draw(self, context):
        layout = self.layout

        box = layout.box()
        box.label(text="Companion Files", icon='FILE_FOLDER')
        box.prop(self, "export_splines_ai")
        box.prop(self, "export_startpoints_bed")
        box.prop(self, "export_splitpoints_bed")

        # Auto-generation section. Each checkbox is disabled when every section
        # already has the corresponding curve (nothing to do). CenterLine
        # generation runs before TargetLine so a ribbon-only section ends up
        # with both boundaries, then CenterLine, then TargetLine — all four
        # curves auto-created in one export.
        box = layout.box()
        box.label(text="Auto-generation", icon='NODETREE')

        # CenterLine controls
        any_center_missing = _any_section_missing_centerline()
        row = box.row()
        row.enabled = any_center_missing
        row.prop(self, "auto_generate_center")
        if not any_center_missing:
            info = box.row()
            info.enabled = False
            info.label(text="All sections already have a CenterLine",
                       icon='CHECKMARK')
        else:
            picker = box.row()
            picker.enabled = self.auto_generate_center
            picker.prop(self, "center_offset")

        box.separator()

        # TargetLine controls — method-dependent sub-widgets
        any_target_missing = _any_section_missing_targetline()
        row = box.row()
        row.enabled = any_target_missing
        row.prop(self, "auto_generate_target")
        if not any_target_missing:
            info = box.row()
            info.enabled = False
            info.label(text="All sections already have a TargetLine",
                       icon='CHECKMARK')
        else:
            sub = box.column()
            sub.enabled = self.auto_generate_target
            sub.prop(self, "target_method")
            if self.target_method == 'SMOOTH':
                sub.prop(self, "target_lerp", slider=True)
                sub.prop(self, "target_smooth_iters")
            else:  # DUPLICATE
                sub.prop(self, "target_source", expand=True)

        # Speed-hint tuning. Applies to from-scratch node generation only —
        # sections that already have node empties (roundtrip) read the stored
        # fo2_speed_hint verbatim and are unaffected by these knobs.
        box = layout.box()
        box.label(text="Speed hint (AI cornering)", icon='AUTO')
        box.prop(self, "generate_speed_hints")
        sub = box.column()
        sub.enabled = self.generate_speed_hints
        sub.prop(self, "speed_lookahead")
        sub.prop(self, "speed_radius_threshold")

        # Boundary side override. Enabled only when there's actually a
        # from-scratch derivation about to happen; a stale True while every
        # section already has explicit L/R curves would be a silent no-op.
        box = layout.box()
        box.label(text="Boundaries", icon='MOD_MIRROR')
        box.prop(self, "flip_boundaries")

    def execute(self, context):
        # Only pass the auto-gen flags when they can actually do something —
        # matches the disabled-checkbox UX so a stale True doesn't try to
        # generate over already-existing curves.
        any_target_missing = _any_section_missing_targetline()
        any_center_missing = _any_section_missing_centerline()
        options = {
            'export_splines_ai': self.export_splines_ai,
            'export_startpoints_bed': self.export_startpoints_bed,
            'export_splitpoints_bed': self.export_splitpoints_bed,
            'auto_generate_target': self.auto_generate_target and any_target_missing,
            'target_method': self.target_method,
            'target_source': self.target_source,
            'target_lerp': float(self.target_lerp),
            'target_smooth_iters': int(self.target_smooth_iters),
            'auto_generate_center': self.auto_generate_center and any_center_missing,
            'center_offset': float(self.center_offset),
            'speed_lookahead': int(self.speed_lookahead),
            'speed_radius_threshold': float(self.speed_radius_threshold),
            'generate_speed_hints': self.generate_speed_hints,
            'flip_boundaries': self.flip_boundaries,
        }
        try:
            result = export_trackai(self.filepath, context, options)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            import traceback; traceback.print_exc()
            return {'CANCELLED'}

        # Custom properties we set on collections (fo2_section_index, fo2_is_closed,
        # fo2_ai_bvh_*, fo2_splines_ai, fo2_*_bed, ...) and any auto-generated
        # curve objects (LeftBoundary/RightBoundary from Ribbon, TargetLine) don't
        # auto-refresh the Properties editor / Outliner. Tag every visible area
        # for redraw so the new state is visible without needing a click-away.
        _refresh_ui(context)

        self.report({'INFO'}, f"Exported Track AI to {os.path.basename(self.filepath)}")
        return result


def _refresh_ui(context):
    """Force redraw of all Blender areas so newly-set custom properties on
    collections and newly-created curve objects become visible in the
    Properties/Outliner panels immediately. Blender does not auto-refresh
    those panels when ID properties are added or objects are linked outside
    of a depsgraph-tracked change."""
    try:
        # Nudge the depsgraph so newly-linked objects are recognised by the
        # active view layer before the panels redraw.
        context.view_layer.update()
    except Exception:
        pass
    try:
        wm = context.window_manager
        for window in wm.windows:
            for area in window.screen.areas:
                area.tag_redraw()
    except Exception:
        pass  # non-fatal — worst case user needs to click the collection


def menu_func_export(self, context):
    self.layout.operator(ExportTrackAI.bl_idname, text="FlatOut 2 TrackAI (.bin)")


def menu_func_object(self, context):
    # Show only when a TrackAI_Path{N} collection is active.
    col = context.collection
    if col is not None and _SECTION_NAME_RE.match(col.name):
        self.layout.separator()
        self.layout.operator(TRACKAI_OT_ribbon_from_boundaries.bl_idname,
                             icon='MESH_GRID')
        self.layout.operator(TRACKAI_OT_boundaries_from_ribbon.bl_idname,
                             icon='CURVE_NCURVE')

    self.layout.separator()
    self.layout.operator(FO2_OT_MakeTrackAIHierarchy.bl_idname)
    self.layout.operator(FO2_OT_AddStandardStartpoints.bl_idname)
    self.layout.operator(FO2_OT_SnapStartpointsToRibbon.bl_idname)
    self.layout.operator(TRACKAI_OT_rotate_splitpoints_to_route.bl_idname)
    self.layout.separator()
    self.layout.operator(FO2_OT_ReverseTrack.bl_idname)
    self.layout.operator(FO2_OT_ReverseNodeIndexes.bl_idname)
    self.layout.operator(FO2_OT_ReverseSplitpointIndexes.bl_idname)
    self.layout.separator()
    self.layout.operator(FO2_OT_PreviewTrackAI.bl_idname)


_CLASSES = (
    FO2_OT_AddStandardStartpoints,
    FO2_OT_SnapStartpointsToRibbon,
    FO2_OT_ReverseTrack,
    FO2_OT_PreviewTrackAI,
    FO2_OT_MakeTrackAIHierarchy,
    FO2_OT_ReverseNodeIndexes,
    FO2_OT_ReverseSplitpointIndexes,
    TRACKAI_OT_rotate_splitpoints_to_route,
    TRACKAI_OT_ribbon_from_boundaries,
    TRACKAI_OT_boundaries_from_ribbon,
    ExportTrackAI,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)
    bpy.types.VIEW3D_MT_object.append(menu_func_object)


def unregister():
    bpy.types.VIEW3D_MT_object.remove(menu_func_object)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
