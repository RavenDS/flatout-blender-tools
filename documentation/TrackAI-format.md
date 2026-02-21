# FlatOut 2 — Track File Format Specification

Research by ravenDS (https://github.com/RavenDS/flatout-blender-tools)
21/02/2026

## Overview

```
- trackai.bin           AI racing paths + startpoints + splitpoints + AI BVH
- splines.ai            AI border splines (text, Lua-like format)
- splitpoints.bed       Checkpoint gate definitions (text)
- startpoints.bed       Start grid positions (text)
```

All multi-byte values are **little-endian**. Coordinates use a **Y-up** system:
X = right, Y = up, Z = forward.

---

# `trackai.bin` — AI Racing Paths

## File Structure

```
┌─────────────────────────────┐
│  File Header  (8 bytes)     │
├─────────────────────────────┤
│  Spline Section 0           │  Main racing line
│    Section Header (8 bytes) │
│    Node 0  (208 bytes)      │
│    ...                      │
│    Node N-1                 │
│    Section Footer (20 bytes)│
├─────────────────────────────┤
│  Section Separator (4 bytes)│  0x00260276
├─────────────────────────────┤
│  Spline Section 1..N        │
│    (same structure)         │
├─────────────────────────────┤
│  Section Separator          │  Also present after LAST section
├─────────────────────────────┤
│  Extra Data Block           │
│    TAG_EXTRA_START          │
│    Startpoints              │
│    Splitpoints              │
│    AI BVH Tree              │
│    TAG_FILE_END             │
└─────────────────────────────┘
```


## Tag Constants

```c
#define TAG_FILE_HEADER     0x00270276   // File start
#define TAG_SPLINE_SECTION  0x00290276   // Section start (node count follows)
#define TAG_NODE_START      0x00230276   // AI node begin
#define TAG_NODE_END        0x00240276   // AI node end
#define TAG_SECTION_SEP     0x00260276   // Separator between AND after sections
#define TAG_FILE_END        0x00280276   // File end
#define TAG_EXTRA_START     0x00280876   // Extra data block start
#define TAG_STARTPOINTS     0x00300876   // Startpoints sub-block
#define TAG_SPLITPOINTS     0x00310876   // Splitpoints sub-block
#define TAG_AI_BVH_1        0x00020976   // AI BVH sub-block start
#define TAG_AI_BVH_2        0x00290876   // AI BVH secondary tag
```


## C Structures

### File Header

```c
struct TrackAIHeader {
    uint32_t tag;            // TAG_FILE_HEADER (0x00270276)
    uint32_t num_sections;   // Number of spline sections (typically 1–4)
};
```

### AI Node (208 bytes)

Each node is a waypoint along a spline. Nodes form a linked list via `index` and
`prev_index`. For closed loops, the last node wraps back to index 0. For open paths
that branch from another section, the last node's `index` may cross-reference into a
different section.

```c
struct AINode {                     // 208 bytes total
    // ── Header (20 bytes) ──
    uint32_t tag;                   //   0: TAG_NODE_START (0x00230276)
    uint32_t index;                 //   4: Linked-list index (NOT sequential;
                                    //      see "Indexing" section below)
    uint32_t unk1;                  //   8: Usually 0; occasionally 1
    uint32_t prev_index;            //  12: Previous node's linked-list index
    uint32_t unk2;                  //  16: Usually 0

    // ── Orientation (36 bytes) ──
    float    rotation[9];           //  20: 3×3 rotation matrix (row-major)
                                    //      Columns: [right_dir | up | forward]

    // ── Positions (60 bytes) ──
    float    center[3];             //  56: Center of track at this waypoint
    float    left[3];               //  68: Left track boundary
    float    right[3];              //  80: Right track boundary
    float    mid[3];                //  92: Midpoint between left and right
    float    target[3];             // 104: AI target position (may differ from mid)

    // ── Directions (24 bytes) ──
    float    forward[3];            // 116: Forward direction (unit vector)
    float    right_dir[3];          // 128: Right/lateral direction (unit vector)

    // ── Interpolation & Widths (20 bytes) ──
    float    interp_weights[3];     // 140: Interpolation weights for blending
    float    width_left;            // 152: Distance from center to left boundary
    float    width_right;           // 156: Distance from center to right boundary

    // ── Distance & Speed (20 bytes) ──
    float    cumul_distance;        // 160: Cumulative arc length from section start
    float    unk_neg1;              // 164: Usually -1.0; occasionally a distance value
    float    speed_hint;            // 168: AI speed/priority hint (higher = faster)

    // ── Flags & Metadata (28 bytes) ──
    int32_t  unk3;                  // 172: 0 or -1 (0xFFFFFFFF)
    int32_t  sentinel1;             // 176: Always -1
    float    speed_hint2;           // 180: Copy of speed_hint
    uint32_t flag;                  // 184: 0 or 1; marks special waypoints
    uint32_t seq_index;             // 188: Sequential file position (0, 1, 2, ... N-1)
    uint32_t unk4;                  // 192: Usually 0
    int32_t  sentinel2;             // 196: Usually -1; occasionally 0
    uint32_t unk5;                  // 200: Usually 0

    // ── Footer (4 bytes) ──
    uint32_t end_tag;               // 204: TAG_NODE_END (0x00240276)
};
```


### Indexing — `index` vs `seq_index`

| Field       | Offset | Purpose                              | Sequential? |
|-------------|--------|--------------------------------------|-------------|
| `index`     | 4      | Linked-list pointer for AI pathfinding | **No**     |
| `seq_index` | 188    | File position (0, 1, 2 ... N-1)       | **Yes**    |

**Closed loop** (e.g. main racing line):
```
File pos:    0    1    2   ...   74   75
index:       1    2    3   ...   75    0   ← wraps
prev_index: 75    0    1   ...   73   74   ← wraps
seq_index:   0    1    2   ...   74   75   ← always sequential
```

**Open path** (branching from another section):
```
File pos:    0    1    2    3   ...   10   11
index:       1    2    3    4   ...   11    3   ← merges to sec 0 node 3
prev_index: 66    0    1    2   ...    9   10   ← branches from sec 0 node 66
```

**Critical:** Tools must preserve file order using `seq_index`, never sort by `index`.


### Section Footer

```c
struct SectionFooter {              // 20 bytes
    uint32_t section_id;            // 0 for main racing line, 1+ for alternates
    float    speed_factor;          // Usually 0.5 or 0.7
    uint32_t reserved1;             // 0
    uint32_t reserved2;             // 0
    uint32_t next_section_id;       // Cross-reference to another section
};
```


### Section Roles

| Section | Typical Role         | Loop   | Notes                                    |
|---------|----------------------|--------|------------------------------------------|
| 0       | Main racing line     | Closed | Always present; defines the lap          |
| 1       | Shortcut / pit lane  | Open   | Branches from/merges to section 0        |
| 2       | Alternate path       | Open   | Usually branches from section 0          |
| 3       | Secondary shortcut   | Open   | May branch from section 0 or 1           |


## Extra Data Block

After all sections and the trailing separator, the extra data block contains:

### Startpoints

```c
// Tag: TAG_EXTRA_START (0x00280876)
// Tag: TAG_STARTPOINTS (0x00300876)
// uint32_t count

struct Startpoint {                 // 48 bytes
    float position[3];              // World-space grid position
    float rotation[9];              // 3×3 orientation matrix
};
```

Typically 8 positions for the start grid.


### Splitpoints

```c
// Tag: TAG_SPLITPOINTS (0x00310876)
// uint32_t subtag                  // Always 0x00010976
// uint32_t count

struct Splitpoint {                 // 36 bytes
    float position[3];              // Gate center position
    float left[3];                  // Left edge of checkpoint gate
    float right[3];                 // Right edge of checkpoint gate
};
```

Checkpoint gates for lap tracking and wrong-way detection.


### AI BVH (Embedded Spatial Lookup Tree)

After splitpoints, an AI-specific bounding volume hierarchy is embedded for fast
spatial queries (e.g. "which track segment is the car nearest to?").

```c
// Header (24 bytes):
uint32_t tag1;         // TAG_AI_BVH_1 (0x00020976)
uint32_t tag2;         // 0x00290876
uint32_t tag3;         // TAG_SPLINE_SECTION (0x00290276) — reused tag
uint32_t total_nodes;  // Total AI node count across all sections
uint32_t tag4;         // 0x00020376
uint32_t reserved;     // 0
```

Followed by an array of BVH entries. Each entry is 32 bytes:

```c
struct AIBVHEntry {                 // 32 bytes
    uint32_t node_ref;              // Packed: (section_index << 24) | node_index
    float    aabb_min[3];           // 2D AABB minimum (x, 0, z) — Y always 0
    float    aabb_max[3];           // 2D AABB maximum (x, 0, z) — Y always 0
    uint32_t tree_index;            // Sequential tree node index
};
```

The `node_ref` field encodes which section and node this entry covers:
- Bits 24-31: section index (0, 1, 2, 3)
- Bits 0-23: node index within that section

The bounding boxes are projected onto the XZ ground plane (Y is always 0).

The first `total_nodes` entries are leaf nodes (one per AI node segment) with valid
AABB data. Additional entries after those are internal tree nodes with zero positions
and child index references. The block ends with tags and `TAG_FILE_END` (0x00280276).

**Note:** This AI BVH is separate from `track_bvh.gen` which handles visibility
culling for track geometry.

---

# Companion Text Files

### `splines.ai`

AI border splines in a Lua-like text format — invisible walls that keep AI on track:

```
["border_left"]
[0] = { 21.39, 92.19, 234.31 }
[1] = { 20.22, 93.54, 252.92 }
...

["border_right"]
[0] = { 24.05, 92.49, 234.46 }
...
```

Common spline names: `border_left`, `border_right`, `shortcut_left`, `shortcut_right`.


### `splitpoints.bed` / `startpoints.bed`

Text equivalents of the embedded startpoints and splitpoints in `trackai.bin`.
Used by track editor tools. Stored in a different geometry format.
The binary versions in `trackai.bin` take priority at runtime.


---

# Notes

For lossless `trackai.bin` import/export:

1. **Node file order** — write nodes in the sequence they were read.
   Sort by `seq_index` (offset 188) or a file-position counter, **never** by `index`.

2. **Trailing separator** — `TAG_SECTION_SEP` appears after the *last* section too.

3. **Extra data block** — the AI BVH and any trailing tags must be preserved verbatim.

4. **Signed fields** — `unk3`, `sentinel1`, `sentinel2` can hold -1 (0xFFFFFFFF).
   Store as `int32_t` to avoid overflow in tools with signed-only integer properties.

5. **All node fields** — recomputing derived values like `rotation`, `forward`, `right_dir`,
   `mid`, `cumul_distance`. Recomputation may introduce floating-point drift.
