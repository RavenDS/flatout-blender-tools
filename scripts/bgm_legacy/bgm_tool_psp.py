#!/usr/bin/env python3
"""
FlatOut 2 BGM Converter: PC -> PSP (v1)
by ravenDS (github.com/ravenDS)

Input:  FO2 PC BGM file (version 0x20000, float vertex buffers)
Output: PSP BGM file for FlatOut Head On

PSP BGM format overview
-----------------------
Overall file structure is identical to FO2 PC:
  uint32  version = 0x00020000
  uint32  num_materials
  [materials — identical to FO2 PC]
  uint32  num_streams = 1
  [1 stream: dt=3, fc=0, vc=<blob_bytes>, vs=1, data=<raw GE blob>]
  uint32  num_surfaces
  [surface records — 32 bytes each, see below]
  uint32  num_models
  [models — identical to FO2 PC]
  uint32  num_meshes
  uint32  num_objects

GE display list blob
--------------------
All surfaces are packed into a single contiguous blob.
Surfaces are GROUPED BY MATERIAL ID in the blob (not in surface-index order).
Within each material group, surfaces appear in their original surface-index order.

Per-surface GE chunk layout
----------------------------
  [20 bytes] GE command header
    0x14000000  ORIGIN()            SCE_GE_CMD_ORIGIN  — capture current addr as origin
    0x10000000  BASE(0)             SCE_GE_CMD_BASE    — upper address bits = 0
    0x01000014  VADR(20)            SCE_GE_CMD_VADR    — vertex data = origin + 20
    0x04pp00nn  PRIM(pp, nn)        SCE_GE_CMD_PRIM    — pp=prim type, nn=vertex count
                                      pp=03 SCE_GE_PRIM_TRIANGLES (PC poly_mode 4)
                                      pp=04 SCE_GE_PRIM_TRIANGLE_STRIP (PC poly_mode 5)
    0x0b000000  RET()               SCE_GE_CMD_RET     — return from display list
  [nv * stride bytes] vertex data
  [0-15 bytes] zero padding to 16-byte alignment

Vertex formats
--------------
  Standard (non-shadow) — 14 bytes:
    int16   u, v         (UV,     multiply by 1/4096 to get float)
    int8    nx, ny, nz   (normal, multiply by 1/127  to get float)
    int8    pad = 0
    int16   px, py, pz   (pos,    multiply by 1/1024 to get float)

  Shadow — 6 bytes:
    int16   px, py, pz   (pos,    multiply by 1/1024 to get float)

Surface record — 32 bytes (8 x uint32)
---------------------------------------
  [0] flags = 0x00001000  (always)
  [1] mid   = material index
  [2] nv    = vertex count
  [3] nb    = 1           (always; no batch splitting on PSP)
  [4] f4    = 0x00000027  (always; engine constant)
  [5] f5    = 0           (always)
  [6] boff  = byte offset of this surface's GE chunk within the blob
  [7] bsz   = byte size   of this surface's GE chunk

Poly mode mapping (PC -> PSP GE primitive type)
------------------------------------------------
  PC poly_mode 4 (triangle list)  -> GE pp=03  GU_TRIANGLES
  PC poly_mode 5 (triangle strip) -> GE pp=04  GU_TRIANGLE_STRIP

Strip optimization
------------------
  PM=4 (triangle list) surfaces are ALWAYS converted to TRI_STRIP via
  NvTriStrip, regardless of flags.  This matches Bugbear's original
  behaviour (all PSP BGM surfaces are TRI_STRIP) and saves ~30 % of
  vertex data for list-mode surfaces.

  PM=5 (triangle strip) surfaces are passed through unchanged.  The PC
  strip is already a highly optimised sub-strip sequence; re-stripifying
  from canonical triangles produces a larger result for this data.

  -strip flag: raises NvTriStrip quality for PM=4 from 8 to num_samples
        candidate starting faces (default 16 with -strip).  The extra
        quality is marginal (~0 to 1 % additional saving) at the cost of
        longer conversion time.

  -samples N  (default 16): NvTriStrip sample count when -strip is active.
        Higher values try more candidate starting faces per iteration,
        producing longer strips at the cost of more conversion time.
"""

import struct, sys, os
from collections import defaultdict

# encoding scales
POS_SCALE  = 1024.0   # float pos  -> int16
UV_SCALE   = 2048.0   # float UV   -> uint16 (scale factor)
UV_OFFSET  = 16384    # float UV   -> uint16 (base offset)

# The PSP game engine sets up a texture UV transform (SU=8, TU=-3.5):
#   game_uv = (raw_UV - 14336) / 4096
# (the game decodes raw as a float in that space)

# PSP UV encoded as:
#   PSP_game_uv = PC_UV * 0.5 + 0.5   (halved range, shifted to [0.5, 1.0])
# Therefore the raw encoding is:
#   raw_UV  = round(PC_UV * UV_SCALE) + UV_OFFSET
#           = round(PC_UV * 2048) + 16384
#   game_uv = PC_UV * 0.5 + 0.5 (raw range [16384, 18432] for PC UV [0.0, 1.0])

NORM_SCALE = 127.0    # float norm -> int8

# vertex types
VTYPE_STANDARD = 0
VTYPE_SHADOW   = 1

# PSP GE primitive type codes — bits [18:16] of the PRIM command word
# SCE_GE_SET_PRIM(_count, _prim) = (0x04<<24) | (_prim<<16) | _count
GE_PRIM_TRIANGLES = 3   # SCE_GE_PRIM_TRIANGLES       (PC poly_mode == 4)
GE_PRIM_STRIP     = 4   # SCE_GE_PRIM_TRIANGLE_STRIP  (PC poly_mode == 5)


# ── helpers ──

def clamp_i16(v):
    return max(-32768, min(32767, round(v)))

def clamp_i8(v):
    return max(-128, min(127, round(v)))

def clamp_u16(v):
    return max(0, min(65535, round(v)))

def ge_cmd(cmd8, param24):
    """Pack one 4-byte PSP GE command word."""
    return struct.pack('<I', ((cmd8 & 0xFF) << 24) | (param24 & 0xFFFFFF))

def read_string(f):
    s = b''
    while True:
        c = f.read(1)
        if not c or c == b'\x00':
            break
        s += c
    return s.decode('ascii', errors='replace')

def write_string(f, s):
    f.write(s.encode('ascii') + b'\x00')


# ── FO2 PC parser ──

class FO2Parser:
    def __init__(self, path):
        self.path      = path
        self.version   = 0
        self.materials = []
        self.vbufs     = {}   # stream_idx -> (vc, vs, flags, data)
        self.ibufs     = {}   # stream_idx -> (ic, data)
        self.surfaces  = []
        self.models    = []
        self.meshes    = []
        self.objects   = []

    def parse(self):
        with open(self.path, 'rb') as f:
            self.version = struct.unpack('<I', f.read(4))[0]
            if self.version != 0x20000:
                raise ValueError(f"Expected FO2 version 0x20000, got 0x{self.version:X}. "
                                 f"Only FO2 PC BGM files are supported.")
            self._parse_materials(f)
            self._parse_streams(f)
            self._parse_surfaces(f)
            self._parse_models(f)
            self._parse_meshes(f)
            self._parse_objects(f)

    def _parse_materials(self, f):
        for _ in range(struct.unpack('<I', f.read(4))[0]):
            ident = struct.unpack('<I', f.read(4))[0]
            name  = read_string(f)
            alpha = struct.unpack('<i', f.read(4))[0]
            v92, n_num_tex, shader_id, n_use_colormap, v74 = struct.unpack('<5i', f.read(20))
            v108  = f.read(12)
            v109  = f.read(12)
            v98   = f.read(16)
            v99   = f.read(16)
            v100  = f.read(16)
            v101  = f.read(16)
            v102  = struct.unpack('<i', f.read(4))[0]
            tex_names = [read_string(f) for _ in range(3)]
            self.materials.append({
                'ident': ident, 'name': name, 'alpha': alpha,
                'v92': v92, 'n_num_tex': n_num_tex, 'shader_id': shader_id,
                'n_use_colormap': n_use_colormap, 'v74': v74,
                'v108': v108, 'v109': v109,
                'v98': v98, 'v99': v99, 'v100': v100, 'v101': v101,
                'v102': v102, 'tex_names': tex_names,
            })

    def _parse_streams(self, f):
        for i in range(struct.unpack('<I', f.read(4))[0]):
            dt = struct.unpack('<I', f.read(4))[0]
            if dt == 1:
                fc, vc, vs, fl = struct.unpack('<4I', f.read(16))
                self.vbufs[i] = (vc, vs, fl, f.read(vc * vs))
            elif dt == 2:
                fc, ic = struct.unpack('<2I', f.read(8))
                self.ibufs[i] = (ic, f.read(ic * 2))
            elif dt == 3:
                fc, vc, vs = struct.unpack('<3I', f.read(12))
                self.vbufs[i] = (vc, vs, 0, f.read(vc * vs))

    def _parse_surfaces(self, f):
        for _ in range(struct.unpack('<I', f.read(4))[0]):
            isveg, mid, vc, flags, pc, pm, niu = struct.unpack('<7i', f.read(28))
            # FO2: no extra bytes between fields and stream list
            nst  = struct.unpack('<i', f.read(4))[0]
            sids = []; softs = []
            for _ in range(nst):
                sid, soff = struct.unpack('<2I', f.read(8))
                sids.append(sid); softs.append(soff)
            self.surfaces.append({
                'isveg': isveg, 'mid': mid, 'vc': vc, 'flags': flags,
                'pc': pc, 'pm': pm, 'niu': niu,
                'nst': nst, 'sids': sids, 'soffs': softs,
            })

    def _parse_models(self, f):
        for _ in range(struct.unpack('<I', f.read(4))[0]):
            ident    = struct.unpack('<I', f.read(4))[0]
            unk      = struct.unpack('<i', f.read(4))[0]
            name     = read_string(f)
            center   = struct.unpack('<3f', f.read(12))
            radius   = struct.unpack('<3f', f.read(12))
            f_radius = struct.unpack('<f',  f.read(4))[0]
            ns       = struct.unpack('<I',  f.read(4))[0]
            surfs    = [struct.unpack('<i', f.read(4))[0] for _ in range(ns)]
            self.models.append({'ident': ident, 'unk': unk, 'name': name,
                                 'center': center, 'radius': radius,
                                 'f_radius': f_radius, 'surfaces': surfs})

    def _parse_meshes(self, f):
        for _ in range(struct.unpack('<I', f.read(4))[0]):
            ident  = struct.unpack('<I',   f.read(4))[0]
            name1  = read_string(f)
            name2  = read_string(f)
            flags  = struct.unpack('<I',   f.read(4))[0]
            group  = struct.unpack('<i',   f.read(4))[0]
            matrix = struct.unpack('<16f', f.read(64))
            nm     = struct.unpack('<i',   f.read(4))[0]
            mids   = [struct.unpack('<i',  f.read(4))[0] for _ in range(nm)]
            self.meshes.append({'ident': ident, 'name1': name1, 'name2': name2,
                                 'flags': flags, 'group': group, 'matrix': matrix,
                                 'model_ids': mids})

    def _parse_objects(self, f):
        for _ in range(struct.unpack('<I', f.read(4))[0]):
            ident  = struct.unpack('<I',   f.read(4))[0]
            name1  = read_string(f)
            name2  = read_string(f)
            flags  = struct.unpack('<I',   f.read(4))[0]
            matrix = struct.unpack('<16f', f.read(64))
            self.objects.append({'ident': ident, 'name1': name1, 'name2': name2,
                                  'flags': flags, 'matrix': matrix})

    def get_vtype(self, surf):
        if surf['nst'] < 1:
            return VTYPE_STANDARD
        vid = surf['sids'][0]
        if vid not in self.vbufs:
            return VTYPE_STANDARD
        _, vs, fl, _ = self.vbufs[vid]
        if fl == 0x0002 or vs == 12:
            return VTYPE_SHADOW
        return VTYPE_STANDARD

    def get_vertices(self, surf):
        """Return list of (pos, normal, uv) tuples for a surface, in index order.

        PC surfaces use indexed rendering: the vertex buffer holds unique verts
        and the index buffer defines the strip/triangle order.  We expand through
        the index buffer so the returned list is in the correct draw order.

        Index values are ABSOLUTE vertex indices into the full shared vertex stream
        (not relative to this surface's soffs[0]).  soffs[0] is the byte offset of
        the surface's vertex range; indices already encode the absolute position.

        pos/normal are float[3], uv is float[2].
        Shadow surfaces return only pos; normal/uv will be zero.
        """
        if surf['nst'] < 1:
            return []
        vid = surf['sids'][0]
        if vid not in self.vbufs:
            return []
        vc, vs, fl, vdata = self.vbufs[vid]
        vtype    = self.get_vtype(surf)
        has_norm = bool(fl & 0x10)
        has_color= bool(fl & 0x40)
        has_uv   = bool(fl & 0x300)

        def read_vert_abs(idx):
            """Read vertex by absolute stream index."""
            off = idx * vs
            if off + vs > len(vdata):
                return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0))
            pos    = struct.unpack_from('<3f', vdata, off)
            normal = (0.0, 0.0, 0.0)
            uv     = (0.0, 0.0)
            if vtype != VTYPE_SHADOW:
                fo = 12
                if has_norm:  normal = struct.unpack_from('<3f', vdata, off + fo); fo += 12
                if has_color: fo += 4
                if has_uv:    uv = struct.unpack_from('<2f', vdata, off + fo)
            return (pos, normal, uv)

        def read_vert_rel(i):
            """Read vertex by surface-relative index (fallback path)."""
            return read_vert_abs(surf['soffs'][0] // vs + i)

        # if an index buffer is present, expand through it (index order = draw order).
        # every surface in FO2 has an index buffer (nst == 2).
        if len(surf['sids']) >= 2:
            iid = surf['sids'][1]
            if iid in self.ibufs:
                ic, idata = self.ibufs[iid]
                ibase = surf['soffs'][1]
                indices = [struct.unpack_from('<H', idata, ibase + j * 2)[0]
                           for j in range(surf['niu'])]
                return [read_vert_abs(idx) for idx in indices]

        # fallback: no index buffer — iterate relative to surface start.
        return [read_vert_rel(i) for i in range(surf['vc'])]

    def get_raw_indices(self, surf):
        """Return the raw index list for this surface (absolute vertex indices, in draw order)."""
        if surf['nst'] < 1:
            return []
        vid = surf['sids'][0]
        if vid not in self.vbufs:
            return []
        _, vs, fl, _ = self.vbufs[vid]
        if len(surf['sids']) >= 2:
            iid = surf['sids'][1]
            if iid in self.ibufs:
                ic, idata = self.ibufs[iid]
                ibase = surf['soffs'][1]
                return [struct.unpack_from('<H', idata, ibase + j * 2)[0]
                        for j in range(surf['niu'])]
        # fallback: sequential from surface vertex range start
        ibase = surf['soffs'][0] // vs
        return list(range(ibase, ibase + surf['vc']))

    def expand_index_list(self, surf, idx_list):
        """Expand an arbitrary index sequence to vertex tuples using this surface's vbuf."""
        if surf['nst'] < 1 or not idx_list:
            return []
        vid = surf['sids'][0]
        if vid not in self.vbufs:
            return []
        vc, vs, fl, vdata = self.vbufs[vid]
        vtype    = self.get_vtype(surf)
        has_norm = bool(fl & 0x10)
        has_color= bool(fl & 0x40)
        has_uv   = bool(fl & 0x300)

        def read_abs(idx):
            off = idx * vs
            if off + vs > len(vdata):
                return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0))
            pos    = struct.unpack_from('<3f', vdata, off)
            normal = (0.0, 0.0, 0.0)
            uv     = (0.0, 0.0)
            if vtype != VTYPE_SHADOW:
                fo = 12
                if has_norm:  normal = struct.unpack_from('<3f', vdata, off + fo); fo += 12
                if has_color: fo += 4
                if has_uv:    uv = struct.unpack_from('<2f', vdata, off + fo)
            return (pos, normal, uv)

        return [read_abs(i) for i in idx_list]


# ── strip optimization ──

def _tris_from_strip(indices):
    """Extract canonical CCW triangles from a FO2 PC strip index sequence.

    FO2 PC strips are rendered by the GPU with automatic winding correction for
    odd-numbered triangles.  We undo that so every output tuple is CCW.
      even i: canonical = (v_i, v_{i+1}, v_{i+2})
      odd  i: canonical = (v_{i+1}, v_i, v_{i+2})   [first two swapped]
    Degenerate triangles (any two vertices identical) are silently skipped.
    """
    tris = []
    for i in range(len(indices) - 2):
        a, b, c = indices[i], indices[i+1], indices[i+2]
        if a == b or b == c or a == c:
            continue
        tris.append((a, b, c) if i % 2 == 0 else (b, a, c))
    return tris


def _tris_from_list(indices):
    """Extract canonical CCW triangles from a FO2 PC triangle-list index sequence."""
    tris = []
    for i in range(0, len(indices) - 2, 3):
        a, b, c = indices[i], indices[i+1], indices[i+2]
        if a != b and b != c and a != c:
            tris.append((a, b, c))
    return tris


def _nvts_stripify(tris, num_samples=10):
    """NvTriStrip-based triangle strip builder.

    Ported from NvTriStrip.py, itself ported from
    NVidia's NvTriStrip library via the RuneBlade Foundation.
    Original: http://developer.nvidia.com/view.asp?IO=nvtristrip_library

    Args:
        tris: list of (v0, v1, v2) tuples (canonical CCW winding)
        num_samples: experiment samples per iteration (higher = better quality)

    Returns: list of strips, each a list of vertex indices.
    """
    if not tris:
        return []

    # ── face ───
    class _Face:
        __slots__ = ('index', 'verts', '_adj')
        def __init__(self, index, v0, v1, v2):
            self.index = index
            self.verts = (v0, v1, v2)
            self._adj  = {v0: [], v1: [], v2: []}
        def get_next_vertex(self, v):
            vs = self.verts
            if v == vs[0]: return vs[1]
            if v == vs[1]: return vs[2]
            if v == vs[2]: return vs[0]
            return None
        def get_adjacent_faces(self, vi):
            return self._adj.get(vi, [])

    # ── mesh ──
    faces = []
    for v0, v1, v2 in tris:
        f = _Face(len(faces), v0, v1, v2)
        faces.append(f)

    edge_map = defaultdict(list)
    for face in faces:
        v0, v1, v2 = face.verts
        edge_map[(min(v1,v2), max(v1,v2))].append((face, v0))
        edge_map[(min(v0,v2), max(v0,v2))].append((face, v1))
        edge_map[(min(v0,v1), max(v0,v1))].append((face, v2))
    for _, face_list in edge_map.items():
        for i, (fi, oi) in enumerate(face_list):
            for j, (fj, _oj) in enumerate(face_list):
                if i != j:
                    fi._adj[oi].append(fj)

    def discard_face(face):
        for vi in face.verts:
            for adj in face._adj[vi]:
                for ovi in adj.verts:
                    try: adj._adj[ovi].remove(face)
                    except (ValueError, KeyError): pass
        for vi in face.verts:
            face._adj[vi] = []

    # ── TriangleStrip ──
    class _Strip:
        def __init__(self, stripped_faces=None):
            self.faces     = []
            self.vertices  = []
            self.reversed_ = False
            self.stripped_faces = stripped_faces if stripped_faces is not None else set()

        def _unstripped_adj(self, face, vi):
            for f in face.get_adjacent_faces(vi):
                if f.index not in self.stripped_faces:
                    return f
            return None

        def _traverse(self, start_vertex, start_face, forward):
            """Extend strip. pv1/pv2 computed from start_vertex+start_face (not passed in)."""
            count = 0
            pv0 = start_vertex
            pv1 = start_face.get_next_vertex(pv0)
            if pv1 is None: return 0
            pv2 = start_face.get_next_vertex(pv1)
            if pv2 is None: return 0
            next_face = self._unstripped_adj(start_face, pv0)
            while next_face:
                self.stripped_faces.add(next_face.index)
                count += 1
                if count & 1:
                    if forward:
                        pv0 = pv1
                        pv1 = next_face.get_next_vertex(pv0)
                        if pv1 is None: break
                        self.vertices.append(pv1)
                        self.faces.append(next_face)
                    else:
                        pv0 = pv2
                        pv2 = next_face.get_next_vertex(pv1)
                        if pv2 is None: break
                        self.vertices.insert(0, pv2)
                        self.faces.insert(0, next_face)
                        self.reversed_ = not self.reversed_
                else:
                    if forward:
                        pv0 = pv2
                        pv2 = next_face.get_next_vertex(pv1)
                        if pv2 is None: break
                        self.vertices.append(pv2)
                        self.faces.append(next_face)
                    else:
                        pv0 = pv1
                        pv1 = next_face.get_next_vertex(pv0)
                        if pv1 is None: break
                        self.vertices.insert(0, pv1)
                        self.faces.insert(0, next_face)
                        self.reversed_ = not self.reversed_
                next_face = self._unstripped_adj(next_face, pv0)
            return count

        def build(self, start_vertex, start_face):
            del self.faces[:]
            del self.vertices[:]
            self.reversed_ = False
            v0 = start_vertex
            v1 = start_face.get_next_vertex(v0)
            if v1 is None: return 0
            v2 = start_face.get_next_vertex(v1)
            if v2 is None: return 0
            self.stripped_faces.add(start_face.index)
            self.faces.append(start_face)
            self.vertices += [v0, v1, v2]
            self._traverse(v0, start_face, True)
            return self._traverse(v2, start_face, False)

        def get_strip(self):
            if self.reversed_:
                if len(self.vertices) & 1:
                    return list(reversed(self.vertices))
                elif len(self.vertices) == 4:
                    return [self.vertices[i] for i in (0, 2, 1, 3)]
                else:
                    s = list(self.vertices); s.insert(0, s[0]); return s
            return list(self.vertices)

    # ── Experiment ──
    class _Experiment:
        def __init__(self, start_vertex, start_face):
            self.stripped_faces = set()
            self.start_vertex   = start_vertex
            self.start_face     = start_face
            self.strips         = []

        def build(self):
            strip = _Strip(stripped_faces=self.stripped_faces)
            strip.build(self.start_vertex, self.start_face)
            self.strips.append(strip)
            nf = len(strip.faces)
            if nf >= 4:
                fi = nf >> 1
                self._build_adj(strip, fi)
                self._build_adj(strip, fi + 1)
            elif nf == 3:
                if not self._build_adj(strip, 0):
                    self._build_adj(strip, 2)
                self._build_adj(strip, 1)
            elif nf == 2:
                self._build_adj(strip, 0)
                self._build_adj(strip, 1)
            elif nf == 1:
                self._build_adj(strip, 0)

        def _build_adj(self, strip, face_index):
            if face_index >= len(strip.faces): return False
            opp_v = strip.vertices[face_index + 1]
            face  = strip.faces[face_index]
            other = next((f for f in face.get_adjacent_faces(opp_v)
                          if f.index not in self.stripped_faces), None)
            if other is None: return False
            winding = strip.reversed_
            if face_index & 1: winding = not winding
            other_v = (strip.vertices[face_index] if winding else
                       strip.vertices[face_index + 2]
                       if face_index + 2 < len(strip.vertices)
                       else strip.vertices[face_index])
            other_strip = _Strip(stripped_faces=self.stripped_faces)
            fi2 = other_strip.build(other_v, other)
            self.strips.append(other_strip)
            if fi2 > (len(other_strip.faces) >> 1):
                self._build_adj(other_strip, fi2 - 1)
            elif fi2 < len(other_strip.faces) - 1:
                self._build_adj(other_strip, fi2 + 1)
            return True

        @property
        def score(self):
            if not self.strips: return 0.0
            return sum(len(s.faces) for s in self.strips) / len(self.strips)

    # ── Main loop ──
    def det_sample(population, k):
        if not population or k <= 0: return []
        if k >= len(population): return list(population)
        return [population[int(i*(len(population)-1)/(k-1))] for i in range(k)]

    all_strips = []
    unstripped = set(range(len(faces)))

    while unstripped:
        samples = det_sample(list(unstripped), min(num_samples, len(unstripped)))
        best_score = -1.0
        best_exp   = None
        for idx in samples:
            sf = faces[idx]
            for sv in sf.verts:
                exp = _Experiment(sv, sf)
                exp.build()
                sc = exp.score
                if sc > best_score:
                    best_score = sc
                    best_exp   = exp
        if best_exp is None:
            break
        unstripped -= best_exp.stripped_faces
        for strip in best_exp.strips:
            for f in strip.faces:
                discard_face(f)
            all_strips.append(strip.get_strip())

    return all_strips


def _split_pc_strip(indices):
    """Split a FlatOut 2 PC triangle strip into clean sub-strips.

    FO2 PC strips join sub-strips with a degenerate run of the form:
      [..., A, A, B, B, B, C, ...]
    where A is the last vertex of the preceding sub-strip (repeated once)
    and B is the first vertex of the next sub-strip (repeated twice before
    the actual sub-strip content begins).

    This function removes all such join runs and returns the list of
    sub-strip index sequences (each at least 3 elements long).
    """
    subs = []
    n = len(indices)
    i = 0
    while i < n:
        start = i
        # advance j to the first position where indices[j] == indices[j+1]
        j = i
        while j < n - 1 and indices[j] != indices[j + 1]:
            j += 1
        sub = indices[start : j + 1]
        if len(sub) >= 3:
            subs.append(sub)
        if j >= n - 1:
            break
        # skip the A-run (repeated last vert of this sub-strip)
        A = indices[j]
        i = j
        while i < n and indices[i] == A:
            i += 1
        if i >= n:
            break
        # advance to the last B in the B-run (first vert of next sub-strip)
        B = indices[i]
        while i < n - 1 and indices[i + 1] == B:
            i += 1
        # i now points to the last B, which is the first vert of the next sub-strip
    return subs


def _join_strips(strips):
    """Concatenate strips with parity-aware degenerate joins.

    Uses a 2-vert join when the current accumulated length is even, or a 3-vert
    join when odd, so the first triangle of each sub-strip always falls at an
    even position in the combined strip and renders with correct CCW winding.
    """
    if not strips:
        return []
    result = list(strips[0])
    for s in strips[1:]:
        n = len(result)
        result.append(result[-1])      # always repeat last vert
        if n % 2 == 1:
            result.append(result[-1])  # extra repeat: 3-vert join to fix parity
        result.append(s[0])            # first vert of next strip
        result.extend(s)
    return result


def _strip_with_fallback(strips_or_subs, tris):
    """Join strips and append any uncovered input triangles as isolated fallback strips.

    Returns the final joined index list (always in strip form).
    """
    joined = _join_strips(strips_or_subs)

    covered = set()
    n = len(joined)
    for i in range(n - 2):
        a, b, c = joined[i], joined[i + 1], joined[i + 2]
        if a != b and b != c and a != c:
            covered.add(frozenset((a, b, c)))

    input_fs = {}
    for t in tris:
        fs = frozenset(t)
        if fs not in input_fs:
            input_fs[fs] = t

    missing = [t for fs, t in input_fs.items() if fs not in covered]
    if missing:
        joined = _join_strips(strips_or_subs + [[a, b, c] for a, b, c in missing])
    return joined


def optimise_strip(indices, poly_mode, num_samples=16):
    """Optimise a PC index sequence for PSP output.

    poly_mode 5 (PC triangle strip):
        Passed through unchanged.  The PC strip already contains well-optimised
        sub-strips; the degenerate join verts (A,A,B,B,B format) are valid PSP
        GE and produce zero-area triangles that are silently culled.
        Re-stripifying via NvTriStrip is counter-productive here: it loses the
        original sub-strip structure, creates more joins and a larger index list
        than the PC source.

    poly_mode 4 (PC triangle list):
        Runs the NvTriStrip algorithm to convert the triangle list into a
        compact triangle strip, then rejoins sub-strips with parity-aware
        2- or 3-vert degenerate bridges.  Any triangles not covered by the
        stripifier are appended as isolated 3-vert fallback strips.
        This matches Bugbear's behaviour: all original PSP BGM surfaces are
        TRI_STRIP, never TRI_LIST.

    num_samples: number of candidate starting faces tried per NvTriStrip
        iteration for PM=4 surfaces.  Higher values produce longer (better)
        strips at the cost of more CPU time during conversion.  Default 16.

    Returns: (index list, psp_poly_mode) where psp_poly_mode is 4 or 5
        (GE_PRIM_TRIANGLES or GE_PRIM_TRIANGLE_STRIP).
    """
    if poly_mode == 5:
        return indices, 5       # pass through — PC strip is already optimal
    else:
        tris = _tris_from_list(indices)
        if not tris:
            return indices, 4   # empty, keep as list
        strips = _nvts_stripify(tris, num_samples=num_samples)
        return _strip_with_fallback(strips, tris), 5





def encode_shadow_vertex(pos):
    """6 bytes: int16[3] position."""
    return struct.pack('<3h',
        clamp_i16(pos[0] * POS_SCALE),
        clamp_i16(pos[1] * POS_SCALE),
        clamp_i16(pos[2] * POS_SCALE))

def encode_standard_vertex(pos, normal, uv):
    """14 bytes: uint16[2] UV + int8[3] norm + int8 pad + int16[3] pos.

    GE vertex layout (from gevtx.h / libgu.h):
      SCEGU_TEXTURE_USHORT : 2 × unsigned short  (bits [1:0] = 2)
      SCEGU_NORMAL_BYTE    : 3 × signed   char   (bits [6:5] = 1)  + 1 byte pad
      SCEGU_VERTEX_SHORT   : 3 × signed   short  (bits [8:7] = 2)
    VTYPE = SCEGU_TEXTURE_USHORT | SCEGU_NORMAL_BYTE | SCEGU_VERTEX_SHORT = 0x122

    UV is encoded as: raw = round(float_uv * UV_SCALE) + UV_OFFSET = round(float_uv * 2048) + 16384
    The game engine applies the inverse transform (SU=8, TU=-3.5) to recover
    a game_uv = PC_UV * 0.5 + 0.5, sampling the texture in the [0.5, 1.0] range.
    """
    return struct.pack('<2H 3b b 3h',
        clamp_u16(round(uv[0] * UV_SCALE) + UV_OFFSET),  # unsigned short (SCEGU_TEXTURE_USHORT)
        clamp_u16(round(uv[1] * UV_SCALE) + UV_OFFSET),  # unsigned short
        clamp_i8 (normal[0] * NORM_SCALE),                # signed char    (SCEGU_NORMAL_BYTE)
        clamp_i8 (normal[1] * NORM_SCALE),
        clamp_i8 (normal[2] * NORM_SCALE),
        0,                                                  # pad byte
        clamp_i16(pos[0]    * POS_SCALE),                 # signed short   (SCEGU_VERTEX_SHORT)
        clamp_i16(pos[1]    * POS_SCALE),
        clamp_i16(pos[2]    * POS_SCALE))


# ── GE chunk builder ──

def build_ge_chunk(verts, vtype, poly_mode):
    """Build one complete PSP GE display list chunk for a single surface.

    Returns bytes: 20-byte header + encoded vertex data + padding to 16B alignment.
    """
    if vtype == VTYPE_SHADOW:
        vdata = b''.join(encode_shadow_vertex(pos) for pos, _n, _u in verts)
    else:
        vdata = b''.join(encode_standard_vertex(pos, normal, uv)
                         for pos, normal, uv in verts)

    nv        = len(verts)
    prim_type = GE_PRIM_STRIP if poly_mode == 5 else GE_PRIM_TRIANGLES

    header  = ge_cmd(0x14, 0)                            # ORIGIN()   SCE_GE_CMD_ORIGIN  — capture current address as origin
    header += ge_cmd(0x10, 0)                            # BASE(0)    SCE_GE_CMD_BASE    — upper address bits = 0
    header += ge_cmd(0x01, 0x14)                         # VADR(0x14) SCE_GE_CMD_VADR    — vertex data at origin+20 (right after this header)
    header += ge_cmd(0x04, (prim_type << 16) | nv)      # PRIM(type, count)  SCE_GE_CMD_PRIM  — draw primitive
    header += ge_cmd(0x0b, 0)                            # RET()      SCE_GE_CMD_RET     — return from display list

    chunk   = header + vdata
    pad_len = (16 - len(chunk) % 16) % 16
    return chunk + b'\x00' * pad_len


# ── material area computation ──

def compute_material_areas(pc):
    """Compute per-material 3D surface area and UV-space area from PC geometry.

    v109 = total 3D surface area of all surfaces for each material.
    v108 = sqrt(v109 * uv_area), a geometric mean of 3D and UV-space areas.
           For shadow surfaces (no UV), both area_uv and v108 are 0.

    Returns two dicts keyed by material index: {mid: area3d}, {mid: area_uv}.
    """
    import math
    from collections import defaultdict
    area3d = defaultdict(float)
    area_uv = defaultdict(float)

    for surf in pc.surfaces:
        vid = surf['sids'][0] if surf['sids'] else -1
        if vid not in pc.vbufs:
            continue
        vc, vs, fl, vdata = pc.vbufs[vid]
        is_shadow = fl == 0x0002 or vs == 12
        has_norm  = bool(fl & 0x10)
        has_color = bool(fl & 0x40)
        has_uv    = bool(fl & 0x300)

        def get_vert(idx, vdata=vdata, vs=vs, has_norm=has_norm,
                     has_color=has_color, has_uv=has_uv, is_shadow=is_shadow):
            off = idx * vs
            if off + vs > len(vdata):
                return (0.0, 0.0, 0.0), (0.0, 0.0)
            pos = struct.unpack_from('<3f', vdata, off)
            uv  = (0.0, 0.0)
            if not is_shadow and has_uv:
                fo = 12
                if has_norm:  fo += 12
                if has_color: fo += 4
                uv = struct.unpack_from('<2f', vdata, off + fo)
            return pos, uv

        # expand indices
        if len(surf['sids']) >= 2 and surf['sids'][1] in pc.ibufs:
            ic, idata = pc.ibufs[surf['sids'][1]]
            ibase  = surf['soffs'][1]
            indices = [struct.unpack_from('<H', idata, ibase + j * 2)[0]
                       for j in range(surf['niu'])]
        else:
            base_idx = surf['soffs'][0] // vs
            indices  = [base_idx + i for i in range(surf['vc'])]

        # build triangle list from indices
        if surf['pm'] == 4:
            tris = [(indices[i * 3], indices[i * 3 + 1], indices[i * 3 + 2])
                    for i in range(len(indices) // 3)]
        else:
            tris = []
            for i in range(len(indices) - 2):
                a, b, c = indices[i], indices[i + 1], indices[i + 2]
                if a != b and b != c and a != c:
                    tris.append((a, b, c))

        for a, b, c in tris:
            pa, ua = get_vert(a)
            pb, ub = get_vert(b)
            pc_, uc = get_vert(c)
            # 3D area
            ab3 = (pb[0]-pa[0], pb[1]-pa[1], pb[2]-pa[2])
            ac3 = (pc_[0]-pa[0], pc_[1]-pa[1], pc_[2]-pa[2])
            cx  = ab3[1]*ac3[2] - ab3[2]*ac3[1]
            cy  = ab3[2]*ac3[0] - ab3[0]*ac3[2]
            cz  = ab3[0]*ac3[1] - ab3[1]*ac3[0]
            area3d[surf['mid']] += math.sqrt(cx*cx + cy*cy + cz*cz) / 2
            # UV area
            if has_uv and not is_shadow:
                ab2 = (ub[0]-ua[0], ub[1]-ua[1])
                ac2 = (uc[0]-ua[0], uc[1]-ua[1])
                area_uv[surf['mid']] += abs(ab2[0]*ac2[1] - ab2[1]*ac2[0]) / 2

    return area3d, area_uv


# ── main converter ──

def convert_surface(pc, si, do_strip=False, num_samples=16):
    """Convert one PC surface to a PSP GE chunk.
    Returns (ge_chunk_bytes, nv).

    PM=4 (triangle list): always converted to TRI_STRIP via NvTriStrip.
        This matches Bugbear's behaviour and saves ~30 % vs passing the
        list through.  num_samples controls NvTriStrip quality.

    PM=5 (triangle strip): passed through unchanged as TRI_STRIP.
        The PC strip is already highly optimised; re-stripifying from scratch
        produces a larger result for this data.

    do_strip: when True, num_samples is raised to the caller-supplied value
        (default 16) for higher-quality PM=4 stripification.  When False,
        a baseline of 8 samples is used (fast, still clearly better than
        no conversion).

    Shadow surfaces: no conversion, always passed through.
    """
    surf  = pc.surfaces[si]
    vtype = pc.get_vtype(surf)

    if vtype != VTYPE_SHADOW and surf['pm'] == 4:
        # Always NvTriStrip for PM=4; do_strip raises the quality ceiling
        effective_samples = num_samples if do_strip else 8
        raw_idx = pc.get_raw_indices(surf)
        if raw_idx:
            opt_idx, psp_pm = optimise_strip(raw_idx, 4, effective_samples)
            verts = pc.expand_index_list(surf, opt_idx)
            chunk = build_ge_chunk(verts, vtype, psp_pm)
            return chunk, len(verts)

    verts = pc.get_vertices(surf)
    if not verts:
        chunk = build_ge_chunk([], vtype, surf['pm'])
        return chunk, 0
    chunk = build_ge_chunk(verts, vtype, surf['pm'])
    return chunk, len(verts)


def write_psp_bgm(pc, output_path, do_strip=False, num_samples=16):
    import math
    print(f"\nConverting to PSP{' (strip optimised)' if do_strip else ''}...")

    # ── compute per-material area data for v108/v109 ──
    area3d, area_uv = compute_material_areas(pc)

    # ── build per-surface GE chunks ──
    # We must place them in the blob grouped by material ID.
    # Strategy: compute chunks first, then sort surfaces by mid for blob ordering.

    chunks = []   # indexed by surface index
    nverts = []
    for si in range(len(pc.surfaces)):
        chunk, nv = convert_surface(pc, si, do_strip, num_samples)
        chunks.append(chunk)
        nverts.append(nv)
        surf  = pc.surfaces[si]
        vtype = pc.get_vtype(surf)
        pm_label = 'SHD' if vtype == VTYPE_SHADOW else f'pm{surf["pm"]}'
        print(f"  S{si:2d} m={surf['mid']:2d} {pm_label} "
              f"{surf['niu']:4d}idx -> {nv:4d}v  {len(chunk):5d}B")

    # sort surface indices by material ID, preserving original order within each material
    sorted_indices = sorted(range(len(pc.surfaces)), key=lambda si: pc.surfaces[si]['mid'])

    # build the blob and record byte offsets
    blob   = bytearray()
    boffs  = [0] * len(pc.surfaces)   # boff per surface index
    bszs   = [0] * len(pc.surfaces)   # bsz  per surface index
    for si in sorted_indices:
        boffs[si] = len(blob)
        bszs[si]  = len(chunks[si])
        blob      += chunks[si]

    total_v = sum(nverts)
    print(f"  Total: {total_v}v  blob={len(blob):,}B")

    # ── write file ──
    with open(output_path, 'wb') as f:

        # version
        f.write(struct.pack('<I', pc.version))

        # materials — v108/v109 computed from geometry; rest identical to FO2 PC
        f.write(struct.pack('<I', len(pc.materials)))
        for mi, m in enumerate(pc.materials):
            a9  = area3d.get(mi, 0.0)
            a8  = math.sqrt(a9 * area_uv.get(mi, 0.0)) if a9 > 0.0 else 0.0
            f.write(struct.pack('<I', m['ident']))
            write_string(f, m['name'])
            f.write(struct.pack('<i', m['alpha']))
            f.write(struct.pack('<5i', m['v92'], m['n_num_tex'], m['shader_id'],
                                       m['n_use_colormap'], m['v74']))
            f.write(struct.pack('<3f', a8, 0.0, 0.0))   # v108: (sqrt(a3d*auv), 0, 0)
            f.write(struct.pack('<3f', a9, a9,  a9))    # v109: (a3d, a3d, a3d)
            f.write(m['v98'])
            f.write(m['v99'])
            f.write(m['v100'])
            f.write(m['v101'])
            f.write(struct.pack('<i', m['v102']))
            for tn in m['tex_names']:
                write_string(f, tn)

        # streams — exactly 1, dt=3, fc=0, vc=blob_size, vs=1
        f.write(struct.pack('<I', 1))
        f.write(struct.pack('<I', 3))    # dt
        f.write(struct.pack('<3I', 0, len(blob), 1))   # fc=0, vc=len, vs=1
        f.write(blob)

        # surface records — 32 bytes each (8 x uint32)
        f.write(struct.pack('<I', len(pc.surfaces)))
        for si, surf in enumerate(pc.surfaces):
            f.write(struct.pack('<8I',
                0x00001000,       # [0] flags
                surf['mid'],      # [1] material index
                nverts[si],       # [2] vertex count
                1,                # [3] nb (always 1)
                0x00000027,       # [4] engine constant
                0,                # [5] always 0
                boffs[si],        # [6] blob offset
                bszs[si],         # [7] blob size
            ))

        # models — identical to FO2 PC
        f.write(struct.pack('<I', len(pc.models)))
        for m in pc.models:
            f.write(struct.pack('<I', m['ident']))
            f.write(struct.pack('<i', m['unk']))
            write_string(f, m['name'])
            f.write(struct.pack('<3f', *m['center']))
            f.write(struct.pack('<3f', *m['radius']))
            f.write(struct.pack('<f',  m['f_radius']))
            f.write(struct.pack('<I', len(m['surfaces'])))
            for s in m['surfaces']:
                f.write(struct.pack('<i', s))

        # meshes — identical to FO2 PC
        f.write(struct.pack('<I', len(pc.meshes)))
        for mesh in pc.meshes:
            f.write(struct.pack('<I', mesh['ident']))
            write_string(f, mesh['name1'])
            write_string(f, mesh['name2'])
            f.write(struct.pack('<I', mesh['flags']))
            f.write(struct.pack('<i', mesh['group']))
            f.write(struct.pack('<16f', *mesh['matrix']))
            f.write(struct.pack('<i', len(mesh['model_ids'])))
            for mid in mesh['model_ids']:
                f.write(struct.pack('<i', mid))

        # objects — identical to FO2 PC
        f.write(struct.pack('<I', len(pc.objects)))
        for obj in pc.objects:
            f.write(struct.pack('<I', obj['ident']))
            write_string(f, obj['name1'])
            write_string(f, obj['name2'])
            f.write(struct.pack('<I', obj['flags']))
            f.write(struct.pack('<16f', *obj['matrix']))

    isz = os.path.getsize(pc.path)
    osz = os.path.getsize(output_path)
    print(f"\n  {os.path.basename(pc.path)}: {isz:,}B  ->  "
          f"{os.path.basename(output_path)}: {osz:,}B  ({osz/isz*100:.1f}%)")


# ── main ──

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} [-strip] [-samples N] <input_fo2.bgm> [output_psp.bgm]")
        sys.exit(1)

    args = sys.argv[1:]
    do_strip = '-strip' in args
    args = [a for a in args if a != '-strip']

    num_samples = 16
    if '-samples' in args:
        idx = args.index('-samples')
        try:
            num_samples = int(args[idx + 1])
        except (IndexError, ValueError):
            print("Error: -samples requires an integer argument.")
            sys.exit(1)
        args = args[:idx] + args[idx + 2:]

    if not args:
        print(f"Usage: {sys.argv[0]} [-strip] [-samples N] <input_fo2.bgm> [output_psp.bgm]")
        sys.exit(1)

    inp = args[0]
    out = args[1] if len(args) >= 2 else \
          os.path.splitext(inp)[0] + "_psp" + os.path.splitext(inp)[1]

    pc = FO2Parser(inp)
    pc.parse()
    write_psp_bgm(pc, out, do_strip, num_samples)
    print("Done!")


if __name__ == '__main__':
    main()
