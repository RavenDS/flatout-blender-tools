#!/usr/bin/env python3
"""
FlatOut 2 BGM Converter: PC -> PS2 (v3)
by ravenDS (github.com/ravenDS)

v3 changelog:
  1. mode=5 strips preserved directly from PC data - NO re-stripification
  2. single-skip strip breaks within batches (2-skip only at batch start)
  3. zero padding between batches (pad only final blob to 16B alignment)
  4. better winding detection for mode=4 re-stripified data

ADC byte encoding (4th component of normal/vertex packed data):
  bit 7 = skip (1=skip, 0=draw)
  bit 5 = winding/parity (0=even, 1=odd)
  values: 0x80=-128 SKIP+even, 0xA0=-96 SKIP+odd, 0x00=0 DRAW+even, 0x20=32 DRAW+odd

VIF formats per vertex buffer type:
  flags 0x0112 (32B: pos+norm+uv):      CL=3 WL=1, V3-16@7 + V2-16@8 + V4-8@9
  flags 0x0152 (36B: pos+norm+color+uv): CL=3 WL=1, V4-16@7 + V2-16@8 + V4-8@9
  flags 0x0002 (12B: pos only, shadow):  CL=1 WL=1, V4-16@5
"""

import struct, sys, os, math
from dataclasses import dataclass, field
from typing import List, Tuple
from collections import defaultdict

POS_SCALE = 1024.0; UV_SCALE = 4096.0; NORM_SCALE = 127.0
MAX_BATCH_V3 = 77; MAX_BATCH_V3_ALPHA = 34; MAX_BATCH_V4 = 22; MAX_BATCH_SHADOW = 155

# per-shader STD batch limits from PS2
# opaque: shader 8 (common) proven max=55; all others use 77.
# alpha: shader 6 (windows) proven max=34; shaders 11,12 proven max=32.
SHADER_STD_OPAQUE_LIMIT = { 8: 55 }
SHADER_STD_ALPHA_LIMIT  = { 11: 32, 12: 32 }  # shader 6 uses default 34
VIF_STCYCL = 0x01; VIF_ITOP = 0x04; VIF_MSCNT = 0x17
VIF_UNPACK_V3_16 = 0x69; VIF_UNPACK_V4_16 = 0x6D
VIF_UNPACK_V2_16 = 0x65; VIF_UNPACK_V4_8 = 0x6E
VTYPE_STANDARD = 0; VTYPE_COLORED = 1; VTYPE_SHADOW = 2

@dataclass
class Material:
    identifier: int = 0x4354414D; name: str = ""; alpha: int = 0
    v92: int = 0; num_textures: int = 0; shader_id: int = 0
    use_colormap: int = 0; v74: int = 0
    v108: bytes = b'\x00'*12; v109: bytes = b'\x00'*12
    v98: bytes = b'\x00'*16; v99: bytes = b'\x00'*16
    v100: bytes = b'\x00'*16; v101: bytes = b'\x00'*16
    v102: int = 0; tex_names: list = field(default_factory=lambda: ["","",""])

@dataclass
class Vertex:
    pos: Tuple[float,float,float] = (0,0,0)
    normal: Tuple[float,float,float] = (0,0,0)
    uv: Tuple[float,float] = (0,0)
    color: Tuple[int,int,int,int] = (255,255,255,255)

@dataclass
class PCSurface:
    is_vegetation: int = 0; material_id: int = 0; vertex_count: int = 0
    flags: int = 0; poly_count: int = 0; poly_mode: int = 0
    num_indices_used: int = 0
    center: Tuple[float,float,float] = (0,0,0)
    radius: Tuple[float,float,float] = (0,0,0)
    num_streams: int = 0
    stream_ids: list = field(default_factory=list)
    stream_offsets: list = field(default_factory=list)

@dataclass
class Model:
    identifier: int = 0x444F4D42; unk: int = 0; name: str = ""
    center: Tuple[float,float,float] = (0,0,0)
    radius: Tuple[float,float,float] = (0,0,0)
    f_radius: float = 0; surfaces: list = field(default_factory=list)

@dataclass
class CompactMesh:
    identifier: int = 0x4853454D; name1: str = ""; name2: str = ""
    flags: int = 0; group: int = -1; matrix: tuple = tuple([0.0]*16)
    models: list = field(default_factory=list)

@dataclass
class Object:
    identifier: int = 0x434A424F; name1: str = ""; name2: str = ""
    flags: int = 0; matrix: tuple = tuple([0.0]*16)

def read_string(f):
    s = b""
    while True:
        c = f.read(1)
        if not c or c == b'\x00': break
        s += c
    return s.decode('ascii', errors='replace')

def write_string(f, s):
    f.write(s.encode('ascii') + b'\x00')

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def get_vbuf_type(flags, vs):
    if flags == 0x0002 or vs == 12: return VTYPE_SHADOW
    if flags & 0x0040: return VTYPE_COLORED
    return VTYPE_STANDARD


class PCBGMParser:
    def __init__(self, path):
        self.path = path; self.version = 0
        self.materials = []; self.vertex_buffers = {}
        self.index_buffers = {}; self.surfaces = []
        self.models = []; self.meshes = []; self.objects = []

    def parse(self):
        with open(self.path, 'rb') as f:
            self.version = struct.unpack('<I', f.read(4))[0]
            self._parse_materials(f); self._parse_streams(f)
            self._parse_surfaces(f); self._parse_models(f)
            self._parse_bgm_meshes(f); self._parse_objects(f)

    def _parse_materials(self, f):
        for _ in range(struct.unpack('<I', f.read(4))[0]):
            m = Material(); m.identifier = struct.unpack('<I', f.read(4))[0]
            m.name = read_string(f); m.alpha = struct.unpack('<i', f.read(4))[0]
            if self.version >= 0x10004:
                m.v92, m.num_textures, m.shader_id, m.use_colormap, m.v74 = struct.unpack('<5i', f.read(20))
                m.v108 = f.read(12); m.v109 = f.read(12)
            m.v98 = f.read(16); m.v99 = f.read(16); m.v100 = f.read(16); m.v101 = f.read(16)
            m.v102 = struct.unpack('<i', f.read(4))[0]
            m.tex_names = [read_string(f) for _ in range(3)]
            self.materials.append(m)

    def _parse_streams(self, f):
        for i in range(struct.unpack('<I', f.read(4))[0]):
            dt = struct.unpack('<I', f.read(4))[0]
            if dt == 1:
                fc, vc, vs, fl = struct.unpack('<4I', f.read(16))
                self.vertex_buffers[i] = (vc, vs, fl, f.read(vc*vs))
            elif dt == 2:
                fc, ic = struct.unpack('<2I', f.read(8))
                self.index_buffers[i] = (ic, f.read(ic*2))
            elif dt == 3:
                fc, vc, vs = struct.unpack('<3I', f.read(12))
                self.vertex_buffers[i] = (vc, vs, 0, f.read(vc*vs))

    def _parse_surfaces(self, f):
        for _ in range(struct.unpack('<I', f.read(4))[0]):
            s = PCSurface()
            s.is_vegetation, s.material_id, s.vertex_count, s.flags, \
                s.poly_count, s.poly_mode, s.num_indices_used = struct.unpack('<7i', f.read(28))
            if self.version < 0x20000:
                s.center = struct.unpack('<3f', f.read(12))
                s.radius = struct.unpack('<3f', f.read(12))
            s.num_streams = struct.unpack('<i', f.read(4))[0]
            for _ in range(s.num_streams):
                sid, soff = struct.unpack('<2I', f.read(8))
                s.stream_ids.append(sid); s.stream_offsets.append(soff)
            self.surfaces.append(s)

    def _parse_models(self, f):
        for _ in range(struct.unpack('<I', f.read(4))[0]):
            m = Model(); m.identifier = struct.unpack('<I', f.read(4))[0]
            m.unk = struct.unpack('<i', f.read(4))[0]; m.name = read_string(f)
            m.center = struct.unpack('<3f', f.read(12))
            m.radius = struct.unpack('<3f', f.read(12))
            m.f_radius = struct.unpack('<f', f.read(4))[0]
            ns = struct.unpack('<I', f.read(4))[0]
            m.surfaces = [struct.unpack('<i', f.read(4))[0] for _ in range(ns)]
            self.models.append(m)

    def _parse_bgm_meshes(self, f):
        for _ in range(struct.unpack('<I', f.read(4))[0]):
            m = CompactMesh(); m.identifier = struct.unpack('<I', f.read(4))[0]
            m.name1 = read_string(f); m.name2 = read_string(f)
            m.flags = struct.unpack('<I', f.read(4))[0]
            m.group = struct.unpack('<i', f.read(4))[0]
            m.matrix = struct.unpack('<16f', f.read(64))
            nm = struct.unpack('<i', f.read(4))[0]
            m.models = [struct.unpack('<i', f.read(4))[0] for _ in range(nm)]
            self.meshes.append(m)

    def _parse_objects(self, f):
        for _ in range(struct.unpack('<I', f.read(4))[0]):
            o = Object(); o.identifier = struct.unpack('<I', f.read(4))[0]
            o.name1 = read_string(f); o.name2 = read_string(f)
            o.flags = struct.unpack('<I', f.read(4))[0]
            o.matrix = struct.unpack('<16f', f.read(64))
            self.objects.append(o)

    def get_surface_vtype(self, surf):
        if surf.num_streams < 1: return VTYPE_STANDARD
        vid = surf.stream_ids[0]
        if vid not in self.vertex_buffers: return VTYPE_STANDARD
        _, vs, fl, _ = self.vertex_buffers[vid]
        return get_vbuf_type(fl, vs)

    def get_surface_vertices(self, surf):
        if surf.num_streams < 1: return []
        vid = surf.stream_ids[0]
        if vid not in self.vertex_buffers: return []
        vc, vs, fl, data = self.vertex_buffers[vid]
        verts = []
        hn = bool(fl & 0x10); hc = bool(fl & 0x40); hu = bool(fl & 0x300)
        for i in range(surf.vertex_count):
            off = surf.stream_offsets[0] + i * vs
            if off + vs > len(data): break
            v = Vertex(); v.pos = struct.unpack_from('<3f', data, off); fo = 12
            if hn: v.normal = struct.unpack_from('<3f', data, off+fo); fo += 12
            if hc: v.color = struct.unpack_from('<4B', data, off+fo); fo += 4
            if hu: v.uv = struct.unpack_from('<2f', data, off+fo)
            verts.append(v)
        return verts

    def get_surface_indices(self, surf):
        if surf.num_streams < 2: return []
        iid = surf.stream_ids[1]
        if iid not in self.index_buffers: return []
        ic, data = self.index_buffers[iid]
        return [struct.unpack_from('<H', data, surf.stream_offsets[1]+i*2)[0]
                for i in range(surf.num_indices_used)
                if surf.stream_offsets[1]+i*2+2 <= len(data)]



# strip extraction (for mode=5 PC data)
def extract_strips_from_pc_strip(indices):
    """Extract sub-strips from PC mode=5 strip (split at degenerate triangles).
    Returns list of (strip_vertex_indices, winding_offset) tuples.
    winding_offset: 0 = first triangle uses even winding, 1 = odd winding.
    
    In a standard triangle strip, triangle i uses vertices (i, i+1, i+2).
    If i is even → even winding (v0,v1,v2), if i is odd → odd winding (v1,v0,v2).
    
    Key insight: when we find a degenerate at position i, the last VALID triangle
    was at position i-1, which uses vertices[i-1], vertices[i], vertices[i+1].
    So the sub-strip must include up to index i+1 inclusive = indices[start : i+2].
    """
    if len(indices) < 3:
        return []
    
    strips = []
    i = 0
    strip_start = 0
    
    while i + 2 < len(indices):
        a, b, c = indices[i], indices[i+1], indices[i+2]
        is_degen = (a == b or b == c or a == c)
        
        if is_degen:
            # end current strip: include vertices up to i+1 (last valid tri uses i-1,i,i+1)
            end = i + 2  # exclusive: indices[strip_start:i+2] gives vertices through index i+1
            if end - strip_start >= 3:
                sub = indices[strip_start:end]
                strips.append((sub, strip_start % 2))
            # advance past this and any subsequent degenerates
            i += 1
            while i + 2 < len(indices):
                a2, b2, c2 = indices[i], indices[i+1], indices[i+2]
                if a2 == b2 or b2 == c2 or a2 == c2:
                    i += 1
                else:
                    break
            strip_start = i
        else:
            i += 1
    
    # Final strip
    if len(indices) - strip_start >= 3:
        sub = indices[strip_start:len(indices)]
        strips.append((sub, strip_start % 2))
    
    return strips


# stripification for triangle lists (mode=4)
def stripify_triangles(triangles):
    """Greedy stripifier for mode=4 triangle lists."""
    if not triangles: return []
    edge_tris = defaultdict(list)
    for ti, (a, b, c) in enumerate(triangles):
        for e in [(a,b),(b,c),(c,a)]:
            edge_tris[(min(e),max(e))].append(ti)
    tri_adj = defaultdict(set)
    for tris in edge_tris.values():
        for i in range(len(tris)):
            for j in range(i+1, len(tris)):
                tri_adj[tris[i]].add(tris[j]); tri_adj[tris[j]].add(tris[i])
    used = [False]*len(triangles)
    strips = []
    order = sorted(range(len(triangles)), key=lambda t: len(tri_adj[t]))

    def find_next(e0, e1):
        edge = (min(e0,e1), max(e0,e1))
        best, best_d = None, 999
        for ti in edge_tris[edge]:
            if used[ti]: continue
            d = sum(1 for n in tri_adj[ti] if not used[n])
            if d < best_d: best, best_d = ti, d
        return best

    for start in order:
        if used[start]: continue
        tri = triangles[start]; best_strip = best_local = None
        for rot in range(3):
            a, b, c = tri[rot], tri[(rot+1)%3], tri[(rot+2)%3]
            save = list(used); used[start] = True; local = {start}
            strip = [a, b, c]
            while True:
                ti = find_next(strip[-2], strip[-1])
                if ti is None: break
                third = None
                for v in triangles[ti]:
                    if v != strip[-2] and v != strip[-1]: third = v; break
                if third is None: break
                used[ti] = True; local.add(ti); strip.append(third)
            while True:
                ti = find_next(strip[1], strip[0])
                if ti is None: break
                third = None
                for v in triangles[ti]:
                    if v != strip[1] and v != strip[0]: third = v; break
                if third is None: break
                used[ti] = True; local.add(ti); strip.insert(0, third)
            if best_strip is None or len(strip) > len(best_strip):
                best_strip = strip; best_local = local.copy()
            for i, u in enumerate(save): used[i] = u
        if best_strip:
            for ti in best_local: used[ti] = True
            strips.append(best_strip)
    return strips


def detect_winding_for_strip(strip, vertices, orig_normals):
    """Detect correct winding offset (0 or 1) for a stripified strip by
    checking the first triangle's face normal against the original triangles."""
    if len(strip) < 3:
        return 0
    for w in [0, 1]:
        if w == 0:
            t = (strip[0], strip[1], strip[2])
        else:
            t = (strip[1], strip[0], strip[2])
        key = frozenset(t)
        if key in orig_normals and all(0 <= x < len(vertices) for x in t):
            p0, p1, p2 = vertices[t[0]].pos, vertices[t[1]].pos, vertices[t[2]].pos
            e1 = (p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2])
            e2 = (p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2])
            sn = (e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0])
            dot = sum(a*b for a,b in zip(sn, orig_normals[key]))
            if dot > 0:
                return w
    return 0


# ADC strip Packing (v3: single-skip within batches)
def pack_strips_with_adc(strips_with_winding, max_batch):
    """Pack strips into batches with PS2-compatible ADC flags.

    Uses 2-skip for ALL strip breaks (both at batch start and within batch).
    This is safe because it ensures the first drawn triangle uses only vertices
    from the new strip, avoiding spurious triangles.

    The PS2 retail data uses a mix of 1-skip and 2-skip, but 1-skip requires
    consecutive strips to share an edge, which our re-stripifier doesn't guarantee.

    ADC byte encoding:
      bit 7 = skip (1=skip, 0=draw)
      bit 5 = winding/parity (0=even, 1=odd)

    Each strip = (vertex_indices, winding_offset).
    Returns list of batches. Each batch = list of (vertex_index, adc_byte).
    """
    batches: List[List[tuple]] = []
    current: List[tuple] = []

    for strip_idx, (strip, winding) in enumerate(strips_with_winding):
        if len(strip) < 3:
            continue

        # split strip if it exceeds max_batch (overlap by 2 for continuity)
        sub_strips = []
        if len(strip) > max_batch:
            i = 0
            while i < len(strip):
                end = min(i + max_batch, len(strip))
                if end - i < 3:
                    break
                sub = strip[i:end]
                w = (winding + i) % 2 if i > 0 else winding
                sub_strips.append((sub, w))
                i = end - 2 if end < len(strip) else end
        else:
            sub_strips = [(strip, winding)]

        for ss, sw in sub_strips:
            # check if strip fits in current batch (2-skip needs len(ss) vertices)
            if current and len(current) + len(ss) > max_batch:
                # Flush current batch
                batches.append(current)
                current = []

            # always use 2-skip for strip breaks
            skip_adc = 0x80 | (0x20 if sw else 0x00)  # SKIP + winding
            skip_adc_signed = skip_adc if skip_adc < 128 else skip_adc - 256

            for i, vidx in enumerate(ss):
                if i < 2:
                    adc = skip_adc_signed
                else:
                    # Draw vertices: winding alternates from position 2 onwards
                    tri_pos = (i - 2 + sw) % 2
                    adc = 32 if tri_pos else 0
                current.append((vidx, adc))

    if current:
        batches.append(current)
    return batches


# VIF batch builders (v3: no per-batch padding)
def build_vif_standard(batch, vertices, first):
    """Build VIF data for standard vertex batch (V3-16 pos + V2-16 uv + V4-8 norm+adc)."""
    n = len(batch)
    d = bytearray()
    if first:
        d += struct.pack('<I', (VIF_STCYCL << 24) | (1 << 8) | 3)
    d += struct.pack('<I', (VIF_UNPACK_V3_16 << 24) | (n << 16) | 0x8007)
    for vidx, _ in batch:
        v = vertices[vidx]
        d += struct.pack('<3h', *(clamp(int(round(c * POS_SCALE)), -32768, 32767) for c in v.pos))
    while len(d) % 4:
        d += b'\x00'
    d += struct.pack('<I', (VIF_UNPACK_V2_16 << 24) | (n << 16) | 0x8008)
    for vidx, _ in batch:
        v = vertices[vidx]
        d += struct.pack('<2h', *(clamp(int(round(c * UV_SCALE)), -32768, 32767) for c in v.uv))
    d += struct.pack('<I', (VIF_UNPACK_V4_8 << 24) | (n << 16) | 0x8009)
    for vidx, adc in batch:
        v = vertices[vidx]
        d += struct.pack('<4b',
            *(clamp(int(round(c * NORM_SCALE)), -127, 127) for c in v.normal),
            adc)
    d += struct.pack('<I', (VIF_ITOP << 24) | n)
    d += struct.pack('<I', VIF_MSCNT << 24)
    # NO padding here, batches are packed tightly (matching PS2 original)
    return bytes(d)

def build_vif_colored(batch, vertices, first):
    """Build VIF data for colored vertex batch (V4-16 pos + V2-16 uv + V4-8 norm+adc)."""
    n = len(batch)
    d = bytearray()
    if first:
        d += struct.pack('<I', (VIF_STCYCL << 24) | (1 << 8) | 3)
    d += struct.pack('<I', (VIF_UNPACK_V4_16 << 24) | (n << 16) | 0x8007)
    for vidx, _ in batch:
        v = vertices[vidx]
        d += struct.pack('<4h', *(clamp(int(round(c * POS_SCALE)), -32768, 32767) for c in v.pos), 0)
    d += struct.pack('<I', (VIF_UNPACK_V2_16 << 24) | (n << 16) | 0x8008)
    for vidx, _ in batch:
        v = vertices[vidx]
        d += struct.pack('<2h', *(clamp(int(round(c * UV_SCALE)), -32768, 32767) for c in v.uv))
    d += struct.pack('<I', (VIF_UNPACK_V4_8 << 24) | (n << 16) | 0x8009)
    for vidx, adc in batch:
        v = vertices[vidx]
        d += struct.pack('<4b',
            *(clamp(int(round(c * NORM_SCALE)), -127, 127) for c in v.normal),
            adc)
    d += struct.pack('<I', (VIF_ITOP << 24) | n)
    d += struct.pack('<I', VIF_MSCNT << 24)
    return bytes(d)

def build_vif_shadow(batch, vertices, first):
    """Build VIF data for shadow vertex batch (V4-16 pos+adc)."""
    n = len(batch)
    d = bytearray()
    if first:
        d += struct.pack('<I', (VIF_STCYCL << 24) | (1 << 8) | 1)
    d += struct.pack('<I', (VIF_UNPACK_V4_16 << 24) | (n << 16) | 0x8005)
    for vidx, adc in batch:
        v = vertices[vidx]
        d += struct.pack('<4h',
            *(clamp(int(round(c * POS_SCALE)), -32768, 32767) for c in v.pos),
            adc)
    d += struct.pack('<I', (VIF_ITOP << 24) | n)
    d += struct.pack('<I', VIF_MSCNT << 24)
    return bytes(d)


def pad_blob_to_16(data):
    """Pad a complete VIF blob (all batches concatenated) to 16-byte alignment.
    Only pads at the END, matching PS2 original behavior."""
    while len(data) % 16:
        data += b'\x00\x00\x00\x00'  # NOP words
    return data


# material v108/v109
def compute_material_areas(pc):
    nm = len(pc.materials)
    m3d = [0.0]*nm; m108v = [0.0]*nm
    for surf in pc.surfaces:
        mid = surf.material_id
        vtype = pc.get_surface_vtype(surf)
        vid = surf.stream_ids[0] if surf.num_streams >= 1 else -1
        if vid < 0 or vid not in pc.vertex_buffers: continue
        _, vs, fl, data = pc.vertex_buffers[vid]
        base = surf.stream_offsets[0] // vs if vs > 0 else 0
        indices = pc.get_surface_indices(surf)
        if not indices: continue
        indices = [idx - base for idx in indices]
        tris = []
        if surf.poly_mode == 5:
            for i in range(len(indices)-2):
                a,b,c = indices[i],indices[i+1],indices[i+2]
                if a==b or b==c or a==c: continue
                tris.append((a,b,c) if i%2==0 else (b,a,c))
        else:
            for i in range(0, len(indices)-2, 3):
                tris.append((indices[i],indices[i+1],indices[i+2]))
        if vtype == VTYPE_SHADOW:
            verts = []
            for i in range(surf.vertex_count):
                off = surf.stream_offsets[0] + i*vs
                if off+vs > len(data): break
                verts.append(struct.unpack_from('<3f', data, off))
            for a,b,c in tris:
                if not all(0<=x<len(verts) for x in (a,b,c)): continue
                pa,pb,pc_=verts[a],verts[b],verts[c]
                e1=(pb[0]-pa[0],pb[1]-pa[1],pb[2]-pa[2])
                e2=(pc_[0]-pa[0],pc_[1]-pa[1],pc_[2]-pa[2])
                cx=e1[1]*e2[2]-e1[2]*e2[1]; cy=e1[2]*e2[0]-e1[0]*e2[2]; cz=e1[0]*e2[1]-e1[1]*e2[0]
                m3d[mid] += 0.5*math.sqrt(cx*cx+cy*cy+cz*cz)
        else:
            verts = pc.get_surface_vertices(surf)
            if not verts: continue
            for a,b,c in tris:
                if not all(0<=x<len(verts) for x in (a,b,c)): continue
                va,vb,vc_=verts[a],verts[b],verts[c]
                e1=(vb.pos[0]-va.pos[0],vb.pos[1]-va.pos[1],vb.pos[2]-va.pos[2])
                e2=(vc_.pos[0]-va.pos[0],vc_.pos[1]-va.pos[1],vc_.pos[2]-va.pos[2])
                cx=e1[1]*e2[2]-e1[2]*e2[1]; cy=e1[2]*e2[0]-e1[0]*e2[2]; cz=e1[0]*e2[1]-e1[1]*e2[0]
                t3d = 0.5*math.sqrt(cx*cx+cy*cy+cz*cz)
                u1=vb.uv[0]-va.uv[0]; v1=vb.uv[1]-va.uv[1]
                u2=vc_.uv[0]-va.uv[0]; v2=vc_.uv[1]-va.uv[1]
                tuv = 0.5*abs(u1*v2 - u2*v1)
                m3d[mid] += t3d
                if t3d > 0 and tuv > 0: m108v[mid] += math.sqrt(t3d*tuv)
    for i in range(nm):
        pc.materials[i].v109 = struct.pack('<3f', m3d[i], m3d[i], m3d[i])
        pc.materials[i].v108 = struct.pack('<3f', m108v[i], 0.0, 0.0)


# surface converter (v3)
def convert_surface(pc, si):
    surf = pc.surfaces[si]
    vtype = pc.get_surface_vtype(surf)
    vid = surf.stream_ids[0] if surf.num_streams >= 1 else -1
    vs = 0
    if vid >= 0 and vid in pc.vertex_buffers:
        _, vs, _, _ = pc.vertex_buffers[vid]

    if vtype == VTYPE_SHADOW:
        vertices = []
        if vid >= 0:
            vc2, vs2, _, dat2 = pc.vertex_buffers[vid]
            for i in range(surf.vertex_count):
                off = surf.stream_offsets[0] + i * vs2
                if off + vs2 > len(dat2): break
                v = Vertex()
                v.pos = struct.unpack_from('<3f', dat2, off)
                vertices.append(v)
    else:
        vertices = pc.get_surface_vertices(surf)

    if not vertices:
        return b'', 0, 0, []

    indices = pc.get_surface_indices(surf)
    base = surf.stream_offsets[0] // vs if vs > 0 else 0
    if indices and base > 0:
        indices = [idx - base for idx in indices]
    if not indices:
        indices = list(range(len(vertices)))
    indices = [idx for idx in indices if 0 <= idx < len(vertices)]

    if not indices or len(indices) < 3:
        return b'', 0, 0, []

    # mode=5 extracts triangles then re-strips,
    # mode=4 directly strips triangle list
    if surf.poly_mode == 5:
        # mode 5: triangle strip, extract triangles, then re-stripify
        # the PC uses degenerate triangles for strip breaks which don't map
        # cleanly to PS2 ADC flags, so we re-strip for optimal PS2 layout
        tris = []
        for i in range(len(indices) - 2):
            a, b, c = indices[i], indices[i+1], indices[i+2]
            if a == b or b == c or a == c:
                continue
            if all(0 <= x < len(vertices) for x in (a, b, c)):
                if i % 2 == 0:
                    tris.append((a, b, c))
                else:
                    tris.append((b, a, c))
        if not tris:
            return b'', 0, 0, []

        raw_strips = stripify_triangles(tris)

        orig_normals = {}
        for a, b, c in tris:
            if all(0 <= x < len(vertices) for x in (a, b, c)):
                p0, p1, p2 = vertices[a].pos, vertices[b].pos, vertices[c].pos
                e1 = (p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2])
                e2 = (p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2])
                n = (e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0])
                orig_normals[frozenset([a, b, c])] = n

        strips_with_winding = []
        for strip in raw_strips:
            w = detect_winding_for_strip(strip, vertices, orig_normals)
            strips_with_winding.append((strip, w))

    elif surf.poly_mode == 4:
        # mode 4: triangle list,need to stripify
        tris = [(indices[i], indices[i+1], indices[i+2])
                for i in range(0, len(indices) - 2, 3)
                if all(0 <= indices[j] < len(vertices) for j in (i, i+1, i+2))]
        if not tris:
            return b'', 0, 0, []

        raw_strips = stripify_triangles(tris)

        # detect correct winding per strip
        orig_normals = {}
        for a, b, c in tris:
            if all(0 <= x < len(vertices) for x in (a, b, c)):
                p0, p1, p2 = vertices[a].pos, vertices[b].pos, vertices[c].pos
                e1 = (p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2])
                e2 = (p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2])
                n = (e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0])
                orig_normals[frozenset([a, b, c])] = n

        strips_with_winding = []
        for strip in raw_strips:
            w = detect_winding_for_strip(strip, vertices, orig_normals)
            strips_with_winding.append((strip, w))
    else:
        return b'', 0, 0, []

    if not strips_with_winding:
        return b'', 0, 0, []

    # per-shader VU1 vertex buffer limits
    # different VU1 microprograms have different buffer sizes. 
    # shader 8 (common) proven max=55 across most PS2 files.
    mat = pc.materials[surf.material_id]
    is_alpha = (mat.alpha == 1)
    if vtype == VTYPE_STANDARD:
        if is_alpha:
            max_batch = SHADER_STD_ALPHA_LIMIT.get(mat.shader_id, MAX_BATCH_V3_ALPHA)
        else:
            max_batch = SHADER_STD_OPAQUE_LIMIT.get(mat.shader_id, MAX_BATCH_V3)
    elif vtype == VTYPE_COLORED:
        max_batch = MAX_BATCH_V4
    else:
        max_batch = MAX_BATCH_SHADOW
    batches = pack_strips_with_adc(strips_with_winding, max_batch)
    if not batches:
        return b'', 0, 0, []

    builders = [build_vif_standard, build_vif_colored, build_vif_shadow]
    builder = builders[vtype]

    # build all batches WITHOUT inter-batch padding (matching PS2 original)
    vif = bytearray()
    total_verts = 0
    for bi, batch in enumerate(batches):
        vif += builder(batch, vertices, bi == 0)
        total_verts += len(batch)

    # pad only the final blob to 16-byte alignment
    vif = pad_blob_to_16(vif)

    return bytes(vif), total_verts, len(batches), batches


# writer
def write_ps2_bgm(pc, output_path):
    print(f"\nConverting to PS2...")
    compute_material_areas(pc)

    sinfo = []; vif_blob = bytearray(); all_batches = []
    for si in range(len(pc.surfaces)):
        vd, nv, nb, batches = convert_surface(pc, si)
        sinfo.append((nv, nb, len(vif_blob), len(vd)))
        all_batches.append(batches)
        vif_blob += vd
        s = pc.surfaces[si]; vt = pc.get_surface_vtype(s)
        print(f"  S{si:2d} m={s.material_id:2d} {['STD','CLR','SHD'][vt]} "
              f"{s.vertex_count:4d}->{nv:4d}v b={nb:2d} {len(vd):5d}B")

    print(f"  Total: {sum(x[0] for x in sinfo)}v {sum(x[1] for x in sinfo)}bat {len(vif_blob)}B")

    with open(output_path, 'wb') as f:
        f.write(struct.pack('<I', pc.version))
        f.write(struct.pack('<I', len(pc.materials)))
        for mat in pc.materials:
            f.write(struct.pack('<I', mat.identifier)); write_string(f, mat.name)
            f.write(struct.pack('<i', mat.alpha))
            if pc.version >= 0x10004:
                f.write(struct.pack('<5i', mat.v92, mat.num_textures, mat.shader_id, mat.use_colormap, mat.v74))
                f.write(mat.v108); f.write(mat.v109)
            f.write(mat.v98); f.write(mat.v99); f.write(mat.v100); f.write(mat.v101)
            f.write(struct.pack('<i', mat.v102))
            for tn in mat.tex_names: write_string(f, tn)

        f.write(struct.pack('<I', 1))
        f.write(struct.pack('<I', 3))
        f.write(struct.pack('<3I', 0, len(vif_blob), 1))
        f.write(vif_blob)

        f.write(struct.pack('<I', len(pc.surfaces)))
        for si, surf in enumerate(pc.surfaces):
            nv, nb, bo, bs = sinfo[si]
            f.write(struct.pack('<10I', 0, 0x1000, surf.material_id,
                                nv, nb, 1, 0x0E, 0, bo, bs))

        f.write(struct.pack('<I', len(pc.models)))
        for m in pc.models:
            f.write(struct.pack('<Ii', m.identifier, m.unk)); write_string(f, m.name)
            f.write(struct.pack('<3f', *m.center)); f.write(struct.pack('<3f', *m.radius))
            f.write(struct.pack('<f', m.f_radius))
            f.write(struct.pack('<I', len(m.surfaces)))
            for s in m.surfaces: f.write(struct.pack('<i', s))

        f.write(struct.pack('<I', len(pc.meshes)))
        for mesh in pc.meshes:
            f.write(struct.pack('<I', mesh.identifier))
            write_string(f, mesh.name1); write_string(f, mesh.name2)
            f.write(struct.pack('<Ii', mesh.flags, mesh.group))
            f.write(struct.pack('<16f', *mesh.matrix))
            f.write(struct.pack('<i', len(mesh.models)))
            for m in mesh.models: f.write(struct.pack('<i', m))

        f.write(struct.pack('<I', len(pc.objects)))
        for obj in pc.objects:
            f.write(struct.pack('<I', obj.identifier))
            write_string(f, obj.name1); write_string(f, obj.name2)
            f.write(struct.pack('<I', obj.flags)); f.write(struct.pack('<16f', *obj.matrix))

    sz = os.path.getsize(output_path); isz = os.path.getsize(pc.path)
    print(f"\n  {os.path.basename(pc.path)}: {isz:,}B -> {os.path.basename(output_path)}: {sz:,}B ({sz/isz*100:.1f}%)")

    # generate crash.dat if PC crash.dat exists
    pc_crash_path = getattr(pc, '_crash_dat_override', None)
    if not pc_crash_path or not os.path.exists(pc_crash_path):
        pc_crash_path = os.path.join(os.path.dirname(pc.path), os.path.basename(pc.path).replace('.bgm', '-crash.dat'))
        if not os.path.exists(pc_crash_path):
            for pattern in ['-crash.dat', '_crash.dat', '-pc-crash.dat']:
                alt = os.path.join(os.path.dirname(pc.path), os.path.basename(pc.path).replace('.bgm', pattern))
                if os.path.exists(alt): pc_crash_path = alt; break

    if os.path.exists(pc_crash_path):
        crash_out = os.path.join(os.path.dirname(output_path),
                                 os.path.basename(output_path).replace('.bgm', '-crash.dat'))
        generate_ps2_crash_dat(pc, pc_crash_path, output_path, vif_blob, sinfo, all_batches, crash_out)
    else:
        print(f"  No PC crash.dat found (tried {pc_crash_path}), skipping crash.dat generation")


# crash.dat
def parse_pc_crash_dat(path):
    data = open(path, 'rb').read()
    off = 0
    nc = struct.unpack_from('<I', data, off)[0]; off += 4
    nodes = []
    for _ in range(nc):
        name_end = data.index(0, off) + 1
        name = data[off:name_end-1].decode('ascii', errors='replace')
        off = name_end
        ns = struct.unpack_from('<I', data, off)[0]; off += 4
        surfs = []
        for _ in range(ns):
            nv = struct.unpack_from('<I', data, off)[0]; off += 4
            nvb = struct.unpack_from('<I', data, off)[0]; off += 4
            vs = nvb // nv if nv > 0 else 0
            nf = vs // 4
            vtx = [list(struct.unpack_from(f'<{nf}f', data, off + vi*vs)) for vi in range(nv)]
            off += nvb
            wgt = [list(struct.unpack_from('<12f', data, off + vi*48)) for vi in range(nv)]
            off += nv * 48
            surfs.append((nv, vs, vtx, wgt))
        nodes.append((name, surfs))
    return nodes


def compute_vif_batch_offsets(vif_data, vtype):
    off = 0; end = len(vif_data)
    batches = []
    pos_off = adc_off = batch_size = 0

    while off < end:
        w = struct.unpack_from('<I', vif_data, off)[0]
        cmd = (w >> 24) & 0x7F

        if cmd == 0x01 or cmd == 0x00:
            off += 4
        elif cmd == 0x04:
            batch_size = w & 0x3FF
            off += 4
        elif cmd == 0x17:
            batches.append((batch_size, pos_off, adc_off))
            off += 4
        elif cmd >= 0x60:
            vn = (cmd >> 2) & 3; vl = cmd & 3
            num = (w >> 16) & 0xFF
            addr = w & 0x3FF
            eb = [4,2,1,0][vl]; comp = [1,2,3,4][vn]
            db = (num * comp * eb + 3) & ~3
            data_start = off + 4

            if addr == 7:
                pos_off = data_start
            elif addr == 9:
                adc_off = data_start
            elif addr == 5:
                pos_off = data_start
                adc_off = data_start

            off += 4 + db
        else:
            off += 4

    return batches


def generate_ps2_crash_dat(pc, pc_crash_path, ps2_bgm_path, vif_blob, sinfo, all_batches, output_path):
    pc_nodes = parse_pc_crash_dat(pc_crash_path)

    model_surfs = {}
    for m in pc.models:
        model_surfs[m.name] = m.surfaces

    out = bytearray()
    out += struct.pack('<I', len(pc_nodes))

    for name, pc_surfs in pc_nodes:
        base_model = name.replace('_crash', '')
        out += name.encode('ascii') + b'\x00'

        if base_model not in model_surfs:
            print(f"  crash.dat: WARNING model '{base_model}' not found, writing 0 surfaces")
            out += struct.pack('<I', 0)
            continue

        bgm_surf_ids = model_surfs[base_model]
        out += struct.pack('<I', len(bgm_surf_ids))

        for surf_idx, bgm_si in enumerate(bgm_surf_ids):
            nv, nb, blob_off, blob_sz = sinfo[bgm_si]
            batches = all_batches[bgm_si]
            surf = pc.surfaces[bgm_si]
            vtype = pc.get_surface_vtype(surf)

            vif_data = vif_blob[blob_off:blob_off+blob_sz]
            vif_batches = compute_vif_batch_offsets(vif_data, vtype)

            out += struct.pack('<I', len(vif_batches))
            for bs, _, _ in vif_batches:
                out += struct.pack('<I', bs)
            for _, po, _ in vif_batches:
                out += struct.pack('<I', po)
            for _, _, ao in vif_batches:
                out += struct.pack('<I', ao)
            out += struct.pack('<I', nv)

            pc_wgts = None
            if surf_idx < len(pc_surfs):
                pc_nv, pc_vs, pc_vtx, pc_wgt = pc_surfs[surf_idx]
                pc_wgts = pc_wgt

            pc_vertices = pc.get_surface_vertices(surf)
            if vtype == VTYPE_SHADOW:
                vid = surf.stream_ids[0] if surf.num_streams >= 1 else -1
                if vid >= 0 and vid in pc.vertex_buffers:
                    vc2, vs2, _, dat2 = pc.vertex_buffers[vid]
                    pc_vertices = []
                    for i in range(surf.vertex_count):
                        off = surf.stream_offsets[0]+i*vs2
                        if off+vs2>len(dat2): break
                        v = Vertex(); v.pos = struct.unpack_from('<3f', dat2, off)
                        pc_vertices.append(v)

            for batch in batches:
                for vidx, adc in batch:
                    if vidx < len(pc_vertices):
                        v = pc_vertices[vidx]
                        px, py, pz = v.pos
                        nx, ny, nz = v.normal
                    else:
                        px = py = pz = nx = ny = nz = 0.0

                    if pc_wgts and vidx < len(pc_wgts):
                        w = pc_wgts[vidx]
                        dpx, dpy, dpz = w[3], w[4], w[5]
                        dnx, dny, dnz = w[9], w[10], w[11]
                    else:
                        dpx, dpy, dpz = px, py, pz
                        dnx, dny, dnz = nx, ny, nz

                    out += struct.pack('<12f',
                        px, py, pz,
                        dpx, dpy, dpz,
                        nx, ny, nz,
                        dnx, dny, dnz)

    with open(output_path, 'wb') as f:
        f.write(out)
    print(f"  Generated {os.path.basename(output_path)}: {len(out):,}B")

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.bgm> [output.bgm] [--crash <pc-crash.dat>]"); sys.exit(1)
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) >= 3 and not sys.argv[2].startswith('--') else os.path.splitext(inp)[0]+"_ps2"+os.path.splitext(inp)[1]

    crash_path = None
    for i, arg in enumerate(sys.argv):
        if arg == '--crash' and i + 1 < len(sys.argv):
            crash_path = sys.argv[i + 1]

    pc = PCBGMParser(inp); pc.parse()

    if crash_path:
        pc._crash_dat_override = crash_path

    write_ps2_bgm(pc, out); print("Done!")

if __name__ == '__main__':
    main()