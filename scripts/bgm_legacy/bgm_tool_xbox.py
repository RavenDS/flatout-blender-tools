#!/usr/bin/env python3
"""
FlatOut 2 BGM Converter: PC -> Xbox (OG Xbox)
by ravenDS (github.com/ravenDS)

Xbox BGM structure (FO2 v0x00020000):

  4 streams:
    stream 0  dt=1  vs=12  flags=0x0000  shadow VB  (float xyz, 12B per vertex)
    stream 1  dt=1  vs=16  flags=0x0000  main VB    (int16 packed, 16B per vertex)
    stream 2  dt=4                       main IB  (small/medium surfaces + shadow)
    stream 3  dt=5                       large IB (main surfaces with vc >= 700)

  Xbox vs=16 vertex layout (16 bytes):
    int16[3]  xyz       scale 1/1024
    int16     pad = 0
    uint8[4]  normals   [0]=nz  [1]=ny  [2]=nx  [3]=255
              b = clamp(round((n+1)*127), 0, 255)
    int16[2]  uv        scale 1/2048

  Main VB packing order:
    All non-shadow surfaces sorted by material_id, then by PC VB byte-offset.
    Both stream2 and stream3 surfaces share this single main VB.

  Stream 2 layout (dt=4):
    [12B real-hdr]   0x17FC 0x0004 N 0 0x1800 X    <- surf 0
    [niu*2B indices]
    for each surface i = 1..n-1:
      [16B null-hdr]  0x17FC 0x0004 0 0 0 0 0 0
      [12B real-hdr]  0x17FC 0x0004 N 0 0x1800 X
      [niu*2B indices]
    [12B trailing mini-hdr]  0x17FC 0x0004 0 0 0 0   <- N=0 X=0 always present
    where N=6 (strip pm=5) or N=5 (list pm=4),  X=0x4000+niu*2

  Stream 3 layout (dt=5) — large surfaces (vc >= S3_THRESHOLD):
    for each surface i = 0..n-1:
      [12B real-hdr]
      [niu*2B indices]
      [16B null-hdr]
      [K bytes zeros]    K = ((niu*2+47)&~15) - (niu*2+28)  (16-byte chunk alignment)

  Seek table (between stream 3 and surface table):
    uint32  count  = len(stream2_surfs) - 1
    uint32  unk    = 0
    for i = 1..count:
      uint32  size    = 28 + niu_i * 2  (approximate; exact formula unknown)
      uint32  ib_off  = byte offset from stream 2 file-start to surface i's null-hdr

  Mystery block (between seek table and surface table):
    Encodes per-surface chunk metrics for stream 3.
    mystery[0] = 0
    for each stream3 surface i:
      zeros_i = ((niu_i*2+47)&~15) - (niu_i*2+28)
      off     = 8 if zeros_i % 8 == 4 else 12
      mystery.append(chunk_i - off)
      if not last: mystery.append(cumulative_chunk_sum_to_i)   [= start of surf i+1]

  Surface table (52 bytes per surface, no stream references):
    int32   isveg
    int32   mid
    int32   vc
    int32   flags   always 0x0112 for non-shadow, 0x0002 for shadow
    int32   pc
    int32   pm
    int32   niu
    uint32  e0      VB base vertex (cumulative vc in main or shadow VB)
    uint32  e1 = 2
    uint32  e2      1 = main surface,  0 = shadow surface
    uint32  e3 = 0
    uint32  e4 = 0
    uint32  e5      position in the surface's IB stream (stream2 or stream3, independent)

  crash.dat:
    Same structure as PC.  vtx: vs=32/36 -> vs=16.  Weights (nv*48) unchanged.
"""

import struct, sys, os
from dataclasses import dataclass, field

VERSION_FO2   = 0x00020000
SHADOW_FLAGS  = 0x0002
S3_NIU_THRESH = 1200      # niu >= this -> stream 3/large IB (verified body, menucar_3, menucar_4)

XB_VS         = 16        # main VB vertex stride
XB_SHADOW_VS  = 12        # shadow VB vertex stride (float xyz)
POS_SCALE     = 1024.0
UV_SCALE      = 2048.0
NORM_SCALE    = 127.0

TRAIL_HDR  = struct.pack('<6H', 0x17FC, 0x0004, 0, 0, 0, 0)          # 12B trailing (N=0,X=0)


# dataclasses

@dataclass
class Material:
    identifier:   int   = 0x4354414D
    name:         str   = ""
    alpha:        int   = 0
    v92:          int   = 0
    num_textures: int   = 0
    shader_id:    int   = 0
    use_colormap: int   = 0
    v74:          int   = 0
    v108:         bytes = b'\x00'*12
    v109:         bytes = b'\x00'*12
    v98:          bytes = b'\x00'*16
    v99:          bytes = b'\x00'*16
    v100:         bytes = b'\x00'*16
    v101:         bytes = b'\x00'*16
    v102:         int   = 0
    tex_names:    list  = field(default_factory=lambda: ["","",""])

@dataclass
class PCSurface:
    is_vegetation:    int  = 0
    material_id:      int  = 0
    vertex_count:     int  = 0
    flags:            int  = 0
    poly_count:       int  = 0
    poly_mode:        int  = 0
    num_indices_used: int  = 0
    num_streams:      int  = 0
    stream_ids:       list = field(default_factory=list)
    stream_offsets:   list = field(default_factory=list)

@dataclass
class Model:
    identifier: int   = 0x444F4D42
    unk:        int   = 0
    name:       str   = ""
    center:     tuple = (0.,0.,0.)
    radius:     tuple = (0.,0.,0.)
    f_radius:   float = 0.
    surfaces:   list  = field(default_factory=list)

@dataclass
class CompactMesh:
    identifier: int   = 0x4853454D
    name1:      str   = ""
    name2:      str   = ""
    flags:      int   = 0
    group:      int   = -1
    matrix:     tuple = tuple([0.]*16)
    models:     list  = field(default_factory=list)

@dataclass
class Object:
    identifier: int   = 0x434A424F
    name1:      str   = ""
    name2:      str   = ""
    flags:      int   = 0
    matrix:     tuple = tuple([0.]*16)


# helpers

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def read_string(f):
    s = b""
    while True:
        c = f.read(1)
        if not c or c == b'\x00': break
        s += c
    return s.decode('ascii', errors='replace')

def write_string(f, s):
    f.write(s.encode('ascii') + b'\x00')

def null_hdr_size(payload):
    """Dynamic null-header size so the next real-header starts 16B-aligned."""
    base = (-payload) % 16
    if base == 0: base = 16
    while base < 12: base += 16
    return base

def make_null_hdr(size):
    """SET_BEGIN_END(0) [8B] + (size-8) zero bytes."""
    return struct.pack('<2I', 0x000417FC, 0) + b'\x00' * (size - 8)

def make_surface_pb(pm, indices_bytes):
    """Complete NV2A push-buffer block for one surface.

    For niu > 4094 (11-bit draw count field caps at 2047 dwords = 4094 indices),
    emits multiple consecutive DRAW_INLINE_ARRAY commands within one BEGIN/END pair.
    All full batches use exactly 4094 indices (cnt=2047).
    The final batch handles the remaining indices normally (even or odd via 0x1808 tail).
    """
    MAX_BATCH = 4094  # 2047 dwords × 2 indices/dword
    niu     = len(indices_bytes) // 2
    N       = 6 if pm == 5 else 5
    begin   = struct.pack('<2I', 0x000417FC, N)
    out     = begin

    offset  = 0
    remaining = niu
    while remaining > 0:
        batch = min(remaining, MAX_BATCH)
        draw_cnt = batch // 2
        has_odd  = batch & 1
        draw_cmd = struct.pack('<I', 0x40000000 | (draw_cnt << 18) | 0x1800)
        out += draw_cmd + indices_bytes[offset * 2 : offset * 2 + draw_cnt * 4]
        if has_odd:
            last_idx = struct.unpack_from('<H', indices_bytes, offset * 2 + draw_cnt * 4)[0]
            out += struct.pack('<I', 0x00041808) + struct.pack('<HH', last_idx, 0)
        offset    += batch
        remaining -= batch

    return out


# PC BGM parser

class PCBGMParser:
    def __init__(self, path):
        self.path = path
        self.version = 0
        self.materials   = []
        self.vertex_buffers  = {}   # stream_idx -> (vc, vs, flags, data)
        self.index_buffers   = {}   # stream_idx -> (ic, data)
        self.surfaces = []
        self.models   = []
        self.meshes   = []
        self.objects  = []

    def parse(self):
        with open(self.path, 'rb') as f:
            self.version = struct.unpack('<I', f.read(4))[0]
            if self.version != VERSION_FO2:
                raise ValueError(f'{self.path}: expected FO2 0x{VERSION_FO2:08x}, '
                                 f'got 0x{self.version:08x}')
            self._parse_materials(f)
            self._parse_streams(f)
            self._parse_surfaces(f)
            self._parse_models(f)
            self._parse_meshes(f)
            self._parse_objects(f)

    def _parse_materials(self, f):
        for _ in range(struct.unpack('<I', f.read(4))[0]):
            m = Material()
            m.identifier    = struct.unpack('<I', f.read(4))[0]
            m.name          = read_string(f)
            m.alpha         = struct.unpack('<i', f.read(4))[0]
            m.v92, m.num_textures, m.shader_id, m.use_colormap, m.v74 = \
                struct.unpack('<5i', f.read(20))
            m.v108 = f.read(12); m.v109 = f.read(12)
            m.v98  = f.read(16); m.v99  = f.read(16)
            m.v100 = f.read(16); m.v101 = f.read(16)
            m.v102 = struct.unpack('<i', f.read(4))[0]
            m.tex_names = [read_string(f) for _ in range(3)]
            self.materials.append(m)

    def _parse_streams(self, f):
        for i in range(struct.unpack('<I', f.read(4))[0]):
            dt = struct.unpack('<I', f.read(4))[0]
            if dt == 1:
                fc, vc, vs, fl = struct.unpack('<4I', f.read(16))
                self.vertex_buffers[i] = (vc, vs, fl, f.read(vc * vs))
            elif dt == 2:
                fc, ic = struct.unpack('<2I', f.read(8))
                self.index_buffers[i]  = (ic, f.read(ic * 2))
            elif dt == 3:
                fc, vc, vs = struct.unpack('<3I', f.read(12))
                self.vertex_buffers[i] = (vc, vs, 0, f.read(vc * vs))

    def _parse_surfaces(self, f):
        for _ in range(struct.unpack('<I', f.read(4))[0]):
            s = PCSurface()
            (s.is_vegetation, s.material_id, s.vertex_count,
             s.flags, s.poly_count, s.poly_mode,
             s.num_indices_used) = struct.unpack('<7i', f.read(28))
            s.num_streams = struct.unpack('<i', f.read(4))[0]
            for _ in range(s.num_streams):
                sid, soff = struct.unpack('<2I', f.read(8))
                s.stream_ids.append(sid); s.stream_offsets.append(soff)
            self.surfaces.append(s)

    def _parse_models(self, f):
        for _ in range(struct.unpack('<I', f.read(4))[0]):
            m = Model()
            m.identifier = struct.unpack('<I', f.read(4))[0]
            m.unk        = struct.unpack('<i', f.read(4))[0]
            m.name       = read_string(f)
            m.center     = struct.unpack('<3f', f.read(12))
            m.radius     = struct.unpack('<3f', f.read(12))
            m.f_radius   = struct.unpack('<f',  f.read(4))[0]
            ns = struct.unpack('<I', f.read(4))[0]
            m.surfaces = [struct.unpack('<i', f.read(4))[0] for _ in range(ns)]
            self.models.append(m)

    def _parse_meshes(self, f):
        for _ in range(struct.unpack('<I', f.read(4))[0]):
            m = CompactMesh()
            m.identifier = struct.unpack('<I', f.read(4))[0]
            m.name1      = read_string(f); m.name2 = read_string(f)
            m.flags      = struct.unpack('<I', f.read(4))[0]
            m.group      = struct.unpack('<i', f.read(4))[0]
            m.matrix     = struct.unpack('<16f', f.read(64))
            nm = struct.unpack('<i', f.read(4))[0]
            m.models = [struct.unpack('<i', f.read(4))[0] for _ in range(nm)]
            self.meshes.append(m)

    def _parse_objects(self, f):
        for _ in range(struct.unpack('<I', f.read(4))[0]):
            o = Object()
            o.identifier = struct.unpack('<I', f.read(4))[0]
            o.name1      = read_string(f); o.name2 = read_string(f)
            o.flags      = struct.unpack('<I', f.read(4))[0]
            o.matrix     = struct.unpack('<16f', f.read(64))
            self.objects.append(o)

    def is_shadow(self, s):
        return s.flags == SHADOW_FLAGS

    def is_stream3(self, s):
        return not self.is_shadow(s) and s.num_indices_used >= S3_NIU_THRESH


# vertex conversion

def pc_to_xb_vert(vdata, vs, vi):
    """Convert one PC float vertex (vs=32 or 36) to Xbox int16 vs=16.
    vs=32: pos(12) norm(12) uv(8)
    vs=36: pos(12) norm(12) color(4) uv(8)

    XB vs=16 layout:
      int16[3]  pos  (×1/1024)
      uint16    pad = 0
      uint32    normal D3DVSDT_NORMPACKED3:
                  bits[31:22] = int(nz × 511)  & 0x3FF  (10-bit signed)
                  bits[21:11] = int(ny × 1023) & 0x7FF  (11-bit signed)
                  bits[10:0]  = int(nx × 1023) & 0x7FF  (11-bit signed)
      int16[2]  uv   (×1/2048)
    """
    b      = vi * vs
    uv_off = 28 if vs == 36 else 24
    px, py, pz = struct.unpack_from('<3f', vdata, b)
    nx, ny, nz = struct.unpack_from('<3f', vdata, b + 12)
    u,  v      = struct.unpack_from('<2f', vdata, b + uv_off)
    ix  = clamp(round(px * POS_SCALE), -32767, 32767)
    iy  = clamp(round(py * POS_SCALE), -32767, 32767)
    iz  = clamp(round(pz * POS_SCALE), -32767, 32767)
    # NORMPACKED3: truncate toward zero (int()), not round
    x_raw = max(-1023, min(1023, int(nx * 1023))) & 0x7FF
    y_raw = max(-1023, min(1023, int(ny * 1023))) & 0x7FF
    z_raw = max(-511,  min(511,  int(nz * 511)))  & 0x3FF
    norm  = (z_raw << 22) | (y_raw << 11) | x_raw
    iu  = clamp(round(u * UV_SCALE), -32767, 32767)
    iv  = clamp(round(v * UV_SCALE), -32767, 32767)
    return struct.pack('<3hHI2h', ix, iy, iz, 0, norm, iu, iv)


# VB / IB builders

def build_vbs(pc):
    """
    Build main VB (stream1, vs=16) and shadow VB (stream0, vs=12).

    Main VB contains ALL non-shadow surfaces (both stream2 and stream3),
    with stream3 surfaces first (sorted by material_id then PC VB byte-offset),
    then stream2 surfaces (same sort order).

    Returns:
        main_vb    bytes
        shadow_vb  bytes
        e0_map     {surface_index: (vb_base_vertex, is_shadow)}
        ib_order   list of surface indices in main VB packing order
    """
    by_mid = {}
    for si, s in enumerate(pc.surfaces):
        if pc.is_shadow(s) or s.vertex_count == 0 or s.num_streams < 1:
            continue
        by_mid.setdefault(s.material_id, []).append((s.stream_offsets[0], si))

    ib_order = []
    for mid in sorted(by_mid):
        for _, si in sorted(by_mid[mid]):
            ib_order.append(si)
    # zero-vc non-shadow surfaces appended for e5 completeness
    seen = set(ib_order)
    for si, s in enumerate(pc.surfaces):
        if si not in seen and not pc.is_shadow(s):
            ib_order.append(si)

    e0_map = {}

    # main VB
    main_vb = bytearray()
    cur = 0
    for si in ib_order:
        s = pc.surfaces[si]
        if s.vertex_count == 0 or s.num_streams < 1:
            e0_map[si] = (0, False); continue
        vc, vs, fl, vdata = pc.vertex_buffers[s.stream_ids[0]]
        vb0 = s.stream_offsets[0] // vs
        e0_map[si] = (cur, False)
        for vi in range(s.vertex_count):
            main_vb += pc_to_xb_vert(vdata, vs, vb0 + vi)
        cur += s.vertex_count

    # shadow VB (float xyz copied directly)
    shadow_vb = bytearray()
    scur = 0
    for si, s in enumerate(pc.surfaces):
        if not pc.is_shadow(s): continue
        if s.vertex_count == 0 or s.num_streams < 1:
            e0_map[si] = (0, True); continue
        vc, vs, fl, vdata = pc.vertex_buffers[s.stream_ids[0]]
        vb0 = s.stream_offsets[0] // vs
        e0_map[si] = (scur, True)
        for vi in range(s.vertex_count):
            base = (vb0 + vi) * vs
            shadow_vb += vdata[base:base + XB_SHADOW_VS]
        scur += s.vertex_count

    return bytes(main_vb), bytes(shadow_vb), e0_map, ib_order


def get_adjusted_indices(pc, si, e0_map):
    """Return uint16 index data for surface si, adjusted to Xbox VB base."""
    s = pc.surfaces[si]
    if s.num_streams < 2 or s.num_indices_used == 0:
        return b''
    ic, ibuf = pc.index_buffers[s.stream_ids[1]]
    ib_off   = s.stream_offsets[1]
    vc, vs, fl, vdata = pc.vertex_buffers[s.stream_ids[0]]
    pc_vbase = s.stream_offsets[0] // vs
    xb_e0, _ = e0_map[si]
    adj  = xb_e0 - pc_vbase
    out  = bytearray(s.num_indices_used * 2)
    for k in range(s.num_indices_used):
        idx = struct.unpack_from('<H', ibuf, ib_off + k * 2)[0] + adj
        struct.pack_into('<H', out, k * 2, clamp(idx, 0, 65534))
    return bytes(out)


def build_stream2(pc, e0_map, ib_order):
    """
    Build stream2 (dt=4) IB data blob and seek table entries.

    Surfaces included: all non-stream3 surfaces (small/medium main + all shadow).
    Order: stream2 mains in ib_order sequence, then shadow surfaces.

    Returns:
        blob         bytes   — data field (outer 16B header added by caller)
        seek_entries list    — [(size, ib_off), …] for surfaces 1..n-1
    """
    # stream2 surface list: main (non-stream3) in ib_order, shadow appended
    s2_mains  = [si for si in ib_order
                 if not pc.is_stream3(pc.surfaces[si])
                 and pc.surfaces[si].num_indices_used > 0]
    s2_shadow = [si for si, s in enumerate(pc.surfaces)
                 if pc.is_shadow(s) and s.num_indices_used > 0]
    s2_surfs  = s2_mains + s2_shadow

    if not s2_surfs:
        return b'', [], 12

    def spb(si):
        s = pc.surfaces[si]
        return make_surface_pb(s.poly_mode, get_adjusted_indices(pc, si, e0_map))

    blob = spb(s2_surfs[0])

    seek_entries = []
    for k in range(1, len(s2_surfs)):
        si         = s2_surfs[k]
        prev_si    = s2_surfs[k - 1]
        prev_pay   = len(spb(prev_si))
        n_size     = null_hdr_size(prev_pay)
        size       = 12 + prev_pay              # seek size = trail(12) + prev payload
        ib_off     = len(blob) + n_size         # offset to this real-hdr from s2 data start
        blob      += make_null_hdr(n_size) + spb(si)
        seek_entries.append((size, ib_off))

    sentinel_size = 12 + len(spb(s2_surfs[-1]))

    blob += TRAIL_HDR   # 12B trailing N=0 X=0
    return bytes(blob), seek_entries, sentinel_size


def _s3_chunk_size(niu):
    """Stream3 chunk size = 16-byte aligned: real(12)+data(niu*2)+null(16)+zeros."""
    return (niu * 2 + 47) & ~15


def build_stream3(pc, e0_map, ib_order):
    """
    Build stream3 (dt=5) IB data blob for large surfaces (vc >= S3_THRESHOLD).

    Each surface chunk: real-hdr(12) + data(niu*2) + null-hdr(16) + zeros(K)
    where K = _s3_chunk_size(niu) - (28 + niu*2)  [pads chunk to 16B boundary]

    Returns:
        blob         bytes
        s3_chunks    list of (niu, pb_size, chunk_size) in s3 order (for mystery block)
    """
    s3_surfs = [si for si in ib_order if pc.is_stream3(pc.surfaces[si])
                and pc.surfaces[si].num_indices_used > 0]
    blob = bytearray()
    s3_chunks = []
    for si in s3_surfs:
        s   = pc.surfaces[si]
        niu = s.num_indices_used
        raw  = get_adjusted_indices(pc, si, e0_map)
        pb   = make_surface_pb(s.poly_mode, raw)
        # Dynamic null-header to pad pb+null to 16-byte boundary (same rule as stream2)
        ns   = null_hdr_size(len(pb))
        chunk_size = len(pb) + ns
        blob += pb + make_null_hdr(ns)
        s3_chunks.append((niu, len(pb), chunk_size))
    return bytes(blob), s3_chunks


def make_mystery(s3_chunks):
    """
    Build mystery block bytes.
    For n_s3 = 0: 16 zero bytes.
    For n_s3 > 0: 2*n_s3 uint32s = 8*n_s3 bytes.
      [0]           = 0
      for each surface i:
        append pb_size + 12
        if not last: append cumulative chunk offset (= start of surf i+1)
    s3_chunks: list of (niu, pb_size, chunk_size) from build_stream3.
    """
    if not s3_chunks:
        return b'\x00' * 16
    result = [0]
    cumsum = 0
    for i, (niu, pb_size, chunk_size) in enumerate(s3_chunks):
        result.append(pb_size + 12)
        if i < len(s3_chunks) - 1:
            cumsum += chunk_size
            result.append(cumsum)
    return struct.pack(f'<{len(result)}I', *result)


# Xbox BGM writer

def write_xbox_bgm(pc, output_path):
    print(f"\nConverting {os.path.basename(pc.path)} -> Xbox BGM ...")

    has_shadow = any(pc.is_shadow(s) for s in pc.surfaces)
    main_vb, shadow_vb, e0_map, ib_order = build_vbs(pc)

    if has_shadow:
        print(f"  main VB   : {len(main_vb)//XB_VS} verts  ({len(main_vb)} B)")
        print(f"  shadow VB : {len(shadow_vb)//XB_SHADOW_VS} verts  ({len(shadow_vb)} B)")
    else:
        print(f"  VB        : {len(main_vb)//XB_VS} verts  ({len(main_vb)} B)  [no shadow]")

    s2_blob, seek_entries, sentinel_size = build_stream2(pc, e0_map, ib_order)
    s3_blob, s3_chunks     = build_stream3(pc, e0_map, ib_order)
    print(f"  stream IB-small: {len(s2_blob)} B  ({len(seek_entries)+1} surfaces)")
    print(f"  stream IB-large: {len(s3_blob)} B  ({len(s3_chunks)} large surfaces)")

    def pack_vb(vc, vs, flags, data):
        return struct.pack('<5I', 1, 0, vc, vs, flags) + data

    def pack_ib(dt, blob):
        return struct.pack('<4I', dt, 0, len(blob), 1) + blob

    # stream objects
    if has_shadow:
        # body type: 4 streams — shadow VB (dt=1 vs=12), main VB (dt=1 vs=16),
        #            small IB (dt=4), large IB (dt=5)
        stream0 = pack_vb(len(shadow_vb)//XB_SHADOW_VS, XB_SHADOW_VS, 0, shadow_vb)
        stream1 = pack_vb(len(main_vb)//XB_VS,           XB_VS,        0, main_vb)
        stream2 = pack_ib(4, s2_blob)
        stream3 = pack_ib(5, s3_blob)
        streams = [stream0, stream1, stream2, stream3]
    else:
        # menucar type: 3 streams — combined VB (dt=1 vs=16),
        #               small IB (dt=4), large IB (dt=5)
        stream0 = pack_vb(len(main_vb)//XB_VS, XB_VS, 0, main_vb)
        stream1 = pack_ib(4, s2_blob)
        stream2 = pack_ib(5, s3_blob)
        streams = [stream0, stream1, stream2]

    def pack_seek():
        all_entries = seek_entries + [(sentinel_size, len(s3_order))]
        out = struct.pack('<2I', len(all_entries), 0)
        for size, off in all_entries:
            out += struct.pack('<2I', size, off)
        return out

    mystery_blob = make_mystery(s3_chunks)

    # e5 maps
    s2_mains  = [si for si in ib_order
                 if not pc.is_stream3(pc.surfaces[si])
                 and pc.surfaces[si].num_indices_used > 0]
    s2_shadow = [si for si, s in enumerate(pc.surfaces)
                 if pc.is_shadow(s) and s.num_indices_used > 0]
    s2_order  = s2_mains + s2_shadow
    s3_order  = [si for si in ib_order
                 if pc.is_stream3(pc.surfaces[si])
                 and pc.surfaces[si].num_indices_used > 0]
    e5_s2 = {si: pos for pos, si in enumerate(s2_order)}
    e5_s3 = {si: pos for pos, si in enumerate(s3_order)}
    e5_ib = {**e5_s2, **e5_s3}

    def pack_surfaces():
        s3_set = set(s3_order)
        out = struct.pack('<I', len(pc.surfaces))
        for si, s in enumerate(pc.surfaces):
            e0, is_shad = e0_map.get(si, (0, pc.is_shadow(s)))
            # e2=1 for body-type non-shadow surfaces; e2=0 for shadow or no-shadow model
            e2 = 1 if (has_shadow and not is_shad) else 0
            e4 = 1 if si in s3_set else 0
            e5 = e5_ib.get(si, 0)
            xb_flags = SHADOW_FLAGS if is_shad else 0x0112
            out += struct.pack('<7i', s.is_vegetation, s.material_id, s.vertex_count,
                               xb_flags, s.poly_count, s.poly_mode, s.num_indices_used)
            out += struct.pack('<6I', e0, 2, e2, 0, e4, e5)
        return out

    with open(output_path, 'wb') as f:
        f.write(struct.pack('<I', pc.version))
        f.write(struct.pack('<I', len(pc.materials)))
        for m in pc.materials:
            f.write(struct.pack('<I',  m.identifier))
            write_string(f, m.name)
            f.write(struct.pack('<i',  m.alpha))
            f.write(struct.pack('<5i', m.v92, m.num_textures,
                                m.shader_id, m.use_colormap, m.v74))
            f.write(m.v108); f.write(m.v109)
            f.write(m.v98);  f.write(m.v99)
            f.write(m.v100); f.write(m.v101)
            f.write(struct.pack('<i', m.v102))
            for tn in m.tex_names: write_string(f, tn)

        f.write(struct.pack('<I', len(streams)))
        for st in streams:
            f.write(st)

        f.write(pack_seek())
        f.write(mystery_blob)
        f.write(pack_surfaces())

        f.write(struct.pack('<I', len(pc.models)))
        for m in pc.models:
            f.write(struct.pack('<I', m.identifier))
            f.write(struct.pack('<i', m.unk))
            write_string(f, m.name)
            f.write(struct.pack('<3f', *m.center))
            f.write(struct.pack('<3f', *m.radius))
            f.write(struct.pack('<f',  m.f_radius))
            f.write(struct.pack('<I',  len(m.surfaces)))
            for s in m.surfaces: f.write(struct.pack('<i', s))

        f.write(struct.pack('<I', len(pc.meshes)))
        for mesh in pc.meshes:
            f.write(struct.pack('<I', mesh.identifier))
            write_string(f, mesh.name1); write_string(f, mesh.name2)
            f.write(struct.pack('<I', mesh.flags))
            f.write(struct.pack('<i', mesh.group))
            f.write(struct.pack('<16f', *mesh.matrix))
            f.write(struct.pack('<i',   len(mesh.models)))
            for mid in mesh.models: f.write(struct.pack('<i', mid))

        f.write(struct.pack('<I', len(pc.objects)))
        for o in pc.objects:
            f.write(struct.pack('<I', o.identifier))
            write_string(f, o.name1); write_string(f, o.name2)
            f.write(struct.pack('<I', o.flags))
            f.write(struct.pack('<16f', *o.matrix))

    isz = os.path.getsize(pc.path)
    osz = os.path.getsize(output_path)
    print(f"\n  {os.path.basename(pc.path)}: {isz:,}B -> "
          f"{os.path.basename(output_path)}: {osz:,}B  ({osz/isz*100:.1f}%)")


# crash.dat conversion

def convert_crash_dat(input_path, output_path):
    """Convert PC crash.dat to Xbox. vtx: vs=32/36 -> vs=16. Weights unchanged."""
    data = open(input_path, 'rb').read()
    out  = bytearray()
    off  = 0
    nc   = struct.unpack_from('<I', data, off)[0]; off += 4
    out += struct.pack('<I', nc)
    for _ in range(nc):
        end  = data.index(0, off)
        out += data[off:end + 1]; off = end + 1
        ns = struct.unpack_from('<I', data, off)[0]; off += 4
        out += struct.pack('<I', ns)
        for _ in range(ns):
            nv  = struct.unpack_from('<I', data, off)[0]; off += 4
            nvb = struct.unpack_from('<I', data, off)[0]; off += 4
            vs  = nvb // nv if nv > 0 else 32
            out += struct.pack('<I', nv)
            out += struct.pack('<I', nv * XB_VS)
            vtx = data[off:off + nvb]; off += nvb
            for vi in range(nv):
                out += pc_to_xb_vert(vtx, vs, vi)
            wgt_sz = nv * 48
            out += data[off:off + wgt_sz]; off += wgt_sz
    with open(output_path, 'wb') as f:
        f.write(out)
    isz = os.path.getsize(input_path)
    print(f"  crash: {os.path.basename(input_path)}: {isz:,}B -> "
          f"{os.path.basename(output_path)}: {len(out):,}B")


# main

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.bgm> [output.bgm] [--crash <pc-crash.dat>]")
        sys.exit(1)

    inp = sys.argv[1]
    out = (sys.argv[2]
           if len(sys.argv) >= 3 and not sys.argv[2].startswith('--')
           else os.path.splitext(inp)[0] + '_xbox' + os.path.splitext(inp)[1])

    crash_in = None
    for i, arg in enumerate(sys.argv):
        if arg == '--crash' and i + 1 < len(sys.argv):
            crash_in = sys.argv[i + 1]

    pc = PCBGMParser(inp)
    pc.parse()
    write_xbox_bgm(pc, out)

    if not crash_in:
        base = os.path.splitext(inp)[0]
        for suffix in ['-crash.dat', '_crash.dat']:
            candidate = base + suffix
            if os.path.exists(candidate):
                crash_in = candidate; break

    if crash_in and os.path.exists(crash_in):
        crash_out = os.path.join(
            os.path.dirname(out),
            os.path.basename(out).replace('.bgm', '') + '-crash.dat')
        convert_crash_dat(crash_in, crash_out)
    elif crash_in:
        print(f"  crash.dat not found: {crash_in}")
    else:
        print("  No crash.dat found, skipping.")

    print("Done!")

if __name__ == '__main__':
    main()
