#!/usr/bin/env python3
"""
FlatOut BGM Tool 2.3.0 — multi-platform BGM converter & optimizer
https://github.com/RavenDS/flatout-blender-tools

Supported input formats  (auto-detected):
  PC FO1 / FO2 / FOUC   — float vertex / index streams (types 1 + 2)
  PS2                    — VIF packet blob (stream type 3, 40-byte surface records)
  PSP                    — GE display-list blob (stream type 3, 32-byte surface records)
  Xbox (OG)              — NV2A push-buffer streams (types 4/5, 52-byte surface records)

Conversion targets (-convert <fmt>):
  FO1, FO2, FOUC         — existing PC format conversions
  PS2                    — PlayStation 2 (VIF packed, ADC triangle strips)
  PSP                    — PSP / FlatOut Head On (GE display list)
  XBOX                   — Original Xbox (NV2A push-buffer, NORMPACKED3 VB)

Console input: auto-detects PS2/PSP/Xbox and converts geometry to PC FO2.
  All PC operations (-clean, -optimize, etc.) can then be applied.
  crash.dat is converted/generated where supported.

Operations (can be combined):
  -convert <fmt>
      Convert the BGM (and its crash.dat) to a different game format.

  -clean
      Remove unreferenced vertex/index buffer streams.  The source tool often
      writes two stream pairs per surface but only registers one; the other is
      dead weight.  Renumbers surviving stream indices in the surface table.
      <name>_crash.dat is copied alongside unchanged (index reordering does not
      affect vertex data or positions)

  -optimize
      Vertex deduplication + stream merging.
      1. Deduplicate: the source tool writes one vertex per index entry with no
         sharing.  We rebuild each vertex buffer keeping only distinct records
         and rewrite the index buffer to reference them.  Typical saving: 60-65%.
      2. Merge: collapse to one vbuf per vertex format (FO2-style shared streams)
         and one shared ibuf.  Matches the layout of original FO2 files.
      <name>_crash.dat vertex arrays are remapped to match the new vertex ordering

  -full
      Shortcut for -clean & -optimize

  -menucar
      Reorder surfaces to FO2 menucar draw order

  -windflip
      Experimental fix for inconsistent triangle winding on FOUC (+ FO2) BGM files.
      This corrects the "missing faces" visual artifact seen on some FOUC car parts 
      where a subset of triangles were authored with reversed winding relative to their normals.
      Operates directly on the index buffer streams; vertex data is unchanged.
      crash.dat is copied alongside unchanged (vertex ordering is not affected).

  -lighthacks
      Fix for additional light surfaces when converting from FOUC to FO2/FO1.
      Default: duplicate all _b light variants, assign common material
      Override: <target1>,<target2>,<target3>,..

  -lightorder
      Reorder materials and surfaces by draw priority (mirrors fo2_bgm_export).
      Light _b surfaces are drawn before active lights; suspension drawn before body.
      Edit the MATERIAL_PRIORITIES table below to customise.

Usage:
  bgm_tool.py <input.bgm> [output.bgm] -clean [-optimize]
  bgm_tool.py <input.bgm> [output.bgm] -convert PS2
  bgm_tool.py <input_ps2.bgm> [output_pc.bgm]
  bgm_tool.py <input_ps2.bgm> [output.bgm] -convert PSP
"""

import struct, sys, os, shutil, copy, math
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# LIGHT ORDER — material draw priority table
# ─────────────────────────────────────────────────────────────────────────────

MATERIAL_PRIORITIES = {
    # suspension
    "shearshock": 1, "shearhock": 1, "shearspring": 2,
    # lights — _b variants (inactive/base state) drawn first
    "light_brake_b": 1, "light_brake_b_2": 1, "light_brake_2_b": 1,
    "light_brake_l_b": 1, "light_brake_r_b": 1,
    "light_brake_l_b_2": 1, "light_brake_r_b_2": 1,
    "light_brake_l_2_b": 1, "light_brake_r_2_b": 1,
    "light_reverse_b": 1, "light_reverse_b_2": 1, "light_reverse_2_b": 1,
    "light_reverse_l_b": 1, "light_reverse_r_b": 1,
    "light_reverse_l_b_2": 1, "light_reverse_r_b_2": 1,
    "light_reverse_l_2_b": 1, "light_reverse_r_2_b": 1,
    "light_front_b": 1, "light_front_b_2": 1, "light_front_2_b": 1,
    "light_front_l_b": 1, "light_front_r_b": 1,
    "light_front_l_b_2": 1, "light_front_r_b_2": 1,
    "light_front_l_2_b": 1, "light_front_r_2_b": 1,
    # lights — active/illuminated variants drawn last
    "light_brake": 2, "light_brake_2": 2,
    "light_brake_l": 2, "light_brake_r": 2,
    "light_brake_l_2": 2, "light_brake_r_2": 2,
    "light_reverse": 2, "light_reverse_2": 2,
    "light_reverse_l": 2, "light_reverse_r": 2,
    "light_reverse_l_2": 2, "light_reverse_r_2": 2,
    "light_front": 2, "light_front_2": 2,
    "light_front_l": 2, "light_front_r": 2,
    "light_front_l_2": 2, "light_front_r_2": 2,
}


def _read_string(f):
    s = b""
    while True:
        c = f.read(1)
        if not c or c == b'\x00':
            break
        s += c
    return s.decode('ascii', errors='replace')



# ─────────────────────────────────────────────────────────────────────────────
# LIGHTWEIGHT BINARY READER
# ─────────────────────────────────────────────────────────────────────────────

class _BinReader:
    """Lightweight binary reader (no Blender dependency)."""
    __slots__ = ('data', 'pos')

    def __init__(self, data):
        self.data = data
        self.pos  = 0

    def read(self, n):
        r = self.data[self.pos:self.pos + n]
        self.pos += n
        return r

    def u8(self):
        v = self.data[self.pos]; self.pos += 1; return v

    def u16(self):
        v = struct.unpack_from('<H', self.data, self.pos)[0]; self.pos += 2; return v

    def u32(self):
        v = struct.unpack_from('<I', self.data, self.pos)[0]; self.pos += 4; return v

    def i32(self):
        v = struct.unpack_from('<i', self.data, self.pos)[0]; self.pos += 4; return v

    def f32(self):
        v = struct.unpack_from('<f', self.data, self.pos)[0]; self.pos += 4; return v

    def vec3f(self):
        v = struct.unpack_from('<3f', self.data, self.pos); self.pos += 12; return v

    def read_string(self):
        start = self.pos
        while self.data[self.pos] != 0:
            self.pos += 1
        s = self.data[start:self.pos].decode('ascii', errors='replace')
        self.pos += 1
        return s


# ─────────────────────────────────────────────────────────────────────────────
# PC BGM PARSER / WRITER  (FO1 / FO2 / FOUC)
# ─────────────────────────────────────────────────────────────────────────────

def parse_bgm(path):
    """Parse a PC BGM file.

    Returns (version, materials_raw, streams, surfaces, models, rest) where:
      materials_raw : list of material dicts  {ident, name, n_alpha, v92,
                                               n_num_tex, shader_id, n_use_colormap,
                                               v74, v108, v109, v98..v101, v102, tex_names}
      streams       : list of dicts  {dt, fc, data, [vc, vs, flags] or [ic]}
      surfaces      : list of dicts  {isveg, mid, vc, flags, pc, pm, niu,
                                       extra, nst, sids, soffs}
      models        : list of dicts  {ident, unk, name, center, radius,
                                       f_radius, surfaces}
      meshes        : list of dicts  {ident, name1, name2, flags, group, matrix, model_ids}
      objects       : list of dicts  {ident, name1, name2, flags, matrix}
    """
    with open(path, 'rb') as f:
        version = struct.unpack('<I', f.read(4))[0]

        # materials — parsed as structured dicts so shader IDs and v92 can be patched
        nm = struct.unpack('<I', f.read(4))[0]
        materials_raw = []
        for _ in range(nm):
            ident  = struct.unpack('<I', f.read(4))[0]
            name   = _read_string(f)
            n_alpha = struct.unpack('<i', f.read(4))[0]
            v92 = n_num_tex = shader_id = n_use_colormap = v74 = 0
            v108 = v109 = (0, 0, 0)
            if version >= 0x10004:
                v92            = struct.unpack('<i', f.read(4))[0]
                n_num_tex      = struct.unpack('<i', f.read(4))[0]
                shader_id      = struct.unpack('<i', f.read(4))[0]
                n_use_colormap = struct.unpack('<i', f.read(4))[0]
                v74            = struct.unpack('<i', f.read(4))[0]
                v108           = struct.unpack('<3i', f.read(12))
                v109           = struct.unpack('<3i', f.read(12))
            v98  = struct.unpack('<4i', f.read(16))
            v99  = struct.unpack('<4i', f.read(16))
            v100 = struct.unpack('<4i', f.read(16))
            v101 = struct.unpack('<4i', f.read(16))
            v102 = struct.unpack('<i', f.read(4))[0]
            tex_names = [_read_string(f) for _ in range(3)]
            materials_raw.append({
                'ident': ident, 'name': name, 'n_alpha': n_alpha,
                'v92': v92, 'n_num_tex': n_num_tex, 'shader_id': shader_id,
                'n_use_colormap': n_use_colormap, 'v74': v74,
                'v108': v108, 'v109': v109,
                'v98': v98, 'v99': v99, 'v100': v100, 'v101': v101,
                'v102': v102, 'tex_names': tex_names,
            })

        # streams
        ns = struct.unpack('<I', f.read(4))[0]
        streams = []
        for _ in range(ns):
            dt = struct.unpack('<I', f.read(4))[0]
            fc = struct.unpack('<I', f.read(4))[0]
            if dt == 1:
                vc, vs, fl = struct.unpack('<3I', f.read(12))
                data = f.read(vc * vs)
                streams.append({'dt': 1, 'fc': fc, 'vc': vc, 'vs': vs,
                                 'flags': fl, 'data': data})
            elif dt == 2:
                ic = struct.unpack('<I', f.read(4))[0]
                data = f.read(ic * 2)
                streams.append({'dt': 2, 'fc': fc, 'ic': ic, 'data': data})
            elif dt == 3:
                vc, vs = struct.unpack('<2I', f.read(8))
                data = f.read(vc * vs)
                streams.append({'dt': 3, 'fc': fc, 'vc': vc, 'vs': vs,
                                 'flags': 0, 'data': data})
            else:
                streams.append({'dt': dt, 'fc': fc, 'data': b''})

        # FOUC detection: any stream with fc (foucExtraFormat) > 0
        is_fouc = any(s['fc'] > 0 for s in streams)

        # surfaces
        nsurf = struct.unpack('<I', f.read(4))[0]
        surfaces = []
        for _ in range(nsurf):
            isveg, mid, vc, flags, pc, pm, niu = struct.unpack('<7i', f.read(28))
            extra = b''
            if version < 0x20000:
                # FO1: center (3f) + radius (3f) = 24 bytes
                extra = f.read(24)
            elif is_fouc:
                # FOUC: foucVertexMultiplier (4f) = 16 bytes
                # version is 0x20000 (same as FO2) but distinguished by stream fc > 0
                extra = f.read(16)
            nst = struct.unpack('<i', f.read(4))[0]
            sids = []; softs = []
            for _ in range(nst):
                sid, soff = struct.unpack('<2I', f.read(8))
                sids.append(sid); softs.append(soff)
            surfaces.append({
                'isveg': isveg, 'mid': mid, 'vc': vc, 'flags': flags,
                'pc': pc, 'pm': pm, 'niu': niu, 'extra': extra,
                'nst': nst, 'sids': list(sids), 'soffs': list(softs),
            })

        # models, needed for crash.dat remapping
        nmod = struct.unpack('<I', f.read(4))[0]
        models = []
        for _ in range(nmod):
            ident    = struct.unpack('<I', f.read(4))[0]
            unk      = struct.unpack('<i', f.read(4))[0]
            name     = _read_string(f)
            center   = struct.unpack('<3f', f.read(12))
            radius   = struct.unpack('<3f', f.read(12))
            f_radius = struct.unpack('<f',  f.read(4))[0]
            ns2      = struct.unpack('<I',  f.read(4))[0]
            surfs2   = [struct.unpack('<i', f.read(4))[0] for _ in range(ns2)]
            models.append({'ident': ident, 'unk': unk, 'name': name,
                           'center': center, 'radius': radius,
                           'f_radius': f_radius, 'surfaces': surfs2})

        # compact meshes (MESH)
        nmesh = struct.unpack('<I', f.read(4))[0]
        meshes = []
        for _ in range(nmesh):
            ident  = struct.unpack('<I', f.read(4))[0]
            name1  = _read_string(f)
            name2  = _read_string(f)
            flags  = struct.unpack('<I', f.read(4))[0]
            group  = struct.unpack('<i', f.read(4))[0]
            matrix = list(struct.unpack('<16f', f.read(64)))
            nm2    = struct.unpack('<i', f.read(4))[0]
            mids   = [struct.unpack('<i', f.read(4))[0] for _ in range(nm2)]
            meshes.append({'ident': ident, 'name1': name1, 'name2': name2,
                           'flags': flags, 'group': group,
                           'matrix': matrix, 'model_ids': mids})

        # objects / dummies (OBJC)
        nobj = struct.unpack('<I', f.read(4))[0]
        objects = []
        for _ in range(nobj):
            ident  = struct.unpack('<I', f.read(4))[0]
            name1  = _read_string(f)
            name2  = _read_string(f)
            flags  = struct.unpack('<I', f.read(4))[0]
            matrix = list(struct.unpack('<16f', f.read(64)))
            objects.append({'ident': ident, 'name1': name1, 'name2': name2,
                            'flags': flags, 'matrix': matrix})

    is_fo1 = (version < 0x20000) and not is_fouc
    return version, materials_raw, streams, surfaces, models, meshes, objects, is_fouc, is_fo1


def write_bgm(path, version, materials_raw, streams, surfaces, models, meshes, objects):
    """Write a BGM file.  meshes and objects replace the old raw 'rest' blob."""
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', version))

        f.write(struct.pack('<I', len(materials_raw)))
        for m in materials_raw:
            f.write(struct.pack('<I', m['ident']))
            f.write(m['name'].encode('ascii') + b'\x00')
            f.write(struct.pack('<i', m['n_alpha']))
            if version >= 0x10004:
                f.write(struct.pack('<i', m['v92']))
                f.write(struct.pack('<i', m['n_num_tex']))
                f.write(struct.pack('<i', m['shader_id']))
                f.write(struct.pack('<i', m['n_use_colormap']))
                f.write(struct.pack('<i', m['v74']))
                f.write(struct.pack('<3i', *m['v108']))
                f.write(struct.pack('<3i', *m['v109']))
            f.write(struct.pack('<4i', *m['v98']))
            f.write(struct.pack('<4i', *m['v99']))
            f.write(struct.pack('<4i', *m['v100']))
            f.write(struct.pack('<4i', *m['v101']))
            f.write(struct.pack('<i', m['v102']))
            for t in m['tex_names']:
                f.write(t.encode('ascii') + b'\x00')

        f.write(struct.pack('<I', len(streams)))
        for s in streams:
            f.write(struct.pack('<2I', s['dt'], s['fc']))
            if s['dt'] == 1:
                f.write(struct.pack('<3I', s['vc'], s['vs'], s['flags']))
                f.write(s['data'])
            elif s['dt'] == 2:
                f.write(struct.pack('<I', s['ic']))
                f.write(s['data'])
            elif s['dt'] == 3:
                f.write(struct.pack('<2I', s['vc'], s['vs']))
                f.write(s['data'])
            else:
                f.write(s['data'])

        f.write(struct.pack('<I', len(surfaces)))
        for s in surfaces:
            f.write(struct.pack('<7i',
                s['isveg'], s['mid'], s['vc'], s['flags'],
                s['pc'], s['pm'], s['niu']))
            f.write(s['extra'])
            f.write(struct.pack('<i', s['nst']))
            for j in range(s['nst']):
                f.write(struct.pack('<2I', s['sids'][j], s['soffs'][j]))

        f.write(struct.pack('<I', len(models)))
        for m in models:
            f.write(struct.pack('<I', m['ident']))
            f.write(struct.pack('<i', m['unk']))
            f.write(m['name'].encode('ascii') + b'\x00')
            f.write(struct.pack('<3f', *m['center']))
            f.write(struct.pack('<3f', *m['radius']))
            f.write(struct.pack('<f',   m['f_radius']))
            f.write(struct.pack('<I', len(m['surfaces'])))
            for sid in m['surfaces']:
                f.write(struct.pack('<i', sid))

        f.write(struct.pack('<I', len(meshes)))
        for m in meshes:
            f.write(struct.pack('<I', m['ident']))
            f.write(m['name1'].encode('ascii') + b'\x00')
            f.write(m['name2'].encode('ascii') + b'\x00')
            f.write(struct.pack('<I', m['flags']))
            f.write(struct.pack('<i', m['group']))
            f.write(struct.pack('<16f', *m['matrix']))
            f.write(struct.pack('<i', len(m['model_ids'])))
            for mid in m['model_ids']:
                f.write(struct.pack('<i', mid))

        f.write(struct.pack('<I', len(objects)))
        for o in objects:
            f.write(struct.pack('<I', o['ident']))
            f.write(o['name1'].encode('ascii') + b'\x00')
            f.write(o['name2'].encode('ascii') + b'\x00')
            f.write(struct.pack('<I', o['flags']))
            f.write(struct.pack('<16f', *o['matrix']))



# ─────────────────────────────────────────────────────────────────────────────
# PC CRASH.DAT
# ─────────────────────────────────────────────────────────────────────────────

def _find_crash_dat(bgm_path):
    """Locate a crash.dat file alongside a BGM file.

    Checks:
      1. <stem>_crash.dat / <stem>_crash.DAT  — named, paired with this BGM
      2. crash.dat / CRASH.DAT               — standalone, shared in same folder

    If both are present the user is prompted to choose.
    Returns (path, is_standalone) where is_standalone=False for the named variant
    and True for the bare crash.dat.  Returns (None, None) if nothing found.
    """
    base   = os.path.splitext(bgm_path)[0]
    folder = os.path.dirname(os.path.abspath(bgm_path))

    named = None
    for suffix in ('_crash.dat', '-crash.dat', '_crash.DAT', '-crash.DAT'):
        p = base + suffix
        if os.path.exists(p):
            named = p
            break

    standalone = None
    for name in ('crash.dat', 'CRASH.DAT'):
        p = os.path.join(folder, name)
        if os.path.exists(p):
            standalone = p
            break

    if named and standalone:
        print(f"  crash.dat: found both '{os.path.basename(named)}'"
              f" and '{os.path.basename(standalone)}'.")
        while True:
            ans = input("  Which one to use? [1] named  [2] standalone (crash.dat): ").strip()
            if ans == '1': return named, False
            if ans == '2': return standalone, True
            print("  Please enter 1 or 2.")
    if named:
        return named, False
    if standalone:
        return standalone, False  # treat as input-only; output goes to <stem>_crash.dat
    return None, None


def _crash_dst_path(output_bgm, is_standalone):
    """Compute the output crash.dat path for a given output BGM path."""
    if is_standalone:
        return os.path.join(os.path.dirname(os.path.abspath(output_bgm)), 'crash.dat')
    return os.path.splitext(output_bgm)[0] + '_crash.dat'


def _parse_crash_dat(path, is_fouc=False):
    """Parse a crash.dat file.

    FO2 / FO1 surface layout per surface:
      uint32  num_verts
      uint32  num_bytes  (= num_verts * vertex_size)
      bytes   vertex_buffer[num_bytes]
      bytes   weights[num_verts * 48]    (tCrashDataWeights: 4 × float[3])

    FOUC surface layout per surface:
      uint32  num_verts
      bytes   weights[num_verts * 40]    (tCrashDataWeightsFOUC, no vbuf)
    """
    data = open(path, 'rb').read()
    off  = 0
    nc   = struct.unpack_from('<I', data, off)[0]; off += 4
    nodes = []
    for _ in range(nc):
        name_end = data.index(0, off)
        name = data[off:name_end].decode('ascii', errors='replace')
        off  = name_end + 1
        ns   = struct.unpack_from('<I', data, off)[0]; off += 4
        surfs = []
        for _ in range(ns):
            nv = struct.unpack_from('<I', data, off)[0]; off += 4
            if is_fouc:
                # FOUC: no vbuf, 40-byte weights (tCrashDataWeightsFOUC)
                wgt = data[off : off + nv * 40]; off += nv * 40
                surfs.append({'vtx': b'', 'wgt': wgt, 'vs': 0})
            else:
                # FO2 / FO1: vbuf + 48-byte weights (tCrashDataWeights)
                nvb = struct.unpack_from('<I', data, off)[0]; off += 4
                vs  = nvb // nv if nv > 0 else 0
                vtx = data[off : off + nvb];    off += nvb
                wgt = data[off : off + nv * 48]; off += nv * 48
                surfs.append({'vtx': vtx, 'wgt': wgt, 'vs': vs})
        nodes.append((name, surfs))
    return nodes


def _write_crash_dat(path, nodes, is_fouc=False):
    """Write a crash.dat file in the correct format for the target game version."""
    out = bytearray()
    out += struct.pack('<I', len(nodes))
    for name, surfs in nodes:
        out += name.encode('ascii') + b'\x00'
        out += struct.pack('<I', len(surfs))
        for s in surfs:
            if is_fouc:
                # FOUC: vcount + 40-byte weights, no vbuf
                nv = len(s['wgt']) // 40
                out += struct.pack('<I', nv)
                out += s['wgt']
            else:
                # FO2 / FO1: vcount + vbytes + vbuf + 48-byte weights
                nv  = len(s['wgt']) // 48
                nvb = len(s['vtx'])
                out += struct.pack('<2I', nv, nvb)
                out += s['vtx']
                out += s['wgt']
    with open(path, 'wb') as f:
        f.write(out)


def _copy_crash_dat(input_bgm, output_bgm):
    src, is_standalone = _find_crash_dat(input_bgm)
    if not src:
        print(f"  crash.dat: none found alongside "
              f"{os.path.basename(input_bgm)}, skipping")
        return
    dst = _crash_dst_path(output_bgm, is_standalone)
    if os.path.abspath(src) == os.path.abspath(dst):
        print(f"  crash.dat: same path, no copy needed ({os.path.basename(src)})")
    else:
        shutil.copy2(src, dst)
        print(f"  crash.dat: copied  {os.path.basename(src)}"
              f" -> {os.path.basename(dst)}")


def _parse_ps2_crash_dat(path):
    """Parse a PS2 crash.dat into a list of (node_name, surfs) where each surf has
    'tv' (total vertex count) and 'verts' (list of 12-float weight tuples).

    PS2 crash.dat per-surface layout:
      uint32 num_batches
      uint32[nb]  batch_vertex_counts
      uint32[nb]  pos_vif_offsets    (not needed for PC conversion)
      uint32[nb]  adc_vif_offsets    (not needed for PC conversion)
      uint32      total_verts
      float32[12 * total_verts]   base_pos[3], crash_pos[3], base_norm[3], crash_norm[3]
    """
    data = open(path, 'rb').read()
    off = 0
    nc = struct.unpack_from('<I', data, off)[0]; off += 4
    nodes = []
    for _ in range(nc):
        name_end = data.index(0, off)
        name = data[off:name_end].decode('ascii', 'replace'); off = name_end + 1
        ns = struct.unpack_from('<I', data, off)[0]; off += 4
        surfs = []
        for _ in range(ns):
            nb = struct.unpack_from('<I', data, off)[0]; off += 4
            off += nb * 4 * 3   # skip batch_sizes, pos_offs, adc_offs
            tv = struct.unpack_from('<I', data, off)[0]; off += 4
            verts = []
            for _ in range(tv):
                verts.append(struct.unpack_from('<12f', data, off)); off += 48
            surfs.append({'tv': tv, 'verts': verts})
        nodes.append((name, surfs))
    return nodes


def _convert_ps2_crash_to_pc(ps2_crash_path, output_path, streams, surfaces, models):
    """Convert a PS2 crash.dat to PC FO2 format.

    The PS2 crash.dat stores vertices in VIF batch order (with duplicates at batch
    boundaries). The PC crash.dat needs vertices in the same order as the PC surface's
    vertex buffer, with one 48-byte weight block per vertex.

    Conversion strategy:
      1. Parse PS2 crash weights; build a position→weight lookup using base_pos.
         PS2 base_pos values are stored as the original PC float positions (written
         by generate_ps2_crash_dat), so they match the reconstructed PC VB exactly.
      2. For each PC surface, walk its vertex buffer. For each vertex, look up its
         position in the PS2 weight table and emit the crash weight.
         Fallback: identity weight (base == crash) for any unmatched vertex.

    PC crash.dat per-surface layout:
      uint32 nv, uint32 nvb, vbuf[nvb], weights[nv * 48]
    """
    if not os.path.exists(ps2_crash_path):
        print(f"  crash.dat: PS2 source not found ({os.path.basename(ps2_crash_path)}), skipping")
        return

    ps2_nodes = _parse_ps2_crash_dat(ps2_crash_path)
    model_surf_map = {m['name']: m['surfaces'] for m in models}

    out = bytearray()
    out += struct.pack('<I', len(ps2_nodes))

    for node_name, ps2_surfs in ps2_nodes:
        out += node_name.encode('ascii') + b'\x00'
        base_model = node_name.replace('_crash', '')
        surf_ids = model_surf_map.get(base_model)

        if surf_ids is None:
            print(f"  crash.dat: WARNING '{base_model}' not found in models, writing 0 surfaces")
            out += struct.pack('<I', 0)
            continue

        out += struct.pack('<I', len(surf_ids))

        for j, bgm_si in enumerate(surf_ids):
            if j >= len(ps2_surfs) or bgm_si >= len(surfaces):
                # no crash data for this surface — write nv=0
                out += struct.pack('<2I', 0, 0)
                continue

            surf    = surfaces[bgm_si]
            ps2_s   = ps2_surfs[j]
            vid     = surf['sids'][0] if surf['nst'] >= 1 else -1

            if vid < 0 or vid >= len(streams):
                out += struct.pack('<2I', 0, 0)
                continue

            sv    = streams[vid]
            vs    = sv['vs']
            vdata = sv['data']
            voff  = surf['soffs'][0]
            nv    = surf['vc']

            # Build position → weight lookup from PS2 crash surface.
            #
            # The PS2 crash.dat stores base_pos as the *original PC float* positions
            # (written by generate_ps2_crash_dat from the PC BGM's vertex buffer).
            # The reconstructed PC vertex buffer holds VIF-decoded positions:
            #   int16_value / 1024.0
            # whose max error from the original float is ± 0.5/1024 ≈ 0.000488 per axis.
            # This means round(orig_float * 1024) and round(vif_decoded * 1024) can differ
            # by ±1.  We therefore index every crash vertex under all 27 surrounding
            # int16-space positions so the lookup always succeeds within quantisation error.
            #
            # Key: (int16_x, int16_y, int16_z) = (round(x * 1024), ...)
            ps2_pos_wgt = {}
            for wv in ps2_s['verts']:
                cx = round(wv[0] * 1024)
                cy = round(wv[1] * 1024)
                cz = round(wv[2] * 1024)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            k = (cx + dx, cy + dy, cz + dz)
                            if k not in ps2_pos_wgt:   # first (exact) entry wins
                                ps2_pos_wgt[k] = wv

            # vtx_data slice for this surface
            vtx_slice = vdata[voff : voff + nv * vs]
            nvb = len(vtx_slice)
            out += struct.pack('<2I', nv, nvb)
            out += vtx_slice

            # weights in PC vertex order
            # PC crash.dat convention: base_pos/base_norm = rest-pose vertex (== vtx_data),
            # crash_pos/crash_norm = deformed position under impact.
            # The PS2 crash stores both relative to the PS2 geometry.  We keep crash_pos and
            # crash_norm from the PS2 data but replace base_pos/base_norm with the PC VB
            # values so the rest-pose exactly matches the vertex buffer (as in the PC original).
            missing = 0
            for vi in range(nv):
                px, py, pz = struct.unpack_from('<3f', vdata, voff + vi * vs)
                nx = ny = nz = 0.0
                if vs >= 24:
                    nx, ny, nz = struct.unpack_from('<3f', vdata, voff + vi * vs + 12)
                key = (round(px * 1024), round(py * 1024), round(pz * 1024))
                wv = ps2_pos_wgt.get(key)
                if wv is not None:
                    # base_pos/base_norm from PC VB; crash_pos/crash_norm from PS2 crash
                    out += struct.pack('<12f',
                                      px,    py,    pz,      # base_pos  = PC VB (exact)
                                      wv[3], wv[4], wv[5],   # crash_pos = PS2 crash
                                      nx,    ny,    nz,      # base_norm = PC VB
                                      wv[9], wv[10],wv[11])  # crash_norm= PS2 crash
                else:
                    # identity fallback: no crash deformation
                    missing += 1
                    out += struct.pack('<12f', px, py, pz, px, py, pz, nx, ny, nz, nx, ny, nz)

            if missing:
                print(f"  crash.dat: WARNING {node_name} surf{j}: "
                      f"{missing}/{nv} verts had no PS2 weight match (identity fallback used)")

    with open(output_path, 'wb') as f:
        f.write(out)

    src_sz = os.path.getsize(ps2_crash_path)
    out_sz = os.path.getsize(output_path)
    print(f"  crash.dat: PS2→PC  {src_sz:,} B → {out_sz:,} B  ({os.path.basename(output_path)})")


def _convert_xbox_crash_to_pc(xbox_crash_path, output_path, streams, surfaces, models):
    """Convert an Xbox crash.dat to PC FO2 format.

    Xbox crash.dat is PC-format (nv, nvb, vtx_vs16, weights×48) but with vs=16
    (NORMPACKED3) vertex buffers and PC float weights. We replace vtx_vs16 with
    the corresponding slice of the PC surface's vertex buffer (already decoded to
    vs=32 float format by xbox_to_pc), keeping the 48-byte weights unchanged.
    Position matching is used to align Xbox crash vertex order to PC vertex order.
    """
    if not os.path.exists(xbox_crash_path):
        print(f"  crash.dat: Xbox source not found ({os.path.basename(xbox_crash_path)}), skipping")
        return

    # parse Xbox crash.dat (same PC format, but vtx has vs=16)
    xb_nodes = _parse_crash_dat(xbox_crash_path, is_fouc=False)
    # xb_nodes: list of (name, surfs) where surf = {'vtx': bytes(vs16), 'wgt': bytes(48*nv), 'vs': 16}

    model_surf_map = {m['name']: m['surfaces'] for m in models}
    out = bytearray()
    out += struct.pack('<I', len(xb_nodes))

    for node_name, xb_surfs in xb_nodes:
        out += node_name.encode('ascii') + b'\x00'
        base_model = node_name.replace('_crash', '')
        surf_ids = model_surf_map.get(base_model)

        if surf_ids is None:
            print(f"  crash.dat: WARNING '{base_model}' not found, writing 0 surfaces")
            out += struct.pack('<I', 0); continue

        out += struct.pack('<I', len(surf_ids))

        for j, bgm_si in enumerate(surf_ids):
            if j >= len(xb_surfs) or bgm_si >= len(surfaces):
                out += struct.pack('<2I', 0, 0); continue

            surf  = surfaces[bgm_si]
            xb_s  = xb_surfs[j]
            vid   = surf['sids'][0] if surf['nst'] >= 1 else -1

            if vid < 0 or vid >= len(streams):
                out += struct.pack('<2I', 0, 0); continue

            sv    = streams[vid]; vs = sv['vs']
            vdata = sv['data'];   voff = surf['soffs'][0]; nv = surf['vc']

            vtx_slice = vdata[voff : voff + nv * vs]
            nvb = len(vtx_slice)

            # Build Xbox crash position → weight lookup
            xb_vs  = xb_s['vs']   # 16 for main, 12 for shadow
            xb_vtx = xb_s['vtx']
            xb_wgt = xb_s['wgt']
            xb_nv  = len(xb_wgt) // 48

            xb_pos_wgt = {}
            for vi in range(xb_nv):
                if xb_vs == 16:
                    px_i, py_i, pz_i = struct.unpack_from('<3h', xb_vtx, vi * 16)
                    # Use exact int16 keys:
                    # both Xbox crash VTX and the converted PC VB positions are 
                    # decoded from the same Xbox int16/1024 source, so
                    # they match exactly without any floating-point rounding tolerance.
                    key = (int(px_i), int(py_i), int(pz_i))
                elif xb_vs == 12:
                    px, py, pz = struct.unpack_from('<3f', xb_vtx, vi * 12)
                    key = (round(px * 1024), round(py * 1024), round(pz * 1024))
                else:
                    continue
                if key not in xb_pos_wgt:
                    xb_pos_wgt[key] = xb_wgt[vi * 48 : vi * 48 + 48]

            out += struct.pack('<2I', nv, nvb)
            out += vtx_slice

            missing = 0
            for vi in range(nv):
                px, py, pz = struct.unpack_from('<3f', vdata, voff + vi * vs)
                nx = ny = nz = 0.0
                if vs >= 24:
                    nx, ny, nz = struct.unpack_from('<3f', vdata, voff + vi * vs + 12)
                key = (round(px * 1024), round(py * 1024), round(pz * 1024))
                wgt_bytes = xb_pos_wgt.get(key)
                if wgt_bytes is not None:
                    # Keep crash_pos and crash_norm from Xbox crash weights,
                    # but replace base_pos/base_norm with the exact PC VB values.
                    # The export plugin always writes base_pos == vtx_data position
                    # exactly (same float), so the game can match by exact comparison.
                    # Xbox crash base_pos is int16/1024 decoded (slightly different).
                    xb_wgt = struct.unpack_from('<12f', wgt_bytes)
                    out += struct.pack('<12f',
                                      px, py, pz,            # base_pos  = PC VB exact
                                      xb_wgt[3], xb_wgt[4], xb_wgt[5],   # crash_pos from Xbox
                                      nx, ny, nz,            # base_norm = PC VB exact
                                      xb_wgt[9], xb_wgt[10], xb_wgt[11]) # crash_norm from Xbox
                else:
                    missing += 1
                    out += struct.pack('<12f', px, py, pz, px, py, pz, nx, ny, nz, nx, ny, nz)

            if missing:
                print(f"  crash.dat: WARNING {node_name} surf{j}: "
                      f"{missing}/{nv} verts had no Xbox weight match")

    with open(output_path, 'wb') as f:
        f.write(out)

    src_sz = os.path.getsize(xbox_crash_path)
    out_sz = os.path.getsize(output_path)
    print(f"  crash.dat: Xbox→PC  {src_sz:,} B → {out_sz:,} B  ({os.path.basename(output_path)})")


def _remap_crash_dat(crash_path, output_bgm, models,
                     surf_seen, surf_vs, surf_original_vc,
                     streams_orig, surfaces_orig,
                     is_fouc=False, is_standalone=False):
    """Remap crash.dat vertex arrays to match deduplicated vertex ordering.

    FO2 / FO1: remaps both the crash vbuf (vtx) and the 48-byte weight array.
    FOUC:       no vbuf; remaps only the 40-byte weight array.
    In both cases the vertex-to-new-index lookup uses the original BGM VB data
    (same raw-bytes key used by op_optimize's surf_seen dict).
    is_standalone=True means the source was a bare crash.dat (not <stem>_crash.dat).
    """

    crash_nodes    = _parse_crash_dat(crash_path, is_fouc=is_fouc)
    model_by_name  = {m['name']: m['surfaces'] for m in models}
    wgt_size       = 40 if is_fouc else 48

    new_nodes = []
    for node_name, surfs in crash_nodes:
        base = node_name.replace('_crash', '')
        if base not in model_by_name:
            print(f"  crash.dat: WARNING node '{node_name}' → model '{base}'"
                  f" not found, kept unchanged")
            new_nodes.append((node_name, surfs))
            continue

        bgm_surf_ids = model_by_name[base]
        new_surfs = []

        for j, crash_surf in enumerate(surfs):
            if j >= len(bgm_surf_ids):
                print(f"  crash.dat: WARNING node '{node_name}' surf {j}"
                      f" has no matching BGM surface, kept unchanged")
                new_surfs.append(crash_surf)
                continue

            si = bgm_surf_ids[j]
            if si not in surf_seen:
                new_surfs.append(crash_surf)
                continue

            seen        = surf_seen[si]
            vs          = surf_vs[si]
            original_vc = surf_original_vc[si]
            surf_o      = surfaces_orig[si]
            vid         = surf_o['sids'][0]
            voff        = surf_o['soffs'][0]
            vdata       = streams_orig[vid]['data']

            new_vc  = len(seen)
            old_wgt = crash_surf['wgt']

            if is_fouc:
                # FOUC: no vtx buffer; remap 40-byte weights only
                new_wgt = bytearray(new_vc * wgt_size)
                for old_i in range(original_vc):
                    if old_i >= len(old_wgt) // wgt_size:
                        break
                    rec = bytes(vdata[voff + old_i*vs : voff + old_i*vs + vs])
                    if rec not in seen:
                        continue
                    new_i = seen[rec]
                    new_wgt[new_i*wgt_size : new_i*wgt_size + wgt_size] = \
                        old_wgt[old_i*wgt_size : old_i*wgt_size + wgt_size]
                new_surfs.append({'vtx': b'', 'wgt': bytes(new_wgt), 'vs': 0})
            else:
                # FO2 / FO1: remap both vtx buffer and 48-byte weights
                new_vtx = bytearray(new_vc * vs)
                new_wgt = bytearray(new_vc * wgt_size)
                old_vtx = crash_surf['vtx']
                for old_i in range(original_vc):
                    if old_i >= len(old_wgt) // wgt_size:
                        break
                    rec = bytes(vdata[voff + old_i*vs : voff + old_i*vs + vs])
                    if rec not in seen:
                        continue
                    new_i = seen[rec]
                    new_vtx[new_i*vs  : new_i*vs + vs]               = \
                        old_vtx[old_i*vs  : old_i*vs + vs]
                    new_wgt[new_i*wgt_size : new_i*wgt_size + wgt_size] = \
                        old_wgt[old_i*wgt_size : old_i*wgt_size + wgt_size]
                new_surfs.append({'vtx': bytes(new_vtx),
                                  'wgt': bytes(new_wgt), 'vs': vs})

        new_nodes.append((node_name, new_surfs))

    dst = _crash_dst_path(output_bgm, is_standalone)
    _write_crash_dat(dst, new_nodes, is_fouc=is_fouc)
    old_sz = os.path.getsize(crash_path)
    new_sz = os.path.getsize(dst)
    print(f"  crash.dat: {old_sz:,} B → {new_sz:,} B  ({os.path.basename(dst)})")



# ─────────────────────────────────────────────────────────────────────────────
# PC OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def op_strip(streams, surfaces):
    """Remove unreferenced streams.  Returns (new_streams, new_surfaces).
    Mutates surfaces in-place (sids remapped); returns a new stream list."""
    referenced = set()
    for s in surfaces:
        for sid in s['sids']:
            referenced.add(sid)

    kept    = sorted(referenced)
    dropped = [i for i in range(len(streams)) if i not in referenced]

    print(f"\n── Clean orphan streams ──")
    print(f"  Streams total:          {len(streams)}")
    print(f"  Referenced (kept):      {len(kept)}")
    print(f"  Unreferenced (dropped): {len(dropped)}")

    if not dropped:
        print("  Nothing to remove — already clean.")
        return streams, surfaces

    dropped_bytes = sum(len(streams[i]['data']) +
                        (12 if streams[i]['dt'] == 1 else
                         4  if streams[i]['dt'] == 2 else 8)
                        for i in dropped)
    print(f"  Bytes removed:          {dropped_bytes:,}")

    old_to_new = {old: new for new, old in enumerate(kept)}
    new_streams = [streams[i] for i in kept]
    for s in surfaces:
        s['sids'] = [old_to_new[sid] for sid in s['sids']]

    return new_streams, surfaces


def op_optimize(streams, surfaces):
    """Vertex deduplication and stream merging.

    Returns (new_streams, new_surfaces, surf_seen, surf_vs,
             surf_original_vc, streams_snapshot, surfaces_snapshot)
    The last three items are snapshots taken *before* mutation, needed for
    crash.dat remapping.
    """
    # Snapshots before any mutation
    streams_orig  = copy.deepcopy(streams)
    surfaces_orig = copy.deepcopy(surfaces)

    surf_unique_vdata = {}
    surf_new_indices  = {}
    surf_vs           = {}
    surf_vflags       = {}
    surf_seen         = {}
    surf_original_vc  = {}

    total_raw   = 0
    total_dedup = 0

    print(f"\n── Optimize: vertex deduplication ──")

    for si, s in enumerate(surfaces):
        if s['nst'] < 1:
            continue

        if s['nst'] < 2:
            # vbuf-only surface (shadow): dedup the vertex buffer directly
            vid   = s['sids'][0]
            sv    = streams[vid]
            vs    = sv['vs']
            vdata = sv['data']
            voff  = s['soffs'][0]
            seen = {}; unique = bytearray()
            for i in range(s['vc']):
                rec = bytes(vdata[voff + i*vs : voff + i*vs + vs])
                if rec not in seen:
                    seen[rec] = len(seen)
                    unique += rec
            surf_unique_vdata[si] = bytes(unique)
            surf_new_indices[si]  = None
            surf_vs[si]           = vs
            surf_vflags[si]       = sv['flags']
            surf_seen[si]         = seen
            surf_original_vc[si]  = s['vc']
            total_raw   += s['vc'] * vs
            total_dedup += len(unique)
            s['vc'] = len(unique) // vs
            continue

        vid   = s['sids'][0]
        iid   = s['sids'][1]
        sv    = streams[vid]
        sv2   = streams[iid]
        vs    = sv['vs']
        vdata = sv['data']
        idata = sv2['data']
        voff  = s['soffs'][0]
        ioff  = s['soffs'][1]
        vbase = voff // vs if vs > 0 else 0

        surf_original_vc[si] = s['vc']

        raw_idx   = [struct.unpack_from('<H', idata, ioff + i*2)[0]
                     for i in range(s['niu'])]
        local_idx = [x - vbase for x in raw_idx]

        seen = {}; unique = bytearray(); new_idx = []
        for li in local_idx:
            if li < 0 or li >= s['vc']:
                new_idx.append(0)
                continue
            rec = bytes(vdata[voff + li*vs : voff + li*vs + vs])
            if rec not in seen:
                seen[rec] = len(seen)
                unique += rec
            new_idx.append(seen[rec])

        surf_unique_vdata[si] = bytes(unique)
        surf_new_indices[si]  = new_idx
        surf_vs[si]           = vs
        surf_vflags[si]       = sv['flags']
        surf_seen[si]         = seen
        total_raw   += s['vc'] * vs
        total_dedup += len(unique)

        n_before = s['vc']
        n_after  = len(unique) // vs
        print(f"  S{si:>2d}  {n_before:>5} → {n_after:>5} verts"
              f"  (-{n_before-n_after:>5}, {(n_before-n_after)/n_before*100:.0f}%)")
        s['vc'] = n_after

    print(f"\n  vbuf: {total_raw:>10,} B → {total_dedup:>10,} B"
          f"  (saved {total_raw-total_dedup:,} B,"
          f" {(total_raw-total_dedup)/total_raw*100:.1f}%)")

    print(f"\n── Optimize: stream merging ──")

    # one merged vbuf per (vs, flags) format group, one shared ibuf
    # also track the fc (foucExtraFormat) value for each group — FOUC streams
    # have fc > 0 (typically 22) and this must be preserved so that FOUC
    # detection still works when the merged file is read back.
    fmt_order   = []
    fmt_seen_set = set()
    fmt_fc      = {}   # key -> fc value to use for the merged stream
    for si in range(len(surfaces)):
        if si not in surf_vs:
            continue
        vid = surfaces[si]['sids'][0] if surfaces[si]['nst'] >= 1 else None
        fc  = streams_orig[vid]['fc'] if vid is not None and vid < len(streams_orig) else 0
        key = (surf_vs[si], surf_vflags[si])
        if key not in fmt_seen_set:
            fmt_seen_set.add(key)
            fmt_order.append(key)
            fmt_fc[key] = fc
        elif fc > 0:
            # prefer non-zero fc if any surface in this group has one
            fmt_fc[key] = fc

    merged_vbufs  = {key: bytearray() for key in fmt_order}
    surf_vbuf_off = {}

    for si, s in enumerate(surfaces):
        if si not in surf_vs:
            continue
        key = (surf_vs[si], surf_vflags[si])
        surf_vbuf_off[si] = len(merged_vbufs[key])
        merged_vbufs[key] += surf_unique_vdata[si]

    merged_ibuf  = bytearray()
    surf_ibuf_off = {}

    for si, s in enumerate(surfaces):
        if si not in surf_new_indices or surf_new_indices[si] is None:
            continue
        vs       = surf_vs[si]
        abs_base = surf_vbuf_off[si] // vs
        surf_ibuf_off[si] = len(merged_ibuf)
        for li in surf_new_indices[si]:
            merged_ibuf += struct.pack('<H', abs_base + li)

    new_streams = []
    fmt_to_idx  = {}
    for key in fmt_order:
        vs, fl = key
        buf = merged_vbufs[key]
        vc  = len(buf) // vs
        idx = len(new_streams)
        fmt_to_idx[key] = idx
        new_streams.append({'dt': 1, 'fc': fmt_fc.get(key, 0), 'vc': vc, 'vs': vs,
                             'flags': fl, 'data': bytes(buf)})
        print(f"  vbuf stream {idx}: vs={vs} flags=0x{fl:04X}"
              f"  {vc:,} verts  {len(buf):,} B")

    ibuf_idx = len(new_streams)
    ic = len(merged_ibuf) // 2
    new_streams.append({'dt': 2, 'fc': 0, 'ic': ic, 'data': bytes(merged_ibuf)})
    print(f"  ibuf stream {ibuf_idx}: {ic:,} indices  {len(merged_ibuf):,} B")
    print(f"\n  {len(streams)} streams → {len(new_streams)} streams")

    # update surface stream references
    for si, s in enumerate(surfaces):
        if si not in surf_vs:
            continue
        key         = (surf_vs[si], surf_vflags[si])
        vstream_idx = fmt_to_idx[key]
        vbuf_off    = surf_vbuf_off[si]
        if surf_new_indices[si] is not None:
            s['sids']  = [vstream_idx, ibuf_idx]
            s['soffs'] = [vbuf_off, surf_ibuf_off[si]]
            s['niu']   = len(surf_new_indices[si])
        else:
            s['sids']  = [vstream_idx]
            s['soffs'] = [vbuf_off]

    return (new_streams, surfaces,
            surf_seen, surf_vs, surf_original_vc,
            streams_orig, surfaces_orig)


def _menucar_sort_key(name):
    n = name.lower()
    if n.startswith('body'):                                         return 0
    if n.startswith('common'):                                       return 1
    if n.startswith('interior'):                                     return 2
    if n.startswith('window'):                                       return 3
    if n.startswith('light'):                                        return 4
    if n in ('shear', 'shearhock', 'shearshock', 'shearspring'):    return 5
    if n.startswith('scalespring') or n.startswith('shearspring'):  return 6
    if n.startswith('scaleshock') or n.startswith('shearshock') \
            or n.startswith('shearhock'):                            return 7
    if n.startswith('scale') or n.startswith('shock') \
            or n.startswith('spring'):                               return 8
    if n.startswith('tire'):                                         return 9
    if n.startswith('rim'):                                          return 10
    if n.startswith('ground'):                                       return 10  # same group as rim, stable sort preserves relative order
    return 99


def op_menucar(surfaces, models, materials_raw, version):
    """Reorder surfaces within each model to match FO2 menucar draw order.

    Surface ordering (stable within each group):
      body -> common -> interior -> window* -> light* ->
      shear* -> scalespring* -> scaleshock* -> scale*/shock*/spring* ->
      tire* -> rim* -> ground* -> (everything else)

    Surfaces not owned by any model are left in place at the end.
    """
    import struct

    mat_names = [m['name'] for m in materials_raw]

    print(f"\n── Menucar surface ordering ──")

    # build set of surface indices owned by at least one model
    owned = set()
    for m in models:
        owned.update(m['surfaces'])

    # for each model, stable-sort its surface list by material name priority
    total_moved = 0
    for mi, m in enumerate(models):
        orig = list(m['surfaces'])
        # Sort by (priority, original_position) for stable sort within group
        sorted_surfs = sorted(
            orig,
            key=lambda si: _menucar_sort_key(
                mat_names[surfaces[si]['mid']] if surfaces[si]['mid'] < len(mat_names) else ''
            )
        )
        moved = sum(1 for a, b in zip(orig, sorted_surfs) if a != b)
        total_moved += moved
        if moved:
            old_names = [mat_names[surfaces[si]['mid']] for si in orig]
            new_names = [mat_names[surfaces[si]['mid']] for si in sorted_surfs]
            print(f"  Model '{m['name']}': reordered {moved} surfaces")
            for oi, (o, n) in enumerate(zip(old_names, new_names)):
                if o != n:
                    print(f"    [{oi}] {o} → {n}")
        m['surfaces'] = sorted_surfs

    if total_moved == 0:
        print("  Already in correct order — nothing to do.")
    else:
        print(f"  Total surfaces reordered: {total_moved}")

    # rebuild the flat surfaces list so its order matches the model surface lists
    # surfaces appear in the order they are first referenced by a model, unowned surfaces are appended at the end

    new_order = []
    seen = set()
    for m in models:
        for si in m['surfaces']:
            if si not in seen:
                new_order.append(si)
                seen.add(si)
    for si in range(len(surfaces)):
        if si not in seen:
            new_order.append(si)

    # build old->new index map and remap model surface lists

    old_to_new = {old: new for new, old in enumerate(new_order)}
    new_surfaces = [surfaces[i] for i in new_order]
    for m in models:
        m['surfaces'] = [old_to_new[si] for si in m['surfaces']]

    return new_surfaces, models


def op_lightorder(surfaces, models, materials_raw):
    """Reorder materials and surfaces by draw priority (mirrors fo2_bgm_export).

    Two-pass operation matching the export plugin behaviour:

    Pass 1 — Material reordering:
      Sort the materials_raw list by MATERIAL_PRIORITIES[name] (stable sort,
      unlisted materials stay at priority 0).  Remap every surface's 'mid'
      field to point at the new material indices.

    Pass 2 — Surface reordering within each model:
      For each model, stable-sort its surface list so surfaces with lower-
      priority materials come first (mirrors mesh_sort_key in the exporter).
    """
    print(f"\n── Light order ──")

    # Pass 1: sort materials and remap surface mid references
    old_order = list(range(len(materials_raw)))
    old_order.sort(key=lambda i: MATERIAL_PRIORITIES.get(
        materials_raw[i]['name'].lower(), 0))

    mat_moved = sum(1 for new, old in enumerate(old_order) if new != old)

    # build old-index → new-index map and reorder
    old_to_new_mid = {old: new for new, old in enumerate(old_order)}
    materials_raw[:] = [materials_raw[i] for i in old_order]

    # remap all surface mid references
    for s in surfaces:
        s['mid'] = old_to_new_mid.get(s['mid'], s['mid'])

    if mat_moved:
        print(f"  Materials reordered: {mat_moved}")
        for new_i, old_i in enumerate(old_order):
            if new_i != old_i:
                print(f"    [{old_i}] → [{new_i}]  {materials_raw[new_i]['name']}")
    else:
        print("  Materials: already in correct order")

    # Pass 2: sort surfaces within each model by material priority
    mat_names = [m['name'] for m in materials_raw]
    surf_moved = 0
    for m in models:
        orig = list(m['surfaces'])
        sorted_surfs = sorted(
            orig,
            key=lambda si: MATERIAL_PRIORITIES.get(
                mat_names[surfaces[si]['mid']].lower()
                if surfaces[si]['mid'] < len(mat_names) else '', 0)
        )
        moved = sum(1 for a, b in zip(orig, sorted_surfs) if a != b)
        surf_moved += moved
        if moved:
            print(f"  Model '{m['name']}': reordered {moved} surface(s)")
        m['surfaces'] = sorted_surfs

    # rebuild flat surfaces list in model-reference order (mirrors op_menucar)
    new_order = []
    seen = set()
    for m in models:
        for si in m['surfaces']:
            if si not in seen:
                new_order.append(si)
                seen.add(si)
    for si in range(len(surfaces)):
        if si not in seen:
            new_order.append(si)

    old_to_new_si = {old: new for new, old in enumerate(new_order)}
    new_surfaces = [surfaces[i] for i in new_order]
    for m in models:
        m['surfaces'] = [old_to_new_si[si] for si in m['surfaces']]

    if surf_moved == 0:
        print("  Surfaces: already in correct order")
    else:
        print(f"  Total surfaces reordered: {surf_moved}")

    return new_surfaces, models, materials_raw


def _is_lighthacks_target(mat_name, targets):
    """Return True if mat_name should be processed by -lighthacks.

    targets=None  : any material starting with "light" and ending with "_b".
    targets=set() : exact match against the set (case-insensitive).
    """
    n = mat_name.lower()
    if targets is None:
        return n.startswith('light') and n.endswith('_b')
    return n in {t.lower() for t in targets}


def op_lighthacks(streams, surfaces, models, materials_raw, objects, targets=None):
    """List light materials and objects, display _b surfaces, then duplicate
    and remap to common.

    targets=None  : duplicate all surfaces whose material starts with "light"
                    and ends with "_b".
    targets=set() : duplicate surfaces whose material name is in the set
                    (exact match, case-insensitive).

    Returns (streams, surfaces, b_dupe_map) where b_dupe_map is used by
    main() to mirror the duplicates in crash.dat:
      b_dupe_map: list of (model_name, orig_pos_in_model) for each duplicate.
    """
    import copy as _copy
    from collections import defaultdict

    mat_names = [m['name'] for m in materials_raw]

    # light materials
    print(f"\n── Light hacks: light materials ──")
    light_mats = [m['name'] for m in materials_raw if m['name'].lower().startswith('light')]
    if light_mats:
        for name in light_mats:
            print(f"  {name}")
        print(f"\n  {len(light_mats)} material(s)")
    else:
        print("  None found.")

    # light objects (dummies)
    print(f"\n── Light hacks: light objects ──")
    light_objs = [o['name1'] for o in objects if o['name1'].lower().startswith('light')]
    if light_objs:
        for name in light_objs:
            print(f"  {name}")
        print(f"\n  {len(light_objs)} object(s)")
    else:
        print("  None found.")

    # _b surface check
    print(f"\n── Light hacks: _b surface check ──")

    # build model name -> surface-index list map and reverse lookup
    model_surfs   = defaultdict(list)
    surf_to_model = {}
    for m in models:
        for sid in m['surfaces']:
            model_surfs[m['name']].append(sid)
            surf_to_model[sid] = m['name']

    b_hits = []   # (model_name, mat_name, surf_index)
    for si, s in enumerate(surfaces):
        name = mat_names[s['mid']] if s['mid'] < len(mat_names) else ''
        if _is_lighthacks_target(name, targets):
            b_hits.append((surf_to_model.get(si, '(unowned)'), name, si))

    if not b_hits:
        print("  No light _b surfaces found.")
        return streams, surfaces, []

    by_model = defaultdict(list)
    for model_name, mat_name, si in b_hits:
        by_model[model_name].append(mat_name)
    for model_name, mats in sorted(by_model.items()):
        print(f"  {model_name}")
        for mat in mats:
            print(f"    {mat}")
    print(f"\n  {len(b_hits)} surface(s) across {len(by_model)} model(s)")

    # find "common" material index
    common_mid = next(
        (i for i, m in enumerate(materials_raw) if m['name'].lower() == 'common'),
        None
    )
    if common_mid is None:
        print("\n── Light hacks: duplication ──")
        print("  No 'common' material found — skipping.")
        return streams, surfaces, []

    # duplicate each _b surface, remapped to common
    #
    # for each light_*_b surface:
    # - copy VB stream verbatim -> new stream entry
    # - copy IB stream verbatim -> new stream entry
    # - build duplicate surface dict with mid = common_mid
    # - append to surfaces list and owning model's surface list
    #
    # b_dupe_map records (model_name, orig_pos_in_model) for each duplicate
    # so main() can mirror the operation in crash.dat

    print(f"\n── Light hacks: duplication ──")

    b_dupe_map = []   # [(model_name, orig_pos_in_model), ...]

    for model_name, mat_name, si in b_hits:
        s = surfaces[si]

        # copy VB stream
        src_vb     = streams[s['sids'][0]]
        new_vb_sid = len(streams)
        streams.append(dict(src_vb, data=bytes(src_vb['data'])))

        # copy IB stream (if present)
        new_sids  = [new_vb_sid]
        new_soffs = [0]
        if len(s['sids']) >= 2:
            src_ib     = streams[s['sids'][1]]
            new_ib_sid = len(streams)
            streams.append(dict(src_ib, data=bytes(src_ib['data'])))
            new_sids.append(new_ib_sid)
            new_soffs.append(0)

        # build duplicate surface remapped to common
        new_surf          = _copy.copy(s)
        new_surf['mid']   = common_mid
        new_surf['sids']  = new_sids
        new_surf['soffs'] = new_soffs
        new_surf['nst']   = len(new_sids)

        new_si = len(surfaces)
        surfaces.append(new_surf)

        # append to owning model and record position for crash.dat mirroring
        for m in models:
            if si in m['surfaces']:
                orig_pos = m['surfaces'].index(si)
                m['surfaces'].append(new_si)
                b_dupe_map.append((m['name'], orig_pos))
                break

        print(f"  {model_name}: {mat_name} → common")

    print(f"\n  {len(b_hits)} surface(s) duplicated")

    return streams, surfaces, b_dupe_map


def op_windflip(streams, surfaces, is_fouc, models=None):
    """Fix triangles whose geometric winding disagrees with their stored vertex normal.

    For each triangle: compute the geometric face normal (cross product of edges
    after applying our import winding reversal), dot it against the stored vertex
    normal.  If the dot product is negative the winding is wrong — swap the two
    non-anchor index buffer entries to correct it.

    Vertex normal decoding:
      FOUC: uint8[4] at vbuf offset 16; [0]=FO2.z, [1]=FO2.y, [2]=FO2.x
            formula: (byte / 127.0) - 1.0
      FO2 / FO1: float[3] at vbuf offset 12 (when VERTEX_NORMAL flag 0x10 set)

    Position decoding:
      FOUC: int16[3] at offset 0, multiplied by foucVertexMultiplier[3] (from
            surface extra bytes), offset by foucVertexMultiplier[0,1,2].
      FO2 / FO1: float[3] at offset 0.

    The IB stream data is converted to mutable bytearrays in-place.
    Vertex buffer data is not modified.
    """
    VERTEX_NORMAL = 0x10

    # make all IB streams mutable so we can patch individual index entries
    for s in streams:
        if s['dt'] == 2 and not isinstance(s['data'], bytearray):
            s['data'] = bytearray(s['data'])

    def _cross(a, b):
        return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
    def _dot(a, b):
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
    def _sub(a, b):
        return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

    total_flipped = 0
    surface_stats = []

    for si, s in enumerate(surfaces):
        if s['nst'] < 2 or s['pc'] == 0:
            continue
        if s['pm'] != 4:
            # only triangle lists (pm=4) supported; strips rare in practice
            continue

        vid = s['sids'][0]; iid = s['sids'][1]
        if vid >= len(streams) or iid >= len(streams):
            continue
        sv = streams[vid]
        si_buf = streams[iid]
        if sv['dt'] not in (1, 3) or si_buf['dt'] != 2:
            continue

        vs     = sv['vs']
        vdata  = sv['data']
        idata  = si_buf['data']   # mutable bytearray
        voff   = s['soffs'][0]
        ioff   = s['soffs'][1]    # byte offset into IB
        vbase  = voff // vs if vs > 0 else 0

        # decode foucVertexMultiplier from surface extra bytes (FOUC only)
        if is_fouc and len(s['extra']) >= 16:
            mult = struct.unpack_from('<4f', s['extra'], 0)
            ox, oy, oz, scale = mult[0], mult[1], mult[2], mult[3]
            if scale == 0.0:
                scale = 1.0 / 1024.0
        else:
            ox = oy = oz = 0.0; scale = 1.0

        def get_pos_nrm(abs_idx):
            """Return (pos, nrm) in Blender space (Y↔Z swapped from FO2) for absolute vertex index."""
            off = abs_idx * vs
            if is_fouc:
                px, py, pz = struct.unpack_from('<3h', vdata, off)
                # FO2→Blender: (x,y,z)→(x,z,y)
                pos = ((px + ox) * scale,
                       (pz + oz) * scale,
                       (py + oy) * scale)
                nb = struct.unpack_from('<4B', vdata, off + 16)
                # buffer[0]=FO2.z  [1]=FO2.y  [2]=FO2.x, then swap to Blender
                nx = (nb[2] / 127.0) - 1.0  # FO2.x
                ny = (nb[1] / 127.0) - 1.0  # FO2.y
                nz = (nb[0] / 127.0) - 1.0  # FO2.z
                nrm = (nx, nz, ny)           # Blender: (FO2.x, FO2.z, FO2.y)
            else:
                fx, fy, fz = struct.unpack_from('<3f', vdata, off)
                pos = (fx, fz, fy)           # FO2→Blender Y↔Z swap
                if sv['flags'] & VERTEX_NORMAL:
                    nx, ny, nz = struct.unpack_from('<3f', vdata, off + 12)
                    nrm = (nx, nz, ny)
                else:
                    nrm = (0.0, 0.0, 1.0)   # no stored normal — can't determine
            return pos, nrm

        flipped = 0
        for t in range(s['pc']):
            base = ioff + t * 6
            i0, i1, i2 = struct.unpack_from('<3H', idata, base)

            # our importer reverses: file (i0,i1,i2) -> render (i2,i1,i0)
            ri0, ri1, ri2 = i2, i1, i0

            if not (0 <= ri0 - vbase < s['vc'] and
                    0 <= ri1 - vbase < s['vc'] and
                    0 <= ri2 - vbase < s['vc']):
                continue

            p0, n0 = get_pos_nrm(ri0)
            p1, _  = get_pos_nrm(ri1)
            p2, _  = get_pos_nrm(ri2)

            e1 = _sub(p1, p0); e2 = _sub(p2, p0)
            gn = _cross(e1, e2)
            L  = (gn[0]**2 + gn[1]**2 + gn[2]**2) ** 0.5
            if L < 1e-12:
                continue

            if _dot(gn, n0) < 0:
                # geometric normal disagrees with stored normal
                # flip by swapping i1 and i2 in the file's IB entry
                # file entry (i0,i1,i2) -> reversed (i2,i1,i0)

                # swapping file i1<->i2 gives file (i0,i2,i1) -> reversed (i1,i2,i0),
                # which has the opposite geometric orientation
                struct.pack_into('<3H', idata, base, i0, i2, i1)
                flipped += 1

        surface_stats.append((si, flipped, s['pc']))
        total_flipped += flipped

    print(f"\n── Winding flip ──")
    print(f"  Format: {'FOUC' if is_fouc else 'FO2/FO1'}")

    # Build surface→model name map
    surf_to_model = {}
    if models:
        for m in models:
            for sid in m['surfaces']:
                surf_to_model[sid] = m['name']

    # Aggregate flipped counts per model
    model_flipped = {}   # model_name -> (total_flipped, total_tris)
    unowned_flipped = 0; unowned_tris = 0
    for si, flipped, pc in surface_stats:
        mname = surf_to_model.get(si)
        if mname:
            mf, mt = model_flipped.get(mname, (0, 0))
            model_flipped[mname] = (mf + flipped, mt + pc)
        else:
            unowned_flipped += flipped
            unowned_tris    += pc

    if total_flipped == 0:
        print("  No winding errors found — nothing to flip.")
    else:
        # Print models that had at least one flip, sorted descending by count
        model_lines = [(name, f, t) for name, (f, t) in model_flipped.items() if f > 0]
        model_lines.sort(key=lambda x: -x[1])
        for name, f, t in model_lines:
            print(f"  {name}: {f} / {t} tris ({f/t*100:.1f}%)")
        if unowned_flipped:
            print(f"  (unowned surfaces): {unowned_flipped} / {unowned_tris} tris")

    print(f"  Total: {total_flipped} tris flipped across {sum(1 for _,f,_ in surface_stats if f>0)} surfaces")

    # convert mutable IB bytearrays back to bytes and update ic counts
    for s in streams:
        if s['dt'] == 2 and isinstance(s['data'], bytearray):
            s['data'] = bytes(s['data'])

    return streams, surfaces



# ─────────────────────────────────────────────────────────────────────────────
# PC FORMAT CONVERSION  (FO1 / FO2 / FOUC)
# ─────────────────────────────────────────────────────────────────────────────

FOUC_SCALE     = 1.0 / 1024.0   # int16 * this = metres (FOUC default)

FOUC_SCALE_INV = 1024.0

FOUC_VERTEX_FLAGS = 0x2242       # INT16 | UV2 | COLOR | POSITION

FO2_VERTEX_FLAGS_FULL = 0x0152   # POSITION | NORMAL | COLOR | UV  (36 B)

VERTEX_POSITION = 0x0002

VERTEX_NORMAL   = 0x0010

VERTEX_COLOR    = 0x0040

VERTEX_UV       = 0x0100

FO2_TO_FOUC_SHADER = {
    38: 39,   # Ghost Body     → Ghost Body
    39: 40,   # Static Nonlit  → Static Nonlit
    40: 41,   # Dynamic Nonlit → Dynamic Nonlit
    41: 50,   # Racemap        → Racemap
}

FOUC_TO_FO2_SHADER = {
    38: 15,   # Horizon              → Default
    39: 38,   # Ghost Body           → Ghost Body
    40: 39,   # Static Nonlit        → Static Nonlit
    41: 40,   # Dynamic Nonlit       → Dynamic Nonlit
    42: 15,   # Skid Marks           → Default
    43:  7,   # Car Interior         → Car Diffuse
    44:  9,   # Car Tire             → Car Tire
    45: 15,   # Puddle               → Default
    46: 15,   # Ambient Shadow       → Default
    47: 25,   # Local Water          → Water
    48: 15,   # Static Specular      → Default
    49: 15,   # Lightmapped Refl.    → Default
    50: 41,   # Racemap              → Racemap
    51: 15,   # HDR Default          → Default
    52: 15,   # Ambient Particle     → Default
    53: 22,   # Videoscreen Dynamic  → Particle
    54: 22,   # Videoscreen Static   → Particle
}

LIGHT_SHADER_IDS = {10, 14}

V92_FO1   = 0   # FO1: v92 = 0 for lights

V92_FO2UC = 2   # FO2 / FOUC: v92 = 2 for lights


def _fo2_stride(flags):
    """Return the FO2 vertex stride in bytes for the given flags."""
    s = 12  # position always present (3 floats)
    if flags & VERTEX_NORMAL: s += 12
    if flags & VERTEX_COLOR:  s += 4
    if flags & VERTEX_UV:     s += 8
    return s


def _fouc_vert_to_fo2(vdata, vcount, mult):
    """Convert FOUC 32-byte vertices to FO2 float vertices (flags 0x0152, 36 B).

    FOUC layout (32 bytes per vertex):
      int16[3] pos + uint16 pad
      uint8[4] tangents   (offset  8)
      uint8[4] bitangents (offset 12)
      uint8[4] normals    (offset 16)  [0]=FO2.z [1]=FO2.y [2]=FO2.x
      uint8[4] colors     (offset 20)  RGBA
      int16[2] UV1        (offset 24)
      int16[2] UV2        (offset 28)

    Output FO2 layout (36 bytes per vertex, flags 0x0152):
      float[3] pos   (12 B)
      float[3] nrm   (12 B)
      uint32   color  (4 B)   packed RGBA
      float[2] uv     (8 B)
    """
    ox, oy, oz = mult[0], mult[1], mult[2]
    sc = mult[3] if mult[3] != 0.0 else FOUC_SCALE

    out = bytearray()
    for i in range(vcount):
        base = i * 32
        px, py, pz = struct.unpack_from('<3h', vdata, base)
        pos_x = (px + ox) * sc
        pos_y = (py + oy) * sc
        pos_z = (pz + oz) * sc
        out += struct.pack('<3f', pos_x, pos_y, pos_z)

        nb = struct.unpack_from('<4B', vdata, base + 16)
        # buffer: [0]=FO2.z [1]=FO2.y [2]=FO2.x  formula: (b/127)-1
        nx = (nb[2] / 127.0) - 1.0
        ny = (nb[1] / 127.0) - 1.0
        nz = (nb[0] / 127.0) - 1.0
        out += struct.pack('<3f', nx, ny, nz)

        col = struct.unpack_from('<4B', vdata, base + 20)
        # FO2 color is stored as RGBA uint32 little-endian: R in byte 0
        out += struct.pack('<4B', col[0], col[1], col[2], col[3])

        uv = struct.unpack_from('<2h', vdata, base + 24)
        out += struct.pack('<2f', uv[0] / 2048.0, uv[1] / 2048.0)

    return bytes(out), FO2_VERTEX_FLAGS_FULL, 36


def _fo2_vert_to_fouc(vdata, vcount, vs, flags, voff=0):
    """Convert FO2 float vertices to FOUC 32-byte int16 vertices.

    Returns (new_vbuf_bytes, multiplier_4f).
    multiplier = [0, 0, 0, FOUC_SCALE] (default scale 1/1024).
    """
    mult = (0.0, 0.0, 0.0, FOUC_SCALE)

    def _enc_nrm(v):
        return max(0, min(255, int(round((v + 1.0) * 127.0))))

    out = bytearray()
    for i in range(vcount):
        base = voff + i * vs
        px, py, pz = struct.unpack_from('<3f', vdata, base)
        # float -> int16 via scale 1024; clamp to int16 range
        ix = max(-32767, min(32767, int(round(px * FOUC_SCALE_INV))))
        iy = max(-32767, min(32767, int(round(py * FOUC_SCALE_INV))))
        iz = max(-32767, min(32767, int(round(pz * FOUC_SCALE_INV))))

        cur = 3  # float offset after position
        if flags & VERTEX_NORMAL:
            nx, ny, nz = struct.unpack_from('<3f', vdata, base + cur * 4)
            cur += 3
        else:
            nx = ny = nz = 0.0

        if flags & VERTEX_COLOR:
            col = struct.unpack_from('<4B', vdata, base + cur * 4)
            cur += 1
        else:
            col = (255, 255, 255, 255)

        if flags & VERTEX_UV:
            fu, fv = struct.unpack_from('<2f', vdata, base + cur * 4)
        else:
            fu = fv = 0.0

        iu = max(-32767, min(32767, int(round(fu * 2048.0))))
        iv = max(-32767, min(32767, int(round(fv * 2048.0))))

        # normals: buffer [0]=FO2.z [1]=FO2.y [2]=FO2.x
        bn0 = _enc_nrm(nz)
        bn1 = _enc_nrm(ny)
        bn2 = _enc_nrm(nx)

        out += struct.pack('<3hH4B4B4B4B2h2h',
            ix, iy, iz, 0,
            128, 128, 128, 255,      # tangents (neutral)
            128, 128, 128, 255,      # bitangents (neutral)
            bn0, bn1, bn2, 255,      # normals [z,y,x,pad]
            col[0], col[1], col[2], col[3],  # colors RGBA
            iu, iv,                  # UV1
            0, 0,                    # UV2
        )

    return bytes(out), mult


def _compute_surface_aabb(streams, surface):
    """Compute FO2-space AABB for a surface's vertex range.
    Returns (center_3f, radius_3f) suitable for FO1 surface extra bytes."""
    if not surface['sids']:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    vid   = surface['sids'][0]
    if vid >= len(streams):
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    sv    = streams[vid]
    vs    = sv['vs']
    vdata = sv['data']
    voff  = surface['soffs'][0]
    vc    = surface['vc']

    mn = [1e18, 1e18, 1e18]
    mx = [-1e18, -1e18, -1e18]
    for i in range(vc):
        px, py, pz = struct.unpack_from('<3f', vdata, voff + i * vs)
        for k, v in enumerate((px, py, pz)):
            if v < mn[k]: mn[k] = v
            if v > mx[k]: mx[k] = v

    center = ((mn[0]+mx[0])*0.5, (mn[1]+mx[1])*0.5, (mn[2]+mx[2])*0.5)
    radius = (abs(mx[0]-mn[0])*0.5, abs(mx[1]-mn[1])*0.5, abs(mx[2]-mn[2])*0.5)
    return center, radius


def _remap_shaders(materials_raw, src_is_fouc, src_is_fo1,
                   dst_is_fouc, dst_is_fo1, version):
    """Remap nShaderId and v92 in all materials for a format conversion.

    Returns a new shallow-copied list; originals are not modified.
    No-op for pre-0x10004 files (they have no shader_id / v92 fields).
    """
    import copy as _c
    materials_raw = [_c.copy(m) for m in materials_raw]

    if version < 0x10004:
        return materials_raw

    if src_is_fouc and not dst_is_fouc:
        shader_map = FOUC_TO_FO2_SHADER
    elif not src_is_fouc and dst_is_fouc:
        shader_map = FO2_TO_FOUC_SHADER
    else:
        shader_map = {}   # FO1<->FO2: same IDs, only v92 may change

    remapped = 0
    v92_fixed = 0
    for m in materials_raw:
        old_sid = m['shader_id']
        new_sid = shader_map.get(old_sid, old_sid)
        if new_sid != old_sid:
            m['shader_id'] = new_sid
            remapped += 1

        if m['shader_id'] in LIGHT_SHADER_IDS:
            target_v92 = V92_FO1 if dst_is_fo1 else V92_FO2UC
            if m['v92'] != target_v92:
                m['v92'] = target_v92
                v92_fixed += 1

    if remapped:
        print(f"  Shader IDs remapped: {remapped} material(s)")
    if v92_fixed:
        print(f"  Light v92 updated:   {v92_fixed} material(s) "
              f"({'→ 0 (FO1)' if dst_is_fo1 else '→ 2 (FO2/FOUC)'})")

    return materials_raw


def _convert_crash_dat_file(crash_path, output_path,
                             src_is_fouc, dst_is_fouc,
                             surfaces, streams_src, streams_dst,
                             models=None, surfaces_dst=None):
    """Convert a crash.dat file between FO2/FO1 and FOUC formats.

    FO2/FO1 crash surface:
      uint32 nv + uint32 nvb + vbuf[nvb] + weights[nv x 48]
      weights: base_pos[3f] crash_pos[3f] base_nrm[3f] crash_nrm[3f]

    FOUC crash surface:
      uint32 nv + weights[nv x 40]
      weights: base_pos[3h] crash_pos[3h]
               baseUnkBump1[4B] crashUnkBump1[4B]
               baseUnkBump2[4B] crashUnkBump2[4B]
               baseNrm[4B] crashNrm[4B]     (order: [0]=FO2.z [1]=FO2.y [2]=FO2.x)
               baseUV[2H]
    """
    if src_is_fouc == dst_is_fouc:
        # same format -- just copy
        shutil.copy2(crash_path, output_path)
        return

    nodes = _parse_crash_dat(crash_path, is_fouc=src_is_fouc)

    # build model-name -> surface_ids lookup so each crash node maps to the correct BGM surfaces 
    # (NOT a sequential flat-list index)
    model_surf_map = {}
    if models:
        for m in models:
            model_surf_map[m['name']] = m['surfaces']

    def _enc_nrm(v):
        return max(0, min(255, int(round((v + 1.0) * 127.0))))

    new_nodes = []
    surf_idx = 0   # fallback sequential counter (used when model map unavailable)

    for node_name, surfs in nodes:
        new_surfs = []
        base_name = node_name.replace('_crash', '')
        model_surf_ids = model_surf_map.get(base_name)

        for j, crash_surf in enumerate(surfs):
            if src_is_fouc and not dst_is_fouc:
                # FOUC -> FO2: reconstruct vtx from dst streams, convert 40B->48B weights.
                #
                # surface lookup: prefer model-name-based lookup so each crash surface maps 
                # to the correct BGM surface (fixes wrong vtx slice / wrong vs).
                bgm_surf = None
                if model_surf_ids is not None and j < len(model_surf_ids):
                    surf_id = model_surf_ids[j]
                    if surf_id < len(surfaces):
                        bgm_surf = surfaces[surf_id]
                elif surf_idx < len(surfaces):
                    bgm_surf = surfaces[surf_idx]

                vtx = b''
                dst_vs = 36   # FO2 default; updated below if stream found
                # Use surfaces_dst (post-convert) to find the correct per-surface stream.
                # Pre-convert surfaces all have sids[0]=0 (shared FOUC stream), which is
                # wrong after per-surface VB conversion where each surface owns its stream.
                bgm_surf_dst = None
                if surfaces_dst is not None:
                    if model_surf_ids is not None and j < len(model_surf_ids):
                        sid_dst = model_surf_ids[j]
                        if sid_dst < len(surfaces_dst):
                            bgm_surf_dst = surfaces_dst[sid_dst]
                    elif surf_idx < len(surfaces_dst):
                        bgm_surf_dst = surfaces_dst[surf_idx]
                if bgm_surf_dst is not None and bgm_surf_dst['sids']:
                    vid = bgm_surf_dst['sids'][0]
                    if vid < len(streams_dst):
                        sv_dst = streams_dst[vid]
                        dst_vs = sv_dst['vs']
                        voff   = bgm_surf_dst['soffs'][0]  # already 0-based after conversion
                        vtx    = sv_dst['data'][voff : voff + bgm_surf_dst['vc'] * dst_vs]

                # Per-surface foucVertexMultiplier for correct position decoding.
                # Read from pre-convert bgm_surf — extra is stripped during conversion.
                mult = (0.0, 0.0, 0.0, FOUC_SCALE)
                if bgm_surf is not None and len(bgm_surf.get('extra', b'')) >= 16:
                    mult = struct.unpack_from('<4f', bgm_surf['extra'], 0)
                ox, oy, oz = mult[0], mult[1], mult[2]
                sc = mult[3] if mult[3] != 0.0 else FOUC_SCALE

                nv  = len(crash_surf['wgt']) // 40
                new_wgt = bytearray()
                for i in range(nv):
                    base = i * 40
                    bp = struct.unpack_from('<3h', crash_surf['wgt'], base)
                    cp = struct.unpack_from('<3h', crash_surf['wgt'], base + 6)
                    bn = struct.unpack_from('<4B', crash_surf['wgt'], base + 28)
                    cn = struct.unpack_from('<4B', crash_surf['wgt'], base + 32)
                    # int16 pos -> float, applying per-surface offset and scale
                    bpf = ((bp[0] + ox) * sc, (bp[1] + oy) * sc, (bp[2] + oz) * sc)
                    cpf = ((cp[0] + ox) * sc, (cp[1] + oy) * sc, (cp[2] + oz) * sc)
                    # uint8 nrm [0]=z [1]=y [2]=x -> float
                    bnf = ((bn[2]/127.0)-1.0, (bn[1]/127.0)-1.0, (bn[0]/127.0)-1.0)
                    cnf = ((cn[2]/127.0)-1.0, (cn[1]/127.0)-1.0, (cn[0]/127.0)-1.0)
                    new_wgt += struct.pack('<12f', *bpf, *cpf, *bnf, *cnf)

                new_surfs.append({'vtx': vtx, 'wgt': bytes(new_wgt), 'vs': dst_vs})

            else:
                # FO2 -> FOUC: drop vtx, convert 48B->40B weights, extract UVs from vtx
                vs    = crash_surf['vs']
                nv    = len(crash_surf['wgt']) // 48
                vtx   = crash_surf['vtx']
                # detect if vtx has UV (flags must have VERTEX_UV and stride ≥ 32)
                has_uv = vs >= 32 and vs in (32, 36)
                uv_off_in_vert = 24 if vs == 32 else 28  # pos+nrm+[col]+uv

                new_wgt = bytearray()
                for i in range(nv):
                    base = i * 48
                    bp = struct.unpack_from('<3f', crash_surf['wgt'], base)
                    cp = struct.unpack_from('<3f', crash_surf['wgt'], base + 12)
                    bn = struct.unpack_from('<3f', crash_surf['wgt'], base + 24)
                    cn = struct.unpack_from('<3f', crash_surf['wgt'], base + 36)
                    # float pos → int16
                    bpi = tuple(max(-32767, min(32767, int(round(v * FOUC_SCALE_INV)))) for v in bp)
                    cpi = tuple(max(-32767, min(32767, int(round(v * FOUC_SCALE_INV)))) for v in cp)
                    # float nrm → uint8 [0]=z [1]=y [2]=x
                    bni = (_enc_nrm(bn[2]), _enc_nrm(bn[1]), _enc_nrm(bn[0]), 255)
                    cni = (_enc_nrm(cn[2]), _enc_nrm(cn[1]), _enc_nrm(cn[0]), 255)
                    # base UV from vtx buffer (int16, scale 1/2048)
                    if has_uv and vtx and i * vs + uv_off_in_vert + 4 <= len(vtx):
                        fu, fv = struct.unpack_from('<2f', vtx, i * vs + uv_off_in_vert)
                        uvi = (max(-32767, min(32767, int(round(fu * 2048.0)))),
                               max(-32767, min(32767, int(round(fv * 2048.0)))))
                    else:
                        uvi = (0, 0)

                    new_wgt += struct.pack('<3h3h4B4B4B4B4B4B2H',
                        *bpi, *cpi,
                        128, 128, 128, 255,   # baseUnkBump1 (neutral tangent)
                        128, 128, 128, 255,   # crashUnkBump1
                        128, 128, 128, 255,   # baseUnkBump2 (neutral bitangent)
                        128, 128, 128, 255,   # crashUnkBump2
                        *bni,                 # baseNormals
                        *cni,                 # crashNormals
                        *uvi,                 # baseUV
                    )

                new_surfs.append({'vtx': b'', 'wgt': bytes(new_wgt), 'vs': 0})

            surf_idx += 1

        new_nodes.append((node_name, new_surfs))

    _write_crash_dat(output_path, new_nodes, is_fouc=dst_is_fouc)


def op_convert(streams, surfaces, objects, materials_raw,
               version, is_fouc, is_fo1, target):
    """Convert streams, surfaces, objects and materials to the target game format.

    target: 'FO1' | 'FO2' | 'FOUC'

    Returns (new_streams, new_surfaces, new_objects, new_materials,
             new_version, new_is_fouc, new_is_fo1).
    crash.dat conversion is handled separately by _convert_crash_dat_file().

    What changes per transition:
      All paths    : nShaderId remapped (38-41+ range diverges FO2↔FOUC);
                     light shader v92 updated (0 for FO1, 2 for FO2/FOUC)
      FO1→FO2/FOUC : version; strip surface center/radius; OBJC flags 0x00→0xE0F9
      FO1/FO2→FOUC : VB float→int16; add multiplier extra (16 B); fc→22
      FO2→FO1      : version; compute & add surface center/radius; OBJC flags 0xE0F9→0x00
      FOUC→FO2/FO1 : VB int16→float; strip multiplier extra; fc→0
      FOUC→FO1     : all of the above
    """
    import copy as _copy

    TARGET_VERSION = {'FO1': 0x00010004, 'FO2': 0x00020000, 'FOUC': 0x00020000}
    new_version = TARGET_VERSION[target]
    new_is_fouc = (target == 'FOUC')
    new_is_fo1  = (target == 'FO1')

    src_label = 'FOUC' if is_fouc else ('FO1' if is_fo1 else 'FO2')
    print(f"\n── Convert {src_label} → {target} ──")

    if src_label == target:
        print("  Source and target formats are identical — nothing to do.")
        return streams, surfaces, objects, materials_raw, version, is_fouc, is_fo1

    streams  = _copy.deepcopy(streams)
    surfaces = _copy.deepcopy(surfaces)

    need_vb_convert = (is_fouc != new_is_fouc)

    # vertex buffer conversion
    if need_vb_convert:
        if is_fouc:
            # FOUC -> FO2/FO1: per-surface conversion with per-shader vertex format.
            #
            # FOUC uses one shared VB stream for all surfaces. FO2 requires
            # different vertex formats per shader:
            #   shadow  (13):     pos only          12 B  flags 0x0002
            #   body/skin (5/26): pos+nrm+col+uv    36 B  flags 0x0152
            #   everything else:  pos+nrm+uv         32 B  flags 0x0112  (no color)
            #
            # One independent VB+IB pair is created per surface, matching the
            # Blender plugin output. IB indices are rebased to 0 within each VB.

            FO2_FLAGS_NO_COLOR = VERTEX_POSITION | VERTEX_NORMAL | VERTEX_UV  # 0x0112

            def _fo2_fmt_for_surface(surface):
                mid      = surface['mid']
                fouc_sid = materials_raw[mid]['shader_id'] if mid < len(materials_raw) else 8
                fo2_sid  = FOUC_TO_FO2_SHADER.get(fouc_sid, fouc_sid)
                if fo2_sid == 13:        return VERTEX_POSITION, 12
                if fo2_sid in (5, 26):   return FO2_VERTEX_FLAGS_FULL, 36
                return FO2_FLAGS_NO_COLOR, 32

            # Build per-vertex multiplier table.
            stream_vert_mults = {}
            for si, sv in enumerate(streams):
                if sv['dt'] in (1, 3):
                    stream_vert_mults[si] = [(0.0, 0.0, 0.0, FOUC_SCALE)] * sv['vc']
            for s in surfaces:
                if not s['sids'] or len(s['extra']) < 16:
                    continue
                vid   = s['sids'][0]
                if vid not in stream_vert_mults:
                    continue
                vbase = s['soffs'][0] // 32
                mult  = struct.unpack_from('<4f', s['extra'], 0)
                for vi in range(s['vc']):
                    if vbase + vi < len(stream_vert_mults[vid]):
                        stream_vert_mults[vid][vbase + vi] = mult

            # helper: convert a slice of vc vertices starting at abs vertex
            # index vbase from FOUC stream sv into a float VB of tgt_vs bytes/vert
            def _conv_verts(sv, vmults, vbase, vc, tgt_vs):
                out_vb = bytearray(vc * tgt_vs)
                for vi in range(vc):
                    abs_vi  = vbase + vi
                    m       = vmults[abs_vi] if abs_vi < len(vmults) else (0.0, 0.0, 0.0, FOUC_SCALE)
                    ox, oy, oz = m[0], m[1], m[2]
                    sc      = m[3] if m[3] != 0.0 else FOUC_SCALE
                    src_off = abs_vi * 32
                    dst     = vi * tgt_vs
                    px, py, pz = struct.unpack_from('<3h', sv['data'], src_off)
                    struct.pack_into('<3f', out_vb, dst,
                                     (px + ox) * sc, (py + oy) * sc, (pz + oz) * sc)
                    if tgt_vs >= 32:
                        nb = struct.unpack_from('<4B', sv['data'], src_off + 16)
                        struct.pack_into('<3f', out_vb, dst + 12,
                                         (nb[2] / 127.0) - 1.0,
                                         (nb[1] / 127.0) - 1.0,
                                         (nb[0] / 127.0) - 1.0)
                        uv = struct.unpack_from('<2h', sv['data'], src_off + 24)
                        if tgt_vs == 36:
                            col = struct.unpack_from('<4B', sv['data'], src_off + 20)
                            struct.pack_into('<4B', out_vb, dst + 24,
                                             col[0], col[1], col[2], col[3])
                            struct.pack_into('<2f', out_vb, dst + 28,
                                             uv[0] / 2048.0, uv[1] / 2048.0)
                        else:
                            struct.pack_into('<2f', out_vb, dst + 24,
                                             uv[0] / 2048.0, uv[1] / 2048.0)
                return bytes(out_vb)

            # Snapshot original sids/soffs before any surface is mutated.
            orig_sids  = [list(s['sids'])  for s in surfaces]
            orig_soffs = [list(s['soffs']) for s in surfaces]

            new_stream_list = []

            # per-surface pass: each surface gets its own independent VB + IB
            for si, s in enumerate(surfaces):
                if not s['sids']:
                    continue
                vid        = orig_sids[si][0]
                sv         = streams[vid]
                tgt_flags, tgt_vs = _fo2_fmt_for_surface(s)
                vbase      = orig_soffs[si][0] // 32
                vc         = s['vc']
                vmults     = stream_vert_mults.get(vid, [])

                # per-surface VB
                out_vb     = _conv_verts(sv, vmults, vbase, vc, tgt_vs)
                new_vb_sid = len(new_stream_list)
                new_stream_list.append({'dt': sv['dt'], 'fc': 0,
                                        'vc': vc, 'vs': tgt_vs, 'flags': tgt_flags,
                                        'data': out_vb})

                if len(orig_sids[si]) >= 2:
                    iid  = orig_sids[si][1]
                    isv  = streams[iid]
                    ioff = orig_soffs[si][1]
                    niu  = s['niu']
                    new_idata = bytearray(niu * 2)
                    for ii in range(niu):
                        raw_idx = struct.unpack_from('<H', isv['data'], ioff + ii * 2)[0]
                        struct.pack_into('<H', new_idata, ii * 2, raw_idx - vbase)
                    new_ib_sid = len(new_stream_list)
                    new_stream_list.append({'dt': 2, 'fc': 0, 'ic': niu,
                                            'data': bytes(new_idata)})
                    s['sids']  = [new_vb_sid, new_ib_sid]
                    s['soffs'] = [0, 0]
                else:
                    s['sids']  = [new_vb_sid]
                    s['soffs'] = [0]

                s['flags'] = tgt_flags

            streams = new_stream_list
            n_vb = sum(1 for sv in streams if sv['dt'] in (1, 3))
            print(f"  Converted {n_vb} vertex buffer(s): int16→float, per-shader formats")

        else:
            # FO2/FO1 -> FOUC: uniform 32 B format, one new stream per original stream.
            old_vs_map = {si: sv['vs'] for si, sv in enumerate(streams) if sv['dt'] in (1, 3)}
            new_stream_list = []
            sid_map = {}
            for si, sv in enumerate(streams):
                if sv['dt'] in (1, 3):
                    new_data, _ = _fo2_vert_to_fouc(sv['data'], sv['vc'], sv['vs'], sv['flags'])
                    new_stream_list.append({'dt': sv['dt'], 'fc': 22,
                        'vc': sv['vc'], 'vs': 32, 'flags': FOUC_VERTEX_FLAGS, 'data': new_data})
                    sid_map[si] = len(new_stream_list) - 1
                else:
                    new_stream_list.append(dict(sv))
                    sid_map[si] = len(new_stream_list) - 1

            for s in surfaces:
                if s['sids']:
                    old_vb_sid = s['sids'][0]
                    old_vs     = old_vs_map.get(old_vb_sid, 1)
                    new_vb_sid = sid_map.get(old_vb_sid, old_vb_sid)
                    new_vs     = new_stream_list[new_vb_sid]['vs']
                    vbase      = s['soffs'][0] // old_vs if old_vs else 0
                    s['soffs'][0] = vbase * new_vs
                    s['flags'] = new_stream_list[new_vb_sid]['flags']
                s['sids'] = [sid_map.get(sid, sid) for sid in s['sids']]

            streams = new_stream_list
            n_vb = sum(1 for sv in streams if sv['dt'] in (1, 3))
            print(f"  Converted {n_vb} vertex buffer(s): float→int16 (32 B, flags 0x2242)")
    # surface extra bytes
    for s in surfaces:
        # remove source-specific extra first
        if is_fo1:
            # strip FO1 center/radius (24 B)
            s['extra'] = b''
        elif is_fouc:
            # strip FOUC multiplier (16 B)
            s['extra'] = b''

        # add target-specific extra
        if new_is_fo1:
            # compute AABB from (already float) vertex buffer
            center, radius = _compute_surface_aabb(streams, s)
            s['extra'] = struct.pack('<6f', *center, *radius)
        elif new_is_fouc:
            # default multiplier: no offset, scale = 1/1024
            s['extra'] = struct.pack('<4f', 0.0, 0.0, 0.0, FOUC_SCALE)
        # FO2 target: extra stays empty (already cleared above)

    print(f"  Updated extra bytes on {len(surfaces)} surfaces  "
          f"({'24 B center/radius' if new_is_fo1 else '16 B multiplier' if new_is_fouc else 'none'})")

    # shader ID + v92 remapping
    materials_raw = _remap_shaders(materials_raw, is_fouc, is_fo1,
                                   new_is_fouc, new_is_fo1, version)

    # OBJC flags
    # FO1 uses 0x00000000 for all object/dummy flags.
    # FO2 and FOUC use 0x0000E0F9.
    objects = _copy.deepcopy(objects)
    objc_flag_fo1  = 0x00000000
    objc_flag_fo2  = 0x0000E0F9
    target_objc    = objc_flag_fo1 if new_is_fo1 else objc_flag_fo2
    src_objc       = objc_flag_fo1 if is_fo1     else objc_flag_fo2
    if target_objc != src_objc:
        updated = 0
        for o in objects:
            if o['flags'] == src_objc:
                o['flags'] = target_objc
                updated += 1
        print(f"  OBJC flags: 0x{src_objc:08X} → 0x{target_objc:08X}"
              f"  ({updated}/{len(objects)} objects updated)")

    # version
    print(f"  Version: 0x{version:X} → 0x{new_version:X}")

    return streams, surfaces, objects, materials_raw, new_version, new_is_fouc, new_is_fo1


def _parse_bgm_streams_only(path):
    """Parse just the stream list from a BGM file (lightweight; used for is_fouc detection)."""
    with open(path, 'rb') as f:
        version = struct.unpack('<I', f.read(4))[0]
        nm = struct.unpack('<I', f.read(4))[0]
        for _ in range(nm):
            f.read(4)
            while f.read(1) not in (b'\x00', b''): pass
            f.read(4)
            if version >= 0x10004:
                f.read(20); f.read(12); f.read(12)
            f.read(64); f.read(4)
            for _ in range(3):
                while f.read(1) not in (b'\x00', b''): pass
        ns = struct.unpack('<I', f.read(4))[0]
        streams = []
        for _ in range(ns):
            dt = struct.unpack('<I', f.read(4))[0]
            fc = struct.unpack('<I', f.read(4))[0]
            streams.append({'dt': dt, 'fc': fc})
            if dt == 1:
                vc, vs, fl = struct.unpack('<3I', f.read(12)); f.read(vc * vs)
            elif dt == 2:
                ic = struct.unpack('<I', f.read(4))[0];       f.read(ic * 2)
            elif dt == 3:
                vc, vs = struct.unpack('<2I', f.read(8));      f.read(vc * vs)
    return streams



# ─────────────────────────────────────────────────────────────────────────────
# FORMAT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_bgm_format(path):
    """Identify BGM file format.
    Returns one of: 'PC_FO1', 'PC_FO2', 'FOUC', 'PS2', 'PSP', 'XBOX', 'UNKNOWN'.
    """
    try:
        with open(path, 'rb') as f:
            raw = f.read()
        r = _BinReader(raw)
        version = r.u32()
        nm = r.u32()
        for _ in range(nm):
            r.u32(); r.read_string(); r.i32()
            if version >= 0x10004:
                r.read(20 + 12 + 12)
            r.read(16 + 16 + 16 + 16 + 4)
            for _ in range(3): r.read_string()
        ns = r.u32()
        has_type1=False; has_type2=False; has_type3=False
        has_type4=False; has_type5=False; has_fouc=False
        for _ in range(ns):
            dt = r.i32()
            if dt == 1:
                fc = r.u32()
                if fc > 0: has_fouc = True
                vc=r.u32(); vs=r.u32(); r.u32(); r.read(vc*vs); has_type1=True
            elif dt == 2:
                r.u32(); ic=r.u32(); r.read(ic*2); has_type2=True
            elif dt == 3:
                r.u32(); vc=r.u32(); vs=r.u32(); r.read(vc*vs); has_type3=True
            elif dt == 4:
                r.u32(); bl=r.u32(); r.u32(); r.read(bl); has_type4=True
            elif dt == 5:
                r.u32(); bl=r.u32(); r.u32(); r.read(bl); has_type5=True
            else:
                break
        if has_type4 or has_type5: return 'XBOX'
        if has_fouc: return 'FOUC'
        if has_type1 or has_type2:
            return 'PC_FO1' if version < 0x20000 else 'PC_FO2'
        if has_type3:
            nsf = r.u32()
            if nsf == 0: return 'PS2'
            w0=r.u32(); w1=r.u32()
            if w0 == 0 and w1 == 0x1000: return 'PS2'
            if w0 == 0x1000: return 'PSP'
            return 'PS2'
        return 'UNKNOWN'
    except Exception:
        return 'UNKNOWN'


# ─────────────────────────────────────────────────────────────────────────────
# SHARED CONSOLE PARSER HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_console_materials(r, version):
    materials = []
    nm = r.u32()
    for _ in range(nm):
        ident  = r.u32()
        name   = r.read_string()
        n_alpha= r.i32()
        v92=n_num_tex=shader_id=n_use_colormap=v74=0
        v108=v109=(0,0,0)
        if version >= 0x10004:
            v92=r.i32(); n_num_tex=r.i32(); shader_id=r.i32()
            n_use_colormap=r.i32(); v74=r.i32()
            v108=struct.unpack_from('<3i',r.read(12))
            v109=struct.unpack_from('<3i',r.read(12))
        v98=struct.unpack_from('<4i',r.read(16)); v99=struct.unpack_from('<4i',r.read(16))
        v100=struct.unpack_from('<4i',r.read(16)); v101=struct.unpack_from('<4i',r.read(16))
        v102=r.i32()
        tex_names=[r.read_string() for _ in range(3)]
        materials.append({'ident':ident,'name':name,'n_alpha':n_alpha,
            'v92':v92,'n_num_tex':n_num_tex,'shader_id':shader_id,
            'n_use_colormap':n_use_colormap,'v74':v74,'v108':v108,'v109':v109,
            'v98':v98,'v99':v99,'v100':v100,'v101':v101,'v102':v102,'tex_names':tex_names})
    return materials

def _parse_console_models(r):
    models=[]; nmod=r.u32()
    for _ in range(nmod):
        ident=r.u32(); unk=r.i32(); name=r.read_string()
        center=r.vec3f(); radius=r.vec3f(); f_radius=r.f32()
        surfs=[r.i32() for _ in range(r.u32())]
        models.append({'ident':ident,'unk':unk,'name':name,'center':center,
                        'radius':radius,'f_radius':f_radius,'surfaces':surfs})
    return models

def _parse_console_meshes(r):
    meshes=[]; nmesh=r.u32()
    for _ in range(nmesh):
        ident=r.u32(); name1=r.read_string(); name2=r.read_string()
        flags=r.u32(); group=r.i32()
        matrix=list(struct.unpack_from('<16f',r.read(64)))
        mids=[r.i32() for _ in range(r.i32())]
        meshes.append({'ident':ident,'name1':name1,'name2':name2,
                        'flags':flags,'group':group,'matrix':matrix,'model_ids':mids})
    return meshes

def _parse_console_objects(r):
    objects=[]; nobj=r.u32()
    for _ in range(nobj):
        ident=r.u32(); name1=r.read_string(); name2=r.read_string()
        flags=r.u32(); matrix=list(struct.unpack_from('<16f',r.read(64)))
        objects.append({'ident':ident,'name1':name1,'name2':name2,
                         'flags':flags,'matrix':matrix})
    return objects

def _write_console_header(f, version, materials_raw):
    """Write version + materials section shared by all console formats."""
    f.write(struct.pack('<I', version))
    f.write(struct.pack('<I', len(materials_raw)))
    for m in materials_raw:
        f.write(struct.pack('<I', m['ident']))
        f.write(m['name'].encode('ascii') + b'\x00')
        f.write(struct.pack('<i', m['n_alpha']))
        if version >= 0x10004:
            f.write(struct.pack('<i', m['v92'])); f.write(struct.pack('<i', m['n_num_tex']))
            f.write(struct.pack('<i', m['shader_id'])); f.write(struct.pack('<i', m['n_use_colormap']))
            f.write(struct.pack('<i', m['v74']))
            v108=m['v108']
            f.write(v108 if isinstance(v108,(bytes,bytearray)) else struct.pack('<3i',*v108))
            v109=m['v109']
            f.write(v109 if isinstance(v109,(bytes,bytearray)) else struct.pack('<3i',*v109))
        f.write(struct.pack('<4i',*m['v98'])); f.write(struct.pack('<4i',*m['v99']))
        f.write(struct.pack('<4i',*m['v100'])); f.write(struct.pack('<4i',*m['v101']))
        f.write(struct.pack('<i', m['v102']))
        for tn in m['tex_names']: f.write(tn.encode('ascii') + b'\x00')

def _write_models_meshes_objects(f, models, meshes, objects):
    """Write models, meshes, objects — identical layout across all platforms."""
    f.write(struct.pack('<I', len(models)))
    for m in models:
        f.write(struct.pack('<I', m['ident'])); f.write(struct.pack('<i', m['unk']))
        f.write(m['name'].encode('ascii') + b'\x00')
        f.write(struct.pack('<3f',*m['center'])); f.write(struct.pack('<3f',*m['radius']))
        f.write(struct.pack('<f', m['f_radius'])); f.write(struct.pack('<I', len(m['surfaces'])))
        for s in m['surfaces']: f.write(struct.pack('<i', s))
    f.write(struct.pack('<I', len(meshes)))
    for mesh in meshes:
        f.write(struct.pack('<I', mesh['ident']))
        f.write(mesh['name1'].encode('ascii') + b'\x00')
        f.write(mesh['name2'].encode('ascii') + b'\x00')
        f.write(struct.pack('<I', mesh['flags'])); f.write(struct.pack('<i', mesh['group']))
        f.write(struct.pack('<16f',*mesh['matrix']))
        f.write(struct.pack('<i', len(mesh['model_ids'])))
        for mid in mesh['model_ids']: f.write(struct.pack('<i', mid))
    f.write(struct.pack('<I', len(objects)))
    for obj in objects:
        f.write(struct.pack('<I', obj['ident']))
        f.write(obj['name1'].encode('ascii') + b'\x00')
        f.write(obj['name2'].encode('ascii') + b'\x00')
        f.write(struct.pack('<I', obj['flags'])); f.write(struct.pack('<16f',*obj['matrix']))

def _tris_to_pc_surface(tris, is_shadow, vb_stream_idx, ib_stream_idx, material_id,
                         shader_id=0):
    """Build one PC-format surface (VB bytes, IB bytes, surface dict) from a triangle list.
    Each element of tris is ((pos,uv,norm),(pos,uv,norm),(pos,uv,norm)).

    Vertex format follows the export plugin's get_vertex_format() logic:
      shader 13 (shadow):      pos(12)                     flags 0x0002  vs=12
      shader 5/26 (body/skin): pos(12)+norm(12)+color(4)+uv(8) flags 0x0152  vs=36
      everything else:         pos(12)+norm(12)+uv(8)      flags 0x0112  vs=32

    Winding: IB stores (v2,v1,v0) so the PC importer's (i2,i1,i0) reversal restores
    the original console winding.
    """
    if not tris: return None

    _SHADER_SHADOW = 13
    _SHADER_BODY   = {5, 26}   # car body / skinning: need color channel

    if is_shadow or shader_id == _SHADER_SHADOW:
        vs    = 12
        flags = VERTEX_POSITION
    elif shader_id in _SHADER_BODY:
        vs    = 36
        flags = VERTEX_POSITION | VERTEX_NORMAL | VERTEX_COLOR | VERTEX_UV
    else:
        vs    = 32
        flags = VERTEX_POSITION | VERTEX_NORMAL | VERTEX_UV

    seen = {}; vb = bytearray(); ib = bytearray()

    for v0, v1, v2 in tris:
        # Build VB entries for all three vertices
        for (pos, uv, norm) in (v0, v1, v2):
            if is_shadow or shader_id == _SHADER_SHADOW:
                key = (round(pos[0], 5), round(pos[1], 5), round(pos[2], 5))
            elif shader_id in _SHADER_BODY:
                key = (round(pos[0], 5), round(pos[1], 5), round(pos[2], 5),
                       round(uv[0],  4), round(uv[1],  4),
                       round(norm[0],4), round(norm[1],4), round(norm[2],4))
            else:
                key = (round(pos[0], 5), round(pos[1], 5), round(pos[2], 5),
                       round(uv[0],  4), round(uv[1],  4),
                       round(norm[0],4), round(norm[1],4), round(norm[2],4))
            if key not in seen:
                seen[key] = len(seen)
                if is_shadow or shader_id == _SHADER_SHADOW:
                    vb += struct.pack('<3f', *pos)
                elif shader_id in _SHADER_BODY:
                    vb += struct.pack('<3f', *pos)
                    vb += struct.pack('<3f', *norm)
                    vb += struct.pack('<4B', 255, 255, 255, 255)  # white vertex color
                    vb += struct.pack('<2f', *uv)
                else:
                    vb += struct.pack('<3f', *pos)
                    vb += struct.pack('<3f', *norm)
                    vb += struct.pack('<2f', *uv)

        # IB: store reversed (v2,v1,v0) so PC import's (i2,i1,i0) reversal restores (v0,v1,v2)
        for (pos, uv, norm) in (v2, v1, v0):
            if is_shadow or shader_id == _SHADER_SHADOW:
                key = (round(pos[0], 5), round(pos[1], 5), round(pos[2], 5))
            else:
                key = (round(pos[0], 5), round(pos[1], 5), round(pos[2], 5),
                       round(uv[0],  4), round(uv[1],  4),
                       round(norm[0],4), round(norm[1],4), round(norm[2],4))
            ib += struct.pack('<H', seen[key])

    vc = len(seen); niu = len(tris) * 3; pc = len(tris)
    surf = {'isveg': 0, 'mid': material_id, 'vc': vc, 'flags': flags,
            'pc': pc, 'pm': 4, 'niu': niu, 'extra': b'',
            'nst': 2, 'sids': [vb_stream_idx, ib_stream_idx], 'soffs': [0, 0]}
    return bytes(vb), bytes(ib), surf

def _console_tris_to_streams_surfaces(per_surf_tris, is_shadow_list, material_ids,
                                       materials=None):
    """Pack per-surface triangle lists into PC streams+surfaces.
    One independent VB+IB pair per surface.
    materials: list of material dicts (with 'shader_id') — used to pick the right vertex format.
    """
    streams = []; surfaces = []
    for si, (tris, is_shadow, mid) in enumerate(zip(per_surf_tris, is_shadow_list, material_ids)):
        shader_id = 0
        if materials and mid < len(materials):
            shader_id = materials[mid].get('shader_id', 0)
        result = _tris_to_pc_surface(tris, is_shadow, len(streams), len(streams)+1,
                                     mid, shader_id=shader_id)
        if result is None: continue
        vb_bytes, ib_bytes, surf = result
        vs    = surf['flags'] and (12 if surf['flags'] == VERTEX_POSITION else
                                   36 if surf['flags'] & VERTEX_COLOR else 32)
        # determine vs from flags properly
        if surf['flags'] == VERTEX_POSITION:               vs = 12
        elif surf['flags'] & VERTEX_COLOR:                 vs = 36
        else:                                              vs = 32
        vb_si = len(streams); ib_si = vb_si + 1
        streams.append({'dt': 1, 'fc': 0, 'vc': len(vb_bytes) // vs,
                         'vs': vs, 'flags': surf['flags'], 'data': vb_bytes})
        streams.append({'dt': 2, 'fc': 0, 'ic': len(ib_bytes) // 2, 'data': ib_bytes})
        surf['sids'] = [vb_si, ib_si]; surf['soffs'] = [0, 0]
        surfaces.append(surf)
    return streams, surfaces


# ─────────────────────────────────────────────────────────────────────────────
# PS2 SUPPORT — READER
# ─────────────────────────────────────────────────────────────────────────────

def parse_ps2_bgm(path):
    """Parse a PS2 BGM file.
    Returns (version, materials, ps2_surfs, blob, models, meshes, objects).
    """
    with open(path,'rb') as f: raw=f.read()
    r=_BinReader(raw); version=r.u32()
    materials=_parse_console_materials(r,version)
    blob=b''; ns=r.u32()
    for _ in range(ns):
        dt=r.i32()
        if dt==3: r.i32(); vc=r.u32(); vs=r.u32(); blob=r.read(vc*vs)
        elif dt==1: r.i32(); vc=r.u32(); vs=r.u32(); r.u32(); r.read(vc*vs)
        elif dt==2: r.i32(); ic=r.u32(); r.read(ic*2)
    ps2_surfs=[]; nsf=r.u32()
    for _ in range(nsf):
        vals=struct.unpack_from('<10I',raw,r.pos); r.read(40)
        ps2_surfs.append({'unk0':vals[0],'unk1':vals[1],'material_id':vals[2],
            'total_verts':vals[3],'num_batches':vals[4],'unk5':vals[5],
            'unk6':vals[6],'unk7':vals[7],'blob_offset':vals[8],'blob_size':vals[9]})
    models=_parse_console_models(r); meshes=_parse_console_meshes(r)
    objects=_parse_console_objects(r)
    print(f"[PS2 BGM] Parsed {path}: {len(materials)} mats, "
          f"{len(ps2_surfs)} surfs, {len(models)} models, blob={len(blob):,}B")
    return version,materials,ps2_surfs,blob,models,meshes,objects

def _ps2_extract_batches(blob, surf):
    """Extract per-batch vertex data from PS2 VIF packets.
    Returns list of dicts with keys 'pos', 'uv', 'norm', 'adc'.
    Ported from fo2_bgm_import/bgm_ps2.py.
    """
    off=surf['blob_offset']; end=off+surf['blob_size']
    batches=[]; cur_pos=[]; cur_uv=[]; cur_norm=[]; cur_adc=[]
    while off < end:
        if off+4>len(blob): break
        w=struct.unpack_from('<I',blob,off)[0]; cmd=(w>>24)&0x7F
        if cmd in (0x00,0x01,0x04): off+=4; continue
        if cmd==0x17:
            if cur_pos: batches.append({'pos':cur_pos,'uv':cur_uv,'norm':cur_norm,'adc':cur_adc})
            cur_pos=[]; cur_uv=[]; cur_norm=[]; cur_adc=[]; off+=4; continue
        if cmd>=0x60:
            vn=(cmd>>2)&3; vl=cmd&3; num=(w>>16)&0xFF; addr=w&0x3FF
            eb=[4,2,1,0][vl]; comp=[1,2,3,4][vn]; db=(num*comp*eb+3)&~3; doff=off+4
            if vn==2 and vl==1 and addr==7:
                for vi in range(num):
                    x,y,z=struct.unpack_from('<3h',blob,doff+vi*6)
                    cur_pos.append((x/1024.,y/1024.,z/1024.))
            elif vn==3 and vl==1 and addr==7:
                for vi in range(num):
                    x,y,z,_=struct.unpack_from('<4h',blob,doff+vi*8)
                    cur_pos.append((x/1024.,y/1024.,z/1024.))
            elif vn==3 and vl==1 and addr==5:
                for vi in range(num):
                    x,y,z,adc=struct.unpack_from('<4h',blob,doff+vi*8)
                    cur_pos.append((x/1024.,y/1024.,z/1024.)); cur_adc.append(adc)
            elif vn==1 and vl==1 and addr==8:
                for vi in range(num):
                    u,v=struct.unpack_from('<2h',blob,doff+vi*4)
                    cur_uv.append((u/4096.,v/4096.))
            elif vn==3 and vl==2 and addr==9:
                for vi in range(num):
                    nx,ny,nz,adc=struct.unpack_from('<4b',blob,doff+vi*4)
                    cur_norm.append((nx/127.,ny/127.,nz/127.)); cur_adc.append(adc)
            off+=4+db
        else: off+=4
    if cur_pos: batches.append({'pos':cur_pos,'uv':cur_uv,'norm':cur_norm,'adc':cur_adc})
    return batches

def _ps2_triangles_from_adc(positions, uvs, normals, adcs):
    """Decode an ADC-flagged PS2 triangle strip into triangles.
    Each triangle is ((pos,uv,norm),(pos,uv,norm),(pos,uv,norm)).
    Ported from fo2_bgm_import/bgm_ps2.py.
    """
    tris=[]; verts=[]
    for i in range(len(positions)):
        p=positions[i]; uv=uvs[i] if i<len(uvs) else (0.,0.)
        n=normals[i] if i<len(normals) else (0.,0.,1.)
        verts.append((p,uv,n))
        adc=adcs[i] if i<len(adcs) else 0
        adc_u=adc&0xFF; skip=(adc_u>>7)&1; odd=(adc_u>>5)&1
        if not skip and i>=2:
            v0,v1,v2=verts[i-2],verts[i-1],verts[i]
            tris.append((v0,v1,v2) if odd else (v1,v0,v2))
    return tris

def ps2_to_pc(version, materials, ps2_surfs, blob, models, meshes, objects):
    """Convert PS2 BGM geometry to PC FO2 format."""
    per_surf_tris=[]; is_shadow_list=[]; material_ids=[]
    for surf in ps2_surfs:
        batches=_ps2_extract_batches(blob,surf)
        if not batches:
            per_surf_tris.append([]); is_shadow_list.append(False)
            material_ids.append(surf['material_id']); continue
        all_pos=[]; all_uv=[]; all_norm=[]; all_adc=[]
        for b in batches:
            all_pos+=b['pos']; all_uv+=b['uv']; all_norm+=b['norm']; all_adc+=b['adc']
        is_shadow=(not all_uv and not all_norm) or surf['unk1']!=0x1000
        tris=_ps2_triangles_from_adc(all_pos,all_uv,all_norm,all_adc)
        per_surf_tris.append(tris); is_shadow_list.append(is_shadow)
        material_ids.append(surf['material_id'])
    streams,surfaces=_console_tris_to_streams_surfaces(per_surf_tris,is_shadow_list,material_ids,materials=materials)
    old_to_new={}; new_si=0
    for old_si,tris in enumerate(per_surf_tris):
        if tris: old_to_new[old_si]=new_si; new_si+=1
    for m in models:
        m['surfaces']=[old_to_new[s] for s in m['surfaces'] if s in old_to_new]
    out_version=max(version,0x20000)
    print(f"[PS2→PC] {len(ps2_surfs)} PS2 surfaces → {len(surfaces)} PC surfaces, {len(streams)} streams")
    return out_version,materials,streams,surfaces,models,meshes,objects,False,False


# ─────────────────────────────────────────────────────────────────────────────
# PS2 SUPPORT — WRITER
# ─────────────────────────────────────────────────────────────────────────────

# PS2 vertex type constants
_PS2_VTYPE_STD=0; _PS2_VTYPE_COLORED=1; _PS2_VTYPE_SHADOW=2
_PS2_MAX_BATCH_V3=77; _PS2_MAX_BATCH_V3_ALPHA=34; _PS2_MAX_BATCH_V4=22; _PS2_MAX_BATCH_SHADOW=155
_PS2_SHADER_STD_OPAQUE_LIMIT={8:55}; _PS2_SHADER_STD_ALPHA_LIMIT={11:32,12:32}
_VIF_STCYCL=0x01; _VIF_ITOP=0x04; _VIF_MSCNT=0x17
_VIF_UNPACK_V3_16=0x69; _VIF_UNPACK_V4_16=0x6D; _VIF_UNPACK_V2_16=0x65; _VIF_UNPACK_V4_8=0x6E
_PS2_POS_SCALE=1024.; _PS2_UV_SCALE=4096.; _PS2_NORM_SCALE=127.

def _ps2_get_vtype(streams, surf):
    if surf['nst']<1: return _PS2_VTYPE_STD
    vid=surf['sids'][0]
    if vid>=len(streams): return _PS2_VTYPE_STD
    s=streams[vid]
    if s['dt'] not in (1,3): return _PS2_VTYPE_STD
    fl=s.get('flags',0); vs=s['vs']
    if fl==VERTEX_POSITION or vs==12: return _PS2_VTYPE_SHADOW
    if fl&VERTEX_COLOR: return _PS2_VTYPE_COLORED
    return _PS2_VTYPE_STD

def _ps2_get_verts_from_streams(streams, surf):
    """Extract (pos,norm,uv,color) tuples for a surface."""
    if surf['nst']<1: return []
    vid=surf['sids'][0]
    if vid>=len(streams): return []
    sv=streams[vid]; vc=surf['vc']; vs=sv['vs']; fl=sv.get('flags',0); vdata=sv['data']
    voff=surf['soffs'][0]
    hn=bool(fl&VERTEX_NORMAL); hc=bool(fl&VERTEX_COLOR); hu=bool(fl&(VERTEX_UV|0x200))
    verts=[]
    for i in range(vc):
        off=voff+i*vs
        if off+vs>len(vdata): break
        pos=struct.unpack_from('<3f',vdata,off); fo=12
        norm=(0.,0.,0.); col=(255,255,255,255); uv=(0.,0.)
        if hn: norm=struct.unpack_from('<3f',vdata,off+fo); fo+=12
        if hc: col=struct.unpack_from('<4B',vdata,off+fo); fo+=4
        if hu: uv=struct.unpack_from('<2f',vdata,off+fo)
        verts.append((pos,norm,uv,col))
    return verts

def _ps2_get_indices_from_streams(streams, surf):
    if surf['nst']<2: return []
    iid=surf['sids'][1]
    if iid>=len(streams): return []
    isv=streams[iid]; idata=isv['data']; ioff=surf['soffs'][1]
    vid=surf['sids'][0]; sv=streams[vid]; vs=sv['vs']
    vbase=surf['soffs'][0]//vs if vs>0 else 0
    raw=[struct.unpack_from('<H',idata,ioff+i*2)[0] for i in range(surf['niu']) if ioff+i*2+2<=len(idata)]
    return [x-vbase for x in raw]

def _ps2_stripify_triangles(triangles):
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


def _ps2_detect_winding(strip, vertices, orig_normals):
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
            p0, p1, p2 = vertices[t[0]][0], vertices[t[1]][0], vertices[t[2]][0]
            e1 = (p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2])
            e2 = (p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2])
            sn = (e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0])
            dot = sum(a*b for a,b in zip(sn, orig_normals[key]))
            if dot > 0:
                return w
    return 0


def _ps2_pack_strips_adc(strips_with_winding, max_batch):
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


def _ps2_build_vif_standard(batch, vertices, first):
    """Build VIF data for standard vertex batch (V3-16 pos + V2-16 uv + V4-8 norm+adc)."""
    n = len(batch)
    d = bytearray()
    if first:
        d += struct.pack('<I', (_VIF_STCYCL << 24) | (1 << 8) | 3)
    d += struct.pack('<I', (_VIF_UNPACK_V3_16 << 24) | (n << 16) | 0x8007)
    for vidx, _ in batch:
        v = vertices[vidx]
        d += struct.pack('<3h', *(_xb_clamp(int(round(c * _PS2_POS_SCALE)), -32768, 32767) for c in v[0]))
    while len(d) % 4:
        d += b'\x00'
    d += struct.pack('<I', (_VIF_UNPACK_V2_16 << 24) | (n << 16) | 0x8008)
    for vidx, _ in batch:
        v = vertices[vidx]
        d += struct.pack('<2h', *(_xb_clamp(int(round(c * _PS2_UV_SCALE)), -32768, 32767) for c in v[2]))
    d += struct.pack('<I', (_VIF_UNPACK_V4_8 << 24) | (n << 16) | 0x8009)
    for vidx, adc in batch:
        v = vertices[vidx]
        d += struct.pack('<4b',
            *(_xb_clamp(int(round(c * _PS2_NORM_SCALE)), -127, 127) for c in v[1]),
            adc)
    d += struct.pack('<I', (_VIF_ITOP << 24) | n)
    d += struct.pack('<I', _VIF_MSCNT << 24)
    # NO padding here, batches are packed tightly (matching PS2 original)
    return bytes(d)


def _ps2_build_vif_colored(batch, vertices, first):
    """Build VIF data for colored vertex batch (V4-16 pos + V2-16 uv + V4-8 norm+adc)."""
    n = len(batch)
    d = bytearray()
    if first:
        d += struct.pack('<I', (_VIF_STCYCL << 24) | (1 << 8) | 3)
    d += struct.pack('<I', (_VIF_UNPACK_V4_16 << 24) | (n << 16) | 0x8007)
    for vidx, _ in batch:
        v = vertices[vidx]
        d += struct.pack('<4h', *(_xb_clamp(int(round(c * _PS2_POS_SCALE)), -32768, 32767) for c in v[0]), 0)
    d += struct.pack('<I', (_VIF_UNPACK_V2_16 << 24) | (n << 16) | 0x8008)
    for vidx, _ in batch:
        v = vertices[vidx]
        d += struct.pack('<2h', *(_xb_clamp(int(round(c * _PS2_UV_SCALE)), -32768, 32767) for c in v[2]))
    d += struct.pack('<I', (_VIF_UNPACK_V4_8 << 24) | (n << 16) | 0x8009)
    for vidx, adc in batch:
        v = vertices[vidx]
        d += struct.pack('<4b',
            *(_xb_clamp(int(round(c * _PS2_NORM_SCALE)), -127, 127) for c in v[1]),
            adc)
    d += struct.pack('<I', (_VIF_ITOP << 24) | n)
    d += struct.pack('<I', _VIF_MSCNT << 24)
    return bytes(d)


def _ps2_build_vif_shadow(batch, vertices, first):
    """Build VIF data for shadow vertex batch (V4-16 pos+adc)."""
    n = len(batch)
    d = bytearray()
    if first:
        d += struct.pack('<I', (_VIF_STCYCL << 24) | (1 << 8) | 1)
    d += struct.pack('<I', (_VIF_UNPACK_V4_16 << 24) | (n << 16) | 0x8005)
    for vidx, adc in batch:
        v = vertices[vidx]
        d += struct.pack('<4h',
            *(_xb_clamp(int(round(c * _PS2_POS_SCALE)), -32768, 32767) for c in v[0]),
            adc)
    d += struct.pack('<I', (_VIF_ITOP << 24) | n)
    d += struct.pack('<I', _VIF_MSCNT << 24)
    return bytes(d)


def _ps2_pad_blob_to_16(data):
    """Pad a complete VIF blob (all batches concatenated) to 16-byte alignment.
    Only pads at the END, matching PS2 original behavior."""
    while len(data) % 16:
        data += b'\x00\x00\x00\x00'  # NOP words
    return data








def _ps2_parse_pc_crash_dat(path):
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


def _ps2_compute_vif_batch_offsets(vif_data, vtype):
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





# ─────────────────────────────────────────────────────────────────────────────
# PS2 WRITER — STREAM/SURFACE BRIDGE
# ─────────────────────────────────────────────────────────────────────────────


def _ps2_write_bgm(output_path, version, materials_raw, streams, surfaces, models, meshes, objects, input_path=''):
    """Write a PS2 BGM file from PC-format data (stream/surface dicts).
    Implements the same logic as bgm_tool_ps2.py write_ps2_bgm but operates
    on the unified dict-based data model instead of PCBGMParser.
    """
    print(f"\nConverting to PS2...")
    # compute material areas
    _ps2_compute_material_areas_dicts(streams, surfaces, materials_raw)

    sinfo=[]; vif_blob=bytearray(); all_batches_list=[]
    for si in range(len(surfaces)):
        vd,nv,nb,batches = _ps2_convert_surface_dicts(streams, surfaces, si, materials_raw)
        sinfo.append((nv,nb,len(vif_blob),len(vd)))
        all_batches_list.append(batches)
        vif_blob+=vd
        surf=surfaces[si]; vt=_ps2_get_vtype(streams,surf)
        print(f"  S{si:2d} m={surf['mid']:2d} {['STD','CLR','SHD'][vt]} "
              f"{surf['vc']:4d}->{nv:4d}v b={nb:2d} {len(vd):5d}B")
    print(f"  Total: {sum(x[0] for x in sinfo)}v {sum(x[1] for x in sinfo)}bat {len(vif_blob)}B")

    with open(output_path,'wb') as f:
        _write_console_header(f, version, materials_raw)
        f.write(struct.pack('<I',1)); f.write(struct.pack('<I',3))
        f.write(struct.pack('<3I',0,len(vif_blob),1)); f.write(vif_blob)
        f.write(struct.pack('<I',len(surfaces)))
        for si,surf in enumerate(surfaces):
            nv,nb,bo,bs=sinfo[si]
            f.write(struct.pack('<10I',0,0x1000,surf['mid'],nv,nb,1,0x0E,0,bo,bs))
        _write_models_meshes_objects(f, models, meshes, objects)

    isz=os.path.getsize(input_path) if input_path and os.path.exists(input_path) else 0
    osz=os.path.getsize(output_path)
    if isz: print(f"\n  {osz:,}B ({osz/isz*100:.1f}% of input)")
    else:   print(f"\n  {osz:,}B")

    # crash.dat
    crash_src,_=_find_crash_dat(input_path) if input_path else (None,None)
    if crash_src and os.path.exists(crash_src):
        crash_out=os.path.splitext(output_path)[0]+'-crash.dat'
        _ps2_gen_crash_dat_dicts(streams,surfaces,models,all_batches_list,
                                  sinfo,vif_blob,crash_src,crash_out)
    else:
        print("  No PC crash.dat found, skipping PS2 crash.dat generation")


def _ps2_compute_material_areas_dicts(streams, surfaces, materials):
    nm=len(materials)
    m3d=[0.]*nm; m108v=[0.]*nm
    for surf in surfaces:
        mid=surf['mid']
        if mid>=nm: continue
        vtype=_ps2_get_vtype(streams,surf)
        verts=_ps2_get_verts_from_streams(streams,surf)
        indices=_ps2_get_indices_from_streams(streams,surf)
        if not verts or not indices: continue
        pm=surf['pm']
        if pm==4:
            tris=[(indices[i],indices[i+1],indices[i+2]) for i in range(0,len(indices)-2,3)]
        else:
            tris=[(indices[i],indices[i+1],indices[i+2]) for i in range(len(indices)-2)
                  if indices[i]!=indices[i+1] and indices[i+1]!=indices[i+2]]
        is_shadow=(vtype==_PS2_VTYPE_SHADOW)
        for a,b,c in tris:
            if not all(0<=x<len(verts) for x in (a,b,c)): continue
            pa,pb,pc_=verts[a][0],verts[b][0],verts[c][0]
            e1=(pb[0]-pa[0],pb[1]-pa[1],pb[2]-pa[2])
            e2=(pc_[0]-pa[0],pc_[1]-pa[1],pc_[2]-pa[2])
            cx=e1[1]*e2[2]-e1[2]*e2[1]; cy=e1[2]*e2[0]-e1[0]*e2[2]; cz=e1[0]*e2[1]-e1[1]*e2[0]
            t3d=0.5*math.sqrt(cx*cx+cy*cy+cz*cz); m3d[mid]+=t3d
            if not is_shadow:
                ua,ub,uc=verts[a][2],verts[b][2],verts[c][2]
                u1=ub[0]-ua[0]; v1=ub[1]-ua[1]; u2=uc[0]-ua[0]; v2=uc[1]-ua[1]
                tuv=0.5*abs(u1*v2-u2*v1)
                if t3d>0 and tuv>0: m108v[mid]+=math.sqrt(t3d*tuv)
    for i in range(nm):
        materials[i]=dict(materials[i])
        materials[i]['v109']=struct.pack('<3f',m3d[i],m3d[i],m3d[i])
        materials[i]['v108']=struct.pack('<3f',m108v[i],0.,0.)


def _ps2_convert_surface_dicts(streams, surfaces, si, materials):
    """Convert one PC surface (dict-based) to PS2 VIF blob data.
    Returns (vif_bytes, total_verts, num_batches, batches).
    """
    surf=surfaces[si]; vtype=_ps2_get_vtype(streams,surf)
    verts=_ps2_get_verts_from_streams(streams,surf)
    if not verts: return b'',0,0,[]
    indices=_ps2_get_indices_from_streams(streams,surf)
    if not indices: indices=list(range(len(verts)))
    indices=[i for i in indices if 0<=i<len(verts)]
    if len(indices)<3: return b'',0,0,[]
    pm=surf['pm']
    if pm==5:
        tris=[]
        for i in range(len(indices)-2):
            a,b,c=indices[i],indices[i+1],indices[i+2]
            if a==b or b==c or a==c: continue
            if all(0<=x<len(verts) for x in (a,b,c)):
                tris.append((a,b,c) if i%2==0 else (b,a,c))
        if not tris: return b'',0,0,[]
        raw_strips=_ps2_stripify_triangles(tris)
    elif pm==4:
        tris=[(indices[i],indices[i+1],indices[i+2])
              for i in range(0,len(indices)-2,3)
              if all(0<=indices[j]<len(verts) for j in (i,i+1,i+2))]
        if not tris: return b'',0,0,[]
        raw_strips=_ps2_stripify_triangles(tris)
    else:
        return b'',0,0,[]
    orig_normals={}
    for a,b,c in tris:
        if all(0<=x<len(verts) for x in (a,b,c)):
            p0,p1,p2=verts[a][0],verts[b][0],verts[c][0]
            e1=(p1[0]-p0[0],p1[1]-p0[1],p1[2]-p0[2])
            e2=(p2[0]-p0[0],p2[1]-p0[1],p2[2]-p0[2])
            n=(e1[1]*e2[2]-e1[2]*e2[1],e1[2]*e2[0]-e1[0]*e2[2],e1[0]*e2[1]-e1[1]*e2[0])
            orig_normals[frozenset([a,b,c])]=n
    strips_with_winding=[(s,_ps2_detect_winding(s,verts,orig_normals)) for s in raw_strips]
    if not strips_with_winding: return b'',0,0,[]
    mat=materials[surf['mid']] if surf['mid']<len(materials) else None
    is_alpha=(mat['n_alpha']!=0) if mat else False
    if vtype==_PS2_VTYPE_STD:
        if is_alpha:
            max_batch=_PS2_SHADER_STD_ALPHA_LIMIT.get(mat['shader_id'] if mat else 0,_PS2_MAX_BATCH_V3_ALPHA)
        else:
            max_batch=_PS2_SHADER_STD_OPAQUE_LIMIT.get(mat['shader_id'] if mat else 0,_PS2_MAX_BATCH_V3)
    elif vtype==_PS2_VTYPE_COLORED:
        max_batch=_PS2_MAX_BATCH_V4
    else:
        max_batch=_PS2_MAX_BATCH_SHADOW
    batches=_ps2_pack_strips_adc(strips_with_winding,max_batch)
    if not batches: return b'',0,0,[]
    builders=[_ps2_build_vif_standard,_ps2_build_vif_colored,_ps2_build_vif_shadow]
    builder=builders[vtype]
    vif=bytearray(); total_verts=0
    for bi,batch in enumerate(batches):
        vif+=builder(batch,verts,bi==0); total_verts+=len(batch)
    vif=_ps2_pad_blob_to_16(vif)
    return bytes(vif),total_verts,len(batches),batches


def _ps2_gen_crash_dat_dicts(streams, surfaces, models, all_batches, sinfo, vif_blob, pc_crash_path, output_path):
    """Generate PS2 crash.dat from PC crash.dat + PS2 VIF data (dict-based)."""
    pc_nodes=_parse_crash_dat(pc_crash_path,is_fouc=False)
    model_surfs={m['name']:m['surfaces'] for m in models}
    out=bytearray(); out+=struct.pack('<I',len(pc_nodes))
    for name,pc_surfs in pc_nodes:
        base_model=name.replace('_crash','')
        out+=name.encode('ascii')+b'\x00'
        if base_model not in model_surfs:
            print(f"  crash.dat: WARNING model '{base_model}' not found, writing 0 surfaces")
            out+=struct.pack('<I',0); continue
        bgm_surf_ids=model_surfs[base_model]
        out+=struct.pack('<I',len(bgm_surf_ids))
        for surf_idx,bgm_si in enumerate(bgm_surf_ids):
            if bgm_si>=len(sinfo): out+=struct.pack('<I',0); continue
            nv,nb,blob_off,blob_sz=sinfo[bgm_si]
            batches=all_batches[bgm_si]; surf=surfaces[bgm_si]
            vtype=_ps2_get_vtype(streams,surf); verts=_ps2_get_verts_from_streams(streams,surf)
            vif_data=vif_blob[blob_off:blob_off+blob_sz]
            vif_batches_info=_ps2_compute_vif_batch_offsets(vif_data,vtype)
            out+=struct.pack('<I',len(vif_batches_info))
            for bs,_,_ in vif_batches_info: out+=struct.pack('<I',bs)
            for _,po,_ in vif_batches_info: out+=struct.pack('<I',po)
            for _,_,ao in vif_batches_info: out+=struct.pack('<I',ao)
            out+=struct.pack('<I',nv)
            pc_wgts=None
            if surf_idx<len(pc_surfs):
                cs=pc_surfs[surf_idx]; wgt_raw=cs['wgt']
                pc_wgts=[struct.unpack_from('<12f',wgt_raw,i*48) for i in range(len(wgt_raw)//48)]
            for batch in batches:
                for vidx,adc in batch:
                    if vidx<len(verts): px,py,pz=verts[vidx][0]; nx,ny,nz=verts[vidx][1]
                    else: px=py=pz=nx=ny=nz=0.
                    if pc_wgts and vidx<len(pc_wgts):
                        w=pc_wgts[vidx]; dpx,dpy,dpz=w[3],w[4],w[5]; dnx,dny,dnz=w[9],w[10],w[11]
                    else: dpx,dpy,dpz=px,py,pz; dnx,dny,dnz=nx,ny,nz
                    out+=struct.pack('<12f',px,py,pz,dpx,dpy,dpz,nx,ny,nz,dnx,dny,dnz)
    with open(output_path,'wb') as f: f.write(out)
    print(f"  Generated {os.path.basename(output_path)}: {len(out):,}B")


# ─────────────────────────────────────────────────────────────────────────────
# PSP SUPPORT — READER
# ─────────────────────────────────────────────────────────────────────────────

def parse_psp_bgm(path):
    """Parse a PSP BGM file.
    Returns (version, materials, psp_surfs, blob, models, meshes, objects).
    """
    with open(path,'rb') as f: raw=f.read()
    r=_BinReader(raw); version=r.u32()
    materials=_parse_console_materials(r,version)
    blob=b''; ns=r.u32()
    for _ in range(ns):
        dt=r.i32()
        if dt==3: r.i32(); vc=r.u32(); vs=r.u32(); blob=r.read(vc*vs)
        elif dt==1: r.i32(); vc=r.u32(); vs=r.u32(); r.u32(); r.read(vc*vs)
        elif dt==2: r.i32(); ic=r.u32(); r.read(ic*2)
    psp_surfs=[]; nsf=r.u32()
    for _ in range(nsf):
        vals=struct.unpack_from('<8I',raw,r.pos); r.read(32)
        psp_surfs.append({'flags':vals[0],'material_id':vals[1],'nv':vals[2],
            'nb':vals[3],'f4':vals[4],'f5':vals[5],'blob_offset':vals[6],'blob_size':vals[7]})
    models=_parse_console_models(r); meshes=_parse_console_meshes(r)
    objects=_parse_console_objects(r)
    print(f"[PSP BGM] Parsed {path}: {len(materials)} mats, "
          f"{len(psp_surfs)} surfs, {len(models)} models, blob={len(blob):,}B")
    return version,materials,psp_surfs,blob,models,meshes,objects

# PSP constants
_PSP_POS_SCALE=1024.; _PSP_UV_SCALE=2048.; _PSP_UV_OFFSET=16384; _PSP_NORM_SCALE=127.
_GE_PRIM_TRIANGLES=3; _GE_PRIM_STRIP=4

def _psp_ge_cmd(cmd8, param24):
    return struct.pack('<I',((cmd8&0xFF)<<24)|(param24&0xFFFFFF))

def _psp_clamp_i16(v): return max(-32768,min(32767,round(v)))
def _psp_clamp_i8(v):  return max(-128,min(127,round(v)))
def _psp_clamp_u16(v): return max(0,min(65535,round(v)))

def _psp_tris_from_strip(indices):
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


def _psp_tris_from_list(indices):
    """Extract canonical CCW triangles from a FO2 PC triangle-list index sequence."""
    tris = []
    for i in range(0, len(indices) - 2, 3):
        a, b, c = indices[i], indices[i+1], indices[i+2]
        if a != b and b != c and a != c:
            tris.append((a, b, c))
    return tris


def _psp_nvts_stripify(tris, num_samples=10):
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


def _psp_split_pc_strip(indices):
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


def _psp_join_strips(strips):
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


def _psp_strip_with_fallback(strips_or_subs, tris):
    """Join strips and append any uncovered input triangles as isolated fallback strips.

    Returns the final joined index list (always in strip form).
    """
    joined = _psp_join_strips(strips_or_subs)

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
        joined = _psp_join_strips(strips_or_subs + [[a, b, c] for a, b, c in missing])
    return joined


def _psp_optimise_strip(indices, poly_mode, num_samples=16):
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
        (_GE_PRIM_TRIANGLES or GE_PRIM_TRIANGLE_STRIP).
    """
    if poly_mode == 5:
        return indices, 5       # pass through — PC strip is already optimal
    else:
        tris = _psp_tris_from_list(indices)
        if not tris:
            return indices, 4   # empty, keep as list
        strips = _psp_nvts_stripify(tris, num_samples=num_samples)
        return _psp_strip_with_fallback(strips, tris), 5


def _psp_encode_shadow_vertex(pos):
    """6 bytes: int16[3] position."""
    return struct.pack('<3h',
        _psp_clamp_i16(pos[0] * _PSP_POS_SCALE),
        _psp_clamp_i16(pos[1] * _PSP_POS_SCALE),
        _psp_clamp_i16(pos[2] * _PSP_POS_SCALE))


def _psp_encode_standard_vertex(pos, normal, uv):
    """14 bytes: uint16[2] UV + int8[3] norm + int8 pad + int16[3] pos.

    GE vertex layout (from gevtx.h / libgu.h):
      SCEGU_TEXTURE_USHORT : 2 × unsigned short  (bits [1:0] = 2)
      SCEGU_NORMAL_BYTE    : 3 × signed   char   (bits [6:5] = 1)  + 1 byte pad
      SCEGU_VERTEX_SHORT   : 3 × signed   short  (bits [8:7] = 2)
    VTYPE = SCEGU_TEXTURE_USHORT | SCEGU_NORMAL_BYTE | SCEGU_VERTEX_SHORT = 0x122

    UV is encoded as: raw = round(float_uv * _PSP_UV_SCALE) + _PSP_UV_OFFSET = round(float_uv * 2048) + 16384
    The game engine applies the inverse transform (SU=8, TU=-3.5) to recover
    a game_uv = PC_UV * 0.5 + 0.5, sampling the texture in the [0.5, 1.0] range.
    """
    return struct.pack('<2H 3b b 3h',
        _psp_clamp_u16(round(uv[0] * _PSP_UV_SCALE) + _PSP_UV_OFFSET),  # unsigned short (SCEGU_TEXTURE_USHORT)
        _psp_clamp_u16(round(uv[1] * _PSP_UV_SCALE) + _PSP_UV_OFFSET),  # unsigned short
        _psp_clamp_i8 (normal[0] * _PSP_NORM_SCALE),                # signed char    (SCEGU_NORMAL_BYTE)
        _psp_clamp_i8 (normal[1] * _PSP_NORM_SCALE),
        _psp_clamp_i8 (normal[2] * _PSP_NORM_SCALE),
        0,                                                  # pad byte
        _psp_clamp_i16(pos[0]    * _PSP_POS_SCALE),                 # signed short   (SCEGU_VERTEX_SHORT)
        _psp_clamp_i16(pos[1]    * _PSP_POS_SCALE),
        _psp_clamp_i16(pos[2]    * _PSP_POS_SCALE))


def _psp_build_ge_chunk(verts, vtype, poly_mode):
    """Build one complete PSP GE display list chunk for a single surface.

    Returns bytes: 20-byte header + encoded vertex data + padding to 16B alignment.
    """
    if vtype == VTYPE_SHADOW:
        vdata = b''.join(_psp_encode_shadow_vertex(pos) for pos, _n, _u in verts)
    else:
        vdata = b''.join(_psp_encode_standard_vertex(pos, normal, uv)
                         for pos, normal, uv in verts)

    nv        = len(verts)
    prim_type = _GE_PRIM_STRIP if poly_mode == 5 else _GE_PRIM_TRIANGLES

    header  = _psp_ge_cmd(0x14, 0)                            # ORIGIN()   SCE_GE_CMD_ORIGIN  — capture current address as origin
    header += _psp_ge_cmd(0x10, 0)                            # BASE(0)    SCE_GE_CMD_BASE    — upper address bits = 0
    header += _psp_ge_cmd(0x01, 0x14)                         # VADR(0x14) SCE_GE_CMD_VADR    — vertex data at origin+20 (right after this header)
    header += _psp_ge_cmd(0x04, (prim_type << 16) | nv)      # PRIM(type, count)  SCE_GE_CMD_PRIM  — draw primitive
    header += _psp_ge_cmd(0x0b, 0)                            # RET()      SCE_GE_CMD_RET     — return from display list

    chunk   = header + vdata
    pad_len = (16 - len(chunk) % 16) % 16
    return chunk + b'\x00' * pad_len




def _psp_infer_stride(nv, bsz):
    for stride in (14,6):
        raw=20+nv*stride; pad=(16-raw%16)%16
        if raw+pad==bsz: return stride
    return 14

def _psp_extract_surface_tris(blob, surf):
    """Decode one PSP GE surface chunk into triangles ((pos,uv,norm),...)."""
    nv=surf['nv']; bsz=surf['blob_size']
    if nv==0 or bsz<20: return []
    off=surf['blob_offset']
    prim_word=struct.unpack_from('<I',blob,off+12)[0]
    prim_type=(prim_word>>16)&0xFF
    stride=_psp_infer_stride(nv,bsz); is_shadow=(stride==6)
    vdata_start=off+20; verts=[]
    for i in range(nv):
        voff=vdata_start+i*stride
        if voff+stride>len(blob): break
        if is_shadow:
            px,py,pz=struct.unpack_from('<3h',blob,voff)
            verts.append(((px/1024.,py/1024.,pz/1024.),(0.,0.),(0.,0.,1.)))
        else:
            u_raw,v_raw=struct.unpack_from('<2H',blob,voff)
            nx,ny,nz,_pad=struct.unpack_from('<4b',blob,voff+4)
            px,py,pz=struct.unpack_from('<3h',blob,voff+8)
            verts.append(((px/1024.,py/1024.,pz/1024.),
                          ((u_raw-16384)/2048.,(v_raw-16384)/2048.),
                          (nx/127.,ny/127.,nz/127.)))
    tris=[]
    if prim_type==3:
        for i in range(0,len(verts)-2,3):
            v0,v1,v2=verts[i],verts[i+1],verts[i+2]
            if v0[0]==v1[0] or v1[0]==v2[0] or v0[0]==v2[0]: continue
            tris.append((v1,v0,v2))
    elif prim_type==4:
        for i in range(len(verts)-2):
            v0,v1,v2=verts[i],verts[i+1],verts[i+2]
            if v0[0]==v1[0] or v1[0]==v2[0] or v0[0]==v2[0]: continue
            if i%2==0: tris.append((v1,v0,v2))
            else:      tris.append((v0,v1,v2))
    return tris

def psp_to_pc(version, materials, psp_surfs, blob, models, meshes, objects):
    """Convert PSP BGM geometry to PC FO2 format."""
    per_surf_tris=[]; is_shadow_list=[]; material_ids=[]
    for surf in psp_surfs:
        tris=_psp_extract_surface_tris(blob,surf)
        stride=_psp_infer_stride(surf['nv'],surf['blob_size'])
        per_surf_tris.append(tris); is_shadow_list.append(stride==6)
        material_ids.append(surf['material_id'])
    streams,surfaces=_console_tris_to_streams_surfaces(per_surf_tris,is_shadow_list,material_ids,materials=materials)
    old_to_new={}; new_si=0
    for old_si,tris in enumerate(per_surf_tris):
        if tris: old_to_new[old_si]=new_si; new_si+=1
    for m in models:
        m['surfaces']=[old_to_new[s] for s in m['surfaces'] if s in old_to_new]
    out_version=max(version,0x20000)
    print(f"[PSP→PC] {len(psp_surfs)} PSP surfaces → {len(surfaces)} PC surfaces, {len(streams)} streams")
    return out_version,materials,streams,surfaces,models,meshes,objects,False,False

def _psp_convert_surface_dicts(streams, surfaces, si, do_strip=False, num_samples=16):
    """Convert one PC surface (dict) to PSP GE chunk. Returns (chunk_bytes, nv)."""
    surf=surfaces[si]; vtype=_ps2_get_vtype(streams,surf)
    verts_raw=_ps2_get_verts_from_streams(streams,surf)
    if not verts_raw:
        return _psp_build_ge_chunk_dicts([],vtype,4),0
    indices=_ps2_get_indices_from_streams(streams,surf)
    if not indices: indices=list(range(len(verts_raw)))
    indices=[i for i in indices if 0<=i<len(verts_raw)]
    if vtype!=2 and surf['pm']==4:
        eff_samples=num_samples if do_strip else 8
        opt_idx,psp_pm=_psp_optimise_strip(indices,4,eff_samples)
        verts_out=[verts_raw[i] for i in opt_idx if 0<=i<len(verts_raw)]
        chunk=_psp_build_ge_chunk_dicts(verts_out,vtype,psp_pm)
        return chunk,len(verts_out)
    verts_out=[verts_raw[i] for i in indices if 0<=i<len(verts_raw)]
    chunk=_psp_build_ge_chunk_dicts(verts_out,vtype,surf['pm'])
    return chunk,len(verts_out)

def _psp_build_ge_chunk_dicts(verts, vtype, poly_mode):
    """Build PSP GE chunk from (pos,norm,uv,color) tuples."""
    prim_type=_GE_PRIM_STRIP if poly_mode==5 else _GE_PRIM_TRIANGLES
    nv=len(verts)
    if vtype==2:  # shadow — pos only, 6 bytes
        vdata=b''.join(_psp_encode_shadow_vertex(v[0]) for v in verts)
    else:  # standard — 14 bytes: uv(uint16x2) norm(int8x3+pad) pos(int16x3)
        vdata=b''.join(_psp_encode_standard_vertex(v[0],v[1],v[2]) for v in verts)
    header=(_psp_ge_cmd(0x14,0)+_psp_ge_cmd(0x10,0)+_psp_ge_cmd(0x01,0x14)
            +_psp_ge_cmd(0x04,(prim_type<<16)|nv)+_psp_ge_cmd(0x0b,0))
    chunk=header+vdata
    pad_len=(16-len(chunk)%16)%16
    return chunk+b'\x00'*pad_len

def _psp_write_bgm(output_path, version, materials_raw, streams, surfaces, models, meshes, objects,
                   input_path='', do_strip=False, num_samples=16):
    """Write a PSP BGM file from PC-format data."""
    print(f"\nConverting to PSP{' (strip optimised)' if do_strip else ''}...")
    # material areas
    area3d=defaultdict(float); area_uv=defaultdict(float)
    for surf in surfaces:
        mid=surf['mid']; verts=_ps2_get_verts_from_streams(streams,surf)
        indices=_ps2_get_indices_from_streams(streams,surf)
        if not verts or not indices: continue
        is_shadow=(_ps2_get_vtype(streams,surf)==2); pm=surf['pm']
        tris=([(indices[i],indices[i+1],indices[i+2]) for i in range(0,len(indices)-2,3)]
              if pm==4 else [(indices[i],indices[i+1],indices[i+2]) for i in range(len(indices)-2)
                             if indices[i]!=indices[i+1] and indices[i+1]!=indices[i+2]])
        for a,b,c in tris:
            if not all(0<=x<len(verts) for x in (a,b,c)): continue
            pa,pb,pc_=verts[a][0],verts[b][0],verts[c][0]
            ab3=(pb[0]-pa[0],pb[1]-pa[1],pb[2]-pa[2]); ac3=(pc_[0]-pa[0],pc_[1]-pa[1],pc_[2]-pa[2])
            cx=ab3[1]*ac3[2]-ab3[2]*ac3[1]; cy=ab3[2]*ac3[0]-ab3[0]*ac3[2]; cz=ab3[0]*ac3[1]-ab3[1]*ac3[0]
            area3d[mid]+=math.sqrt(cx*cx+cy*cy+cz*cz)/2
            if not is_shadow:
                ua,ub,uc=verts[a][2],verts[b][2],verts[c][2]
                ab2=(ub[0]-ua[0],ub[1]-ua[1]); ac2=(uc[0]-ua[0],uc[1]-ua[1])
                area_uv[mid]+=abs(ab2[0]*ac2[1]-ab2[1]*ac2[0])/2
    chunks=[]; nverts=[]
    for si in range(len(surfaces)):
        chunk,nv=_psp_convert_surface_dicts(streams,surfaces,si,do_strip,num_samples)
        chunks.append(chunk); nverts.append(nv)
        surf=surfaces[si]; vt=_ps2_get_vtype(streams,surf)
        pm_label='SHD' if vt==2 else f'pm{surf["pm"]}'
        print(f"  S{si:2d} m={surf['mid']:2d} {pm_label} "
              f"{surf['niu']:4d}idx -> {nv:4d}v  {len(chunk):5d}B")
    sorted_indices=sorted(range(len(surfaces)),key=lambda si:surfaces[si]['mid'])
    blob=bytearray(); boffs=[0]*len(surfaces); bszs=[0]*len(surfaces)
    for si in sorted_indices: boffs[si]=len(blob); bszs[si]=len(chunks[si]); blob+=chunks[si]
    print(f"  Total: {sum(nverts)}v  blob={len(blob):,}B")
    with open(output_path,'wb') as f:
        f.write(struct.pack('<I',version)); f.write(struct.pack('<I',len(materials_raw)))
        for mi,m in enumerate(materials_raw):
            a9=area3d.get(mi,0.); a8=math.sqrt(a9*area_uv.get(mi,0.)) if a9>0 else 0.
            f.write(struct.pack('<I',m['ident'])); f.write(m['name'].encode('ascii')+b'\x00')
            f.write(struct.pack('<i',m['n_alpha']))
            f.write(struct.pack('<5i',m['v92'],m['n_num_tex'],m['shader_id'],m['n_use_colormap'],m['v74']))
            f.write(struct.pack('<3f',a8,0.,0.)); f.write(struct.pack('<3f',a9,a9,a9))
            f.write(struct.pack('<4i',*m['v98'])); f.write(struct.pack('<4i',*m['v99']))
            f.write(struct.pack('<4i',*m['v100'])); f.write(struct.pack('<4i',*m['v101']))
            f.write(struct.pack('<i',m['v102']))
            for tn in m['tex_names']: f.write(tn.encode('ascii')+b'\x00')
        f.write(struct.pack('<I',1)); f.write(struct.pack('<I',3))
        f.write(struct.pack('<3I',0,len(blob),1)); f.write(blob)
        f.write(struct.pack('<I',len(surfaces)))
        for si,surf in enumerate(surfaces):
            f.write(struct.pack('<8I',0x00001000,surf['mid'],nverts[si],1,0x00000027,0,boffs[si],bszs[si]))
        _write_models_meshes_objects(f,models,meshes,objects)
    isz=os.path.getsize(input_path) if input_path and os.path.exists(input_path) else 0
    osz=os.path.getsize(output_path)
    if isz: print(f"\n  {osz:,}B ({osz/isz*100:.1f}% of input)")
    else:   print(f"\n  {osz:,}B")


# ─────────────────────────────────────────────────────────────────────────────
# XBOX SUPPORT — READER
# ─────────────────────────────────────────────────────────────────────────────

def parse_xbox_bgm(path):
    """Parse an Xbox BGM file.
    Returns (version, materials, xbox_surfs, main_vb, shadow_vb, s2_blob, s3_blob,
             models, meshes, objects).
    """
    with open(path,'rb') as f: raw=f.read()
    r=_BinReader(raw); version=r.u32()
    materials=_parse_console_materials(r,version)
    main_vb=b''; shadow_vb=b''; s2_blob=b''; s3_blob=b''
    ns=r.u32()
    for _ in range(ns):
        dt=r.i32()
        if dt==1:
            r.i32(); vc=r.u32(); vs=r.u32(); r.u32(); data=r.read(vc*vs)
            if vs==12: shadow_vb=data
            else:      main_vb=data
        elif dt==2: r.i32(); ic=r.u32(); r.read(ic*2)
        elif dt==3: r.i32(); vc=r.u32(); vs=r.u32(); r.read(vc*vs)
        elif dt in (4,5):
            r.i32(); bl=r.u32(); r.u32(); blob=r.read(bl)
            if dt==4: s2_blob=blob
            else:     s3_blob=blob
    seek_count=r.u32(); r.u32(); last_off=0
    for _ in range(seek_count): r.u32(); last_off=r.u32()
    n_s3=last_off if seek_count>0 else 0
    mystery_size=16 if n_s3==0 else 8*n_s3; r.read(mystery_size)
    xbox_surfs=[]; nsf=r.u32()
    for _ in range(nsf):
        isveg,mid,vc,flags,pc,pm,niu=struct.unpack_from('<7i',raw,r.pos)
        e0,e1,e2,e3,e4,e5=struct.unpack_from('<6I',raw,r.pos+28); r.read(52)
        xbox_surfs.append({'isveg':isveg,'mid':mid,'vc':vc,'flags':flags,
            'pc':pc,'pm':pm,'niu':niu,'e0':e0,'e1':e1,'e2':e2,'e3':e3,'e4':e4,'e5':e5})
    models=_parse_console_models(r); meshes=_parse_console_meshes(r)
    objects=_parse_console_objects(r)
    print(f"[Xbox BGM] Parsed {path}: {len(materials)} mats, "
          f"{len(xbox_surfs)} surfs, {len(models)} models")
    return version,materials,xbox_surfs,main_vb,shadow_vb,s2_blob,s3_blob,models,meshes,objects

def _xbox_parse_ib_blob(blob):
    """Parse an Xbox NV2A push-buffer blob into a list of (poly_mode, indices) per surface."""
    surfaces=[]; off=0; n=len(blob)
    while off+8<=n:
        word0=struct.unpack_from('<I',blob,off)[0]
        if word0!=0x000417FC: off+=4; continue
        off+=4; word1=struct.unpack_from('<I',blob,off)[0]; off+=4
        if word1==0: continue
        pm=5 if word1==6 else 4; indices=[]
        while off+4<=n:
            cmd=struct.unpack_from('<I',blob,off)[0]
            if cmd==0x000417FC: break
            off+=4
            if (cmd>>30)&3==1:
                cnt=(cmd>>18)&0x7FF
                for _ in range(cnt):
                    if off+4>n: break
                    a,b=struct.unpack_from('<2H',blob,off); off+=4
                    indices.append(a); indices.append(b)
            elif cmd==0x00041808:
                if off+4<=n: idx_odd=struct.unpack_from('<H',blob,off)[0]; off+=4; indices.append(idx_odd)
        if indices: surfaces.append((pm,indices))
    return surfaces

def _xbox_decode_main_vert(vb_data, abs_idx):
    """Decode one Xbox main VB vertex (vs=16, NORMPACKED3). Returns (pos,uv,norm)."""
    off=abs_idx*16
    if off+16>len(vb_data): return (0.,0.,0.),(0.,0.),(0.,0.,1.)
    px_i,py_i,pz_i=struct.unpack_from('<3h',vb_data,off)
    norm_w=struct.unpack_from('<I',vb_data,off+8)[0]
    u_i,v_i=struct.unpack_from('<2h',vb_data,off+12)
    pos=(px_i/1024.,py_i/1024.,pz_i/1024.)
    x_r=norm_w&0x7FF;       nx=(x_r-2048 if x_r>=1024 else x_r)/1023.
    y_r=(norm_w>>11)&0x7FF;  ny=(y_r-2048 if y_r>=1024 else y_r)/1023.
    z_r=(norm_w>>22)&0x3FF;  nz=(z_r-1024 if z_r>=512  else z_r)/511.
    return pos,(u_i/2048.,v_i/2048.),(nx,ny,nz)

def _xbox_decode_shadow_vert(vb_data, abs_idx):
    off=abs_idx*12
    if off+12>len(vb_data): return (0.,0.,0.),(0.,0.),(0.,0.,1.)
    return struct.unpack_from('<3f',vb_data,off),(0.,0.),(0.,0.,1.)

def xbox_to_pc(version, materials, xbox_surfs, main_vb, shadow_vb, s2_blob, s3_blob,
               models, meshes, objects):
    """Convert Xbox BGM geometry to PC FO2 format."""
    s2_parsed=_xbox_parse_ib_blob(s2_blob); s3_parsed=_xbox_parse_ib_blob(s3_blob)
    ib_s2={e5:entry for e5,entry in enumerate(s2_parsed)}
    ib_s3={e5:entry for e5,entry in enumerate(s3_parsed)}
    per_surf_tris=[]; is_shadow_list=[]; material_ids=[]
    for surf in xbox_surfs:
        is_shadow=(surf['flags']==0x0002); mid=surf['mid']
        e0=surf['e0']; e4=surf['e4']; e5=surf['e5']
        pm_entry=ib_s3.get(e5) if e4 else ib_s2.get(e5)
        if pm_entry is None or surf['niu']==0:
            per_surf_tris.append([]); is_shadow_list.append(is_shadow); material_ids.append(mid); continue
        pm,raw_indices=pm_entry
        vb_data=shadow_vb if is_shadow else main_vb
        decode=_xbox_decode_shadow_vert if is_shadow else _xbox_decode_main_vert
        tris=[]
        if pm==4:
            for i in range(0,len(raw_indices)-2,3):
                i0,i1,i2=raw_indices[i],raw_indices[i+1],raw_indices[i+2]
                p0,uv0,n0=decode(vb_data,i0); p1,uv1,n1=decode(vb_data,i1); p2,uv2,n2=decode(vb_data,i2)
                if p0==p1 or p1==p2 or p0==p2: continue
                tris.append(((p2,uv2,n2),(p1,uv1,n1),(p0,uv0,n0)))
        elif pm==5:
            flip=False
            for i in range(len(raw_indices)-2):
                i0,i1,i2=raw_indices[i],raw_indices[i+1],raw_indices[i+2]
                # Degenerate detection MUST use index equality, not position equality.
                # Two different indices can decode to the same float position at UV seams
                # (different UV / normal but same geometric position). Treating those as
                # degenerate skips valid triangles and corrupts flip parity for the rest
                # of the strip. Index repeats are the correct strip-restart signal.
                if i0==i1 or i1==i2 or i0==i2: flip=not flip; continue
                p0,uv0,n0=decode(vb_data,i0); p1,uv1,n1=decode(vb_data,i1); p2,uv2,n2=decode(vb_data,i2)
                if flip: tris.append(((p0,uv0,n0),(p1,uv1,n1),(p2,uv2,n2)))
                else:    tris.append(((p2,uv2,n2),(p1,uv1,n1),(p0,uv0,n0)))
                flip=not flip
        per_surf_tris.append(tris); is_shadow_list.append(is_shadow); material_ids.append(mid)
    streams,surfaces=_console_tris_to_streams_surfaces(per_surf_tris,is_shadow_list,material_ids,materials=materials)
    old_to_new={}; new_si=0
    for old_si,tris in enumerate(per_surf_tris):
        if tris: old_to_new[old_si]=new_si; new_si+=1
    for m in models:
        m['surfaces']=[old_to_new[s] for s in m['surfaces'] if s in old_to_new]
    out_version=max(version,0x20000)
    print(f"[Xbox→PC] {len(xbox_surfs)} Xbox surfaces → {len(surfaces)} PC surfaces, {len(streams)} streams")
    return out_version,materials,streams,surfaces,models,meshes,objects,False,False


# ─────────────────────────────────────────────────────────────────────────────
# XBOX SUPPORT — WRITER
# ─────────────────────────────────────────────────────────────────────────────

# Xbox constants
_XB_VS=16; _XB_SHADOW_VS=12; _XB_POS_SCALE=1024.; _XB_UV_SCALE=2048.
_XB_S3_NIU_THRESH=1200
_XB_TRAIL_HDR=struct.pack('<6H',0x17FC,0x0004,0,0,0,0)

def _xb_is_shadow(surf): return surf.get('flags',0)==0x0002
def _xb_is_stream3(surf): return not _xb_is_shadow(surf) and surf.get('niu',0)>=_XB_S3_NIU_THRESH

def _pc_to_xb_vert(vdata, vs, vi):
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
    ix  = _xb_clamp(round(px * _XB_POS_SCALE), -32767, 32767)
    iy  = _xb_clamp(round(py * _XB_POS_SCALE), -32767, 32767)
    iz  = _xb_clamp(round(pz * _XB_POS_SCALE), -32767, 32767)
    # NORMPACKED3: truncate toward zero (int()), not round
    x_raw = max(-1023, min(1023, int(nx * 1023))) & 0x7FF
    y_raw = max(-1023, min(1023, int(ny * 1023))) & 0x7FF
    z_raw = max(-511,  min(511,  int(nz * 511)))  & 0x3FF
    norm  = (z_raw << 22) | (y_raw << 11) | x_raw
    iu  = _xb_clamp(round(u * _XB_UV_SCALE), -32767, 32767)
    iv  = _xb_clamp(round(v * _XB_UV_SCALE), -32767, 32767)
    return struct.pack('<3hHI2h', ix, iy, iz, 0, norm, iu, iv)










def _xb_make_mystery(s3_chunks):
    """
    Build mystery block bytes.
    For n_s3 = 0: 16 zero bytes.
    For n_s3 > 0: 2*n_s3 uint32s = 8*n_s3 bytes.
      [0]           = 0
      for each surface i:
        append pb_size + 12
        if not last: append cumulative chunk offset (= start of surf i+1)
    s3_chunks: list of (niu, pb_size, chunk_size) from _xb_build_stream3.
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




def _xb_convert_crash_dat(input_path, output_path):
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
            out += struct.pack('<I', nv * _XB_VS)
            vtx = data[off:off + nvb]; off += nvb
            for vi in range(nv):
                out += _pc_to_xb_vert(vtx, vs, vi)
            wgt_sz = nv * 48
            out += data[off:off + wgt_sz]; off += wgt_sz
    with open(output_path, 'wb') as f:
        f.write(out)
    isz = os.path.getsize(input_path)
    print(f"  crash: {os.path.basename(input_path)}: {isz:,}B -> "
          f"{os.path.basename(output_path)}: {len(out):,}B")


def _xb_null_hdr_size(payload):
    """Dynamic null-header size so the next real-header starts 16B-aligned."""
    base = (-payload) % 16
    if base == 0: base = 16
    while base < 12: base += 16
    return base


def _xb_make_null_hdr(size):
    """SET_BEGIN_END(0) [8B] + (size-8) zero bytes."""
    return struct.pack('<2I', 0x000417FC, 0) + b'\x00' * (size - 8)


def _xb_make_surface_pb(pm, indices_bytes):
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


def _xb_clamp(v,lo,hi): return max(lo,min(hi,v))

def _xb_pc_to_xb_vert_dict(streams, surf, vi):
    """Convert surface vertex vi from PC stream to Xbox vs=16."""
    vid=surf['sids'][0]; sv=streams[vid]; vs=sv['vs']; vdata=sv['data']
    vb0=surf['soffs'][0]//vs
    return _pc_to_xb_vert(vdata, vs, vb0+vi)

def write_xbox_bgm(output_path, version, materials_raw, streams, surfaces, models, meshes, objects, input_path=''):
    """Write an Xbox BGM file from PC-format data (dict-based)."""
    print(f"\nConverting to Xbox BGM...")
    has_shadow=any(_xb_is_shadow(s) for s in surfaces)
    # Build VBs
    main_vb=bytearray(); shadow_vb=bytearray()
    by_mid={}
    for si,s in enumerate(surfaces):
        if _xb_is_shadow(s) or s['vc']==0 or s['nst']<1: continue
        by_mid.setdefault(s['mid'],[]).append((s['soffs'][0],si))
    ib_order=[]
    for mid in sorted(by_mid):
        for _,si in sorted(by_mid[mid]): ib_order.append(si)
    seen_ibo=set(ib_order)
    for si,s in enumerate(surfaces):
        if si not in seen_ibo and not _xb_is_shadow(s): ib_order.append(si)
    e0_map={}; cur=0
    for si in ib_order:
        s=surfaces[si]
        if s['vc']==0 or s['nst']<1: e0_map[si]=(0,False); continue
        vid=s['sids'][0]
        if vid>=len(streams): e0_map[si]=(0,False); continue
        sv=streams[vid]; vs=sv['vs']; vdata=sv['data']; vb0=s['soffs'][0]//vs
        e0_map[si]=(cur,False)
        for vi in range(s['vc']): main_vb+=_pc_to_xb_vert(vdata,vs,vb0+vi)
        cur+=s['vc']
    scur=0
    for si,s in enumerate(surfaces):
        if not _xb_is_shadow(s): continue
        if s['vc']==0 or s['nst']<1: e0_map[si]=(0,True); continue
        vid=s['sids'][0]
        if vid>=len(streams): e0_map[si]=(0,True); continue
        sv=streams[vid]; vs=sv['vs']; vdata=sv['data']; vb0=s['soffs'][0]//vs
        e0_map[si]=(scur,True)
        for vi in range(s['vc']):
            base=(vb0+vi)*vs; shadow_vb+=vdata[base:base+_XB_SHADOW_VS]
        scur+=s['vc']
    main_vb=bytes(main_vb); shadow_vb=bytes(shadow_vb)
    if has_shadow:
        print(f"  main VB   : {len(main_vb)//_XB_VS} verts  ({len(main_vb)} B)")
        print(f"  shadow VB : {len(shadow_vb)//_XB_SHADOW_VS} verts  ({len(shadow_vb)} B)")
    else:
        print(f"  VB        : {len(main_vb)//_XB_VS} verts  ({len(main_vb)} B)  [no shadow]")
    # Build IBs
    def get_adj_idx(si):
        s=surfaces[si]
        if s['nst']<2 or s['niu']==0: return b''
        iid=s['sids'][1]
        if iid>=len(streams): return b''
        isv=streams[iid]; idata=isv['data']; ib_off=s['soffs'][1]
        vid=s['sids'][0]; sv=streams[vid]; vs=sv['vs']
        pc_vbase=s['soffs'][0]//vs; xb_e0,_=e0_map[si]; adj=xb_e0-pc_vbase
        out=bytearray(s['niu']*2)
        for k in range(s['niu']):
            if ib_off+k*2+2>len(idata): break
            idx=struct.unpack_from('<H',idata,ib_off+k*2)[0]+adj
            struct.pack_into('<H',out,k*2,max(0,min(65534,idx)))
        return bytes(out)
    def spb(si):
        s=surfaces[si]; return _xb_make_surface_pb(s['pm'],get_adj_idx(si))
    s2_mains=[si for si in ib_order if not _xb_is_stream3(surfaces[si]) and surfaces[si]['niu']>0]
    s2_shadow=[si for si,s in enumerate(surfaces) if _xb_is_shadow(s) and s['niu']>0]
    s2_surfs=s2_mains+s2_shadow
    if s2_surfs:
        s2_blob_b=spb(s2_surfs[0]); seek_entries=[]
        for k in range(1,len(s2_surfs)):
            si=s2_surfs[k]; prev_si=s2_surfs[k-1]; prev_pay=len(spb(prev_si))
            n_size=_xb_null_hdr_size(prev_pay)
            size=12+prev_pay; ib_off=len(s2_blob_b)+n_size
            s2_blob_b+=_xb_make_null_hdr(n_size)+spb(si)
            seek_entries.append((size,ib_off))
        sentinel_size=12+len(spb(s2_surfs[-1]))
        s2_blob_b+=_XB_TRAIL_HDR
    else:
        s2_blob_b=b''; seek_entries=[]; sentinel_size=12
    s3_surfs=[si for si in ib_order if _xb_is_stream3(surfaces[si]) and surfaces[si]['niu']>0]
    s3_blob_b=bytearray(); s3_chunks=[]
    for si in s3_surfs:
        s=surfaces[si]; niu=s['niu']; raw=get_adj_idx(si)
        pb=_xb_make_surface_pb(s['pm'],raw)
        ns=_xb_null_hdr_size(len(pb)); chunk_size=len(pb)+ns
        s3_blob_b+=pb+_xb_make_null_hdr(ns)
        s3_chunks.append((niu,len(pb),chunk_size))
    s3_blob_b=bytes(s3_blob_b)
    print(f"  stream IB-small: {len(s2_blob_b)} B  ({len(seek_entries)+1} surfaces)")
    print(f"  stream IB-large: {len(s3_blob_b)} B  ({len(s3_chunks)} large surfaces)")
    mystery_blob=_xb_make_mystery(s3_chunks)
    def pack_vb(vc,vs,flags,data): return struct.pack('<5I',1,0,vc,vs,flags)+data
    def pack_ib(dt,blob): return struct.pack('<4I',dt,0,len(blob),1)+blob
    if has_shadow:
        xb_streams=[pack_vb(len(shadow_vb)//_XB_SHADOW_VS,_XB_SHADOW_VS,0,shadow_vb),
                    pack_vb(len(main_vb)//_XB_VS,_XB_VS,0,main_vb),
                    pack_ib(4,s2_blob_b),pack_ib(5,s3_blob_b)]
    else:
        xb_streams=[pack_vb(len(main_vb)//_XB_VS,_XB_VS,0,main_vb),
                    pack_ib(4,s2_blob_b),pack_ib(5,s3_blob_b)]
    e5_s2={si:pos for pos,si in enumerate(s2_surfs)}
    e5_s3={si:pos for pos,si in enumerate(s3_surfs)}
    e5_ib={**e5_s2,**e5_s3}; s3_set=set(s3_surfs)
    all_entries=seek_entries+[(sentinel_size,len(s3_surfs))]
    with open(output_path,'wb') as f:
        f.write(struct.pack('<I',version)); f.write(struct.pack('<I',len(materials_raw)))
        for m in materials_raw:
            f.write(struct.pack('<I',m['ident'])); f.write(m['name'].encode('ascii')+b'\x00')
            f.write(struct.pack('<i',m['n_alpha']))
            if version>=0x10004:
                f.write(struct.pack('<i',m['v92'])); f.write(struct.pack('<i',m['n_num_tex']))
                f.write(struct.pack('<i',m['shader_id'])); f.write(struct.pack('<i',m['n_use_colormap']))
                f.write(struct.pack('<i',m['v74']))
                v108=m['v108']; f.write(v108 if isinstance(v108,(bytes,bytearray)) else struct.pack('<3i',*v108))
                v109=m['v109']; f.write(v109 if isinstance(v109,(bytes,bytearray)) else struct.pack('<3i',*v109))
            f.write(struct.pack('<4i',*m['v98'])); f.write(struct.pack('<4i',*m['v99']))
            f.write(struct.pack('<4i',*m['v100'])); f.write(struct.pack('<4i',*m['v101']))
            f.write(struct.pack('<i',m['v102']))
            for tn in m['tex_names']: f.write(tn.encode('ascii')+b'\x00')
        f.write(struct.pack('<I',len(xb_streams)))
        for st in xb_streams: f.write(st)
        f.write(struct.pack('<2I',len(all_entries),0))
        for size,off in all_entries: f.write(struct.pack('<2I',size,off))
        f.write(mystery_blob)
        f.write(struct.pack('<I',len(surfaces)))
        for si,s in enumerate(surfaces):
            e0,is_shad=e0_map.get(si,(0,_xb_is_shadow(s)))
            e2=1 if (has_shadow and not is_shad) else 0
            e4=1 if si in s3_set else 0; e5=e5_ib.get(si,0)
            xb_flags=0x0002 if is_shad else 0x0112
            f.write(struct.pack('<7i',s['isveg'],s['mid'],s['vc'],xb_flags,s['pc'],s['pm'],s['niu']))
            f.write(struct.pack('<6I',e0,2,e2,0,e4,e5))
        _write_models_meshes_objects(f,models,meshes,objects)
    isz=os.path.getsize(input_path) if input_path and os.path.exists(input_path) else 0
    osz=os.path.getsize(output_path)
    if isz: print(f"\n  {osz:,}B ({osz/isz*100:.1f}% of input)")
    else:   print(f"\n  {osz:,}B")
    crash_src,_=_find_crash_dat(input_path) if input_path else (None,None)
    if crash_src and os.path.exists(crash_src):
        crash_out=os.path.splitext(output_path)[0]+'_crash.dat'
        _xb_convert_crash_dat_file(crash_src,crash_out)
    else:
        print("  No crash.dat found, skipping Xbox crash.dat generation")

def _xb_convert_crash_dat_file(input_path, output_path):
    """Convert PC crash.dat to Xbox: vtx vs=32/36 → vs=16, weights unchanged."""
    data=open(input_path,'rb').read(); out=bytearray(); off=0
    nc=struct.unpack_from('<I',data,off)[0]; off+=4; out+=struct.pack('<I',nc)
    for _ in range(nc):
        end=data.index(0,off); out+=data[off:end+1]; off=end+1
        ns=struct.unpack_from('<I',data,off)[0]; off+=4; out+=struct.pack('<I',ns)
        for _ in range(ns):
            nv=struct.unpack_from('<I',data,off)[0]; off+=4
            nvb=struct.unpack_from('<I',data,off)[0]; off+=4
            vs=nvb//nv if nv>0 else 32
            out+=struct.pack('<2I',nv,nv*_XB_VS)
            vtx=data[off:off+nvb]; off+=nvb
            for vi in range(nv): out+=_pc_to_xb_vert(vtx,vs,vi)
            wgt_sz=nv*48; out+=data[off:off+wgt_sz]; off+=wgt_sz
    with open(output_path,'wb') as f: f.write(out)
    isz=os.path.getsize(input_path)
    print(f"  crash.dat: {os.path.basename(input_path)}: {isz:,}B → {len(out):,}B  ({os.path.basename(output_path)})")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE CRASH.DAT → PC CONVERSION  (PS2 / Xbox)
# ─────────────────────────────────────────────────────────────────────────────

def _console_convert_crash_to_pc(src_fmt, input_bgm_path, output_bgm_path,
                                   streams, surfaces, models):
    """Convert a console crash.dat to PC FO2 format alongside a console→PC BGM conversion.

    src_fmt: 'PS2' | 'XBOX' | 'PSP' (PSP has no crash.dat — silently skipped)

    Looks for a crash.dat adjacent to input_bgm_path using the normal
    _find_crash_dat search order, then calls the appropriate converter.
    The output crash.dat is placed alongside output_bgm_path.
    """
    if src_fmt == 'PSP':
        return   # PSP games have no crash.dat

    crash_src, _ = _find_crash_dat(input_bgm_path)
    if not crash_src:
        print(f"  crash.dat: none found alongside {os.path.basename(input_bgm_path)}, skipping")
        return

    crash_dst = os.path.splitext(output_bgm_path)[0] + '_crash.dat'

    if src_fmt == 'PS2':
        _convert_ps2_crash_to_pc(crash_src, crash_dst, streams, surfaces, models)
    elif src_fmt == 'XBOX':
        _convert_xbox_crash_to_pc(crash_src, crash_dst, streams, surfaces, models)
    else:
        # Unknown console format — just copy so the caller at least has something
        shutil.copy2(crash_src, crash_dst)
        print(f"  crash.dat: copied {os.path.basename(crash_src)} "
              f"(no converter for {src_fmt})")


_CONSOLE_TARGETS = {'PS2', 'PSP', 'XBOX'}
_PC_TARGETS      = {'FO1', 'FO2', 'FOUC'}
_ALL_TARGETS     = _CONSOLE_TARGETS | _PC_TARGETS


def main():
    args = sys.argv[1:]
    do_full       = '-full'       in args
    do_clean      = '-clean'      in args or do_full
    do_optimize   = '-optimize'   in args or do_full
    do_menucar    = '-menucar'    in args
    do_lightorder = '-lightorder' in args
    do_windflip   = '-windflip'   in args
    do_lighthacks = '-lighthacks' in args
    lighthacks_targets = None
    if do_lighthacks:
        lh_idx = args.index('-lighthacks')
        if lh_idx+1<len(args) and not args[lh_idx+1].startswith('-'):
            lighthacks_targets = set(args[lh_idx+1].split(','))
    convert_target = None
    for i,a in enumerate(args):
        if a=='-convert' and i+1<len(args) and args[i+1].upper() in _ALL_TARGETS:
            convert_target=args[i+1].upper(); break
    do_strip = '-strip' in args
    num_samples = 16
    if '-samples' in args:
        idx=args.index('-samples')
        try: num_samples=int(args[idx+1])
        except: pass

    flags={'-clean','-optimize','-full','-menucar','-windflip',
           '-lighthacks','-lightorder','-convert','-strip','-samples',
           'FO1','FO2','FOUC','PS2','PSP','XBOX'}
    if lighthacks_targets is not None:
        lh_idx=args.index('-lighthacks'); flags.add(args[lh_idx+1])
    if '-samples' in args:
        idx=args.index('-samples')
        if idx+1<len(args): flags.add(args[idx+1])
    pos_args=[a for a in args if a not in flags]
    any_op=(do_clean or do_optimize or do_menucar or do_windflip or
            do_lighthacks or do_lightorder or convert_target)

    if not pos_args:
        print("Usage: bgm_tool.py <input.bgm> [output.bgm] <flags>")
        print()
        print("  -clean            Remove unreferenced (orphan) streams")
        print("  -optimize         Vertex deduplication + stream merging")
        print("  -full             Shortcut for -clean -optimize")
        print("  -menucar          Reorder surfaces to FO2 menucar draw order")
        print("  -lightorder       Reorder materials+surfaces by draw priority")
        print("  -windflip         Fix triangles with inverted winding")
        print("  -lighthacks [m,..]  Duplicate light _b surfaces")
        print("  -convert <fmt>    FO1, FO2, FOUC, PS2, PSP, XBOX")
        print("  -strip            (PSP output) higher-quality stripification")
        print("  -samples N        NvTriStrip sample count (default 16)")
        print()
        print("Auto-detects PS2/PSP/Xbox input and converts to PC FO2 when no -convert given.")
        sys.exit(1)

    inp=pos_args[0]
    if not os.path.exists(inp):
        print(f"ERROR: file not found: {inp}",file=sys.stderr); sys.exit(1)

    fmt=detect_bgm_format(inp)
    print(f"Reading  {os.path.basename(inp)}  ({os.path.getsize(inp):,} B)  [{fmt}]")
    is_console_input=(fmt in ('PS2','PSP','XBOX'))

    if len(pos_args)>=2:
        out=pos_args[1]
    else:
        stem,ext=os.path.splitext(inp)
        if convert_target:
            out=stem+f'_{convert_target.lower()}'+ext
        elif is_console_input and not convert_target:
            out=stem+'_pc'+ext
        elif do_optimize:
            out=stem+'_opt'+ext
        elif do_windflip and not do_clean and not do_menucar:
            out=stem+'_wf'+ext
        elif do_menucar and not do_clean:
            out=stem+'_mc'+ext
        else:
            out=stem+'_clean'+ext

    # ── PATH A: console input → reconstruct PC geometry ──
    # ── Console → FOUC two-step shortcut ─────────────────────────────────
    # Direct console→FOUC conversion is unreliable (stream merging issues).
    # Instead: console→FO2 PC (temp) then FO2 PC→FOUC, then remove temp.
    if is_console_input and convert_target == 'FOUC':
        import subprocess as _sp
        # Place the temp FO2 file alongside the input (clean stem, no crash ambiguity)
        _inp_stem = os.path.splitext(inp)[0]
        tmp_fo2 = _inp_stem + '__tmp_fo2__.bgm'
        try:
            # Step 1: console → FO2 PC
            print(f'  [two-step] {fmt} → FO2 PC ...')
            _sp.run([sys.executable, __file__, inp, tmp_fo2, '-convert', 'FO2'], check=True)
            # Step 2: FO2 PC → FOUC (reads tmp_fo2 + its crash.dat)
            print(f'  [two-step] FO2 PC → FOUC ...')
            _sp.run([sys.executable, __file__, tmp_fo2, out,
                     '-convert', 'FOUC'], check=True)
        finally:
            # Remove temp FO2 BGM and its crash.dat (any _crash.dat or -crash.dat)
            for _suf in ('.bgm', '_crash.dat', '-crash.dat'):
                _p = _inp_stem + '__tmp_fo2__' + _suf
                try:
                    if os.path.exists(_p): os.remove(_p)
                except OSError: pass
        print('\nDone!'); return

    if is_console_input:
        if fmt=='PS2':
            raw_data=parse_ps2_bgm(inp)
            (version,materials_raw,streams,surfaces,models,
             meshes,objects,is_fouc,is_fo1)=ps2_to_pc(*raw_data)
        elif fmt=='PSP':
            raw_data=parse_psp_bgm(inp)
            (version,materials_raw,streams,surfaces,models,
             meshes,objects,is_fouc,is_fo1)=psp_to_pc(*raw_data)
        elif fmt=='XBOX':
            raw_data=parse_xbox_bgm(inp)
            (version,materials_raw,streams,surfaces,models,
             meshes,objects,is_fouc,is_fo1)=xbox_to_pc(*raw_data)

        if do_lightorder: surfaces,models,materials_raw=op_lightorder(surfaces,models,materials_raw)
        if do_clean:      streams,surfaces=op_strip(streams,surfaces)
        if do_menucar:    surfaces,models=op_menucar(surfaces,models,materials_raw,version)
        if do_optimize:
            (streams,surfaces,surf_seen,surf_vs,surf_original_vc,
             streams_orig,surfaces_orig)=op_optimize(streams,surfaces)
        if do_windflip:   streams,surfaces=op_windflip(streams,surfaces,is_fouc,models=models)

        if convert_target and convert_target in _PC_TARGETS:
            # Save FO2-float streams/surfaces before op_convert changes them to FOUC int16.
            # _console_convert_crash_to_pc needs float VB positions for matching.
            streams_for_crash, surfaces_for_crash = streams, surfaces
            streams,surfaces,objects,materials_raw,version,is_fouc,is_fo1=op_convert(
                streams,surfaces,objects,materials_raw,version,is_fouc,is_fo1,convert_target)
            print(f"\nWriting  {os.path.basename(out)} ...")
            write_bgm(out,version,materials_raw,streams,surfaces,models,meshes,objects)
            print(f"  {os.path.getsize(out):,} B")
            print()
            _console_convert_crash_to_pc(fmt,inp,out,streams_for_crash,surfaces_for_crash,models)
        elif convert_target=='PS2':
            _ps2_write_bgm(out,version,materials_raw,streams,surfaces,models,meshes,objects,input_path=inp)
        elif convert_target=='PSP':
            _psp_write_bgm(out,version,materials_raw,streams,surfaces,models,meshes,objects,
                           input_path=inp,do_strip=do_strip,num_samples=num_samples)
        elif convert_target=='XBOX':
            write_xbox_bgm(out,version,materials_raw,streams,surfaces,models,meshes,objects,input_path=inp)
        else:
            # default: console input -> PC FO2
            print(f"\nWriting  {os.path.basename(out)} ...")
            write_bgm(out,version,materials_raw,streams,surfaces,models,meshes,objects)
            isz=os.path.getsize(inp); osz=os.path.getsize(out); delta=isz-osz
            print(f"  {isz:>10,} B  →  {osz:>10,} B"
                  f"  ({'saved' if delta>=0 else 'grew by'} {abs(delta):,} B, {abs(delta)/isz*100:.1f}%)")
            print()
            _console_convert_crash_to_pc(fmt,inp,out,streams,surfaces,models)
        print("\nDone!"); return

    # ── PATH B: PC input (FO1 / FO2 / FOUC) ──
    version,materials_raw,streams,surfaces,models,meshes,objects,is_fouc,is_fo1=parse_bgm(inp)
    if is_fouc:  ver_label='FOUC'
    elif is_fo1: ver_label=f'FO1 (0x{version:X})'
    else:        ver_label='FO2'
    print(f"  version=0x{version:X}  {ver_label}  "
          f"{len(materials_raw)} mat  {len(streams)} streams  "
          f"{len(surfaces)} surfaces  {len(models)} models")

    if not any_op:
        print("  No operation specified — nothing to do."); sys.exit(0)

    # ── FOUC → console two-step shortcut ──────────────────────────────────
    # FOUC→console: first convert FOUC→FO2 PC (with -full + any user flags),
    # then convert FO2 PC→console (no extra flags).  Temp file placed
    # alongside the input BGM.
    if is_fouc and convert_target in _CONSOLE_TARGETS:
        import subprocess as _sp
        _inp_stem = os.path.splitext(inp)[0]
        tmp_fo2 = _inp_stem + '__tmp_fo2__.bgm'
        # Collect user-supplied flags to forward (everything except positional
        # args and the -convert <target> pair).
        _raw = sys.argv[1:]
        _extra = []
        _skip_next = False
        for _i, _a in enumerate(_raw):
            if _skip_next: _skip_next = False; continue
            if _a in (inp, out): continue
            if _a == '-convert': _skip_next = True; continue
            if _a.upper() in _CONSOLE_TARGETS: continue
            _extra.append(_a)
        try:
            # Step 1: FOUC → FO2 PC  (-convert FO2 -full <user flags>)
            print(f'  [two-step] FOUC → FO2 PC ...')
            _sp.run([sys.executable, __file__, inp, tmp_fo2,
                     '-convert', 'FO2', '-full'] + _extra, check=True)
            # Step 2: FO2 PC → console  (no extra flags)
            print(f'  [two-step] FO2 PC → {convert_target} ...')
            _sp.run([sys.executable, __file__, tmp_fo2, out,
                     '-convert', convert_target], check=True)
        finally:
            for _suf in ('.bgm', '_crash.dat', '-crash.dat'):
                _p = _inp_stem + '__tmp_fo2__' + _suf
                try:
                    if os.path.exists(_p): os.remove(_p)
                except OSError: pass
        print('\nDone!'); return

    if convert_target=='PS2':
        _ps2_write_bgm(out,version,materials_raw,streams,surfaces,models,meshes,objects,input_path=inp)
        print("\nDone!"); return
    if convert_target=='PSP':
        _psp_write_bgm(out,version,materials_raw,streams,surfaces,models,meshes,objects,
                       input_path=inp,do_strip=do_strip,num_samples=num_samples)
        print("\nDone!"); return
    if convert_target=='XBOX':
        write_xbox_bgm(out,version,materials_raw,streams,surfaces,models,meshes,objects,input_path=inp)
        print("\nDone!"); return

    # ── standard PC pipeline ──
    crash_src,crash_standalone=_find_crash_dat(inp)
    opt_crash_info=None; src_is_fouc=is_fouc; b_dupe_map=[]
    converting_from_fouc=is_fouc and convert_target in ('FO1','FO2')

    if converting_from_fouc:
        if do_windflip: streams,surfaces=op_windflip(streams,surfaces,is_fouc,models=models)
        if do_lightorder: surfaces,models,materials_raw=op_lightorder(surfaces,models,materials_raw)
        streams_pre_convert=streams; surfaces_pre_convert=surfaces
        if convert_target:
            streams,surfaces,objects,materials_raw,version,is_fouc,is_fo1=op_convert(
                streams,surfaces,objects,materials_raw,version,is_fouc,is_fo1,convert_target)
        if do_lighthacks:
            streams,surfaces,b_dupe_map=op_lighthacks(
                streams,surfaces,models,materials_raw,objects,targets=lighthacks_targets)
        if do_clean: streams,surfaces=op_strip(streams,surfaces)
        if do_menucar: surfaces,models=op_menucar(surfaces,models,materials_raw,version)
        if do_optimize:
            (streams,surfaces,surf_seen,surf_vs,surf_original_vc,
             streams_orig,surfaces_orig)=op_optimize(streams,surfaces)
            opt_crash_info=(surf_seen,surf_vs,surf_original_vc,streams_orig,surfaces_orig)
    else:
        if do_lightorder: surfaces,models,materials_raw=op_lightorder(surfaces,models,materials_raw)
        if do_clean: streams,surfaces=op_strip(streams,surfaces)
        if do_menucar: surfaces,models=op_menucar(surfaces,models,materials_raw,version)
        if do_optimize:
            (streams,surfaces,surf_seen,surf_vs,surf_original_vc,
             streams_orig,surfaces_orig)=op_optimize(streams,surfaces)
            opt_crash_info=(surf_seen,surf_vs,surf_original_vc,streams_orig,surfaces_orig)
        if do_windflip: streams,surfaces=op_windflip(streams,surfaces,is_fouc,models=models)
        streams_pre_convert=streams; surfaces_pre_convert=surfaces
        if convert_target:
            streams,surfaces,objects,materials_raw,version,is_fouc,is_fo1=op_convert(
                streams,surfaces,objects,materials_raw,version,is_fouc,is_fo1,convert_target)
        if do_lighthacks:
            streams,surfaces,b_dupe_map=op_lighthacks(
                streams,surfaces,models,materials_raw,objects,targets=lighthacks_targets)

    print(f"\nWriting  {os.path.basename(out)} ...")
    write_bgm(out,version,materials_raw,streams,surfaces,models,meshes,objects)
    isz=os.path.getsize(inp); osz=os.path.getsize(out); delta=isz-osz
    print(f"  {isz:>10,} B  →  {osz:>10,} B"
          f"  ({'saved' if delta>=0 else 'grew by'} {abs(delta):,} B, {abs(delta)/isz*100:.1f}%)")

    print()
    dst_is_fouc=is_fouc
    if not crash_src:
        print(f"  crash.dat: none found alongside {os.path.basename(inp)}, skipping")
    elif opt_crash_info is not None:
        print(f"  crash.dat: remapping {os.path.basename(crash_src)} ...")
        if src_is_fouc!=dst_is_fouc:
            tmp_bgm=out+'.__tmp__'
            _remap_crash_dat(crash_src,tmp_bgm,models,*opt_crash_info,
                             is_fouc=src_is_fouc,is_standalone=crash_standalone)
            tmp_crash=_crash_dst_path(tmp_bgm,crash_standalone)
            dst_crash=_crash_dst_path(out,crash_standalone)
            print(f"  crash.dat: converting format "
                  f"({'FOUC→FO2/FO1' if src_is_fouc else 'FO2/FO1→FOUC'}) ...")
            _convert_crash_dat_file(tmp_crash,dst_crash,src_is_fouc,dst_is_fouc,
                                     surfaces_pre_convert,streams_pre_convert,streams,
                                     models=models,surfaces_dst=surfaces)
            if os.path.exists(tmp_crash): os.remove(tmp_crash)
            if os.path.exists(tmp_bgm):   os.remove(tmp_bgm)
            print(f"  crash.dat: done → {os.path.getsize(dst_crash):,} B  ({os.path.basename(dst_crash)})")
        else:
            _remap_crash_dat(crash_src,out,models,*opt_crash_info,
                             is_fouc=src_is_fouc,is_standalone=crash_standalone)
    elif convert_target and src_is_fouc!=dst_is_fouc:
        dst_crash=_crash_dst_path(out,crash_standalone)
        print(f"  crash.dat: converting format "
              f"({'FOUC→FO2/FO1' if src_is_fouc else 'FO2/FO1→FOUC'}) ...")
        _convert_crash_dat_file(crash_src,dst_crash,src_is_fouc,dst_is_fouc,
                                 surfaces_pre_convert,streams_pre_convert,streams,
                                 models=models,surfaces_dst=surfaces)
        print(f"  crash.dat: {os.path.getsize(crash_src):,} B → {os.path.getsize(dst_crash):,} B  ({os.path.basename(dst_crash)})")
    else:
        _copy_crash_dat(inp,out)

    if do_lighthacks and b_dupe_map and crash_src:
        dst_crash=_crash_dst_path(out,crash_standalone)
        if not os.path.exists(dst_crash): shutil.copy2(crash_src,dst_crash)
        crash_nodes=_parse_crash_dat(dst_crash,is_fouc=dst_is_fouc)
        crash_node_idx={}
        for ni,(node_name,_) in enumerate(crash_nodes):
            crash_node_idx[node_name.replace('_crash','')]=ni
        import copy as _copy2; patched=0
        for model_name,orig_pos in b_dupe_map:
            ni=crash_node_idx.get(model_name)
            if ni is None: continue
            node_name,crash_surfs=crash_nodes[ni]
            if orig_pos<len(crash_surfs):
                crash_surfs.append(_copy2.copy(crash_surfs[orig_pos]))
                crash_nodes[ni]=(node_name,crash_surfs); patched+=1
        if patched:
            _write_crash_dat(dst_crash,crash_nodes,is_fouc=dst_is_fouc)
            print(f"  crash.dat: mirrored {patched} _b duplicate(s) ({os.path.basename(dst_crash)})")
    print("\nDone!")


if __name__ == '__main__':
    main()
