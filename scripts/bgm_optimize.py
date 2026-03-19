#!/usr/bin/env python3
"""
FO2 BGM optimizer — FlatOut 2 PC BGM file optimizer: clean orphan streams and/or optimize geometry.

by ravenDS (github.com/ravenDS)

Operations (can be combined; -clean always runs before -optimize):
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

  -menucar
      Reorder surfaces to FO2 menucar draw order

  -windflip
      Experimental fix for inconsistent triangle winding on FOUC (+ FO2) BGM files.
      This corrects the "missing faces" visual artifact seen on some FOUC car parts 
      where a subset of triangles were authored with reversed winding relative to their normals.
      Operates directly on the index buffer streams; vertex data is unchanged.
      crash.dat is copied alongside unchanged (vertex ordering is not affected).

  -full
      Shortcut for -clean & -optimize

Usage:
  bgm_optimize.py <input.bgm> [output.bgm] -clean [-optimize]
  bgm_optimize.py <input.bgm> [output.bgm] -optimize [-clean]
  bgm_optimize.py <input.bgm> [output.bgm] -windflip

Default output suffix: _clean.bgm (-clean only), _opt.bgm (-optimize only or both),
                       _wf.bgm (-windflip only)
"""

import struct, sys, os, shutil, copy

def _read_string(f):
    s = b""
    while True:
        c = f.read(1)
        if not c or c == b'\x00':
            break
        s += c
    return s.decode('ascii', errors='replace')


# parser

def parse_bgm(path):
    """Parse a PC BGM file.

    Returns (version, materials_raw, streams, surfaces, models, rest) where:
      materials_raw : list of raw byte blobs (round-trip exact)
      streams       : list of dicts  {dt, fc, data, [vc, vs, flags] or [ic]}
      surfaces      : list of dicts  {isveg, mid, vc, flags, pc, pm, niu,
                                       extra, nst, sids, soffs}
      models        : list of dicts  {ident, unk, name, center, radius,
                                       f_radius, surfaces}
      rest          : raw bytes for meshes + objects (round-trip exact)
    """
    with open(path, 'rb') as f:
        version = struct.unpack('<I', f.read(4))[0]

        # materials — raw blobs for exact round-trip
        nm = struct.unpack('<I', f.read(4))[0]
        materials_raw = []
        for _ in range(nm):
            start = f.tell()
            f.read(4)                                    # identifier
            while f.read(1) not in (b'\x00', b''):       # name (null-terminated)
                pass
            f.read(4)                                    # alpha
            if version >= 0x10004:
                f.read(20); f.read(12); f.read(12)
            f.read(64)                                   # v98-v101
            f.read(4)                                    # v102
            for _ in range(3):
                while f.read(1) not in (b'\x00', b''):   # tex names
                    pass
            end = f.tell()
            f.seek(start)
            materials_raw.append(f.read(end - start))

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

        # meshes + objects, raw round-trip
        rest = f.read()

    is_fo1 = (version < 0x20000) and not is_fouc
    return version, materials_raw, streams, surfaces, models, rest, is_fouc, is_fo1


# writer

def write_bgm(path, version, materials_raw, streams, surfaces, models, rest):
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', version))

        f.write(struct.pack('<I', len(materials_raw)))
        for blob in materials_raw:
            f.write(blob)

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

        f.write(rest)


# crash.dat helpers

def _find_crash_dat(bgm_path):
    base = os.path.splitext(bgm_path)[0]
    for suffix in ('_crash.dat', '_crash.DAT'):
        p = base + suffix
        if os.path.exists(p):
            return p, suffix
    return None, None


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
    src, suffix = _find_crash_dat(input_bgm)
    if not src:
        print(f"  crash.dat: none found alongside "
              f"{os.path.basename(input_bgm)}, skipping")
        return
    dst = os.path.splitext(output_bgm)[0] + suffix
    if os.path.abspath(src) == os.path.abspath(dst):
        print(f"  crash.dat: same path, no copy needed ({os.path.basename(src)})")
    else:
        shutil.copy2(src, dst)
        print(f"  crash.dat: copied  {os.path.basename(src)}"
              f" -> {os.path.basename(dst)}")


def _remap_crash_dat(crash_path, output_bgm, models,
                     surf_seen, surf_vs, surf_original_vc,
                     streams_orig, surfaces_orig,
                     is_fouc=False):
    """Remap crash.dat vertex arrays to match deduplicated vertex ordering.

    FO2 / FO1: remaps both the crash vbuf (vtx) and the 48-byte weight array.
    FOUC:       no vbuf; remaps only the 40-byte weight array.
    In both cases the vertex-to-new-index lookup uses the original BGM VB data
    (same raw-bytes key used by op_optimize's surf_seen dict).
    """
    suffix = '_crash.dat'
    for s in ('_crash.dat', '_crash.DAT'):
        if crash_path.endswith(s):
            suffix = s
            break

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

    dst = os.path.splitext(output_bgm)[0] + suffix
    _write_crash_dat(dst, new_nodes, is_fouc=is_fouc)
    old_sz = os.path.getsize(crash_path)
    new_sz = os.path.getsize(dst)
    print(f"  crash.dat: {old_sz:,} B → {new_sz:,} B  ({os.path.basename(dst)})")


# clean orphan streams

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


# vertex dedup + stream merge

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


# menucar surface ordering

# priority by material name prefix, lower number renders first (observed on FO2 menucar files)
# stable sort: ties preserve original order.
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

    # parse material names from raw blobs (name = null-terminated string after 4-byte identifier)
    def mat_name(blob):
        off = 4  # skip identifier
        end = blob.index(0, off)
        return blob[off:end].decode('ascii', errors='replace')

    mat_names = [mat_name(b) for b in materials_raw]

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




# winding flip

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
            # Only triangle lists (pm=4) supported; strips rare in practice
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
                # flip by swapping i1 and i2 in the file's entry
                # file entry (i0,i1,i2) -> reversed (i2,i1,i0)
                # swapping i1<->i2 gives file (i0,i2,i1) -> reversed (i1,i2,i0)
                #   which has the opposite geometric orientation
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

    # Convert mutable IB bytearrays back to bytes and update ic counts
    for s in streams:
        if s['dt'] == 2 and isinstance(s['data'], bytearray):
            s['data'] = bytes(s['data'])

    return streams, surfaces


def main():
    args = sys.argv[1:]

    do_full      = '-full'       in args
    do_clean     = '-clean'      in args or do_full
    do_optimize  = '-optimize'   in args or do_full
    do_menucar   = '-menucar'    in args
    do_windflip  = '-windflip'   in args
    flags        = {'-clean', '-optimize', '-full', '-menucar', '-windflip'}
    pos_args     = [a for a in args if a not in flags]

    if not (do_clean or do_optimize or do_menucar or do_windflip) or not pos_args:
        print("Usage: bgm_optimize.py <input.bgm> [output.bgm] <flags>")
        print()
        print("  -clean     Remove unreferenced (orphan) streams")
        print("  -optimize  Vertex deduplication + stream merging")
        print("  -full      Shortcut for -clean -optimize")
        print("  -menucar   Reorder surfaces to FO2 menucar draw order")
        print("  -windflip  Fix triangles with inverted winding (FOUC/FO2)")
        print()
        print("Flags can be combined; execution order: -clean -> -menucar -> -optimize -> -windflip")
        sys.exit(1)

    inp = pos_args[0]
    if not os.path.exists(inp):
        print(f"ERROR: file not found: {inp}", file=sys.stderr)
        sys.exit(1)

    if len(pos_args) >= 2:
        out = pos_args[1]
    else:
        stem, ext = os.path.splitext(inp)
        if do_optimize:
            out = stem + '_opt' + ext
        elif do_windflip and not do_clean and not do_menucar:
            out = stem + '_wf' + ext
        elif do_menucar and not do_clean:
            out = stem + '_mc' + ext
        else:
            out = stem + '_clean' + ext

    print(f"Reading  {os.path.basename(inp)}  ({os.path.getsize(inp):,} B)")
    version, materials_raw, streams, surfaces, models, rest, is_fouc, is_fo1 = parse_bgm(inp)
    if is_fouc:
        ver_label = 'FOUC'
    elif is_fo1:
        ver_label = f'FO1 (0x{version:X})'
    else:
        ver_label = 'FO2'
    print(f"  version=0x{version:X}  {ver_label}  "
          f"{len(materials_raw)} mat  {len(streams)} streams  "
          f"{len(surfaces)} surfaces  {len(models)} models")

    # crash.dat
    crash_src, crash_suffix = _find_crash_dat(inp)
    opt_crash_info = None

    # -clean (always first)
    if do_clean:
        streams, surfaces = op_strip(streams, surfaces)

    # -menuorder (after clean, before optimize)
    if do_menucar:
        surfaces, models = op_menucar(surfaces, models, materials_raw, version)

    # -optimize (always last before windflip)
    if do_optimize:
        (streams, surfaces,
         surf_seen, surf_vs, surf_original_vc,
         streams_orig, surfaces_orig) = op_optimize(streams, surfaces)
        opt_crash_info = (surf_seen, surf_vs, surf_original_vc,
                          streams_orig, surfaces_orig)

    # -windflip (always last)
    if do_windflip:
        streams, surfaces = op_windflip(streams, surfaces, is_fouc, models=models)

    # write BGM
    print(f"\nWriting  {os.path.basename(out)} ...")
    write_bgm(out, version, materials_raw, streams, surfaces, models, rest)
    isz = os.path.getsize(inp)
    osz = os.path.getsize(out)
    print(f"  {isz:>10,} B  →  {osz:>10,} B"
          f"  (saved {isz-osz:,} B, {(isz-osz)/isz*100:.1f}%)")

    # crash.dat
    print()
    if not crash_src:
        print(f"  crash.dat: none found alongside {os.path.basename(inp)}, skipping")
    elif opt_crash_info is not None:
        # optimize ran: remap vertex ordering
        print(f"  crash.dat: remapping {os.path.basename(crash_src)} ...")
        _remap_crash_dat(crash_src, out, models, *opt_crash_info, is_fouc=is_fouc)
    else:
        # strip / windflip only: vertex data unchanged, plain copy
        _copy_crash_dat(inp, out)

    print("\nDone!")


if __name__ == '__main__':
    main()
