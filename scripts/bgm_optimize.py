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

  -full
      Shortcut for -clean & -optimize

Usage:
  bgm_process.py <input.bgm> [output.bgm] -clean [-optimize]
  bgm_process.py <input.bgm> [output.bgm] -optimize [-clean]

Default output suffix: _clean.bgm (-clean only), _opt.bgm (-optimize only or both)
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

        # surfaces
        nsurf = struct.unpack('<I', f.read(4))[0]
        surfaces = []
        for _ in range(nsurf):
            isveg, mid, vc, flags, pc, pm, niu = struct.unpack('<7i', f.read(28))
            extra = b''
            if version < 0x20000:
                extra = f.read(24)
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

    return version, materials_raw, streams, surfaces, models, rest


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


def _parse_crash_dat(path):
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
            nv  = struct.unpack_from('<I', data, off)[0]; off += 4
            nvb = struct.unpack_from('<I', data, off)[0]; off += 4
            vs  = nvb // nv if nv > 0 else 0
            vtx = data[off : off + nvb];   off += nvb
            wgt = data[off : off + nv*48]; off += nv * 48
            surfs.append({'vtx': vtx, 'wgt': wgt, 'vs': vs})
        nodes.append((name, surfs))
    return nodes


def _write_crash_dat(path, nodes):
    out = bytearray()
    out += struct.pack('<I', len(nodes))
    for name, surfs in nodes:
        out += name.encode('ascii') + b'\x00'
        out += struct.pack('<I', len(surfs))
        for s in surfs:
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
                     streams_orig, surfaces_orig):
    """Remap crash.dat vertex arrays to match deduplicated vertex ordering."""
    suffix = '_crash.dat'  # default
    for s in ('_crash.dat', '_crash.DAT'):
        if crash_path.endswith(s):
            suffix = s
            break
    crash_nodes = _parse_crash_dat(crash_path)
    model_by_name = {m['name']: m['surfaces'] for m in models}

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
            new_vtx = bytearray(new_vc * vs)
            new_wgt = bytearray(new_vc * 48)
            old_vtx = crash_surf['vtx']
            old_wgt = crash_surf['wgt']

            for old_i in range(original_vc):
                if old_i >= len(old_wgt) // 48:
                    break
                rec = bytes(vdata[voff + old_i*vs : voff + old_i*vs + vs])
                if rec not in seen:
                    continue
                new_i = seen[rec]
                new_vtx[new_i*vs  : new_i*vs+vs] = old_vtx[old_i*vs  : old_i*vs+vs]
                new_wgt[new_i*48  : new_i*48+48] = old_wgt[old_i*48  : old_i*48+48]

            new_surfs.append({'vtx': bytes(new_vtx),
                              'wgt': bytes(new_wgt), 'vs': vs})

        new_nodes.append((node_name, new_surfs))

    dst = os.path.splitext(output_bgm)[0] + suffix
    _write_crash_dat(dst, new_nodes)
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
    fmt_order = []
    fmt_seen_set = set()
    for si in range(len(surfaces)):
        if si not in surf_vs:
            continue
        key = (surf_vs[si], surf_vflags[si])
        if key not in fmt_seen_set:
            fmt_seen_set.add(key)
            fmt_order.append(key)

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
        new_streams.append({'dt': 1, 'fc': 0, 'vc': vc, 'vs': vs,
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



def main():
    args = sys.argv[1:]

    do_full      = '-full'       in args
    do_clean     = '-clean'      in args or do_full
    do_optimize  = '-optimize'   in args or do_full
    do_menucar   = '-menucar'    in args
    flags        = {'-clean', '-optimize', '-full', '-menucar'}
    pos_args     = [a for a in args if a not in flags]

    if not (do_clean or do_optimize or do_menucar) or not pos_args:
        print("Usage: bgm_process.py <input.bgm> [output.bgm] <flags>")
        print()
        print("  -clean     Remove unreferenced (orphan) streams")
        print("  -optimize  Vertex deduplication + stream merging")
        print("  -full      Shortcut for -clean -optimize")
        print("  -menucar   Reorder surfaces to FO2 menucar draw order")
        print()
        print("Flags can be combined; execution order: -clean -> -menucar -> -optimize")
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
        elif do_menucar and not do_clean:
            out = stem + '_mc' + ext
        else:
            out = stem + '_clean' + ext

    print(f"Reading  {os.path.basename(inp)}  ({os.path.getsize(inp):,} B)")
    version, materials_raw, streams, surfaces, models, rest = parse_bgm(inp)
    print(f"  version=0x{version:X}  {len(materials_raw)} mat"
          f"  {len(streams)} streams  {len(surfaces)} surfaces"
          f"  {len(models)} models")

    # crash.dat
    crash_src, crash_suffix = _find_crash_dat(inp)
    opt_crash_info = None

    # -clean (always first)
    if do_clean:
        streams, surfaces = op_strip(streams, surfaces)

    # -menuorder (after clean, before optimize)
    if do_menucar:
        surfaces, models = op_menucar(surfaces, models, materials_raw, version)

    # -optimize (always last)
    if do_optimize:
        (streams, surfaces,
         surf_seen, surf_vs, surf_original_vc,
         streams_orig, surfaces_orig) = op_optimize(streams, surfaces)
        opt_crash_info = (surf_seen, surf_vs, surf_original_vc,
                          streams_orig, surfaces_orig)

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
        _remap_crash_dat(crash_src, out, models, *opt_crash_info)
    else:
        # strip only: vertex data unchanged, plain copy
        _copy_crash_dat(inp, out)

    print("\nDone!")


if __name__ == '__main__':
    main()
