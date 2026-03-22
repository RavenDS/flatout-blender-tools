#!/usr/bin/env python3
"""
DDS Normal Map Converter for FlatOut Ultimate Carnage
Converts between standard normal maps (X=R, Y=G) and FOUC's DXT5nm format (X=A, Y=G, R=0, B=0).
Always outputs DXT5.

https://github.com/RavenDS/flatout-blender-tools

Usage:
  python dds_normal.py -to_fouc   input.dds
  python dds_normal.py -from_fouc input.dds
  python dds_normal.py -to_fouc   *.dds
  python dds_normal.py -from_fouc folder/
  python dds_normal.py -to_fouc   input.dds output.dds
"""

import sys
import os
import glob


def _import_sibling(name):
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, name + '.py')
    if not os.path.isfile(path):
        raise ImportError(
            f"Could not find {name}.py next to {__file__}\n"
            "Make sure dds2tga.py and tga2dds.py are in the same directory."
        )
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

dds2tga = _import_sibling('dds2tga')
tga2dds = _import_sibling('tga2dds')

read_dds            = dds2tga.read_dds
decode_dxt1_block   = dds2tga.decode_dxt1_block
decode_dxt3_block   = dds2tga.decode_dxt3_block
decode_dxt5_block   = dds2tga.decode_dxt5_block
decompress_dxt      = dds2tga.decompress_dxt
decode_uncompressed = dds2tga.decode_uncompressed
DDPF_FOURCC         = dds2tga.DDPF_FOURCC
DDPF_RGB            = dds2tga.DDPF_RGB
DXT1                = dds2tga.DXT1
DXT3                = dds2tga.DXT3
DXT5                = dds2tga.DXT5

compress_to_dxt = tga2dds.compress_to_dxt
write_dds       = tga2dds.write_dds


def decode_dds(dds):
    pf_flags = dds['pf_flags']
    fourcc   = dds['fourcc']

    if pf_flags & DDPF_FOURCC:
        if fourcc == DXT1:
            return decompress_dxt(dds, lambda b: decode_dxt1_block(b, has_alpha=True), 8)
        elif fourcc == DXT3:
            return decompress_dxt(dds, decode_dxt3_block, 16)
        elif fourcc == DXT5:
            return decompress_dxt(dds, decode_dxt5_block, 16)
        else:
            raise ValueError(f"Unsupported compressed format: {fourcc.decode('ascii', errors='replace')}")
    elif pf_flags & DDPF_RGB:
        return decode_uncompressed(dds)
    else:
        raise ValueError(f"Unsupported DDS pixel format flags: 0x{pf_flags:08X}")


def remap_to_fouc(pixels):
    """
    Standard → FOUC DXT5nm:  (R,G,B,A) → (0, G, 0, R)
    X normal moves from R to A, Y stays in G, R and B are zeroed.
    """
    return [(0, g, 0, r) for r, g, b, a in pixels]


def remap_from_fouc(pixels):
    """
    FOUC DXT5nm → Standard:  (0, G, 0, X_in_alpha) → (X, G, reconstructed_B, 255)
    X is recovered from A back to R, Y stays in G.
    B is reconstructed via sqrt(1 - X² - Y²) from the unit normal constraint.
    Note: DXT5 compression artifacts may push X²+Y² slightly above 1, clamped to 0.
    """
    import math
    out = []
    for r, g, b, a in pixels:
        nx = (a / 127.5) - 1.0
        ny = (g / 127.5) - 1.0
        nz = math.sqrt(max(0.0, 1.0 - nx * nx - ny * ny))
        out.append((a, g, min(255, int((nz + 1.0) * 127.5)), 255))
    return out


def convert_normalmap(src_path, dst_path, to_fouc):
    print(f"  Reading:     {src_path}")
    dds = read_dds(src_path)
    width, height = dds['width'], dds['height']

    src_fmt_name = (dds['fourcc'].decode('ascii', errors='replace')
                    if dds['pf_flags'] & DDPF_FOURCC
                    else f"Uncompressed {dds['rgb_bitcount']}-bit")
    print(f"  Source:      {width}x{height}  [{src_fmt_name}]")

    pixels = decode_dds(dds)

    if to_fouc:
        remapped = remap_to_fouc(pixels)
        print(f"  Remapping:   (R,G,B,A) to (0,G,0,R)  [standard → FOUC DXT5nm]")
    else:
        remapped = remap_from_fouc(pixels)
        print(f"  Remapping:   (0,G,0,A) to (A,G,0,255)  [FOUC DXT5nm → standard]")

    print(f"  Compressing: DXT5")
    compressed = compress_to_dxt(width, height, remapped, 'DXT5')
    write_dds(dst_path, width, height, compressed, 'DXT5')


def main():
    to_fouc  = None
    raw_args = []

    for arg in sys.argv[1:]:
        low = arg.lower()
        if low == '-to_fouc':
            to_fouc = True
        elif low == '-from_fouc':
            to_fouc = False
        else:
            raw_args.append(arg)

    if to_fouc is None or not raw_args:
        print("Source: https://github.com/RavenDS/flatout-blender-tools")
        print()
        print("Usage: dds_normal.py -to_fouc|-from_fouc <input.dds|*.dds|folder/> [output.dds]")
        print()
        print("Always outputs DXT5.")
        print()
        print("  -to_fouc    Standard (X=R, Y=G) → FOUC DXT5nm (X=A, Y=G, R=0, B=0)")
        print("  -from_fouc  FOUC DXT5nm (X=A, Y=G) → Standard (X=R, Y=G, B=reconstructed, A=255)")
        print()
        print("Note: -from_fouc reconstructs B via sqrt(1 - X² - Y²). Not bit-perfect due to DXT5 artifacts.")
        print()
        print("Examples:")
        print("  python dds_normal.py -to_fouc   normal.dds")
        print("  python dds_normal.py -from_fouc normal.dds out.dds")
        print("  python dds_normal.py -to_fouc   *.dds")
        print("  python dds_normal.py -from_fouc normalmaps/")
        sys.exit(1)

    inputs = []
    output = None

    if len(raw_args) == 1 and os.path.isdir(raw_args[0]):
        inputs = glob.glob(os.path.join(raw_args[0], '*.dds'))
        if not inputs:
            print(f"No .dds files found in: {raw_args[0]}")
            sys.exit(1)
    else:
        for arg in raw_args:
            expanded = glob.glob(arg)
            inputs.extend(expanded if expanded else [arg])

        if len(raw_args) == 2 and raw_args[1].lower().endswith('.dds'):
            inputs = [raw_args[0]]
            output = raw_args[1]

    converted = 0
    for src in inputs:
        if not src.lower().endswith('.dds'):
            continue
        dst = output or src  # overwrite in-place if no explicit output
        try:
            convert_normalmap(src, dst, to_fouc)
            converted += 1
        except Exception as e:
            print(f"  ERROR: {src}: {e}")

    print(f"\nDone. Converted {converted} file(s).")


if __name__ == '__main__':
    main()
