#!/usr/bin/env python3
"""
DDS Resizer
Downscales a DDS texture by a power-of-two factor and recompresses (preserving
source format by default). Requires dds2tga.py and tga2dds.py in the same directory.

https://github.com/RavenDS/flatout-blender-tools

Usage:
  python dds_resize.py -2x  input.dds [output.dds]
  python dds_resize.py -4x  input.dds [output.dds]
  python dds_resize.py -8x  *.dds
  python dds_resize.py -16x folder/
  python dds_resize.py -xbox input.dds          # DXT3, ÷2
  python dds_resize.py -ps2  input.dds          # DXT3, ÷2
  python dds_resize.py -xbox -loskin input.dds  # DXT3, ÷4
  python dds_resize.py -2x   -loskin input.dds  # source fmt, ÷4
  python dds_resize.py -2x   -tga    input.dds  # output resized TGA
"""

import struct
import sys
import os
import glob

# helpers from sibling scripts

def _import_sibling(name):
    """Import dds2tga / tga2dds from the same directory as this script."""
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

# re-export the pieces we actually use
read_dds        = dds2tga.read_dds
decode_dxt5_block  = dds2tga.decode_dxt5_block
decode_dxt3_block  = dds2tga.decode_dxt3_block
decode_dxt1_block  = dds2tga.decode_dxt1_block
decompress_dxt  = dds2tga.decompress_dxt
decode_uncompressed = dds2tga.decode_uncompressed
DDPF_FOURCC     = dds2tga.DDPF_FOURCC
DDPF_RGB        = dds2tga.DDPF_RGB
DXT1 = dds2tga.DXT1
DXT3 = dds2tga.DXT3
DXT5 = dds2tga.DXT5

compress_to_dxt = tga2dds.compress_to_dxt
write_dds       = tga2dds.write_dds
write_tga       = dds2tga.write_tga


# detect the DXT format name of a parsed DDS dict (for preserving it by default)

def detect_fmt(dds):
    """Return 'DXT1', 'DXT3', 'DXT5', or None for uncompressed."""
    if dds['pf_flags'] & DDPF_FOURCC:
        fc = dds['fourcc']
        if fc == DXT1: return 'DXT1'
        if fc == DXT3: return 'DXT3'
        if fc == DXT5: return 'DXT5'
    return None  # uncompressed — caller decides fallback




def decode_dds(dds):
    """Return (width, height, [(r,g,b,a), ...]) from a parsed DDS dict."""
    pf_flags = dds['pf_flags']
    fourcc   = dds['fourcc']
    width    = dds['width']
    height   = dds['height']

    if pf_flags & DDPF_FOURCC:
        fmt = fourcc.decode('ascii', errors='replace')
        if fourcc == DXT1:
            pixels = decompress_dxt(dds, lambda b: decode_dxt1_block(b, has_alpha=True), 8)
        elif fourcc == DXT3:
            pixels = decompress_dxt(dds, decode_dxt3_block, 16)
        elif fourcc == DXT5:
            pixels = decompress_dxt(dds, decode_dxt5_block, 16)
        else:
            raise ValueError(f"Unsupported compressed format: {fmt}")
    elif pf_flags & DDPF_RGB:
        pixels = decode_uncompressed(dds)
    else:
        raise ValueError(f"Unsupported DDS pixel format flags: 0x{pf_flags:08X}")

    return width, height, pixels


# box-filter downsampler

def downsample(pixels, src_w, src_h, factor):
    """
    Average factor×factor pixel blocks into one output pixel (box filter).
    Works for any integer factor; handles non-multiple dimensions by
    clamping sample coordinates to the image edge.
    """
    dst_w = max(1, src_w // factor)
    dst_h = max(1, src_h // factor)
    out   = []

    for dy in range(dst_h):
        for dx in range(dst_w):
            r_acc = g_acc = b_acc = a_acc = 0
            count = 0
            for ky in range(factor):
                for kx in range(factor):
                    sx = min(dx * factor + kx, src_w - 1)
                    sy = min(dy * factor + ky, src_h - 1)
                    r, g, b, a = pixels[sy * src_w + sx]
                    r_acc += r; g_acc += g; b_acc += b; a_acc += a
                    count += 1
            out.append((r_acc // count, g_acc // count,
                         b_acc // count, a_acc // count))

    return dst_w, dst_h, out


# top-level conversion

def resize_dds(src_path, dst_path, factor, out_fmt=None, tga_out=False):
    print(f"  Reading:     {src_path}")
    dds = read_dds(src_path)
    src_w, src_h, pixels = decode_dds(dds)

    src_fmt_name = (dds['fourcc'].decode('ascii', errors='replace')
                    if dds['pf_flags'] & DDPF_FOURCC
                    else f"Uncompressed {dds['rgb_bitcount']}-bit")
    print(f"  Source:      {src_w}x{src_h}  [{src_fmt_name}]")

    # resolve output format: explicit override > source format > DXT3 fallback
    resolved_fmt = out_fmt or detect_fmt(dds) or 'DXT3'

    dst_w, dst_h, scaled = downsample(pixels, src_w, src_h, factor)
    print(f"  Downscale:   ÷{factor}  →  {dst_w}x{dst_h}")

    if tga_out:
        print(f"  Output:      TGA (32-bit BGRA)")
        write_tga(dst_path, dst_w, dst_h, scaled)
    else:
        print(f"  Compressing: {resolved_fmt}")
        compressed = compress_to_dxt(dst_w, dst_h, scaled, resolved_fmt)
        write_dds(dst_path, dst_w, dst_h, compressed, resolved_fmt)

    return dst_path


# CLI

VALID_FACTORS = {2, 4, 8, 16, 32}

def make_output_path(src_path, factor, tga_out=False):
    base, _ = os.path.splitext(src_path)
    ext = '.tga' if tga_out else '.dds'
    return f"{base}_{factor}x{ext}"


def main():
    base_factor  = None   # set by -Nx / -xbox / -ps2
    out_fmt      = None   # None = preserve source format
    loskin       = False  # doubles the final factor
    tga_out      = False  # output TGA instead of DDS
    raw_args     = []

    for arg in sys.argv[1:]:
        low = arg.lower()
        if low.endswith('x') and low[1:-1].isdigit() and low.startswith('-'):
            base_factor = int(low[1:-1])
        elif low.startswith('-dxt') and low[4:].isdigit():
            out_fmt = low[1:].upper()
        elif low == '-xbox':
            base_factor = 2
            out_fmt = 'DXT3'
        elif low == '-ps2':
            base_factor = 2
            out_fmt = 'DXT3'
        elif low == '-loskin':
            loskin = True
        elif low == '-tga':
            tga_out = True
        else:
            raw_args.append(arg)

    # -loskin doubles the division ratio regardless of how it was set
    factor = base_factor * 2 if loskin and base_factor else base_factor

    if factor is None or not raw_args:
        print("Source: https://github.com/RavenDS/flatout-blender-tools")
        print()
        print("Usage: dds_resize.py <scale> [format] [options] <input.dds|*.dds|folder/> [output.dds]")
        print()
        print("Scale (pick one):")
        print("  -Nx        Explicit factor: -2x (half), -4x (quarter), -8x, -16x, -32x")
        print("  -xbox      Shortcut: DXT3 + -2x  (Xbox port quality)")
        print("  -ps2       Shortcut: DXT3 + -2x  (PS2 port quality)")
        print()
        print("Format (optional, default = same as source):")
        print("  -dxt1      No/1-bit alpha  (smallest)")
        print("  -dxt3      Explicit 4-bit alpha")
        print("  -dxt5      Interpolated alpha  (best quality)")
        print()
        print("Options:")
        print("  -loskin    Double the division ratio (2x→4x, 4x→8x, …)")
        print("  -tga       Output resized TGA instead of DDS")
        print()
        print("Examples:")
        print("  python dds_resize.py -2x texture.dds")
        print("  python dds_resize.py -4x texture.dds small.dds")
        print("  python dds_resize.py -2x -dxt5 *.dds")
        print("  python dds_resize.py -xbox input.dds           # DXT3, ÷2")
        print("  python dds_resize.py -xbox -loskin input.dds   # DXT3, ÷4")
        print("  python dds_resize.py -ps2  -loskin input.dds   # DXT3, ÷4")
        print("  python dds_resize.py -2x   -loskin input.dds   # source fmt, ÷4")
        print("  python dds_resize.py -2x   -tga    input.dds   # resized TGA output")
        sys.exit(1)

    if factor not in VALID_FACTORS:
        print(f"ERROR: -{factor}x is not a supported factor. Choose from: "
              + ', '.join(f'-{f}x' for f in sorted(VALID_FACTORS)))
        sys.exit(1)

    if out_fmt is not None and out_fmt not in ('DXT1', 'DXT3', 'DXT5'):
        print(f"ERROR: unsupported output format '{out_fmt}'. Use -dxt1, -dxt3 or -dxt5.")
        sys.exit(1)

    # resolve inputs
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

        # "input.dds output.dds/tga" to explicit output path (single file only)
        out_exts = ('.dds', '.tga')
        if len(raw_args) == 2 and raw_args[1].lower().endswith(out_exts):
            inputs = [raw_args[0]]
            output = raw_args[1]

    converted = 0
    for src in inputs:
        if not src.lower().endswith('.dds'):
            continue
        dst = output or make_output_path(src, factor, tga_out)
        try:
            resize_dds(src, dst, factor, out_fmt, tga_out)
            converted += 1
        except Exception as e:
            print(f"  ERROR: {src}: {e}")

    print(f"\nDone. Converted {converted} file(s).")


if __name__ == '__main__':
    main()
