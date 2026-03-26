#!/usr/bin/env python3
"""
png2dds: PNG Color (+ optional Alpha) -> 32-bit TGA -> DDS
--------------------------------------------------
https://github.com/RavenDS/flatout-blender-tools

If <n>_alpha.png exists alongside <n>.png it is merged in as the alpha channel.
If no alpha file is found the PNG's own alpha channel is used as-is (RGB PNGs without alpha are treated as fully opaque).

Conversion back-end (in priority order):
  1. nvdxt.exe (same folder as this script)
  2. tga2dds.py (pure-Python fallback, mipmaps enabled)

PNG reading uses only Python stdlib (zlib + struct) - no external libraries.
TGA writing uses write_tga() from dds2tga.py.

Usage:
  python png_merge_to_dds.py [options] <folder>
  python png_merge_to_dds.py [options] <n>.png

General options:
  -rd                   Recurse into sub-folders (folder mode only)
  -skiptga              Delete intermediate TGA files after conversion
  -dxt1 | -dxt3 | -dxt5 | -bc4 | -bc5
                        DDS compression format (default: -dxt5)

nvdxt-only options (ignored when falling back to tga2dds):
  -quick                Fast compression method (replaces -quality_highest)
  -quality_normal       Normal quality compression (replaces -quality_highest)
  -quality_production   Production quality compression (replaces -quality_highest)
  -sharpenMethod <m>    Sharpen method for MIP maps
  -nmips <n>            Number of MIP maps to generate

"""

import os
import sys
import glob
import struct
import zlib
import subprocess

# ---------------------------------------------------------------------------
# Bootstrap: make sure sibling scripts are importable
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from dds2tga import write_tga               # noqa: E402  (after path fix)
from tga2dds import convert_tga_to_dds      # noqa: E402


# ---------------------------------------------------------------------------
# Valid formats
# ---------------------------------------------------------------------------

# Maps CLI flag -> tga2dds format string
_FORMAT_MAP = {
    '-dxt1': 'DXT1',
    '-dxt3': 'DXT3',
    '-dxt5': 'DXT5',
    '-bc4':  'ATI1',
    '-bc5':  'ATI2',
}

# nvdxt uses slightly different flag names for bc4/bc5
_NVDXT_FORMAT_FLAG = {
    'DXT1': '-dxt1',
    'DXT3': '-dxt3',
    'DXT5': '-dxt5',
    'ATI1': '-bc4',
    'ATI2': '-bc5',
}

# nvdxt quality flags (mutually exclusive; exactly one goes on the command line)
_NVDXT_QUALITY_FLAGS = {'-quick', '-quality_normal', '-quality_production', '-quality_highest'}

# nvdxt-only flags that consume a following value token
_NVDXT_VALUE_FLAGS = {'-sharpenMethod', '-nmips'}


# ---------------------------------------------------------------------------
# Pure-stdlib PNG reader
# ---------------------------------------------------------------------------

def _paeth(a, b, c):
    """Paeth predictor as defined in the PNG spec."""
    p  = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def read_png(filepath):
    """
    Read a PNG file using only stdlib (zlib + struct).
    Returns (width, height, pixels) where pixels is a list of (R, G, B, A) tuples.

    Supported:
      - colour types 0 (grayscale), 2 (RGB), 3 (indexed/palette),
        4 (grayscale+alpha), 6 (RGBA)
      - bit depth 8 only
      - non-interlaced only
    """
    with open(filepath, 'rb') as f:
        raw = f.read()

    if raw[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError(f"Not a valid PNG file: {filepath}")

    # ---- chunk parsing ----
    pos        = 8
    ihdr       = None
    plte       = None
    trns       = None
    idat_parts = []

    while pos + 12 <= len(raw):
        length     = struct.unpack_from('>I', raw, pos)[0]
        chunk_type = raw[pos + 4 : pos + 8]
        chunk_data = raw[pos + 8 : pos + 8 + length]
        pos       += 12 + length

        if   chunk_type == b'IHDR': ihdr = chunk_data
        elif chunk_type == b'PLTE': plte = chunk_data
        elif chunk_type == b'tRNS': trns = chunk_data
        elif chunk_type == b'IDAT': idat_parts.append(chunk_data)
        elif chunk_type == b'IEND': break

    if ihdr is None:
        raise ValueError(f"Missing IHDR chunk: {filepath}")

    width      = struct.unpack_from('>I', ihdr, 0)[0]
    height     = struct.unpack_from('>I', ihdr, 4)[0]
    bit_depth  = ihdr[8]
    color_type = ihdr[9]
    interlace  = ihdr[12]

    if interlace != 0:
        raise ValueError(f"Interlaced PNG not supported: {filepath}")
    if bit_depth != 8:
        raise ValueError(f"Only 8-bit-per-channel PNG supported (got {bit_depth}): {filepath}")

    # ---- decompress ----
    scanline_data = zlib.decompress(b''.join(idat_parts))

    _ch_count = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    if color_type not in _ch_count:
        raise ValueError(f"Unsupported PNG colour type {color_type}: {filepath}")
    channels = _ch_count[color_type]
    stride   = width * channels

    # ---- reconstruct scanlines (undo PNG filters) ----
    rows   = []
    prev   = bytearray(stride)
    offset = 0

    for _ in range(height):
        filter_type = scanline_data[offset];  offset += 1
        row         = bytearray(scanline_data[offset : offset + stride])
        offset     += stride

        if filter_type == 0:    # None
            pass
        elif filter_type == 1:  # Sub
            for x in range(channels, stride):
                row[x] = (row[x] + row[x - channels]) & 0xFF
        elif filter_type == 2:  # Up
            for x in range(stride):
                row[x] = (row[x] + prev[x]) & 0xFF
        elif filter_type == 3:  # Average
            for x in range(stride):
                a = row[x - channels] if x >= channels else 0
                row[x] = (row[x] + ((a + prev[x]) >> 1)) & 0xFF
        elif filter_type == 4:  # Paeth
            for x in range(stride):
                a = row[x - channels] if x >= channels else 0
                b = prev[x]
                c = prev[x - channels] if x >= channels else 0
                row[x] = (row[x] + _paeth(a, b, c)) & 0xFF
        else:
            raise ValueError(f"Unknown PNG filter type {filter_type}")

        rows.append(bytes(row))
        prev = row

    # ---- convert rows to (R, G, B, A) tuples ----
    pixels = []
    for row in rows:
        for x in range(width):
            if color_type == 0:     # grayscale
                v = row[x]
                if trns is not None and len(trns) >= 2:
                    key = struct.unpack_from('>H', trns, 0)[0]
                    a = 0 if v == key else 255
                else:
                    a = 255
                pixels.append((v, v, v, a))

            elif color_type == 2:   # RGB
                r, g, b = row[x*3], row[x*3+1], row[x*3+2]
                pixels.append((r, g, b, 255))

            elif color_type == 3:   # indexed / palette
                if plte is None:
                    raise ValueError("Indexed PNG has no PLTE chunk")
                idx = row[x]
                r = plte[idx * 3];  g = plte[idx * 3 + 1];  b = plte[idx * 3 + 2]
                a = trns[idx] if (trns is not None and idx < len(trns)) else 255
                pixels.append((r, g, b, a))

            elif color_type == 4:   # grayscale + alpha
                v = row[x*2];  a = row[x*2 + 1]
                pixels.append((v, v, v, a))

            elif color_type == 6:   # RGBA
                r, g, b, a = row[x*4], row[x*4+1], row[x*4+2], row[x*4+3]
                pixels.append((r, g, b, a))

    return width, height, pixels


# ---------------------------------------------------------------------------
# Core merge + convert logic
# ---------------------------------------------------------------------------

def _alpha_from_pixel(r, g, b, a):
    """
    Extract a single alpha value from an alpha-PNG pixel.
    Works whether the file was saved as true grayscale or RGB with equal channels.
    """
    return (int(r) + int(g) + int(b)) // 3


def build_tga(color_png, alpha_png, tga_path):
    """
    Read color_png (and optionally alpha_png) and write a 32-bit TGA at tga_path.
    If alpha_png is None every pixel gets alpha=255.
    Returns (width, height).
    """
    cw, ch, color_px = read_png(color_png)

    if alpha_png is not None:
        aw, ah, alpha_px = read_png(alpha_png)
        if cw != aw or ch != ah:
            raise ValueError(
                f"Size mismatch: '{os.path.basename(color_png)}' ({cw}x{ch}) "
                f"vs '{os.path.basename(alpha_png)}' ({aw}x{ah})"
            )
        merged = [
            (r, g, b, _alpha_from_pixel(ar, ag, ab, aa))
            for (r, g, b, _), (ar, ag, ab, aa) in zip(color_px, alpha_px)
        ]
    else:
        merged = color_px  # use alpha already in the PNG (255 for opaque types)

    write_tga(tga_path, cw, ch, merged)
    return cw, ch


def convert_tga(tga_path, fmt, nvdxt_extra):
    """
    Convert tga_path to a DDS using nvdxt.exe if present, else tga2dds.

    fmt         : tga2dds format string, e.g. 'DXT5'
    nvdxt_extra : extra flags for nvdxt only (quality, -sharpenMethod, -nmips, ...)
    """
    nvdxt = os.path.join(_SCRIPT_DIR, 'nvdxt.exe')

    if os.path.isfile(nvdxt):
        fmt_flag = _NVDXT_FORMAT_FLAG[fmt]

        # Separate quality flag from the rest of the nvdxt_extra tokens
        quality_flag = '-quality_highest'   # default
        other_extra  = []
        for token in nvdxt_extra:
            if token in _NVDXT_QUALITY_FLAGS:
                quality_flag = token
            else:
                other_extra.append(token)

        cmd = [nvdxt, quality_flag, fmt_flag] + other_extra + ['-file', tga_path]
        print(f"  nvdxt: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            stderr = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"nvdxt failed (code {result.returncode}): {stderr}")
        expected_dds = os.path.splitext(tga_path)[0] + '.dds'
        if not os.path.isfile(expected_dds):
            raise RuntimeError(f"nvdxt did not produce expected output: {expected_dds}")
        print(f"  Saved: {expected_dds}")

    else:
        dds_path = os.path.splitext(tga_path)[0] + '.dds'
        convert_tga_to_dds(tga_path, dds_path, fmt, mipmaps=True)


def process_png(color_png, alpha_png, fmt, nvdxt_extra, skip_tga):
    """
    Full pipeline for one PNG:  build TGA -> convert to DDS -> optionally remove TGA.
    alpha_png may be None (standalone PNG, alpha=255).
    """
    base     = os.path.splitext(color_png)[0]
    tga_path = base + '.tga'

    print(f"\n[{os.path.basename(color_png)}]")
    print(f"  Color : {color_png}")
    if alpha_png:
        print(f"  Alpha : {alpha_png}")
    else:
        print(f"  Alpha : (none - using PNG's own alpha channel)")

    try:
        build_tga(color_png, alpha_png, tga_path)
        convert_tga(tga_path, fmt, nvdxt_extra)
        if skip_tga and os.path.isfile(tga_path):
            os.remove(tga_path)
            print(f"  Removed: {os.path.basename(tga_path)}")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        if skip_tga and os.path.isfile(tga_path):
            try:
                os.remove(tga_path)
            except OSError:
                pass
        return False


# ---------------------------------------------------------------------------
# Folder scanning
# ---------------------------------------------------------------------------

def find_pngs(folder):
    """
    Yield (color_png, alpha_png_or_None) for every non-alpha PNG in folder.
    Alpha PNGs (*_alpha.png) are paired with their base and never yielded standalone.
    """
    all_pngs   = set(glob.glob(os.path.join(folder, '*.png')))
    alpha_set  = {p for p in all_pngs if os.path.basename(p).lower().endswith('_alpha.png')}
    color_pngs = sorted(all_pngs - alpha_set)

    for png in color_pngs:
        base      = os.path.splitext(png)[0]
        alpha_png = base + '_alpha.png'
        yield png, (alpha_png if alpha_png in alpha_set else None)


def process_folder(folder, fmt, nvdxt_extra, skip_tga, recursive=False):
    """Process all PNGs in folder (and optionally its sub-folders)."""
    folders = []
    if recursive:
        for dirpath, dirnames, _ in os.walk(folder):
            dirnames.sort()
            folders.append(dirpath)
    else:
        folders = [folder]

    total_ok  = 0
    total_err = 0

    for f in folders:
        pngs = list(find_pngs(f))
        if not pngs:
            if not recursive:
                print(f"No PNG files found in: {f}")
            continue

        if recursive and f != folder:
            print(f"\n=== {f} ===")

        for color_png, alpha_png in pngs:
            if process_png(color_png, alpha_png, fmt, nvdxt_extra, skip_tga):
                total_ok  += 1
            else:
                total_err += 1

    return total_ok, total_err


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _usage():
    print("Source: https://github.com/RavenDS/flatout-blender-tools")
    print()
    print("Usage:")
    print("  python png2dds.py [options] <folder>")
    print("  python png2dds.py [options] <n>.png")
    print()
    print("General options:")
    print("  -rd                   Recurse into sub-folders (folder mode only)")
    print("  -skiptga              Delete intermediate TGA after conversion")
    print("  -dxt1 | -dxt3 | -dxt5 | -bc4 | -bc5")
    print("                        Output format (default: -dxt5)")
    print()
    print("nvdxt-only options (ignored when falling back to tga2dds):")
    print("  -quick                Fast compression (replaces -quality_highest)")
    print("  -quality_normal       Normal quality   (replaces -quality_highest)")
    print("  -quality_production   Production quality (replaces -quality_highest)")
    print("  -sharpenMethod <m>    Sharpen method for MIP maps")
    print("  -nmips <n>            Number of MIP maps to generate")
    print()
    print("Notes:")
    print("  If <n>_alpha.png is present it is used as the alpha channel.")
    print("  If not, the alpha channel already embedded in the PNG is used.")
    print("  (RGB PNGs with no alpha are treated as fully opaque.)")
    print("  Output: <n>.dds (and <n>.tga unless -skiptga) next to the source PNG.")


def main():
    raw_args = sys.argv[1:]

    if not raw_args:
        _usage()
        sys.exit(1)

    # ---- parse flags ----
    recursive   = False
    skip_tga    = False
    fmt         = 'DXT5'    # default format
    nvdxt_extra = []        # forwarded to nvdxt only
    positional  = []

    i = 0
    while i < len(raw_args):
        arg = raw_args[i]
        low = arg.lower()

        if low == '-rd':
            recursive = True
        elif low == '-skiptga':
            skip_tga = True
        elif low in _FORMAT_MAP:
            fmt = _FORMAT_MAP[low]
        elif low in _NVDXT_QUALITY_FLAGS:
            nvdxt_extra.append(low)
        elif low in _NVDXT_VALUE_FLAGS:
            if i + 1 >= len(raw_args):
                print(f"ERROR: {arg} requires a value argument.")
                sys.exit(1)
            nvdxt_extra.append(arg)
            i += 1
            nvdxt_extra.append(raw_args[i])
        else:
            positional.append(arg)

        i += 1

    if not positional:
        _usage()
        sys.exit(1)

    target = positional[0]

    # ------------------------------------------------------------------
    # Single explicit PNG
    # ------------------------------------------------------------------
    if target.lower().endswith('.png'):
        if not os.path.isfile(target):
            print(f"ERROR: File not found: {target}")
            sys.exit(1)

        base = os.path.splitext(os.path.abspath(target))[0]
        # Correct silently if user passed the alpha PNG by mistake
        if base.endswith('_alpha'):
            base = base[: -len('_alpha')]

        color_png = base + '.png'
        alpha_png = base + '_alpha.png'

        if not os.path.isfile(color_png):
            print(f"ERROR: Color PNG not found: {color_png}")
            sys.exit(1)

        alpha_arg = alpha_png if os.path.isfile(alpha_png) else None
        ok = process_png(color_png, alpha_arg, fmt, nvdxt_extra, skip_tga)
        print(f"\nDone. {'1 converted.' if ok else '0 converted (error).'}")
        sys.exit(0 if ok else 1)

    # ------------------------------------------------------------------
    # Folder
    # ------------------------------------------------------------------
    if os.path.isdir(target):
        label = f"'{target}'" + (" recursively" if recursive else "")
        print(f"Scanning {label} ...")

        ok, err = process_folder(target, fmt, nvdxt_extra, skip_tga, recursive)
        print(f"\nDone. {ok} converted, {err} error(s).")
        sys.exit(0 if err == 0 else 1)

    print(f"ERROR: '{target}' is neither a PNG file nor a directory.")
    sys.exit(1)


if __name__ == '__main__':
    main()
