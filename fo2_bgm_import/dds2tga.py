#!/usr/bin/env python3
"""
DDS to TGA Converter (32-bit with Alpha)
Converts DDS files (DXT1, DXT3, DXT5, uncompressed RGBA) to 32-bit TGA with alpha preserved.

https://github.com/RavenDS/flatout-blender-tools
"""

import struct
import sys
import os
import glob

# DDS constants

DDSD_CAPS        = 0x1
DDSD_HEIGHT      = 0x2
DDSD_WIDTH       = 0x4
DDSD_PIXELFORMAT = 0x1000

DDPF_ALPHAPIXELS = 0x1
DDPF_FOURCC      = 0x4
DDPF_RGB         = 0x40

DXT1 = b'DXT1'
DXT3 = b'DXT3'
DXT5 = b'DXT5'


# DDS parsing

def read_dds(filepath):
    with open(filepath, 'rb') as f:
        magic = f.read(4)
        if magic != b'DDS ':
            raise ValueError(f"Not a valid DDS file: {filepath}")

        header = f.read(124)
        size        = struct.unpack_from('<I', header, 0)[0]
        flags       = struct.unpack_from('<I', header, 4)[0]
        height      = struct.unpack_from('<I', header, 8)[0]
        width       = struct.unpack_from('<I', header, 12)[0]
        pitch       = struct.unpack_from('<I', header, 16)[0]
        depth       = struct.unpack_from('<I', header, 20)[0]
        mip_count   = struct.unpack_from('<I', header, 24)[0]

        # pixel format at offset 72
        pf_size     = struct.unpack_from('<I', header, 72)[0]
        pf_flags    = struct.unpack_from('<I', header, 76)[0]
        pf_fourcc   = header[80:84]
        pf_rgbbit   = struct.unpack_from('<I', header, 84)[0]
        pf_rmask    = struct.unpack_from('<I', header, 88)[0]
        pf_gmask    = struct.unpack_from('<I', header, 92)[0]
        pf_bmask    = struct.unpack_from('<I', header, 96)[0]
        pf_amask    = struct.unpack_from('<I', header, 100)[0]

        data = f.read()

    return {
        'width': width,
        'height': height,
        'pf_flags': pf_flags,
        'fourcc': pf_fourcc,
        'rgb_bitcount': pf_rgbbit,
        'r_mask': pf_rmask,
        'g_mask': pf_gmask,
        'b_mask': pf_bmask,
        'a_mask': pf_amask,
        'data': data,
    }


# DXT decompression (not from me, lost the original source)

def unpack_565(c):
    r = ((c >> 11) & 0x1F) * 255 // 31
    g = ((c >> 5)  & 0x3F) * 255 // 63
    b = (c & 0x1F) * 255 // 31
    return (r, g, b)


def decode_dxt1_block(block, has_alpha=False):
    c0 = struct.unpack_from('<H', block, 0)[0]
    c1 = struct.unpack_from('<H', block, 2)[0]
    bits = struct.unpack_from('<I', block, 4)[0]

    r0, g0, b0 = unpack_565(c0)
    r1, g1, b1 = unpack_565(c1)

    colors = [(r0, g0, b0, 255), (r1, g1, b1, 255)]

    if c0 > c1:
        colors.append(((2*r0 + r1) // 3, (2*g0 + g1) // 3, (2*b0 + b1) // 3, 255))
        colors.append(((r0 + 2*r1) // 3, (g0 + 2*g1) // 3, (b0 + 2*b1) // 3, 255))
    else:
        colors.append(((r0 + r1) // 2, (g0 + g1) // 2, (b0 + b1) // 2, 255))
        colors.append((0, 0, 0, 0) if has_alpha else (0, 0, 0, 255))

    pixels = []
    for i in range(16):
        idx = (bits >> (i * 2)) & 0x3
        pixels.append(colors[idx])
    return pixels


def decode_dxt3_block(block):
    # first 8 bytes: explicit alpha (4 bits per pixel)
    alpha_data = struct.unpack_from('<Q', block, 0)[0]
    alphas = []
    for i in range(16):
        a = (alpha_data >> (i * 4)) & 0xF
        alphas.append(a * 255 // 15)

    # next 8 bytes: DXT1 color block
    color_pixels = decode_dxt1_block(block[8:16])

    pixels = []
    for i in range(16):
        r, g, b, _ = color_pixels[i]
        pixels.append((r, g, b, alphas[i]))
    return pixels


def decode_dxt5_block(block):
    # first 8 bytes: interpolated alpha
    a0 = block[0]
    a1 = block[1]

    alpha_bits = 0
    for i in range(6):
        alpha_bits |= block[2 + i] << (8 * i)

    alphas_lut = [a0, a1]
    if a0 > a1:
        for i in range(1, 7):
            alphas_lut.append(((7 - i) * a0 + i * a1) // 7)
    else:
        for i in range(1, 5):
            alphas_lut.append(((5 - i) * a0 + i * a1) // 5)
        alphas_lut.append(0)
        alphas_lut.append(255)

    alphas = []
    for i in range(16):
        idx = (alpha_bits >> (i * 3)) & 0x7
        alphas.append(alphas_lut[idx])

    # next 8 bytes: DXT1 color block
    color_pixels = decode_dxt1_block(block[8:16])

    pixels = []
    for i in range(16):
        r, g, b, _ = color_pixels[i]
        pixels.append((r, g, b, alphas[i]))
    return pixels


def decompress_dxt(dds, decoder, block_size):
    width, height = dds['width'], dds['height']
    data = dds['data']
    pixels = [(0, 0, 0, 255)] * (width * height)

    bw = (width + 3) // 4
    bh = (height + 3) // 4

    offset = 0
    for by in range(bh):
        for bx in range(bw):
            block = data[offset:offset + block_size]
            offset += block_size
            if len(block) < block_size:
                break
            block_pixels = decoder(block)

            for py in range(4):
                for px in range(4):
                    x = bx * 4 + px
                    y = by * 4 + py
                    if x < width and y < height:
                        pixels[y * width + x] = block_pixels[py * 4 + px]

    return pixels


# uncompressed RGBA

def get_shift_and_size(mask):
    if mask == 0:
        return 0, 0
    shift = 0
    while (mask >> shift) & 1 == 0:
        shift += 1
    size = 0
    while (mask >> (shift + size)) & 1 == 1:
        size += 1
    return shift, size


def scale_channel(value, size):
    if size == 0:
        return 255
    max_val = (1 << size) - 1
    return (value * 255 + max_val // 2) // max_val


def decode_uncompressed(dds):
    width, height = dds['width'], dds['height']
    bpp = dds['rgb_bitcount']
    byte_pp = bpp // 8
    data = dds['data']

    r_shift, r_size = get_shift_and_size(dds['r_mask'])
    g_shift, g_size = get_shift_and_size(dds['g_mask'])
    b_shift, b_size = get_shift_and_size(dds['b_mask'])
    a_shift, a_size = get_shift_and_size(dds['a_mask'])

    has_alpha = dds['pf_flags'] & DDPF_ALPHAPIXELS and dds['a_mask'] != 0

    pixels = []
    for y in range(height):
        for x in range(width):
            offset = (y * width + x) * byte_pp
            if byte_pp <= 4:
                val = int.from_bytes(data[offset:offset + byte_pp], 'little')
            else:
                val = int.from_bytes(data[offset:offset + 4], 'little')

            r = scale_channel((val >> r_shift) & ((1 << r_size) - 1), r_size)
            g = scale_channel((val >> g_shift) & ((1 << g_size) - 1), g_size)
            b = scale_channel((val >> b_shift) & ((1 << b_size) - 1), b_size)
            if has_alpha:
                a = scale_channel((val >> a_shift) & ((1 << a_size) - 1), a_size)
            else:
                a = 255
            pixels.append((r, g, b, a))

    return pixels


# TGA writing

def write_tga(filepath, width, height, pixels):
    """Write 32-bit uncompressed TGA with alpha (BGRA order)."""
    with open(filepath, 'wb') as f:
        # TGA header (18 bytes)
        f.write(struct.pack('<B', 0))       # ID length
        f.write(struct.pack('<B', 0))       # color map type
        f.write(struct.pack('<B', 2))       # image type: uncompressed true-color
        f.write(b'\x00' * 5)                # color map spec
        f.write(struct.pack('<H', 0))       # X origin
        f.write(struct.pack('<H', 0))       # Y origin
        f.write(struct.pack('<H', width))   # width
        f.write(struct.pack('<H', height))  # height
        f.write(struct.pack('<B', 32))      # bpp
        f.write(struct.pack('<B', 0x28))    # image descriptor: top-left origin + 8 alpha bits

        # pixel data: BGRA order, top-to-bottom
        for r, g, b, a in pixels:
            f.write(struct.pack('BBBB', b, g, r, a))

    print(f"  Saved: {filepath} ({width}x{height}, 32-bit BGRA)")


# main conversion

def convert_dds_to_tga(dds_path, tga_path=None):
    if tga_path is None:
        tga_path = os.path.splitext(dds_path)[0] + '.tga'

    print(f"  Reading: {dds_path}")
    dds = read_dds(dds_path)
    width, height = dds['width'], dds['height']

    pf_flags = dds['pf_flags']
    fourcc = dds['fourcc']

    if pf_flags & DDPF_FOURCC:
        fmt_name = fourcc.decode('ascii', errors='replace')
        print(f"  Format: {fmt_name} ({width}x{height})")

        if fourcc == DXT1:
            pixels = decompress_dxt(dds, lambda b: decode_dxt1_block(b, has_alpha=True), 8)
        elif fourcc == DXT3:
            pixels = decompress_dxt(dds, decode_dxt3_block, 16)
        elif fourcc == DXT5:
            pixels = decompress_dxt(dds, decode_dxt5_block, 16)
        else:
            raise ValueError(f"Unsupported compressed format: {fmt_name}")

    elif pf_flags & DDPF_RGB:
        bpp = dds['rgb_bitcount']
        has_alpha = bool(pf_flags & DDPF_ALPHAPIXELS)
        print(f"  Format: Uncompressed {bpp}-bit {'RGBA' if has_alpha else 'RGB'} ({width}x{height})")
        pixels = decode_uncompressed(dds)

    else:
        raise ValueError(f"Unsupported DDS pixel format flags: 0x{pf_flags:08X}")

    write_tga(tga_path, width, height, pixels)
    return tga_path


# CLI

def main():
    if len(sys.argv) < 2:
        print("Source: https://github.com/RavenDS/flatout-blender-tools")
        print()
        print("Usage: dds_to_tga.py <input.dds or *.dds> [output.tga]")
        print()
        print("Examples:")
        print("  python dds_to_tga.py texture.dds")
        print("  python dds_to_tga.py texture.dds output.tga")
        print("  python dds_to_tga.py *.dds")
        print("  python dds_to_tga.py folder/")

        sys.exit(1)

    inputs = []
    output = None

    # check if first arg is a directory
    if os.path.isdir(sys.argv[1]):
        inputs = glob.glob(os.path.join(sys.argv[1], '*.dds'))
        if not inputs:
            print(f"No .dds files found in {sys.argv[1]}")
            sys.exit(1)
    else:
        # expand globs
        for arg in sys.argv[1:]:
            expanded = glob.glob(arg)
            if expanded:
                inputs.extend(expanded)
            else:
                inputs.append(arg)

        # if exactly 2 args and second ends with .tga, treat as output path
        if len(sys.argv) == 3 and sys.argv[2].lower().endswith('.tga'):
            inputs = [sys.argv[1]]
            output = sys.argv[2]

    converted = 0
    for dds_path in inputs:
        if not dds_path.lower().endswith('.dds'):
            continue
        try:
            convert_dds_to_tga(dds_path, output)
            converted += 1
        except Exception as e:
            print(f"  ERROR converting {dds_path}: {e}")

    print(f"\nDone. Converted {converted} file(s).")


if __name__ == '__main__':
    main()
