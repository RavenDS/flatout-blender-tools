#!/usr/bin/env python3
"""
TGA to DDS Converter (DXT1, DXT3, DXT5)
Converts 32-bit TGA with alpha to compressed DDS.
Use -dxt1, -dxt3, -dxt5, -bc5, -bc4 to select compression format.

https://github.com/RavenDS/flatout-blender-tools
"""

import struct
import sys
import os
import glob


# TGA reading

def decode_tga_rle(data, pixel_count, bytes_per_pixel):
    """Decode RLE-compressed TGA pixel data."""
    out = bytearray()
    i = 0
    while len(out) // bytes_per_pixel < pixel_count:
        if i >= len(data):
            break
        hdr = data[i]; i += 1
        count = (hdr & 0x7F) + 1
        if hdr & 0x80:                              # RLE packet
            px = data[i:i + bytes_per_pixel]; i += bytes_per_pixel
            out += px * count
        else:                                       # raw packet
            out += data[i:i + count * bytes_per_pixel]
            i   += count * bytes_per_pixel
    return bytes(out)


def read_tga(filepath):
    with open(filepath, 'rb') as f:
        raw = f.read()

    id_length  = raw[0]
    image_type = raw[2]
    width      = struct.unpack_from('<H', raw, 12)[0]
    height     = struct.unpack_from('<H', raw, 14)[0]
    bpp        = raw[16]
    image_desc = raw[17]

    if image_type not in (2, 10):
        raise ValueError(
            f"Unsupported TGA image type {image_type} "
            "(only uncompressed=2 or RLE=10 true-color supported)"
        )
    if bpp != 32:
        raise ValueError(f"Expected 32-bit TGA, got {bpp}-bit")

    pixel_data = raw[18 + id_length:]
    if image_type == 10:
        pixel_data = decode_tga_rle(pixel_data, width * height, 4)

    top_left = bool((image_desc >> 5) & 1)

    # BGRA to RGBA
    pixels = []
    for i in range(width * height):
        b = pixel_data[i * 4 + 0]
        g = pixel_data[i * 4 + 1]
        r = pixel_data[i * 4 + 2]
        a = pixel_data[i * 4 + 3]
        pixels.append((r, g, b, a))

    if not top_left:
        rows = [pixels[y * width:(y + 1) * width] for y in range(height)]
        rows.reverse()
        pixels = [p for row in rows for p in row]

    return width, height, pixels


# helpers

def to_rgb565(r, g, b):
    r5 = min(31, (r * 31 + 127) // 255)
    g6 = min(63, (g * 63 + 127) // 255)
    b5 = min(31, (b * 31 + 127) // 255)
    return (r5 << 11) | (g6 << 5) | b5


def from_rgb565(c):
    r = ((c >> 11) & 0x1F) * 255 // 31
    g = ((c >> 5)  & 0x3F) * 255 // 63
    b = (c & 0x1F) * 255 // 31
    return r, g, b


def sq_dist(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


# DXT color block encoder (shared by all three formats)

def encode_color_block(pixels, four_color_mode=True):
    """
    Encode 16 RGBA pixels into an 8-byte DXT color block.

    four_color_mode = True  = c0 > c1  (4 opaque colours, used by DXT3/DXT5 and opaque DXT1 blocks)
    four_color_mode = False = c0 <= c1 (3 colours + 1-bit transparent, used by DXT1 blocks with alpha)
    """
    rgb = [(r, g, b) for r, g, b, a in pixels]

    # axis-aligned bounding box for a quick but reasonable endpoint pair
    min_r = min(p[0] for p in rgb);  max_r = max(p[0] for p in rgb)
    min_g = min(p[1] for p in rgb);  max_g = max(p[1] for p in rgb)
    min_b = min(p[2] for p in rgb);  max_b = max(p[2] for p in rgb)

    c0_565 = to_rgb565(max_r, max_g, max_b)
    c1_565 = to_rgb565(min_r, min_g, min_b)

    # enforce the ordering required by the chosen mode
    if four_color_mode:
        if c0_565 < c1_565:
            c0_565, c1_565 = c1_565, c0_565
        elif c0_565 == c1_565 and c0_565 > 0:
            c1_565 -= 1                             # guarantee c0 > c1
    else:
        if c0_565 > c1_565:
            c0_565, c1_565 = c1_565, c0_565
        elif c0_565 == c1_565 and c1_565 < 0xFFFF:
            c1_565 += 1                             # guarantee c0 < c1

    # reconstruct palette from the quantised (rounded) endpoints
    c0r, c0g, c0b = from_rgb565(c0_565)
    c1r, c1g, c1b = from_rgb565(c1_565)

    if c0_565 > c1_565:                             # 4-colour mode
        palette = [
            (c0r, c0g, c0b),
            (c1r, c1g, c1b),
            ((2*c0r + c1r) // 3, (2*c0g + c1g) // 3, (2*c0b + c1b) // 3),
            ((c0r + 2*c1r) // 3, (c0g + 2*c1g) // 3, (c0b + 2*c1b) // 3),
        ]
        transparent_idx = None
    else:                                           # 3-colour + transparent
        palette = [
            (c0r, c0g, c0b),
            (c1r, c1g, c1b),
            ((c0r + c1r) // 2, (c0g + c1g) // 2, (c0b + c1b) // 2),
            None,
        ]
        transparent_idx = 3

    bits = 0
    for i, (r, g, b, a) in enumerate(pixels):
        if transparent_idx is not None and a < 128:
            idx = transparent_idx
        else:
            idx = min(
                (j for j in range(4) if palette[j] is not None),
                key=lambda j: sq_dist((r, g, b), palette[j])
            )
        bits |= idx << (i * 2)

    return struct.pack('<HHI', c0_565, c1_565, bits)


# block encoders

def encode_dxt1_block(pixels):
    """8 bytes. Uses 1-bit alpha (transparent) if any pixel has a (alpha) < 128."""
    has_transparent = any(a < 128 for _, _, _, a in pixels)
    return encode_color_block(pixels, four_color_mode=not has_transparent)


def encode_dxt3_block(pixels):
    """16 bytes. Explicit 4-bit alpha per pixel."""
    alpha_bits = 0
    for i, (_, _, _, a) in enumerate(pixels):
        a4 = min(15, (a * 15 + 127) // 255)
        alpha_bits |= a4 << (i * 4)
    color_block = encode_color_block(pixels, four_color_mode=True)
    return struct.pack('<Q', alpha_bits) + color_block


# DXT5 / ATI2 alpha-block encoder (shared)

def _encode_alpha_block(values: list) -> bytes:
    """
    Encode 16 single-channel values (0-255) into an 8-byte DXT5-style alpha block.
    Shared by encode_dxt5_block (alpha channel) and encode_ati2_block (R and G channels).
    """
    a0 = max(values)    # endpoint 0 – stored as byte 0
    a1 = min(values)    # endpoint 1 – stored as byte 1

    # Build 8-entry decoder LUT matching the endpoints we will write
    if a0 > a1:         # 8-value interpolation mode
        lut = [a0, a1] + [((7 - i) * a0 + i * a1) // 7 for i in range(1, 7)]
    else:               # 6-value + 0 + 255 mode
        lut = [a0, a1] + [((5 - i) * a0 + i * a1) // 5 for i in range(1, 5)] + [0, 255]

    indices = 0
    for i, v in enumerate(values):
        idx      = min(range(8), key=lambda j: abs(v - lut[j]))
        indices |= idx << (i * 3)

    return bytes([a0, a1]) + bytes((indices >> (8 * k)) & 0xFF for k in range(6))


# DXT5

def encode_dxt5_block(pixels):
    """16 bytes. Interpolated 8-bit alpha with 2 endpoints + 3-bit indices."""
    alpha_block = _encode_alpha_block([a for _, _, _, a in pixels])
    color_block = encode_color_block(pixels, four_color_mode=True)
    return alpha_block + color_block


# ATI2 / BC5

def encode_ati2_block(pixels):
    """
    16 bytes. ATI2/BC5 normal-map format.
    [8 bytes: Red/X channel] [8 bytes: Green/Y channel]
    Blue and Alpha from the input pixels are ignored; Z is reconstructed on decode.
    """
    r_block = _encode_alpha_block([r for r, g, b, a in pixels])
    g_block = _encode_alpha_block([g for r, g, b, a in pixels])
    return r_block + g_block


# ATI1 / BC4

def encode_ati1_block(pixels):
    """
    8 bytes. ATI1/BC4 single-channel format.
    Only the Red channel of the input pixels is stored.
    Green and Blue are ignored; they will be reconstructed as R=G=B on decode.
    """
    return _encode_alpha_block([r for r, g, b, a in pixels])


# main compression

def compress_to_dxt(width, height, pixels, fmt):
    encoder = {
        'DXT1': encode_dxt1_block,
        'DXT3': encode_dxt3_block,
        'DXT5': encode_dxt5_block,
        'ATI1': encode_ati1_block,
        'ATI2': encode_ati2_block,
    }[fmt]

    bw = (width  + 3) // 4
    bh = (height + 3) // 4
    output = bytearray()

    for by in range(bh):
        for bx in range(bw):
            block = []
            for py in range(4):
                for px in range(4):
                    x = min(bx * 4 + px, width  - 1)
                    y = min(by * 4 + py, height - 1)
                    block.append(pixels[y * width + x])
            output += encoder(block)

    return bytes(output)


# DDS writing

def write_dds(filepath, width, height, compressed_data, fmt):
    fourcc     = fmt.encode('ascii')
    block_size = 8 if fmt in ('DXT1', 'ATI1') else 16   # ATI1/DXT1=8, DXT3/DXT5/ATI2=16
    linear_size = (
        max(1, (width + 3) // 4) *
        max(1, (height + 3) // 4) *
        block_size
    )

    DDSD_CAPS        = 0x000001
    DDSD_HEIGHT      = 0x000002
    DDSD_WIDTH       = 0x000004
    DDSD_PIXELFORMAT = 0x001000
    DDSD_LINEARSIZE  = 0x080000
    DDSCAPS_TEXTURE  = 0x001000
    DDPF_FOURCC      = 0x000004

    flags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_LINEARSIZE

    with open(filepath, 'wb') as f:
        f.write(b'DDS ')

        # DDS_HEADER (124 bytes)
        f.write(struct.pack('<I', 124))           # dwSize
        f.write(struct.pack('<I', flags))         # dwFlags
        f.write(struct.pack('<I', height))        # dwHeight
        f.write(struct.pack('<I', width))         # dwWidth
        f.write(struct.pack('<I', linear_size))   # dwPitchOrLinearSize
        f.write(struct.pack('<I', 0))             # dwDepth
        f.write(struct.pack('<I', 1))             # dwMipMapCount
        f.write(b'\x00' * 44)                     # dwReserved1[11]

        # DDS_PIXELFORMAT (32 bytes)
        f.write(struct.pack('<I', 32))            # dwSize
        f.write(struct.pack('<I', DDPF_FOURCC))   # dwFlags
        f.write(fourcc)                           # dwFourCC
        f.write(struct.pack('<I', 0))             # dwRGBBitCount
        f.write(struct.pack('<I', 0))             # dwRBitMask
        f.write(struct.pack('<I', 0))             # dwGBitMask
        f.write(struct.pack('<I', 0))             # dwBBitMask
        f.write(struct.pack('<I', 0))             # dwABitMask

        # caps (5 × 4 bytes)
        f.write(struct.pack('<I', DDSCAPS_TEXTURE))
        f.write(struct.pack('<I', 0))             # dwCaps2
        f.write(struct.pack('<I', 0))             # dwCaps3
        f.write(struct.pack('<I', 0))             # dwCaps4
        f.write(struct.pack('<I', 0))             # dwReserved2

        f.write(compressed_data)

    print(f"  Saved: {filepath} ({width}x{height}, {fmt})")


# top-level conversion

def convert_tga_to_dds(tga_path, dds_path, fmt):
    print(f"  Reading: {tga_path}")
    width, height, pixels = read_tga(tga_path)
    print(f"  Compressing: {width}x{height} → {fmt}")
    data = compress_to_dxt(width, height, pixels, fmt)
    write_dds(dds_path, width, height, data, fmt)
    return dds_path


# CLI

def main():
    fmt  = None
    args = []

    for arg in sys.argv[1:]:
        low = arg.lower()
        if low in ('-dxt1', '-dxt3', '-dxt5', '-ati2', '-bc5', '-ati1', '-bc4'):
            if   low in ('-ati2', '-bc5'): fmt = 'ATI2'
            elif low in ('-ati1', '-bc4'): fmt = 'ATI1'
            else:                          fmt = low[1:].upper()
        else:
            args.append(arg)

    if not args or fmt is None:
        print("Source: https://github.com/RavenDS/flatout-blender-tools")
        print()
        print("Usage: tga2dds.py -dxt1|-dxt3|-dxt5|-ati1|-bc4|-ati2|-bc5 <input.tga or *.tga> [output.dds]")
        print()
        print("Format guide:")
        print("  -dxt1        No alpha / 1-bit alpha  (smallest file)")
        print("  -dxt3        Sharp / binary alpha     (explicit 4-bit alpha per pixel)")
        print("  -dxt5        Smooth alpha gradients   (interpolated alpha, best quality)")
        print("  -ati1/-bc4   Single-channel grayscale (BC4: R only, R=G=B on decode)")
        print("  -ati2/-bc5   Two-channel normal map   (BC5: R=X, G=Y, Z reconstructed on decode)")
        print()
        print("Examples:")
        print("  python tga2dds.py -dxt5 texture.tga")
        print("  python tga2dds.py -dxt1 texture.tga output.dds")
        print("  python tga2dds.py -dxt3 *.tga")
        print("  python tga2dds.py -dxt5 folder/")
        sys.exit(1)

    inputs = []
    output = None

    if len(args) == 1 and os.path.isdir(args[0]):
        inputs = glob.glob(os.path.join(args[0], '*.tga'))
        if not inputs:
            print(f"No .tga files found in: {args[0]}")
            sys.exit(1)
    else:
        for arg in args:
            expanded = glob.glob(arg)
            inputs.extend(expanded if expanded else [arg])

        if len(args) == 2 and args[1].lower().endswith('.dds'):
            inputs = [args[0]]
            output = args[1]

    converted = 0
    for tga_path in inputs:
        if not tga_path.lower().endswith('.tga'):
            continue
        dds_path = output or (os.path.splitext(tga_path)[0] + '.dds')
        try:
            convert_tga_to_dds(tga_path, dds_path, fmt)
            converted += 1
        except Exception as e:
            print(f"  ERROR converting {tga_path}: {e}")

    print(f"\nDone. Converted {converted} file(s).")


if __name__ == '__main__':
    main()
