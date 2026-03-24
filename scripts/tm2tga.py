#!/usr/bin/env python3
"""
TM2 / TEX ↔ TGA Converter
Converts PS2 .tm2 and PSP .tex indexed textures to/from 32-bit TGA with alpha.

  PS2 TM2  : PSMT8 swizzle (default), PS2 CLUT unswizzle applied
  PSP TEX  : 16×8 tile swizzle, no CLUT swizzle
  Alpha    : PS2/PSP alpha scale (0-128) correctly expanded to (0-255)
  Mip      : Mip 0 only by default; use -mip N to extract a specific level
  Reverse  : TGA → TM2 (PS2) via Pillow quantisation + PSMT8 swizzle

Usage:
  python tm2tga.py input.tm2                  → input.tga  (mip 0, PSMT8)
  python tm2tga.py input.tex                  → input.tga  (mip 0)
  python tm2tga.py *.tm2                      → batch, all mip 0
  python tm2tga.py folder/                    → all .tm2 and .tex in folder
  python tm2tga.py input.tm2 output.tga       → explicit output path
  python tm2tga.py input.tm2 -mip 2           → mip level 2
  python tm2tga.py input.tga output.tm2       → TGA → PS2 TM2 (requires Pillow)
  python tm2tga.py input.tga output.tm2 -swizzle none

Flags:
  -mip N        Extract mip level N (default: 0)
  -swizzle X    Swizzle mode for TM2: psmt8 (default), psmt4 (decode only), none
  -allalpha     Force alpha to 255 on output (discard stored alpha)

https://github.com/ravenDS
"""

import struct
import sys
import os
import glob


# PS2 PSMT8 swizzle / unswizzle

_PAT_A = [
    (0,  0), (1,  4), (0,  8), (1, 12),
    (0,  1), (1,  5), (0,  9), (1, 13),
    (0,  2), (1,  6), (0, 10), (1, 14),
    (0,  3), (1,  7), (0, 11), (1, 15),
]
_PAT_B = [
    (0,  4), (1,  0), (0, 12), (1,  8),
    (0,  5), (1,  1), (0, 13), (1,  9),
    (0,  6), (1,  2), (0, 14), (1, 10),
    (0,  7), (1,  3), (0, 15), (1, 11),
]

def _psmt8_pat(ab_swapped, sub_group):
    if ab_swapped:
        return _PAT_A if sub_group == 1 else _PAT_B
    else:
        return _PAT_A if sub_group == 0 else _PAT_B


def can_swizzle_psmt8(byte_width, height):
    return (byte_width >= 32 and byte_width % 32 == 0
            and height >= 4 and height % 4 == 0)


def _psmt8_core(src, dst, byte_width, height, writing):
    """
    Shared inner loop for both swizzle and unswizzle.
    writing=True  → src is linear, dst is swizzled   (Swizzle)
    writing=False → src is swizzled, dst is linear   (Unswizzle)
    """
    hw  = byte_width // 2
    bph = hw // 16

    for rg in range(height // 4):
        base = rg * 4
        ab   = (rg % 2) == 1

        for pi in range(2):
            lr1 = base + pi
            lr2 = base + pi + 2

            for hf in range(2):
                swiz_row = rg * 4 + pi * 2 + hf
                col_base = hf * hw
                seq_pos  = swiz_row * byte_width

                for b in range(bph):
                    bc = col_base + b * 16

                    for sg in range(2):
                        p = _psmt8_pat(ab, sg)

                        for i in range(16):
                            lin_row = lr1 if p[i][0] == 0 else lr2
                            lin_col = bc  + p[i][1]
                            lin_idx = lin_row * byte_width + lin_col

                            if writing:           # linear → swizzled
                                dst[seq_pos] = src[lin_idx]
                            else:                 # swizzled → linear
                                dst[lin_idx] = src[seq_pos]
                            seq_pos += 1


def unswizzle_psmt8(data, byte_width, height):
    """Decode PS2 PSMT8 tile layout → linear pixel data."""
    result = bytearray(len(data))
    _psmt8_core(data, result, byte_width, height, writing=False)
    return bytes(result)


def swizzle_psmt8(data, byte_width, height):
    """Encode linear pixel data → PS2 PSMT8 tile layout."""
    result = bytearray(len(data))
    _psmt8_core(data, result, byte_width, height, writing=True)
    return bytes(result)


# PS2 CLUT (palette) unswizzle
# calling it twice returns the original

def unswizzle_clut(palette):
    """
    Unswizzle (or re-swizzle) a 256-colour PS2 palette (1024 bytes).
    Swaps rows 8-15 and 16-23 within every 32-colour block.
    No-op for 16-colour (64-byte) palettes.
    """
    if len(palette) != 1024:
        return palette
    result = bytearray(palette)
    for block in range(0, 256, 32):
        for i in range(8):
            a = (block + 8  + i) * 4
            b = (block + 16 + i) * 4
            for j in range(4):
                result[a + j], result[b + j] = result[b + j], result[a + j]
    return bytes(result)


def palette_to_rgba(palette_bytes, bpp, ps2_clut_unswizzle=True):
    """
    Convert raw palette bytes (RGBA, PS2/PSP alpha 0-128) to a list of
    (R, G, B, A) tuples where A is scaled to 0-255.
    ps2_clut_unswizzle: apply PS2 CLUT reorder (True for TM2, False for TEX).
    """
    if bpp == 8 and ps2_clut_unswizzle:
        palette_bytes = unswizzle_clut(palette_bytes)
    n = len(palette_bytes) // 4
    colors = []
    for i in range(n):
        r = palette_bytes[i * 4]
        g = palette_bytes[i * 4 + 1]
        b = palette_bytes[i * 4 + 2]
        a = min(palette_bytes[i * 4 + 3] * 2, 255)   # PS2/PSP: 128 → 255
        colors.append((r, g, b, a))
    return colors


def rgba_to_ps2_palette(colors, apply_clut_swizzle=True):
    """
    Convert a list of (R,G,B,A) tuples (A in 0-255) to raw PS2 palette bytes
    (RGBA, alpha 0-128).  apply_clut_swizzle should be True for TM2.
    """
    raw = bytearray(len(colors) * 4)
    for i, (r, g, b, a) in enumerate(colors):
        raw[i * 4]     = r
        raw[i * 4 + 1] = g
        raw[i * 4 + 2] = b
        raw[i * 4 + 3] = round(a * 128 / 255)
    if apply_clut_swizzle and len(colors) == 256:
        raw = unswizzle_clut(bytes(raw))   # its own inverse → re-swizzles
    return bytes(raw)


# PSP tile swizzle / unswizzle  (16×8 pixel tiles, 128 bytes each)

def can_swizzle_psp(width, height, bpp):
    if bpp == 4:
        return width >= 32 and height >= 8
    return width >= 16 and height >= 8


def _psp_core(src, dst, byte_width, height, writing):
    """Shared inner loop for PSP tile swizzle/unswizzle."""
    row_blocks = byte_width // 16
    for y in range(height):
        for x in range(byte_width):
            bx  = x  // 16
            by  = y  // 8
            bi  = by * row_blocks + bx
            bof = bi * 128 + (y % 8) * 16 + (x % 16)
            lin = y  * byte_width + x
            if writing:           # linear → swizzled
                dst[bof] = src[lin]
            else:                 # swizzled → linear
                dst[lin] = src[bof]


def unswizzle_psp(data, width, height, bpp):
    """
    Decode PSP tile layout → linear pixel data.
    For 4bpp: byte_width = width // 2 (two pixels per byte).
    """
    bw = (width + 1) // 2 if bpp == 4 else width
    if not can_swizzle_psp(width, height, bpp):
        return data
    result = bytearray(len(data))
    _psp_core(data, result, bw, height, writing=False)
    return bytes(result)


def swizzle_psp(data, width, height, bpp):
    """Encode linear pixel data → PSP tile layout."""
    bw = (width + 1) // 2 if bpp == 4 else width
    if not can_swizzle_psp(width, height, bpp):
        return data
    result = bytearray(len(data))
    _psp_core(data, result, bw, height, writing=True)
    return bytes(result)


# PSP row-pitch helpers (GE requires minimum 16-byte rows)

def _psp_pitch(width, bpp):
    return max(width, 16) if bpp == 8 else max((width + 1) // 2, 8)

def _psp_disk_size(width, height, bpp):
    return height * _psp_pitch(width, bpp)

def _psp_strip_pitch(data, width, height, bpp):
    pitch   = _psp_pitch(width, bpp)
    row_b   = width if bpp == 8 else (width + 1) // 2
    if pitch == row_b:
        return data
    result = bytearray(row_b * height)
    for row in range(height):
        result[row * row_b:(row + 1) * row_b] = data[row * pitch:row * pitch + row_b]
    return bytes(result)

def _psp_add_pitch(data, width, height, bpp):
    pitch   = _psp_pitch(width, bpp)
    row_b   = width if bpp == 8 else (width + 1) // 2
    if pitch == row_b:
        return data
    result = bytearray(height * pitch)
    for row in range(height):
        result[row * pitch:row * pitch + row_b] = data[row * row_b:(row + 1) * row_b]
    return bytes(result)


# indexed pixel expansion -> RGBA

def expand_indexed(indices, bpp, colors, force_opaque=False):
    """
    Expand palette-indexed pixel bytes to a list of (R,G,B,A) tuples.
    4bpp: two pixels per byte, low nibble first.
    """
    n = len(colors)
    def lookup(idx):
        c = colors[idx] if idx < n else (0, 0, 0, 255)
        return (c[0], c[1], c[2], 255) if force_opaque else c

    pixels = []
    if bpp == 8:
        for idx in indices:
            pixels.append(lookup(idx))
    else:   # 4bpp
        for byte in indices:
            pixels.append(lookup(byte & 0xF))
            pixels.append(lookup((byte >> 4) & 0xF))
    return pixels


# TM2 (PS2) parsing
# ParseTM2Header / ExtractMipLevels

TM2_MAGIC = b'TIM2'

def parse_tm2(data, offset=0):
    """
    Parse a TM2 file header.  Returns a dict with all geometry needed to
    locate pixel data and palette for any mip level.
    """
    if data[offset:offset + 4] != TM2_MAGIC:
        raise ValueError(f"Not a TIM2 file (bad magic at offset 0x{offset:X})")

    fmt      = data[offset + 5]
    pic_ofs  = offset + (16 if fmt == 0 else 128)

    image_size   = struct.unpack_from('<I', data, pic_ofs +  8)[0]
    clut_size    = struct.unpack_from('<I', data, pic_ofs +  4)[0]
    clut_colors  = struct.unpack_from('<H', data, pic_ofs + 14)[0]
    image_type   = data[pic_ofs + 19]
    mip_count    = data[pic_ofs + 17]
    width        = struct.unpack_from('<H', data, pic_ofs + 20)[0]
    height       = struct.unpack_from('<H', data, pic_ofs + 22)[0]
    header_size  = struct.unpack_from('<H', data, pic_ofs + 12)[0]

    if   image_type == 4: bpp = 4
    elif image_type == 5: bpp = 8
    else: raise ValueError(f"Unsupported TM2 image type {image_type} "
                           "(only 4=4bpp, 5=8bpp supported)")

    pix_ofs = pic_ofs + header_size
    pal_ofs = pix_ofs + image_size

    # extended mip-slot sizes table (header_size ≥ 128, fmt=1 files)
    mip_declared = None
    gs_miptbp2   = 0
    if header_size >= 128:
        mip_declared = [
            struct.unpack_from('<Q', data, pic_ofs + 64 + i * 8)[0]
            for i in range(4)
        ]
        gs_miptbp2 = struct.unpack_from('<Q', data, pic_ofs + 56)[0]

    return {
        'width': width, 'height': height, 'bpp': bpp,
        'mip_count': mip_count,
        'image_size': image_size, 'clut_size': clut_size, 'clut_colors': clut_colors,
        'pix_ofs': pix_ofs, 'pal_ofs': pal_ofs,
        'mip_declared': mip_declared, 'gs_miptbp2': gs_miptbp2,
    }


def _mip_size(w, h, bpp):
    if bpp == 8: return w * h
    if bpp == 4: return (w * h) // 2
    raise ValueError(f"Unsupported bpp {bpp}")


def read_tm2_mip(data, info, mip_level=0, swizzle='psmt8'):
    """
    Return (width, height, linear_pixel_bytes, rgba_colors_list) for one mip.

    swizzle: 'psmt8' (default), 'none'
      'psmt4' is recognised but not decoded (data returned as-is with a warning).
    """
    w, h, bpp = info['width'], info['height'], info['bpp']

    # Walk forward through mip offsets
    offset = info['pix_ofs']
    for lvl in range(mip_level):
        offset += _mip_size(w, h, bpp)
        w //= 2; h //= 2

    size = _mip_size(w, h, bpp)
    if offset + size > info['pal_ofs']:
        raise ValueError(
            f"Mip {mip_level} extends past palette – file may not have this level "
            f"(max mip count = {info['mip_count']})")

    raw = data[offset:offset + size]

    # unswizzle
    if swizzle == 'psmt8':
        bw = w // 2 if bpp == 4 else w
        if can_swizzle_psmt8(bw, h):
            raw = unswizzle_psmt8(raw, bw, h)
        # else: too small to swizzle, leave as-is
    elif swizzle == 'psmt4':
        print(f"  WARNING: PSMT4 decode not implemented; pixel data returned unswizzled.")
    # 'none': leave raw as-is

    # palette
    pal_bytes = data[info['pal_ofs']:info['pal_ofs'] + info['clut_size']]
    colors = palette_to_rgba(pal_bytes, bpp, ps2_clut_unswizzle=(bpp == 8))

    return w, h, raw, colors


# TEX (PSP) parsing
# ParseTexHeader / ExtractTexMipLevels

def parse_tex(data):
    """
    Parse a PSP .TEX file header.

    Layout (all uint32 LE):
      0:  version (= 1)
      4:  bpp (4 or 8)
      8:  width
     12:  height
     16:  GS-data end offset  → palette at +8
     20:  entry count (≈ mip count)
     24+: offset table (entry_count × uint32)
          MipOffsets[i]+8 = start of mip[i]'s pixel data
    """
    if len(data) < 28:
        raise ValueError("File too short for .TEX header")

    bpp        = struct.unpack_from('<I', data,  4)[0]
    width      = struct.unpack_from('<I', data,  8)[0]
    height     = struct.unpack_from('<I', data, 12)[0]
    gs_end     = struct.unpack_from('<I', data, 16)[0]
    entry_cnt  = struct.unpack_from('<I', data, 20)[0]

    pal_ofs   = gs_end + 8
    clut_size = 64 if bpp == 4 else 1024

    offsets = [
        struct.unpack_from('<I', data, 24 + i * 4)[0]
        for i in range(entry_cnt)
    ]

    return {
        'width': width, 'height': height, 'bpp': bpp,
        'entry_count': entry_cnt,
        'pal_ofs': pal_ofs, 'clut_size': clut_size,
        'mip_offsets': offsets,
    }


def _tex_mip_count(info, data_len):
    """Count usable mip levels (mirrors CountMips in PSPSwizzle.vb)."""
    count = 0
    w, h = info['width'], info['height']
    while count < info['entry_count'] and w >= 1 and h >= 1:
        disk_sz   = _psp_disk_size(w, h, info['bpp'])
        pix_start = info['mip_offsets'][count] + 8
        if pix_start + disk_sz > data_len:
            break
        if count < info['entry_count'] - 1:
            stride = info['mip_offsets'][count + 1] - info['mip_offsets'][count]
            if stride > disk_sz:
                break
        count += 1
        w //= 2; h //= 2
    return count


def read_tex_mip(data, info, mip_level=0):
    """
    Return (width, height, linear_pixel_bytes, rgba_colors_list) for one mip.
    PSP does not use PS2-style CLUT swizzle.
    """
    mip_count = _tex_mip_count(info, len(data))
    if mip_level >= mip_count:
        raise ValueError(
            f"Mip {mip_level} not available (found {mip_count} usable levels)")

    w   = info['width']  >> mip_level
    h   = info['height'] >> mip_level
    bpp = info['bpp']

    pix_start = info['mip_offsets'][mip_level] + 8
    disk_sz   = _psp_disk_size(w, h, bpp)

    pitched = data[pix_start:pix_start + disk_sz]
    raw     = _psp_strip_pitch(pitched, w, h, bpp)
    linear  = unswizzle_psp(raw, w, h, bpp)

    pal_bytes = data[info['pal_ofs']:info['pal_ofs'] + info['clut_size']]
    colors = palette_to_rgba(pal_bytes, bpp, ps2_clut_unswizzle=False)

    return w, h, linear, colors


#  TGA writing

def write_tga(path, width, height, pixels):
    """Write a 32-bit uncompressed TGA (BGRA, top-left origin)."""
    with open(path, 'wb') as f:
        f.write(struct.pack('<B', 0))        # ID length
        f.write(struct.pack('<B', 0))        # colour map type
        f.write(struct.pack('<B', 2))        # image type: uncompressed true-colour
        f.write(b'\x00' * 5)                 # colour map spec (unused)
        f.write(struct.pack('<H', 0))        # X origin
        f.write(struct.pack('<H', 0))        # Y origin
        f.write(struct.pack('<HH', width, height))
        f.write(struct.pack('<B', 32))       # bits per pixel
        f.write(struct.pack('<B', 0x28))     # image descriptor: top-left + 8 alpha bits
        for r, g, b, a in pixels:
            f.write(struct.pack('4B', b, g, r, a))
    print(f"  Saved: {path} ({width}×{height}, 32-bit BGRA TGA)")


# TGA reading  (for reverse conversion)

def _decode_tga_rle(data, pixel_count, bpp):
    out = bytearray()
    i   = 0
    while len(out) // bpp < pixel_count:
        if i >= len(data):
            break
        hdr = data[i]; i += 1
        count = (hdr & 0x7F) + 1
        if hdr & 0x80:
            px = data[i:i + bpp]; i += bpp
            out += px * count
        else:
            out += data[i:i + count * bpp]; i += count * bpp
    return bytes(out)


def read_tga(path):
    """Return (width, height, list_of_(R,G,B,A)) for a 32-bit TGA."""
    with open(path, 'rb') as f:
        raw = f.read()

    id_len     = raw[0]
    img_type   = raw[2]
    width      = struct.unpack_from('<H', raw, 12)[0]
    height     = struct.unpack_from('<H', raw, 14)[0]
    bpp        = raw[16]
    img_desc   = raw[17]

    if img_type not in (2, 10):
        raise ValueError(f"Unsupported TGA type {img_type} (need 2=uncompressed or 10=RLE)")
    if bpp != 32:
        raise ValueError(f"Expected 32-bit TGA, got {bpp}-bit")

    pxdata = raw[18 + id_len:]
    if img_type == 10:
        pxdata = _decode_tga_rle(pxdata, width * height, 4)

    top_left = bool((img_desc >> 5) & 1)
    pixels = [
        (pxdata[i*4+2], pxdata[i*4+1], pxdata[i*4+0], pxdata[i*4+3])
        for i in range(width * height)
    ]
    if not top_left:
        rows = [pixels[y * width:(y + 1) * width] for y in range(height)]
        pixels = [p for row in reversed(rows) for p in row]

    return width, height, pixels


# standalone TM2 builder  (reverse direction)
# BuildStandaloneTM2 (fmt=0, single mip)

def build_standalone_tm2(pixel_data, width, height, bpp, palette_bytes):
    """
    Build a minimal single-mip TM2 (fmt=0, 16-byte file header, 48-byte picture header).
    pixel_data must already be swizzled; palette_bytes must already be CLUT-swizzled.
    """
    FILE_HDR = 16
    PIC_HDR  = 48
    img_type   = 4 if bpp == 4 else 5
    clut_colors = 16 if bpp == 4 else 256
    img_size   = len(pixel_data)
    clut_size  = len(palette_bytes)
    total_pic  = PIC_HDR + img_size + clut_size

    out = bytearray()

    # file header (16 bytes)
    out += b'TIM2'
    out += bytes([4, 0])              # version=4, fmt=0 → 16-byte file header
    out += struct.pack('<H', 1)       # picture count
    out += struct.pack('<Q', 0)       # 8 bytes reserved

    # picture header (48 bytes)
    out += struct.pack('<I', total_pic)    # +0  total picture size
    out += struct.pack('<I', clut_size)   # +4  CLUT size
    out += struct.pack('<I', img_size)    # +8  image size
    out += struct.pack('<H', PIC_HDR)     # +12 header size = 48
    out += struct.pack('<H', clut_colors) # +14 CLUT colours
    out += bytes([0, 1, 3, img_type])     # +16 pict_fmt / mip_count=1 / clut_type / img_type
    out += struct.pack('<HH', width, height)  # +20 +22
    out += struct.pack('<QQ', 0, 0)       # +24 GsTex0/1 (zeroed for simple case)
    out += struct.pack('<II', 0, 0)       # +40 GsRegs / GsTexClut

    # pixel data + palette
    out += pixel_data
    out += palette_bytes

    return bytes(out)


# conversion helpers

def detect_format(path):
    """Return 'tm2', 'tex', or None based on extension and/or magic."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.tm2':
        return 'tm2'
    if ext == '.tex':
        return 'tex'
    # Fallback: check TIM2 magic
    try:
        with open(path, 'rb') as f:
            magic = f.read(4)
        if magic == TM2_MAGIC:
            return 'tm2'
    except OSError:
        pass
    return None


def convert_to_tga(src_path, tga_path=None, mip_level=0,
                   swizzle='psmt8', force_opaque=False):
    """Convert a TM2 or TEX file to a 32-bit TGA."""
    if tga_path is None:
        base = os.path.splitext(src_path)[0]
        suffix = '' if mip_level == 0 else f'_mip{mip_level}'
        tga_path = base + suffix + '.tga'

    print(f"  Reading: {src_path}")
    with open(src_path, 'rb') as f:
        data = f.read()

    fmt = detect_format(src_path)
    if fmt == 'tm2':
        info = parse_tm2(data)
        print(f"  TM2: {info['width']}×{info['height']} "
              f"{info['bpp']}bpp, {info['mip_count']} mip(s)")
        w, h, indices, colors = read_tm2_mip(data, info, mip_level, swizzle)
    elif fmt == 'tex':
        info = parse_tex(data)
        n = _tex_mip_count(info, len(data))
        print(f"  TEX: {info['width']}×{info['height']} "
              f"{info['bpp']}bpp, {n} mip(s)")
        w, h, indices, colors = read_tex_mip(data, info, mip_level)
    else:
        raise ValueError(f"Unknown format: {src_path} "
                         "(expected .tm2 or .tex extension, or TIM2 magic)")

    pixels = expand_indexed(indices, info['bpp'], colors, force_opaque)
    write_tga(tga_path, w, h, pixels)
    return tga_path


def convert_tga_to_tm2(tga_path, tm2_path, swizzle='psmt8'):
    """
    Convert a 32-bit TGA to a PS2 TM2 file.
    Requires Pillow (pip install Pillow) for palette quantisation.
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "TGA → TM2 conversion requires Pillow.\n"
            "  Install: pip install Pillow")

    print(f"  Reading: {tga_path}")
    img = Image.open(tga_path).convert('RGBA')
    width, height = img.size
    print(f"  Quantising {width}×{height} → 256 colours…")

    # quantise RGB, keep alpha separately
    img_rgb = img.convert('RGB')
    img_q   = img_rgb.quantize(colors=256, dither=Image.Dither.FLOYDSTEINBERG)
    indices = list(img_q.tobytes())           # 1 byte = palette index per pixel
    pal_rgb = img_q.getpalette()              # flat [R,G,B, R,G,B, …] × 256

    # average alpha per palette entry from the original RGBA image
    orig_raw    = img.tobytes()               # flat RGBA bytes
    orig_pixels = [
        (orig_raw[i*4], orig_raw[i*4+1], orig_raw[i*4+2], orig_raw[i*4+3])
        for i in range(width * height)
    ]
    alpha_sum   = [0] * 256
    alpha_cnt   = [0] * 256
    for pix_idx, pal_idx in enumerate(indices):
        alpha_sum[pal_idx] += orig_pixels[pix_idx][3]
        alpha_cnt[pal_idx] += 1

    # build RGBA colour list (A already in 0-255)
    colors = []
    for i in range(256):
        avg_a = alpha_sum[i] // alpha_cnt[i] if alpha_cnt[i] else 255
        colors.append((pal_rgb[i * 3], pal_rgb[i * 3 + 1], pal_rgb[i * 3 + 2], avg_a))

    # encode to PS2 palette (RGBA, alpha 0-128, CLUT-swizzled)
    pal_bytes = rgba_to_ps2_palette(colors, apply_clut_swizzle=True)

    # encode pixel indices
    pixel_bytes = bytes(indices)

    # swizzle pixel data
    if swizzle == 'psmt8' and can_swizzle_psmt8(width, height):
        pixel_bytes = swizzle_psmt8(pixel_bytes, width, height)
    elif swizzle != 'none':
        print(f"  WARNING: swizzle '{swizzle}' not supported for export; "
              "using 'none' instead.")

    tm2_data = build_standalone_tm2(pixel_bytes, width, height, 8, pal_bytes)
    with open(tm2_path, 'wb') as f:
        f.write(tm2_data)
    print(f"  Saved: {tm2_path} ({width}×{height}, 8bpp, "
          f"{'PSMT8' if swizzle == 'psmt8' else 'unswizzled'})")
    return tm2_path


# CLI

def _usage():
    print("Source: https://github.com/RavenDS/flatout-blender-tools")
    print()
    print("Usage:")
    print("  tm2tga.py <input.tm2|input.tex> [output.tga] [-mip N] [-swizzle MODE] [-allalpha]")
    print("  tm2tga.py <input.tga> <output.tm2>            [-swizzle MODE]")
    print("  tm2tga.py *.tm2                               (batch, mip 0)")
    print("  tm2tga.py folder/                             (all .tm2 and .tex)")
    print()
    print("Flags:")
    print("  -mip N        Mip level to extract (default: 0, i.e. largest)")
    print("  -swizzle X    Swizzle mode: psmt8 (default), none")
    print("  -allalpha     Output fully-opaque alpha (ignore stored alpha channel)")
    print()
    print("Formats:")
    print("  .tm2  PS2 TIM2 – PSMT8 unswizzle + PS2 CLUT unswizzle applied")
    print("  .tex  PSP TEX  – 16×8 tile unswizzle; no CLUT unswizzle")
    print()
    print("Examples:")
    print("  python tm2tga.py skin1.tm2")
    print("  python tm2tga.py skin1.tm2 -mip 1")
    print("  python tm2tga.py skin1.tm2 out.tga -swizzle none")
    print("  python tm2tga.py lowskin1.tex")
    print("  python tm2tga.py mytexture.tga skin1.tm2          # requires Pillow")


def main():
    args = sys.argv[1:]
    if not args:
        _usage(); sys.exit(1)

    # parse flags
    mip_level   = 0
    swizzle     = 'psmt8'
    force_opaque = False
    positional  = []

    i = 0
    while i < len(args):
        a = args[i]
        if   a == '-mip'      and i + 1 < len(args):
            mip_level = int(args[i + 1]); i += 2
        elif a == '-swizzle'  and i + 1 < len(args):
            swizzle = args[i + 1].lower(); i += 2
        elif a == '-allalpha':
            force_opaque = True; i += 1
        else:
            positional.append(a); i += 1

    if not positional:
        _usage(); sys.exit(1)

    # detect reverse mode: TGA -> TM2
    if (len(positional) == 2
            and positional[0].lower().endswith('.tga')
            and positional[1].lower().endswith('.tm2')):
        try:
            convert_tga_to_tm2(positional[0], positional[1], swizzle)
            print("\nDone. Converted 1 file.")
        except Exception as e:
            print(f"  ERROR: {e}")
        return

    # forward mode: TM2/TEX -> TGA
    inputs = []
    output = None

    if len(positional) == 1 and os.path.isdir(positional[0]):
        # entire directory
        folder = positional[0]
        inputs  = (glob.glob(os.path.join(folder, '*.tm2')) +
                   glob.glob(os.path.join(folder, '*.tex')))
        if not inputs:
            print(f"No .tm2 or .tex files found in: {folder}")
            sys.exit(1)
    else:
        for arg in positional:
            expanded = glob.glob(arg)
            inputs.extend(expanded if expanded else [arg])

        # if last positional is an explicit .tga output path, treat it as such
        if (len(positional) >= 2
                and positional[-1].lower().endswith('.tga')
                and not os.path.isdir(positional[-1])):
            output  = positional[-1]
            inputs  = []
            for arg in positional[:-1]:
                expanded = glob.glob(arg)
                inputs.extend(expanded if expanded else [arg])

    ok = err = 0
    for src in inputs:
        ext = os.path.splitext(src)[1].lower()
        if ext not in ('.tm2', '.tex'):
            continue
        try:
            convert_to_tga(src, output, mip_level, swizzle, force_opaque)
            ok += 1
        except Exception as e:
            print(f"  ERROR converting {src}: {e}")
            err += 1

    print(f"\nDone. Converted {ok} file(s)" + (f", {err} error(s)." if err else "."))


if __name__ == '__main__':
    main()
