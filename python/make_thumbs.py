#!/usr/bin/env python3
"""
make_thumbs.py — turn a folder of exported photos into a thumbs/ + full/
pair, sized for the Africa 2026 site's tile grid, and print the HTML
markup ready to paste into a page.

Workflow this is meant to slot into:
  1. In Photoshop Elements, filter by tag (e.g. "lions"), select all.
  2. File > Export As New File(s) -> dump them into a plain folder,
     e.g. ~/Desktop/lions_export, with a common base name so they sort
     predictably (lion-01.jpg, lion-02.jpg, ...).
  3. Run this script against that folder.
  4. Paste the printed <a class="tile">...</a> block into the page,
     inside a <div class="grid">.

What it does to each image:
  - full/<name>.jpg   — capped at --full-max px on the long side (default
                         2000), so multi-MB camera originals don't get
                         served as-is. Skipped (just copied) if already
                         smaller than that.
  - thumbs/<name>.jpg — resized-and-center-cropped to exactly
                         --tile-w x --tile-h at --scale x (default 340x256,
                         i.e. 2x a 170x128 CSS tile, for retina sharpness).
  - EXIF orientation is respected before resizing (so sideways phone/
    camera shots come out right-side up); EXIF is stripped from the
    output to keep files small, since GPS/timestamp metadata is already
    tracked separately for this project.
  - Caption -> alt/title text: if you've typed a Caption in Elements'
    Organizer for a photo, this reads it back out of the file (checked in
    order: IPTC Caption-Abstract, EXIF ImageDescription, XMP dc:description
    — Elements writes to all three) and uses it as that image's alt/title
    text, instead of the generic --label. Falls back to --label for any
    photo with no caption.
    IMPORTANT: Elements only embeds the caption in the file itself once
    you've told it to — select the photo(s) and run File > Save Metadata
    to Files before exporting, or the caption stays in the catalog
    database only and this script won't see it.

Usage:
  python3 make_thumbs.py /path/to/exported_folder --label lions
"""

import argparse
import html
import re
import sys
from pathlib import Path

from PIL import Image, ImageOps, IptcImagePlugin

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic"}
IPTC_CAPTION_KEY = (2, 120)  # Caption-Abstract
EXIF_IMAGE_DESCRIPTION = 0x010E


def read_caption(path):
    """Best-effort read of an embedded caption, checking the three fields
    Elements can write to, in the order Adobe documents them being written.
    Returns None if nothing usable is found (caller should fall back to
    --label)."""
    try:
        img = Image.open(path)
    except Exception:
        return None

    # 1. Legacy IPTC IIM — Caption-Abstract
    try:
        iptc = IptcImagePlugin.getiptcinfo(img)
        if iptc and IPTC_CAPTION_KEY in iptc:
            val = iptc[IPTC_CAPTION_KEY]
            if isinstance(val, bytes):
                val = val.decode("utf-8", "replace")
            val = val.strip()
            if val:
                return val
    except Exception:
        pass

    # 2. EXIF ImageDescription
    try:
        exif = img.getexif()
        val = exif.get(EXIF_IMAGE_DESCRIPTION)
        if val:
            if isinstance(val, bytes):
                val = val.decode("utf-8", "replace")
            val = val.strip()
            if val:
                return val
    except Exception:
        pass

    # 3. XMP dc:description (raw packet regex — Pillow doesn't parse XMP
    # into a structured dict, so this is a light-touch extraction rather
    # than a full XML parse)
    try:
        xmp = img.info.get("xmp")
        if xmp:
            if isinstance(xmp, bytes):
                xmp = xmp.decode("utf-8", "replace")
            m = re.search(r"<dc:description>.*?<rdf:li[^>]*>(.*?)</rdf:li>", xmp, re.S)
            if not m:
                m = re.search(r'dc:description="([^"]*)"', xmp)
            if m:
                val = html.unescape(m.group(1)).strip()
                if val:
                    return val
    except Exception:
        pass

    return None


def load_upright(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # fix sideways EXIF-rotated photos
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img


def save_jpeg(img, path, quality):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path, "JPEG", quality=quality, optimize=True)


def make_full(img, out_path, max_dim, quality):
    w, h = img.size
    longest = max(w, h)
    if longest <= max_dim:
        save_jpeg(img, out_path, quality)
        return
    scale = max_dim / float(longest)
    resized = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    save_jpeg(resized, out_path, quality)


def make_thumb(img, out_path, tile_w, tile_h, quality):
    """Resize to cover a tile_w x tile_h box, then center-crop to it exactly
    — the same visual result as CSS object-fit:cover, but baked into the
    file so the browser doesn't have to download more than it shows."""
    w, h = img.size
    target_ratio = tile_w / tile_h
    src_ratio = w / h

    if src_ratio > target_ratio:
        # source is relatively wider than target -> crop left/right
        new_h = h
        new_w = round(h * target_ratio)
    else:
        # source is relatively taller than target -> crop top/bottom
        new_w = w
        new_h = round(w / target_ratio)

    left = (w - new_w) // 2
    top = (h - new_h) // 2
    cropped = img.crop((left, top, left + new_w, top + new_h))
    resized = cropped.resize((tile_w, tile_h), Image.LANCZOS)
    save_jpeg(resized, out_path, quality)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="Folder of exported photos")
    ap.add_argument("--out", help="Output root (default: alongside source, i.e. its parent)")
    ap.add_argument("--web-root", default="",
                     help="Path to the output root AS IT SHOULD APPEAR IN THE HTML, relative to "
                          "the page it'll be pasted into, e.g. 'images/animals-birds'. Controls "
                          "only the printed markup's href/src paths, independent of --out (which "
                          "controls where files are physically written). Default: '' (thumbs/full "
                          "assumed to sit right next to the HTML file).")
    ap.add_argument("--tile-w", type=int, default=170, help="CSS tile width in px (default 170)")
    ap.add_argument("--tile-h", type=int, default=128, help="CSS tile height in px (default 128)")
    ap.add_argument("--scale", type=int, default=2, help="Thumbnail oversample factor for retina (default 2x)")
    ap.add_argument("--full-max", type=int, default=2000, help="Max long-side px for the 'full' version (default 2000)")
    ap.add_argument("--quality", type=int, default=85, help="JPEG quality (default 85)")
    ap.add_argument("--label", default="photo", help="Alt-text label used in the printed markup (default 'photo')")
    args = ap.parse_args()

    src_dir = Path(args.source).expanduser()
    if not src_dir.is_dir():
        sys.exit(f"Not a folder: {src_dir}")

    out_root = Path(args.out).expanduser() if args.out else src_dir.parent
    thumbs_dir = out_root / "thumbs"
    full_dir = out_root / "full"

    web_root = args.web_root.strip("/")
    web_prefix = f"{web_root}/" if web_root else ""

    tile_w, tile_h = args.tile_w, args.tile_h
    thumb_w, thumb_h = tile_w * args.scale, tile_h * args.scale

    files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not files:
        sys.exit(f"No images found in {src_dir}")

    markup_lines = ['<div class="grid">']
    n_ok, n_fail, n_captioned = 0, 0, 0

    for p in files:
        out_name = p.stem + ".jpg"
        try:
            caption = read_caption(p)
            if caption:
                n_captioned += 1
            text = html.escape(caption or args.label, quote=True)

            img = load_upright(p)
            make_full(img, full_dir / out_name, args.full_max, args.quality)
            make_thumb(img, thumbs_dir / out_name, thumb_w, thumb_h, args.quality)
            markup_lines.append(
                f'  <a class="tile" href="{web_prefix}full/{out_name}" title="{text}">'
                f'<img class="thumb" src="{web_prefix}thumbs/{out_name}" alt="{text}"></a>'
            )
            n_ok += 1
        except Exception as e:
            print(f"!! skipped {p.name}: {e}", file=sys.stderr)
            n_fail += 1

    markup_lines.append("</div>")

    print(f"\n{n_ok} images processed" + (f", {n_fail} skipped" if n_fail else ""), file=sys.stderr)
    print(f"{n_captioned} of {n_ok} had an embedded caption; the rest used --label '{args.label}'", file=sys.stderr)
    if n_ok and n_captioned == 0:
        print("(0 captions found — if you expected some, check you ran File > Save Metadata", file=sys.stderr)
        print(" to Files on these in Elements before exporting them.)", file=sys.stderr)
    print(f"thumbs/ -> {thumbs_dir}  ({thumb_w}x{thumb_h}px)", file=sys.stderr)
    print(f"full/   -> {full_dir}  (long side capped at {args.full_max}px)", file=sys.stderr)
    print("\n--- paste this into the page ---\n", file=sys.stderr)
    print("\n".join(markup_lines))


if __name__ == "__main__":
    main()
