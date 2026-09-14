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

Photos are processed in capture-time order (EXIF DateTimeOriginal, falling
back to file modified time), not filename order — so this works whether
you rename on export or keep original camera filenames like IMG_5711.jpg,
which won't sort chronologically once photos from different cameras or
renumbered rolls get mixed into one folder.

Each thumbnail links to a small generated photo page (in pages/) rather
than straight to the full-size jpg — that page shows the image plus
Previous / Home / Next links, so you can browse the whole batch without
shuttling back to the gallery grid each time. --home tells it which page
to treat as "Home" (default: index.html).

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
    photo with no caption. In practice Elements writes the caption into
    the file's metadata as soon as you save it in the Organizer, so this
    should just work without any extra export step — if a batch comes
    through with zero captions found and you expected some, that's the
    first thing worth double-checking, but it isn't the normal case.

Usage:
  python3 make_thumbs.py /path/to/exported_folder --label lions
"""

import argparse
import html
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps, IptcImagePlugin

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic"}
IPTC_CAPTION_KEY = (2, 120)  # Caption-Abstract
EXIF_IMAGE_DESCRIPTION = 0x010E
EXIF_IFD_TAG = 0x8769
EXIF_DATETIME_ORIGINAL = 0x9003
EXIF_DATETIME = 0x0132


def capture_time(path):
    """When the photo was actually taken, for sorting — since keeping
    original camera filenames means alphabetical order no longer tracks
    chronological order (different cameras, renumbered rolls, etc.).
    Prefers EXIF DateTimeOriginal, falls back to EXIF DateTime, then to
    the file's own modified time, then (only if truly nothing else is
    available) pushes it to the end rather than erroring out."""
    try:
        img = Image.open(path)
        exif = img.getexif()
        dt = None
        try:
            exif_ifd = exif.get_ifd(EXIF_IFD_TAG)
            dt = exif_ifd.get(EXIF_DATETIME_ORIGINAL)
        except Exception:
            pass
        if not dt:
            dt = exif.get(EXIF_DATETIME)
        if dt:
            if isinstance(dt, bytes):
                dt = dt.decode("utf-8", "replace")
            return datetime.strptime(dt.strip(), "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass

    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except Exception:
        return datetime.max


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


def reset_dir(path):
    """Wipe a directory this script owns (thumbs/, full/, or pages/) so a
    re-run doesn't leave orphaned files behind for photos that were dropped
    from the batch. Only ever called on those three specific subfolders,
    never on --out itself, to keep the blast radius small."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def rel_up(web_root):
    """'../' prefix needed to climb from <web_root>/pages/ back to whatever
    directory web_root itself is anchored in — i.e. the folder the HTML
    page that embeds these thumbnails lives in."""
    depth = (len([s for s in web_root.split("/") if s]) if web_root else 0) + 1
    return "../" * depth


PHOTO_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<link rel="stylesheet" href="{up}style.css">
</head>
<body>
<div class="photopage">
  <p class="photonav">
    <a href="{prev}">&larr; Previous</a>
    &middot;
    <a href="{home}">Home</a>
    &middot;
    <a href="{next}">Next &rarr;</a>
  </p>
  <img class="full" src="../full/{img_name}" alt="{caption}">
  <p class="photocaption">{caption}</p>
  <p class="photonav">
    <a href="{prev}">&larr; Previous</a>
    &middot;
    <a href="{home}">Home</a>
    &middot;
    <a href="{next}">Next &rarr;</a>
  </p>
</div>
</body>
</html>
"""


def write_photo_page(out_path, img_name, caption, prev_href, next_href, home_href, up):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        PHOTO_PAGE_TEMPLATE.format(
            title=caption or "photo",
            up=up,
            prev=prev_href,
            next=next_href,
            home=home_href,
            img_name=img_name,
            caption=caption,
        ),
        encoding="utf-8",
    )


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
    ap.add_argument("--home", default="index.html",
                     help="Filename (relative to the site root) that the generated photo pages' "
                          "'Home' link should point to, e.g. 'animals-birds.html'. Default: 'index.html'.")
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
    pages_dir = out_root / "pages"

    for d in (thumbs_dir, full_dir, pages_dir):
        reset_dir(d)
    print("cleared existing thumbs/, full/, and pages/ before regenerating", file=sys.stderr)

    web_root = args.web_root.strip("/")
    web_prefix = f"{web_root}/" if web_root else ""
    up = rel_up(web_root)
    home_href = up + args.home

    tile_w, tile_h = args.tile_w, args.tile_h
    thumb_w, thumb_h = tile_w * args.scale, tile_h * args.scale

    files = sorted(
        (p for p in src_dir.iterdir() if p.suffix.lower() in IMG_EXTS),
        key=capture_time,
    )
    if not files:
        sys.exit(f"No images found in {src_dir}")

    # pass 1: resize/crop every image, collect per-image data
    items = []  # (out_name, text_for_alt, caption_or_empty)
    n_fail, n_captioned = 0, 0

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
            items.append((out_name, text))
        except Exception as e:
            print(f"!! skipped {p.name}: {e}", file=sys.stderr)
            n_fail += 1

    n_ok = len(items)

    # pass 2: now that we know the final ordered list, write each photo
    # page with correct Previous/Next links (wraps around at the ends)
    markup_lines = ['<div class="grid">']
    for i, (out_name, text) in enumerate(items):
        page_name = Path(out_name).stem + ".html"
        prev_page = Path(items[(i - 1) % n_ok][0]).stem + ".html"
        next_page = Path(items[(i + 1) % n_ok][0]).stem + ".html"

        write_photo_page(
            pages_dir / page_name,
            img_name=out_name,
            caption=text,
            prev_href=prev_page,
            next_href=next_page,
            home_href=home_href,
            up=up,
        )

        markup_lines.append(
            f'  <a class="tile" href="{web_prefix}pages/{page_name}" title="{text}">'
            f'<img class="thumb" src="{web_prefix}thumbs/{out_name}" alt="{text}"></a>'
        )

    markup_lines.append("</div>")

    print(f"\n{n_ok} images processed" + (f", {n_fail} skipped" if n_fail else ""), file=sys.stderr)
    print(f"{n_captioned} of {n_ok} had an embedded caption; the rest used --label '{args.label}'", file=sys.stderr)
    if n_ok and n_captioned == 0:
        print("(0 captions found — if you expected some, worth a quick check that these", file=sys.stderr)
        print(" particular photos actually had a Caption typed and saved in the Organizer.)", file=sys.stderr)
    print(f"thumbs/ -> {thumbs_dir}  ({thumb_w}x{thumb_h}px)", file=sys.stderr)
    print(f"full/   -> {full_dir}  (long side capped at {args.full_max}px)", file=sys.stderr)
    print(f"pages/  -> {pages_dir}  (one photo page per image, with Previous/Home/Next)", file=sys.stderr)
    print("\n--- paste this into the page ---\n", file=sys.stderr)
    print("\n".join(markup_lines))


if __name__ == "__main__":
    main()
