#!/usr/bin/env python3
"""
transcribe_notes.py -- turn evening voice-memo recap recordings into a
day-by-day trip journal, offline.

THE IDEA
--------
Each evening at camp, record ONE voice memo (2-3 min) talking through the
day: where you were, what you saw, what surprised you, who you met, which
photos to flag. Don't type, don't edit -- just talk. This script does the
rest once you're home (or at a camp with a laptop and no need for internet):

  1. Transcribes every voice memo with a local Whisper model -- no internet
     required at run time, only for the one-time model download below.
  2. Figures out which trip day each memo belongs to, either from a date in
     the filename or the file's timestamp, and looks up the location/camp
     for that day from itinerary.csv.
  3. Optionally scans a folder of photos and lists, per day, the filenames
     and times of everything you shot that day (from EXIF), so you have a
     skeleton to pick highlights from later -- you don't have to remember
     which shot was which.
  4. Writes one markdown file per day plus a combined trip-journal.md.

ONE-TIME SETUP (needs internet, do this before you leave)
-----------------------------------------------------------
    pip install faster-whisper pillow pillow-heif --break-system-packages

Then run the script once on any short test audio file so it downloads and
caches the model (default: "small", a good speed/accuracy balance for a
laptop CPU). After that first run it never touches the network again.

INTEL MAC NOTE: faster-whisper's native backend (ctranslate2) can segfault
on Intel Macs due to a duplicate-OpenMP conflict with PyTorch/numpy. If you
hit "exit code 139 / SIGSEGV", first try:
    KMP_DUPLICATE_LIB_OK=TRUE python transcribe_notes.py ...
If it still crashes, force the pure-PyTorch backend instead:
    pip install openai-whisper --break-system-packages
    python transcribe_notes.py --engine whisper ...

EXPORTING VOICE MEMOS FROM YOUR IPHONE
---------------------------------------
Voice Memos app -> select a memo -> Share -> Save to Files -> a folder you
sync to your laptop (iCloud Drive / a folder you AirDrop into). The
filenames Apple gives them (e.g. "New Recording 3.m4a") don't carry a date
you can read, so the script falls back to the file's modified-time, which
survives AirDrop/iCloud sync reasonably well. If you want to be certain,
rename each file to start with the date, e.g. "2026-08-17 recap.m4a" --
the script prefers that if it's there.

USAGE
-----
    python transcribe_notes.py \\
        --memos ~/Safari2026/voice_memos \\
        --photos ~/Safari2026/photos \\
        --itinerary itinerary.csv \\
        --out ~/Safari2026/notes

All arguments except --memos are optional (photos and itinerary just add
more context if you have them).
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".aac", ".caf"}
PHOTO_EXTS = {".jpg", ".jpeg", ".heic", ".heif", ".png", ".dng", ".cr2", ".nef"}

DATE_IN_NAME = re.compile(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})")


# ---------------------------------------------------------------- itinerary

# Spreadsheet apps love to silently reformat a "date" column the moment you
# open and save the file (ISO "2026-08-16" becomes "8/16/26", etc). Rather
# than requiring ISO format, try a handful of common ones and normalize to
# ISO internally so the lookup still works no matter how Excel/Numbers/
# Google Sheets last mangled it.
CSV_DATE_FORMATS = [
    "%Y-%m-%d",   # 2026-08-16
    "%m/%d/%y",   # 8/16/26
    "%m/%d/%Y",   # 8/16/2026
    "%d/%m/%Y",   # 16/08/2026
    "%d-%m-%Y",   # 16-08-2026
    "%d %b %Y",   # 16 Aug 2026
    "%B %d, %Y",  # August 16, 2026
]


def parse_csv_date(s):
    s = s.strip()
    for fmt in CSV_DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def load_itinerary(path):
    """date (any common format) -> dict(country, location, camp, note),
    keyed internally by ISO date string."""
    lookup = {}
    if not path:
        return lookup
    p = Path(path)
    if not p.exists():
        print(f"  (no itinerary file at {p}, skipping location lookup)")
        return lookup
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = row.get("date", "")
            d = parse_csv_date(raw)
            if d is None:
                print(f"  (couldn't parse itinerary date '{raw}', skipping that row)")
                continue
            lookup[d.isoformat()] = row
    return lookup


# ------------------------------------------------------------- date sniffing

def date_from_filename(path: Path):
    m = DATE_IN_NAME.search(path.stem)
    if not m:
        return None
    try:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).date()
    except ValueError:
        return None


def date_from_mtime(path: Path):
    return datetime.fromtimestamp(path.stat().st_mtime).date()


def recording_date(path: Path):
    return date_from_filename(path) or date_from_mtime(path)


# ------------------------------------------------------------- transcription

def get_transcriber(model_size: str, engine: str = "auto"):
    """Returns a function(path) -> text, backed by whichever local Whisper
    implementation is installed.

    engine: "auto" tries faster-whisper first, then openai-whisper.
            "faster-whisper" or "whisper" forces that one specifically --
            useful on Intel Macs, where faster-whisper's native backend
            (ctranslate2) sometimes segfaults due to a duplicate-OpenMP
            conflict with PyTorch/numpy. If that happens, either set
            KMP_DUPLICATE_LIB_OK=TRUE before running, or pass
            --engine whisper to use the pure-PyTorch implementation instead.
    """
    if engine in ("auto", "faster-whisper"):
        try:
            from faster_whisper import WhisperModel

            model = WhisperModel(model_size, device="cpu", compute_type="int8")

            def transcribe(path):
                segments, _ = model.transcribe(str(path))
                return " ".join(seg.text.strip() for seg in segments)

            print(f"  using faster-whisper (model: {model_size})")
            return transcribe
        except ImportError:
            if engine == "faster-whisper":
                sys.exit(
                    "faster-whisper is not installed. Run:\n"
                    "    pip install faster-whisper --break-system-packages"
                )

    if engine in ("auto", "whisper"):
        try:
            import whisper

            model = whisper.load_model(model_size)

            def transcribe(path):
                return model.transcribe(str(path))["text"].strip()

            print(f"  using openai-whisper (model: {model_size})")
            return transcribe
        except ImportError:
            if engine == "whisper":
                sys.exit(
                    "openai-whisper is not installed. Run:\n"
                    "    pip install openai-whisper --break-system-packages\n"
                    "(also needs ffmpeg: brew install ffmpeg)"
                )

    sys.exit(
        "No local Whisper install found. Run:\n"
        "    pip install faster-whisper --break-system-packages\n"
        "(or openai-whisper) and re-run this script."
    )


# ------------------------------------------------------------------ photos

def photo_exif_datetime(path: Path):
    """Best-effort EXIF capture time; falls back to file mtime."""
    try:
        from PIL import Image, ExifTags

        if path.suffix.lower() in (".heic", ".heif"):
            try:
                import pillow_heif

                pillow_heif.register_heif_opener()
            except ImportError:
                pass

        img = Image.open(path)
        exif = img.getexif()
        for tag_id, value in exif.items():
            tag = ExifTags.TAGS.get(tag_id)
            if tag == "DateTime":
                return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return datetime.fromtimestamp(path.stat().st_mtime)


def index_photos(photos_dir):
    """date -> sorted list of (time_str, filename)"""
    by_date = defaultdict(list)
    if not photos_dir:
        return by_date
    p = Path(photos_dir)
    if not p.exists():
        print(f"  (no photos folder at {p}, skipping)")
        return by_date
    for f in sorted(p.iterdir()):
        if f.suffix.lower() in PHOTO_EXTS:
            dt = photo_exif_datetime(f)
            by_date[dt.date()].append((dt.strftime("%H:%M"), f.name))
    for d in by_date:
        by_date[d].sort()
    return by_date


# --------------------------------------------------------------- markdown

def day_heading(date, itinerary):
    row = itinerary.get(date.isoformat())
    if not row:
        return f"# {date:%A, %d %B %Y}"
    loc = row.get("camp") or row.get("location") or ""
    place = row.get("location", "")
    heading = f"# {date:%A, %d %B %Y} -- {place}"
    if row.get("camp"):
        heading += f" ({row['camp']})"
    if row.get("note"):
        heading += f"\n\n*{row['note']}*"
    return heading


def build_day_markdown(date, recordings, photos, itinerary):
    lines = [day_heading(date, itinerary), ""]
    for time_label, text in recordings:
        lines.append(f"## Evening recap ({time_label})")
        lines.append("")
        lines.append(text if text else "*(empty transcript)*")
        lines.append("")
    if photos:
        lines.append("## Photos from today")
        lines.append("")
        for time_str, name in photos:
            lines.append(f"- `{time_str}` {name}")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--memos", required=True, help="folder of voice memo audio files")
    ap.add_argument("--photos", default=None, help="folder of photos (optional)")
    ap.add_argument("--itinerary", default="itinerary.csv", help="date/location lookup CSV")
    ap.add_argument("--out", default="notes", help="output folder for markdown")
    ap.add_argument("--model", default="small",
                     help="Whisper model size: tiny, base, small, medium, large (default: small)")
    ap.add_argument("--engine", default="auto", choices=["auto", "faster-whisper", "whisper"],
                     help="Whisper backend to use. Default 'auto' tries faster-whisper then "
                          "openai-whisper. On Intel Mac, if faster-whisper segfaults, pass "
                          "'whisper' to force the pure-PyTorch implementation.")
    args = ap.parse_args()

    memos_dir = Path(args.memos)
    if not memos_dir.exists():
        sys.exit(f"Memos folder not found: {memos_dir}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading itinerary...")
    itinerary = load_itinerary(args.itinerary)

    print("Indexing photos...")
    photos_by_date = index_photos(args.photos)

    audio_files = sorted(
        f for f in memos_dir.iterdir() if f.suffix.lower() in AUDIO_EXTS
    )
    if not audio_files:
        sys.exit(f"No audio files found in {memos_dir}")

    print(f"Found {len(audio_files)} voice memo(s). Loading Whisper model...")
    transcribe = get_transcriber(args.model, args.engine)

    recordings_by_date = defaultdict(list)
    for f in audio_files:
        date = recording_date(f)
        time_label = datetime.fromtimestamp(f.stat().st_mtime).strftime("%H:%M")
        print(f"  transcribing {f.name}  ->  {date}")
        text = transcribe(f)
        recordings_by_date[date].append((time_label, text))

    all_dates = sorted(set(recordings_by_date) | set(photos_by_date))
    journal_parts = []

    for date in all_dates:
        recs = sorted(recordings_by_date.get(date, []))
        photos = photos_by_date.get(date, [])
        md = build_day_markdown(date, recs, photos, itinerary)
        day_path = out_dir / f"{date.isoformat()}.md"
        day_path.write_text(md, encoding="utf-8")
        journal_parts.append(md)
        print(f"  wrote {day_path}")

    journal_path = out_dir / "trip-journal.md"
    journal_path.write_text("\n\n---\n\n".join(journal_parts), encoding="utf-8")
    print(f"\nDone. {len(all_dates)} day(s) written to {out_dir}/")
    print(f"Combined journal: {journal_path}")


if __name__ == "__main__":
    main()
