#!/bin/bash
#
# Retrieve SG_NAVPOSJ2000 mnemonics over a time range and downsample to 30 s.
#
# Usage:
#   ./retrieve_navpos.sh -s START -d DURATION -o OUTFILE
#
#   -s  Start date/time, e.g. '2025-12-01 00:00:00'
#   -d  Duration relative to start, e.g. '+210 days'
#   -o  Output CSV (the 30 s downsampled file)
#
set -euo pipefail

usage() {
    echo "Usage: $0 -s START -d DURATION -o OUTFILE" >&2
    echo "  -s  Start date/time, e.g. '2025-12-01 00:00:00'" >&2
    echo "  -d  Duration, e.g. '+210 days'" >&2
    echo "  -o  Output CSV filename (30 s downsampled)" >&2
    exit 1
}

START=""
DURATION=""
OUTFILE=""

while getopts "s:d:o:h" opt; do
    case "$opt" in
        s) START="$OPTARG" ;;
        d) DURATION="$OPTARG" ;;
        o) OUTFILE="$OPTARG" ;;
        h|*) usage ;;
    esac
done

if [[ -z "$START" || -z "$DURATION" || -z "$OUTFILE" ]]; then
    echo "ERROR: -s, -d and -o are all required." >&2
    usage
fi

# Intermediate 1 s file derived from the output name (foo.csv -> foo_1s.csv)
BASE="${OUTFILE%.csv}"
RAWFILE="${BASE}_1s.csv"

# 1. Start the apptainer / set up the ISOC PROD environment.
#    Sourced so the environment persists for the commands below.
source /sdf/data/fermi/a/isoc/s3df/bin/isoc.sh PROD

# 2. Retrieve the mnemonics.
MnemRet.py -b "$START" -e "$DURATION" --csv "$RAWFILE" \
    SG_NAVPOSJ2000_1 SG_NAVPOSJ2000_2 SG_NAVPOSJ2000_3

# 3. Downsample: keep rows at whole and half minutes (:00. and :30.).
perl -ne 'print if (/:00\./ or /:30\./)' "$RAWFILE" > "$OUTFILE"

echo "Done. Raw: $RAWFILE  Downsampled: $OUTFILE"
