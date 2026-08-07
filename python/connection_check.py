#!/usr/bin/env python3
"""
connection_check.py -- how much time does a connection *actually* leave you?

Pulls recent operating history for a flight (and, optionally, the flight you are
connecting to) from FlightAware's AeroAPI v4, pairs the legs up by date, and
reports the real gate-in -> gate-out interval for each day alongside the
scheduled one.

Why this is the right quantity: the published connection time is
scheduled_in(inbound) -> scheduled_out(outbound). What you actually get on the
day is actual_in(inbound) -> actual_out(outbound), and the useful part of that
is smaller still, because the outbound closes its door some minutes before
pushback. The tool reports all three.


THE NORMAL WAY TO USE THIS: BY HAND, FROM A CSV
===============================================
No account, no API key, no billing. Ten minutes of typing gets you everything
the paid path gets except per-day gate numbers.

Step 1. Make a blank sheet. It will refuse to clobber an existing file, so you
        can run this without fear:

    python connection_check.py --inbound LH455 --outbound LH590 \
        --write-csv-template mytrip.csv

Step 2. Open these two pages and read the times off them. Note the ICAO form of
        the flight number -- DLH455, not LH455. That trips everyone up:

    https://www.flightaware.com/live/flight/DLH455/history
    https://www.flightaware.com/live/flight/DLH590/history

        Flightera is a good second source and sometimes shows more history:

    https://www.flightera.net/en/flight/LH455
    https://www.flightera.net/en/flight/LH590

        For each day you want, you need four times: the inbound's scheduled and
        actual ARRIVAL, and the outbound's scheduled and actual DEPARTURE. Use
        the gate/block times ("Gate Arrival", "Gate Departure") rather than
        takeoff and landing, if the page distinguishes them.

Step 3. One row per day. Times are HH:MM in the connecting airport's local
        time -- they only ever get differenced, so the zone cancels out and you
        don't need to think about CEST vs EAT. Blank cells are fine. Suffix a
        time with +1 if it falls after midnight. A filled sheet looks like:

    date,sched_in,actual_in,sched_out,actual_out,gate_in,gate_out
    2026-07-28,10:25,10:53,11:25,11:44,Z52,Z55
    2026-07-29,10:25,10:32,11:25,11:34,Z52,Z55

        The gate columns are optional -- leave them blank if the page doesn't
        show them. Everything else still works.

Step 4. Analyse and plot:

    python connection_check.py --inbound LH455 --outbound LH590 --via FRA \
        --csv mytrip.csv --plot fra.html

Then append new rows as the days roll by and re-run; it is cheap and the sample
only gets better.

Why these numbers and not the published connection time
-------------------------------------------------------
The advertised connection is scheduled_in(inbound) -> scheduled_out(outbound).
What you actually get on the day is actual_in -> actual_out, and the usable part
is smaller still, because the outbound's door shuts some minutes before
pushback (--door-close, default 20). The tool reports all three, and flags any
day that fell below the airport's minimum connecting time.

Read the tail, not the average. A connection that is comfortable nine days in
ten and impossible on the tenth is a bad connection. Recombine the extremes by
hand too: the worst inbound arrival you observed meeting the most punctual
outbound departure you observed is a plausible bad day even if those two never
coincided in your sample.

The optional API path
---------------------
AeroAPI v4, GET /flights/{ident}, needs a key and gives you real gate
assignments plus live same-day tracking. The free "Personal" tier is ample
(this tool costs 1-2 queries per run) but the signup is genuinely confusing:
there is no "generate key" button, and the developer portal is documentation
only. The key is created as a side effect of subscribing to a tier. The $5 is a
monthly credit against usage billing, not a hard cap, so expect to be asked for
a payment method. Also note the 10-day lookback on non-history endpoints;
reaching further back needs the History endpoints and a $200/month Standard
subscription, which is not worth it for personal travel.

    https://www.flightaware.com/aeroapi/portal/
    https://support.flightaware.com/hc/en-us/sections/32586090657175-AeroAPI

    export AEROAPI_KEY=...
    python connection_check.py --inbound LH455 --outbound LH590 --plot fra.html
    python connection_check.py --inbound LH455          # punctuality only

Responses are cached under ~/.cache/connection_check, so re-plotting is free.

    # synthetic data, to check the code runs; the numbers are FICTION
    python connection_check.py --inbound LH455 --outbound LH590 --demo

Requires: bokeh, numpy (stdlib otherwise).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

BASE_URL = "https://aeroapi.flightaware.com/aeroapi"
MAX_LOOKBACK_DAYS = 10  # AeroAPI non-history endpoint limit

# AeroAPI keys flights by ICAO ident. IATA usually resolves, but not always,
# so try the ICAO form first for airlines we know about.
IATA_TO_ICAO = {
    "LH": "DLH", "KQ": "KQA", "LX": "SWR", "OS": "AUA", "SN": "BEL",
    "UA": "UAL", "AA": "AAL", "DL": "DAL", "BA": "BAW", "AF": "AFR",
    "KL": "KLM", "EK": "UAE", "QR": "QTR", "ET": "ETH", "TK": "THY",
    "SA": "SAA", "4Z": "LNK", "PW": "PRF", "WB": "RWD",
}

# Published minimum connecting times, international-to-international, minutes.
# Indicative only -- the airline's own MCT is what a booking engine enforces.
MCT_MINUTES = {
    "EDDF": 45, "FRA": 45,     # Frankfurt
    "EDDM": 35, "MUC": 35,     # Munich
    "EHAM": 50, "AMS": 50,     # Amsterdam
    "LFPG": 60, "CDG": 60,     # Paris CDG
    "EGLL": 75, "LHR": 75,     # Heathrow
    "LTFM": 60, "IST": 60,     # Istanbul
    "OMDB": 60, "DXB": 60,     # Dubai
    "HKJK": 45, "NBO": 45,     # Nairobi
    "FAOR": 60, "JNB": 60,     # Johannesburg
}
DEFAULT_MCT = 45

# Fallback so an IATA code on the command line still matches ICAO-only data.
ICAO_TO_IATA = {
    "EDDF": "FRA", "EDDM": "MUC", "EHAM": "AMS", "LFPG": "CDG",
    "EGLL": "LHR", "LTFM": "IST", "OMDB": "DXB", "HKJK": "NBO",
    "FAOR": "JNB", "KSFO": "SFO", "HTKJ": "JRO", "FVFA": "VFA",
    "FBSK": "GBE", "FBMN": "MUB", "LSZH": "ZRH", "LOWW": "VIE",
    "EBBR": "BRU", "HKNW": "WIL",
}

# Doors typically close this many minutes before scheduled pushback, so this
# much of the nominal window is not usable for walking to the gate.
DEFAULT_DOOR_CLOSE_MIN = 20


# --------------------------------------------------------------------------- #
# data model
# --------------------------------------------------------------------------- #

def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    """AeroAPI timestamps are ISO-8601 UTC, e.g. 2026-08-07T10:25:00Z."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass
class Leg:
    """One operation of a flight number on one day."""
    ident: str
    fa_flight_id: str = ""
    origin: str = ""
    destination: str = ""
    origin_iata: str = ""
    destination_iata: str = ""
    scheduled_out: Optional[datetime] = None
    actual_out: Optional[datetime] = None
    scheduled_in: Optional[datetime] = None
    actual_in: Optional[datetime] = None
    scheduled_on: Optional[datetime] = None   # scheduled landing
    actual_on: Optional[datetime] = None      # actual landing
    gate_origin: str = ""
    gate_destination: str = ""
    terminal_origin: str = ""
    terminal_destination: str = ""
    status: str = ""
    cancelled: bool = False
    diverted: bool = False

    @classmethod
    def from_api(cls, d: dict) -> "Leg":
        def apt(key: str) -> str:
            node = d.get(key) or {}
            return node.get("code_icao") or node.get("code") or ""

        def apt_iata(key: str) -> str:
            node = d.get(key) or {}
            return node.get("code_iata") or ""

        return cls(
            ident=d.get("ident") or "",
            fa_flight_id=d.get("fa_flight_id") or "",
            origin=apt("origin"),
            destination=apt("destination"),
            origin_iata=apt_iata("origin"),
            destination_iata=apt_iata("destination"),
            scheduled_out=_parse_ts(d.get("scheduled_out")),
            actual_out=_parse_ts(d.get("actual_out")),
            scheduled_in=_parse_ts(d.get("scheduled_in")),
            actual_in=_parse_ts(d.get("actual_in")),
            scheduled_on=_parse_ts(d.get("scheduled_on")),
            actual_on=_parse_ts(d.get("actual_on")),
            gate_origin=d.get("gate_origin") or "",
            gate_destination=d.get("gate_destination") or "",
            terminal_origin=d.get("terminal_origin") or "",
            terminal_destination=d.get("terminal_destination") or "",
            status=d.get("status") or "",
            cancelled=bool(d.get("cancelled")),
            diverted=bool(d.get("diverted")),
        )

    @property
    def date(self) -> Optional[str]:
        ref = self.scheduled_out or self.actual_out
        return ref.date().isoformat() if ref else None

    @property
    def arrival_delay_min(self) -> Optional[float]:
        """Gate arrival vs schedule, minutes. Negative = early."""
        if self.actual_in and self.scheduled_in:
            return (self.actual_in - self.scheduled_in).total_seconds() / 60.0
        return None

    @property
    def departure_delay_min(self) -> Optional[float]:
        if self.actual_out and self.scheduled_out:
            return (self.actual_out - self.scheduled_out).total_seconds() / 60.0
        return None

    @property
    def usable(self) -> bool:
        """Did this leg actually operate and record both block times?"""
        return not (self.cancelled or self.diverted)


@dataclass
class Connection:
    """An inbound leg paired with the outbound leg it feeds, on one day."""
    date: str
    airport: str
    inbound: Leg
    outbound: Leg

    @property
    def scheduled_gap_min(self) -> Optional[float]:
        if self.inbound.scheduled_in and self.outbound.scheduled_out:
            return (self.outbound.scheduled_out
                    - self.inbound.scheduled_in).total_seconds() / 60.0
        return None

    @property
    def actual_gap_min(self) -> Optional[float]:
        """Real gate-in to real pushback. The honest number."""
        if self.inbound.actual_in and self.outbound.actual_out:
            return (self.outbound.actual_out
                    - self.inbound.actual_in).total_seconds() / 60.0
        return None

    def usable_gap_min(self, door_close: int = DEFAULT_DOOR_CLOSE_MIN
                       ) -> Optional[float]:
        """Time you had to get off, walk, and be at the door before it shut."""
        gap = self.actual_gap_min
        return None if gap is None else gap - door_close

    @property
    def gates(self) -> str:
        a = self.inbound.gate_destination or "?"
        b = self.outbound.gate_origin or "?"
        ta = self.inbound.terminal_destination
        tb = self.outbound.terminal_origin
        left = f"{ta}/{a}" if ta else a
        right = f"{tb}/{b}" if tb else b
        return f"{left} -> {right}"

    @property
    def missed(self) -> bool:
        """Outbound pushed back before the inbound reached its gate."""
        gap = self.actual_gap_min
        return gap is not None and gap <= 0


# --------------------------------------------------------------------------- #
# AeroAPI access
# --------------------------------------------------------------------------- #

class AeroAPIError(RuntimeError):
    pass


def normalize_idents(ident: str) -> list[str]:
    """Candidate idents to try, ICAO form first."""
    ident = ident.strip().upper().replace(" ", "")
    out = []
    for n in (2, 3):
        prefix, rest = ident[:n], ident[n:]
        if rest.isdigit() and prefix in IATA_TO_ICAO:
            out.append(IATA_TO_ICAO[prefix] + rest)
    if ident not in out:
        out.append(ident)
    return out


def _get(path: str, params: dict, api_key: str) -> dict:
    url = f"{BASE_URL}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url, headers={"x-apikey": api_key, "Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")[:400]
        if exc.code == 401:
            raise AeroAPIError("401 -- AEROAPI_KEY missing or rejected") from exc
        if exc.code == 429:
            raise AeroAPIError(
                "429 -- rate limited. The Personal tier throttles bursts; "
                "wait a few seconds between calls."
            ) from exc
        raise AeroAPIError(f"HTTP {exc.code} on {path}: {body}") from exc
    except urllib.error.URLError as exc:
        raise AeroAPIError(f"network error on {path}: {exc.reason}") from exc


def fetch_legs(ident: str, days: int, api_key: str,
               cache_dir: Optional[Path] = None,
               max_pages: int = 2) -> list[Leg]:
    """Recent operations of `ident`, most recent last."""
    days = min(days, MAX_LOOKBACK_DAYS)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    params = {
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "max_pages": max_pages,
    }

    cache_file = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        stamp = end.strftime("%Y%m%dT%H")  # one cache slot per hour
        cache_file = cache_dir / f"{ident.upper()}_{days}d_{stamp}.json"
        if cache_file.exists():
            payload = json.loads(cache_file.read_text())
            return [Leg.from_api(f) for f in payload.get("flights", [])]

    last_error = None
    for candidate in normalize_idents(ident):
        try:
            payload = _get(f"/flights/{urllib.parse.quote(candidate)}",
                           params, api_key)
        except AeroAPIError as exc:
            last_error = exc
            continue
        flights = payload.get("flights", [])
        if flights:
            if cache_file:
                cache_file.write_text(json.dumps(payload, indent=1))
            return [Leg.from_api(f) for f in flights]

    if last_error:
        raise last_error
    return []


# --------------------------------------------------------------------------- #
# pairing and stats
# --------------------------------------------------------------------------- #

def _codes(icao: str, iata: str) -> set[str]:
    """Every code that legitimately names this airport."""
    out = {c for c in (icao.upper(), iata.upper()) if c}
    if icao.upper() in ICAO_TO_IATA:
        out.add(ICAO_TO_IATA[icao.upper()])
    return out


def filter_route(legs: Sequence[Leg], origin: str = "",
                 destination: str = "") -> list[Leg]:
    """Flight numbers get reused across routes; keep only the one we mean."""
    def ok(leg: Leg) -> bool:
        if origin and origin.upper() not in _codes(leg.origin, leg.origin_iata):
            return False
        if destination and destination.upper() not in _codes(
                leg.destination, leg.destination_iata):
            return False
        return True
    return [l for l in legs if ok(l)]


def pair_connections(inbound: Sequence[Leg], outbound: Sequence[Leg],
                     window_hours: float = 8.0) -> list[Connection]:
    """
    Match each inbound leg to the outbound leg it feeds: the first outbound
    departing the same airport after the inbound's scheduled arrival, within
    `window_hours`.
    """
    out_sorted = sorted(
        (l for l in outbound if l.scheduled_out),
        key=lambda l: l.scheduled_out,
    )
    used: set[str] = set()
    pairs: list[Connection] = []

    for inb in sorted((l for l in inbound if l.scheduled_in),
                      key=lambda l: l.scheduled_in):
        best = None
        for out in out_sorted:
            if out.fa_flight_id in used:
                continue
            if inb.destination and out.origin and inb.destination != out.origin:
                continue
            dt = (out.scheduled_out - inb.scheduled_in).total_seconds() / 3600.0
            if -1.0 <= dt <= window_hours:
                best = out
                break
        if best is None:
            continue
        used.add(best.fa_flight_id)
        pairs.append(Connection(
            date=inb.date or "?",
            airport=inb.destination or best.origin,
            inbound=inb,
            outbound=best,
        ))
    return pairs


def _pct(values: Sequence[float], q: float) -> float:
    """Simple linear-interpolation percentile; q in [0, 100]."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q / 100.0
    lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def summarize_connections(pairs: Sequence[Connection], mct: int,
                          door_close: int) -> dict:
    gaps = [p.actual_gap_min for p in pairs if p.actual_gap_min is not None]
    sched = [p.scheduled_gap_min for p in pairs
             if p.scheduled_gap_min is not None]
    usable = [g - door_close for g in gaps]
    return {
        "n": len(gaps),
        "scheduled_gap": statistics.median(sched) if sched else float("nan"),
        "median": statistics.median(gaps) if gaps else float("nan"),
        "mean": statistics.fmean(gaps) if gaps else float("nan"),
        "min": min(gaps) if gaps else float("nan"),
        "max": max(gaps) if gaps else float("nan"),
        "p10": _pct(gaps, 10),
        "p90": _pct(gaps, 90),
        "n_below_mct": sum(1 for g in gaps if g < mct),
        "n_missed": sum(1 for g in gaps if g <= 0),
        "n_usable_under_20": sum(1 for u in usable if u < 20),
        "mct": mct,
        "door_close": door_close,
    }


def summarize_delays(legs: Sequence[Leg]) -> dict:
    d = [l.arrival_delay_min for l in legs if l.arrival_delay_min is not None]
    return {
        "n": len(d),
        "median": statistics.median(d) if d else float("nan"),
        "mean": statistics.fmean(d) if d else float("nan"),
        "min": min(d) if d else float("nan"),
        "max": max(d) if d else float("nan"),
        "p90": _pct(d, 90),
        "on_time_15": (sum(1 for x in d if x <= 15) / len(d) * 100) if d else
                      float("nan"),
    }


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def print_connection_table(pairs: Sequence[Connection], door_close: int,
                           mct: int) -> None:
    hdr = (f"{'date':<11} {'in sched':<9} {'in act':<9} {'out sched':<10} "
           f"{'out act':<9} {'gap':>6} {'usable':>7}  gates")
    print(hdr)
    print("-" * len(hdr))
    for p in pairs:
        def hm(t: Optional[datetime]) -> str:
            return t.strftime("%H:%M") if t else "--:--"
        gap = p.actual_gap_min
        gap_s = f"{gap:6.0f}" if gap is not None else "     ?"
        use = p.usable_gap_min(door_close)
        use_s = f"{use:7.0f}" if use is not None else "      ?"
        flag = ""
        if gap is not None and gap <= 0:
            flag = "  <-- MISSED"
        elif gap is not None and gap < mct:
            flag = "  <-- under MCT"
        print(f"{p.date:<11} {hm(p.inbound.scheduled_in):<9} "
              f"{hm(p.inbound.actual_in):<9} "
              f"{hm(p.outbound.scheduled_out):<10} "
              f"{hm(p.outbound.actual_out):<9} {gap_s} {use_s}  "
              f"{p.gates}{flag}")


def print_connection_summary(s: dict, inbound: str, outbound: str,
                             airport: str) -> None:
    n = s["n"]
    print()
    print(f"{inbound} -> {outbound} at {airport}: {n} observed connections")
    if not n:
        return
    print(f"  scheduled gap (median)      {s['scheduled_gap']:.0f} min")
    print(f"  actual gate-in -> pushback  median {s['median']:.0f} min, "
          f"range {s['min']:.0f} to {s['max']:.0f}, "
          f"10th pct {s['p10']:.0f}")
    print(f"  minus {s['door_close']}-min door close   median usable "
          f"{s['median'] - s['door_close']:.0f} min")
    print(f"  below {s['mct']}-min MCT            {s['n_below_mct']} of {n}")
    print(f"  under 20 min usable         {s['n_usable_under_20']} of {n}")
    print(f"  inbound arrived after out.  {s['n_missed']} of {n}")


def print_delay_summary(s: dict, ident: str) -> None:
    print()
    print(f"{ident} arrival punctuality: {s['n']} legs")
    if not s["n"]:
        return
    print(f"  arrival delay median {s['median']:+.0f} min, "
          f"mean {s['mean']:+.0f}, range {s['min']:+.0f} to {s['max']:+.0f}")
    print(f"  90th percentile      {s['p90']:+.0f} min")
    print(f"  within 15 min        {s['on_time_15']:.0f}%")


# --------------------------------------------------------------------------- #
# bokeh output
# --------------------------------------------------------------------------- #

def make_plot(pairs: Sequence[Connection], legs: Sequence[Leg], path: str,
              title: str, mct: int, door_close: int) -> str:
    from bokeh.layouts import column
    from bokeh.models import (ColumnDataSource, DataTable, Div, HoverTool,
                              NumberFormatter, Span, TableColumn)
    from bokeh.plotting import figure, output_file, save

    output_file(path, title=title, mode="cdn")
    panels = [Div(text=f"<h2 style='font-family:sans-serif'>{title}</h2>",
                  width=900)]

    if pairs:
        # Categorical x-ranges must be unique; a flight can run twice a day.
        labels, seen = [], {}
        for p in pairs:
            seen[p.date] = seen.get(p.date, 0) + 1
            labels.append(p.date if seen[p.date] == 1
                          else f"{p.date} ({seen[p.date]})")

        src = ColumnDataSource(dict(
            date=labels,
            x=list(range(len(pairs))),
            actual=[p.actual_gap_min if p.actual_gap_min is not None
                    else float("nan") for p in pairs],
            sched=[p.scheduled_gap_min if p.scheduled_gap_min is not None
                   else float("nan") for p in pairs],
            usable=[(p.usable_gap_min(door_close)
                     if p.usable_gap_min(door_close) is not None
                     else float("nan")) for p in pairs],
            gates=[p.gates for p in pairs],
            in_delay=[p.inbound.arrival_delay_min or 0.0 for p in pairs],
            out_delay=[p.outbound.departure_delay_min or 0.0 for p in pairs],
            colour=["#c0392b" if (p.actual_gap_min or 0) < mct else "#1a7f5a"
                    for p in pairs],
        ))

        f1 = figure(width=900, height=380,
                    title="Connection time actually available, by day",
                    x_axis_label="", y_axis_label="minutes",
                    x_range=labels,
                    tools="pan,wheel_zoom,box_zoom,reset,save")
        f1.vbar(x="date", top="actual", width=0.55, source=src,
                fill_color="colour", line_color=None, alpha=0.85,
                legend_label="actual gate-in -> pushback")
        f1.scatter(x="date", y="sched", source=src, size=11, marker="dash",
                   line_color="#333", line_width=3,
                   legend_label="scheduled gap")
        f1.scatter(x="date", y="usable", source=src, size=9, marker="circle",
                   fill_color="white", line_color="#333",
                   legend_label=f"usable (minus {door_close} min door close)")
        f1.add_layout(Span(location=mct, dimension="width",
                           line_color="#b8860b", line_dash="dashed",
                           line_width=2))
        f1.add_tools(HoverTool(tooltips=[
            ("date", "@date"),
            ("actual gap", "@actual{0} min"),
            ("scheduled gap", "@sched{0} min"),
            ("usable", "@usable{0} min"),
            ("inbound arr delay", "@in_delay{+0} min"),
            ("outbound dep delay", "@out_delay{+0} min"),
            ("gates", "@gates"),
        ]))
        f1.xaxis.major_label_orientation = 0.8
        f1.legend.location = "top_left"
        f1.legend.label_text_font_size = "9pt"
        finite_usable = [v for v in src.data["usable"] if v == v]
        if finite_usable:
            f1.y_range.start = min(0, min(finite_usable) - 5)
        panels.append(f1)

        gaps = sorted(v for v in src.data["actual"] if v == v)
        if len(gaps) > 1:
            f2 = figure(width=900, height=300,
                        title="Empirical distribution of available "
                              "connection time",
                        x_axis_label="minutes available",
                        y_axis_label="fraction of days at or below",
                        tools="pan,wheel_zoom,box_zoom,reset,save")
            frac = [(i + 1) / len(gaps) for i in range(len(gaps))]
            f2.step(gaps, frac, line_width=2, mode="after",
                    line_color="#1a5276")
            f2.scatter(gaps, frac, size=7, fill_color="#1a5276",
                       line_color=None)
            f2.add_layout(Span(location=mct, dimension="height",
                               line_color="#b8860b", line_dash="dashed",
                               line_width=2))
            f2.add_tools(HoverTool(tooltips=[("minutes", "$x{0}"),
                                             ("fraction", "$y{0.00}")],
                                   mode="vline"))
            panels.append(f2)

        cols = [
            TableColumn(field="date", title="date", width=90),
            TableColumn(field="sched", title="sched gap (min)",
                        formatter=NumberFormatter(format="0")),
            TableColumn(field="actual", title="actual gap (min)",
                        formatter=NumberFormatter(format="0")),
            TableColumn(field="usable", title="usable (min)",
                        formatter=NumberFormatter(format="0")),
            TableColumn(field="in_delay", title="inb arr delay",
                        formatter=NumberFormatter(format="+0")),
            TableColumn(field="out_delay", title="outb dep delay",
                        formatter=NumberFormatter(format="+0")),
            TableColumn(field="gates", title="gates", width=180),
        ]
        panels.append(DataTable(source=src, columns=cols, width=900,
                                height=min(320, 30 + 26 * len(pairs)),
                                index_position=None))

    delays = [l.arrival_delay_min for l in legs
              if l.arrival_delay_min is not None]
    if len(delays) > 1:
        import numpy as np
        lo = min(min(delays), -20)
        hi = max(max(delays), 40)
        edges = np.linspace(lo, hi, 13)
        counts, edges = np.histogram(delays, bins=edges)
        f3 = figure(width=900, height=280,
                    title="Inbound arrival delay (gate), minutes",
                    x_axis_label="minutes late (negative = early)",
                    y_axis_label="days",
                    tools="pan,wheel_zoom,box_zoom,reset,save")
        f3.quad(top=counts, bottom=0, left=edges[:-1], right=edges[1:],
                fill_color="#5b8fa8", line_color="white")
        f3.add_layout(Span(location=0, dimension="height",
                           line_color="#333", line_dash="dotted"))
        panels.append(f3)

    save(column(*panels))
    return path


# --------------------------------------------------------------------------- #
# CSV input, for when you have the numbers but not an API key
# --------------------------------------------------------------------------- #

CSV_TEMPLATE = """\
# Read the times off the flight's history page. Use the ICAO form of the
# flight number -- DLH455, not LH455. That trips everyone up:
#
#   https://www.flightaware.com/live/flight/DLH455/history   <- inbound
#   https://www.flightaware.com/live/flight/DLH590/history   <- outbound
#   https://www.flightera.net/en/flight/LH455                <- second source
#
# One row per day. You need the inbound's scheduled and actual ARRIVAL, and the
# outbound's scheduled and actual DEPARTURE. Prefer gate/block times over
# takeoff and landing where the page distinguishes them.
#
# Times are HH:MM in the connecting airport's local time -- they only ever get
# differenced, so the zone cancels and you needn't convert anything. Leave a
# cell blank if you don't have it; the gate columns are entirely optional.
# Suffix a time with +1 if it falls after midnight.
#
# Replace the example row below with your own.
date,sched_in,actual_in,sched_out,actual_out,gate_in,gate_out
2026-07-28,10:25,10:41,11:25,11:38,Z52,Z55
"""


def _csv_time(day: str, hhmm: str) -> Optional[datetime]:
    hhmm = (hhmm or "").strip()
    if not hhmm:
        return None
    plus_day = 0
    if hhmm.endswith("+1"):
        plus_day, hhmm = 1, hhmm[:-2].strip()
    try:
        base = datetime.strptime(f"{day} {hhmm}", "%Y-%m-%d %H:%M")
    except ValueError:
        return None
    return base.replace(tzinfo=timezone.utc) + timedelta(days=plus_day)


def load_csv(path: str, inbound: str, outbound: str,
             via: str = "FRA") -> tuple[list[Leg], list[Leg]]:
    """Read hand-entered history. See CSV_TEMPLATE for the format."""
    import csv as _csv

    inb, out = [], []
    with open(path, newline="") as fh:
        rows = [r for r in fh if r.strip() and not r.lstrip().startswith("#")]
    for i, row in enumerate(_csv.DictReader(rows)):
        day = (row.get("date") or "").strip()
        if not day:
            continue
        via_u = via.upper()
        inb.append(Leg(
            ident=inbound, fa_flight_id=f"csv-in-{i}",
            destination=via_u, destination_iata=via_u,
            scheduled_in=_csv_time(day, row.get("sched_in", "")),
            actual_in=_csv_time(day, row.get("actual_in", "")),
            scheduled_out=_csv_time(day, row.get("sched_in", "")),
            gate_destination=(row.get("gate_in") or "").strip(),
            status="Arrived",
        ))
        out.append(Leg(
            ident=outbound, fa_flight_id=f"csv-out-{i}",
            origin=via_u, origin_iata=via_u,
            scheduled_out=_csv_time(day, row.get("sched_out", "")),
            actual_out=_csv_time(day, row.get("actual_out", "")),
            gate_origin=(row.get("gate_out") or "").strip(),
            status="Arrived",
        ))
    return inb, out


# --------------------------------------------------------------------------- #
# synthetic data, for testing without a key
# --------------------------------------------------------------------------- #

def demo_legs(inbound: str, outbound: str, days: int = 10, seed: int = 7):
    """Plausible fake history: long-haul arrival delays are right-skewed."""
    rng = random.Random(seed)
    base = datetime.now(timezone.utc).replace(
        hour=8, minute=25, second=0, microsecond=0) - timedelta(days=days)
    inb, out = [], []
    for i in range(days):
        day = base + timedelta(days=i)
        sched_in = day
        arr_delay = max(-18, rng.gauss(9, 14) + rng.expovariate(1 / 12) - 8)
        sched_out = day + timedelta(minutes=60)
        dep_delay = max(-4, rng.gauss(6, 9))
        # a badly late inbound tends to drag the outbound a little
        if arr_delay > 45:
            dep_delay += min(30, 0.4 * (arr_delay - 45))
        inb.append(Leg(
            ident=inbound, fa_flight_id=f"demo-in-{i}",
            origin="KSFO", destination="EDDF",
            origin_iata="SFO", destination_iata="FRA",
            scheduled_out=day - timedelta(hours=11, minutes=45),
            actual_out=day - timedelta(hours=11, minutes=41),
            scheduled_in=sched_in,
            actual_in=sched_in + timedelta(minutes=arr_delay),
            gate_destination=rng.choice(["Z50", "Z52", "Z54", "A44", "Z25"]),
            terminal_destination="1", status="Arrived",
        ))
        out.append(Leg(
            ident=outbound, fa_flight_id=f"demo-out-{i}",
            origin="EDDF", destination="HKJK",
            origin_iata="FRA", destination_iata="NBO",
            scheduled_out=sched_out,
            actual_out=sched_out + timedelta(minutes=dep_delay),
            scheduled_in=sched_out + timedelta(hours=8, minutes=10),
            actual_in=sched_out + timedelta(hours=8, minutes=10 + dep_delay),
            gate_origin=rng.choice(["Z55", "Z57", "Z59", "A50"]),
            terminal_origin="1", status="Arrived",
        ))
    return inb, out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Compare a flight's actual arrival against the departure "
                    "of the flight you are connecting to.")
    p.add_argument("--inbound", required=True,
                   help="inbound flight number, e.g. LH455")
    p.add_argument("--outbound", default=None,
                   help="connecting flight number, e.g. LH590")
    p.add_argument("--days", type=int, default=MAX_LOOKBACK_DAYS,
                   help=f"lookback, max {MAX_LOOKBACK_DAYS} on non-history "
                        "endpoints (default: %(default)s)")
    p.add_argument("--from", dest="origin", default="",
                   help="restrict inbound to this origin (IATA or ICAO)")
    p.add_argument("--via", default="",
                   help="the connecting airport, if the flight number is "
                        "used on more than one route")
    p.add_argument("--to", dest="final", default="",
                   help="restrict outbound to this destination")
    p.add_argument("--mct", type=int, default=None,
                   help="minimum connecting time in minutes "
                        "(default: looked up per airport)")
    p.add_argument("--door-close", type=int, default=DEFAULT_DOOR_CLOSE_MIN,
                   help="minutes before pushback the door shuts "
                        "(default: %(default)s)")
    p.add_argument("--plot", default=None, help="write a bokeh HTML file here")
    p.add_argument("--cache", default="~/.cache/connection_check",
                   help="response cache dir; set to '' to disable")
    p.add_argument("--api-key", default=os.environ.get("AEROAPI_KEY", ""),
                   help="AeroAPI key (default: $AEROAPI_KEY)")
    p.add_argument("--demo", action="store_true",
                   help="use synthetic data instead of the API")
    p.add_argument("--csv", default=None,
                   help="read hand-entered history from this CSV instead of "
                        "calling the API")
    p.add_argument("--write-csv-template", default=None, metavar="FILE",
                   help="write a blank CSV template here and exit; will not "
                        "clobber an existing file")
    p.add_argument("--force", action="store_true",
                   help="allow --write-csv-template to overwrite an existing "
                        "file")
    p.add_argument("--json", dest="json_out", default=None,
                   help="also write the parsed results here as JSON")
    args = p.parse_args(argv)

    cache = Path(os.path.expanduser(args.cache)) if args.cache else None

    if args.write_csv_template:
        target = Path(args.write_csv_template)
        if target.exists() and not args.force:
            print(f"{target} already exists -- refusing to overwrite it.\n"
                  f"Your filled-in data is safe. Pick another filename, or "
                  f"pass --force if you really do want a blank template here.",
                  file=sys.stderr)
            return 3
        target.write_text(CSV_TEMPLATE)
        print(f"wrote {target} -- fill it in, then re-run "
              f"with --csv {target}")
        return 0

    if args.csv:
        if not args.outbound:
            p.error("--csv describes a connection, so --outbound is required")
        inb_legs, out_legs = load_csv(args.csv, args.inbound, args.outbound,
                                      args.via or "FRA")
        print(f"[read {len(inb_legs)} rows from {args.csv}]\n")
    elif args.demo:
        inb_legs, out_legs = demo_legs(args.inbound,
                                       args.outbound or "LH590", args.days)
        if not args.outbound:
            out_legs = []
        print("[demo mode: synthetic data, no API calls]\n")
    else:
        if not args.api_key:
            print("No API key. Set AEROAPI_KEY or pass --api-key, or try "
                  "--demo.\nFree key: "
                  "https://www.flightaware.com/aeroapi/portal/",
                  file=sys.stderr)
            return 2
        try:
            inb_legs = fetch_legs(args.inbound, args.days, args.api_key, cache)
            out_legs = (fetch_legs(args.outbound, args.days, args.api_key,
                                   cache) if args.outbound else [])
        except AeroAPIError as exc:
            print(f"AeroAPI: {exc}", file=sys.stderr)
            return 1

    inb_legs = filter_route(inb_legs, args.origin, args.via)
    out_legs = filter_route(out_legs, args.via, args.final)
    inb_legs = [l for l in inb_legs if l.usable]
    out_legs = [l for l in out_legs if l.usable]

    if not inb_legs:
        print(f"No usable legs found for {args.inbound} in the last "
              f"{args.days} days.", file=sys.stderr)
        return 1

    airport = args.via.upper() or inb_legs[-1].destination
    mct = args.mct if args.mct is not None else MCT_MINUTES.get(
        airport, MCT_MINUTES.get(airport[-3:], DEFAULT_MCT))

    pairs = pair_connections(inb_legs, out_legs) if out_legs else []

    if pairs:
        print_connection_table(pairs, args.door_close, mct)
        summary = summarize_connections(pairs, mct, args.door_close)
        print_connection_summary(summary, args.inbound, args.outbound, airport)
    else:
        summary = None
        if args.outbound:
            print("Could not pair any legs -- check --via, or the outbound "
                  "may not depart within the matching window.",
                  file=sys.stderr)

    print_delay_summary(summarize_delays(inb_legs), args.inbound)

    if args.json_out:
        payload = {
            "inbound": args.inbound, "outbound": args.outbound,
            "airport": airport, "mct": mct, "summary": summary,
            "connections": [{
                "date": c.date,
                "scheduled_gap_min": c.scheduled_gap_min,
                "actual_gap_min": c.actual_gap_min,
                "usable_gap_min": c.usable_gap_min(args.door_close),
                "gates": c.gates,
                "inbound_arrival_delay_min": c.inbound.arrival_delay_min,
                "outbound_departure_delay_min":
                    c.outbound.departure_delay_min,
            } for c in pairs],
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json_out}")

    if args.plot:
        title = (f"{args.inbound} -> {args.outbound} at {airport}"
                 if args.outbound else f"{args.inbound} punctuality")
        make_plot(pairs, inb_legs, args.plot, title, mct, args.door_close)
        print(f"wrote {args.plot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
