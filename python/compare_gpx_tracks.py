"""
compare_gpx_tracks.py

Compare two GPX tracks recorded simultaneously (e.g. Apple Watch vs iPhone)
to see where their distance totals diverge: uniformly over the walk, or in
discrete episodes (GPS jitter, signal loss among buildings, etc).

Usage:
    python3 compare_gpx_tracks.py watch.gpx iphone.gpx -o comparison.html

Requires: bokeh, numpy  (no pandas dependency, kept deliberately lightweight)
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from typing import List

import numpy as np
from bokeh.io import save
from bokeh.layouts import column, gridplot
from bokeh.models import ColumnDataSource, HoverTool, Div, Span
from bokeh.palettes import Category10
from bokeh.plotting import figure

EARTH_RADIUS_M = 6371000.0
GPX_NS = {"g": "http://www.topografix.com/GPX/1/1"}


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized great-circle distance in meters between consecutive points."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return EARTH_RADIUS_M * c


@dataclass
class GPXTrack:
    """A single parsed GPX track with derived distance/cadence fields."""

    label: str
    path: str
    lat: np.ndarray = field(default=None, repr=False)
    lon: np.ndarray = field(default=None, repr=False)
    ele: np.ndarray = field(default=None, repr=False)
    time: List[datetime] = field(default=None, repr=False)
    elapsed_s: np.ndarray = field(default=None, repr=False)   # seconds since this track's own start
    step_dist_m: np.ndarray = field(default=None, repr=False)  # distance of point i-1 -> i (len N, [0]=0)
    cum_dist_m: np.ndarray = field(default=None, repr=False)
    dt_s: np.ndarray = field(default=None, repr=False)         # time delta of point i-1 -> i (len N, [0]=0)

    @classmethod
    def from_gpx(cls, path: str, label: str) -> "GPXTrack":
        tree = ET.parse(path)
        root = tree.getroot()
        pts = root.findall(".//g:trkpt", GPX_NS)

        lat, lon, ele, time = [], [], [], []
        for pt in pts:
            lat.append(float(pt.get("lat")))
            lon.append(float(pt.get("lon")))
            ele_el = pt.find("g:ele", GPX_NS)
            ele.append(float(ele_el.text) if ele_el is not None else np.nan)
            time_el = pt.find("g:time", GPX_NS)
            time.append(datetime.fromisoformat(time_el.text.replace("Z", "+00:00")))

        track = cls(label=label, path=path)
        track.lat = np.array(lat)
        track.lon = np.array(lon)
        track.ele = np.array(ele)
        track.time = time
        track._compute_derived()
        return track

    def _compute_derived(self) -> None:
        t0 = self.time[0]
        self.elapsed_s = np.array([(t - t0).total_seconds() for t in self.time])

        self.dt_s = np.diff(self.elapsed_s, prepend=self.elapsed_s[0])
        self.dt_s[0] = 0.0

        step = haversine_m(self.lat[:-1], self.lon[:-1], self.lat[1:], self.lon[1:])
        self.step_dist_m = np.concatenate([[0.0], step])
        self.cum_dist_m = np.cumsum(self.step_dist_m)

    @property
    def n_points(self) -> int:
        return len(self.lat)

    @property
    def total_distance_mi(self) -> float:
        return self.cum_dist_m[-1] / 1609.344

    @property
    def start_time(self) -> datetime:
        return self.time[0]

    @property
    def end_time(self) -> datetime:
        return self.time[-1]

    def detect_spikes(self, max_walk_speed_mps: float = 3.0, snap_fraction: float = 0.4) -> List[int]:
        """
        Flag indices that look like a single-fix GPS outlier: the point is
        reached at an implausible walking speed and then immediately leaves
        at an implausible speed back toward where it came from (a "teleport
        and snap back"), rather than being part of a real, sustained path.

        Returns a list of point indices (not the first/last point).
        """
        spikes = []
        for i in range(1, self.n_points - 1):
            dt_in, dt_out = self.dt_s[i], self.dt_s[i + 1]
            if dt_in <= 0 or dt_out <= 0:
                continue
            v_in = self.step_dist_m[i] / dt_in
            v_out = self.step_dist_m[i + 1] / dt_out
            if v_in < max_walk_speed_mps or v_out < max_walk_speed_mps:
                continue
            direct = haversine_m(self.lat[i - 1], self.lon[i - 1], self.lat[i + 1], self.lon[i + 1])
            round_trip = self.step_dist_m[i] + self.step_dist_m[i + 1]
            if direct < snap_fraction * round_trip:
                spikes.append(i)
        return spikes

    def cadence_summary(self) -> str:
        """Return a short text summary of sampling interval stats."""
        dts = self.dt_s[1:]  # drop the synthetic leading 0
        vals, counts = np.unique(np.round(dts, 1), return_counts=True)
        mode_val = vals[np.argmax(counts)]
        return (
            f"{self.label}: {self.n_points} points, "
            f"median dt={np.median(dts):.2f}s, mode dt={mode_val:.1f}s, "
            f"min={dts.min():.2f}s, max={dts.max():.2f}s, "
            f"non-{mode_val:.1f}s samples={int(np.sum(np.round(dts,1) != mode_val))}/{len(dts)}"
        )


class TrackComparison:
    """
    Aligns two GPXTracks on a shared, uniform 1-second wall-clock grid
    (built from whichever track starts later / ends earlier, so both are
    covered) and produces a set of Bokeh diagnostic plots.
    """

    def __init__(self, track_a: GPXTrack, track_b: GPXTrack, grid_step_s: float = 1.0,
                 known_distance_m: float = None, max_walk_speed_mps: float = 3.0):
        self.a = track_a
        self.b = track_b
        self.grid_step_s = grid_step_s
        self.known_distance_m = known_distance_m
        self._build_common_grid()
        self.a_spikes = self.a.detect_spikes(max_walk_speed_mps=max_walk_speed_mps)
        self.b_spikes = self.b.detect_spikes(max_walk_speed_mps=max_walk_speed_mps)

    def _build_common_grid(self) -> None:
        start = max(self.a.start_time, self.b.start_time)
        end = min(self.a.end_time, self.b.end_time)
        n = int((end - start).total_seconds() // self.grid_step_s) + 1
        self.grid_s = np.arange(n) * self.grid_step_s  # seconds since `start`
        self.grid_start = start

        a_offset = (start - self.a.start_time).total_seconds()
        b_offset = (start - self.b.start_time).total_seconds()

        self.a_cum_on_grid = np.interp(self.grid_s + a_offset, self.a.elapsed_s, self.a.cum_dist_m)
        self.b_cum_on_grid = np.interp(self.grid_s + b_offset, self.b.elapsed_s, self.b.cum_dist_m)

        # instantaneous speed on the grid (m/s), from the 1-s-spaced cumulative distance
        self.a_speed = np.diff(self.a_cum_on_grid, prepend=self.a_cum_on_grid[0])
        self.b_speed = np.diff(self.b_cum_on_grid, prepend=self.b_cum_on_grid[0])

        self.diff_cum_m = self.b_cum_on_grid - self.a_cum_on_grid  # b minus a
        self.diff_speed = self.b_speed - self.a_speed

    def text_summary(self) -> str:
        lines = [
            f"<b>{self.a.label}</b>: {self.a.total_distance_mi:.3f} mi over "
            f"{self.a.elapsed_s[-1]/60:.1f} min ({self.a.n_points} pts)",
            f"<b>{self.b.label}</b>: {self.b.total_distance_mi:.3f} mi over "
            f"{self.b.elapsed_s[-1]/60:.1f} min ({self.b.n_points} pts)",
            f"Difference over overlapping window: {self.diff_cum_m[-1]:.1f} m "
            f"({self.diff_cum_m[-1]/1609.344:.3f} mi), "
            f"{'b/' + self.b.label + ' larger' if self.diff_cum_m[-1] > 0 else self.a.label + ' larger'}",
            self.a.cadence_summary(),
            self.b.cadence_summary(),
            f"Single-fix GPS spikes detected: {self.a.label}={len(self.a_spikes)}, "
            f"{self.b.label}={len(self.b_spikes)} "
            "(a point reached and left at an implausible walking speed, snapping back toward its neighbors)",
        ]
        if self.known_distance_m is not None:
            known_mi = self.known_distance_m / 1609.344
            for track in (self.a, self.b):
                err_m = track.cum_dist_m[-1] - self.known_distance_m
                err_pct = 100 * err_m / self.known_distance_m
                lines.append(
                    f"<b>{track.label} vs. known distance</b>: ground truth={known_mi:.3f} mi, "
                    f"measured={track.total_distance_mi:.3f} mi, "
                    f"error={err_m:+.1f} m ({err_pct:+.1f}%)"
                )
        return "<br>".join(lines)

    def spike_table(self) -> str:
        """HTML list of detected spikes with timestamp and coordinates, for inspection."""
        rows = []
        for track, idxs in ((self.a, self.a_spikes), (self.b, self.b_spikes)):
            for i in idxs:
                rows.append(
                    f"<tr><td>{track.label}</td><td>{track.time[i].isoformat()}</td>"
                    f"<td>{track.lat[i]:.6f}</td><td>{track.lon[i]:.6f}</td></tr>"
                )
        if not rows:
            return "<p>No spikes detected.</p>"
        return (
            "<table border='1' cellpadding='4' style='border-collapse:collapse'>"
            "<tr><th>Track</th><th>Time (UTC)</th><th>lat</th><th>lon</th></tr>"
            + "".join(rows) + "</table>"
        )

    def build_layout(self, focus_lat: float = None, focus_lon: float = None, focus_pad_deg: float = 0.0015):
        colors = Category10[3]
        time_axis = [self.grid_start.isoformat()] * 0  # placeholder, bokeh wants datetime axis via ms since epoch
        x = self.grid_s / 60.0  # minutes elapsed, easier to read than raw seconds

        src = ColumnDataSource(data=dict(
            x=x,
            a_cum_mi=self.a_cum_on_grid / 1609.344,
            b_cum_mi=self.b_cum_on_grid / 1609.344,
            diff_mi=self.diff_cum_m / 1609.344,
            diff_m=self.diff_cum_m,
            a_speed=self.a_speed,
            b_speed=self.b_speed,
            diff_speed=self.diff_speed,
        ))

        tools = "pan,wheel_zoom,box_zoom,reset,save"

        # 1. Cumulative distance, both tracks overlaid
        p1 = figure(title="Cumulative distance vs. time", x_axis_label="elapsed minutes",
                    y_axis_label="cumulative miles", tools=tools, height=320, width=850)
        p1.line("x", "a_cum_mi", source=src, legend_label=self.a.label, color=colors[0], line_width=2)
        p1.line("x", "b_cum_mi", source=src, legend_label=self.b.label, color=colors[1], line_width=2)
        p1.add_tools(HoverTool(tooltips=[("min", "@x{0.0}"), (self.a.label, "@a_cum_mi{0.000} mi"),
                                          (self.b.label, "@b_cum_mi{0.000} mi")]))
        p1.legend.location = "top_left"
        p1.legend.click_policy = "hide"

        # 2. Running difference in cumulative distance (does the gap grow steadily or in jumps?)
        p2 = figure(title=f"Cumulative distance difference ({self.b.label} minus {self.a.label})",
                    x_axis_label="elapsed minutes", y_axis_label="difference (miles)",
                    tools=tools, height=320, width=850, x_range=p1.x_range)
        p2.line("x", "diff_mi", source=src, color=colors[2], line_width=2)
        zero = Span(location=0, dimension="width", line_color="gray", line_dash="dashed")
        p2.add_layout(zero)
        p2.add_tools(HoverTool(tooltips=[("min", "@x{0.0}"), ("diff", "@diff_mi{0.000} mi")]))

        # 3. Per-second speed for both tracks (reveals episodic spikes/jitter)
        p3 = figure(title="Per-second speed (step distance / 1s) — spikes indicate GPS jitter",
                    x_axis_label="elapsed minutes", y_axis_label="m/s",
                    tools=tools, height=320, width=850, x_range=p1.x_range)
        p3.line("x", "a_speed", source=src, legend_label=self.a.label, color=colors[0], alpha=0.8)
        p3.line("x", "b_speed", source=src, legend_label=self.b.label, color=colors[1], alpha=0.8)
        p3.add_tools(HoverTool(tooltips=[("min", "@x{0.0}"), (self.a.label, "@a_speed{0.00} m/s"),
                                          (self.b.label, "@b_speed{0.00} m/s")]))
        p3.legend.location = "top_left"
        p3.legend.click_policy = "hide"

        # 4. Difference in per-second speed (where the gap is actively opening vs flat)
        p4 = figure(title="Per-second speed difference (where the gap is actively opening)",
                    x_axis_label="elapsed minutes", y_axis_label="m/s difference",
                    tools=tools, height=320, width=850, x_range=p1.x_range)
        p4.vbar(x="x", top="diff_speed", source=src, width=self.grid_step_s / 60.0 * 0.9, color=colors[2])
        p4.add_layout(Span(location=0, dimension="width", line_color="gray", line_dash="dashed"))

        # 5. Sampling interval (cadence) histogram for both tracks
        a_dt = self.a.dt_s[1:]
        b_dt = self.b.dt_s[1:]
        p5 = figure(title="Sampling interval histogram (cadence)", x_axis_label="dt between fixes (s)",
                    y_axis_label="count", tools=tools, height=300, width=420)
        for dt, label, color in ((a_dt, self.a.label, colors[0]), (b_dt, self.b.label, colors[1])):
            hist, edges = np.histogram(dt, bins=np.arange(0, max(dt.max(), 2) + 0.5, 0.5))
            p5.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
                    fill_color=color, line_color="white", alpha=0.6, legend_label=label)
        p5.legend.click_policy = "hide"

        # 6. GPS track map (lat/lon) for both, to visually spot where paths diverge
        p6 = figure(title="Track paths (lon/lat)", x_axis_label="longitude", y_axis_label="latitude",
                    tools=tools, height=300, width=420, match_aspect=True)
        p6.line(self.a.lon, self.a.lat, color=colors[0], legend_label=self.a.label, line_width=2)
        p6.line(self.b.lon, self.b.lat, color=colors[1], legend_label=self.b.label, line_width=2)
        if self.a_spikes:
            p6.scatter(self.a.lon[self.a_spikes], self.a.lat[self.a_spikes], marker="x",
                       size=10, color="red", legend_label=f"{self.a.label} spike")
        if self.b_spikes:
            p6.scatter(self.b.lon[self.b_spikes], self.b.lat[self.b_spikes], marker="x",
                       size=10, color="black", legend_label=f"{self.b.label} spike")
        p6.legend.click_policy = "hide"

        summary_div = Div(text=f"<h2>GPX Track Comparison</h2><p>{self.text_summary()}</p>", width=850)
        plots = [summary_div, p1, p2, p3, p4, gridplot([[p5, p6]])]

        # 7. Optional zoomed-in map around a region of interest, with points shown
        # individually (not just connected lines) and detected spikes marked, so
        # jittery vs. plausible paths can be told apart.
        if focus_lat is not None and focus_lon is not None:
            p7 = figure(title=f"Zoomed path detail near ({focus_lat:.4f}, {focus_lon:.4f})",
                        x_axis_label="longitude", y_axis_label="latitude",
                        tools=tools, height=500, width=850, match_aspect=True,
                        x_range=(focus_lon - focus_pad_deg, focus_lon + focus_pad_deg),
                        y_range=(focus_lat - focus_pad_deg, focus_lat + focus_pad_deg))

            for track, color, spikes in ((self.a, colors[0], self.a_spikes), (self.b, colors[1], self.b_spikes)):
                mask = ((track.lat > focus_lat - focus_pad_deg) & (track.lat < focus_lat + focus_pad_deg) &
                        (track.lon > focus_lon - focus_pad_deg) & (track.lon < focus_lon + focus_pad_deg))
                sub_src = ColumnDataSource(data=dict(
                    lon=track.lon[mask], lat=track.lat[mask],
                    t=[t.strftime("%H:%M:%S") for t, m in zip(track.time, mask) if m],
                ))
                p7.line("lon", "lat", source=sub_src, color=color, legend_label=track.label, line_width=1.5, alpha=0.7)
                r = p7.scatter("lon", "lat", source=sub_src, color=color, size=5, legend_label=track.label)
                p7.add_tools(HoverTool(renderers=[r], tooltips=[("track", track.label), ("time", "@t"),
                                                                 ("lat", "@lat{0.000000}"), ("lon", "@lon{0.000000}")]))
                spike_mask = [i for i in spikes if mask[i]]
                if spike_mask:
                    p7.scatter(track.lon[spike_mask], track.lat[spike_mask], marker="x", size=14,
                               line_width=3, color="red" if track is self.a else "black",
                               legend_label=f"{track.label} spike")
            p7.legend.click_policy = "hide"
            plots.append(p7)
            plots.append(Div(text=f"<h3>Detected single-fix GPS spikes</h3>{self.spike_table()}", width=850))

        return column(*plots)

    def save(self, out_path: str, title: str = "GPX Track Comparison",
             focus_lat: float = None, focus_lon: float = None) -> None:
        from bokeh.plotting import output_file
        from bokeh.resources import INLINE
        output_file(out_path, title=title)
        save(self.build_layout(focus_lat=focus_lat, focus_lon=focus_lon), resources=INLINE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("track_a", help="Path to first GPX file (e.g. Watch)")
    parser.add_argument("track_b", help="Path to second GPX file (e.g. iPhone)")
    parser.add_argument("--label-a", default="Watch")
    parser.add_argument("--label-b", default="iPhone")
    parser.add_argument("-o", "--output", default="gpx_comparison.html")
    parser.add_argument("--focus-lat", type=float, default=None,
                         help="Latitude of a region to zoom in on in an extra plot")
    parser.add_argument("--focus-lon", type=float, default=None,
                         help="Longitude of a region to zoom in on in an extra plot")
    parser.add_argument("--known-distance-m", type=float, default=None,
                         help="Ground-truth distance in meters (e.g. laps of a track) to compute "
                              "each device's percent error against, instead of only comparing them "
                              "to each other")
    parser.add_argument("--max-walk-speed-mps", type=float, default=3.0,
                         help="Speed threshold used by the spike detector; raise this if the "
                              "recording includes jogging/running rather than walking")
    args = parser.parse_args()

    a = GPXTrack.from_gpx(args.track_a, args.label_a)
    b = GPXTrack.from_gpx(args.track_b, args.label_b)
    cmp = TrackComparison(a, b, known_distance_m=args.known_distance_m,
                           max_walk_speed_mps=args.max_walk_speed_mps)
    cmp.save(args.output, focus_lat=args.focus_lat, focus_lon=args.focus_lon)
    print(cmp.text_summary().replace("<br>", "\n").replace("<b>", "").replace("</b>", ""))
    print(f"\nSaved plot to {args.output}")


if __name__ == "__main__":
    main()
