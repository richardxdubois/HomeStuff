"""rubin_streak_id.py

Identify and plot satellite streaks (public catalog + targeted Fermi ephemeris)
overlapping Rubin/LSSTCam visits, using Bokeh.
"""
import argparse

import numpy as np
import requests
import yaml

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord

import lsst.daf.butler as dafButler
import lsst.geom
import lsst.sphgeom
from lsst.obs.lsst import LsstCam
from lsst.sphgeom import LonLat
from rubin_nights.consdb_query import ConsDbSql

from bokeh.plotting import figure, output_file, save
from bokeh.models import HoverTool, Legend, Label, ColumnDataSource
from bokeh.palettes import Category20_20
from bokeh.layouts import gridplot


# ---------------------------------------------------------------------------
# SatChecker helpers
# ---------------------------------------------------------------------------

def get_satchecker_ephemeris(satellite, start_jd, stop_jd, step_jd,
                             latitude, longitude, elevation):
    """Query SatChecker's name-jdstep endpoint for one satellite's topocentric
    RA/Dec over a time range (single call).

    Returns a list of [ra_deg, dec_deg, jd]; empty on failure.
    """
    url = (
        "https://satchecker.cps.iau.org/ephemeris/name-jdstep/"
        f"?name={satellite}"
        f"&elevation={elevation}"
        f"&latitude={latitude}"
        f"&longitude={longitude}"
        f"&startjd={start_jd}"
        f"&stopjd={stop_jd}"
        f"&stepjd={step_jd}"
        "&min_altitude=-90"
    )
    positions = []
    try:
        info = requests.get(url, timeout=60).json()
        fields = info["fields"]
        ra_index = fields.index("right_ascension_deg")
        dec_index = fields.index("declination_deg")

        jd_index = None
        for cand in ("julian_date", "julian_date_jd", "jd", "mjd"):
            if cand in fields:
                jd_index = fields.index(cand)
                break

        for row in info["data"]:
            ra, dec = row[ra_index], row[dec_index]
            jd = row[jd_index] if jd_index is not None else None
            if ra is not None and dec is not None:
                positions.append([float(ra), float(dec),
                                  float(jd) if jd is not None else None])
    except Exception as e:
        print(f"  WARNING: ephemeris query failed for {satellite}: {e}")
    return positions


# ---------------------------------------------------------------------------
# Butler / detector geometry helpers
# ---------------------------------------------------------------------------

def get_visit_summary(butler, visit, dataset_type=None):
    """Fetch a visit summary, supporting legacy and newer dataset names."""
    dataset_types = (
        [dataset_type] if dataset_type is not None
        else ["visit_summary", "preliminary_visit_summary", "visitSummary"]
    )
    for dt in dataset_types:
        try:
            return butler.get(dt, visit=visit)
        except LookupError:
            pass
    raise LookupError(
        f"Visit summary for visit {visit!r} not found in any of {dataset_types!r}."
    )


def get_detector_corners(butler, visit):
    """Return (corners_list, detectors_list) of RA/Dec corners for a visit."""
    camera = LsstCam().getCamera()
    detectors = [det.getId() for det in camera]
    corners_list, detectors_list = [], []

    try:
        pvs = get_visit_summary(butler, visit)
    except LookupError:
        print("No visit summary dataset found. Constructing corners from raws. "
              "THIS IS SLOW.")
        for detector in detectors:
            if detector >= 189:
                continue
            data_id = {"exposure": visit, "detector": detector}
            try:
                exposure = butler.get("raw", data_id)
            except (dafButler.DatasetNotFoundError,
                    dafButler.MissingDatasetTypeError):
                print(f"Unable to compute corners for detector {detector}.")
                corners_list.append([(None, None)] * 4)
                detectors_list.append(detector)
                continue
            wcs = exposure.getWcs()
            bbox = exposure.getBBox()
            sph_points = wcs.pixelToSky(lsst.geom.Box2D(bbox).getCorners())
            ra_corners = [float(p.getRa().asDegrees()) for p in sph_points]
            dec_corners = [float(p.getDec().asDegrees()) for p in sph_points]
            corners_list.append(list(zip(ra_corners, dec_corners)))
            detectors_list.append(detector)
    else:
        for row in pvs:
            corners_list.append(list(zip(row["raCorners"], row["decCorners"])))
            detectors_list.append(row["id"])

    return corners_list, detectors_list


# ---------------------------------------------------------------------------
# Main plotter class
# ---------------------------------------------------------------------------

class VisitStreakPlotter:
    """Build Bokeh plots of satellite streaks overlapping Rubin/LSSTCam visits."""

    GOLD = "#FFB300"
    DETECTOR_COLOR = "#058B8C"

    def __init__(self, butler, instrument="LSSTCam", fov_radius=3,
                 consdb_site="usdf"):
        self.butler = butler
        self.instrument = instrument
        self.fov_radius = fov_radius
        self.consdb = ConsDbSql(site=consdb_site)

        location = EarthLocation.of_site("Rubin")
        self.latitude = location.lat.value
        self.longitude = location.lon.value
        self.elevation = location.height.value

    # -- ConsDB -------------------------------------------------------------

    def _query_visit(self, visit_id):
        """Return the ConsDB visit row, or None if unavailable."""
        query = f"select * from cdb_lsstcam.visit1 where visit_id={visit_id}"
        try:
            info = self.consdb.query(query)
        except Exception as e:
            print(f"ERROR: Could not query ConsDB for visit {visit_id}: {e}")
            return None
        if len(info) == 0:
            print(f"WARNING: No ConsDB data for visit {visit_id}")
            return None
        return info

    @staticmethod
    def _parse_region(region_cdb):
        """Parse an s_region string into a list of [lon_deg, lat_deg] vertices."""
        region = lsst.sphgeom.Region.from_ivoa_pos(
            "".join(region_cdb.split("ICRS")).upper()
        )
        return [[np.float64(LonLat.longitudeOf(v).asDegrees()),
                 np.float64(LonLat.latitudeOf(v).asDegrees())]
                for v in region.getVertices()]

    # -- SatChecker ---------------------------------------------------------

    def _query_fov_satellites(self, ra_center, dec_center,
                              start_time_jd, duration):
        """Query the SatChecker FOV endpoint; return {name: [[ra, dec, jd], ...]}."""
        url = (
            "https://satchecker.cps.iau.org/fov/satellite-passes/"
            f"?latitude={self.latitude}&longitude={self.longitude}"
            f"&elevation={self.elevation}&start_time_jd={start_time_jd}"
            f"&duration={duration}&ra={ra_center}&dec={dec_center}"
            f"&fov_radius={self.fov_radius}&group_by=satellite&async=False"
        )
        try:
            data = requests.get(url, timeout=60).json()
        except Exception as e:
            print(f"WARNING: SatChecker FOV request failed: {e}")
            data = {"data": {"satellites": {}}}

        satellites = {}
        for name, sat_data in data["data"]["satellites"].items():
            satellites[name] = [
                [pos["ra"], pos["dec"], pos["julian_date"]]
                for pos in sat_data["positions"]
            ]
        print(f"Public catalog satellites: {len(satellites)} found")
        return satellites

    def _fermi_ephemeris(self, overlaps_list, exp_midpt):
        """Targeted Fermi ephemeris centered on closest approach."""
        t_close = exp_midpt.jd
        if (overlaps_list and isinstance(overlaps_list[0], dict)
                and "closest_approach_time" in overlaps_list[0]):
            t_close = Time(overlaps_list[0]["closest_approach_time"]).utc.jd

        half_window_s = 120.0  # +/- 2 min
        start_jd = t_close - half_window_s / 86400.0
        stop_jd = t_close + half_window_s / 86400.0
        step_jd = (stop_jd - start_jd) / 30.0  # ~8 s sampling

        eph = get_satchecker_ephemeris(
            "GLAST", start_jd, stop_jd, step_jd,
            self.latitude, self.longitude, self.elevation,
        )
        if eph:
            print(f"  SatChecker ephemeris: Fermi found, {len(eph)} positions")
        else:
            print("  SatChecker ephemeris: no Fermi positions "
                  "(check satellite name / endpoint)")
        return eph

    # -- diagnostics --------------------------------------------------------

    @staticmethod
    def _report_closest_approach(fermi_overlaps, fermi_eph,
                                 ra_center, dec_center, exp_midpt):
        """Print pipeline vs SatChecker closest-approach comparison."""
        for overlap in fermi_overlaps:
            trajectory = overlap.get("trajectory")
            if not trajectory:
                continue
            ra_points = [p["ra"] for p in trajectory]
            dec_points = [p["dec"] for p in trajectory]
            time_points = [p["time"] for p in trajectory]

            print(f"  Fermi trajectory: {len(ra_points)} points")

            d_ra = (np.array(ra_points) - ra_center) * np.cos(np.radians(dec_center))
            d_dec = np.array(dec_points) - dec_center
            distances = np.sqrt(d_ra ** 2 + d_dec ** 2)
            min_idx = int(np.argmin(distances))

            print("\n  Your matching code says:")
            print(f"    Fermi closest approach at: {time_points[min_idx]}")
            print(f"    Position: RA={ra_points[min_idx]:.3f}deg, "
                  f"Dec={dec_points[min_idx]:.3f}deg")
            print(f"    Distance from field center: {distances[min_idx]:.3f}deg")

            if fermi_eph:
                mid_jd = exp_midpt.jd
                j = min(range(len(fermi_eph)),
                        key=lambda k: abs(fermi_eph[k][2] - mid_jd))
                print("\n  SatChecker ephemeris nearest exp_midpt:")
                print(f"    RA={fermi_eph[j][0]:.3f}deg, Dec={fermi_eph[j][1]:.3f}deg")
                dra = ((fermi_eph[j][0] - ra_points[min_idx])
                       * np.cos(np.radians(dec_points[min_idx])))
                ddec = fermi_eph[j][1] - dec_points[min_idx]
                print(f"    Offset from your trajectory: "
                      f"{np.sqrt(dra**2 + ddec**2):.3f}deg "
                      "(should be small if both agree)")

    # -- plotting -----------------------------------------------------------

    def create_visit_plot(self, visit_id, fermi_overlaps):
        """Create a Bokeh figure for a single visit, or None on failure."""
        print(f"\n{'=' * 60}\nProcessing visit {visit_id}\n{'=' * 60}")

        info = self._query_visit(visit_id)
        if info is None:
            return None

        region_cdb = info["s_region"][0]
        if region_cdb is None:
            print(f"WARNING: No s_region for visit {visit_id}, skipping")
            return None
        try:
            vertices = self._parse_region(region_cdb)
        except Exception as e:
            print(f"ERROR: Could not parse s_region for visit {visit_id}: {e}")
            return None

        exp_begin = Time(info["obs_start"][0])
        exp_end = Time(info["obs_end"][0])
        exp_time = exp_end - exp_begin
        exp_midpt = Time(info["exp_midpt"][0])

        start_time_jd = (exp_begin - exp_time).jd
        duration = exp_time.sec * 3
        ra_center = info["s_ra"][0]
        dec_center = info["s_dec"][0]

        print(f"  EXPOSURE: {exp_begin.iso} -> {exp_end.iso} (mid {exp_midpt.iso})")

        try:
            corners_list, detectors_list = get_detector_corners(
                self.butler, visit=visit_id)
        except Exception as e:
            print(f"WARNING: Could not get detector corners for {visit_id}: {e}")
            corners_list, detectors_list = [], []

        satellites = self._query_fov_satellites(
            ra_center, dec_center, start_time_jd, duration)

        overlaps_list = (
            [fermi_overlaps] if isinstance(fermi_overlaps, dict)
            else fermi_overlaps if isinstance(fermi_overlaps, list)
            else []
        )
        fermi_eph = self._fermi_ephemeris(overlaps_list, exp_midpt)

        if fermi_overlaps:
            self._report_closest_approach(
                fermi_overlaps, fermi_eph, ra_center, dec_center, exp_midpt)

        return self._build_figure(
            visit_id, duration, ra_center, dec_center, vertices,
            corners_list, detectors_list, satellites, fermi_eph, fermi_overlaps)

    def _build_figure(self, visit_id, duration, ra_center, dec_center, vertices,
                      corners_list, detectors_list, satellites, fermi_eph,
                      fermi_overlaps):
        p = figure(
            width=800, height=700,
            title=f"Visit {visit_id} (duration {duration:.1f} s)",
            x_axis_label="Right Ascension (deg)",
            y_axis_label="Declination (deg)",
            match_aspect=True,
        )

        # SatChecker search radius circle
        center = SkyCoord(ra=ra_center * u.deg, dec=dec_center * u.deg, frame="icrs")
        position_angles = np.linspace(0, 360, 360) * u.deg
        circle = center.directional_offset_by(position_angles, self.fov_radius * u.deg)
        p.line(circle.ra.deg, circle.dec.deg, color="lightgray",
               line_dash="dashed", line_width=2,
               legend_label="SatChecker search radius", alpha=0.7)

        legend_items = []
        self._plot_satellites(p, satellites, legend_items)
        self._plot_detectors(p, corners_list, detectors_list, vertices)
        self._plot_fermi_ephemeris(p, fermi_eph, legend_items)
        self._plot_fermi_overlaps(p, fermi_overlaps, legend_items)

        legend = Legend(items=legend_items, location="top_left")
        legend.click_policy = "hide"
        p.add_layout(legend, "right")

        p.x_range.flipped = True
        p.grid.grid_line_color = "lightgray"
        p.grid.grid_line_dash = "dashed"
        return p

    def _plot_satellites(self, p, satellites, legend_items):
        colors = Category20_20
        for idx, (sat_name, positions) in enumerate(satellites.items()):
            ra = [pos[0] for pos in positions]
            dec = [pos[1] for pos in positions]
            color = colors[idx % len(colors)]

            line = p.line(ra, dec, color=color, line_width=2, alpha=0.8)
            scatter = p.scatter(ra, dec, color=color, size=6, alpha=0.8)
            p.add_tools(HoverTool(
                renderers=[scatter],
                tooltips=[("Satellite", sat_name),
                          ("RA", "@x{0.000}"), ("Dec", "@y{0.000}")],
            ))
            legend_items.append((sat_name, [line, scatter]))

            if ra:
                i = 0 if ra[0] < ra[-1] else -1
                p.add_layout(Label(x=ra[i], y=dec[i], text=sat_name,
                                   text_font_size="8pt", text_color=color))

    def _plot_detectors(self, p, corners_list, detectors_list, vertices):
        could_not_load = (
            len(corners_list) == 0 or
            all(item is None for corners in corners_list
                for value in corners for item in value)
        )

        for idx, corners in enumerate(corners_list):
            if any(item is None for value in corners for item in value):
                continue
            xs = [c[0] for c in corners] + [corners[0][0]]
            ys = [c[1] for c in corners] + [corners[0][1]]
            p.patch(xs, ys, color=self.DETECTOR_COLOR, alpha=0.2,
                    line_color=self.DETECTOR_COLOR, line_width=1)

            center_x = (corners[0][0] + corners[2][0]) / 2
            center_y = (corners[0][1] + corners[2][1]) / 2
            p.add_layout(Label(x=center_x, y=center_y,
                               text=str(detectors_list[idx]),
                               text_font_size="6pt", text_color="#313333",
                               text_align="center", text_baseline="middle"))

        if could_not_load or len(corners_list) < 160:
            xs = [v[0] for v in vertices] + [vertices[0][0]]
            ys = [v[1] for v in vertices] + [vertices[0][1]]
            p.patch(xs, ys, color=self.DETECTOR_COLOR, alpha=0.2,
                    line_color=self.DETECTOR_COLOR, line_width=1)

    def _plot_fermi_ephemeris(self, p, fermi_eph, legend_items):
        if not fermi_eph:
            return
        eph_ra = [pp[0] for pp in fermi_eph]
        eph_dec = [pp[1] for pp in fermi_eph]
        eph_jd = [pp[2] for pp in fermi_eph]

        line = p.line(eph_ra, eph_dec, color=self.GOLD, line_width=3,
                      alpha=0.9, line_dash="dotted")
        src = ColumnDataSource(data=dict(ra=eph_ra, dec=eph_dec, jd=eph_jd))
        scatter = p.scatter("ra", "dec", source=src, color=self.GOLD,
                            size=7, alpha=0.9, marker="diamond")
        p.add_tools(HoverTool(
            renderers=[scatter],
            tooltips=[("Type", "Fermi (SatChecker)"),
                      ("RA", "@ra{0.000}"), ("Dec", "@dec{0.000}"),
                      ("JD", "@jd{0.00000}")],
        ))
        legend_items.append(("Fermi - SatChecker (33053)", [line, scatter]))

    def _plot_fermi_overlaps(self, p, fermi_overlaps, legend_items):
        if not fermi_overlaps:
            return
        fermi_lines = []
        for i, overlap in enumerate(fermi_overlaps):
            trajectory = overlap.get("trajectory")
            if not trajectory:
                continue
            ra_points = [point["ra"] for point in trajectory]
            dec_points = [point["dec"] for point in trajectory]
            time_points = [point["time"] for point in trajectory]

            line = p.line(ra_points, dec_points, color="red",
                          line_width=3, alpha=0.9, line_dash="solid")
            scatter = p.scatter(ra_points, dec_points, color="red",
                                size=6, alpha=0.9, marker="circle")

            src = ColumnDataSource(data=dict(
                ra=ra_points, dec=dec_points, time=time_points))
            scatter_hover = p.scatter("ra", "dec", source=src, color="red",
                                      size=6, alpha=0.9, marker="circle")
            p.add_tools(HoverTool(
                renderers=[scatter_hover],
                tooltips=[("Type", "Fermi (pipeline)"),
                          ("RA", "@ra{0.3f}deg"), ("Dec", "@dec{0.3f}deg"),
                          ("Time", "@time")],
            ))

            p.scatter([ra_points[0]], [dec_points[0]], color="darkred",
                      size=10, marker="square", alpha=1.0)
            p.scatter([ra_points[-1]], [dec_points[-1]], color="darkred",
                      size=10, marker="triangle", alpha=1.0)

            if i == 0:
                fermi_lines.append(
                    (f"Fermi - pipeline ({len(ra_points)} pts)",
                     [line, scatter]))
        legend_items.extend(fermi_lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_fermi_overlaps(yaml_file):
    with open(yaml_file, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot satellite streaks overlapping Rubin/LSSTCam visits.")
    parser.add_argument(
        "-o", "--overlaps", default="fermi_overlaps.yaml",
        help="Path to the Fermi overlaps YAML file "
             "(default: fermi_overlaps.yaml).")
    parser.add_argument(
        "--repo", default="/repo/main", help="Butler repo (default: /repo/main).")
    parser.add_argument(
        "--instrument", default="LSSTCam", help="Instrument (default: LSSTCam).")
    parser.add_argument(
        "--fov-radius", type=float, default=3,
        help="SatChecker search radius in deg (default: 3).")
    return parser.parse_args()


def main():
    args = parse_args()

    fermi_data = load_fermi_overlaps(args.overlaps)

    collections = [f"{args.instrument}/defaults"]
    butler = dafButler.Butler(args.repo, collections=collections,
                              instrument=args.instrument)

    plotter = VisitStreakPlotter(butler, instrument=args.instrument,
                                 fov_radius=args.fov_radius)

    plots, visit_ids = [], []
    for visit_info in fermi_data["visits"]:
        visit_id = visit_info["visit_id"]
        overlaps = visit_info.get("overlaps", [])
        p = plotter.create_visit_plot(visit_id, overlaps)
        if p is not None:
            plots.append(p)
            visit_ids.append(visit_id)
        else:
            print(f"Skipping visit {visit_id} due to errors")

    if not plots:
        print("ERROR: No valid plots generated!")
        return

    grid = gridplot(plots, ncols=2, sizing_mode="scale_width")
    output_filename = f"fermi_passes_{'_'.join(map(str, visit_ids[:3]))}_etc.html"
    output_file(output_filename)
    save(grid)

    print(f"\n{'=' * 60}")
    print(f"Successfully created {len(plots)} plots, saved to {output_filename}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
