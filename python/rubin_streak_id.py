# "rubin_streak_id_bokeh.py"
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation
import numpy as np
import requests
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import lsst.daf.butler as dafButler
from lsst.obs.lsst import LsstCam
import lsst.geom
from lsst.sphgeom import LonLat
from rubin_nights.consdb_query import ConsDbSql
from datetime import datetime, timezone

from bokeh.plotting import figure, output_file, save
from bokeh.models import HoverTool, Legend, Label, ColumnDataSource
from bokeh.palettes import Category20_20
from bokeh.layouts import gridplot
from bokeh.models import Title

import yaml


def load_fermi_overlaps(yaml_file):
    """Load Fermi overlap data from YAML file."""
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    return data


def getVisitSummaryForVisit(butler, visit, visitSummaryDatasetType=None):
    """Fetch visit summary for a visit, supporting legacy and newer names."""
    datasetTypes = (
        [visitSummaryDatasetType]
        if visitSummaryDatasetType is not None
        else [
            "visit_summary",
            "preliminary_visit_summary",
            "visitSummary",
        ]
    )
    for datasetType in datasetTypes:
        try:
            return butler.get(datasetType, visit=visit)
        except LookupError:
            pass
    raise LookupError(f"Visit summary for visit {visit!r} not found in any of {datasetTypes!r}.")


def getDetRaDecCorners(butler, visit):
    """Compute the RA/Dec corners lists for a given detector in a visit."""
    camera = LsstCam().getCamera()
    detectors = [det.getId() for det in camera]
    detectorsList = []
    cornersList = []
    try:
        pvs = getVisitSummaryForVisit(butler, visit)
    except LookupError:
        print("No visit summary dataset found. Attempting to construct detector corners from raws. THIS IS SLOW.")
        for detector in detectors:
            if detector < 189:
                dataId = {"exposure": visit, "detector": detector}
                try:
                    exposure = butler.get("raw", dataId)
                except (dafButler.DatasetNotFoundError, dafButler.MissingDatasetTypeError):
                    print(f"Unable to compute raCorners and decCorners for detector {detector}.")
                    cornersList.append([(None, None), (None, None), (None, None), (None, None)])
                    detectorsList.append(detector)
                    continue
                else:
                    wcs = exposure.getWcs()
                    bbox = exposure.getBBox()
                    sphPoints = wcs.pixelToSky(lsst.geom.Box2D(bbox).getCorners())
                    raCorners = [float(sph.getRa().asDegrees()) for sph in sphPoints]
                    decCorners = [float(sph.getDec().asDegrees()) for sph in sphPoints]
                    cornersList.append(list(zip(raCorners, decCorners)))
                    detectorsList.append(detector)
    else:
        for row in pvs:
            cornersList.append(list(zip(row["raCorners"], row["decCorners"])))
            detectorsList.append(row["id"])

    return cornersList, detectorsList


def create_visit_plot(obsinfo, fermi_overlaps, butler, collections, instrument, fov_radius=8):
    """Create a Bokeh plot for a single visit."""

    print(f"\n{'=' * 60}")
    print(f"Processing visit {obsinfo}")
    print(f"{'=' * 60}")

    location = EarthLocation.of_site('Rubin')
    latitude = location.lat.value
    longitude = location.lon.value
    elevation = location.height.value

    consdb_sql = ConsDbSql(site="usdf")
    query = f"select * from cdb_lsstcam.visit1 where visit_id={obsinfo}"

    try:
        cdb_visit_info = consdb_sql.query(query)
    except Exception as e:
        print(f"ERROR: Could not query ConsDB for visit {obsinfo}: {e}")
        return None

    if len(cdb_visit_info) == 0:
        print(f"WARNING: No data found in ConsDB for visit {obsinfo}")
        return None

    region_cdb = cdb_visit_info["s_region"][0]

    if region_cdb is None:
        print(f"WARNING: No s_region data for visit {obsinfo}, skipping")
        return None

    try:
        region = lsst.sphgeom.Region.from_ivoa_pos("".join(region_cdb.split("ICRS")).upper())
        vertices = [[np.float64(LonLat.longitudeOf(v).asDegrees()),
                     np.float64(LonLat.latitudeOf(v).asDegrees())]
                    for idx, v in enumerate(region.getVertices())]
    except Exception as e:
        print(f"ERROR: Could not parse s_region for visit {obsinfo}: {e}")
        return None

    exp_begin = Time(cdb_visit_info["obs_start"][0])
    exp_end = Time(cdb_visit_info["obs_end"][0])
    exp_time = exp_end - exp_begin

    start_time_jd = (exp_begin - exp_time).jd
    duration = (exp_time.sec * 3)

    zenith_angle = cdb_visit_info["zenith_distance"][0] * u.deg
    azimuth = cdb_visit_info["azimuth"][0] * u.deg
    exp_midpt = Time(cdb_visit_info["exp_midpt"][0])

    ra_center = cdb_visit_info["s_ra"][0]
    dec_center = cdb_visit_info["s_dec"][0]

    try:
        cornersList, detectorsList = getDetRaDecCorners(butler, visit=obsinfo)
    except Exception as e:
        print(f"WARNING: Could not get detector corners for visit {obsinfo}: {e}")
        cornersList = []
        detectorsList = []

    # Make the SatChecker FOV API request
    url_string = f"https://satchecker.cps.iau.org/fov/satellite-passes/?latitude={latitude}&longitude={longitude}&elevation={elevation}&start_time_jd={start_time_jd}&duration={duration}&ra={ra_center}&dec={dec_center}&fov_radius={fov_radius}&group_by=satellite&async=False"

    try:
        response = requests.get(url_string, timeout=60)
        data = response.json()
    except Exception as e:
        print(f"WARNING: SatChecker API request failed for visit {obsinfo}: {e}")
        data = {'data': {'satellites': {}}}

    # Extract RA/Dec positions for each satellite
    satellites = {}
    for sat_key, sat_data in data['data']['satellites'].items():
        if sat_key not in satellites:
            satellites[sat_key] = []
        for position in sat_data['positions']:
            satellites[sat_key].append([
                position['ra'],
                position['dec'],
                position['julian_date']
            ])

    print(f"Public catalog satellites: {len(satellites)} found")

    # VERIFY FERMI POSITION - Do this before plotting
    if fermi_overlaps:
        for i, overlap in enumerate(fermi_overlaps):
            if 'trajectory' in overlap:
                trajectory = overlap['trajectory']
                ra_points = [point['ra'] for point in trajectory]
                dec_points = [point['dec'] for point in trajectory]
                time_points = [point['time'] for point in trajectory]

                print(f"  Fermi trajectory: {len(ra_points)} points")

                # Find closest approach to field center
                distances = []
                for ra_p, dec_p in zip(ra_points, dec_points):
                    d_ra = (ra_p - ra_center) * np.cos(np.radians(dec_center))
                    d_dec = dec_p - dec_center
                    dist = np.sqrt(d_ra ** 2 + d_dec ** 2)
                    distances.append(dist)

                min_idx = np.argmin(distances)
                closest_ra = ra_points[min_idx]
                closest_dec = dec_points[min_idx]
                closest_time = time_points[min_idx]

                print(f"\n  Your matching code says:")
                print(f"    Fermi closest approach at: {closest_time}")
                print(f"    Position: RA={closest_ra:.3f}°, Dec={closest_dec:.3f}°")
                print(f"    Distance from field center: {distances[min_idx]:.3f}°")

                # Time conversion check
                closest_time_astropy = Time(closest_time)
                query_jd = closest_time_astropy.jd

                print(f"\n  Time conversion check:")
                print(f"    Input time string: {closest_time}")
                print(f"    Astropy Time object: {closest_time_astropy.iso}")
                print(f"    Julian Date: {query_jd}")
                print(f"    Visit exposure midpoint: {exp_midpt.iso}")
                print(f"    Visit JD: {exp_midpt.jd}")
                print(f"    Difference: {(closest_time_astropy.jd - exp_midpt.jd) * 86400:.1f} seconds")

                # Check if a known satellite appears (for testing)
                print(f"\n  Testing: Checking if SatChecker finds NOAA 16 DEB (42428)...")

                test_found = False
                for sat_name in satellites.keys():
                    if '42428' in sat_name or 'NOAA 16 DEB' in sat_name.upper():
                        test_found = True
                        print(f"  ✓ TEST SATELLITE FOUND: {sat_name}")
                        print(f"    (This confirms our checking logic works)")

                if not test_found:
                    print(f"  ✗ Hmm, test satellite not found either - something wrong with logic")

                # Check if Fermi appears in the satellites already found
                print(f"\n  Checking if SatChecker finds Fermi during visit exposure...")
                print(f"  (Using same query as main FOV search)")

                fermi_found = False
                for sat_name in satellites.keys():
                    if '33053' in sat_name or 'FERMI' in sat_name.upper() or 'GLAST' in sat_name.upper():
                        fermi_found = True
                        print(f"  ✓ FERMI FOUND in main query: {sat_name}")

                        # Compare position
                        fermi_positions = satellites[sat_name]
                        if fermi_positions:
                            # Find position closest to exposure midpoint
                            mid_jd = exp_midpt.jd
                            closest_pos_idx = min(range(len(fermi_positions)),
                                                  key=lambda i: abs(fermi_positions[i][2] - mid_jd))

                            satchecker_ra = fermi_positions[closest_pos_idx][0]
                            satchecker_dec = fermi_positions[closest_pos_idx][1]
                            satchecker_jd = fermi_positions[closest_pos_idx][2]

                            print(f"  SatChecker position at JD {satchecker_jd}:")
                            print(f"    RA={satchecker_ra:.3f}°, Dec={satchecker_dec:.3f}°")
                            print(f"  Your matching code at closest approach:")
                            print(f"    RA={closest_ra:.3f}°, Dec={closest_dec:.3f}°")

                if not fermi_found:
                    print(f"  ✗ Fermi NOT in the main SatChecker query results")
                    print(f"  → Fermi (NORAD 33053) is likely not in SatChecker's database")
                    print(f"  → SatChecker focuses on Starlink, OneWeb, and debris")
                    print(f"  → NASA science missions may not be included")

    # Create Bokeh figure
    p = figure(
        width=800,
        height=700,
        title=f"Visit {obsinfo} (duration {duration:.1f} s)",
        x_axis_label='Right Ascension (deg)',
        y_axis_label='Declination (deg)',
        match_aspect=True
    )

    # Query FOV circle
    center = SkyCoord(ra=ra_center * u.deg, dec=dec_center * u.deg, frame='icrs')
    position_angles = np.linspace(0, 360, 360) * u.deg
    circle_points = center.directional_offset_by(position_angles, fov_radius * u.deg)

    p.line(circle_points.ra.deg, circle_points.dec.deg,
           color='lightgray', line_dash='dashed', line_width=2,
           legend_label='Query FOV', alpha=0.7)

    # Satellite tracks
    colors = Category20_20
    legend_items = []

    for idx, sat_name in enumerate(satellites.keys()):
        ra = [pos[0] for pos in satellites[sat_name]]
        dec = [pos[1] for pos in satellites[sat_name]]
        times = [pos[2] for pos in satellites[sat_name]]

        color = colors[idx % len(colors)]

        line = p.line(ra, dec, color=color, line_width=2, alpha=0.8)
        scatter = p.scatter(ra, dec, color=color, size=6, alpha=0.8)

        p.add_tools(HoverTool(
            renderers=[scatter],
            tooltips=[
                ("Satellite", sat_name),
                ("RA", "@x{0.000}"),
                ("Dec", "@y{0.000}"),
            ]
        ))

        legend_items.append((sat_name, [line, scatter]))

        if len(ra) > 0:
            if ra[0] < ra[-1]:
                label = Label(x=ra[0], y=dec[0], text=sat_name,
                              text_font_size='8pt', text_color=color)
            else:
                label = Label(x=ra[-1], y=dec[-1], text=sat_name,
                              text_font_size='8pt', text_color=color)
            p.add_layout(label)

    # Camera detectors
    could_not_load_corners = (len(cornersList) == 0 or
                              all(item is None for corners in cornersList for value in corners for item in value))

    for idx, corners in enumerate(cornersList):
        if not any(item is None for value in corners for item in value):
            xs = [c[0] for c in corners] + [corners[0][0]]
            ys = [c[1] for c in corners] + [corners[0][1]]

            p.patch(xs, ys, color="#058B8C", alpha=0.2, line_color="#058B8C", line_width=1)

            center_x = (corners[0][0] + corners[2][0]) / 2
            center_y = (corners[0][1] + corners[2][1]) / 2
            label = Label(x=center_x, y=center_y, text=str(detectorsList[idx]),
                          text_font_size='6pt', text_color='#313333',
                          text_align='center', text_baseline='middle')
            p.add_layout(label)

    if (could_not_load_corners) or (len(cornersList) < 160):
        xs = [v[0] for v in vertices] + [vertices[0][0]]
        ys = [v[1] for v in vertices] + [vertices[0][1]]
        p.patch(xs, ys, color="#058B8C", alpha=0.2, line_color="#058B8C", line_width=1)

    # Plot Fermi overlaps
    if fermi_overlaps:
        fermi_lines = []
        for i, overlap in enumerate(fermi_overlaps):
            if 'trajectory' in overlap:
                trajectory = overlap['trajectory']
                ra_points = [point['ra'] for point in trajectory]
                dec_points = [point['dec'] for point in trajectory]
                time_points = [point['time'] for point in trajectory]

                fermi_line = p.line(ra_points, dec_points,
                                    color='red', line_width=3, alpha=0.9,
                                    line_dash='solid')

                fermi_scatter = p.scatter(ra_points, dec_points,
                                          color='red', size=6, alpha=0.9,
                                          marker='circle')

                source = ColumnDataSource(data=dict(
                    ra=ra_points,
                    dec=dec_points,
                    time=time_points
                ))

                fermi_scatter_hover = p.scatter('ra', 'dec', source=source,
                                                color='red', size=6, alpha=0.9,
                                                marker='circle')

                p.add_tools(HoverTool(
                    renderers=[fermi_scatter_hover],
                    tooltips=[
                        ("Type", "Fermi"),
                        ("RA", "@ra{0.3f}°"),
                        ("Dec", "@dec{0.3f}°"),
                        ("Time", "@time"),
                    ]
                ))

                p.scatter([ra_points[0]], [dec_points[0]],
                          color='darkred', size=10, marker='square', alpha=1.0)
                p.scatter([ra_points[-1]], [dec_points[-1]],
                          color='darkred', size=10, marker='triangle', alpha=1.0)

                if i == 0:
                    fermi_lines.append((f'Fermi ({len(ra_points)} pts)', [fermi_line, fermi_scatter]))

        legend_items.extend(fermi_lines)

    # Configure legend
    legend = Legend(items=legend_items, location="top_left")
    legend.click_policy = "hide"
    p.add_layout(legend, 'right')

    # Invert x-axis (RA)
    p.x_range.flipped = True

    # Add grid
    p.grid.grid_line_color = 'lightgray'
    p.grid.grid_line_dash = 'dashed'

    return p


# Main execution
yaml_file = 'fermi_closest.yaml'
fermi_data = load_fermi_overlaps(yaml_file)

# LSSTCam configuration
instrument = 'LSSTCam'
repo = '/repo/main'
collections = ['LSSTCam/defaults']
butler = dafButler.Butler(repo, collections=collections, instrument=instrument)

# Create plots for all visits
plots = []
visit_ids = []

for visit_info in fermi_data['visits']:
    obsinfo = visit_info['visit_id']
    fermi_overlaps = visit_info.get('overlaps', [])

    p = create_visit_plot(obsinfo, fermi_overlaps, butler, collections, instrument)

    if p is not None:  # Only add successful plots
        plots.append(p)
        visit_ids.append(obsinfo)
    else:
        print(f"Skipping visit {obsinfo} due to errors")

if len(plots) == 0:
    print("ERROR: No valid plots generated!")
else:
    # Create grid layout (2 columns)
    grid = gridplot(plots, ncols=2, sizing_mode='scale_width')

    # Save to HTML
    output_filename = f"fermi_passes_{'_'.join(map(str, visit_ids[:3]))}_etc.html"
    output_file(output_filename)
    save(grid)

    print(f"\n{'=' * 60}")
    print(f"Successfully created {len(plots)} plots, saved to {output_filename}")
    print(f"{'=' * 60}")
