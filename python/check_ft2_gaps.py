#!/usr/bin/env python

"""
FT2 Data Gap and Location Analysis Tool (with SAA and Rubin Overlay)

This script reads a Fermi FT2 file and analyzes telemetry gaps. It produces
a single, combined HTML file containing two plots:

1. A time series of gap durations ("swiss cheese" map).
2. A Latitude vs. Longitude scatter plot showing where gaps begin, with the
   official Fermi SAA operational boundary and Rubin Observatory's location overlaid.

This script has no external map data dependencies.

Usage:
    python check_ft2_gaps.py /path/to/your/ft2_file.fits
"""
import argparse
import numpy as np
from astropy.io import fits
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord, ITRS
from bokeh.plotting import figure, save, output_file
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.layouts import column


def analyze_gaps_geographically(ft2_path: str):
    """
    Loads an FT2 file, finds time gaps, and generates a combined report plot,
    overlaying the official Fermi SAA boundary and a marker for Rubin Observatory.
    """
    # ... (Steps 1, 2, and 3 are unchanged) ...
    print(f"--- Loading FT2 file: {ft2_path} ---")

    # Step 1: Load and Sort Data
    try:
        with fits.open(ft2_path) as hdul:
            data_table = hdul['SC_DATA'].data
            met_times_raw = np.array(data_table['START'])
            positions_raw = np.array(data_table['SC_POSITION'])
    except Exception as e:
        print(f"ERROR: Could not read FITS file. Details: {e}")
        return

    fermi_epoch = Time('2001-01-01T00:00:00', scale='utc')
    times_unsorted = fermi_epoch + met_times_raw * u.s

    print("Sorting data by time...")
    sort_indices = np.argsort(times_unsorted)
    times = times_unsorted[sort_indices]
    positions = positions_raw[sort_indices]
    print(f"Loaded and sorted {len(times)} records.")

    # Step 2: Identify Gaps
    print("Identifying data gaps...")
    time_diffs_sec = np.diff(times.mjd) * 86400
    gap_threshold_sec = 60.0
    gap_indices = np.where(time_diffs_sec > gap_threshold_sec)[0]

    if len(gap_indices) == 0:
        print("SUCCESS: No significant data gaps found.")
        return

    num_gaps = len(gap_indices)
    gap_durations_sec = time_diffs_sec[gap_indices]
    gap_start_times = times[gap_indices]

    # Step 3: Get Geographic Location of Gap Starts
    print("Calculating geographic start locations for each gap...")
    gap_start_positions = positions[gap_indices]

    gap_starts_gcrs = SkyCoord(
        x=gap_start_positions[:, 0] * u.m, y=gap_start_positions[:, 1] * u.m, z=gap_start_positions[:, 2] * u.m,
        representation_type='cartesian', frame='gcrs', obstime=gap_start_times
    )
    gap_starts_itrs = gap_starts_gcrs.transform_to(ITRS(obstime=gap_start_times))
    gap_lons = gap_starts_itrs.earth_location.lon.wrap_at('180d').deg
    gap_lats = gap_starts_itrs.earth_location.lat.deg

    # --- Step 4: Create Both Plots ---
    print("Generating plot components...")

    # Plot 1: Time Series
    total_gap_time_days = np.sum(gap_durations_sec) / 86400
    ts_title = f"Map of {num_gaps} Data Gaps (> {gap_threshold_sec}s) | Total Missing Time: {total_gap_time_days:.1f} days"
    p_ts = figure(width=1200, height=500, title=ts_title, x_axis_type='datetime', y_axis_type='log',
                  x_axis_label="Date", y_axis_label="Gap Duration (seconds)")
    p_ts.scatter(gap_start_times.datetime, gap_durations_sec, size=5, color="navy", alpha=0.6)
    p_ts.add_tools(
        HoverTool(tooltips=[("Start", "@x{%F %T}"), ("Duration", "@y{0.1f} s")], formatters={'@x': 'datetime'}))

    # Plot 2: Geographic Scatter
    p_map = figure(
        width=1200, height=600,
        title=f"Geographic Location of {num_gaps} Data Gap Starts",
        x_axis_label="Longitude", y_axis_label="Latitude",
        x_range=(-180, 180), y_range=(-90, 90)
    )
    map_source = ColumnDataSource(
        data={'lon': gap_lons, 'lat': gap_lats, 'duration': gap_durations_sec, 'time': gap_start_times.iso})
    p_map.scatter('lon', 'lat', source=map_source, size=5, color="red", alpha=0.5, legend_label="Gap Start Location")
    p_map.add_tools(HoverTool(
        tooltips=[("Lon, Lat", "(@lon{0.2f}, @lat{0.2f})"), ("Time", "@time"), ("Gap Duration", "@duration{0.0f} s")]))

    # Official Fermi SAA Outline
    saa_longitude = [33.9, -30.0, -36.0, -42.0, -58.8, -97.5, -98.5, -86.1, -60.0, -40.0, -20.0, 0.0, 33.9]
    saa_latitude = [-30.0, 3.0, 4.6, 4.6, 0.7, -9.9, -12.5, -30.0, -30.0, -30.0, -30.0, -30.0, -30.0]
    p_map.line(saa_longitude, saa_latitude, line_width=3, color='cyan', alpha=0.8, legend_label="Official SAA Outline")

    # --- NEW: Add a marker for Rubin Observatory ---
    rubin_lon = -70.74755
    rubin_lat = -30.24463
    p_map.scatter(x=[rubin_lon], y=[rubin_lat], marker='star', size=20, color='gold',
                  line_color='black', line_width=1, legend_label='Rubin Observatory')
    # --- END NEW SECTION ---

    p_map.legend.location = "top_left"
    p_map.legend.click_policy = "hide"

    # --- Step 5: Combine Plots and Save ---
    print("Combining plots and saving to HTML...")
    final_layout = column(p_ts, p_map)
    output_filename = "FT2_gap_report_combined.html"
    output_file(filename=output_filename)
    save(final_layout)

    print(f"SUCCESS: Combined report saved to '{output_filename}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('ft2_file', nargs='?', default=None, help='Path to the FT2 FITS file to analyze.')
    parser.add_argument('--ft2_file', dest='ft2_file_named',
                        help='Path to the FT2 FITS file to analyze (alternative flag).')
    args = parser.parse_args()

    file_path = args.ft2_file if args.ft2_file is not None else args.ft2_file_named

    if not file_path:
        parser.error("No FT2 file specified.")

    analyze_gaps_geographically(file_path)
