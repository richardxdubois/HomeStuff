#!/usr/bin/env python

"""
FT2 Data Gap and Location Analysis Tool (with SAA Overlay)

This script reads a Fermi FT2 file and analyzes telemetry gaps.
It produces a single, combined HTML file containing two plots:

1. A time series of gap durations ("swiss cheese" map).
2. A Latitude vs. Longitude scatter plot showing where gaps begin.

It now includes an optional feature to overlay a provided SAA boundary outline
on the geographic plot for definitive comparison.

Usage:
    python check_ft2_gaps.py /path/to/file.fits --saa-outline /path/to/saa.csv
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


def analyze_gaps_geographically(ft2_path: str, saa_outline_file: str = None):
    """
    Loads an FT2 file, finds time gaps, and generates a combined report plot,
    optionally overlaying a provided SAA boundary.
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

    # --- Step 4: Create Both Plots (but don't save yet) ---
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

    # --- NEW: Load and plot the SAA outline if the file is provided ---
    if saa_outline_file:
        print(f"Loading and overlaying SAA outline from: {saa_outline_file}")
        try:
            # Assumes a simple two-column text file: lon,lat
            saa_lon, saa_lat = np.loadtxt(saa_outline_file, delimiter=',', unpack=True)
            p_map.line(saa_lon, saa_lat, line_width=3, color='cyan', alpha=0.8, legend_label="SAA Outline")
        except Exception as e:
            print(f"WARNING: Could not load or plot the SAA outline file. Error: {e}")
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
    # --- NEW: Add argument for the SAA outline file ---
    parser.add_argument('--saa-outline', help='Path to a CSV file (lon,lat) defining the SAA outline.')

    args = parser.parse_args()

    file_path = args.ft2_file if args.ft2_file is not None else args.ft2_file_named

    if not file_path:
        parser.error("No FT2 file specified.")

    # Pass the new argument to the main function
    analyze_gaps_geographically(file_path, saa_outline_file=args.saa_outline)
