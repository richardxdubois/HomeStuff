#!/usr/bin/env python

"""
FT2 vs. TLE Position Comparison Tool (Corrected for True Overlap)

This script validates the TLE-based analysis by comparing its predictions
against the latest available high-precision FT2 ephemeris. It uses the
cached TLE from the main analysis and compares it against the final
data segment of the FT2 file, providing a real-world measure of the
propagation error over a ~2-week period.
"""

import argparse
import numpy as np
from astropy.io import fits
from astropy.time import Time
import astropy.units as u
from skyfield.api import load
from bokeh.plotting import figure, save, output_file
from bokeh.models import HoverTool


def compare_positions(ft2_path: str):
    print(f"--- Loading FT2 file: {ft2_path} ---")

    # --- Step 1: Load and Sort FT2 Data ---
    with fits.open(ft2_path) as hdul:
        data_table = hdul['SC_DATA'].data
        met_times_raw = np.array(data_table['START'])
        positions_raw = np.array(data_table['SC_POSITION'])

    fermi_epoch = Time('2001-01-01T00:00:00', scale='utc')
    times_unsorted = fermi_epoch + met_times_raw * u.s
    sort_indices = np.argsort(times_unsorted)
    ft2_times = times_unsorted[sort_indices]
    ft2_positions = positions_raw[sort_indices]
    print(f"Loaded {len(ft2_times)} records from FT2 file.")
    print(f"FT2 data runs from {ft2_times[0].iso} to {ft2_times[-1].iso}")

    # --- Step 2: Load the OPERATIONAL TLE Data (from cache) ---
    print("\n--- Loading LIVE TLE data for Fermi from Celestrak cache ---")
    load.verbose = True
    ts = load.timescale()
    TLE_URL = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=science&FORMAT=tle'
    try:
        # reload=False ensures we use the existing cache from the Rubin_Fermi run
        satellites = load.tle_file(TLE_URL, reload=False)
        FERMI_NORAD_ID = 33053
        fermi_satellite = {sat.model.satnum: sat for sat in satellites}[FERMI_NORAD_ID]
        print(f"\nSuccessfully loaded TLE for: {fermi_satellite.name}")
        print(f"The loaded TLE has an epoch (is most accurate) on: {fermi_satellite.epoch.to_astropy().iso}")
    except Exception as e:
        print(f"FATAL ERROR: Could not load TLE data. Details: {e}")
        return

    # --- Step 3: Define the Correct Comparison Window ---
    # We will use the LAST 10,000 data points from the FT2 file.
    # This guarantees the data exists and is the closest in time to the TLE epoch.
    num_points_to_compare = 10000
    if len(ft2_times) < num_points_to_compare:
        print(f"Warning: FT2 file has fewer than {num_points_to_compare} points. Using all available.")
        num_points_to_compare = len(ft2_times)

    print(f"\nComparing the last {num_points_to_compare} data points of the FT2 file.")

    # Slice the arrays from the end to get the latest data
    comparison_times = ft2_times[-num_points_to_compare:]
    comparison_ft2_positions = ft2_positions[-num_points_to_compare:]
    print(f"Comparison time range: {comparison_times[0].iso} to {comparison_times[-1].iso}")

    # --- Step 4: Generate TLE positions for these specific timestamps ---
    print("\nGenerating TLE positions for the FT2 timestamps...")
    t_skyfield = ts.from_astropy(comparison_times)
    tle_geocentric = fermi_satellite.at(t_skyfield)
    tle_positions = tle_geocentric.position.m.T

    # --- Step 5: Calculate the Separation Distance ---
    print("Calculating 3D separation...")
    delta_vectors = comparison_ft2_positions - tle_positions
    separation_km = np.linalg.norm(delta_vectors, axis=1) / 1000.0

    # --- Step 6: Print Summary Statistics ---
    print("\n" + "=" * 60)
    print("          FT2 vs. TLE Position Separation Report")
    print("=" * 60)
    print(f"Mean Separation:           {np.mean(separation_km):.3f} km")
    print(f"Median Separation:         {np.median(separation_km):.3f} km")
    print(f"Maximum Separation:        {np.max(separation_km):.3f} km")
    print(f"Minimum Separation:        {np.min(separation_km):.3f} km")
    print("=" * 60 + "\n")

    # --- Step 7: Generate Plot ---
    print("Generating plot of separation distance over time...")
    output_filename = "ft2_vs_tle_separation.html"
    p = figure(width=1200, height=600, title="Separation Distance (FT2 vs. TLE near end of FT2 data)",
               x_axis_label="Time (UTC)", y_axis_label="Separation (km)", x_axis_type='datetime')
    p.line(comparison_times.datetime, separation_km, line_width=2, legend_label="FT2 vs. TLE Separation")
    p.add_tools(
        HoverTool(tooltips=[("Time", "@x{%F %T}"), ("Separation", "@y{0.3f} km")], formatters={'@x': 'datetime'}))
    output_file(filename=output_filename)
    save(p)
    print(f"SUCCESS: Comparison plot saved to '{output_filename}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('ft2_file', type=str, help='Path to the FT2 FITS file to analyze.')
    args = parser.parse_args()
    compare_positions(args.ft2_file)
