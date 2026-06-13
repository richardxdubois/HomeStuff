from dataclasses import dataclass
from astropy.io import fits
from astropy.coordinates import EarthLocation, AltAz, ICRS, GCRS, UnitSphericalRepresentation
from astropy.time import Time
import astropy.units as u
import numpy as np
import pickle
import json

@dataclass
class FermiPass:
    """Container for a single Fermi pass through survey area"""
    start_time: Time
    end_time: Time
    times: np.ndarray  # Time samples during pass
    ra: np.ndarray  # RA trajectory (deg)
    dec: np.ndarray  # Dec trajectory (deg)
    alt: np.ndarray  # Altitude trajectory (deg)
    az: np.ndarray  # Azimuth trajectory (deg)
    max_angular_velocity: float  # deg/sec


class FermiPassCalculator:
    def __init__(self, ft2_file, start_time=None, end_time=None):
        """
        Parameters:
        -----------
        ft2_file : str
            Path to FT2 FITS file
        start_time : astropy.Time or str, optional
            Start of time range to process
        end_time : astropy.Time or str, optional
            End of time range to process
        """
        self.ft2_file = ft2_file
        self.start_time = Time(start_time) if start_time else None
        self.end_time = Time(end_time) if end_time else None

        # Cerro Pachón location
        self.rubin_location = EarthLocation(
            lat=-30.24463 * u.deg,
            lon=-70.74755 * u.deg,
            height=2647 * u.m
        )

        self.times = None
        self.positions = None  # spacecraft positions
        self.sky_coords = None  # computed RA/Dec/Alt/Az
        self.passes = []

    def load_ft2(self, downsample_sec=30):
        """Load FT2 data, filtering by time range and downsampling"""
        with fits.open(self.ft2_file) as hdul:
            sc_data = hdul['SC_DATA'].data

            print("Available columns in FT2:")
            print(sc_data.dtype.names)

            # Get times
            met_times = sc_data['START']
            fermi_epoch = Time('2001-01-01T00:00:00', format='isot', scale='utc')
            times = fermi_epoch + met_times * u.s

            # Filter by time range
            if self.start_time or self.end_time:
                mask = np.ones(len(times), dtype=bool)
                if self.start_time:
                    mask &= (times >= self.start_time)
                if self.end_time:
                    mask &= (times <= self.end_time)

                times = times[mask]
                sc_data = sc_data[mask]

            # Downsample
            if len(times) > 1:
                current_cadence = np.median(np.diff(times.mjd)) * 86400
                downsample_factor = max(1, int(downsample_sec / current_cadence))

                times = times[::downsample_factor]
                sc_data = sc_data[::downsample_factor]

            self.times = times
            self.positions = sc_data['SC_POSITION']

            # Check position magnitudes and changes
            pos_mag = np.sqrt(np.sum(self.positions ** 2, axis=1))
            print(f"  Position magnitude range: {pos_mag.min():.0f} to {pos_mag.max():.0f} m")
            print(f"  Expected: ~6,900,000 m for Fermi orbit")

            # Check if positions are actually changing
            pos_diff = np.diff(self.positions, axis=0)
            pos_diff_mag = np.sqrt(np.sum(pos_diff ** 2, axis=1))
            print(f"  Position change per sample:")
            print(f"    Min: {pos_diff_mag.min():.0f} m")
            print(f"    Max: {pos_diff_mag.max():.0f} m")
            print(f"    Mean: {pos_diff_mag.mean():.0f} m")
            print(f"    Expected: ~450,000 m per 60s at 7.5 km/s orbital velocity")

            print(f"  First 3 position vectors (km):")
            for i in range(3):
                print(f"    {self.positions[i] / 1000}")

            # Check if velocity exists and verify motion
            if 'SC_VELOCITY' in sc_data.columns.names:
                velocities = sc_data['SC_VELOCITY']
                vel_mag = np.sqrt(np.sum(velocities ** 2, axis=1))
                print(f"  Velocity magnitude range: {vel_mag.min():.0f} to {vel_mag.max():.0f} m/s")
                print(f"  First velocity: {velocities[0]}")

    def compute_topocentric(self):
        """Convert spacecraft positions to RA/Dec/Alt/Az from Rubin"""
        if self.times is None or self.positions is None:
            raise ValueError("Must call load_ft2() first")

        from astropy.coordinates import ITRS, CartesianRepresentation

        # Get Rubin's ITRS coordinates at each time
        rubin_itrs = self.rubin_location.get_itrs(obstime=self.times)

        # Convert Rubin from ITRS (Earth-fixed) to GCRS (inertial)
        rubin_gcrs = rubin_itrs.transform_to(GCRS(obstime=self.times))

        # Fermi position in GCRS (SC_POSITION is already in ECI ~ GCRS)
        fermi_x = self.positions[:, 0] * u.m
        fermi_y = self.positions[:, 1] * u.m
        fermi_z = self.positions[:, 2] * u.m

        # DIAGNOSTIC: Check Fermi position changes
        fermi_mag = np.sqrt(fermi_x ** 2 + fermi_y ** 2 + fermi_z ** 2)
        print(f"\nFermi position diagnostics:")
        print(f"  Magnitude range: {fermi_mag.min():.0f} to {fermi_mag.max():.0f}")

        # Check first few Fermi positions vs Rubin positions
        print(f"  First 3 Fermi positions (GCRS, km):")
        for i in range(3):
            print(f"    [{fermi_x[i].value / 1000:.1f}, {fermi_y[i].value / 1000:.1f}, {fermi_z[i].value / 1000:.1f}]")

        print(f"  First 3 Rubin positions (GCRS, km):")
        for i in range(3):
            rx = rubin_gcrs.cartesian.x[i].to(u.km).value
            ry = rubin_gcrs.cartesian.y[i].to(u.km).value
            rz = rubin_gcrs.cartesian.z[i].to(u.km).value
            print(f"    [{rx:.1f}, {ry:.1f}, {rz:.1f}]")

        # Vector FROM Rubin TO Fermi
        dx = fermi_x - rubin_gcrs.cartesian.x
        dy = fermi_y - rubin_gcrs.cartesian.y
        dz = fermi_z - rubin_gcrs.cartesian.z

        print(f"  First 3 topocentric vectors (km):")
        for i in range(3):
            print(f"    [{dx[i].to(u.km).value:.1f}, {dy[i].to(u.km).value:.1f}, {dz[i].to(u.km).value:.1f}]")

        # Create topocentric GCRS coordinates
        topo_gcrs = GCRS(
            CartesianRepresentation(dx, dy, dz),
            obstime=None
        )

        # Transform to ICRS for RA/Dec
        icrs_coords = topo_gcrs.transform_to(ICRS())

        print(f"  First 3 RA/Dec:")
        for i in range(3):
            print(f"    RA={icrs_coords.ra[i].deg:.6f}, Dec={icrs_coords.dec[i].deg:.6f}")

        # Transform to AltAz
        altaz_coords = topo_gcrs.transform_to(AltAz(obstime=self.times, location=self.rubin_location))

        topo_spherical = CartesianRepresentation(dx, dy, dz).represent_as(UnitSphericalRepresentation)

        # Store results
        self.sky_coords = {
            'ra': topo_spherical.lon.deg,
            'dec': topo_spherical.lat.deg,
            'alt': altaz_coords.alt.deg,
            'az': altaz_coords.az.deg
        }

    def find_survey_passes(self, min_altitude=20 * u.deg,
                           dec_range=(-70, 10) * u.deg,
                           min_angular_velocity=0.1):
        """
        Identify continuous passes through Rubin survey area with significant motion

        Parameters:
        -----------
        min_altitude : astropy.units.Quantity
            Minimum altitude above horizon
        dec_range : tuple of astropy.units.Quantity
            (min_dec, max_dec) for survey area
        min_angular_velocity : float
            Minimum angular velocity in deg/s to consider a valid pass

        Returns:
        --------
        list of FermiPass objects
        """
        if self.sky_coords is None:
            raise ValueError("Must call compute_topocentric() first")

        # First compute angular velocities for all points
        ra = self.sky_coords['ra']
        dec = self.sky_coords['dec']

        # Compute angular separations between consecutive points
        ra_diff = np.diff(ra)
        dec_diff = np.diff(dec)

        # Handle RA wraparound
        ra_diff = np.where(ra_diff > 180, ra_diff - 360, ra_diff)
        ra_diff = np.where(ra_diff < -180, ra_diff + 360, ra_diff)

        # Angular separation (small angle approximation with cos(dec) correction)
        dec_avg = (dec[:-1] + dec[1:]) / 2
        ang_sep = np.sqrt(
            (ra_diff * np.cos(np.radians(dec_avg))) ** 2 +
            dec_diff ** 2
        )

        # Time differences in seconds
        time_diff = np.diff(self.times.mjd) * 86400

        # Angular velocity in deg/sec (pad with 0 at end to match array length)
        ang_velocity = np.concatenate([ang_sep / time_diff, [0.0]])

        print(f"\nVelocity calculation diagnostics:")
        print(f"  Number of ang_sep values: {len(ang_sep)}")
        print(f"  ang_sep range: {ang_sep.min():.6f} to {ang_sep.max():.6f} deg")
        print(f"  time_diff range: {time_diff.min():.2f} to {time_diff.max():.2f} sec")
        print(f"  First 5 ang_sep: {ang_sep[:5]}")
        print(f"  First 5 time_diff: {time_diff[:5]}")
        print(f"  First 5 ang_velocity: {ang_velocity[:5]}")
        print(f"  Max ang_velocity overall: {ang_velocity.max():.6f}")

        # Create combined mask: in survey area AND moving fast enough
        alt_mask = self.sky_coords['alt'] >= min_altitude.value
        dec_mask = ((self.sky_coords['dec'] >= dec_range[0].value) &
                    (self.sky_coords['dec'] <= dec_range[1].value))
        velocity_mask = ang_velocity >= min_angular_velocity

        in_survey = alt_mask & dec_mask & velocity_mask

        # Find continuous segments
        transitions = np.diff(in_survey.astype(int))
        starts = np.where(transitions == 1)[0] + 1
        ends = np.where(transitions == -1)[0] + 1

        # Handle edge cases
        if in_survey[0]:
            starts = np.concatenate([[0], starts])
        if in_survey[-1]:
            ends = np.concatenate([ends, [len(in_survey)]])

        # Create FermiPass objects
        self.passes = []
        for start_idx, end_idx in zip(starts, ends):
            pass_obj = FermiPass(
                start_time=self.times[start_idx],
                end_time=self.times[end_idx - 1],
                times=self.times[start_idx:end_idx],
                ra=self.sky_coords['ra'][start_idx:end_idx],
                dec=self.sky_coords['dec'][start_idx:end_idx],
                alt=self.sky_coords['alt'][start_idx:end_idx],
                az=self.sky_coords['az'][start_idx:end_idx],
                max_angular_velocity=0.0  # Will compute in next method
            )
            self.passes.append(pass_obj)

        return self.passes

    def compute_pass_parameters(self):
        """Calculate detailed parameters for each pass (angular velocity, etc)"""
        for pass_obj in self.passes:
            if len(pass_obj.times) < 2:
                pass_obj.max_angular_velocity = 0.0
                continue

            # Compute angular separations between consecutive points
            # Use simple great circle distance approximation
            ra_diff = np.diff(pass_obj.ra)
            dec_diff = np.diff(pass_obj.dec)

            # Handle RA wraparound (360/0 boundary)
            ra_diff = np.where(ra_diff > 180, ra_diff - 360, ra_diff)
            ra_diff = np.where(ra_diff < -180, ra_diff + 360, ra_diff)

            # Angular separation (small angle approximation with cos(dec) correction)
            dec_avg = (pass_obj.dec[:-1] + pass_obj.dec[1:]) / 2
            ang_sep = np.sqrt(
                (ra_diff * np.cos(np.radians(dec_avg))) ** 2 +
                dec_diff ** 2
            )

            # Time differences in seconds
            time_diff = np.diff(pass_obj.times.mjd) * 86400  # convert days to seconds

            # Angular velocity in deg/sec
            ang_velocity = ang_sep / time_diff

            pass_obj.max_angular_velocity = np.max(ang_velocity)


    def filter_night_passes(self, sun_alt_limit=-18 * u.deg):
        """
        Filter passes to only include those during astronomical night

        Parameters:
        -----------
        sun_alt_limit : astropy.units.Quantity
            Maximum sun altitude (typically -12 or -18 deg for twilight)
        """
        from astropy.coordinates import get_sun

        night_passes = []

        print(f"\nChecking sun altitude for {len(self.passes)} passes:")
        for i, pass_obj in enumerate(self.passes):
            # Get sun position at start and end of pass
            sun_start = get_sun(pass_obj.start_time).transform_to(
                AltAz(obstime=pass_obj.start_time, location=self.rubin_location)
            )
            sun_end = get_sun(pass_obj.end_time).transform_to(
                AltAz(obstime=pass_obj.end_time, location=self.rubin_location)
            )

            sun_alt_max = max(sun_start.alt.deg, sun_end.alt.deg)

            print(f"  Pass {i + 1} ({pass_obj.start_time.iso}): sun alt = {sun_alt_max:.1f}°", end="")

            # Keep pass if sun is below limit for entire duration
            if sun_start.alt < sun_alt_limit and sun_end.alt < sun_alt_limit:
                night_passes.append(pass_obj)
                print(" ✓ NIGHT")
            else:
                print(" ✗ DAY/TWILIGHT")

        self.passes = night_passes
        return self.passes


    import pickle
    import json
    from pathlib import Path


    def save_passes(self, output_dir, file_prefix="fermi_passes"):
        """
        Save pass data in multiple formats

        Parameters:
        -----------
        output_dir : str or Path
            Directory to save output files
        file_prefix : str
            Prefix for output filenames
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as pickle for easy Python loading
        pickle_file = output_dir / f"{file_prefix}.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(self.passes, f)
        print(f"Saved {len(self.passes)} passes to {pickle_file}")

        # Save as JSON for readability/portability
        json_data = []
        for i, p in enumerate(self.passes):
            json_data.append({
                'pass_number': i + 1,
                'start_time': p.start_time.iso,
                'end_time': p.end_time.iso,
                'duration_sec': (p.end_time - p.start_time).sec,
                'n_samples': len(p.times),
                'ra_range': [float(p.ra.min()), float(p.ra.max())],
                'dec_range': [float(p.dec.min()), float(p.dec.max())],
                'alt_range': [float(p.alt.min()), float(p.alt.max())],
                'max_angular_velocity': float(p.max_angular_velocity)
            })

        json_file = output_dir / f"{file_prefix}.json"
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"Saved summary to {json_file}")

        # Save as CSV for spreadsheet compatibility
        import csv
        csv_file = output_dir / f"{file_prefix}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Pass', 'Start Time', 'End Time', 'Duration (s)',
                             'Samples', 'RA Min', 'RA Max', 'Dec Min', 'Dec Max',
                             'Alt Min', 'Alt Max', 'Max Velocity (deg/s)'])
            for i, p in enumerate(self.passes):
                writer.writerow([
                    i + 1,
                    p.start_time.iso,
                    p.end_time.iso,
                    (p.end_time - p.start_time).sec,
                    len(p.times),
                    f"{p.ra.min():.2f}",
                    f"{p.ra.max():.2f}",
                    f"{p.dec.min():.2f}",
                    f"{p.dec.max():.2f}",
                    f"{p.alt.min():.2f}",
                    f"{p.alt.max():.2f}",
                    f"{p.max_angular_velocity:.4f}"
                ])
        print(f"Saved CSV to {csv_file}")

    def load_passes(self, pickle_file):
        """
        Load passes from a previously saved pickle file

        Parameters:
        -----------
        pickle_file : str or Path
            Path to pickle file containing saved passes
        """
        pickle_file = Path(pickle_file)

        if not pickle_file.exists():
            raise FileNotFoundError(f"Pickle file not found: {pickle_file}")

        with open(pickle_file, 'rb') as f:
            self.passes = pickle.load(f)

        print(f"Loaded {len(self.passes)} passes from {pickle_file}")
        return self.passes

    def visualize_passes(self, output_dir, file_prefix="fermi_passes"):
        """
        Create Bokeh visualizations of Fermi passes

        Parameters:
        -----------
        output_dir : str or Path
            Directory to save output files
        file_prefix : str
            Prefix for output filenames
        """
        from bokeh.plotting import figure, save, output_file
        from bokeh.models import HoverTool, ColumnDataSource
        from bokeh.layouts import column, row
        from bokeh.palettes import Viridis256
        import numpy as np

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if len(self.passes) == 0:
            print("No passes to visualize")
            return

        # Create sky map of all pass trajectories
        sky_plot = figure(
            width=900,
            height=600,
            title="Fermi Pass Trajectories Through Survey Area",
            x_axis_label="RA (degrees)",
            y_axis_label="Dec (degrees)",
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )

        # Plot each pass as a line
        colors = Viridis256[::len(Viridis256) // min(len(self.passes), 256)]

        for i, p in enumerate(self.passes):
            color = colors[i % len(colors)]
            # Only add legend for first 10 passes to avoid clutter
            if i < 10:
                sky_plot.line(p.ra, p.dec,
                              line_width=2,
                              alpha=0.6,
                              color=color,
                              legend_label=f"Pass {i + 1}")
            else:
                sky_plot.line(p.ra, p.dec,
                              line_width=2,
                              alpha=0.6,
                              color=color)

        # Add survey area boundaries
        sky_plot.line([0, 360], [-70, -70], line_dash='dashed',
                      color='red', alpha=0.3, legend_label="Survey bounds")
        sky_plot.line([0, 360], [10, 10], line_dash='dashed',
                      color='red', alpha=0.3)

        sky_plot.legend.click_policy = "hide"
        sky_plot.legend.location = "top_right"

        # Create timeline plot
        timeline_plot = figure(
            width=900,
            height=400,
            title="Pass Timeline",
            x_axis_label="Date",
            y_axis_label="Duration (minutes)",
            x_axis_type='datetime',
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )

        # Prepare data for timeline
        start_times = [p.start_time.datetime for p in self.passes]
        durations = [(p.end_time - p.start_time).sec / 60.0 for p in self.passes]
        max_velocities = [p.max_angular_velocity for p in self.passes]
        pass_numbers = list(range(1, len(self.passes) + 1))

        # Color by velocity
        from bokeh.transform import linear_cmap
        mapper = linear_cmap(field_name='velocity',
                             palette=Viridis256,
                             low=min(max_velocities),
                             high=max(max_velocities))

        source = ColumnDataSource(data={
            'start': start_times,
            'duration': durations,
            'velocity': max_velocities,
            'pass_num': pass_numbers
        })

        timeline_plot.scatter('start', 'duration', size=10, source=source,
                             color=mapper, alpha=0.8)

        # Add hover tool
        hover = HoverTool(tooltips=[
            ("Pass", "@pass_num"),
            ("Start", "@start{%F %T}"),
            ("Duration", "@duration{0.1f} min"),
            ("Max Velocity", "@velocity{0.3f} deg/s")
        ], formatters={'@start': 'datetime'})

        timeline_plot.add_tools(hover)

        # Create velocity distribution histogram
        hist_plot = figure(
            width=900,
            height=400,
            title="Angular Velocity Distribution",
            x_axis_label="Max Angular Velocity (deg/s)",
            y_axis_label="Number of Passes",
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )

        hist, edges = np.histogram(max_velocities, bins=20)
        hist_plot.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
                       fill_color="navy", alpha=0.7, line_color="white")

        # Save combined layout
        layout = column(sky_plot, timeline_plot, hist_plot)

        html_file = output_dir / f"{file_prefix}_visualization.html"
        output_file(html_file)
        save(layout)

        print(f"Saved visualization to {html_file}")

import argparse
import yaml
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Calculate Fermi satellite passes through Rubin survey area'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML configuration file'
    )

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    mode = config.get('mode', 'compute')
    output_dir = config.get('output_dir', None)
    verbose = config.get('verbose', True)

    if mode == 'visualize':
        # Visualization-only mode
        passes_file = config.get('passes_file')
        if not passes_file:
            raise ValueError("In visualize mode, must specify 'passes_file' in config")

        if verbose:
            print(f"Loading passes from {passes_file}")

        # Create calculator just for visualization
        calculator = FermiPassCalculator(ft2_file=None)
        calculator.load_passes(passes_file)

        if output_dir:
            if verbose:
                print(f"\nCreating visualizations in {output_dir}")
            calculator.visualize_passes(output_dir)
        else:
            print("Warning: No output_dir specified, skipping visualization save")

    elif mode == 'compute':
        # Full computation mode
        ft2_file = config['ft2_file']
        start_time = config.get('start_time', None)
        end_time = config.get('end_time', None)
        downsample_sec = config.get('downsample_sec', 30)
        min_altitude = config.get('min_altitude', 20)
        dec_range = config.get('dec_range', [-70, 10])
        min_angular_velocity = config.get('min_angular_velocity', 0.1)
        sun_alt_limit = config.get('sun_alt_limit', -18)

        # Run calculation
        if verbose:
            print(f"Loading FT2 file: {ft2_file}")
        calculator = FermiPassCalculator(ft2_file, start_time, end_time)

        if verbose:
            print(f"Loading and downsampling to {downsample_sec}s cadence...")
        calculator.load_ft2(downsample_sec=downsample_sec)
        if verbose:
            print(f"  Loaded {len(calculator.times)} time samples")
            print(f"  Time range: {calculator.times[0].iso} to {calculator.times[-1].iso}")
            print(f"  Duration: {(calculator.times[-1] - calculator.times[0]).to(u.day):.1f}")

        if verbose:
            print("Computing topocentric coordinates...")
        calculator.compute_topocentric()

        if verbose:
            # Diagnostic: check if coordinates are changing at all
            print(f"\nDiagnostic info:")
            print(f"  Total time span: {calculator.times[0].iso} to {calculator.times[-1].iso}")
            print(f"  Number of samples: {len(calculator.times)}")
            print(f"  RA range: {calculator.sky_coords['ra'].min():.2f} to {calculator.sky_coords['ra'].max():.2f}")
            print(f"  Dec range: {calculator.sky_coords['dec'].min():.2f} to {calculator.sky_coords['dec'].max():.2f}")
            print(f"  Alt range: {calculator.sky_coords['alt'].min():.2f} to {calculator.sky_coords['alt'].max():.2f}")

            # Check how many points are above horizon
            above_horizon = np.sum(calculator.sky_coords['alt'] > 0)
            above_20 = np.sum(calculator.sky_coords['alt'] > 20)
            print(f"  Points above horizon (alt > 0): {above_horizon}")
            print(f"  Points above 20 deg: {above_20}")

            # Check a few consecutive samples
            print(f"\nFirst 5 positions:")
            for i in range(min(5, len(calculator.times))):
                print(
                    f"    {calculator.times[i].iso}: RA={calculator.sky_coords['ra'][i]:.6f}, Dec={calculator.sky_coords['dec'][i]:.6f}, Alt={calculator.sky_coords['alt'][i]:.2f}")

        if verbose:
            print(f"Finding passes (alt > {min_altitude}°, dec in {dec_range})...")
        passes = calculator.find_survey_passes(
            min_altitude=min_altitude * u.deg,
            dec_range=(dec_range[0] * u.deg, dec_range[1] * u.deg),
            min_angular_velocity=min_angular_velocity
        )

        if verbose:
            print("Computing pass parameters...")
        calculator.compute_pass_parameters()

        if verbose:
            print(f"Filtering for night-time passes (sun < {sun_alt_limit}°)...")
        night_passes = calculator.filter_night_passes(sun_alt_limit=sun_alt_limit * u.deg)

        if verbose:
            print(f"\nFound {len(night_passes)} night-time passes through survey area:")

            # Show monthly distribution
            from collections import defaultdict
            monthly_counts = defaultdict(int)
            for p in night_passes:
                month_key = p.start_time.datetime.strftime('%Y-%m')
                monthly_counts[month_key] += 1

            print(f"\nMonthly distribution:")
            for month in sorted(monthly_counts.keys()):
                print(f"  {month}: {monthly_counts[month]} passes")

            for i, p in enumerate(night_passes):
                duration = (p.end_time - p.start_time).sec
                n_samples = len(p.times)
                print(f"  Pass {i + 1}: {p.start_time.iso} - {p.end_time.iso}")
                print(
                    f"    Duration: {duration:.1f}s, Samples: {n_samples}, Max velocity: {p.max_angular_velocity:.3f} deg/s")
                print(f"    RA range: {p.ra.min():.2f} - {p.ra.max():.2f}°")
                print(f"    Dec range: {p.dec.min():.2f} - {p.dec.max():.2f}°")

        # Save and visualize results
        if output_dir and len(night_passes) > 0:
            if verbose:
                print(f"\nSaving results to {output_dir}")
            calculator.save_passes(output_dir)
            calculator.visualize_passes(output_dir)
        elif output_dir and len(night_passes) == 0:
            print("\nNo passes to save")

    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'compute' or 'visualize'")


if __name__ == '__main__':
    main()
