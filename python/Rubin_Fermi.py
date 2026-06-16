from dataclasses import dataclass
from astropy.io import fits
from astropy.coordinates import EarthLocation, AltAz, ICRS, GCRS, CartesianRepresentation
from astropy.time import Time
import astropy.units as u
import numpy as np
import pickle
import json

# in collaboration with Anthropic Claude - 2026

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

    # Filter flags - which constraints does this pass meet?
    passes_altitude: bool = False  # alt >= min_altitude
    passes_declination: bool = False  # dec in range
    passes_velocity: bool = False  # velocity >= threshold
    passes_night: bool = False  # sun < sun_alt_limit

    @property
    def passes_all_filters(self):
        """Does this pass meet ALL filter criteria?"""
        return (self.passes_altitude and
                self.passes_declination and
                self.passes_velocity and
                self.passes_night)


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

        from astropy.coordinates import SkyCoord, AltAz

        # Direct spherical conversion from ECI Cartesian
        x = self.positions[:, 0]
        y = self.positions[:, 1]
        z = self.positions[:, 2]

        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        # Standard spherical coordinate conversion
        ra_rad = np.arctan2(y, x)
        dec_rad = np.arcsin(z / r)

        # Convert to degrees
        ra_deg = np.degrees(ra_rad)
        dec_deg = np.degrees(dec_rad)

        # Wrap RA to [0, 360)
        ra_deg = (ra_deg + 360) % 360

        # For Alt/Az, create SkyCoord with these RA/Dec
        fermi_coords = SkyCoord(
            ra=ra_deg * u.deg,
            dec=dec_deg * u.deg,
            distance=r * u.m,
            frame='icrs',
            obstime=self.times
        )

        # Transform to AltAz
        altaz = fermi_coords.transform_to(
            AltAz(obstime=self.times, location=self.rubin_location)
        )

        # Store results
        self.sky_coords = {
            'ra': ra_deg,
            'dec': dec_deg,
            'alt': altaz.alt.deg,
            'az': altaz.az.deg
        }

        # DIAGNOSTIC
        print(f"\nDirect conversion check (first 10 samples):")
        for i in range(10):
            print(f"  {self.times[i].iso}: RA={ra_deg[i]:.2f}, Dec={dec_deg[i]:.2f}")

    def find_survey_passes(self, min_altitude=20 * u.deg,
                           dec_range=(-70, 10) * u.deg,
                           min_angular_velocity=0.1):
        """
        Find ALL continuous pass segments, then evaluate which constraints each meets

        Parameters:
        -----------
        min_altitude : astropy.units.Quantity
            Minimum altitude above horizon
        dec_range : tuple of astropy.units.Quantity
            (min_dec, max_dec) for survey area
        min_angular_velocity : float
            Minimum angular velocity in deg/s

        Returns:
        --------
        list of FermiPass objects with filter flags set
        """
        if self.sky_coords is None:
            raise ValueError("Must call compute_topocentric() first")

        ra = self.sky_coords['ra']
        dec = self.sky_coords['dec']
        alt = self.sky_coords['alt']

        # Compute angular velocities
        ra_diff = np.diff(ra)
        dec_diff = np.diff(dec)

        # Handle RA wraparound
        ra_diff = np.where(ra_diff > 180, ra_diff - 360, ra_diff)
        ra_diff = np.where(ra_diff < -180, ra_diff + 360, ra_diff)

        # Angular separation with cos(dec) correction
        dec_avg = (dec[:-1] + dec[1:]) / 2
        ang_sep = np.sqrt((ra_diff * np.cos(np.radians(dec_avg))) ** 2 + dec_diff ** 2)

        # Time differences in seconds
        time_diff = np.diff(self.times.mjd) * 86400

        # Angular velocity in deg/sec (pad with 0 at end)
        ang_velocity = np.concatenate([ang_sep / time_diff, np.array([0.0])])

        # Find continuous data segments (gaps < 5 minutes indicate data continuity)
        continuous_mask = np.concatenate([time_diff < 300, [True]])

        # Find segment boundaries
        transitions = np.diff(continuous_mask.astype(int))
        starts = np.where(transitions == 1)[0] + 1
        ends = np.where(transitions == -1)[0] + 1

        # Handle edge cases
        if continuous_mask[0]:
            starts = np.concatenate([[0], starts])
        if continuous_mask[-1]:
            ends = np.concatenate([ends, [len(continuous_mask)]])

        # Create passes and evaluate filters
        self.passes = []

        for start_idx, end_idx in zip(starts, ends):
            # Skip very short segments
            if end_idx - start_idx < 2:
                continue

            # Create pass object
            pass_obj = FermiPass(
                start_time=self.times[start_idx],
                end_time=self.times[end_idx - 1],
                times=self.times[start_idx:end_idx],
                ra=ra[start_idx:end_idx],
                dec=dec[start_idx:end_idx],
                alt=alt[start_idx:end_idx],
                az=self.sky_coords['az'][start_idx:end_idx],
                max_angular_velocity=0.0  # Will compute in compute_pass_parameters()
            )

            # Evaluate filter constraints

            # Altitude: Are ALL points above minimum?
            pass_obj.passes_altitude = np.all(pass_obj.alt >= min_altitude.value)

            # Declination: Are ALL points in range?
            pass_obj.passes_declination = np.all(
                (pass_obj.dec >= dec_range[0].value) &
                (pass_obj.dec <= dec_range[1].value)
            )

            # Velocity: Is ANY point above threshold?
            pass_velocities = ang_velocity[start_idx:end_idx - 1]
            if len(pass_velocities) > 0:
                pass_obj.passes_velocity = np.max(pass_velocities) >= min_angular_velocity
            else:
                pass_obj.passes_velocity = False

            # Night filter will be evaluated later in filter_night_passes()
            pass_obj.passes_night = False  # Default to False

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
        Evaluate which passes meet night-time criteria
        Sets the passes_night flag but does NOT remove any passes

        Parameters:
        -----------
        sun_alt_limit : astropy.units.Quantity
            Maximum sun altitude (typically -12 or -18 deg for twilight)
        """
        from astropy.coordinates import get_sun

        for pass_obj in self.passes:
            # Get sun position at start and end of pass
            sun_start = get_sun(pass_obj.start_time).transform_to(
                AltAz(obstime=pass_obj.start_time, location=self.rubin_location)
            )
            sun_end = get_sun(pass_obj.end_time).transform_to(
                AltAz(obstime=pass_obj.end_time, location=self.rubin_location)
            )

            # Set flag if sun is below limit for entire duration
            pass_obj.passes_night = (sun_start.alt < sun_alt_limit and
                                     sun_end.alt < sun_alt_limit)

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
                'max_angular_velocity': float(p.max_angular_velocity),
                # Filter flags
                'passes_altitude': bool(p.passes_altitude),
                'passes_declination': bool(p.passes_declination),
                'passes_velocity': bool(p.passes_velocity),
                'passes_night': bool(p.passes_night),
                'passes_all_filters': bool(p.passes_all_filters)
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
                             'Alt Min', 'Alt Max', 'Max Velocity (deg/s)',
                             'Passes Altitude', 'Passes Dec', 'Passes Velocity',
                             'Passes Night', 'Passes All'])
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
                    f"{p.max_angular_velocity:.4f}",
                    p.passes_altitude,
                    p.passes_declination,
                    p.passes_velocity,
                    p.passes_night,
                    p.passes_all_filters
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

    def visualize_passes(self, output_dir, file_prefix="fermi_passes", filter_mode='all', max_passes=500):
        """
        Create Bokeh visualizations of Fermi passes

        Parameters:
        -----------
        output_dir : str or Path
            Directory to save output files
        file_prefix : str
            Prefix for output filenames
        filter_mode : str
            Which passes to visualize: 'all', 'passes_all', 'night_only', etc.
        max_passes : int
            Maximum number of passes to plot (samples intelligently if exceeded)
        """
        from bokeh.plotting import figure, save, output_file
        from bokeh.models import HoverTool, ColumnDataSource, Legend
        from bokeh.layouts import column
        from bokeh.palettes import Category10
        import numpy as np

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if len(self.passes) == 0:
            print("No passes to visualize")
            return

        # Filter passes based on mode
        if filter_mode == 'all':
            passes_to_plot = self.passes
            title_suffix = "All Passes"
        elif filter_mode == 'passes_all':
            passes_to_plot = [p for p in self.passes if p.passes_all_filters]
            title_suffix = "Passes Meeting All Filters"
        elif filter_mode == 'night_only':
            passes_to_plot = [p for p in self.passes if p.passes_night]
            title_suffix = "Night-Time Passes"
        elif filter_mode == 'day_only':
            passes_to_plot = [p for p in self.passes if not p.passes_night]
            title_suffix = "Day-Time Passes"
        else:
            passes_to_plot = self.passes
            title_suffix = f"Filter: {filter_mode}"

        if len(passes_to_plot) == 0:
            print(f"No passes match filter mode: {filter_mode}")
            return

        # Intelligent sampling if too many passes
        original_count = len(passes_to_plot)
        if len(passes_to_plot) > max_passes:
            print(f"  Sampling {max_passes} passes from {original_count} for visualization performance")

            # Stratified sampling: ensure we get examples from different filter states
            passes_all = [p for p in passes_to_plot if p.passes_all_filters]
            passes_night = [p for p in passes_to_plot if p.passes_night and not p.passes_all_filters]
            passes_day = [p for p in passes_to_plot if not p.passes_night]

            # Sample proportionally
            n_all = min(len(passes_all), max_passes // 3)
            n_night = min(len(passes_night), max_passes // 3)
            n_day = min(max_passes - n_all - n_night, len(passes_day))

            sampled = []
            if n_all > 0:
                indices = np.linspace(0, len(passes_all) - 1, n_all, dtype=int)
                sampled.extend([passes_all[i] for i in indices])
            if n_night > 0:
                indices = np.linspace(0, len(passes_night) - 1, n_night, dtype=int)
                sampled.extend([passes_night[i] for i in indices])
            if n_day > 0:
                indices = np.linspace(0, len(passes_day) - 1, n_day, dtype=int)
                sampled.extend([passes_day[i] for i in indices])

            passes_to_plot = sampled
            title_suffix += f" (showing {len(passes_to_plot)} of {original_count})"

        # Create sky map of pass trajectories
        sky_plot = figure(
            width=1000,
            height=600,
            title=f"Fermi Pass Trajectories - {title_suffix}",
            x_axis_label="RA (degrees)",
            y_axis_label="Dec (degrees)",
            tools="pan,wheel_zoom,box_zoom,reset,save,hover"
        )

        # Color code by filter status
        def get_pass_color(p):
            if p.passes_all_filters:
                return 'green'  # Passes everything
            elif p.passes_night:
                return 'blue'  # Night but fails other filters
            elif p.passes_altitude and p.passes_declination and p.passes_velocity:
                return 'orange'  # Day but passes other filters
            else:
                return 'red'  # Fails multiple filters

        # Plot each pass
        legend_items = []
        colors_used = set()

        for i, p in enumerate(passes_to_plot):
            color = get_pass_color(p)
            alpha = 0.8 if p.passes_all_filters else 0.4
            line_width = 3 if p.passes_all_filters else 2

            line = sky_plot.line(
                p.ra, p.dec,
                line_width=line_width,
                alpha=alpha,
                color=color
            )

            # Add to legend (avoid duplicates)
            if color not in colors_used:
                colors_used.add(color)
                if color == 'green':
                    label = "Passes All Filters"
                elif color == 'blue':
                    label = "Night Only"
                elif color == 'orange':
                    label = "Day (Alt+Dec+Vel)"
                else:
                    label = "Fails Multiple"
                legend_items.append((label, [line]))

        # Add legend
        if legend_items:
            legend = Legend(items=legend_items, location="top_right")
            legend.click_policy = "hide"
            sky_plot.add_layout(legend)

        # Add survey area boundaries
        sky_plot.line([0, 360], [-70, -70], line_dash='dashed',
                      color='gray', alpha=0.5, line_width=2)
        sky_plot.line([0, 360], [10, 10], line_dash='dashed',
                      color='gray', alpha=0.5, line_width=2)

        # Configure hover tool
        hover = sky_plot.select_one(HoverTool)
        hover.tooltips = [
            ("RA", "$x{0.2f}°"),
            ("Dec", "$y{0.2f}°")
        ]

        # Timeline plot - use ALL passes_to_plot (already sampled)
        timeline_plot = figure(
            width=1000,
            height=400,
            title="Pass Timeline",
            x_axis_label="Date",
            y_axis_label="Duration (minutes)",
            x_axis_type='datetime',
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )

        # Prepare data for timeline
        start_times = [p.start_time.datetime for p in passes_to_plot]
        durations = [(p.end_time - p.start_time).sec / 60.0 for p in passes_to_plot]
        max_velocities = [p.max_angular_velocity for p in passes_to_plot]
        pass_numbers = list(range(1, len(passes_to_plot) + 1))
        colors = [get_pass_color(p) for p in passes_to_plot]

        # Filter status strings for hover
        filter_status = []
        for p in passes_to_plot:
            status_parts = []
            if p.passes_altitude:
                status_parts.append("Alt✓")
            else:
                status_parts.append("Alt✗")
            if p.passes_declination:
                status_parts.append("Dec✓")
            else:
                status_parts.append("Dec✗")
            if p.passes_velocity:
                status_parts.append("Vel✓")
            else:
                status_parts.append("Vel✗")
            if p.passes_night:
                status_parts.append("Night✓")
            else:
                status_parts.append("Night✗")
            filter_status.append(" ".join(status_parts))

        source = ColumnDataSource(data={
            'start': start_times,
            'duration': durations,
            'velocity': max_velocities,
            'pass_num': pass_numbers,
            'color': colors,
            'filter_status': filter_status
        })

        timeline_plot.scatter('start', 'duration', size=10, source=source,
                              color='color', alpha=0.8)

        # Add hover tool to timeline
        timeline_hover = HoverTool(tooltips=[
            ("Pass", "@pass_num"),
            ("Start", "@start{%F %T}"),
            ("Duration", "@duration{0.1f} min"),
            ("Max Velocity", "@velocity{0.3f} deg/s"),
            ("Filters", "@filter_status")
        ], formatters={'@start': 'datetime'})

        timeline_plot.add_tools(timeline_hover)

        # Create filter statistics bar chart (use ALL passes, not sampled)
        stats_plot = figure(
            width=1000,
            height=400,
            title=f"Filter Statistics (all {len(self.passes)} passes)",
            x_range=['Altitude', 'Declination', 'Velocity', 'Night', 'All Filters'],
            y_axis_label="Number of Passes",
            tools="save"
        )

        filter_counts = [
            sum(p.passes_altitude for p in self.passes),
            sum(p.passes_declination for p in self.passes),
            sum(p.passes_velocity for p in self.passes),
            sum(p.passes_night for p in self.passes),
            sum(p.passes_all_filters for p in self.passes)
        ]

        stats_plot.vbar(x=['Altitude', 'Declination', 'Velocity', 'Night', 'All Filters'],
                        top=filter_counts, width=0.8,
                        color=['blue', 'green', 'orange', 'purple', 'red'],
                        alpha=0.8)

        # Add text labels on bars
        from bokeh.models import Label
        for i, (x, count) in enumerate(
                zip(['Altitude', 'Declination', 'Velocity', 'Night', 'All Filters'], filter_counts)):
            label = Label(x=i, y=count, text=str(count),
                          text_align='center', text_baseline='bottom',
                          text_font_size='10pt')
            stats_plot.add_layout(label)

        # Save combined layout
        layout = column(sky_plot, timeline_plot, stats_plot)

        html_file = output_dir / f"{file_prefix}_visualization.html"
        output_file(html_file)
        save(layout)

        print(f"Saved visualization to {html_file}")
        print(f"  Plotted {len(passes_to_plot)} passes (filter mode: {filter_mode})")
        if original_count != len(passes_to_plot):
            print(f"  (sampled from {original_count} total passes for performance)")


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
            viz_filter_mode = config.get('viz_filter_mode', 'all')
            calculator.visualize_passes(output_dir, filter_mode=viz_filter_mode)
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
            print(f"\nCoordinate diagnostics:")
            print(f"  RA range: {calculator.sky_coords['ra'].min():.2f} to {calculator.sky_coords['ra'].max():.2f}°")
            print(f"  Dec range: {calculator.sky_coords['dec'].min():.2f} to {calculator.sky_coords['dec'].max():.2f}°")
            print(f"  Alt range: {calculator.sky_coords['alt'].min():.2f} to {calculator.sky_coords['alt'].max():.2f}°")
            print(f"  Samples above horizon (alt > 0): {np.sum(calculator.sky_coords['alt'] > 0)}")
            print(f"  Samples above {min_altitude}°: {np.sum(calculator.sky_coords['alt'] > min_altitude)}")

        if verbose:
            print(
                f"\nFinding passes (evaluating alt >= {min_altitude}°, dec in {dec_range}, vel >= {min_angular_velocity} deg/s)...")
        passes = calculator.find_survey_passes(
            min_altitude=min_altitude * u.deg,
            dec_range=(dec_range[0] * u.deg, dec_range[1] * u.deg),
            min_angular_velocity=min_angular_velocity
        )

        if verbose:
            print(f"  Found {len(passes)} continuous pass segments")

        if verbose:
            print("Computing pass parameters...")
        calculator.compute_pass_parameters()

        if verbose:
            print(f"Evaluating night-time constraint (sun < {sun_alt_limit}°)...")
        calculator.filter_night_passes(sun_alt_limit=sun_alt_limit * u.deg)

        if verbose:
            # Filter statistics
            print(f"\nFilter Statistics:")
            print(f"  Total passes found: {len(passes)}")

            n_alt = sum(p.passes_altitude for p in passes)
            n_dec = sum(p.passes_declination for p in passes)
            n_vel = sum(p.passes_velocity for p in passes)
            n_night = sum(p.passes_night for p in passes)
            n_all = sum(p.passes_all_filters for p in passes)

            print(f"  Passes altitude filter (>= {min_altitude}°): {n_alt} ({100 * n_alt / len(passes):.1f}%)")
            print(
                f"  Passes declination filter ({dec_range[0]}° to {dec_range[1]}°): {n_dec} ({100 * n_dec / len(passes):.1f}%)")
            print(
                f"  Passes velocity filter (>= {min_angular_velocity} deg/s): {n_vel} ({100 * n_vel / len(passes):.1f}%)")
            print(f"  Passes night filter (sun < {sun_alt_limit}°): {n_night} ({100 * n_night / len(passes):.1f}%)")
            print(f"  Passes ALL filters: {n_all} ({100 * n_all / len(passes) if len(passes) > 0 else 0:.1f}%)")

            # Show breakdown of filter combinations
            print(f"\nFilter Combinations:")
            alt_dec = sum(p.passes_altitude and p.passes_declination for p in passes)
            alt_dec_vel = sum(p.passes_altitude and p.passes_declination and p.passes_velocity for p in passes)
            print(f"  Alt + Dec: {alt_dec}")
            print(f"  Alt + Dec + Vel: {alt_dec_vel}")
            print(f"  Alt + Dec + Vel + Night: {n_all}")

            # Monthly distribution of passes meeting all filters
            if n_all > 0:
                from collections import defaultdict
                monthly_counts = defaultdict(int)
                for p in passes:
                    if p.passes_all_filters:
                        month_key = p.start_time.datetime.strftime('%Y-%m')
                        monthly_counts[month_key] += 1

                print(f"\nMonthly distribution (passes meeting ALL filters):")
                for month in sorted(monthly_counts.keys()):
                    print(f"  {month}: {monthly_counts[month]} passes")

            # Show sample passes from different filter categories
            print(f"\nSample passes by filter status:")

            # Passes all filters
            all_filter_passes = [p for p in passes if p.passes_all_filters]
            if all_filter_passes:
                print(f"\n  Passes ALL filters ({len(all_filter_passes)} total):")
                for p in all_filter_passes[:5]:
                    print(f"    {p.start_time.iso}: duration={(p.end_time - p.start_time).sec:.0f}s, "
                          f"RA={p.ra.min():.1f}°-{p.ra.max():.1f}°, "
                          f"Dec={p.dec.min():.1f}°-{p.dec.max():.1f}°, "
                          f"Alt={p.alt.min():.1f}°-{p.alt.max():.1f}°, "
                          f"vel={p.max_angular_velocity:.3f} deg/s")

            # Passes everything except night
            day_passes = [p for p in passes if (p.passes_altitude and p.passes_declination
                                                and p.passes_velocity and not p.passes_night)]
            if day_passes:
                print(f"\n  Passes all EXCEPT night ({len(day_passes)} total, showing first 5):")
                for p in day_passes[:5]:
                    print(f"    {p.start_time.iso}: duration={(p.end_time - p.start_time).sec:.0f}s, "
                          f"Alt={p.alt.min():.1f}°-{p.alt.max():.1f}°, "
                          f"vel={p.max_angular_velocity:.3f} deg/s")

            # Night passes that fail other filters
            night_only = [p for p in passes if (p.passes_night and
                                                not (p.passes_altitude and p.passes_declination and p.passes_velocity))]
            if night_only:
                print(f"\n  Passes night but fails other filters ({len(night_only)} total, showing first 5):")
                for p in night_only[:5]:
                    flags = []
                    if not p.passes_altitude: flags.append("alt")
                    if not p.passes_declination: flags.append("dec")
                    if not p.passes_velocity: flags.append("vel")
                    print(f"    {p.start_time.iso}: fails {','.join(flags)}, "
                          f"Alt={p.alt.min():.1f}°-{p.alt.max():.1f}°")

            # Passes that are close (fail only one or two filters)
            close_passes = [p for p in passes if
                            sum([p.passes_altitude, p.passes_declination,
                                 p.passes_velocity, p.passes_night]) >= 3]
            close_passes = [p for p in close_passes if not p.passes_all_filters]
            if close_passes:
                print(f"\n  Close calls (pass 3 of 4 filters, {len(close_passes)} total, showing first 5):")
                for p in close_passes[:5]:
                    fails = []
                    if not p.passes_altitude: fails.append(f"alt({p.alt.min():.1f}°-{p.alt.max():.1f}°)")
                    if not p.passes_declination: fails.append(f"dec({p.dec.min():.1f}°-{p.dec.max():.1f}°)")
                    if not p.passes_velocity: fails.append(f"vel({p.max_angular_velocity:.3f})")
                    if not p.passes_night: fails.append("night")
                    print(f"    {p.start_time.iso}: fails {', '.join(fails)}")

        # Save and visualize results
        if output_dir:
            if verbose:
                print(f"\nSaving results to {output_dir}")
            calculator.save_passes(output_dir)

            viz_filter_mode = config.get('viz_filter_mode', 'all')
            viz_max_passes = config.get('viz_max_passes', 500)
            if verbose:
                print(f"Creating visualizations (filter mode: {viz_filter_mode}, max: {viz_max_passes})...")
            calculator.visualize_passes(output_dir, filter_mode=viz_filter_mode, max_passes=viz_max_passes)
        elif len(passes) > 0:
            print("\nNote: No output_dir specified, results not saved")

    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'compute' or 'visualize'")


if __name__ == '__main__':
    main()
