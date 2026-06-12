from dataclasses import dataclass
from astropy.io import fits
from astropy.coordinates import EarthLocation, AltAz, ICRS, GCRS
from astropy.time import Time
import astropy.units as u
import numpy as np


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
            self.geo_lat = sc_data['LAT_GEO']
            self.geo_lon = sc_data['LON_GEO']
            self.geo_rad = sc_data['RAD_GEO']
            self.positions = sc_data['SC_POSITION']

            print(f"  Geographic coordinate ranges:")
            print(f"    Lat: {self.geo_lat.min():.2f} to {self.geo_lat.max():.2f}")
            print(f"    Lon: {self.geo_lon.min():.2f} to {self.geo_lon.max():.2f}")
            print(f"    Radius: {self.geo_rad.min():.0f} to {self.geo_rad.max():.0f} m")
            print(f"  First few geographic positions:")
            for i in range(min(5, len(times))):
                print(
                    f"    {times[i].iso}: lat={self.geo_lat[i]:.2f}, lon={self.geo_lon[i]:.2f}, "
                    f"r={self.geo_rad[i]:.0f}")

    def compute_topocentric(self):
        """Convert spacecraft positions to RA/Dec/Alt/Az from Rubin"""
        if self.times is None:
            raise ValueError("Must call load_ft2() first")

        from astropy.coordinates import ITRS, CartesianRepresentation

        # RAD_GEO might be altitude, not radius - add Earth radius
        EARTH_RADIUS = 6371000  # meters
        r = self.geo_rad + EARTH_RADIUS  # Total radius from Earth center

        # Geographic coordinates to Cartesian ITRS
        lat_rad = np.radians(self.geo_lat)
        lon_rad = np.radians(self.geo_lon)

        # Spherical to Cartesian conversion
        x = r * np.cos(lat_rad) * np.cos(lon_rad)
        y = r * np.cos(lat_rad) * np.sin(lon_rad)
        z = r * np.sin(lat_rad)

        # Create ITRS coordinates (Earth-fixed)
        itrs_coords = ITRS(
            CartesianRepresentation(x * u.m, y * u.m, z * u.m),
            obstime=self.times
        )

        # Transform to GCRS (inertial celestial frame)
        gcrs_coords = itrs_coords.transform_to(GCRS(obstime=self.times))

        # Transform to topocentric AltAz at Rubin
        altaz_frame = AltAz(obstime=self.times, location=self.rubin_location)
        altaz_coords = gcrs_coords.transform_to(altaz_frame)

        # Get RA/Dec (ICRS)
        icrs_coords = gcrs_coords.transform_to(ICRS())

        # Store results
        self.sky_coords = {
            'ra': icrs_coords.ra.deg,
            'dec': icrs_coords.dec.deg,
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
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for pass data (optional)'
    )

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Extract parameters
    ft2_file = config['ft2_file']
    start_time = config.get('start_time', None)
    end_time = config.get('end_time', None)
    downsample_sec = config.get('downsample_sec', 30)
    min_altitude = config.get('min_altitude', 20)
    dec_range = config.get('dec_range', [-70, 10])
    output_file = config.get('output_file', None)
    min_angular_velocity = config.get('min_angular_velocity', 0.1)
    verbose = config.get('verbose', False)

    # Run calculation
    print(f"Loading FT2 file: {ft2_file}")
    calculator = FermiPassCalculator(ft2_file, start_time, end_time)

    print(f"Loading and downsampling to {downsample_sec}s cadence...")
    calculator.load_ft2(downsample_sec=downsample_sec)
    print(f"  Loaded {len(calculator.times)} time samples")

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
                f"    {calculator.times[i].iso}: RA={calculator.sky_coords['ra'][i]:.6f}, "
                f"Dec={calculator.sky_coords['dec'][i]:.6f}, Alt={calculator.sky_coords['alt'][i]:.2f}")

    print(f"Finding passes (alt > {min_altitude}°, dec in {dec_range})...")
    passes = calculator.find_survey_passes(
        min_altitude=min_altitude * u.deg,
        dec_range=(dec_range[0] * u.deg, dec_range[1] * u.deg),
        min_angular_velocity=min_angular_velocity
    )

    print("Computing pass parameters...")
    calculator.compute_pass_parameters()

    if verbose:
        print(f"Finding passes (alt > {min_altitude}°, dec in {dec_range})...")
        print(f"  Velocity threshold: {min_angular_velocity} deg/s")

        # Diagnostic: show velocity statistics
        alt_mask = calculator.sky_coords['alt'] >= min_altitude
        dec_mask = ((calculator.sky_coords['dec'] >= dec_range[0]) &
                    (calculator.sky_coords['dec'] <= dec_range[1]))
        in_area = alt_mask & dec_mask

        # Compute velocities (same as in find_survey_passes)
        ra = calculator.sky_coords['ra']
        dec = calculator.sky_coords['dec']
        ra_diff = np.diff(ra)
        dec_diff = np.diff(dec)
        ra_diff = np.where(ra_diff > 180, ra_diff - 360, ra_diff)
        ra_diff = np.where(ra_diff < -180, ra_diff + 360, ra_diff)
        dec_avg = (dec[:-1] + dec[1:]) / 2
        ang_sep = np.sqrt((ra_diff * np.cos(np.radians(dec_avg))) ** 2 + dec_diff ** 2)
        time_diff = np.diff(calculator.times.mjd) * 86400
        ang_velocity = ang_sep / time_diff

        print(f"  Velocity stats in survey area:")
        print(f"    Max: {np.max(ang_velocity[in_area[:-1]]):.4f} deg/s")
        print(f"    Median: {np.median(ang_velocity[in_area[:-1]]):.4f} deg/s")
        print(f"    95th percentile: {np.percentile(ang_velocity[in_area[:-1]], 95):.4f} deg/s")

    print(f"\nFound {len(passes)} passes through survey area:")
    for i, p in enumerate(passes):
        duration = (p.end_time - p.start_time).sec
        n_samples = len(p.times)
        print(f"  Pass {i + 1}: {p.start_time.iso} - {p.end_time.iso}")
        print(f"    Duration: {duration:.1f}s, Samples: {n_samples}, Max velocity: {p.max_angular_velocity:.3f} deg/s")
        print(f"    RA range: {p.ra.min():.2f} - {p.ra.max():.2f}°")
        print(f"    Dec range: {p.dec.min():.2f} - {p.dec.max():.2f}°")

        # Diagnostic: show first few positions if multiple samples
        if n_samples > 1:
            print(f"    First 3 RAs: {p.ra[:3]}")
            print(f"    First 3 Decs: {p.dec[:3]}")


    # Optional: save results
    if args.output:
        print(f"\nSaving results to {args.output}")
        # TODO: implement saving (pickle, HDF5, or custom format)


if __name__ == '__main__':
    main()
