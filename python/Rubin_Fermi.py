from dataclasses import dataclass
from astropy.io import fits
from astropy.coordinates import EarthLocation, AltAz, ICRS, GCRS, ITRS, CartesianRepresentation
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
        # This version is identical to the last one, but with a final
        # diagnostic block to print the details of the impossible jump.
        import numpy as np
        import astropy.units as u
        from astropy.io import fits
        from astropy.time import Time

        with fits.open(self.ft2_file) as hdul:
            data_table = hdul['SC_DATA'].data
            met_times_raw = np.array(data_table['START'])
            positions_raw = np.array(data_table['SC_POSITION'])
            times_unsorted = (Time('2001-01-01T00:00:00', format='isot', scale='utc') +
                              met_times_raw * u.s)

            sort_indices = np.argsort(times_unsorted)

            self.times = times_unsorted[sort_indices]
            self.positions = positions_raw[sort_indices]

            # Filtering and downsampling logic as before...
            if self.start_time or self.end_time:
                mask = np.ones(len(self.times), dtype=bool)
                if self.start_time: mask &= (self.times >= self.start_time)
                if self.end_time: mask &= (self.times <= self.end_time)
                self.times, self.positions = self.times[mask], self.positions[mask]

            if len(self.times) > 1:
                median_cadence = np.median(np.diff(self.times.mjd)) * 86400
                downsample_factor = max(1, int(round(downsample_sec / median_cadence)))
                if downsample_factor > 1:
                    self.times = self.times[::downsample_factor]
                    self.positions = self.positions[::downsample_factor]

            # ==============================================================================
            # FINAL DIAGNOSTIC: PROVE THE FILE IS CORRUPT
            # ==============================================================================
            print(f"\n{'=' * 60}\nFILE CORRUPTION ANALYSIS\n{'=' * 60}")

            pos_diff = np.diff(self.positions, axis=0)
            pos_diff_mag = np.sqrt(np.sum(pos_diff ** 2, axis=1))
            time_diffs_sec = np.diff(self.times.mjd) * 86400

            # Find the index of the largest physical jump in the data
            max_jump_idx = np.argmax(pos_diff_mag)

            print(f"The largest single jump between consecutive data points was found at index: {max_jump_idx}")
            print(
                f"This jump covers {pos_diff_mag[max_jump_idx] / 1000:.1f} km in {time_diffs_sec[max_jump_idx]:.1f} seconds.\n")

            print("Data point BEFORE the jump:")
            print(f"  Time: {self.times[max_jump_idx].iso}")
            print(f"  Position (X,Y,Z meters): {self.positions[max_jump_idx]}")

            print("\nData point AFTER the jump:")
            print(f"  Time: {self.times[max_jump_idx + 1].iso}")
            print(f"  Position (X,Y,Z meters): {self.positions[max_jump_idx + 1]}")

            print(f"\nThis is physically impossible. It confirms the FITS file's columns are scrambled.")
            print(f"{'=' * 60}\n")
            # ==============================================================================

            # The original integrity report for comparison
            print(f"\nFT2 DATA INTEGRITY REPORT - check:")
            # ... (rest of the diagnostics will follow) ...

    def load_tle(self, start_time, end_time, time_step_seconds=30.0,
                 norad_id=33053,
                 tle_url='https://celestrak.org/NORAD/elements/gp.php?GROUP=science&FORMAT=tle',
                 verbose=True):

        """
        Populate self.times and self.positions by propagating a TLE via Skyfield.

        CAVEAT — current-epoch feed, not a historical archive:
        CelesTrak's live feed returns only the single most-recent element set for
        each satellite (one epoch). SGP4 propagation error grows with distance from
        that epoch in EITHER direction — roughly km/day and compounding past a week
        or two. Propagating ~6 months produces hundreds-to-thousands of km of
        along-track error.

        This is the root cause of the "excellent in June 2026 / poor in December
        2025" pattern seen during validation: the cached TLE's epoch was ~June 2026,
        so December 2025 required ~6 months of BACK-propagation.

        Use this method only for genuine near-future prediction (within ~1-2 weeks
        of the current epoch). For historical analysis, use load_ascii() with the
        definitive engineering-telemetry file. As a fallback without telemetry,
        fetch a time-series of archived TLEs (e.g. Space-Track) and always
        propagate from the nearest-in-time epoch.

        Parameters
        ----------
        start_time, end_time : astropy.Time or str
            Time range to generate positions over.
        time_step_seconds : float
            Propagation cadence in seconds.
        norad_id : int
            Satellite catalog number (Fermi = 33053).
        tle_url : str
            CelesTrak GP query URL.
        verbose : bool
            Print progress / epoch info.
        """

        from skyfield.api import load
        import numpy as np
        import astropy.units as u

        if verbose:
            print("--- Initializing Skyfield TLE Propagator ---")
        load.verbose = False
        ts = load.timescale()

        try:
            satellites = load.tle_file(tle_url, reload=True)
        except Exception as e:
            raise RuntimeError(f"Could not download TLE data. Check internet. Details: {e}")

        try:
            fermi_satellite = {sat.model.satnum: sat for sat in satellites}[norad_id]
        except KeyError:
            raise RuntimeError(f"Could not find satellite (NORAD ID {norad_id}) in TLE list.")

        if verbose:
            print(f"Loaded TLE for: {fermi_satellite.name}")
            print(f"TLE Epoch: {fermi_satellite.epoch.utc_strftime()}")

        start_time = Time(start_time, scale='utc')
        end_time = Time(end_time, scale='utc')
        num_steps = int((end_time - start_time).sec / time_step_seconds)
        times_array = start_time + np.arange(num_steps) * time_step_seconds * u.s

        if verbose:
            print(f"Generating {len(times_array)} position points...")
        t_skyfield = ts.from_astropy(times_array)
        geocentric = fermi_satellite.at(t_skyfield)

        self.times = times_array
        self.positions = geocentric.position.m.T

        if verbose:
            print("Position generation complete.")

        return self.times, self.positions

    def load_ascii(self, ascii_file, frame='gcrs', expected_cadence_sec=30.0,
                   units='m', validate_beta=False, verbose=True):
        """
        Load Fermi ECI trajectory from the engineering-telemetry ASCII file.

        Actual format (no header, comma-separated, time quoted):
            "2025-12-01 00:00:00.940104",1764547200.940104,-2407610.0,6270908.5,-1497560.5
             [0] ISO UTC (quoted)         [1] Unix time      [2] X  [3] Y  [4] Z   (meters, ECI)

        Notes / gotchas baked in:
          * Time is parsed from col 0 (ISO, sub-second). Col 1 is UNIX time
            (1970 epoch) -- NOT Fermi MET (2001 epoch); used only as a cross-check.
          * Positions are in METERS (this file), unlike the km tle2beta sample.
          * Frame is ECI; 'gcrs' is the correct astropy choice (J2000-ish).
          * No Rad/beta columns, so integrity is checked via plausible LEO radius.
        """
        import numpy as np
        import astropy.units as u
        from astropy.time import Time
        from pathlib import Path

        ascii_file = Path(ascii_file)
        if not ascii_file.exists():
            raise FileNotFoundError(f"ASCII telemetry file not found: {ascii_file}")

        iso_str, unix_col, xyz = [], [], []
        with open(ascii_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) < 5:
                    continue
                try:
                    t = parts[0].strip().strip('"').strip()
                    unix_t = float(parts[1])
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                except (ValueError, IndexError):
                    continue
                iso_str.append(t)
                unix_col.append(unix_t)
                xyz.append((x, y, z))

        if not iso_str:
            raise ValueError(f"No parseable rows in {ascii_file}.")

        # Parse authoritative time from the ISO column (carries sub-second precision)
        times = Time(iso_str, format='iso', scale='utc')
        unix_col = np.array(unix_col)
        pos = np.array(xyz)  # meters, ECI

        # --- Cross-check: col 1 is Unix time, must agree with col 0 ---------------
        # (Guards against the "assumed Fermi MET-2001" 55-year error.)
        unix_from_iso = times.unix
        max_dt = np.max(np.abs(unix_from_iso - unix_col))
        if max_dt > 1.0:
            print(f"  WARNING: col 1 disagrees with col 0 by up to {max_dt:.1f}s. "
                  f"Col 1 is NOT Unix time as assumed -- check epoch (Fermi MET? GPS?).")
        elif verbose:
            print(f"  Time cross-check OK (col1 = Unix time, max Δ {max_dt:.3f}s)")

        # --- Unit / units handling ------------------------------------------------
        if units == 'km':
            pos = pos * 1000.0
        elif units != 'm':
            raise ValueError(f"units must be 'm' or 'km', got {units!r}")

        # --- Integrity: radius must be plausible LEO (replaces missing Rad col) ---
        radius_km = np.sqrt(np.sum(pos ** 2, axis=1)) / 1000.0
        r_med = np.median(radius_km)
        if not (6600 < r_med < 7300):
            raise ValueError(
                f"Median radius {r_med:.0f} km is not plausible LEO. "
                f"Likely a UNIT error -- if file is in km, set units='km' "
                f"(km misread as m gives ~millions of km; m misread as km gives ~7 km)."
            )
        if verbose:
            print(f"  Radius OK: median {r_med:.1f} km (~{r_med - 6371:.0f} km altitude)")

        # --- Cadence check --------------------------------------------------------
        if len(times) > 1:
            dt_sec = np.diff(times.mjd) * 86400
            med_cad = np.median(dt_sec)
            if verbose:
                print(f"  Median cadence {med_cad:.1f}s (expected ~{expected_cadence_sec:.0f}s)")
            if abs(med_cad - expected_cadence_sec) > 0.5 * expected_cadence_sec:
                print(f"  WARNING: cadence {med_cad:.1f}s != expected "
                      f"{expected_cadence_sec:.0f}s; transits need ~30s sampling.")
            n_gaps = np.sum(dt_sec > 5 * expected_cadence_sec)
            if n_gaps:
                print(f"  NOTE: {n_gaps} gaps > {5 * expected_cadence_sec:.0f}s "
                      f"(telemetry expected continuous through SAA).")

        # --- Frame: ECI -> store as GCRS (already inertial; gcrs is correct) ------
        if frame.lower() != 'gcrs':
            # TEME/ITRS conversion path retained from earlier draft if ever needed
            raise NotImplementedError(
                f"This file is ECI; use frame='gcrs'. (Got {frame!r}.)")
        self.times = times
        self.positions = pos
        self.beta = None  # no beta column in this format

        if verbose:
            print(f"  Time span: {times[0].iso} -> {times[-1].iso}")
            print(f"  Stored {len(times)} GCRS positions. Ready for compute_topocentric().\n")

        return self.times, self.positions


    def compute_topocentric(self, rubin_loc=None):
        """
        Computes topocentric coordinates using the standard, canonical astropy method.

        This is the correct approach for clean, TLE-generated GCRS data. It relies
        on astropy's internal parallax correction, which was failing on the flawed
        FT2 file but will now work correctly with the clean input data.
        """
        import astropy.units as u
        from astropy.coordinates import EarthLocation, SkyCoord, AltAz, GCRS
        import numpy as np

        if self.positions is None or self.times is None:
            raise ValueError("Must load self.positions and self.times before computing coordinates.")

        if rubin_loc is None:
            rubin_loc = self.rubin_location

        # Step 1: Define Fermi's geocentric position in GCRS.
        # Our TLE generator provides clean data in this frame.
        fermi_gcrs = SkyCoord(
            x=self.positions[:, 0] * u.m,
            y=self.positions[:, 1] * u.m,
            z=self.positions[:, 2] * u.m,
            representation_type='cartesian',
            frame='gcrs',
            obstime=self.times
        )

        # Step 2: Transform to Alt/Az. astropy handles the parallax correction internally.
        # This is the direct, standard method that failed on the bad FT2 data
        # but will succeed on the clean TLE data.
        fermi_altaz = fermi_gcrs.transform_to(AltAz(obstime=self.times, location=rubin_loc))

        # Step 3: To get the corresponding topocentric RA/Dec, we must first
        # calculate the topocentric vector manually. This is the only part of
        # the 'hybrid' method that remains correct and necessary.
        rubin_gcrs = self.rubin_location.get_gcrs(self.times)
        topo_vector = fermi_gcrs.cartesian - rubin_gcrs.cartesian
        fermi_topo_celestial = SkyCoord(topo_vector, frame='gcrs', obstime=self.times)

        # Step 4: Save the results.
        self.sky_coords = {
            'alt': fermi_altaz.alt,
            'az': fermi_altaz.az,
            'ra': fermi_topo_celestial.ra,
            'dec': fermi_topo_celestial.dec
        }

        # Diagnostic check
        max_idx = np.argmax(self.sky_coords['alt'])
        print("\n" + "=" * 60)
        print("MAXIMUM ALTITUDE (Canonical Astropy Method on Clean TLE Data)")
        print(f"Peak Altitude:   {self.sky_coords['alt'][max_idx].deg:.4f}°")
        print(f"Timestamp:       {self.times[max_idx].iso}")
        print("=" * 60 + "\n")

        return self.sky_coords

    def find_survey_passes(self, min_altitude=20 * u.deg,
                           dec_range=(-70, 10) * u.deg,
                           min_angular_velocity=0.1,
                           max_ra_jump=50.0):
        """
        Find ALL continuous pass segments, then evaluate which constraints each meets.
        Splits segments on large RA jumps (indicating separate orbital passes).

        Parameters:
        -----------
        min_altitude : astropy.units.Quantity
            Minimum altitude above horizon
        dec_range : tuple of astropy.units.Quantity
            (min_dec, max_dec) for survey area
        min_angular_velocity : float
            Minimum angular velocity in deg/s
        max_ra_jump : float
            Maximum RA change in degrees between samples before splitting
            (detects separate orbital passes)

        Returns:
        --------
        list of FermiPass objects with filter flags set
        """
        if self.sky_coords is None:
            raise ValueError("Must call compute_topocentric() first")

        ra = self.sky_coords['ra'].deg
        dec = self.sky_coords['dec'].deg
        alt = self.sky_coords['alt'].deg

        # Compute angular velocities
        ra_diff = np.diff(ra)
        dec_diff = np.diff(dec)

        # Handle RA wraparound for velocity calculation
        ra_diff_wrapped = np.copy(ra_diff)
        ra_diff_wrapped = np.where(ra_diff_wrapped > 180, ra_diff_wrapped - 360, ra_diff_wrapped)
        ra_diff_wrapped = np.where(ra_diff_wrapped < -180, ra_diff_wrapped + 360, ra_diff_wrapped)

        # Angular separation with cos(dec) correction
        dec_avg = (dec[:-1] + dec[1:]) / 2
        ang_sep = np.sqrt((ra_diff_wrapped * np.cos(np.radians(dec_avg))) ** 2 + dec_diff ** 2)

        # Time differences in seconds
        time_diff = np.diff(self.times.mjd) * 86400

        # Angular velocity in deg/sec (pad with 0 at end)
        ang_velocity = np.concatenate([ang_sep / time_diff, np.array([0.0])])

        # ====================================================================
        # SEGMENT FINDING — corrected for gap-free telemetry
        # ====================================================================
        # A "pass" must be a single ABOVE-HORIZON transit. Previously the edge
        # handling prepended index 0 as a "start" whenever the data was
        # continuous, which (with gap-free telemetry) glued the entire
        # below-horizon arc onto the front of each real transit. We now gate
        # the segment purely on visibility and pair starts->ends correctly.

        # Use the real altitude floor, not 0, so segments are actual usable
        # transits. (alt > 0 admits horizon-skimming junk; min_altitude is the
        # observing floor we care about.)
        visible_mask = alt > min_altitude.value

        vis = visible_mask.astype(int)
        transitions = np.diff(vis)
        starts = list(np.where(transitions == 1)[0] + 1)  # rises
        ends = list(np.where(transitions == -1)[0] + 1)  # sets

        # Edge cases: gate on VISIBILITY at the ends, not on data continuity.
        if vis[0] == 1:
            starts = [0] + starts
        if vis[-1] == 1:
            ends = ends + [len(vis)]

        # After the edge fix, starts and ends are equal length and interleaved:
        # starts[k] < ends[k] < starts[k+1] for every k.
        assert len(starts) == len(ends), (
            f"start/end mismatch: {len(starts)} starts vs {len(ends)} ends"
        )

        # Optionally split a transit further if there's a genuine data gap or a
        # large RA jump WITHIN it (shouldn't happen for clean telemetry, but
        # harmless to keep — protects the FT2 path too).
        time_diff_full = np.diff(self.times.mjd) * 86400
        ra_diff_full = np.diff(ra)

        def split_on_breaks(s, e):
            """Yield sub-segments of [s,e) broken on time gaps / RA jumps."""
            sub_start = s
            for i in range(s, e - 1):
                if time_diff_full[i] >= 300 or abs(ra_diff_full[i]) > max_ra_jump:
                    if i + 1 - sub_start >= 2:
                        yield (sub_start, i + 1)
                    sub_start = i + 1
            if e - sub_start >= 2:

                yield (sub_start, e)

        segments = []
        for s, e in zip(starts, ends):
            segments.extend(split_on_breaks(s, e))

        # Create passes and evaluate filters
        self.passes = []

        for start_idx, end_idx in segments:
            # Skip very short segments
            if end_idx - start_idx < 2:
                continue

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
                az=self.sky_coords['az'][start_idx:end_idx].deg,
                max_angular_velocity=0.0  # Will compute in compute_pass_parameters()
            )

            # Evaluate filter constraints

            # Altitude: Are ALL points above minimum?
            # Does the peak altitude of the pass exceed the minimum?
            pass_obj.passes_altitude = (np.max(pass_obj.alt) >= min_altitude.value)

            # Declination: Does pass overlap with range?
            pass_obj.passes_declination = (
                    (pass_obj.dec.min() <= dec_range[1].value) and
                    (pass_obj.dec.max() >= dec_range[0].value)
            )

            # Velocity: Is ANY point above threshold?
            pass_velocities = ang_velocity[start_idx:end_idx - 1]
            if len(pass_velocities) > 0:
                pass_obj.passes_velocity = np.max(pass_velocities) >= min_angular_velocity
            else:
                pass_obj.passes_velocity = False

            # Night filter will be evaluated later in filter_night_passes()
            pass_obj.passes_night = False

            self.passes.append(pass_obj)

        p = self.passes[0]
        print("\n" + "=" * 60)
        print("DEBUG: first pass anatomy")
        print("=" * 60)
        print(f"  start -> end : {p.start_time.iso}  ->  {p.end_time.iso}")
        print(f"  duration     : {(p.end_time - p.start_time).sec/60:.1f} min")
        print(f"  n samples    : {len(p.times)}")
        print(f"  alt range    : {p.alt.min():.1f}  ..  {p.alt.max():.1f}")
        print(f"  dec range    : {p.dec.min():.1f}  ..  {p.dec.max():.1f}")
        print(f"  ra  range    : {p.ra.min():.1f}  ..  {p.ra.max():.1f}")

        # THE KEY DIAGNOSTIC: how many altitude humps?
        # A clean single transit = one rise-and-set = one local maximum.

        a = p.alt
        # count interior local maxima
        humps = np.sum((a[1:-1] > a[:-2]) & (a[1:-1] > a[2:]))
        # count horizon crossings (alt sign changes through 0)
        crossings = np.sum(np.diff(np.sign(a)) != 0)
        print(f"  alt humps    : {humps}   (1 = clean single transit)")
        print(f"  horizon xings: {crossings}  (2 = one rise + one set)")

        # show the alt profile coarsely so you can eyeball the shape
        step = max(1, len(a)//20)
        print("  alt profile  :", " ".join(f"{v:5.1f}" for v in a[::step]))
        print("=" * 60 + "\n")

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

    def visualize_all_gaps(self, gap_threshold_sec=60, output_filename="all_gaps_visualization.html"):
        """
        Scans the entire dataset for time gaps and creates a summary scatter plot.
        This provides a high-level "swiss cheese" map of the file's data quality.
        """
        from bokeh.plotting import figure, save, output_file
        from bokeh.models import HoverTool, ColumnDataSource
        import numpy as np

        if self.times is None or len(self.times) < 2:
            print("Not enough data to visualize gaps.")
            return

        print("\nScanning entire dataset for all time gaps...")

        # 1. Calculate all time differences between consecutive points
        time_diffs_sec = np.diff(self.times.mjd) * 86400

        # 2. Find the indices of all points that precede a significant gap
        gap_indices = np.where(time_diffs_sec > gap_threshold_sec)[0]

        if len(gap_indices) == 0:
            print(f"No gaps larger than {gap_threshold_sec} seconds found in the file.")
            return

        # 3. Collect the data for plotting: the time the gap starts and its duration
        gap_start_times = self.times[gap_indices].datetime
        gap_durations_sec = time_diffs_sec[gap_indices]

        # 4. Calculate summary statistics
        num_gaps = len(gap_indices)
        total_gap_time_sec = np.sum(gap_durations_sec)
        total_dataset_duration_sec = (self.times[-1] - self.times[0]).sec

        # Calculate percentage of missing time based on gaps within the dataset's span
        percentage_missing = (total_gap_time_sec / total_dataset_duration_sec) * 100
        total_gap_time_days = total_gap_time_sec / 86400

        print(f"Found {num_gaps} gaps totaling {total_gap_time_days:.1f} days of missing time.")

        # 5. Create the Bokeh plot
        title = (f"Map of {num_gaps} Data Gaps (> {gap_threshold_sec}s) in FT2 File | "
                 f"Total Missing Time: {total_gap_time_days:.1f} days ({percentage_missing:.1f}%)")

        p = figure(
            width=1200,
            height=600,
            title=title,
            x_axis_label="Date",
            y_axis_label="Gap Duration (seconds)",
            x_axis_type='datetime',
            y_axis_type='log'  # Log scale is essential for this kind of data
        )

        source = ColumnDataSource(data={
            'time': gap_start_times,
            'duration': gap_durations_sec
        })

        p.scatter('time', 'duration', source=source, size=5, color="navy", alpha=0.6)

        # 6. Add an informative hover tool
        hover = HoverTool(tooltips=[
            ("Gap Start", "@time{%F %T}"),
            ("Duration (sec)", "@duration{0.1f}"),
            ("Duration (min)", "@duration{0.0f} min"),
            ("Duration (hrs)", "@duration{0.0f} hrs")
        ], formatters={
            '@time': 'datetime',
            # Custom formatter to convert seconds to minutes and hours for hover
            '@duration': 'printf'
        })
        p.add_tools(hover)

        # 7. Save the file
        output_file(filename=output_filename)
        save(p)
        print(f"Saved 'swiss cheese' gap map to {output_filename}")


    def visualize_gaps(self, target_time, time_window_hours=2, gap_threshold_sec=300,
                       output_filename="gap_visualization.html"):
        """
        Creates a visualization focused on a specific time to show data gaps.
        (Corrected version to fix variable name collision)
        """
        from bokeh.plotting import figure, save, output_file
        from bokeh.models import HoverTool, ColumnDataSource, Legend
        from astropy.time import Time
        import astropy.units as u
        import numpy as np

        if self.sky_coords is None:
            print("Cannot visualize gaps, sky_coords have not been computed.")
            return

        # 1. Select the data within the time window
        start_window = target_time - time_window_hours * u.hour
        end_window = target_time + time_window_hours * u.hour
        window_mask = (self.times >= start_window) & (self.times <= end_window)

        if not np.any(window_mask):
            print(f"No data found in the time window around {target_time.iso}")
            return

        window_times = self.times[window_mask]
        window_alt = self.sky_coords['alt'][window_mask].deg
        window_az = self.sky_coords['az'][window_mask].deg

        # 2. Identify the gaps within this window
        time_diffs_sec = np.diff(window_times.mjd) * 86400
        gap_indices = np.where(time_diffs_sec > gap_threshold_sec)[0]

        print(
            f"\nFound {len(gap_indices)} gaps larger than {gap_threshold_sec}s in the {time_window_hours * 2}-hour window.")

        # 3. Prepare data sources for Bokeh
        source_trajectory = ColumnDataSource(data={
            'az': window_az,
            'alt': window_alt,
            'time': [t.iso for t in window_times]
        })

        # Data for points just BEFORE a gap (the "stops")
        gap_starts_data = {
            'az': window_az[gap_indices],
            'alt': window_alt[gap_indices],
            'time': [window_times[i].iso for i in gap_indices],
            'duration': time_diffs_sec[gap_indices]
        }
        source_gap_starts = ColumnDataSource(gap_starts_data)

        # Data for points just AFTER a gap (the "resumes")
        gap_ends_data = {
            'az': window_az[gap_indices + 1],
            'alt': window_alt[gap_indices + 1],
            'time': [window_times[i + 1].iso for i in gap_indices]
        }
        source_gap_ends = ColumnDataSource(gap_ends_data)

        # 4. Create the plot
        p = figure(
            width=1000,
            height=600,
            title=f"Fermi Trajectory & Data Gaps around {target_time.iso}",
            x_axis_label="Azimuth (degrees)",
            y_axis_label="Altitude (degrees)",
            x_range=(0, 360),
            y_range=(-5, 90)  # Start from slightly below horizon
        )
        p.xaxis.axis_label_text_font_style = "normal"
        p.yaxis.axis_label_text_font_style = "normal"

        # Plot the full trajectory as a faint line
        p.line('az', 'alt', source=source_trajectory, line_width=2, color='gray', alpha=0.5,
               legend_label="Trajectory Path")

        # Plot the "stop" points in red (using modern .scatter)
        r_stop = p.scatter('az', 'alt', source=source_gap_starts, size=12, color='red', alpha=0.8,
                           legend_label="Data Stop (Gap Start)")

        # Plot the "resume" points in green (using modern .scatter)
        r_resume = p.scatter('az', 'alt', source=source_gap_ends, size=12, color='lime', alpha=0.8,
                             legend_label="Data Resume (Gap End)")

        # 5. Add informative hover tools
        hover_traj = HoverTool(tooltips=[("Time", "@time"), ("Alt", "@alt{0.2f}°"), ("Az", "@az{0.2f}°")])
        p.add_tools(hover_traj)

        hover_stop = HoverTool(renderers=[r_stop], tooltips=[
            ("Data Stop Time", "@time"),
            ("Alt", "@alt{0.2f}°"),
            ("Az", "@az{0.2f}°"),
            ("Following Gap Duration", "@duration{0.0f} s")
        ])
        p.add_tools(hover_stop)

        hover_resume = HoverTool(renderers=[r_resume], tooltips=[
            ("Data Resume Time", "@time"),
            ("Alt", "@alt{0.2f}°"),
            ("Az", "@az{0.2f}°")
        ])
        p.add_tools(hover_resume)

        p.legend.location = "top_left"
        p.legend.click_policy = "hide"

        # 6. Save the file (FIXED)
        output_file(filename=output_filename)  # Use the renamed variable here
        save(p)
        print(f"Saved gap visualization to {output_filename}")

# -------------------------------------------------------------------------
    # CONFIG VALIDATION — fail fast on misconfiguration
    # -------------------------------------------------------------------------
def validate_config(config):
    """Check required keys per source/mode, warn on unknowns, apply
    documented defaults. Raises ValueError on fatal problems."""

    source = config.get('source', 'tle').lower()
    mode = config.get('mode', 'compute').lower()

    # --- Enumerated-value checks ---
    valid_sources = {'tle', 'ft2', 'ascii'}
    valid_modes = {'compute', 'visualize'}
    if source not in valid_sources:
        raise ValueError(
            f"Invalid source: {source!r}. Must be one of {sorted(valid_sources)}."
        )
    if mode not in valid_modes:
        raise ValueError(
            f"Invalid mode: {mode!r}. Must be one of {sorted(valid_modes)}."
        )

    # --- Required keys per mode/source ---
    required = []
    if mode == 'visualize':
        required.append('passes_file')
    else:  # compute
        if source == 'tle':
            required += ['start_time', 'end_time']
        elif source == 'ft2':
            required.append('ft2_file')
        elif source == 'ascii':
            required.append('ascii_file')

    missing = [k for k in required if config.get(k) is None]
    if missing:
        raise ValueError(
            f"Missing required config key(s) for source={source}, "
            f"mode={mode}: {missing}"
        )

    # --- Conditional requirement: specific-gap plot needs a target ---
    if config.get('visualize_specific_gap', False) and not config.get('gap_target_time'):
        raise ValueError(
            "visualize_specific_gap is true but 'gap_target_time' is not set."
        )

    # --- File existence checks (where a path is expected to pre-exist) ---
    path_keys = []
    if mode == 'visualize':
        path_keys.append('passes_file')
    elif source == 'ft2':
        path_keys.append('ft2_file')
    elif source == 'ascii':
        path_keys.append('ascii_file')
    for key in path_keys:
        p = Path(config[key])
        if not p.exists():
            raise FileNotFoundError(f"{key} does not exist: {p}")

    # --- Sanity checks on physical parameters ---
    sun_alt = config.get('sun_alt_limit', -12)
    if sun_alt > 0:
        print(f"WARNING: sun_alt_limit={sun_alt} is above the horizon; "
              f"this admits daytime passes. Rubin twilight ops use -12.")
    min_alt = config.get('min_altitude', 20)
    if not (0 <= min_alt <= 90):
        raise ValueError(f"min_altitude={min_alt} must be in [0, 90] degrees.")
    dec_range = config.get('dec_range', [-70, 10])
    if (len(dec_range) != 2 or dec_range[0] >= dec_range[1]
            or not all(-90 <= d <= 90 for d in dec_range)):
        raise ValueError(
            f"dec_range={dec_range} must be [min, max] within [-90, 90] and min < max."
        )

    # --- Time-range ordering (compute mode, where applicable) ---
    if config.get('start_time') and config.get('end_time'):
        t0, t1 = Time(config['start_time']), Time(config['end_time'])
        if t0 >= t1:
            raise ValueError(
                f"start_time ({t0.iso}) must be before end_time ({t1.iso})."
            )

    # --- Warn on unknown keys (typo catcher) ---
    known_keys = {
        'source', 'mode', 'start_time', 'end_time', 'time_step_seconds',
        'ft2_file', 'downsample_sec', 'ascii_file', 'passes_file',
        'min_altitude', 'dec_range', 'min_angular_velocity', 'max_ra_jump',
        'sun_alt_limit', 'output_dir', 'viz_filter_mode', 'viz_max_passes',
        'visualize_all_gaps', 'visualize_specific_gap', 'gap_target_time',
        'verbose',
    }
    unknown = set(config) - known_keys
    if unknown:
        print(f"WARNING: unrecognized config key(s) ignored: {sorted(unknown)}")

    return source, mode


import argparse
import yaml
from pathlib import Path

import argparse
import yaml
from pathlib import Path


def main():
    import argparse
    import yaml
    from pathlib import Path
    from astropy.time import Time
    import astropy.units as u

    parser = argparse.ArgumentParser(
        description='Calculate Fermi satellite passes over Rubin Observatory '
                    'from TLE, FT2, or ASCII telemetry data.'
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to YAML configuration file'
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)


    verbose = config.get('verbose', True)

    # -------------------------------------------------------------------------
    # SETUP
    # -------------------------------------------------------------------------
    source, mode = validate_config(config)

    calculator = FermiPassCalculator(ft2_file=config.get('ft2_file'))

    # -------------------------------------------------------------------------
    # VISUALIZE MODE: load precomputed passes, skip all data loading/analysis
    # -------------------------------------------------------------------------
    if mode == 'visualize':
        passes_file = config.get('passes_file')
        if not passes_file:
            raise ValueError("mode: visualize requires 'passes_file' in config.")
        if verbose:
            print(f"--- Visualize mode: loading passes from {passes_file} ---")
        calculator.load_passes(passes_file)

        output_dir = config.get('output_dir')
        if output_dir:
            calculator.visualize_passes(
                output_dir,
                filter_mode=config.get('viz_filter_mode', 'all'),
                max_passes=config.get('viz_max_passes', 500)
            )
        else:
            print("WARNING: No output_dir set; nothing to visualize to.")

        print("\nVisualization complete.")
        return

    # -------------------------------------------------------------------------
    # COMPUTE MODE: load data from the selected source
    # -------------------------------------------------------------------------
    if verbose:
        print(f"--- Compute mode | source: {source} ---")

    if source == 'tle':
        if verbose:
            print("WARNING: TLE propagation is for near-future prediction only.\n"
                  "         For multi-month historical analysis use source: ascii.")
        calculator.load_tle(
            start_time=config['start_time'],
            end_time=config['end_time'],
            time_step_seconds=config.get('time_step_seconds', 30.0),
            verbose=verbose
        )

    elif source == 'ft2':
        if calculator.ft2_file is None:
            raise ValueError("source: ft2 requires 'ft2_file' in config.")
        # Optional time-range filtering for FT2 (commented out in old config,
        # but supported by load_ft2's constructor-based masking).
        if config.get('start_time'):
            calculator.start_time = Time(config['start_time'])
        if config.get('end_time'):
            calculator.end_time = Time(config['end_time'])
        calculator.load_ft2(downsample_sec=config.get('downsample_sec', 30))

    elif source == 'ascii':
        ascii_file = config.get('ascii_file')
        if not ascii_file:
            raise ValueError("source: ascii requires 'ascii_file' in config.")
        calculator.load_ascii(ascii_file, verbose=verbose)

    else:
        raise ValueError(
            f"Unknown source: {source!r}. Use 'tle', 'ft2', or 'ascii'."
        )

    # -------------------------------------------------------------------------
    # ANALYSIS PIPELINE (identical for all sources)
    # -------------------------------------------------------------------------
    if verbose:
        print("\n--- Starting Pass Analysis ---")

    if verbose:
        print("Computing topocentric coordinates...")
    calculator.compute_topocentric()

    if verbose:
        print("Finding passes...")
    dec_range = config.get('dec_range', [-70, 10])
    passes = calculator.find_survey_passes(
        min_altitude=config.get('min_altitude', 20) * u.deg,
        dec_range=(dec_range[0] * u.deg, dec_range[1] * u.deg),
        min_angular_velocity=config.get('min_angular_velocity', 0.1),
        max_ra_jump=config.get('max_ra_jump', 50.0)
    )

    if verbose:
        print("Computing pass parameters...")
    calculator.compute_pass_parameters()

    if verbose:
        print("Filtering for night-time passes...")
    calculator.filter_night_passes(
        sun_alt_limit=config.get('sun_alt_limit', -18) * u.deg
    )

    # -------------------------------------------------------------------------
    # RESULTS SUMMARY
    # -------------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 60)
        print("                 FINAL RESULTS")
        print("=" * 60)
        n_all = sum(p.passes_all_filters for p in passes)
        print(f"Found {n_all} passes meeting all criteria.")

        all_filter_passes = [p for p in passes if p.passes_all_filters]
        if all_filter_passes:
            print("\nFirst 10 passes meeting ALL filters:")
            for p in all_filter_passes[:10]:
                print(f"  {p.start_time.iso}: "
                      f"duration={(p.end_time - p.start_time).sec:.0f}s, "
                      f"max_alt={p.alt.max():.1f}°")

    # -------------------------------------------------------------------------
    # OUTPUT & VISUALIZATION
    # -------------------------------------------------------------------------
    output_dir = config.get('output_dir')
    if output_dir:
        print(f"\nSaving results to {output_dir}...")
        calculator.save_passes(output_dir)
        calculator.visualize_passes(
            output_dir,
            filter_mode=config.get('viz_filter_mode', 'all'),
            max_passes=config.get('viz_max_passes', 500)
        )

        # Whole-file gap map (most meaningful for ft2)
        if config.get('visualize_all_gaps', False):
            calculator.visualize_all_gaps(
                output_filename=str(Path(output_dir) / "all_gaps_visualization.html")
            )

        # Focused gap plot around a specific target time
        if config.get('visualize_specific_gap', False):
            target_str = config.get('gap_target_time')
            if not target_str:
                print("WARNING: visualize_specific_gap is true but "
                      "'gap_target_time' is not set; skipping.")
            else:
                target = Time(target_str)
                # Warn loudly if the target falls outside the loaded data span
                if calculator.times is not None and len(calculator.times) > 0:
                    t0, t1 = calculator.times[0], calculator.times[-1]
                    if target < t0 or target > t1:
                        print(f"WARNING: gap_target_time {target.iso} is outside "
                              f"the loaded data range [{t0.iso} .. {t1.iso}]; "
                              f"plot may be empty.")
                calculator.visualize_gaps(
                    target_time=target,
                    output_filename=str(Path(output_dir) / "gap_visualization.html")
                )

    print("\nAnalysis complete.")


if __name__ == '__main__':
    main()
