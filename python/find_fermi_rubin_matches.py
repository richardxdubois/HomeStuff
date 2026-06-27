#!/usr/bin/env python
"""
find_fermi_rubin_matches.py

Search for overlaps between Fermi satellite passes and Rubin Observatory visits.
Loads pre-computed Fermi passes and queries Rubin observation database for
potential detections. Now includes CCD-level matching.
"""

import argparse
import pickle
import yaml
import numpy as np
from pathlib import Path
from astropy.time import Time
import astropy.units as u
from Rubin_Fermi import FermiPass

# For Rubin data access
try:
    from lsst.daf.butler import Butler, Timespan

    HAVE_BUTLER = True
except ImportError:
    HAVE_BUTLER = False
    print("Warning: lsst.daf.butler not available. Butler queries will fail.")


def point_to_arc_distance(ra_point, dec_point, ra_start, dec_start, ra_end, dec_end):
    """
    Minimum angular distance from a point to a great circle arc

    Parameters:
    -----------
    ra_point, dec_point : float
        Point coordinates (degrees)
    ra_start, dec_start : float
        Arc start point (degrees)
    ra_end, dec_end : float
        Arc end point (degrees)

    Returns:
    --------
    min_distance : float
        Minimum angular distance in degrees
    """
    # Handle RA wraparound by adjusting to continuous range
    ra_s, ra_e, ra_p = ra_start, ra_end, ra_point

    # If arc crosses 0/360 boundary, shift to continuous range
    if abs(ra_e - ra_s) > 180:
        if ra_e < ra_s:
            ra_e += 360
        else:
            ra_s += 360
        # Adjust point too if needed
        if abs(ra_p - ra_s) > 180:
            if ra_p < ra_s:
                ra_p += 360
        if abs(ra_p - ra_e) > 180:
            if ra_p < ra_e:
                ra_p += 360

    # Convert to radians
    ra_p_rad, dec_p_rad = np.radians(ra_p), np.radians(dec_point)
    ra_s_rad, dec_s_rad = np.radians(ra_s), np.radians(dec_start)
    ra_e_rad, dec_e_rad = np.radians(ra_e), np.radians(dec_end)

    # Convert to Cartesian unit vectors
    def radec_to_cartesian(ra, dec):
        x = np.cos(dec) * np.cos(ra)
        y = np.cos(dec) * np.sin(ra)
        z = np.sin(dec)
        return np.array([x, y, z])

    p = radec_to_cartesian(ra_p_rad, dec_p_rad)
    s = radec_to_cartesian(ra_s_rad, dec_s_rad)
    e = radec_to_cartesian(ra_e_rad, dec_e_rad)

    # Normal to the great circle plane
    normal = np.cross(s, e)
    normal_mag = np.linalg.norm(normal)

    if normal_mag < 1e-10:
        # Degenerate case: start and end are same/antipodal
        dist_to_start = np.arccos(np.clip(np.dot(p, s), -1, 1))
        return np.degrees(dist_to_start)

    normal = normal / normal_mag

    # Distance from point to the great circle
    perp_dist = np.arcsin(np.clip(np.abs(np.dot(p, normal)), 0, 1))

    # Project point onto the great circle plane
    p_proj = p - np.dot(p, normal) * normal
    p_proj_norm = np.linalg.norm(p_proj)

    if p_proj_norm < 1e-10:
        # Point is at pole of the great circle
        dist_to_start = np.arccos(np.clip(np.dot(p, s), -1, 1))
        dist_to_end = np.arccos(np.clip(np.dot(p, e), -1, 1))
        return np.degrees(min(dist_to_start, dist_to_end))

    p_proj = p_proj / p_proj_norm

    # Check if projected point is on the arc between start and end
    angle_s_to_e = np.arccos(np.clip(np.dot(s, e), -1, 1))
    angle_s_to_p = np.arccos(np.clip(np.dot(s, p_proj), -1, 1))
    angle_p_to_e = np.arccos(np.clip(np.dot(p_proj, e), -1, 1))

    # If p_proj is on the arc (within tolerance)
    if np.abs((angle_s_to_p + angle_p_to_e) - angle_s_to_e) < 1e-4:
        return np.degrees(perp_dist)
    else:
        # Closest point is an endpoint
        dist_to_start = np.arccos(np.clip(np.dot(p, s), -1, 1))
        dist_to_end = np.arccos(np.clip(np.dot(p, e), -1, 1))
        return np.degrees(min(dist_to_start, dist_to_end))


def load_fermi_passes(pickle_file, start_time=None, end_time=None):
    """
    Load Fermi passes from pickle file with optional time filtering

    Parameters:
    -----------
    pickle_file : str or Path
        Path to pickle file containing passes
    start_time : str or astropy.time.Time, optional
        Only include passes that end after this time (ISO format string or Time object)
    end_time : str or astropy.time.Time, optional
        Only include passes that start before this time (ISO format string or Time object)

    Returns:
    --------
    passes : list
        List of FermiPass objects (filtered by time if specified)
    """
    with open(pickle_file, 'rb') as f:
        passes = pickle.load(f)

    print(f"Loaded {len(passes)} Fermi passes from {pickle_file}")

    # Apply time filtering if requested
    if start_time is not None or end_time is not None:
        # Convert to Time objects if needed
        if start_time is not None and isinstance(start_time, str):
            start_time = Time(start_time, format='iso')
        if end_time is not None and isinstance(end_time, str):
            end_time = Time(end_time, format='iso')

        original_count = len(passes)
        filtered_passes = []

        for p in passes:
            # Keep pass if it overlaps the time window
            if start_time is not None and p.end_time < start_time:
                continue
            if end_time is not None and p.start_time > end_time:
                continue
            filtered_passes.append(p)

        passes = filtered_passes
        print(f"Time filter applied: {original_count} -> {len(passes)} passes")
        if start_time:
            print(f"  After: {start_time.iso}")
        if end_time:
            print(f"  Before: {end_time.iso}")

    return passes


def get_rubin_visits_butler(butler, start_time, end_time, instrument='LSSTComCam'):
    """
    Get Rubin visits in time range using Butler

    Parameters:
    -----------
    butler : lsst.daf.butler.Butler
    start_time : astropy.time.Time
    end_time : astropy.time.Time
    instrument : str
        Instrument name (e.g., 'LSSTComCam', 'LATISS')

    Returns:
    --------
    visits : list of dict
        Each dict has: visit_id, obs_time, pointing_ra, pointing_dec
    """
    # Convert to TAI
    start_tai = start_time.tai.datetime64
    end_tai = end_time.tai.datetime64
    timespan = Timespan(begin=start_time, end=end_time)

    try:
        where_clause = "instrument = 'LSSTCam' AND visit.timespan OVERLAPS timespan"
        bind_params = {"timespan": timespan}

        visit_records = butler.query_dimension_records(
            "visit",
            where=where_clause,
            bind=bind_params
        )
    except Exception as e:
        print(f"  Warning: Butler query failed: {e}")
        return []

    visits = []
    for visit in visit_records:
        try:
            # Extract pointing
            ra = visit.region.center.getRa().asDegrees()
            dec = visit.region.center.getDec().asDegrees()

            # Get observation time (midpoint)
            mid_mjd = (visit.timespan.begin.tai + visit.timespan.end.tai) / 2
            obs_time = Time(mid_mjd, format='mjd', scale='tai')

            visits.append({
                'visit_id': visit.id,
                'obs_time': obs_time,
                'pointing_ra': ra,
                'pointing_dec': dec,
                'instrument': instrument,
                'timespan': visit.timespan
            })
        except Exception as e:
            print(f"  Warning: Could not process visit {visit.id}: {e}")
            continue

    return visits


def find_ccd_matches(butler, visit_id, fermi_ra, fermi_dec, instrument='LSSTComCam', verbose=False):
    """
    Determine which CCDs Fermi's position falls on.

    Parameters:
    -----------
    butler : lsst.daf.butler.Butler
        Butler instance
    visit_id : int
        Visit identifier
    fermi_ra : array-like
        Fermi RA positions during exposure (degrees)
    fermi_dec : array-like
        Fermi Dec positions during exposure (degrees)
    instrument : str
        Instrument name
    verbose : bool
        Print detailed information

    Returns:
    --------
    list of dicts with CCD match information
    """
    ccd_matches = []

    try:
        # Get camera geometry
        camera = butler.get('camera', instrument=instrument)

        if verbose:
            print(f"    Camera has {len(camera)} detectors")

        # Check each detector
        for detector in camera:
            detector_id = detector.getId()
            detector_name = detector.getName()

            try:
                # Get WCS for this detector/visit
                wcs = butler.get('wcs',
                                 dataId={'visit': visit_id,
                                         'detector': detector_id,
                                         'instrument': instrument})

                # Get detector bounding box
                bbox = detector.getBBox()

                # Get corners in sky coordinates
                corners_pix = [
                    (bbox.getMinX(), bbox.getMinY()),
                    (bbox.getMaxX(), bbox.getMinY()),
                    (bbox.getMaxX(), bbox.getMaxY()),
                    (bbox.getMinX(), bbox.getMaxY())
                ]

                corners_sky = [wcs.pixelToSky(x, y) for x, y in corners_pix]

                # Get RA/Dec ranges
                corner_ras = [c.getRa().asDegrees() for c in corners_sky]
                corner_decs = [c.getDec().asDegrees() for c in corners_sky]

                ra_min, ra_max = min(corner_ras), max(corner_ras)
                dec_min, dec_max = min(corner_decs), max(corner_decs)

                # Check if any Fermi position falls within this detector
                # Simple bounding box check
                for i, (fra, fdec) in enumerate(zip(fermi_ra, fermi_dec)):
                    # Handle RA wraparound if needed
                    if ra_max - ra_min > 180:  # Wraparound case
                        in_ra = (fra >= ra_min or fra <= ra_max)
                    else:
                        in_ra = (ra_min <= fra <= ra_max)

                    in_dec = (dec_min <= fdec <= dec_max)

                    if in_ra and in_dec:
                        # Found a match - try to get pixel coordinates
                        try:
                            pixel_pos = wcs.skyToPixel(corners_sky[0].__class__(
                                fra * corners_sky[0].getRa().Units,
                                fdec * corners_sky[0].getDec().Units
                            ))
                            pixel_x, pixel_y = pixel_pos
                        except:
                            pixel_x, pixel_y = None, None

                        ccd_matches.append({
                            'detector_id': detector_id,
                            'detector_name': detector_name,
                            'fermi_ra': fra,
                            'fermi_dec': fdec,
                            'pixel_x': pixel_x,
                            'pixel_y': pixel_y,
                            'time_index': i
                        })
                        break  # Just record that this CCD has a match

            except Exception as e:
                # WCS might not be available for all detectors
                if verbose:
                    print(f"    Could not check detector {detector_name}: {e}")
                continue

    except Exception as e:
        if verbose:
            print(f"    Could not load camera geometry: {e}")
        return []

    if ccd_matches and verbose:
        print(f"    Fermi crosses {len(ccd_matches)} CCDs: " +
              ", ".join([m['detector_name'] for m in ccd_matches]))

    return ccd_matches


def find_pass_overlaps(fermi_pass, rubin_visits, butler, fov_radius=1.75, buffer=0.1,
                       check_ccds=True, instrument='LSSTComCam', verbose=False,
                       n_trajectory_points=10):
    """
    Find Rubin visits that potentially overlap with a Fermi pass

    Parameters:
    -----------
    fermi_pass : FermiPass
        Fermi pass object with trajectory
    rubin_visits : list of dict
        Rubin visit information
    butler : lsst.daf.butler.Butler or None
        Butler instance for CCD queries
    fov_radius : float
        Rubin field of view radius in degrees (default 1.75 for 3.5° diameter)
    buffer : float
        Additional buffer in degrees for conservative matching
    check_ccds : bool
        If True, determine which CCDs Fermi crosses
    instrument : str
        Instrument name for CCD queries
    verbose : bool
        Print detailed CCD information
    n_trajectory_points : int
        Number of trajectory points to include around closest approach

    Returns:
    --------
    overlaps : list of dict
        Candidate overlaps with distance and timing info, including CCD matches
        and trajectory points
    """
    overlaps = []
    min_dist_seen = float('inf')

    for visit in rubin_visits:
        # Quick time check
        if not (fermi_pass.start_time <= visit['obs_time'] <= fermi_pass.end_time):
            continue

        # Geometric check: minimum distance to ANY point on Fermi's actual trajectory
        # Calculate angular separation to all Fermi positions
        visit_ra = visit['pointing_ra']
        visit_dec = visit['pointing_dec']

        # Calculate separation to each point in Fermi's trajectory
        # Handle RA wraparound
        ra_diffs = fermi_pass.ra - visit_ra
        ra_diffs = np.where(ra_diffs > 180, ra_diffs - 360, ra_diffs)
        ra_diffs = np.where(ra_diffs < -180, ra_diffs + 360, ra_diffs)

        dec_diffs = fermi_pass.dec - visit_dec

        # Angular separation using spherical geometry
        # Convert to radians
        ra_diffs_rad = np.radians(ra_diffs)
        dec_diffs_rad = np.radians(dec_diffs)
        visit_dec_rad = np.radians(visit_dec)
        fermi_dec_rad = np.radians(fermi_pass.dec)

        # Haversine formula for great circle distance
        a = np.sin(dec_diffs_rad / 2) ** 2 + \
            np.cos(visit_dec_rad) * np.cos(fermi_dec_rad) * np.sin(ra_diffs_rad / 2) ** 2
        separations = 2 * np.arcsin(np.sqrt(a))
        separations_deg = np.degrees(separations)

        dist = np.min(separations_deg)

        # Track minimum
        if dist < min_dist_seen:
            min_dist_seen = dist

        if dist < (fov_radius + buffer):
            # Get Fermi positions during this visit
            visit_start = visit['timespan'].begin if not hasattr(visit['timespan'].begin, 'astropy') else visit[
                'timespan'].begin.astropy
            visit_end = visit['timespan'].end if not hasattr(visit['timespan'].end, 'astropy') else visit[
                'timespan'].end.astropy

            # Find Fermi positions during exposure
            time_mask = (fermi_pass.times >= visit_start) & (fermi_pass.times <= visit_end)
            fermi_ra_during = fermi_pass.ra[time_mask]
            fermi_dec_during = fermi_pass.dec[time_mask]
            fermi_times_during = fermi_pass.times[time_mask]

            # Find index of closest approach
            closest_idx = np.argmin(separations_deg)

            # Extract trajectory points around closest approach
            # Get indices for points before and after closest approach
            half_points = n_trajectory_points // 2
            start_idx = max(0, closest_idx - half_points)
            end_idx = min(len(fermi_pass.ra), closest_idx + half_points + 1)

            trajectory_points = []
            for i in range(start_idx, end_idx):
                trajectory_points.append({
                    'ra': float(fermi_pass.ra[i]),
                    'dec': float(fermi_pass.dec[i]),
                    'time': fermi_pass.times[i].iso,
                    'time_mjd': float(fermi_pass.times[i].mjd),
                    'alt': float(fermi_pass.alt[i]) if hasattr(fermi_pass, 'alt') else None,
                    'az': float(fermi_pass.az[i]) if hasattr(fermi_pass, 'az') else None,
                    'is_closest': (i == closest_idx)
                })

            # Find CCD matches if requested
            ccd_matches = []
            if check_ccds and butler is not None and len(fermi_ra_during) > 0:
                ccd_matches = find_ccd_matches(
                    butler,
                    visit['visit_id'],
                    fermi_ra_during,
                    fermi_dec_during,
                    instrument=instrument,
                    verbose=verbose
                )

            overlaps.append({
                'visit_id': visit['visit_id'],
                'obs_time': visit['obs_time'],
                'pointing_ra': visit['pointing_ra'],
                'pointing_dec': visit['pointing_dec'],
                'distance_to_fermi_path': dist,
                'closest_approach_ra': float(fermi_pass.ra[closest_idx]),
                'closest_approach_dec': float(fermi_pass.dec[closest_idx]),
                'closest_approach_time': fermi_pass.times[closest_idx].iso,
                'within_fov': dist < fov_radius,
                'instrument': visit['instrument'],
                'ccd_matches': ccd_matches,
                'num_ccds': len(ccd_matches),
                'trajectory_points': trajectory_points
            })

    # DEBUG: print if we had visits but no matches
    if len(rubin_visits) > 0 and len(overlaps) == 0 and verbose:
        print(f"    Closest visit was {min_dist_seen:.2f}° from Fermi path (threshold: {fov_radius + buffer:.2f}°)")

    return overlaps

def search_all_passes(passes, butler, instrument='LSSTComCam',
                      filter_passes=True, fov_radius=1.75, check_ccds=True, verbose=True):
    """
    Search all Fermi passes for overlaps with Rubin observations

    Parameters:
    -----------
    passes : list
        List of FermiPass objects
    butler : lsst.daf.butler.Butler or None
        Butler instance for querying (None to skip)
    instrument : str
        Rubin instrument name
    filter_passes : bool
        If True, only search passes meeting all filter criteria
    fov_radius : float
        Field of view radius in degrees
    check_ccds : bool
        If True, determine which CCDs Fermi crosses
    verbose : bool
        Print progress

    Returns:
    --------
    all_overlaps : list of dict
        All candidate overlaps found
    """
    # Filter passes if requested
    if filter_passes:
        search_passes = [p for p in passes if p.passes_all_filters]
        if verbose:
            print(f"\nSearching {len(search_passes)} passes that meet all filter criteria")
    else:
        search_passes = passes
        if verbose:
            print(f"\nSearching all {len(search_passes)} passes")

    if len(search_passes) == 0:
        print("No passes to search!")
        return []

    all_overlaps = []

    for i, fermi_pass in enumerate(search_passes):
        if verbose:
            print(f"\nPass {i + 1}/{len(search_passes)}: {fermi_pass.start_time.iso} to {fermi_pass.end_time.iso}")
            print(f"  Duration: {(fermi_pass.end_time - fermi_pass.start_time).sec / 60:.1f} min")
            print(f"  RA: {fermi_pass.ra.min():.1f}° to {fermi_pass.ra.max():.1f}°")
            print(f"  Dec: {fermi_pass.dec.min():.1f}° to {fermi_pass.dec.max():.1f}°")

        # Query Rubin visits during this pass
        if butler is None:
            if verbose:
                print("  No Butler available - skipping Rubin query")
            continue

        rubin_visits = get_rubin_visits_butler(
            butler,
            fermi_pass.start_time,
            fermi_pass.end_time,
            instrument=instrument
        )

        if verbose:
            print(f"  Found {len(rubin_visits)} Rubin visits during this time")

        if len(rubin_visits) == 0:
            continue

        # Find overlaps
        overlaps = find_pass_overlaps(
            fermi_pass,
            rubin_visits,
            butler,
            fov_radius=fov_radius,
            check_ccds=check_ccds,
            instrument=instrument,
            verbose=verbose
        )

        if len(overlaps) > 0:
            if verbose:
                print(f"  *** Found {len(overlaps)} potential overlaps! ***")

            # Add pass information to overlaps
            for overlap in overlaps:
                overlap['fermi_pass_index'] = i
                overlap['fermi_start_time'] = fermi_pass.start_time
                overlap['fermi_end_time'] = fermi_pass.end_time
                overlap['fermi_ra_range'] = (fermi_pass.ra.min(), fermi_pass.ra.max())
                overlap['fermi_dec_range'] = (fermi_pass.dec.min(), fermi_pass.dec.max())
                overlap['fermi_max_velocity'] = fermi_pass.max_angular_velocity

            all_overlaps.extend(overlaps)

            if verbose:
                for overlap in overlaps:
                    status = "WITHIN FOV" if overlap['within_fov'] else "nearby"
                    ccd_info = f", {overlap['num_ccds']} CCDs" if check_ccds else ""
                    print(
                        f"    Visit {overlap['visit_id']}: {status}, dist={overlap['distance_to_fermi_path']:.2f}°{ccd_info}")

    return all_overlaps


def save_overlaps(overlaps, output_file):
    """Save overlaps to file"""
    import json

    output_file = Path(output_file)

    # Convert to JSON-serializable format
    output_data = []
    for overlap in overlaps:
        data = {
            'fermi_pass_index': overlap['fermi_pass_index'],
            'fermi_start_time': overlap['fermi_start_time'].iso,
            'fermi_end_time': overlap['fermi_end_time'].iso,
            'fermi_ra_range': overlap['fermi_ra_range'],
            'fermi_dec_range': overlap['fermi_dec_range'],
            'fermi_max_velocity_deg_per_s': overlap['fermi_max_velocity'],
            'rubin_visit_id': overlap['visit_id'],
            'rubin_obs_time': overlap['obs_time'].iso,
            'rubin_pointing_ra': overlap['pointing_ra'],
            'rubin_pointing_dec': overlap['pointing_dec'],
            'distance_to_fermi_path_deg': overlap['distance_to_fermi_path'],
            'within_fov': overlap['within_fov'],
            'instrument': overlap['instrument'],
            'num_ccds': overlap['num_ccds'],
            'ccd_matches': overlap['ccd_matches']
        }
        output_data.append(data)

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved {len(overlaps)} overlaps to {output_file}")


def query_and_cache_visits(butler, start_time, end_time, instrument, output_file):
    """
    Query Butler for visits in time range and cache to pickle file.

    Parameters:
    -----------
    butler : lsst.daf.butler.Butler
        Butler instance
    start_time : astropy.time.Time
        Start of time range
    end_time : astropy.time.Time
        End of time range
    instrument : str
        Instrument name
    output_file : str or Path
        Path to save pickle file

    Returns:
    --------
    visits : list of dict
        Visit information
    """
    from lsst.daf.butler import Timespan
    import lsst.geom as geom

    print(f"\n{'=' * 60}")
    print(f"QUERYING VISITS")
    print(f"{'=' * 60}")
    print(f"Time range: {start_time.iso} to {end_time.iso}")
    print(f"Instrument: {instrument}")
    print(f"Output: {output_file}")
    print()

    # Create timespan for query
    timespan = Timespan(begin=start_time, end=end_time)

    try:
        where_clause = f"instrument = '{instrument}' AND visit.timespan OVERLAPS timespan"
        bind_params = {"timespan": timespan}

        print("Executing Butler query...")
        visit_records = butler.query_dimension_records(
            "visit",
            where=where_clause,
            bind=bind_params
        )

        # Convert to list to get count
        visit_list = list(visit_records)
        print(f"Found {len(visit_list)} visits")

    except Exception as e:
        print(f"ERROR: Butler query failed: {e}")
        return []

    print("\nProcessing visit records...")
    visits = []
    for i, visit in enumerate(visit_list):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(visit_list)} visits...")

        try:
            # Extract pointing - handle both old and new region types
            if hasattr(visit.region, 'center'):
                # Old style with center attribute
                ra = visit.region.center.getRa().asDegrees()
                dec = visit.region.center.getDec().asDegrees()
            else:
                # New style - region is a ConvexPolygon, compute centroid
                # Get bounding circle center as approximation
                bounding_circle = visit.region.getBoundingCircle()
                center_vector = bounding_circle.getCenter()

                # Convert UnitVector3d to lon/lat
                # UnitVector3d has x, y, z components
                lon = np.arctan2(center_vector.y(), center_vector.x())
                lat = np.arcsin(center_vector.z())

                # Convert to degrees
                ra = np.degrees(lon)
                dec = np.degrees(lat)

                # Normalize RA to [0, 360)
                if ra < 0:
                    ra += 360.0

            # Get observation time (midpoint)
            # Handle both old (with .astropy) and new (already astropy Time) formats
            begin_time = visit.timespan.begin
            end_time = visit.timespan.end

            if hasattr(begin_time, 'astropy'):
                # Old format - convert to astropy
                begin_astropy = begin_time.astropy
                end_astropy = end_time.astropy
            else:
                # New format - already astropy Time objects
                begin_astropy = begin_time
                end_astropy = end_time

            mid_astropy = begin_astropy + (end_astropy - begin_astropy) / 2

            visits.append({
                'visit_id': visit.id,
                'obs_time': mid_astropy,
                'pointing_ra': ra,
                'pointing_dec': dec,
                'instrument': instrument,
                'timespan': visit.timespan
            })
        except Exception as e:
            print(f"  Warning: Could not process visit {visit.id}: {e}")
            continue

    print(f"\nSuccessfully processed {len(visits)} visits")

    # Save to pickle
    output_file = Path(output_file)
    with open(output_file, 'wb') as f:
        pickle.dump(visits, f)

    print(f"Saved visits to {output_file}")

    # Print summary statistics
    print(f"\n{'=' * 60}")
    print(f"VISIT SUMMARY")
    print(f"{'=' * 60}")
    if len(visits) > 0:
        ras = [v['pointing_ra'] for v in visits]
        decs = [v['pointing_dec'] for v in visits]
        times = [v['obs_time'].iso for v in visits]

        print(f"Total visits: {len(visits)}")
        print(f"RA range: {min(ras):.2f}° to {max(ras):.2f}°")
        print(f"Dec range: {min(decs):.2f}° to {max(decs):.2f}°")
        print(f"First observation: {min(times)}")
        print(f"Last observation: {max(times)}")

    return visits

def search_all_passes_with_cache(passes, cached_visits, instrument='LSSTComCam',
                                 filter_passes=True, fov_radius=1.75,
                                 check_ccds=False, butler=None, verbose=True,
                                 n_trajectory_points=10):
    """
    Search all Fermi passes for overlaps using pre-cached visits.

    Parameters:
    -----------
    passes : list
        List of FermiPass objects
    cached_visits : list of dict
        Pre-loaded visit information
    instrument : str
        Rubin instrument name (for filtering)
    filter_passes : bool
        If True, only search passes meeting all filter criteria
    fov_radius : float
        Field of view radius in degrees
    check_ccds : bool
        If True, determine which CCDs Fermi crosses (requires butler)
    butler : lsst.daf.butler.Butler or None
        Butler instance (only needed if check_ccds=True)
    verbose : bool
        Print progress

    Returns:
    --------
    all_overlaps : list of dict
        All candidate overlaps found
    """
    # Filter passes if requested
    if filter_passes:
        search_passes = [p for p in passes if p.passes_all_filters]
        if verbose:
            print(f"\nSearching {len(search_passes)} passes that meet all filter criteria")
    else:
        search_passes = passes
        if verbose:
            print(f"\nSearching all {len(search_passes)} passes")

    if len(search_passes) == 0:
        print("No passes to search!")
        return []

    # Filter cached visits by instrument
    instrument_visits = [v for v in cached_visits if v['instrument'] == instrument]
    if verbose:
        print(f"Using {len(instrument_visits)} cached {instrument} visits")

    if check_ccds and butler is None:
        print("WARNING: check_ccds=True but no Butler provided - skipping CCD matching")
        check_ccds = False

    all_overlaps = []

    for i, fermi_pass in enumerate(search_passes):
        if verbose:
            print(f"\nPass {i + 1}/{len(search_passes)}: {fermi_pass.start_time.iso} to {fermi_pass.end_time.iso}")
            print(f"  Duration: {(fermi_pass.end_time - fermi_pass.start_time).sec / 60:.1f} min")
            print(f"  RA: {fermi_pass.ra.min():.1f}° to {fermi_pass.ra.max():.1f}°")
            print(f"  Dec: {fermi_pass.dec.min():.1f}° to {fermi_pass.dec.max():.1f}°")

        # Filter cached visits to those during this pass
        pass_visits = [
            v for v in instrument_visits
            if fermi_pass.start_time <= v['obs_time'] <= fermi_pass.end_time
        ]

        if verbose:
            print(f"  Found {len(pass_visits)} visits during this time")

        if len(pass_visits) == 0:
            continue

        # Find overlaps
        overlaps = find_pass_overlaps(
            fermi_pass,
            pass_visits,
            butler,
            fov_radius=fov_radius,
            check_ccds=check_ccds,
            instrument=instrument,
            verbose=verbose,
            n_trajectory_points=n_trajectory_points
        )

        if len(overlaps) > 0:
            if verbose:
                print(f"  *** Found {len(overlaps)} potential overlaps! ***")

            # Add pass information to overlaps
            for overlap in overlaps:
                overlap['fermi_pass_index'] = i
                overlap['fermi_start_time'] = fermi_pass.start_time
                overlap['fermi_end_time'] = fermi_pass.end_time
                overlap['fermi_ra_range'] = (fermi_pass.ra.min(), fermi_pass.ra.max())
                overlap['fermi_dec_range'] = (fermi_pass.dec.min(), fermi_pass.dec.max())
                overlap['fermi_max_velocity'] = fermi_pass.max_angular_velocity

            all_overlaps.extend(overlaps)

            if verbose:
                for overlap in overlaps:
                    status = "WITHIN FOV" if overlap['within_fov'] else "nearby"
                    ccd_info = f", {overlap['num_ccds']} CCDs" if check_ccds else ""
                    print(
                        f"    Visit {overlap['visit_id']}: {status}, dist={overlap['distance_to_fermi_path']:.2f}°{ccd_info}")

    return all_overlaps


def save_overlaps(overlaps, output_file):
    """Save overlaps to YAML file with trajectory points"""
    import yaml

    output_file = Path(output_file)

    # Group overlaps by visit_id
    from collections import defaultdict
    visits_data = defaultdict(list)

    for overlap in overlaps:
        visit_id = overlap['visit_id']

        # Format trajectory points
        trajectory = []
        for point in overlap['trajectory_points']:
            # Extract just the time portion (HH:MM:SS)
            time_str = point['time'].split('T')[1].split('.')[0] if 'T' in point['time'] else point['time']

            trajectory.append({
                'ra': round(point['ra'], 1),
                'dec': round(point['dec'], 1),
                'time': time_str
            })

        overlap_data = {
            'fermi_pass_index': overlap['fermi_pass_index'],
            'fermi_start_time': overlap['fermi_start_time'].iso,
            'fermi_end_time': overlap['fermi_end_time'].iso,
            'rubin_obs_time': overlap['obs_time'].iso,
            'rubin_pointing_ra': round(overlap['pointing_ra'], 2),
            'rubin_pointing_dec': round(overlap['pointing_dec'], 2),
            'distance_to_fermi_path_deg': round(overlap['distance_to_fermi_path'], 2),
            'closest_approach_ra': round(overlap['closest_approach_ra'], 2),
            'closest_approach_dec': round(overlap['closest_approach_dec'], 2),
            'closest_approach_time': overlap['closest_approach_time'],
            'within_fov': overlap['within_fov'],
            'num_ccds': overlap['num_ccds'],
            'trajectory': trajectory
        }

        visits_data[visit_id].append(overlap_data)

    # Build output structure
    output_data = {'visits': []}

    for visit_id, overlap_list in sorted(visits_data.items()):
        visit_entry = {
            'visit_id': visit_id,
            'overlaps': overlap_list
        }
        output_data['visits'].append(visit_entry)

    # Write YAML with custom formatting
    with open(output_file, 'w') as f:
        # Write custom YAML with cleaner formatting for trajectory
        f.write("visits:\n\n")

        for visit in output_data['visits']:
            f.write(f"  - visit_id: {visit['visit_id']}\n")
            f.write(f"    overlaps:\n")

            for overlap in visit['overlaps']:
                f.write(f"      - fermi_pass_index: {overlap['fermi_pass_index']}\n")
                f.write(f"        fermi_start_time: {overlap['fermi_start_time']}\n")
                f.write(f"        fermi_end_time: {overlap['fermi_end_time']}\n")
                f.write(f"        rubin_obs_time: {overlap['rubin_obs_time']}\n")
                f.write(f"        rubin_pointing_ra: {overlap['rubin_pointing_ra']}\n")
                f.write(f"        rubin_pointing_dec: {overlap['rubin_pointing_dec']}\n")
                f.write(f"        distance_to_fermi_path_deg: {overlap['distance_to_fermi_path_deg']}\n")
                f.write(f"        closest_approach_ra: {overlap['closest_approach_ra']}\n")
                f.write(f"        closest_approach_dec: {overlap['closest_approach_dec']}\n")
                f.write(f"        closest_approach_time: {overlap['closest_approach_time']}\n")
                f.write(f"        within_fov: {overlap['within_fov']}\n")
                f.write(f"        num_ccds: {overlap['num_ccds']}\n")
                f.write(f"        trajectory:\n")

                for point in overlap['trajectory']:
                    # Inline format for trajectory points
                    f.write(f"          - {{ra: {point['ra']}, dec: {point['dec']}, time: \"{point['time']}\"}}\n")

                f.write("\n")

    print(f"\nSaved {len(overlaps)} overlaps to {output_file}")

def query_and_cache_visits(butler, start_time, end_time, instrument, output_file):
    """
    Query Butler for visits in time range and cache to pickle file.

    Parameters:
    -----------
    butler : lsst.daf.butler.Butler
        Butler instance
    start_time : astropy.time.Time
        Start of time range
    end_time : astropy.time.Time
        End of time range
    instrument : str
        Instrument name
    output_file : str or Path
        Path to save pickle file

    Returns:
    --------
    visits : list of dict
        Visit information
    """
    from lsst.daf.butler import Timespan
    import lsst.geom as geom

    print(f"\n{'=' * 60}")
    print(f"QUERYING VISITS")
    print(f"{'=' * 60}")
    print(f"Time range: {start_time.iso} to {end_time.iso}")
    print(f"Instrument: {instrument}")
    print(f"Output: {output_file}")
    print()

    # Create timespan for query
    timespan = Timespan(begin=start_time, end=end_time)

    try:
        where_clause = f"instrument = '{instrument}' AND visit.timespan OVERLAPS timespan"
        bind_params = {"timespan": timespan}

        print("Executing Butler query...")
        visit_records = butler.query_dimension_records(
            "visit",
            where=where_clause,
            bind=bind_params,
            limit=150000
        )

        # Convert to list to get count
        visit_list = list(visit_records)
        print(f"Found {len(visit_list)} visits")

    except Exception as e:
        print(f"ERROR: Butler query failed: {e}")
        return []

    print("\nProcessing visit records...")
    visits = []
    for i, visit in enumerate(visit_list):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(visit_list)} visits...")

        try:
            # Extract pointing - handle both old and new region types
            if hasattr(visit.region, 'center'):
                # Old style with center attribute
                ra = visit.region.center.getRa().asDegrees()
                dec = visit.region.center.getDec().asDegrees()
            else:
                # New style - region is a ConvexPolygon, compute centroid
                # Get bounding circle center as approximation
                bounding_circle = visit.region.getBoundingCircle()
                center_vector = bounding_circle.getCenter()

                # Convert UnitVector3d to lon/lat
                # UnitVector3d has x, y, z components
                lon = np.arctan2(center_vector.y(), center_vector.x())
                lat = np.arcsin(center_vector.z())

                # Convert to degrees
                ra = np.degrees(lon)
                dec = np.degrees(lat)

                # Normalize RA to [0, 360)
                if ra < 0:
                    ra += 360.0

            # Get observation time (midpoint)
            # Handle both old (with .astropy) and new (already astropy Time) formats
            begin_time = visit.timespan.begin
            end_time = visit.timespan.end

            if hasattr(begin_time, 'astropy'):
                # Old format - convert to astropy
                begin_astropy = begin_time.astropy
                end_astropy = end_time.astropy
            else:
                # New format - already astropy Time objects
                begin_astropy = begin_time
                end_astropy = end_time

            mid_astropy = begin_astropy + (end_astropy - begin_astropy) / 2

            visits.append({
                'visit_id': visit.id,
                'obs_time': mid_astropy,
                'pointing_ra': ra,
                'pointing_dec': dec,
                'instrument': instrument,
                'timespan': visit.timespan
            })
        except Exception as e:
            print(f"  Warning: Could not process visit {visit.id}: {e}")
            continue

    print(f"\nSuccessfully processed {len(visits)} visits")

    # Save to pickle
    output_file = Path(output_file)
    with open(output_file, 'wb') as f:
        pickle.dump(visits, f)

    print(f"Saved visits to {output_file}")

    # Print summary statistics
    print(f"\n{'=' * 60}")
    print(f"VISIT SUMMARY")
    print(f"{'=' * 60}")
    if len(visits) > 0:
        ras = [v['pointing_ra'] for v in visits]
        decs = [v['pointing_dec'] for v in visits]
        times = [v['obs_time'].iso for v in visits]

        print(f"Total visits: {len(visits)}")
        print(f"RA range: {min(ras):.2f}° to {max(ras):.2f}°")
        print(f"Dec range: {min(decs):.2f}° to {max(decs):.2f}°")
        print(f"First observation: {min(times)}")
        print(f"Last observation: {max(times)}")

    return visits


def load_cached_visits(cache_file):
    """
    Load previously cached visits from pickle file.

    Parameters:
    -----------
    cache_file : str or Path
        Path to pickle file

    Returns:
    --------
    visits : list of dict
        Visit information
    """
    cache_file = Path(cache_file)

    if not cache_file.exists():
        print(f"ERROR: Cache file not found: {cache_file}")
        return []

    print(f"Loading cached visits from {cache_file}")
    with open(cache_file, 'rb') as f:
        visits = pickle.load(f)

    print(f"Loaded {len(visits)} visits from cache")
    return visits

def print_visit_histogram(visits, label="cached visits"):
    """Print a month-by-month histogram of visits and the largest time gaps."""
    from collections import Counter

    if len(visits) == 0:
        print("No visits to histogram.")
        return

    times = Time([v['obs_time'] for v in visits])

    print(f"\n{'=' * 60}")
    print(f"VISIT TIME DISTRIBUTION ({label}: {len(visits)} visits)")
    print(f"{'=' * 60}")
    print(f"Span: {times.min().iso}  ->  {times.max().iso}\n")

    # Bucket by calendar month via ISO 'YYYY-MM' prefix
    months = [t.iso[:7] for t in times]
    counts = Counter(months)
    maxc = max(counts.values())

    print(f"{'Month':<10} {'Visits':>8}   Histogram")
    print("-" * 60)
    for month in sorted(counts):
        n = counts[month]
        bar = '#' * int(40 * n / maxc)
        print(f"{month:<10} {n:>8}   {bar}")
    print("-" * 60)
    print(f"{'TOTAL':<10} {len(visits):>8}")

    # Largest gaps = observing-run boundaries
    print("\nLargest gaps between consecutive visits (> 0.5 d):")
    mjd_sorted = np.sort(times.mjd)
    gaps = np.diff(mjd_sorted)
    for i in np.argsort(gaps)[::-1][:10]:
        if gaps[i] < 0.5:
            continue
        t0 = Time(mjd_sorted[i], format='mjd')
        t1 = Time(mjd_sorted[i + 1], format='mjd')
        print(f"  {gaps[i]:6.1f} d : {t0.iso[:16]}  ->  {t1.iso[:16]}")
    print(f"{'=' * 60}\n")

def main():
    parser = argparse.ArgumentParser(
        description='Find overlaps between Fermi passes and Rubin observations'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--query-only',
        action='store_true',
        help='Only query and cache visits, do not search for overlaps'
    )
    parser.add_argument(
        '--use-cache',
        type=str,
        help='Use cached visits from pickle file instead of querying Butler'
    )

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Extract parameters
    passes_file = config['passes_file']
    butler_repo = config.get('butler_repo', None)
    if butler_repo == 'None':
        butler_repo = None
    instrument = config.get('instrument', 'LSSTComCam')
    filter_passes = config.get('filter_passes_only', True)
    fov_radius = config.get('fov_radius', 1.75)
    fov_buffer = config.get('fov_buffer', 0.1)
    check_ccds = config.get('check_ccds', True)
    output_file = config.get('output_file', 'fermi_rubin_overlaps.json')
    verbose = config.get('verbose', True)

    # Time filtering parameters
    pass_start_time = config.get('pass_start_time', None)
    pass_end_time = config.get('pass_end_time', None)

    # Visit cache parameters
    visit_cache_file = config.get('visit_cache_file', 'rubin_visits_cache.pkl')
    visit_query_start = config.get('visit_query_start', pass_start_time)
    visit_query_end = config.get('visit_query_end', pass_end_time)

    # Initialize Butler if needed
    butler = None
    need_butler = args.query_only or (check_ccds and args.use_cache) or (not args.use_cache)

    if need_butler and butler_repo:
        if not HAVE_BUTLER:
            print("ERROR: Butler requested but lsst.daf.butler not available")
            print("Install Rubin Science Pipelines or set butler_repo to null in config")
            return 1

        if verbose:
            print(f"Initializing Butler with repo: {butler_repo}")

        try:
            butler = Butler(butler_repo)
        except Exception as e:
            print(f"ERROR: Could not initialize Butler: {e}")
            return 1

    # Handle query-only mode
    if args.query_only:
        if butler is None:
            print("ERROR: --query-only requires butler_repo in config")
            return 1

        if visit_query_start is None or visit_query_end is None:
            print("ERROR: --query-only requires visit_query_start and visit_query_end in config")
            return 1

        start_time = Time(visit_query_start, format='iso')
        end_time = Time(visit_query_end, format='iso')

        query_and_cache_visits(butler, start_time, end_time, instrument, visit_cache_file)
        print("\nQuery complete. Run without --query-only to search for overlaps.")
        return 0

    # Load Fermi passes with time filtering
    if verbose:
        print(f"Loading Fermi passes from {passes_file}")
    passes = load_fermi_passes(passes_file,
                               start_time=pass_start_time,
                               end_time=pass_end_time)

    if len(passes) == 0:
        print("No passes to search!")
        return 0

    # Search for overlaps
    if args.use_cache:
        # Use cached visits
        if verbose:
            print(f"\nUsing cached visits from {args.use_cache}")
        cached_visits = load_cached_visits(args.use_cache)

        if len(cached_visits) == 0:
            return 1

        print_visit_histogram(cached_visits, label=args.use_cache)

        overlaps = search_all_passes_with_cache(
            passes,
            cached_visits,
            instrument=instrument,
            filter_passes=filter_passes,
            fov_radius=fov_radius,
            check_ccds=check_ccds,
            butler=butler,  # Pass butler if available for CCD checking
            verbose=verbose
        )
    else:
        # Query Butler per-pass (original approach)
        if butler is None:
            print("ERROR: Need Butler connection when not using cache")
            return 1

        overlaps = search_all_passes(
            passes,
            butler,
            instrument=instrument,
            filter_passes=filter_passes,
            fov_radius=fov_radius,
            check_ccds=check_ccds,
            verbose=verbose
        )

    # Summary
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total Fermi passes searched: {len([p for p in passes if not filter_passes or p.passes_all_filters])}")
    print(f"Total potential overlaps found: {len(overlaps)}")

    if len(overlaps) > 0:
        within_fov = sum(1 for o in overlaps if o['within_fov'])
        print(f"  Within FOV (< {fov_radius}°): {within_fov}")
        print(f"  Nearby (< {fov_radius + fov_buffer}°): {len(overlaps) - within_fov}")

        if check_ccds:
            total_ccds = sum(o['num_ccds'] for o in overlaps)
            print(f"  Total CCD matches: {total_ccds}")

        # Group by Fermi pass
        from collections import defaultdict
        by_pass = defaultdict(int)
        for o in overlaps:
            by_pass[o['fermi_pass_index']] += 1

        print(f"\nOverlaps by Fermi pass:")
        for pass_idx in sorted(by_pass.keys()):
            print(f"  Pass {pass_idx + 1}: {by_pass[pass_idx]} Rubin visits")

    # Save results
    if len(overlaps) > 0 and output_file:
        save_overlaps(overlaps, output_file)
    elif len(overlaps) == 0:
        print("\nNo overlaps found - nothing to save")

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main())

