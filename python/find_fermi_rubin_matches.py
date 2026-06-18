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
    from lsst.daf.butler import Butler

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

    try:
        visit_records = butler.registry.queryDimensionRecords(
            'visit',
            where="visit.timespan OVERLAPS T(?,?) AND instrument = ?",
            bind=(start_tai, end_tai, instrument)
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
                       check_ccds=True, instrument='LSSTComCam', verbose=False):
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

    Returns:
    --------
    overlaps : list of dict
        Candidate overlaps with distance and timing info, including CCD matches
    """
    # Fermi trajectory endpoints
    ra_start = fermi_pass.ra[0]
    dec_start = fermi_pass.dec[0]
    ra_end = fermi_pass.ra[-1]
    dec_end = fermi_pass.dec[-1]

    overlaps = []

    for visit in rubin_visits:
        # Quick time check
        if not (fermi_pass.start_time <= visit['obs_time'] <= fermi_pass.end_time):
            continue

        # Geometric check: distance to Fermi's path
        dist = point_to_arc_distance(
            visit['pointing_ra'], visit['pointing_dec'],
            ra_start, dec_start,
            ra_end, dec_end
        )

        if dist < (fov_radius + buffer):
            # Get Fermi positions during this visit
            visit_start = Time(visit['timespan'].begin.tai, format='mjd', scale='tai')
            visit_end = Time(visit['timespan'].end.tai, format='mjd', scale='tai')

            # Find Fermi positions during exposure
            time_mask = (fermi_pass.times >= visit_start) & (fermi_pass.times <= visit_end)
            fermi_ra_during = fermi_pass.ra[time_mask]
            fermi_dec_during = fermi_pass.dec[time_mask]

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
                'within_fov': dist < fov_radius,
                'instrument': visit['instrument'],
                'ccd_matches': ccd_matches,
                'num_ccds': len(ccd_matches)
            })

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


def main():
    parser = argparse.ArgumentParser(
        description='Find overlaps between Fermi passes and Rubin observations'
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

    # Extract parameters
    passes_file = config['passes_file']
    butler_repo = config.get('butler_repo', None)
    instrument = config.get('instrument', 'LSSTComCam')
    filter_passes = config.get('filter_passes_only', True)
    fov_radius = config.get('fov_radius', 1.75)
    fov_buffer = config.get('fov_buffer', 0.1)
    check_ccds = config.get('check_ccds', True)
    output_file = config.get('output_file', 'fermi_rubin_overlaps.json')
    verbose = config.get('verbose', True)

    # Time filtering parameters
    pass_start_time = config.get('pass_start_time', None)  # e.g., "2025-12-20"
    pass_end_time = config.get('pass_end_time', None)  # e.g., "2026-01-10"

    # Load Fermi passes with time filtering
    if verbose:
        print(f"Loading Fermi passes from {passes_file}")
    passes = load_fermi_passes(passes_file,
                               start_time=pass_start_time,
                               end_time=pass_end_time)

    # Initialize Butler if available
    butler = None
    if butler_repo:
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
    else:
        print("WARNING: No butler_repo specified - will not query Rubin observations")
        print("Set butler_repo in config to enable overlap search")

    # Search for overlaps
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
