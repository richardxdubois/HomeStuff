"""
Stained Glass Pattern Analyzer
===============================
Analyzes stained glass pattern images to detect pieces, measure their
properties, and perform quality assurance checks.

Features:
    - Piece detection from black-line-on-white-background pattern images
    - Measurement of piece area, width, angles, and complexity
    - QA warnings for sharp angles, narrow pieces, tiny pieces, etc.
    - Gap detection in the lead line network
    - Suspiciously large piece detection (possible merged pieces)
    - Interactive Bokeh HTML visualization with hover/click/filter
    - Static annotated images and text reports

Usage:
    python pattern_analyzer.py pattern.jpg --panel-width 24 -o ./output

    For a pattern image where the actual panel is 24 inches wide.
    The tool auto-calculates DPI from the panel dimensions.

Dependencies:
    Required: opencv-python, numpy, scikit-image
    Optional: bokeh (>=3.8), pillow (for interactive visualization)

    pip install opencv-python numpy scikit-image bokeh pillow

Author: Collaborative development with Claude
Date: 2024
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Piece:
    """Represents a single piece of glass in the pattern.

    Attributes:
        id: Unique piece number (1-based)
        contour: OpenCV contour array defining the piece boundary
        area: Area in pixels
        area_sq_inches: Area in square inches (based on DPI)
        centroid: (x, y) center point of the piece
        bounding_box: (x, y, width, height) bounding rectangle
        num_vertices: Number of vertices in simplified polygon
        min_angle: Smallest interior angle in degrees
        min_width_inches: Narrowest dimension in inches
        max_width_inches: Widest dimension in inches
        warnings: List of QA warning strings
    """
    id: int
    contour: np.ndarray
    area: float
    area_sq_inches: float
    centroid: tuple
    bounding_box: tuple
    num_vertices: int
    min_angle: float
    min_width_inches: float
    max_width_inches: float
    warnings: list = field(default_factory=list)


# =============================================================================
# Main Analyzer Class
# =============================================================================

class PatternAnalyzer:
    """Analyzes a stained glass pattern image.

    The analyzer processes a pattern image (black lines on white background)
    to detect individual glass pieces, measure their properties, and flag
    potential quality issues.

    Typical workflow:
        1. Initialize with image path and scale info
        2. preprocess() - threshold and clean the image
        3. detect_pieces() - find closed regions
        4. detect_line_gaps() - find disconnected lines
        5. run_qa() - check for quality issues
        6. generate outputs (images, reports, interactive viz)

    Example:
        analyzer = PatternAnalyzer("pattern.jpg", panel_width=24)
        analyzer.preprocess()
        analyzer.detect_pieces()
        analyzer.detect_line_gaps()
        analyzer.run_qa()
        analyzer.generate_all(prefix="my_pattern")
    """

    def __init__(self, image_path, scale_dpi=150, output_dir=".",
                 panel_width=None, panel_height=None):
        """Initialize the analyzer with an image.

        Args:
            image_path: Path to the pattern image file (JPG, PNG, etc.)
            scale_dpi: Dots per inch for measurements (default 150).
                       Overridden if panel_width or panel_height is given.
            output_dir: Directory for output files (created if needed)
            panel_width: Actual panel width in inches (auto-calculates DPI)
            panel_height: Actual panel height in inches (auto-calculates DPI)

        Raises:
            FileNotFoundError: If the image file cannot be loaded
        """
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.pieces = []
        self.gaps = []
        self.binary = None
        self.image_path = image_path

        img_h, img_w = self.image.shape[:2]

        # Calculate DPI from panel dimensions if provided
        if panel_width:
            self.scale_dpi = img_w / panel_width
            print(f"Calculated DPI from panel width "
                  f"({panel_width}\"): {self.scale_dpi:.1f}")
        elif panel_height:
            self.scale_dpi = img_h / panel_height
            print(f"Calculated DPI from panel height "
                  f"({panel_height}\"): {self.scale_dpi:.1f}")
        else:
            self.scale_dpi = scale_dpi

        # Set up output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loaded image: {img_w}x{img_h} pixels")
        print(f"At {self.scale_dpi:.1f} DPI that's approximately "
              f"{img_w / self.scale_dpi:.1f}\" x "
              f"{img_h / self.scale_dpi:.1f}\"")
        print(f"Output directory: {self.output_dir.resolve()}")
        print(f"Pixel value range: {self.gray.min()} - {self.gray.max()}")
        print(f"Mean pixel value: {self.gray.mean():.1f}")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _output_path(self, filename):
        """Build a full output path from a filename.

        Args:
            filename: Output filename (e.g., "pattern_analyzed.png")

        Returns:
            Full path string within the output directory
        """
        return str(self.output_dir / filename)

    # =========================================================================
    # Image Preprocessing
    # =========================================================================

    def preprocess(self, line_threshold=128, close_kernel_size=3,
                   dilate_iterations=0):
        """Convert the pattern image to a clean binary image.

        Produces a binary image where:
            - Lead lines = BLACK (0)
            - Glass regions = WHITE (255)

        Steps:
            1. Threshold grayscale to separate lines from background
            2. Morphological closing to seal small gaps in lines
            3. Optional dilation to thicken lines

        Args:
            line_threshold: Pixel value threshold (0-255). Pixels darker
                           than this become lines. Default 128.
            close_kernel_size: Kernel size for morphological closing.
                              Seals small gaps from JPG compression.
                              Set to 0 to disable. Default 3.
            dilate_iterations: Number of dilation passes to thicken lines.
                              Helps seal larger gaps. Default 0.

        Returns:
            The binary image (also stored as self.binary)
        """
        # Analyze histogram for diagnostic info
        hist = cv2.calcHist([self.gray], [0], None, [256], [0, 256])
        dark_peak = np.argmax(hist[:128])
        light_peak = np.argmax(hist[128:]) + 128
        print(f"Dark peak at: {dark_peak}, Light peak at: {light_peak}")

        # Otsu's method for reference
        otsu_thresh, _ = cv2.threshold(
            self.gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        print(f"Otsu's suggested threshold: {otsu_thresh}")

        # Apply threshold: dark pixels -> black (lines), light -> white (glass)
        _, binary = cv2.threshold(
            self.gray, line_threshold, 255, cv2.THRESH_BINARY
        )
        cv2.imwrite(self._output_path("debug_00_threshold_only.png"), binary)
        print("Saved debug_00_threshold_only.png")

        # Morphological operations to clean up the line network
        if close_kernel_size > 0:
            # Work on inverted image so lines are white (foreground)
            lines = cv2.bitwise_not(binary)
            kernel = np.ones(
                (close_kernel_size, close_kernel_size), np.uint8
            )
            lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, kernel)

            if dilate_iterations > 0:
                lines = cv2.dilate(
                    lines, kernel, iterations=dilate_iterations
                )

            # Invert back: lines=black, glass=white
            binary = cv2.bitwise_not(lines)

        self.binary = binary

        # Save debug images
        cv2.imwrite(self._output_path("debug_01_binary.png"), binary)
        debug_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        debug_color[binary == 0] = [0, 0, 255]  # lines in red
        cv2.imwrite(
            self._output_path("debug_02_lines_highlighted.png"), debug_color
        )

        # Count regions for diagnostic
        num_labels, _ = cv2.connectedComponents(binary)
        print(f"Connected white regions found: {num_labels - 1}")
        print(f"Saved debug images to {self.output_dir}")

        return binary

    # =========================================================================
    # Piece Detection
    # =========================================================================

    def detect_pieces(self, min_area_pixels=200, max_area_ratio=0.5):
        """Find individual glass pieces in the preprocessed binary image.

        Each closed white region between lead lines is detected as a piece.
        Uses contour hierarchy to exclude container regions (e.g., the outer
        boundary that encloses all pieces).

        Args:
            min_area_pixels: Minimum area in pixels to count as a piece.
                            Smaller regions are treated as noise. Default 200.
            max_area_ratio: Maximum area as fraction of total image area.
                           Larger regions are treated as background. Default 0.5.

        Returns:
            List of Piece objects (also stored as self.pieces)
        """
        if self.binary is None:
            self.preprocess()

        # Use RETR_TREE for full hierarchy
        contours, hierarchy = cv2.findContours(
            self.binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        if hierarchy is None:
            print("No contours found!")
            return []

        hierarchy = hierarchy[0]  # reshape from (1, N, 4) to (N, 4)
        total_area = self.image.shape[0] * self.image.shape[1]
        max_area = total_area * max_area_ratio

        print(f"Found {len(contours)} raw contours")

        # Diagnostic: area distribution
        areas = [cv2.contourArea(c) for c in contours]
        areas_nonzero = [a for a in areas if a > 0]
        if areas_nonzero:
            print(f"Contour area range: {min(areas_nonzero):.0f} - "
                  f"{max(areas_nonzero):.0f} pixels")

        piece_id = 0
        rejected_small = 0
        rejected_large = 0
        rejected_parent = 0

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            if area < min_area_pixels:
                rejected_small += 1
                continue
            if area > max_area:
                rejected_large += 1
                continue

            # Hierarchy check: reject contours that contain significant
            # children (these are containers, not glass pieces)
            # hierarchy[i] = [next_sibling, prev_sibling, first_child, parent]
            if self._has_significant_children(
                    i, contours, hierarchy, area, min_area_pixels):
                rejected_parent += 1
                continue

            # Calculate piece properties
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            x, y, w, h = cv2.boundingRect(contour)
            epsilon = 0.015 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            min_angle = self._min_interior_angle(approx)
            min_width, max_width = self._piece_widths(contour)
            area_sq_inches = area / (self.scale_dpi ** 2)

            piece_id += 1
            piece = Piece(
                id=piece_id,
                contour=contour,
                area=area,
                area_sq_inches=area_sq_inches,
                centroid=(cx, cy),
                bounding_box=(x, y, w, h),
                num_vertices=len(approx),
                min_angle=min_angle,
                min_width_inches=min_width,
                max_width_inches=max_width,
            )
            self.pieces.append(piece)

        print(f"\nDetected {len(self.pieces)} pieces")
        print(f"Rejected: {rejected_small} too small, "
              f"{rejected_large} too large, "
              f"{rejected_parent} container contours")

        return self.pieces

    def _has_significant_children(self, contour_idx, contours, hierarchy,
                                  parent_area, min_area):
        """Check if a contour contains child contours of significant size.

        A contour with large children is likely a "container" that encloses
        other pieces, not a glass piece itself.

        Args:
            contour_idx: Index of the contour to check
            contours: List of all contours
            hierarchy: Contour hierarchy array
            parent_area: Area of the parent contour
            min_area: Minimum area to consider significant

        Returns:
            True if the contour has significant children
        """
        first_child = hierarchy[contour_idx][2]
        if first_child < 0:
            return False

        child_idx = first_child
        while child_idx >= 0:
            child_area = cv2.contourArea(contours[child_idx])
            if child_area > parent_area * 0.05 and child_area > min_area:
                return True
            child_idx = hierarchy[child_idx][0]  # next sibling
        return False

    # =========================================================================
    # Geometry Measurements
    # =========================================================================

    def _min_interior_angle(self, approx_contour):
        """Find the sharpest interior angle in a polygon.

        Sharp interior angles (< 35°) are difficult or impossible to
        score and break cleanly in glass cutting.

        Args:
            approx_contour: Simplified polygon contour from approxPolyDP

        Returns:
            Minimum interior angle in degrees (0-180)
        """
        points = approx_contour.reshape(-1, 2)
        n = len(points)
        if n < 3:
            return 180.0

        min_angle = 180.0
        for i in range(n):
            p1 = points[(i - 1) % n].astype(float)
            p2 = points[i].astype(float)
            p3 = points[(i + 1) % n].astype(float)

            v1 = p1 - p2
            v2 = p3 - p2

            norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
            if norm_product < 1e-6:
                continue

            cos_angle = np.dot(v1, v2) / norm_product
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            min_angle = min(min_angle, angle)

        return min_angle

    def _piece_widths(self, contour):
        """Estimate the minimum and maximum width of a piece.

        Uses two strategies:
            1. Minimum area rotated rectangle (resolution-independent)
            2. Distance transform with skeleton analysis (for interior
               narrow passages)

        Args:
            contour: OpenCV contour defining the piece boundary

        Returns:
            Tuple of (min_width_inches, max_width_inches)
        """
        mask = np.zeros(self.gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # Strategy 1: Rotated rectangle dimensions
        if len(contour) >= 5:
            rect = cv2.minAreaRect(contour)
            rect_w, rect_h = rect[1]
            min_dim = min(rect_w, rect_h) / self.scale_dpi
            max_dim = max(rect_w, rect_h) / self.scale_dpi
        else:
            x, y, w, h = cv2.boundingRect(contour)
            min_dim = min(w, h) / self.scale_dpi
            max_dim = max(w, h) / self.scale_dpi

        # Strategy 2: Distance transform for interior analysis
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        if dist.max() > 0:
            max_width_dt = dist.max() * 2 / self.scale_dpi

            interior_pixels = cv2.countNonZero(mask)
            if interior_pixels > 100:
                skeleton = self._skeletonize(mask)
                skeleton_distances = dist[skeleton > 0]
                if len(skeleton_distances) > 2:
                    min_width_dt = (
                            np.percentile(skeleton_distances, 10) * 2
                            / self.scale_dpi
                    )
                else:
                    min_width_dt = min_dim
            else:
                min_width_dt = min_dim

            # Use the more conservative (larger) estimate for min width
            min_width = max(min_width_dt, min_dim * 0.8)
            max_width = max(max_width_dt, max_dim)
        else:
            min_width = min_dim
            max_width = max_dim

        return min_width, max_width

    def _skeletonize(self, binary_image):
        """Reduce a binary region to its single-pixel-wide skeleton.

        Uses iterative erosion and subtraction to find the medial axis.

        Args:
            binary_image: Binary image (white foreground on black)

        Returns:
            Skeleton image (single-pixel-wide white lines)
        """
        skeleton = np.zeros_like(binary_image)
        temp = binary_image.copy()
        kernel = np.ones((3, 3), np.uint8)
        while True:
            eroded = cv2.erode(temp, kernel)
            dilated = cv2.dilate(eroded, kernel)
            diff = cv2.subtract(temp, dilated)
            skeleton = cv2.bitwise_or(skeleton, diff)
            temp = eroded.copy()
            if cv2.countNonZero(temp) == 0:
                break
        return skeleton

    # =========================================================================
    # Line Gap Detection
    # =========================================================================

    def detect_line_gaps(self, max_gap_pixels=20, min_gap_pixels=2,
                         suspicious_ratio=2.5):
        """Detect gaps in lead lines using morphological erosion.

        Strategy: progressively erode white regions (thicken black lines).
        When a previously single piece splits into two, a gap has been
        closed by the thickened lines, revealing where the original
        gap was.

        For each split, finds the closest points between the two new
        pieces to pinpoint the gap location.

        Args:
            max_gap_pixels: Maximum gap width in pixels to detect.
                           Controls how far erosion proceeds. Default 20.
            min_gap_pixels: Minimum gap width in pixels. Default 2.
            suspicious_ratio: Multiplier for flagging suspiciously large
                            pieces (stored for QA use). Default 2.5.
        """
        self.suspicious_ratio = suspicious_ratio
        binary = self.binary

        from skimage.measure import label

        original_labels = label(binary, connectivity=1)
        original_count = original_labels.max()
        print(f"Original piece count: {original_count}")

        gaps = []
        prev_labels = original_labels.copy()
        prev_count = original_count

        # ERODE white regions (= thicken black lines)
        # This CLOSES gaps, splitting merged pieces
        for r in range(1, max_gap_pixels // 2 + 1):
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)
            )
            # Erode white regions = dilate black lines directly on binary
            eroded = cv2.erode(binary, kernel, iterations=1)

            new_labels = label(eroded, connectivity=1)
            new_count = new_labels.max()

            if new_count > prev_count:
                pieces_split = new_count - prev_count
                print(f"  Erosion r={r} ({2 * r}px): "
                      f"{prev_count} → {new_count} pieces "
                      f"({pieces_split} split)")

                # Find which previous piece split
                for prev_id in range(1, prev_count + 1):
                    prev_mask = (prev_labels == prev_id)
                    # What new labels exist in this previous region?
                    overlapping_new = np.unique(new_labels[prev_mask])
                    overlapping_new = overlapping_new[overlapping_new > 0]

                    if len(overlapping_new) < 2:
                        continue

                    print(f"    Piece {prev_id} split into "
                          f"{overlapping_new.tolist()}")

                    # Find closest points between the split parts
                    for i in range(len(overlapping_new)):
                        for j in range(i + 1, len(overlapping_new)):
                            id_a = overlapping_new[i]
                            id_b = overlapping_new[j]

                            mask_a = (new_labels == id_a).astype(np.uint8)
                            mask_b = (new_labels == id_b).astype(np.uint8)

                            contours_a, _ = cv2.findContours(
                                mask_a, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE
                            )
                            contours_b, _ = cv2.findContours(
                                mask_b, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE
                            )

                            if not contours_a or not contours_b:
                                continue

                            pts_a = contours_a[0].reshape(-1, 2)
                            pts_b = contours_b[0].reshape(-1, 2)

                            if len(pts_a) > 500:
                                pts_a = pts_a[::len(pts_a) // 500]
                            if len(pts_b) > 500:
                                pts_b = pts_b[::len(pts_b) // 500]

                            min_dist = float('inf')
                            best_a, best_b = None, None

                            for pa in pts_a:
                                dists = np.sqrt(
                                    (pts_b[:, 0] - pa[0]) ** 2 +
                                    (pts_b[:, 1] - pa[1]) ** 2
                                )
                                idx = np.argmin(dists)
                                if dists[idx] < min_dist:
                                    min_dist = dists[idx]
                                    best_a = tuple(pa)
                                    best_b = tuple(pts_b[idx])

                            if best_a is not None:
                                gap_width = max(
                                    int(min_dist), min_gap_pixels
                                )

                                print(f"      GAP between {id_a}&{id_b}: "
                                      f"{best_a} - {best_b} "
                                      f"dist={min_dist:.1f}px "
                                      f"({min_dist / self.scale_dpi:.2f}\")")

                                gaps.append({
                                    'from': best_a,
                                    'to': best_b,
                                    'center': (
                                        (best_a[0] + best_b[0]) // 2,
                                        (best_a[1] + best_b[1]) // 2
                                    ),
                                    'distance_px': gap_width,
                                    'distance_in': (
                                        gap_width / self.scale_dpi
                                    ),
                                })

            prev_labels = new_labels
            prev_count = new_count
            if new_count <= 1:
                break

        clustered = self._cluster_gaps_morphological(gaps)
        print(f"Detected {len(clustered)} line gap(s)")
        self.gaps = clustered

    def _cluster_gaps_morphological(self, gaps, radius=20):
        """Cluster gap detections by center proximity, keep smallest.

        Multiple erosion levels may detect the same physical gap.
        This groups detections within 'radius' pixels and keeps
        the smallest (most accurate) measurement.

        Args:
            gaps: List of gap dicts with 'center' and 'distance_px'
            radius: Maximum pixel distance to consider same gap

        Returns:
            List of deduplicated gap dicts, sorted by distance
        """
        if not gaps:
            return []

        # Sort smallest first
        sorted_gaps = sorted(gaps, key=lambda g: g['distance_px'])

        clusters = []
        for g in sorted_gaps:
            cx, cy = g.get('center', g['from'])

            merged = False
            for cluster in clusters:
                ccx, ccy = cluster.get('center', cluster['from'])
                if abs(cx - ccx) <= radius and abs(cy - ccy) <= radius:
                    merged = True
                    break

            if not merged:
                clusters.append(g)

        clusters.sort(key=lambda g: g['distance_px'])
        return clusters

    def _save_gap_debug_image(self):
        """Save debug image showing detected gap locations.

        Draws red circles at gap endpoints and red lines connecting
        them, with distance labels.
        """
        ep_image = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)

        for gap in self.gaps:
            pt1 = (int(gap['from'][0]), int(gap['from'][1]))
            pt2 = (int(gap['to'][0]), int(gap['to'][1]))
            cv2.circle(ep_image, pt1, 10, (0, 0, 255), 3)
            cv2.circle(ep_image, pt2, 10, (0, 255, 0), 3)
            cv2.line(ep_image, pt1, pt2, (0, 0, 255), 2)
            mid_x = (pt1[0] + pt2[0]) // 2
            mid_y = (pt1[1] + pt2[1]) // 2
            label = f"{gap['distance_in']:.2f}\""
            cv2.putText(ep_image, label, (mid_x + 10, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imwrite(self._output_path("debug_04_gaps_found.png"), ep_image)

    # =========================================================================
    # Quality Assurance
    # =========================================================================

    def run_qa(self):
        """Run all quality assurance checks on detected pieces.

        Checks performed:
            - TINY: Piece area < 0.25 sq in
            - SMALL: Piece area < 0.5 sq in
            - SUSPICIOUSLY LARGE: Area > Nx median (possible merged pieces)
            - VERY NARROW: Min width < 3/16"
            - NARROW: Min width < 1/4"
            - VERY SHARP ANGLE: Min angle < 20°
            - SHARP ANGLE: Min angle < 35°
            - COMPLEX: More than 12 vertices
            - VERY ELONGATED: Aspect ratio > 6:1
            - ADJACENT TO GAP: Near a detected line gap

        Results are stored as warning strings in each Piece's warnings list.
        """
        print("\nRunning QA checks...")

        # Calculate median for suspicious size detection
        median_area = None
        if len(self.pieces) > 3:
            median_area = np.median(
                [p.area_sq_inches for p in self.pieces]
            )

        for piece in self.pieces:
            self._check_piece_size(piece, median_area)
            self._check_piece_width(piece)
            self._check_piece_angles(piece)
            self._check_piece_complexity(piece)
            self._check_piece_elongation(piece)

        # Check adjacency to gaps
        self._check_gap_adjacency()

        # Report summary
        warning_count = sum(len(p.warnings) for p in self.pieces)
        pieces_with_warnings = sum(
            1 for p in self.pieces if p.warnings
        )
        print(f"Found {warning_count} warnings "
              f"across {pieces_with_warnings} pieces")

    def _check_piece_size(self, piece, median_area):
        """Check if a piece is too small or suspiciously large."""
        if piece.area_sq_inches < 0.25:
            piece.warnings.append(
                f"TINY: Only {piece.area_sq_inches:.2f} sq in "
                f"— very difficult to cut"
            )
        elif piece.area_sq_inches < 0.5:
            piece.warnings.append(
                f"SMALL: {piece.area_sq_inches:.2f} sq in "
                f"— challenging to cut"
            )

        if median_area and (
                piece.area_sq_inches / median_area > self.suspicious_ratio):
            piece.warnings.append(
                f"SUSPICIOUSLY LARGE: {piece.area_sq_inches:.1f} sq in "
                f"is {piece.area_sq_inches / median_area:.0f}x the median "
                f"— possible merged pieces from line gap"
            )

    def _check_piece_width(self, piece):
        """Check if a piece has dangerously narrow sections."""
        if piece.min_width_inches < 0.1875:  # 3/16"
            piece.warnings.append(
                f"VERY NARROW: Min width ~{piece.min_width_inches:.3f}\" "
                f"— will likely break"
            )
        elif piece.min_width_inches < 0.25:  # 1/4"
            piece.warnings.append(
                f"NARROW: Min width ~{piece.min_width_inches:.3f}\" "
                f"— fragile"
            )

    def _check_piece_angles(self, piece):
        """Check if a piece has sharp interior angles."""
        if piece.min_angle < 20:
            piece.warnings.append(
                f"VERY SHARP ANGLE: {piece.min_angle:.0f}° "
                f"— nearly impossible to cut"
            )
        elif piece.min_angle < 35:
            piece.warnings.append(
                f"SHARP ANGLE: {piece.min_angle:.0f}° "
                f"— difficult, may need grinding"
            )

    def _check_piece_complexity(self, piece):
        """Check if a piece has too many vertices."""
        if piece.num_vertices > 12:
            piece.warnings.append(
                f"COMPLEX: {piece.num_vertices} vertices "
                f"— consider simplifying"
            )

    def _check_piece_elongation(self, piece):
        """Check if a piece is dangerously elongated."""
        x, y, w, h = piece.bounding_box
        if w > 0 and h > 0:
            aspect = max(w, h) / min(w, h)
            if aspect > 6:
                piece.warnings.append(
                    f"VERY ELONGATED: {aspect:.1f}:1 aspect ratio "
                    f"— fragile"
                )

    def _check_gap_adjacency(self):
        """Flag pieces that are adjacent to detected line gaps."""
        for gap in self.gaps:
            for piece in self.pieces:
                for pt in [gap['from'], gap['to']]:
                    dist = cv2.pointPolygonTest(
                        piece.contour,
                        (float(pt[0]), float(pt[1])),
                        True
                    )
                    if abs(dist) < 10:
                        msg = (
                            f"ADJACENT TO GAP: Line gap of "
                            f"{gap['distance_in']:.2f}\" at "
                            f"({pt[0]}, {pt[1]}) — "
                            f"lead lines may not connect properly"
                        )
                        if msg not in piece.warnings:
                            piece.warnings.append(msg)

    # =========================================================================
    # Output Generation — Static Images
    # =========================================================================

    def generate_annotated_image(self, prefix="pattern"):
        """Create an annotated image with piece numbers and QA highlights.

        - Clean pieces: numbered only
        - Warning pieces: orange highlight + number
        - Critical pieces: red highlight + number
        - Gaps: red circles and lines

        Args:
            prefix: Output filename prefix

        Returns:
            Path to the saved image
        """
        output_path = self._output_path(f"{prefix}_analyzed.png")
        annotated = self.image.copy()

        WARNING_COLORS = {
            'TINY': (0, 0, 255),
            'SMALL': (0, 128, 255),
            'VERY NARROW': (0, 0, 255),
            'NARROW': (0, 128, 255),
            'VERY SHARP': (0, 0, 255),
            'SHARP': (0, 128, 255),
            'COMPLEX': (255, 0, 255),
            'VERY ELONGATED': (255, 0, 0),
            'SUSPICIOUSLY': (0, 165, 255),
            'ADJACENT TO GAP': (0, 0, 255),
        }

        for piece in self.pieces:
            cx, cy = piece.centroid

            # Highlight pieces with warnings
            if piece.warnings:
                color = (0, 128, 255)  # default orange
                for key, c in WARNING_COLORS.items():
                    if any(key in w for w in piece.warnings):
                        if c == (0, 0, 255):
                            color = c
                            break
                        color = c

                overlay = annotated.copy()
                cv2.drawContours(
                    overlay, [piece.contour], -1, color, -1
                )
                cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0, annotated)
                cv2.drawContours(
                    annotated, [piece.contour], -1, color, 2
                )

            # Draw piece number label
            font_scale = max(
                0.3, min(0.7, np.sqrt(piece.area_sq_inches) * 0.3)
            )
            text = str(piece.id)
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            cv2.rectangle(
                annotated,
                (cx - tw // 2 - 2, cy - th // 2 - 2),
                (cx + tw // 2 + 2, cy + th // 2 + 2),
                (255, 255, 255), -1
            )
            cv2.putText(
                annotated, text,
                (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1
            )

        # Draw gap markers
        for gap in self.gaps:
            pt1 = (int(gap['from'][0]), int(gap['from'][1]))
            pt2 = (int(gap['to'][0]), int(gap['to'][1]))
            cv2.circle(annotated, pt1, 8, (0, 0, 255), 2)
            cv2.circle(annotated, pt2, 8, (0, 0, 255), 2)
            cv2.line(
                annotated, pt1, pt2,
                (0, 0, 255), 2, lineType=cv2.LINE_AA
            )

        cv2.imwrite(output_path, annotated)
        print(f"Saved annotated image: {output_path}")
        return output_path

    def generate_qa_overlay(self, prefix="pattern"):
        """Create an image showing only QA issues on a faded background.

        Args:
            prefix: Output filename prefix
        """
        output_path = self._output_path(f"{prefix}_qa.png")
        qa_image = cv2.addWeighted(
            self.image, 0.4,
            np.ones_like(self.image) * 255, 0.6, 0
        )

        for piece in self.pieces:
            if not piece.warnings:
                continue
            cx, cy = piece.centroid
            mask = np.zeros(self.gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [piece.contour], -1, 255, -1)
            qa_image[mask > 0] = (
                    qa_image[mask > 0] * 0.5 +
                    np.array([0, 0, 200]) * 0.5
            ).astype(np.uint8)

            short_warning = piece.warnings[0].split(":")[0]
            cv2.putText(
                qa_image, f"#{piece.id}", (cx - 10, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1
            )
            cv2.putText(
                qa_image, short_warning, (cx - 30, cy + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 150), 1
            )

        # Draw gaps prominently
        for gap in self.gaps:
            pt1 = (int(gap['from'][0]), int(gap['from'][1]))
            pt2 = (int(gap['to'][0]), int(gap['to'][1]))
            cv2.circle(qa_image, pt1, 12, (0, 0, 255), 3)
            cv2.circle(qa_image, pt2, 12, (0, 0, 255), 3)
            cv2.line(
                qa_image, pt1, pt2,
                (0, 0, 255), 3, lineType=cv2.LINE_AA
            )

        cv2.imwrite(output_path, qa_image)
        print(f"Saved QA overlay: {output_path}")

    # =========================================================================
    # Output Generation — Text Report
    # =========================================================================

    def generate_report(self, prefix="pattern"):
        """Generate a detailed text report of the analysis.

        Includes:
            - Image and panel dimensions
            - Line gap detection results
            - Piece size distribution statistics
            - Full piece table with measurements
            - QA warning summary and details

        Args:
            prefix: Output filename prefix

        Returns:
            Report text as string
        """
        output_path = self._output_path(f"{prefix}_report.txt")
        img_h, img_w = self.image.shape[:2]

        lines = []
        lines.append("=" * 76)
        lines.append("STAINED GLASS PATTERN ANALYSIS REPORT")
        lines.append("=" * 76)
        lines.append("")
        lines.append(f"Image size: {img_w}x{img_h} pixels")
        lines.append(f"Scale: {self.scale_dpi:.1f} DPI")
        lines.append(
            f"Estimated panel size: "
            f"{img_w / self.scale_dpi:.1f}\" x "
            f"{img_h / self.scale_dpi:.1f}\""
        )
        lines.append(f"Total pieces detected: {len(self.pieces)}")
        lines.append("")

        # Line gaps section
        lines.append("-" * 40)
        lines.append("LINE GAP DETECTION")
        lines.append("-" * 40)

        if self.gaps:
            lines.append(f"  *** FOUND {len(self.gaps)} LINE GAP(S) ***")
            lines.append("")
            for i, gap in enumerate(self.gaps, 1):
                lines.append(f"  Gap {i}:")
                lines.append(
                    f"    From: ({gap['from'][0]}, {gap['from'][1]})"
                )
                lines.append(
                    f"    To:   ({gap['to'][0]}, {gap['to'][1]})"
                )
                lines.append(
                    f"    Distance: {gap['distance_px']:.0f}px "
                    f"({gap['distance_in']:.2f}\")"
                )
                lines.append(
                    f"    ⚠ Lead lines do not connect — "
                    f"pieces may be incorrectly merged"
                )
                lines.append("")
        else:
            lines.append(
                "  ✓ No line gaps detected — all lines appear connected"
            )

        lines.append("")

        # Size distribution
        lines.append("-" * 40)
        lines.append("PIECE SIZE DISTRIBUTION")
        lines.append("-" * 40)
        areas = [p.area_sq_inches for p in self.pieces]
        if areas:
            lines.append(f"  Smallest: {min(areas):.2f} sq in")
            lines.append(f"  Largest:  {max(areas):.2f} sq in")
            lines.append(f"  Average:  {np.mean(areas):.2f} sq in")
            lines.append(f"  Median:   {np.median(areas):.2f} sq in")
            lines.append(f"  Total glass area: {sum(areas):.1f} sq in")
        lines.append("")

        # Piece table
        lines.append("-" * 76)
        lines.append("ALL PIECES")
        lines.append("-" * 76)
        lines.append(
            f"{'ID':>4} {'Area(sq in)':>11} {'MinW':>8} {'MaxW':>8} "
            f"{'Min Angle':>10} {'Vertices':>8} {'Warnings':>8}"
        )
        lines.append("-" * 76)

        for p in sorted(self.pieces, key=lambda x: x.id):
            warn_flag = "***" if p.warnings else ""
            lines.append(
                f"{p.id:>4} {p.area_sq_inches:>11.2f} "
                f"{p.min_width_inches:>7.3f}\" "
                f"{p.max_width_inches:>7.3f}\" "
                f"{p.min_angle:>9.0f}° "
                f"{p.num_vertices:>8} {warn_flag:>8}"
            )

        # Warnings summary
        lines.append("")
        lines.append("-" * 40)
        lines.append("QA WARNINGS")
        lines.append("-" * 40)

        warning_pieces = [p for p in self.pieces if p.warnings]
        if warning_pieces:
            # Group by type
            warning_type_counts = defaultdict(int)
            for p in self.pieces:
                for w in p.warnings:
                    wtype = w.split(":")[0]
                    warning_type_counts[wtype] += 1

            lines.append("\n  Warning Summary:")
            for wtype, count in sorted(
                    warning_type_counts.items(), key=lambda x: -x[1]):
                lines.append(f"    {wtype}: {count} pieces")

            lines.append(f"\n  Details:")
            for p in sorted(warning_pieces, key=lambda x: x.id):
                for w in p.warnings:
                    lines.append(f"    Piece {p.id:>3}: {w}")
        else:
            lines.append("  ✓ No warnings — pattern looks good!")

        lines.append("")
        lines.append("=" * 76)
        lines.append("END OF REPORT")
        lines.append("=" * 76)

        report_text = "\n".join(lines)
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"\nSaved report: {output_path}")
        print("\n" + report_text)

        return report_text

    # =========================================================================
    # Output Generation — Interactive Bokeh Visualization
    # =========================================================================

    def generate_bokeh_visualization(self, prefix="pattern"):
        """Generate an interactive HTML visualization using Bokeh.

        Features:
            - Pan/zoom on the pattern image
            - Hover over any piece for measurements and warnings
            - Click a piece for detailed info panel
            - Filter by QA status (OK / Warning / Critical)
            - Warning breakdown bar chart
            - Piece size distribution histogram
            - Gap locations shown as red dashed lines

        Requires: bokeh>=3.8

        Args:
            prefix: Output filename prefix

        Returns:
            Path to the saved HTML file
        """
        from bokeh.plotting import figure, save, output_file
        from bokeh.models import (
            ColumnDataSource, HoverTool,
            CustomJS, Div, CheckboxGroup,
        )
        from bokeh.layouts import column, row

        output_path = self._output_path(f"{prefix}_interactive.html")
        output_file(output_path, title="Stained Glass Pattern Analysis")

        img_height, img_width = self.image.shape[:2]

        # --- Prepare background image as RGBA uint32 ---
        bg_image = self._prepare_bokeh_background(img_height, img_width)

        # --- Prepare piece data ---
        source, source_full, piece_data = self._prepare_bokeh_piece_data(
            img_height
        )

        # --- Build figure ---
        p = self._build_bokeh_figure(
            bg_image, source, img_width, img_height
        )

        # --- Build side panels ---
        right_panel = self._build_bokeh_panels(
            source, source_full, piece_data, img_width, img_height
        )

        # --- Add gap visualization ---
        if self.gaps:
            self._add_bokeh_gaps(p, img_height)

        layout = row(p, right_panel)
        save(layout)
        print(f"Saved interactive visualization: {output_path}")
        return output_path

    def _prepare_bokeh_background(self, img_height, img_width):
        """Encode the pattern image as a uint32 RGBA array for Bokeh.

        Args:
            img_height: Image height in pixels
            img_width: Image width in pixels

        Returns:
            uint32 array suitable for Bokeh's image_rgba
        """
        img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        img_rgba = np.zeros((img_height, img_width, 4), dtype=np.uint8)
        img_rgba[:, :, :3] = img_rgb
        img_rgba[:, :, 3] = 255
        img_rgba = np.flipud(img_rgba)
        img_uint32 = np.zeros((img_height, img_width), dtype=np.uint32)
        view = img_uint32.view(dtype=np.uint8).reshape(
            (img_height, img_width, 4)
        )
        view[:, :, 0] = img_rgba[:, :, 0]
        view[:, :, 1] = img_rgba[:, :, 1]
        view[:, :, 2] = img_rgba[:, :, 2]
        view[:, :, 3] = img_rgba[:, :, 3]
        return img_uint32

    def _prepare_bokeh_piece_data(self, img_height):
        """Build ColumnDataSource with all piece data for Bokeh.

        Args:
            img_height: Image height (for y-axis flip)

        Returns:
            Tuple of (source, source_full, piece_data_dict)
        """
        from bokeh.models import ColumnDataSource

        data = defaultdict(list)

        for piece in self.pieces:
            # Simplify contour for performance
            epsilon = 0.01 * cv2.arcLength(piece.contour, True)
            simplified = cv2.approxPolyDP(piece.contour, epsilon, True)
            points = simplified.reshape(-1, 2)

            # Flip y for Bokeh coordinate system and close polygon
            xs = points[:, 0].tolist()
            ys = (img_height - points[:, 1]).tolist()
            xs.append(xs[0])
            ys.append(ys[0])

            data['xs'].append(xs)
            data['ys'].append(ys)
            data['piece_id'].append(piece.id)
            data['area'].append(round(piece.area_sq_inches, 3))
            data['min_width'].append(round(piece.min_width_inches, 3))
            data['max_width'].append(round(piece.max_width_inches, 3))
            data['min_angle'].append(round(piece.min_angle, 1))
            data['vertices'].append(piece.num_vertices)
            data['cx'].append(piece.centroid[0])
            data['cy'].append(img_height - piece.centroid[1])

            warn_text = ("; ".join(piece.warnings)
                         if piece.warnings else "None")
            data['warnings'].append(warn_text)
            data['warning_count'].append(len(piece.warnings))

            # Categorize QA status
            if not piece.warnings:
                data['qa_status'].append("OK")
                data['fill_color'].append("rgba(46, 204, 113, 0.0)")
            elif any("VERY" in w or "TINY" in w or "SUSPICIOUSLY" in w
                     or "GAP" in w for w in piece.warnings):
                data['qa_status'].append("Critical")
                data['fill_color'].append("rgba(231, 76, 60, 0.35)")
            else:
                data['qa_status'].append("Warning")
                data['fill_color'].append("rgba(243, 156, 18, 0.35)")

        source = ColumnDataSource(data=dict(data))
        source_full = ColumnDataSource(data=dict(data))
        return source, source_full, dict(data)

    def _build_bokeh_figure(self, bg_image, source, img_width, img_height):
        """Build the main Bokeh figure with image, patches, and tools.

        Args:
            bg_image: uint32 background image array
            source: ColumnDataSource with piece data
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            Bokeh figure object
        """
        from bokeh.plotting import figure
        from bokeh.models import HoverTool

        p = figure(
            title="Stained Glass Pattern Analysis",
            width=800,
            height=int(800 * img_height / img_width),
            x_range=(0, img_width),
            y_range=(0, img_height),
            tools="pan,wheel_zoom,box_zoom,reset,save,tap",
            active_scroll="wheel_zoom",
            match_aspect=True,
        )

        # Background image
        p.image_rgba(
            image=[bg_image],
            x=0, y=0,
            dw=img_width, dh=img_height,
        )

        # Piece polygons (transparent by default, colored for warnings)
        patches = p.patches(
            'xs', 'ys',
            source=source,
            fill_color='fill_color',
            fill_alpha=1.0,
            line_color="rgba(0, 0, 0, 0)",
            line_width=0,
            selection_fill_color="rgba(52, 152, 219, 0.4)",
            selection_line_color="#2980b9",
            selection_line_width=3,
            nonselection_fill_alpha=1.0,
            nonselection_line_alpha=0,
        )

        # Piece number labels
        from bokeh.models import ColumnDataSource
        label_source = ColumnDataSource(data=dict(
            x=source.data['cx'],
            y=source.data['cy'],
            text=[str(pid) for pid in source.data['piece_id']],
        ))
        p.text(
            'x', 'y', 'text',
            source=label_source,
            text_font_size="8pt",
            text_align="center",
            text_baseline="middle",
            text_color="#333333",
            text_font_style="bold",
        )

        # Hover tooltip
        hover = HoverTool(
            renderers=[patches],
            tooltips="""
            <div style="background: white; padding: 8px; border-radius: 4px;
                        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
                <div style="font-weight: bold; font-size: 14px;
                            margin-bottom: 4px;">Piece #@piece_id</div>
                <table>
                    <tr><td>Area:</td><td><b>@area sq in</b></td></tr>
                    <tr><td>Min Width:</td><td><b>@min_width"</b></td></tr>
                    <tr><td>Max Width:</td><td><b>@max_width"</b></td></tr>
                    <tr><td>Min Angle:</td><td><b>@min_angle°</b></td></tr>
                    <tr><td>Vertices:</td><td><b>@vertices</b></td></tr>
                    <tr><td>Status:</td><td><b>@qa_status</b></td></tr>
                </table>
                <div style="margin-top: 4px; font-size: 11px;
                            color: #666;">@warnings</div>
            </div>
            """,
        )
        p.add_tools(hover)
        p.axis.visible = False
        p.grid.visible = False

        return p

    def _add_bokeh_gaps(self, p, img_height):
        """Add gap visualization to the Bokeh figure.

        Args:
            p: Bokeh figure
            img_height: Image height for y-flip
        """
        from bokeh.models import ColumnDataSource

        gap_xs = []
        gap_ys = []
        gap_pts_x = []
        gap_pts_y = []

        for gap in self.gaps:
            gap_xs.append([gap['from'][0], gap['to'][0]])
            gap_ys.append([
                img_height - gap['from'][1],
                img_height - gap['to'][1]
            ])
            gap_pts_x.extend([gap['from'][0], gap['to'][0]])
            gap_pts_y.extend([
                img_height - gap['from'][1],
                img_height - gap['to'][1]
            ])

        gap_source = ColumnDataSource(data=dict(xs=gap_xs, ys=gap_ys))
        p.multi_line(
            'xs', 'ys', source=gap_source,
            line_color="red", line_width=4, line_dash="dashed"
        )

        # Gap endpoint markers
        p.scatter(
            gap_pts_x, gap_pts_y, size=10,
            fill_color="red", fill_alpha=0.5,
            line_color="red", line_width=2,
            marker="circle",
        )

    def _build_bokeh_panels(self, source, source_full, piece_data,
                            img_width, img_height):
        """Build the right-side info panels for the Bokeh layout.

        Includes summary, detail panel, filter checkboxes,
        warning chart, and size histogram.

        Args:
            source: Piece data ColumnDataSource
            source_full: Unfiltered copy of piece data
            piece_data: Dict of piece data arrays
            img_width: Image width
            img_height: Image height

        Returns:
            Bokeh column layout for right panel
        """
        from bokeh.plotting import figure as bokeh_figure
        from bokeh.models import (
            ColumnDataSource, CustomJS, Div, CheckboxGroup
        )
        from bokeh.layouts import column

        areas = piece_data['area']
        qa_status = piece_data['qa_status']

        total = len(self.pieces)
        ok_count = sum(1 for s in qa_status if s == "OK")
        warn_count = sum(1 for s in qa_status if s == "Warning")
        crit_count = sum(1 for s in qa_status if s == "Critical")

        # --- Summary panel ---
        gap_html = ""
        if self.gaps:
            gap_html = f"""
            <h3 style="color: #e74c3c;">⚠ Line Gaps: {len(self.gaps)}</h3>
            """

        summary_div = Div(text=f"""
        <div style="font-family: Arial, sans-serif; padding: 15px;
                    background: #f8f9fa; border-radius: 8px; width: 320px;">
            <h2 style="margin-top: 0;">Pattern Summary</h2>
            <table style="width: 100%; font-size: 14px;">
                <tr><td><b>Total Pieces:</b></td><td>{total}</td></tr>
                <tr><td><b>Scale:</b></td>
                    <td>{self.scale_dpi:.1f} DPI</td></tr>
                <tr><td><b>Panel Size:</b></td>
                    <td>{img_width / self.scale_dpi:.1f}" x
                        {img_height / self.scale_dpi:.1f}"</td></tr>
                <tr><td><b>Total Glass:</b></td>
                    <td>{sum(areas):.0f} sq in</td></tr>
            </table>
            {gap_html}
            <h3>QA Status</h3>
            <table style="width: 100%; font-size: 14px;">
                <tr>
                    <td><span style="color: #2ecc71; font-size: 1.5em;">
                        ●</span> OK:</td>
                    <td><b>{ok_count}</b>
                        ({100 * ok_count / max(total, 1):.0f}%)</td>
                </tr>
                <tr>
                    <td><span style="color: #f39c12; font-size: 1.5em;">
                        ●</span> Warning:</td>
                    <td><b>{warn_count}</b>
                        ({100 * warn_count / max(total, 1):.0f}%)</td>
                </tr>
                <tr>
                    <td><span style="color: #e74c3c; font-size: 1.5em;">
                        ●</span> Critical:</td>
                    <td><b>{crit_count}</b>
                        ({100 * crit_count / max(total, 1):.0f}%)</td>
                </tr>
            </table>
            <h3>Instructions</h3>
            <ul style="font-size: 12px; color: #555;">
                <li><b>Hover</b> for piece details</li>
                <li><b>Click</b> to highlight</li>
                <li><b>Scroll</b> to zoom</li>
                <li><b>Drag</b> to pan</li>
            </ul>
        </div>
        """)

        # --- Detail panel (updates on click) ---
        detail_div = Div(text="""
        <div style="font-family: Arial, sans-serif; padding: 15px;
                    background: #fff3cd; border-radius: 8px; width: 320px;">
            <h3 style="margin-top: 0;">Piece Detail</h3>
            <p><i>Click a piece to see details</i></p>
        </div>
        """, width=340)

        # JavaScript callback for click selection
        tap_callback = CustomJS(
            args=dict(source=source, detail_div=detail_div),
            code=self._bokeh_tap_callback_js()
        )
        source.selected.js_on_change('indices', tap_callback)

        # --- Filter checkboxes ---
        filter_callback = CustomJS(
            args=dict(source=source, source_full=source_full),
            code=self._bokeh_filter_callback_js()
        )

        filter_div = Div(text="""
        <div style="font-family: Arial; padding: 10px;
                    background: #e8e8e8; border-radius: 8px; width: 320px;">
            <h4 style="margin-top: 0;">Filter by QA Status:</h4>
        </div>
        """)

        checkbox = CheckboxGroup(
            labels=[
                f"✓ OK ({ok_count})",
                f"⚠ Warning ({warn_count})",
                f"✖ Critical ({crit_count})"
            ],
            active=[0, 1, 2],
        )
        checkbox.js_on_change('active', filter_callback)

        # --- Warning breakdown chart ---
        warn_chart = self._build_warning_chart()

        # --- Area histogram ---
        area_hist = self._build_area_histogram(areas)

        return column(
            summary_div, detail_div,
            filter_div, checkbox,
            warn_chart, area_hist,
        )

    def _build_warning_chart(self):
        """Build a bar chart showing warning type distribution.

        Returns:
            Bokeh figure or Div if no warnings
        """
        from bokeh.plotting import figure as bokeh_figure
        from bokeh.models import ColumnDataSource, Div

        warning_types = defaultdict(int)
        for piece in self.pieces:
            for w in piece.warnings:
                wtype = w.split(":")[0]
                warning_types[wtype] += 1

        if not warning_types:
            return Div(text="""
            <div style="padding: 10px; background: #d4edda;
                        border-radius: 8px;">
                <b>✓ No warnings — all pieces OK!</b>
            </div>
            """)

        warn_labels = list(warning_types.keys())
        warn_values = list(warning_types.values())

        warn_source = ColumnDataSource(data=dict(
            labels=warn_labels,
            counts=warn_values,
            colors=[
                "#e74c3c" if "VERY" in l or "TINY" in l
                             or "SUSPICIOUSLY" in l or "GAP" in l
                else "#f39c12"
                for l in warn_labels
            ]
        ))

        chart = bokeh_figure(
            title="Warning Breakdown",
            x_range=warn_labels,
            width=340, height=200,
            tools="", toolbar_location=None,
        )
        chart.vbar(
            x='labels', top='counts', width=0.8,
            source=warn_source, color='colors',
        )
        chart.xaxis.major_label_orientation = 0.8
        chart.yaxis.axis_label = "Count"

        return chart

    def _build_area_histogram(self, areas):
        """Build a histogram of piece areas.

        Args:
            areas: List of piece areas in square inches

        Returns:
            Bokeh figure
        """
        from bokeh.plotting import figure as bokeh_figure
        from bokeh.models import ColumnDataSource

        hist_values, hist_edges = np.histogram(areas, bins=20)
        hist_source = ColumnDataSource(data=dict(
            top=hist_values.tolist(),
            left=hist_edges[:-1].tolist(),
            right=hist_edges[1:].tolist(),
        ))

        hist = bokeh_figure(
            title="Piece Size Distribution",
            width=340, height=200,
            tools="", toolbar_location=None,
        )
        hist.quad(
            top='top', bottom=0, left='left', right='right',
            source=hist_source,
            fill_color="#3498db", line_color="white", alpha=0.8,
        )
        hist.xaxis.axis_label = "Area (sq in)"
        hist.yaxis.axis_label = "Count"

        return hist

    @staticmethod
    def _bokeh_tap_callback_js():
        """Return JavaScript code for the piece-click callback."""
        return """
            const indices = source.selected.indices;
            if (indices.length === 0) {
                detail_div.text = `
                <div style="font-family: Arial; padding: 15px;
                            background: #fff3cd; border-radius: 8px;
                            width: 320px;">
                    <h3 style="margin-top: 0;">Piece Detail</h3>
                    <p><i>Click a piece to see details</i></p>
                </div>`;
                return;
            }

            const idx = indices[0];
            const data = source.data;
            const status = data.qa_status[idx];
            const status_color = status === "OK" ? "#2ecc71" :
                                status === "Warning" ? "#f39c12" : "#e74c3c";
            const bg_color = status === "OK" ? "#d4edda" :
                            status === "Warning" ? "#fff3cd" : "#f8d7da";

            const warnings = data.warnings[idx];
            const warn_html = warnings === "None" ?
                "<p style='color: #2ecc71;'><b>✓ No issues</b></p>" :
                warnings.split("; ").map(w =>
                    "<p style='color: #c0392b; margin: 2px 0;'>⚠ "
                    + w + "</p>"
                ).join("");

            detail_div.text = `
            <div style="font-family: Arial; padding: 15px;
                        background: ${bg_color}; border-radius: 8px;
                        width: 320px;">
                <h3 style="margin-top: 0;">
                    Piece #${data.piece_id[idx]}
                    <span style="color: ${status_color}; font-size: 0.8em;">
                        (${status})</span>
                </h3>
                <table style="width: 100%; font-size: 14px;">
                    <tr><td>Area:</td>
                        <td><b>${data.area[idx]} sq in</b></td></tr>
                    <tr><td>Min Width:</td>
                        <td><b>${data.min_width[idx]}"</b></td></tr>
                    <tr><td>Max Width:</td>
                        <td><b>${data.max_width[idx]}"</b></td></tr>
                    <tr><td>Min Angle:</td>
                        <td><b>${data.min_angle[idx]}°</b></td></tr>
                    <tr><td>Vertices:</td>
                        <td><b>${data.vertices[idx]}</b></td></tr>
                </table>
                <h4 style="margin-bottom: 4px;">Warnings:</h4>
                ${warn_html}
            </div>`;
        """

    @staticmethod
    def _bokeh_filter_callback_js():
        """Return JavaScript code for the QA status filter callback."""
        return """
            const checkboxes = cb_obj.active;
            const show_ok = checkboxes.includes(0);
            const show_warn = checkboxes.includes(1);
            const show_crit = checkboxes.includes(2);

            const full = source_full.data;
            const filtered = {};
            for (const key of Object.keys(full)) {
                filtered[key] = [];
            }

            for (let i = 0; i < full.qa_status.length; i++) {
                const status = full.qa_status[i];
                const show = (status === "OK" && show_ok) ||
                           (status === "Warning" && show_warn) ||
                           (status === "Critical" && show_crit);
                if (show) {
                    for (const key of Object.keys(full)) {
                        filtered[key].push(full[key][i]);
                    }
                }
            }

            source.data = filtered;
            source.change.emit();
        """

    # =========================================================================
    # Convenience Method
    # =========================================================================

    def generate_all(self, prefix="pattern"):
        """Generate all output files.

        Runs the full pipeline and generates all outputs:
            - Annotated image
            - QA overlay image
            - Text report
            - Interactive Bokeh visualization (if available)

        Args:
            prefix: Output filename prefix
        """
        self.generate_annotated_image(prefix=prefix)
        self.generate_qa_overlay(prefix=prefix)
        self.generate_report(prefix=prefix)

        try:
            self.generate_bokeh_visualization(prefix=prefix)
        except ImportError as e:
            print(f"\nBokeh visualization skipped: {e}")
            print("Install with: pip install bokeh")
        except Exception as e:
            print(f"\nBokeh visualization failed: {e}")
            import traceback
            traceback.print_exc()


# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Stained Glass Pattern Analyzer — '
                    'Detect pieces, measure properties, and check quality.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with auto DPI
  %(prog)s pattern.png -o ./output

  # Specify actual panel width for correct measurements
  %(prog)s pattern.jpg --panel-width 24 -o ./output

  # Specify panel height instead
  %(prog)s pattern.jpg --panel-height 48 -o ./output

  # Adjust line detection for faint lines
  %(prog)s pattern.jpg --threshold 100 --dilate 2 -o ./output

  # Skip Bokeh visualization
  %(prog)s pattern.jpg --panel-width 24 --no-bokeh -o ./output
        """
    )

    # Required
    parser.add_argument(
        'image',
        help='Path to pattern image (JPG, PNG, etc.)'
    )

    # Scale options
    scale_group = parser.add_argument_group('Scale Options')
    scale_group.add_argument(
        '--dpi', type=int, default=150,
        help='Scale DPI for measurements (default: 150). '
             'Overridden by --panel-width or --panel-height.'
    )
    scale_group.add_argument(
        '--panel-width', type=float, default=None,
        help='Actual panel width in inches (auto-calculates DPI)'
    )
    scale_group.add_argument(
        '--panel-height', type=float, default=None,
        help='Actual panel height in inches (auto-calculates DPI)'
    )

    # Image processing options
    proc_group = parser.add_argument_group('Image Processing')
    proc_group.add_argument(
        '--threshold', type=int, default=128,
        help='Line detection threshold 0-255 (default: 128). '
             'Lower = detect fainter lines.'
    )
    proc_group.add_argument(
        '--close-kernel', type=int, default=3,
        help='Morphological closing kernel size (default: 3). '
             'Seals small gaps from JPG compression.'
    )
    proc_group.add_argument(
        '--dilate', type=int, default=1,
        help='Line dilation iterations (default: 1). '
             'Thickens lines to seal larger gaps.'
    )
    proc_group.add_argument(
        '--min-area', type=int, default=200,
        help='Minimum piece area in pixels (default: 200). '
             'Smaller regions treated as noise.'
    )

    # Gap detection
    gap_group = parser.add_argument_group('Gap Detection')
    gap_group.add_argument(
        '--max-gap', type=int, default=20,
        help='Maximum gap distance in pixels to detect (default: 20)'
    )
    gap_group.add_argument(
        '--min-gap', type=int, default=2,
        help='Minimum gap distance in pixels (default: 2)'
    )
    gap_group.add_argument(
        '--suspicious-ratio', type=float, default=8.0,
        help='Flag pieces larger than this multiple of median area '
             'as suspiciously large / possible merged pieces (default: 8.0)'
    )

    # Output options
    out_group = parser.add_argument_group('Output')
    out_group.add_argument(
        '--prefix', type=str, default='pattern',
        help='Output filename prefix (default: pattern)'
    )
    out_group.add_argument(
        '--output-dir', '-o', type=str, default='./output',
        help='Output directory (default: ./output)'
    )
    out_group.add_argument(
        '--no-bokeh', action='store_true',
        help='Skip interactive Bokeh HTML visualization'
    )

    args = parser.parse_args()

    # --- Run analysis pipeline ---
    print("Stained Glass Pattern Analyzer")
    print("=" * 40)

    analyzer = PatternAnalyzer(
        args.image,
        scale_dpi=args.dpi,
        output_dir=args.output_dir,
        panel_width=args.panel_width,
        panel_height=args.panel_height,
    )

    print("\n--- Preprocessing ---")
    analyzer.preprocess(
        line_threshold=args.threshold,
        close_kernel_size=args.close_kernel,
        dilate_iterations=args.dilate
    )

    print("\n--- Detecting Pieces ---")
    analyzer.detect_pieces(min_area_pixels=args.min_area)

    print("\n--- Detecting Line Gaps ---")
    analyzer.detect_line_gaps(
        max_gap_pixels=args.max_gap,
        min_gap_pixels=args.min_gap,
        suspicious_ratio=args.suspicious_ratio,
    )

    print("\n--- Running QA ---")
    analyzer.run_qa()

    print("\n--- Generating Outputs ---")
    analyzer.generate_annotated_image(prefix=args.prefix)
    analyzer.generate_qa_overlay(prefix=args.prefix)
    analyzer.generate_report(prefix=args.prefix)

    if not args.no_bokeh:
        print("\n--- Generating Interactive Visualization ---")
        try:
            analyzer.generate_bokeh_visualization(prefix=args.prefix)
        except ImportError as e:
            print(f"Bokeh visualization skipped: {e}")
            print("Install with: pip install bokeh")
        except Exception as e:
            print(f"Bokeh visualization failed: {e}")
            import traceback
            traceback.print_exc()

    # --- Summary ---
    print("\n--- Done! ---")
    output_dir = Path(args.output_dir).resolve()
    print(f"\nAll output files in: {output_dir}")
    print(f"  debug_00_threshold_only.png    — raw threshold result")
    print(f"  debug_01_binary.png            — after morphological cleanup")
    print(f"  debug_02_lines_highlighted.png — lines shown in red")
    print(f"  debug_04_gaps_found.png        — detected line gaps")
    print(f"  {args.prefix}_analyzed.png     — numbered pieces + QA")
    print(f"  {args.prefix}_qa.png           — QA issues only")
    print(f"  {args.prefix}_report.txt       — full text report")
    if not args.no_bokeh:
        print(f"  {args.prefix}_interactive.html — interactive visualization")


if __name__ == "__main__":
    main()
