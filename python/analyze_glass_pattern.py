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
    sampled_color: tuple = None
    color_group_id: int = None
    color_group_center: tuple = None
    color_group_name: str = None

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
                 panel_width=None, panel_height=None,
                 zinc_channel_depth=0.25, zinc_face_width=0.5,
                 came_width=None, technique='lead'):

        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.pieces = []
        self.gaps = []
        self.binary = None
        self.image_path = image_path
        self.zinc_channel_depth = zinc_channel_depth
        self.zinc_face_width = zinc_face_width

        img_h, img_w = self.image.shape[:2]

        # Panel dimensions: user specifies outer (zinc) dimension
        # Pattern (glass) dimension is smaller by 2x channel depth
        if panel_width:
            self.outer_width = panel_width
            self.pattern_width = panel_width - 2 * zinc_channel_depth
            self.scale_dpi = img_w / self.pattern_width
            self.outer_height = img_h / self.scale_dpi + 2 * zinc_channel_depth
            self.pattern_height = img_h / self.scale_dpi
            print(f"Zinc outer dimension: "
                  f"{self.outer_width:.1f}\" x {self.outer_height:.1f}\"")
            print(f"Pattern (glass) dimension: "
                  f"{self.pattern_width:.1f}\" x {self.pattern_height:.1f}\"")
            print(f"Zinc channel depth: {zinc_channel_depth}\"")
            print(f"Calculated DPI from pattern width: "
                  f"{self.scale_dpi:.1f}")
        elif panel_height:
            self.outer_height = panel_height
            self.pattern_height = panel_height - 2 * zinc_channel_depth
            self.scale_dpi = img_h / self.pattern_height
            self.outer_width = img_w / self.scale_dpi + 2 * zinc_channel_depth
            self.pattern_width = img_w / self.scale_dpi
            print(f"Zinc outer dimension: "
                  f"{self.outer_width:.1f}\" x {self.outer_height:.1f}\"")
            print(f"Pattern (glass) dimension: "
                  f"{self.pattern_width:.1f}\" x {self.pattern_height:.1f}\"")
            print(f"Zinc channel depth: {zinc_channel_depth}\"")
            print(f"Calculated DPI from pattern height: "
                  f"{self.scale_dpi:.1f}")
        else:
            self.scale_dpi = scale_dpi
            self.pattern_width = img_w / self.scale_dpi
            self.pattern_height = img_h / self.scale_dpi
            self.outer_width = self.pattern_width + 2 * zinc_channel_depth
            self.outer_height = self.pattern_height + 2 * zinc_channel_depth

        # Set up output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loaded image: {img_w}x{img_h} pixels")
        print(f"Output directory: {self.output_dir.resolve()}")
        print(f"Pixel value range: {self.gray.min()} - {self.gray.max()}")
        print(f"Mean pixel value: {self.gray.mean():.1f}")

        # Came/foil width for template scissors
        # Lead scissors remove 1/16" strip
        # Foil scissors remove ~1.5 mil strip
        if came_width:
            self.came_width = came_width
        elif technique == 'foil':
            self.came_width = 0.0015  # 1.5 mil
        else:
            self.came_width = 1 / 16  # 0.0625"

        self.technique = technique

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

    def calculate_frame(self):
        """Calculate zinc frame requirements for the panel."""
        img_h, img_w = self.image.shape[:2]

        panel_w = self.pattern_width
        panel_h = self.pattern_height
        outer_w = self.outer_width
        outer_h = self.outer_height
        depth = self.zinc_channel_depth
        face = self.zinc_face_width

        # Detect panel shape
        contours, hierarchy = cv2.findContours(
            self.binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        image_area = img_h * img_w
        panel_shape = "rectangular"

        if contours:
            sorted_contours = sorted(
                contours, key=cv2.contourArea, reverse=True
            )
            for c in sorted_contours:
                area = cv2.contourArea(c)
                if area > image_area * 0.95:
                    continue
                if area < image_area * 0.3:
                    break
                rect = cv2.minAreaRect(c)
                rect_area = rect[1][0] * rect[1][1]
                if rect_area > 0:
                    rectangularity = area / rect_area
                else:
                    rectangularity = 1.0
                if rectangularity < 0.9:
                    perimeter_px = cv2.arcLength(c, True)
                    vertices = cv2.approxPolyDP(
                        c, 0.02 * perimeter_px, True
                    )
                    if len(vertices) > 8:
                        panel_shape = "curved/elliptical"
                    elif len(vertices) > 4:
                        panel_shape = "polygonal"
                break

        # Rectangular zinc perimeter uses OUTER dimensions
        rect_perimeter = 2 * (outer_w + outer_h)

        # Zinc frame inventory for rectangular panels
        if panel_shape == "rectangular":
            zinc_pieces = [
                {'label': 'Top', 'length_inches': round(outer_w, 2),
                 'orientation': 'horizontal'},
                {'label': 'Bottom', 'length_inches': round(outer_w, 2),
                 'orientation': 'horizontal'},
                {'label': 'Left', 'length_inches': round(outer_h, 2),
                 'orientation': 'vertical'},
                {'label': 'Right', 'length_inches': round(outer_h, 2),
                 'orientation': 'vertical'},
            ]
        else:
            zinc_pieces = None

        # Interior came calculation
        total_piece_perimeter = 0
        for piece in self.pieces:
            perim_px = cv2.arcLength(piece.contour, True)
            total_piece_perimeter += perim_px / self.scale_dpi

        pattern_perimeter = 2 * (panel_w + panel_h)
        interior_came = (total_piece_perimeter / 2) - pattern_perimeter
        interior_came = max(0, interior_came)

        waste_factor = 1.10

        self.frame_info = {
            'panel_shape': panel_shape,
            'outer_width': outer_w,
            'outer_height': outer_h,
            'pattern_width': panel_w,
            'pattern_height': panel_h,
            'zinc_channel_depth': depth,
            'zinc_face_width': face,
            'zinc_pieces': zinc_pieces,
            'zinc_perimeter': round(rect_perimeter, 1),
            'zinc_perimeter_with_waste': round(
                rect_perimeter * waste_factor, 1),
            'interior_came_inches': round(interior_came, 1),
            'interior_came_feet': round(interior_came / 12, 2),
            'interior_came_with_waste': round(
                interior_came * waste_factor / 12, 2),
            'total_came_inches': round(
                rect_perimeter + interior_came, 1),
            'total_came_feet': round(
                (rect_perimeter + interior_came) / 12, 2),
            'total_with_waste_feet': round(
                (rect_perimeter + interior_came) * waste_factor / 12, 2),
        }

        print(f"\nFrame calculation:")
        print(f"  Panel shape: {panel_shape}")
        print(f"  Zinc outer: {outer_w:.1f}\" x {outer_h:.1f}\"")
        print(f"  Pattern (glass): {panel_w:.1f}\" x {panel_h:.1f}\"")
        print(f"  Interior came: {interior_came:.1f}\" "
              f"({interior_came / 12:.1f} ft)")
        print(f"  Total came needed (with 10% waste): "
              f"{(rect_perimeter + interior_came) * waste_factor / 12:.1f} ft")

        return self.frame_info

    def analyze_colors_from_source(self, source_image_path, num_colors=6):
        """Sample colors from a source image for each detected piece.

        Loads a reference image (photo, painting, etc.) that corresponds
        to the pattern, and samples the dominant color within each piece's
        region. Then clusters all piece colors into groups for glass
        purchasing.

        The source image must align with the pattern image — same scene,
        same boundaries. It will be resized to match the pattern if needed.

        Args:
            source_image_path: Path to the reference image
            num_colors: Number of color groups to cluster into (default 6)

        Returns:
            Dict with color group info:
                {group_id: {'color_bgr': (b,g,r), 'color_name': str,
                            'piece_ids': [...], 'total_area_sq_in': float}}
        """
        # Load source image
        source = cv2.imread(source_image_path)
        if source is None:
            raise FileNotFoundError(
                f"Could not load source image: {source_image_path}"
            )

        # Resize source to match pattern if needed
        pattern_h, pattern_w = self.image.shape[:2]
        source_h, source_w = source.shape[:2]

        if (source_w, source_h) != (pattern_w, pattern_h):
            print(f"Resizing source image from {source_w}x{source_h} "
                  f"to {pattern_w}x{pattern_h}")
            source = cv2.resize(source, (pattern_w, pattern_h),
                                interpolation=cv2.INTER_AREA)

        self.source_image = source

        # Sample dominant color for each piece
        print(f"\nSampling colors from source image...")
        piece_colors = []

        for piece in self.pieces:
            # Create mask for this piece
            mask = np.zeros(self.gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [piece.contour], -1, 255, -1)

            # Erode mask slightly to avoid sampling from lead lines
            erode_kernel = np.ones((5, 5), np.uint8)
            mask_eroded = cv2.erode(mask, erode_kernel, iterations=2)

            # If erosion eliminated the mask (tiny piece), use original
            if cv2.countNonZero(mask_eroded) < 10:
                mask_eroded = mask

            # Sample pixels within the mask
            pixels = source[mask_eroded > 0]

            if len(pixels) == 0:
                piece.sampled_color = (128, 128, 128)  # default grey
            else:
                # Use median for robustness against outliers
                median_color = np.median(pixels, axis=0).astype(int)
                piece.sampled_color = tuple(median_color)

            piece_colors.append(piece.sampled_color)
            print(f"  Piece {piece.id:>3}: BGR={piece.sampled_color}")

        # Cluster colors into groups using K-means
        color_groups = self._cluster_colors(piece_colors, num_colors)

        # Check if absolute names are distinctive
        abs_names = [g['color_name'] for g in self.color_groups.values()]
        unique_names = len(set(abs_names))

        if unique_names < len(abs_names) * 0.7:
            # Too many duplicate names — use relative naming
            print(f"\nAbsolute naming produced only {unique_names} "
                  f"unique names for {len(abs_names)} groups — "
                  f"switching to relative naming")
            self._name_colors_relative()
        else:
            print(f"\nUsing absolute color names "
                  f"({unique_names} unique names)")

        return color_groups

    def _cluster_colors(self, piece_colors, num_colors):
        """Cluster piece colors into groups using K-means.

        Args:
            piece_colors: List of (B, G, R) tuples, one per piece
            num_colors: Number of clusters

        Returns:
            Dict of color groups with piece assignments and areas
        """
        from sklearn.cluster import KMeans

        # Prepare data for clustering
        color_array = np.array(piece_colors, dtype=np.float32)

        # Don't request more clusters than pieces
        num_colors = min(num_colors, len(self.pieces))

        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
        labels = kmeans.fit_predict(color_array)
        centers = kmeans.cluster_centers_.astype(int)

        # Build color group info
        color_groups = {}
        for group_id in range(num_colors):
            group_pieces = [
                self.pieces[i] for i in range(len(self.pieces))
                if labels[i] == group_id
            ]

            center_bgr = tuple(centers[group_id])
            color_name = self._name_color(center_bgr)

            total_area = sum(p.area_sq_inches for p in group_pieces)
            piece_ids = [p.id for p in group_pieces]

            color_groups[group_id] = {
                'color_bgr': center_bgr,
                'color_rgb': (center_bgr[2], center_bgr[1], center_bgr[0]),
                'color_name': color_name,
                'piece_ids': piece_ids,
                'piece_count': len(piece_ids),
                'total_area_sq_in': round(total_area, 2),
            }

            # Assign group info back to each piece
            for piece in group_pieces:
                piece.color_group_id = group_id
                piece.color_group_center = center_bgr
                piece.color_group_name = color_name

            print(f"  Group {group_id}: {color_name} "
                  f"BGR={center_bgr} — "
                  f"{len(piece_ids)} pieces, "
                  f"{total_area:.1f} sq in")

        self.color_groups = color_groups
        return color_groups

    def _name_color(self, bgr):
        """Generate a human-readable name for a BGR color.

        Uses HSV conversion to classify into basic color categories.

        Args:
            bgr: (B, G, R) color tuple

        Returns:
            String color name like "Steel Blue", "Warm Tan", etc.
        """
        # Convert to HSV for easier classification
        pixel = np.uint8([[list(bgr)]])
        hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

        # Low saturation = grey/white/black
        if s < 30:
            if v < 60:
                return "Dark Grey"
            elif v < 130:
                return "Medium Grey"
            elif v < 200:
                return "Light Grey"
            else:
                return "White"

        # Low value = dark
        if v < 50:
            return "Black"

        # Classify by hue
        # OpenCV hue range is 0-179
        brightness = "Dark" if v < 120 else "Medium" if v < 180 else "Light"

        if h < 10 or h >= 170:
            base = "Red"
        elif h < 22:
            base = "Orange" if s > 100 else "Brown"
        elif h < 35:
            base = "Gold" if v > 150 else "Brown"
        elif h < 45:
            base = "Yellow"
        elif h < 75:
            base = "Green"
        elif h < 100:
            base = "Teal"
        elif h < 130:
            base = "Blue"
        elif h < 145:
            base = "Steel Blue"
        elif h < 160:
            base = "Purple"
        else:
            base = "Pink"

        return f"{brightness} {base}"

    def generate_color_report(self, prefix="pattern"):
        """Generate a color summary report for glass purchasing.

        Args:
            prefix: Output filename prefix

        Returns:
            Report text as string
        """
        if not hasattr(self, 'color_groups') or not self.color_groups:
            print("No color analysis done yet. "
                  "Run analyze_colors_from_source() first.")
            return ""

        output_path = self._output_path(f"{prefix}_color_report.txt")

        lines = []
        lines.append("=" * 60)
        lines.append("GLASS COLOR SUMMARY — PURCHASING GUIDE")
        lines.append("=" * 60)
        lines.append("")

        total_area = sum(
            g['total_area_sq_in'] for g in self.color_groups.values()
        )

        lines.append(f"Total glass area: {total_area:.1f} sq in")
        lines.append(f"Color groups: {len(self.color_groups)}")
        lines.append("")

        lines.append("-" * 60)
        lines.append(
            f"{'#':>3} {'Color':>20} {'Pieces':>8} {'Area (sq in)':>14} {'%':>6}"
        )
        lines.append("-" * 60)

        for gid, group in sorted(
                self.color_groups.items(),
                key=lambda x: -x[1]['total_area_sq_in']):
            pct = 100 * group['total_area_sq_in'] / total_area
            rgb = group['color_rgb']
            lines.append(
                f"{gid:>3} {group['color_name']:>20} "
                f"{group['piece_count']:>8} "
                f"{group['total_area_sq_in']:>14.1f} "
                f"{pct:>5.1f}%"
            )
            lines.append(
                f"{'':>24} RGB=({rgb[0]},{rgb[1]},{rgb[2]}) "
                f"Pieces: {group['piece_ids']}"
            )

        lines.append("")
        lines.append("-" * 60)
        lines.append("PER-PIECE COLOR ASSIGNMENTS")
        lines.append("-" * 60)
        lines.append(
            f"{'ID':>4} {'Color Group':>20} {'Area':>10} {'Sampled BGR':>20}"
        )

        for p in sorted(self.pieces, key=lambda x: x.id):
            name = getattr(p, 'color_group_name', 'Unassigned')
            bgr = getattr(p, 'sampled_color', (0, 0, 0))
            lines.append(
                f"{p.id:>4} {name:>20} "
                f"{p.area_sq_inches:>9.2f} "
                f"({bgr[0]:>3},{bgr[1]:>3},{bgr[2]:>3})"
            )

        lines.append("")
        lines.append("=" * 60)

        report_text = "\n".join(lines)
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"\nSaved color report: {output_path}")
        print("\n" + report_text)

        return report_text

    def generate_colored_pattern(self, prefix="pattern"):
        """Generate a pattern image with pieces filled by color group.

        Args:
            prefix: Output filename prefix

        Returns:
            Path to saved image
        """
        if not hasattr(self, 'color_groups') or not self.color_groups:
            print("No color analysis done yet. "
                  "Run analyze_colors_from_source() first.")
            return None

        output_path = self._output_path(f"{prefix}_colored.png")

        # Start with white background
        colored = np.ones_like(self.image) * 255

        # Fill each piece with its cluster color
        for piece in self.pieces:
            fill_color = getattr(piece, 'color_group_center', (200, 200, 200))
            fill_color = self._boost_color(fill_color)
            cv2.drawContours(
                colored, [piece.contour], -1, fill_color, -1
            )

        # Redraw the lead lines on top
        colored[self.binary == 0] = [0, 0, 0]

        # Add piece numbers
        for piece in self.pieces:
            cx, cy = piece.centroid
            font_scale = max(
                0.3, min(0.7, np.sqrt(piece.area_sq_inches) * 0.3)
            )
            text = str(piece.id)
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            cv2.putText(
                colored, text,
                (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0), 1
            )

        cv2.imwrite(output_path, colored)
        print(f"Saved colored pattern: {output_path}")
        return output_path

    def _boost_color(self, bgr, saturation_boost=2.5, value_boost=1.4):
        """Boost saturation and brightness to make subtle colors visible.

        Args:
            bgr: (B, G, R) color tuple
            saturation_boost: Multiplier for saturation (default 2.5)
            value_boost: Multiplier for brightness (default 1.4)

        Returns:
            Boosted (B, G, R) tuple
        """
        pixel = np.uint8([[list(bgr)]])
        hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
        h = hsv[0]
        s = min(255, int(hsv[1] * saturation_boost))
        v = min(255, int(hsv[2] * value_boost))
        boosted = cv2.cvtColor(
            np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR
        )[0][0]
        return tuple(int(c) for c in boosted)

    def _name_color(self, bgr):
        """Generate a human-readable name for a BGR color."""
        pixel = np.uint8([[list(bgr)]])
        hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

        # Determine warmth from the original BGR
        b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
        warmth = r - b  # positive = warm, negative = cool

        # Low saturation — grey family, distinguish by value and warmth
        if s < 40:
            if v < 60:
                temp = "Warm " if warmth > 5 else "Cool " if warmth < -5 else ""
                return f"Dark {temp}Grey"
            elif v < 100:
                temp = "Warm " if warmth > 5 else "Cool " if warmth < -5 else ""
                return f"Medium {temp}Grey"
            elif v < 160:
                temp = "Warm " if warmth > 5 else "Cool " if warmth < -5 else ""
                return f"Light {temp}Grey"
            elif v < 210:
                return "Pale Grey" if warmth < 3 else "Cream"
            else:
                return "White"

        # Low value = very dark
        if v < 50:
            return "Black"

        # Classify by hue
        brightness = "Dark" if v < 120 else "Medium" if v < 180 else "Light"

        if h < 10 or h >= 170:
            base = "Red"
        elif h < 22:
            base = "Orange" if s > 100 else "Brown"
        elif h < 35:
            base = "Gold" if v > 150 else "Brown"
        elif h < 45:
            base = "Yellow"
        elif h < 75:
            base = "Green"
        elif h < 100:
            base = "Teal"
        elif h < 130:
            base = "Blue"
        elif h < 145:
            base = "Steel Blue"
        elif h < 160:
            base = "Purple"
        else:
            base = "Pink"

        return f"{brightness} {base}"

    def _name_colors_relative(self):
        """Rename color groups based on relative brightness and warmth.

        Instead of absolute HSV thresholds, ranks the clusters
        from darkest to lightest and coolest to warmest, producing
        more distinctive names.

        Call after _cluster_colors has set self.color_groups.
        """
        if not self.color_groups:
            return

        groups = list(self.color_groups.items())

        # Calculate brightness and warmth for each group
        for gid, group in groups:
            b, g, r = group['color_bgr']
            group['brightness'] = int(r) + int(g) + int(b)
            group['warmth'] = int(r) - int(b)

        # Sort by brightness
        groups_by_bright = sorted(groups, key=lambda x: x[1]['brightness'])
        n = len(groups_by_bright)

        # Assign brightness tier
        for i, (gid, group) in enumerate(groups_by_bright):
            if n <= 3:
                tiers = ["Dark", "Medium", "Light"]
            elif n <= 5:
                tiers = ["Darkest", "Dark", "Medium", "Light", "Lightest"]
            else:
                tiers = ["Darkest", "Dark", "Medium Dark",
                         "Medium Light", "Light", "Lightest"]
            group['bright_tier'] = tiers[min(i, len(tiers) - 1)]

        # Determine warmth descriptor
        warmths = [g['warmth'] for _, g in groups]
        avg_warmth = np.mean(warmths)

        for gid, group in groups:
            w = group['warmth']
            if w > avg_warmth + 5:
                temp = "Warm"
            elif w < avg_warmth - 5:
                temp = "Cool"
            else:
                temp = "Neutral"

            rgb = group['color_rgb']
            name = f"{group['bright_tier']} {temp}"
            group['color_name'] = name

            # Update pieces
            for piece in self.pieces:
                if getattr(piece, 'color_group_id', None) == gid:
                    piece.color_group_name = name

        print("\nRelative color naming:")
        for gid, group in sorted(groups,
                                 key=lambda x: x[1]['brightness']):
            print(f"  Group {gid}: {group['color_name']} "
                  f"RGB={group['color_rgb']} "
                  f"(brightness={group['brightness']}, "
                  f"warmth={group['warmth']})")

    # =========================================================================
    # Output Generation — Static Images
    # =========================================================================

    def generate_template(self, prefix="pattern"):
        """Generate a clean numbered pattern for printing as templates.

        Lines are drawn at the actual came/foil width so that pattern
        scissors (which remove a strip equal to the came width) cut
        correctly. DPI metadata is set for actual-size printing.

        Args:
            prefix: Output filename prefix

        Returns:
            Path to saved image
        """
        output_path = self._output_path(f"{prefix}_template.png")

        # Calculate line width in pixels from came width
        line_width_px = max(1, int(round(
            self.came_width * self.scale_dpi
        )))

        print(f"Template line width: {self.came_width}\" = "
              f"{line_width_px}px at {self.scale_dpi:.0f} DPI "
              f"({self.technique})")

        # Start with white background
        img_h, img_w = self.gray.shape
        template = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255

        # Draw each piece contour at the correct came width
        for piece in self.pieces:
            cv2.drawContours(
                template, [piece.contour], -1,
                (0, 0, 0), line_width_px, lineType=cv2.LINE_AA
            )

        # Draw outer border at zinc face width
        zinc_border_px = max(1, int(round(
            self.zinc_face_width * self.scale_dpi
        )))
        cv2.rectangle(
            template,
            (0, 0),
            (img_w - 1, img_h - 1),
            (0, 0, 0), zinc_border_px
        )

        # Add piece numbers
        for piece in self.pieces:
            cx, cy = piece.centroid
            font_scale = max(
                0.3, min(0.7, np.sqrt(piece.area_sq_inches) * 0.3)
            )
            text = str(piece.id)
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )

            cv2.rectangle(
                template,
                (cx - tw // 2 - 2, cy - th // 2 - 2),
                (cx + tw // 2 + 2, cy + th // 2 + 2),
                (255, 255, 255), -1
            )
            cv2.putText(
                template, text,
                (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0), 1
            )

        # Add dimension info at bottom margin
        technique_label = (
            "copper foil" if self.technique == 'foil' else "lead came"
        )
        if hasattr(self, 'frame_info'):
            f = self.frame_info
            dim_text = (
                f"Pattern: {f['pattern_width']:.1f}\" x "
                f"{f['pattern_height']:.1f}\"  |  "
                f"Zinc: {f['outer_width']:.1f}\" x "
                f"{f['outer_height']:.1f}\"  |  "
                f"{len(self.pieces)} pieces  |  "
                f"{technique_label} {self.came_width:.4g}\"  |  "
                f"Print at {self.scale_dpi:.0f} DPI"
            )
        else:
            dim_text = (
                f"Pattern: {img_w / self.scale_dpi:.1f}\" x "
                f"{img_h / self.scale_dpi:.1f}\"  |  "
                f"{len(self.pieces)} pieces  |  "
                f"{technique_label} {self.came_width:.4g}\"  |  "
                f"Print at {self.scale_dpi:.0f} DPI"
            )

        label_h = 40
        label_strip = np.ones((label_h, img_w, 3), dtype=np.uint8) * 255
        cv2.putText(
            label_strip, dim_text,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 0, 0), 1
        )
        template = np.vstack([template, label_strip])

        # Save with DPI metadata using Pillow
        try:
            from PIL import Image
            template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(template_rgb)
            dpi = int(round(self.scale_dpi))
            pil_image.save(output_path, dpi=(dpi, dpi))
            print(f"Saved template: {output_path} "
                  f"(DPI set to {dpi} for actual-size printing)")
        except ImportError:
            cv2.imwrite(output_path, template)
            print(f"Saved template: {output_path} "
                  f"(Install Pillow for automatic DPI metadata)")

        return output_path

    def generate_packed_templates(self, prefix="pattern",
                                  page_width=8.5, page_height=11.0,
                                  margin=0.5, piece_gap=0.25,
                                  overlap=0.5):
        """Generate packed piece templates as a single multi-page PDF.

        Arranges individual pieces onto pages for efficient printing.
        Pieces larger than a single page are split across multiple
        pages with overlap marks for alignment.

        Args:
            prefix: Output filename prefix
            page_width: Paper width in inches (default 8.5)
            page_height: Paper height in inches (default 11.0)
            margin: Page margin in inches (default 0.5)
            piece_gap: Gap between pieces in inches (default 0.25)
            overlap: Overlap for multi-page pieces in inches (default 0.5)

        Returns:
            Path to saved PDF file
        """
        from PIL import Image

        dpi = int(round(self.scale_dpi))

        # Usable area in pixels
        usable_w = page_width - 2 * margin
        usable_h = page_height - 2 * margin - 0.4  # reserve for footer
        usable_w_px = int(usable_w * dpi)
        usable_h_px = int(usable_h * dpi)
        page_w_px = int(page_width * dpi)
        page_h_px = int(page_height * dpi)
        margin_px = int(margin * dpi)
        gap_px = int(piece_gap * dpi)
        overlap_px = int(overlap * dpi)

        line_width_px = max(1, int(round(self.came_width * dpi)))

        print(f"\nPacking pieces onto {page_width}\" x {page_height}\" pages")
        print(f"  Usable area: {usable_w:.1f}\" x {usable_h:.1f}\"")
        print(f"  Piece gap: {piece_gap}\"")
        print(f"  Line width: {self.came_width}\" ({line_width_px}px)")

        # Get bounding box for each piece
        piece_rects = []
        for piece in self.pieces:
            x, y, w, h = piece.bounding_box
            piece_rects.append({
                'piece': piece,
                'width': w + line_width_px,
                'height': h + line_width_px,
            })

        # Separate oversized pieces
        normal_pieces = []
        oversized_pieces = []

        for pr in piece_rects:
            if pr['width'] > usable_w_px or pr['height'] > usable_h_px:
                oversized_pieces.append(pr)
            else:
                normal_pieces.append(pr)

        if oversized_pieces:
            print(f"  {len(oversized_pieces)} oversized piece(s) "
                  f"will span multiple pages")

        # Sort normal pieces by height (tallest first) for shelf packing
        normal_pieces.sort(key=lambda p: p['height'], reverse=True)

        # Shelf packing
        page_assignments = self._shelf_pack(
            normal_pieces, usable_w_px, usable_h_px, gap_px
        )

        # Count total pages for footer
        oversized_page_count = 0
        for pr in oversized_pieces:
            bx, by, bw, bh = pr['piece'].bounding_box
            step_x = usable_w_px - overlap_px
            step_y = usable_h_px - overlap_px
            cols = max(1, int(np.ceil(bw / step_x)))
            rows = max(1, int(np.ceil(bh / step_y)))
            oversized_page_count += cols * rows

        total_pages = len(page_assignments) + oversized_page_count

        # Render all pages
        all_pages = []
        page_num = 0

        for page_pieces in page_assignments:
            page_num += 1
            page_img = self._render_template_page(
                page_pieces, page_w_px, page_h_px,
                margin_px, line_width_px, gap_px,
                page_num, total_pages,
                page_width, page_height
            )
            all_pages.append(page_img)

            # Log pieces on this page
            piece_ids = [pr['piece'].id for pr, _, _ in page_pieces]
            print(f"  Page {page_num}: Pieces {piece_ids}")

        for pr in oversized_pieces:
            piece = pr['piece']
            piece_pages = self._render_oversized_piece(
                piece, page_w_px, page_h_px,
                margin_px, usable_w_px, usable_h_px,
                line_width_px, overlap_px,
                page_width, page_height,
                page_num, total_pages
            )

            for sub_img, sub_label in piece_pages:
                page_num += 1
                all_pages.append(sub_img)
                print(f"  Page {page_num}: Piece {piece.id} — {sub_label}")

        # Save as multi-page PDF
        output_path = self._output_path(f"{prefix}_templates.pdf")

        pil_pages = []
        for page_img in all_pages:
            rgb = cv2.cvtColor(page_img, cv2.COLOR_BGR2RGB)
            pil_page = Image.fromarray(rgb)
            pil_pages.append(pil_page)

        if pil_pages:
            pil_pages[0].save(
                output_path,
                save_all=True,
                append_images=pil_pages[1:],
                resolution=dpi,
            )
            print(f"\nSaved {len(pil_pages)}-page template PDF: {output_path}")

        return output_path

    def _shelf_pack(self, pieces, usable_w, usable_h, gap):
        """Pack pieces onto pages using shelf algorithm.

        Places pieces left-to-right on horizontal shelves.
        When a piece doesn't fit on the current shelf, starts
        a new shelf below. When a shelf doesn't fit on the
        current page, starts a new page.

        Args:
            pieces: List of piece dicts sorted by height (tallest first)
            usable_w: Usable width in pixels
            usable_h: Usable height in pixels
            gap: Gap between pieces in pixels

        Returns:
            List of pages, each a list of (piece_dict, x, y) tuples
        """
        pages = []
        current_page = []
        shelf_y = 0
        shelf_h = 0
        cursor_x = 0

        for pr in pieces:
            w, h = pr['width'], pr['height']

            # Try to place on current shelf
            if cursor_x + w <= usable_w and shelf_y + h <= usable_h:
                current_page.append((pr, cursor_x, shelf_y))
                cursor_x += w + gap
                shelf_h = max(shelf_h, h)
                continue

            # Try new shelf on current page
            new_shelf_y = shelf_y + shelf_h + gap
            if new_shelf_y + h <= usable_h:
                shelf_y = new_shelf_y
                shelf_h = h
                cursor_x = w + gap
                current_page.append((pr, 0, shelf_y))
                continue

            # Need a new page
            if current_page:
                pages.append(current_page)
            current_page = [(pr, 0, 0)]
            cursor_x = w + gap
            shelf_y = 0
            shelf_h = h

        if current_page:
            pages.append(current_page)

        return pages

    def _render_template_page(self, page_pieces, page_w, page_h,
                              margin, line_width, gap,
                              page_num, total_pages,
                              page_width_in, page_height_in):
        """Render a single page of packed piece templates.

        Each piece is drawn with its outline at came width,
        numbered prominently, and labeled with color group
        if available.

        Args:
            page_pieces: List of (piece_dict, x, y) tuples
            page_w, page_h: Page size in pixels
            margin: Margin in pixels
            line_width: Came line width in pixels
            gap: Gap between pieces in pixels
            page_num: Current page number
            total_pages: Total number of pages
            page_width_in, page_height_in: Page size in inches

        Returns:
            Page image as numpy array
        """
        page = np.ones((page_h, page_w, 3), dtype=np.uint8) * 255

        for pr, px, py in page_pieces:
            piece = pr['piece']
            bx, by, bw, bh = piece.bounding_box

            # Offset contour to packed position
            offset_x = margin + px - bx + line_width // 2
            offset_y = margin + py - by + line_width // 2
            shifted = piece.contour.copy()
            shifted[:, :, 0] += offset_x
            shifted[:, :, 1] += offset_y

            # Draw piece outline
            cv2.drawContours(
                page, [shifted], -1,
                (0, 0, 0), line_width, lineType=cv2.LINE_AA
            )

            # Calculate centroid in page coordinates
            cx = piece.centroid[0] - bx + margin + px + line_width // 2
            cy = piece.centroid[1] - by + margin + py + line_width // 2

            # Piece number — prominent
            font_scale = max(
                0.5, min(1.0, np.sqrt(piece.area_sq_inches) * 0.35)
            )
            text = str(piece.id)
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
            )

            cv2.rectangle(
                page,
                (cx - tw // 2 - 3, cy - th // 2 - 3),
                (cx + tw // 2 + 3, cy + th // 2 + 3),
                (255, 255, 255), -1
            )
            cv2.putText(
                page, text,
                (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0), 2
            )

            # Color group name below number
            color_name = getattr(piece, 'color_group_name', None)
            if color_name:
                (cw, ch), _ = cv2.getTextSize(
                    color_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1
                )
                cv2.putText(
                    page, color_name,
                    (cx - cw // 2, cy + th // 2 + ch + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (80, 80, 80), 1
                )

        # Footer
        footer_y = page_h - margin // 2
        technique_label = (
            "copper foil" if self.technique == 'foil' else "lead came"
        )
        footer = (
            f"Page {page_num}/{total_pages}  |  "
            f"{technique_label} {self.came_width:.4g}\"  |  "
            f"Print at {self.scale_dpi:.0f} DPI for actual size"
        )
        cv2.putText(
            page, footer,
            (margin, footer_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (100, 100, 100), 1
        )

        return page

    def _render_oversized_piece(self, piece, page_w, page_h,
                                margin, usable_w, usable_h,
                                line_width, overlap,
                                page_width_in, page_height_in,
                                current_page_num, total_pages):
        """Render an oversized piece across multiple pages with overlap marks.

        Args:
            piece: Piece object
            page_w, page_h: Page size in pixels
            margin: Margin in pixels
            usable_w, usable_h: Usable area in pixels
            line_width: Came line width in pixels
            overlap: Overlap region in pixels
            page_width_in, page_height_in: Page size in inches
            current_page_num: Starting page number
            total_pages: Total pages in document

        Returns:
            List of (page_image, label_string) tuples
        """
        bx, by, bw, bh = piece.bounding_box

        step_x = usable_w - overlap
        step_y = usable_h - overlap
        cols = max(1, int(np.ceil(bw / step_x)))
        rows = max(1, int(np.ceil(bh / step_y)))

        pages = []
        mark_color = (0, 0, 200)  # red in BGR
        mark_len = int(0.25 * self.scale_dpi)

        for row_idx in range(rows):
            for col_idx in range(cols):
                page = np.ones((page_h, page_w, 3), dtype=np.uint8) * 255

                # Source region offset
                src_x = bx + col_idx * step_x
                src_y = by + row_idx * step_y
                offset_x = margin - src_x
                offset_y = margin - src_y

                shifted = piece.contour.copy()
                shifted[:, :, 0] += offset_x
                shifted[:, :, 1] += offset_y

                cv2.drawContours(
                    page, [shifted], -1,
                    (0, 0, 0), line_width, lineType=cv2.LINE_AA
                )

                # Overlap marks — right edge
                if col_idx < cols - 1:
                    mark_x = margin + usable_w - overlap
                    for my in range(margin, margin + usable_h,
                                    int(1.0 * self.scale_dpi)):
                        cv2.line(page,
                                 (mark_x, my - mark_len),
                                 (mark_x, my + mark_len),
                                 mark_color, 2)
                        cv2.putText(page, ">>>",
                                    (mark_x + 5, my + 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                                    mark_color, 1)

                # Overlap marks — left edge
                if col_idx > 0:
                    mark_x = margin + overlap
                    for my in range(margin, margin + usable_h,
                                    int(1.0 * self.scale_dpi)):
                        cv2.line(page,
                                 (mark_x, my - mark_len),
                                 (mark_x, my + mark_len),
                                 mark_color, 2)
                        cv2.putText(page, "<<<",
                                    (mark_x - 30, my + 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                                    mark_color, 1)

                # Overlap marks — bottom edge
                if row_idx < rows - 1:
                    mark_y = margin + usable_h - overlap
                    for mx in range(margin, margin + usable_w,
                                    int(1.0 * self.scale_dpi)):
                        cv2.line(page,
                                 (mx - mark_len, mark_y),
                                 (mx + mark_len, mark_y),
                                 mark_color, 2)

                # Overlap marks — top edge
                if row_idx > 0:
                    mark_y = margin + overlap
                    for mx in range(margin, margin + usable_w,
                                    int(1.0 * self.scale_dpi)):
                        cv2.line(page,
                                 (mx - mark_len, mark_y),
                                 (mx + mark_len, mark_y),
                                 mark_color, 2)

                # Piece number if centroid is on this page
                cx = piece.centroid[0] + offset_x
                cy = piece.centroid[1] + offset_y
                if (margin < cx < margin + usable_w and
                        margin < cy < margin + usable_h):
                    font_scale = max(
                        0.5, min(1.0, np.sqrt(piece.area_sq_inches) * 0.35)
                    )
                    text = str(piece.id)
                    (tw, th), _ = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
                    )
                    cv2.rectangle(
                        page,
                        (cx - tw // 2 - 3, cy - th // 2 - 3),
                        (cx + tw // 2 + 3, cy + th // 2 + 3),
                        (255, 255, 255), -1
                    )
                    cv2.putText(
                        page, text,
                        (cx - tw // 2, cy + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 0, 0), 2
                    )

                # Footer
                label = (f"section ({col_idx + 1},{row_idx + 1}) "
                         f"of ({cols},{rows})")
                page_n = current_page_num + row_idx * cols + col_idx + 1
                footer_y = page_h - margin // 2
                footer = (
                    f"Page {page_n}/{total_pages}  |  "
                    f"Piece {piece.id} — {label}  |  "
                    f"Overlap: {overlap / self.scale_dpi:.1f}\""
                )
                cv2.putText(
                    page, footer,
                    (margin, footer_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (100, 100, 100), 1
                )

                pages.append((page, label))

        return pages

    def _save_with_dpi(self, image, path):
        """Save an image with DPI metadata.

        Args:
            image: numpy array (BGR)
            path: Output file path
        """
        try:
            from PIL import Image
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            dpi = int(round(self.scale_dpi))
            pil_image.save(path, dpi=(dpi, dpi))
        except ImportError:
            cv2.imwrite(path, image)


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

        # Frame/came requirements
        if hasattr(self, 'frame_info'):
            lines.append("-" * 40)
            lines.append("FRAME & CAME REQUIREMENTS")
            lines.append("-" * 40)
            f = self.frame_info

            lines.append(f"  Panel shape: {f['panel_shape']}")
            lines.append(
                f"  Zinc outer dimension: "
                f"{f['outer_width']:.1f}\" x {f['outer_height']:.1f}\""
            )
            lines.append(
                f"  Pattern (glass) dimension: "
                f"{f['pattern_width']:.1f}\" x "
                f"{f['pattern_height']:.1f}\""
            )
            lines.append(
                f"  Zinc U-channel: {f['zinc_face_width']}\" face, "
                f"{f['zinc_channel_depth']}\" channel depth"
            )
            lines.append("")

            lines.append("  Zinc outer frame (U-channel):")
            if f.get('zinc_pieces'):
                lines.append(
                    f"    {'Piece':<10} {'Length':>10} {'Cut':>12}"
                )
                lines.append(f"    {'-' * 34}")
                for zp in f['zinc_pieces']:
                    length = zp['length_inches']
                    lines.append(
                        f"    {zp['label']:<10} "
                        f"{length:>9.1f}\" "
                        f"{'miter 45°':>12}"
                    )
                lines.append(f"    {'-' * 34}")
                total_zinc = sum(
                    zp['length_inches'] for zp in f['zinc_pieces']
                )
                lines.append(
                    f"    {'Total':<10} {total_zinc:>9.1f}\""
                )
                waste_zinc = total_zinc * 1.10
                lines.append(
                    f"    {'+ 10% waste':<10} "
                    f"{waste_zinc:>9.1f}\" "
                    f"({waste_zinc / 12:.1f} ft)"
                )
            else:
                lines.append(
                    f"    Non-rectangular — total perimeter: "
                    f"{f['zinc_perimeter']}\" "
                    f"({f['zinc_perimeter'] / 12:.1f} ft)"
                )

            lines.append("")
            lines.append("  Interior lead came (H-channel):")
            lines.append(
                f"    Total length: "
                f"{f['interior_came_inches']}\" "
                f"({f['interior_came_feet']} ft)"
            )
            lines.append(
                f"    + 10% waste: "
                f"{f['interior_came_with_waste']} ft"
            )
            lines.append("")
            lines.append("  Combined total:")
            lines.append(
                f"    All came/zinc: "
                f"{f['total_came_inches']}\" "
                f"({f['total_came_feet']} ft)"
            )
            lines.append(
                f"    + 10% waste: "
                f"{f['total_with_waste_feet']} ft"
            )
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

        # --- Add colored figure if color analysis was done ---
        color_fig = self._build_bokeh_colored_figure(
            source, img_width, img_height
        )

        if color_fig:
            # Stack the two figures vertically on the left
            figures = column(p, color_fig)
            layout = row(figures, right_panel)
        else:
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

    def _build_bokeh_colored_figure(self, source, img_width, img_height):
        """Build a second Bokeh figure showing pieces filled with sampled colors.

        Args:
            source: ColumnDataSource with piece data
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            Bokeh figure object, or None if no color analysis done
        """
        if not hasattr(self, 'color_groups') or not self.color_groups:
            return None

        from bokeh.plotting import figure
        from bokeh.models import ColumnDataSource, HoverTool

        # Build color data source
        color_data = defaultdict(list)

        for piece in self.pieces:
            # Simplify contour for performance
            epsilon = 0.01 * cv2.arcLength(piece.contour, True)
            simplified = cv2.approxPolyDP(piece.contour, epsilon, True)
            points = simplified.reshape(-1, 2)

            xs = points[:, 0].tolist()
            ys = (img_height - points[:, 1]).tolist()
            xs.append(xs[0])
            ys.append(ys[0])

            color_data['xs'].append(xs)
            color_data['ys'].append(ys)
            color_data['piece_id'].append(piece.id)
            color_data['area'].append(round(piece.area_sq_inches, 3))
            color_data['cx'].append(piece.centroid[0])
            color_data['cy'].append(img_height - piece.centroid[1])

            # Get color info
            bgr = getattr(piece, 'color_group_center', (200, 200, 200))
            boosted = self._boost_color(bgr)
            r, g, b = int(boosted[2]), int(boosted[1]), int(boosted[0])
            color_data['fill_color'].append(f"rgb({r},{g},{b})")

            name = getattr(piece, 'color_group_name', 'Unassigned')
            color_data['color_name'].append(name)
            color_data['color_rgb'].append(f"({r},{g},{b})")

            # Sampled color (actual, not cluster center)
            sampled = getattr(piece, 'sampled_color', (128, 128, 128))
            sr, sg, sb = int(sampled[2]), int(sampled[1]), int(sampled[0])
            color_data['sampled_rgb'].append(f"({sr},{sg},{sb})")

        color_source = ColumnDataSource(data=dict(color_data))

        p = figure(
            title="Color Analysis — Glass Color Groups",
            width=800,
            height=int(800 * img_height / img_width),
            x_range=(0, img_width),
            y_range=(0, img_height),
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
            match_aspect=True,
        )

        # White background
        p.rect(
            x=img_width / 2, y=img_height / 2,
            width=img_width, height=img_height,
            fill_color="white", line_color=None
        )

        # Colored piece polygons
        patches = p.patches(
            'xs', 'ys',
            source=color_source,
            fill_color='fill_color',
            fill_alpha=0.85,
            line_color="black",
            line_width=2,
        )

        # Piece number labels
        label_source = ColumnDataSource(data=dict(
            x=color_data['cx'],
            y=color_data['cy'],
            text=[str(pid) for pid in color_data['piece_id']],
        ))
        p.text(
            'x', 'y', 'text',
            source=label_source,
            text_font_size="9pt",
            text_align="center",
            text_baseline="middle",
            text_color="#000000",
            text_font_style="bold",
        )

        # Hover tooltip with color info
        hover = HoverTool(
            renderers=[patches],
            tooltips="""
            <div style="background: white; padding: 8px; border-radius: 4px;
                        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
                <div style="font-weight: bold; font-size: 14px;
                            margin-bottom: 4px;">Piece #@piece_id</div>
                <table>
                    <tr><td>Color Group:</td>
                        <td><b>@color_name</b></td></tr>
                    <tr><td>Group RGB:</td>
                        <td><b>@color_rgb</b></td></tr>
                    <tr><td>Sampled RGB:</td>
                        <td><b>@sampled_rgb</b></td></tr>
                    <tr><td>Area:</td>
                        <td><b>@area sq in</b></td></tr>
                </table>
                <div style="margin-top: 4px; width: 40px; height: 20px;
                            background: rgb@color_rgb; border: 1px solid #333;">
                </div>
            </div>
            """,
        )
        p.add_tools(hover)
        p.axis.visible = False
        p.grid.visible = False

        return p

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

    # In the Frame group or Scale group:
    frame_group = parser.add_argument_group('Frame')
    frame_group.add_argument(
        '--zinc-channel-depth', type=float, default=0.25,
        help='Zinc U-channel depth in inches (default: 0.25)'
    )
    frame_group.add_argument(
        '--zinc-face-width', type=float, default=0.5,
        help='Zinc U-channel total face width in inches (default: 0.5)'
    )

    frame_group.add_argument(
        '--came-width', type=float, default=None,
        help='Width of material between glass pieces in inches. '
             'For lead: heart width, typically 1/16 (0.0625). '
             'For copper foil: 2x foil thickness, typically 1/32 (0.03125). '
             'This is the strip the pattern scissors remove.'
    )
    frame_group.add_argument(
        '--technique', type=str, default='lead',
        choices=['lead', 'foil'],
        help='Construction technique (default: lead). '
             'Sets default came width if --came-width not specified.'
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
    # In Output group:
    out_group.add_argument(
        '--page-width', type=float, default=8.5,
        help='Template page width in inches (default: 8.5)'
    )
    out_group.add_argument(
        '--page-height', type=float, default=11.0,
        help='Template page height in inches (default: 11.0)'
    )

    color_group = parser.add_argument_group('Color Analysis')
    color_group.add_argument(
        '--source-image', type=str, default=None,
        help='Reference image for color sampling (photo, painting, etc.)'
    )
    color_group.add_argument(
        '--num-colors', type=int, default=6,
        help='Number of color groups for clustering (default: 6)'
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
        zinc_channel_depth=args.zinc_channel_depth,
        zinc_face_width=args.zinc_face_width,
        came_width=args.came_width,
        technique=args.technique,
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

    print("\n--- Calculating Frame Requirements ---")
    analyzer.calculate_frame()

    if args.source_image:
        print("\n--- Analyzing Colors ---")
        analyzer.analyze_colors_from_source(
            args.source_image,
            num_colors=args.num_colors
        )
        analyzer.generate_colored_pattern(prefix=args.prefix)
        analyzer.generate_color_report(prefix=args.prefix)

    print("\n--- Generating Outputs ---")
    analyzer.generate_annotated_image(prefix=args.prefix)
    analyzer.generate_qa_overlay(prefix=args.prefix)
    analyzer.generate_report(prefix=args.prefix)
    analyzer.generate_template(prefix=args.prefix)
    analyzer.generate_packed_templates(
        prefix=args.prefix,
        page_width=args.page_width,
        page_height=args.page_height,
    )

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
    print(f"  {args.prefix}_template.png     — full numbered template")
    print(f"  {args.prefix}_templates.pdf     — packed templates for printing")
    if not args.no_bokeh:
        print(f"  {args.prefix}_interactive.html — interactive visualization")


if __name__ == "__main__":
    main()
