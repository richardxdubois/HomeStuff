import cv2
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import json


@dataclass
class Piece:
    id: int
    contour: np.ndarray
    area: float
    area_sq_inches: float
    centroid: tuple
    bounding_box: tuple  # x, y, w, h
    num_vertices: int
    min_angle: float
    min_width_inches: float
    warnings: list = field(default_factory=list)


class PatternAnalyzer:
    def __init__(self, image_path, scale_dpi=150, output_dir="."):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.scale_dpi = scale_dpi
        self.pieces = []
        self.binary = None

        # Set up output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loaded image: {self.image.shape[1]}x{self.image.shape[0]} pixels")
        print(f"At {scale_dpi} DPI that's approximately "
              f"{self.image.shape[1] / scale_dpi:.1f}\" x {self.image.shape[0] / scale_dpi:.1f}\"")
        print(f"Output directory: {self.output_dir.resolve()}")

        # Debug: show pixel value range
        print(f"Pixel value range: {self.gray.min()} - {self.gray.max()}")
        print(f"Mean pixel value: {self.gray.mean():.1f}")

    def _output_path(self, filename):
        """Helper to build output paths."""
        return str(self.output_dir / filename)

    def preprocess(self, line_threshold=128, close_kernel_size=3, dilate_iterations=0):
        """
        Clean up the pattern image for contour detection.

        Goal: produce a binary image where:
        - Lead lines = BLACK (0)
        - Glass regions = WHITE (255)

        In the source image, lines are dark (low pixel values) and
        background is light (high pixel values).

        cv2.threshold with THRESH_BINARY:
        - pixels ABOVE threshold -> 255 (white) = glass regions ✓
        - pixels BELOW threshold -> 0 (black) = lead lines ✓
        """
        # First, let's look at the histogram to choose a good threshold
        hist = cv2.calcHist([self.gray], [0], None, [256], [0, 256])

        # Find the two peaks (should be one near 0 for lines, one near 255 for background)
        dark_peak = np.argmax(hist[:128])
        light_peak = np.argmax(hist[128:]) + 128
        print(f"Dark peak at: {dark_peak}, Light peak at: {light_peak}")

        # Use Otsu's method to find optimal threshold automatically
        otsu_thresh, binary_otsu = cv2.threshold(
            self.gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        print(f"Otsu's suggested threshold: {otsu_thresh}")

        # Use the provided threshold, but apply it correctly
        # THRESH_BINARY: pixel > threshold -> 255, else -> 0
        # So dark lines (low values) become 0 (black) ✓
        # And light background (high values) become 255 (white) ✓
        _, binary = cv2.threshold(self.gray, line_threshold, 255, cv2.THRESH_BINARY)

        # Save pre-morphology for debugging
        cv2.imwrite(self._output_path("debug_00_threshold_only.png"), binary)
        print(f"Saved debug_00_threshold_only.png")

        # Morphological closing on the LINE network to seal small gaps
        # We need to operate on the lines (black), so invert first
        if close_kernel_size > 0:
            lines = cv2.bitwise_not(binary)  # lines become white
            kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
            lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, kernel)

            # Optional: dilate lines to make them thicker / seal gaps
            if dilate_iterations > 0:
                lines = cv2.dilate(lines, kernel, iterations=dilate_iterations)

            binary = cv2.bitwise_not(lines)  # back to: lines=black, glass=white

        self.binary = binary

        # Save preprocessed image for inspection
        debug_path = self._output_path("debug_01_binary.png")
        cv2.imwrite(debug_path, binary)

        # Also save a version with lines highlighted in red for easy checking
        debug_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        debug_color[binary == 0] = [0, 0, 255]  # lines in red
        cv2.imwrite(self._output_path("debug_02_lines_highlighted.png"), debug_color)

        # Count approximate regions to sanity-check
        num_labels, labels = cv2.connectedComponents(binary)
        print(f"Connected white regions found: {num_labels - 1}")  # -1 for background
        print(f"Saved debug images to {self.output_dir}")

        return binary

    def detect_pieces(self, min_area_pixels=200, max_area_ratio=0.5):
        """
        Find closed regions between lead lines.
        Each closed white region in the binary image = one piece of glass.
        """
        if self.binary is None:
            self.preprocess()

        # Find contours on the binary image
        # The white regions (255) are the glass pieces
        # findContours finds boundaries of white regions
        contours, hierarchy = cv2.findContours(
            self.binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )

        if hierarchy is None:
            print("No contours found!")
            return []

        total_area = self.image.shape[0] * self.image.shape[1]
        max_area = total_area * max_area_ratio

        print(f"Found {len(contours)} raw contours")

        # Debug: print area distribution
        areas = [cv2.contourArea(c) for c in contours]
        areas_nonzero = [a for a in areas if a > 0]
        if areas_nonzero:
            print(f"Contour area range: {min(areas_nonzero):.0f} - {max(areas_nonzero):.0f} pixels")

        piece_id = 0
        rejected_small = 0
        rejected_large = 0

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            # Skip tiny regions (noise, JPG artifacts)
            if area < min_area_pixels:
                rejected_small += 1
                continue

            # Skip the outer boundary / huge regions
            if area > max_area:
                rejected_large += 1
                continue

            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Approximate polygon for angle analysis
            epsilon = 0.015 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Calculate minimum angle
            min_angle = self._min_interior_angle(approx)

            # Calculate minimum width using distance transform
            min_width_inches = self._min_width(contour)

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
                min_width_inches=min_width_inches
            )

            self.pieces.append(piece)

        print(f"\nDetected {len(self.pieces)} pieces")
        print(f"Rejected: {rejected_small} too small, {rejected_large} too large")

        return self.pieces

    def _min_interior_angle(self, approx_contour):
        """Find the sharpest interior angle in the piece."""
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

    def _min_width(self, contour):
        """
        Estimate the minimum width of a piece using distance transform.
        """
        mask = np.zeros(self.gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

        if dist.max() == 0:
            return 0.0

        distances = dist[dist > 0]

        if len(distances) == 0:
            return 0.0

        narrow_width = np.percentile(distances, 5) * 2

        return narrow_width / self.scale_dpi

    def run_qa(self):
        """Run quality assurance checks on all pieces."""
        print("\nRunning QA checks...")

        for piece in self.pieces:
            if piece.area_sq_inches < 0.25:
                piece.warnings.append(
                    f"TINY: Only {piece.area_sq_inches:.2f} sq in — very difficult to cut"
                )
            elif piece.area_sq_inches < 0.5:
                piece.warnings.append(
                    f"SMALL: {piece.area_sq_inches:.2f} sq in — challenging to cut"
                )

            if piece.min_width_inches < 0.1875:
                piece.warnings.append(
                    f"VERY NARROW: Min width ~{piece.min_width_inches:.3f}\" — will likely break"
                )
            elif piece.min_width_inches < 0.25:
                piece.warnings.append(
                    f"NARROW: Min width ~{piece.min_width_inches:.3f}\" — fragile"
                )

            if piece.min_angle < 20:
                piece.warnings.append(
                    f"VERY SHARP ANGLE: {piece.min_angle:.0f}° — nearly impossible to cut"
                )
            elif piece.min_angle < 35:
                piece.warnings.append(
                    f"SHARP ANGLE: {piece.min_angle:.0f}° — difficult, may need grinding"
                )

            if piece.num_vertices > 12:
                piece.warnings.append(
                    f"COMPLEX: {piece.num_vertices} vertices — consider simplifying"
                )

            x, y, w, h = piece.bounding_box
            if w > 0 and h > 0:
                aspect = max(w, h) / min(w, h)
                if aspect > 6:
                    piece.warnings.append(
                        f"VERY ELONGATED: {aspect:.1f}:1 aspect ratio — fragile"
                    )

        warning_count = sum(len(p.warnings) for p in self.pieces)
        pieces_with_warnings = sum(1 for p in self.pieces if p.warnings)
        print(f"Found {warning_count} warnings across {pieces_with_warnings} pieces")

    def generate_annotated_image(self, prefix="pattern"):
        """Create output image with piece numbers and QA highlights."""
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
        }

        for piece in self.pieces:
            cx, cy = piece.centroid

            if piece.warnings:
                color = (0, 128, 255)
                for key, c in WARNING_COLORS.items():
                    if any(key in w for w in piece.warnings):
                        if c == (0, 0, 255):
                            color = c
                            break
                        color = c

                overlay = annotated.copy()
                cv2.drawContours(overlay, [piece.contour], -1, color, -1)
                cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0, annotated)
                cv2.drawContours(annotated, [piece.contour], -1, color, 2)

            font_scale = max(0.25, min(0.5, piece.area_sq_inches * 0.5))

            text = str(piece.id)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(annotated,
                          (cx - tw // 2 - 2, cy - th // 2 - 2),
                          (cx + tw // 2 + 2, cy + th // 2 + 2),
                          (255, 255, 255), -1)
            cv2.putText(annotated, text,
                        (cx - tw // 2, cy + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)

        cv2.imwrite(output_path, annotated)
        print(f"Saved annotated image: {output_path}")
        return output_path

    def generate_qa_overlay(self, prefix="pattern"):
        """Separate image showing ONLY the QA issues."""
        output_path = self._output_path(f"{prefix}_qa.png")

        qa_image = cv2.addWeighted(self.image, 0.4,
                                   np.ones_like(self.image) * 255, 0.6, 0)

        for piece in self.pieces:
            if not piece.warnings:
                continue

            cx, cy = piece.centroid

            mask = np.zeros(self.gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [piece.contour], -1, 255, -1)
            qa_image[mask > 0] = (qa_image[mask > 0] * 0.5 +
                                  np.array([0, 0, 200]) * 0.5).astype(np.uint8)

            short_warning = piece.warnings[0].split(":")[0]
            cv2.putText(qa_image, f"#{piece.id}", (cx - 10, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
            cv2.putText(qa_image, short_warning, (cx - 20, cy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 150), 1)

        cv2.imwrite(output_path, qa_image)
        print(f"Saved QA overlay: {output_path}")

    def generate_report(self, prefix="pattern"):
        """Generate a complete text report."""
        output_path = self._output_path(f"{prefix}_report.txt")

        lines = []
        lines.append("=" * 60)
        lines.append("STAINED GLASS PATTERN ANALYSIS REPORT")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Image size: {self.image.shape[1]}x{self.image.shape[0]} pixels")
        lines.append(f"Scale: {self.scale_dpi} DPI")
        lines.append(f"Estimated panel size: "
                     f"{self.image.shape[1] / self.scale_dpi:.1f}\" x "
                     f"{self.image.shape[0] / self.scale_dpi:.1f}\"")
        lines.append(f"Total pieces detected: {len(self.pieces)}")
        lines.append("")

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

        lines.append("-" * 60)
        lines.append("ALL PIECES")
        lines.append("-" * 60)
        lines.append(f"{'ID':>4} {'Area(sq in)':>11} {'Min Width':>10} "
                     f"{'Min Angle':>10} {'Vertices':>8} {'Warnings':>8}")
        lines.append("-" * 60)

        for p in sorted(self.pieces, key=lambda x: x.id):
            warn_flag = "***" if p.warnings else ""
            lines.append(f"{p.id:>4} {p.area_sq_inches:>11.2f} "
                         f"{p.min_width_inches:>9.3f}\" "
                         f"{p.min_angle:>9.0f}° "
                         f"{p.num_vertices:>8} {warn_flag:>8}")

        lines.append("")
        lines.append("-" * 40)
        lines.append("QA WARNINGS")
        lines.append("-" * 40)

        warning_pieces = [p for p in self.pieces if p.warnings]
        if warning_pieces:
            for p in sorted(warning_pieces, key=lambda x: x.id):
                for w in p.warnings:
                    lines.append(f"  Piece {p.id:>3}: {w}")
        else:
            lines.append("  No warnings — pattern looks good!")

        lines.append("")
        lines.append("=" * 60)
        lines.append("END OF REPORT")
        lines.append("=" * 60)

        report_text = "\n".join(lines)

        with open(output_path, 'w') as f:
            f.write(report_text)

        print(f"\nSaved report: {output_path}")
        print("\n" + report_text)

        return report_text


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Stained Glass Pattern Analyzer')
    parser.add_argument('image', help='Path to pattern image')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Scale DPI (default: 150)')
    parser.add_argument('--threshold', type=int, default=128,
                        help='Line detection threshold 0-255 (default: 128)')
    parser.add_argument('--close-kernel', type=int, default=3,
                        help='Morphological closing kernel size (default: 3)')
    parser.add_argument('--dilate', type=int, default=1,
                        help='Line dilation iterations (default: 1)')
    parser.add_argument('--min-area', type=int, default=200,
                        help='Minimum piece area in pixels (default: 200)')
    parser.add_argument('--prefix', type=str, default='pattern',
                        help='Output file prefix (default: pattern)')
    parser.add_argument('--output-dir', '-o', type=str, default='./output',
                        help='Output directory (default: ./output)')

    args = parser.parse_args()

    print("Stained Glass Pattern Analyzer")
    print("=" * 40)

    analyzer = PatternAnalyzer(
        args.image,
        scale_dpi=args.dpi,
        output_dir=args.output_dir
    )

    print("\n--- Preprocessing ---")
    analyzer.preprocess(
        line_threshold=args.threshold,
        close_kernel_size=args.close_kernel,
        dilate_iterations=args.dilate
    )

    print("\n--- Detecting Pieces ---")
    analyzer.detect_pieces(min_area_pixels=args.min_area)

    print("\n--- Running QA ---")
    analyzer.run_qa()

    print("\n--- Generating Outputs ---")
    analyzer.generate_annotated_image(prefix=args.prefix)
    analyzer.generate_qa_overlay(prefix=args.prefix)
    analyzer.generate_report(prefix=args.prefix)

    print("\n--- Done! ---")
    output_dir = Path(args.output_dir).resolve()
    print(f"All output files in: {output_dir}")
    print(f"  debug_00_threshold_only.png — raw threshold result")
    print(f"  debug_01_binary.png         — after morphological cleanup")
    print(f"  debug_02_lines_highlighted.png — lines shown in red")
    print(f"  {args.prefix}_analyzed.png  — numbered pieces with QA highlights")
    print(f"  {args.prefix}_qa.png        — QA issues only")
    print(f"  {args.prefix}_report.txt    — full text report")


if __name__ == "__main__":
    main()
