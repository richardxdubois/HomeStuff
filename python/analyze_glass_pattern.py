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
    max_width_inches: float
    warnings: list = field(default_factory=list)


class PatternAnalyzer:
    def __init__(self, image_path, scale_dpi=150, output_dir=".",
                 panel_width=None, panel_height=None):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.pieces = []
        self.binary = None
        self.image_path = image_path

        img_h, img_w = self.image.shape[:2]

        # Calculate DPI from panel dimensions if provided
        if panel_width:
            self.scale_dpi = img_w / panel_width
            print(f"Calculated DPI from panel width ({panel_width}\"): {self.scale_dpi:.1f}")
        elif panel_height:
            self.scale_dpi = img_h / panel_height
            print(f"Calculated DPI from panel height ({panel_height}\"): {self.scale_dpi:.1f}")
        else:
            self.scale_dpi = scale_dpi

        # Set up output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loaded image: {img_w}x{img_h} pixels")
        print(f"At {self.scale_dpi:.1f} DPI that's approximately "
              f"{img_w / self.scale_dpi:.1f}\" x {img_h / self.scale_dpi:.1f}\"")
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
        """
        # Histogram analysis
        hist = cv2.calcHist([self.gray], [0], None, [256], [0, 256])
        dark_peak = np.argmax(hist[:128])
        light_peak = np.argmax(hist[128:]) + 128
        print(f"Dark peak at: {dark_peak}, Light peak at: {light_peak}")

        # Otsu's method for reference
        otsu_thresh, _ = cv2.threshold(
            self.gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        print(f"Otsu's suggested threshold: {otsu_thresh}")

        # Apply threshold
        _, binary = cv2.threshold(self.gray, line_threshold, 255, cv2.THRESH_BINARY)

        # Save pre-morphology for debugging
        cv2.imwrite(self._output_path("debug_00_threshold_only.png"), binary)
        print(f"Saved debug_00_threshold_only.png")

        # Morphological operations on the LINE network
        if close_kernel_size > 0:
            lines = cv2.bitwise_not(binary)  # lines become white
            kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
            lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, kernel)

            if dilate_iterations > 0:
                lines = cv2.dilate(lines, kernel, iterations=dilate_iterations)

            binary = cv2.bitwise_not(lines)  # back to: lines=black, glass=white

        self.binary = binary

        # Save debug images
        cv2.imwrite(self._output_path("debug_01_binary.png"), binary)

        debug_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        debug_color[binary == 0] = [0, 0, 255]
        cv2.imwrite(self._output_path("debug_02_lines_highlighted.png"), debug_color)

        # Count regions
        num_labels, labels = cv2.connectedComponents(binary)
        print(f"Connected white regions found: {num_labels - 1}")
        print(f"Saved debug images to {self.output_dir}")

        return binary

    def detect_pieces(self, min_area_pixels=200, max_area_ratio=0.5):
        """Find closed regions between lead lines."""
        if self.binary is None:
            self.preprocess()

        contours, hierarchy = cv2.findContours(
            self.binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )

        if hierarchy is None:
            print("No contours found!")
            return []

        total_area = self.image.shape[0] * self.image.shape[1]
        max_area = total_area * max_area_ratio

        print(f"Found {len(contours)} raw contours")

        areas = [cv2.contourArea(c) for c in contours]
        areas_nonzero = [a for a in areas if a > 0]
        if areas_nonzero:
            print(f"Contour area range: {min(areas_nonzero):.0f} - {max(areas_nonzero):.0f} pixels")

        piece_id = 0
        rejected_small = 0
        rejected_large = 0

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            if area < min_area_pixels:
                rejected_small += 1
                continue

            if area > max_area:
                rejected_large += 1
                continue

            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            x, y, w, h = cv2.boundingRect(contour)

            epsilon = 0.015 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            min_angle = self._min_interior_angle(approx)
            min_width_inches, max_width_inches = self._piece_widths(contour)
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
                min_width_inches=min_width_inches,
                max_width_inches=max_width_inches,
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

    def _piece_widths(self, contour):
        """
        Estimate min and max width of a piece.

        Uses multiple strategies and picks the most reliable:
        1. Distance transform (good for high-res images)
        2. Rotating calipers / min-area rectangle (resolution-independent)
        3. Area / length estimate (robust fallback)
        """
        mask = np.zeros(self.gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # Strategy 1: Min-area rotated rectangle
        # This gives the tightest-fitting rotated rectangle
        # Its short side = min width, long side = max length
        if len(contour) >= 5:
            rect = cv2.minAreaRect(contour)
            rect_w, rect_h = rect[1]  # (width, height) of the rotated rect

            min_dim = min(rect_w, rect_h) / self.scale_dpi
            max_dim = max(rect_w, rect_h) / self.scale_dpi
        else:
            x, y, w, h = cv2.boundingRect(contour)
            min_dim = min(w, h) / self.scale_dpi
            max_dim = max(w, h) / self.scale_dpi

        # Strategy 2: Distance transform for narrowest internal passage
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

        if dist.max() > 0:
            # Max inscribed circle diameter
            max_width_dt = dist.max() * 2 / self.scale_dpi

            # For min width, use skeleton only if we have enough pixels
            interior_pixels = cv2.countNonZero(mask)

            if interior_pixels > 100:
                # Enough resolution for skeleton analysis
                skeleton = np.zeros_like(mask)
                temp = mask.copy()
                kernel = np.ones((3, 3), np.uint8)

                while True:
                    eroded = cv2.erode(temp, kernel)
                    dilated = cv2.dilate(eroded, kernel)
                    diff = cv2.subtract(temp, dilated)
                    skeleton = cv2.bitwise_or(skeleton, diff)
                    temp = eroded.copy()
                    if cv2.countNonZero(temp) == 0:
                        break

                skeleton_distances = dist[skeleton > 0]

                if len(skeleton_distances) > 2:
                    # Use 10th percentile to avoid single-pixel noise
                    min_width_dt = np.percentile(skeleton_distances, 10) * 2 / self.scale_dpi
                else:
                    min_width_dt = min_dim
            else:
                # Too few pixels — fall back to rectangle estimate
                min_width_dt = min_dim

            # Use the more conservative (larger) of the two estimates
            # for min width — the rectangle method is more reliable
            # at low resolution
            min_width = max(min_width_dt, min_dim * 0.8)
            max_width = max(max_width_dt, max_dim)
        else:
            min_width = min_dim
            max_width = max_dim

        return min_width, max_width

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
        lines.append("=" * 70)
        lines.append("STAINED GLASS PATTERN ANALYSIS REPORT")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"Image size: {self.image.shape[1]}x{self.image.shape[0]} pixels")
        lines.append(f"Scale: {self.scale_dpi:.1f} DPI")

        img_h, img_w = self.image.shape[:2]
        lines.append(f"Estimated panel size: "
                     f"{img_w / self.scale_dpi:.1f}\" x "
                     f"{img_h / self.scale_dpi:.1f}\"")
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

        lines.append("-" * 76)
        lines.append("ALL PIECES")
        lines.append("-" * 76)
        lines.append(f"{'ID':>4} {'Area(sq in)':>11} {'MinW':>8} {'MaxW':>8} "
                     f"{'Min Angle':>10} {'Vertices':>8} {'Warnings':>8}")
        lines.append("-" * 76)

        for p in sorted(self.pieces, key=lambda x: x.id):
            warn_flag = "***" if p.warnings else ""
            lines.append(f"{p.id:>4} {p.area_sq_inches:>11.2f} "
                         f"{p.min_width_inches:>7.3f}\" "
                         f"{p.max_width_inches:>7.3f}\" "
                         f"{p.min_angle:>9.0f}° "
                         f"{p.num_vertices:>8} {warn_flag:>8}")

        lines.append("")
        lines.append("-" * 40)
        lines.append("QA WARNINGS")
        lines.append("-" * 40)

        warning_pieces = [p for p in self.pieces if p.warnings]
        if warning_pieces:
            # Group by warning type for summary
            warning_type_counts = defaultdict(int)
            for p in self.pieces:
                for w in p.warnings:
                    wtype = w.split(":")[0]
                    warning_type_counts[wtype] += 1

            lines.append("\n  Warning Summary:")
            for wtype, count in sorted(warning_type_counts.items(),
                                       key=lambda x: -x[1]):
                lines.append(f"    {wtype}: {count} pieces")

            lines.append(f"\n  Details:")
            for p in sorted(warning_pieces, key=lambda x: x.id):
                for w in p.warnings:
                    lines.append(f"    Piece {p.id:>3}: {w}")
        else:
            lines.append("  No warnings — pattern looks good!")

        lines.append("")
        lines.append("=" * 70)
        lines.append("END OF REPORT")
        lines.append("=" * 70)

        report_text = "\n".join(lines)

        with open(output_path, 'w') as f:
            f.write(report_text)

        print(f"\nSaved report: {output_path}")
        print("\n" + report_text)

        return report_text

    def generate_bokeh_visualization(self, prefix="pattern"):
        """
        Generate an interactive Bokeh HTML visualization.
        """
        from bokeh.plotting import figure, save, output_file
        from bokeh.models import (
            ColumnDataSource, HoverTool,
            CustomJS, Div, CheckboxGroup,
        )
        from bokeh.layouts import column, row
        import base64

        output_path = self._output_path(f"{prefix}_interactive.html")
        output_file(output_path, title="Stained Glass Pattern Analysis")

        img_height, img_width = self.image.shape[:2]

        # --- Prepare the background image as RGBA array for image_rgba ---
        # Convert BGR to RGBA
        img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        img_rgba = np.zeros((img_height, img_width, 4), dtype=np.uint8)
        img_rgba[:, :, :3] = img_rgb
        img_rgba[:, :, 3] = 255  # fully opaque

        # Flip vertically for Bokeh (y=0 at bottom)
        img_rgba = np.flipud(img_rgba)

        # Pack RGBA into uint32 for image_rgba
        img_uint32 = np.zeros((img_height, img_width), dtype=np.uint32)
        view = img_uint32.view(dtype=np.uint8).reshape((img_height, img_width, 4))
        view[:, :, 0] = img_rgba[:, :, 0]  # R
        view[:, :, 1] = img_rgba[:, :, 1]  # G
        view[:, :, 2] = img_rgba[:, :, 2]  # B
        view[:, :, 3] = img_rgba[:, :, 3]  # A

        # --- Prepare piece polygon data ---
        xs_all = []
        ys_all = []
        piece_ids = []
        areas = []
        min_widths = []
        max_widths = []
        min_angles = []
        vertices = []
        centroids_x = []
        centroids_y = []
        warning_texts = []
        warning_counts = []
        qa_status = []
        qa_colors = []

        for piece in self.pieces:
            # Use a slightly larger epsilon for smoother polygons at low res
            epsilon = 0.01 * cv2.arcLength(piece.contour, True)
            simplified = cv2.approxPolyDP(piece.contour, epsilon, True)
            points = simplified.reshape(-1, 2)

            # Flip y for Bokeh coordinate system
            xs = points[:, 0].tolist()
            ys = (img_height - points[:, 1]).tolist()

            # Close the polygon
            xs.append(xs[0])
            ys.append(ys[0])

            xs_all.append(xs)
            ys_all.append(ys)

            piece_ids.append(piece.id)
            areas.append(round(piece.area_sq_inches, 3))
            min_widths.append(round(piece.min_width_inches, 3))
            max_widths.append(round(piece.max_width_inches, 3))
            min_angles.append(round(piece.min_angle, 1))
            vertices.append(piece.num_vertices)
            centroids_x.append(piece.centroid[0])
            centroids_y.append(img_height - piece.centroid[1])

            warn_text = "; ".join(piece.warnings) if piece.warnings else "None"
            warning_texts.append(warn_text)
            warning_counts.append(len(piece.warnings))

            if not piece.warnings:
                qa_status.append("OK")
                qa_colors.append("rgba(46, 204, 113, 0.0)")  # transparent when OK
            elif any("VERY" in w or "TINY" in w for w in piece.warnings):
                qa_status.append("Critical")
                qa_colors.append("rgba(231, 76, 60, 0.35)")  # semi-transparent red
            else:
                qa_status.append("Warning")
                qa_colors.append("rgba(243, 156, 18, 0.35)")  # semi-transparent orange

        source = ColumnDataSource(data=dict(
            xs=xs_all,
            ys=ys_all,
            piece_id=piece_ids,
            area=areas,
            min_width=min_widths,
            max_width=max_widths,
            min_angle=min_angles,
            vertices=vertices,
            cx=centroids_x,
            cy=centroids_y,
            warnings=warning_texts,
            warning_count=warning_counts,
            qa_status=qa_status,
            fill_color=qa_colors,
        ))

        source_full = ColumnDataSource(data=dict(source.data))

        # --- Main figure ---
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

        # Background image using image_rgba (much more reliable)
        p.image_rgba(
            image=[img_uint32],
            x=0, y=0,
            dw=img_width, dh=img_height,
        )

        # Draw piece polygons — transparent fill by default,
        # only warning pieces get colored
        patches = p.patches(
            'xs', 'ys',
            source=source,
            fill_color='fill_color',
            fill_alpha=1.0,  # alpha is baked into the rgba color
            line_color="rgba(0, 0, 0, 0)",  # no outline by default
            line_width=0,

            # On selection (click), highlight the piece
            selection_fill_color="rgba(52, 152, 219, 0.4)",
            selection_line_color="#2980b9",
            selection_line_width=3,

            # Non-selected pieces stay as they are
            nonselection_fill_alpha=1.0,
            nonselection_line_alpha=0,
        )

        # Piece number labels at centroids
        label_source = ColumnDataSource(data=dict(
            x=centroids_x,
            y=centroids_y,
            text=[str(pid) for pid in piece_ids],
        ))

        p.text(
            'x', 'y', 'text',
            source=label_source,
            text_font_size="7pt",
            text_align="center",
            text_baseline="middle",
            text_color="#333333",
            text_font_style="bold",
        )

        # Hover tool
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

        # --- Summary panel ---
        total = len(self.pieces)
        ok_count = sum(1 for s in qa_status if s == "OK")
        warn_count = sum(1 for s in qa_status if s == "Warning")
        crit_count = sum(1 for s in qa_status if s == "Critical")

        summary_html = f"""
        <div style="font-family: Arial, sans-serif; padding: 15px; 
                    background: #f8f9fa; border-radius: 8px; width: 320px;">
            <h2 style="margin-top: 0;">Pattern Summary</h2>
            <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                <tr><td><b>Total Pieces:</b></td><td>{total}</td></tr>
                <tr><td><b>Image:</b></td>
                    <td>{img_width}x{img_height}px @ {self.scale_dpi:.1f} DPI</td></tr>
                <tr><td><b>Panel Size:</b></td>
                    <td>{img_width / self.scale_dpi:.1f}" x 
                        {img_height / self.scale_dpi:.1f}"</td></tr>
                <tr><td><b>Total Glass:</b></td>
                    <td>{sum(areas):.0f} sq in</td></tr>
            </table>

            <h3>QA Status</h3>
            <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                <tr>
                    <td><span style="color: #2ecc71; font-size: 1.5em;">●</span> OK:</td>
                    <td><b>{ok_count}</b> ({100 * ok_count / max(total, 1):.0f}%)</td>
                </tr>
                <tr>
                    <td><span style="color: #f39c12; font-size: 1.5em;">●</span> Warning:</td>
                    <td><b>{warn_count}</b> ({100 * warn_count / max(total, 1):.0f}%)</td>
                </tr>
                <tr>
                    <td><span style="color: #e74c3c; font-size: 1.5em;">●</span> Critical:</td>
                    <td><b>{crit_count}</b> ({100 * crit_count / max(total, 1):.0f}%)</td>
                </tr>
            </table>

            <h3>Size Range</h3>
            <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                <tr><td>Smallest:</td><td>{min(areas):.2f} sq in</td></tr>
                <tr><td>Largest:</td><td>{max(areas):.2f} sq in</td></tr>
                <tr><td>Average:</td><td>{np.mean(areas):.2f} sq in</td></tr>
            </table>

            <h3 style="margin-top: 15px;">Instructions</h3>
            <ul style="font-size: 12px; color: #555;">
                <li><b>Hover</b> over a piece for details</li>
                <li><b>Click</b> a piece to highlight it</li>
                <li><b>Scroll</b> to zoom in/out</li>
                <li><b>Drag</b> to pan</li>
                <li>Use filters below to show/hide by status</li>
            </ul>
        </div>
        """
        summary_div = Div(text=summary_html)

        # --- Detail panel ---
        detail_div = Div(text="""
        <div style="font-family: Arial, sans-serif; padding: 15px;
                    background: #fff3cd; border-radius: 8px; width: 320px;">
            <h3 style="margin-top: 0;">Piece Detail</h3>
            <p><i>Click a piece on the pattern to see details here</i></p>
        </div>
        """, width=340)

        tap_callback = CustomJS(args=dict(source=source, detail_div=detail_div), code="""
            const indices = source.selected.indices;
            if (indices.length === 0) {
                detail_div.text = `
                <div style="font-family: Arial, sans-serif; padding: 15px;
                            background: #fff3cd; border-radius: 8px; width: 320px;">
                    <h3 style="margin-top: 0;">Piece Detail</h3>
                    <p><i>Click a piece on the pattern to see details here</i></p>
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
                "<p style='color: #2ecc71;'><b>✓ No issues found</b></p>" :
                warnings.split("; ").map(w => 
                    "<p style='color: #c0392b; margin: 2px 0;'>⚠ " + w + "</p>"
                ).join("");

            detail_div.text = `
            <div style="font-family: Arial, sans-serif; padding: 15px;
                        background: ${bg_color}; border-radius: 8px; width: 320px;">
                <h3 style="margin-top: 0;">
                    Piece #${data.piece_id[idx]}
                    <span style="color: ${status_color}; font-size: 0.8em;">
                        (${status})
                    </span>
                </h3>
                <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                    <tr><td>Area:</td><td><b>${data.area[idx]} sq in</b></td></tr>
                    <tr><td>Min Width:</td><td><b>${data.min_width[idx]}"</b></td></tr>
                    <tr><td>Max Width:</td><td><b>${data.max_width[idx]}"</b></td></tr>
                    <tr><td>Min Angle:</td><td><b>${data.min_angle[idx]}°</b></td></tr>
                    <tr><td>Vertices:</td><td><b>${data.vertices[idx]}</b></td></tr>
                </table>
                <h4 style="margin-bottom: 4px;">Warnings:</h4>
                ${warn_html}
            </div>`;
        """)

        source.selected.js_on_change('indices', tap_callback)

        # --- Filter controls ---
        filter_callback = CustomJS(args=dict(
            source=source,
            source_full=source_full
        ), code="""
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
        """)

        filter_div = Div(text="""
        <div style="font-family: Arial, sans-serif; padding: 10px;
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
        warning_types = defaultdict(int)
        for piece in self.pieces:
            for w in piece.warnings:
                wtype = w.split(":")[0]
                warning_types[wtype] += 1

        if warning_types:
            warn_labels = list(warning_types.keys())
            warn_values = list(warning_types.values())

            warn_source = ColumnDataSource(data=dict(
                labels=warn_labels,
                counts=warn_values,
                colors=[
                    "#e74c3c" if "VERY" in l or "TINY" in l else "#f39c12"
                    for l in warn_labels
                ]
            ))

            warn_chart = figure(
                title="Warning Breakdown",
                x_range=warn_labels,
                width=340,
                height=200,
                tools="",
                toolbar_location=None,
            )
            warn_chart.vbar(
                x='labels', top='counts', width=0.8,
                source=warn_source,
                color='colors',
            )
            warn_chart.xaxis.major_label_orientation = 0.8
            warn_chart.yaxis.axis_label = "Count"
        else:
            warn_chart = Div(text="""
            <div style="padding: 10px; background: #d4edda; border-radius: 8px;">
                <b>✓ No warnings — all pieces OK!</b>
            </div>
            """)

        # --- Area histogram ---
        hist_values, hist_edges = np.histogram(areas, bins=20)
        hist_source = ColumnDataSource(data=dict(
            top=hist_values.tolist(),
            left=hist_edges[:-1].tolist(),
            right=hist_edges[1:].tolist(),
        ))

        area_hist = figure(
            title="Piece Size Distribution",
            width=340,
            height=200,
            tools="",
            toolbar_location=None,
        )
        area_hist.quad(
            top='top', bottom=0, left='left', right='right',
            source=hist_source,
            fill_color="#3498db",
            line_color="white",
            alpha=0.8,
        )
        area_hist.xaxis.axis_label = "Area (sq in)"
        area_hist.yaxis.axis_label = "Count"

        # --- Layout ---
        right_panel = column(
            summary_div,
            detail_div,
            filter_div,
            checkbox,
            warn_chart,
            area_hist,
        )

        layout = row(p, right_panel)

        save(layout)
        print(f"Saved interactive visualization: {output_path}")
        return output_path

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Stained Glass Pattern Analyzer')
    parser.add_argument('image', help='Path to pattern image')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Scale DPI (default: 150)')
    parser.add_argument('--panel-width', type=float, default=None,
                        help='Actual panel width in inches (auto-calculates DPI)')
    parser.add_argument('--panel-height', type=float, default=None,
                        help='Actual panel height in inches (auto-calculates DPI)')
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
    parser.add_argument('--no-bokeh', action='store_true',
                        help='Skip Bokeh interactive visualization')

    args = parser.parse_args()

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
            print("Install with: pip install bokeh pillow")
        except Exception as e:
            print(f"Bokeh visualization failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n--- Done! ---")
    output_dir = Path(args.output_dir).resolve()
    print(f"All output files in: {output_dir}")
    print(f"  debug_00_threshold_only.png    — raw threshold result")
    print(f"  debug_01_binary.png            — after morphological cleanup")
    print(f"  debug_02_lines_highlighted.png — lines shown in red")
    print(f"  {args.prefix}_analyzed.png     — numbered pieces with QA highlights")
    print(f"  {args.prefix}_qa.png           — QA issues only")
    print(f"  {args.prefix}_report.txt       — full text report")
    if not args.no_bokeh:
        print(f"  {args.prefix}_interactive.html — interactive Bokeh visualization")


if __name__ == "__main__":
    main()
