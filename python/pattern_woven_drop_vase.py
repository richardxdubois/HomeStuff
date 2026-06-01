from bokeh.plotting import figure, output_file, save
from bokeh.models import Range1d
from bokeh.layouts import column
import math


class StainedGlassPattern:
    """Generate printable patterns for stained glass projects."""

    def __init__(self, disk_diameter_inches=9, strip_width_inches=0.25,
                 strip_gap_inches=0.25, dpi=96):
        """
        Initialize the pattern generator.

        Args:
            disk_diameter_inches: Diameter of the base disk
            strip_width_inches: Width of each strip
            strip_gap_inches: Gap between strips
            dpi: Dots per inch for screen display (96 is standard)
        """
        self.disk_diameter = disk_diameter_inches
        self.strip_width = strip_width_inches
        self.strip_gap = strip_gap_inches
        self.dpi = dpi
        self.repeat_pattern = strip_width_inches + strip_gap_inches

        # Calculate number of strips needed
        self.num_strips = int(math.ceil(disk_diameter_inches / self.repeat_pattern))

        # Convert inches to pixels for bokeh (at 96 dpi)
        self.disk_diameter_px = int(disk_diameter_inches * dpi)

    def calculate_strip_positions(self):
        """Calculate center positions for all strips."""
        radius = self.disk_diameter / 2
        positions = []

        # Center strip
        positions.append(0)

        # Strips on each side of center
        for i in range(1, self.num_strips):
            offset = i * self.repeat_pattern
            if offset <= radius:
                positions.append(offset)
                positions.append(-offset)

        return sorted(positions)

    def calculate_strip_length(self, offset_from_center):
        """Calculate the length of a strip at given offset from center."""
        radius = self.disk_diameter / 2
        offset = abs(offset_from_center)

        if offset >= radius:
            return 0

        # Using Pythagorean theorem: length = 2 * sqrt(r^2 - offset^2)
        half_length = math.sqrt(radius ** 2 - offset ** 2)
        return 2 * half_length

    def draw_half_circle(self, p, radius, side='left'):
        """Draw a half circle arc on the plot."""
        # Generate points for just the half we want
        if side == 'left':
            # Left half: theta from pi/2 (top) through pi (left) to 3pi/2 (bottom)
            theta = [math.pi / 2 + i * math.pi / 100 for i in range(101)]
        else:  # right
            # Right half: theta from -pi/2 (bottom) through 0 (right) to pi/2 (top)
            theta = [-math.pi / 2 + i * math.pi / 100 for i in range(101)]

        circle_x = [radius * math.cos(t) for t in theta]
        circle_y = [radius * math.sin(t) for t in theta]

        # Draw the arc
        p.line(circle_x, circle_y, line_width=2, color="black", legend_label="Half disk edge")

    def create_half_pattern(self, side='left', title_suffix="Left Half"):
        """Create a half-circle pattern that fits on letter paper."""

        radius = self.disk_diameter / 2

        # Make it square so match_aspect works correctly
        # Use 9" x 9" but only show half horizontally
        # Half circle needs to be 4.5" wide x 9" tall to maintain aspect
        width_px = int(4.5 * self.dpi)
        height_px = int(9 * self.dpi)

        p = figure(
            width=width_px,
            height=height_px,
            title=f"{self.disk_diameter}\" Woven Glass Pattern - {title_suffix}",
            toolbar_location="right",
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )

        if side == 'left':
            p.x_range = Range1d(-radius - 0.2, 0.2)
            p.y_range = Range1d(-radius - 0.2, radius + 0.2)
        else:  # right
            p.x_range = Range1d(-0.2, radius + 0.2)
            p.y_range = Range1d(-radius - 0.2, radius + 0.2)

        # Draw half circle
        self.draw_half_circle(p, radius, side)

        # Draw center line
        p.line([0, 0], [-radius, radius], line_width=2, color="black", line_dash="dashed",
               legend_label="Center line (align halves here)")

        # Get strip positions
        positions = self.calculate_strip_positions()

        # Draw horizontal strips (warp)
        for pos in positions:
            strip_length = self.calculate_strip_length(pos)
            if strip_length > 0:
                half_length = strip_length / 2

                if side == 'left':
                    x_start = -half_length
                    x_end = 0
                else:  # right
                    x_start = 0
                    x_end = half_length

                # Draw center line of strip
                p.line([x_start, x_end], [pos, pos],
                       line_width=1, color="blue", alpha=0.5)
                # Draw strip edges
                top_edge = pos + self.strip_width / 2
                bottom_edge = pos - self.strip_width / 2
                p.line([x_start, x_end], [top_edge, top_edge],
                       line_width=0.5, color="blue", alpha=0.3, line_dash="dashed")
                p.line([x_start, x_end], [bottom_edge, bottom_edge],
                       line_width=0.5, color="blue", alpha=0.3, line_dash="dashed")

        # Draw vertical strips (weft)
        for pos in positions:
            if side == 'left' and pos > 0:
                continue
            if side == 'right' and pos < 0:
                continue

            strip_length = self.calculate_strip_length(pos)
            if strip_length > 0:
                half_length = strip_length / 2
                # Draw center line of strip
                p.line([pos, pos], [-half_length, half_length],
                       line_width=1, color="red", alpha=0.5)
                # Draw strip edges
                right_edge = pos + self.strip_width / 2
                left_edge = pos - self.strip_width / 2
                p.line([right_edge, right_edge], [-half_length, half_length],
                       line_width=0.5, color="red", alpha=0.3, line_dash="dashed")
                p.line([left_edge, left_edge], [-half_length, half_length],
                       line_width=0.5, color="red", alpha=0.3, line_dash="dotted")

        # Add grid lines at 1" intervals
        for i in range(-int(radius), int(radius) + 1):
            if i != 0:
                if side == 'left':
                    if i <= 0:
                        p.line([i, i], [-radius, radius],
                               line_width=0.3, color="gray", alpha=0.3, line_dash="dotted")
                else:
                    if i >= 0:
                        p.line([i, i], [-radius, radius],
                               line_width=0.3, color="gray", alpha=0.3, line_dash="dotted")

                x_min = -radius if side == 'left' else 0
                x_max = 0 if side == 'left' else radius
                p.line([x_min, x_max], [i, i],
                       line_width=0.3, color="gray", alpha=0.3, line_dash="dotted")

        # Style the plot
        p.xaxis.axis_label = "Inches"
        p.yaxis.axis_label = "Inches"
        p.legend.location = "bottom_left" if side == 'left' else "bottom_right"

        # Hide axes for cleaner printing
        p.xaxis.visible = False
        p.yaxis.visible = False

        return p

    def create_pattern(self, output_filename="glass_pattern.html"):
        """Create the full-size printable pattern with full circle and two halves."""

        # Create full circle pattern
        p_full = figure(
            width=self.disk_diameter_px,
            height=self.disk_diameter_px,
            title=f"{self.disk_diameter}\" Woven Glass Pattern - Full Circle (Reference)",
            toolbar_location="right",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            match_aspect=True
        )

        # Set exact ranges to match disk diameter
        radius = self.disk_diameter / 2
        p_full.x_range = Range1d(-radius, radius)
        p_full.y_range = Range1d(-radius, radius)

        # Draw outer circle (base disk)
        theta = [i * 2 * math.pi / 100 for i in range(101)]
        circle_x = [radius * math.cos(t) for t in theta]
        circle_y = [radius * math.sin(t) for t in theta]
        p_full.line(circle_x, circle_y, line_width=2, color="black",
                    legend_label="Base disk edge")

        # Get strip positions
        positions = self.calculate_strip_positions()

        # Draw horizontal strips (warp)
        for pos in positions:
            strip_length = self.calculate_strip_length(pos)
            if strip_length > 0:
                half_length = strip_length / 2
                # Draw center line of strip
                p_full.line([-half_length, half_length], [pos, pos],
                            line_width=1, color="blue", alpha=0.5)
                # Draw strip edges
                top_edge = pos + self.strip_width / 2
                bottom_edge = pos - self.strip_width / 2
                p_full.line([-half_length, half_length], [top_edge, top_edge],
                            line_width=0.5, color="blue", alpha=0.3, line_dash="dashed")
                p_full.line([-half_length, half_length], [bottom_edge, bottom_edge],
                            line_width=0.5, color="blue", alpha=0.3, line_dash="dashed")

        # Draw vertical strips (weft)
        for pos in positions:
            strip_length = self.calculate_strip_length(pos)
            if strip_length > 0:
                half_length = strip_length / 2
                # Draw center line of strip
                p_full.line([pos, pos], [-half_length, half_length],
                            line_width=1, color="red", alpha=0.5)
                # Draw strip edges
                right_edge = pos + self.strip_width / 2
                left_edge = pos - self.strip_width / 2
                p_full.line([right_edge, right_edge], [-half_length, half_length],
                            line_width=0.5, color="red", alpha=0.3, line_dash="dashed")
                p_full.line([left_edge, left_edge], [-half_length, half_length],
                            line_width=0.5, color="red", alpha=0.3, line_dash="dotted")

        # Add grid lines at 1" intervals for reference
        for i in range(-int(radius), int(radius) + 1):
            if i != 0:
                p_full.line([i, i], [-radius, radius], line_width=0.3,
                            color="gray", alpha=0.3, line_dash="dotted")
                p_full.line([-radius, radius], [i, i], line_width=0.3,
                            color="gray", alpha=0.3, line_dash="dotted")

        # Style the plot
        p_full.xaxis.axis_label = "Inches"
        p_full.yaxis.axis_label = "Inches"
        p_full.legend.location = "top_right"
        p_full.legend.click_policy = "hide"

        # Add text annotations
        p_full.text([0], [radius + 0.3],
                    text=[f"Blue = Horizontal strips | Red = Vertical strips"],
                    text_align="center", text_font_size="10pt")
        p_full.text([0], [radius + 0.6],
                    text=[
                        f"Strip width: {self.strip_width}\" | Gap: {self.strip_gap}\" | Total strips: {len(positions) * 2}"],
                    text_align="center", text_font_size="10pt")

        # Create half patterns (separate, not in a row)
        p_left = self.create_half_pattern(side='left', title_suffix="Left Half (Print on 8.5x11 Landscape)")
        p_right = self.create_half_pattern(side='right', title_suffix="Right Half (Print on 8.5x11 Landscape)")

        # Combine all plots vertically
        layout = column(p_full, p_left, p_right)

        # Save the pattern
        output_file(output_filename)
        save(layout)

        print(f"Pattern saved to {output_filename}")
        print(f"Total strips needed: {len(positions) * 2}")
        print(f"\nPattern includes:")
        print(f"1. Full circle (reference - use save tool to download PNG)")
        print(f"2. Left half circle (fits 8.5x11\" paper in landscape)")
        print(f"3. Right half circle (fits 8.5x11\" paper in landscape)")
        print(f"\nTo use as a guide underneath:")
        print(f"1. Print left and right halves separately in landscape mode")
        print(f"2. Align them at the center dashed line")
        print(f"3. Tape together to create full circle pattern")
        print(f"4. Place under your clear base disk while weaving")

        return layout

    def generate_cutting_list(self):
        """Generate a list of strip lengths needed."""
        positions = self.calculate_strip_positions()
        cutting_list = []

        for pos in positions:
            length = self.calculate_strip_length(pos)
            if length > 0:
                cutting_list.append({
                    'position': round(pos, 2),
                    'length': round(length, 2),
                    'cut_length': round(length + 0.5, 2)  # Add 0.5" for handling
                })

        print("\nCutting List (each strip needed in both directions):")
        print("-" * 60)
        print(f"{'Position (from center)':<25} {'Exact Length':<15} {'Cut Length'}")
        print("-" * 60)

        for strip in cutting_list:
            print(
                f"{strip['position']:>7.2f}\"  {'':<16} {strip['length']:>6.2f}\"  {'':<7} {strip['cut_length']:>6.2f}\"")

        print("-" * 60)
        print(f"Total unique lengths: {len(cutting_list)}")
        print(f"Each length needed: 2 times (horizontal and vertical)")
        print(f"Total strips: {len(cutting_list) * 2}")


        return cutting_list

# Create the pattern for your project
if __name__ == "__main__":
    pattern = StainedGlassPattern(
        disk_diameter_inches=9,
        strip_width_inches=0.25,
        strip_gap_inches=0.25,
        dpi=96
    )

    pattern.create_pattern("/Volumes/Data/Home/stained_glass/woven_drop_vase/woven_glass_pattern.html")
    pattern.generate_cutting_list()
