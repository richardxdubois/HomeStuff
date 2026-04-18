"""
Stained Glass & Fused Glass Tessellation Generator
====================================================
Converts an image into a glass cutting pattern optimized for
real fabrication constraints. Supports:
  - Lead came construction
  - Copper foil (Tiffany) construction
  - Fused glass (no gaps, no border)
  - Multiple cutting tool profiles (standard, ring saw, etc.)
  - Named projects for organization
  - Watertight tessellation (no gaps, no overlaps)

Requirements:
    pip install numpy opencv-python scipy scikit-image bokeh shapely Pillow

Usage:
    python stained_glass.py -i input.jpg -p "rose_window" --pieces 150
    python stained_glass.py -i input.jpg -p "kitchen_panel" --mode foil --tools ring_saw
    python stained_glass.py -i input.jpg -p "fused_bowl" --mode fused -n 80
"""

import numpy as np
import cv2
from scipy.spatial import Voronoi, cKDTree
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import slic, find_boundaries
from skimage.measure import find_contours, regionprops, label
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box
from shapely.ops import unary_union
from shapely.validation import make_valid
from bokeh.plotting import figure, show, save, output_file
from bokeh.models import (
    ColumnDataSource, HoverTool, ColorBar,
    LinearColorMapper, Label, Div, CustomJS,
    Range1d, Title
)
from bokeh.layouts import row, column, gridplot
from bokeh.palettes import Turbo256
from bokeh.io import export_svg
from PIL import Image as PILImage, ImageDraw
from enum import Enum
import argparse
import json
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# Tool Capability Profiles
# =============================================================================

@dataclass
class ToolProfile:
    """Defines what cutting operations are possible with available tools."""
    name: str
    description: str
    max_concavity: float
    min_interior_angle: float
    min_inside_radius_in: float
    min_outside_radius_in: float
    min_neck_width_in: float
    max_complexity: float
    min_piece_dimension_in: float
    min_piece_area_in2: float
    can_inside_curves: bool = False
    can_sharp_inside_corners: bool = False
    can_deep_concavity: bool = False
    can_narrow_necks: bool = False


TOOL_PROFILES: Dict[str, ToolProfile] = {
    "standard": ToolProfile(
        name="Standard Glass Cutter",
        description="Score-and-break with running pliers. Grozing pliers for minor shaping.",
        max_concavity=0.15, min_interior_angle=30.0,
        min_inside_radius_in=2.0, min_outside_radius_in=0.5,
        min_neck_width_in=1.0, max_complexity=25.0,
        min_piece_dimension_in=0.75, min_piece_area_in2=1.0,
    ),
    "standard_with_grinder": ToolProfile(
        name="Standard Cutter + Grinder",
        description="Score-and-break with grinder cleanup. Allows tighter fits.",
        max_concavity=0.25, min_interior_angle=25.0,
        min_inside_radius_in=1.0, min_outside_radius_in=0.375,
        min_neck_width_in=0.75, max_complexity=30.0,
        min_piece_dimension_in=0.625, min_piece_area_in2=0.75,
    ),
    "ring_saw": ToolProfile(
        name="Ring Saw",
        description="Diamond ring blade allows interior cuts, deep concavities, tight inside curves.",
        max_concavity=0.85, min_interior_angle=10.0,
        min_inside_radius_in=0.25, min_outside_radius_in=0.125,
        min_neck_width_in=0.375, max_complexity=60.0,
        min_piece_dimension_in=0.375, min_piece_area_in2=0.5,
        can_inside_curves=True, can_sharp_inside_corners=True,
        can_deep_concavity=True, can_narrow_necks=True,
    ),
    "ring_saw_with_grinder": ToolProfile(
        name="Ring Saw + Grinder",
        description="Maximum capability: ring saw for complex cuts, grinder for precision cleanup.",
        max_concavity=0.92, min_interior_angle=8.0,
        min_inside_radius_in=0.1875, min_outside_radius_in=0.1,
        min_neck_width_in=0.3125, max_complexity=75.0,
        min_piece_dimension_in=0.3125, min_piece_area_in2=0.375,
        can_inside_curves=True, can_sharp_inside_corners=True,
        can_deep_concavity=True, can_narrow_necks=True,
    ),
    "band_saw": ToolProfile(
        name="Band Saw (glass)",
        description="Glass band saw. Good for inside curves but less precise than ring saw.",
        max_concavity=0.65, min_interior_angle=15.0,
        min_inside_radius_in=0.5, min_outside_radius_in=0.25,
        min_neck_width_in=0.5, max_complexity=45.0,
        min_piece_dimension_in=0.5, min_piece_area_in2=0.625,
        can_inside_curves=True, can_deep_concavity=True,
    ),
    "water_jet": ToolProfile(
        name="Water Jet Cutter",
        description="CNC water jet. Extremely precise, virtually no shape limitations.",
        max_concavity=0.98, min_interior_angle=3.0,
        min_inside_radius_in=0.0625, min_outside_radius_in=0.0625,
        min_neck_width_in=0.1875, max_complexity=100.0,
        min_piece_dimension_in=0.1875, min_piece_area_in2=0.25,
        can_inside_curves=True, can_sharp_inside_corners=True,
        can_deep_concavity=True, can_narrow_necks=True,
    ),
}


# =============================================================================
# Construction Method Definitions
# =============================================================================

class ConstructionMethod(Enum):
    COPPER_FOIL = "copper_foil"
    LEAD_CAME = "lead_came"
    FUSED = "fused"


@dataclass
class CameProfile:
    """Physical dimensions of the came or foil. All measurements in inches."""
    name: str
    method: ConstructionMethod
    heart_width: float
    face_width: float
    channel_depth: float
    scissor_allowance: float

    @property
    def total_visible_width(self) -> float:
        return self.heart_width + 2 * self.face_width

    def piece_inset(self) -> float:
        return self.heart_width / 2.0


@dataclass
class ZincFrame:
    """Zinc border frame dimensions and calculations."""
    came_profile: CameProfile
    panel_width_in: float
    panel_height_in: float

    @property
    def channel_depth_in(self) -> float:
        return self.came_profile.channel_depth

    @property
    def outer_width_in(self) -> float:
        return self.panel_width_in + 2 * self.came_profile.face_width

    @property
    def outer_height_in(self) -> float:
        return self.panel_height_in + 2 * self.came_profile.face_width

    @property
    def glass_width_in(self) -> float:
        return self.panel_width_in + 2 * self.channel_depth_in

    @property
    def glass_height_in(self) -> float:
        return self.panel_height_in + 2 * self.channel_depth_in

    @property
    def frame_perimeter_in(self) -> float:
        return 2 * (self.outer_width_in + self.outer_height_in)

    @property
    def mitre_cut_lengths(self) -> Dict[str, float]:
        top_bottom = self.outer_width_in + 2 * self.channel_depth_in
        left_right = self.outer_height_in + 2 * self.channel_depth_in
        return {
            "top": top_bottom, "bottom": top_bottom,
            "left": left_right, "right": left_right,
            "total_linear": 2 * top_bottom + 2 * left_right,
        }

    @property
    def frame_pieces_description(self) -> str:
        cuts = self.mitre_cut_lengths
        return "\n".join([
            "ZINC FRAME CUT LIST",
            f"  Profile: {self.came_profile.name}",
            f"  Channel depth: {self.channel_depth_in:.3f}\"",
            f"",
            f"  Visible panel: {self.panel_width_in:.2f}\" x {self.panel_height_in:.2f}\"",
            f"  Outer frame:   {self.outer_width_in:.2f}\" x {self.outer_height_in:.2f}\"",
            f"  Glass area:    {self.glass_width_in:.2f}\" x {self.glass_height_in:.2f}\"",
            f"",
            f"  Frame pieces (mitre-cut):",
            f"    Top:    {cuts['top']:.3f}\"",
            f"    Bottom: {cuts['bottom']:.3f}\"",
            f"    Left:   {cuts['left']:.3f}\"",
            f"    Right:  {cuts['right']:.3f}\"",
            f"",
            f"  Total zinc needed: {cuts['total_linear']:.2f}\"",
            f"  (Buy at least: {cuts['total_linear'] + 2:.1f}\" for waste)",
        ])

    def to_dict(self) -> dict:
        cuts = self.mitre_cut_lengths
        return {
            "profile": self.came_profile.name,
            "channel_depth_in": self.channel_depth_in,
            "panel_visible_width_in": self.panel_width_in,
            "panel_visible_height_in": self.panel_height_in,
            "outer_frame_width_in": self.outer_width_in,
            "outer_frame_height_in": self.outer_height_in,
            "glass_area_width_in": self.glass_width_in,
            "glass_area_height_in": self.glass_height_in,
            "frame_perimeter_in": self.frame_perimeter_in,
            "mitre_cuts": cuts,
        }


CAME_PROFILES: Dict[str, CameProfile] = {
    "copper_foil_3_16": CameProfile(
        name="Copper Foil (3/16\" face)", method=ConstructionMethod.COPPER_FOIL,
        heart_width=0.0, face_width=3 / 32, channel_depth=0.0, scissor_allowance=1 / 32,
    ),
    "copper_foil_7_32": CameProfile(
        name="Copper Foil (7/32\" face)", method=ConstructionMethod.COPPER_FOIL,
        heart_width=0.0, face_width=7 / 64, channel_depth=0.0, scissor_allowance=1 / 32,
    ),
    "lead_3_16": CameProfile(
        name="3/16\" Lead Came (H)", method=ConstructionMethod.LEAD_CAME,
        heart_width=3 / 16, face_width=1 / 16, channel_depth=1 / 4, scissor_allowance=7 / 64,
    ),
    "lead_1_4": CameProfile(
        name="1/4\" Lead Came (H)", method=ConstructionMethod.LEAD_CAME,
        heart_width=1 / 4, face_width=5 / 64, channel_depth=1 / 4, scissor_allowance=1 / 8,
    ),
    "lead_3_8": CameProfile(
        name="3/8\" Lead Came (H)", method=ConstructionMethod.LEAD_CAME,
        heart_width=3 / 8, face_width=3 / 32, channel_depth=5 / 16, scissor_allowance=3 / 16,
    ),
    "zinc_1_4": CameProfile(
        name="1/4\" Zinc Came (U-border)", method=ConstructionMethod.LEAD_CAME,
        heart_width=1 / 4, face_width=3 / 32, channel_depth=5 / 16, scissor_allowance=1 / 8,
    ),
    "zinc_3_8": CameProfile(
        name="3/8\" Zinc Came (U-border)", method=ConstructionMethod.LEAD_CAME,
        heart_width=3 / 8, face_width=1 / 8, channel_depth=3 / 8, scissor_allowance=3 / 16,
    ),
    "fused_none": CameProfile(
        name="Fused Glass (no came)", method=ConstructionMethod.FUSED,
        heart_width=0.0, face_width=0.0, channel_depth=0.0, scissor_allowance=0.0,
    ),
}


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class GlassPiece:
    """Represents a single piece of stained glass."""
    id: int
    polygon: Polygon
    color_rgb: Tuple[int, int, int]
    color_hex: str
    area: float
    perimeter: float
    centroid: Tuple[float, float]
    complexity: float
    is_feasible: bool = True
    feasibility_issues: List[str] = field(default_factory=list)


@dataclass
class FabricationConstraints:
    """Real-world constraints for stained glass fabrication."""
    tool_profile: ToolProfile = field(
        default_factory=lambda: TOOL_PROFILES["standard"]
    )
    min_piece_area_in2: Optional[float] = None
    max_piece_area_in2: float = 36.0
    construction_mode: ConstructionMethod = ConstructionMethod.LEAD_CAME
    came_profile: CameProfile = field(
        default_factory=lambda: CAME_PROFILES["lead_3_16"]
    )
    border_came_profile: Optional[CameProfile] = field(
        default_factory=lambda: CAME_PROFILES["zinc_1_4"]
    )
    ppi: float = 50.0
    available_colors: Optional[int] = 24
    fuse_overlap_in: float = 0.0
    fuse_border_margin_in: float = 0.125

    @property
    def effective_min_area_in2(self) -> float:
        return self.min_piece_area_in2 if self.min_piece_area_in2 is not None \
            else self.tool_profile.min_piece_area_in2

    @property
    def min_piece_area(self) -> float:
        return self.effective_min_area_in2 * self.ppi ** 2

    @property
    def max_piece_area(self) -> float:
        return self.max_piece_area_in2 * self.ppi ** 2

    @property
    def min_angle(self) -> float:
        return self.tool_profile.min_interior_angle

    @property
    def max_complexity(self) -> float:
        return self.tool_profile.max_complexity

    @property
    def max_concavity(self) -> float:
        return self.tool_profile.max_concavity

    @property
    def min_inside_radius_px(self) -> float:
        return self.tool_profile.min_inside_radius_in * self.ppi

    @property
    def min_neck_width_px(self) -> float:
        return self.tool_profile.min_neck_width_in * self.ppi

    @property
    def heart_width_px(self) -> float:
        return self.came_profile.heart_width * self.ppi

    @property
    def scissor_allowance_px(self) -> float:
        return self.came_profile.scissor_allowance * self.ppi

    @property
    def piece_inset_px(self) -> float:
        return self.came_profile.piece_inset() * self.ppi

    @property
    def border_inset_px(self) -> float:
        if self.border_came_profile:
            return self.border_came_profile.piece_inset() * self.ppi
        return self.piece_inset_px

    @property
    def lead_width_px(self) -> float:
        return self.came_profile.total_visible_width * self.ppi

    @property
    def fuse_overlap_px(self) -> float:
        return self.fuse_overlap_in * self.ppi

    @property
    def is_fused(self) -> bool:
        return self.construction_mode == ConstructionMethod.FUSED

    @property
    def has_border_frame(self) -> bool:
        return not self.is_fused and self.border_came_profile is not None


# =============================================================================
# Piece Geometry
# =============================================================================

class PieceGeometry:
    """Manages design_line, cut_line, and glass_edge polygon variants."""

    def __init__(self, design_polygon: Polygon,
                 constraints: FabricationConstraints,
                 is_border: bool = False):
        self.design_polygon = design_polygon
        self.constraints = constraints
        self.is_border = is_border
        self._compute_derived_polygons()

    def _compute_derived_polygons(self):
        c = self.constraints
        if c.is_fused:
            overlap = c.fuse_overlap_px
            if overlap > 0:
                self.glass_polygon = self._safe_buffer(self.design_polygon, overlap)
            elif overlap < 0:
                self.glass_polygon = self._safe_inset(self.design_polygon, abs(overlap))
            else:
                self.glass_polygon = self.design_polygon
            self.cut_polygon = self.glass_polygon
        else:
            inset = c.border_inset_px if self.is_border else c.piece_inset_px
            self.glass_polygon = self._safe_inset(self.design_polygon, inset) \
                if inset > 0 else self.design_polygon
            scissor = c.scissor_allowance_px
            self.cut_polygon = self._safe_inset(self.design_polygon, scissor) \
                if scissor > 0 else self.glass_polygon

    @staticmethod
    def _safe_inset(polygon: Polygon, distance: float) -> Polygon:
        if distance <= 0:
            return polygon
        try:
            inset = polygon.buffer(-distance, join_style=2, mitre_limit=2.0)
            if inset.is_empty:
                return polygon
            if isinstance(inset, MultiPolygon):
                inset = max(inset.geoms, key=lambda g: g.area)
            if inset.area < polygon.area * 0.1:
                return polygon
            return inset
        except Exception:
            return polygon

    @staticmethod
    def _safe_buffer(polygon: Polygon, distance: float) -> Polygon:
        if distance <= 0:
            return polygon
        try:
            expanded = polygon.buffer(distance, join_style=2, mitre_limit=2.0)
            if expanded.is_empty:
                return polygon
            if isinstance(expanded, MultiPolygon):
                expanded = max(expanded.geoms, key=lambda g: g.area)
            return expanded
        except Exception:
            return polygon

    @property
    def is_viable(self) -> bool:
        return (not self.glass_polygon.is_empty and
                self.glass_polygon.area > self.constraints.min_piece_area * 0.5)


# =============================================================================
# Image Analysis
# =============================================================================

class ImageAnalyzer:
    """Analyzes input image to find optimal tessellation seed points."""

    def __init__(self, image_path: str, max_dimension: int = 800):
        self.image_path = image_path
        self.original = self._load_image(image_path)

        h, w = self.original.shape[:2]
        scale = min(max_dimension / max(h, w), 1.0)
        self.image = cv2.resize(self.original, (int(w * scale), int(h * scale)),
                                interpolation=cv2.INTER_AREA)
        self.height, self.width = self.image.shape[:2]
        self.rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self._compute_edge_map()
        self._compute_detail_map()

    @staticmethod
    def _load_image(image_path: str) -> np.ndarray:
        """Robust image loader with Pillow fallback."""
        cleaned = image_path.strip().strip("'\"")
        path = Path(cleaned).expanduser()
        if not path.is_absolute():
            path = path.resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"Image not found:\n"
                f"  As given:    {image_path}\n"
                f"  Cleaned to:  {cleaned}\n"
                f"  Final path:  {path}\n"
                f"  CWD:         {Path.cwd()}"
            )

        # cv2 via numpy buffer (bypasses path encoding issues)
        try:
            buf = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is not None:
                return img
        except Exception:
            pass

        # Pillow fallback
        try:
            pil_img = PILImage.open(str(path))
            if pil_img.mode == 'RGBA':
                background = PILImage.new('RGB', pil_img.size, (255, 255, 255))
                background.paste(pil_img, mask=pil_img.split()[3])
                pil_img = background
            elif pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            rgb_array = np.array(pil_img)
            return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        except Exception:
            pass

        raise ValueError(
            f"Could not decode image: {path}\n"
            f"  File size: {path.stat().st_size:,} bytes\n"
            f"  Suffix: {path.suffix}\n"
            f"  Try converting to PNG and retry."
        )

    def _compute_edge_map(self):
        edges_gray = cv2.Canny(self.gray, 50, 150)
        edges_color = np.zeros_like(self.gray, dtype=np.float64)
        for channel in range(3):
            ch = self.lab[:, :, channel]
            sx = cv2.Sobel(ch, cv2.CV_64F, 1, 0, ksize=3)
            sy = cv2.Sobel(ch, cv2.CV_64F, 0, 1, ksize=3)
            edges_color += np.sqrt(sx ** 2 + sy ** 2)
        edges_color = (edges_color / max(edges_color.max(), 1e-8) * 255).astype(np.uint8)
        combined = cv2.addWeighted(edges_gray, 0.5, edges_color, 0.5, 0)
        self.edge_map = cv2.GaussianBlur(combined, (5, 5), 1.0)

    def _compute_detail_map(self):
        local_mean = cv2.blur(self.gray.astype(np.float64), (21, 21))
        local_sq_mean = cv2.blur(self.gray.astype(np.float64) ** 2, (21, 21))
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))
        edge_density = cv2.blur(self.edge_map.astype(np.float64), (31, 31))
        detail = 0.5 * (local_std / max(local_std.max(), 1)) + \
                 0.5 * (edge_density / max(edge_density.max(), 1))
        self.detail_map = detail / max(detail.max(), 1e-8)

    def get_adaptive_seeds(self, num_seeds: int) -> np.ndarray:
        prob_map = 0.3 + 0.7 * self.detail_map
        prob_map = prob_map / prob_map.sum()
        flat_probs = prob_map.ravel()
        indices = np.random.choice(len(flat_probs), size=num_seeds,
                                   replace=False, p=flat_probs)
        ys, xs = np.unravel_index(indices, prob_map.shape)
        seeds = np.column_stack([xs.astype(float), ys.astype(float)])
        border_seeds = self._generate_border_seeds(spacing=30)
        return np.vstack([seeds, border_seeds])

    def _generate_border_seeds(self, spacing: int = 30) -> np.ndarray:
        w, h = self.width, self.height
        top = np.column_stack([np.arange(0, w, spacing),
                               np.zeros(len(range(0, w, spacing)))])
        bottom = np.column_stack([np.arange(0, w, spacing),
                                  np.full(len(range(0, w, spacing)), h - 1)])
        left = np.column_stack([np.zeros(len(range(0, h, spacing))),
                                np.arange(0, h, spacing)])
        right = np.column_stack([np.full(len(range(0, h, spacing)), w - 1),
                                 np.arange(0, h, spacing)])
        return np.vstack([top, bottom, left, right])

    def get_region_color(self, mask: np.ndarray) -> Tuple[int, int, int]:
        pixels = self.rgb[mask]
        if len(pixels) == 0:
            return (128, 128, 128)
        return tuple(int(c) for c in pixels.mean(axis=0))

    def quantize_color(self, color: Tuple[int, int, int],
                       palette: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
        color_arr = np.array(color, dtype=np.float64)
        palette_arr = np.array(palette, dtype=np.float64)
        distances = np.sqrt(((palette_arr - color_arr) ** 2).sum(axis=1))
        return tuple(int(c) for c in palette[np.argmin(distances)])


# =============================================================================
# Watertight Tessellation Engine
# =============================================================================

class WatertightTessellator:
    """
    Produces a guaranteed gap-free, overlap-free tessellation.

    Pipeline:
        1. Generate segments (label map) — every pixel assigned
        2. Fill any unlabelled pixels
        3. Extend edge segments to image boundary
        4. Extract polygons from label map
        5. Remove overlaps (largest-first priority)
        6. Clip to image rectangle, fill remaining gaps
        7. Merge undersized pieces and slivers
        8. Validate: full coverage, no overlaps
    """

    def __init__(self, analyzer: ImageAnalyzer,
                 constraints: FabricationConstraints):
        self.analyzer = analyzer
        self.constraints = constraints
        self.width = analyzer.width
        self.height = analyzer.height
        self.image_rect = box(0, 0, self.width, self.height)

    def tessellate(self, num_pieces: int, method: str = "slic") -> List[GlassPiece]:
        """Generate a watertight, overlap-free tessellation."""
        print(f"  Generating label map ({method})...")
        label_map = self._generate_label_map(num_pieces, method)

        print(f"  Filling gaps and extending to boundary...")
        label_map = self._fill_gaps(label_map)
        label_map = self._extend_to_boundary(label_map)

        print(f"  Extracting polygons...")
        pieces = self._extract_pieces(label_map)
        print(f"    Raw pieces: {len(pieces)}")

        print(f"  Removing overlaps...")
        pieces = self._remove_overlaps(pieces)
        print(f"    After de-overlap: {len(pieces)}")

        print(f"  Clipping and filling coverage gaps...")
        pieces = self._clip_and_fill(pieces)

        print(f"  Merging small pieces and slivers...")
        pieces = self._merge_small_pieces(pieces)
        print(f"    After merge: {len(pieces)}")

        print(f"  Validating pieces...")
        self._validate_pieces(pieces)

        print(f"  Checking coverage and overlaps...")
        self._validate_coverage(pieces)

        return pieces

    # ---- Label Map Generation ----

    def _generate_label_map(self, num_pieces: int, method: str) -> np.ndarray:
        if method in ("slic", "hybrid"):
            segments = slic(
                self.analyzer.rgb,
                n_segments=num_pieces,
                compactness=15,
                sigma=1.5,
                start_label=0,
                enforce_connectivity=True,
                min_size_factor=0.3,
            )
        elif method == "voronoi":
            segments = self._voronoi_to_labelmap(num_pieces)
        else:
            raise ValueError(f"Unknown method: {method}")
        return segments.astype(np.int32)

    def _voronoi_to_labelmap(self, num_pieces: int) -> np.ndarray:
        seeds = self.analyzer.get_adaptive_seeds(num_pieces)
        for _ in range(10):
            tree = cKDTree(seeds)
            yy, xx = np.mgrid[0:self.height, 0:self.width]
            pixels = np.column_stack([xx.ravel(), yy.ravel()])
            _, labels = tree.query(pixels)
            label_map = labels.reshape(self.height, self.width)
            new_seeds = []
            for i in range(len(seeds)):
                mask = label_map == i
                if mask.any():
                    ys, xs = np.where(mask)
                    new_seeds.append([xs.mean(), ys.mean()])
                else:
                    new_seeds.append(seeds[i])
            seeds = np.array(new_seeds)

        tree = cKDTree(seeds)
        yy, xx = np.mgrid[0:self.height, 0:self.width]
        pixels = np.column_stack([xx.ravel(), yy.ravel()])
        _, labels = tree.query(pixels)
        return labels.reshape(self.height, self.width).astype(np.int32)

    # ---- Gap Filling ----

    def _fill_gaps(self, label_map: np.ndarray) -> np.ndarray:
        unlabelled = label_map < 0
        if not unlabelled.any():
            return label_map
        _, nearest_indices = distance_transform_edt(
            unlabelled, return_distances=True, return_indices=True)
        label_map[unlabelled] = label_map[
            nearest_indices[0][unlabelled],
            nearest_indices[1][unlabelled]]
        return label_map

    def _extend_to_boundary(self, label_map: np.ndarray) -> np.ndarray:
        h, w = label_map.shape
        label_map[0, :] = label_map[1, :]
        label_map[h - 1, :] = label_map[h - 2, :]
        label_map[:, 0] = label_map[:, 1]
        label_map[:, w - 1] = label_map[:, w - 2]
        return label_map

    # ---- Polygon Extraction ----

    def _extract_pieces(self, label_map: np.ndarray) -> List[GlassPiece]:
        pieces = []
        unique_labels = np.unique(label_map)
        for seg_id in unique_labels:
            if seg_id < 0:
                continue
            mask = label_map == seg_id
            if not mask.any():
                continue
            polygon = self._mask_to_polygon(mask)
            if polygon is None:
                continue
            color = self.analyzer.get_region_color(mask)
            piece = GlassPiece(
                id=len(pieces), polygon=polygon, color_rgb=color,
                color_hex='#{:02x}{:02x}{:02x}'.format(*color),
                area=polygon.area, perimeter=polygon.length,
                centroid=polygon.centroid.coords[0],
                complexity=(polygon.length ** 2) / max(polygon.area, 1),
            )
            pieces.append(piece)
        return pieces

    def _mask_to_polygon(self, mask: np.ndarray) -> Optional[Polygon]:
        padded = np.pad(mask, pad_width=1, mode='constant',
                        constant_values=False)
        contours = find_contours(padded.astype(np.float64), 0.5)
        if not contours:
            return None
        contour = max(contours, key=len)
        if len(contour) < 4:
            return None

        coords = np.column_stack([contour[:, 1] - 1, contour[:, 0] - 1])
        coords[:, 0] = np.clip(coords[:, 0], 0, self.width)
        coords[:, 1] = np.clip(coords[:, 1], 0, self.height)

        try:
            poly = Polygon(coords.tolist())
            poly = make_valid(poly)
            if poly.geom_type == 'MultiPolygon':
                poly = max(poly.geoms, key=lambda g: g.area)
            if poly.is_empty or poly.area < 1 or poly.geom_type != 'Polygon':
                return None
            poly = poly.simplify(tolerance=1.0, preserve_topology=True)
            if poly.is_empty or poly.area < 1 or poly.geom_type != 'Polygon':
                return None
            return poly
        except Exception:
            return None

    def _polygon_to_mask(self, polygon: Polygon) -> np.ndarray:
        """Convert polygon to boolean mask for color sampling."""
        if polygon.is_empty:
            return np.zeros((self.height, self.width), dtype=bool)
        coords = [(int(round(x)), int(round(y)))
                   for x, y in polygon.exterior.coords]
        img = PILImage.new('L', (self.width, self.height), 0)
        ImageDraw.Draw(img).polygon(coords, fill=1)
        return np.array(img, dtype=bool)

    # ---- Overlap Removal ----

    def _remove_overlaps(self, pieces: List[GlassPiece]) -> List[GlassPiece]:
        """
        Remove overlapping regions between pieces.
        Largest pieces get priority; smaller overlapping pieces are trimmed.
        """
        if not pieces:
            return pieces

        pieces.sort(key=lambda p: -p.area)
        print(f"    Checking {len(pieces)} pieces for overlaps...")

        cleaned = []
        claimed_union = None
        overlap_count = 0
        total_overlap_area = 0.0

        for piece in pieces:
            if claimed_union is None:
                claimed_union = piece.polygon
                cleaned.append(piece)
                continue

            try:
                overlap = piece.polygon.intersection(claimed_union)

                if overlap.is_empty or overlap.area < 1.0:
                    claimed_union = claimed_union.union(piece.polygon)
                    cleaned.append(piece)
                    continue

                overlap_count += 1
                total_overlap_area += overlap.area

                trimmed = piece.polygon.difference(claimed_union)

                if trimmed.is_empty:
                    print(f"      #{piece.id} fully contained \u2014 dropping")
                    continue

                if trimmed.geom_type == 'MultiPolygon':
                    trimmed = max(trimmed.geoms, key=lambda g: g.area)

                if trimmed.geom_type != 'Polygon' or trimmed.area < 1.0:
                    print(f"      #{piece.id} trimmed to nothing \u2014 dropping")
                    continue

                piece.polygon = trimmed
                piece.area = trimmed.area
                piece.perimeter = trimmed.length
                try:
                    piece.centroid = trimmed.centroid.coords[0]
                except Exception:
                    piece.centroid = (
                        (trimmed.bounds[0] + trimmed.bounds[2]) / 2,
                        (trimmed.bounds[1] + trimmed.bounds[3]) / 2,
                    )
                piece.complexity = (trimmed.length ** 2) / max(trimmed.area, 1)

                try:
                    mask = self._polygon_to_mask(trimmed)
                    color = self.analyzer.get_region_color(mask)
                    piece.color_rgb = color
                    piece.color_hex = '#{:02x}{:02x}{:02x}'.format(*color)
                except Exception:
                    pass

                claimed_union = claimed_union.union(trimmed)
                cleaned.append(piece)

            except Exception:
                try:
                    claimed_union = claimed_union.union(piece.polygon)
                except Exception:
                    pass
                cleaned.append(piece)

        if overlap_count > 0:
            print(f"    Resolved {overlap_count} overlaps "
                  f"({total_overlap_area:.0f} px\u00b2 total)")
        else:
            print(f"    No overlaps found \u2714")

        for i, piece in enumerate(cleaned):
            piece.id = i

        return cleaned

    # ---- Clip, Fill, Merge ----

    def _clip_and_fill(self, pieces: List[GlassPiece]) -> List[GlassPiece]:
        clipped = []
        for piece in pieces:
            try:
                cp = piece.polygon.intersection(self.image_rect)
                if cp.is_empty:
                    continue
                if cp.geom_type == 'MultiPolygon':
                    cp = max(cp.geoms, key=lambda g: g.area)
                if cp.geom_type != 'Polygon' or cp.area < 1:
                    continue
                piece.polygon = cp
                piece.area = cp.area
                piece.perimeter = cp.length
                piece.centroid = cp.centroid.coords[0]
                piece.complexity = (cp.length ** 2) / max(cp.area, 1)
                clipped.append(piece)
            except Exception:
                continue

        try:
            all_union = unary_union([p.polygon for p in clipped])
            gaps = self.image_rect.difference(all_union)
            if not gaps.is_empty and gaps.area > 1:
                gap_area = gaps.area
                print(f"    Filling {gap_area:.0f} px\u00b2 of gaps...")
                self._assign_gaps(clipped, gaps)
        except Exception as e:
            print(f"    Warning: gap detection failed: {e}")

        return clipped

    def _assign_gaps(self, pieces: List[GlassPiece], gaps) -> None:
        gap_polys = []
        if gaps.geom_type == 'Polygon':
            gap_polys = [gaps]
        elif gaps.geom_type == 'MultiPolygon':
            gap_polys = list(gaps.geoms)
        elif gaps.geom_type == 'GeometryCollection':
            gap_polys = [g for g in gaps.geoms
                         if g.geom_type == 'Polygon' and g.area > 0.5]

        for gap in gap_polys:
            if gap.area < 0.5:
                continue
            gap_centroid = gap.centroid
            best_piece, best_dist = None, float('inf')
            for piece in pieces:
                try:
                    d = piece.polygon.distance(gap_centroid)
                    if d < best_dist:
                        best_dist, best_piece = d, piece
                except Exception:
                    continue
            if best_piece is not None:
                try:
                    merged = unary_union([best_piece.polygon, gap])
                    if merged.geom_type == 'MultiPolygon':
                        merged = max(merged.geoms, key=lambda g: g.area)
                    if merged.geom_type == 'Polygon' and merged.area > 0:
                        best_piece.polygon = merged
                        best_piece.area = merged.area
                        best_piece.perimeter = merged.length
                        best_piece.centroid = merged.centroid.coords[0]
                        best_piece.complexity = (
                            merged.length ** 2) / max(merged.area, 1)
                except Exception:
                    pass

    def _merge_small_pieces(self, pieces: List[GlassPiece]) -> List[GlassPiece]:
        """Merge undersized pieces AND slivers into their best neighbor."""
        min_area = self.constraints.min_piece_area
        max_complexity = self.constraints.max_complexity
        sliver_complexity = max_complexity * 2.5

        changed = True
        max_iterations = len(pieces) * 2
        iteration = 0

        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            pieces.sort(key=lambda p: p.area)

            for i, piece in enumerate(pieces):
                should_merge = (
                    piece.area < min_area or
                    piece.complexity > sliver_complexity or
                    self._is_sliver(piece.polygon)
                )
                if not should_merge:
                    continue

                # Find best touching neighbor
                best_j, best_shared = None, 0
                for j, other in enumerate(pieces):
                    if i == j:
                        continue
                    try:
                        if (piece.polygon.touches(other.polygon) or
                                piece.polygon.intersects(other.polygon)):
                            shared = piece.polygon.boundary.intersection(
                                other.polygon.buffer(1.0)).length
                            if shared > best_shared:
                                best_shared, best_j = shared, j
                    except Exception:
                        continue

                # Fallback: nearest centroid
                if best_j is None:
                    best_dist = float('inf')
                    for j, other in enumerate(pieces):
                        if i == j:
                            continue
                        d = np.sqrt(
                            (piece.centroid[0] - other.centroid[0]) ** 2 +
                            (piece.centroid[1] - other.centroid[1]) ** 2)
                        if d < best_dist:
                            best_dist, best_j = d, j

                if best_j is not None:
                    try:
                        merged = unary_union([
                            piece.polygon, pieces[best_j].polygon])
                        if merged.geom_type == 'MultiPolygon':
                            merged = max(merged.geoms, key=lambda g: g.area)
                        if merged.geom_type == 'Polygon' and merged.area > 0:
                            pieces[best_j].polygon = merged
                            pieces[best_j].area = merged.area
                            pieces[best_j].perimeter = merged.length
                            pieces[best_j].centroid = merged.centroid.coords[0]
                            pieces[best_j].complexity = (
                                merged.length ** 2) / max(merged.area, 1)
                            try:
                                mask = self._polygon_to_mask(merged)
                                color = self.analyzer.get_region_color(mask)
                                pieces[best_j].color_rgb = color
                                pieces[best_j].color_hex = \
                                    '#{:02x}{:02x}{:02x}'.format(*color)
                            except Exception:
                                pass
                            pieces.pop(i)
                            changed = True
                            break
                    except Exception:
                        pass

        if iteration >= max_iterations:
            print(f"    Warning: merge iteration limit reached")

        for i, piece in enumerate(pieces):
            piece.id = i
        return pieces

    @staticmethod
    def _is_sliver(polygon: Polygon) -> bool:
        """Detect sliver polygons too thin to be useful."""
        try:
            min_rect = polygon.minimum_rotated_rectangle
            rect_coords = list(min_rect.exterior.coords)
            edge1 = np.sqrt(
                (rect_coords[0][0] - rect_coords[1][0]) ** 2 +
                (rect_coords[0][1] - rect_coords[1][1]) ** 2)
            edge2 = np.sqrt(
                (rect_coords[1][0] - rect_coords[2][0]) ** 2 +
                (rect_coords[1][1] - rect_coords[2][1]) ** 2)
            short_edge = min(edge1, edge2)
            long_edge = max(edge1, edge2)
            if short_edge < 1e-6:
                return True
            return (long_edge / short_edge) > 8.0 or short_edge < 3.0
        except Exception:
            return False

    # ---- Validation ----

    def _validate_pieces(self, pieces: List[GlassPiece]) -> None:
        c = self.constraints
        tool = c.tool_profile
        ppi = c.ppi

        for piece in pieces:
            piece.feasibility_issues = []

            if piece.area < c.min_piece_area:
                piece.feasibility_issues.append(
                    f"Too small: {piece.area / ppi ** 2:.2f}in\u00b2 < "
                    f"{c.effective_min_area_in2:.2f}in\u00b2")
            if piece.area > c.max_piece_area:
                piece.feasibility_issues.append(
                    f"Too large: {piece.area / ppi ** 2:.2f}in\u00b2 > "
                    f"{c.max_piece_area_in2:.2f}in\u00b2")
            if piece.complexity > c.max_complexity:
                piece.feasibility_issues.append(
                    f"Too complex for {tool.name}: "
                    f"{piece.complexity:.1f} > {c.max_complexity:.1f}")

            try:
                convex_area = piece.polygon.convex_hull.area
                if convex_area > 0:
                    concavity = (convex_area - piece.area) / convex_area
                    if concavity > c.max_concavity:
                        piece.feasibility_issues.append(
                            f"Too concave for {tool.name}: "
                            f"{concavity:.2f} > {c.max_concavity:.2f}")
            except Exception:
                pass

            try:
                coords = list(piece.polygon.exterior.coords)
                min_ang = self._min_interior_angle(coords)
                if min_ang < c.min_angle:
                    piece.feasibility_issues.append(
                        f"Angle too tight for {tool.name}: "
                        f"{min_ang:.1f}\u00b0 < {c.min_angle:.1f}\u00b0")
            except Exception:
                pass

            try:
                min_w = self._estimate_min_width(piece.polygon)
                min_w_in = min_w / ppi
                if min_w_in < tool.min_neck_width_in:
                    piece.feasibility_issues.append(
                        f"Too narrow for {tool.name}: "
                        f"{min_w_in:.3f}\" < {tool.min_neck_width_in:.3f}\"")
            except Exception:
                pass

            if not tool.can_inside_curves:
                try:
                    if self._has_significant_concavity(piece.polygon):
                        piece.feasibility_issues.append(
                            "Has inside curves \u2014 needs ring/band saw")
                except Exception:
                    pass

            piece.is_feasible = len(piece.feasibility_issues) == 0

    def _validate_coverage(self, pieces: List[GlassPiece]) -> None:
        """Verify full coverage and no overlaps."""
        try:
            all_union = unary_union([p.polygon for p in pieces])
            image_area = self.width * self.height
            covered = all_union.area
            coverage = covered / image_area

            if coverage < 0.999:
                print(f"  WARNING: Coverage is {coverage:.1%} "
                      f"({image_area - covered:.0f} px\u00b2 uncovered)")
            else:
                print(f"  Coverage: {coverage:.1%} \u2714")

            # Check for overlaps via total area
            total_piece_area = sum(p.area for p in pieces)
            if total_piece_area > image_area * 1.01:
                excess = total_piece_area - image_area
                print(f"  WARNING: {excess:.0f} px\u00b2 overlap remains "
                      f"(total={total_piece_area:.0f}, "
                      f"image={image_area:.0f})")
            else:
                print(f"  No overlaps \u2714 "
                      f"(total={total_piece_area:.0f}, "
                      f"image={image_area:.0f})")

            # Flag tiny remnants
            ppi = self.constraints.ppi
            tiny = [p for p in pieces
                    if p.area < self.constraints.min_piece_area * 0.5]
            if tiny:
                print(f"  WARNING: {len(tiny)} very small pieces:")
                for p in tiny:
                    print(f"    #{p.id}: {p.area / ppi ** 2:.3f} in\u00b2 "
                          f"at ({p.centroid[0]:.0f}, {p.centroid[1]:.0f})")

        except Exception as e:
            print(f"  WARNING: Could not validate: {e}")

    # ---- Geometry Helpers ----

    @staticmethod
    def _min_interior_angle(coords: list) -> float:
        if len(coords) < 4:
            return 180.0
        min_angle, n = 180.0, len(coords) - 1
        for i in range(n):
            v1 = np.array(coords[(i - 1) % n]) - np.array(coords[i])
            v2 = np.array(coords[(i + 1) % n]) - np.array(coords[i])
            l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if l1 < 1e-10 or l2 < 1e-10:
                continue
            cos_a = np.clip(np.dot(v1, v2) / (l1 * l2), -1, 1)
            min_angle = min(min_angle, np.degrees(np.arccos(cos_a)))
        return min_angle

    @staticmethod
    def _estimate_min_width(polygon: Polygon) -> float:
        area = polygon.area
        if area < 1:
            return 0.0
        lo, hi = 0.0, np.sqrt(area) / 2
        for _ in range(15):
            mid = (lo + hi) / 2
            inset = polygon.buffer(-mid, join_style=2)
            if inset.is_empty or isinstance(inset, MultiPolygon):
                hi = mid
            else:
                lo = mid
        return lo * 2

    @staticmethod
    def _has_significant_concavity(polygon: Polygon) -> bool:
        ch = polygon.convex_hull
        if ch.area < 1:
            return False
        return (ch.area - polygon.area) / ch.area > 0.05


# =============================================================================
# Visualization (Bokeh)
# =============================================================================

class StainedGlassVisualizer:
    """Bokeh-based interactive visualization."""

    def __init__(self, analyzer: ImageAnalyzer, pieces: List[GlassPiece],
                 constraints: FabricationConstraints,
                 project_name: str = "untitled",
                 zinc_frame: Optional[ZincFrame] = None):
        self.analyzer = analyzer
        self.pieces = pieces
        self.constraints = constraints
        self.project_name = project_name
        self.zinc_frame = zinc_frame
        self.width = analyzer.width
        self.height = analyzer.height

        self.geometries = [
            PieceGeometry(p.polygon, constraints,
                          self._is_border_piece(p.polygon))
            for p in pieces
        ]

    def _is_border_piece(self, polygon: Polygon) -> bool:
        if self.constraints.is_fused:
            return False
        b = polygon.bounds
        m = 2.0
        return (b[0] < m or b[1] < m or
                b[2] > self.width - m or b[3] > self.height - m)

    def create_visualization(self, output_path: str):
        output_file(output_path,
                    title=f"{self.project_name} \u2014 Glass Pattern")

        glass_plot = self._create_glass_panel()
        cut_plot = self._create_cut_pattern_panel()
        analysis_plot = self._create_analysis_panel()
        stats_div = self._create_stats_panel()
        construction_div = self._create_construction_info_panel()

        mode_name = self.constraints.construction_mode.value.replace(
            "_", " ").title()
        tool_name = self.constraints.tool_profile.name
        display_name = self.project_name.replace("_", " ").title()

        layout = column(
            Div(text=f"""
                <h1 style='text-align:center; font-family:Georgia, serif;
                    color:#2c3e50; margin-bottom:2px;'>
                    &#128311; {display_name}
                </h1>
                <p style='text-align:center; color:#7f8c8d;
                    font-family:sans-serif; margin-top:2px;'>
                    Mode: <b>{mode_name}</b> | Tools: <b>{tool_name}</b> |
                    Pieces: <b>{len(self.pieces)}</b><br/>
                    Hover for details. Red outlines = feasibility issues.
                </p>
            """),
            row(glass_plot, cut_plot, sizing_mode="stretch_width"),
            row(analysis_plot,
                column(construction_div, sizing_mode="stretch_width"),
                sizing_mode="stretch_width"),
            stats_div,
            sizing_mode="stretch_width"
        )

        save(layout)
        print(f"  Visualization: {output_path}")
        return layout

    def _make_piece_labels(self, p_figure):
        """Add scaled piece number labels with white halo."""
        label_x, label_y, label_text, label_size = [], [], [], []
        for pc in self.pieces:
            label_x.append(pc.centroid[0])
            label_y.append(pc.centroid[1])
            label_text.append(str(pc.id))
            fs = max(6, min(np.sqrt(pc.area) * 0.3, 14))
            label_size.append(f"{fs:.0f}pt")

        src = ColumnDataSource(dict(
            x=label_x, y=label_y, text=label_text, size=label_size))

        # White halo for readability
        p_figure.text('x', 'y', 'text', source=src,
                      text_font_size='size', text_color="white",
                      text_align="center", text_baseline="middle",
                      text_font_style="bold")
        # Foreground text
        p_figure.text('x', 'y', 'text', source=src,
                      text_font_size='size', text_color="#333",
                      text_align="center", text_baseline="middle")

    def _create_glass_panel(self) -> figure:
        p = figure(
            width=self.width + 80, height=self.height + 80,
            title=f"{self.project_name} \u2014 Finished",
            x_range=Range1d(-10, self.width + 10),
            y_range=Range1d(self.height + 10, -10),
            tools="pan,wheel_zoom,box_zoom,reset,save",
            match_aspect=True,
            background_fill_color=(
                "#1a1a1a" if not self.constraints.is_fused else "#333"),
        )
        p.title.text_font = "Georgia"
        p.title.text_font_size = "14pt"
        p.grid.visible = False
        p.axis.visible = False

        lw = (0.5 if self.constraints.is_fused
              else max(self.constraints.lead_width_px * 0.5, 1.5))
        lc = "#555" if self.constraints.is_fused else "#2c2c2c"

        xs_f, ys_f, cf, if_, af, xf, isf = [], [], [], [], [], [], []
        xs_p, ys_p, cp, ip, ap, xp, isp = [], [], [], [], [], [], []

        for piece in self.pieces:
            coords = list(piece.polygon.exterior.coords)
            px, py = [c[0] for c in coords], [c[1] for c in coords]
            if piece.is_feasible:
                xs_f.append(px); ys_f.append(py); cf.append(piece.color_hex)
                if_.append(piece.id); af.append(f"{piece.area:.0f}")
                xf.append(f"{piece.complexity:.1f}"); isf.append("None")
            else:
                xs_p.append(px); ys_p.append(py); cp.append(piece.color_hex)
                ip.append(piece.id); ap.append(f"{piece.area:.0f}")
                xp.append(f"{piece.complexity:.1f}")
                isp.append("; ".join(piece.feasibility_issues))

        if xs_f:
            src_f = ColumnDataSource(dict(
                xs=xs_f, ys=ys_f, color=cf, piece_id=if_,
                area=af, complexity=xf, issues=isf))
            r_f = p.patches('xs', 'ys', source=src_f,
                            fill_color='color', fill_alpha=0.92,
                            line_color=lc, line_width=lw, line_join='round')

        if xs_p:
            src_p = ColumnDataSource(dict(
                xs=xs_p, ys=ys_p, color=cp, piece_id=ip,
                area=ap, complexity=xp, issues=isp))
            r_p = p.patches('xs', 'ys', source=src_p,
                            fill_color='color', fill_alpha=0.7,
                            line_color='#ff4444', line_width=lw + 1,
                            line_join='round', line_dash='dashed')

        if self.zinc_frame and self.constraints.border_came_profile:
            fo = (self.constraints.border_came_profile.face_width *
                  self.constraints.ppi)
            p.rect(x=self.width / 2, y=self.height / 2,
                   width=self.width + 2 * fo,
                   height=self.height + 2 * fo,
                   fill_color=None, line_color="#888",
                   line_width=fo * 0.8)

        # Hover on all piece renderers
        renderers = []
        if xs_f:
            renderers.append(r_f)
        if xs_p:
            renderers.append(r_p)
        p.add_tools(HoverTool(
            renderers=renderers,
            tooltips=[
                ("Piece #", "@piece_id"),
                ("Area (px\u00b2)", "@area"),
                ("Complexity", "@complexity"),
                ("Issues", "@issues")]))

        self._make_piece_labels(p)
        return p

    def _create_cut_pattern_panel(self) -> figure:
        c = self.constraints
        title = f"{self.project_name} \u2014 Cut Pattern"
        if c.is_fused:
            title += " (fused)"
        else:
            title += f" ({c.came_profile.name})"

        p = figure(
            width=self.width + 80, height=self.height + 80, title=title,
            x_range=Range1d(-10, self.width + 10),
            y_range=Range1d(self.height + 10, -10),
            tools="pan,wheel_zoom,box_zoom,reset,save",
            match_aspect=True, background_fill_color="#f5f5f0",
        )
        p.title.text_font = "Georgia"
        p.title.text_font_size = "14pt"
        p.grid.visible = False
        p.axis.visible = False

        if c.is_fused:
            xs, ys, cols, ids = [], [], [], []
            for piece in self.pieces:
                coords = list(piece.polygon.exterior.coords)
                xs.append([co[0] for co in coords])
                ys.append([co[1] for co in coords])
                cols.append(piece.color_hex); ids.append(piece.id)

            source = ColumnDataSource(dict(
                xs=xs, ys=ys, color=cols, piece_id=ids))
            renderer = p.patches(
                'xs', 'ys', source=source,
                fill_color='color', fill_alpha=0.5,
                line_color='#000', line_width=1.0,
                legend_label='Cut line')

            hover = HoverTool(
                renderers=[renderer],
                tooltips=[("Piece #", "@piece_id")])
            p.add_tools(hover)

        else:
            # Design lines
            xs_d, ys_d = [], []
            for piece in self.pieces:
                coords = list(piece.polygon.exterior.coords)
                xs_d.append([co[0] for co in coords])
                ys_d.append([co[1] for co in coords])
            p.patches('xs', 'ys',
                      source=ColumnDataSource(dict(xs=xs_d, ys=ys_d)),
                      fill_color=None, fill_alpha=0, line_color='#bbb',
                      line_width=0.5, legend_label='Design line')

            # Cut lines — primary interactive layer
            xs_c, ys_c, cc, ic, tc, ac, xc = [], [], [], [], [], [], []
            for i, geom in enumerate(self.geometries):
                coords = list(geom.cut_polygon.exterior.coords)
                xs_c.append([co[0] for co in coords])
                ys_c.append([co[1] for co in coords])
                piece = self.pieces[i]
                cc.append(piece.color_hex)
                ic.append(piece.id)
                tc.append("border" if geom.is_border else "interior")
                ac.append(f"{piece.area:.0f}")
                xc.append(f"{piece.complexity:.1f}")

            source_cut = ColumnDataSource(dict(
                xs=xs_c, ys=ys_c, color=cc, piece_id=ic,
                piece_type=tc, area=ac, complexity=xc))
            cut_renderer = p.patches(
                'xs', 'ys', source=source_cut,
                fill_color='color', fill_alpha=0.25,
                line_color='#000', line_width=1.2,
                legend_label='Cut line')

            # Glass edges
            xs_g, ys_g = [], []
            for geom in self.geometries:
                coords = list(geom.glass_polygon.exterior.coords)
                xs_g.append([co[0] for co in coords])
                ys_g.append([co[1] for co in coords])
            p.patches('xs', 'ys',
                      source=ColumnDataSource(dict(xs=xs_g, ys=ys_g)),
                      fill_color=None, fill_alpha=0, line_color='#0066cc',
                      line_width=0.8, line_dash='dashed',
                      legend_label='Glass edge')

            # Hover only on cut lines
            hover = HoverTool(
                renderers=[cut_renderer],
                tooltips=[
                    ("Piece #", "@piece_id"),
                    ("Type", "@piece_type"),
                    ("Area (px\u00b2)", "@area"),
                    ("Complexity", "@complexity"),
                ])
            p.add_tools(hover)

        self._make_piece_labels(p)

        # Scale bar
        ppi = c.ppi
        bar_y = self.height - 15
        p.line([10, 10 + ppi], [bar_y, bar_y],
               line_color="#333", line_width=2)
        p.line([10, 10], [bar_y - 3, bar_y + 3],
               line_color="#333", line_width=1.5)
        p.line([10 + ppi, 10 + ppi], [bar_y - 3, bar_y + 3],
               line_color="#333", line_width=1.5)
        p.add_layout(Label(
            x=10 + ppi / 2, y=bar_y - 8, text="1 inch",
            text_font_size="9pt", text_color="#333",
            text_align="center"))

        # Legend outside plot area
        if not c.is_fused:
            p.legend.location = "top_right"
            p.legend.label_text_font_size = "8pt"
            p.legend.background_fill_alpha = 0.85
            p.legend.click_policy = "hide"
        p.add_layout(p.legend[0], 'right')

        return p

    def _create_analysis_panel(self) -> figure:
        c = self.constraints
        areas = [pc.area for pc in self.pieces]
        complexities = [pc.complexity for pc in self.pieces]

        p = figure(
            width=500, height=400,
            title=f"{self.project_name} \u2014 Area vs Complexity",
            x_axis_label="Area (px\u00b2)", y_axis_label="Complexity",
            tools="pan,wheel_zoom,box_zoom,reset,hover,save",
            background_fill_color="#fafafa")
        p.title.text_font_size = "13pt"

        source = ColumnDataSource(dict(
            area=areas, complexity=complexities,
            color=[pc.color_hex for pc in self.pieces],
            outline=['#27ae60' if pc.is_feasible else '#e74c3c'
                     for pc in self.pieces],
            piece_id=[pc.id for pc in self.pieces],
            status=['OK' if pc.is_feasible else 'Issue'
                    for pc in self.pieces],
        ))
        p.scatter('area', 'complexity', source=source, size=8,
                  marker='circle', fill_color='color',
                  line_color='outline', line_width=2, alpha=0.8)

        mc = max(complexities) * 1.1 if complexities else 30
        ma = max(areas) * 1.1 if areas else 10000
        p.line([c.min_piece_area] * 2, [0, mc], line_color='#e74c3c',
               line_dash='dashed', line_width=1.5, legend_label='Min area')
        p.line([c.max_piece_area] * 2, [0, mc], line_color='#e67e22',
               line_dash='dashed', line_width=1.5, legend_label='Max area')
        p.line([0, ma], [c.max_complexity] * 2, line_color='#9b59b6',
               line_dash='dashed', line_width=1.5,
               legend_label=f'Max complexity ({c.tool_profile.name})')

        p.legend.location = "top_right"
        p.legend.label_text_font_size = "9pt"
        p.hover.tooltips = [
            ("Piece #", "@piece_id"), ("Area", "@area{0.0}"),
            ("Complexity", "@complexity{0.0}"),
            ("Status", "@status")]
        return p

    def _create_construction_info_panel(self) -> Div:
        c = self.constraints
        tool = c.tool_profile
        cap = lambda v: "&#10003;" if v else "&#10007;"
        ccol = lambda v: "#27ae60" if v else "#e74c3c"

        html = f"""
        <div style="font-family:sans-serif; padding:15px; background:#e3f2fd;
                    border-radius:8px; border:1px solid #90caf9; margin:5px;">
            <h3 style="color:#1565c0; margin-top:0;">
                &#128295; {tool.name}</h3>
            <p style="color:#555; font-size:0.9em;">{tool.description}</p>
            <table style="border-collapse:collapse; width:100%;">
                <tr style="background:#bbdefb;">
                    <td style="padding:3px 8px;">Max concavity</td>
                    <td><b>{tool.max_concavity:.0%}</b></td></tr>
                <tr><td style="padding:3px 8px;">Min angle</td>
                    <td><b>{tool.min_interior_angle:.0f}\u00b0</b></td></tr>
                <tr style="background:#bbdefb;">
                    <td style="padding:3px 8px;">Min inside radius</td>
                    <td><b>{tool.min_inside_radius_in:.3f}"</b></td></tr>
                <tr><td style="padding:3px 8px;">Min neck</td>
                    <td><b>{tool.min_neck_width_in:.3f}"</b></td></tr>
                <tr style="background:#bbdefb;">
                    <td style="padding:3px 8px;">Max complexity</td>
                    <td><b>{tool.max_complexity:.0f}</b></td></tr>
            </table>
            <table style="margin-top:8px;">
                <tr><td style="color:{ccol(tool.can_inside_curves)}">
                    {cap(tool.can_inside_curves)}</td>
                    <td>Inside curves</td></tr>
                <tr><td style="color:{ccol(tool.can_deep_concavity)}">
                    {cap(tool.can_deep_concavity)}</td>
                    <td>Deep concavity</td></tr>
                <tr><td style="color:{ccol(tool.can_sharp_inside_corners)}">
                    {cap(tool.can_sharp_inside_corners)}</td>
                    <td>Sharp inside corners</td></tr>
                <tr><td style="color:{ccol(tool.can_narrow_necks)}">
                    {cap(tool.can_narrow_necks)}</td>
                    <td>Narrow necks</td></tr>
            </table>
        </div>"""

        if c.is_fused:
            ppi = c.ppi
            html += f"""
            <div style="font-family:sans-serif; padding:15px;
                        background:#e8f5e9; border-radius:8px;
                        border:1px solid #a5d6a7; margin:5px;">
                <h3 style="color:#2e7d32; margin-top:0;">Fused Glass</h3>
                <table>
                    <tr><td style="padding:3px 8px;">Panel</td>
                        <td>{self.width/ppi:.2f}" x
                            {self.height/ppi:.2f}"</td></tr>
                    <tr><td style="padding:3px 8px;">Overlap</td>
                        <td>{c.fuse_overlap_in:.3f}"</td></tr>
                </table>
            </div>"""
        else:
            profile = c.came_profile
            html += f"""
            <div style="font-family:sans-serif; padding:15px;
                        background:#fff8e1; border-radius:8px;
                        border:1px solid #ffe082; margin:5px;">
                <h3 style="color:#e65100; margin-top:0;">Construction</h3>
                <table style="width:100%;">
                    <tr style="background:#fff3e0;">
                        <td style="padding:3px 8px;"><b>Interior</b></td>
                        <td>{profile.name}</td></tr>
                    <tr><td style="padding:3px 8px;">Heart</td>
                        <td>{profile.heart_width:.4f}"
                            ({c.heart_width_px:.1f}px)</td></tr>
                    <tr style="background:#fff3e0;">
                        <td style="padding:3px 8px;">Scissor</td>
                        <td>{profile.scissor_allowance:.4f}"/side</td></tr>
                    <tr><td style="padding:3px 8px;">Visible</td>
                        <td>{profile.total_visible_width:.4f}"</td></tr>"""

            if c.border_came_profile:
                bp = c.border_came_profile
                html += f"""
                    <tr style="background:#fff3e0;">
                        <td style="padding:3px 8px;"><b>Border</b></td>
                        <td>{bp.name}</td></tr>
                    <tr><td style="padding:3px 8px;">Channel</td>
                        <td>{bp.channel_depth:.4f}"</td></tr>"""

            if self.zinc_frame:
                zf = self.zinc_frame
                cuts = zf.mitre_cut_lengths
                html += f"""
                    <tr><td colspan="2" style="padding:8px 0 2px;">
                        <b style="color:#e65100;">
                        &#9634; Frame</b></td></tr>
                    <tr style="background:#fff3e0;">
                        <td style="padding:3px 8px;">Visible</td>
                        <td>{zf.panel_width_in:.2f}" x
                            {zf.panel_height_in:.2f}"</td></tr>
                    <tr><td style="padding:3px 8px;">Outer</td>
                        <td>{zf.outer_width_in:.2f}" x
                            {zf.outer_height_in:.2f}"</td></tr>
                    <tr style="background:#fff3e0;">
                        <td style="padding:3px 8px;">Top/Bottom</td>
                        <td>{cuts['top']:.3f}" ea</td></tr>
                    <tr><td style="padding:3px 8px;">Left/Right</td>
                        <td>{cuts['left']:.3f}" ea</td></tr>
                    <tr style="background:#fff3e0;">
                        <td style="padding:3px 8px;">
                            <b>Total zinc</b></td>
                        <td><b>{cuts['total_linear']:.2f}"</b>
                            (buy {cuts['total_linear']+2:.0f}"+)
                        </td></tr>"""

            html += "</table></div>"

        return Div(text=html, width=450)

    def _create_stats_panel(self) -> Div:
        total = len(self.pieces)
        feasible = sum(1 for p in self.pieces if p.is_feasible)
        areas = [p.area for p in self.pieces]
        complexities = [p.complexity for p in self.pieces]
        ppi = self.constraints.ppi
        unique_colors = sorted(set(p.color_hex for p in self.pieces))

        swatches = "".join(
            f'<span style="display:inline-block;width:18px;height:18px;'
            f'background:{c};border:1px solid #333;margin:1px;'
            f'border-radius:2px;" title="{c}"></span>'
            for c in unique_colors[:60]
        )
        if len(unique_colors) > 60:
            swatches += f" +{len(unique_colors) - 60} more"

        issues = {}
        for p in self.pieces:
            for iss in p.feasibility_issues:
                k = iss.split(":")[0]
                issues[k] = issues.get(k, 0) + 1
        iss_html = "".join(
            f"<li><b>{k}</b>: {v}</li>"
            for k, v in sorted(issues.items(), key=lambda x: -x[1]))
        if not iss_html:
            iss_html = "<li>All feasible &#10003;</li>"

        html = f"""
        <div style="font-family:sans-serif; padding:15px;
                    background:#f8f9fa; border-radius:8px;
                    border:1px solid #dee2e6; margin-top:10px;">
            <h3 style="color:#2c3e50; margin-top:0;">
                &#128202; {self.project_name} \u2014 Statistics</h3>
            <div style="display:flex; gap:40px; flex-wrap:wrap;">
                <div><h4>Pieces</h4><table>
                    <tr><td style="padding:2px 10px;">Total:</td>
                        <td><b>{total}</b></td></tr>
                    <tr><td style="padding:2px 10px;">Feasible:</td>
                        <td style="color:#27ae60;">
                            <b>{feasible}</b></td></tr>
                    <tr><td style="padding:2px 10px;">Issues:</td>
                        <td style="color:#e74c3c;">
                            <b>{total - feasible}</b></td></tr>
                </table></div>
                <div><h4>Area</h4><table>
                    <tr><td style="padding:2px 10px;">Min:</td>
                        <td><b>{min(areas)/ppi**2:.2f}
                            in\u00b2</b></td></tr>
                    <tr><td style="padding:2px 10px;">Max:</td>
                        <td><b>{max(areas)/ppi**2:.2f}
                            in\u00b2</b></td></tr>
                    <tr><td style="padding:2px 10px;">Median:</td>
                        <td><b>{np.median(areas)/ppi**2:.2f}
                            in\u00b2</b></td></tr>
                </table></div>
                <div><h4>Complexity</h4><table>
                    <tr><td style="padding:2px 10px;">Min:</td>
                        <td><b>{min(complexities):.1f}</b></td></tr>
                    <tr><td style="padding:2px 10px;">Max:</td>
                        <td><b>{max(complexities):.1f}</b></td></tr>
                    <tr><td style="padding:2px 10px;">Median:</td>
                        <td><b>{np.median(complexities):.1f}</b></td></tr>
                </table></div>
                <div><h4>Issues</h4>
                    <ul style="margin:0;padding-left:20px;">
                        {iss_html}</ul></div>
            </div>
            <div style="margin-top:10px;">
                <h4>Palette ({len(unique_colors)} colors)</h4>
                {swatches}
            </div>
        </div>"""
        return Div(text=html, sizing_mode="stretch_width")


# =============================================================================
# SVG & Workshop Export
# =============================================================================

class PatternExporter:
    """Export cut patterns in fabrication-ready formats."""

    def __init__(self, pieces: List[GlassPiece], width: int, height: int,
                 constraints: FabricationConstraints,
                 project_name: str = "untitled",
                 zinc_frame: Optional[ZincFrame] = None):
        self.pieces = pieces
        self.width = width
        self.height = height
        self.constraints = constraints
        self.project_name = project_name
        self.zinc_frame = zinc_frame

        self.geometries = [
            PieceGeometry(p.polygon, constraints,
                          self._is_border(p.polygon))
            for p in pieces
        ]

    def _is_border(self, polygon: Polygon) -> bool:
        if self.constraints.is_fused:
            return False
        b = polygon.bounds
        m = 2.0
        return (b[0] < m or b[1] < m or
                b[2] > self.width - m or b[3] > self.height - m)

    def export_svg(self, path: str, scale: float = 1.0,
                   show_layers: str = "all"):
        sw, sh = self.width * scale, self.height * scale
        c = self.constraints

        if c.is_fused and show_layers == "all":
            show_layers = "fused"

        lines = [
            f'<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg"',
            f'  width="{sw:.1f}" height="{sh:.1f}" '
            f'viewBox="0 0 {sw:.1f} {sh:.1f}">',
            f'  <!-- Project: {self.project_name} | '
            f'Mode: {c.construction_mode.value} | '
            f'Tools: {c.tool_profile.name} -->',
        ]
        if not c.is_fused:
            lines.append(
                f'  <!-- Came: {c.came_profile.name} | '
                f'Heart: {c.came_profile.heart_width}" -->')
        lines.append(
            f'  <rect width="{sw}" height="{sh}" fill="#f5f5f0"/>')

        def pts(poly):
            return " ".join(
                f"{x * scale:.1f},{y * scale:.1f}"
                for x, y in poly.exterior.coords)

        if show_layers == "fused":
            lines.append(f'  <g id="fused-pieces">')
            for i, piece in enumerate(self.pieces):
                g = self.geometries[i]
                s = "#000" if piece.is_feasible else "#f00"
                lines.append(
                    f'    <polygon points="{pts(g.glass_polygon)}" '
                    f'fill="{piece.color_hex}" fill-opacity="0.85" '
                    f'stroke="{s}" stroke-width="{0.8 * scale}"/>')
            lines.append('  </g>')
            lines.append(f'  <g id="labels" font-family="Arial">')
            for p in self.pieces:
                fs = max(6, min(p.area ** 0.3, 12)) * scale
                lines.append(
                    f'    <text x="{p.centroid[0]*scale:.1f}" '
                    f'y="{p.centroid[1]*scale:.1f}" '
                    f'text-anchor="middle" dominant-baseline="middle" '
                    f'font-size="{fs:.1f}" fill="#333">{p.id}</text>')
            lines.append('  </g>')
        else:
            if show_layers in ("all", "design"):
                lines.append(
                    f'  <g id="design-lines" stroke="#999" '
                    f'stroke-width="{0.5 * scale}" '
                    f'fill="none" opacity="0.5">')
                for p in self.pieces:
                    lines.append(
                        f'    <polygon points="{pts(p.polygon)}"/>')
                lines.append('  </g>')
            if show_layers in ("all", "cut", "workshop"):
                lines.append(
                    f'  <g id="cut-lines" stroke="#000" '
                    f'stroke-width="{1.0 * scale}" fill="none">')
                for i, g in enumerate(self.geometries):
                    lines.append(
                        f'    <polygon points="{pts(g.cut_polygon)}" '
                        f'fill="{self.pieces[i].color_hex}" '
                        f'fill-opacity="0.3"/>')
                lines.append('  </g>')
            if show_layers in ("all", "glass"):
                lines.append(
                    f'  <g id="glass-edges" stroke="#06c" '
                    f'stroke-width="{0.8 * scale}" '
                    f'stroke-dasharray="{2 * scale},{2 * scale}" '
                    f'fill="none">')
                for g in self.geometries:
                    lines.append(
                        f'    <polygon '
                        f'points="{pts(g.glass_polygon)}"/>')
                lines.append('  </g>')
            if show_layers in ("all", "workshop"):
                lines.append(
                    f'  <g id="color-fill" opacity="0.85">')
                for i, g in enumerate(self.geometries):
                    p = self.pieces[i]
                    lines.append(
                        f'    <polygon '
                        f'points="{pts(g.glass_polygon)}" '
                        f'fill="{p.color_hex}" stroke="#333" '
                        f'stroke-width='
                        f'"{c.heart_width_px * scale:.1f}" '
                        f'stroke-linejoin="round"/>')
                lines.append('  </g>')
                lines.append(
                    f'  <g id="labels" font-family="Arial">')
                for p in self.pieces:
                    fs = max(6, min(p.area ** 0.3, 12)) * scale
                    lines.append(
                        f'    <text x="{p.centroid[0]*scale:.1f}" '
                        f'y="{p.centroid[1]*scale:.1f}" '
                        f'text-anchor="middle" '
                        f'dominant-baseline="middle" '
                        f'font-size="{fs:.1f}" '
                        f'fill="#333">{p.id}</text>')
                    lines.append(
                        f'    <text x="{p.centroid[0]*scale:.1f}" '
                        f'y="{p.centroid[1]*scale + fs:.1f}" '
                        f'text-anchor="middle" '
                        f'dominant-baseline="middle" '
                        f'font-size="{fs * 0.6:.1f}" '
                        f'fill="#666">{p.color_hex}</text>')
                lines.append('  </g>')

        # Scale bar
        ry = sh - 15 * scale
        oi = c.ppi * scale
        lines.append(f'  <g id="scale">')
        lines.append(
            f'    <line x1="{10 * scale}" y1="{ry}" '
            f'x2="{10 * scale + oi}" y2="{ry}" '
            f'stroke="#333" stroke-width="{1.5 * scale}"/>')
        lines.append(
            f'    <text x="{10 * scale + oi / 2}" '
            f'y="{ry - 4 * scale}" text-anchor="middle" '
            f'font-size="{8 * scale}" '
            f'font-family="Arial">1 inch</text>')
        lines.append(f'  </g>')
        lines.append('</svg>')

        with open(path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"  SVG ({show_layers}): {path}")

    def export_cut_list(self, path: str):
        c = self.constraints
        data = {
            "project": self.project_name,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "construction_mode": c.construction_mode.value,
            "tool_profile": {
                "name": c.tool_profile.name,
                "max_concavity": c.tool_profile.max_concavity,
                "min_interior_angle": c.tool_profile.min_interior_angle,
                "min_inside_radius_in": c.tool_profile.min_inside_radius_in,
                "min_neck_width_in": c.tool_profile.min_neck_width_in,
                "max_complexity": c.tool_profile.max_complexity,
            },
            "panel_width_in": self.width / c.ppi,
            "panel_height_in": self.height / c.ppi,
            "ppi": c.ppi,
            "total_pieces": len(self.pieces),
        }

        if not c.is_fused:
            data["came_profile"] = {
                "name": c.came_profile.name,
                "heart_width_in": c.came_profile.heart_width,
                "scissor_allowance_in": c.came_profile.scissor_allowance,
            }
            if self.zinc_frame:
                data["zinc_frame"] = self.zinc_frame.to_dict()
        else:
            data["fused"] = {
                "overlap_in": c.fuse_overlap_in,
                "margin_in": c.fuse_border_margin_in}

        data["pieces"] = [{
            "id": p.id, "color_hex": p.color_hex,
            "color_rgb": list(p.color_rgb),
            "area_in2": round(p.area / c.ppi ** 2, 3),
            "perimeter_in": round(p.perimeter / c.ppi, 3),
            "complexity": round(p.complexity, 2),
            "feasible": p.is_feasible,
            "issues": p.feasibility_issues,
            "is_border": self.geometries[i].is_border,
            "design_vertices": [
                [round(x, 1), round(y, 1)]
                for x, y in p.polygon.exterior.coords],
            "cut_vertices": [
                [round(x, 1), round(y, 1)]
                for x, y in
                self.geometries[i].cut_polygon.exterior.coords],
            "glass_vertices": [
                [round(x, 1), round(y, 1)]
                for x, y in
                self.geometries[i].glass_polygon.exterior.coords],
            "centroid": [round(p.centroid[0], 1),
                         round(p.centroid[1], 1)],
        } for i, p in enumerate(self.pieces)]

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  Cut list: {path}")

    def export_workshop_sheet(self, path: str):
        c = self.constraints
        ppi = c.ppi
        tool = c.tool_profile
        display_name = self.project_name.replace("_", " ").upper()

        tl = [
            f"{display_name} \u2014 CUT SHEET",
            "=" * 85,
            f"Date:  {time.strftime('%Y-%m-%d %H:%M')}",
            f"Mode:  "
            f"{c.construction_mode.value.replace('_', ' ').title()}",
            f"Tools: {tool.name}",
            f"       Concavity: {tool.max_concavity:.0%} | "
            f"Angle: {tool.min_interior_angle:.0f}\u00b0 | "
            f"Neck: {tool.min_neck_width_in:.3f}\"",
        ]

        if not c.is_fused:
            pr = c.came_profile
            tl.extend([
                f"Came:  {pr.name}",
                f"       Heart: {pr.heart_width}\" | "
                f"Scissor: {pr.scissor_allowance}\"/side",
            ])
        else:
            tl.append(f"Overlap: {c.fuse_overlap_in}\"")

        tl.extend([
            f"Panel: {self.width / ppi:.1f}\" x "
            f"{self.height / ppi:.1f}\"",
            f"Scale: {ppi:.0f} px/in",
            f"Pieces: {len(self.pieces)}",
        ])

        if self.zinc_frame:
            tl.extend(["", self.zinc_frame.frame_pieces_description])

        tl.extend([
            "",
            f"{'#':>4} {'Color':>8} {'Area':>8} {'Perim':>8} "
            f"{'Cmplx':>7} {'Bdr':>4} {'OK':>4} {'Notes'}",
            "-" * 85,
        ])

        for i, piece in enumerate(self.pieces):
            g = self.geometries[i]
            tl.append(
                f"{piece.id:>4} {piece.color_hex:>8} "
                f"{piece.area / ppi ** 2:>7.2f}\" "
                f"{piece.perimeter / ppi:>7.2f}\" "
                f"{piece.complexity:>7.1f} "
                f"{'Y' if g.is_border else ' ':>4} "
                f"{'Y' if piece.is_feasible else 'N':>4} "
                f"{'; '.join(piece.feasibility_issues)}")

        tl.append("")
        tl.append("=" * 85)
        feasible = sum(1 for p in self.pieces if p.is_feasible)
        tl.append(f"Feasible: {feasible}/{len(self.pieces)}")

        if not c.is_fused:
            tp = sum(p.perimeter for p in self.pieces) / 2
            tl.append(
                f"Est. interior came: ~{tp / ppi:.1f}\" "
                f"(buy {tp / ppi * 1.15:.0f}\"+)")

        with open(path, 'w') as f:
            f.write('\n'.join(tl))
        print(f"  Workshop sheet: {path}")

    def export_project_manifest(self, path: str, image_path: str,
                                method: str, elapsed: float):
        c = self.constraints
        manifest = {
            "project": self.project_name,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(elapsed, 2),
            "source_image": str(
                Path(image_path).expanduser().resolve()
                if not Path(image_path).is_absolute()
                else Path(image_path)),
            "settings": {
                "method": method,
                "mode": c.construction_mode.value,
                "tools": c.tool_profile.name,
                "came": (c.came_profile.name
                         if not c.is_fused else None),
                "border_came": (c.border_came_profile.name
                                if c.has_border_frame else None),
                "target_pieces": len(self.pieces),
                "ppi": c.ppi,
                "min_area_in2": c.effective_min_area_in2,
                "max_area_in2": c.max_piece_area_in2,
                "fuse_overlap_in": (c.fuse_overlap_in
                                    if c.is_fused else None),
            },
            "results": {
                "total_pieces": len(self.pieces),
                "feasible_pieces": sum(
                    1 for p in self.pieces if p.is_feasible),
                "unique_colors": len(
                    set(p.color_hex for p in self.pieces)),
                "panel_width_in": self.width / c.ppi,
                "panel_height_in": self.height / c.ppi,
            },
            "files": {
                "visualization": f"{self.project_name}.html",
                "cut_list": f"{self.project_name}_cut_list.json",
                "workshop_sheet": f"{self.project_name}_workshop.txt",
                "svg_files": [],
            }
        }

        if c.is_fused:
            manifest["files"]["svg_files"] = [
                f"{self.project_name}_cut.svg"]
        else:
            manifest["files"]["svg_files"] = [
                f"{self.project_name}_all_layers.svg",
                f"{self.project_name}_workshop.svg",
                f"{self.project_name}_cut_only.svg",
            ]

        if self.zinc_frame:
            manifest["zinc_frame"] = self.zinc_frame.to_dict()

        with open(path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"  Manifest: {path}")


# =============================================================================
# Main Pipeline
# =============================================================================

def generate_stained_glass(
    image_path: str,
    project_name: str = "untitled",
    num_pieces: int = 150,
    method: str = "slic",
    mode: str = "lead",
    tools: str = "standard",
    came_key: str = "lead_3_16",
    border_came_key: str = "zinc_1_4",
    output_dir: str = "./projects",
    min_piece_area_in2: Optional[float] = None,
    max_piece_area_in2: float = 36.0,
    ppi: float = 50.0,
    max_dimension: int = 800,
    fuse_overlap_in: float = 0.0,
) -> Tuple[List[GlassPiece], ImageAnalyzer]:
    """Full pipeline: image -> tessellation -> validation -> output."""

    start_time = time.time()

    safe_name = "".join(
        c if c.isalnum() or c in "-_" else "_"
        for c in project_name)
    project_dir = Path(output_dir) / safe_name
    project_dir.mkdir(parents=True, exist_ok=True)

    if tools not in TOOL_PROFILES:
        raise ValueError(
            f"Unknown tools: '{tools}'. "
            f"Available: {', '.join(TOOL_PROFILES.keys())}")
    tool_profile = TOOL_PROFILES[tools]

    if mode == "fused":
        construction_mode = ConstructionMethod.FUSED
        came_profile = CAME_PROFILES["fused_none"]
        border_profile = None
    elif mode == "foil":
        construction_mode = ConstructionMethod.COPPER_FOIL
        if came_key not in CAME_PROFILES:
            raise ValueError(f"Unknown came: '{came_key}'")
        came_profile = CAME_PROFILES[came_key]
        border_profile = CAME_PROFILES.get(border_came_key)
    else:
        construction_mode = ConstructionMethod.LEAD_CAME
        if came_key not in CAME_PROFILES:
            raise ValueError(f"Unknown came: '{came_key}'")
        came_profile = CAME_PROFILES[came_key]
        border_profile = CAME_PROFILES.get(border_came_key)

    print(f"\n{'=' * 60}")
    print(f"Project: {project_name}")
    print(f"{'=' * 60}")

    print(f"\nLoading: {image_path}")
    analyzer = ImageAnalyzer(image_path, max_dimension=max_dimension)
    print(f"  Image: {analyzer.width} x {analyzer.height}")
    print(f"  Panel: {analyzer.width / ppi:.1f}\" x "
          f"{analyzer.height / ppi:.1f}\"")
    print(f"  Mode:  {mode} | Tools: {tool_profile.name}")
    if not mode == "fused":
        print(f"  Came:  {came_profile.name}")

    constraints = FabricationConstraints(
        tool_profile=tool_profile,
        min_piece_area_in2=min_piece_area_in2,
        max_piece_area_in2=max_piece_area_in2,
        construction_mode=construction_mode,
        came_profile=came_profile,
        border_came_profile=border_profile,
        ppi=ppi, fuse_overlap_in=fuse_overlap_in,
    )

    zinc_frame = None
    if constraints.has_border_frame and border_profile:
        zinc_frame = ZincFrame(
            border_profile,
            analyzer.width / ppi, analyzer.height / ppi)
        print(f"  Frame: {zinc_frame.outer_width_in:.2f}\" x "
              f"{zinc_frame.outer_height_in:.2f}\"")
        print(f"  Zinc:  "
              f"{zinc_frame.mitre_cut_lengths['total_linear']:.2f}\"")

    print(f"\nTessellating ({method}, target {num_pieces})...")
    tessellator = WatertightTessellator(analyzer, constraints)
    pieces = tessellator.tessellate(num_pieces, method=method)

    feasible = sum(1 for p in pieces if p.is_feasible)
    print(f"\n  Result: {len(pieces)} pieces, {feasible} feasible")

    pn = safe_name
    print(f"\nWriting to: {project_dir}/")

    print(f"\nTessellating ({method}, target {num_pieces})...")
    tessellator = WatertightTessellator(analyzer, constraints)
    pieces = tessellator.tessellate(num_pieces, method=method)

    feasible = sum(1 for p in pieces if p.is_feasible)
    print(f"\n  Result: {len(pieces)} pieces, {feasible} feasible")

    # ---- DIAGNOSTIC START ----
    from shapely.geometry import Point as DiagPoint

    test_points = [
        DiagPoint(230, 70), DiagPoint(240, 80), DiagPoint(235, 75),
        DiagPoint(225, 65), DiagPoint(245, 90), DiagPoint(250, 75),
        DiagPoint(230, 60), DiagPoint(240, 65), DiagPoint(235, 85),
    ]

    print("\n=== Which piece contains each test point? ===")
    for pt in test_points:
        found = False
        for piece in pieces:
            if piece.polygon.contains(pt):
                print(f"  ({pt.x}, {pt.y}) -> Piece #{piece.id}")
                found = True
                break
        if not found:
            print(f"  ({pt.x}, {pt.y}) -> *** GAP - NO PIECE ***")

    print("\n=== Remaining gaps ===")
    image_rect = box(0, 0, analyzer.width, analyzer.height)
    all_union = unary_union([p.polygon for p in pieces])
    gaps = image_rect.difference(all_union)

    if gaps.is_empty:
        print("  No gaps found")
    else:
        print(f"  Total gap area: {gaps.area:.1f} px\u00b2")
        if gaps.geom_type == 'Polygon':
            gap_list = [gaps]
        elif gaps.geom_type == 'MultiPolygon':
            gap_list = list(gaps.geoms)
        else:
            gap_list = [g for g in gaps.geoms
                        if hasattr(g, 'area') and g.area > 0]

        for i, gap in enumerate(gap_list):
            if gap.area > 1:
                print(f"\n  Gap {i}:")
                print(f"    Area: {gap.area:.1f} px\u00b2 "
                      f"({gap.area / ppi ** 2:.3f} in\u00b2)")
                print(f"    Bounds: "
                      f"{tuple(round(b, 1) for b in gap.bounds)}")
                print(f"    Centroid: ({gap.centroid.x:.0f}, "
                      f"{gap.centroid.y:.0f})")
                print(f"    Bordered by:")
                gap_buffered = gap.buffer(2)
                for piece in pieces:
                    if piece.polygon.intersects(gap_buffered):
                        print(f"      #{piece.id} "
                              f"(area={piece.area / ppi ** 2:.2f} in\u00b2)")
    # ---- DIAGNOSTIC END ----

    pn = safe_name
    print(f"\nWriting to: {project_dir}/")

    viz = StainedGlassVisualizer(
        analyzer, pieces, constraints, project_name, zinc_frame)
    viz.create_visualization(str(project_dir / f"{pn}.html"))

    exporter = PatternExporter(
        pieces, analyzer.width, analyzer.height,
        constraints, project_name, zinc_frame)

    if constraints.is_fused:
        exporter.export_svg(
            str(project_dir / f"{pn}_cut.svg"),
            show_layers="fused")
    else:
        exporter.export_svg(
            str(project_dir / f"{pn}_all_layers.svg"),
            show_layers="all")
        exporter.export_svg(
            str(project_dir / f"{pn}_workshop.svg"),
            show_layers="workshop")
        exporter.export_svg(
            str(project_dir / f"{pn}_cut_only.svg"),
            show_layers="cut")

    exporter.export_cut_list(
        str(project_dir / f"{pn}_cut_list.json"))
    exporter.export_workshop_sheet(
        str(project_dir / f"{pn}_workshop.txt"))

    elapsed = time.time() - start_time
    exporter.export_project_manifest(
        str(project_dir / f"{pn}_manifest.json"),
        image_path, method, elapsed)

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"{'=' * 60}\n")

    return pieces, analyzer


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert an image into a glass cutting pattern.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Tool profiles:
{chr(10).join(f'  {k:30s} {v.description[:55]}'
              for k, v in TOOL_PROFILES.items())}

Came profiles:
{chr(10).join(f'  {k:25s} {v.name}'
              for k, v in CAME_PROFILES.items())}

Examples:
  python stained_glass.py -i photo.jpg -p "rose_window" -n 150
  python stained_glass.py -i photo.jpg -p "kitchen" -t ring_saw --mode foil -m hybrid
  python stained_glass.py -i photo.jpg -p "fused_dish" --mode fused -t ring_saw -n 80
  python stained_glass.py -i photo.jpg -p "first_panel" -n 40 --came lead_3_8 --min-area 3.0
        """
    )
    parser.add_argument("--image", "-i", required=True,
                        help="Input image")
    parser.add_argument("--project", "-p", default="untitled",
                        help="Project name (default: untitled)")
    parser.add_argument("--pieces", "-n", type=int, default=150,
                        help="Target pieces (150)")
    parser.add_argument("--method", "-m", default="slic",
                        choices=["voronoi", "slic", "hybrid"],
                        help="Tessellation (slic)")
    parser.add_argument("--mode", default="lead",
                        choices=["lead", "foil", "fused"],
                        help="Construction mode (lead)")
    parser.add_argument("--tools", "-t", default="standard",
                        choices=list(TOOL_PROFILES.keys()),
                        help="Tool profile (standard)")
    parser.add_argument("--came", "-c", default="lead_3_16",
                        choices=list(CAME_PROFILES.keys()),
                        help="Interior came (lead_3_16)")
    parser.add_argument("--border-came", default="zinc_1_4",
                        choices=list(CAME_PROFILES.keys()),
                        help="Border came (zinc_1_4)")
    parser.add_argument("--output", "-o", default="./projects",
                        help="Base output dir (./projects)")
    parser.add_argument("--min-area", type=float, default=None,
                        help="Min piece area in\u00b2 (from tool)")
    parser.add_argument("--max-area", type=float, default=36.0,
                        help="Max piece area in\u00b2 (36)")
    parser.add_argument("--ppi", type=float, default=50.0,
                        help="Pixels per inch (50)")
    parser.add_argument("--max-dim", type=int, default=800,
                        help="Max image dimension (800)")
    parser.add_argument("--fuse-overlap", type=float, default=0.0,
                        help="Fused piece overlap inches (0)")

    args = parser.parse_args()

    generate_stained_glass(
        image_path=args.image,
        project_name=args.project,
        num_pieces=args.pieces,
        method=args.method,
        mode=args.mode,
        tools=args.tools,
        came_key=args.came,
        border_came_key=args.border_came,
        output_dir=args.output,
        min_piece_area_in2=args.min_area,
        max_piece_area_in2=args.max_area,
        ppi=args.ppi,
        max_dimension=args.max_dim,
        fuse_overlap_in=args.fuse_overlap,
    )


if __name__ == "__main__":
    main()
