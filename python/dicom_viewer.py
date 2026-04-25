"""
DICOM Medical Image Viewer
===========================
A Bokeh-based interactive medical image viewer supporting X-Ray, CT, MRI,
and Ultrasound DICOM files.

Conda env: pydicom
Requirements: pydicom (conda); pylibjpeg[all] (pip); scikit-image; bokeh; pyyaml

Invoke:
    bokeh serve dicom_viewer.py --args --app_config "/Volumes/Data/Home/dicom_viewer.yaml"

View in browser at:
    localhost:5006/dicom_viewer
"""

import os
import logging
import numpy as np
import pydicom
from tornado.ioloop import IOLoop
from skimage import transform
import yaml
import argparse
from pathlib import Path
import pylibjpeg
import re
from dataclasses import dataclass, field
from typing import Optional

from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row, layout
from bokeh.models import (
    ColumnDataSource, LinearColorMapper, ColorBar, HoverTool,
    Button, Slider, Div, TapTool, Select, RadioButtonGroup,
    Toggle, TextInput, Range1d
)
from bokeh.palettes import Greys256

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
XRAY_SCALE = 0.5
CT_SCALE = 1.5
MRI_SCALE = 4.0
US_SCALE = 1.5

MRI_GAMMA = 2.0
US_GAMMA = 2.0
DEFAULT_GAMMA = 1.0
DEFAULT_WINDOW = 1.0

GAMMA_SLIDER_MAX = 10.0
WINDOW_SLIDER_MAX = 2.0
MAX_LOG_MESSAGES = 10
DEFAULT_REFRESH_MS = 500.0
NORMAL_THRESHOLD = 0.95
ANIMATION_SLICE_RESET = 0

# Window/Level defaults (used when no DICOM presets available)
WL_CENTER_DEFAULT = 2000.0
WL_WIDTH_DEFAULT = 4000.0
WL_CENTER_SLIDER_MIN = -2000.0
WL_CENTER_SLIDER_MAX = 20000.0
WL_WIDTH_SLIDER_MIN = 1.0
WL_WIDTH_SLIDER_MAX = 40000.0
WL_SLIDER_STEP = 10.0

MODALITY_XRAY = "X-Ray"
MODALITY_CT = "CT"
MODALITY_MRI = "MRI"
MODALITY_US = "US"

WL_MANUAL_LABEL = "Manual"

logger = logging.getLogger("dicom_viewer")
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class ViewerConfig:
    """Holds application configuration loaded from YAML."""
    debug: bool = False
    gamma_def: float = DEFAULT_GAMMA
    window_def: float = DEFAULT_WINDOW
    starter_images: str = ""
    data_db: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "ViewerConfig":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(
            debug=data.get("debug", False),
            gamma_def=data.get("gamma_def", DEFAULT_GAMMA),
            window_def=data.get("window_def", DEFAULT_WINDOW),
            starter_images=data.get("starter_images", ""),
            data_db=data.get("data_db", {}),
        )


# ---------------------------------------------------------------------------
# Image processing helpers
# ---------------------------------------------------------------------------
class ImageProcessor:
    """Stateless image processing utilities for DICOM images."""

    @staticmethod
    def apply_photometric(image: np.ndarray, ds: pydicom.Dataset) -> np.ndarray:
        """Invert grayscale only for MONOCHROME1 images."""
        photometric = ds.get("PhotometricInterpretation", "MONOCHROME2")
        if photometric == "MONOCHROME1":
            return np.max(image) - image
        return image.copy()

    @staticmethod
    def apply_rescale(image: np.ndarray, ds: pydicom.Dataset) -> np.ndarray:
        """Apply DICOM RescaleSlope and RescaleIntercept (e.g. CT Hounsfield)."""
        slope = getattr(ds, "RescaleSlope", None)
        intercept = getattr(ds, "RescaleIntercept", None)
        if slope is not None and intercept is not None:
            image = image.astype(np.float64) * float(slope) + float(intercept)
        return image

    @staticmethod
    def ensure_2d(image: np.ndarray) -> np.ndarray:
        """Ensure image is 2-D (handle multi-frame or RGB)."""
        if image.ndim == 3:
            if image.shape[2] <= 4:
                return np.mean(image, axis=2).astype(image.dtype)
            else:
                return image[0]
        return image

    @staticmethod
    def perform_gamma(image: np.ndarray, gamma: float,
                      original_dtype: np.dtype) -> np.ndarray:
        """Apply gamma correction to image."""
        img = image.astype(np.float64)
        max_val = np.max(img)
        if max_val == 0:
            return image
        img = img / max_val
        img = img ** gamma
        if np.issubdtype(original_dtype, np.integer):
            img = (img * np.iinfo(original_dtype).max).astype(original_dtype)
        else:
            img = (img * max_val).astype(original_dtype)
        return img

    @staticmethod
    def extract_wl_presets(ds: pydicom.Dataset) -> list:
        """
        Extract window/level presets from DICOM metadata.

        Returns a list of dicts: [{"center": float, "width": float, "name": str}, ...]
        Returns empty list if no presets found.
        """
        presets = []
        try:
            centers = ds[0x0028, 0x1050].value
            widths = ds[0x0028, 0x1051].value

            # Normalize to lists (can be single value or MultiValue)
            if not isinstance(centers, (list, pydicom.multival.MultiValue)):
                centers = [centers]
                widths = [widths]

            # Try to get descriptive names
            try:
                names = ds[0x0028, 0x1055].value
                if not isinstance(names, (list, pydicom.multival.MultiValue)):
                    names = [names]
            except (KeyError, AttributeError):
                names = [f"Preset {i + 1}" for i in range(len(centers))]

            # Pad names if fewer than centers
            while len(names) < len(centers):
                names.append(f"Preset {len(names) + 1}")

            for c, w, n in zip(centers, widths, names):
                presets.append({
                    "center": float(c),
                    "width": float(w),
                    "name": str(n),
                })
        except (KeyError, AttributeError):
            pass

        return presets

    @staticmethod
    def rotated_rectangle_properties(corners):
        """
        Calculate rotation angle and bounding box from 4 corner points.

        Parameters
        ----------
        corners : list of [x, y] pairs

        Returns
        -------
        tuple : (rotation_angle_degrees, min_x, max_x, min_y, max_y, center)
        """
        corners = np.array(corners)
        center = np.mean(corners, axis=0)

        max_dist = 0
        corner1_index, corner2_index = 0, 1
        for i in range(4):
            for j in range(i + 1, 4):
                dist = np.linalg.norm(corners[i] - corners[j])
                if dist > max_dist:
                    max_dist = dist
                    corner1_index = i
                    corner2_index = j

        corner1 = corners[corner1_index]
        corner2 = corners[corner2_index]

        dx = corner2[0] - corner1[0]
        dy = corner2[1] - corner1[1]
        rotation_angle_degrees = np.degrees(np.arctan2(dy, dx))

        if rotation_angle_degrees > 90:
            rotation_angle_degrees -= 180
        elif rotation_angle_degrees < -90:
            rotation_angle_degrees += 180

        angle_rad = np.radians(rotation_angle_degrees)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad),  np.cos(angle_rad)],
        ])

        rotated_corners = []
        for corner in corners:
            v = corner - center
            rotated_v = np.dot(rotation_matrix, v)
            rotated_corners.append(rotated_v + center)
        rotated_corners = np.array(rotated_corners)

        min_x = np.min(rotated_corners[:, 0])
        max_x = np.max(rotated_corners[:, 0])
        min_y = np.min(rotated_corners[:, 1])
        max_y = np.max(rotated_corners[:, 1])

        logger.debug(
            f"Rotation: {rotation_angle_degrees:.2f} deg, "
            f"X:[{min_x:.1f},{max_x:.1f}], Y:[{min_y:.1f},{max_y:.1f}], "
            f"Center:{center}"
        )
        return rotation_angle_degrees, min_x, max_x, min_y, max_y, center

# ---------------------------------------------------------------------------
# Series manager
# ---------------------------------------------------------------------------
class SeriesManager:
    """Manages DICOM series metadata for CT/MRI datasets."""

    def __init__(self):
        self.series_map: dict = {}
        self.series: list = []
        self.series_extrema: dict = {}
        self.series_pos_index: int = 2

    def categorize(self, images_list: list, data_dir: str,
                   debug: bool = False, log_callback=None) -> None:
        """Read metadata from all DICOM files and organize by series."""
        self.series_map = {}
        for name in images_list:
            path = os.path.join(data_dir, name)
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True)
            except Exception as e:
                logger.warning(f"Could not read {name}: {e}")
                continue

            try:
                series = str(ds[0x0020, 0x0011].value)
            except (KeyError, AttributeError):
                continue

            self.series_map.setdefault(series, {})
            try:
                instance = str(ds[0x0020, 0x0013].value)
            except (KeyError, AttributeError):
                instance = "0"

            try:
                image_pos = list(ds[0x0020, 0x0032].value)
            except (KeyError, AttributeError):
                image_pos = [-999.0, -999.0, -999.0]
                if debug and log_callback:
                    log_callback(f"image_pos not available: {name}, instance {instance}")

            try:
                image_dir = list(ds[0x0020, 0x0037].value)
                row_direction = np.array(image_dir[:3])
                col_direction = np.array(image_dir[3:])
                normal_direction = np.cross(row_direction, col_direction)
            except (KeyError, AttributeError):
                image_dir = [1.0, 1.0, 1.0]
                normal_direction = np.array([0.0, 0.0, 0.0])
                if debug and log_callback:
                    log_callback(f"directions not available: {name}, instance {instance}")

            self.series_map[series][name] = [
                instance, image_pos, image_dir, normal_direction
            ]

        self.series = sorted(self.series_map.keys(), key=self._key_func)
        self.series_extrema = {}
        for s in self.series_map:
            self.series_extrema[s] = self._get_position_range(s)

    def get_series_images(self, series_key: str) -> list:
        """Return sorted list of image filenames in a series."""
        return sorted(self.series_map.get(series_key, {}).keys(),
                      key=self._key_func)

    def _get_position_range(self, series: str) -> list:
        """Calculate min/max x, y, z positions for a series."""
        positions = [self.series_map[series][s][1] for s in self.series_map[series]]
        if not positions:
            return [0, 0, 0, 0, 0, 0]
        min_x = min(p[0] for p in positions)
        max_x = max(p[0] for p in positions)
        min_y = min(p[1] for p in positions)
        max_y = max(p[1] for p in positions)
        min_z = min(p[2] for p in positions)
        max_z = max(p[2] for p in positions)
        return [min_x, max_x, min_y, max_y, min_z, max_z]

    def determine_axis(self, series_key: str, images: list) -> int:
        """Determine the principal imaging axis from slice normals. Returns 0, 1, or 2."""
        if not images or series_key not in self.series_map:
            return 2
        first_image = images[0]
        if first_image not in self.series_map[series_key]:
            return 2
        normal = self.series_map[series_key][first_image][3]
        if isinstance(normal, (list, tuple)):
            normal = np.array(normal)
        for axis, unit_vec in enumerate([
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        ]):
            if abs(np.dot(normal, unit_vec)) > NORMAL_THRESHOLD:
                return axis
        return 2

    @staticmethod
    def _key_func(filename: str) -> tuple:
        """Natural sort key."""
        match = re.match(r"([a-zA-Z]*)(\d+)", str(filename))
        if not match:
            try:
                return ("", int(filename))
            except ValueError:
                return (str(filename), 0)
        return (match.group(1), int(match.group(2)))


# ---------------------------------------------------------------------------
# Main viewer class
# ---------------------------------------------------------------------------
class DicomViewer:
    """Interactive DICOM medical image viewer built on Bokeh."""

    def __init__(self):
        # Parse arguments and load config
        parser = argparse.ArgumentParser(description="DICOM Image Viewer")
        parser.add_argument(
            "--app_config",
            default="/Volumes/Data/Home/dicom_viewer.yaml",
            help="Path to YAML configuration file",
        )
        args = parser.parse_args()
        self.config = ViewerConfig.from_yaml(args.app_config)
        self.debug = self.config.debug

        if self.debug:
            logging.getLogger("dicom_viewer").setLevel(logging.DEBUG)

        # State initialization
        self.clip_points: list = []
        self.clipee: int = 0
        self.image_type: str = MODALITY_XRAY
        self.image_scale: float = MRI_SCALE
        self.current_slice: int = 0
        self.series_animate_refresh_rate: float = DEFAULT_REFRESH_MS
        self.current_series: list = []
        self.selected_series: str = ""
        self.gamma: float = self.config.gamma_def
        self.window: float = self.config.window_def
        self.message_log: list = []
        self.series_anim_cb_id: Optional[object] = None
        self.title_postfix: str = ""
        self.wl_presets: list = []

        # Helpers
        self.img_proc = ImageProcessor()
        self.series_mgr = SeriesManager()

        # Load initial dataset
        try:
            self.data_db = self.config.data_db
            self.starter_images = self.config.starter_images
            self.data_dir = self.data_db[self.starter_images]
            self._find_images()
            self.image_name = self.images_list[0]
            self.path = os.path.join(self.data_dir, self.image_name)
            self._prepare_images()

            data = {"image": [self.processed_image]}
            self.source = ColumnDataSource(data)

            if self.is_series:
                self.series_mgr.categorize(
                    self.images_list, self.data_dir, self.debug, self._log
                )
                self.current_series = self.series_mgr.get_series_images(
                    self.series_mgr.series[0]
                )
                self.selected_series = self.series_mgr.series[0]

            if self.debug:
                logger.debug(str(self.ds))
        except FileNotFoundError:
            logger.error("DICOM file not found. Check your configuration.")
            exit(1)
        except Exception as e:
            logger.error(f"Error reading DICOM file: {e}")
            exit(1)

        # Color mapper — initial range from image data
        low = float(np.min(self.processed_image))
        high = float(np.max(self.processed_image))
        self.color_mapper = LinearColorMapper(
            palette=Greys256, low=low, high=high
        )

        # Apply first W/L preset if available
        if self.wl_presets:
            self._apply_window_level(
                self.wl_presets[0]["center"],
                self.wl_presets[0]["width"],
            )

        # Build UI
        self._create_figures()
        self._create_widgets()
        self._build_layout()

    # ------------------------------------------------------------------
    # Figure creation
    # ------------------------------------------------------------------
    def _create_figures(self):
        """Create Bokeh figure objects."""
        self.fig_image = figure(
            width=self.width_scl, height=self.height_scl,
            title="",
            x_range=(0, self.width_scl),
            y_range=(0, self.height_scl),
        )
        self._set_image_fig_title(self.image_name)

        self.f_img = self.fig_image.image(
            image="image", x=0, y=0,
            dw=self.width_scl, dh=self.height_scl,
            source=self.source,
            color_mapper=self.color_mapper,
        )

        hover = HoverTool(tooltips=[
            ("x", "$x"), ("y", "$y"), ("value", "@image"),
        ])
        self.fig_image.add_tools(hover)

        self.color_bar = ColorBar(
            color_mapper=self.color_mapper, label_standoff=12
        )
        self.fig_image.add_layout(self.color_bar, "right")

        self.fig_image.grid.grid_line_color = None
        self.fig_image.axis.axis_line_color = None
        self.fig_image.axis.major_tick_line_color = None
        self.fig_image.axis.major_label_standoff = 0

        # Series positions scatter plot
        self.fig_series_positions = figure(
            title="z positions", height=500, width=640,
            visible=self.is_series,
            x_axis_label="position", y_axis_label="instance",
        )
        self.fig_MRI_source = ColumnDataSource(data={"y": [], "x": []})
        self.fig_series_positions.scatter(
            x="x", y="y", color="blue", source=self.fig_MRI_source,
        )
        self.fig_MRI_source_series = ColumnDataSource(data={"y": [], "x": []})
        self.fig_series_positions.scatter(
            x="x", y="y", color="black", source=self.fig_MRI_source_series,
        )
        self.fig_series_positions_titles = [
            "x positions vs image instance",
            "y positions vs image instance",
            "z positions vs image instance",
        ]

    # ------------------------------------------------------------------
    # Widget creation
    # ------------------------------------------------------------------
    def _create_widgets(self):
        """Create all Bokeh widgets and register callbacks."""

        # Dataset selector
        db_list = list(self.data_db.keys())
        self.db_dropdown = Select(
            title="Pick imaging", value=self.starter_images, options=db_list,
        )
        self.db_dropdown.on_change("value", self._db_dropdown_cb)

        # Modality indicator
        self.mode = RadioButtonGroup(
            labels=["XRay", "CT", "MRI", "US"],
            active=self._modality_index(), visible=True,
        )
        self.mode_div = Div(text="mode", visible=False)

        ct_visible = self.is_series

        # Series animation controls
        self.series_slider_slice = Slider(
            start=0, end=max(len(self.current_series), 1),
            value=0, step=1, title="slice", visible=ct_visible,
        )
        self.series_slider_slice.on_change(
            "value_throttled", self._series_slider_slice_cb
        )

        self.series_toggle_anim = Toggle(
            label="Start Animation", button_type="success",
            active=False, visible=ct_visible,
        )
        self.series_toggle_anim.on_click(self._series_toggle_anim_cb)

        self.CT_text_refresh = TextInput(
            title="refresh (ms)",
            value=str(int(self.series_animate_refresh_rate)),
            visible=ct_visible,
        )
        self.CT_text_refresh.on_change("value", self._refresh_rate_cb)

        # Series selector
        self.series_pulldown = Select(
            title="Pick series", value="", visible=self.is_series,
        )

        # Increment / Decrement
        self.increment_button = Button(
            label="Increment", button_type="success", visible=self.is_series,
        )
        self.increment_button.on_click(self._increment_cb)
        self.decrement_button = Button(
            label="Decrement", button_type="danger", visible=self.is_series,
        )
        self.decrement_button.on_click(self._decrement_cb)

        # Image name selector
        self.name_dropdown = Select(
            title="Pick image", value=self.image_name, options=self.images_list,
        )

        if self.is_series:
            self.series_pulldown.options = self.series_mgr.series
            self.series_pulldown.value = self.series_mgr.series[0]
            self.name_dropdown.options = self.current_series
            self._histogram_positions()
            self._series_scatter_pos(self.image_name)

        self.series_pulldown.on_change("value", self._series_cb)
        self.name_dropdown.on_change("value", self._name_cb)

        # Clip button
        self.clip_button = Button(label="Clip", button_type="danger")
        self.clip_button.on_click(self._clip_reset_cb)

        # Gamma slider
        self.gamma_slider = Slider(
            start=0, end=GAMMA_SLIDER_MAX,
            value=self.gamma, step=0.1, title="Gamma",
        )
        self.gamma_slider.on_change("value_throttled", self._gamma_cb)

        # Legacy window slider (simple brightness multiplier)
        self.window_slider = Slider(
            start=0, end=WINDOW_SLIDER_MAX,
            value=self.window, step=0.05, title="Window (legacy)",
        )
        self.window_slider.on_change("value_throttled", self._window_cb)

        # --- Window/Level preset controls ---
        self.wl_preset_dropdown = Select(
            title="W/L Preset", value=WL_MANUAL_LABEL,
            options=self._build_wl_preset_options(),
        )
        self.wl_preset_dropdown.on_change("value", self._wl_preset_cb)

        # Determine initial W/L slider values
        if self.wl_presets:
            init_center = self.wl_presets[0]["center"]
            init_width = self.wl_presets[0]["width"]
        else:
            init_center = self.max_bright / 2.0 if self.max_bright > 0 else WL_CENTER_DEFAULT
            init_width = self.max_bright if self.max_bright > 0 else WL_WIDTH_DEFAULT

        # Compute slider ranges from image data
        wl_center_max = max(WL_CENTER_SLIDER_MAX, self.max_bright)
        wl_width_max = max(WL_WIDTH_SLIDER_MAX, self.max_bright * 2)

        self.wl_center_slider = Slider(
            start=WL_CENTER_SLIDER_MIN, end=wl_center_max,
            value=init_center, step=WL_SLIDER_STEP,
            title="Window Center",
        )
        self.wl_center_slider.on_change("value_throttled", self._wl_manual_cb)

        self.wl_width_slider = Slider(
            start=WL_WIDTH_SLIDER_MIN, end=wl_width_max,
            value=init_width, step=WL_SLIDER_STEP,
            title="Window Width",
        )
        self.wl_width_slider.on_change("value_throttled", self._wl_manual_cb)

        # Reset button
        self.reset_button = Button(label="Reset", button_type="warning")
        self.reset_button.on_click(self._reset_cb)

        # Log display
        self.log_div = Div(text="Log:<br>", width=400, height=200)

        # Exit button
        self.exit_button = Button(label="Exit", button_type="danger")
        self.exit_button.on_click(self._stop_server)

        # Tap tool for clipping
        self.taptool = TapTool()
        self.fig_image.add_tools(self.taptool)
        self.fig_image.on_event("tap", self._tap_callback)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_layout(self):
        """Assemble the Bokeh layout and attach to curdoc."""
        ct_layout = row(
            self.series_toggle_anim,
            self.CT_text_refresh,
            self.series_slider_slice,
            self.increment_button,
            self.decrement_button,
        )

        wl_controls = column(
            self.wl_preset_dropdown,
            self.wl_center_slider,
            self.wl_width_slider,
        )

        adjustment_controls = column(
            self.gamma_slider,
            self.window_slider,
        )

        control_widgets = row(
            column(
                row(
                    self.exit_button,
                    self.reset_button,
                    self.db_dropdown,
                    column(self.mode_div, self.mode),
                    self.clip_button,
                    adjustment_controls,
                    wl_controls,
                    self.name_dropdown,
                ),
                self.series_pulldown,
                ct_layout,
            ),
            self.log_div,
        )
        image_glyph = row(
            self.fig_image,
            column(self.fig_series_positions),
        )
        canvas_layout = layout(column(control_widgets, image_glyph))

        curdoc().clear()
        curdoc().add_root(canvas_layout)
        curdoc().title = "DICOM Viewer"

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _name_cb(self, attr, old, new):
        """Handle image selection change."""
        self.image_name = new
        self._log(f"Get new file {self.image_name}")
        self.path = os.path.join(self.data_dir, self.image_name)
        self._prepare_images()
        self.source.data["image"] = [self.processed_image]
        self._size_figures()
        self._set_image_fig_title(new)
        self._refresh_wl_presets()

    def _db_dropdown_cb(self, attr, old, new):
        """Handle dataset change — full reset."""
        self.data_dir = self.data_db[new]
        self._find_images()
        self.image_name = self.images_list[0]
        self.path = os.path.join(self.data_dir, self.image_name)

        self.name_dropdown.remove_on_change("value", self._name_cb)
        self.name_dropdown.options = self.images_list
        self.name_dropdown.value = self.image_name
        self.name_dropdown.on_change("value", self._name_cb)

        self._prepare_images()

        if self.is_series:
            self.series_mgr.categorize(
                self.images_list, self.data_dir, self.debug, self._log,
            )
            self.selected_series = self.series_mgr.series[0]
            self.current_series = self.series_mgr.get_series_images(
                self.selected_series
            )
            self.series_pulldown.options = self.series_mgr.series
            self.current_slice = 0

            self._histogram_positions()
            self._series_scatter_pos(self.current_series[0])

            self.series_slider_slice.remove_on_change(
                "value_throttled", self._series_slider_slice_cb
            )
            self.series_slider_slice.value = self.current_slice
            self.series_slider_slice.end = len(self.current_series)
            self.series_slider_slice.on_change(
                "value_throttled", self._series_slider_slice_cb
            )

            self.series_pulldown.remove_on_change("value", self._series_cb)
            self.series_pulldown.value = self.selected_series
            self.series_pulldown.on_change("value", self._series_cb)

            self.name_dropdown.remove_on_change("value", self._name_cb)
            self.name_dropdown.options = self.current_series
            self.name_dropdown.value = self.current_series[0]
            self.name_dropdown.on_change("value", self._name_cb)

        self._size_figures()
        self._update_visibility()
        self._reset_adjustments()

        self.mode.active = self._modality_index()
        self._set_image_fig_title(self.image_name)
        self.source.data["image"] = [self.processed_image]
        self._refresh_wl_presets()
        self._log(f"Get new imaging {new}")

    def _series_cb(self, attr, old, new):
        """Handle series selection change."""
        self.selected_series = new
        self._log(f"Get new series {self.selected_series}")

        self.current_series = self.series_mgr.get_series_images(
            self.selected_series
        )

        self.name_dropdown.remove_on_change("value", self._name_cb)
        self.name_dropdown.options = self.current_series
        if self.current_series:
            self.name_dropdown.value = self.current_series[0]
        self.name_dropdown.on_change("value", self._name_cb)

        if self.current_series:
            self.path = os.path.join(self.data_dir, self.current_series[0])

        self._histogram_positions()
        if self.current_series:
            self._series_scatter_pos(self.current_series[0])

        self.fig_series_positions.title.text = (
            self.fig_series_positions_titles[self.series_mgr.series_pos_index]
        )

        self.series_toggle_anim.button_type = "success"
        self.series_toggle_anim.label = "Start Animation"
        self._prepare_images()
        if self.current_series:
            self._set_image_fig_title(self.current_series[0])
        self.current_slice = 0
        self._size_figures()
        self.source.data["image"] = [self.processed_image]

        self.series_slider_slice.remove_on_change(
            "value_throttled", self._series_slider_slice_cb
        )
        self.series_slider_slice.value = self.current_slice
        self.series_slider_slice.end = max(len(self.current_series), 1)
        self.series_slider_slice.on_change(
            "value_throttled", self._series_slider_slice_cb
        )

        self._refresh_wl_presets()
        self._log(f"Selected new series {new}")

    def _gamma_cb(self, attr, old, new):
        """Handle gamma slider change."""
        self.gamma = new
        self.processed_image = ImageProcessor.perform_gamma(
            self.clipped_image, self.gamma, self.original_dtype
        )
        self.source.data["image"] = [self.processed_image]
        self.max_bright = float(np.max(self.processed_image))

        # Refresh W/L since gamma changes the value space
        self._refresh_wl_presets()

        self._log(f"Set gamma to {self.gamma:.1f}")

    def _window_cb(self, attr, old, new):
        """Handle legacy window slider change."""
        self.window = new
        # Only apply if no W/L presets are active
        if self.wl_preset_dropdown.value == WL_MANUAL_LABEL and not self.wl_presets:
            self.color_mapper.high = self.max_bright * new
            self._log(f"Set window scale to {new:.2f}")
        else:
            self._log(f"Legacy window ignored — using W/L controls "
                      f"(scale={new:.2f})")

    def _refresh_rate_cb(self, attr, old, new):
        """Handle animation refresh rate change."""
        try:
            rate = float(new)
            if rate > 0:
                self.series_animate_refresh_rate = rate
                self._log(f"Refresh rate set to {rate:.0f} ms")
            else:
                self._log("Refresh rate must be positive")
        except ValueError:
            self._log(f"Invalid refresh rate: {new}")

    # --- Window/Level callbacks ---

    def _wl_preset_cb(self, attr, old, new):
        """Handle window/level preset dropdown selection."""
        if new == WL_MANUAL_LABEL:
            self._log("Switched to manual window/level")
            return

        for p in self.wl_presets:
            label = self._wl_preset_label(p)
            if label == new:
                # Update sliders without triggering their callbacks
                self.wl_center_slider.remove_on_change(
                    "value_throttled", self._wl_manual_cb
                )
                self.wl_width_slider.remove_on_change(
                    "value_throttled", self._wl_manual_cb
                )

                self.wl_center_slider.value = p["center"]
                self.wl_width_slider.value = p["width"]

                self.wl_center_slider.on_change(
                    "value_throttled", self._wl_manual_cb
                )
                self.wl_width_slider.on_change(
                    "value_throttled", self._wl_manual_cb
                )

                self._apply_window_level(p["center"], p["width"])
                self._log(f"W/L preset: {p['name']} "
                          f"(C:{p['center']:.0f} W:{p['width']:.0f})")
                break

    def _wl_manual_cb(self, attr, old, new):
        """Handle manual window center/width slider changes."""
        center = self.wl_center_slider.value
        width = self.wl_width_slider.value
        self._apply_window_level(center, width)

        # Switch dropdown to Manual since user is overriding presets
        self.wl_preset_dropdown.remove_on_change("value", self._wl_preset_cb)
        self.wl_preset_dropdown.value = WL_MANUAL_LABEL
        self.wl_preset_dropdown.on_change("value", self._wl_preset_cb)

        self._log(f"Manual W/L: center={center:.0f}, width={width:.0f}")

    def _reset_cb(self):
        """Reset gamma, window, and W/L to defaults."""
        # Reset gamma
        self.gamma = self.config.gamma_def
        self.window = self.config.window_def

        self.gamma_slider.remove_on_change("value_throttled", self._gamma_cb)
        self.gamma_slider.value = self.gamma
        self.gamma_slider.on_change("value_throttled", self._gamma_cb)

        self.window_slider.remove_on_change("value_throttled", self._window_cb)
        self.window_slider.value = self.window
        self.window_slider.on_change("value_throttled", self._window_cb)

        # Reprocess image with default gamma
        self.processed_image = ImageProcessor.perform_gamma(
            self.clipped_image, self.gamma, self.original_dtype
        )
        self.max_bright = float(np.max(self.processed_image))
        self.source.data["image"] = [self.processed_image]

        # Reset W/L — let _refresh_wl_presets handle everything
        self._refresh_wl_presets()

        self._log("Reset all adjustments to defaults")

    def _clip_reset_cb(self):
        """Reset clip state."""
        self.clip_points = []
        self.clipee = 0
        self._log("Clip mode reset. Tap 4 corners on the image.")

    # ------------------------------------------------------------------
    # Series animation
    # ------------------------------------------------------------------
    def _increment_cb(self):
        """Advance one slice forward."""
        images = self.current_series if self.is_series else self.images_list
        if self.current_slice < len(images) - 1:
            self.current_slice += 1
            self._sync_slice_slider()
            self._get_new_image()
            self._log(f"Increment image to {self.current_slice}")
        else:
            self._log(f"At max image {self.current_slice}")

    def _decrement_cb(self):
        """Go back one slice."""
        if self.current_slice > 0:
            self.current_slice -= 1
            self._sync_slice_slider()
            self._get_new_image()
            self._log(f"Decrement image to {self.current_slice}")
        else:
            self._log(f"At min image {self.current_slice}")

    def _series_slider_slice_cb(self, attr, old, new):
        """Handle slice slider change."""
        self.current_slice = int(new)
        self._get_new_image()
        self._log(f"Slider set to {self.current_slice}")

    def _animate_series(self):
        """Periodic callback to advance animation one frame."""
        if not self.series_toggle_anim.active:
            return
        images = self.current_series if self.is_series else self.images_list
        if not images:
            return

        image_name = images[self.current_slice]
        self.path = os.path.join(self.data_dir, image_name)
        self._prepare_images()

        if self.is_series:
            self._series_scatter_pos(image_name)

        self.fig_image.title.text = image_name + self.title_postfix
        self.source.data["image"] = [self.processed_image]

        self.current_slice += 1
        if self.current_slice >= len(images):
            self.current_slice = ANIMATION_SLICE_RESET

        self._sync_slice_slider()

        if self.debug:
            self._log(f"Animation image {image_name}")

    def _series_toggle_anim_cb(self, active):
        """Start or stop animation."""
        if active:
            self.series_toggle_anim.button_type = "danger"
            self.series_toggle_anim.label = "Stop Animation"
            self.series_anim_cb_id = curdoc().add_periodic_callback(
                self._animate_series, int(self.series_animate_refresh_rate),
            )
        else:
            self.series_toggle_anim.button_type = "success"
            self.series_toggle_anim.label = "Start Animation"
            if self.series_anim_cb_id is not None:
                curdoc().remove_periodic_callback(self.series_anim_cb_id)
                self.series_anim_cb_id = None

    # ------------------------------------------------------------------
    # Tap / clip
    # ------------------------------------------------------------------
    def _tap_callback(self, event):
        """Handle tap event for clip-and-rotate (4 corners)."""
        try:
            selected_index = [
                self.source.selected.image_indices[self.clipee]["i"],
                self.source.selected.image_indices[self.clipee]["j"],
            ]
        except (IndexError, KeyError):
            self._log("Hit whitespace! Try again")
            return

        self.clip_points.append(selected_index)
        self.clipee += 1
        self._log(f"Selected point {self.clipee}: {selected_index}")

        if self.clipee == 4:
            rotation_angle, min_x, max_x, min_y, max_y, center = (
                ImageProcessor.rotated_rectangle_properties(self.clip_points)
            )

            self.clipped_image = transform.rotate(
                image=self.clipped_image,
                angle=-rotation_angle * 0.6,
                center=(center[1], center[0]),
                preserve_range=True,
            )
            self.processed_image = ImageProcessor.perform_gamma(
                self.clipped_image, self.gamma, self.original_dtype
            )

            dw = int(max_x - min_x)
            dh = int(max_y - min_y)

            self.fig_image.renderers = [
                r for r in self.fig_image.renderers if r != self.f_img
            ]
            self.f_img = self.fig_image.image(
                image="image", x=min_x, y=min_y, dw=dw, dh=dh,
                source=self.source, color_mapper=self.color_mapper,
            )

            self.fig_image.height = dh
            self.fig_image.y_range.start = min_y
            self.fig_image.y_range.end = max_y
            self.fig_image.width = dw
            self.fig_image.x_range.start = min_x
            self.fig_image.x_range.end = max_x

            self.source.data["image"] = [self.processed_image]
            self._log(f"Clipped and rotated by {rotation_angle:.2f} degrees")

            self.clipee = 0
            self.clip_points = []
            self.source.selected.image_indices = []

    # ------------------------------------------------------------------
    # Server control
    # ------------------------------------------------------------------
    def _stop_server(self):
        """Gracefully shut down the Bokeh server."""
        curdoc().add_next_tick_callback(
            lambda: self._async_update_log("Server is shutting down...")
        )
        curdoc().add_next_tick_callback(
            lambda: self._async_change_button(self.exit_button, "light")
        )
        curdoc().add_next_tick_callback(self._exit_server)

    async def _async_update_log(self, message: str):
        self.log_div.text = message

    async def _exit_server(self):
        logger.info("Server is shutting down...")
        IOLoop.current().stop()

    async def _async_change_button(self, button, color: str):
        button.button_type = color

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_images(self):
        """Read and process the current DICOM file."""
        self.ds = pydicom.dcmread(self.path)
        self.ds_SOP = self.ds.file_meta[0x0002, 0x0002].value
        self.sop_class_name = pydicom.uid.UID(self.ds_SOP).name

        if self.debug:
            logger.debug(self.sop_class_name)

        self.dicom_image = self.ds.pixel_array
        self.original_dtype = self.dicom_image.dtype

        # Handle multi-frame / RGB
        self.dicom_image = ImageProcessor.ensure_2d(self.dicom_image)

        # Apply rescale slope/intercept
        self.dicom_image = ImageProcessor.apply_rescale(self.dicom_image, self.ds)

        # Apply photometric interpretation
        self.clipped_image = ImageProcessor.apply_photometric(self.dicom_image, self.ds)

        # Flip vertically for display
        self.clipped_image = np.flipud(self.clipped_image)

        # Determine modality and scale
        if "X-Ray" in self.sop_class_name:
            self.image_scale = XRAY_SCALE
            self.image_type = MODALITY_XRAY
        elif "CT" in self.sop_class_name:
            self.image_scale = CT_SCALE
            self.image_type = MODALITY_CT
        elif "MR" in self.sop_class_name:
            self.image_scale = MRI_SCALE
            self.image_type = MODALITY_MRI
            if self.gamma == self.config.gamma_def:
                self.gamma = MRI_GAMMA
        else:
            self.image_scale = US_SCALE
            self.image_type = MODALITY_US
            if self.gamma == self.config.gamma_def:
                self.gamma = US_GAMMA

        self.is_series = (self.image_type != MODALITY_XRAY)
        self.height, self.width = self.clipped_image.shape
        self.height_scl = int(self.height * self.image_scale)
        self.width_scl = int(self.width * self.image_scale)

        # Apply gamma correction
        self.processed_image = ImageProcessor.perform_gamma(
            self.clipped_image, self.gamma, self.original_dtype
        )
        self.max_bright = float(np.max(self.processed_image))

        # Extract W/L presets from this image's metadata
        self.wl_presets = ImageProcessor.extract_wl_presets(self.ds)

    def _size_figures(self):
        """Update figure dimensions to match current image."""
        self.fig_image.x_range.end = self.width_scl
        self.fig_image.y_range.end = self.height_scl
        self.fig_image.width = self.width_scl
        self.fig_image.height = self.height_scl
        self.f_img.glyph.dw = self.width_scl
        self.f_img.glyph.dh = self.height_scl

    def _set_image_fig_title(self, image_name: str):
        """Set figure title with patient metadata."""
        try:
            patient = (
                self.ds.PatientName.given_name + " " +
                self.ds.PatientName.family_name
            )
        except (AttributeError, TypeError):
            patient = "Unknown"

        try:
            proc_date = self.ds.PerformedProcedureStepStartDate
        except AttributeError:
            proc_date = "Unknown date"

        try:
            protocol = self.ds.ProtocolName
        except AttributeError:
            protocol = "Unknown protocol"

        self.title_postfix = f" {patient} {proc_date} {protocol}"
        self.fig_image.title.text = image_name + self.title_postfix

    def _get_new_image(self):
        """Load and display the image at current_slice."""
        images = self.current_series if self.is_series else self.images_list
        if not images or self.current_slice >= len(images):
            return
        image_name = images[self.current_slice]
        self.path = os.path.join(self.data_dir, image_name)
        self._prepare_images()
        self._set_image_fig_title(image_name)
        self.source.data["image"] = [self.processed_image]

        # Apply current W/L settings to new image
        # (preserve user's current slider positions)
        center = self.wl_center_slider.value
        width = self.wl_width_slider.value
        self._apply_window_level(center, width)

    def _sync_slice_slider(self):
        """Update the slice slider without triggering its callback."""
        self.series_slider_slice.remove_on_change(
            "value_throttled", self._series_slider_slice_cb
        )
        self.series_slider_slice.value = self.current_slice
        self.series_slider_slice.on_change(
            "value_throttled", self._series_slider_slice_cb
        )

    def _histogram_positions(self):
        """Populate the series position scatter plot data."""
        if not self.current_series or not self.selected_series:
            return

        sm = self.series_mgr
        axis = sm.determine_axis(self.selected_series, self.current_series)
        sm.series_pos_index = axis

        i_n = []
        z_i = []
        for img in self.current_series:
            if img not in sm.series_map.get(self.selected_series, {}):
                continue
            entry = sm.series_map[self.selected_series][img]
            i_n.append(entry[0])
            z_i.append(entry[1][axis])

        self.fig_MRI_source_series.data = {"x": z_i, "y": i_n}

    def _series_scatter_pos(self, image_name: str):
        """Highlight the current slice on the position scatter plot."""
        sm = self.series_mgr
        if (self.selected_series not in sm.series_map or
                image_name not in sm.series_map[self.selected_series]):
            return

        entry = sm.series_map[self.selected_series][image_name]
        z_i = entry[1][sm.series_pos_index]
        i_n = entry[0]
        self.fig_MRI_source.data = {"x": [z_i], "y": [i_n]}

    def _find_images(self):
        """Scan data directory for image files."""
        path = Path(self.data_dir)
        files = [
            f.name for f in path.iterdir()
            if f.is_file() and not f.name.startswith(".")
        ]
        self.images_list = sorted(files, key=SeriesManager._key_func)

    def _modality_index(self) -> int:
        """Return RadioButtonGroup index for current modality."""
        mapping = {
            MODALITY_XRAY: 0, MODALITY_CT: 1,
            MODALITY_MRI: 2, MODALITY_US: 3,
        }
        return mapping.get(self.image_type, 0)

    def _update_visibility(self):
        """Update widget visibility based on current modality."""
        is_s = self.is_series
        self.series_pulldown.visible = is_s
        self.series_toggle_anim.visible = is_s
        self.series_slider_slice.visible = is_s
        self.CT_text_refresh.visible = is_s
        self.fig_series_positions.visible = is_s
        self.increment_button.visible = is_s
        self.decrement_button.visible = is_s

    def _reset_adjustments(self):
        """Reset gamma and window sliders to defaults."""
        self.gamma = self.config.gamma_def
        self.window = self.config.window_def

        self.gamma_slider.remove_on_change("value_throttled", self._gamma_cb)
        self.gamma_slider.value = self.gamma
        self.gamma_slider.on_change("value_throttled", self._gamma_cb)

        self.window_slider.remove_on_change("value_throttled", self._window_cb)
        self.window_slider.value = self.window
        self.window_slider.on_change("value_throttled", self._window_cb)

    # ------------------------------------------------------------------
    # Window/Level helpers
    # ------------------------------------------------------------------
    def _apply_window_level(self, center: float, width: float):
        """Apply window/level by adjusting the color mapper range."""
        low = center - width / 2.0
        high = center + width / 2.0
        self.color_mapper.low = low
        self.color_mapper.high = high

    @staticmethod
    def _wl_preset_label(preset: dict) -> str:
        """Generate a display label for a W/L preset."""
        return f"{preset['name']} (C:{preset['center']:.0f} W:{preset['width']:.0f})"

    def _build_wl_preset_options(self) -> list:
        """Build the options list for the W/L preset dropdown."""
        options = [WL_MANUAL_LABEL]
        for p in self.wl_presets:
            options.append(self._wl_preset_label(p))
        return options

    def _refresh_wl_presets(self):
        """Update W/L controls for the current image's metadata."""
        # Recalculate image range from current processed image
        img_min = float(np.min(self.processed_image))
        img_max = float(np.max(self.processed_image))
        if img_max <= img_min:
            img_max = img_min + 1.0

        # Update dropdown options
        options = self._build_wl_preset_options()

        self.wl_preset_dropdown.remove_on_change("value", self._wl_preset_cb)
        self.wl_preset_dropdown.options = options

        # Update slider ranges based on current image data
        wl_center_min = min(WL_CENTER_SLIDER_MIN, img_min)
        wl_center_max = max(WL_CENTER_SLIDER_MAX, img_max)
        wl_width_max = max(WL_WIDTH_SLIDER_MAX, (img_max - img_min) * 2)

        self.wl_center_slider.remove_on_change("value_throttled", self._wl_manual_cb)
        self.wl_width_slider.remove_on_change("value_throttled", self._wl_manual_cb)

        self.wl_center_slider.start = wl_center_min
        self.wl_center_slider.end = wl_center_max
        self.wl_width_slider.end = wl_width_max

        if self.wl_presets:
            # Apply first preset
            first = self.wl_presets[0]

            # Preset values are in original DICOM value space.
            # If gamma has been applied, we need to check whether
            # the preset values are compatible with the displayed range.
            # Use preset values directly — they set the color mapper
            # range on the gamma-corrected data.
            center = first["center"]
            width = first["width"]

            # Clamp preset values to be within the actual image range
            # if they're wildly out of range (can happen when gamma
            # transforms the value space)
            if center < img_min or center > img_max:
                # Preset is in a different value space — auto-calculate instead
                center = (img_min + img_max) / 2.0
                width = img_max - img_min
                self._log(f"W/L preset out of range, using auto: "
                          f"C:{center:.0f} W:{width:.0f}")
            else:
                self._log(f"W/L preset applied: {first['name']} "
                          f"(C:{center:.0f} W:{width:.0f})")

            self.wl_center_slider.value = center
            self.wl_width_slider.value = width
            self.wl_preset_dropdown.value = options[1] if len(options) > 1 else WL_MANUAL_LABEL
            self._apply_window_level(center, width)
        else:
            # No presets — use image range
            center = (img_min + img_max) / 2.0
            width = img_max - img_min
            if width <= 0:
                width = 1.0

            self.wl_center_slider.value = center
            self.wl_width_slider.value = width
            self.wl_preset_dropdown.value = WL_MANUAL_LABEL
            self._apply_window_level(center, width)

        self.wl_center_slider.on_change("value_throttled", self._wl_manual_cb)
        self.wl_width_slider.on_change("value_throttled", self._wl_manual_cb)
        self.wl_preset_dropdown.on_change("value", self._wl_preset_cb)

    def _log(self, message: str):
        """Add a message to the on-screen log."""
        self.message_log.append(message)
        if len(self.message_log) > MAX_LOG_MESSAGES:
            self.message_log.pop(0)
        self.log_div.text = "Log:<br>" + "<br>".join(self.message_log)
        logger.info(message)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
viewer = DicomViewer()
