import os
import numpy as np
import pydicom
from tornado.ioloop import IOLoop
from skimage import transform
import yaml
import argparse
from pathlib import Path
import pylibjpeg
import re

from bokeh.plotting import figure, show, curdoc
from bokeh.layouts import column, row, layout
from bokeh.models import (ColumnDataSource, LinearColorMapper, ColorBar, HoverTool, Button, Slider, Div, TapTool,
                          Select, RadioButtonGroup, Toggle, TextInput, Range1d, LinearAxis)

from bokeh.palettes import Greys256  # For grayscale

# https://pydicom.github.io/pydicom/stable/index.html

#   Conda env: pydicom
#   Requirements:  pydicom (conda); ppylibjpeg[all] (pip install); scikit-image; bokeh
#
#   Invoke: bokeh serve dicom_viewer.py --args --app_config "/Volumes/Data/Home/dicom_viewer.yaml"
#
#   View in browser at: http://localhost:5006/dicom_viewer


class dicom_viewer():
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--app_config',
                            default="/Volumes/Data/Home/dicom_viewer.yaml",
                            help="overall app config file")
        args = parser.parse_args()

        with open(args.app_config, "r") as f:
            data = yaml.safe_load(f)

        self.debug = data["debug"]

        self.clip_points = []
        self.clipee = 0
        self.clipped = False

        self.image_type = "X-Ray"
        self.image_scale = 4.

        self.current_slice = 0
        self.series_animate_refresh_rate = 500.

        self.current_series = []

        self.gamma_def = data["gamma_def"]
        self.gamma = self.gamma_def

        self.window_def = data["window_def"]
        self.window = self.window_def

        self.message_log = []

        # 1. Load the DICOM image
        try:
            self.data_db = data["data_db"]
            self.starter_images = data["starter_images"]

            self.data_dir = self.data_db[self.starter_images]

            rc = self.find_images()

            self.image_name = self.images_list[0]
            self.path = self.data_dir + self.image_name

            rc = self.prepare_images()
            data = {'image': [self.processed_image]}

            self.source = ColumnDataSource(data)

            if self.is_series:
                rc = self.categorize_series()

            if self.debug:
                print(self.ds)

        except FileNotFoundError:
            print("Error: DICOM file not found. Please specify the correct path.")
            exit()
        except Exception as e:
            print(f"Error reading DICOM file: {e}")
            exit()

        # 3. Create a LinearColorMapper for grayscale mapping
        try:
            low = np.min(self.processed_image)
            high = np.max(self.processed_image)

            self.color_mapper = LinearColorMapper(palette=Greys256, low=low, high=high) #Use Greys256 for grayscale
        except Exception as e:
            print(f"Error creating color mapper: {e}")
            exit()


        # Create the Bokeh figure

        self.fig_image = figure(width=self.width_scl, height=self.height_scl, title="",
                                x_range=(0, self.width_scl), y_range=(0, self.height_scl))
        rc = self.set_image_fig_title(self.image_name)


        # Add the image to the plot
        self.f_img = self.fig_image.image(image='image', x=0, y=0, dw=self.width_scl, dh=self.height_scl, source=self.source,
                             color_mapper=self.color_mapper)

        # Add a HoverTool
        hover = HoverTool(tooltips=[
            ("x", "$x"),
            ("y", "$y"),
            ("value", f"@image"),
        ])
        self.fig_image.add_tools(hover)

        # 6. Add a color bar (optional)
        self.color_bar = ColorBar(color_mapper=self.color_mapper, label_standoff=12)
        self.fig_image.add_layout(self.color_bar, 'right')

        # Remove grid lines and axis ticks
        self.fig_image.grid.grid_line_color = None
        self.fig_image.axis.axis_line_color = None
        self.fig_image.axis.major_tick_line_color = None
        self.fig_image.axis.major_label_standoff = 0

        self.fig_series_positions = figure(title="z positions", height=500, width=640, visible=False,
                                        x_axis_label="position", y_axis_label="instance")
        self.fig_MRI_source = ColumnDataSource(data=dict({"y": [], "x": []}))
        self.fig_series_positions.scatter(x="x", y="y", color="blue", source=self.fig_MRI_source)
        self.fig_MRI_source_series = ColumnDataSource(data=dict({"y": [], "x": []}))
        self.fig_series_positions.scatter(x="x", y="y", color="black", source=self.fig_MRI_source_series)
        self.fig_series_positions_titles = ["x positions vs image instance", "y positions vs image instance",
                                            "z positions vs image instance"]

        self.fig_series_positions.visible = self.is_series

        rc = self.create_widgets()

        CT_layout = row(self.series_toggle_anim, self.CT_text_refresh, self.series_slider_slice,
                        self.increment_button, self.decrement_button)
        control_widgets = row(column(row(self.exit_button, self.db_dropdown, column(self.mode_div, self.mode),
                                         self.clip_button,
                                     column(self.gamma_slider, self.window_slider), self.name_dropdown),
                                     self.series_pulldown, CT_layout),
                              self.log_div)

        image_glyph = row(self.fig_image, column(self.fig_series_positions))
        canvas_layout = layout(column(control_widgets, image_glyph))

        # Add the layout to the current document

        curdoc().clear()
        curdoc().add_root(canvas_layout)
        curdoc().title = "DICOM viewer"

    def size_figures(self):

        self.fig_image.x_range.end = self.width_scl
        self.fig_image.y_range.end = self.height_scl

        self.f_img.glyph.dw = self.width_scl
        self.f_img.glyph.dh = self.height_scl

    def prepare_images(self):

        self.ds = pydicom.dcmread(self.path)
        self.ds_SOP = self.ds.file_meta[0x0002, 0x0002].value
        self.sop_class_name = pydicom.uid.UID(self.ds_SOP).name

        if self.debug:
            print(self.sop_class_name)

        self.dicom_image = self.ds.pixel_array

        # Invert the grayscale
        self.clipped_image = np.max(self.dicom_image) - self.dicom_image  # Subtract image from the maximum value

        # Flip the image vertically
        self.clipped_image = np.flipud(self.clipped_image)

        # rescale for visibility

        if "X-Ray" in self.sop_class_name:
            self.image_scale = 0.5
            self.image_type = "X-Ray"
        elif "CT" in self.sop_class_name:
            self.image_scale = 1.5
            self.image_type = "CT"
        elif "MR" in self.sop_class_name:
            self.image_scale = 4.
            self.gamma = 2.
            self.image_type = "MRI"
        else:
            self.image_scale = 1.5
            self.gamma = 2.
            self.image_type = "US"

        self.is_series = (self.image_type != "X-Ray")

        self.height, self.width = self.clipped_image.shape
        self.height_scl = int(self.height * self.image_scale)
        self.width_scl = int(self.width * self.image_scale)
        self.max_bright = np.max(self.clipped_image)

        # Define clip ranges
        width_start = 0
        width_end = self.width
        height_start = 0
        height_end = self.height

        # Clip the image
        self.clipped_image = self.clipped_image[height_start:height_end, width_start:width_end]

        # Gamma Correction
        self.processed_image = self.perform_gamma(self.clipped_image)
        self.max_bright = np.max(self.processed_image)

    def categorize_series(self):

        self.series_map = {}

        for n in self.images_list:
            p = self.data_dir + n

            ds = pydicom.dcmread(p)

            # some images may not be actual images
            try:
                series = str(ds[0x0020, 0x0011].value)
            except:
                continue

            self.series_map.setdefault(series, {})

            instance = str(ds[0x0020, 0x0013].value)
            try:
                image_pos = ds[0x0020, 0x0032].value
            except:
                image_pos = ["-999", "-999", "-999"]
                if self.debug:
                    self.generate_log_message(self.log_div, f"image_pos not available: {n, instance}")

            try:
                image_dir = ds[0x0020, 0x0037].value
                row_direction = np.array(image_dir[:3])
                col_direction = np.array(image_dir[3:])
                normal_direction = np.cross(row_direction, col_direction)
                if self.debug:
                    self.generate_log_message(self.log_div, f"directions not available: {n, instance}")

            except:
                image_dir = [1., 1., 1.]
                normal_direction = [0, 0, 0]

            self.series_map[series][n] = [instance, image_pos, image_dir, normal_direction]

        self.series = []
        for s in self.series_map:
            self.series.append(str(s))

        self.series = sorted(self.series, key=self.key_func)

        current_series = []
        for i in self.series_map[self.series[0]]:
            current_series.append(str(i))

        self.series_extrema = {}

        for i in self.series_map:
            extrema = self.get_position_range(i)
            self.series_extrema[i] = extrema

        self.current_series = current_series

    def get_position_range(self, series):
        """Calculates the range of positions within a series of DICOM slices."""
        positions = [self.series_map[series][s][1] for s in self.series_map[series]]

        min_x = min(p[0] for p in positions)
        max_x = max(p[0] for p in positions)
        min_y = min(p[1] for p in positions)
        max_y = max(p[1] for p in positions)
        min_z = min(p[2] for p in positions)
        max_z = max(p[2] for p in positions)
        return [min_x, max_x, min_y, max_y, min_z, max_z]

    def perform_gamma(self, image):
        # Normalize to range 0-1
        image = image.astype(np.float64) / np.max(image)  # Important: Normalize *before* gamma

        # Gamma Correction
        image = image ** (self.gamma)

        # Scale back to original range (Crucial after gamma)
        image = (image * np.iinfo(self.ds.pixel_array.dtype).max).astype(self.ds.pixel_array.dtype)

        return image

    def create_widgets(self):

        db_list = list(self.data_db.keys())
        self.db_dropdown = Select(title="Pick imaging", value=self.starter_images, options=db_list)
        self.db_dropdown.on_change("value", self.db_dropdown_cb)

        self.mode = RadioButtonGroup(labels=["XRay", "CT", "MRI", "US"], active=0, visible=True)
        self.mode_div = Div(text="mode", visible=False)

        ct_visible = False

        if "X-Ray" in self.sop_class_name:
            self.mode.active = 0
        elif "CT" in self.sop_class_name:
            self.mode.active = 1
            ct_visible = True
        elif "MR" in self.sop_class_name:
            self.mode.active = 2
            ct_visible = True
        elif "Ultra" in self.sop_class_name:
            self.mode.active = 3
            ct_visible = True

        #if self.is_series:
        #    rc = self.categorize_series()

        # CT animation

        # Create a slider for CT refresh rate
        step = len(self.images_list)/100.
        self.series_slider_slice = Slider(start=0, end=len(self.images_list), value=0, step=step,
                                     title="slice", visible=ct_visible)
        self.series_slider_slice.on_change("value_throttled", self.series_slider_slice_cb)

        # Create a Toggle button for CT start/stop
        self.series_toggle_anim = Toggle(label="Start Animation", button_type="success", active=False, visible=ct_visible)
        self.series_toggle_anim.on_click(self.series_toggle_anim_cb)

        self.CT_text_refresh = TextInput(title="refresh (ms)",
                                         value=str(self.series_animate_refresh_rate), visible=ct_visible)

        self.series_pulldown = Select(title="Pick series", value=None,
                                          visible=self.is_series)

        self.increment_button = Button(label="Increment", button_type="success", visible=self.is_series)
        self.increment_button.on_click(self.increment_cb)
        self.decrement_button = Button(label="Decrement", button_type="danger", visible=self.is_series)
        self.decrement_button.on_click(self.decrement_cb)


        self.name_dropdown = Select(title="Pick image", value=self.image_name, options=self.images_list)

        if self.is_series:
            self.series_pulldown.options = self.series
            self.series_pulldown.value = self.series[0]
            self.selected_series = self.series[0]
            self.name_dropdown.options = self.current_series
            self.histogram_positions()
            self.series_scatter_pos(self.image_name)

        self.series_pulldown.on_change("value", self.series_cb)
        self.name_dropdown.on_change('value', self.name_cb)

        self.clip_button = Button(label="Clip", button_type="danger")

        self.gamma_slider = Slider(start=0, end=10, value=self.gamma, step=0.1, title="Gamma")
        self.gamma_slider.on_change('value_throttled', self.gamma_cb)

        step = 0.05
        self.window_slider = Slider(start=0, end=2., value=self.window, step=step, title="Window")
        self.window_slider.on_change('value_throttled', self.window_cb)

        self.log_div = Div(text="Log:<br>", width=400, height=200)

        self.exit_button = Button(label="Exit", button_type="danger")
        self.exit_button.on_click(self.stop_server)

        # Attach the callback to the TapTool's event
        self.taptool = TapTool()

        self.fig_image.add_tools(self.taptool)
        self.fig_image.on_event('tap', self.tap_callback)

    def histogram_positions(self):

        i_n = []
        z_i = []

        once = True
        self.series_pos_index = 2
        unit_normal = [0., 0., 1.]

        for i in self.current_series:
            if once:
                # decide which direction the image was taken in
                normal = self.series_map[self.selected_series][i][3]
                unit_normal = [0., 0., 1.]
                n_dot_z = np.dot(normal, unit_normal)
                if n_dot_z > 0.95:
                    self.series_pos_index = 2
                else:
                    unit_normal = [0., 1., 0.]
                    n_dot_y = np.dot(normal, unit_normal)
                    if n_dot_y > 0.95:
                        self.series_pos_index = 1
                    else:
                        unit_normal = [1., 0., 0.]
                        n_dot_x = np.dot(normal, unit_normal)
                        if n_dot_x > 0.95:
                            self.series_pos_index = 0

            sel_key_list = list(self.series_map[self.selected_series].keys())
            normal = self.series_map[self.selected_series][i][3]
            try:
                #i_n.append(float(i))
                i_n.append(self.series_map[self.selected_series][i][0])
            except ValueError:
                i_floated = float(sel_key_list.index(i))
                i_n.append(i_floated)

            z_i.append(self.series_map[self.selected_series][i][1][self.series_pos_index])

        self.fig_MRI_source_series.data = dict({"x": z_i, "y": i_n})

    def series_scatter_pos(self, image_name):

        sel_key_list = list(self.series_map[self.selected_series].keys())

        z_i = self.series_map[self.selected_series][image_name][1][self.series_pos_index]
        i_n = image_name
        try:
            #i_n = float(image_name)
            i_n = self.series_map[self.selected_series][image_name][0]
        except ValueError:
            i_n = float(sel_key_list.index(image_name))

        self.fig_MRI_source.data = dict(x=[z_i], y=[i_n])
        self.fig_series_positions.scatter(x="x", y="y", source=self.fig_MRI_source, color='red', size=6)

    def name_cb(self, attr, old, new):
        self.image_name = new
        self.generate_log_message(self.log_div, f"Get new file {self.image_name}")

        self.path = self.data_dir + self.image_name
        rc = self.prepare_images()
        self.source.data["image"] = [self.processed_image]
        self.size_figures()

        rc = self.set_image_fig_title(new)

    def set_image_fig_title(self, image_name):

        patient = self.ds.PatientName.given_name + " " + self.ds.PatientName.family_name
        try:
            proc_date = self.ds.PerformedProcedureStepStartDate
        except:
            proc_date = "1971-01-01"
        try:
            protocol = self.ds.ProtocolName
        except:
            protocol = "unknown"

        self.title_postfix = (" " + patient + " " + proc_date + " " +
                              protocol)
        self.fig_image.title.text = image_name + self.title_postfix

    def db_dropdown_cb(self, attr, old, new):
        """
        Change in dataset means essentially a restart and reset to most things

        :param attr:
        :param old:
        :param new:
        :return:
        """
        self.data_dir = self.data_db[new]

        rc = self.find_images()

        self.image_name = self.images_list[0]
        self.path = self.data_dir + self.image_name
        self.name_dropdown.options = self.images_list
        self.name_dropdown.remove_on_change("value", self.name_cb)
        self.name_dropdown.value = self.image_name
        self.name_dropdown.on_change("value", self.name_cb)

        rc = self.prepare_images()

        if self.is_series:
            rc = self.categorize_series()

            self.selected_series = self.series[0]
            self.series_pulldown.options = self.series
            self.current_slice = 0
            self.histogram_positions()
            self.series_scatter_pos(self.current_series[0])

            self.series_slider_slice.remove_on_change("value_throttled", self.series_slider_slice_cb)
            self.series_slider_slice.value = self.current_slice
            self.series_slider_slice.on_change("value_throttled", self.series_slider_slice_cb)
            self.series_slider_slice.end = len(self.current_series)

            self.series_pulldown.remove_on_change("value", self.series_cb)
            self.series_pulldown.value = self.current_series[0]
            self.series_pulldown.on_change("value", self.series_cb)

            self.name_dropdown.options = self.current_series

        self.size_figures()

        self.series_pulldown.visible = (self.is_series)
        self.series_toggle_anim.button_type = "success"

        self.gamma_slider.remove_on_change("value_throttled", self.gamma_cb)
        self.window_slider.remove_on_change("value_throttled", self.window_cb)

        self.gamma_slider.value = self.gamma_def
        self.window_slider.value = self.window_def

        self.gamma_slider.on_change("value_throttled", self.gamma_cb)
        self.window_slider.on_change("value_throttled", self.window_cb)

        self.CT_text_refresh.visible = self.is_series
        self.series_toggle_anim.visible = self.is_series
        self.series_slider_slice.visible = self.is_series
        self.fig_series_positions.visible = self.is_series
        self.increment_button.visible = self.is_series
        self.decrement_button.visible = self.is_series

        if "X-Ray" in self.sop_class_name:
            self.mode.active = 0
        elif "CT" in self.sop_class_name:
            self.mode.active = 1
        elif "MR" in self.sop_class_name:
            self.mode.active = 2
        else:  # ultrasound
            self.mode.active = 3

        rc = self.set_image_fig_title(self.image_name)

        self.source.data["image"] = [self.processed_image]

        self.generate_log_message(self.log_div, f"Get new imaging {new}")

    def series_cb(self, attr, old, new):
        """
        Select a new series. Set up for first image in that series.

        :param attr:
        :param old:
        :param new:
        :return:
        """
        self.selected_series = new
        self.generate_log_message(self.log_div, f"Get new series {self.selected_series}")

        #rc = self.categorize_series()

        current_series = []
        for i in self.series_map[self.selected_series]:
            current_series.append(str(i))

        self.current_series = current_series
        self.name_dropdown.options = self.current_series
        self.path = self.data_dir + self.current_series[0]

        self.histogram_positions()
        self.series_scatter_pos(self.current_series[0])
        self.fig_series_positions.title.text = self.fig_series_positions_titles[self.series_pos_index]
        self.series_toggle_anim.button_type = "success"

        rc = self.prepare_images()
        rc = self.set_image_fig_title(self.current_series[0])

        self.current_slice = 0
        self.size_figures()

        self.source.data["image"] = [self.processed_image]

        self.series_slider_slice.remove_on_change("value_throttled", self.series_slider_slice_cb)

        self.series_slider_slice.value = self.current_slice
        self.series_slider_slice.end = len(self.current_series)

        self.series_slider_slice.on_change("value_throttled", self.series_slider_slice_cb)

        self.generate_log_message(self.log_div, f"select new MRI series {new}")

    def gamma_cb(self, attr, old, new):

        self.gamma = new
        self.processed_image = self.perform_gamma(self.clipped_image)
        self.source.data["image"] = [self.processed_image]
        self.generate_log_message(self.log_div, f"set gamma to {self.gamma}")

    def window_cb(self, attr, old, new):

        self.color_mapper.high = self.max_bright * new
        self.generate_log_message(self.log_div, f"set window scale to {new}")

    def stop_server(self):
        """
         Stops the Bokeh server gracefully.

         This method updates the log message, changes the button color,
         and then stops the Tornado IOLoop to shut down the server.
         """

        curdoc().add_next_tick_callback(lambda: self.async_generate_log_message(self.log_div,
                                                                                "Server is shutting down..."))
        curdoc().add_next_tick_callback(lambda: self.change_button_color(self.exit_button, "light"))
        curdoc().add_next_tick_callback(self.exit_server)

    def generate_log_message(self, log_div, message):
        self.message_log.append(message)

        if len(self.message_log) > 10:
            self.message_log.pop(0)

        self.log_div.text = "Log: <br>" + "<br>".join(self.message_log)
        curdoc().add_next_tick_callback(lambda: None)

    async def async_generate_log_message(self, log_div, message):
        """
        Asynchronously updates the log message in the Div widget.

        Args:
            log_div (Div): The Div widget to update.
            message (str): The message to display in the log_div.
        """
        log_div.text = message

    async def exit_server(self):
        """
        Stops the Tornado IOLoop, effectively shutting down the server.
        """

        print("Server is shutting down...")
        IOLoop.current().stop()

    async def change_button_color(self, button, color):
        """
        Asynchronously changes the button color.

        Args:
            button (Button): The button widget whose color will be changed.
            color (str): The new color for the button.
        """
        button.button_type = color

    def tap_callback(self, event):
        """
        Handles the tap event: click on 4 corners of rectangle
        """
        selected = self.source.selected
        try:
            selected_index = [self.source.selected.image_indices[self.clipee]["i"],
                              self.source.selected.image_indices[self.clipee]["j"]]
            self.clip_points.append(selected_index)
            self.clipee = self.clipee + 1

            self.generate_log_message(self.log_div, f"Selected point {self.clipee} : {selected_index}")

            if self.clipee == 4:
                rotation_angle_degrees, min_x, max_x, min_y, max_y, center = (
                    self.rotated_rectangle_properties(self.clip_points))

                self.clipped_image = transform.rotate(image=self.clipped_image, angle=-rotation_angle_degrees*0.6,
                                                        center=(center[1], center[0]), preserve_range=True)

                self.processed_image = self.perform_gamma(self.clipped_image)

                dw = int(max_x - min_x)
                dh = int(max_y - min_y)

                print(dw, dh)
                self.fig_image.renderers = [r for r in self.fig_image.renderers if r != self.f_img]
                self.f_img = self.fig_image.image(image='image', x=min_x, y=min_y, dw=dw, dh=dh, source=self.source,
                                     color_mapper=self.color_mapper)
                self.fig_image.height = dh
                self.fig_image.y_range.start = min_y
                self.fig_image.y_range.end = max_y

                self.fig_image.width = dw
                self.fig_image.x_range.end = max_x
                self.fig_image.x_range.start = min_x

                self.source.data["image"] = [self.processed_image]
                self.generate_log_message(self.log_div, f"clipped and rotated by {rotation_angle_degrees:.2f}")
                self.clipee = 0
                self.clip_points = []
                self.source.selected.image_indices = []

        except IndexError:
            self.generate_log_message(self.log_div, "Hit whitespace! Try again")
            return

    def get_new_image(self):

        if not self.is_series:
            images_list = self.images_list
        else:
            images_list = self.current_series

        self.path = self.data_dir + images_list[self.current_slice]

        rc = self.prepare_images()
        rc = self.set_image_fig_title(images_list[self.current_slice])

        self.source.data["image"] = [self.processed_image]

    def increment_cb(self):

        if self.current_slice < len(self.current_series):
            self.current_slice += 1

            self.series_slider_slice.remove_on_change("value_throttled", self.series_slider_slice_cb)
            self.series_slider_slice.value = self.current_slice
            self.series_slider_slice.on_change("value_throttled", self.series_slider_slice_cb)

            self.get_new_image()
            self.generate_log_message(self.log_div, f"Increment image to {self.current_slice}")
        else:
            self.generate_log_message(self.log_div, f"No increment. At max image {self.current_slice}")

    def decrement_cb(self):

        if self.current_slice > 0:
            self.current_slice -= 1
            self.get_new_image()

            self.series_slider_slice.remove_on_change("value_throttled", self.series_slider_slice_cb)
            self.series_slider_slice.value = self.current_slice
            self.series_slider_slice.on_change("value_throttled", self.series_slider_slice_cb)

            self.generate_log_message(self.log_div, f"Decrement image to {self.current_slice}")
        else:
            self.generate_log_message(self.log_div, f"No decrement. At min image {self.current_slice}")

    def series_slider_slice_cb(self, attr, old, new):

        self.current_slice = int(new)

        self.get_new_image()

        self.generate_log_message(self.log_div, f"Slider reset to {self.current_slice}")

    def animate_series(self):
        if self.series_toggle_anim.active:
            if not self.is_series:
                images_list = self.images_list
            else:
                images_list = self.current_series

            self.name_dropdown.options = images_list

            image_name = images_list[self.current_slice]
            self.path = self.data_dir + image_name
            rc = self.prepare_images()

            if self.is_series:
                self.series_scatter_pos(image_name)

            self.fig_image.title.text = image_name + self.title_postfix
            self.source.data["image"] = [self.processed_image]

            self.current_slice += 1
            if self.current_slice >= len(images_list):
                self.current_slice = 0

            self.series_slider_slice.remove_on_change("value_throttled", self.series_slider_slice_cb)

            self.series_slider_slice.value = self.current_slice

            self.series_slider_slice.on_change("value_throttled", self.series_slider_slice_cb)

            if self.debug:
                self.generate_log_message(self.log_div, f"Animation image {image_name}")

    def series_toggle_anim_cb(self, active):

        self.series_toggle_active = active

        if self.series_toggle_active:
            self.series_toggle_anim.button_type = "success"
            self.series_anim_cb_ID = curdoc().add_periodic_callback(self.animate_series, int(self.series_animate_refresh_rate))
        else:
            self.series_toggle_anim.button_type = "danger"
            curdoc().remove_periodic_callback(self.series_anim_cb_ID)

    def rotated_rectangle_properties(self, corners):
        """
        Calculates the rotation angle, and min/max x and y values of a rotated rectangle.

        Args:
          corners: A list of four (x, y) tuples representing the corners of the rectangle,
                   in any order.  The order doesn't matter as long as they define a rectangle.

        Returns:
          A tuple containing:
            - rotation_angle: The rotation angle in degrees (float).  The angle will be
                              in the range -90 to +90 degrees.
            - min_x: The minimum x-coordinate of the rotated rectangle (float).
            - max_x: The maximum x-coordinate of the rotated rectangle (float).
            - min_y: The minimum y-coordinate of the rotated rectangle (float).
            - max_y: The maximum y-coordinate of the rotated rectangle (float).
        """

        # 1. Convert corners to a NumPy array
        corners = np.array(corners)

        # 2. Calculate the center of the rectangle
        center = np.mean(corners, axis=0)

        # 3. Find the two corners that form (approximately) the longer side.
        # This is done by finding the maximum distance between corners.
        max_dist = 0
        corner1_index = 0
        corner2_index = 1
        for i in range(4):
            for j in range(i + 1, 4):
                dist = np.linalg.norm(corners[i] - corners[j])
                if dist > max_dist:
                    max_dist = dist
                    corner1_index = i
                    corner2_index = j

        corner1 = corners[corner1_index]
        corner2 = corners[corner2_index]

        # 4. Calculate the rotation angle
        dx = corner2[0] - corner1[0]
        dy = corner2[1] - corner1[1]
        rotation_angle_radians = np.arctan2(dy, dx)
        rotation_angle_degrees = np.degrees(rotation_angle_radians)

        # Adjust the angle to be within the range of -90 to +90
        if rotation_angle_degrees > 90:
            rotation_angle_degrees -= 180
        elif rotation_angle_degrees < -90:
            rotation_angle_degrees += 180

        # 5. Create a rotation matrix
        angle_rad = np.radians(rotation_angle_degrees)
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                    [np.sin(angle_rad), np.cos(angle_rad)]])

        # 6. Rotate the corners around the center
        rotated_corners = []
        for corner in corners:
            v = corner - center
            rotated_v = np.dot(rotation_matrix, v)
            rotated_corner = rotated_v + center
            rotated_corners.append(rotated_corner)
        rotated_corners = np.array(rotated_corners)

        # 7. Find the min/max x and y values of the rotated rectangle
        min_x = np.min(rotated_corners[:, 0])
        max_x = np.max(rotated_corners[:, 0])
        min_y = np.min(rotated_corners[:, 1])
        max_y = np.max(rotated_corners[:, 1])

        print(f"Rotation Angle: {rotation_angle_degrees:.2f} degrees")
        print(f"Min X: {min_x:.2f}")
        print(f"Max X: {max_x:.2f}")
        print(f"Min Y: {min_y:.2f}")
        print(f"Max Y: {max_y:.2f}")
        print("Center: ", center)

        return rotation_angle_degrees, min_x, max_x, min_y, max_y, center

    def key_func(self, filename):
        """
        Extracts the character and number parts from the filename (if present)
        and returns a tuple that can be used for sorting.
        """
        match = re.match(r"([a-zA-Z]*)(\d+)", filename)  # Allow for optional characters
        if not match:
            try:
                # Try converting the filename to an integer directly if it's purely numeric
                num = int(filename)
                return ("", num)  # Treat as having empty characters and a number
            except ValueError:
                # Handle filenames that are neither character+number nor purely numeric
                return (filename, 0)  # Assign a low sort value
        chars = match.group(1)
        nums = int(match.group(2))
        return (chars, nums)

    def find_images(self):
        """
        locate all the pickle files in the data dir
        :return:
        """
        path = Path(self.data_dir)
        img_list = list(path.glob('*'))  # '*/' for non-recursive

        posix_list = np.sort([Path(file.as_posix()).name for file in img_list])
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        self.images_list = sorted(files, key=self.key_func)


d = dicom_viewer()
