import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, layout, row
from bokeh.models import ColumnDataSource, Slider, Button, Toggle, CustomJS, Div
from bokeh.plotting import figure, save, show, output_file

import yaml
import argparse
from pyproj import Transformer
from timezonefinder import TimezoneFinder
import pytz
import xyzservices.providers as xyz
import gpxpy
from geopy.distance import great_circle
import simplekml
from datetime import datetime, timedelta
import os
from pathlib import Path


class animate_gps_trajectories():
    def __init__(self, config=None):

        with open(config, "r") as f:
            data = yaml.safe_load(f)

        self.data_dir = data['data_dir']
        self.html_dir = data['html_dir']
        self.infile_list = data["infile_list"]
        self.map = data["map"]
        self.refresh_rate = data["refresh_rate"]
        self.slider_start = data["slider_start"]
        self.slider_end = data["slider_end"]

        self.html_name = self.html_dir + Path(self.infile_list[0]).stem + ".html"

        # Create a KML object for export to Google maps

        self.kml = simplekml.Kml()

        # Create a TimezoneFinder instance
        self.tf = TimezoneFinder()

        # Generate the sine wave data
        x = np.linspace(0, 2 * np.pi, 100)  # 100 points from 0 to 2*pi
        y = np.sin(x)
        idx = []

        self.source = ColumnDataSource()
        self.mercator_x = []
        self.mercator_y = []

        # Create a ColumnDataSource
        #source = ColumnDataSource(data=dict(x=[], y=[], idx=[]))

        # Create a Bokeh figure
        #p = figure(title="Sine Wave Animation", x_axis_label='X', y_axis_label='Y', width=600, height=400,
        #           x_range=(0, 2 * np.pi), y_range=(-1., 1.))
        #p.scatter('x', 'y', source=source, size=10)

        # Initial parameters
        #index = 0
        #refresh_rate = 500  # Default refresh rate in milliseconds

        # Create a slider for refresh rate
        step = (self.slider_end - self.slider_start)/100.
        self.refresh_slider = Slider(start=self.slider_start, end=self.slider_end, value=self.refresh_rate, step=step,
                                     title="Refresh Rate (ms)")

        # Create a Toggle button for start/stop
        self.toggle = Toggle(label="Start", button_type="success", active=False)

        for infile in self.infile_list:
            g_file = self.data_dir + "/" + infile
            with (open(g_file, 'r') as gpx_file):
                gpx = gpxpy.parse(gpx_file)

            rc = self.process_gpx(gpx=gpx)

            idx = np.zeros(2)

            self.source_anim = ColumnDataSource(data=dict(x=[], y=[]))
            # Create the JavaScript callback for updating the plot
            self.callback = CustomJS(args=dict(source=self.source_anim, x=self.mercator_x, y=self.mercator_y, index=idx),
                                     code="""

                        const data = source.data;
                        const length = x.length;
                        //console.log('Entered callback for ', length, index[0]);

                        // Update data with the next two points based on the index
                        data['x'] = [x[index[0]], x[(index[0] + 1) % length]];
                        data['y'] = [y[index[0]], y[(index[0] + 1) % length]];
                        // data['idx'] = [index[0], index[0]]
                        //console.log('x,y = ', source.data['x'], source.data['y']);
                        source.change.emit();

                        // Update the index
                        index[0] = (index[0] + 1) % length;
                        //console.log('Updated index', index[0]);
                    """)

            div_text, anim = self.setup_figures(name=infile)

        # Toggle button callback
        self.toggle.js_on_change("active", CustomJS(args=dict(slider=self.refresh_slider,
                                                              toggle=self.toggle, callback=self.callback), code="""
                    if (toggle.active) {
                        toggle.label = "Stop";  // Change button label to Stop
                        // Start the animation loop
                        let idx = 0
                        const refreshRate = slider.value;  // Get refresh rate from the slider
                        //console.log('Slider - Refresh rate', refreshRate);
                        const animate = () => {
                            if (toggle.active) {
                                //console.log('toggle - idx ');
                                callback.execute();  // Call the update function
                                setTimeout(animate, refreshRate);  // Schedule the next update
                            }
                        };

                        animate();  // Start the animation sequence
                    } else {
                        toggle.label = "Start";  // Change button label to Start
                    }
                """))

        # Layout the application
        canvas_layout = column(div_text, self.toggle, self.refresh_slider, anim)
        output_file(self.html_name)

        show(canvas_layout)

    def latlon_to_mercator(self, lat, lon):
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857")
        return transformer.transform(lat, lon)

    def process_gpx(self, gpx):

        self.lat = []
        self.lon = []
        self.time = []
        self.speed = []
        self.elev = []
        self.total_distance = []

        self.next_merc_x = []
        self.next_merc_y = []

        running_distance = 0.

        # Access data in the GPX file
        for track in gpx.tracks:
            for segment in track.segments:
                line = self.kml.newlinestring()
                coords = []
                for ip, point in enumerate(segment.points):
                    # print(f'Latitude: {point.latitude}, Longitude: {point.longitude}, Elevation: {point.elevation}')
                    self.lat.append(point.latitude)
                    self.lon.append(point.longitude)

                    timezone_name = self.tf.timezone_at(lat=point.latitude, lng=point.longitude)
                    timezone = pytz.timezone(timezone_name)
                    time_tz = point.time.astimezone(timezone)
                    self.time.append(time_tz)

                    self.elev.append(point.elevation * 3.28084)
                    distance = 0.

                    if ip > 0:
                        distance = great_circle((self.lat[ip - 1], self.lon[ip - 1]),
                                                (self.lat[ip], self.lon[ip])).miles

                        time_diff = (self.time[ip] - self.time[ip - 1]).total_seconds() / 3600.  # hrs

                        if time_diff > 0.:
                            p_speed = distance / time_diff
                        else:
                            p_speed = 0
                    else:
                        p_speed = 0

                    self.speed.append(p_speed)
                    running_distance += distance
                    self.total_distance.append(running_distance)

                    # line.coords = [(point.longitude, point.latitude, point.elevation) for point in segment.points]
                    coords.append((point.longitude, point.latitude, point.elevation))
                    line.altitudemode = simplekml.AltitudeMode.clamptoground
                line.coords = coords

        # Convert the latitude and longitude coordinates to Web Mercator
        self.mercator_x, self.mercator_y = zip(*(self.latlon_to_mercator(lat_t, lon_t)
                                                 for lat_t, lon_t in zip(self.lat, self.lon)))

        # Append all but the last element
        for i in range(len(self.mercator_x) - 1):
            self.next_merc_x.append(self.mercator_x[i + 1])
        self.next_merc_x.append(self.mercator_x[-1])

        for i in range(len(self.mercator_y) - 1):
            self.next_merc_y.append(self.mercator_y[i + 1])
        self.next_merc_y.append(self.mercator_y[-1])

        lat_min, lat_max = min(self.lat), max(self.lat)
        buffer_lat = 0.05 * (lat_max - lat_min)
        lon_min, lon_max = min(self.lon), max(self.lon)
        buffer_lon = 0.05 * (lon_max - lon_min)

        # buffer_lon = 0
        # buffer_lat = 0.

        b_lat_min = lat_min - buffer_lat
        b_lat_max = lat_max + buffer_lat
        b_lon_min = lon_min - buffer_lon
        b_lon_max = lon_max + buffer_lon

        # Convert lat/lon to Web Mercator (required for tile providers)
        self.x_min, self.y_min = self.latlon_to_mercator(b_lat_min, b_lon_min)
        self.x_max, self.y_max = self.latlon_to_mercator(b_lat_max, b_lon_max)

        print(buffer_lat, buffer_lon)
        print(lat_min, lat_max, lon_min, lon_max)
        print(b_lat_min, b_lat_max, b_lon_min, b_lon_max)
        print(self.x_min, self.x_max, self.y_min, self.y_max)

    def setup_figures(self, name=None):

        self.source = ColumnDataSource(
            data=dict(lat=self.lat, lon=self.lon, time=self.time, merc_x=self.mercator_x, merc_y=self.mercator_y,
                      speed=self.speed,
                      next_merc_x=self.next_merc_x, next_merc_y=self.next_merc_y,
                      elev=self.elev, dist=self.total_distance))

        tooltips = [('Latitude', '@lat'), ('Longitude', '@lon'), ('Speed', '@speed'),
                    ('Time (UTC-7)', '@time{%F %H:%M}'), ('Elevation', '@elev'), ('Distance', '@dist')]

        t_hist = figure(title="Latitude vs Longitude",
                        x_axis_label='Longitude (deg)', y_axis_label='Latitude (deg)',
                        width=1000, tooltips=tooltips,
                        x_range=(self.x_min, self.x_max), y_range=(self.y_min, self.y_max),
                        x_axis_type="mercator", y_axis_type="mercator")

        # https://xyzservices.readthedocs.io/en/stable/introduction.html
        t_hist.add_tile(getattr(xyz.Esri, self.map))
        t_hist.scatter(x="x", y="y", source=self.source_anim, color="red", size=10)

        del_div = Div(text=name + " Run on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        return del_div, t_hist


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--app_config',
                        default="/Volumes/Data/Home/animate_gps.yaml",
                        help="overall app config file")

    args = parser.parse_args()

    a = animate_gps_trajectories(config=args.app_config)
