import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, layout, row
from bokeh.models import ColumnDataSource, Slider, Button, Toggle, CustomJS, Div, Legend, Span
from bokeh.plotting import figure, save, show, output_file
from bokeh.palettes import Viridis256

from PIL import Image, IptcImagePlugin
from PIL.ExifTags import TAGS
import datetime

import yaml
import argparse
from timezonefinder import TimezoneFinder
import pytz
import gpxpy
from geopy.distance import great_circle
import simplekml
from datetime import datetime, timedelta
import os
from pathlib import Path


class correlate_photo_gps():
    """
    Find the corresponding time to photo creation in a gpx file and extract the (lat, lon) the photot
    was taken at.

    If a null list of photos is provided in the yaml file, it will look for all JPGs in the photos directory.
    """
    def __init__(self, config=None):

        with open(config, "r") as f:
            data = yaml.safe_load(f)

        self.gpx_dir = data['gpx_dir']
        self.photo_dir = data['photo_dir']
        self.gpx_list = data["gpx_list"]
        self.photo_list = data["photo_list"]

        self.kml = simplekml.Kml()

        self.lat = []
        self.lon = []
        self.time = []

        # Create a TimezoneFinder instance
        self.tf = TimezoneFinder()

        self.current_photo = photo()
        self.photo_properties = {}

        for g in self.gpx_list:

            self.photo_properties.setdefault(g, {})

            gpx_file = self.gpx_dir + g
            with (open(gpx_file, 'r') as g_file):
                gpx = gpxpy.parse(g_file)

            self.process_gpx(gpx=gpx)

            if len(self.photo_list) == 0:
                path = Path(self.photo_dir)
                p_list = list(path.glob('*.JPG'))  # '*/' for non-recursive

                posix_p_list = np.sort([Path(file.as_posix()).name for file in p_list])
            else:
                posix_p_list = self.photo_list

            for p in posix_p_list:

                photo_file = self.photo_dir + p

                self.current_photo.get_image(photo_file)
                timezone_name = self.tf.timezone_at(lat=self.lat[0], lng=self.lon[0])
                timezone = pytz.timezone(timezone_name)
                t_photo = timezone.localize(self.current_photo.create_date)
                time_tz = t_photo.astimezone(timezone)

                if time_tz < self.time[0] or time_tz > self.time[-1]:
                    print("Photo not taken during gps route")
                    break

                t_diff = []
                for i in range(len(self.time)):
                    t_diff.append((self.time[i] - t_photo).seconds)

                closest_to_zero = 100000
                closest_index = -1
                for i in range(len(t_diff)):
                    if t_diff[i] < closest_to_zero:
                        closest_to_zero = t_diff[i]
                        closest_index = i

                self.photo_properties[g].setdefault(p, {})
                self.photo_properties[g][p]["name"] = p
                self.photo_properties[g][p]["lat"] = self.lat[closest_index]
                self.photo_properties[g][p]["lon"] = self.lon[closest_index]
                self.photo_properties[g][p]["time"] = self.time[closest_index]

        print("  Name          LAT        LON              Time")
        for g in self.gpx_list:
            print(g)
            for p in self.photo_properties[g]:
                print(p, self.photo_properties[g][p]["lat"], self.photo_properties[g][p]["lon"],
                      self.photo_properties[g][p]["time"])

    def process_gpx(self, gpx):
        """
        Take the gpx file and extract (lat, lon, time)
        :param gpx: input gpx file
        :return:
        """

        self.lat = []
        self.lon = []
        self.time = []

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

                    distance = 0.

                    if ip > 0:
                        distance = great_circle((self.lat[ip - 1], self.lon[ip - 1]),
                                                (self.lat[ip], self.lon[ip])).miles

                    running_distance += distance

                    # line.coords = [(point.longitude, point.latitude, point.elevation) for point in segment.points]
                    coords.append((point.longitude, point.latitude, point.elevation))
                    line.altitudemode = simplekml.AltitudeMode.clamptoground
                line.coords = coords


class photo():

    def __init__(self, name=None):

        self.name = name
        self.caption = ""
        self.create_date = ""
        self.camera = ""
        self.file_location = ""
        self.image_size = ""

        self.big_loc = ""
        self.tn_loc = ""

        self.image_obj = ""

    def get_image(self, file_location=None):

        self.file_location = file_location

        self.image_obj = Image.open(file_location)
        self.image_size = self.image_obj.size

        # https://hhsprings.bitbucket.io/docs/programming/examples/python/PIL/ExifTags.html
        _TAGS_r = dict(((v, k) for k, v in TAGS.items()))
        exifd = self.image_obj._getexif()
        keys = list(exifd.keys())
        try:
            t = exifd[36868]
        except KeyError:
            t = exifd[36867]

        self.create_date = datetime.strptime(t, '%Y:%m:%d %H:%M:%S')
        self.camera = exifd[272]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--app_config',
                        default="/Volumes/Data/Home/correlate_photo_gps.yaml",
                        help="overall app config file")

    args = parser.parse_args()

    a = correlate_photo_gps(config=args.app_config)
