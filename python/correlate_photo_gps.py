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
        self.do_all_photos = data["do_all_photos"]

        self.kml = simplekml.Kml()

    def correlate_gpx_photos(self):

        self.lat = []
        self.lon = []
        self.time = []

        # Create a TimezoneFinder instance
        self.tf = TimezoneFinder()

        self.current_photo = photo()
        self.photo_properties = {}

        if len(self.photo_list) == 0:
            path = Path(self.photo_dir)
            p_list = list(path.glob("*.[jJ][pP][gG]"))  # case-insensitive

            posix_p_list = np.sort([Path(file.as_posix()).name for file in p_list])
        else:
            posix_p_list = self.photo_list

        got_gps = "got_gps"
        self.photo_properties.setdefault(got_gps, {})

        print("Processing photos with gps already")
        for p in posix_p_list:
            p_loc = self.photo_dir + p
            p_name = str(p)
            self.current_photo.get_image(p_loc)
            if self.current_photo.latitude is None:
                continue

            self.photo_properties[got_gps].setdefault(p_name, {})
            self.photo_properties[got_gps][p_name]["name"] = p
            self.photo_properties[got_gps][p_name]["lat"] = self.current_photo.latitude
            self.photo_properties[got_gps][p_name]["lon"] = self.current_photo.longitude
            self.photo_properties[got_gps][p_name]["time"] = self.current_photo.create_date
            self.photo_properties[got_gps][p_name]["filespec"] = p_loc

        for g in self.gpx_list:

            g_name = str(Path(g).name)
            print("Processing gpx file ", g_name)

            self.photo_properties.setdefault(g_name, {})

            gpx_file = self.gpx_dir + g
            with (open(gpx_file, 'r') as g_file):
                gpx = gpxpy.parse(g_file)

            for pl in posix_p_list:

                p = str(pl)
                p_name = str(Path(p).name)

                photo_file = self.photo_dir + p

                self.current_photo.get_image(photo_file)

                if self.current_photo.latitude is not None:
                    continue

                rc = self.process_gpx(gpx=gpx)

                timezone_name = self.tf.timezone_at(lat=self.lat[0], lng=self.lon[0])
                timezone = pytz.timezone(timezone_name)
                time_tz = self.current_photo.create_date.astimezone(timezone)

                if time_tz < self.time[0] or time_tz > self.time[-1]:
                    #print("Photo not taken during gps route")
                    continue

                t_diff = []
                for i in range(len(self.time)):
                    t_diff.append((self.time[i] - time_tz).seconds)

                closest_to_zero = 100000
                closest_index = -1
                for i in range(len(t_diff)):
                    if t_diff[i] < closest_to_zero:
                        closest_to_zero = t_diff[i]
                        closest_index = i
                latitude = self.lat[closest_index]
                longitude = self.lon[closest_index]
                time = self.time[closest_index]

                self.photo_properties[g_name].setdefault(p_name, {})
                self.photo_properties[g_name][p_name]["name"] = p
                self.photo_properties[g_name][p_name]["lat"] = latitude
                self.photo_properties[g_name][p_name]["lon"] = longitude
                self.photo_properties[g_name][p_name]["time"] = time
                self.photo_properties[g_name][p_name]["filespec"] = photo_file

        return self.photo_properties

    def print_photos_gps(self):

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
                    #self.time.append(point.time)

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
        self.caption = None
        self.create_date = ""
        self.camera = ""
        self.file_location = ""
        self.image_size = ""

        self.big_loc = ""
        self.tn_loc = ""

        self.image_obj = ""

        self.latitude = None
        self.longitude = None

    def dms_to_dd(self, deg, minutes, seconds, hemi):
        sign = -1 if hemi.upper() in ('S', 'W') else 1

        return float(sign * (abs(deg) + minutes / 60 + seconds / 3600))

    def get_image(self, file_location=None):

        self.file_location = file_location

        self.image_obj = Image.open(file_location)
        self.image_size = self.image_obj.size

        iptc = IptcImagePlugin.getiptcinfo(self.image_obj)
        self.caption = None

        try:
            self.caption = iptc[(2, 120)].decode("utf-8")
        except:
            try:
                self.caption = iptc[(2, 120)].decode("cp1252")
            except:
                #print("Failed to get caption", file_location, iptc)
                pass
        self.longitude = None
        self.latitude = None

        # https://hhsprings.bitbucket.io/docs/programming/examples/python/PIL/ExifTags.html
        _TAGS_r = dict(((v, k) for k, v in TAGS.items()))
        exifd = self.image_obj._getexif()
        keys = list(exifd.keys())
        try:
            t = exifd[36868]
        except:
            try:
                t = exifd[36867]
            except:
                t = exifd[306]    # unusual location used by Photoshop saving HEIC files as JPG
                pass

        self.create_date = datetime.strptime(t, '%Y:%m:%d %H:%M:%S')
        try:
            self.camera = exifd[272]
        except:
            self.camera = "Unknown camera"
            pass
        try:
            gpsinfo = exifd[_TAGS_r["GPSInfo"]]

            self.latitude = self.dms_to_dd(deg=gpsinfo[2][0], minutes=gpsinfo[2][1],
                                 seconds=gpsinfo[2][2], hemi=gpsinfo[1])
            self.longitude = self.dms_to_dd(deg=gpsinfo[4][0], minutes=gpsinfo[4][1],
                                  seconds=gpsinfo[4][2], hemi=gpsinfo[3])
        except:
            pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--app_config',
                        default="/Volumes/Data/Home/correlate_photo_gps.yaml",
                        help="overall app config file")

    args = parser.parse_args()

    a = correlate_photo_gps(config=args.app_config)

    p = a.correlate_gpx_photos()
    rc = a.print_photos_gps()
