import gpxpy
from geopy.distance import great_circle
import simplekml
import argparse
from datetime import datetime, timedelta
from pyproj import Transformer
import os
from pathlib import Path
import numpy as np
from timezonefinder import TimezoneFinder
import pytz
from scipy.signal import savgol_filter

from bokeh.models import BasicTicker, HoverTool, ColumnDataSource, WheelZoomTool, LinearColorMapper, ColorBar
from bokeh.plotting import figure, output_file, reset_output, show, save
from bokeh.layouts import row, layout, column
from bokeh.models.widgets import Div
from bokeh.palettes import Viridis256
from bokeh.transform import linear_cmap

import xyzservices.providers as xyz

parser = argparse.ArgumentParser(description='process gpx files')
parser.add_argument('--infile',
                    default="/Users/richarddubois/Code/Home/Strava_gpx/Paris_2025/",
                    help="input gpx file")
parser.add_argument('--map',
                    default="NatGeoWorldMap",
                    help="Esri map type")
parser.add_argument('--merge',
                    default="no",
                    help="Merge all gpx files in current directory")
parser.add_argument('--time_delta',
                    default=1.,
                    help="# secs per step")

args = parser.parse_args()

# Set up coordinate transformation (lat/lon to Web Mercator)


def latlon_to_mercator(lat, lon):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857")
    return transformer.transform(lat, lon)


def smooth_speeds(speeds, window_length=41, polyorder=2):
    """Smooths speed data using a Savitzky-Golay filter."""
    if len(speeds) < window_length:
        return speeds  # Not enough data for smoothing
    if window_length % 2 == 0:
       window_length += 1 # window_length must be odd, so add 1 if even
    try:
        smoothed_speeds = savgol_filter(speeds, window_length, polyorder)
        return smoothed_speeds
    except Exception as e:
        print(f"Savgol filter failed {e}")
        return speeds  #Return data as is

#infile = "/Users/richarddubois/Code/Home/Strava_gpx/Dolphin_Lodge_Rio_Panayacu_downstream_Afternoon_Kayaking_2024_10_27.gpx"
#infile = "/Users/richarddubois/Code/Home/Strava_gpx/Dolphin_Lodge_to_Napo_aborted_midway_2024_10_29.gpx"
#infile = "/Users/richarddubois/Code/Home/Strava_gpx/Dolphin_Lodge_Rio_Panayacu_Morning_Kayaking_2024_10_28.gpx"

if args.merge == "yes":

    # List all .gpx files in the directory
    gpx_files = [file for file in os.listdir(args.infile) if file.endswith('.gpx')]

    # Create a new GPX object
    merged_gpx = gpxpy.gpx.GPX()

    for file in gpx_files:
        print("working on " + file)
        with open(file, "r") as gpx_file:
            gpx = gpxpy.parse(gpx_file)
            for track in gpx.tracks:
                merged_gpx.tracks.append(track)

    # Save the merged GPX file
    with open("merged_tracks.gpx", "w") as f:
        f.write(merged_gpx.to_xml())

    # Create a new KML object
    kml = simplekml.Kml()

    # Extract tracks from the GPX file and add them to KML
    for track in merged_gpx.tracks:
        for segment in track.segments:
            kept = []
            last_time = None
            for point in segment.points:
                if last_time is None or (point.time - last_time) >= timedelta(seconds=args.time_delta):
                    kept.append(point)
                    last_time = point.time
            kml_coords = [(point.longitude, point.latitude) for point in kept]
            kml.newlinestring(name=track.name, coords=kml_coords)

    # Save the KML file
    kml.save(args.infile + "/merged_tracks.kml")
    exit(0)

infile = args.infile

html_stem = Path(infile).stem
html_name = html_stem + ".html"
print(infile, html_name)

# Open the GPX file
with (open(infile, 'r') as gpx_file):
    gpx = gpxpy.parse(gpx_file)

lat = []
lon = []
time = []
t_diff_secs = []
speed = []
elev = []
total_distance = []

next_merc_x = []
next_merc_y = []

# Create a KML object for export to Google maps

kml = simplekml.Kml()

# Create a TimezoneFinder instance
tf = TimezoneFinder()

running_distance = 0.

# Access data in the GPX file
for track in gpx.tracks:
    for segment in track.segments:
        line = kml.newlinestring()
        coords = []
        for ip, point in enumerate(segment.points):
            #print(f'Latitude: {point.latitude}, Longitude: {point.longitude}, Elevation: {point.elevation}')
            lat.append(point.latitude)
            lon.append(point.longitude)

            timezone_name = tf.timezone_at(lat=point.latitude, lng=point.longitude)
            timezone = pytz.timezone(timezone_name)
            time_tz = point.time.astimezone(timezone)
            time.append(time_tz)

            elev.append(point.elevation*3.28084)
            distance = 0.

            if ip > 0:
                distance = great_circle((lat[ip-1], lon[ip-1]), (lat[ip], lon[ip])).miles

                time_diff = (time[ip] - time[ip-1]).total_seconds()/3600.  #hrs

                if time_diff > 0.:
                    p_speed = distance / time_diff
                else:
                    p_speed = 0
            else:
                p_speed = 0
                time_diff = 0.

            t_diff_secs.append(time_diff*3600.)
            speed.append(p_speed)
            running_distance += distance
            total_distance.append(running_distance)

            #line.coords = [(point.longitude, point.latitude, point.elevation) for point in segment.points]
            coords.append((point.longitude, point.latitude, point.elevation))
            line.altitudemode = simplekml.AltitudeMode.clamptoground
        line.coords = coords

kml_name = html_stem + ".kml"
kml.save(infile + "/" + kml_name)

smoothed_speeds = smooth_speeds(speed)


# Convert the latitude and longitude coordinates to Web Mercator
mercator_x, mercator_y = zip(*(latlon_to_mercator(lat_t, lon_t) for lat_t, lon_t in zip(lat, lon)))

# Append all but the last element
for i in range(len(mercator_x) - 1):
    next_merc_x.append(mercator_x[i+1])
next_merc_x.append(mercator_x[-1])

for i in range(len(mercator_y) - 1):
    next_merc_y.append(mercator_y[i+1])
next_merc_y.append(mercator_y[-1])
source = ColumnDataSource(data=dict(lat=lat, lon=lon, time=time, merc_x=mercator_x, merc_y=mercator_y,
                                    smoothed_speed=smoothed_speeds, speed=speed,
                                    next_merc_x=next_merc_x, next_merc_y=next_merc_y, elev=elev, dist=total_distance,))

tooltips = [('Latitude', '@lat'), ('Longitude', '@lon'), ('Speed', '@speed'),
            ('Time (UTC-7)', '@time{%F %H:%M}'), ('Elevation', '@elev'), ('Distance', '@dist')]

lat_min, lat_max = min(lat), max(lat)
buffer_lat = 0.05 * (lat_max - lat_min)
lon_min, lon_max = min(lon), max(lon)
buffer_lon = 0.05 * (lon_max - lon_min)

#buffer_lon = 0
#buffer_lat = 0.

b_lat_min = lat_min-buffer_lat
b_lat_max = lat_max+buffer_lat
b_lon_min = lon_min-buffer_lon
b_lon_max = lon_max+buffer_lon

# Convert lat/lon to Web Mercator (required for tile providers)
x_min, y_min = latlon_to_mercator(b_lat_min, b_lon_min)
x_max, y_max = latlon_to_mercator(b_lat_max, b_lon_max)

print(buffer_lat, buffer_lon)
print(lat_min, lat_max, lon_min, lon_max)
print(b_lat_min, b_lat_max, b_lon_min, b_lon_max)
print(x_min, x_max, y_min, y_max)

t_hist = figure(title="Latitude vs Longitude",
                    x_axis_label='Longitude (deg)', y_axis_label='Latitude (deg)',
                    width=1000, tooltips=tooltips, x_range=(x_min, x_max), y_range=(y_min, y_max),
                x_axis_type="mercator", y_axis_type="mercator")

# https://xyzservices.readthedocs.io/en/stable/introduction.html
t_hist.add_tile(getattr(xyz.Esri,args.map))
t_hist.line(x="merc_x", y="merc_y", source=source, color="red")

print("min speed", min(smoothed_speeds), "max speed", max(smoothed_speeds))
# Create a linear color mapper based on speed
#mapper = LinearColorMapper(palette=Viridis256, low=0., high=1.)
#t_hist.segment(x0='merc_x', y0='merc_y', x1='next_merc_x', y1='next_merc_y', line_width=2,
#               color=linear_cmap('speed', palette=Viridis256, low=min(speed), high=max(speed)), source=source)
# Add a color bar
#color_bar = ColorBar(color_mapper=mapper, location=(0, 0))
#_hist.add_layout(color_bar, 'right')

# Add HoverTool with formatting
hover = HoverTool(tooltips=tooltips, formatters={'@time': 'datetime'})
t_hist.add_tools(hover)
wheel_zoom = WheelZoomTool()
t_hist.add_tools(wheel_zoom)

e_hist = figure(title="Elevation vs Time", x_axis_label='Time', y_axis_label='Elevation', width=750,
                x_axis_type="datetime", tooltips=tooltips)
e_hist.line(x="time", y="elev", source=source, color="blue")
e_hist.add_tools(hover)

d_hist = figure(title="Elevation vs Distance", x_axis_label='Distance (miles)', y_axis_label='Elevation', width=750,
                tooltips=tooltips)
d_hist.line(x="dist", y="elev", source=source, color="red")
d_hist.add_tools(hover)

delta_t_hist = figure(title="Time differences between measurements")

delta_t_h, delta_t_edges = np.histogram(t_diff_secs, bins=100, range=(0.8, 1.2))
width = delta_t_edges[1] - delta_t_edges[0]
delta_t_hist.vbar(top=delta_t_h, x=delta_t_edges[1:], width=width, alpha=0.3, fill_color="red" )

speed_hist = figure(title="Speed vs Distance", x_axis_label='Distance (miles)', y_axis_label='Speed', width=750,
                tooltips=tooltips)
speed_hist.line(x="dist", y="speed", source=source, color="red")
speed_hist.line(x="dist", y="smoothed_speed", source=source, color="blue")
speed_hist.add_tools(hover)

fn = gpx_file.buffer.name
del_div = Div(text= fn + " Run on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

output_file(html_name)
l = layout( del_div, t_hist, e_hist, d_hist, delta_t_hist, speed_hist)
save(l, title="Strava Track Lat vs Lon")
