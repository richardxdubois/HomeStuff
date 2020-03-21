from datetime import datetime, timedelta, date
import calendar
import argparse
import math
import pandas
import geopandas as gpd
import json
from collections import OrderedDict
from unidecode import unidecode
from scipy import stats
from scipy.interpolate import interp1d
from bokeh.plotting import figure, output_file, reset_output, show, save, curdoc
from bokeh.layouts import row, layout, column
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar, LogColorMapper
from bokeh.palettes import Category20
from bokeh.models.widgets import Tabs, Panel, DataTable, TableColumn, DateFormatter, \
    NumberFormatter, Div
from bokeh.models import ColumnDataSource, Span, Label, HoverTool, DatetimeTickFormatter
import numpy as np
from bokeh.palettes import Viridis256 as palette
from bokeh.palettes import Plasma256 as palette2
from bokeh.palettes import Paired as palette3

"""
Analysis of Johns Hopkins U covid-19 data:

Read a csv file of CV-19 data

"""

# borrowed from scipy example: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """


    if x.ndim != 1:
        print("smooth only accepts 1 dimension arrays.")
        raise ValueError


    if x.size < window_len:
        print("Input vector needs to be bigger than window size.")
        raise ValueError

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        print("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
        raise ValueError

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[int((window_len/2-1)+1):-int(window_len/2)]  # note restricted length returned


def substitute_name(name):
    new_name = name
    if name == "United States of America":
        new_name = "US"
    elif name == "United Kingdom of Great Britain and Northern Ireland":
        new_name = "United Kingdom"
    elif name == "Republic of Korea":
        new_name = "Korea, South"
    elif name == "Iran (Islamic Republic of)":
        new_name = "Iran"
    elif name == "Republic of Moldova":
        new_name = "Moldova"

    return new_name


def find_start(t_array=None, c_array=None, thresh=5.):

    t_start = datetime.strptime("1/1/20", '%m/%d/%y')
    running = 0

    for idt, c in enumerate(c_array):
        running += c
        if running > thresh:
            t_start = t_array[idt]
            break

    return t_start


def make_time_plot(place_1, place_dict, place_2=None):

    x_1 = np.array(list(place_dict[place_1].keys()))
    y_1 = np.array(list(place_dict[place_1].values()))

    # just to be sure, order the dates
    order = np.argsort(np.array(x_1))
    x_sorted = np.array(x_1)[order]
    y_sorted = np.array(y_1)[order]

    y_smooth = smooth(y_1)

    if place_2 is not None:
        x_2 = np.array(list(place_dict[place_2].keys()))
        y_2 = np.array(list(place_dict[place_2].values()))

        # just to be sure, order the dates
        order = np.argsort(np.array(x_2))
        x_2_sorted = np.array(x_2)[order]
        y_2_sorted = np.array(y_2)[order]
        y_2_smooth = smooth(y_2_sorted)

        len_diff = len(y_sorted) - len(y_2_sorted)
        if len_diff > 0:
            y_2_smooth = np.append(y_2_smooth, np.zeros(len_diff))

        m_hover = ColumnDataSource(dict({"date": x_sorted, "counts": y_sorted, "counts2": y_2_smooth}))
        m_title = "counts vs Date: " + args.type + " : " + place_1 + " overlaid with " + place_2 + " (black)"
    else:
        m_hover = ColumnDataSource(dict({"date": x_sorted, "counts": y_sorted}))
        m_title = "counts vs Date: " + args.type + " : " + place_1

    m = figure(tools=TOOLS, title=m_title, x_axis_type="datetime", x_axis_label='Date',
               y_axis_label='counts', width=750, y_axis_type="log")

    m_hover_tool = HoverTool(tooltips=[("date", "@date{%F}"), ("counts", "@counts")])
    m_hover_tool.formatters = {"date": "datetime"}
    m.add_tools(m_hover_tool)
    m.line(x="date", y="counts", color="navy", source=m_hover)
    m.line(x=x_sorted, y=y_smooth, line_color="red")

    if place_2 is not None:
        m.line(x=x_sorted, y=y_2_smooth, line_color="black")

    return m


# Command line arguments
parser = argparse.ArgumentParser(description='Analyze the SRM-derived WBS information')

parser.add_argument('--files', default='jhu_files.txt',help="lists of files (default=%(default)s)")
parser.add_argument('-o', '--output', default='jhu_analysis.html',
                    help="output bokeh html file (default=%(default)s)")
parser.add_argument('-c', '--country', default=None,
                    help="select country (default=%(default)s)")
parser.add_argument('-t', '--type', default="Deaths",
                    help="select country (default=%(default)s)")

args = parser.parse_args()

# get list of data files

jhu_files = {}
with open(args.files) as f:
    for line in f:
        (key, val) = line.split()
        jhu_files.setdefault(key, [])
        jhu_files[key].append(val)

country_counts = OrderedDict()
dates_counts = OrderedDict()
country_dates_counts = OrderedDict()

state_counts = OrderedDict()
state_dates_counts = OrderedDict()

n_dates = 0

today = datetime.today()

# parse the JHU csv files

for cats in jhu_files[args.type]:
    print("Using file ", cats)
    csv_assign = pandas.read_csv(cats, header=0, skipinitialspace=True)
    csv_drop_cols = csv_assign.dropna(axis="columns", how="all")
    csv_final = csv_drop_cols.fillna(0.)

    # loop over dates per country
    n_dates = 0

    for index, c_row in csv_final.iterrows():
        country_region = unidecode(c_row["Country/Region"])
        if "Province" in country_region:  # ignore the header row
            continue
        if "Mainland" in country_region:
            country_region = "China"

        # loop over dates per country
        n_dates = 0
        for id, jhu_dates in enumerate(csv_final.columns):

            try:
                dt_date = datetime.strptime(jhu_dates, '%m/%d/%y')
            except ValueError:
                try:
                    dt_date = datetime.strptime(jhu_dates, '%m/%d/%y %H:%M')
                except ValueError:
                    continue

            if id > 4:  # account for first 4 rows of header as non-dates
                day_count = float(c_row[id]) - float(c_row[id-1])
            else:
                day_count = float(c_row[id])

            if day_count == 0:
                continue

            try:
                country_counts[country_region] += day_count
            except KeyError:
                country_counts[country_region] = day_count

            c = country_dates_counts.setdefault(country_region, OrderedDict())
            d = country_dates_counts[country_region].setdefault(dt_date, 0)
            country_dates_counts[country_region][dt_date] += day_count

            if country_region == "US":
                state = c_row["Province/State"]
                if "," not in state:
                    s = state_dates_counts.setdefault(state, OrderedDict())
                    t = state_dates_counts[state].setdefault(dt_date, 0)
                    state_dates_counts[state][dt_date] += day_count

                    sc = state_counts.setdefault(state, 0)
                    state_counts[state] += day_count

            try:
                dates_counts[dt_date] += day_count
            except KeyError:
                dates_counts[dt_date] = day_count

country_counts_sorted = OrderedDict({k: v for k, v in sorted(country_counts.items(), key=lambda item: item[1],
                                                            reverse=True)})
print(country_counts)
print(dates_counts)

state_counts_sorted = OrderedDict({k: v for k, v in sorted(state_counts.items(), key=lambda item: item[1],
                                                            reverse=True)})

TOOLS = "reset,save"
x = []
y = []

if args.country is None:
    x = list(dates_counts.keys())
    y = list(dates_counts.values())
else:
    x = list(country_dates_counts[args.country].keys())
    y = list(country_dates_counts[args.country].values())


target = make_time_plot(place_1="Italy", place_dict=country_dates_counts)
overlay = make_time_plot(place_1="China", place_dict=country_dates_counts, place_2="Italy")
#overlay = make_time_plot(place_1="Italy", place_dict=country_dates_counts, place_2="US")

# create plot of counts per country. Reverse sort x axis (names) by count - ie most to least

cutoff = 20
a_x = list(country_counts.keys())
a_y = list(country_counts.values())
a_order = np.argsort(-np.array(a_y))
a_y_sorted = np.array(a_y)[a_order]
a_x_sorted = np.array(a_x)[a_order]
a_x_grid = [0.5+x for x in range(cutoff)]

a_hover = ColumnDataSource(dict({"country": a_x_sorted[0:cutoff-1], "counts": a_y_sorted[0:cutoff-1]}))


a = figure(tools=TOOLS, title="counts per country", x_axis_label='country', y_axis_label='counts', y_axis_type="log",
           x_range=a_x_sorted[0:cutoff-1], width=750, tooltips=[("country", "@country"), ("Count", "@counts")])

a.vbar(top="counts", x="country", width=0.5, fill_color='red', fill_alpha=0.2, source=a_hover, bottom=0.001)
a.xaxis.major_label_orientation = math.pi/2

# World heatmap - guided by -
# https://towardsdatascience.com/a-complete-guide-to-an-interactive-geographical-map-using-python-f4c5197e23e0

# from: https://www.naturalearthdata.com/downloads/110m-cultural-vectors/

shapefile = '/Users/richarddubois/Code/Home/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp'
# Read shapefile using Geopandas

gdf = gpd.read_file(shapefile)[['ADMIN', 'ADM0_A3', 'geometry']]

# Rename columns.
gdf.columns = ['country', 'country_code', 'geometry']
gdf.head()
print(gdf[gdf['country'] == 'Antarctica'])
# Drop row corresponding to 'Antarctica'
gdf = gdf.drop(gdf.index[159])

# from: https://unstats.un.org/unsd/methodology/m49/

country_code = {}
codes_file = "/Users/richarddubois/Code/Home/Country_codes.csv"
csv_assign = pandas.read_csv(codes_file, header=0, skipinitialspace=True)
csv_frame = csv_assign.set_index('country_code', drop=False)
id_col = csv_frame["country_code"]

h_x_code = []
h_x_country = []
h_y_counts = []
h_start = []
h_st_hover = []

for country in id_col:
    try:
        chk_country = substitute_name(country)
        code = csv_frame.loc[country, "code"]
        h_x_code.append(code)
        h_x_country.append(chk_country)
        idx = np.where(a_x_sorted == chk_country)
        id = idx[0][0]
        h_y_counts.append(a_y_sorted[id])
        c_start = find_start(list(country_dates_counts[chk_country].keys()),
                             list(country_dates_counts[chk_country].values()))
        h_start.append(int(c_start.timestamp())*1000.)
        h_st_hover.append(c_start.strftime('%m/%d/%y'))
    except:
        h_y_counts.append(0.)
        h_start.append(datetime.strptime("1/1/20", '%m/%d/%y').timestamp()*1000.)
        h_st_hover.append("1/1/20")
        continue

pd_source = pandas.DataFrame(dict({"country_code": h_x_country, "code": h_x_code, "counts": h_y_counts,
                                   "start": h_start, "date": h_st_hover}))

# Merge dataframes gdf and pd_source.
merged = gdf.merge(pd_source, left_on = 'country_code', right_on = 'code')

# Read data to json.
merged_json = json.loads(merged.to_json())

# Convert to String like object.
json_data = json.dumps(merged_json)

# Input GeoJSON source that contains features for plotting.
geosource = GeoJSONDataSource(geojson=json_data)

# Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.
color_mapper = LogColorMapper(palette = palette, low = 0, high = np.max(a_y_sorted))

# Create color bar.
color_bar = ColorBar(color_mapper=color_mapper, label_standoff=8, width=500, height=20,
                     border_line_color=None, location=(0, 0), orientation='horizontal')

TOOLS = "reset,save"


# Create figure object.
r = figure(title = 'World Map', plot_height = 600 , plot_width = 950, tools=TOOLS)
r.xgrid.grid_line_color = None
r.ygrid.grid_line_color = None

r_hover_tool = HoverTool(tooltips=[("country", "@country"), ("Count", "@counts")])
r.add_tools(r_hover_tool)
# Add patch renderer to figure.
r.patches('xs', 'ys', source=geosource, fill_color={'field': 'counts', 'transform': color_mapper},
          line_color = 'black', line_width = 0.25, fill_alpha = 1)
# Specify figure layout.
r.add_layout(color_bar, 'below')

# world map of start dates

# Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.
palette2 = palette2[::-1]

t_color_mapper = LinearColorMapper(palette=palette2,
                                low=int(datetime.strptime("1/10/20", '%m/%d/%y').timestamp()) * 1000.,
                                high=int(datetime.today().timestamp()) * 1000.)

# Create color bar.
t_color_bar = ColorBar(color_mapper=t_color_mapper, label_standoff=8, width=500, height=20,
                       border_line_color=None, location=(0, 0), orientation='horizontal',
                       formatter=DatetimeTickFormatter())

# Create figure object.
t = figure(title = 'World Map: Start dates', plot_height = 600 , plot_width = 950, tools=TOOLS)
t.xgrid.grid_line_color = None
t.ygrid.grid_line_color = None

t_hover_tool = HoverTool(tooltips=[("country", "@country"), ("Date", "@date")])
t.add_tools(t_hover_tool)
# Add patch renderer to figure.
t.patches('xs', 'ys', source=geosource, fill_color={'field': 'start', 'transform': t_color_mapper},
          line_color = 'black', line_width = 0.25, fill_alpha = 1)
# Specify figure layout.
t.add_layout(t_color_bar, 'below')

# create plot of counts per state. Reverse sort x axis (names) by count - ie most to least

cutoff = 20
s_x = list(state_counts.keys())
s_y = list(state_counts.values())
s_order = np.argsort(-np.array(s_y))
s_y_sorted = np.array(s_y)[s_order]
s_x_sorted = np.array(s_x)[s_order]
s_x_grid = [0.5+x for x in range(cutoff)]

s_hover = ColumnDataSource(dict({"state": s_x_sorted[0:cutoff-1], "counts": s_y_sorted[0:cutoff-1]}))


sa = figure(tools=TOOLS, title="counts per state", x_axis_label='state', y_axis_label='counts', y_axis_type="log",
           x_range=s_x_sorted[0:cutoff-1], width=750, tooltips=[("state", "@state"), ("Count", "@counts")])

sa.vbar(top="counts", x="state", width=0.5, fill_color='red', fill_alpha=0.2, source=s_hover, bottom=0.001)
sa.xaxis.major_label_orientation = math.pi/2

# US state heatmap

shapefile = '/Users/richarddubois/Code/Home/ne_110m_admin_1_states_provinces/ne_110m_admin_1_states_provinces.shp'
# Read shapefile using Geopandas
gdf = gpd.read_file(shapefile)[['name', 'geometry']]

# Rename columns.
gdf.columns = ['state', 'geometry']
gdf.head()
gdf_frame = gdf.set_index('state', drop=False)
gdf_col = gdf_frame["state"]

s_x_state = []
s_y_counts = []

for state in gdf_col:
    try:
        chk_s = state_counts_sorted[state]
        s_x_state.append(state)
        s_y_counts.append(state_counts_sorted[state])
    except KeyError:
        s_x_state.append(state)
        s_y_counts.append(0.)


st_source = pandas.DataFrame(dict({"state": s_x_state,  "counts": s_y_counts}))

# Merge dataframes gdf and pd_source.
# https://stackoverflow.com/questions/53645882/pandas-merging-101

merged = gdf.merge(st_source, on = 'state')
# Read data to json.
merged_json = json.loads(merged.to_json())
# Convert to String like object.
json_data = json.dumps(merged_json)
# Input GeoJSON source that contains features for plotting.
geosource = GeoJSONDataSource(geojson=json_data)

# Create figure object.

high_state = max(s_y_counts)

s_color_mapper = LogColorMapper(palette=palette, low = 0, high=high_state)
s_color_bar = ColorBar(color_mapper=s_color_mapper, label_standoff=8, width=500, height=20,
                       border_line_color=None, location=(0, 0), orientation='horizontal')

sf = figure(title = 'US State Map', plot_height = 600 , plot_width = 950, tools=TOOLS)
sf.xgrid.grid_line_color = None
sf.ygrid.grid_line_color = None

sf_hover_tool = HoverTool(tooltips=[("state", "@state"), ("Count", "@counts")])
sf.add_tools(sf_hover_tool)
# Add patch renderer to figure.
sf.patches('xs', 'ys', source=geosource, fill_color={'field': 'counts', 'transform': s_color_mapper},
          line_color = 'black', line_width = 0.25, fill_alpha = 1)
# Specify figure layout.

sf.add_layout(s_color_bar, 'below')


# header

header_text = "Run on: " + datetime.now().strftime("%Y-%m-%d") +  "  " + \
            "<a href= 'https://www.arcgis.com/apps/opsdashboard/index.html#/bda7594740fd40299423467b48e9ecf6' \
            >'Data source - Johns Hopkins'</a>"
att_div = Div(text=header_text)

l = layout(att_div, row(target, overlay), row(a, r), t, row(sa, sf))
output_file(args.output)
save(l, title="JHU Analysis")
