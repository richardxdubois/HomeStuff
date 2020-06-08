from datetime import datetime, timedelta, date
import calendar
import argparse
import math
import pandas
import geopandas as gpd
import json
import us
import itertools
from collections import OrderedDict
from unidecode import unidecode
from scipy import stats
from scipy.stats import linregress
from bokeh.plotting import figure, output_file, reset_output, show, save, curdoc
from bokeh.layouts import row, layout, column
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar, LogColorMapper, Range1d
from bokeh.models.widgets import Tabs, Panel, DataTable, TableColumn, DateFormatter, \
    NumberFormatter, Div
from bokeh.models import ColumnDataSource, Span, Label, HoverTool, DatetimeTickFormatter
import numpy as np
from bokeh.palettes import Viridis256 as palette
from bokeh.palettes import Plasma256 as palette2
from bokeh.palettes import brewer

"""
Analysis of Johns Hopkins U covid-19 data:

Read a csv file of CV-19 data:

eg.

python JHU_analysis.py --files /Users/richarddubois/Code/Home/jhu_covid_data.txt -t Deaths --country Italy

Imports notes:

 - us package obtained by pip install


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
    if args.source == "JHU":
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
    elif args.source == "ECDC":
        if name == "Republic of Korea":
            new_name = "South Korea"
        elif name == "United Kingdom of Great Britain and Northern Ireland":
            new_name = "United Kingdom"
        elif name == "Republic of Moldova":
            new_name = "Moldova"
        elif name == "Iran (Islamic Republic of)":
            new_name = "Iran"
        elif name == "Russian Federation":
            new_name = "Russia"
        elif name == "Viet Nam":
            new_name = "Vietnam"
        elif name == "Lao People's Democratic Republic":
            new_name = "Laos"
        elif name == "Democratic Republic of the Congo":
            new_name = "Congo"
        elif "Ivoire" in name:
            new_name = "Ivory Coast"
        elif name == "Syrian Arab Republic":
            new_name = "Syria"
        elif name == "Bolivia (Plurinational State of)":
            new_name = "Bolivia"
        elif name == "Venezuela (Bolivarian Republic of)":
            new_name = "Venezuala"

    return new_name


def find_start(t_array=None, c_array=None, thresh=5.):

    # just to be sure, order the dates
    order = np.argsort(np.array(t_array), kind="stable")
    x_sorted = np.array(t_array)[order]
    y_sorted = np.array(c_array)[order]

    t_start = datetime.strptime("1/1/20", '%m/%d/%y')
    running = 0

    for idt, c in enumerate(y_sorted):
        running += c
        if running > thresh:
            t_start = x_sorted[idt]
            break

    return t_start


def calculate_doubling(in_array, window=7):

    # fit in_array to semi-log, assuming 1 day binning
    # doubling time is dt = ln2/b   for the fit c = a*exp(bt)

    running = []
    running_sum = 0.
    l_array = len(in_array)

    for i in range(l_array):
        running_sum += in_array[i]
        if i > l_array - 8:
            running.append(running_sum)

    y = np.log(np.array(running))
    t = [x for x in range(len(y))]

    slope, intercept, r_value, p_value, std_err = stats.linregress(t, y)

    doubling_time = math.log(2.)/slope

    return doubling_time


def make_time_plot(place_1, place_dict, place_2=None, window_len=11, y_axis="log"):

    x_1 = np.array(list(place_dict[place_1].keys()))
    y_1 = np.array(list(place_dict[place_1].values()))

    # just to be sure, order the dates
    order = np.argsort(np.array(x_1), kind="stable")
    x_sorted = np.array(x_1)[order]
    y_sorted = np.array(y_1)[order]

    y_smooth = smooth(y_sorted, window_len=window_len)

    if place_2 is not None:
        x_2 = np.array(list(place_dict[place_2].keys()))
        y_2 = np.array(list(place_dict[place_2].values()))

        # just to be sure, order the dates
        order = np.argsort(np.array(x_2), kind="stable")
        x_2_sorted = np.array(x_2)[order]
        y_2_sorted = np.array(y_2)[order]
        y_2_smooth = smooth(y_2_sorted)

        len_diff = len(y_sorted) - len(y_2_sorted)
        if len_diff > 0:
            y_2_smooth = np.append(y_2_smooth, np.zeros(len_diff))

        m_hover = ColumnDataSource(dict({"date": x_sorted, "counts": y_sorted, "counts2": y_2_smooth}))
        m_title = "counts vs Date: " + args.type + " : " + place_1 + " overlaid with " + place_2  \
                  + " (black), shifted to " + place_1 + " first date"
    else:
        m_hover = ColumnDataSource(dict({"date": x_sorted, "counts": y_sorted}))
        m_title = "counts vs Date: " + args.type + " : " + place_1

    m = figure(tools=TOOLS, title=m_title, x_axis_type="datetime", x_axis_label='Date',
               y_axis_label='counts', width=750, y_axis_type=y_axis)

    m_hover_tool = HoverTool(tooltips=[("date", "@date{%F}"), ("counts", "@counts")])
    m_hover_tool.formatters = {"date": "datetime"}
    m.add_tools(m_hover_tool)
    m.line(x="date", y="counts", color="navy", source=m_hover)
    m.line(x=x_sorted, y=y_smooth, line_color="red")

    if place_2 is not None:
        m.line(x=x_sorted, y=y_2_smooth, line_color="black")

    return m


def overlay_smooths(place_list, place_dict, window_len=11):

    m_title = "Smoothed counts/day vs Day from first: " + args.type
    m = figure(tools=TOOLS, title=m_title, x_axis_label='Days from first',
               y_axis_label='counts', width=750, y_axis_type="log")

    # colors has a list of colors which can be used in plots
    lines_palette = brewer['Dark2'][8]
    colors = itertools.cycle(lines_palette)

    max_len = 0
    for p in place_dict:
        len_place = len(list(place_dict[p].keys()))
        if len_place > max_len:
            max_len = len_place

    x_days = [float(i) for i in range(max_len)]

    ic = 0
    for place in place_list:

        x = np.array(list(place_dict[place].keys()))
        y = np.array(list(place_dict[place].values()))

        # just to be sure, order the dates
        order = np.argsort(np.array(x))
        x_sorted = np.array(x)[order]
        y_sorted = np.array(y)[order]
        print(place, len(x), x_sorted[0])

        y_smooth = smooth(y_sorted, window_len=window_len)

        len_diff = max_len - len(y_smooth)
        if len_diff > 0:
            y_smooth = np.append(y_smooth, np.zeros(len_diff), axis=None)

        m.line(x=x_days, y=y_smooth, line_color=next(colors), line_width=2, legend_label=place)

        ic += 1

    m.legend.location = "bottom_right"

    return m


def make_start_hist(t_array, title=None):

    # histogram of start dates

    actual_starts = [xt for xt in t_array if xt > datetime.strptime("1/1/20", '%m/%d/%y').timestamp() * 1000.]

    hist, edges = np.histogram(actual_starts, bins=15)

    p_hist = figure(tools=TOOLS, title="Start Dates of outbreak (running sum exceeds 5)",
                    x_axis_label='date', y_axis_label='counts',
                    width=750, x_axis_type="datetime", )

    p_hist.vbar(top=hist, x=edges[:-1], width=0.5e8, fill_color='red', fill_alpha=0.2, bottom=0.001)

    return p_hist


def make_starts_map(s_x_state, s_starts, s_st_hover, maxed_out=False):

    # US state heatmap

    shapefile = '/Users/richarddubois/Code/Home/ne_110m_admin_1_states_provinces/ne_110m_admin_1_states_provinces.shp'
    # Read shapefile using Geopandas
    gdf = gpd.read_file(shapefile)[['name', 'geometry']]

    # Rename columns.
    gdf.columns = ['state', 'geometry']
    gdf.head()
    gdf_frame = gdf.set_index('state', drop=False)
    gdf_col = gdf_frame["state"]

    st_source = pandas.DataFrame(dict({"state": s_x_state,  "date": s_st_hover, "start": s_starts}))

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

    high_state = int(datetime.today().timestamp()) * 1000.
    if maxed_out:
        high_state = max(s_starts)

    t_color_mapper = LinearColorMapper(palette=palette2,
                                       low=int(datetime.strptime("3/01/20", '%m/%d/%y').timestamp()) * 1000.,
                                       high=high_state)
    t_color_bar = ColorBar(color_mapper=t_color_mapper, label_standoff=8, width=500, height=20,
                           border_line_color=None, location=(0, 0), orientation='horizontal',
                           formatter=DatetimeTickFormatter())

    sf = figure(title = 'States Map: Start dates (yellow = not started yet)', plot_height = 600 , plot_width = 950,
                tools=TOOLS)
    sf.xgrid.grid_line_color = None
    sf.ygrid.grid_line_color = None

    sf_hover_tool = HoverTool(tooltips=[("state", "@state"), ("Start", "@date")])
    sf.add_tools(sf_hover_tool)
    # Add patch renderer to figure.
    sf.patches('xs', 'ys', source=geosource, fill_color={'field': 'start', 'transform': t_color_mapper},
               line_color='black', line_width=0.25, fill_alpha=1)
    # Specify figure layout.

    sf.add_layout(t_color_bar, 'below')

    return sf


def parse_counties(csv_files=None):

    county_counts = OrderedDict()
    county_dates_counts = OrderedDict()
    county_dates_counts_inv = OrderedDict()

    n_dates = 0

    bay_area_counties = ["San Mateo", "West Santa Clara", "San Francisco", "Alameda", "San Jose", "Oakland",
                         "South San Francisco", "Half Moon Bay", "Fremont", "Berkeley", "Hayward", "Santa Cruz",
                         "Los Angeles", "Orange", "San Diego"]

    today = datetime.today()

    for cats in csv_files:
        print("Using file ", cats)
        csv_final = pandas.read_csv(cats, header=0, skipinitialspace=True)

        # loop over dates per state
        n_dates = 0

        for index, c_row in csv_final.iterrows():

            state = c_row["state"]
            county = c_row["county"]
            if county == "Santa Clara":
                county = "West Santa Clara"

            if state != "California" or county not in bay_area_counties:
                continue

            if args.type == "Deaths":
                day_count = c_row["deaths"]
            else:
                day_count = c_row["cases"]

            if day_count == 0:
                continue

            c_date = str(c_row["date"])
            dt_date = datetime.strptime(c_date, '%Y-%m-%d')

            s = county_dates_counts_inv.setdefault(county, OrderedDict())
            county_dates_counts_inv[county][dt_date] = day_count

        # post process the dates to reverse the order of getting per day quantities

    for county in county_dates_counts_inv:
        s_x = list(county_dates_counts_inv[county].keys())
        s_y = list(county_dates_counts_inv[county].values())
        s_order = np.argsort(np.array(s_x))
        s_y_sorted = np.array(s_y)[s_order]
        s_x_sorted = np.array(s_x)[s_order]
        county_counts[county] = s_y_sorted[-1]  # last date is the sum

        for it, dt in enumerate(s_x_sorted):
            s = county_dates_counts.setdefault(county, OrderedDict())
            t = county_dates_counts[county].setdefault(dt, s_y_sorted[it])
            if it > 0:
                diff = county_dates_counts[county][dt] - s_y_sorted[it - 1]
                if diff > 0:
                    county_dates_counts[county][dt] -= s_y_sorted[it - 1]

    return n_dates, county_counts, county_dates_counts


def parse_states(csv_files=None):

    state_counts = OrderedDict()
    state_dates_counts = OrderedDict()
    state_dates_counts_inv = OrderedDict()

    n_dates = 0

    today = datetime.today()

    for cats in csv_files:
        print("Using file ", cats)
        csv_final = pandas.read_csv(cats, header=0, skipinitialspace=True)
#        csv_drop_cols = csv_assign.dropna(axis="columns", how="all")
#        csv_final = csv_drop_cols.fillna(0.)

        # loop over dates per state  - unpack for post processing since data is reverse ordered, summed
        n_dates = 0

        for index, c_row in csv_final.iterrows():

            state_abbrev = unidecode(c_row["state"])
            if state_abbrev == "state":  # ignore header line
                continue

            state = str(us.states.lookup(state_abbrev))
            if args.type == "Deaths":
                day_count = c_row["death"]
            else:
                day_count = c_row["positive"]

            if math.isnan(day_count) or day_count == 0:
                continue

            ecdc_date = str(c_row["date"])
            dt_date = datetime.strptime(ecdc_date, '%Y%m%d')

            s = state_dates_counts_inv.setdefault(state, OrderedDict())
            state_dates_counts_inv[state][dt_date] = day_count

# post process the dates to reverse the order of getting per day quantities

    for state in state_dates_counts_inv:
        s_x = list(state_dates_counts_inv[state].keys())
        s_y = list(state_dates_counts_inv[state].values())
        s_order = np.argsort(np.array(s_x))
        s_y_sorted = np.array(s_y)[s_order]
        s_x_sorted = np.array(s_x)[s_order]
        state_counts[state] = s_y_sorted[-1]   # last date is the sum

        for it, dt in enumerate(s_x_sorted):
            s = state_dates_counts.setdefault(state, OrderedDict())
            t = state_dates_counts[state].setdefault(dt, s_y_sorted[it])
            if it > 0:
                diff = state_dates_counts[state][dt] - s_y_sorted[it - 1]
#                if diff > 0:
#                state_dates_counts[state][dt] -= s_y_sorted[it-1]
                state_dates_counts[state][dt] = diff

    state_counts_sorted = OrderedDict({k: v for k, v in sorted(state_counts.items(), key=lambda item: item[1],
                                                                 reverse=True)})

    print(state_counts)

    return n_dates, state_counts_sorted, state_dates_counts


def parse_ECDC(csv_files=None):

    country_counts = OrderedDict()
    dates_counts = OrderedDict()
    country_dates_counts = {}

    state_counts = OrderedDict()
    state_dates_counts = OrderedDict()

    n_dates = 0

    today = datetime.today()

    for cats in csv_files:
        print("Using file ", cats)
        csv_final = pandas.read_excel(cats, header=0, skipinitialspace=True)
#        csv_drop_cols = csv_assign.dropna(axis="columns", how="all")
#       csv_final = csv_drop_cols.fillna(0.)

        # loop over dates per state
        n_dates = 0

        for index, c_row in csv_final.iterrows():

            country_region = unidecode(c_row["countriesAndTerritories"])
            if country_region == "countriesAndTerritories":  # ignore header line
                continue
            country_region = country_region.replace("_", " ").strip()
            if args.type == "Deaths":
                day_count = c_row["deaths"]
            else:
                day_count = c_row["cases"]

            if day_count == 0:
                continue

            ecdc_date = c_row["dateRep"]
            dt_date = ecdc_date.to_pydatetime()

            try:
                country_counts[country_region] += day_count
            except KeyError:
                country_counts[country_region] = day_count

            c = country_dates_counts.setdefault(country_region, OrderedDict())
            d = country_dates_counts[country_region].setdefault(dt_date, 0)
            country_dates_counts[country_region][dt_date] += day_count

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

    return n_dates, country_counts_sorted, country_dates_counts,  \
        dates_counts, country_counts, state_counts


def parse_JHU(csv_files=None):

    country_counts = OrderedDict()
    dates_counts = OrderedDict()
    country_dates_counts = OrderedDict()

    state_counts = OrderedDict()
    state_dates_counts = OrderedDict()

    n_dates = 0

    today = datetime.today()

    # parse the JHU csv files

    for cats in csv_files:
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
                    day_count = float(c_row[id]) - float(c_row[id - 1])
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
                    try:
                        if "," not in state:  # in case state data has vanished
                            s = state_dates_counts.setdefault(state, OrderedDict())
                            t = state_dates_counts[state].setdefault(dt_date, 0)
                            state_dates_counts[state][dt_date] += day_count

                            sc = state_counts.setdefault(state, 0)
                            state_counts[state] += day_count
                    except TypeError:
                        pass

                try:
                    dates_counts[dt_date] += day_count
                except KeyError:
                    dates_counts[dt_date] = day_count

    country_counts_sorted = OrderedDict({k: v for k, v in sorted(country_counts.items(), key=lambda item: item[1],
                                                                 reverse=True)})
    print(country_counts)
    print(dates_counts)

    return n_dates, country_counts_sorted, country_dates_counts,  \
        dates_counts, country_counts, state_counts


def parse_iso_order_dates(list_file):

    csv_assign = pandas.read_csv(list_file, header=0, skipinitialspace=True)

    start_by_place = {}
    for index, c_row in csv_assign.iterrows():
        place = c_row["placeLabel"]
        dt = c_row["isoStartDate"]
        start_by_place[place] = dt

    return start_by_place


# Command line arguments
parser = argparse.ArgumentParser(description='Analyze the SRM-derived WBS information')

parser.add_argument('--source', default='ECDC', help="type of source data [ECDC or JHU] (default=%(default)s)")
parser.add_argument('--files', default='jhu_files.txt', help="lists of files (default=%(default)s)")
parser.add_argument('--isoDate', default='/Users/richarddubois/Code/Home/US-isolation-dates.csv',
                    help="file with isolation order dates (default=%(default)s)")
parser.add_argument('-o', '--output', default='jhu_analysis.html',
                    help="output bokeh html file (default=%(default)s)")
parser.add_argument('-c', '--country', default="Italy",
                    help="select country (default=%(default)s)")
parser.add_argument('--comp_country', default="China",
                    help="select country to overlay (default=%(default)s)")
parser.add_argument('-s', '--state', default="New York",
                    help="select state (default=%(default)s)")
parser.add_argument('--county', default="Los Angeles",
                    help="select CA county (default=%(default)s)")
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

TOOLS = "reset,save"

#  County data

n_dates, county_counts, county_dates_counts = parse_counties(jhu_files["Counties"])

#  county data

la_target = make_time_plot(place_1=args.county, place_dict=county_dates_counts, y_axis="linear")

# create plot of counts per county. Reverse sort x axis (names) by count - ie most to least

c_x = list(county_counts.keys())
c_y = list(county_counts.values())
c_order = np.argsort(-np.array(c_y))
c_y_sorted = np.array(c_y)[c_order]
c_x_sorted = np.array(c_x)[c_order]
c_x_grid = [0.5+x for x in range(len(c_x))]

c_hover = ColumnDataSource(dict({"county": c_x_sorted, "counts": c_y_sorted}))

ca = figure(tools=TOOLS, title="counts per county", x_axis_label='county', y_axis_label='counts',
           x_range=c_x_sorted, width=750, tooltips=[("county", "@county"), ("Count", "@counts")])

ca.vbar(top="counts", x="county", width=0.5, fill_color='red', fill_alpha=0.2, source=c_hover, bottom=0.001)
ca.xaxis.major_label_orientation = math.pi/2

# US county heatmap

#shapefile = '/Users/richarddubois/Code/Home/CA_Counties/CA_Counties_TIGER2016.shp'
shapefile = '/Users/richarddubois/Code/Home/cb_2018_06_cousub_500k/cb_2018_06_cousub_500k.shp'
#shapefile = '/Users/richarddubois/Code/Home/cb_2018_us_county_5m/cb_2018_us_county_5m.shp'

# Read shapefile using Geopandas
#gdf = gpd.read_file(shapefile)
gdf = gpd.read_file(shapefile)[['NAME', 'STATEFP', 'geometry']]

# Rename columns.
gdf.columns = ['county', 'state_fips', 'geometry']
gdf.head()
gdf_frame = gdf.set_index('county', drop=False)
gdf_col = gdf_frame["county"]

c_x_county = []
c_y_counts = []
c_starts = []
c_st_hover = []

for county in gdf_col:
    try:
        state_fips = str(gdf_frame.loc[county, "state_fips"])
        if state_fips != "06":  # California
            continue
        chk = county_counts[county]
        c_x_county.append(county)
        c_y_counts.append(county_counts[county])
        c_start = find_start(list(county_dates_counts[county].keys()),
                             list(county_dates_counts[county].values()))
        c_starts.append(int(c_start.timestamp())*1000.)
        c_st_hover.append(c_start.strftime('%m/%d/%y'))
    except KeyError:
        c_x_county.append(county)
        c_y_counts.append(0.)
        c_starts.append(datetime.strptime("1/1/20", '%m/%d/%y').timestamp()*1000.)
        c_st_hover.append("1/1/20")

st_source = pandas.DataFrame(dict({"county": c_x_county,  "counts": c_y_counts}))

# Merge dataframes gdf and pd_source.
# https://stackoverflow.com/questions/53645882/pandas-merging-101

merged = gdf.merge(st_source, on = 'county')
# Read data to json.
merged_json = json.loads(merged.to_json())
# Convert to String like object.
json_data = json.dumps(merged_json)
# Input GeoJSON source that contains features for plotting.
geosource = GeoJSONDataSource(geojson=json_data)

# Create figure object.

high_county = max(c_y_counts)

c_color_mapper = LinearColorMapper(palette=palette, low = 0, high=high_county)
c_color_bar = ColorBar(color_mapper=c_color_mapper, label_standoff=8, width=500, height=20,
                       border_line_color=None, location=(0, 0), orientation='horizontal')

cf = figure(title = 'Bay Area County Map', plot_height = 600 , plot_width = 600, tools=TOOLS, x_range=(-122.75, -121.75),
            y_range=(37, 38.5))
cf.xgrid.grid_line_color = None
cf.ygrid.grid_line_color = None

cf_hover_tool = HoverTool(tooltips=[("county", "@county"), ("Count", "@counts")])
cf.add_tools(cf_hover_tool)
# Add patch renderer to figure.
cf.patches('xs', 'ys', source=geosource, fill_color={'field': 'counts', 'transform': c_color_mapper},
           line_color='black', line_width=0.25, fill_alpha=1)
# Specify figure layout.

cf.add_layout(c_color_bar, 'below')

if args.source == "JHU":
    n_dates, country_counts_sorted, country_dates_counts, dates_counts, \
        country_counts, state_counts = parse_JHU(jhu_files[args.type])
else:
    n_dates, country_counts_sorted, country_dates_counts, dates_counts, \
        country_counts, state_counts = parse_ECDC(jhu_files[args.type])

x = []
y = []

if args.country is None:
    x = list(dates_counts.keys())
    y = list(dates_counts.values())
else:
    x = list(country_dates_counts[args.country].keys())
    y = list(country_dates_counts[args.country].values())


target = make_time_plot(place_1=args.country, place_dict=country_dates_counts)
overlay = make_time_plot(place_1=args.comp_country, place_dict=country_dates_counts, place_2=args.country)

tops = list(country_counts_sorted.keys())[0:8]
over_smooths = overlay_smooths(place_list=tops, place_dict=country_dates_counts)

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
           x_range=a_x_sorted[0:cutoff-1], width=750, y_range=Range1d(1., 1.2*a_y_sorted[0]),
           tooltips=[("country", "@country"), ("Count", "@counts")])

a.vbar(top="counts", x="country", width=0.5, fill_color='red', fill_alpha=0.2, source=a_hover, bottom=0.001)
a.xaxis.major_label_orientation = math.pi/2

# create plot of last count per country. Reverse sort x axis (names) by count - ie most to least

ac_y = []
doubling_country = []
for c in a_x_sorted[0:cutoff-1]:
    c_dt_dict = country_dates_counts[c]
    c_x = list(country_dates_counts[c].keys())
    c_y = list(country_dates_counts[c].values())
    c_a_order = np.argsort(np.array(c_x))
    c_y_sorted = np.array(c_y)[c_a_order]
    ac_y.append(c_y_sorted[-1])
    doubling_time = calculate_doubling(c_y_sorted)
    doubling_country.append(doubling_time)


c_hover = ColumnDataSource(dict({"country": a_x_sorted[0:cutoff-1], "counts": ac_y}))


cta = figure(tools=TOOLS, title="most recent count per country", x_axis_label='country', y_axis_label='counts',
            y_axis_type="log",
           x_range=a_x_sorted[0:cutoff-1], width=750, y_range=Range1d(1., 1.2*max(ac_y)),
           tooltips=[("country", "@country"), ("Count", "@counts")])

cta.vbar(top="counts", x="country", width=0.5, fill_color='red', fill_alpha=0.2, source=c_hover, bottom=0.001)
cta.xaxis.major_label_orientation = math.pi/2

# plot doubling time

doub_hover = ColumnDataSource(dict({"country": a_x_sorted[0:cutoff-1], "time": doubling_country}))


doub = figure(tools=TOOLS, title="doubling time per country", x_axis_label='country', y_axis_label='time (days)',
           x_range=a_x_sorted[0:cutoff-1], width=750, y_axis_type="log",
           tooltips=[("country", "@country"), ("Days", "@time")])

doub.vbar(top="time", x="country", width=1., fill_color='red', fill_alpha=0.2, source=doub_hover, bottom=0.001)
doub.xaxis.major_label_orientation = math.pi/2



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

    chk_country = substitute_name(country)
    code = csv_frame.loc[country, "code"]
    h_x_code.append(code)
    h_x_country.append(chk_country)
    try:
        idx = np.where(a_x_sorted == chk_country)
        id = idx[0][0]
        h_y_counts.append(a_y_sorted[id])
    except:
        h_y_counts.append(0.)
        h_start.append(datetime.strptime("1/1/20", '%m/%d/%y').timestamp() * 1000.)
        h_st_hover.append("1/1/20")
        continue

    try:
        c_start = find_start(list(country_dates_counts[chk_country].keys()),
                             list(country_dates_counts[chk_country].values()))
        h_start.append(int(c_start.timestamp())*1000.)
        h_st_hover.append(c_start.strftime('%m/%d/%y'))
    except:
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

# histogram of start dates

actual_starts = [xt for xt in h_start if xt > datetime.strptime("1/1/20", '%m/%d/%y').timestamp()*1000.]

hist, edges = np.histogram(actual_starts, bins=15)

p_hist = figure(tools=TOOLS, title="Start Dates of outbreak (running sum exceeds 5)",
                x_axis_label='date', y_axis_label='counts',
                width=750, x_axis_type="datetime",)

p_hist.vbar(top=hist, x=edges[:-1], width=0.25e9, fill_color='red', fill_alpha=0.2, bottom=0.001)


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
t = figure(title='World Map: Start dates (yellow = not started yet)', plot_height=600, plot_width=950, tools=TOOLS)
t.xgrid.grid_line_color = None
t.ygrid.grid_line_color = None

t_hover_tool = HoverTool(tooltips=[("country", "@country"), ("Date", "@date")])
t.add_tools(t_hover_tool)
# Add patch renderer to figure.
t.patches('xs', 'ys', source=geosource, fill_color={'field': 'start', 'transform': t_color_mapper},
          line_color='black', line_width=0.25, fill_alpha=1)
# Specify figure layout.
t.add_layout(t_color_bar, 'below')

n_dates, state_counts_sorted, state_dates_counts = parse_states(jhu_files["States"])

if len(state_counts_sorted) != 0:
    # create plot of counts per state. Reverse sort x axis (names) by count - ie most to least

    cutoff = 20
    s_x = list(state_counts_sorted.keys())
    s_y = list(state_counts_sorted.values())
    s_order = np.argsort(-np.array(s_y))
    s_y_sorted = np.array(s_y)[s_order]
    s_x_sorted = np.array(s_x)[s_order]
    s_x_grid = [0.5+x for x in range(cutoff)]

    s_hover = ColumnDataSource(dict({"state": s_x_sorted[0:cutoff-1], "counts": s_y_sorted[0:cutoff-1]}))

    sa = figure(tools=TOOLS, title="counts per state", x_axis_label='state', y_axis_label='counts', y_axis_type="log",
               x_range=s_x_sorted[0:cutoff-1], width=750, y_range=Range1d(1., 1.2*s_y_sorted[0]),
                tooltips=[("state", "@state"), ("Count", "@counts")])

    sa.vbar(top="counts", x="state", width=0.5, fill_color='red', fill_alpha=0.2, source=s_hover, bottom=0.001)
    sa.xaxis.major_label_orientation = math.pi/2

    doubling_state = []
    sc_y = []
    iso_dates_all = parse_iso_order_dates(list_file=args.isoDate)
    iso_dates = []
    iso_dates_str = []

    for s in s_x_sorted[0:cutoff-1]:
        s_dt_dict = state_dates_counts[s]
        sd_x = list(state_dates_counts[s].keys())
        sd_y = list(state_dates_counts[s].values())
        sd_a_order = np.argsort(np.array(sd_x))
        sd_y_sorted = np.array(sd_y)[sd_a_order]
        sc_y.append(sd_y_sorted[-1])
        doubling_time_state = calculate_doubling(sd_y_sorted)
        doubling_state.append(doubling_time_state)
        try:   # not in list
            id = iso_dates_all[s]
        except KeyError:
            id = "2020-03-15T00:00:00Z"
        id_dt = datetime.strptime(id, "%Y-%m-%dT00:00:00Z").timestamp()*1000.
        iso_dates.append(id_dt)
        iso_dates_str.append(id)


# plot most recent

    sr_hover = ColumnDataSource(dict({"state": s_x_sorted[0:cutoff - 1], "counts": sc_y}))

    sr = figure(tools=TOOLS, title="most recent count per state", x_axis_label='state', y_axis_label='counts',
                 y_axis_type="log",
                 x_range=s_x_sorted[0:cutoff - 1], width=750,
                 tooltips=[("state", "@state"), ("Count", "@counts")])

    sr.vbar(top="counts", x="state", width=0.5, fill_color='red', fill_alpha=0.2, source=sr_hover, bottom=0.001)
    sr.xaxis.major_label_orientation = math.pi / 2

# plot doubling time

    doub_state_hover = ColumnDataSource(dict({"state": s_x_sorted[0:cutoff - 1], "time": doubling_state}))

    doub_state = figure(tools=TOOLS, title="doubling time per state", x_axis_label='state', y_axis_label='time (days)',
                  x_range=s_x_sorted[0:cutoff - 1], width=750,
                  tooltips=[("state", "@state"), ("Days", "@time")])

    doub_state.vbar(top="time", x="state", width=1., fill_color='red', fill_alpha=0.2, source=doub_state_hover)
    doub_state.xaxis.major_label_orientation = math.pi / 2

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
    s_starts = []
    s_st_hover = []

    for state in gdf_col:
        try:
            chk_s = state_counts_sorted[state]
            s_x_state.append(state)
            s_y_counts.append(state_counts_sorted[state])
            c_start = find_start(list(state_dates_counts[state].keys()),
                                 list(state_dates_counts[state].values()))
            s_starts.append(int(c_start.timestamp())*1000.)
            s_st_hover.append(c_start.strftime('%m/%d/%y'))
        except KeyError:
            s_x_state.append(state)
            s_y_counts.append(0.)
            s_starts.append(datetime.strptime("1/1/20", '%m/%d/%y').timestamp()*1000.)
            s_st_hover.append("1/1/20")

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
               line_color='black', line_width=0.25, fill_alpha=1)
    # Specify figure layout.

    sf.add_layout(s_color_bar, 'below')

    ps = make_start_hist(s_starts)
    sf_st = make_starts_map(s_x_state, s_starts, s_st_hover, maxed_out=True)

    st_tops = list(state_counts_sorted.keys())[0:7]
    st_over_smooths = overlay_smooths(place_list=st_tops, place_dict=state_dates_counts)
    st_target = make_time_plot(place_1=args.state, place_dict=state_dates_counts, window_len=7)

# doubling rate vs isolation order date

    iso_state_hover = ColumnDataSource(dict({"state": s_x_sorted[0:cutoff - 1], "double": doubling_state,
                                             "iso_date": iso_dates, "iso_str": iso_dates_str}))

    iso_state = figure(tools=TOOLS, title="doubling time vs isol date", x_axis_label='date',
                       y_axis_label='doubling rate', width=750,  x_axis_type="datetime",
                  tooltips=[("state", "@state"), ("Date", "@iso_str"), ("Rate", "@double")])

    iso_state.circle(x="iso_date", y="double", fill_color='blue', fill_alpha=0.2, source=iso_state_hover, size=10)
    iso_state.xaxis.major_label_orientation = math.pi / 2

# header

source_url = "https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution" \
             "-covid-19-cases-worldwide"

if args.source == "JHU":
    source_url = "https://www.arcgis.com/apps/opsdashboard/index.html#/bda7594740fd40299423467b48e9ecf6"

states_url = "https://covidtracking.com/"
counties_url = "https://github.com/nytimes/covid-19-data"

header_text = "Run on: " + datetime.now().strftime("%Y-%m-%d") + "  " + \
            "<a href= " + source_url + " > Global Data source - " + args.source + "</a></p>" + \
            "<a href= " + states_url + " > States Data source </a></p>" + \
            "<a href= " + counties_url + " > Counties Data source </a>"

att_div = Div(text=header_text)

global_layout = layout(row(target, overlay), row(over_smooths, cta), doub, row(a, r), row(p_hist, t))


state_layout = layout(row(sa, sf), row(ps, sf_st), row(st_over_smooths, st_target), row(doub_state, sr),
               iso_state)

county_layout = row(ca, la_target)

tab1 = Panel(child=global_layout, title="World")
tab2 = Panel(child=state_layout, title="US States")
tab3 = Panel(child=county_layout, title="CA Counties")
tabs = Tabs(tabs=[tab1, tab2, tab3])

full_layout = layout(att_div, tabs)

output_file(args.output)
save(full_layout, title=args.source + " Analysis: " + args.type)
