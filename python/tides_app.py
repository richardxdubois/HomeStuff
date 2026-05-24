# tide_app.py - Interactive Bokeh Server App for Tide Analysis

from bokeh.plotting import figure, curdoc
from bokeh.models import (CheckboxGroup, DateRangeSlider, Select, Div,
                         HoverTool, BoxAnnotation, Button, DateRangePicker)
from bokeh.layouts import column, row
from tornado.ioloop import IOLoop

import pandas as pd
import os
import signal

from datetime import datetime, timedelta, time

from Tide_analyzer import TideAnalyzer, TideConfig

# Global state
current_analyzer = None
current_dates = []
date_categories = {'good': [], 'marginal': [], 'other': []}
all_plots = {}

# Configuration
CONFIG_PATH = '/Volumes/Data/Home/tides_predictor/tides_kayak.yaml'
config = TideConfig(CONFIG_PATH)
window_config = config.get_window()
locations = config.get_locations()

print(f"DEBUG: locations = {locations}")
print(f"DEBUG: window_config = {window_config}")
date_list = config.get_date_list()
print(f"DEBUG: date_list from config = {date_list}")
print(f"DEBUG: date_list length = {len(date_list) if date_list else 0}")

# Create UI controls
location_select = Select(
    title="Location:",
    value=locations[0]['key'],
    options=[(loc['key'], loc['name']) for loc in locations],
    width=200
)

window_start_select = Select(
    title="Window Start:",
    value=str(window_config['start_hour']),
    options=[str(i) for i in range(0, 24)],
    width=100
)

window_end_select = Select(
    title="Window End:",
    value=str(window_config['end_hour']),
    options=[str(i) for i in range(0, 24)],
    width=100
)

# Date range selector
date_list = config.get_date_list()
if date_list:
    min_date = pd.to_datetime(min(date_list))
    max_date = pd.to_datetime(max(date_list))
else:
    min_date = datetime.now()
    max_date = datetime.now() + timedelta(days=30)

date_range_slider = DateRangeSlider(
    title="Date Range:",
    start=min_date,
    end=max_date,
    value=(min_date, max_date),
    step=1,
    width=400
)

date_range_picker = DateRangePicker(
    title="Select Date Range:",
    value=(min_date.date(), max_date.date()),
    min_date=min_date.date(),
    max_date=max_date.date(),
    width=300
)
weekends_only_checkbox = CheckboxGroup(
    labels=["Weekends Only"],
    active=[0] if config.config.get('dates', {}).get('weekends_only', False) else [],
    width=150
)

# Category filter checkboxes
category_filter = CheckboxGroup(
    labels=["✓ Good Days", "~ Marginal Days", "✗ Other Days"],
    active=[0, 1, 2],  # All checked by default
    width=400
)

# Load data button
load_button = Button(label="Load Tide Data", button_type="primary", width=150)

# Exit button
exit_button = Button(label="Exit App", button_type="danger", width=100)


def exit_callback():
    """Shut down the Bokeh server."""
    status_div.text = "<p style='color: blue; font-size: 16px;'>✓ Exiting application...</p>"

    # Give the browser time to display the message before shutdown
    import threading
    def delayed_exit():
        import sys
        print("Exit button shutdown")
        # Kill entire process group
        os.killpg(os.getpgid(0), signal.SIGTERM)

    timer = threading.Timer(0.5, delayed_exit)
    timer.start()

# ------------------------------------------------------------------
    # Server control
    # ------------------------------------------------------------------
def _stop_server(self):
    """Gracefully shut down the Bokeh server."""
    #curdoc().add_next_tick_callback(
    #    lambda: self._async_update_log("Server is shutting down...")
    #
    #curdoc().add_next_tick_callback(
    #    lambda: self._async_change_button(self.exit_button, "light")
    #)
    status_div.text = "<p style='color: blue; font-size: 16px;'>✓ Exiting application...</p>"

    curdoc().add_next_tick_callback(_exit_server)

async def _async_update_log(self, message: str):
    self.log_div.text = message

async def _exit_server():
    print("Server is shutting down...")

    IOLoop.current().stop()

async def _async_change_button(self, button, color: str):
    button.button_type = color

#exit_button.on_click(exit_callback)
exit_button.on_click(_stop_server)

def slider_change_callback(attr, old, new):
    """Auto-reload when slider changes (after user stops dragging)."""
    load_data_callback()

# Auto-reload callbacks for all controls
location_select.on_change('value', lambda attr, old, new: load_data_callback())

weekends_only_checkbox.on_change('active', lambda attr, old, new: load_data_callback())

# Use value_throttled instead of value - only fires when dragging stops
date_range_slider.on_change('value_throttled', slider_change_callback)


def window_change_callback(attr, old, new):
    """Re-categorize and update display when window times change."""
    global date_categories, all_plots

    if not current_analyzer or current_analyzer.df_hilo is None:
        return  # No data loaded yet

    window_start = int(window_start_select.value)
    window_end = int(window_end_select.value)

    # Re-categorize with new window
    date_categories = categorize_dates(current_analyzer, current_dates, window_start, window_end)

    # Update summary table
    summary_html = create_summary_table(current_dates, date_categories, window_start, window_end)
    summary_div.text = summary_html

    # Update plots with new window (they need to redraw the green box)
    # This requires recreating plots since window annotation is built-in
    for date_str in current_dates:
        if date_str in date_categories['good']:
            category = 'good'
        elif date_str in date_categories['marginal']:
            category = 'marginal'
        else:
            category = 'other'

        plot = create_day_plot(date_str, window_start, window_end, category)
        if plot:
            all_plots[date_str] = plot

    # Update layout
    plots_column.children = list(all_plots.values())
    update_plots()
    update_day_selector()

    status_div.text = f"<p style='color: green;'>✓ Window updated. Found {len(date_categories['good'])} good, {len(date_categories['marginal'])} marginal, {len(date_categories['other'])} other.</p>"


# Wire up the new callback
window_start_select.on_change('value', window_change_callback)
window_end_select.on_change('value', window_change_callback)


def sync_picker_from_slider(attr, old, new):
    """Update date picker when slider moves."""
    start_ts, end_ts = new

    start_date = pd.Timestamp(start_ts, unit='ms').date()
    end_date = pd.Timestamp(end_ts, unit='ms').date()

    # Update picker without triggering callback
    date_range_picker.remove_on_change('value', date_range_picker_cb)
    date_range_picker.value = (start_date, end_date)
    date_range_picker.on_change('value', date_range_picker_cb)


# Add to existing slider callback
date_range_slider.on_change('value', sync_picker_from_slider)

# Status display
status_div = Div(text="<p>Select parameters and click 'Load Tide Data'</p>", width=800)

# Summary table div
summary_div = Div(text="", width=900)

# Container for plots
plots_column = column()

# Single day selector
day_select = Select(
    title="Jump to specific day:",
    value="-- Show All Days --",
    options=[("-- Show All Days --", "-- Show All Days --")],
    width=300
)

def get_selected_location():
    """Get the currently selected location info."""
    loc_key = location_select.value
    for loc in locations:
        if loc['key'] == loc_key:
            return loc
    return locations[0]


def categorize_dates(analyzer, dates, window_start, window_end):
    """Categorize dates as good, marginal, or other."""
    good = []
    marginal = []
    other = []

    for date_str in dates:
        target_date = pd.to_datetime(date_str).date()
        day_hilo = analyzer.df_hilo[analyzer.df_hilo.index.date == target_date]

        if day_hilo.empty:
            continue

        highs = day_hilo[day_hilo['type'] == 'H']

        # Check if good
        good_highs = [t for t in highs.index if window_start <= t.hour <= window_end]
        if good_highs:
            good.append(date_str)
            continue

        # Check if marginal
        marginal_highs = [
            t for t in highs.index
            if (window_start - 1 <= t.hour < window_start) or
               (window_end < t.hour <= window_end + 1)
        ]
        if marginal_highs:
            marginal.append(date_str)
        else:
            other.append(date_str)

    return {'good': good, 'marginal': marginal, 'other': other}


def create_summary_table(dates, categories, window_start, window_end):
    """Create HTML summary table."""
    global current_analyzer

    if not current_analyzer or current_analyzer.df_hilo is None:
        return ""

    rows_html = []

    for date_str in dates:
        target_date = pd.to_datetime(date_str).date()
        day_hilo = current_analyzer.df_hilo[current_analyzer.df_hilo.index.date == target_date]

        if day_hilo.empty:
            continue

        highs = day_hilo[day_hilo['type'] == 'H']
        lows = day_hilo[day_hilo['type'] == 'L']

        # Determine category
        if date_str in categories['good']:
            status = "✓"
            row_class = "good-day"
        elif date_str in categories['marginal']:
            status = "~"
            row_class = "marginal-day"
        else:
            status = "✗"
            row_class = "other-day"

        day_name = pd.to_datetime(date_str).strftime('%A')

        # Format high tide
        if not highs.empty:
            high_strs = []
            for idx, high in highs.iterrows():
                # Mark if it's in the target window
                in_window = window_start <= idx.hour <= window_end
                marker = " ⭐" if in_window else ""
                high_strs.append(f"{idx.strftime('%I:%M %p')} ({high['v']:.1f} ft){marker}")
            high_str = "<br>".join(high_strs)
        else:
            high_str = "N/A"

        # Format low tides
        if not lows.empty:
            low_strs = [f"{idx.strftime('%I:%M %p')} ({low['v']:.1f} ft)"
                        for idx, low in lows.iterrows()]
            low_str = "<br>".join(low_strs)
        else:
            low_str = "N/A"

        rows_html.append(
            f'<tr class="{row_class}">'
            f'<td style="text-align: center;">{status}</td>'
            f'<td>{date_str}</td>'
            f'<td>{day_name}</td>'
            f'<td>{high_str}</td>'
            f'<td>{low_str}</td>'
            f'</tr>'
        )

    loc = get_selected_location()

    html = f"""
    <div style="font-family: Arial, sans-serif; margin: 20px 0;">
        <h2>Summary - {loc['name']}</h2>
        <p><strong>Target window:</strong> {window_start}:00 - {window_end}:00</p>
        <p><strong>Good:</strong> {len(categories['good'])} | 
           <strong>Marginal:</strong> {len(categories['marginal'])} | 
           <strong>Other:</strong> {len(categories['other'])}</p>

        <table style="border-collapse: collapse; width: 100%;">
            <thead>
                <tr style="background-color: #f0f0f0;">
                    <th style="padding: 8px; border: 1px solid #ddd; width: 40px;">Status</th>
                    <th style="padding: 8px; border: 1px solid #ddd;">Date</th>
                    <th style="padding: 8px; border: 1px solid #ddd;">Day</th>
                    <th style="padding: 8px; border: 1px solid #ddd;">High Tide</th>
                    <th style="padding: 8px; border: 1px solid #ddd;">Low Tides</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows_html)}
            </tbody>
        </table>
    </div>

    <style>
        .good-day {{
            background-color: #d4edda;
        }}
        .marginal-day {{
            background-color: #fff3cd;
        }}
        .other-day {{
            background-color: #f8d7da;
        }}
        table td {{
            padding: 6px 12px;
            border: 1px solid #ddd;
            vertical-align: top;
        }}
    </style>
    """

    return html


def create_day_plot(date_str, window_start, window_end, category):
    """Create a plot for a single day."""
    global current_analyzer

    if not current_analyzer:
        return None

    target_date = pd.to_datetime(date_str).date()

    # Filter to single day
    day_data = current_analyzer.df_hourly[current_analyzer.df_hourly.index.date == target_date].copy()
    day_hilo = current_analyzer.df_hilo[current_analyzer.df_hilo.index.date == target_date].copy()

    if day_data.empty:
        return None

    # Color code by category
    border_color = 'green' if category == 'good' else 'orange' if category == 'marginal' else 'red'

    p = figure(
        x_axis_type='datetime',
        title=f'{get_selected_location()["name"]} - {date_str} ({pd.to_datetime(date_str).strftime("%A")})',
        width=900,
        height=350,
        toolbar_location='above',
        outline_line_color=border_color,
        outline_line_width=3
    )

    # Shaded target window
    window_start_dt = datetime.combine(target_date, time(window_start, 0))
    window_end_dt = datetime.combine(target_date, time(window_end, 0))

    box = BoxAnnotation(
        left=window_start_dt,
        right=window_end_dt,
        fill_alpha=0.1,
        fill_color='green',
        line_color='green',
        line_dash='dashed',
        line_alpha=0.5
    )
    p.add_layout(box)

    # Tide curve
    p.line(day_data.index, day_data['v'], line_width=2, color='navy', legend_label='Tide Level')

    # High/low markers
    highs = day_hilo[day_hilo['type'] == 'H']
    lows = day_hilo[day_hilo['type'] == 'L']

    p.scatter(highs.index, highs['v'], size=12, color='green', marker='triangle', legend_label='High Tide')
    p.scatter(lows.index, lows['v'], size=12, color='red', marker='inverted_triangle', legend_label='Low Tide')

    # Hover tool
    hover = HoverTool(
        tooltips=[('Time', '@x{%H:%M}'), ('Height', '@y{0.1f} ft')],
        formatters={'@x': 'datetime'},
        mode='vline'
    )
    p.add_tools(hover)

    # Styling
    p.yaxis.axis_label = 'Height (feet, MLLW)'
    p.xaxis.axis_label = 'Time'
    p.legend.location = 'top_right'
    p.legend.click_policy = 'hide'
    p.xaxis.formatter.days = '%H:%M'

    return p


def update_plots():
    """Update the displayed plots based on category filter."""
    global all_plots, date_categories, plots_column

    active_categories = category_filter.active
    show_good = 0 in active_categories
    show_marginal = 1 in active_categories
    show_other = 2 in active_categories

    # Instead of rebuilding children list, just set visibility
    for date_str, plot in all_plots.items():
        if date_str in date_categories['good']:
            plot.visible = show_good
        elif date_str in date_categories['marginal']:
            plot.visible = show_marginal
        elif date_str in date_categories['other']:
            plot.visible = show_other


def update_day_selector():
    """Update the day selector dropdown with current dates and their status."""
    global date_categories

    options = [("-- Show All Days --", "-- Show All Days --")]

    for date_str in sorted(current_dates):
        if date_str in date_categories['good']:
            label = f"✓ {date_str} (Good)"
        elif date_str in date_categories['marginal']:
            label = f"~ {date_str} (Marginal)"
        else:
            label = f"✗ {date_str} (Poor)"

        options.append((date_str, label))

    day_select.options = options
    day_select.value = "-- Show All Days --"


def day_select_callback(attr, old, new):
    """Handle single day selection."""
    global all_plots

    if new == "-- Show All Days --":
        update_plots()  # Use filter-based visibility
    else:
        # Hide all, show only selected
        for date_str, plot in all_plots.items():
            plot.visible = (date_str == new)


def load_data_callback():
    """Load tide data based on current selections."""
    global current_analyzer, current_dates, date_categories, all_plots

    status_div.text = "<p>Loading tide data...</p>"

    # Get parameters
    loc = get_selected_location()
    window_start = int(window_start_select.value)
    window_end = int(window_end_select.value)

    # Get date range - convert from milliseconds
    start_date = pd.Timestamp(date_range_slider.value[0], unit='ms')
    end_date = pd.Timestamp(date_range_slider.value[1], unit='ms')

    # Generate date list
    date_range = pd.date_range(start_date, end_date, freq='D')
    dates = date_range.strftime('%Y-%m-%d').tolist()

    # Filter to weekends if requested
    if 0 in weekends_only_checkbox.active:
        dates = [d for d in dates if pd.to_datetime(d).dayofweek in [5, 6]]

    if not dates:
        status_div.text = "<p style='color: red;'>Error: No dates in range. Check weekend filter.</p>"
        return

    current_dates = dates

    # Create analyzer and fetch data
    try:
        current_analyzer = TideAnalyzer(loc['station_id'], loc['name'])

        fetch_start = pd.to_datetime(min(dates)).strftime('%Y%m%d')
        fetch_end = pd.to_datetime(max(dates)).strftime('%Y%m%d')

        current_analyzer.fetch_data(fetch_start, fetch_end)

        # Categorize dates
        date_categories = categorize_dates(current_analyzer, dates, window_start, window_end)

        # Create summary table
        summary_html = create_summary_table(dates, date_categories, window_start, window_end)
        summary_div.text = summary_html

        # Create all plots
        all_plots = {}
        for date_str in dates:
            if date_str in date_categories['good']:
                category = 'good'
            elif date_str in date_categories['marginal']:
                category = 'marginal'
            else:
                category = 'other'

            plot = create_day_plot(date_str, window_start, window_end, category)
            if plot:
                all_plots[date_str] = plot

        # Add ALL plots to layout once
        plots_column.children = list(all_plots.values())

        # Then update visibility based on current filter
        update_plots()
        update_day_selector()

        status_div.text = f"<p style='color: green;'>✓ Loaded {len(dates)} dates. Found {len(date_categories['good'])} good, {len(date_categories['marginal'])} marginal, {len(date_categories['other'])} other.</p>"

    except Exception as e:
        import traceback
        status_div.text = f"<p style='color: red;'>Error: {str(e)}<br><pre>{traceback.format_exc()}</pre></p>"

# Wire up callbacks
load_button.on_click(load_data_callback)
category_filter.on_change('active', lambda attr, old, new: update_plots())
day_select.on_change('value', day_select_callback)


def date_range_picker_cb(attr, old, new):
    """Handle date range picker change."""
    if new and len(new) == 2:
        start_date, end_date = new

        # Convert dates to timestamps for the slider
        start_ts = pd.Timestamp(start_date).value / 1e6  # milliseconds
        end_ts = pd.Timestamp(end_date).value / 1e6

        # Update slider (without triggering its callback to avoid loop)
        date_range_slider.value = (start_ts, end_ts)

        # Trigger reload
        load_data_callback()

        status_div.text = f"Date range set to {start_date} to {end_date}"


date_range_picker.on_change('value', date_range_picker_cb)

# Layout
title_div = Div(text="""
<h1 style='font-family: Arial, sans-serif;'>🌊 Tide Prediction Tool for Kayaking</h1>
<p style='font-family: Arial, sans-serif;'>Interactive tide analysis for planning kayaking trips</p>
""", width=900)

controls_row1 = row(location_select, window_start_select, window_end_select, weekends_only_checkbox)
controls_row2 = row(date_range_slider, date_range_picker, load_button, exit_button)
controls_row3 = row(category_filter, day_select)

layout = column(
    title_div,
    controls_row1,
    controls_row2,
    status_div,
    controls_row3,
    summary_div,
    plots_column
)

# Add to document
curdoc().add_root(layout)
curdoc().title = "Tide Predictor"
