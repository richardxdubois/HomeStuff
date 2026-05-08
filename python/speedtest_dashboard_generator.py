import pandas as pd
import yaml
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta

from bokeh.plotting import figure, save
from bokeh.models import (
    ColumnDataSource, DateRangeSlider, Div, DataTable, TableColumn,  HoverTool,
    HTMLTemplateFormatter, CustomJS, Range1d, LinearAxis
)
from bokeh.layouts import column
from bokeh.resources import CDN


class DashboardGenerator:
    """
    Generates a Bokeh dashboard from the speed monitor log file.
    """

    def __init__(self, app_config=None):
        """
        Initializes the generator with paths to configuration and data.

        Args:
            app_config: path to yaml
        """
        with open(app_config, "r") as f:
            config_data = yaml.safe_load(f)

        self.config_path = config_data["config_path"]
        self.log_path = config_data["log_path"]
        self.output_path = config_data["output_path"]
        self.table_history = config_data["table_history"]

        self.thresholds = self._load_thresholds()
        self.data = self._load_and_prepare_data()

    def _load_thresholds(self):
        """Loads performance thresholds from the config file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('thresholds', {})
        except FileNotFoundError:
            print(f"Warning: Config file '{self.config_path}' not found. Thresholds will not be applied.")
            return {}

    def _load_and_prepare_data(self):
        """Loads and cleans data from the log file into a pandas DataFrame."""
        try:
            df = pd.read_csv(self.log_path)
            # Convert timestamp to datetime objects for plotting
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Convert numeric columns, coercing errors (like 'N/A') to NaN
            for col in ['ping_ms', 'download_mbps', 'upload_mbps']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        except FileNotFoundError:
            print(f"Error: Log file '{self.log_path}' not found. Cannot generate dashboard.")
            return pd.DataFrame()

    def _create_plot(self, source, original_source):
        """Creates the main time-series plot with a secondary Y-axis."""
        # Calculate Y-axis ranges to provide some padding
        y_max_speed = max(self.data['download_mbps'].max(), self.data['upload_mbps'].max()) * 1.1
        y_max_ping = self.data['ping_ms'].max() * 1.1

        tooltips = [
            ("Time", "@timestamp{%F}"),
            ("Download", "@download_mbps{0.2f} Mbps"),
            ("Upload", "@upload_mbps{0.2f} Mbps"),
            ("Ping", "@ping_ms{0.2f} ms"),
        ]

        tooltips = HoverTool(tooltips=tooltips, formatters={"@Time": "datetime"})
        p = figure(
            height=350, sizing_mode="stretch_width",
            x_axis_type="datetime",
            y_range=(0, y_max_speed), title="Internet Performance Over Time"
        )
        p.add_tools(tooltips)
        p.xaxis.axis_label = "Date"
        p.yaxis.axis_label = "Speed (Mbps)"

        # Primary Y-axis for speeds
        # Conditionally render a line or a scatter plot based on the number of data points.
        num_points = len(source.data['timestamp'])

        #if num_points > 1:
        # If we have enough points, draw lines
        p.line(x='timestamp', y='download_mbps', source=source, legend_label="Download", color="dodgerblue",
               width=2)
        p.line(x='timestamp', y='upload_mbps', source=source, legend_label="Upload", color="seagreen", width=2)
        p.line(x='timestamp', y='ping_ms', source=source, legend_label="Ping", color="tomato", width=2,
               y_range_name="ping_range")
        #elif num_points == 1:
        # If we have only one point, draw circles (scatter plot)
        p.scatter(x='timestamp', y='download_mbps', source=source, legend_label="Download", color="dodgerblue",
                  size=4)
        p.scatter(x='timestamp', y='upload_mbps', source=source, legend_label="Upload", color="seagreen", size=4)
        p.scatter(x='timestamp', y='ping_ms', source=source, legend_label="Ping", color="tomato", size=4,
                  y_range_name="ping_range")

        # Secondary Y-axis for ping
        p.extra_y_ranges = {"ping_range": Range1d(start=0, end=y_max_ping)}
        p.add_layout(LinearAxis(y_range_name="ping_range", axis_label="Ping (ms)"), 'right')
        p.line(x='timestamp', y='ping_ms', source=source, legend_label="Ping", color="tomato", width=2,
               y_range_name="ping_range")

        p.legend.location = "bottom_left"
        p.legend.click_policy = "hide"

        return p

    def _create_date_slider(self, data_frame, fig, source):
        """Creates a DateRangeSlider linked to the plot's data source via JavaScript."""
        if data_frame.empty:
            return Div(text="Not enough data for a slider.")

        start_date = data_frame['timestamp'].min()
        end_date = data_frame['timestamp'].max()

        one_month_ago = pd.Timestamp.today() - pd.DateOffset(months=1)

        # Don't let the initial value fall before the slider's start
        initial_start = max(one_month_ago, start_date)

        slider = DateRangeSlider(
            title="Filter Date Range",
            start=start_date, end=end_date,
            value=(initial_start, end_date), step=1,
            sizing_mode="stretch_width"
        )

        fig.x_range = Range1d(start=initial_start, end=end_date)
        # JavaScript callback to filter the plot data based on the slider's range
        callback = CustomJS(args=dict(source=source, original_source=ColumnDataSource(self.data), fig=fig)
                            , code="""
            const data = source.data;
            const original_data = original_source.data;
            const start = cb_obj.value[0];
            const end = cb_obj.value[1];
            fig.x_range.start = start;
            fig.x_range.end = end;

            // Clear existing data
            Object.keys(data).forEach(key => data[key] = []);

            // Repopulate with filtered data
            for (let i = 0; i < original_data.timestamp.length; i++) {
                if (original_data.timestamp[i] >= start && original_data.timestamp[i] <= end) {
                    Object.keys(data).forEach(key => data[key].push(original_data[key][i]));
                }
            }
            source.change.emit();
        """)
        slider.js_on_change('value', callback)
        return slider

    def _create_table(self):
        """Creates a data table of the last 5 days with conditional formatting."""
        days_ago = datetime.now() - timedelta(days=self.table_history)
        recent_data = self.data[self.data['timestamp'] >= days_ago].copy()
        recent_data.sort_values('timestamp', ascending=False, inplace=True)

        if recent_data.empty:
            return Div(text="No data from the last 5 days to display in table.")

        recent_data['ping_threshold'] = self.thresholds.get('ping_ms', float('inf'))
        recent_data['download_threshold'] = self.thresholds.get('download_mbps', 0)
        recent_data['upload_threshold'] = self.thresholds.get('upload_mbps', 0)

        table_source = ColumnDataSource(recent_data)

        # Base template for the background color logic
        base_template = """
        <div style="background-color:<%= 
            (
                (data['status'] === 'FAILED') ||
                (
                    (
                        (cell_name === 'ping_ms' && value > data['ping_threshold']) ||
                        (cell_name === 'download_mbps' && value < data['download_threshold']) ||
                        (cell_name === 'upload_mbps' && value < data['upload_threshold'])
                    ) && data['status'] !== 'FAILED'
                )
            ) ? 'lightcoral' : 'transparent' 
        %>;">
        <%= SCRIPT_REPLACEMENT %> 
        </div>
        """

        # --- THIS IS THE FIX ---
        # Formatter for NUMERIC columns that correctly checks for both null and NaN.
        numeric_script = "(value === null || isNaN(value)) ? 'N/A' : value.toFixed(2)"
        numeric_formatter = HTMLTemplateFormatter(template=base_template.replace("SCRIPT_REPLACEMENT", numeric_script))

        # Formatter for TEXT columns. A simple null check is sufficient here.
        text_script = "(value === null) ? 'N/A' : value"
        text_formatter = HTMLTemplateFormatter(template=base_template.replace("SCRIPT_REPLACEMENT", text_script))

        # Formatter for TIMESTAMPS
        time_formatter = HTMLTemplateFormatter(
            template='<%= new Date(value).toISOString().replace("T", " ").substring(0, 19) %>')

        """
        columns = [
            TableColumn(field="timestamp", title="Timestamp", formatter=time_formatter),
            TableColumn(field="ping_ms", title="Ping (ms)", formatter=numeric_formatter),
            TableColumn(field="download_mbps", title="Download (Mbps)", formatter=numeric_formatter),
            TableColumn(field="upload_mbps", title="Upload (Mbps)", formatter=numeric_formatter),
            TableColumn(field="alerts", title="Alerts", formatter=text_formatter),
            TableColumn(field="server_sponsor", title="Server", formatter=text_formatter),
        ]
        """
        columns = [
            TableColumn(field="timestamp", title="Timestamp", formatter=time_formatter),
            TableColumn(field="ping_ms", title="Ping (ms)"),
            TableColumn(field="download_mbps", title="Download (Mbps)"),
            TableColumn(field="upload_mbps", title="Upload (Mbps)"),
            TableColumn(field="alerts", title="Alerts"),
            TableColumn(field="server_sponsor", title="Server"),
        ]

        return DataTable(source=table_source, columns=columns, sizing_mode="stretch_width", index_position=None)

    def generate_dashboard(self):
        """Assembles and saves the complete Bokeh dashboard to an HTML file."""
        if self.data.empty:
            return

        source = ColumnDataSource(self.data)

        # Create components
        timestamp_div = Div(text=f"<i>Dashboard generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>")
        plot = self._create_plot(source, ColumnDataSource(self.data))
        slider = self._create_date_slider(self.data, plot, source)
        table = self._create_table()

        # Arrange layout
        layout = column(timestamp_div, slider, plot, Div(text="<b>Recent Results (Last " + str(self.table_history) +
                                                              " Days)</b>"), table,
                        sizing_mode="stretch_width")

        # Save to file
        save(layout, self.output_path, title="Internet Speed Dashboard", resources=CDN)
        print(f"Dashboard successfully generated at: {self.output_path}")


if __name__ == '__main__':
    # --- How to use the class ---
    # Create an instance of the generator
    dashboard = DashboardGenerator(app_config="/Users/richarddubois/Code/Home/home_checker/speedtest_dashboard.yaml")
    # Generate the dashboard file
    dashboard.generate_dashboard()
