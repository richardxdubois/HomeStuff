from noaa_coops import Station
import pandas as pd
from datetime import datetime, time, timedelta
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, BoxAnnotation, Div
from bokeh.io import output_file
from bokeh.layouts import column, row
import yaml
import pickle
import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

class TideAnalyzer:
    """Analyze tides for kayaking trip planning."""

    def __init__(self, station_id, station_name=None, output_dir='./tide_reports',
                 cache_dir='./tide_cache'):
        """
        Initialize analyzer for a specific location.

        Args:
            station_id: NOAA station ID (e.g., '9414523')
            station_name: Human-readable name (e.g., 'Redwood City')
            output_dir: Directory for saving output files
            cache_dir: Directory for caching tide data
        """
        self.station_id = station_id
        self.station_name = station_name or station_id
        self.station = Station(id=station_id)
        self.df_hourly = None
        self.df_hilo = None
        self.output_dir = Path(output_dir)
        self.cache = TideDataCache(cache_dir)

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_data(self, start_date, end_date, units='english', datum='MLLW',
                   use_cache=True, cache_max_age_days=7):
        """
        Fetch tide predictions from NOAA (or cache).

        Args:
            start_date: 'YYYYMMDD' format
            end_date: 'YYYYMMDD' format
            units: 'english' (feet) or 'metric' (meters)
            datum: 'MLLW' (Mean Lower Low Water) is standard
            use_cache: Whether to use cached data if available
            cache_max_age_days: Maximum age of cache to use (days)
        """
        print(f"Fetching tide data for {self.station_name}...")

        # Try cache first for hourly data
        if use_cache:
            self.df_hourly = self.cache.get(
                self.station_id, start_date, end_date, 'h',
                max_age_days=cache_max_age_days
            )

        # Fetch from NOAA if not cached
        if self.df_hourly is None:
            print("  Fetching hourly data from NOAA...")
            self.df_hourly = self.station.get_data(
                begin_date=start_date,
                end_date=end_date,
                product="predictions",
                datum=datum,
                interval="h",
                units=units,
                time_zone="lst_ldt"  # Local Standard/Local Daylight Time
            )

            if use_cache:
                self.cache.set(self.station_id, start_date, end_date, 'h', self.df_hourly)

        # Try cache for high/low data
        if use_cache:
            self.df_hilo = self.cache.get(
                self.station_id, start_date, end_date, 'hilo',
                max_age_days=cache_max_age_days
            )

        # Fetch from NOAA if not cached
        if self.df_hilo is None:
            print("  Fetching high/low data from NOAA...")
            self.df_hilo = self.station.get_data(
                begin_date=start_date,
                end_date=end_date,
                product="predictions",
                datum=datum,
                interval="hilo",
                units=units,
                time_zone="lst_ldt"  # Local Standard/Local Daylight Time
            )

            if use_cache:
                self.cache.set(self.station_id, start_date, end_date, 'hilo', self.df_hilo)

        print(f"Loaded {len(self.df_hourly)} hourly points and {len(self.df_hilo)} high/low tides")


    def find_good_days(self, window_start=9, window_end=14, weekends_only=False):
        """
        Find days with high tide in target window.

        Args:
            window_start: Hour (24h format, e.g., 9 for 9am)
            window_end: Hour (24h format, e.g., 14 for 2pm)
            weekends_only: Filter to Sat/Sun only

        Returns:
            DataFrame with good kayaking days
        """
        if self.df_hilo is None:
            raise ValueError("No data loaded. Call fetch_data() first.")

        # Filter to high tides only
        df_highs = self.df_hilo[self.df_hilo['type'] == 'H'].copy()
        df_highs['hour'] = df_highs.index.hour
        df_highs['date'] = df_highs.index.date

        # Find highs in window
        in_window = df_highs[(df_highs['hour'] >= window_start) &
                             (df_highs['hour'] <= window_end)]

        # Weekend filter if requested
        if weekends_only:
            in_window = in_window[in_window.index.dayofweek.isin([5, 6])]

        # Format for display
        result = in_window[['v']].copy()
        result['high_tide_time'] = in_window.index.strftime('%I:%M %p')
        result['day_of_week'] = in_window.index.strftime('%A')
        result.columns = ['height_ft', 'high_tide_time', 'day_of_week']

        return result

    def find_marginal_days(self, window_start=9, window_end=14,
                           margin_hours=1, weekends_only=False):
        """
        Find days where high tide is close to the window (e.g., 8am or 3pm).

        Args:
            window_start: Hour for window start
            window_end: Hour for window end
            margin_hours: Hours before/after window to consider
            weekends_only: Filter to Sat/Sun only

        Returns:
            DataFrame with marginal days
        """
        if self.df_hilo is None:
            raise ValueError("No data loaded. Call fetch_data() first.")

        df_highs = self.df_hilo[self.df_hilo['type'] == 'H'].copy()
        df_highs['hour'] = df_highs.index.hour

        # In margin but not in main window
        in_margin = df_highs[
            ((df_highs['hour'] >= window_start - margin_hours) &
             (df_highs['hour'] < window_start)) |
            ((df_highs['hour'] > window_end) &
             (df_highs['hour'] <= window_end + margin_hours))
            ]

        if weekends_only:
            in_margin = in_margin[in_margin.index.dayofweek.isin([5, 6])]

        result = in_margin[['v']].copy()
        result['high_tide_time'] = in_margin.index.strftime('%I:%M %p')
        result['day_of_week'] = in_margin.index.strftime('%A')
        result.columns = ['height_ft', 'high_tide_time', 'day_of_week']

        return result

    def plot_day(self, date_str, window_start=9, window_end=14, show_plot=True):
        """
        Plot tide chart for a single day with info table.

        Args:
            date_str: 'YYYY-MM-DD' format
            window_start: Hour for kayaking window start
            window_end: Hour for kayaking window end
            show_plot: Whether to display immediately

        Returns:
            Bokeh layout object
        """
        if self.df_hourly is None or self.df_hilo is None:
            raise ValueError("No data loaded. Call fetch_data() first.")

        target_date = pd.to_datetime(date_str).date()

        # Filter to single day
        day_data = self.df_hourly[self.df_hourly.index.date == target_date].copy()
        day_hilo = self.df_hilo[self.df_hilo.index.date == target_date].copy()

        if day_data.empty:
            print(f"No data for {date_str}")
            return None

        p = figure(
            x_axis_type='datetime',
            title=f'{self.station_name} Tides - {date_str}',
            width=900,
            height=400,
            toolbar_location='above'
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
        p.line(
            day_data.index,
            day_data['v'],
            line_width=2,
            color='navy',
            legend_label='Tide Level'
        )

        # High/low markers
        highs = day_hilo[day_hilo['type'] == 'H']
        lows = day_hilo[day_hilo['type'] == 'L']

        p.scatter(
            highs.index,
            highs['v'],
            size=12,
            color='green',
            marker='triangle',
            legend_label='High Tide'
        )

        p.scatter(
            lows.index,
            lows['v'],
            size=12,
            color='red',
            marker='inverted_triangle',
            legend_label='Low Tide'
        )

        # Hover tool
        hover = HoverTool(
            tooltips=[
                ('Time', '@x{%H:%M}'),
                ('Height', '@y{0.1f} ft')
            ],
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

        # Create info table and timestamp
        timestamp_div = self._create_timestamp_div()
        info_table = self._create_info_table(date_str, window_start, window_end)

        # Combine into layout - only add components that exist
        components = [timestamp_div]
        if info_table is not None:
            components.append(info_table)
        components.append(p)

        layout = column(*components)

        if show_plot:
            # Create safe filename
            safe_name = self.station_name.replace(' ', '_').lower()
            output_path = self.output_dir / f'tide_{safe_name}_{date_str}.html'
            output_file(str(output_path))
            show(layout)
            print(f"  Saved: {output_path}")

        return layout

    def plot_comparison(self, date_list, window_start=9, window_end=14, filename=None):
        """
        Plot multiple days stacked vertically for comparison.

        Args:
            date_list: List of date strings in 'YYYY-MM-DD' format
            window_start: Hour for window start
            window_end: Hour for window end
            filename: Optional custom filename (without extension)
        """
        components = []

        # Add timestamp at the very top
        components.append(self._create_timestamp_div())

        # Add summary table
        summary_div = self._create_summary_table(date_list, window_start, window_end)
        if summary_div:
            components.append(summary_div)

        # Add individual day plots
        for date_str in date_list:
            day_layout = self.plot_day(date_str, window_start, window_end, show_plot=False)
            if day_layout:
                components.append(day_layout)

        if components:
            full_layout = column(*components)

            if filename is None:
                safe_name = self.station_name.replace(' ', '_').lower()
                filename = f'tide_comparison_{safe_name}'

            output_path = self.output_dir / f'{filename}.html'
            output_file(str(output_path))
            show(full_layout)
            print(f"  Saved: {output_path}")
        else:
            print("No valid plots generated")


    def _create_info_table(self, date_str, window_start, window_end):
        """
        Create an HTML table with tide information for a specific date.

        Args:
            date_str: Date in 'YYYY-MM-DD' format
            window_start: Hour for window start
            window_end: Hour for window end

        Returns:
            Bokeh Div with HTML table
        """
        if self.df_hilo is None:
            return None

        target_date = pd.to_datetime(date_str).date()
        day_hilo = self.df_hilo[self.df_hilo.index.date == target_date].copy()

        if day_hilo.empty:
            return None

        # Build table rows
        rows_html = []
        for idx, row in day_hilo.iterrows():
            tide_type = "High" if row['type'] == 'H' else "Low"
            time_str = idx.strftime('%I:%M %p')
            height = f"{row['v']:.1f} ft"

            # Check if in target window
            hour = idx.hour
            in_window = (tide_type == "High" and
                         window_start <= hour <= window_end)

            row_class = 'in-window' if in_window else ''
            rows_html.append(
                f'<tr class="{row_class}">'
                f'<td>{tide_type}</td>'
                f'<td>{time_str}</td>'
                f'<td>{height}</td>'
                f'</tr>'
            )

        # Check if any high tide is in window
        highs = day_hilo[day_hilo['type'] == 'H']
        good_for_kayaking = any(
            (window_start <= t.hour <= window_end)
            for t in highs.index
        )

        status = "✓ Good for kayaking" if good_for_kayaking else "✗ Not ideal"
        status_class = "good" if good_for_kayaking else "not-good"

        html = f"""
        <div style="font-family: Arial, sans-serif; margin-bottom: 20px;">
            <h3>{self.station_name} - {date_str} ({pd.to_datetime(date_str).strftime('%A')})</h3>
            <p><strong>Target window:</strong> {window_start}:00 - {window_end}:00</p>
            <p class="{status_class}" style="font-weight: bold;">{status}</p>

            <table style="border-collapse: collapse; margin-top: 10px;">
                <thead>
                    <tr style="background-color: #f0f0f0;">
                        <th style="padding: 8px; border: 1px solid #ddd;">Type</th>
                        <th style="padding: 8px; border: 1px solid #ddd;">Time</th>
                        <th style="padding: 8px; border: 1px solid #ddd;">Height</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows_html)}
                </tbody>
            </table>
        </div>

        <style>
            .in-window {{
                background-color: #d4edda;
                font-weight: bold;
            }}
            .good {{
                color: #28a745;
            }}
            .not-good {{
                color: #dc3545;
            }}
            table td {{
                padding: 6px 12px;
                border: 1px solid #ddd;
            }}
        </style>
        """

        return Div(text=html)

    def _create_timestamp_div(self):
        """Create a Div with generation timestamp."""
        timestamp = datetime.now().strftime('%Y-%m-%d %I:%M %p')
        html = f"""
        <div style="font-family: Arial, sans-serif; color: #666; font-size: 12px; margin-bottom: 10px;">
            Report generated: {timestamp}
        </div>
        """
        return Div(text=html)


    def _create_summary_table(self, date_list, window_start, window_end):
        """
        Create a summary table for multiple dates.

        Args:
            date_list: List of date strings
            window_start: Hour for window start
            window_end: Hour for window end

        Returns:
            Bokeh Div with summary table
        """
        if self.df_hilo is None:
            return None

        rows_html = []

        for date_str in date_list:
            target_date = pd.to_datetime(date_str).date()
            day_hilo = self.df_hilo[self.df_hilo.index.date == target_date]

            if day_hilo.empty:
                continue

            # Find high tides
            highs = day_hilo[day_hilo['type'] == 'H']
            lows = day_hilo[day_hilo['type'] == 'L']

            # Check if any high tide in window
            good_highs = [
                (t, h['v']) for t, h in highs.iterrows()
                if window_start <= t.hour <= window_end
            ]

            day_name = pd.to_datetime(date_str).strftime('%A')

            # Format high tide info
            if good_highs:
                high_time, high_height = good_highs[0]
                high_str = f"{high_time.strftime('%I:%M %p')} ({high_height:.1f} ft)"
                status = "✓"
                row_class = "good-day"
            else:
                # Find closest high
                if not highs.empty:
                    closest_high = highs.iloc[0]
                    high_str = f"{closest_high.name.strftime('%I:%M %p')} ({closest_high['v']:.1f} ft)"
                else:
                    high_str = "N/A"
                status = "✗"
                row_class = "not-good-day"

            # Format low tide info - show all lows for the day
            if not lows.empty:
                low_strs = []
                for idx, low in lows.iterrows():
                    low_strs.append(f"{idx.strftime('%I:%M %p')} ({low['v']:.1f} ft)")
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

        html = f"""
        <div style="font-family: Arial, sans-serif; margin: 20px 0;">
            <h2>Summary - {self.station_name}</h2>
            <p><strong>Target window:</strong> {window_start}:00 - {window_end}:00</p>

            <table style="border-collapse: collapse; width: 100%; max-width: 900px;">
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
            .not-good-day {{
                background-color: #f8d7da;
            }}
            table td {{
                padding: 6px 12px;
                border: 1px solid #ddd;
                vertical-align: top;
            }}
        </style>
        """

        return Div(text=html)


class TideLocationManager:
    """Manage multiple kayaking locations."""

    LOCATIONS = {
        'redwood_city': ('9414523', 'Redwood City'),
        'alameda': ('9414750', 'Alameda'),
        'half_moon_bay': ('9414131', 'Half Moon Bay'),
        'monterey': ('9413450', 'Monterey'),
        'bodega_bay': ('9414304', 'Bodega Bay'),
    }

    def __init__(self):
        self.analyzers = {}

    def get_analyzer(self, location_key):
        """Get or create analyzer for a location."""
        if location_key not in self.analyzers:
            if location_key not in self.LOCATIONS:
                raise ValueError(f"Unknown location: {location_key}")

            station_id, name = self.LOCATIONS[location_key]
            self.analyzers[location_key] = TideAnalyzer(station_id, name)

        return self.analyzers[location_key]

    def compare_locations(self, location_keys, date_str, window_start=9, window_end=14):
        """Compare same day across multiple locations."""
        plots = []

        for loc_key in location_keys:
            analyzer = self.get_analyzer(loc_key)
            # Fetch data if not already loaded
            if analyzer.df_hourly is None:
                date_obj = pd.to_datetime(date_str)
                start = date_obj.strftime('%Y%m%d')
                end = (date_obj + timedelta(days=1)).strftime('%Y%m%d')
                analyzer.fetch_data(start, end)

            p = analyzer.plot_day(date_str, window_start, window_end, show_plot=False)
            if p:
                plots.append(p)

        if plots:
            layout = column(plots)
            output_file(f'location_comparison_{date_str}.html')
            show(layout)


class TideConfig:
    """Load and parse YAML configuration for tide analysis."""

    def __init__(self, config_path='kayak_plan.yaml'):
        """Load configuration from YAML file."""
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get_date_list(self):
        """
        Generate list of dates to analyze based on config.

        Returns:
            List of date strings in 'YYYY-MM-DD' format
        """
        dates_config = self.config.get('dates', {})
        all_dates = set()

        # Specific dates
        if 'specific' in dates_config:
            specific = dates_config['specific']
            if specific:  # check not None
                all_dates.update([str(d) for d in specific])

        # Date range
        if 'range' in dates_config:
            range_config = dates_config['range']
            start = pd.to_datetime(range_config['start'])
            end = pd.to_datetime(range_config['end'])

            date_range = pd.date_range(start, end, freq='D')
            all_dates.update(date_range.strftime('%Y-%m-%d').tolist())

        # Convert to sorted list
        date_list = sorted(list(all_dates))

        # Apply weekend filter if specified
        if dates_config.get('weekends_only', False):
            date_list = [d for d in date_list
                         if pd.to_datetime(d).dayofweek in [5, 6]]

        return date_list

    def get_locations(self):
        """
        Get location configurations.

        Returns:
            List of dicts with location info
        """
        return self.config.get('locations', [])

    def get_window(self):
        """
        Get kayaking time window configuration.

        Returns:
            Dict with start_hour, end_hour, margin_hours
        """
        window = self.config.get('window', {})
        return {
            'start_hour': window.get('start_hour', 9),
            'end_hour': window.get('end_hour', 14),
            'margin_hours': window.get('margin_hours', 1)
        }

    def get_output_preferences(self):
        """Get output configuration."""
        output = self.config.get('output', {})
        # Ensure directory exists
        output_dir = Path(output.get('directory') + '/tide_reports')
        output_dir.mkdir(parents=True, exist_ok=True)
        output['directory'] = str(output_dir)
        return output

    def get_date_range_for_fetch(self):
        """
        Get min/max dates for NOAA data fetch.

        Returns:
            Tuple of (start_date, end_date) in YYYYMMDD format
        """
        date_list = self.get_date_list()

        if not date_list:
            raise ValueError("No dates specified in config")

        min_date = min(date_list)
        max_date = max(date_list)

        # Convert to YYYYMMDD format for NOAA API
        start = pd.to_datetime(min_date).strftime('%Y%m%d')
        end = pd.to_datetime(max_date).strftime('%Y%m%d')

        return start, end

    def get_force_display_all(self):
        """
        Check if all dates should be displayed regardless of tide quality.

        Returns:
            Boolean
        """
        dates_config = self.config.get('dates', {})
        return dates_config.get('force_display_all', False)


class TidePlanner:
    """Main planner that uses config and analyzers."""

    def __init__(self, config_path='kayak_plan.yaml'):
        """Initialize planner with config file."""
        self.config = TideConfig(config_path)
        self.analyzers = {}
        self.results = {}

    def run_analysis(self):
        """Run full analysis based on config."""
        window = self.config.get_window()
        locations = self.config.get_locations()
        date_list = self.config.get_date_list()
        force_all = self.config.get_force_display_all()
        output_prefs = self.config.get_output_preferences()

        print(f"Analyzing {len(date_list)} dates across {len(locations)} location(s)")
        print(f"Window: {window['start_hour']}:00 - {window['end_hour']}:00")

        if force_all:
            print(f"Force display mode: showing ALL dates regardless of tide quality")

        # Get date range for fetching
        fetch_start, fetch_end = self.config.get_date_range_for_fetch()

        # Process each location
        for loc in locations:
            loc_key = loc['key']
            print(f"\n{'=' * 60}")
            print(f"Location: {loc['name']}")
            print(f"{'=' * 60}")

            # Create analyzer
            analyzer = TideAnalyzer(loc['station_id'], loc['name'])
            self.analyzers[loc_key] = analyzer

            # Fetch data
            analyzer.fetch_data(fetch_start, fetch_end)

            # Find good days
            good_days = analyzer.find_good_days(
                window_start=window['start_hour'],
                window_end=window['end_hour'],
                weekends_only=False
            )

            # Filter to our date list
            good_days_filtered = good_days[
                good_days.index.strftime('%Y-%m-%d').isin(date_list)
            ]

            # Find marginal days
            marginal_days = analyzer.find_marginal_days(
                window_start=window['start_hour'],
                window_end=window['end_hour'],
                margin_hours=window['margin_hours'],
                weekends_only=False
            )

            marginal_days_filtered = marginal_days[
                marginal_days.index.strftime('%Y-%m-%d').isin(date_list)
            ]

            # Handle force display
            other_days = pd.DataFrame()
            if force_all:
                # Get dates not in good or marginal
                good_dates = set(good_days_filtered.index.strftime('%Y-%m-%d').tolist())
                marginal_dates = set(marginal_days_filtered.index.strftime('%Y-%m-%d').tolist())
                covered_dates = good_dates | marginal_dates

                remaining_dates = [d for d in date_list if d not in covered_dates]

                if remaining_dates:
                    # Get high tides for remaining dates
                    all_highs = analyzer.df_hilo[analyzer.df_hilo['type'] == 'H'].copy()

                    for date_str in remaining_dates:
                        target_date = pd.to_datetime(date_str).date()
                        day_highs = all_highs[all_highs.index.date == target_date]

                        if not day_highs.empty:
                            # Take first high tide of the day
                            first_high = day_highs.iloc[[0]]
                            result = first_high[['v']].copy()
                            result['high_tide_time'] = first_high.index.strftime('%I:%M %p')
                            result['day_of_week'] = first_high.index.strftime('%A')
                            result.columns = ['height_ft', 'high_tide_time', 'day_of_week']
                            other_days = pd.concat([other_days, result])

            # Store results
            self.results[loc_key] = {
                'good': good_days_filtered,
                'marginal': marginal_days_filtered,
                'other': other_days
            }

            # Display results
            print(f"\nGood days ({len(good_days_filtered)}):")
            if not good_days_filtered.empty:
                print(good_days_filtered.to_string())
            else:
                print("  None")

            print(f"\nMarginal days ({len(marginal_days_filtered)}):")
            if not marginal_days_filtered.empty:
                print(marginal_days_filtered.to_string())
            else:
                print("  None")

            if not other_days.empty:
                print(f"\nOther days ({len(other_days)}) - not ideal tides:")
                print(other_days.to_string())

            # Determine which dates to plot
            dates_to_plot = set()

            if force_all:
                # Plot everything
                dates_to_plot.update(date_list)
            else:
                # Plot based on preferences
                if output_prefs.get('plot_good_days', True):
                    if not good_days_filtered.empty:
                        dates_to_plot.update(
                            good_days_filtered.index.strftime('%Y-%m-%d').unique().tolist()
                        )

                if output_prefs.get('plot_marginal_days', False):
                    if not marginal_days_filtered.empty:
                        dates_to_plot.update(
                            marginal_days_filtered.index.strftime('%Y-%m-%d').unique().tolist()
                        )

            if dates_to_plot:
                print(f"\nGenerating plots for {len(dates_to_plot)} day(s)...")
                analyzer.plot_comparison(
                    sorted(list(dates_to_plot)),
                    window_start=window['start_hour'],
                    window_end=window['end_hour']
                )

            # Save CSV if requested
            if output_prefs.get('save_results_csv', True):
                csv_path = f'results_{loc_key}.csv'
                combined = pd.concat([
                    good_days_filtered.assign(category='good'),
                    marginal_days_filtered.assign(category='marginal'),
                    other_days.assign(category='other')
                ])
                if not combined.empty:
                    combined.to_csv(csv_path)
                    print(f"\nResults saved to: {csv_path}")

        # Location comparison plots
        if output_prefs.get('compare_locations', False) and len(locations) > 1:
            self._compare_locations_on_good_days(window)

    def _compare_locations_on_good_days(self, window):
        """Create side-by-side plots for days good at any location."""
        print("\nGenerating location comparison plots...")

        # Find dates that are good at ANY location
        all_good_dates = set()
        for loc_key, results in self.results.items():
            if not results['good'].empty:
                dates = results['good'].index.strftime('%Y-%m-%d').unique()
                all_good_dates.update(dates)

        # Plot each date across all locations
        from bokeh.layouts import column

        for date_str in sorted(all_good_dates):
            plots = []
            for loc_key, analyzer in self.analyzers.items():
                p = analyzer.plot_day(
                    date_str,
                    window_start=window['start_hour'],
                    window_end=window['end_hour'],
                    show_plot=False
                )
                if p:
                    plots.append(p)

            if plots:
                layout = column(plots)
                output_file(f'comparison_{date_str}.html')
                show(layout)

class TideDataCache:
    """Cache NOAA tide data locally to avoid repeated API calls."""

    def __init__(self, cache_dir='./tide_cache'):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / 'cache_metadata.json'
        self.metadata = self._load_metadata()

    def _load_metadata(self):
        """Load cache metadata (what's cached and when)."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Save cache metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _get_cache_key(self, station_id, start_date, end_date, interval):
        """Generate cache key for a data request."""
        return f"{station_id}_{start_date}_{end_date}_{interval}"

    def _get_cache_path(self, cache_key):
        """Get file path for cached data."""
        return self.cache_dir / f"{cache_key}.pkl"

    def get(self, station_id, start_date, end_date, interval='h', max_age_days=7):
        """
        Retrieve cached data if available and not too old.

        Args:
            station_id: NOAA station ID
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            interval: 'h' (hourly) or 'hilo' (high/low only)
            max_age_days: Maximum age of cache in days (default 7)

        Returns:
            DataFrame if cached and fresh, None otherwise
        """
        cache_key = self._get_cache_key(station_id, start_date, end_date, interval)
        cache_path = self._get_cache_path(cache_key)

        # Check if cache exists
        if not cache_path.exists():
            return None

        # Check cache age
        if cache_key in self.metadata:
            cached_date = datetime.fromisoformat(self.metadata[cache_key]['cached_at'])
            age_days = (datetime.now() - cached_date).days

            if age_days > max_age_days:
                print(f"  Cache expired ({age_days} days old), fetching fresh data...")
                return None

            print(f"  Using cached data (age: {age_days} days)")

        # Load and return cached data
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"  Error loading cache: {e}")
            return None

    def set(self, station_id, start_date, end_date, interval, data):
        """
        Cache data for future use.

        Args:
            station_id: NOAA station ID
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            interval: 'h' or 'hilo'
            data: DataFrame to cache
        """
        cache_key = self._get_cache_key(station_id, start_date, end_date, interval)
        cache_path = self._get_cache_path(cache_key)

        # Save data
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)

        # Update metadata
        self.metadata[cache_key] = {
            'station_id': station_id,
            'start_date': start_date,
            'end_date': end_date,
            'interval': interval,
            'cached_at': datetime.now().isoformat(),
            'rows': len(data)
        }
        self._save_metadata()

        print(f"  Cached {len(data)} rows")

    def clear(self, station_id=None, older_than_days=None):
        """
        Clear cache entries.

        Args:
            station_id: Only clear for specific station (None = all)
            older_than_days: Only clear entries older than N days (None = all)
        """
        keys_to_remove = []

        for cache_key, meta in self.metadata.items():
            should_remove = True

            # Filter by station if specified
            if station_id and meta['station_id'] != station_id:
                should_remove = False

            # Filter by age if specified
            if older_than_days:
                cached_date = datetime.fromisoformat(meta['cached_at'])
                age_days = (datetime.now() - cached_date).days
                if age_days < older_than_days:
                    should_remove = False

            if should_remove:
                keys_to_remove.append(cache_key)
                cache_path = self._get_cache_path(cache_key)
                if cache_path.exists():
                    cache_path.unlink()

        # Update metadata
        for key in keys_to_remove:
            del self.metadata[key]

        self._save_metadata()
        print(f"Cleared {len(keys_to_remove)} cache entries")

    def list_cache(self):
        """Display what's in the cache."""
        if not self.metadata:
            print("Cache is empty")
            return

        print(f"\n{'Station':<15} {'Dates':<25} {'Type':<8} {'Age (days)':<12} {'Rows':<8}")
        print("=" * 80)

        for cache_key, meta in sorted(self.metadata.items()):
            cached_date = datetime.fromisoformat(meta['cached_at'])
            age_days = (datetime.now() - cached_date).days

            date_range = f"{meta['start_date']}-{meta['end_date']}"

            print(f"{meta['station_id']:<15} {date_range:<25} {meta['interval']:<8} "
                  f"{age_days:<12} {meta['rows']:<8}")


# Usage
if __name__ == "__main__":
    planner = TidePlanner('/Volumes/Data/Home/tides_predictor/tides_kayak.yaml')
    planner.run_analysis()
