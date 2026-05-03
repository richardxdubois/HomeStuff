import pandas as pd
from datetime import datetime, timedelta
import os

from bokeh.plotting import figure
from bokeh.models import (ColumnDataSource, HoverTool, Span, Label,
                          Div, NumeralTickFormatter,
                          DatetimeTickFormatter)
from bokeh.layouts import column, row
from bokeh.io import save, output_file
from bokeh.palettes import Category10_6


class DiskDashboard:

    DARK_BG = '#0e1117'
    CARD_BG = '#1a1a2e'
    BORDER = '#2a2a4a'
    TEXT = '#fafafa'
    TEXT_DIM = '#888888'
    GRID = '#1a1a2e'

    CSS = """
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                         Roboto, sans-serif;
            background-color: #0e1117; color: #fafafa;
        }
        .card-container {
            display: flex; gap: 15px; flex-wrap: wrap;
            margin-bottom: 20px;
        }
        .card {
            background: #1a1a2e; border-radius: 10px;
            padding: 20px; min-width: 220px; flex: 1;
            border: 1px solid #2a2a4a;
        }
        .card-title { font-size: 14px; color: #888; margin-bottom: 8px; }
        .card-value { font-size: 28px; font-weight: bold; margin-bottom: 5px; }
        .card-delta { font-size: 14px; margin-bottom: 8px; }
        .card-caption { font-size: 12px; color: #666; }
        .ok { color: #00cc96; }
        .low { color: #ef553b; }
        .unmounted { color: #888; }
        .delta-pos { color: #00cc96; }
        .delta-neg { color: #ef553b; }
        .section-title {
            color: #c0c0c0; font-size: 20px; font-weight: bold;
            margin: 30px 0 10px 0;
        }
        .subtitle { color: #888; font-size: 14px; margin-bottom: 20px; }
        table {
            width: 100%; border-collapse: collapse;
            font-size: 14px; margin: 10px 0;
        }
        th {
            background: #1a1a2e; color: #888;
            padding: 10px 12px; text-align: left;
            border-bottom: 2px solid #2a2a4a;
        }
        td { padding: 8px 12px; border-bottom: 1px solid #1a1a2e; }
        tr:hover { background: #1a1a2e; }
        tr.alert-low { background: #2d1215; }
        tr.alert-unmounted { background: #1a1a1a; }
        .success { color: #00cc96; }
    </style>
    """

    def __init__(self, log_file="/Users/richarddubois/Code/Home/home_checker/disk_space.csv",
                 output_file_path="/Users/richarddubois/Code/Home/home_checker/disk_dashboard.html",
                 days=4):
        self.log_file = log_file
        self.output_file = output_file_path
        self.days = days

    def generate(self):
        """Generate static HTML dashboard. Returns True on success."""

        if not os.path.exists(self.log_file):
            print(f"DiskDashboard: No log file found at {self.log_file}")
            return False

        df = pd.read_csv(self.log_file, parse_dates=['timestamp'])
        df['label'] = df['label'].str.strip('"')

        if df.empty:
            print("DiskDashboard: No data in log file")
            return False

        cutoff = datetime.now() - timedelta(days=self.days)
        df = df[df['timestamp'] >= cutoff]

        if df.empty:
            print(f"DiskDashboard: No data in last {self.days} days")
            return False

        disks = sorted(df['label'].unique())
        colors = {disk: Category10_6[i % len(Category10_6)]
                  for i, disk in enumerate(disks)}

        # Build components
        generated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        header = Div(text=f"""
            {self.CSS}
            <h1>💾 Disk Space Monitor</h1>
            <div class="subtitle">
                Generated: {generated} ·
                Showing last {self.days} days ·
                Log: {self.log_file}
            </div>
        """, sizing_mode='stretch_width')

        status_cards = Div(
            text=self._build_status_cards(df, disks, colors),
            sizing_mode='stretch_width'
        )

        free_title = Div(
            text='<div class="section-title">Free Space Over Time</div>',
            sizing_mode='stretch_width'
        )
        free_chart = self._build_free_space_chart(df, disks, colors)

        pct_title = Div(
            text='<div class="section-title">Used Percentage Over Time</div>',
            sizing_mode='stretch_width'
        )
        pct_chart = self._build_used_pct_chart(df, disks, colors)

        alert_div = Div(
            text=self._build_alert_table(df),
            sizing_mode='stretch_width'
        )

        recent_div = Div(
            text=self._build_recent_table(df),
            sizing_mode='stretch_width'
        )

        layout = column(
            header,
            status_cards,
            free_title, free_chart,
            pct_title, pct_chart,
            alert_div,
            recent_div,
            sizing_mode='stretch_width'
        )

        output_file(self.output_file, title="Disk Space Monitor")
        save(layout)

        print(f"DiskDashboard: Written to {self.output_file}")
        return True

    def _styled_figure(self, title="", y_label="", height=420):
        """Create a dark-themed Bokeh figure."""

        fig = figure(
            x_axis_type='datetime',
            height=height,
            sizing_mode='stretch_width',
            tools="pan,wheel_zoom,box_zoom,reset,save",
            background_fill_color=self.DARK_BG,
            border_fill_color=self.DARK_BG,
            outline_line_color=None,
            title=title,
        )

        fig.title.text_color = self.TEXT
        fig.yaxis.axis_label = y_label
        fig.yaxis.axis_label_text_color = self.TEXT_DIM
        fig.yaxis.major_label_text_color = self.TEXT_DIM
        fig.xaxis.major_label_text_color = self.TEXT_DIM
        fig.xaxis.formatter = DatetimeTickFormatter(
            hours='%m/%d %H:%M',
            days='%m/%d',
            months='%Y/%m'
        )
        fig.grid.grid_line_color = self.GRID
        fig.grid.grid_line_alpha = 0.3

        # NO legend styling here — legend doesn't exist yet

        return fig

    def _style_legend(self, fig):
        """Apply legend styling after glyphs have been added."""
        if fig.legend:
            fig.legend.background_fill_color = self.DARK_BG
            fig.legend.background_fill_alpha = 0.8
            fig.legend.label_text_color = self.TEXT
            fig.legend.border_line_color = None
            fig.legend.location = 'top_left'
            fig.legend.click_policy = 'hide'

    def _build_status_cards(self, df, disks, colors):
        """Build status card HTML for each disk."""

        cards = '<div class="card-container">'

        for disk in disks:
            disk_data = df[df['label'] == disk].sort_values('timestamp')
            if disk_data.empty:
                continue

            latest = disk_data.iloc[-1]
            free = latest['free_gb']
            used = latest['used_pct']
            total = latest['total_gb']
            threshold = latest['threshold_gb']
            status = latest['status']

            if status == "OK":
                icon, status_class = "🟢", "ok"
            elif status == "LOW":
                icon, status_class = "🔴", "low"
            elif status == "UNMOUNTED":
                icon, status_class = "⚫", "unmounted"
            else:
                icon, status_class = "🟡", "low"

            delta_html = ""
            if len(disk_data) >= 2:
                prev = disk_data.iloc[-2]
                delta = free - prev['free_gb']
                delta_class = "delta-pos" if delta >= 0 else "delta-neg"
                arrow = "▲" if delta >= 0 else "▼"
                delta_html = (f'<div class="card-delta {delta_class}">'
                              f'{arrow} {delta:+.1f} GB</div>')

            cards += f"""
            <div class="card">
                <div class="card-title">{icon} {disk}</div>
                <div class="card-value {status_class}">
                    {free:.1f} GB free
                </div>
                {delta_html}
                <div class="card-caption">
                    {used:.1f}% used · {total:.0f} GB total ·
                    Threshold: {threshold:.0f} GB
                </div>
            </div>"""

        cards += '</div>'
        return cards

    def _build_free_space_chart(self, df, disks, colors):
        """Build free space over time chart."""

        fig = self._styled_figure(y_label="Free Space (GB)")

        for disk in disks:
            disk_data = df[df['label'] == disk].sort_values('timestamp')
            if disk_data.empty:
                continue

            color = colors[disk]
            threshold = disk_data['threshold_gb'].iloc[0]

            source = ColumnDataSource(data={
                'timestamp': disk_data['timestamp'],
                'free_gb': disk_data['free_gb'].round(2),
                'used_pct': disk_data['used_pct'].round(1),
                'label': disk_data['label'],
            })

            fig.line('timestamp', 'free_gb', source=source,
                     line_width=2, color=color, legend_label=disk)
            fig.scatter('timestamp', 'free_gb', source=source,
                        size=5, color=color, alpha=0.7,
                        legend_label=disk)

            # Threshold line
            threshold_span = Span(
                location=threshold,
                dimension='width',
                line_color=color,
                line_dash='dashed',
                line_width=1,
                line_alpha=0.6
            )
            fig.add_layout(threshold_span)

            label = Label(
                x=10, y=threshold,
                x_units='screen',
                text=f'{disk} threshold ({threshold:.0f} GB)',
                text_color=color,
                text_font_size='10px',
                text_alpha=0.7,
                y_offset=5
            )
            fig.add_layout(label)

        hover = HoverTool(
            tooltips=[
                ("Disk", "@label"),
                ("Time", "@timestamp{%Y-%m-%d %H:%M}"),
                ("Free", "@free_gb{0.1f} GB"),
                ("Used", "@used_pct{0.1f}%"),
            ],
            formatters={'@timestamp': 'datetime'},
            mode='mouse'
        )
        fig.add_tools(hover)
        self._style_legend(fig)

        return fig

    def _build_used_pct_chart(self, df, disks, colors):
        """Build used percentage over time chart."""

        fig = self._styled_figure(y_label="Used (%)")
        fig.y_range.start = 0
        fig.y_range.end = 100

        for disk in disks:
            disk_data = df[df['label'] == disk].sort_values('timestamp')
            if disk_data.empty:
                continue

            color = colors[disk]

            source = ColumnDataSource(data={
                'timestamp': disk_data['timestamp'],
                'used_pct': disk_data['used_pct'].round(1),
                'free_gb': disk_data['free_gb'].round(1),
                'label': disk_data['label'],
            })

            fig.line('timestamp', 'used_pct', source=source,
                     line_width=2, color=color, legend_label=disk)
            fig.scatter('timestamp', 'used_pct', source=source,
                        size=5, color=color, alpha=0.7,
                        legend_label=disk)

        # Warning line at 90%
        warn_span = Span(location=90, dimension='width',
                         line_color='orange', line_dash='dashed',
                         line_width=1)
        fig.add_layout(warn_span)
        fig.add_layout(Label(
            x=10, y=90, x_units='screen',
            text='90% Warning', text_color='orange',
            text_font_size='10px', y_offset=5
        ))

        # Critical line at 95%
        crit_span = Span(location=95, dimension='width',
                         line_color='red', line_dash='dashed',
                         line_width=1)
        fig.add_layout(crit_span)
        fig.add_layout(Label(
            x=10, y=95, x_units='screen',
            text='95% Critical', text_color='red',
            text_font_size='10px', y_offset=5
        ))

        hover = HoverTool(
            tooltips=[
                ("Disk", "@label"),
                ("Time", "@timestamp{%Y-%m-%d %H:%M}"),
                ("Used", "@used_pct{0.1f}%"),
                ("Free", "@free_gb{0.1f} GB"),
            ],
            formatters={'@timestamp': 'datetime'},
            mode='mouse'
        )
        fig.add_tools(hover)
        self._style_legend(fig)

        return fig

    def _build_alert_table(self, df):
        """Build alert history table HTML."""

        alerts = df[df['status'] != 'OK'].sort_values(
            'timestamp', ascending=False
        )

        if alerts.empty:
            return ('<div class="section-title">⚠️ Alert History</div>'
                    '<p class="success">✅ No alerts in this period.</p>')

        rows = ""
        for _, r in alerts.head(50).iterrows():
            css = ""
            if r['status'] == 'LOW':
                css = 'class="alert-low"'
            elif r['status'] == 'UNMOUNTED':
                css = 'class="alert-unmounted"'

            rows += f"""<tr {css}>
                <td>{r['timestamp'].strftime('%Y-%m-%d %H:%M')}</td>
                <td>{r['label']}</td>
                <td>{r['status']}</td>
                <td>{r['free_gb']:.1f}</td>
                <td>{r['used_pct']:.1f}%</td>
                <td>{r['threshold_gb']:.0f}</td>
            </tr>"""

        return f"""
        <div class="section-title">
            ⚠️ Alert History ({len(alerts)} events)
        </div>
        <table>
            <tr>
                <th>Time</th><th>Disk</th><th>Status</th>
                <th>Free (GB)</th><th>Used %</th>
                <th>Threshold (GB)</th>
            </tr>
            {rows}
        </table>"""

    def _build_recent_table(self, df):
        """Build recent readings table HTML."""

        recent = df.sort_values('timestamp', ascending=False).head(100)

        rows = ""
        for _, r in recent.iterrows():
            css = 'class="alert-low"' if r['status'] == 'LOW' else ''

            rows += f"""<tr {css}>
                <td>{r['timestamp'].strftime('%Y-%m-%d %H:%M')}</td>
                <td>{r['label']}</td>
                <td>{r['status']}</td>
                <td>{r['free_gb']:.1f}</td>
                <td>{r['total_gb']:.0f}</td>
                <td>{r['used_pct']:.1f}%</td>
                <td>{r['threshold_gb']:.0f}</td>
            </tr>"""

        return f"""
        <div class="section-title">Recent Readings</div>
        <table>
            <tr>
                <th>Time</th><th>Disk</th><th>Status</th>
                <th>Free (GB)</th><th>Total (GB)</th>
                <th>Used %</th><th>Threshold (GB)</th>
            </tr>
            {rows}
        </table>"""


if __name__ == "__main__":
    d = DiskDashboard()
    d.generate()
