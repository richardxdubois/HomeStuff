#!/usr/bin/env python3
"""
Blood pressure trend analysis + Bokeh chart.

Reads BP/weight readings from a local file or a Google Sheet URL, prints a
statistical summary to the console, and writes an interactive Bokeh chart
(bp_trend.html) with a trend line and the dose-change date marked.

Usage:
    python3 bp_report.py <source> [--start-date 2026-01-01] [--dose-date 2025-12-01] [--out bp_trend.html]

<source> can be:
    - a local path to a .txt/.csv/.tsv file
    - a Google Sheets URL (edit link, share link, or export link)

--start-date drops readings before that date (default: use full history).

Expected columns (any delimiter pandas can sniff): Date, Systolic, Diastolic,
pulse, weight. Extra columns are ignored.
"""

import argparse
import re
import sys

import pandas as pd
from scipy import stats
from bokeh.plotting import figure, save
from bokeh.models import ColumnDataSource, Span, Label, HoverTool, Legend, BoxAnnotation
from bokeh.io import output_file


def resolve_source(source: str) -> str:
    """If `source` is a Google Sheets URL, rewrite it as a direct CSV export link."""
    m = re.search(r"docs\.google\.com/spreadsheets/d/([a-zA-Z0-9_-]+)", source)
    if not m:
        return source  # local file path, or already a direct link

    sheet_id = m.group(1)
    gid_match = re.search(r"[#&?]gid=(\d+)", source)
    gid = gid_match.group(1) if gid_match else "0"
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"


def load_data(source: str) -> pd.DataFrame:
    url_or_path = resolve_source(source)
    # sep=None + engine='python' auto-detects comma, tab, etc.
    df = pd.read_csv(url_or_path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def linreg_trend(sub: pd.DataFrame, col: str):
    t = (sub["Date"] - sub["Date"].min()).dt.days
    slope, intercept, r, p, se = stats.linregress(t, sub[col])
    return slope, intercept, r, p


def print_summary(df: pd.DataFrame, dose_date: pd.Timestamp) -> None:
    before = df[df["Date"] < dose_date]
    after = df[df["Date"] >= dose_date]

    print(f"\n--- Before {dose_date.date()} (n={len(before)}) ---")
    print(before[["Systolic", "Diastolic", "pulse"]].mean())
    print(f"\n--- After {dose_date.date()} (n={len(after)}) ---")
    print(after[["Systolic", "Diastolic", "pulse"]].mean())

    print("\nMonthly averages:")
    monthly = df.copy()
    monthly["month"] = monthly["Date"].dt.to_period("M")
    print(monthly.groupby("month")[["Systolic", "Diastolic", "pulse"]].mean())

    slope, _, r, p = linreg_trend(df, "Systolic")
    print(f"\nOverall systolic trend: {slope*30:.3f} mmHg/month, r={r:.3f}, p={p:.4f}")
    slope_d, _, r_d, p_d = linreg_trend(df, "Diastolic")
    print(f"Overall diastolic trend: {slope_d*30:.3f} mmHg/month, r={r_d:.3f}, p={p_d:.4f}")

    print("\nSystolic variability by quarter:")
    q_start = df["Date"].min().to_period("Q").start_time
    q_end = df["Date"].max().to_period("Q").end_time
    for period_start in pd.period_range(q_start, q_end, freq="Q"):
        sub = df[(df["Date"] >= period_start.start_time) & (df["Date"] <= period_start.end_time)]
        if len(sub) == 0:
            continue
        print(f"  {period_start}: n={len(sub)}, mean={sub['Systolic'].mean():.1f}, "
              f"std={sub['Systolic'].std():.1f}, range={sub['Systolic'].min()}-{sub['Systolic'].max()}")

    wdf = df.dropna(subset=["weight"])
    if len(wdf) > 1:
        sw, _, rw, pw = linreg_trend(wdf, "weight")
        print(f"\nWeight trend: {sw*30:.3f} lb/month, r={rw:.3f}, p={pw:.4f}, n={len(wdf)}")
        print(f"Correlation weight vs systolic: {wdf['weight'].corr(wdf['Systolic']):.3f}")

    n_high_before = (before["Systolic"] >= 130).sum()
    n_high_after = (after["Systolic"] >= 130).sum()
    print(f"\nReadings >=130 systolic: before={n_high_before}/{len(before)}, after={n_high_after}/{len(after)}")


def make_plot(df: pd.DataFrame, dose_date: pd.Timestamp, out_path: str) -> None:
    output_file(out_path, title="Blood Pressure Trend")

    p = figure(x_axis_type="datetime", width=1000, height=450,
               title=f"Systolic / Diastolic BP since {df['Date'].min().date()}",
               tools="pan,wheel_zoom,box_zoom,reset,save")

    src = ColumnDataSource(df)
    r1 = p.scatter("Date", "Systolic", source=src, size=6, color="crimson", alpha=0.6)
    r2 = p.scatter("Date", "Diastolic", source=src, size=6, color="navy", alpha=0.6)

    span = Span(location=dose_date.timestamp() * 1000, dimension="height",
                line_color="gray", line_dash="dashed", line_width=1.5)
    p.add_layout(span)
    p.add_layout(Label(x=dose_date.timestamp() * 1000, y=148, text=" dose change",
                        text_font_size="9pt", text_color="gray"))

    # trend line from the last local minimum-average quarter onward (recent drift)
    recent = df[df["Date"] >= dose_date].copy()
    legend_items = [("Systolic", [r1]), ("Diastolic", [r2])]
    if len(recent) > 2:
        slope, intercept, r, p_val = linreg_trend(recent, "Systolic")
        xs = pd.date_range(recent["Date"].min(), df["Date"].max())
        ys = intercept + slope * (xs - recent["Date"].min()).days
        r3 = p.line(xs, ys, color="crimson", line_width=2, line_dash="dotted")
        legend_items.append((f"Systolic trend since dose change: {slope*30:+.1f} mmHg/mo", [r3]))

    legend = Legend(items=legend_items, click_policy="hide", label_text_font_size="9pt")
    p.add_layout(legend, "right")

    p.xaxis.axis_label = "Date"
    p.yaxis.axis_label = "mmHg"
    p.add_tools(HoverTool(tooltips=[("Date", "@Date{%F}"), ("Systolic", "@Systolic"), ("Diastolic", "@Diastolic")],
                           formatters={"@Date": "datetime"}))

    save(p)
    print(f"\nChart saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Blood pressure trend analysis + chart")
    parser.add_argument("source", help="Local file path or Google Sheets URL with BP data")
    parser.add_argument("--start-date", default=None,
                         help="Ignore readings before this date (YYYY-MM-DD); default is full history")
    parser.add_argument("--dose-date", default="2025-12-01",
                         help="Date medication dose changed (YYYY-MM-DD), default 2025-12-01")
    parser.add_argument("--out", default="bp_trend.html", help="Output HTML file for the chart")
    args = parser.parse_args()

    df = load_data(args.source)
    if args.start_date:
        start = pd.Timestamp(args.start_date)
        df = df[df["Date"] >= start].reset_index(drop=True)
        if df.empty:
            sys.exit(f"No readings on or after {start.date()}.")
    dose_date = pd.Timestamp(args.dose_date)

    print_summary(df, dose_date)
    make_plot(df, dose_date, args.out)


if __name__ == "__main__":
    main()
