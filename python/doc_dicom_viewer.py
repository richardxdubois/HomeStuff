"""
DICOM Viewer Documentation Generator
=====================================
Generates multi-page HTML User Manual and Developer Reference Manual
from structured content definitions.

Usage:
    python doc_generator.py --output_dir ./docs

This will create:
    ./docs/user_manual/
        index.html
        installation.html
        configuration.html
        launching.html
        interface.html
        workflows.html
        troubleshooting.html
    ./docs/reference_manual/
        index.html
        architecture.html
        dependencies.html
        constants.html
        viewerconfig.html
        imageprocessor.html
        seriesmanager.html
        dicomviewer.html
        dataflow.html
        yaml_schema.html
        extending.html
        limitations.html
"""

import os
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures for documentation content
# ---------------------------------------------------------------------------
@dataclass
class TableRow:
    """A single row of data for an HTML table."""
    cells: list


@dataclass
class Table:
    """An HTML table with headers and rows."""
    headers: list
    rows: list  # list of TableRow
    css_class: str = ""


@dataclass
class Section:
    """A content section within a page (rendered as <h3> or <h4>)."""
    title: str
    content: str = ""  # Raw HTML content
    level: int = 3  # h3 by default


@dataclass
class Page:
    """A single documentation page."""
    filename: str
    title: str
    nav_title: str  # Short title for navigation
    sections: list = field(default_factory=list)  # list of Section objects
    raw_content: str = ""  # Alternative: raw HTML body


@dataclass
class Manual:
    """A complete manual (collection of pages)."""
    title: str
    subtitle: str
    output_dir: str
    pages: list = field(default_factory=list)  # list of Page objects
    accent_color: str = "#2563eb"
    icon: str = "&#x1F4D6;"


# ---------------------------------------------------------------------------
# HTML rendering engine
# ---------------------------------------------------------------------------
class HTMLRenderer:
    """Renders Manual objects into multi-page HTML sites."""

    @staticmethod
    def render_table(table: Table) -> str:
        """Render a Table object to HTML."""
        css = f' class="{table.css_class}"' if table.css_class else ""
        lines = [f"<table{css}>"]
        lines.append("<tr>" + "".join(f"<th>{h}</th>" for h in table.headers) + "</tr>")
        for r in table.rows:
            lines.append("<tr>" + "".join(f"<td>{c}</td>" for c in r.cells) + "</tr>")
        lines.append("</table>")
        return "\n".join(lines)

    @staticmethod
    def get_css(accent: str) -> str:
        """Return the complete CSS stylesheet."""
        return f"""
:root {{
    --accent: {accent};
    --bg: #f8fafc;
    --card: #ffffff;
    --text: #1e293b;
    --muted: #64748b;
    --border: #e2e8f0;
    --code-bg: #1e293b;
    --nav-bg: #1e293b;
    --nav-text: #e2e8f0;
    --nav-hover: {accent};
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.7;
    display: flex; min-height: 100vh;
}}
nav {{
    width: 260px; min-width: 260px; background: var(--nav-bg); color: var(--nav-text);
    padding: 1.5rem 1rem; position: fixed; height: 100vh; overflow-y: auto;
}}
nav h2 {{
    font-size: 1.1rem; margin-bottom: 1rem; color: white;
    padding-bottom: 0.5rem; border-bottom: 1px solid #374151;
}}
nav ul {{ list-style: none; padding: 0; }}
nav li {{ margin-bottom: 0.3rem; }}
nav a {{
    color: var(--nav-text); text-decoration: none; display: block;
    padding: 0.35rem 0.6rem; border-radius: 4px; font-size: 0.92em;
    transition: background 0.15s, color 0.15s;
}}
nav a:hover, nav a.active {{
    background: var(--nav-hover); color: white;
}}
main {{
    margin-left: 260px; padding: 2rem 3rem; max-width: 900px; flex: 1;
}}
h1 {{
    font-size: 2rem; margin-bottom: 0.5rem; color: var(--accent);
}}
h2 {{
    font-size: 1.5rem; margin: 2rem 0 0.75rem; padding-bottom: 0.3rem;
    border-bottom: 2px solid var(--accent);
}}
h3 {{
    font-size: 1.15rem; margin: 1.5rem 0 0.5rem; color: var(--accent);
}}
h4 {{ font-size: 1rem; margin: 1rem 0 0.4rem; }}
p, li {{ margin-bottom: 0.5rem; }}
ul, ol {{ padding-left: 1.5rem; }}
code {{
    background: #f1f5f9; padding: 0.15em 0.4em; border-radius: 4px;
    font-size: 0.9em; font-family: "SF Mono", "Fira Code", Consolas, monospace;
}}
pre {{
    background: var(--code-bg); color: #e2e8f0; padding: 1rem;
    border-radius: 8px; overflow-x: auto; margin: 1rem 0;
    font-size: 0.85em; line-height: 1.5;
}}
pre code {{ background: none; padding: 0; color: inherit; }}
table {{
    width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.92em;
}}
th, td {{
    text-align: left; padding: 0.5rem 0.7rem; border: 1px solid var(--border);
}}
th {{ background: var(--accent); color: white; }}
tr:nth-child(even) {{ background: #f1f5f9; }}
.card {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 1.25rem; margin: 1rem 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}}
.warning {{
    border-left: 4px solid #f59e0b; background: #fffbeb;
    padding: 0.75rem 1rem; border-radius: 0 6px 6px 0; margin: 1rem 0;
}}
.info {{
    border-left: 4px solid var(--accent); background: #eff6ff;
    padding: 0.75rem 1rem; border-radius: 0 6px 6px 0; margin: 1rem 0;
}}
.method-sig {{
    background: #faf5ff; border: 1px solid #e9d5ff;
    padding: 0.6rem 1rem; border-radius: 6px;
    font-family: monospace; font-size: 0.92em; margin: 0.75rem 0;
}}
.subtitle {{ color: var(--muted); font-size: 1.1rem; margin-bottom: 2rem; }}
.flow-diagram {{
    background: white; border: 1px solid var(--border);
    border-radius: 8px; padding: 1.5rem;
    font-family: monospace; font-size: 0.9em; line-height: 2;
}}
.flow-diagram .arrow {{ color: var(--accent); font-weight: bold; }}
.attr-table td:first-child {{ font-family: monospace; font-size: 0.88em; }}
.attr-table td:nth-child(2) {{ font-family: monospace; font-size: 0.88em; color: var(--accent); }}
footer {{
    margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border);
    color: var(--muted); font-size: 0.85rem;
}}
@media (max-width: 768px) {{
    nav {{ display: none; }}
    main {{ margin-left: 0; padding: 1rem; }}
}}
"""

    @staticmethod
    def render_page(manual: Manual, page: Page, all_pages: list) -> str:
        """Render a single page to complete HTML."""
        nav_items = []
        for p in all_pages:
            active = ' class="active"' if p.filename == page.filename else ""
            nav_items.append(
                f'<li><a href="{p.filename}"{active}>{p.nav_title}</a></li>'
            )
        nav_html = "\n".join(nav_items)

        if page.raw_content:
            body_html = page.raw_content
        else:
            parts = []
            for section in page.sections:
                tag = f"h{section.level}"
                parts.append(f"<{tag}>{section.title}</{tag}>")
                if section.content:
                    parts.append(section.content)
            body_html = "\n".join(parts)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{page.title} &mdash; {manual.title}</title>
<style>
{HTMLRenderer.get_css(manual.accent_color)}
</style>
</head>
<body>

<nav>
    <h2>{manual.icon} {manual.title}</h2>
    <ul>
        {nav_html}
    </ul>
</nav>

<main>
    <h1>{page.title}</h1>
    {body_html}
    <footer>
        {manual.title} &copy; 2026. A collaboration between Richard Dubois and Claude (Anthropic). Built with Python, Bokeh, and pydicom.
    </footer>
</main>

</body>
</html>"""

    @staticmethod
    def write_manual(manual: Manual):
        """Write all pages of a manual to disk."""
        out_dir = Path(manual.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for page in manual.pages:
            html = HTMLRenderer.render_page(manual, page, manual.pages)
            filepath = out_dir / page.filename
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"  Wrote: {filepath}")

# ---------------------------------------------------------------------------
# User Manual content definition (UPDATED with W/L documentation)
# ---------------------------------------------------------------------------
def build_user_manual(output_dir: str) -> Manual:
    """Build the complete User Manual content structure."""

    manual = Manual(
        title="DICOM Viewer User Manual",
        subtitle="Version 2.3 &mdash; April 2026",
        output_dir=os.path.join(output_dir, "user_manual"),
        accent_color="#2563eb",
        icon="&#x1F3E5;",
    )

    # ---- Index / Overview ----
    manual.pages.append(Page(
        filename="index.html",
        title="Overview",
        nav_title="Overview",
        raw_content="""
<p class="subtitle">User Manual &mdash; Version 2.3 &mdash; April 2026</p>

<div class="card">
<p>The DICOM Medical Image Viewer is an interactive, browser-based application for viewing
medical images stored in DICOM format. It supports the following modalities:</p>
<ul>
    <li><strong>X-Ray</strong> &mdash; single-frame radiographs</li>
    <li><strong>CT (Computed Tomography)</strong> &mdash; multi-slice axial imaging</li>
    <li><strong>MRI (Magnetic Resonance Imaging)</strong> &mdash; multi-series volumetric imaging</li>
    <li><strong>Ultrasound</strong> &mdash; sonographic imaging</li>
</ul>
</div>

<h2>Key Features</h2>
<ul>
    <li>Interactive image display with hover pixel readout</li>
    <li>Gamma correction and brightness windowing</li>
    <li>DICOM Window/Level presets auto-loaded from image metadata</li>
    <li>Manual Window Center / Width sliders</li>
    <li>10 color maps including Grayscale, Inferno, Viridis, Hot, and PET</li>
    <li>Image inversion (positive/negative toggle)</li>
    <li>Zoom/pan lock for persistent viewing across slices</li>
    <li>Position-sorted slices for anatomically correct ordering</li>
    <li><strong>DICOM metadata inspector</strong> &mdash; view all tags for any image</li>
    <li><strong>Pixel intensity histogram</strong> with W/L range overlay</li>
    <li><strong>Mousewheel slice scrolling</strong> for rapid navigation</li>
    <li>Multi-series navigation for CT and MRI</li>
    <li>Animated slice playback with adjustable speed</li>
    <li>Clip-and-rotate tool for region-of-interest extraction</li>
    <li>Series spatial position visualization</li>
    <li>Configurable via YAML with multiple dataset support</li>
</ul>

<h2>Quick Start</h2>
<ol>
    <li>Install dependencies (see <a href="installation.html">Installation</a>)</li>
    <li>Create a YAML config file (see <a href="configuration.html">Configuration</a>)</li>
    <li>Launch the server (see <a href="launching.html">Launching</a>)</li>
    <li>Open your browser to <code>localhost:5006/dicom_viewer</code></li>
</ol>

<h2>What's New in Version 2.3</h2>
<div class="info">
    <ul>
        <li><strong>DICOM Metadata Inspector:</strong> Toggle a panel showing all DICOM tags
            for the current image. Tags update automatically when you navigate to a new image.
            Binary data (pixel arrays, overlays) is summarized rather than displayed.</li>
        <li><strong>Pixel Intensity Histogram:</strong> Toggle a histogram showing the distribution
            of pixel values in the current image. The title shows min, max, mean, and standard
            deviation. Red dashed lines indicate the current Window/Level display range.
            Updates in real-time as you adjust gamma or W/L settings.</li>
        <li><strong>Mousewheel Slice Scrolling:</strong> Enable mousewheel scrolling over the
            image to rapidly navigate through slices. Disabled by default to avoid conflict
            with Bokeh's zoom wheel. Toggle on when you want fast hands-free navigation.</li>
    </ul>
</div>

<h2>Version History</h2>
<table>
    <tr><th>Version</th><th>Date</th><th>Highlights</th></tr>
    <tr><td>2.3</td><td>April 2026</td><td>Metadata inspector, histogram, mousewheel scrolling</td></tr>
    <tr><td>2.2</td><td>April 2026</td><td>Color maps, invert, zoom lock, position sorting</td></tr>
    <tr><td>2.1</td><td>April 2026</td><td>DICOM Window/Level presets, center/width sliders</td></tr>
    <tr><td>2.0</td><td>April 2026</td><td>Refactored architecture, bug fixes, MONOCHROME support</td></tr>
    <tr><td>1.0</td><td>2024</td><td>Initial release</td></tr>
</table>
""",
    ))

    # ---- Installation (unchanged) ----
    manual.pages.append(Page(
        filename="installation.html",
        title="Installation",
        nav_title="Installation",
        raw_content="""
<h2>Prerequisites</h2>
<ul>
    <li>Python 3.8 or later</li>
    <li>Conda (recommended) or pip</li>
    <li>A modern web browser (Chrome, Firefox, Safari, Edge)</li>
</ul>

<h2>Setup with Conda</h2>
<pre><code># Create and activate environment
conda create -n pydicom python=3.11
conda activate pydicom

# Install dependencies
conda install pydicom numpy scikit-image pyyaml
pip install "pylibjpeg[all]"
pip install bokeh</code></pre>

<h2>Setup with pip only</h2>
<pre><code>python -m venv dicom_env
source dicom_env/bin/activate
pip install pydicom numpy scikit-image pyyaml bokeh
pip install "pylibjpeg[all]"</code></pre>

<h2>Verify Installation</h2>
<pre><code>python -c "import pydicom, bokeh, skimage, yaml; print('All packages OK')"</code></pre>

<h2>Required Packages</h2>
<table>
    <tr><th>Package</th><th>Install</th><th>Purpose</th></tr>
    <tr><td><code>pydicom</code></td><td>conda/pip</td><td>DICOM file reading</td></tr>
    <tr><td><code>pylibjpeg[all]</code></td><td>pip</td><td>JPEG DICOM support</td></tr>
    <tr><td><code>numpy</code></td><td>conda/pip</td><td>Array operations</td></tr>
    <tr><td><code>scikit-image</code></td><td>conda/pip</td><td>Image rotation</td></tr>
    <tr><td><code>bokeh</code></td><td>pip</td><td>Web UI framework</td></tr>
    <tr><td><code>pyyaml</code></td><td>conda/pip</td><td>Configuration parsing</td></tr>
</table>
""",
    ))

    # ---- Configuration (unchanged) ----
    manual.pages.append(Page(
        filename="configuration.html",
        title="Configuration",
        nav_title="Configuration",
        raw_content="""
<h2>Configuration File</h2>
<pre><code># dicom_viewer.yaml
debug: false
gamma_def: 1.0
window_def: 1.0
starter_images: "my_xrays"
data_db:
  my_xrays: "/path/to/xray/dicom/files/"
  chest_ct: "/path/to/ct/dicom/files/"
  brain_mri: "/path/to/mri/dicom/files/"</code></pre>

<h2>Configuration Fields</h2>
<table>
    <tr><th>Field</th><th>Type</th><th>Default</th><th>Description</th></tr>
    <tr><td><code>debug</code></td><td>boolean</td><td><code>false</code></td>
        <td>Verbose debug output to terminal</td></tr>
    <tr><td><code>gamma_def</code></td><td>float</td><td><code>1.0</code></td>
        <td>Default gamma correction</td></tr>
    <tr><td><code>window_def</code></td><td>float</td><td><code>1.0</code></td>
        <td>Default legacy window multiplier</td></tr>
    <tr><td><code>starter_images</code></td><td>string</td><td>&mdash;</td>
        <td>Initial dataset key from <code>data_db</code></td></tr>
    <tr><td><code>data_db</code></td><td>dict</td><td>&mdash;</td>
        <td>Named datasets mapping to directory paths</td></tr>
</table>

<div class="warning">
    <strong>Important:</strong> Directories should contain only DICOM files.
    Hidden files (starting with <code>.</code>) are excluded. Paths must be absolute.
</div>
""",
    ))

    # ---- Launching (unchanged) ----
    manual.pages.append(Page(
        filename="launching.html",
        title="Launching the Viewer",
        nav_title="Launching",
        raw_content="""
<h2>Starting the Server</h2>
<pre><code>bokeh serve dicom_viewer.py --args --app_config "/path/to/dicom_viewer.yaml"</code></pre>

<h2>Opening in Browser</h2>
<pre><code>localhost:5006/dicom_viewer</code></pre>

<h2>Custom Port</h2>
<pre><code>bokeh serve dicom_viewer.py --port 8080 --args --app_config "config.yaml"</code></pre>

<h2>Stopping the Server</h2>
<ul>
    <li>Click <strong>Exit</strong> in the viewer</li>
    <li>Press <code>Ctrl+C</code> in the terminal</li>
</ul>
""",
    ))

    # ---- Interface Guide (UPDATED with Phase 2) ----
    manual.pages.append(Page(
        filename="interface.html",
        title="User Interface Guide",
        nav_title="Interface Guide",
        raw_content="""
<h2>Layout Overview</h2>
<div class="card">
    <ul>
        <li><strong>Top row:</strong> Control widgets (buttons, dropdowns, sliders, panel toggles)</li>
        <li><strong>Bottom-left:</strong> Main image display with color bar</li>
        <li><strong>Bottom-right:</strong> Panels column &mdash; series positions, histogram,
            and metadata (each toggled independently)</li>
    </ul>
</div>

<h2>Control Widgets</h2>
<table>
    <tr><th>Widget</th><th>Description</th></tr>
    <tr><td><strong>Exit</strong></td><td>Shuts down the server.</td></tr>
    <tr><td><strong>Reset</strong></td>
        <td>Restores all settings to defaults: gamma, window, W/L, color map,
            inversion, zoom lock. Closes metadata and histogram panels.
            Disables mousewheel scrolling.</td></tr>
    <tr><td><strong>Pick Imaging</strong></td><td>Switch datasets.</td></tr>
    <tr><td><strong>Mode</strong></td><td>Auto-detected modality indicator.</td></tr>
    <tr><td><strong>Clip</strong></td><td>Tap 4 corners to clip and rotate.</td></tr>
    <tr><td><strong>Gamma</strong></td><td>Gamma correction curve (0&ndash;10).</td></tr>
    <tr><td><strong>Window (legacy)</strong></td><td>Simple brightness multiplier (0&ndash;2).</td></tr>
    <tr><td><strong>W/L Preset</strong></td><td>DICOM window/level presets.</td></tr>
    <tr><td><strong>Window Center</strong></td><td>Display range center.</td></tr>
    <tr><td><strong>Window Width</strong></td><td>Display range width.</td></tr>
    <tr><td><strong>Color Map</strong></td><td>10 color lookup tables.</td></tr>
    <tr><td><strong>Invert</strong></td><td>Reverse current color map.</td></tr>
    <tr><td><strong>Lock Zoom</strong></td><td>Preserve zoom/pan across slices.</td></tr>
    <tr><td><strong>Show Metadata</strong></td>
        <td>Toggle the DICOM metadata inspector panel. Shows all tags for
            the current image. Green when active.</td></tr>
    <tr><td><strong>Show Histogram</strong></td>
        <td>Toggle the pixel intensity histogram. Shows distribution with
            statistics and W/L range lines. Green when active.</td></tr>
    <tr><td><strong>Pick Image</strong></td><td>Select specific image file.</td></tr>
    <tr><td><strong>Pick Series</strong></td><td>Select series (CT/MRI).</td></tr>
    <tr><td><strong>Start/Stop Animation</strong></td><td>Auto-play slices.</td></tr>
    <tr><td><strong>Refresh (ms)</strong></td><td>Animation speed.</td></tr>
    <tr><td><strong>Slice Slider</strong></td><td>Scrub through slices.</td></tr>
    <tr><td><strong>Increment / Decrement</strong></td><td>Step by one slice.</td></tr>
    <tr><td><strong>Sort by Position / Filename</strong></td>
        <td>Toggle slice sort order (CT/MRI).</td></tr>
    <tr><td><strong>Scroll Slices</strong></td>
        <td>Enable mousewheel slice scrolling over the image. Green when active.
            Only visible for CT/MRI/Ultrasound. Disabled by default.</td></tr>
</table>

<h2>Metadata Inspector</h2>
<div class="card">
    <p>The metadata panel displays DICOM tags for the current image in a scrollable
    text area on the right side of the display. It shows:</p>
    <ul>
        <li>The current filename and SOP Class</li>
        <li>Up to 80 DICOM data elements</li>
        <li>Tag number, keyword/name, and value for each element</li>
        <li>Binary data types (pixel data, overlays) are summarized with byte count</li>
        <li>Long values are truncated to 80 characters</li>
    </ul>
    <p><strong>Usage:</strong></p>
    <ol>
        <li>Click <strong>Show Metadata</strong> (button turns green)</li>
        <li>The panel appears below the series position plot</li>
        <li>Navigate to different images &mdash; metadata updates automatically</li>
        <li>Click <strong>Hide Metadata</strong> to close</li>
    </ol>
    <div class="info">
        <strong>Tip:</strong> Use the metadata panel to check specific DICOM tags like
        <code>PixelSpacing</code>, <code>SliceThickness</code>, <code>WindowCenter</code>,
        or <code>Manufacturer</code> without needing an external DICOM tool.
    </div>
</div>

<h2>Pixel Intensity Histogram</h2>
<div class="card">
    <p>The histogram shows the distribution of pixel values in the current image:</p>
    <ul>
        <li><strong>Blue bars:</strong> Pixel value frequency (256 bins)</li>
        <li><strong>Red dashed lines:</strong> Current Window/Level display range
            (low and high bounds)</li>
        <li><strong>Title bar:</strong> Shows min, max, mean, and standard deviation</li>
    </ul>
    <p><strong>Usage:</strong></p>
    <ol>
        <li>Click <strong>Show Histogram</strong> (button turns green)</li>
        <li>The histogram appears in the right panel column</li>
        <li>Adjust <strong>Gamma</strong> &mdash; the histogram updates to show the
            redistributed values</li>
        <li>Adjust <strong>W/L Center</strong> or <strong>Width</strong> &mdash; the red
            lines move to show what portion of the histogram is being displayed</li>
        <li>Navigate to different images &mdash; the histogram updates automatically</li>
    </ol>
    <p><strong>Reading the histogram:</strong></p>
    <ul>
        <li>A <strong>narrow peak</strong> indicates most pixels have similar values
            (low contrast)</li>
        <li>A <strong>wide spread</strong> indicates high dynamic range</li>
        <li>The <strong>red lines</strong> should bracket the region of interest &mdash;
            pixels outside the lines are clipped to black or white</li>
        <li>If the red lines are too narrow, you'll see high contrast but lose detail
            outside the range</li>
        <li>If the red lines are too wide, the image will appear low-contrast</li>
    </ul>
    <div class="info">
        <strong>Tip:</strong> Use the histogram to guide your W/L settings. Move the
        center to the peak of interest and narrow the width until you have good contrast
        in the region you care about.
    </div>
</div>

<h2>Mousewheel Slice Scrolling</h2>
<div class="card">
    <p>When enabled, scrolling the mousewheel over the image advances or reverses
    through slices:</p>
    <ul>
        <li><strong>Scroll up:</strong> Next slice (increment)</li>
        <li><strong>Scroll down:</strong> Previous slice (decrement)</li>
    </ul>
    <p><strong>Why it's disabled by default:</strong> Bokeh uses the mousewheel for
    zooming. When scroll-slicing is enabled, the mousewheel is captured for navigation
    instead. Use the Bokeh toolbar zoom tools if you need to zoom while scrolling
    is active.</p>
    <p><strong>Recommended workflow:</strong></p>
    <ol>
        <li>Zoom to your region of interest using the mousewheel or toolbar</li>
        <li>Click <strong>Lock Zoom</strong></li>
        <li>Enable <strong>Scroll Slices</strong></li>
        <li>Scroll through slices with the mousewheel &mdash; your zoom is preserved</li>
        <li>Disable <strong>Scroll Slices</strong> when done to restore zoom wheel</li>
    </ol>
</div>

<h2>Brightness Controls</h2>
<div class="card">
    <table>
        <tr><th>Control</th><th>What It Does</th><th>When to Use</th></tr>
        <tr><td><strong>Gamma</strong></td><td>Brightness curve shape</td>
            <td>Enhance dark or bright detail</td></tr>
        <tr><td><strong>Window (legacy)</strong></td><td>Simple multiplier</td>
            <td>Quick adjustment</td></tr>
        <tr><td><strong>W/L Center &amp; Width</strong></td><td>Value range</td>
            <td>Standard radiological viewing</td></tr>
    </table>
    <p>Use the <strong>Histogram</strong> panel to visualize how these controls
    affect the pixel value mapping.</p>
</div>

<h2>Color Maps</h2>
<table>
    <tr><th>Color Map</th><th>Best For</th></tr>
    <tr><td>Grayscale</td><td>General diagnostic viewing</td></tr>
    <tr><td>Grayscale (Inverted)</td><td>Alternative viewing</td></tr>
    <tr><td>Hot</td><td>Intensity highlighting</td></tr>
    <tr><td>PET</td><td>Nuclear medicine</td></tr>
    <tr><td>Inferno / Viridis</td><td>Scientific, perceptually uniform</td></tr>
    <tr><td>Cividis</td><td>Colorblind accessible</td></tr>
    <tr><td>Turbo / Plasma / Magma</td><td>General scientific</td></tr>
</table>

<h2>Series Position Plot</h2>
<p>Shows spatial position of each slice. Black/blue = all slices, red = current.</p>

<h2>Log Panel</h2>
<p>Shows the last 10 actions including panel toggles, scroll events, and all operations.</p>
""",
    ))

    # ---- Workflows (UPDATED with Phase 2) ----
    manual.pages.append(Page(
        filename="workflows.html",
        title="Common Workflows",
        nav_title="Workflows",
        raw_content="""
<h2>Viewing an X-Ray</h2>
<ol>
    <li>Select your X-Ray dataset from <strong>Pick Imaging</strong>.</li>
    <li>Adjust brightness using W/L presets or sliders.</li>
    <li>Open <strong>Show Histogram</strong> to see the pixel distribution.</li>
    <li>Use the histogram to guide your W/L settings.</li>
    <li>Open <strong>Show Metadata</strong> to check imaging parameters.</li>
</ol>

<h2>Browsing a CT/MRI Series</h2>
<ol>
    <li>Select your dataset and series.</li>
    <li>Slices are position-sorted by default.</li>
    <li>Zoom into the area of interest.</li>
    <li>Click <strong>Lock Zoom</strong>.</li>
    <li>Enable <strong>Scroll Slices</strong> for mousewheel navigation.</li>
    <li>Scroll through slices with the mousewheel.</li>
</ol>

<h2>Tracking a Structure Through Slices</h2>
<div class="card">
    <ol>
        <li>Ensure <strong>Sort by Position</strong> is active.</li>
        <li>Navigate to a slice where the structure is visible.</li>
        <li>Zoom in and click <strong>Lock Zoom</strong>.</li>
        <li>Set appropriate <strong>W/L Preset</strong>.</li>
        <li>Enable <strong>Scroll Slices</strong>.</li>
        <li>Open <strong>Show Histogram</strong> to monitor pixel values.</li>
        <li>Scroll through slices &mdash; zoom, W/L, and histogram all update together.</li>
    </ol>
</div>

<h2>Investigating Image Parameters</h2>
<div class="card">
    <p>To understand the technical details of your imaging:</p>
    <ol>
        <li>Click <strong>Show Metadata</strong>.</li>
        <li>Look for key tags:
            <table>
                <tr><th>Tag</th><th>What It Tells You</th></tr>
                <tr><td>PixelSpacing</td><td>Physical size of each pixel (mm)</td></tr>
                <tr><td>SliceThickness</td><td>Thickness of each slice (mm)</td></tr>
                <tr><td>WindowCenter / WindowWidth</td><td>Scanner-recommended brightness</td></tr>
                <tr><td>Manufacturer</td><td>Scanner manufacturer</td></tr>
                <tr><td>MagneticFieldStrength</td><td>MRI field strength (Tesla)</td></tr>
                <tr><td>KVP</td><td>X-Ray tube voltage</td></tr>
                <tr><td>RepetitionTime / EchoTime</td><td>MRI sequence timing</td></tr>
                <tr><td>ConvolutionKernel</td><td>CT reconstruction filter</td></tr>
            </table>
        </li>
        <li>Navigate between images &mdash; metadata updates automatically.</li>
    </ol>
</div>

<h2>Using the Histogram for Optimal Brightness</h2>
<div class="card">
    <ol>
        <li>Open <strong>Show Histogram</strong>.</li>
        <li>The blue bars show where pixel values are concentrated.</li>
        <li>The red dashed lines show your current W/L display range.</li>
        <li>Adjust <strong>Window Center</strong> to move the red lines to
            bracket the peak of interest.</li>
        <li>Adjust <strong>Window Width</strong> to narrow or widen the range:
            <ul>
                <li>Narrower = higher contrast in the selected range</li>
                <li>Wider = more values visible but lower contrast</li>
            </ul>
        </li>
        <li>Change <strong>Gamma</strong> and watch the histogram reshape &mdash;
            gamma redistributes where pixel values fall.</li>
    </ol>
    <div class="info">
        <strong>Example:</strong> For a CT chest scan, you might see two peaks in the
        histogram &mdash; one for air (low values) and one for tissue (higher values).
        Set your W/L center between the peaks and width to cover just the tissue peak
        for optimal soft tissue viewing.
    </div>
</div>

<h2>Animating with Panels Open</h2>
<ol>
    <li>Set your W/L, color map, and zoom.</li>
    <li>Open the panels you want to monitor (histogram, metadata).</li>
    <li>Note: panels update on manual navigation (increment, decrement, slider, scroll)
        but <strong>not</strong> during animation for performance. Stop the animation
        to see updated panels for the current frame.</li>
</ol>

<h2>Clipping and Rotating</h2>
<ol>
    <li>Click <strong>Clip</strong>, tap 4 corners, image is cropped and rotated.</li>
</ol>
<div class="warning">
    <strong>Note:</strong> Cannot be undone. Select a different image to restore.
</div>

<h2>Using Color Maps</h2>
<table>
    <tr><th>Use Case</th><th>Map</th></tr>
    <tr><td>Standard reading</td><td>Grayscale</td></tr>
    <tr><td>Bone/intensity</td><td>Hot</td></tr>
    <tr><td>Nuclear medicine</td><td>PET</td></tr>
    <tr><td>Subtle differences</td><td>Inferno / Viridis</td></tr>
    <tr><td>Colorblind viewers</td><td>Cividis</td></tr>
</table>
""",
    ))

    # ---- Troubleshooting (UPDATED with Phase 2) ----
    manual.pages.append(Page(
        filename="troubleshooting.html",
        title="Troubleshooting",
        nav_title="Troubleshooting",
        raw_content="""
<h2>Common Issues</h2>
<table>
    <tr><th>Problem</th><th>Solution</th></tr>
    <tr>
        <td>Blank or white image</td>
        <td>Try a different W/L Preset. Open the <strong>Histogram</strong> to see where
            pixel values are concentrated and adjust W/L accordingly.
            Click <strong>Reset</strong>.</td>
    </tr>
    <tr>
        <td>Image too dark or too bright</td>
        <td>Open the <strong>Histogram</strong>. Move the red W/L lines to bracket
            the main peak. Adjust Gamma. Try <strong>Invert</strong>.</td>
    </tr>
    <tr>
        <td>Colors look wrong</td>
        <td>Check <strong>Color Map</strong> dropdown and <strong>Invert</strong> toggle.
            Click <strong>Reset</strong> to restore Grayscale.</td>
    </tr>
    <tr>
        <td>Mousewheel zooms instead of scrolling slices</td>
        <td>Enable <strong>Scroll Slices</strong> (button turns green). The mousewheel
            is captured for slice navigation when this is active.</td>
    </tr>
    <tr>
        <td>Can't zoom while scroll-slicing is enabled</td>
        <td>Disable <strong>Scroll Slices</strong>, zoom to desired level using
            the mousewheel, click <strong>Lock Zoom</strong>, then re-enable
            <strong>Scroll Slices</strong>.</td>
    </tr>
    <tr>
        <td>Zoom resets when changing slices</td>
        <td>Click <strong>Lock Zoom</strong> before navigating.</td>
    </tr>
    <tr>
        <td>Slices out of anatomical order</td>
        <td>Ensure <strong>Sort by Position</strong> is active (green).</td>
    </tr>
    <tr>
        <td>Metadata panel shows limited tags</td>
        <td>The panel displays up to 80 tags. Binary data types (pixel data, overlays)
            are summarized with byte counts. Use an external DICOM tool for complete
            tag exploration.</td>
    </tr>
    <tr>
        <td>Histogram red lines don't appear</td>
        <td>The W/L range lines may be outside the histogram's visible range.
            Try adjusting the W/L sliders to bring them into the plotted range.
            You can click the "W/L Range" legend entry to toggle line visibility.</td>
    </tr>
    <tr>
        <td>Panels don't update during animation</td>
        <td>Metadata and histogram panels update on manual navigation only (not during
            animation) for performance. Stop the animation to see current values.</td>
    </tr>
    <tr>
        <td>W/L Preset shows only "Manual"</td>
        <td>The DICOM image lacks Window Center/Width metadata.</td>
    </tr>
    <tr>
        <td>Server won't start</td>
        <td>Check YAML config path, verify directories exist.</td>
    </tr>
</table>

<h2>Debug Mode</h2>
<p>Set <code>debug: true</code> in YAML for verbose output.</p>

<h2>Diagnostic Checklist</h2>
<div class="info">
    <ol>
        <li>Enable <code>debug: true</code></li>
        <li>Check terminal for errors</li>
        <li>Check the Log panel</li>
        <li>Open <strong>Show Metadata</strong> to inspect DICOM tags</li>
        <li>Open <strong>Show Histogram</strong> to understand the pixel data</li>
        <li>Click <strong>Reset</strong> to restore all defaults</li>
        <li>Try a different dataset to isolate the issue</li>
    </ol>
</div>
""",
    ))

    return manual

# ---------------------------------------------------------------------------
# Reference Manual content definition (UPDATED with W/L documentation)
# ---------------------------------------------------------------------------
def build_reference_manual(output_dir: str) -> Manual:
    """Build the complete Developer Reference Manual."""

    manual = Manual(
        title="DICOM Viewer Reference",
        subtitle="Developer API &amp; Architecture Guide &mdash; Version 2.3 &mdash; April 2026",
        output_dir=os.path.join(output_dir, "reference_manual"),
        accent_color="#7c3aed",
        icon="&#x1F527;",
    )

    # ---- Architecture ----
    manual.pages.append(Page(
        filename="index.html",
        title="Architecture Overview",
        nav_title="Architecture",
        raw_content="""
<p class="subtitle">Developer API &amp; Architecture Guide &mdash; Version 2.3</p>

<h2>Module Structure</h2>
<table>
    <tr><th>Component</th><th>Responsibility</th></tr>
    <tr><td><code>ViewerConfig</code></td><td>YAML configuration dataclass</td></tr>
    <tr><td><code>ImageProcessor</code></td><td>Stateless image processing and metadata extraction</td></tr>
    <tr><td><code>SeriesManager</code></td><td>Series metadata, spatial analysis, position/filename sorting</td></tr>
    <tr><td><code>DicomViewer</code></td><td>Main UI: widgets, callbacks, color maps, zoom, panels, scroll</td></tr>
</table>

<h2>Module-Level Functions</h2>
<table>
    <tr><th>Function</th><th>Purpose</th></tr>
    <tr><td><code>_build_hot_palette(n=256)</code></td><td>Generates Hot colormap</td></tr>
    <tr><td><code>_build_pet_palette(n=256)</code></td><td>Generates PET colormap</td></tr>
</table>

<h2>Module-Level Data</h2>
<table>
    <tr><th>Name</th><th>Type</th><th>Description</th></tr>
    <tr><td><code>COLOR_LUTS</code></td><td>dict</td><td>10 color palettes (256 entries each)</td></tr>
</table>

<h2>Design Principles</h2>
<ul>
    <li>Separation of concerns: processing, metadata, and UI are independent</li>
    <li>Named constants for all magic numbers</li>
    <li>Specific exception types (no bare <code>except</code>)</li>
    <li>Panel updates via central <code>_update_panels()</code> method</li>
    <li>Mousewheel scroll uses CustomJS &rarr; hidden TextInput &rarr; Python callback pattern</li>
    <li>LUT composability: inversion as a modifier on any base palette</li>
    <li>Zoom state saved/restored explicitly via range tuples</li>
</ul>
""",
    ))

    # ---- Dependencies ----
    manual.pages.append(Page(
        filename="dependencies.html",
        title="Dependencies",
        nav_title="Dependencies",
        raw_content="""
<h2>Required Packages</h2>
<table>
    <tr><th>Package</th><th>Min Version</th><th>Purpose</th></tr>
    <tr><td><code>pydicom</code></td><td>&ge;2.0</td><td>DICOM reading</td></tr>
    <tr><td><code>pylibjpeg[all]</code></td><td>&ge;1.0</td><td>JPEG DICOM</td></tr>
    <tr><td><code>numpy</code></td><td>&ge;1.20</td><td>Arrays</td></tr>
    <tr><td><code>scikit-image</code></td><td>&ge;0.18</td><td>Rotation</td></tr>
    <tr><td><code>bokeh</code></td><td>&ge;3.0</td><td>Web UI</td></tr>
    <tr><td><code>pyyaml</code></td><td>&ge;5.0</td><td>YAML</td></tr>
</table>

<h2>Bokeh Imports</h2>
<pre><code>from bokeh.palettes import (
    Greys256, Inferno256, Viridis256, Turbo256,
    Cividis256, Plasma256, Magma256
)
from bokeh.models import (
    ColumnDataSource, LinearColorMapper, ColorBar,
    HoverTool, Button, Slider, Div, TapTool, Select,
    RadioButtonGroup, Toggle, TextInput,
    CustomJS, PreText        # New in v2.3
)
from bokeh.events import MouseWheel  # New in v2.3</code></pre>
""",
    ))

    # ---- Constants (UPDATED) ----
    manual.pages.append(Page(
        filename="constants.html",
        title="Module Constants",
        nav_title="Constants",
        raw_content="""
<h2>Display Scale</h2>
<table>
    <tr><th>Constant</th><th>Value</th></tr>
    <tr><td><code>XRAY_SCALE</code></td><td>0.5</td></tr>
    <tr><td><code>CT_SCALE</code></td><td>1.5</td></tr>
    <tr><td><code>MRI_SCALE</code></td><td>4.0</td></tr>
    <tr><td><code>US_SCALE</code></td><td>1.5</td></tr>
</table>

<h2>Gamma</h2>
<table>
    <tr><th>Constant</th><th>Value</th></tr>
    <tr><td><code>MRI_GAMMA / US_GAMMA</code></td><td>2.0</td></tr>
    <tr><td><code>DEFAULT_GAMMA</code></td><td>1.0</td></tr>
    <tr><td><code>DEFAULT_WINDOW</code></td><td>1.0</td></tr>
</table>

<h2>Window/Level</h2>
<table>
    <tr><th>Constant</th><th>Value</th><th>Description</th></tr>
    <tr><td><code>WL_CENTER_DEFAULT</code></td><td>2000.0</td><td>Fallback center</td></tr>
    <tr><td><code>WL_WIDTH_DEFAULT</code></td><td>4000.0</td><td>Fallback width</td></tr>
    <tr><td><code>WL_CENTER_SLIDER_MIN</code></td><td>-2000.0</td><td>Slider min</td></tr>
    <tr><td><code>WL_CENTER_SLIDER_MAX</code></td><td>20000.0</td><td>Slider default max</td></tr>
    <tr><td><code>WL_WIDTH_SLIDER_MIN</code></td><td>1.0</td><td>Slider min</td></tr>
    <tr><td><code>WL_WIDTH_SLIDER_MAX</code></td><td>40000.0</td><td>Slider default max</td></tr>
    <tr><td><code>WL_SLIDER_STEP</code></td><td>10.0</td><td>Step size</td></tr>
    <tr><td><code>WL_MANUAL_LABEL</code></td><td>"Manual"</td><td>Dropdown label</td></tr>
</table>

<h2>Color Map</h2>
<table>
    <tr><th>Constant</th><th>Description</th></tr>
    <tr><td><code>COLOR_LUTS</code></td><td>Dict of 10 palette names &rarr; 256-entry hex lists</td></tr>
    <tr><td><code>DEFAULT_LUT</code></td><td>"Grayscale"</td></tr>
</table>

<h2>Histogram (New in v2.3)</h2>
<table>
    <tr><th>Constant</th><th>Value</th><th>Description</th></tr>
    <tr><td><code>HISTOGRAM_BINS</code></td><td>256</td><td>Number of histogram bins</td></tr>
    <tr><td><code>HISTOGRAM_HEIGHT</code></td><td>300</td><td>Figure height in pixels</td></tr>
    <tr><td><code>HISTOGRAM_WIDTH</code></td><td>500</td><td>Figure width in pixels</td></tr>
</table>

<h2>Metadata Panel (New in v2.3)</h2>
<table>
    <tr><th>Constant</th><th>Value</th><th>Description</th></tr>
    <tr><td><code>METADATA_MAX_LINES</code></td><td>80</td>
        <td>Maximum DICOM elements to display</td></tr>
    <tr><td><code>METADATA_WIDTH</code></td><td>500</td><td>Panel width in pixels</td></tr>
    <tr><td><code>METADATA_HEIGHT</code></td><td>400</td><td>Panel height in pixels</td></tr>
    <tr><td><code>METADATA_SKIP_VR</code></td><td>{"OB","OW","OF","SQ","UN","UC","UR"}</td>
        <td>VR types to summarize instead of display</td></tr>
</table>

<h2>UI and Analysis</h2>
<table>
    <tr><th>Constant</th><th>Value</th></tr>
    <tr><td><code>GAMMA_SLIDER_MAX</code></td><td>10.0</td></tr>
    <tr><td><code>WINDOW_SLIDER_MAX</code></td><td>2.0</td></tr>
    <tr><td><code>MAX_LOG_MESSAGES</code></td><td>10</td></tr>
    <tr><td><code>DEFAULT_REFRESH_MS</code></td><td>500.0</td></tr>
    <tr><td><code>NORMAL_THRESHOLD</code></td><td>0.95</td></tr>
    <tr><td><code>ANIMATION_SLICE_RESET</code></td><td>0</td></tr>
</table>
""",
    ))

    # ---- ViewerConfig (unchanged) ----
    manual.pages.append(Page(
        filename="viewerconfig.html",
        title="Class: ViewerConfig",
        nav_title="ViewerConfig",
        raw_content="""
<h2>Overview</h2>
<div class="method-sig">@dataclass<br>class ViewerConfig:</div>
<h2>Fields</h2>
<table class="attr-table">
    <tr><th>Field</th><th>Type</th><th>Default</th></tr>
    <tr><td>debug</td><td>bool</td><td>False</td></tr>
    <tr><td>gamma_def</td><td>float</td><td>1.0</td></tr>
    <tr><td>window_def</td><td>float</td><td>1.0</td></tr>
    <tr><td>starter_images</td><td>str</td><td>""</td></tr>
    <tr><td>data_db</td><td>dict</td><td>{}</td></tr>
</table>
<h2>Methods</h2>
<div class="method-sig">@classmethod from_yaml(cls, path: str) &rarr; ViewerConfig</div>
""",
    ))

    # ---- ImageProcessor (unchanged) ----
    manual.pages.append(Page(
        filename="imageprocessor.html",
        title="Class: ImageProcessor",
        nav_title="ImageProcessor",
        raw_content="""
<h2>Overview</h2>
<p>Stateless utilities. All <code>@staticmethod</code>.</p>
<h2>Image Processing</h2>
<div class="method-sig">apply_photometric(image, ds) &rarr; ndarray</div>
<div class="method-sig">apply_rescale(image, ds) &rarr; ndarray</div>
<div class="method-sig">ensure_2d(image) &rarr; ndarray</div>
<div class="method-sig">perform_gamma(image, gamma, original_dtype) &rarr; ndarray</div>
<h2>Metadata Extraction</h2>
<div class="method-sig">extract_wl_presets(ds) &rarr; list[dict]</div>
<h2>Geometry</h2>
<div class="method-sig">rotated_rectangle_properties(corners) &rarr; tuple</div>
""",
    ))

    # ---- SeriesManager (unchanged from v2.2) ----
    manual.pages.append(Page(
        filename="seriesmanager.html",
        title="Class: SeriesManager",
        nav_title="SeriesManager",
        raw_content="""
<h2>Overview</h2>
<p>Series metadata with filename and position-based sorting.</p>
<h2>Attributes</h2>
<table class="attr-table">
    <tr><th>Attribute</th><th>Type</th><th>Description</th></tr>
    <tr><td>series_map</td><td>dict</td><td>Nested series/image metadata</td></tr>
    <tr><td>series</td><td>list</td><td>Sorted series numbers</td></tr>
    <tr><td>series_extrema</td><td>dict</td><td>Position ranges</td></tr>
    <tr><td>series_pos_index</td><td>int</td><td>Active axis (0/1/2)</td></tr>
    <tr><td>sort_by_position</td><td>bool</td><td>Default: True</td></tr>
</table>
<h2>Methods</h2>
<div class="method-sig">categorize(images_list, data_dir, debug, log_callback)</div>
<div class="method-sig">get_series_images(series_key) &rarr; list</div>
<div class="method-sig">get_series_images_by_position(series_key, axis=2) &rarr; list</div>
<div class="method-sig">determine_axis(series_key, images) &rarr; int</div>
""",
    ))

    # ---- DicomViewer (UPDATED with Phase 2) ----
    manual.pages.append(Page(
        filename="dicomviewer.html",
        title="Class: DicomViewer",
        nav_title="DicomViewer",
        raw_content="""
<h2>Overview</h2>
<p>Main application class with full UI management including analysis panels
and mousewheel navigation.</p>

<h2>Key Instance Attributes</h2>
<table class="attr-table">
    <tr><th>Attribute</th><th>Type</th><th>Description</th></tr>
    <tr><td>config</td><td>ViewerConfig</td><td>Configuration</td></tr>
    <tr><td>img_proc</td><td>ImageProcessor</td><td>Processing</td></tr>
    <tr><td>series_mgr</td><td>SeriesManager</td><td>Series management</td></tr>
    <tr><td>ds</td><td>pydicom.Dataset</td><td>Current DICOM dataset</td></tr>
    <tr><td>processed_image</td><td>ndarray</td><td>Display image</td></tr>
    <tr><td>wl_presets</td><td>list[dict]</td><td>W/L presets</td></tr>
    <tr><td>color_mapper</td><td>LinearColorMapper</td><td>Display range + palette</td></tr>
    <tr><td>current_lut</td><td>str</td><td>Active color map name</td></tr>
    <tr><td>is_inverted</td><td>bool</td><td>Palette inversion state</td></tr>
    <tr><td>zoom_locked</td><td>bool</td><td>Zoom persistence state</td></tr>
    <tr><td>saved_x_range</td><td>tuple/None</td><td>Saved x range</td></tr>
    <tr><td>saved_y_range</td><td>tuple/None</td><td>Saved y range</td></tr>
    <tr><td colspan="3" style="background:#faf5ff;font-weight:bold;">New in v2.3</td></tr>
    <tr><td>metadata_pre</td><td>PreText</td>
        <td>Scrollable text widget for DICOM tag display</td></tr>
    <tr><td>fig_histogram</td><td>Figure</td>
        <td>Bokeh figure for pixel intensity histogram</td></tr>
    <tr><td>histogram_source</td><td>ColumnDataSource</td>
        <td>Data for histogram bars: <code>{top, left, right}</code></td></tr>
    <tr><td>histogram_wl_source</td><td>ColumnDataSource</td>
        <td>Data for W/L range lines: <code>{x, y}</code> with NaN separator</td></tr>
    <tr><td>scroll_input</td><td>TextInput</td>
        <td>Hidden widget bridging JS mousewheel events to Python</td></tr>
    <tr><td>scroll_js</td><td>CustomJS</td>
        <td>JavaScript callback attached to MouseWheel event</td></tr>
</table>

<h2>Phase 2 Methods (New in v2.3)</h2>

<h3>Metadata Panel</h3>
<table>
    <tr><th>Method</th><th>Description</th></tr>
    <tr><td><code>_metadata_toggle_cb(active)</code></td>
        <td>Shows/hides metadata panel. Calls <code>_update_metadata()</code> on show.</td></tr>
    <tr><td><code>_update_metadata()</code></td>
        <td>Iterates <code>self.ds</code> elements. Skips VR types in
            <code>METADATA_SKIP_VR</code> (shows byte count instead). Truncates
            values &gt;80 chars. Limits to <code>METADATA_MAX_LINES</code> elements.</td></tr>
</table>

<h3>Histogram</h3>
<table>
    <tr><th>Method</th><th>Description</th></tr>
    <tr><td><code>_histogram_toggle_cb(active)</code></td>
        <td>Shows/hides histogram. Calls <code>_update_histogram()</code> on show.</td></tr>
    <tr><td><code>_update_histogram()</code></td>
        <td>Computes <code>np.histogram()</code> with <code>HISTOGRAM_BINS</code> bins on
            <code>processed_image</code>. Updates <code>histogram_source</code>.
            Calls <code>_update_histogram_wl_lines()</code>. Updates title with
            min/max/mean/std statistics.</td></tr>
    <tr><td><code>_update_histogram_wl_lines()</code></td>
        <td>Draws two vertical lines at <code>color_mapper.low</code> and
            <code>.high</code> using NaN-separated line data. Scales to max
            histogram count.</td></tr>
</table>

<h3>Mousewheel Scrolling</h3>
<table>
    <tr><th>Method</th><th>Description</th></tr>
    <tr><td><code>_scroll_toggle_cb(active)</code></td>
        <td>When enabled: attaches <code>scroll_js</code> CustomJS to
            <code>fig_image</code>'s MouseWheel event. When disabled: attaches
            a no-op JS callback (Bokeh doesn't support clean removal).</td></tr>
    <tr><td><code>_scroll_cb(attr, old, new)</code></td>
        <td>Python callback triggered by hidden <code>scroll_input</code> TextInput.
            Computes scroll delta from old/new integer values. Increments or
            decrements <code>current_slice</code>. Calls <code>_get_new_image()</code>
            and <code>_update_panels()</code>.</td></tr>
</table>

<h3>Panel Helper</h3>
<table>
    <tr><th>Method</th><th>Description</th></tr>
    <tr><td><code>_update_panels()</code></td>
        <td>Central method that refreshes metadata and histogram if their panels
            are currently visible. Called by all navigation callbacks:
            <code>_name_cb</code>, <code>_db_dropdown_cb</code>,
            <code>_series_cb</code>, <code>_increment_cb</code>,
            <code>_decrement_cb</code>, <code>_series_slider_slice_cb</code>,
            <code>_scroll_cb</code>.</td></tr>
</table>

<h2>Mousewheel Architecture</h2>
<div class="card">
    <p>Bokeh doesn't provide a native Python callback for mousewheel events on figures.
    The implementation uses a JS&rarr;Python bridge pattern:</p>
    <ol>
        <li><strong>CustomJS</strong> (<code>scroll_js</code>) attached to
            <code>fig_image</code>'s <code>MouseWheel</code> event</li>
        <li>JS reads scroll delta, increments/decrements an integer counter</li>
        <li>JS writes the counter to a <strong>hidden TextInput</strong>
            (<code>scroll_input</code>)</li>
        <li>TextInput's <code>on_change("value")</code> triggers Python
            <code>_scroll_cb</code></li>
        <li>Python computes the delta between old and new counter values</li>
        <li>Python navigates slices accordingly</li>
    </ol>
    <p>This pattern is reusable for any JS event that needs to reach Python.</p>
</div>

<h2>Updated Methods (changed in v2.3)</h2>
<table>
    <tr><th>Method</th><th>What Changed</th></tr>
    <tr><td><code>_name_cb</code></td><td>+ <code>_update_panels()</code></td></tr>
    <tr><td><code>_db_dropdown_cb</code></td><td>+ <code>_update_panels()</code></td></tr>
    <tr><td><code>_series_cb</code></td><td>+ <code>_update_panels()</code></td></tr>
    <tr><td><code>_gamma_cb</code></td><td>+ histogram update when visible</td></tr>
    <tr><td><code>_wl_preset_cb</code></td><td>+ W/L line update when visible</td></tr>
    <tr><td><code>_wl_manual_cb</code></td><td>+ W/L line update when visible</td></tr>
    <tr><td><code>_increment_cb</code></td><td>+ <code>_update_panels()</code></td></tr>
    <tr><td><code>_decrement_cb</code></td><td>+ <code>_update_panels()</code></td></tr>
    <tr><td><code>_series_slider_slice_cb</code></td><td>+ <code>_update_panels()</code></td></tr>
    <tr><td><code>_update_visibility</code></td><td>+ <code>scroll_enabled</code> visibility</td></tr>
    <tr><td><code>_reset_cb</code></td>
        <td>+ closes panels, disables scroll, resets toggle states</td></tr>
    <tr><td><code>_build_layout</code></td>
        <td>+ panel toggles column, right panels column with histogram and metadata,
            hidden scroll_input in layout</td></tr>
    <tr><td><code>_create_widgets</code></td>
        <td>+ metadata toggle/PreText, histogram toggle/figure/sources,
            scroll toggle/input/CustomJS</td></tr>
</table>

<h2>All Callbacks by Category</h2>
<table>
    <tr><th>Category</th><th>Callbacks</th></tr>
    <tr><td>Navigation</td>
        <td><code>_name_cb, _db_dropdown_cb, _series_cb, _increment_cb,
            _decrement_cb, _series_slider_slice_cb, _sort_toggle_cb,
            _scroll_cb</code></td></tr>
    <tr><td>Animation</td>
        <td><code>_series_toggle_anim_cb, _animate_series, _refresh_rate_cb</code></td></tr>
    <tr><td>Image Adjust</td>
        <td><code>_gamma_cb, _window_cb, _wl_preset_cb, _wl_manual_cb</code></td></tr>
    <tr><td>Visual</td>
        <td><code>_lut_cb, _invert_cb, _zoom_lock_cb</code></td></tr>
    <tr><td>Panels</td>
        <td><code>_metadata_toggle_cb, _histogram_toggle_cb, _scroll_toggle_cb</code></td></tr>
    <tr><td>Tools</td>
        <td><code>_clip_reset_cb, _tap_callback</code></td></tr>
    <tr><td>Control</td>
        <td><code>_reset_cb, _stop_server</code></td></tr>
</table>
""",
    ))

    # ---- Data Flow (UPDATED) ----
    manual.pages.append(Page(
        filename="dataflow.html",
        title="Data Flow",
        nav_title="Data Flow",
        raw_content="""
<h2>Initialization Flow</h2>
<div class="flow-diagram">
    YAML &rarr; ViewerConfig &rarr; DicomViewer.__init__()<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    _find_images() &rarr; _prepare_images() &rarr; SeriesManager.categorize()<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    Create color_mapper, apply first W/L preset<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    _create_figures() [image + scatter + <strong>histogram</strong>]<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    _create_widgets() [all widgets + <strong>metadata panel + scroll bridge</strong>]<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    _build_layout() [with <strong>right panels column + hidden scroll_input</strong>]<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    curdoc().add_root()
</div>

<h2>Navigation Flow (any slice change)</h2>
<div class="flow-diagram">
    User action (button, slider, dropdown, <strong>mousewheel</strong>)<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    Update current_slice<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    _get_new_image() [prepare, display, W/L, zoom]<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    <strong>_update_panels()</strong><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; if metadata visible: <strong>_update_metadata()</strong><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x2514; if histogram visible: <strong>_update_histogram()</strong><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#x2514; <strong>_update_histogram_wl_lines()</strong>
</div>

<h2>Mousewheel Scroll Flow</h2>
<div class="flow-diagram">
    User scrolls mousewheel over fig_image<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    <strong>MouseWheel event</strong> (browser/JS)<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    <strong>scroll_js</strong> (CustomJS callback)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; Read scroll delta direction<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x2514; Write incremented counter to scroll_input.value<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    <strong>scroll_input</strong> on_change("value") fires<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    <strong>_scroll_cb</strong> (Python)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; Compute delta = new - old<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; Increment or decrement current_slice<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; _sync_slice_slider()<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; _get_new_image()<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x2514; _update_panels()
</div>

<h2>Histogram Update Flow</h2>
<div class="flow-diagram">
    Image or W/L change<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    _update_histogram()<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; Flatten processed_image, filter NaN/inf<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; np.histogram(flat, bins=256)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; Update histogram_source {top, left, right}<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; Compute min/max/mean/std for title<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x2514; _update_histogram_wl_lines()<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#x251C; low = color_mapper.low<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#x251C; high = color_mapper.high<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#x2514; Two vertical lines with NaN separator
</div>

<h2>Metadata Update Flow</h2>
<div class="flow-diagram">
    _update_metadata()<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    Iterate self.ds elements<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; Skip VR in METADATA_SKIP_VR (show byte count)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; Truncate values &gt; 80 chars<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x2514; Stop at METADATA_MAX_LINES<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    Join lines &rarr; metadata_pre.text
</div>

<h2>Reset Flow</h2>
<div class="flow-diagram">
    _reset_cb()<br>
    &nbsp;&nbsp;&#x251C; Reset gamma, window sliders<br>
    &nbsp;&nbsp;&#x251C; Reprocess image<br>
    &nbsp;&nbsp;&#x251C; _refresh_wl_presets()<br>
    &nbsp;&nbsp;&#x251C; Reset invert, LUT &rarr; _apply_current_lut()<br>
    &nbsp;&nbsp;&#x251C; Unlock zoom<br>
    &nbsp;&nbsp;&#x251C; <strong>Close metadata panel</strong><br>
    &nbsp;&nbsp;&#x251C; <strong>Close histogram panel</strong><br>
    &nbsp;&nbsp;&#x2514; <strong>Disable scroll</strong>
</div>
""",
    ))

    # ---- YAML Schema (unchanged) ----
    manual.pages.append(Page(
        filename="yaml_schema.html",
        title="YAML Configuration Schema",
        nav_title="YAML Schema",
        raw_content="""
<h2>Schema</h2>
<pre><code>debug: bool
gamma_def: float
window_def: float
starter_images: str
data_db:
  name: "/absolute/path/"</code></pre>
<h2>Rules</h2>
<table>
    <tr><th>Rule</th></tr>
    <tr><td><code>starter_images</code> must exist in <code>data_db</code></td></tr>
    <tr><td>All paths must exist and contain DICOM files</td></tr>
    <tr><td><code>gamma_def</code> and <code>window_def</code> must be non-negative</td></tr>
</table>
""",
    ))

    # ---- Extending (UPDATED with Phase 2 patterns) ----
    manual.pages.append(Page(
        filename="extending.html",
        title="Extending the Viewer",
        nav_title="Extending",
        raw_content="""
<h2>Adding a New Analysis Panel</h2>
<p>Follow the pattern established by the metadata and histogram panels:</p>
<pre><code># 1. In _create_widgets():
self.my_panel_toggle = Toggle(label="My Panel", ...)
self.my_panel_toggle.on_click(self._my_panel_toggle_cb)
self.my_panel_widget = PreText(text="", visible=False, ...)
# or: self.my_panel_fig = figure(visible=False, ...)

# 2. In _build_layout():
# Add to right_panels column

# 3. Callbacks:
def _my_panel_toggle_cb(self, active):
    self.my_panel_widget.visible = active
    if active:
        self._update_my_panel()

def _update_my_panel(self):
    # Compute and display content
    pass

# 4. In _update_panels():
if self.my_panel_widget.visible:
    self._update_my_panel()

# 5. In _reset_cb():
self.my_panel_widget.visible = False
self.my_panel_toggle.active = False</code></pre>

<h2>Adding a JS&rarr;Python Bridge</h2>
<p>The mousewheel scroll pattern is reusable for any browser event:</p>
<pre><code># 1. Hidden TextInput as bridge
self.bridge_input = TextInput(value="0", visible=False)
self.bridge_input.on_change("value", self._bridge_cb)

# 2. CustomJS that writes to the bridge
self.bridge_js = CustomJS(
    args=dict(bridge=self.bridge_input),
    code=\"\"\"
    // Process the browser event
    bridge.value = String(some_computed_value);
    \"\"\"
)

# 3. Attach to a Bokeh event
self.fig_image.js_on_event(SomeEvent, self.bridge_js)

# 4. Python callback reads the value
def _bridge_cb(self, attr, old, new):
    value = int(new)
    # Do something with it</code></pre>

<h2>Adding a Custom Color Map</h2>
<pre><code>COLOR_LUTS["MyMap"] = [f"#{r:02x}{g:02x}{b:02x}"
                       for r, g, b in my_color_data]</code></pre>

<h2>Adding Custom W/L Presets</h2>
<pre><code>if self.image_type == MODALITY_CT and not self.wl_presets:
    self.wl_presets = [
        {"center": 40, "width": 350, "name": "Soft Tissue"},
        {"center": 400, "width": 1500, "name": "Bone"},
    ]</code></pre>

<h2>Naming Conventions</h2>
<table>
    <tr><th>Pattern</th><th>Meaning</th></tr>
    <tr><td><code>_method</code></td><td>Private method</td></tr>
    <tr><td><code>_xxx_cb</code></td><td>Bokeh callback</td></tr>
    <tr><td><code>_update_xxx</code></td><td>Panel/data refresh method</td></tr>
    <tr><td><code>_xxx_toggle_cb</code></td><td>Panel visibility toggle</td></tr>
    <tr><td><code>UPPER_CASE</code></td><td>Module constant</td></tr>
</table>

<h2>Testing</h2>
<pre><code># Test histogram computation
import numpy as np
img = np.random.randint(0, 4096, (512, 512), dtype=np.uint16)
hist, edges = np.histogram(img.flatten(), bins=256)
assert len(hist) == 256
assert len(edges) == 257

# Test metadata skip logic
from dicom_viewer import METADATA_SKIP_VR
assert "OW" in METADATA_SKIP_VR  # pixel data
assert "DS" not in METADATA_SKIP_VR  # decimal string - should display</code></pre>
""",
    ))

    # ---- Limitations (UPDATED — final) ----
    manual.pages.append(Page(
        filename="limitations.html",
        title="Known Limitations",
        nav_title="Limitations",
        raw_content="""
<h2>Current Limitations</h2>
<table>
    <tr><th>#</th><th>Limitation</th><th>Status</th></tr>
    <tr><td>1</td><td>No DICOMDIR navigation</td><td>Open</td></tr>
    <tr><td>2</td><td>W/L presets from metadata only</td><td>Partially resolved v2.1</td></tr>
    <tr><td>3</td><td>Color images converted to grayscale</td><td>Open</td></tr>
    <tr><td>4</td><td>Clip/rotate cannot be undone</td><td>Open</td></tr>
    <tr><td>5</td><td>Zoom persistence</td><td>&#x2705; Resolved v2.2</td></tr>
    <tr><td>6</td><td>Sequential series metadata loading</td><td>Open</td></tr>
    <tr><td>7</td><td>No measurement tools</td><td>Future</td></tr>
    <tr><td>8</td><td>DICOM metadata panel</td><td>&#x2705; Resolved v2.3</td></tr>
    <tr><td>9</td><td>No keyboard shortcuts</td><td>Future</td></tr>
    <tr><td>10</td><td>Single-user per session</td><td>By design</td></tr>
    <tr><td>11</td><td>Animation refresh requires restart</td><td>Open</td></tr>
    <tr><td>12</td><td>Legacy window overlaps with W/L</td><td>Documented</td></tr>
    <tr><td>13</td><td>Mousewheel scrolling</td><td>&#x2705; Resolved v2.3</td></tr>
    <tr><td>14</td><td>No ROI statistics</td><td>Future</td></tr>
    <tr><td>15</td><td>No multiplanar reconstruction</td><td>Future</td></tr>
    <tr><td>16</td><td>Panels don't update during animation</td>
        <td>By design (performance)</td></tr>
    <tr><td>17</td><td>Mousewheel scroll conflicts with zoom wheel</td>
        <td>By design (toggle to switch)</td></tr>
</table>

<h2>Feature Roadmap</h2>
<table>
    <tr><th>Phase</th><th>Features</th><th>Status</th></tr>
    <tr><td>1</td><td>Color maps, invert, zoom lock, position sort</td>
        <td><strong>&#x2705; Complete (v2.2)</strong></td></tr>
    <tr><td>2</td><td>Metadata panel, histogram, mousewheel scroll</td>
        <td><strong>&#x2705; Complete (v2.3)</strong></td></tr>
    <tr><td>3</td><td>Measurement tools, ROI stats, annotations</td><td>Future</td></tr>
    <tr><td>4</td><td>MPR, multi-panel, cine controls, keyboard</td><td>Future</td></tr>
    <tr><td>5</td><td>3D rendering, PACS, fusion</td><td>Future</td></tr>
</table>

<h2>DICOM Compliance</h2>
<div class="warning">
    <p>Research and educational use only. Not a certified medical device.</p>
</div>
<ul>
    <li>Reads W/L tags (0028,1050/1051/1055)</li>
    <li>Reads PhotometricInterpretation</li>
    <li>Reads RescaleSlope/Intercept</li>
    <li>Reads ImagePositionPatient for sorting</li>
    <li>Reads ImageOrientationPatient for axis detection</li>
    <li>Metadata panel displays all readable tags</li>
    <li>Does not support VOI LUT Sequences</li>
    <li>Overlays and annotations not rendered</li>
</ul>
""",
    ))

    return manual

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def generate_all_docs(output_dir: str):
    """Generate both User Manual and Reference Manual."""
    print(f"Generating documentation in: {output_dir}")
    print()

    print("=== User Manual ===")
    user_manual = build_user_manual(output_dir)
    HTMLRenderer.write_manual(user_manual)
    print(f"  Total pages: {len(user_manual.pages)}")
    print()

    print("=== Reference Manual ===")
    ref_manual = build_reference_manual(output_dir)
    HTMLRenderer.write_manual(ref_manual)
    print(f"  Total pages: {len(ref_manual.pages)}")
    print()

    print("Done! Open the following files in your browser:")
    print(f"  User Manual:      {os.path.join(output_dir, 'user_manual', 'index.html')}")
    print(f"  Reference Manual: {os.path.join(output_dir, 'reference_manual', 'index.html')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate DICOM Viewer documentation as multi-page HTML"
    )
    parser.add_argument(
        "--output_dir",
        default="./docs",
        help="Output directory for documentation (default: ./docs)",
    )
    args = parser.parse_args()
    generate_all_docs(args.output_dir)

