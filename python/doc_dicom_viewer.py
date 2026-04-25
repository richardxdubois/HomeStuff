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
        {manual.title} &copy; 2024. Built with Python, Bokeh, and pydicom.
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
        subtitle="Version 2.1",
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
<p class="subtitle">User Manual &mdash; Version 2.1</p>

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
    <li><strong>DICOM Window/Level presets</strong> &mdash; auto-loaded from image metadata</li>
    <li>Manual Window Center / Width sliders for fine-tuned control</li>
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

<h2>What's New in Version 2.1</h2>
<div class="info">
    <ul>
        <li><strong>Window/Level Presets:</strong> The viewer now reads DICOM tags (0028,1050)
            and (0028,1051) to provide radiologist-defined brightness presets.</li>
        <li><strong>Manual W/L Sliders:</strong> Fine-tune Window Center and Width with
            dedicated sliders that auto-adjust their range to each image.</li>
        <li><strong>Automatic Preset Application:</strong> The first available W/L preset
            is applied automatically when loading an image.</li>
    </ul>
</div>
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
<pre><code># Create virtual environment
python -m venv dicom_env
source dicom_env/bin/activate  # macOS/Linux
# dicom_env\\Scripts\\activate  # Windows

# Install dependencies
pip install pydicom numpy scikit-image pyyaml bokeh
pip install "pylibjpeg[all]"</code></pre>

<h2>Verify Installation</h2>
<pre><code>python -c "import pydicom, bokeh, skimage, yaml; print('All packages OK')"</code></pre>

<div class="info">
    <strong>Note:</strong> The <code>pylibjpeg[all]</code> package is required for
    viewing JPEG and JPEG2000 compressed DICOM files, which are common in clinical data.
</div>

<h2>Required Packages Summary</h2>
<table>
    <tr><th>Package</th><th>Install Method</th><th>Purpose</th></tr>
    <tr><td><code>pydicom</code></td><td>conda / pip</td><td>DICOM file reading</td></tr>
    <tr><td><code>pylibjpeg[all]</code></td><td>pip</td><td>JPEG compressed DICOM support</td></tr>
    <tr><td><code>numpy</code></td><td>conda / pip</td><td>Array operations</td></tr>
    <tr><td><code>scikit-image</code></td><td>conda / pip</td><td>Image rotation</td></tr>
    <tr><td><code>bokeh</code></td><td>pip</td><td>Web UI framework</td></tr>
    <tr><td><code>pyyaml</code></td><td>conda / pip</td><td>Configuration parsing</td></tr>
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
<p>The viewer is configured via a YAML file. Create a file (e.g., <code>dicom_viewer.yaml</code>):</p>

<pre><code># dicom_viewer.yaml
debug: false
gamma_def: 1.0
window_def: 1.0
starter_images: "my_xrays"
data_db:
  my_xrays: "/path/to/xray/dicom/files/"
  chest_ct: "/path/to/ct/dicom/files/"
  brain_mri: "/path/to/mri/dicom/files/"
  abdominal_us: "/path/to/ultrasound/files/"</code></pre>

<h2>Configuration Fields</h2>
<table>
    <tr><th>Field</th><th>Type</th><th>Default</th><th>Description</th></tr>
    <tr><td><code>debug</code></td><td>boolean</td><td><code>false</code></td>
        <td>Enable verbose debug output to the terminal</td></tr>
    <tr><td><code>gamma_def</code></td><td>float</td><td><code>1.0</code></td>
        <td>Default gamma correction value. 1.0 = no correction. Higher values darken midtones.</td></tr>
    <tr><td><code>window_def</code></td><td>float</td><td><code>1.0</code></td>
        <td>Default brightness window multiplier (legacy control). 1.0 = full range.</td></tr>
    <tr><td><code>starter_images</code></td><td>string</td><td>&mdash;</td>
        <td>Key from <code>data_db</code> to load on startup</td></tr>
    <tr><td><code>data_db</code></td><td>dict</td><td>&mdash;</td>
        <td>Named datasets: maps a label to a directory path containing DICOM files</td></tr>
</table>

<h2>Directory Requirements</h2>
<div class="warning">
    <strong>Important:</strong>
    <ul>
        <li>Each directory in <code>data_db</code> should contain only DICOM files</li>
        <li>Hidden files (starting with <code>.</code>) are automatically excluded</li>
        <li>Paths must be absolute</li>
        <li>Paths should end with a trailing slash <code>/</code></li>
        <li>All DICOM files in a directory are assumed to belong to the same study</li>
    </ul>
</div>

<h2>Example Configurations</h2>

<h3>Minimal Configuration</h3>
<pre><code>debug: false
gamma_def: 1.0
window_def: 1.0
starter_images: "xrays"
data_db:
  xrays: "/Users/me/dicom_data/xrays/"</code></pre>

<h3>Multi-Dataset Configuration</h3>
<pre><code>debug: true
gamma_def: 1.5
window_def: 1.0
starter_images: "brain_mri"
data_db:
  hand_xray: "/Volumes/Data/DICOM/hand/"
  chest_xray: "/Volumes/Data/DICOM/chest/"
  brain_mri: "/Volumes/Data/DICOM/brain_mri/"
  knee_mri: "/Volumes/Data/DICOM/knee_mri/"
  abdominal_ct: "/Volumes/Data/DICOM/abd_ct/"
  cardiac_us: "/Volumes/Data/DICOM/cardiac_us/"</code></pre>
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
<p>After the server starts, open your browser to:</p>
<pre><code>localhost:5006/dicom_viewer</code></pre>

<div class="info">
    <strong>Tip:</strong> The Bokeh server prints the exact URL to the terminal when it starts.
    Look for a line like: <code>Bokeh app running at: ...</code>
</div>

<h2>Command-Line Options</h2>
<table>
    <tr><th>Option</th><th>Description</th></tr>
    <tr><td><code>--app_config PATH</code></td>
        <td>Path to the YAML configuration file. Defaults to
        <code>/Volumes/Data/Home/dicom_viewer.yaml</code></td></tr>
</table>

<h2>Running on a Specific Port</h2>
<pre><code>bokeh serve dicom_viewer.py --port 8080 --args --app_config "config.yaml"</code></pre>
<p>Then open <code>localhost:8080/dicom_viewer</code></p>

<h2>Stopping the Server</h2>
<ul>
    <li>Click the <strong>Exit</strong> button in the viewer interface</li>
    <li>Press <code>Ctrl+C</code> in the terminal where the server is running</li>
</ul>
""",
    ))

    # ---- Interface Guide (UPDATED with W/L controls) ----
    manual.pages.append(Page(
        filename="interface.html",
        title="User Interface Guide",
        nav_title="Interface Guide",
        raw_content="""
<h2>Layout Overview</h2>
<div class="card">
    <p>The interface is divided into three areas:</p>
    <ul>
        <li><strong>Top row:</strong> Control widgets (buttons, dropdowns, sliders)</li>
        <li><strong>Bottom-left:</strong> Main image display with color bar</li>
        <li><strong>Bottom-right:</strong> Series position scatter plot (CT/MRI only)</li>
    </ul>
</div>

<h2>Control Widgets</h2>
<table>
    <tr><th>Widget</th><th>Description</th></tr>
    <tr><td><strong>Exit</strong></td>
        <td>Shuts down the Bokeh server. Button turns light to confirm shutdown.</td></tr>
    <tr><td><strong>Reset</strong></td>
        <td>Restores gamma, window, and W/L settings to their defaults.
            If DICOM presets exist, the first preset is reapplied.</td></tr>
    <tr><td><strong>Pick Imaging</strong></td>
        <td>Dropdown to switch between datasets defined in your YAML config.</td></tr>
    <tr><td><strong>Mode</strong></td>
        <td>Shows auto-detected modality (XRay / CT / MRI / US). Read-only indicator.</td></tr>
    <tr><td><strong>Clip</strong></td>
        <td>Resets clip mode. After clicking, tap 4 corners on the image to clip and rotate.</td></tr>
    <tr><td><strong>Gamma</strong></td>
        <td>Slider (0&ndash;10). Controls gamma correction curve. Higher values darken midtones.</td></tr>
    <tr><td><strong>Window (legacy)</strong></td>
        <td>Slider (0&ndash;2). Simple brightness multiplier on the color map high value.</td></tr>
    <tr><td><strong>W/L Preset</strong></td>
        <td>Dropdown showing DICOM-defined window/level presets for the current image.
            Selecting a preset sets the Window Center and Width automatically.
            Shows "Manual" when using the sliders directly.</td></tr>
    <tr><td><strong>Window Center</strong></td>
        <td>Slider controlling the center of the displayed value range.
            Shifts which brightness values are visible. Range auto-adjusts per image.</td></tr>
    <tr><td><strong>Window Width</strong></td>
        <td>Slider controlling the width of the displayed value range.
            Narrower = higher contrast. Range auto-adjusts per image.</td></tr>
    <tr><td><strong>Pick Image</strong></td>
        <td>Dropdown to select a specific image file from the current dataset or series.</td></tr>
    <tr><td><strong>Pick Series</strong></td>
        <td>Dropdown to select a series (CT/MRI only).</td></tr>
    <tr><td><strong>Start/Stop Animation</strong></td>
        <td>Toggle to auto-play through slices.
            Green = ready to start. Red = click to stop.</td></tr>
    <tr><td><strong>Refresh (ms)</strong></td>
        <td>Animation speed in milliseconds. Type a new value and press Enter.</td></tr>
    <tr><td><strong>Slice Slider</strong></td>
        <td>Scrub through slices manually by dragging.</td></tr>
    <tr><td><strong>Increment / Decrement</strong></td>
        <td>Step forward or backward by exactly one slice.</td></tr>
</table>

<h2>Image Display Features</h2>
<ul>
    <li><strong>Hover:</strong> Move your mouse over the image to see pixel coordinates and intensity.</li>
    <li><strong>Zoom:</strong> Use the scroll wheel or Bokeh's toolbar tools.</li>
    <li><strong>Pan:</strong> Click and drag to pan.</li>
    <li><strong>Color bar:</strong> Shows the grayscale value mapping.
        The range updates when W/L settings change.</li>
    <li><strong>Title:</strong> Shows filename, patient name, procedure date, and protocol.</li>
</ul>

<h2>Understanding the Brightness Controls</h2>
<div class="card">
    <p>The viewer provides <strong>three layers</strong> of brightness control that work together:</p>
    <table>
        <tr><th>Control</th><th>What It Does</th><th>When to Use</th></tr>
        <tr>
            <td><strong>Gamma</strong></td>
            <td>Adjusts the brightness <em>curve shape</em>. Changes how
                midtone values are mapped relative to darks and lights.</td>
            <td>When you want to see more detail in dark or bright areas
                without changing the overall range.</td>
        </tr>
        <tr>
            <td><strong>Window (legacy)</strong></td>
            <td>Simple multiplier on the color map maximum.
                Scales the top of the brightness range up or down.</td>
            <td>Quick brightness adjustment. Good for images without
                DICOM W/L metadata.</td>
        </tr>
        <tr>
            <td><strong>W/L Center &amp; Width</strong></td>
            <td>Controls which <em>range of pixel values</em> maps to the
                display range. Center sets the midpoint; Width sets how
                much range is visible.</td>
            <td>Standard radiological viewing. Use presets for common
                tissue types (bone, soft tissue, lung, brain, etc.).</td>
        </tr>
    </table>
</div>

<div class="info">
    <strong>Tip:</strong> For the best results with CT and MRI images, use the
    <strong>W/L Preset</strong> dropdown first, then fine-tune with the
    Center/Width sliders. The legacy Window slider and Gamma are supplementary.
</div>

<h2>Series Position Plot</h2>
<p>
    For CT and MRI datasets, a scatter plot appears on the right showing the spatial position
    of each slice in the series:
</p>
<ul>
    <li><span style="color:black;font-weight:bold;">Black / Blue</span> &mdash; all slices in the series</li>
    <li><span style="color:red;font-weight:bold;">Red</span> &mdash; the currently displayed slice</li>
</ul>

<h2>Log Panel</h2>
<p>
    The log panel shows the last 10 actions including W/L preset changes,
    slider adjustments, image loads, and error messages.
</p>
""",
    ))

    # ---- Workflows (UPDATED with W/L workflows) ----
    manual.pages.append(Page(
        filename="workflows.html",
        title="Common Workflows",
        nav_title="Workflows",
        raw_content="""
<h2>Viewing an X-Ray</h2>
<ol>
    <li>Select your X-Ray dataset from <strong>Pick Imaging</strong>.</li>
    <li>Choose a specific image from <strong>Pick Image</strong>.</li>
    <li>If a W/L preset is available, it will be applied automatically.</li>
    <li>Adjust <strong>Gamma</strong> and <strong>Window Center/Width</strong> for optimal visibility.</li>
    <li>Hover over areas of interest to read pixel intensity values.</li>
</ol>

<h2>Browsing a CT/MRI Series</h2>
<ol>
    <li>Select your CT or MRI dataset from <strong>Pick Imaging</strong>.</li>
    <li>Choose a series from <strong>Pick Series</strong>.</li>
    <li>Use <strong>Increment/Decrement</strong> buttons or the <strong>Slice slider</strong> to browse.</li>
    <li>Your current W/L settings are preserved as you step through slices.</li>
    <li>Watch the position plot update to show your current spatial location.</li>
</ol>

<h2>Using Window/Level Presets</h2>
<div class="card">
    <h3>What Are W/L Presets?</h3>
    <p>DICOM images often contain one or more <strong>Window/Level presets</strong> embedded
    in their metadata. These are brightness settings chosen by the scanner manufacturer or
    radiologist to optimize viewing of specific tissue types.</p>

    <p>Common CT presets include:</p>
    <table>
        <tr><th>Preset Name</th><th>Typical Center</th><th>Typical Width</th><th>Best For</th></tr>
        <tr><td>Bone</td><td>400</td><td>1500</td><td>Skeletal structures</td></tr>
        <tr><td>Soft Tissue</td><td>40</td><td>350</td><td>Organs, muscles</td></tr>
        <tr><td>Lung</td><td>-600</td><td>1500</td><td>Lung parenchyma</td></tr>
        <tr><td>Brain</td><td>40</td><td>80</td><td>Brain tissue</td></tr>
        <tr><td>Abdomen</td><td>60</td><td>400</td><td>Abdominal organs</td></tr>
    </table>

    <h3>How to Use Presets</h3>
    <ol>
        <li>Load an image &mdash; the first available preset is applied automatically.</li>
        <li>Open the <strong>W/L Preset</strong> dropdown to see all available presets.
            Each shows its name and (Center, Width) values.</li>
        <li>Select a different preset to switch views (e.g., from "Soft Tissue" to "Bone").</li>
        <li>The Window Center and Width sliders update to match the preset.</li>
    </ol>

    <h3>Fine-Tuning</h3>
    <ol>
        <li>Start with a preset that's close to what you want.</li>
        <li>Drag the <strong>Window Center</strong> slider to shift the brightness midpoint.</li>
        <li>Drag the <strong>Window Width</strong> slider to increase or decrease contrast.</li>
        <li>The dropdown switches to "Manual" to indicate you've overridden the preset.</li>
    </ol>
</div>

<div class="info">
    <strong>Note:</strong> Not all DICOM images contain W/L presets. When no presets are
    available, the viewer automatically calculates initial W/L values from the image data range.
    The dropdown will show only "Manual" in this case.
</div>

<h2>Animating Through Slices</h2>
<ol>
    <li>Navigate to the desired series.</li>
    <li>Set your preferred W/L first &mdash; it will be preserved during animation.</li>
    <li>Optionally set the <strong>Refresh (ms)</strong> rate.</li>
    <li>Click <strong>Start Animation</strong>. The button turns red.</li>
    <li>Click again to stop.</li>
</ol>

<h2>Clipping and Rotating a Region</h2>
<ol>
    <li>Click the <strong>Clip</strong> button to enter clip mode.</li>
    <li>Click on <strong>four corners</strong> of the region of interest on the image.</li>
    <li>The image is automatically rotated and cropped.</li>
    <li>The log confirms each point and the final rotation angle.</li>
</ol>
<div class="warning">
    <strong>Note:</strong> The clip operation cannot be undone. Select a different image
    or dataset to restore the original view.
</div>

<h2>Switching Datasets</h2>
<ol>
    <li>Use <strong>Pick Imaging</strong> to select a different dataset.</li>
    <li>All controls reset automatically (gamma, window, W/L, slice position, animation).</li>
    <li>W/L presets update to reflect the new image's DICOM metadata.</li>
    <li>The modality indicator and series controls update automatically.</li>
</ol>

<h2>Adjusting Image Quality</h2>
<div class="card">
    <h3>Gamma Correction</h3>
    <ul>
        <li><strong>Gamma = 1.0:</strong> Linear (no correction)</li>
        <li><strong>Gamma &lt; 1.0:</strong> Brightens dark areas</li>
        <li><strong>Gamma &gt; 1.0:</strong> Darkens midtones, increases contrast</li>
    </ul>
    <p>MRI images default to gamma = 2.0.</p>

    <h3>Window Center / Width (Recommended for CT/MRI)</h3>
    <ul>
        <li><strong>Narrower Width:</strong> Higher contrast (fewer gray levels visible)</li>
        <li><strong>Wider Width:</strong> Lower contrast (more gray levels visible)</li>
        <li><strong>Higher Center:</strong> Biases display toward brighter structures</li>
        <li><strong>Lower Center:</strong> Biases display toward darker structures</li>
    </ul>

    <h3>Legacy Window</h3>
    <ul>
        <li><strong>Window = 1.0:</strong> Full dynamic range</li>
        <li><strong>Window &lt; 1.0:</strong> Brighter image</li>
        <li><strong>Window &gt; 1.0:</strong> Darker image</li>
    </ul>

    <p>Click <strong>Reset</strong> at any time to restore all settings to defaults.</p>
</div>
""",
    ))

    # ---- Troubleshooting (UPDATED with W/L items) ----
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
        <td>Try selecting a different <strong>W/L Preset</strong> from the dropdown.
            If none available, drag the <strong>Window Width</strong> slider wider and
            adjust <strong>Window Center</strong>. Click <strong>Reset</strong> to restore defaults.</td>
    </tr>
    <tr>
        <td>Image too dark</td>
        <td>Decrease <strong>Window Center</strong> or increase <strong>Window Width</strong>.
            For quick adjustment, try the legacy <strong>Window</strong> slider below 1.0.
            Lower <strong>Gamma</strong> below 1.0 to brighten dark areas.</td>
    </tr>
    <tr>
        <td>Image too bright / washed out</td>
        <td>Increase <strong>Window Center</strong> or decrease <strong>Window Width</strong>.
            Increase <strong>Gamma</strong> above 1.0 to darken midtones.</td>
    </tr>
    <tr>
        <td>W/L Preset dropdown shows only "Manual"</td>
        <td>The current DICOM image does not contain Window Center/Width metadata tags
            (0028,1050) and (0028,1051). The viewer falls back to auto-calculated values
            from the image data range. This is normal for some modalities.</td>
    </tr>
    <tr>
        <td>W/L presets don't look right</td>
        <td>The presets are read directly from DICOM metadata. Try adjusting the
            <strong>Gamma</strong> slider as well &mdash; gamma correction changes how
            values within the W/L window are displayed. A gamma of 1.0 gives the most
            faithful representation of the preset.</td>
    </tr>
    <tr>
        <td>Image appears inverted (negative)</td>
        <td>The viewer auto-detects MONOCHROME1 vs MONOCHROME2 photometric interpretation.
            If an image still appears inverted, the DICOM metadata may be incorrect.</td>
    </tr>
    <tr>
        <td>Series controls not visible</td>
        <td>Series controls only appear for CT, MRI, and Ultrasound modalities.</td>
    </tr>
    <tr>
        <td>Animation stutters or is slow</td>
        <td>Increase the <strong>Refresh (ms)</strong> value. 500-1000ms works well for large files.</td>
    </tr>
    <tr>
        <td>"Hit whitespace" error when clipping</td>
        <td>Your click landed outside the image data area. Click directly on visible image content.</td>
    </tr>
    <tr>
        <td>Server won't start</td>
        <td>Check YAML config path, verify directories exist, ensure DICOM files are present.</td>
    </tr>
    <tr>
        <td>"No module named pylibjpeg" error</td>
        <td>Run <code>pip install "pylibjpeg[all]"</code></td>
    </tr>
    <tr>
        <td>CT values look wrong</td>
        <td>The viewer applies RescaleSlope/Intercept for Hounsfield units.
            Use a <strong>W/L Preset</strong> or set Center to ~40, Width to ~350
            for soft tissue viewing.</td>
    </tr>
</table>

<h2>Enabling Debug Mode</h2>
<p>Set <code>debug: true</code> in your YAML configuration for verbose terminal output.</p>

<h2>Getting Help</h2>
<div class="info">
    <strong>Diagnostic checklist:</strong>
    <ol>
        <li>Enable <code>debug: true</code> in the YAML config</li>
        <li>Check the terminal for Python error messages</li>
        <li>Check the in-app Log panel for application messages</li>
        <li>Verify DICOM files open correctly in another viewer</li>
        <li>Try a different dataset to isolate the issue</li>
        <li>Check the Log panel for W/L preset messages when loading images</li>
    </ol>
</div>
""",
    ))

    return manual

# ---------------------------------------------------------------------------
# Reference Manual content definition (UPDATED with W/L documentation)
# ---------------------------------------------------------------------------
def build_reference_manual(output_dir: str) -> Manual:
    """Build the complete Developer Reference Manual content structure."""

    manual = Manual(
        title="DICOM Viewer Reference",
        subtitle="Developer API &amp; Architecture Guide &mdash; Version 2.1",
        output_dir=os.path.join(output_dir, "reference_manual"),
        accent_color="#7c3aed",
        icon="&#x1F527;",
    )

    # ---- Architecture (UPDATED) ----
    manual.pages.append(Page(
        filename="index.html",
        title="Architecture Overview",
        nav_title="Architecture",
        raw_content="""
<p class="subtitle">Developer API &amp; Architecture Guide &mdash; Version 2.1</p>

<h2>Module Structure</h2>
<table>
    <tr><th>Component</th><th>File Role</th><th>Responsibility</th></tr>
    <tr><td><code>ViewerConfig</code></td><td>Configuration</td>
        <td>Dataclass for loading and validating YAML configuration</td></tr>
    <tr><td><code>ImageProcessor</code></td><td>Processing</td>
        <td>Stateless image processing and DICOM metadata extraction (all static methods)</td></tr>
    <tr><td><code>SeriesManager</code></td><td>Data Management</td>
        <td>DICOM series metadata organization and spatial analysis</td></tr>
    <tr><td><code>DicomViewer</code></td><td>Application</td>
        <td>Main UI class: state management, Bokeh widgets, callbacks, W/L preset management</td></tr>
</table>

<h2>Design Principles</h2>
<ul>
    <li><strong>Separation of concerns:</strong> Image processing and metadata extraction are
        stateless and testable independently.</li>
    <li><strong>Named constants:</strong> All magic numbers are defined as module-level constants.</li>
    <li><strong>Specific exceptions:</strong> All <code>try/except</code> blocks catch specific types.</li>
    <li><strong>Python logging:</strong> Uses the standard <code>logging</code> module.</li>
    <li><strong>DICOM compliance:</strong> Reads standard tags for photometric interpretation,
        rescale, and window/level presets.</li>
    <li><strong>Private methods:</strong> <code>_prefix</code> convention. Callbacks use <code>_name_cb</code>.</li>
</ul>

<h2>Runtime Model</h2>
<div class="card">
    <p>The application runs as a <strong>Bokeh server app</strong>:</p>
    <ol>
        <li>Bokeh's Tornado server loads <code>dicom_viewer.py</code> as a module</li>
        <li>A single <code>DicomViewer</code> instance is created at module scope</li>
        <li>The constructor builds the UI layout and attaches it to <code>curdoc()</code></li>
        <li>DICOM W/L presets are extracted and applied on each image load</li>
        <li>User interactions trigger registered Python callbacks</li>
        <li>Callbacks update Bokeh model properties, which sync to the browser via WebSocket</li>
    </ol>
</div>
""",
    ))

    # ---- Dependencies (unchanged) ----
    manual.pages.append(Page(
        filename="dependencies.html",
        title="Dependencies",
        nav_title="Dependencies",
        raw_content="""
<h2>Required Packages</h2>
<table>
    <tr><th>Package</th><th>Min Version</th><th>Purpose</th></tr>
    <tr><td><code>pydicom</code></td><td>&ge; 2.0</td>
        <td>DICOM file reading, metadata access, MultiValue support</td></tr>
    <tr><td><code>pylibjpeg[all]</code></td><td>&ge; 1.0</td>
        <td>JPEG / JPEG2000 compressed DICOM transfer syntax support</td></tr>
    <tr><td><code>numpy</code></td><td>&ge; 1.20</td>
        <td>Array operations, image manipulation, linear algebra</td></tr>
    <tr><td><code>scikit-image</code></td><td>&ge; 0.18</td>
        <td>Image rotation via <code>skimage.transform.rotate</code></td></tr>
    <tr><td><code>bokeh</code></td><td>&ge; 3.0</td>
        <td>Interactive web UI: figures, widgets, server, WebSocket sync</td></tr>
    <tr><td><code>pyyaml</code></td><td>&ge; 5.0</td>
        <td>YAML configuration file parsing</td></tr>
    <tr><td><code>tornado</code></td><td>(via bokeh)</td>
        <td>Async web server; <code>IOLoop</code> for graceful shutdown</td></tr>
</table>

<h2>Import Map</h2>
<pre><code>import os, re, logging, argparse
import numpy as np
import pydicom, pylibjpeg
from tornado.ioloop import IOLoop
from skimage import transform
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row, layout
from bokeh.models import (
    ColumnDataSource, LinearColorMapper, ColorBar,
    HoverTool, Button, Slider, Div, TapTool,
    Select, RadioButtonGroup, Toggle, TextInput
)
from bokeh.palettes import Greys256</code></pre>
""",
    ))

    # ---- Constants (UPDATED with W/L constants) ----
    manual.pages.append(Page(
        filename="constants.html",
        title="Module Constants",
        nav_title="Constants",
        raw_content="""
<h2>Display Scale Constants</h2>
<table>
    <tr><th>Constant</th><th>Value</th><th>Description</th></tr>
    <tr><td><code>XRAY_SCALE</code></td><td>0.5</td><td>Display scale for X-Ray</td></tr>
    <tr><td><code>CT_SCALE</code></td><td>1.5</td><td>Display scale for CT</td></tr>
    <tr><td><code>MRI_SCALE</code></td><td>4.0</td><td>Display scale for MRI</td></tr>
    <tr><td><code>US_SCALE</code></td><td>1.5</td><td>Display scale for Ultrasound</td></tr>
</table>

<h2>Gamma Defaults</h2>
<table>
    <tr><th>Constant</th><th>Value</th><th>Description</th></tr>
    <tr><td><code>MRI_GAMMA</code></td><td>2.0</td><td>Default gamma for MRI</td></tr>
    <tr><td><code>US_GAMMA</code></td><td>2.0</td><td>Default gamma for Ultrasound</td></tr>
    <tr><td><code>DEFAULT_GAMMA</code></td><td>1.0</td><td>Fallback gamma</td></tr>
    <tr><td><code>DEFAULT_WINDOW</code></td><td>1.0</td><td>Fallback legacy window</td></tr>
</table>

<h2>UI Constants</h2>
<table>
    <tr><th>Constant</th><th>Value</th><th>Description</th></tr>
    <tr><td><code>GAMMA_SLIDER_MAX</code></td><td>10.0</td><td>Upper bound for gamma slider</td></tr>
    <tr><td><code>WINDOW_SLIDER_MAX</code></td><td>2.0</td><td>Upper bound for legacy window slider</td></tr>
    <tr><td><code>MAX_LOG_MESSAGES</code></td><td>10</td><td>Max visible log entries</td></tr>
    <tr><td><code>DEFAULT_REFRESH_MS</code></td><td>500.0</td><td>Animation refresh rate (ms)</td></tr>
</table>

<h2>Window/Level Constants</h2>
<table>
    <tr><th>Constant</th><th>Value</th><th>Description</th></tr>
    <tr><td><code>WL_CENTER_DEFAULT</code></td><td>2000.0</td>
        <td>Fallback center when no DICOM presets and no image data</td></tr>
    <tr><td><code>WL_WIDTH_DEFAULT</code></td><td>4000.0</td>
        <td>Fallback width when no DICOM presets and no image data</td></tr>
    <tr><td><code>WL_CENTER_SLIDER_MIN</code></td><td>-2000.0</td>
        <td>Minimum value for center slider (accommodates negative HU values)</td></tr>
    <tr><td><code>WL_CENTER_SLIDER_MAX</code></td><td>20000.0</td>
        <td>Default maximum for center slider (auto-adjusts to image data)</td></tr>
    <tr><td><code>WL_WIDTH_SLIDER_MIN</code></td><td>1.0</td>
        <td>Minimum window width (must be &gt; 0)</td></tr>
    <tr><td><code>WL_WIDTH_SLIDER_MAX</code></td><td>40000.0</td>
        <td>Default maximum for width slider (auto-adjusts to image data)</td></tr>
    <tr><td><code>WL_SLIDER_STEP</code></td><td>10.0</td>
        <td>Step size for W/L sliders</td></tr>
    <tr><td><code>WL_MANUAL_LABEL</code></td><td>"Manual"</td>
        <td>Label shown in preset dropdown when using manual sliders</td></tr>
</table>

<h2>Analysis Constants</h2>
<table>
    <tr><th>Constant</th><th>Value</th><th>Description</th></tr>
    <tr><td><code>NORMAL_THRESHOLD</code></td><td>0.95</td>
        <td>Dot-product threshold for imaging axis detection</td></tr>
    <tr><td><code>ANIMATION_SLICE_RESET</code></td><td>0</td>
        <td>Slice index to reset to on animation loop</td></tr>
</table>

<h2>Modality Strings</h2>
<table>
    <tr><th>Constant</th><th>Value</th></tr>
    <tr><td><code>MODALITY_XRAY</code></td><td>"X-Ray"</td></tr>
    <tr><td><code>MODALITY_CT</code></td><td>"CT"</td></tr>
    <tr><td><code>MODALITY_MRI</code></td><td>"MRI"</td></tr>
    <tr><td><code>MODALITY_US</code></td><td>"US"</td></tr>
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
<p>A <code>@dataclass</code> that holds application configuration loaded from YAML.</p>
<div class="method-sig">@dataclass<br>class ViewerConfig:</div>

<h2>Fields</h2>
<table class="attr-table">
    <tr><th>Field</th><th>Type</th><th>Default</th><th>Description</th></tr>
    <tr><td>debug</td><td>bool</td><td>False</td><td>Debug mode flag</td></tr>
    <tr><td>gamma_def</td><td>float</td><td>1.0</td><td>Default gamma correction</td></tr>
    <tr><td>window_def</td><td>float</td><td>1.0</td><td>Default legacy window multiplier</td></tr>
    <tr><td>starter_images</td><td>str</td><td>""</td><td>Initial dataset key</td></tr>
    <tr><td>data_db</td><td>dict</td><td>{}</td><td>Dataset name &rarr; directory mapping</td></tr>
</table>

<h2>Class Methods</h2>
<div class="method-sig">@classmethod<br>from_yaml(cls, path: str) &rarr; ViewerConfig</div>
<p>Load configuration from a YAML file. Applies defaults for missing fields.</p>
""",
    ))

    # ---- ImageProcessor (UPDATED with extract_wl_presets) ----
    manual.pages.append(Page(
        filename="imageprocessor.html",
        title="Class: ImageProcessor",
        nav_title="ImageProcessor",
        raw_content="""
<h2>Overview</h2>
<p>Stateless image processing utilities and DICOM metadata extraction.
All methods are <code>@staticmethod</code>.</p>

<h2>Image Processing Methods</h2>

<div class="method-sig">apply_photometric(image: ndarray, ds: Dataset) &rarr; ndarray</div>
<p>Checks <code>PhotometricInterpretation</code>. Inverts for MONOCHROME1, copies for MONOCHROME2.</p>

<div class="method-sig">apply_rescale(image: ndarray, ds: Dataset) &rarr; ndarray</div>
<p>Applies <code>RescaleSlope</code> and <code>RescaleIntercept</code>:
<code>output = pixel * slope + intercept</code>. Essential for CT Hounsfield units.</p>

<div class="method-sig">ensure_2d(image: ndarray) &rarr; ndarray</div>
<p>Handles multi-dimensional pixel arrays (RGB averaged to grayscale; multi-frame takes first frame).</p>

<div class="method-sig">perform_gamma(image: ndarray, gamma: float, original_dtype: dtype) &rarr; ndarray</div>
<p>Gamma correction: normalize &rarr; exponentiate &rarr; rescale to original dtype.</p>

<h2>Metadata Extraction Methods</h2>

<div class="method-sig">extract_wl_presets(ds: Dataset) &rarr; list[dict]</div>
<p>Extracts Window/Level presets from DICOM metadata. Returns a list of preset dictionaries.</p>

<h3>DICOM Tags Read</h3>
<table>
    <tr><th>Tag</th><th>Name</th><th>Purpose</th></tr>
    <tr><td>(0028,1050)</td><td>Window Center</td><td>Center of display value range</td></tr>
    <tr><td>(0028,1051)</td><td>Window Width</td><td>Width of display value range</td></tr>
    <tr><td>(0028,1055)</td><td>Window Center &amp; Width Explanation</td>
        <td>Human-readable preset name (e.g., "BONE", "SOFT_TISSUE")</td></tr>
</table>

<h3>Return Format</h3>
<pre><code>[
    {"center": 400.0, "width": 1500.0, "name": "BONE"},
    {"center": 40.0,  "width": 350.0,  "name": "SOFT_TISSUE"},
]</code></pre>

<h3>Edge Cases Handled</h3>
<ul>
    <li><strong>Single value vs. multi-value:</strong> Both <code>400</code> and
        <code>[400, 40]</code> (pydicom MultiValue) are handled.</li>
    <li><strong>Missing explanation tag:</strong> Falls back to "Preset 1", "Preset 2", etc.</li>
    <li><strong>Fewer names than presets:</strong> Remaining presets get auto-generated names.</li>
    <li><strong>No W/L tags at all:</strong> Returns empty list.</li>
</ul>

<h2>Geometry Methods</h2>

<div class="method-sig">rotated_rectangle_properties(corners: list) &rarr; tuple</div>
<p>Calculates rotation angle and bounding box from four corner points.</p>
<table>
    <tr><th>Returns</th><th>Description</th></tr>
    <tr><td>rotation_angle_degrees</td><td>Angle in degrees (-90 to +90)</td></tr>
    <tr><td>min_x, max_x</td><td>Horizontal bounds</td></tr>
    <tr><td>min_y, max_y</td><td>Vertical bounds</td></tr>
    <tr><td>center</td><td>Center point as ndarray</td></tr>
</table>
""",
    ))

    # ---- SeriesManager (unchanged) ----
    manual.pages.append(Page(
        filename="seriesmanager.html",
        title="Class: SeriesManager",
        nav_title="SeriesManager",
        raw_content="""
<h2>Overview</h2>
<p>Manages DICOM series metadata for multi-slice datasets. Reads metadata
using <code>stop_before_pixels=True</code> for performance.</p>

<h2>Instance Attributes</h2>
<table class="attr-table">
    <tr><th>Attribute</th><th>Type</th><th>Description</th></tr>
    <tr><td>series_map</td><td>dict</td>
        <td><code>{series_num: {filename: [instance, pos, dir, normal]}}</code></td></tr>
    <tr><td>series</td><td>list[str]</td><td>Sorted series numbers</td></tr>
    <tr><td>series_extrema</td><td>dict</td><td>Position ranges per series</td></tr>
    <tr><td>series_pos_index</td><td>int</td><td>Active axis (0=x, 1=y, 2=z)</td></tr>
</table>

<h2>Methods</h2>

<div class="method-sig">categorize(images_list, data_dir, debug=False, log_callback=None)</div>
<p>Scans all files, extracts series/instance/position/orientation metadata.</p>

<div class="method-sig">get_series_images(series_key: str) &rarr; list[str]</div>
<p>Returns naturally-sorted filenames for a series.</p>

<div class="method-sig">determine_axis(series_key, images) &rarr; int</div>
<p>Returns principal imaging axis (0, 1, or 2) from slice normal vectors.</p>
""",
    ))

    # ---- DicomViewer (UPDATED with W/L methods) ----
    manual.pages.append(Page(
        filename="dicomviewer.html",
        title="Class: DicomViewer",
        nav_title="DicomViewer",
        raw_content="""
<h2>Overview</h2>
<p>Main application class. Orchestrates UI, state, DICOM loading, and W/L preset management.</p>

<h2>Key Instance Attributes</h2>
<table class="attr-table">
    <tr><th>Attribute</th><th>Type</th><th>Description</th></tr>
    <tr><td>config</td><td>ViewerConfig</td><td>Application configuration</td></tr>
    <tr><td>img_proc</td><td>ImageProcessor</td><td>Image processing utilities</td></tr>
    <tr><td>series_mgr</td><td>SeriesManager</td><td>Series metadata manager</td></tr>
    <tr><td>ds</td><td>pydicom.Dataset</td><td>Current DICOM dataset</td></tr>
    <tr><td>processed_image</td><td>ndarray</td><td>Final display image (after gamma)</td></tr>
    <tr><td>original_dtype</td><td>numpy.dtype</td><td>Pixel data type before processing</td></tr>
    <tr><td>wl_presets</td><td>list[dict]</td>
        <td>W/L presets extracted from current image's DICOM metadata.
            Each dict has keys: <code>center</code>, <code>width</code>, <code>name</code></td></tr>
    <tr><td>image_type</td><td>str</td><td>Current modality string</td></tr>
    <tr><td>is_series</td><td>bool</td><td>Whether current data is multi-slice</td></tr>
    <tr><td>current_slice</td><td>int</td><td>Current slice index</td></tr>
    <tr><td>gamma</td><td>float</td><td>Active gamma value</td></tr>
    <tr><td>window</td><td>float</td><td>Active legacy window multiplier</td></tr>
    <tr><td>source</td><td>ColumnDataSource</td><td>Bokeh data source for image</td></tr>
    <tr><td>color_mapper</td><td>LinearColorMapper</td>
        <td>Controls displayed value range. <code>low</code> and <code>high</code>
            are set by W/L controls.</td></tr>
</table>

<h2>Initialization Methods</h2>
<table>
    <tr><th>Method</th><th>Description</th></tr>
    <tr><td><code>__init__()</code></td>
        <td>Parse args, load config, read initial DICOM, extract W/L presets, build UI</td></tr>
    <tr><td><code>_create_figures()</code></td><td>Create image + scatter plot figures</td></tr>
    <tr><td><code>_create_widgets()</code></td>
        <td>Create all widgets including W/L preset dropdown, center/width sliders</td></tr>
    <tr><td><code>_build_layout()</code></td><td>Assemble layout with W/L controls column</td></tr>
</table>

<h2>Image Processing Methods</h2>
<table>
    <tr><th>Method</th><th>Description</th></tr>
    <tr><td><code>_prepare_images()</code></td>
        <td>Full pipeline including W/L preset extraction via
            <code>ImageProcessor.extract_wl_presets()</code></td></tr>
    <tr><td><code>_size_figures()</code></td><td>Update figure dimensions</td></tr>
    <tr><td><code>_set_image_fig_title(name)</code></td><td>Set title from patient metadata</td></tr>
    <tr><td><code>_get_new_image()</code></td>
        <td>Load image at current_slice; preserves current W/L slider positions</td></tr>
</table>

<h2>Window/Level Methods</h2>
<table>
    <tr><th>Method</th><th>Description</th></tr>
    <tr><td><code>_apply_window_level(center, width)</code></td>
        <td>Sets <code>color_mapper.low</code> and <code>.high</code> from center &amp; width.
            <code>low = center - width/2</code>, <code>high = center + width/2</code></td></tr>
    <tr><td><code>_wl_preset_label(preset)</code></td>
        <td>Static method. Generates display label: <code>"name (C:nnn W:nnn)"</code></td></tr>
    <tr><td><code>_build_wl_preset_options()</code></td>
        <td>Builds options list for dropdown: <code>["Manual", "preset1 label", ...]</code></td></tr>
    <tr><td><code>_refresh_wl_presets()</code></td>
        <td>Called after every image change. Updates dropdown options, slider ranges,
            and applies first preset or falls back to image data range.
            Temporarily removes and re-adds callbacks to prevent cascading triggers.</td></tr>
</table>

<h2>Window/Level Callbacks</h2>
<table>
    <tr><th>Method</th><th>Trigger</th><th>Description</th></tr>
    <tr><td><code>_wl_preset_cb(a,o,n)</code></td><td>W/L Preset dropdown</td>
        <td>Applies selected preset. Updates center/width sliders without triggering
            their callbacks. If "Manual" selected, no action.</td></tr>
    <tr><td><code>_wl_manual_cb(a,o,n)</code></td><td>Center or Width slider</td>
        <td>Applies manual center/width values. Switches dropdown to "Manual"
            to indicate user override.</td></tr>
</table>

<h2>Other Callbacks</h2>
<table>
    <tr><th>Method</th><th>Trigger</th><th>Description</th></tr>
    <tr><td><code>_name_cb(a,o,n)</code></td><td>Image dropdown</td>
        <td>Load image + call <code>_refresh_wl_presets()</code></td></tr>
    <tr><td><code>_db_dropdown_cb(a,o,n)</code></td><td>Dataset dropdown</td>
        <td>Full reset + call <code>_refresh_wl_presets()</code></td></tr>
    <tr><td><code>_series_cb(a,o,n)</code></td><td>Series dropdown</td>
        <td>Switch series + call <code>_refresh_wl_presets()</code></td></tr>
    <tr><td><code>_gamma_cb(a,o,n)</code></td><td>Gamma slider</td>
        <td>Reprocess image; updates <code>max_bright</code></td></tr>
    <tr><td><code>_window_cb(a,o,n)</code></td><td>Legacy window slider</td>
        <td>Scale color mapper high value</td></tr>
    <tr><td><code>_refresh_rate_cb(a,o,n)</code></td><td>Refresh TextInput</td>
        <td>Update animation speed</td></tr>
    <tr><td><code>_reset_cb()</code></td><td>Reset button</td>
        <td>Restore gamma, window, and W/L to defaults. Reapplies first preset if available.</td></tr>
    <tr><td><code>_clip_reset_cb()</code></td><td>Clip button</td><td>Reset clip state</td></tr>
    <tr><td><code>_increment_cb()</code></td><td>Increment button</td><td>Next slice</td></tr>
    <tr><td><code>_decrement_cb()</code></td><td>Decrement button</td><td>Previous slice</td></tr>
    <tr><td><code>_series_slider_slice_cb</code></td><td>Slice slider</td><td>Jump to slice</td></tr>
    <tr><td><code>_series_toggle_anim_cb</code></td><td>Animation toggle</td><td>Start/stop</td></tr>
    <tr><td><code>_animate_series()</code></td><td>Periodic timer</td><td>Advance frame</td></tr>
    <tr><td><code>_tap_callback(event)</code></td><td>Mouse tap</td><td>Clip points</td></tr>
    <tr><td><code>_stop_server()</code></td><td>Exit button</td><td>Shutdown</td></tr>
</table>

<h2>Helper Methods</h2>
<table>
    <tr><th>Method</th><th>Description</th></tr>
    <tr><td><code>_find_images()</code></td><td>Scan directory, exclude hidden files, natural sort</td></tr>
    <tr><td><code>_sync_slice_slider()</code></td><td>Update slider without triggering callback</td></tr>
    <tr><td><code>_histogram_positions()</code></td><td>Populate position scatter data</td></tr>
    <tr><td><code>_series_scatter_pos(name)</code></td><td>Highlight current slice</td></tr>
    <tr><td><code>_modality_index()</code></td><td>Map modality to button index</td></tr>
    <tr><td><code>_update_visibility()</code></td><td>Show/hide series widgets</td></tr>
    <tr><td><code>_reset_adjustments()</code></td><td>Reset gamma/window sliders to defaults</td></tr>
    <tr><td><code>_log(message)</code></td><td>On-screen + Python logger</td></tr>
</table>
""",
    ))

    # ---- Data Flow (UPDATED with W/L flow) ----
    manual.pages.append(Page(
        filename="dataflow.html",
        title="Data Flow",
        nav_title="Data Flow",
        raw_content="""
<h2>Initialization Flow</h2>
<div class="flow-diagram">
    YAML Config File<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    ViewerConfig.from_yaml()<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    DicomViewer.__init__()<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    _find_images() <span class="arrow">&rarr;</span> self.images_list<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    _prepare_images()<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; pydicom.dcmread()<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; ImageProcessor.ensure_2d()<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; ImageProcessor.apply_rescale()<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; ImageProcessor.apply_photometric()<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; np.flipud()<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; ImageProcessor.perform_gamma()<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x2514; <strong>ImageProcessor.extract_wl_presets()</strong> <span class="arrow">&rarr;</span> self.wl_presets<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    Create color_mapper<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    <strong>_apply_window_level(first preset)</strong> [if presets available]<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    SeriesManager.categorize() &nbsp;[if is_series]<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    _create_figures() + _create_widgets() + _build_layout()<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    curdoc().add_root() <span class="arrow">&rarr;</span> Bokeh server running
</div>

<h2>Image Change Flow (any image switch)</h2>
<div class="flow-diagram">
    User action (dropdown, slider, button, animation)<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    Set self.path to new DICOM file<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    _prepare_images()<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; Full processing pipeline<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x2514; <strong>extract_wl_presets()</strong> <span class="arrow">&rarr;</span> self.wl_presets updated<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    self.source.data["image"] = [self.processed_image]<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    <strong>_refresh_wl_presets()</strong><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; Update dropdown options from new wl_presets<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; Update slider ranges from image data<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; Apply first preset OR auto-calculate from range<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x2514; <strong>_apply_window_level(center, width)</strong><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#x251C; color_mapper.low = center - width/2<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#x2514; color_mapper.high = center + width/2<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    Bokeh syncs to browser
</div>

<h2>W/L Preset Selection Flow</h2>
<div class="flow-diagram">
    User selects preset from dropdown<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    _wl_preset_cb(attr, old, new)<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    Find matching preset in self.wl_presets<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    Remove slider callbacks (prevent cascade)<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    Update wl_center_slider.value and wl_width_slider.value<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    Re-add slider callbacks<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    _apply_window_level(center, width)<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    color_mapper updates <span class="arrow">&rarr;</span> browser re-renders
</div>

<h2>Manual W/L Adjustment Flow</h2>
<div class="flow-diagram">
    User drags Center or Width slider<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    _wl_manual_cb(attr, old, new)<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    Read current values from both sliders<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    _apply_window_level(center, width)<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    Switch dropdown to "Manual" (remove/re-add callback)<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    color_mapper updates <span class="arrow">&rarr;</span> browser re-renders
</div>

<h2>Slice Navigation Flow (preserves W/L)</h2>
<div class="flow-diagram">
    User clicks Increment/Decrement or drags slice slider<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    _get_new_image()<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; _prepare_images() [new wl_presets extracted but not yet applied]<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; Update source.data<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x2514; <strong>_apply_window_level(current slider center, current slider width)</strong><br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    User's W/L settings preserved across slices
</div>
""",
    ))

    # ---- YAML Schema (unchanged) ----
    manual.pages.append(Page(
        filename="yaml_schema.html",
        title="YAML Configuration Schema",
        nav_title="YAML Schema",
        raw_content="""
<h2>Complete Schema</h2>
<pre><code># All fields are required unless marked optional

debug: bool              # Enable verbose logging
gamma_def: float         # Default gamma (typically 1.0-2.0)
window_def: float        # Default legacy window scale (typically 1.0)
starter_images: str      # Key into data_db for initial dataset
data_db:                 # Dictionary of named datasets
  name: "/absolute/path/"</code></pre>

<h2>Validation Rules</h2>
<table>
    <tr><th>Rule</th><th>Description</th></tr>
    <tr><td><code>starter_images</code> must exist in <code>data_db</code></td>
        <td>Initial dataset key must match</td></tr>
    <tr><td>All paths must exist</td><td>Directories must be accessible</td></tr>
    <tr><td>Directories must contain DICOM files</td><td>At least one readable file</td></tr>
    <tr><td><code>gamma_def</code> non-negative</td><td>Zero produces black; typical 0.5-3.0</td></tr>
    <tr><td><code>window_def</code> non-negative</td><td>Zero clips to black; typical 0.5-1.5</td></tr>
</table>
""",
    ))

    # ---- Extending (UPDATED with W/L extension example) ----
    manual.pages.append(Page(
        filename="extending.html",
        title="Extending the Viewer",
        nav_title="Extending",
        raw_content="""
<h2>Adding a New Modality</h2>
<ol>
    <li>Add constants: <code>MODALITY_PET = "PET"</code>, <code>PET_SCALE = 2.0</code></li>
    <li>Add detection in <code>_prepare_images()</code></li>
    <li>Update <code>_modality_index()</code> and <code>RadioButtonGroup</code> labels</li>
</ol>

<h2>Adding a New Image Processing Function</h2>
<pre><code>@staticmethod
def apply_edge_detection(image: np.ndarray) -> np.ndarray:
    from skimage import filters
    return filters.sobel(image)</code></pre>

<h2>Adding Custom W/L Presets</h2>
<p>You can add application-defined presets that supplement the DICOM metadata presets.
Modify <code>_prepare_images()</code> or <code>_refresh_wl_presets()</code>:</p>

<pre><code># Example: Add standard CT presets if the image is CT
# Add to end of _prepare_images() or beginning of _refresh_wl_presets()
if self.image_type == MODALITY_CT and not self.wl_presets:
    self.wl_presets = [
        {"center": 40.0,   "width": 350.0,  "name": "Soft Tissue"},
        {"center": 400.0,  "width": 1500.0, "name": "Bone"},
        {"center": -600.0, "width": 1500.0, "name": "Lung"},
        {"center": 40.0,   "width": 80.0,   "name": "Brain"},
    ]</code></pre>

<h2>Adding a DICOM Metadata Panel</h2>
<pre><code># In _create_widgets():
self.metadata_div = Div(text="", width=400, height=300, visible=False)
self.metadata_button = Toggle(label="Show Metadata", active=False)
self.metadata_button.on_click(self._metadata_toggle_cb)

def _metadata_toggle_cb(self, active):
    if active:
        info = []
        for elem in self.ds:
            if elem.VR not in ("OW", "OB"):
                info.append(f"{elem.tag} {elem.name}: {elem.value}")
        self.metadata_div.text = "&lt;br&gt;".join(info[:50])
    self.metadata_div.visible = active</code></pre>

<h2>Naming Conventions</h2>
<table>
    <tr><th>Pattern</th><th>Meaning</th><th>Example</th></tr>
    <tr><td><code>_method_name</code></td><td>Private method</td><td><code>_prepare_images</code></td></tr>
    <tr><td><code>_xxx_cb</code></td><td>Bokeh callback</td><td><code>_wl_preset_cb</code></td></tr>
    <tr><td><code>_async_xxx</code></td><td>Async method</td><td><code>_async_update_log</code></td></tr>
    <tr><td><code>UPPER_CASE</code></td><td>Module constant</td><td><code>WL_CENTER_DEFAULT</code></td></tr>
</table>

<h2>Testing Strategy</h2>
<div class="card">
<pre><code>import numpy as np
import pydicom
from dicom_viewer import ImageProcessor

def test_gamma_identity():
    img = np.array([[100, 200], [50, 150]], dtype=np.uint16)
    result = ImageProcessor.perform_gamma(img, 1.0, np.uint16)
    assert result.shape == img.shape

def test_extract_wl_presets_empty():
    ds = pydicom.Dataset()
    presets = ImageProcessor.extract_wl_presets(ds)
    assert presets == []

def test_extract_wl_presets_single():
    ds = pydicom.Dataset()
    ds.add_new(0x00281050, 'DS', '400')
    ds.add_new(0x00281051, 'DS', '1500')
    ds.add_new(0x00281055, 'LO', 'BONE')
    presets = ImageProcessor.extract_wl_presets(ds)
    assert len(presets) == 1
    assert presets[0]["center"] == 400.0
    assert presets[0]["name"] == "BONE"

def test_extract_wl_presets_multi():
    ds = pydicom.Dataset()
    ds.add_new(0x00281050, 'DS', ['400', '40'])
    ds.add_new(0x00281051, 'DS', ['1500', '350'])
    ds.add_new(0x00281055, 'LO', ['BONE', 'SOFT'])
    presets = ImageProcessor.extract_wl_presets(ds)
    assert len(presets) == 2</code></pre>
</div>
""",
    ))

    # ---- Limitations (UPDATED) ----
    manual.pages.append(Page(
        filename="limitations.html",
        title="Known Limitations",
        nav_title="Limitations",
        raw_content="""
<h2>Current Limitations</h2>
<table>
    <tr><th>#</th><th>Limitation</th><th>Impact</th><th>Status</th></tr>
    <tr><td>1</td>
        <td>No DICOMDIR navigation</td>
        <td>Must point to flat directories</td>
        <td>Open</td></tr>
    <tr><td>2</td>
        <td>W/L presets from metadata only</td>
        <td>If DICOM lacks tags (0028,1050/1051), no presets are shown.
            Auto-fallback uses image data range.</td>
        <td><em>Partially resolved in v2.1</em></td></tr>
    <tr><td>3</td>
        <td>Color images converted to grayscale</td>
        <td>RGB, YBR_FULL images lose color</td>
        <td>Open</td></tr>
    <tr><td>4</td>
        <td>Clip/rotate cannot be undone</td>
        <td>Must reload image</td>
        <td>Open</td></tr>
    <tr><td>5</td>
        <td>No zoom/pan persistence</td>
        <td>View resets on image switch</td>
        <td>Open</td></tr>
    <tr><td>6</td>
        <td>Sequential series metadata loading</td>
        <td>Slow for large datasets</td>
        <td>Open</td></tr>
    <tr><td>7</td>
        <td>No measurement tools</td>
        <td>Cannot measure distance/area</td>
        <td>Open</td></tr>
    <tr><td>8</td>
        <td>No DICOM metadata panel</td>
        <td>Cannot view tags in UI</td>
        <td>Open (see <a href="extending.html">Extending</a>)</td></tr>
    <tr><td>9</td>
        <td>No keyboard shortcuts</td>
        <td>Mouse-only interaction</td>
        <td>Open</td></tr>
    <tr><td>10</td>
        <td>Single-user per session</td>
        <td>By design (Bokeh)</td>
        <td>N/A</td></tr>
    <tr><td>11</td>
        <td>Animation refresh change requires restart</td>
        <td>Must stop and restart animation</td>
        <td>Open</td></tr>
    <tr><td>12</td>
        <td>Legacy window slider overlaps with W/L</td>
        <td>Both affect color_mapper; can produce confusing interaction
            if used simultaneously</td>
        <td>Documented; legacy slider may be removed in future version</td></tr>
</table>

<h2>DICOM Compliance Notes</h2>
<div class="warning">
    <p>This viewer is for <strong>research and educational use only</strong>.
    It is <strong>not</strong> a certified medical device.</p>
</div>
<ul>
    <li>Reads standard W/L tags (0028,1050), (0028,1051), (0028,1055)</li>
    <li>Reads PhotometricInterpretation for correct grayscale handling</li>
    <li>Reads RescaleSlope/Intercept for CT Hounsfield units</li>
    <li>Does not support VOI LUT Sequences (more complex W/L definitions)</li>
    <li>Overlay and annotation layers are not rendered</li>
    <li>Structured Reports (SR) are not parsed</li>
    <li>Patient orientation labels are not displayed</li>
</ul>
""",
    ))

    return manual

# ---------------------------------------------------------------------------
# Reference Manual content definition
# ---------------------------------------------------------------------------
def build_reference_manual(output_dir: str) -> Manual:
    """Build the complete Developer Reference Manual content structure."""

    manual = Manual(
        title="DICOM Viewer Reference",
        subtitle="Developer API &amp; Architecture Guide &mdash; Version 2.0",
        output_dir=os.path.join(output_dir, "reference_manual"),
        accent_color="#7c3aed",
        icon="&#x1F527;",
    )

    # ---- Index / Architecture ----
    manual.pages.append(Page(
        filename="index.html",
        title="Architecture Overview",
        nav_title="Architecture",
        raw_content="""
<p class="subtitle">Developer API &amp; Architecture Guide &mdash; Version 2.0</p>

<h2>Module Structure</h2>
<p>The application follows a modular architecture with four principal components:</p>

<table>
    <tr><th>Component</th><th>File Role</th><th>Responsibility</th></tr>
    <tr><td><code>ViewerConfig</code></td><td>Configuration</td>
        <td>Dataclass for loading and validating YAML configuration</td></tr>
    <tr><td><code>ImageProcessor</code></td><td>Processing</td>
        <td>Stateless image processing utilities (all static methods)</td></tr>
    <tr><td><code>SeriesManager</code></td><td>Data Management</td>
        <td>DICOM series metadata organization and spatial analysis</td></tr>
    <tr><td><code>DicomViewer</code></td><td>Application</td>
        <td>Main UI class: state management, Bokeh widgets, callbacks</td></tr>
</table>

<h2>Design Principles</h2>
<ul>
    <li><strong>Separation of concerns:</strong> Image processing is stateless and testable
        independently of the UI.</li>
    <li><strong>Named constants:</strong> All magic numbers are defined as module-level constants.</li>
    <li><strong>Specific exceptions:</strong> All <code>try/except</code> blocks catch specific
        exception types.</li>
    <li><strong>Python logging:</strong> Uses the standard <code>logging</code> module alongside
        the in-app log panel.</li>
    <li><strong>Path handling:</strong> Uses <code>os.path.join()</code> and <code>pathlib.Path</code>
        consistently.</li>
    <li><strong>Private methods:</strong> Internal methods use the <code>_prefix</code> convention.
        Callbacks use <code>_name_cb</code> suffix.</li>
</ul>

<h2>Runtime Model</h2>
<div class="card">
    <p>The application runs as a <strong>Bokeh server app</strong>:</p>
    <ol>
        <li>Bokeh's Tornado server loads <code>dicom_viewer.py</code> as a module</li>
        <li>A single <code>DicomViewer</code> instance is created at module scope</li>
        <li>The constructor builds the UI layout and attaches it to <code>curdoc()</code></li>
        <li>User interactions trigger registered Python callbacks</li>
        <li>Callbacks update Bokeh model properties, which sync to the browser via WebSocket</li>
    </ol>
</div>
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
    <tr><td><code>pydicom</code></td><td>&ge; 2.0</td>
        <td>DICOM file reading and metadata access</td></tr>
    <tr><td><code>pylibjpeg[all]</code></td><td>&ge; 1.0</td>
        <td>JPEG / JPEG2000 compressed DICOM transfer syntax support</td></tr>
    <tr><td><code>numpy</code></td><td>&ge; 1.20</td>
        <td>Array operations, image manipulation, linear algebra</td></tr>
    <tr><td><code>scikit-image</code></td><td>&ge; 0.18</td>
        <td>Image rotation via <code>skimage.transform.rotate</code></td></tr>
    <tr><td><code>bokeh</code></td><td>&ge; 3.0</td>
        <td>Interactive web UI: figures, widgets, server, WebSocket sync</td></tr>
    <tr><td><code>pyyaml</code></td><td>&ge; 5.0</td>
        <td>YAML configuration file parsing</td></tr>
    <tr><td><code>tornado</code></td><td>(via bokeh)</td>
        <td>Async web server; <code>IOLoop</code> for graceful shutdown</td></tr>
</table>

<h2>Import Map</h2>
<pre><code>import os, re, logging, argparse
import numpy as np
import pydicom, pylibjpeg
from tornado.ioloop import IOLoop
from skimage import transform
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row, layout
from bokeh.models import (
    ColumnDataSource, LinearColorMapper, ColorBar,
    HoverTool, Button, Slider, Div, TapTool,
    Select, RadioButtonGroup, Toggle, TextInput
)
from bokeh.palettes import Greys256</code></pre>
""",
    ))

    # ---- Constants ----
    manual.pages.append(Page(
        filename="constants.html",
        title="Module Constants",
        nav_title="Constants",
        raw_content="""
<h2>Display Scale Constants</h2>
<table>
    <tr><th>Constant</th><th>Value</th><th>Description</th></tr>
    <tr><td><code>XRAY_SCALE</code></td><td>0.5</td><td>Display scale multiplier for X-Ray images</td></tr>
    <tr><td><code>CT_SCALE</code></td><td>1.5</td><td>Display scale multiplier for CT images</td></tr>
    <tr><td><code>MRI_SCALE</code></td><td>4.0</td><td>Display scale multiplier for MRI images</td></tr>
    <tr><td><code>US_SCALE</code></td><td>1.5</td><td>Display scale multiplier for Ultrasound images</td></tr>
</table>

<h2>Gamma Defaults</h2>
<table>
    <tr><th>Constant</th><th>Value</th><th>Description</th></tr>
    <tr><td><code>MRI_GAMMA</code></td><td>2.0</td><td>Default gamma for MRI modality</td></tr>
    <tr><td><code>US_GAMMA</code></td><td>2.0</td><td>Default gamma for Ultrasound modality</td></tr>
    <tr><td><code>DEFAULT_GAMMA</code></td><td>1.0</td><td>Fallback gamma value</td></tr>
    <tr><td><code>DEFAULT_WINDOW</code></td><td>1.0</td><td>Fallback window value</td></tr>
</table>

<h2>UI Constants</h2>
<table>
    <tr><th>Constant</th><th>Value</th><th>Description</th></tr>
    <tr><td><code>GAMMA_SLIDER_MAX</code></td><td>10.0</td><td>Upper bound for gamma slider</td></tr>
    <tr><td><code>WINDOW_SLIDER_MAX</code></td><td>2.0</td><td>Upper bound for window slider</td></tr>
    <tr><td><code>MAX_LOG_MESSAGES</code></td><td>10</td><td>Maximum visible entries in log panel</td></tr>
    <tr><td><code>DEFAULT_REFRESH_MS</code></td><td>500.0</td><td>Default animation refresh rate (ms)</td></tr>
</table>

<h2>Analysis Constants</h2>
<table>
    <tr><th>Constant</th><th>Value</th><th>Description</th></tr>
    <tr><td><code>NORMAL_THRESHOLD</code></td><td>0.95</td>
        <td>Dot-product threshold for determining imaging axis from slice normals</td></tr>
    <tr><td><code>ANIMATION_SLICE_RESET</code></td><td>0</td>
        <td>Slice index to reset to when animation loops</td></tr>
</table>

<h2>Modality Strings</h2>
<table>
    <tr><th>Constant</th><th>Value</th><th>Used For</th></tr>
    <tr><td><code>MODALITY_XRAY</code></td><td>"X-Ray"</td><td>SOP class name matching</td></tr>
    <tr><td><code>MODALITY_CT</code></td><td>"CT"</td><td>SOP class name matching</td></tr>
    <tr><td><code>MODALITY_MRI</code></td><td>"MRI"</td><td>SOP class name matching</td></tr>
    <tr><td><code>MODALITY_US</code></td><td>"US"</td><td>Fallback modality</td></tr>
</table>
""",
    ))

    # ---- ViewerConfig ----
    manual.pages.append(Page(
        filename="viewerconfig.html",
        title="Class: ViewerConfig",
        nav_title="ViewerConfig",
        raw_content="""
<h2>Overview</h2>
<p>A <code>@dataclass</code> that holds application configuration loaded from YAML.</p>
<div class="method-sig">@dataclass<br>class ViewerConfig:</div>

<h2>Fields</h2>
<table class="attr-table">
    <tr><th>Field</th><th>Type</th><th>Default</th><th>Description</th></tr>
    <tr><td>debug</td><td>bool</td><td>False</td><td>Debug mode flag</td></tr>
    <tr><td>gamma_def</td><td>float</td><td>1.0</td><td>Default gamma correction value</td></tr>
    <tr><td>window_def</td><td>float</td><td>1.0</td><td>Default window multiplier</td></tr>
    <tr><td>starter_images</td><td>str</td><td>""</td><td>Initial dataset key from data_db</td></tr>
    <tr><td>data_db</td><td>dict</td><td>{}</td><td>Dataset name &rarr; directory path mapping</td></tr>
</table>

<h2>Class Methods</h2>
<div class="method-sig">@classmethod<br>from_yaml(cls, path: str) &rarr; ViewerConfig</div>
<p>Load configuration from a YAML file and return a new <code>ViewerConfig</code> instance.
Applies defaults for any missing fields.</p>

<h3>Parameters</h3>
<table>
    <tr><th>Parameter</th><th>Type</th><th>Description</th></tr>
    <tr><td>path</td><td>str</td><td>Absolute path to the YAML configuration file</td></tr>
</table>

<h3>Returns</h3>
<p>A new <code>ViewerConfig</code> instance populated from the YAML data.</p>

<h3>Raises</h3>
<table>
    <tr><th>Exception</th><th>When</th></tr>
    <tr><td><code>FileNotFoundError</code></td><td>YAML file does not exist</td></tr>
    <tr><td><code>yaml.YAMLError</code></td><td>YAML file is malformed</td></tr>
</table>
""",
    ))

    # ---- ImageProcessor ----
    manual.pages.append(Page(
        filename="imageprocessor.html",
        title="Class: ImageProcessor",
        nav_title="ImageProcessor",
        raw_content="""
<h2>Overview</h2>
<p>Stateless image processing utilities for DICOM images.
All methods are <code>@staticmethod</code> — no instance required.</p>

<h2>Methods</h2>

<div class="method-sig">apply_photometric(image: ndarray, ds: Dataset) &rarr; ndarray</div>
<p>Checks the DICOM <code>PhotometricInterpretation</code> tag. Inverts grayscale for
<code>MONOCHROME1</code> images (where high pixel values = dark). Returns a copy for
<code>MONOCHROME2</code> (high values = bright).</p>
<table>
    <tr><th>Parameter</th><th>Type</th><th>Description</th></tr>
    <tr><td>image</td><td>ndarray</td><td>2-D pixel array</td></tr>
    <tr><td>ds</td><td>pydicom.Dataset</td><td>DICOM dataset with metadata</td></tr>
</table>

<div class="method-sig">apply_rescale(image: ndarray, ds: Dataset) &rarr; ndarray</div>
<p>Applies DICOM <code>RescaleSlope</code> and <code>RescaleIntercept</code> if present.
Essential for CT Hounsfield unit conversion: <code>HU = pixel * slope + intercept</code>.</p>

<div class="method-sig">ensure_2d(image: ndarray) &rarr; ndarray</div>
<p>Handles multi-dimensional pixel arrays:</p>
<ul>
    <li><strong>RGB/RGBA</strong> (shape [H, W, 3] or [H, W, 4]): averaged to grayscale</li>
    <li><strong>Multi-frame</strong> (shape [N, H, W]): first frame extracted</li>
    <li><strong>2-D</strong>: returned unchanged</li>
</ul>

<div class="method-sig">perform_gamma(image: ndarray, gamma: float, original_dtype: dtype) &rarr; ndarray</div>
<p>Applies gamma correction:</p>
<ol>
    <li>Normalizes pixel values to [0, 1]</li>
    <li>Applies <code>pixel = pixel ** gamma</code></li>
    <li>Scales back to the original data type range</li>
</ol>
<table>
    <tr><th>Parameter</th><th>Type</th><th>Description</th></tr>
    <tr><td>image</td><td>ndarray</td><td>2-D pixel array</td></tr>
    <tr><td>gamma</td><td>float</td><td>Gamma exponent (&gt; 1 darkens midtones, &lt; 1 brightens)</td></tr>
    <tr><td>original_dtype</td><td>numpy.dtype</td><td>Original pixel data type for rescaling</td></tr>
</table>

<div class="method-sig">apply_window_level(image: ndarray, window_center: float,
                     window_width: float) &rarr; ndarray</div>
<p>Standard DICOM window/level transform. Returns values clipped to [0, 1].
<em>Available for future integration with DICOM metadata presets.</em></p>

<div class="method-sig">rotated_rectangle_properties(corners: list) &rarr; tuple</div>
<p>Calculates rotation angle and bounding box from four corner points defined
by user taps on the image.</p>
<table>
    <tr><th>Parameter</th><th>Type</th><th>Description</th></tr>
    <tr><td>corners</td><td>list of [x, y]</td><td>Four corner coordinate pairs</td></tr>
</table>
<table>
    <tr><th>Returns (tuple element)</th><th>Type</th><th>Description</th></tr>
    <tr><td>rotation_angle_degrees</td><td>float</td><td>Angle in degrees (-90 to +90)</td></tr>
    <tr><td>min_x</td><td>float</td><td>Left bound of rotated bounding box</td></tr>
    <tr><td>max_x</td><td>float</td><td>Right bound</td></tr>
    <tr><td>min_y</td><td>float</td><td>Bottom bound</td></tr>
    <tr><td>max_y</td><td>float</td><td>Top bound</td></tr>
    <tr><td>center</td><td>ndarray</td><td>Center point [x, y]</td></tr>
</table>

<h3>Algorithm</h3>
<ol>
    <li>Compute centroid of four corners</li>
    <li>Find the pair of corners with maximum distance (longest side)</li>
    <li>Compute angle of that side via <code>arctan2</code></li>
    <li>Construct rotation matrix and rotate all corners around centroid</li>
    <li>Extract axis-aligned bounding box from rotated corners</li>
</ol>
""",
    ))

    # ---- SeriesManager ----
    manual.pages.append(Page(
        filename="seriesmanager.html",
        title="Class: SeriesManager",
        nav_title="SeriesManager",
        raw_content="""
<h2>Overview</h2>
<p>Manages DICOM series metadata for multi-slice datasets (CT, MRI, Ultrasound).
Reads metadata without loading pixel data for performance.</p>

<h2>Instance Attributes</h2>
<table class="attr-table">
    <tr><th>Attribute</th><th>Type</th><th>Description</th></tr>
    <tr><td>series_map</td><td>dict</td>
        <td>Nested dict: <code>{series_num: {filename: [instance, pos, dir, normal]}}</code></td></tr>
    <tr><td>series</td><td>list[str]</td><td>Sorted list of series numbers</td></tr>
    <tr><td>series_extrema</td><td>dict</td>
        <td>Position ranges per series: <code>{series: [min_x, max_x, ...]}</code></td></tr>
    <tr><td>series_pos_index</td><td>int</td>
        <td>Active spatial axis: 0=x, 1=y, 2=z</td></tr>
</table>

<h2>Methods</h2>

<div class="method-sig">categorize(images_list, data_dir, debug=False, log_callback=None) &rarr; None</div>
<p>Reads DICOM metadata from all files using <code>stop_before_pixels=True</code> for performance.
Populates <code>series_map</code>, <code>series</code>, and <code>series_extrema</code>.</p>
<table>
    <tr><th>Parameter</th><th>Type</th><th>Description</th></tr>
    <tr><td>images_list</td><td>list[str]</td><td>Filenames to scan</td></tr>
    <tr><td>data_dir</td><td>str</td><td>Directory containing the DICOM files</td></tr>
    <tr><td>debug</td><td>bool</td><td>Enable debug logging for missing metadata</td></tr>
    <tr><td>log_callback</td><td>callable</td><td>Function accepting a string message for UI logging</td></tr>
</table>

<h3>DICOM Tags Read</h3>
<table>
    <tr><th>Tag</th><th>Name</th><th>Purpose</th></tr>
    <tr><td>(0020,0011)</td><td>Series Number</td><td>Group images into series</td></tr>
    <tr><td>(0020,0013)</td><td>Instance Number</td><td>Order within series</td></tr>
    <tr><td>(0020,0032)</td><td>Image Position (Patient)</td><td>3D spatial position [x,y,z]</td></tr>
    <tr><td>(0020,0037)</td><td>Image Orientation (Patient)</td><td>Row/column direction cosines</td></tr>
</table>

<div class="method-sig">get_series_images(series_key: str) &rarr; list[str]</div>
<p>Returns a naturally-sorted list of filenames belonging to the specified series.</p>

<div class="method-sig">determine_axis(series_key: str, images: list) &rarr; int</div>
<p>Determines the principal imaging axis by computing the slice normal vector
(cross product of row and column directions) and comparing against unit vectors
using a dot-product threshold of <code>NORMAL_THRESHOLD</code> (0.95).</p>
<table>
    <tr><th>Return Value</th><th>Meaning</th></tr>
    <tr><td>0</td><td>Sagittal (x-axis)</td></tr>
    <tr><td>1</td><td>Coronal (y-axis)</td></tr>
    <tr><td>2</td><td>Axial (z-axis) — default</td></tr>
</table>

<h2>Private Methods</h2>
<div class="method-sig">_get_position_range(series: str) &rarr; list</div>
<p>Returns <code>[min_x, max_x, min_y, max_y, min_z, max_z]</code> for all slices in a series.</p>

<div class="method-sig">@staticmethod<br>_key_func(filename: str) &rarr; tuple</div>
<p>Natural sort key: splits filename into <code>(alpha_prefix, numeric_suffix)</code>.
Enables sorting like <code>img2</code> before <code>img10</code>.</p>
""",
    ))

    # ---- DicomViewer ----
    manual.pages.append(Page(
        filename="dicomviewer.html",
        title="Class: DicomViewer",
        nav_title="DicomViewer",
        raw_content="""
<h2>Overview</h2>
<p>Main application class. Orchestrates the UI, application state, DICOM loading,
and Bokeh integration. A single instance is created at module scope.</p>

<h2>Key Instance Attributes</h2>
<table class="attr-table">
    <tr><th>Attribute</th><th>Type</th><th>Description</th></tr>
    <tr><td>config</td><td>ViewerConfig</td><td>Application configuration</td></tr>
    <tr><td>img_proc</td><td>ImageProcessor</td><td>Image processing utilities</td></tr>
    <tr><td>series_mgr</td><td>SeriesManager</td><td>Series metadata manager</td></tr>
    <tr><td>ds</td><td>pydicom.Dataset</td><td>Current DICOM dataset</td></tr>
    <tr><td>dicom_image</td><td>ndarray</td><td>Raw pixel array from DICOM</td></tr>
    <tr><td>clipped_image</td><td>ndarray</td><td>After photometric + flip</td></tr>
    <tr><td>processed_image</td><td>ndarray</td><td>Final display image (after gamma)</td></tr>
    <tr><td>original_dtype</td><td>numpy.dtype</td><td>Pixel data type before processing</td></tr>
    <tr><td>image_type</td><td>str</td><td>Current modality string</td></tr>
    <tr><td>is_series</td><td>bool</td><td>Whether current data has multiple slices</td></tr>
    <tr><td>current_slice</td><td>int</td><td>Index of current slice</td></tr>
    <tr><td>current_series</td><td>list[str]</td><td>Filenames in selected series</td></tr>
    <tr><td>selected_series</td><td>str</td><td>Currently selected series key</td></tr>
    <tr><td>gamma</td><td>float</td><td>Active gamma correction value</td></tr>
    <tr><td>window</td><td>float</td><td>Active window multiplier</td></tr>
    <tr><td>source</td><td>ColumnDataSource</td><td>Bokeh data source for image display</td></tr>
    <tr><td>fig_image</td><td>Figure</td><td>Main image figure</td></tr>
    <tr><td>clip_points</td><td>list</td><td>Accumulated tap points for clipping (max 4)</td></tr>
</table>

<h2>Initialization Methods</h2>
<table>
    <tr><th>Method</th><th>Description</th></tr>
    <tr><td><code>__init__()</code></td><td>Parse CLI args, load config, read initial DICOM, build complete UI</td></tr>
    <tr><td><code>_create_figures()</code></td><td>Create Bokeh figure objects (image display + scatter plot)</td></tr>
    <tr><td><code>_create_widgets()</code></td><td>Create all widgets and register their callbacks</td></tr>
    <tr><td><code>_build_layout()</code></td><td>Assemble widget layout and attach to <code>curdoc()</code></td></tr>
</table>

<h2>Image Processing Methods</h2>
<table>
    <tr><th>Method</th><th>Description</th></tr>
    <tr><td><code>_prepare_images()</code></td>
        <td>Full pipeline: read DICOM &rarr; ensure_2d &rarr; rescale &rarr;
            photometric &rarr; flip &rarr; detect modality &rarr; gamma</td></tr>
    <tr><td><code>_size_figures()</code></td>
        <td>Update figure/glyph dimensions to match current image scale</td></tr>
    <tr><td><code>_set_image_fig_title(name)</code></td>
        <td>Set figure title from patient metadata (name, date, protocol)</td></tr>
    <tr><td><code>_get_new_image()</code></td>
        <td>Convenience: load and display the image at <code>current_slice</code></td></tr>
</table>

<h2>Callback Methods</h2>
<table>
    <tr><th>Method</th><th>Trigger</th><th>Description</th></tr>
    <tr><td><code>_name_cb(a,o,n)</code></td><td>Image dropdown</td><td>Load selected image</td></tr>
    <tr><td><code>_db_dropdown_cb(a,o,n)</code></td><td>Dataset dropdown</td>
        <td>Full dataset switch with complete state reset</td></tr>
    <tr><td><code>_series_cb(a,o,n)</code></td><td>Series dropdown</td><td>Switch to selected series</td></tr>
    <tr><td><code>_gamma_cb(a,o,n)</code></td><td>Gamma slider</td><td>Reprocess image with new gamma</td></tr>
    <tr><td><code>_window_cb(a,o,n)</code></td><td>Window slider</td><td>Adjust color mapper high bound</td></tr>
    <tr><td><code>_refresh_rate_cb(a,o,n)</code></td><td>Refresh TextInput</td>
        <td>Update animation speed (validates input)</td></tr>
    <tr><td><code>_reset_cb()</code></td><td>Reset button</td>
        <td>Restore gamma/window to config defaults</td></tr>
    <tr><td><code>_clip_reset_cb()</code></td><td>Clip button</td><td>Reset clip state for new selection</td></tr>
    <tr><td><code>_increment_cb()</code></td><td>Increment button</td><td>Advance one slice (bounds-checked)</td></tr>
    <tr><td><code>_decrement_cb()</code></td><td>Decrement button</td><td>Go back one slice (bounds-checked)</td></tr>
    <tr><td><code>_series_slider_slice_cb(a,o,n)</code></td><td>Slice slider</td><td>Jump to selected slice</td></tr>
    <tr><td><code>_series_toggle_anim_cb(active)</code></td><td>Animation toggle</td>
        <td>Start/stop periodic callback, update button label/color</td></tr>
    <tr><td><code>_animate_series()</code></td><td>Periodic timer</td>
        <td>Advance one animation frame, loop at end</td></tr>
    <tr><td><code>_tap_callback(event)</code></td><td>Mouse tap on image</td>
        <td>Collect clip points; execute clip/rotate on 4th point</td></tr>
    <tr><td><code>_stop_server()</code></td><td>Exit button</td><td>Async graceful shutdown sequence</td></tr>
</table>

<h2>Helper Methods</h2>
<table>
    <tr><th>Method</th><th>Description</th></tr>
    <tr><td><code>_find_images()</code></td>
        <td>Scan data directory for files, exclude hidden files, natural sort</td></tr>
    <tr><td><code>_sync_slice_slider()</code></td>
        <td>Update slider value without triggering its callback (removes/re-adds listener)</td></tr>
    <tr><td><code>_histogram_positions()</code></td>
        <td>Populate series position scatter plot data from SeriesManager</td></tr>
    <tr><td><code>_series_scatter_pos(name)</code></td>
        <td>Highlight current slice position on the scatter plot</td></tr>
    <tr><td><code>_modality_index()</code></td>
        <td>Map modality string to RadioButtonGroup index (0-3)</td></tr>
    <tr><td><code>_update_visibility()</code></td>
        <td>Show/hide series-related widgets based on current modality</td></tr>
    <tr><td><code>_reset_adjustments()</code></td>
        <td>Reset gamma/window sliders to config defaults without triggering callbacks</td></tr>
    <tr><td><code>_log(message)</code></td>
        <td>Add message to on-screen log (max 10) and Python logger</td></tr>
</table>

<h2>Async Methods</h2>
<table>
    <tr><th>Method</th><th>Description</th></tr>
    <tr><td><code>_async_update_log(message)</code></td>
        <td>Async log update used during shutdown sequence</td></tr>
    <tr><td><code>_exit_server()</code></td>
        <td>Stops the Tornado IOLoop</td></tr>
    <tr><td><code>_async_change_button(button, color)</code></td>
        <td>Async button color change</td></tr>
</table>
""",
    ))

    # ---- Data Flow ----
    manual.pages.append(Page(
        filename="dataflow.html",
        title="Data Flow",
        nav_title="Data Flow",
        raw_content="""
<h2>Initialization Flow</h2>
<div class="flow-diagram">
    YAML Config File<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    ViewerConfig.from_yaml()<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    DicomViewer.__init__()<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    _find_images() <span class="arrow">&rarr;</span> self.images_list<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    _prepare_images()<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; pydicom.dcmread() <span class="arrow">&rarr;</span> self.ds<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; ImageProcessor.ensure_2d()<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; ImageProcessor.apply_rescale()<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; ImageProcessor.apply_photometric()<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; np.flipud()<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x2514; ImageProcessor.perform_gamma() <span class="arrow">&rarr;</span> self.processed_image<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    SeriesManager.categorize() &nbsp;[if is_series]<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; dcmread(stop_before_pixels=True) for each file<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x2514; Populates series_map, series, series_extrema<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    _create_figures() <span class="arrow">&rarr;</span> fig_image, fig_series_positions<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    _create_widgets() <span class="arrow">&rarr;</span> all widgets + callbacks registered<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    _build_layout() <span class="arrow">&rarr;</span> curdoc().add_root()<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    Bokeh server running, awaiting user interaction
</div>

<h2>Image Update Flow (any callback that changes the displayed image)</h2>
<div class="flow-diagram">
    User interaction (dropdown, slider, button, animation timer)<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    Callback sets self.path to new DICOM file<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    _prepare_images()<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; Read DICOM <span class="arrow">&rarr;</span> detect modality<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; Process pixel data pipeline<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x2514; Compute display dimensions<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    self.source.data["image"] = [self.processed_image]<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    Bokeh syncs to browser via WebSocket<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    Browser re-renders image
</div>

<h2>Dataset Switch Flow</h2>
<div class="flow-diagram">
    _db_dropdown_cb(attr, old, new)<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    Update self.data_dir from data_db[new]<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    _find_images() <span class="arrow">&rarr;</span> new images_list<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    _prepare_images() on first image<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    [if is_series] SeriesManager.categorize()<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    Reset all widgets (temporarily remove callbacks to avoid cascading triggers)<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    _update_visibility() &mdash; show/hide series controls<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    _reset_adjustments() &mdash; gamma/window to defaults<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    Update source, title, figures
</div>
""",
    ))

    # ---- YAML Schema ----
    manual.pages.append(Page(
        filename="yaml_schema.html",
        title="YAML Configuration Schema",
        nav_title="YAML Schema",
        raw_content="""
<h2>Complete Schema</h2>
<pre><code># All fields are required unless marked optional

# [Required] Enable verbose logging to terminal
debug: bool

# [Required] Default gamma correction value
# Range: 0.0 - 10.0, typical: 1.0 - 2.0
gamma_def: float

# [Required] Default window/brightness multiplier
# Range: 0.0 - 2.0, typical: 1.0
window_def: float

# [Required] Key into data_db for the initial dataset to load
starter_images: str

# [Required] Dictionary of named datasets
# Each key is a human-readable label
# Each value is an absolute path to a directory of DICOM files
data_db:
  dataset_name_1: "/absolute/path/to/dicom/dir1/"
  dataset_name_2: "/absolute/path/to/dicom/dir2/"</code></pre>

<h2>Validation Rules</h2>
<table>
    <tr><th>Rule</th><th>Description</th></tr>
    <tr><td><code>starter_images</code> must exist in <code>data_db</code></td>
        <td>The initial dataset key must match one of the data_db entries</td></tr>
    <tr><td>All paths must exist</td>
        <td>Directory paths must be valid and accessible</td></tr>
    <tr><td>Directories must contain DICOM files</td>
        <td>At least one readable DICOM file must be present</td></tr>
    <tr><td><code>gamma_def</code> must be non-negative</td>
        <td>Zero is allowed (produces black image); typical range 0.5-3.0</td></tr>
    <tr><td><code>window_def</code> must be non-negative</td>
        <td>Zero clips everything to black; typical range 0.5-1.5</td></tr>
</table>

<h2>Type Mapping (YAML &rarr; Python)</h2>
<table>
    <tr><th>YAML Type</th><th>Python Type</th><th>Example</th></tr>
    <tr><td>boolean</td><td>bool</td><td><code>true</code> / <code>false</code></td></tr>
    <tr><td>number (decimal)</td><td>float</td><td><code>1.5</code></td></tr>
    <tr><td>string</td><td>str</td><td><code>"my_dataset"</code></td></tr>
    <tr><td>mapping</td><td>dict</td><td><code>key: value</code></td></tr>
</table>
""",
    ))

    # ---- Extending ----
    manual.pages.append(Page(
        filename="extending.html",
        title="Extending the Viewer",
        nav_title="Extending",
        raw_content="""
<h2>Adding a New Modality</h2>
<ol>
    <li>Add a new constant: <code>MODALITY_PET = "PET"</code></li>
    <li>Add a scale constant: <code>PET_SCALE = 2.0</code></li>
    <li>Add detection logic in <code>DicomViewer._prepare_images()</code>:
<pre><code>elif "PET" in self.sop_class_name:
    self.image_scale = PET_SCALE
    self.image_type = MODALITY_PET</code></pre>
    </li>
    <li>Update <code>_modality_index()</code> to map the new modality to a button index</li>
    <li>Add the label to the <code>RadioButtonGroup</code> in <code>_create_widgets()</code></li>
</ol>

<h2>Adding a New Image Processing Function</h2>
<ol>
    <li>Add a new <code>@staticmethod</code> to <code>ImageProcessor</code>:
<pre><code>@staticmethod
def apply_edge_detection(image: np.ndarray) -> np.ndarray:
    from skimage import filters
    return filters.sobel(image)</code></pre>
    </li>
    <li>Call it from <code>_prepare_images()</code> or a new callback</li>
    <li>The method is independently testable without any UI dependency</li>
</ol>

<h2>Adding a New Widget</h2>
<ol>
    <li>Create the widget in <code>_create_widgets()</code>:
<pre><code>self.my_button = Button(label="My Feature", button_type="primary")
self.my_button.on_click(self._my_feature_cb)</code></pre>
    </li>
    <li>Add it to the layout in <code>_build_layout()</code></li>
    <li>Implement the callback:
<pre><code>def _my_feature_cb(self):
    # Your logic here
    self._log("My feature activated")</code></pre>
    </li>
</ol>

<h2>Adding DICOM Metadata Display</h2>
<pre><code># In _create_widgets():
self.metadata_div = Div(text="", width=400, height=300, visible=False)
self.metadata_button = Toggle(label="Show Metadata", active=False)
self.metadata_button.on_click(self._metadata_toggle_cb)

# New callback:
def _metadata_toggle_cb(self, active):
    if active:
        info = []
        for elem in self.ds:
            if elem.VR != "OW" and elem.VR != "OB":  # Skip pixel data
                info.append(f"{elem.tag} {elem.name}: {elem.value}")
        self.metadata_div.text = "&lt;br&gt;".join(info[:50])
    self.metadata_div.visible = active</code></pre>

<h2>Naming Conventions</h2>
<table>
    <tr><th>Pattern</th><th>Meaning</th><th>Example</th></tr>
    <tr><td><code>_method_name</code></td><td>Private method</td><td><code>_prepare_images</code></td></tr>
    <tr><td><code>_xxx_cb</code></td><td>Bokeh callback</td><td><code>_gamma_cb</code></td></tr>
    <tr><td><code>_async_xxx</code></td><td>Async for next_tick_callback</td><td><code>_async_update_log</code></td></tr>
    <tr><td><code>UPPER_CASE</code></td><td>Module constant</td><td><code>MRI_SCALE</code></td></tr>
    <tr><td><code>ClassName</code></td><td>Class (PascalCase)</td><td><code>DicomViewer</code></td></tr>
</table>

<h2>Testing Strategy</h2>
<div class="card">
    <p><code>ImageProcessor</code> methods can be unit-tested directly:</p>
<pre><code>import numpy as np
from dicom_viewer import ImageProcessor

def test_gamma_identity():
    img = np.array([[100, 200], [50, 150]], dtype=np.uint16)
    result = ImageProcessor.perform_gamma(img, 1.0, np.uint16)
    assert result.shape == img.shape
    assert result.dtype == np.uint16

def test_ensure_2d_rgb():
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    result = ImageProcessor.ensure_2d(rgb)
    assert result.ndim == 2</code></pre>
</div>
""",
    ))

    # ---- Limitations ----
    manual.pages.append(Page(
        filename="limitations.html",
        title="Known Limitations",
        nav_title="Limitations",
        raw_content="""
<h2>Current Limitations</h2>
<table>
    <tr><th>#</th><th>Limitation</th><th>Impact</th><th>Possible Enhancement</th></tr>
    <tr><td>1</td>
        <td>No DICOMDIR navigation</td>
        <td>Must point to flat directories of DICOM files</td>
        <td>Parse DICOMDIR to build file tree</td></tr>
    <tr><td>2</td>
        <td>No DICOM Window Center/Width presets</td>
        <td>Uses a simple brightness multiplier instead of clinical presets</td>
        <td>Read (0028,1050)/(0028,1051) and apply <code>apply_window_level()</code></td></tr>
    <tr><td>3</td>
        <td>Color images converted to grayscale</td>
        <td>RGB, YBR_FULL images lose color information</td>
        <td>Support color display for applicable modalities</td></tr>
    <tr><td>4</td>
        <td>Clip/rotate cannot be undone</td>
        <td>Must reload image to restore original view</td>
        <td>Store original image state for undo</td></tr>
    <tr><td>5</td>
        <td>No zoom/pan persistence</td>
        <td>View resets when switching images</td>
        <td>Save and restore Range1d state</td></tr>
    <tr><td>6</td>
        <td>Sequential series metadata loading</td>
        <td>Slow for datasets with thousands of files</td>
        <td>Parallel loading with ThreadPoolExecutor</td></tr>
    <tr><td>7</td>
        <td>No measurement tools</td>
        <td>Cannot measure distance, angle, or area</td>
        <td>Add Bokeh PolyDrawTool with distance calculation</td></tr>
    <tr><td>8</td>
        <td>No DICOM metadata inspection panel</td>
        <td>Cannot view tags without external tools</td>
        <td>Add metadata Div (see <a href="extending.html">Extending</a>)</td></tr>
    <tr><td>9</td>
        <td>No keyboard shortcuts</td>
        <td>All interaction requires mouse clicks</td>
        <td>Add Bokeh CustomJS for keypress events</td></tr>
    <tr><td>10</td>
        <td>Single-user per session</td>
        <td>Bokeh server creates separate state per browser tab</td>
        <td>By design; each tab is independent</td></tr>
    <tr><td>11</td>
        <td>Animation refresh rate requires restart</td>
        <td>Changing speed while animating requires stop/start</td>
        <td>Dynamically remove and re-add periodic callback</td></tr>
    <tr><td>12</td>
        <td>No JPEG-LS support</td>
        <td>Some DICOM files with JPEG-LS transfer syntax may not load</td>
        <td>Install additional pylibjpeg plugins</td></tr>
</table>

<h2>DICOM Compliance Notes</h2>
<div class="warning">
    <p>This viewer is intended for <strong>research and educational use only</strong>.
    It is <strong>not</strong> a certified medical device and should not be used for
    clinical diagnosis.</p>
</div>
<ul>
    <li>The viewer does not validate DICOM conformance</li>
    <li>Not all DICOM transfer syntaxes are supported</li>
    <li>Overlay and annotation layers are not rendered</li>
    <li>Structured Reports (SR) are not parsed</li>
    <li>Patient orientation labels are not displayed</li>
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

