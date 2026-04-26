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
        subtitle="Version 2.2 &mdash; April 2026",
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
<p class="subtitle">User Manual &mdash; Version 2.2 &mdash; April 2026</p>

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
    <li>Manual Window Center / Width sliders for fine-tuned control</li>
    <li><strong>10 color maps</strong> including Grayscale, Inferno, Viridis, Hot, and PET</li>
    <li><strong>Image inversion</strong> (positive/negative toggle)</li>
    <li><strong>Zoom/pan lock</strong> &mdash; preserves your view when stepping through slices</li>
    <li><strong>Position-sorted slices</strong> &mdash; anatomically correct slice ordering</li>
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

<h2>What's New in Version 2.2</h2>
<div class="info">
    <ul>
        <li><strong>Color Maps:</strong> 10 color lookup tables including clinical PET and
            Hot colormaps, plus scientific palettes (Inferno, Viridis, Plasma, etc.).</li>
        <li><strong>Image Inversion:</strong> Toggle between positive and negative display.
            Works with any color map.</li>
        <li><strong>Zoom/Pan Lock:</strong> Lock your current zoom and pan position. The view
            is preserved as you scroll through slices or animate.</li>
        <li><strong>Position-Sorted Slices:</strong> CT and MRI slices are now sorted by their
            DICOM spatial position by default, ensuring correct anatomical ordering regardless
            of filename conventions.</li>
    </ul>
</div>

<h2>Version History</h2>
<table>
    <tr><th>Version</th><th>Date</th><th>Highlights</th></tr>
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
<pre><code># Create virtual environment
python -m venv dicom_env
source dicom_env/bin/activate  # macOS/Linux
# dicom_env\\Scripts\\activate  # Windows

# Install dependencies
pip install pydicom numpy scikit-image pyyaml bokeh
pip install "pylibjpeg[all]"</code></pre>

<h2>Verify Installation</h2>
<pre><code>python -c "import pydicom, bokeh, skimage, yaml; print('All packages OK')"</code></pre>

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
<p>The viewer is configured via a YAML file:</p>
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
        <td>Default gamma correction value</td></tr>
    <tr><td><code>window_def</code></td><td>float</td><td><code>1.0</code></td>
        <td>Default brightness window multiplier (legacy control)</td></tr>
    <tr><td><code>starter_images</code></td><td>string</td><td>&mdash;</td>
        <td>Key from <code>data_db</code> to load on startup</td></tr>
    <tr><td><code>data_db</code></td><td>dict</td><td>&mdash;</td>
        <td>Named datasets mapping to directory paths</td></tr>
</table>

<div class="warning">
    <strong>Important:</strong>
    <ul>
        <li>Each directory should contain only DICOM files</li>
        <li>Hidden files (starting with <code>.</code>) are automatically excluded</li>
        <li>Paths must be absolute and end with a trailing slash</li>
    </ul>
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

<h2>Running on a Specific Port</h2>
<pre><code>bokeh serve dicom_viewer.py --port 8080 --args --app_config "config.yaml"</code></pre>

<h2>Stopping the Server</h2>
<ul>
    <li>Click <strong>Exit</strong> in the viewer</li>
    <li>Press <code>Ctrl+C</code> in the terminal</li>
</ul>
""",
    ))

    # ---- Interface Guide (UPDATED with Phase 1) ----
    manual.pages.append(Page(
        filename="interface.html",
        title="User Interface Guide",
        nav_title="Interface Guide",
        raw_content="""
<h2>Layout Overview</h2>
<div class="card">
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
        <td>Shuts down the Bokeh server.</td></tr>
    <tr><td><strong>Reset</strong></td>
        <td>Restores all settings to defaults: gamma, window, W/L, color map,
            inversion, and zoom lock.</td></tr>
    <tr><td><strong>Pick Imaging</strong></td>
        <td>Switch between datasets in your YAML config.</td></tr>
    <tr><td><strong>Mode</strong></td>
        <td>Auto-detected modality indicator (XRay / CT / MRI / US).</td></tr>
    <tr><td><strong>Clip</strong></td>
        <td>Reset clip mode. Tap 4 corners on the image to clip and rotate.</td></tr>
    <tr><td><strong>Gamma</strong></td>
        <td>Slider (0&ndash;10). Gamma correction curve.</td></tr>
    <tr><td><strong>Window (legacy)</strong></td>
        <td>Slider (0&ndash;2). Simple brightness multiplier.</td></tr>
    <tr><td><strong>W/L Preset</strong></td>
        <td>DICOM-defined window/level presets for the current image.</td></tr>
    <tr><td><strong>Window Center</strong></td>
        <td>Center of the displayed value range.</td></tr>
    <tr><td><strong>Window Width</strong></td>
        <td>Width of the displayed value range.</td></tr>
    <tr><td><strong>Color Map</strong></td>
        <td>Dropdown to select a color lookup table. Options: Grayscale, Grayscale (Inverted),
            Inferno, Viridis, Turbo, Cividis, Plasma, Magma, Hot, PET.</td></tr>
    <tr><td><strong>Invert</strong></td>
        <td>Toggle to reverse the current color map (positive/negative). Works with
            any color map. Button turns orange when active.</td></tr>
    <tr><td><strong>Lock Zoom</strong></td>
        <td>Toggle to preserve your current zoom and pan position when navigating
            between slices. Button turns green when locked. Automatically unlocks
            on dataset change.</td></tr>
    <tr><td><strong>Pick Image</strong></td>
        <td>Select a specific image file.</td></tr>
    <tr><td><strong>Pick Series</strong></td>
        <td>Select a series (CT/MRI only).</td></tr>
    <tr><td><strong>Start/Stop Animation</strong></td>
        <td>Auto-play through slices.</td></tr>
    <tr><td><strong>Refresh (ms)</strong></td>
        <td>Animation speed in milliseconds.</td></tr>
    <tr><td><strong>Slice Slider</strong></td>
        <td>Scrub through slices manually.</td></tr>
    <tr><td><strong>Increment / Decrement</strong></td>
        <td>Step forward or backward by one slice.</td></tr>
    <tr><td><strong>Sort by Position / Filename</strong></td>
        <td>Toggle between sorting slices by DICOM spatial position (anatomically correct)
            or by filename. Green when position-sorting is active. Only visible for
            CT/MRI/Ultrasound.</td></tr>
</table>

<h2>Understanding the Visual Controls</h2>
<div class="card">
    <h3>Color Maps</h3>
    <p>Color maps change how pixel intensity values are mapped to colors on screen.
    Different maps highlight different features:</p>
    <table>
        <tr><th>Color Map</th><th>Best For</th><th>Description</th></tr>
        <tr><td><strong>Grayscale</strong></td><td>General viewing</td>
            <td>Standard black-to-white mapping. Default for medical imaging.</td></tr>
        <tr><td><strong>Grayscale (Inverted)</strong></td><td>Alternative viewing</td>
            <td>White-to-black. Some radiologists prefer this for certain structures.</td></tr>
        <tr><td><strong>Hot</strong></td><td>Highlighting intensity</td>
            <td>Black &rarr; Red &rarr; Yellow &rarr; White. Good for seeing hot spots.</td></tr>
        <tr><td><strong>PET</strong></td><td>Nuclear medicine</td>
            <td>Black &rarr; Blue &rarr; Green &rarr; Yellow &rarr; Red &rarr; White.
                Standard for PET imaging overlays.</td></tr>
        <tr><td><strong>Inferno</strong></td><td>Scientific visualization</td>
            <td>Perceptually uniform dark-to-light. Colorblind friendly.</td></tr>
        <tr><td><strong>Viridis</strong></td><td>Scientific visualization</td>
            <td>Perceptually uniform purple-to-yellow. Colorblind friendly.</td></tr>
        <tr><td><strong>Turbo</strong></td><td>Rainbow-like</td>
            <td>Improved rainbow colormap with better perceptual properties.</td></tr>
        <tr><td><strong>Plasma</strong></td><td>Scientific visualization</td>
            <td>Purple-to-yellow warm palette.</td></tr>
        <tr><td><strong>Magma</strong></td><td>Scientific visualization</td>
            <td>Black-to-light warm palette.</td></tr>
        <tr><td><strong>Cividis</strong></td><td>Colorblind-safe</td>
            <td>Blue-to-yellow. Designed for deuteranomaly (red-green colorblindness).</td></tr>
    </table>

    <h3>Inversion</h3>
    <p>The <strong>Invert</strong> toggle reverses whatever color map is currently selected.
    For Grayscale, this gives a negative (white-on-black) view. For color maps, it reverses
    the entire color ramp.</p>

    <h3>Zoom Lock</h3>
    <p>When you zoom into a region of interest and want to examine the same area across
    multiple slices:</p>
    <ol>
        <li>Zoom and pan to the area of interest</li>
        <li>Click <strong>Lock Zoom</strong> (button turns green)</li>
        <li>Use Increment/Decrement, the slice slider, or animation</li>
        <li>Your zoom and pan position is preserved on every slice</li>
        <li>Click again to unlock and return to auto-fitting</li>
    </ol>

    <h3>Position Sorting</h3>
    <p>By default, CT and MRI slices are sorted by their DICOM spatial position
    (<code>ImagePositionPatient</code> tag). This ensures anatomically correct ordering
    regardless of how files are named. Toggle to "Sort by Filename" if you prefer
    the original file order.</p>
</div>

<h2>Brightness Controls</h2>
<div class="card">
    <table>
        <tr><th>Control</th><th>What It Does</th><th>When to Use</th></tr>
        <tr>
            <td><strong>Gamma</strong></td>
            <td>Adjusts brightness curve shape</td>
            <td>Enhance dark or bright area detail</td>
        </tr>
        <tr>
            <td><strong>Window (legacy)</strong></td>
            <td>Simple brightness multiplier</td>
            <td>Quick adjustment; images without W/L metadata</td>
        </tr>
        <tr>
            <td><strong>W/L Center &amp; Width</strong></td>
            <td>Controls which value range is displayed</td>
            <td>Standard radiological viewing (recommended for CT/MRI)</td>
        </tr>
    </table>
</div>

<h2>Series Position Plot</h2>
<p>For CT/MRI, shows spatial position of each slice:</p>
<ul>
    <li><span style="color:black;font-weight:bold;">Black / Blue</span> &mdash; all slices</li>
    <li><span style="color:red;font-weight:bold;">Red</span> &mdash; current slice</li>
</ul>

<h2>Log Panel</h2>
<p>Shows the last 10 actions including color map changes, zoom lock events,
sort mode changes, and all other operations.</p>
""",
    ))

    # ---- Workflows (UPDATED with Phase 1) ----
    manual.pages.append(Page(
        filename="workflows.html",
        title="Common Workflows",
        nav_title="Workflows",
        raw_content="""
<h2>Viewing an X-Ray</h2>
<ol>
    <li>Select your X-Ray dataset from <strong>Pick Imaging</strong>.</li>
    <li>Choose a specific image from <strong>Pick Image</strong>.</li>
    <li>Adjust brightness using W/L presets or Gamma/Window sliders.</li>
    <li>Try different <strong>Color Maps</strong> to highlight features.</li>
    <li>Use <strong>Invert</strong> for a negative view.</li>
</ol>

<h2>Browsing a CT/MRI Series</h2>
<ol>
    <li>Select your dataset from <strong>Pick Imaging</strong>.</li>
    <li>Choose a series from <strong>Pick Series</strong>.</li>
    <li>Slices are sorted by spatial position by default (anatomically correct).</li>
    <li>Zoom into the area of interest.</li>
    <li>Click <strong>Lock Zoom</strong> to preserve your view.</li>
    <li>Use <strong>Increment/Decrement</strong> or the slider to browse slices.</li>
    <li>Your zoom position is maintained across all slices.</li>
</ol>

<h2>Tracking a Structure Through Slices</h2>
<div class="card">
    <p>This is one of the most common radiological tasks &mdash; following a structure
    (vessel, lesion, organ boundary) through sequential slices:</p>
    <ol>
        <li>Ensure <strong>Sort by Position</strong> is active (green) for correct ordering.</li>
        <li>Navigate to a slice where the structure is visible.</li>
        <li>Zoom in to the structure.</li>
        <li>Click <strong>Lock Zoom</strong>.</li>
        <li>Set an appropriate <strong>W/L Preset</strong> (e.g., "Soft Tissue" for vessels).</li>
        <li>Use <strong>Increment/Decrement</strong> to step through slices.</li>
        <li>The structure remains centered and at the same zoom level.</li>
    </ol>
</div>

<h2>Using Color Maps for Enhanced Visualization</h2>
<div class="card">
    <h3>Recommended Color Maps by Use Case</h3>
    <table>
        <tr><th>Use Case</th><th>Recommended Map</th><th>Why</th></tr>
        <tr><td>Standard diagnostic reading</td><td>Grayscale</td>
            <td>Industry standard; radiologists are trained on grayscale</td></tr>
        <tr><td>Bone vs. soft tissue contrast</td><td>Hot</td>
            <td>High-intensity structures (bone) appear bright yellow/white</td></tr>
        <tr><td>PET/nuclear medicine</td><td>PET</td>
            <td>Standard nuclear medicine color scale</td></tr>
        <tr><td>Subtle density differences</td><td>Inferno or Viridis</td>
            <td>Perceptually uniform &mdash; equal steps in data appear as equal
                steps in color</td></tr>
        <tr><td>Presentations / colorblind viewers</td><td>Cividis</td>
            <td>Specifically designed for deuteranomaly accessibility</td></tr>
        <tr><td>Quick positive/negative toggle</td><td>Grayscale + Invert</td>
            <td>Fastest way to check for subtle features</td></tr>
    </table>

    <h3>Combining Color Maps with Inversion</h3>
    <p>The <strong>Invert</strong> toggle works with <em>any</em> color map. For example:</p>
    <ul>
        <li><strong>Hot + Invert:</strong> White &rarr; Yellow &rarr; Red &rarr; Black
            (bright areas become dark)</li>
        <li><strong>Viridis + Invert:</strong> Yellow &rarr; Green &rarr; Purple
            (reversed scientific palette)</li>
    </ul>
</div>

<h2>Using Window/Level Presets</h2>
<ol>
    <li>Load an image &mdash; the first preset is applied automatically.</li>
    <li>Open <strong>W/L Preset</strong> to see available presets.</li>
    <li>Select a preset to switch views (e.g., "Soft Tissue" to "Bone").</li>
    <li>Fine-tune with <strong>Window Center</strong> and <strong>Width</strong> sliders.</li>
</ol>

<h2>Animating Through Slices</h2>
<ol>
    <li>Set your preferred W/L, color map, and zoom level first.</li>
    <li>Click <strong>Lock Zoom</strong> if you want to preserve your view.</li>
    <li>Set the <strong>Refresh (ms)</strong> rate.</li>
    <li>Click <strong>Start Animation</strong>.</li>
    <li>All visual settings (W/L, color map, zoom) are preserved during animation.</li>
    <li>Click <strong>Stop Animation</strong> when done.</li>
</ol>

<h2>Clipping and Rotating a Region</h2>
<ol>
    <li>Click <strong>Clip</strong> to enter clip mode.</li>
    <li>Tap <strong>four corners</strong> on the image.</li>
    <li>The image is rotated and cropped automatically.</li>
</ol>
<div class="warning">
    <strong>Note:</strong> Clip cannot be undone. Select a different image to restore.
</div>

<h2>Switching Datasets</h2>
<ol>
    <li>Use <strong>Pick Imaging</strong> to change datasets.</li>
    <li>All settings reset (gamma, window, W/L, color map, zoom lock, sort mode).</li>
    <li>Modality detection and series controls update automatically.</li>
</ol>
""",
    ))

    # ---- Troubleshooting (UPDATED with Phase 1) ----
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
        <td>Try a different <strong>W/L Preset</strong>. Adjust <strong>Window Width</strong>
            wider and <strong>Window Center</strong>. Click <strong>Reset</strong>.</td>
    </tr>
    <tr>
        <td>Image too dark</td>
        <td>Decrease <strong>Window Center</strong> or increase <strong>Window Width</strong>.
            Lower <strong>Gamma</strong> below 1.0. Try the <strong>Invert</strong> toggle.</td>
    </tr>
    <tr>
        <td>Image too bright / washed out</td>
        <td>Increase <strong>Window Center</strong> or decrease <strong>Window Width</strong>.
            Increase <strong>Gamma</strong> above 1.0.</td>
    </tr>
    <tr>
        <td>Colors look wrong</td>
        <td>Check the <strong>Color Map</strong> dropdown &mdash; you may have a non-Grayscale
            map selected. Check if <strong>Invert</strong> is toggled on (button is orange).
            Click <strong>Reset</strong> to restore Grayscale.</td>
    </tr>
    <tr>
        <td>Zoom resets when changing slices</td>
        <td>Click <strong>Lock Zoom</strong> before navigating. The button turns green when
            active. Your zoom and pan position will be preserved.</td>
    </tr>
    <tr>
        <td>Slices appear out of anatomical order</td>
        <td>Ensure <strong>Sort by Position</strong> is active (green button). If the toggle
            shows "Sort by Filename", click it to switch to position-based sorting.</td>
    </tr>
    <tr>
        <td>Position sorting doesn't change anything</td>
        <td>Some DICOM files lack <code>ImagePositionPatient</code> metadata. In this case,
            position sorting falls back to filename sorting. The log panel will indicate
            if position data is unavailable.</td>
    </tr>
    <tr>
        <td>W/L Preset dropdown shows only "Manual"</td>
        <td>The current DICOM image lacks Window Center/Width metadata. The viewer
            uses auto-calculated values from the image data range.</td>
    </tr>
    <tr>
        <td>Image appears inverted unexpectedly</td>
        <td>Check if <strong>Invert</strong> is toggled on (orange button) or if you have
            "Grayscale (Inverted)" selected in the <strong>Color Map</strong> dropdown.</td>
    </tr>
    <tr>
        <td>Animation stutters</td>
        <td>Increase <strong>Refresh (ms)</strong>. 500-1000ms works well for large files.
            Color maps other than Grayscale may be slightly slower.</td>
    </tr>
    <tr>
        <td>"Hit whitespace" error when clipping</td>
        <td>Click directly on visible image content, not the background.</td>
    </tr>
    <tr>
        <td>Server won't start</td>
        <td>Check YAML config path, verify directories exist, ensure DICOM files are present.</td>
    </tr>
</table>

<h2>Enabling Debug Mode</h2>
<p>Set <code>debug: true</code> in your YAML config for verbose terminal output.</p>

<h2>Getting Help</h2>
<div class="info">
    <strong>Diagnostic checklist:</strong>
    <ol>
        <li>Enable <code>debug: true</code> in the YAML config</li>
        <li>Check terminal for Python errors</li>
        <li>Check the in-app Log panel</li>
        <li>Verify DICOM files work in another viewer</li>
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
        subtitle="Developer API &amp; Architecture Guide &mdash; Version 2.2 &mdash; April 2026",
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
<p class="subtitle">Developer API &amp; Architecture Guide &mdash; Version 2.2</p>

<h2>Module Structure</h2>
<table>
    <tr><th>Component</th><th>Responsibility</th></tr>
    <tr><td><code>ViewerConfig</code></td>
        <td>Dataclass for YAML configuration</td></tr>
    <tr><td><code>ImageProcessor</code></td>
        <td>Stateless image processing and metadata extraction</td></tr>
    <tr><td><code>SeriesManager</code></td>
        <td>Series metadata, spatial analysis, position and filename sorting</td></tr>
    <tr><td><code>DicomViewer</code></td>
        <td>Main UI: state, widgets, callbacks, color maps, zoom persistence</td></tr>
</table>

<h2>Module-Level Functions</h2>
<table>
    <tr><th>Function</th><th>Purpose</th></tr>
    <tr><td><code>_build_hot_palette(n=256)</code></td>
        <td>Generates a black&rarr;red&rarr;yellow&rarr;white "Hot" color palette</td></tr>
    <tr><td><code>_build_pet_palette(n=256)</code></td>
        <td>Generates a PET-style multi-color palette</td></tr>
</table>

<h2>Module-Level Data</h2>
<table>
    <tr><th>Name</th><th>Type</th><th>Description</th></tr>
    <tr><td><code>COLOR_LUTS</code></td><td>dict</td>
        <td>Maps color map names to 256-element palette lists</td></tr>
</table>

<h2>Design Principles</h2>
<ul>
    <li><strong>Separation of concerns:</strong> Processing, metadata, and UI are separated</li>
    <li><strong>Named constants:</strong> All magic numbers are module-level constants</li>
    <li><strong>Specific exceptions:</strong> No bare <code>except</code> clauses</li>
    <li><strong>Zoom state management:</strong> View ranges saved/restored explicitly</li>
    <li><strong>Sort abstraction:</strong> Sorting logic in <code>SeriesManager</code>, UI in <code>DicomViewer</code></li>
    <li><strong>LUT composability:</strong> Inversion works as a modifier on any base palette</li>
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
    <tr><td><code>pydicom</code></td><td>&ge; 2.0</td><td>DICOM file reading</td></tr>
    <tr><td><code>pylibjpeg[all]</code></td><td>&ge; 1.0</td><td>JPEG DICOM support</td></tr>
    <tr><td><code>numpy</code></td><td>&ge; 1.20</td><td>Array operations</td></tr>
    <tr><td><code>scikit-image</code></td><td>&ge; 0.18</td><td>Image rotation</td></tr>
    <tr><td><code>bokeh</code></td><td>&ge; 3.0</td><td>Web UI and server</td></tr>
    <tr><td><code>pyyaml</code></td><td>&ge; 5.0</td><td>YAML parsing</td></tr>
</table>

<h2>Bokeh Palettes Used</h2>
<p>The following Bokeh built-in palettes are imported:</p>
<pre><code>from bokeh.palettes import (
    Greys256, Inferno256, Viridis256, Turbo256,
    Cividis256, Plasma256, Magma256
)</code></pre>
<p>Two additional palettes (<code>Hot</code> and <code>PET</code>) are generated
algorithmically by module-level functions.</p>
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
    <tr><th>Constant</th><th>Value</th><th>Description</th></tr>
    <tr><td><code>XRAY_SCALE</code></td><td>0.5</td><td>X-Ray display scale</td></tr>
    <tr><td><code>CT_SCALE</code></td><td>1.5</td><td>CT display scale</td></tr>
    <tr><td><code>MRI_SCALE</code></td><td>4.0</td><td>MRI display scale</td></tr>
    <tr><td><code>US_SCALE</code></td><td>1.5</td><td>Ultrasound display scale</td></tr>
</table>

<h2>Gamma</h2>
<table>
    <tr><th>Constant</th><th>Value</th></tr>
    <tr><td><code>MRI_GAMMA</code></td><td>2.0</td></tr>
    <tr><td><code>US_GAMMA</code></td><td>2.0</td></tr>
    <tr><td><code>DEFAULT_GAMMA</code></td><td>1.0</td></tr>
    <tr><td><code>DEFAULT_WINDOW</code></td><td>1.0</td></tr>
</table>

<h2>Window/Level</h2>
<table>
    <tr><th>Constant</th><th>Value</th><th>Description</th></tr>
    <tr><td><code>WL_CENTER_DEFAULT</code></td><td>2000.0</td><td>Fallback center</td></tr>
    <tr><td><code>WL_WIDTH_DEFAULT</code></td><td>4000.0</td><td>Fallback width</td></tr>
    <tr><td><code>WL_CENTER_SLIDER_MIN</code></td><td>-2000.0</td><td>Min center (neg HU)</td></tr>
    <tr><td><code>WL_CENTER_SLIDER_MAX</code></td><td>20000.0</td><td>Default max center</td></tr>
    <tr><td><code>WL_WIDTH_SLIDER_MIN</code></td><td>1.0</td><td>Min width</td></tr>
    <tr><td><code>WL_WIDTH_SLIDER_MAX</code></td><td>40000.0</td><td>Default max width</td></tr>
    <tr><td><code>WL_SLIDER_STEP</code></td><td>10.0</td><td>Slider step size</td></tr>
    <tr><td><code>WL_MANUAL_LABEL</code></td><td>"Manual"</td><td>Dropdown label</td></tr>
</table>

<h2>Color Map</h2>
<table>
    <tr><th>Constant</th><th>Type</th><th>Description</th></tr>
    <tr><td><code>COLOR_LUTS</code></td><td>dict</td>
        <td>Maps 10 color map names to 256-element hex color lists.
            Keys: Grayscale, Grayscale (Inverted), Inferno, Viridis, Turbo,
            Cividis, Plasma, Magma, Hot, PET</td></tr>
    <tr><td><code>DEFAULT_LUT</code></td><td>str</td>
        <td>"Grayscale" &mdash; initial color map on startup</td></tr>
</table>

<h2>UI and Analysis</h2>
<table>
    <tr><th>Constant</th><th>Value</th><th>Description</th></tr>
    <tr><td><code>GAMMA_SLIDER_MAX</code></td><td>10.0</td><td>Gamma upper bound</td></tr>
    <tr><td><code>WINDOW_SLIDER_MAX</code></td><td>2.0</td><td>Legacy window upper bound</td></tr>
    <tr><td><code>MAX_LOG_MESSAGES</code></td><td>10</td><td>Visible log entries</td></tr>
    <tr><td><code>DEFAULT_REFRESH_MS</code></td><td>500.0</td><td>Animation refresh (ms)</td></tr>
    <tr><td><code>NORMAL_THRESHOLD</code></td><td>0.95</td><td>Axis detection threshold</td></tr>
    <tr><td><code>ANIMATION_SLICE_RESET</code></td><td>0</td><td>Loop reset index</td></tr>
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
<div class="method-sig">@dataclass<br>class ViewerConfig:</div>

<h2>Fields</h2>
<table class="attr-table">
    <tr><th>Field</th><th>Type</th><th>Default</th><th>Description</th></tr>
    <tr><td>debug</td><td>bool</td><td>False</td><td>Debug mode</td></tr>
    <tr><td>gamma_def</td><td>float</td><td>1.0</td><td>Default gamma</td></tr>
    <tr><td>window_def</td><td>float</td><td>1.0</td><td>Default legacy window</td></tr>
    <tr><td>starter_images</td><td>str</td><td>""</td><td>Initial dataset key</td></tr>
    <tr><td>data_db</td><td>dict</td><td>{}</td><td>Dataset mapping</td></tr>
</table>

<h2>Methods</h2>
<div class="method-sig">@classmethod from_yaml(cls, path: str) &rarr; ViewerConfig</div>
""",
    ))

    # ---- ImageProcessor (unchanged from 2.1) ----
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
<p>Reads (0028,1050), (0028,1051), (0028,1055). Returns list of
<code>{"center": float, "width": float, "name": str}</code>.</p>

<h2>Geometry</h2>
<div class="method-sig">rotated_rectangle_properties(corners) &rarr; tuple</div>
""",
    ))

    # ---- SeriesManager (UPDATED with position sort) ----
    manual.pages.append(Page(
        filename="seriesmanager.html",
        title="Class: SeriesManager",
        nav_title="SeriesManager",
        raw_content="""
<h2>Overview</h2>
<p>Manages DICOM series metadata. Supports both filename and position-based sorting.</p>

<h2>Instance Attributes</h2>
<table class="attr-table">
    <tr><th>Attribute</th><th>Type</th><th>Description</th></tr>
    <tr><td>series_map</td><td>dict</td>
        <td><code>{series: {filename: [instance, pos, dir, normal]}}</code></td></tr>
    <tr><td>series</td><td>list[str]</td><td>Sorted series numbers</td></tr>
    <tr><td>series_extrema</td><td>dict</td><td>Position ranges per series</td></tr>
    <tr><td>series_pos_index</td><td>int</td><td>Active axis (0/1/2)</td></tr>
    <tr><td>sort_by_position</td><td>bool</td>
        <td><strong>New in 2.2.</strong> If True, <code>get_series_images_by_position()</code>
            is used; otherwise <code>get_series_images()</code> (filename sort).
            Default: True.</td></tr>
</table>

<h2>Methods</h2>

<div class="method-sig">categorize(images_list, data_dir, debug=False, log_callback=None)</div>
<p>Scans all files with <code>stop_before_pixels=True</code>. Populates series_map.</p>

<div class="method-sig">get_series_images(series_key: str) &rarr; list[str]</div>
<p>Returns filenames sorted by natural filename order (alpha prefix + numeric suffix).</p>

<div class="method-sig">get_series_images_by_position(series_key: str, axis: int = 2) &rarr; list[str]</div>
<p><strong>New in 2.2.</strong> Returns filenames sorted by DICOM spatial position
along the specified axis.</p>
<table>
    <tr><th>Parameter</th><th>Type</th><th>Description</th></tr>
    <tr><td>series_key</td><td>str</td><td>Series identifier</td></tr>
    <tr><td>axis</td><td>int</td><td>0=x (sagittal), 1=y (coronal), 2=z (axial)</td></tr>
</table>
<p><strong>Fallback:</strong> If no valid position data exists (all positions are sentinel
value -999.0), falls back to <code>get_series_images()</code>.</p>

<div class="method-sig">determine_axis(series_key, images) &rarr; int</div>
<p>Returns principal imaging axis from slice normal vectors.</p>
""",
    ))

    # ---- DicomViewer (UPDATED with Phase 1) ----
    manual.pages.append(Page(
        filename="dicomviewer.html",
        title="Class: DicomViewer",
        nav_title="DicomViewer",
        raw_content="""
<h2>Overview</h2>
<p>Main application class with color map management, zoom persistence, and position sorting.</p>

<h2>Key Instance Attributes</h2>
<table class="attr-table">
    <tr><th>Attribute</th><th>Type</th><th>Description</th></tr>
    <tr><td>config</td><td>ViewerConfig</td><td>Configuration</td></tr>
    <tr><td>img_proc</td><td>ImageProcessor</td><td>Processing utilities</td></tr>
    <tr><td>series_mgr</td><td>SeriesManager</td><td>Series manager</td></tr>
    <tr><td>ds</td><td>pydicom.Dataset</td><td>Current DICOM dataset</td></tr>
    <tr><td>processed_image</td><td>ndarray</td><td>Display image</td></tr>
    <tr><td>wl_presets</td><td>list[dict]</td><td>W/L presets from metadata</td></tr>
    <tr><td>color_mapper</td><td>LinearColorMapper</td><td>Controls display range and palette</td></tr>
    <tr><td colspan="3" style="background:#faf5ff;font-weight:bold;">New in v2.2</td></tr>
    <tr><td>current_lut</td><td>str</td>
        <td>Name of active color map (key into <code>COLOR_LUTS</code>)</td></tr>
    <tr><td>is_inverted</td><td>bool</td>
        <td>Whether the current palette is inverted</td></tr>
    <tr><td>zoom_locked</td><td>bool</td>
        <td>Whether zoom/pan persistence is active</td></tr>
    <tr><td>saved_x_range</td><td>tuple or None</td>
        <td>Saved (start, end) for x_range when zoom is locked</td></tr>
    <tr><td>saved_y_range</td><td>tuple or None</td>
        <td>Saved (start, end) for y_range when zoom is locked</td></tr>
</table>

<h2>Phase 1 Methods (New in v2.2)</h2>

<h3>Color Map Methods</h3>
<table>
    <tr><th>Method</th><th>Description</th></tr>
    <tr><td><code>_invert_cb(active)</code></td>
        <td>Toggle callback. Sets <code>is_inverted</code>, calls <code>_apply_current_lut()</code>,
            updates button appearance.</td></tr>
    <tr><td><code>_lut_cb(attr, old, new)</code></td>
        <td>Dropdown callback. Sets <code>current_lut</code>, calls <code>_apply_current_lut()</code>.</td></tr>
    <tr><td><code>_apply_current_lut()</code></td>
        <td>Core LUT application method. Looks up palette from <code>COLOR_LUTS</code>,
            applies inversion if <code>is_inverted</code>, sets
            <code>color_mapper.palette</code>. Makes a copy of the palette list
            to avoid mutating the original.</td></tr>
</table>

<h3>Sort Methods</h3>
<table>
    <tr><th>Method</th><th>Description</th></tr>
    <tr><td><code>_sort_toggle_cb(active)</code></td>
        <td>Toggle callback. Sets <code>series_mgr.sort_by_position</code>,
            calls <code>_resort_current_series()</code>, updates button label.</td></tr>
    <tr><td><code>_resort_current_series()</code></td>
        <td>Re-sorts current series based on sort mode. Updates image dropdown,
            slice slider, and position plot. Tries to preserve the current image
            (finds its index in the new sort order).</td></tr>
</table>

<h3>Zoom Lock Methods</h3>
<table>
    <tr><th>Method</th><th>Description</th></tr>
    <tr><td><code>_zoom_lock_cb(active)</code></td>
        <td>Toggle callback. When locking: saves current <code>x_range</code> and
            <code>y_range</code> start/end values. When unlocking: clears saved ranges.
            Updates button appearance.</td></tr>
    <tr><td><code>_restore_zoom()</code></td>
        <td>Restores saved x/y ranges to the figure. Called by <code>_size_figures()</code>,
            <code>_get_new_image()</code>, and <code>_animate_series()</code> when
            zoom is locked.</td></tr>
</table>

<h2>Updated Methods (changed in v2.2)</h2>
<table>
    <tr><th>Method</th><th>What Changed</th></tr>
    <tr><td><code>_size_figures()</code></td>
        <td>When zoom locked: only updates glyph dw/dh, then calls <code>_restore_zoom()</code>
            instead of resetting ranges.</td></tr>
    <tr><td><code>_get_new_image()</code></td>
        <td>Calls <code>_restore_zoom()</code> after loading new image data.</td></tr>
    <tr><td><code>_name_cb()</code></td>
        <td>Updates <code>current_slice</code> index to match selected image name.</td></tr>
    <tr><td><code>_db_dropdown_cb()</code></td>
        <td>Uses sort mode for series ordering. Unlocks zoom on dataset change.
            Reapplies current LUT after reset.</td></tr>
    <tr><td><code>_series_cb()</code></td>
        <td>Uses sort mode for new series.</td></tr>
    <tr><td><code>_animate_series()</code></td>
        <td>Applies current W/L and restores zoom on each frame.</td></tr>
    <tr><td><code>_update_visibility()</code></td>
        <td>Shows/hides sort toggle based on modality.</td></tr>
    <tr><td><code>_reset_cb()</code></td>
        <td>Resets invert, LUT, and zoom lock in addition to gamma/window/W/L.</td></tr>
    <tr><td><code>_build_layout()</code></td>
        <td>New <code>visual_controls</code> column with LUT dropdown, Invert toggle,
            and Zoom Lock toggle. Sort toggle added to series control row.</td></tr>
    <tr><td><code>_create_widgets()</code></td>
        <td>Creates 4 new widgets: <code>invert_toggle</code>, <code>lut_dropdown</code>,
            <code>sort_toggle</code>, <code>zoom_lock_toggle</code>.</td></tr>
</table>

<h2>All Callbacks</h2>
<table>
    <tr><th>Method</th><th>Trigger</th><th>Category</th></tr>
    <tr><td><code>_name_cb</code></td><td>Image dropdown</td><td>Navigation</td></tr>
    <tr><td><code>_db_dropdown_cb</code></td><td>Dataset dropdown</td><td>Navigation</td></tr>
    <tr><td><code>_series_cb</code></td><td>Series dropdown</td><td>Navigation</td></tr>
    <tr><td><code>_increment_cb</code></td><td>Increment button</td><td>Navigation</td></tr>
    <tr><td><code>_decrement_cb</code></td><td>Decrement button</td><td>Navigation</td></tr>
    <tr><td><code>_series_slider_slice_cb</code></td><td>Slice slider</td><td>Navigation</td></tr>
    <tr><td><code>_series_toggle_anim_cb</code></td><td>Animation toggle</td><td>Animation</td></tr>
    <tr><td><code>_animate_series</code></td><td>Periodic timer</td><td>Animation</td></tr>
    <tr><td><code>_gamma_cb</code></td><td>Gamma slider</td><td>Image adjust</td></tr>
    <tr><td><code>_window_cb</code></td><td>Legacy window slider</td><td>Image adjust</td></tr>
    <tr><td><code>_wl_preset_cb</code></td><td>W/L preset dropdown</td><td>Image adjust</td></tr>
    <tr><td><code>_wl_manual_cb</code></td><td>Center/Width sliders</td><td>Image adjust</td></tr>
    <tr><td><code>_lut_cb</code></td><td>Color map dropdown</td><td>Visual (v2.2)</td></tr>
    <tr><td><code>_invert_cb</code></td><td>Invert toggle</td><td>Visual (v2.2)</td></tr>
    <tr><td><code>_zoom_lock_cb</code></td><td>Zoom lock toggle</td><td>Visual (v2.2)</td></tr>
    <tr><td><code>_sort_toggle_cb</code></td><td>Sort toggle</td><td>Navigation (v2.2)</td></tr>
    <tr><td><code>_refresh_rate_cb</code></td><td>Refresh TextInput</td><td>Animation</td></tr>
    <tr><td><code>_reset_cb</code></td><td>Reset button</td><td>Control</td></tr>
    <tr><td><code>_clip_reset_cb</code></td><td>Clip button</td><td>Tool</td></tr>
    <tr><td><code>_tap_callback</code></td><td>Mouse tap</td><td>Tool</td></tr>
    <tr><td><code>_stop_server</code></td><td>Exit button</td><td>Control</td></tr>
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
    YAML &rarr; ViewerConfig.from_yaml()<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    DicomViewer.__init__()<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    _find_images() &rarr; images_list<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    _prepare_images() [includes extract_wl_presets]<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    SeriesManager.categorize() [if series]<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x2514; <strong>get_series_images_by_position()</strong> [default sort]<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    Create color_mapper with <strong>COLOR_LUTS["Grayscale"]</strong><br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    _apply_window_level(first preset) [if available]<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    _create_figures() + _create_widgets() + _build_layout()<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    curdoc().add_root()
</div>

<h2>Slice Navigation with Zoom Lock</h2>
<div class="flow-diagram">
    User clicks Increment/Decrement or drags slider<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    _get_new_image()<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; _prepare_images()<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; source.data update<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; _apply_window_level(current sliders)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x2514; <strong>if zoom_locked: _restore_zoom()</strong><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#x251C; fig.x_range = saved_x_range<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#x2514; fig.y_range = saved_y_range<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    View preserved &rarr; browser renders same zoom/pan
</div>

<h2>Color Map Change Flow</h2>
<div class="flow-diagram">
    User selects new LUT from dropdown<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    _lut_cb(attr, old, new)<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    self.current_lut = new<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    _apply_current_lut()<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; palette = COLOR_LUTS[current_lut].copy()<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; if is_inverted: palette = reversed(palette)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x2514; color_mapper.palette = palette<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    Bokeh syncs palette to browser
</div>

<h2>Sort Mode Change Flow</h2>
<div class="flow-diagram">
    User toggles Sort by Position / Filename<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    _sort_toggle_cb(active)<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    series_mgr.sort_by_position = active<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    _resort_current_series()<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; if sort_by_position:<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x2502;&nbsp;&nbsp;&nbsp;get_series_images_by_position(series, axis)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; else:<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x2502;&nbsp;&nbsp;&nbsp;get_series_images(series)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; Update name_dropdown options<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; Find current image in new order (preserve selection)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x251C; Update slice_slider range<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&#x2514; Refresh position histogram + scatter
</div>

<h2>Reset Flow</h2>
<div class="flow-diagram">
    _reset_cb()<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    Reset gamma, window sliders to config defaults<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    Reprocess image with default gamma<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    _refresh_wl_presets() [reset W/L]<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    Reset is_inverted = False, invert_toggle<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    Reset current_lut = "Grayscale", lut_dropdown<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    _apply_current_lut() [restore grayscale palette]<br>
    &nbsp;&nbsp;<span class="arrow">&darr;</span><br>
    Unlock zoom: zoom_locked = False, clear saved ranges
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
<pre><code>debug: bool
gamma_def: float
window_def: float
starter_images: str
data_db:
  name: "/absolute/path/"</code></pre>

<h2>Validation Rules</h2>
<table>
    <tr><th>Rule</th><th>Description</th></tr>
    <tr><td><code>starter_images</code> must exist in <code>data_db</code></td>
        <td>Initial dataset key must match</td></tr>
    <tr><td>All paths must exist</td><td>Directories must be accessible</td></tr>
    <tr><td>Directories must contain DICOM files</td><td>At least one readable file</td></tr>
</table>
""",
    ))

    # ---- Extending (UPDATED with Phase 1 examples) ----
    manual.pages.append(Page(
        filename="extending.html",
        title="Extending the Viewer",
        nav_title="Extending",
        raw_content="""
<h2>Adding a Custom Color Map</h2>
<p>Add a new palette to <code>COLOR_LUTS</code> at module level:</p>
<pre><code>def _build_rainbow_palette(n=256):
    palette = []
    for i in range(n):
        h = i / n
        # HSV to RGB conversion
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        palette.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return palette

COLOR_LUTS["Rainbow"] = _build_rainbow_palette()</code></pre>
<p>The new color map automatically appears in the dropdown.</p>

<h2>Adding a New Sort Mode</h2>
<p>Add a new method to <code>SeriesManager</code>:</p>
<pre><code>def get_series_images_by_instance(self, series_key):
    entries = self.series_map.get(series_key, {})
    return sorted(entries.keys(),
                  key=lambda k: int(entries[k][0]))</code></pre>
<p>Then add it as an option in the sort toggle or replace the dropdown.</p>

<h2>Adding Custom W/L Presets</h2>
<pre><code>if self.image_type == MODALITY_CT and not self.wl_presets:
    self.wl_presets = [
        {"center": 40.0,   "width": 350.0,  "name": "Soft Tissue"},
        {"center": 400.0,  "width": 1500.0, "name": "Bone"},
        {"center": -600.0, "width": 1500.0, "name": "Lung"},
    ]</code></pre>

<h2>Adding a New Modality</h2>
<ol>
    <li>Add constants: <code>MODALITY_PET = "PET"</code>, <code>PET_SCALE = 2.0</code></li>
    <li>Add detection in <code>_prepare_images()</code></li>
    <li>Update <code>_modality_index()</code> and RadioButtonGroup labels</li>
</ol>

<h2>Adding Zoom-Aware Tools</h2>
<p>Any new tool that modifies the view should respect zoom lock:</p>
<pre><code>def _my_tool_cb(self):
    # ... do work ...
    if self.zoom_locked:
        self._restore_zoom()</code></pre>

<h2>Naming Conventions</h2>
<table>
    <tr><th>Pattern</th><th>Meaning</th></tr>
    <tr><td><code>_method_name</code></td><td>Private method</td></tr>
    <tr><td><code>_xxx_cb</code></td><td>Bokeh callback</td></tr>
    <tr><td><code>_async_xxx</code></td><td>Async method</td></tr>
    <tr><td><code>UPPER_CASE</code></td><td>Module constant</td></tr>
    <tr><td><code>_build_xxx_palette</code></td><td>Palette generator function</td></tr>
</table>

<h2>Testing</h2>
<pre><code>from dicom_viewer import COLOR_LUTS, _build_hot_palette

def test_all_luts_have_256_entries():
    for name, palette in COLOR_LUTS.items():
        assert len(palette) == 256, f"{name} has {len(palette)} entries"

def test_hot_palette_starts_black():
    p = _build_hot_palette()
    assert p[0] == "#000000"

def test_hot_palette_ends_white():
    p = _build_hot_palette()
    assert p[-1] == "#ffffff"

def test_inversion_reversal():
    original = list(COLOR_LUTS["Grayscale"])
    inverted = list(reversed(original))
    assert original[0] == inverted[-1]
    assert original[-1] == inverted[0]</code></pre>
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
    <tr><th>#</th><th>Limitation</th><th>Status</th></tr>
    <tr><td>1</td><td>No DICOMDIR navigation</td><td>Open</td></tr>
    <tr><td>2</td><td>W/L presets from metadata only (no custom CT presets built-in)</td>
        <td><em>Partially resolved v2.1</em></td></tr>
    <tr><td>3</td><td>Color images converted to grayscale</td><td>Open</td></tr>
    <tr><td>4</td><td>Clip/rotate cannot be undone</td><td>Open</td></tr>
    <tr><td>5</td><td>Zoom persistence requires manual lock toggle</td>
        <td><em>Resolved v2.2</em> (Lock Zoom feature)</td></tr>
    <tr><td>6</td><td>Sequential series metadata loading</td><td>Open</td></tr>
    <tr><td>7</td><td>No measurement tools</td><td>Open (Phase 3)</td></tr>
    <tr><td>8</td><td>No DICOM metadata panel</td><td>Open (Phase 2)</td></tr>
    <tr><td>9</td><td>No keyboard shortcuts</td><td>Open (Phase 4)</td></tr>
    <tr><td>10</td><td>Single-user per session</td><td>By design</td></tr>
    <tr><td>11</td><td>Animation refresh change requires restart</td><td>Open</td></tr>
    <tr><td>12</td><td>Legacy window slider overlaps with W/L</td><td>Documented</td></tr>
    <tr><td>13</td><td>Mousewheel slice scrolling</td><td>Open (Phase 2)</td></tr>
    <tr><td>14</td><td>No ROI statistics</td><td>Open (Phase 3)</td></tr>
    <tr><td>15</td><td>No multiplanar reconstruction</td><td>Open (Phase 4)</td></tr>
</table>

<h2>Feature Roadmap</h2>
<table>
    <tr><th>Phase</th><th>Features</th><th>Status</th></tr>
    <tr><td>1</td><td>Color maps, invert, zoom lock, position sort</td>
        <td><strong>&#x2705; Complete (v2.2)</strong></td></tr>
    <tr><td>2</td><td>Mousewheel scrolling, metadata panel, histogram</td><td>Planned</td></tr>
    <tr><td>3</td><td>Measurement tools, ROI stats, annotations</td><td>Planned</td></tr>
    <tr><td>4</td><td>MPR, multi-panel, cine controls, keyboard shortcuts</td><td>Planned</td></tr>
    <tr><td>5</td><td>3D rendering, PACS connectivity, fusion</td><td>Future</td></tr>
</table>

<h2>DICOM Compliance</h2>
<div class="warning">
    <p>This viewer is for <strong>research and educational use only</strong>.</p>
</div>
<ul>
    <li>Reads W/L tags (0028,1050/1051/1055)</li>
    <li>Reads PhotometricInterpretation</li>
    <li>Reads RescaleSlope/Intercept</li>
    <li>Reads ImagePositionPatient for position sorting</li>
    <li>Reads ImageOrientationPatient for axis detection</li>
    <li>Does not support VOI LUT Sequences</li>
    <li>Overlays and annotations not rendered</li>
    <li>Patient orientation labels not displayed</li>
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

