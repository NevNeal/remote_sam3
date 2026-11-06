"""
Shared paths and loaders for the pv99 analysis. Runs on HPG from the project
root (the flower_sam3 checkout), reading the finished run in place:

    export/pv99/pv99_segment_metadata.parquet       one row per photo
    results/pv99_flower/masks/batch_NNNNN/*.npz     the masks
    image_path column                               the photos on /blue

and writing everything into export/pv99/analysis/. The same code also runs on
an unpacked pack.sh bundle (--project <bundle>): image paths that don't exist
are looked up under the bundle instead.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent))           # pack_select.py

from maskio import load_masks  # noqa: E402
import pack_select  # noqa: E402

IMAGE_ROOT = "/home/neal.nevyn/blue_guralnick/share"   # slurm/settings.sh
RUN = Path("results/pv99_flower")
EXPORT = Path("export/pv99")
ANALYSIS = EXPORT / "analysis"

# Reference palette (light mode).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
             "#0d366b"]


def project_dir(arg=None):
    path = Path(arg).expanduser().resolve() if arg else Path.cwd()
    if not (path / EXPORT / "pv99_segment_metadata.parquet").is_file():
        raise SystemExit(f"no {EXPORT}/pv99_segment_metadata.parquet under {path}: run from "
                         f"the project root after the pv99 export, or pass --project")
    return path


def analysis_dir(project, sub=""):
    out = Path(project) / ANALYSIS / sub
    out.mkdir(parents=True, exist_ok=True)
    return out


def add_args(parser):
    parser.add_argument("--project", help="Project root (default: the current directory)")


def load_metadata(project, columns=None):
    return pd.read_parquet(Path(project) / EXPORT / "pv99_segment_metadata.parquet",
                           columns=columns)


def figure_selection(meta, seed=0):
    """Per family up to 40 masked photos spread across genera (30 + 10 spares)."""
    return pack_select.figure_pool(meta, seed=seed)


def segment_selection(meta):
    """The 200 highest-conf_max masked photos of each of the 30 biggest families."""
    return pack_select.segment_pool(meta, pack_select.top_families(meta))


def local_image(project, image_path):
    path = Path(str(image_path))
    if path.is_file():
        return path
    path, root = str(image_path), IMAGE_ROOT.rstrip("/") + "/"
    rel = path[len(root):] if path.startswith(root) else path.lstrip("/")
    return Path(project) / rel


def photo_masks(project, mask_path):
    """(N x H x W bool masks, scores), most confident first."""
    masks, scores = load_masks(Path(project) / RUN / str(mask_path))
    order = np.argsort(-scores, kind="stable")
    return masks[order], scores[order]


def n_workers():
    """CPUs this job may use (SLURM allocation, not the whole node)."""
    import os
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 2
    return max(1, n - 1)


def safe(value):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(value))


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=9, length=3)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
