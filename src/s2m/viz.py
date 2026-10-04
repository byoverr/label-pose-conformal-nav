"""Shared figure style: fixed colour + marker per calibration arm, recessive axes.

Colours are the first slots of a validated categorical palette (CVD-separated in this order);
every series also has its own marker so identity never relies on colour alone.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

ARM_STYLE = {  # arm -> (label, colour, marker); fixed order, never re-assigned by rank
    "joint": ("joint: labels + pose (ours)", "#2a78d6", "o"),
    "uncalibrated": ("uncalibrated map", "#eb6834", "s"),
    "label_only": ("labels only (no drift in calibration)", "#1baf7a", "^"),
    "pose_only": ("pose only (GT labels)", "#eda100", "v"),
    "separate": ("sum of separate margins", "#e87ba4", "D"),
    "label_cell": ("per-cell label CP (Sundarsingh-style)", "#008300", "P"),
    "geometry": ("geometry only (any obstacle may be the class)", "#4a3aa7", "h"),
    "oracle": ("oracle (true map)", "#52514e", "x"),
}
ARM_LABEL_RU = {  # labels of the same arms for figures in the Russian report
    "joint": "совместный запас (наш метод)",
    "uncalibrated": "без калибровки",
    "label_only": "только метки",
    "pose_only": "только поза",
    "separate": "сумма двух запасов",
    "label_cell": "множества меток (Sundarsingh и др.)",
    "geometry": "только геометрия (любое препятствие)",
    "oracle": "оракул (истинная карта)",
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def setup() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "lines.linewidth": 2.0, "lines.markersize": 6,
        "legend.frameon": False,
    })


def arm_line(ax, arm, x, y, lang: str = "en", **kw):
    label, colour, marker = ARM_STYLE[arm]
    if lang == "ru":
        label = ARM_LABEL_RU[arm]
    return ax.plot(x, y, color=colour, marker=marker, label=label, **kw)


def decimal_comma(*axes) -> None:
    """Russian number format on the y axis (and x axis when it is numeric)."""
    from matplotlib.ticker import FuncFormatter

    fmt = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    for ax in axes:
        ax.yaxis.set_major_formatter(fmt)
