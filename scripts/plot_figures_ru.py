"""Russian figures for the article, redrawn from the result tables (no recomputation).

Writes results/figures/ru/composition_stress.png.
The worked example is drawn by scripts/plot_example.py --lang ru.

Example: python scripts/plot_figures_ru.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from s2m.io import read_csv
from s2m.viz import ARM_LABEL_RU, ARM_STYLE, MUTED, decimal_comma, setup

OUT = Path("results/figures/ru")
CLASSES = ("indoor_plant", "bike")
COMMA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))


def composition():
    rows = read_csv("results/tables/composition_a03_L5.csv")
    alpha = 0.3
    styles = {"aligned": ("--", "там же, где плохие метки"), "random": (":", "случайно"),
              "disjoint": ("-", "там, где метки хорошие")}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, cls, title in zip(axes, CLASSES, ("растение", "велосипед")):
        for arm in ("separate", "joint"):
            _, colour, marker = ARM_STYLE[arm]
            for dep, (ls, dep_ru) in styles.items():
                t = [r for r in rows if r["class"] == cls and r["arm"] == arm and r["dependence"] == dep]
                ax.plot([float(r["frac_fail"]) for r in t], [float(r["coverage_mean"]) for r in t], ls=ls,
                        color=colour, marker=marker, ms=4, label=f"{ARM_LABEL_RU[arm]}; сбои {dep_ru}")
        ax.axhline(1 - alpha, color=MUTED, lw=1)
        ax.text(0, 1 - alpha - 0.005, "цель 0,7", va="top", fontsize=7, color=MUTED)
        ax.axhline(1 - 2 * alpha, color=MUTED, lw=1, ls=":")
        ax.text(0, 1 - 2 * alpha + 0.005, "граница 1 − 2α для суммы запасов", va="bottom", fontsize=7, color=MUTED)
        ax.set_title(title)
        ax.set_xlabel("доля сцен со сбоем локализации")
        ax.xaxis.set_major_formatter(COMMA)
    axes[0].set_ylabel("покрытие на тестовых сценах")
    decimal_comma(*axes)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "composition_stress.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    setup()
    composition()
    print("saved", sorted(p.name for p in OUT.glob("*.png")))
