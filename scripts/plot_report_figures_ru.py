"""Russian versions of the report figures, redrawn from the result tables (no recomputation).

Writes results/figures/ru/coverage_headline.png and results/figures/ru/composition_stress.png.
The worked example is drawn by scripts/plot_example.py --lang ru.

Example: python scripts/plot_report_figures_ru.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from s2m.analysis import load_scores
from s2m.io import DEV_SCENES, LEVELS, level_labels, read_csv
from s2m.viz import ARM_LABEL_RU, ARM_STYLE, INK, MUTED, arm_line, decimal_comma, setup

OUT = Path("results/figures/ru")
CLASSES = ("indoor_plant", "bike")
COMMA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))


def ru_level(note: str) -> str:
    return "точная поза" if note == "GT pose" else note.replace(".", ",").replace(" m", " м").replace(" cm", " см")


def coverage_headline():
    rows = read_csv("results/tables/coverage.csv")
    note = level_labels(load_scores("results/scores", exclude=DEV_SCENES))
    get = lambda lv, arm, key, classes=CLASSES: np.nanmean([float(r[key]) for r in rows if r["level"] == lv
                                                            and r["arm"] == arm and r["class"] in classes])
    x = np.arange(len(LEVELS))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    ax = axes[0]
    ax.axhline(0.9, color=MUTED, ls="--", lw=1)
    ax.text(0.0, 0.88, "цель 1 − α = 0,9", color=MUTED, fontsize=8, va="top")
    for arm in ("joint", "separate", "label_only", "pose_only", "label_cell", "uncalibrated"):
        arm_line(ax, arm, x, [get(lv, arm, "coverage_mean") for lv in LEVELS], lang="ru")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("покрытие (отказ = покрыто)")
    ax.set_title("а) покрытие, растение и велосипед")
    ax = axes[1]
    for arm in ("joint", "separate", "label_only", "pose_only"):
        arm_line(ax, arm, x, [get(lv, arm, "radius_median") for lv in LEVELS], lang="ru")
    ax.set_ylabel("запас, м (медиана)")
    ax.set_title("б) цена гарантии")
    ax = axes[2]
    for cls, marker, name in (("bike", "o", "велосипед"), ("indoor_plant", "s", "растение")):
        ax.plot(x, [1 - get(lv, "joint", "abstain_rate", (cls,)) for lv in LEVELS], color=INK, marker=marker,
                label=f"{name}: сертификат выдан")
        ax.plot(x, [get(lv, "joint", "coverage_cert", (cls,)) for lv in LEVELS], color=MUTED, marker=marker,
                ls="--", label=f"{name}: покрытие при сертификате")
    ax.axhline(0.9, color=MUTED, ls="--", lw=1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("доля разбиений / покрытие")
    ax.set_title("в) совместный запас: отказы")
    ax.legend(loc="lower left", fontsize=7)
    for ax in axes:
        ax.set_xticks(x, [f"{lv}\n{ru_level(note[lv])}" for lv in LEVELS])
        ax.set_xlabel("уровень дрейфа (медианная ATE)")
    decimal_comma(*axes)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=7, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "coverage_headline.png")


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


def risk_curve():
    """Certified risk vs share of tasks with a certified path (scripts/analyze_risk.py)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    names = {"L0": "точная поза", "L2": "2 см", "L3": "7 см", "L4": "45 см"}
    for ax, kind, title in zip(axes, ("pb", "rand"), ("трудные задачи", "случайные задачи")):
        path = Path(f"results/tables/risk_{kind}.csv")
        if not path.exists():
            continue
        rows = read_csv(path)
        for lv, ls in zip(("L0", "L2", "L3", "L4"), ("-", "--", "-.", ":")):
            t = [r for r in rows if r["level"] == lv]
            ax.plot([float(r["target_risk"]) for r in t], [float(r["certified"]) for r in t], ls=ls, color=INK,
                    marker="o", ms=4, label=names[lv])
            ax.plot([float(r["target_risk"]) for r in t], [float(r["unsafe_and_certified"]) for r in t], ls=ls,
                    color=MUTED, lw=1)
        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":")
        ax.set_title(title)
        ax.set_xlabel("допустимый риск a (оба класса)")
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("доля задач")
    axes[0].legend(fontsize=7, loc="upper left", title="дрейф", title_fontsize=7)
    decimal_comma(*axes)
    for ax in axes:
        ax.xaxis.set_major_formatter(COMMA)
    fig.tight_layout()
    fig.savefig(OUT / "risk_curve.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    setup()
    coverage_headline()
    composition()
    risk_curve()
    print("saved", sorted(p.name for p in OUT.glob("*.png")))
