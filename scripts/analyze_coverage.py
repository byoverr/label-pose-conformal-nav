"""Per-class coverage, radius and abstention of every calibration arm vs drift level.

Outputs results/tables/coverage.csv, results/figures/coverage_by_class.png and the headline
figure results/figures/coverage_headline.png.

Example: python scripts/analyze_coverage.py --alpha 0.1 --splits 200
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from s2m.analysis import ARMS, LAM0, load_scores, run_splits
from s2m.entities import AVOID_CLASSES
from s2m.io import DEV_SCENES, LEVELS, level_labels, write_csv
from s2m.viz import MUTED, arm_line, setup

RADIUS_ARMS = ("label_only", "pose_only", "separate", "joint")


def summarise(stats, cls, arm, key, fn):
    v = np.array(stats[cls][arm][key], float)
    return float(fn(v)) if np.isfinite(v).any() else np.nan


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=Path("results/scores"))
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--n-cal", type=int, default=None, help="default: ~60% of scenes, at least 1/alpha - 1")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    rows = load_scores(args.scores, exclude=DEV_SCENES)
    scenes = sorted({r["scene"] for r in rows})
    n_cal = args.n_cal or max(int(np.ceil(1 / args.alpha - 1)), round(0.6 * len(scenes)))
    print(f"{len(scenes)} scenes (dev excluded), n_cal = {n_cal}, alpha = {args.alpha}, lam0 = {LAM0}")

    stats = {lv: run_splits(rows, lv, args.alpha, n_cal, args.splits) for lv in LEVELS}
    table = []
    for lv in LEVELS:
        for cls in AVOID_CLASSES:
            for arm in ARMS:
                table.append({
                    "level": lv, "class": cls, "arm": arm,
                    "coverage_mean": summarise(stats[lv], cls, arm, "coverage", np.mean),
                    "coverage_p10": summarise(stats[lv], cls, arm, "coverage", lambda v: np.percentile(v, 10)),
                    "abstain_rate": summarise(stats[lv], cls, arm, "abstain", np.mean),
                    "radius_median": summarise(stats[lv], cls, arm, "radius", np.nanmedian),
                })
    write_csv(f"results/tables/coverage{args.tag}.csv", table)
    for r in table:
        if r["level"] in ("L0", "L2", "L4"):
            print(f"{r['level']} {r['class']:12s} {r['arm']:12s} cov {r['coverage_mean']:.3f} "
                  f"(p10 {r['coverage_p10']:.3f}) abstain {r['abstain_rate']:.2f} radius {r['radius_median']:.2f}")

    setup()
    x = np.arange(len(LEVELS))
    note = level_labels(rows)
    xt = [f"{lv}\n{note[lv]}" for lv in LEVELS]
    fig, axes = plt.subplots(3, len(AVOID_CLASSES), figsize=(10, 8.2), sharex=True)
    for j, cls in enumerate(AVOID_CLASSES):
        ax = axes[0, j]
        ax.axhline(1 - args.alpha, color=MUTED, ls="--", lw=1)
        for arm in ARMS:
            arm_line(ax, arm, x, [summarise(stats[lv], cls, arm, "coverage", np.mean) for lv in LEVELS])
        ax.set_title(cls.replace("_", " "))
        ax.set_ylim(-0.02, 1.02)
        ax = axes[1, j]
        for arm in RADIUS_ARMS:
            arm_line(ax, arm, x, [summarise(stats[lv], cls, arm, "radius", np.nanmedian) for lv in LEVELS])
        ax = axes[2, j]
        for arm in RADIUS_ARMS:
            arm_line(ax, arm, x, [summarise(stats[lv], cls, arm, "abstain", np.mean) for lv in LEVELS])
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(x, xt)
    axes[0, 0].set_ylabel(f"coverage (target {1 - args.alpha:.2f})")
    axes[1, 0].set_ylabel("keep-out radius, m (median)")
    axes[2, 0].set_ylabel("abstention rate")
    for ax in axes[2]:
        ax.set_xlabel("pose drift level (median ATE over scenes)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Per-class guarantees under pose drift ({len(scenes)} scenes, {args.splits} splits, "
                 f"n_cal = {n_cal}, α = {args.alpha})")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(f"results/figures/coverage_by_class{args.tag}.png")

    # Headline: the two classes the detector localises reliably; coverage and the price paid for it.
    good = ("indoor_plant", "bike")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    ax.axhline(1 - args.alpha, color=MUTED, ls="--", lw=1)
    ax.text(0.0, 1 - args.alpha - 0.02, f"target 1−α = {1 - args.alpha:.2f}", color=MUTED, fontsize=8, va="top")
    for arm in ("joint", "separate", "label_only", "pose_only", "uncalibrated"):
        y = [np.mean([summarise(stats[lv], c, arm, "coverage", np.mean) for c in good]) for lv in LEVELS]
        arm_line(ax, arm, x, y)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("test coverage of true footprints")
    ax.set_title("Coverage (plants and bikes)")
    ax.legend(loc="center left", fontsize=7)
    ax = axes[1]
    for arm in ("joint", "separate", "label_only", "pose_only"):
        y = [np.mean([summarise(stats[lv], c, arm, "radius", np.nanmedian) for c in good]) for lv in LEVELS]
        arm_line(ax, arm, x, y)
    ax.set_ylabel("calibrated keep-out radius, m")
    ax.set_title("Price of the guarantee: extra keep-out distance")
    for ax in axes:
        ax.set_xticks(x, xt)
        ax.set_xlabel("pose drift level (median ATE over scenes)")
    fig.tight_layout()
    fig.savefig(f"results/figures/coverage_headline{args.tag}.png")
    print("saved tables and figures")
