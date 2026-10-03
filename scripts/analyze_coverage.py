"""Coverage of every calibration arm vs drift level over random scene splits.

Outputs results/tables/coverage.csv and results/figures/coverage_vs_drift.png,
results/figures/radius_vs_drift.png.

Example: python scripts/analyze_coverage.py --alpha 0.1 --splits 200
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from s2m.analysis import ARMS, LAM0, load_scores, run_splits
from s2m.conformal import coverage_beta_params
from s2m.viz import ARM_STYLE, MUTED, arm_line, setup

DEV_SCENES = ("apt_0",)
LEVELS = ("L0", "L1", "L2", "L3", "L4", "L5")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=Path("results/scores"))
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--n-cal", type=int, default=None, help="default: about 60% of the scenes")
    args = ap.parse_args()

    rows = load_scores(args.scores, exclude=DEV_SCENES)
    scenes = sorted({r["scene"] for r in rows})
    n_cal = args.n_cal or max(int(np.ceil(1 / args.alpha - 1)), round(0.6 * len(scenes)))
    print(f"{len(scenes)} scenes (dev excluded), n_cal = {n_cal}, alpha = {args.alpha}, lam0 = {LAM0}")

    table, stats = [], {}
    for level in LEVELS:
        stats[level] = run_splits(rows, level, args.alpha, n_cal, args.splits)
        for arm in ARMS:
            st = stats[level][arm]
            cov = np.array(st["coverage"], float)
            table.append({
                "level": level, "arm": arm,
                "coverage_mean": np.nanmean(cov) if np.isfinite(cov).any() else np.nan,
                "coverage_p10": np.nanpercentile(cov, 10) if np.isfinite(cov).any() else np.nan,
                "abstain_rate": float(np.mean(st["abstain"])),
                "radius_median": float(np.nanmedian(st["radius"])) if np.isfinite(st["radius"]).any() else np.nan,
                "lam_median": float(np.nanmedian(st["lam"])) if np.isfinite(st["lam"]).any() else np.nan,
            })

    out = Path("results/tables")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "coverage.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    for r in table:
        print(f"{r['level']} {r['arm']:13s} coverage {r['coverage_mean']:.3f} (p10 {r['coverage_p10']:.3f}) "
              f"abstain {r['abstain_rate']:.2f} radius {r['radius_median']:.2f} lam {r['lam_median']:.2f}")

    setup()
    x = np.arange(len(LEVELS))
    a, b = coverage_beta_params(n_cal, args.alpha)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.axhline(1 - args.alpha, color=MUTED, ls="--", lw=1)
    ax.text(len(LEVELS) - 0.95, 1 - args.alpha + 0.01, f"target 1−α = {1 - args.alpha:.2f}",
            color=MUTED, fontsize=8, ha="right", va="bottom")
    for arm in ARMS:
        y = [np.nanmean(stats[lv][arm]["coverage"]) if not np.all(np.isnan(stats[lv][arm]["coverage"]))
             else np.nan for lv in LEVELS]
        arm_line(ax, arm, x, y)
    joint = np.array([[np.nanpercentile(stats[lv]["joint"]["coverage"], q) for q in (10, 90)] for lv in LEVELS])
    ax.fill_between(x, joint[:, 0], joint[:, 1], color=ARM_STYLE["joint"][1], alpha=0.15, lw=0,
                    label="joint: 10–90% over splits")
    ax.set_xticks(x, [f"{lv}" for lv in LEVELS])
    ax.set_xlabel("pose drift level (L0 = ground truth … L5 = stress test)")
    ax.set_ylabel("test coverage of avoid-class footprints")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Who keeps the promised coverage under pose drift? ({len(scenes)} ReplicaCAD scenes, "
                 f"{args.splits} splits)")
    ax.legend(loc="lower left", fontsize=7)
    fig.savefig("results/figures/coverage_vs_drift.png")

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for arm in ("joint", "separate", "bonferroni"):
        y = [np.nanmedian(stats[lv][arm]["radius"]) if not np.all(np.isnan(stats[lv][arm]["radius"]))
             else np.nan for lv in LEVELS]
        arm_line(ax, arm, x, y)
    ax.set_xticks(x, LEVELS)
    ax.set_xlabel("pose drift level")
    ax.set_ylabel("calibrated keep-out radius, m")
    ax.set_title("Extra keep-out distance chosen by calibration (median over splits)")
    ax.legend(loc="upper left", fontsize=7)
    fig.savefig("results/figures/radius_vs_drift.png")
    print("saved results/tables/coverage.csv and figures")
