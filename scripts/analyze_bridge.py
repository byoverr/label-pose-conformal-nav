"""Sampling-based planners on calibrated (tightened) maps: solved-within-budget vs inflation.

Reads results/tables/planner_bridge*.csv (parts are concatenated); writes
results/tables/planner_bridge_summary.csv and results/figures/planner_bridge.png.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from s2m.viz import INK, MUTED, setup

PLANNERS = {"rrt_connect": ("RRT-Connect, uniform sampling", "#eb6834", "s"),
            "angle_zone": ("angle-limited zone, published rule", "#1baf7a", "^"),
            "angle_zone_adaptive": ("angle-limited zone, widen on blocked extension", "#2a78d6", "o")}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="'' for all avoid classes, '_pb' for plant + bike")
    args = ap.parse_args()
    rows = [r for p in sorted(Path("results/tables").glob(f"planner_bridge{args.tag}_[0-9]*.csv"))
            for r in csv.DictReader(open(p))]
    scales = sorted({float(r["scale"]) for r in rows})
    summary = []
    for m in scales:
        rs = [r for r in rows if float(r["scale"]) == m]
        tasks = {(r["scene"], r["task"]) for r in rs}
        solvable = {(r["scene"], r["task"]) for r in rs if r["solvable"] == "True"}
        entry = {"scale": m, "tasks": len(tasks), "solvable_frac": len(solvable) / max(len(tasks), 1),
                 "free_frac": float(np.mean([float(r["free_frac"]) for r in rs]))}
        for pl in PLANNERS:
            ps = [r for r in rs if r.get("planner") == pl]
            entry[f"{pl}_solved"] = float(np.mean([r["solved"] == "True" for r in ps])) if ps else np.nan
            its = [int(r["iterations"]) for r in ps if r["solved"] == "True"]
            entry[f"{pl}_iters_median"] = float(np.median(its)) if its else np.nan
            lr = [float(r["length_ratio"]) for r in ps if r["solved"] == "True"]
            entry[f"{pl}_length_ratio_median"] = float(np.median(lr)) if lr else np.nan
            ts = [float(r["time_s"]) for r in ps]
            entry[f"{pl}_time_median_s"] = float(np.median(ts)) if ts else np.nan
        summary.append(entry)
        print(entry)
    with open(f"results/tables/planner_bridge{args.tag}_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(summary)

    # Anytime view: a run stops at its first solution, so "solved within budget B" is
    # iterations <= B for every B below the 1500-extension budget actually used.
    budgets = np.unique(np.round(np.logspace(0, np.log10(1500), 40)).astype(int))
    ref_scale = 1.0 if 1.0 in scales else scales[-1]

    setup()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    ax = axes[0]
    ax.plot(scales, [s["solvable_frac"] for s in summary], color=INK, marker="x", label="solvable (exact grid search)")
    ax.plot(scales, [s["free_frac"] for s in summary], color=MUTED, ls="--", marker="x", label="free space left")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Inflation narrows the free space")
    ax.set_ylabel("fraction")
    ax.legend(loc="upper right", fontsize=7)
    ax = axes[1]
    for pl, (label, colour, marker) in PLANNERS.items():
        ps = [r for r in rows if float(r["scale"]) == ref_scale and r.get("planner") == pl]
        its = np.array([int(r["iterations"]) if r["solved"] == "True" else np.inf for r in ps])
        ax.plot(budgets, [(its <= b).mean() for b in budgets], color=colour, label=label)
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("budget B, tree extensions")
    ax.set_ylabel("solved within B (solvable tasks)")
    ax.set_title(f"Anytime success at m = {ref_scale:g}")
    ax.legend(loc="upper left", fontsize=7)
    ax = axes[2]
    for pl, (label, colour, marker) in PLANNERS.items():
        ax.plot(scales, [s[f"{pl}_length_ratio_median"] for s in summary], color=colour, marker=marker, label=label)
    ax.set_ylabel("first path length / shortest (median)")
    ax.set_title("Quality of the first solution")
    for ax in (axes[0], axes[2]):
        ax.set_xlabel("keep-out radius scale m (1 = calibrated)")
        ax.set_xticks(scales)
    fig.tight_layout()
    fig.savefig(f"results/figures/planner_bridge{args.tag}.png")
