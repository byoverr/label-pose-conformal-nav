"""Sampling-based planners on calibrated (tightened) maps: solved-within-budget vs inflation.

Reads results/tables/planner_bridge.csv; writes results/tables/planner_bridge_summary.csv and
results/figures/planner_bridge.png.
"""

import csv

import matplotlib.pyplot as plt
import numpy as np

from s2m.viz import INK, MUTED, setup

PLANNERS = {"rrt_connect": ("RRT-Connect (uniform sampling)", "#eb6834", "s"),
            "angle_zone": ("Bi-RRT with angle-limited zone", "#2a78d6", "o")}

if __name__ == "__main__":
    rows = list(csv.DictReader(open("results/tables/planner_bridge.csv")))
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
    with open("results/tables/planner_bridge_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(summary)

    setup()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for pl, (label, colour, marker) in PLANNERS.items():
        axes[0].plot(scales, [s[f"{pl}_solved"] for s in summary], color=colour, marker=marker, label=label)
        axes[1].plot(scales, [s[f"{pl}_iters_median"] for s in summary], color=colour, marker=marker, label=label)
    axes[0].plot(scales, [s["solvable_frac"] for s in summary], color=MUTED, ls="--", marker="x",
                 label="solvable at all (exact grid search)")
    axes[0].set_ylabel("fraction of tasks")
    axes[0].set_title("Solved within 1500 extensions (solvable tasks)")
    axes[1].set_ylabel("extensions to first solution (median)")
    axes[1].set_title("Search effort")
    for ax in axes:
        ax.set_xlabel("keep-out radius scale m (1 = calibrated joint radius)")
        ax.set_xticks(scales)
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig("results/figures/planner_bridge.png")
