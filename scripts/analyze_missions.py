"""Mission outcomes per arm and drift level, plus H4 (map mIoU vs mission safety).

Reads a results/missions*/ directory and results/tables/scene_metrics.csv; writes
results/tables/missions<tag>.csv, results/figures/missions_vs_drift<tag>.png and (unless --no-h4)
results/figures/miou_vs_violations<tag>.png, results/tables/h4_spearman<tag>.txt.

Example: python scripts/analyze_missions.py --dir results/missions_pb --tag _pb
         python scripts/analyze_missions.py --dir results/missions_pb_vo --tag _pb_vo --levels L0 VO2 VO4 \
                --scores results/scores results/scores_vo --no-h4
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from s2m.analysis import load_scores
from s2m.io import DEV_SCENES, LEVELS, level_labels, mission_rows, rate, read_csv, write_csv
from s2m.viz import arm_line, setup

ARMS = ("joint", "separate", "label_only", "pose_only", "uncalibrated", "label_cell", "oracle")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("results/missions"))
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _pb")
    ap.add_argument("--levels", nargs="+", default=list(LEVELS))
    ap.add_argument("--scores", type=Path, nargs="+", default=[Path("results/scores")],
                    help="score files, only used to label drift levels by their median ATE")
    ap.add_argument("--no-h4", action="store_true", help="skip the mIoU vs violations analysis")
    args = ap.parse_args()
    LEVELS = tuple(args.levels)
    rows = mission_rows(args.dir)
    note = level_labels([r for d in args.scores for r in load_scores(d, exclude=DEV_SCENES)])
    print(f"{len(rows)} mission rows from {len({r['scene'] for r in rows})} scenes")

    table = []
    for lv in LEVELS:
        for arm in ARMS:
            rs = [r for r in rows if r["level"] == lv and r["arm"] == arm]
            planned = [r for r in rs if r["planned"] == "True"]
            ratio = [float(r["length"]) / float(r["oracle_length"]) for r in planned
                     if r["oracle_length"] not in ("", "nan") and np.isfinite(float(r["oracle_length"]))]
            table.append({
                "level": lv, "arm": arm, "n": len(rs),
                "planned": rate(rs, "planned"), "violation": rate(rs, "violation"),
                "collision": rate(rs, "collision"), "success": rate(rs, "success"),
                "abstained": rate(rs, "abstained"),
                "violation_given_planned": rate(planned, "violation"),
                "length_ratio_median": float(np.median(ratio)) if ratio else np.nan,
            })
    write_csv(f"results/tables/missions{args.tag}.csv", table)
    for t in table:
        print(f"{t['level']} {t['arm']:12s} plan {t['planned']:.2f} viol {t['violation']:.3f} "
              f"(|plan {t['violation_given_planned']:.3f}) coll {t['collision']:.3f} succ {t['success']:.2f} "
              f"abst {t['abstained']:.2f} len {t['length_ratio_median']:.2f}")

    setup()
    x = np.arange(len(LEVELS))
    xt = [f"{lv}\n{note.get(lv, '')}" for lv in LEVELS]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for arm in ARMS:
        get = lambda k: [next(t[k] for t in table if t["level"] == lv and t["arm"] == arm) for lv in LEVELS]
        arm_line(axes[0], arm, x, get("violation_given_planned"))
        arm_line(axes[1], arm, x, get("success"))
        arm_line(axes[2], arm, x, get("planned"))
    for ax, title in zip(axes, ("violations among planned paths", "mission success", "path found")):
        ax.set_title(title)
        ax.set_xticks(x, xt)
        ax.set_xlabel("pose error (median ATE over scenes)")
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("fraction of tasks")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(f"results/figures/missions_vs_drift{args.tag}.png")

    if args.no_h4:
        raise SystemExit
    # H4: map quality (mIoU at GT poses) vs safety of the uncalibrated planner at GT poses.
    metrics = {r["scene"]: r for r in read_csv("results/tables/scene_metrics.csv")}
    pts = []
    for scene in sorted({r["scene"] for r in rows}):
        rs = [r for r in rows if r["scene"] == scene and r["level"] == "L0" and r["arm"] == "uncalibrated"
              and r["planned"] == "True"]
        if rs and scene in metrics:
            pts.append((float(metrics[scene]["miou"]), float(metrics[scene]["avoid_iou"]), rate(rs, "violation")))
    pts = np.array(pts)
    rho_m, p_m = spearmanr(pts[:, 0], pts[:, 2])
    rho_a, p_a = spearmanr(pts[:, 1], pts[:, 2])
    print(f"H4: Spearman(mIoU, violations) = {rho_m:.2f} (p={p_m:.2f}); "
          f"Spearman(avoid-class IoU, violations) = {rho_a:.2f} (p={p_a:.2f}); n = {len(pts)} scenes")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for ax, col, name, rho in ((axes[0], 0, "mIoU (all classes)", rho_m), (axes[1], 1, "IoU of avoid classes", rho_a)):
        ax.scatter(pts[:, col], pts[:, 2], color="#2a78d6", s=28)
        ax.set_xlabel(f"{name} at GT poses")
        ax.set_title(f"Spearman ρ = {rho:.2f}")
    axes[0].set_ylabel("violation rate, uncalibrated planner")
    fig.suptitle("Does map quality predict mission safety? (one point per scene, L0)")
    fig.tight_layout()
    fig.savefig(f"results/figures/miou_vs_violations{args.tag}.png")
    with open(f"results/tables/h4_spearman{args.tag}.txt", "w") as f:
        f.write(f"n_scenes={len(pts)} rho_miou={rho_m:.3f} p={p_m:.3f} rho_avoid_iou={rho_a:.3f} p={p_a:.3f}\n")
