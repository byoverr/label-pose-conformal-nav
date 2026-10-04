"""Robustness of the main result to the three simplifications of the main experiment.

  sam : instance masks from MobileSAM instead of box + depth masks (results/scores_sam)
  vo  : real pose error from RGB-D visual odometry instead of synthetic drift (results/scores_vo)
  ood : calibrate on all ReplicaCAD scenes, test on HM3D scenes (results/scores_hm3d)

Writes results/tables/variants_{sam,vo,ood}.csv and results/figures/variants.png.

Example: python scripts/analyze_variants.py --alpha 0.1
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from s2m.analysis import ARMS, LAM0, calibrate_class, covers, load_scores, miss_at, run_splits
from s2m.io import DEV_SCENES, write_csv
from s2m.viz import ARM_STYLE, MUTED, setup

CLASSES = ("indoor_plant", "bike")
LEVELS = ("L0", "L1", "L2", "L3", "L4", "L5")
SHOW_ARMS = ("uncalibrated", "label_only", "pose_only", "separate", "joint")


def summary(stats, cls, arm):
    st = stats[cls][arm]
    r = np.array(st["radius"], float)
    cov, ab = np.array(st["coverage"], float), np.array(st["abstain"], float)
    return {"coverage_mean": float(cov.mean()), "abstain_rate": float(ab.mean()),
            "coverage_cert": float(cov[ab == 0].mean()) if (ab == 0).any() else np.nan,
            "radius_median": float(np.nanmedian(r)) if np.isfinite(r).any() else np.nan}


def ood_stats(cal_rows, test_rows, level, alpha, cls, draws, rng):
    """Calibrate on every ReplicaCAD scene (one random realization each), test on the OOD scenes
    that contain the class. Returns per-arm mean coverage, abstention and median radius."""
    by = {}
    for r in cal_rows:
        by.setdefault((r["scene"], r["level"]), []).append(r)
    scenes = sorted({s for s, _ in by})
    test = [r for r in test_rows if r["level"] == level and r[f"n_ent_{cls}"] > 0]
    acc = {a: {"coverage": [], "abstain": [], "radius": []} for a in ARMS}
    for _ in range(draws):
        cal = [by[(s, level)][rng.integers(len(by[(s, level)]))] for s in scenes]
        cal_L0 = [by[(s, "L0")][0] for s in scenes]
        for arm, p in calibrate_class(cal, cal_L0, cls, alpha).items():
            acc[arm]["coverage"].append(np.mean([covers(t, cls, p) for t in test]))
            acc[arm]["abstain"].append(float(p is None))
            acc[arm]["radius"].append(np.nan if p is None else p.radius)
    return {cls: acc}, len({r["scene"] for r in test})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--n-cal", type=int, default=13)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    base = load_scores(Path("results/scores"), exclude=DEV_SCENES)
    panels = {}

    # SAM masks vs box + depth masks: same splits, same levels.
    if Path("results/scores_sam").exists():
        sam = load_scores(Path("results/scores_sam"), exclude=DEV_SCENES)
        table = []
        for name, rows in (("box", base), ("sam", sam)):
            for lv in LEVELS:
                st = run_splits(rows, lv, args.alpha, args.n_cal, args.splits, classes=CLASSES)
                for cls in CLASSES:
                    for arm in ARMS:
                        table.append({"masks": name, "level": lv, "class": cls, "arm": arm, **summary(st, cls, arm)})
            for cls in CLASSES:
                m = [miss_at(r, cls, LAM0) for r in rows if r["level"] == "L0"]
                print(f"{name}: L0 label miss {cls}: median {np.median(m):.2f} m, max {np.max(m):.2f} m")
        write_csv("results/tables/variants_sam.csv", table)
        panels["sam"] = table

    # Real visual odometry: one realization per scene and VO setting.
    if Path("results/scores_vo").exists():
        vo = load_scores(Path("results/scores_vo"), exclude=DEV_SCENES)
        vo_scenes = {r["scene"] for r in vo}
        if len(vo_scenes) < 15:
            print(f"VO: only {len(vo_scenes)} scenes scored so far, skipped")
            vo = []
    if Path("results/scores_vo").exists() and vo:
        rows = [r for r in base if r["level"] == "L0" and r["scene"] in vo_scenes] + vo
        n = len(vo_scenes)
        table = []
        for lv in sorted({r["level"] for r in vo}):
            ates = np.array([r["ate"] for r in vo if r["level"] == lv])
            n_cal = max(int(np.ceil(1 / args.alpha - 1)), round(0.6 * n))
            st = run_splits(rows, lv, args.alpha, n_cal, args.splits, classes=CLASSES)
            for cls in CLASSES:
                for arm in ARMS:
                    table.append({"level": lv, "n_scenes": n, "n_cal": n_cal, "ate_median": float(np.median(ates)),
                                  "ate_max": float(ates.max()), "class": cls, "arm": arm, **summary(st, cls, arm)})
            print(f"{lv}: {n} scenes, ATE median {np.median(ates) * 100:.1f} cm, "
                  f"quartiles {np.percentile(ates, 25) * 100:.1f}-{np.percentile(ates, 75) * 100:.1f} cm, max {ates.max():.2f} m")
        write_csv("results/tables/variants_vo.csv", table)
        panels["vo"] = table

    # Out of distribution: ReplicaCAD calibration, HM3D test.
    if Path("results/scores_hm3d").exists():
        hm = load_scores(Path("results/scores_hm3d"))
        table = []
        for lv in LEVELS:
            for cls in CLASSES:
                st, n_test = ood_stats(base, hm, lv, args.alpha, cls, 50, rng)
                if n_test == 0:
                    continue
                for arm in ARMS:
                    table.append({"level": lv, "class": cls, "n_test_scenes": n_test, "arm": arm, **summary(st, cls, arm)})
        write_csv("results/tables/variants_ood.csv", table)
        panels["ood"] = table

    for name, table in panels.items():
        print(f"== {name}")
        for t in table:
            if t["arm"] in ("label_only", "separate", "joint") and t["level"] in ("L0", "L2", "L3", "L4", "VO2", "VO4"):
                print("  " + " ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in t.items()))

    # Figure: coverage (top) and keep-out radius (bottom) of every arm in each variant.
    setup()
    cols = [k for k in ("sam", "vo", "ood") if k in panels]
    if not cols:
        raise SystemExit
    fig, axes = plt.subplots(2, len(cols), figsize=(4.2 * len(cols), 5.6), squeeze=False,
                             gridspec_kw={"height_ratios": [1.1, 1]})
    for col, name in enumerate(cols):
        t = panels[name]
        short = {"indoor_plant": "plant", "bike": "bike"}
        if name == "sam":
            groups = [(f"{m}\n{short[c]}", [r for r in t if r["masks"] == m and r["class"] == c and r["level"] == "L3"])
                      for c in CLASSES for m in ("box", "sam")]
            title = "box + depth masks vs MobileSAM (drift L3)"
        elif name == "vo":
            groups = [(f"{lv}\n{short[c]}", [r for r in t if r["level"] == lv and r["class"] == c])
                      for c in CLASSES for lv in sorted({r["level"] for r in t})]
            title = "real RGB-D visual odometry (VO2, VO4 = every 2nd, 4th frame)"
        else:
            groups = [(f"{lv}\n{short[c]}", [r for r in t if r["level"] == lv and r["class"] == c])
                      for c in CLASSES for lv in ("L0", "L3") if any(r["class"] == c for r in t)]
            title = f"calibrated on ReplicaCAD, tested on HM3D ({t[0]['n_test_scenes']} scenes with a visible plant)"
        x = np.arange(len(groups))
        w = 0.8 / len(SHOW_ARMS)
        for j, arm in enumerate(SHOW_ARMS):
            label, colour, marker = ARM_STYLE[arm]
            xs = x + (j - (len(SHOW_ARMS) - 1) / 2) * w
            get = lambda key: np.array([next((r[key] for r in g if r["arm"] == arm), np.nan) for _, g in groups], float)
            cov, rad, abst = get("coverage_mean"), get("radius_median"), get("abstain_rate")
            axes[0][col].bar(xs, cov, width=w * 0.9, color=colour, label=label)
            if arm == "uncalibrated":
                continue
            ok = abst < 0.5
            axes[1][col].bar(xs[ok], np.nan_to_num(rad[ok]), width=w * 0.9, color=colour)
            axes[1][col].bar(xs[~ok], np.where(np.isfinite(rad[~ok]), rad[~ok], 3.0), width=w * 0.9,
                             color="none", edgecolor=colour, hatch="////", lw=0.8)
        axes[0][col].axhline(1 - args.alpha, color=MUTED, lw=1)
        axes[0][col].set_ylim(0, 1.05)
        axes[0][col].set_title(title, fontsize=8)
        for ax in axes[:, col]:
            ax.set_xticks(x, [g[0] for g in groups], fontsize=7)
        axes[1][col].set_ylim(0, 3.1)
    axes[0][0].set_ylabel("coverage")
    axes[1][0].set_ylabel("keep-out radius, m\n(hatched: abstains in >= half of splits)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06), fontsize=7)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig("results/figures/variants.png")
    print("saved results/figures/variants.png")
