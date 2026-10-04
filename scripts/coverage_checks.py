"""Checks of the coverage result that the main table does not show.

  reference : the Beta distribution of split-conformal coverage for the calibration sizes used;
  scores    : how often the premise of Proposition 2 (s_joint <= s_label + s_pose) fails, how often
              a footprint is pushed off the map, how often the label score is uncoverable;
  lam0      : sensitivity of the joint arm to the region threshold lam0;
  n_cal     : keep-out radius and abstention of the joint arm vs the number of calibration scenes;
  both      : coverage of plant and bike at once (what a two-class mission needs);
  layout    : leave-one-layout-out: calibrate on the other layouts, test on one (a mild shift);
  bootstrap : scene-bootstrap 90% intervals for the coverage of every arm (copies of a resampled
              scene always land on the same side of a split).

Writes results/tables/checks_<name>.csv.

Example: python scripts/coverage_checks.py --boot 200
"""

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.stats import beta

from s2m.analysis import ARMS, LAM0, calibrate_class, covers, load_scores, miss_at, run_splits
from s2m.conformal import UNCOVERABLE
from s2m.experiment import R_MAX
from s2m.io import DEV_SCENES, LEVELS, write_csv

CLASSES = ("indoor_plant", "bike")


def layout(scene: str) -> str:
    return scene.split("_staging")[0] if "_staging" in scene else scene


def summary(st):
    cov, ab, rad = (np.array(st[k], float) for k in ("coverage", "abstain", "radius"))
    return {"coverage": float(cov.mean()), "coverage_cert": float(cov[ab == 0].mean()) if (ab == 0).any() else np.nan,
            "abstain": float(ab.mean()), "radius_median": float(np.nanmedian(rad)) if np.isfinite(rad).any() else np.nan}


def grouped_splits(groups: list[list[dict]], level, alpha, n_cal, n_splits, rng, classes=CLASSES):
    """run_splits over groups of scene rows (a group = one scene, possibly repeated by the bootstrap):
    groups are assigned whole to calibration until it holds n_cal scenes."""
    stats = {c: {a: {"coverage": [], "abstain": [], "radius": []} for a in ARMS} for c in classes}
    for _ in range(n_splits):
        order = rng.permutation(len(groups))
        cal, cal_L0, test, n = [], [], [], 0
        for g in order:
            if n < n_cal:
                for rows in groups[g]:
                    by = [r for r in rows if r["level"] == level]
                    cal.append(by[rng.integers(len(by))])
                    cal_L0.append(next(r for r in rows if r["level"] == "L0"))
                    n += 1
            else:
                test += [r for rows in groups[g] for r in rows if r["level"] == level]
        for cls in classes:
            for arm, p in calibrate_class(cal, cal_L0, cls, alpha).items():
                st = stats[cls][arm]
                st["coverage"].append(float(np.mean([covers(t, cls, p) for t in test])))
                st["abstain"].append(float(p is None))
                st["radius"].append(np.nan if p is None else p.radius)
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=Path("results/scores"))
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    run = lambda name: args.only is None or name in args.only
    rows = load_scores(args.scores, exclude=DEV_SCENES)
    scenes = sorted({r["scene"] for r in rows})
    by_scene = {s: [r for r in rows if r["scene"] == s] for s in scenes}
    n_cal = max(int(np.ceil(1 / args.alpha - 1)), round(0.6 * len(scenes)))
    out = Path("results/tables")

    if run("reference"):
        table = []
        for n in (13, 19, 20):
            for a in (0.05, 0.1, 0.2, 0.3):
                m = math.ceil((n + 1) * (1 - a))
                if m > n:
                    continue
                b = beta(m, n + 1 - m)
                table.append({"n_cal": n, "alpha": a, "rank": m, "mean": b.mean(), "p10": b.ppf(0.1), "p90": b.ppf(0.9)})
                print(f"n={n} alpha={a}: rank {m}, coverage ~ Beta({m},{n + 1 - m}): mean {b.mean():.3f}, "
                      f"p10 {b.ppf(0.1):.3f}, p90 {b.ppf(0.9):.3f}")
        write_csv(out / "checks_reference.csv", table)

    if run("scores"):
        table = []
        L0 = {r["scene"]: r for r in rows if r["level"] == "L0"}
        for lv in LEVELS:
            rs = [r for r in rows if r["level"] == lv]
            for cls in CLASSES + ("tv_stand",):
                have = [r for r in rs if r[f"n_ent_{cls}"] > 0]
                s_joint = np.array([miss_at(r, cls, LAM0) for r in have])
                s_sum = np.array([miss_at(L0[r["scene"]], cls, LAM0) + r[f"miss_gtlab_{cls}"] for r in have])
                lab = np.array([r[f"label_score_{cls}"] for r in have])
                t = {"level": lv, "class": cls, "n_rows": len(have),
                     "additivity_fails": float(np.mean(s_joint > s_sum + 1e-9)),
                     "off_grid": float(np.mean([r[f"off_grid_{cls}"] > 0 for r in have])),
                     "label_uncoverable": float(np.mean(lab >= UNCOVERABLE - 1e-9)),
                     "label_score_one": float(np.mean(np.abs(lab - 1.0) < 1e-9)),
                     "joint_at_rmax": float(np.mean(s_joint >= R_MAX - 1e-9))}
                table.append(t)
                print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in t.items()))
        write_csv(out / "checks_scores.csv", table)

    if run("lam0"):
        table = []
        for lam0 in (0.0, 0.02, 0.05, 0.1, 0.2):
            for lv in ("L0", "L2", "L4"):
                st = run_splits(rows, lv, args.alpha, n_cal, args.splits, classes=CLASSES, lam0=lam0)
                for cls in CLASSES:
                    t = {"lam0": lam0, "level": lv, "class": cls, **summary(st[cls]["joint"])}
                    table.append(t)
                    print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in t.items()))
        write_csv(out / "checks_lam0.csv", table)

    if run("n_cal"):
        table = []
        for n in (9, 11, 13, 15, 17, 19):
            for lv in ("L0", "L2", "L4"):
                st = run_splits(rows, lv, args.alpha, n, args.splits, classes=CLASSES)
                for cls in CLASSES:
                    t = {"n_cal": n, "level": lv, "class": cls, **summary(st[cls]["joint"])}
                    table.append(t)
                    print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in t.items()))
        write_csv(out / "checks_n_cal.csv", table)

    if run("both"):
        # coverage of both classes at once for the joint arm (per test realization)
        rng = np.random.default_rng(0)
        table = []
        for lv in LEVELS:
            hits, cert = [], []
            for _ in range(args.splits):
                perm = rng.permutation(scenes)
                cal_s, test_s = perm[:n_cal], perm[n_cal:]
                cal = [(lambda b: b[rng.integers(len(b))])([r for r in by_scene[s] if r["level"] == lv]) for s in cal_s]
                cal_L0 = [next(r for r in by_scene[s] if r["level"] == "L0") for s in cal_s]
                ps = {c: calibrate_class(cal, cal_L0, c, args.alpha)["joint"] for c in CLASSES}
                test = [r for s in test_s for r in by_scene[s] if r["level"] == lv]
                hits.append(np.mean([all(covers(t, c, ps[c]) for c in CLASSES) for t in test]))
                cert.append(all(p is not None for p in ps.values()))
            hits, cert = np.array(hits), np.array(cert)
            t = {"level": lv, "coverage_both": float(hits.mean()), "both_certified": float(cert.mean()),
                 "coverage_both_cert": float(hits[cert].mean()) if cert.any() else np.nan}
            table.append(t)
            print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in t.items()))
        write_csv(out / "checks_both.csv", table)

    if run("layout"):
        rng = np.random.default_rng(0)
        table = []
        for g in sorted({layout(s) for s in scenes}):
            test_s = [s for s in scenes if layout(s) == g]
            cal_s = [s for s in scenes if layout(s) != g]
            for lv in ("L0", "L2", "L4"):
                acc = {c: [] for c in CLASSES}
                for _ in range(50):
                    cal = [(lambda b: b[rng.integers(len(b))])([r for r in by_scene[s] if r["level"] == lv]) for s in cal_s]
                    cal_L0 = [next(r for r in by_scene[s] if r["level"] == "L0") for s in cal_s]
                    test = [r for s in test_s for r in by_scene[s] if r["level"] == lv]
                    for c in CLASSES:
                        p = calibrate_class(cal, cal_L0, c, args.alpha)["joint"]
                        acc[c].append((np.mean([covers(t, c, p) for t in test]), p is None))
                for c in CLASSES:
                    cov = np.array([a for a, _ in acc[c]])
                    ab = np.array([b for _, b in acc[c]])
                    t = {"held_out_layout": g, "n_test": len(test_s), "n_cal": len(cal_s), "level": lv, "class": c,
                         "coverage": float(cov.mean()), "abstain": float(ab.mean())}
                    table.append(t)
                    print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in t.items()))
        write_csv(out / "checks_layout.csv", table)

    if run("bootstrap"):
        rng = np.random.default_rng(1)
        table = []
        for lv in ("L0", "L2", "L3", "L4"):
            boot = {(c, a): [] for c in CLASSES for a in ARMS}
            for _ in range(args.boot):
                draw = rng.integers(len(scenes), size=len(scenes))
                groups = [[by_scene[scenes[i]]] * int(np.sum(draw == i)) for i in np.unique(draw)]
                st = grouped_splits(groups, lv, args.alpha, n_cal, 20, rng)
                for c in CLASSES:
                    for a in ARMS:
                        boot[(c, a)].append(float(np.mean(st[c][a]["coverage"])))
            for (c, a), v in boot.items():
                lo, hi = np.percentile(v, [5, 95])
                t = {"level": lv, "class": c, "arm": a, "boot_mean": float(np.mean(v)), "ci90_lo": lo, "ci90_hi": hi}
                table.append(t)
                print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in t.items()))
        write_csv(out / "checks_bootstrap.csv", table)
