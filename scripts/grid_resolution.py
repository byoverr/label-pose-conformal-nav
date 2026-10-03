"""Why per-cell label calibration degenerates indoors: the role of the grid resolution.

Sundarsingh et al. calibrate per-cell label sets on a 1 m grid. On a fine grid almost every true
footprint has a boundary cell with no evidence for its class, so the calibrated threshold is
lambda* = 0 and every occupied cell becomes "possibly hazardous". This script repeats both
calibrations at several grid resolutions, with known poses and the closed 5-class vocabulary
(their setting), leave-one-scene-out at 1 - alpha = 0.95:
  label_cell : threshold lambda* = 1 - q(label scores), keep-out = cells with p_k >= lambda*,
               dilated by d + grid resolution (as in their planner);
  joint      : region p_k >= LAM0, dilated by d + q(miss distance) (ours; no drift at L0).
Reports lambda*, the radius, LOO coverage and the share of free space the keep-out removes.
Writes results/tables/grid_resolution.csv.

Example: python scripts/grid_resolution.py --res 0.05 0.25 0.5 1.0
"""

import argparse
from pathlib import Path

import numpy as np

from s2m.analysis import LAM0
from s2m.conformal import UNCOVERABLE, conformal_quantile, label_score, miss_distance
from s2m.data import load_scene
from s2m.entities import class_ids, extract_entities
from s2m.experiment import R_MAX
from s2m.mapping import build_map, fixed_spec, floor_class_ids, precompute_observations
from s2m.perception import load_detections
from s2m.planning import SAFETY_DISTANCE, dilate
from s2m.io import DEV_SCENES, write_csv

AVOID = ("indoor_plant", "bike")


def scene_stats(scene, obs, res):
    spec = fixed_spec(scene, obs, res)
    k = max(scene.classes) + 1
    truth = build_map(obs, scene.poses, spec, k, "gt", floor_class_ids(scene))
    pred = build_map(obs, scene.poses, spec, k, "det")
    ids = class_ids(scene.classes, AVOID)
    # entity filters defined in metres so they mean the same at every resolution
    ents = extract_entities(truth, ids, min_cells=max(1, round(0.02 / res ** 2)), merge_cells=max(1, round(0.1 / res)))
    out = {"spec": spec, "pred": pred, "free": truth.free(), "ids": ids}
    for cid, name in zip(ids, AVOID):
        ek = [e for e in ents if e.cls == cid]
        out[name] = {"n": len(ek), "label": label_score(pred, ek),
                     "miss": miss_distance({cid: pred.region(cid, LAM0)}, ek, res, R_MAX), "cid": cid}
    return out


def lost_space(stats, keep):
    area = stats["free"]
    return float((keep & area).sum() / max(area.sum(), 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=float, nargs="+", default=[0.05, 0.25, 0.5, 1.0])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--detections", type=Path, default=Path("data/cache/detections_closed5"))
    args = ap.parse_args()

    scenes = [d for d in sorted(Path("data/replica_cad").iterdir()) if d.name not in DEV_SCENES]
    stats = {r: {} for r in args.res}
    for d in scenes:
        scene = load_scene(d)
        obs = precompute_observations(scene, load_detections(args.detections / f"{d.name}.npz"))
        for r in args.res:
            stats[r][d.name] = scene_stats(scene, obs, r)
        print(d.name, flush=True)

    table = []
    for r in args.res:
        for cls in AVOID:
            lam, rad, cov_l, cov_j, lost_l, lost_j = [], [], [], [], [], []
            names = [s for s in stats[r] if stats[r][s][cls]["n"] > 0]
            for test in names:
                cal = [stats[r][s][cls] for s in stats[r] if s != test]
                st, t = stats[r][test], stats[r][test][cls]
                # per-cell label CP: one threshold for the class
                q = conformal_quantile([c["label"] for c in cal], args.alpha)
                lam_star = max(0.0, 1.0 - q) if q < UNCOVERABLE - 1e-9 else None
                if lam_star is not None:
                    lam.append(lam_star)
                    cov_l.append(t["label"] <= q + 1e-9)
                    hazard = st["pred"].occupied() & (st["pred"].prob(t["cid"]) >= lam_star)
                    lost_l.append(lost_space(st, dilate(hazard, SAFETY_DISTANCE + r, r)))
                # joint miss distance (no drift: equals labels-only)
                qm = conformal_quantile([c["miss"] for c in cal], args.alpha)
                if qm < R_MAX:
                    rad.append(qm)
                    cov_j.append(t["miss"] <= max(qm, r) + 1e-9)
                    lost_j.append(lost_space(st, dilate(st["pred"].region(t["cid"], LAM0), SAFETY_DISTANCE + qm, r)))
            row = {"res": r, "class": cls, "n_scenes": len(names),
                   "label_cell_lambda_median": float(np.median(lam)) if lam else np.nan,
                   "label_cell_coverage": float(np.mean(cov_l)) if cov_l else np.nan,
                   "label_cell_abstain": 1 - len(lam) / len(names),
                   "label_cell_free_lost": float(np.median(lost_l)) if lost_l else np.nan,
                   "joint_radius_median": float(np.median(rad)) if rad else np.nan,
                   "joint_coverage": float(np.mean(cov_j)) if cov_j else np.nan,
                   "joint_abstain": 1 - len(rad) / len(names),
                   "joint_free_lost": float(np.median(lost_j)) if lost_j else np.nan}
            table.append(row)
            print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    write_csv("results/tables/grid_resolution.csv", table)
