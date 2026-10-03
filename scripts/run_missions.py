"""Reach-avoid missions with calibrated arms, leave-one-scene-out.

Missions in a scene use arm parameters calibrated on all OTHER scenes (dev excluded), so no
scene is evaluated with parameters it helped choose; with ~15 scenes this keeps n_cal >= 9,
the minimum for a finite conformal quantile at alpha = 0.1. Parameters are per avoid class;
a class whose certificate abstains falls back to geometry (see s2m.missions.keepout).
Writes results/missions/<scene>.csv (resumable) and results/missions/params_<scene>.csv.

Pose error comes from the synthetic drift levels and, with --vo, from the cached real visual
odometry (levels "VO<step>", one realization per scene; scripts/run_vo.py). --only restricts the
levels; --masks switches perception to the cached SAM masks (scores must match, see --scores).

Example: python scripts/run_missions.py --seeds 5 --tasks 20
         python scripts/run_missions.py --scores results/scores results/scores_vo --vo 2 4 --only L0 VO2 VO4
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from s2m.analysis import ARMS, calibrate_class, load_scores
from s2m.data import load_scene
from s2m.drift import simulate
from s2m.entities import AVOID_CLASSES, class_ids
from s2m.experiment import load_levels, prepare, stable_seed
from s2m.mapping import precompute_observations
from s2m.missions import make_tasks, run_realization
from s2m.odometry import vo_poses
from s2m.perception import load_detections, load_masks

DEV_SCENES = ("apt_0",)


def calibrated_params(rows, cal_scenes, levels, alpha, rng, cls_ids):
    """params[level][arm][class_id] and fallback[level][class_id] (pose-only radius)."""
    by = {}
    for r in rows:
        by.setdefault((r["scene"], r["level"]), []).append(r)
    params, fallback = {}, {}
    for lv in levels:
        cal = [by[(s, lv)][rng.integers(len(by[(s, lv)]))] for s in cal_scenes]
        cal_L0 = [by[(s, "L0")][0] for s in cal_scenes]
        per_class = {cid: calibrate_class(cal, cal_L0, name, alpha) for name, cid in cls_ids.items()}
        params[lv] = {arm: {cid: per_class[cid][arm] for cid in per_class} for arm in ARMS}
        fallback[lv] = {cid: None if per_class[cid]["pose_only"] is None else per_class[cid]["pose_only"].radius
                        for cid in per_class}
    return params, fallback


MISSION_FIELDS = ["scene", "level", "seed", "task", "arm", "abstained", "classes_fell_back", "planned",
                  "violation", "collision", "reached", "length", "success", "oracle_length"]


def write_csv(path, rows, fields=None):
    """Header-only file when there are no rows (e.g. a scene without valid tasks): resumable runs skip it."""
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields or list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/replica_cad"))
    ap.add_argument("--scores", type=Path, nargs="+", default=[Path("results/scores")])
    ap.add_argument("--out", type=Path, default=Path("results/missions"))
    ap.add_argument("--levels", type=Path, default=Path("configs/drift_levels.yaml"))
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--tasks", type=int, default=20)
    ap.add_argument("--part", default="0/1", help="i/n: process every n-th scene starting at i")
    ap.add_argument("--avoid", nargs="+", default=list(AVOID_CLASSES), help="avoid classes used in planning")
    ap.add_argument("--vo", type=int, nargs="*", default=[], help="VO frame steps to add as levels")
    ap.add_argument("--vo-cache", type=Path, default=Path("data/cache/vo"))
    ap.add_argument("--only", nargs="*", default=None, help="run only these levels")
    ap.add_argument("--masks", type=Path, default=None, help="cached SAM masks (default: box + depth)")
    ap.add_argument("--detections", type=Path, default=Path("data/cache/detections"))
    ap.add_argument("--kind", default="pass_by", choices=["pass_by", "random", "approach"], help="task family")
    args = ap.parse_args()

    levels = {lv: m for lv, m in load_levels(args.levels).items() if args.only is None or lv in args.only}
    vo_levels = [f"VO{s}" for s in args.vo if args.only is None or f"VO{s}" in args.only]
    rows = [r for d in args.scores for r in load_scores(d, exclude=DEV_SCENES)]
    # scenes that have every requested level
    have = {}
    for r in rows:
        have.setdefault(r["scene"], set()).add(r["level"])
    scenes = sorted(s for s, lv in have.items() if {"L0", *levels, *vo_levels} <= lv)
    args.out.mkdir(parents=True, exist_ok=True)

    part, nparts = map(int, args.part.split("/"))
    for name in scenes[part::nparts]:
        out = args.out / f"{name}.csv"
        if out.exists():
            continue
        t = time.time()
        scene = load_scene(args.data / name)
        cls_ids = dict(zip(args.avoid, class_ids(scene.classes, args.avoid)))
        params, fallback = calibrated_params(rows, [s for s in scenes if s != name], [*levels, *vo_levels],
                                             args.alpha, np.random.default_rng(stable_seed(0, name)), cls_ids)
        write_csv(args.out / f"params_{name}.csv",
                  [{"scene": name, "level": lv, "arm": arm, "class": cname,
                    "lam": "" if p is None else p.lam, "radius": "" if p is None else p.radius,
                    "fallback_radius": "" if fallback[lv][cid] is None else fallback[lv][cid]}
                   for lv in params for arm in ARMS for cname, cid in cls_ids.items()
                   for p in [params[lv][arm][cid]]])

        masks = None if args.masks is None else load_masks(args.masks / f"{name}.npz")
        setup = prepare(scene, precompute_observations(
            scene, load_detections(args.detections / f"{name}.npz"), masks=masks), args.avoid)
        tasks = make_tasks(setup, args.tasks, seed=0, kind=args.kind)
        results = []
        for lv, model in levels.items():
            for seed in range(1 if lv == "L0" else args.seeds):
                est = simulate(scene.poses, model, np.random.default_rng(stable_seed(7, seed, name)))
                for r in run_realization(setup, tasks, est, params[lv], fallback[lv]):
                    results.append({"scene": name, "level": lv, "seed": seed, **r})
        for lv in vo_levels:
            est = vo_poses(args.vo_cache / f"{name}_s{lv[2:]}.npz", scene.poses)
            for r in run_realization(setup, tasks, est, params[lv], fallback[lv]):
                results.append({"scene": name, "level": lv, "seed": 0, **r})
        write_csv(out, results, MISSION_FIELDS)
        print(f"{name}: {len(tasks)} tasks, {len(results)} rows, {time.time() - t:.0f} s", flush=True)
