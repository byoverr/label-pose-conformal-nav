"""Reach-avoid missions with calibrated arms, leave-one-scene-out.

Missions in a scene use arm parameters calibrated on all OTHER scenes (dev excluded), so no
scene is evaluated with parameters it helped choose; with ~14 scenes this keeps n_cal >= 9,
the minimum for a finite conformal quantile at alpha = 0.1.
Writes results/missions/<scene>.csv (resumable) and results/missions/params_<scene>.csv.

Example: python scripts/run_missions.py --seeds 5 --tasks 20
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from s2m.analysis import calibrate_arms, load_scores
from s2m.data import load_scene
from s2m.drift import simulate
from s2m.experiment import load_levels, prepare
from s2m.mapping import precompute_observations
from s2m.missions import make_tasks, run_realization
from s2m.perception import load_detections

DEV_SCENES = ("apt_0",)


def calibrated_params(rows, cal_scenes, levels, alpha, rng):
    """Arm parameters per level from one random realization of each calibration scene."""
    by = {}
    for r in rows:
        by.setdefault((r["scene"], r["level"]), []).append(r)
    params = {}
    for lv in levels:
        cal = [by[(s, lv)][rng.integers(len(by[(s, lv)]))] for s in cal_scenes]
        cal_L0 = [by[(s, "L0")][0] for s in cal_scenes]
        params[lv] = calibrate_arms(cal, cal_L0, alpha)
    return params


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def run_scene(name, idx, params, args, levels):
    scene = load_scene(args.data / name)
    setup = prepare(scene, precompute_observations(
        scene, load_detections(Path("data/cache/detections") / f"{name}.npz")))
    tasks = make_tasks(setup, args.tasks, seed=0)
    results = []
    for lv, model in levels.items():
        for seed in range(1 if lv == "L0" else args.seeds):
            est = simulate(scene.poses, model, np.random.default_rng([7, seed, idx]))
            for r in run_realization(setup, tasks, est, params[lv]):
                results.append({"scene": name, "level": lv, "seed": seed, **r})
    return tasks, results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/replica_cad"))
    ap.add_argument("--scores", type=Path, default=Path("results/scores"))
    ap.add_argument("--out", type=Path, default=Path("results/missions"))
    ap.add_argument("--levels", type=Path, default=Path("configs/drift_levels.yaml"))
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--tasks", type=int, default=20)
    args = ap.parse_args()

    levels = load_levels(args.levels)
    rows = load_scores(args.scores, exclude=DEV_SCENES)
    scenes = sorted({r["scene"] for r in rows})
    args.out.mkdir(parents=True, exist_ok=True)

    for idx, name in enumerate(scenes):
        out = args.out / f"{name}.csv"
        if out.exists():
            continue
        t = time.time()
        params = calibrated_params(rows, [s for s in scenes if s != name], levels, args.alpha,
                                   np.random.default_rng([0, idx]))
        write_csv(args.out / f"params_{name}.csv",
                  [{"scene": name, "level": lv, "arm": arm,
                    "lam": "" if p is None else p.lam, "radius": "" if p is None else p.radius}
                   for lv, arms in params.items() for arm, p in arms.items()])
        tasks, results = run_scene(name, idx, params, args, levels)
        write_csv(out, results)
        print(f"{name}: {len(tasks)} tasks, {len(results)} rows, {time.time() - t:.0f} s", flush=True)
