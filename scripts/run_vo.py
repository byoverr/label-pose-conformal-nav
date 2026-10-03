"""Real pose error: run RGB-D visual odometry on every scene and score the resulting maps.

For each scene with dense RGB-D frames in data/replica_cad_vo (scripts/download_subset.py
--step 2 --rgbd-only) the script
  1. runs frame-to-frame Open3D RGB-D odometry on every `step`-th frame (cached in data/cache/vo/),
  2. keeps the floor-plane part of its error (s2m.odometry.planarize),
  3. writes one score row per scene and step (level "VO<step>") to results/scores_vo/<scene>.csv,
     in the same format as scripts/compute_scores.py, so the calibration arms can use it.
The trajectory is deterministic, so each scene contributes one realization per VO setting.

Example: python scripts/run_vo.py data/replica_cad/* --steps 2 4
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from s2m.data import load_scene
from s2m.drift import endpoint_drift, path_length
from s2m.experiment import prepare, realization_scores
from s2m.mapping import precompute_observations
from s2m.odometry import VOResult, full_trajectory, load_vo, planarize, rgbd_odometry, save_vo
from s2m.perception import load_detections


def cached_vo(vo_scene, step: int, cache: Path) -> VOResult:
    path = cache / f"{vo_scene.name}_s{step}.npz"
    if path.exists():
        return load_vo(path)
    vo = rgbd_odometry(vo_scene, [f for f in vo_scene.frame_ids() if f % step == 0])
    save_vo(path, vo)
    return vo


def is_dense(vo_dir: Path, step: int) -> bool:
    n = len(np.loadtxt(vo_dir / "traj.txt", ndmin=2)) if (vo_dir / "traj.txt").exists() else 0
    have = {p.stem[len("depth"):] for p in (vo_dir / "results").glob("depth*.png")} if n else set()
    have &= {p.stem[len("frame"):] for p in (vo_dir / "results").glob("frame*.jpg")}
    return n > 0 and all(f"{i:06d}" in have for i in range(0, n, step))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dirs", type=Path, nargs="+")
    ap.add_argument("--vo-data", type=Path, default=Path("data/replica_cad_vo"))
    ap.add_argument("--steps", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--detections", type=Path, default=Path("data/cache/detections"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache/vo"))
    ap.add_argument("--out", type=Path, default=Path("results/scores_vo"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for scene_dir in args.scene_dirs:
        out = args.out / f"{scene_dir.name}.csv"
        vo_dir = args.vo_data / scene_dir.name
        if out.exists() or not is_dense(vo_dir, min(args.steps)):
            print(f"{scene_dir.name}: {'done' if out.exists() else 'dense frames missing, skipped'}", flush=True)
            continue
        t = time.time()
        scene = load_scene(scene_dir)
        vo_scene = load_scene(vo_dir)
        setup = prepare(scene, precompute_observations(scene, load_detections(args.detections / f"{scene.name}.npz")))
        rows = []
        for step in args.steps:
            vo = cached_vo(vo_scene, step, args.cache)
            raw = full_trajectory(scene.poses, vo)
            est = planarize(raw, scene.poses)
            row = {"scene": scene.name, "level": f"VO{step}", "seed": 0, **realization_scores(setup, est)}
            # what planarize removed: height and tilt error of the free 6-DoF estimate
            row["height_err_max"] = float(np.abs(raw[:, 1, 3] - scene.poses[:, 1, 3]).max())
            row["tilt_err_max_deg"] = float(np.degrees(np.arccos(np.clip(
                np.einsum("ki,ki->k", raw[:, :3, 1], scene.poses[:, :3, 1]), -1, 1))).max())
            row["vo_failures"] = int((~vo.success).sum())
            rows.append(row)
            print(f"{scene.name} VO{step}: {len(vo.frames)} frames, ATE {row['ate'] * 100:.1f} cm "
                  f"(removed: height {row['height_err_max'] * 100:.0f} cm, tilt {row['tilt_err_max_deg']:.1f} deg), end drift {endpoint_drift(est, scene.poses) * 100:.1f} % "
                  f"of {path_length(scene.poses):.1f} m, solver failures {row['vo_failures']}", flush=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print(f"{scene.name}: {time.time() - t:.0f} s", flush=True)
