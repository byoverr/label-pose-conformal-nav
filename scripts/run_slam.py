"""Real pose error with loop closures: pose-graph SLAM (s2m.slam) over the cached RGB-D odometry.

For every scene with cached odometry (scripts/run_vo.py, step 2) this runs loop-closure SLAM, caches
the trajectory in data/cache/slam/<scene>_s2.npz (same format as the odometry cache) and writes one
score row per scene with level "SLAM2" to results/scores_slam/<scene>.csv, in the format of
scripts/compute_scores.py.

With several --detections caches the map fuses them (--fuse, as in scripts/compute_scores.py); a trajectory
already in the cache is reused.

Example: python scripts/run_slam.py data/replica_cad/*
"""

import argparse
import time
from pathlib import Path

from s2m.data import load_scene
from s2m.drift import endpoint_drift, path_length
from s2m.experiment import prepare, realization_scores
from s2m.io import write_csv
from s2m.mapping import precompute_observations
from s2m.odometry import full_trajectory, load_vo, planarize, save_vo
from s2m.perception import load_detections
from s2m.slam import loop_closure_slam

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dirs", type=Path, nargs="+")
    ap.add_argument("--vo-data", type=Path, default=Path("data/replica_cad_vo"))
    ap.add_argument("--vo-cache", type=Path, default=Path("data/cache/vo"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache/slam"))
    ap.add_argument("--detections", type=Path, nargs="+", default=[Path("data/cache/detections")])
    ap.add_argument("--fuse", default="max", choices=["max", "mean", "vote2"])
    ap.add_argument("--out", type=Path, default=Path("results/scores_slam"))
    ap.add_argument("--step", type=int, default=2)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for scene_dir in args.scene_dirs:
        out = args.out / f"{scene_dir.name}.csv"
        vo_path = args.vo_cache / f"{scene_dir.name}_s{args.step}.npz"
        if out.exists() or not vo_path.exists():
            continue
        t = time.time()
        scene = load_scene(scene_dir)
        vo_scene = load_scene(args.vo_data / scene_dir.name)
        vo = load_vo(vo_path)
        cache = args.cache / f"{scene.name}_s{args.step}.npz"
        if cache.exists():
            slam, stats = load_vo(cache), None
        else:
            slam, stats, _ = loop_closure_slam(vo_scene, vo)
            save_vo(cache, slam)
        obs = [precompute_observations(scene, load_detections(d / f"{scene.name}.npz")) for d in args.detections]
        setup = prepare(scene, obs[0])
        setup.extra_obs, setup.fuse = obs[1:], args.fuse
        est = planarize(full_trajectory(scene.poses, slam), scene.poses)
        row = {"scene": scene.name, "level": f"SLAM{args.step}", "seed": 0, **realization_scores(setup, est),
               **({"keyframes": stats.keyframes, "loops": stats.kept} if stats else {})}
        vo_est = planarize(full_trajectory(scene.poses, vo), scene.poses)
        write_csv(out, [row])
        from s2m.drift import ate
        print(f"{scene.name}: {stats.kept if stats else 'cached'} loops, ATE odometry {ate(vo_est, scene.poses) * 100:.1f} cm -> SLAM "
              f"{row['ate'] * 100:.1f} cm, end drift {endpoint_drift(est, scene.poses) * 100:.1f} % of "
              f"{path_length(scene.poses):.1f} m, {time.time() - t:.0f} s", flush=True)
