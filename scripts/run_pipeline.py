"""Detector + score computation for every fully downloaded scene (resumable).

Example: python scripts/run_pipeline.py --seeds 10
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from s2m.data import load_scene
from s2m.experiment import load_levels, prepare, scene_rows
from s2m.mapping import precompute_observations
from s2m.perception import (OpenVocabDetector, load_detections, object_vocabulary,
                            save_detections)


def is_complete(scene_dir: Path, step: int) -> bool:
    traj = scene_dir / "traj.txt"
    if not traj.exists():
        return False
    expected = 3 * len(range(0, len(np.loadtxt(traj, ndmin=2)), step))
    return len(list((scene_dir / "results").glob("*.*[gp]"))) >= expected


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/replica_cad"))
    ap.add_argument("--step", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--levels", type=Path, default=Path("configs/drift_levels.yaml"))
    ap.add_argument("--detections", type=Path, default=Path("data/cache/detections"))
    ap.add_argument("--scores", type=Path, default=Path("results/scores"))
    args = ap.parse_args()

    levels = load_levels(args.levels)
    args.scores.mkdir(parents=True, exist_ok=True)
    detector = None
    for scene_dir in sorted(p for p in args.data.iterdir() if p.is_dir()):
        out = args.scores / f"{scene_dir.name}.csv"
        if out.exists():
            continue
        if not is_complete(scene_dir, args.step):
            print(f"{scene_dir.name}: incomplete download, skipped", flush=True)
            continue
        t = time.time()
        scene = load_scene(scene_dir)
        det_path = args.detections / f"{scene.name}.npz"
        if not det_path.exists():
            if detector is None:
                detector = OpenVocabDetector(object_vocabulary(scene.classes))
            save_detections(det_path, {i: detector(scene.rgb(i)) for i in scene.frame_ids()})
        setup = prepare(scene, precompute_observations(scene, load_detections(det_path)))
        rows = list(scene_rows(setup, levels, args.seeds))
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"{scene.name}: {len(setup.entities)} avoid entities, {len(rows)} rows, "
              f"{time.time() - t:.0f} s", flush=True)
