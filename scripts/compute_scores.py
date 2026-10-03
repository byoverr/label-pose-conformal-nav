"""Compute calibration scores for every scene x drift level x realization.

Writes one CSV per scene to results/scores/ (resumable: finished scenes are skipped).
Detections must be cached first (scripts/run_detector.py).

Example: python scripts/compute_scores.py data/replica_cad/* --seeds 10
         python scripts/compute_scores.py data/replica_cad/* --masks data/cache/masks_sam --out results/scores_sam
"""

import argparse
import csv
import time
from pathlib import Path

from s2m.data import load_scene
from s2m.experiment import load_levels, prepare, scene_rows
from s2m.mapping import precompute_observations
from s2m.perception import load_detections, load_masks

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dirs", type=Path, nargs="+")
    ap.add_argument("--levels", type=Path, default=Path("configs/drift_levels.yaml"))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--detections", type=Path, default=Path("data/cache/detections"))
    ap.add_argument("--masks", type=Path, default=None, help="cached instance masks (default: box + depth)")
    ap.add_argument("--out", type=Path, default=Path("results/scores"))
    args = ap.parse_args()

    levels = load_levels(args.levels)
    args.out.mkdir(parents=True, exist_ok=True)
    for scene_dir in args.scene_dirs:
        out = args.out / f"{scene_dir.name}.csv"
        det_path = args.detections / f"{scene_dir.name}.npz"
        if out.exists() or not det_path.exists():
            print(f"{scene_dir.name}: {'done' if out.exists() else 'no detections, skipped'}")
            continue
        t = time.time()
        scene = load_scene(scene_dir)
        masks = None if args.masks is None else load_masks(args.masks / f"{scene_dir.name}.npz")
        setup = prepare(scene, precompute_observations(scene, load_detections(det_path), masks=masks))
        rows = list(scene_rows(setup, levels, args.seeds))
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print(f"{scene.name}: {len(setup.entities)} avoid entities, {len(rows)} rows, {time.time() - t:.0f} s")
