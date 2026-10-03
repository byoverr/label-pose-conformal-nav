"""Out-of-distribution scenes: the HM3D part of OSMa-Bench through the same pipeline.

HM3D scenes are real scans (ReplicaCAD scenes are synthetic) and are published only under the
"no_lights" condition, so they differ from the calibration scenes in geometry, appearance and
lighting at once. For every single-floor HM3D scene the script
  1. maps HM3D categories to ReplicaCAD classes (configs/hm3d_class_map.yaml),
  2. estimates the floor height from ground-truth floor points and shifts it to y = 0,
  3. runs the same YOLO-World vocabulary and writes score rows for every drift level to
     results/scores_hm3d/<scene>.csv (same format as scripts/compute_scores.py).
Calibration still uses ReplicaCAD only; scripts/analyze_variants.py tests the margins on these scenes.

Frames first: python scripts/download_subset.py <scenes> --root data/hm3d --out data/hm3d
Example:      python scripts/run_hm3d.py data/hm3d/*
"""

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from s2m.data import backproject, load_remapped_scene, load_scene, map_class_names
from s2m.experiment import load_levels, prepare, scene_rows
from s2m.grid import FLOOR_CLASSES
from s2m.mapping import precompute_observations
from s2m.perception import OpenVocabDetector, load_detections, object_vocabulary, save_detections
from s2m.io import write_csv

MAX_HEIGHT_RANGE = 0.3  # metres of camera height change; more means stairs / several floors


def floor_height(scene, floor_ids, n_frames: int = 30) -> float:
    """Median world height of ground-truth floor pixels over a sample of frames."""
    ys = []
    frames = scene.frame_ids()
    for i in frames[:: max(1, len(frames) // n_frames)]:
        pts, pix = backproject(scene.depth(i), scene.intrinsics, stride=8)
        lab = scene.semantic(i).ravel()[pix]
        world = pts @ scene.poses[i][:3, :3].T + scene.poses[i][:3, 3]
        ys.append(world[np.isin(lab, floor_ids), 1])
    ys = np.concatenate(ys)
    return float(np.median(ys)) if len(ys) else np.nan


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dirs", type=Path, nargs="+")
    ap.add_argument("--reference", type=Path, default=Path("data/replica_cad/apt_0"),
                    help="a ReplicaCAD scene providing the target class list")
    ap.add_argument("--class-map", type=Path, default=Path("configs/hm3d_class_map.yaml"))
    ap.add_argument("--levels", type=Path, default=Path("configs/drift_levels.yaml"))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--detections", type=Path, default=Path("data/cache/detections_hm3d"))
    ap.add_argument("--out", type=Path, default=Path("results/scores_hm3d"))
    args = ap.parse_args()

    target = load_scene(args.reference).classes
    rules = yaml.safe_load(args.class_map.read_text())
    levels = load_levels(args.levels)
    args.out.mkdir(parents=True, exist_ok=True)
    detector = None
    log = []
    for scene_dir in args.scene_dirs:
        out = args.out / f"{scene_dir.name}.csv"
        if out.exists():
            continue
        t = time.time()
        raw = load_scene(scene_dir)
        h = np.ptp(raw.poses[:, 1, 3])
        lut = map_class_names(raw.classes, target, rules)
        floor_src = [i for i in raw.classes if target[lut[i]] in FLOOR_CLASSES]
        fy = floor_height(raw, floor_src)
        mapped = sorted({target[lut[i]] for i in raw.classes} - {"other"})
        if h > MAX_HEIGHT_RANGE or not np.isfinite(fy):
            print(f"{scene_dir.name}: camera height range {h:.2f} m (several floors), skipped", flush=True)
            continue
        scene = load_remapped_scene(scene_dir, target, rules, fy)
        det_path = args.detections / f"{scene.name}.npz"
        if not det_path.exists():
            detector = detector or OpenVocabDetector(object_vocabulary(target))
            save_detections(det_path, {i: detector(scene.rgb(i)) for i in scene.frame_ids()})
        setup = prepare(scene, precompute_observations(scene, load_detections(det_path)))
        rows = list(scene_rows(setup, levels, args.seeds))
        write_csv(out, rows)
        ents = {target[k]: sum(e.cls == k for e in setup.entities) for k in setup.avoid_ids}
        print(f"{scene.name}: floor at y = {fy:.2f} m, mapped classes {mapped}, avoid entities {ents}, "
              f"{len(rows)} rows, {time.time() - t:.0f} s", flush=True)
