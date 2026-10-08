"""Cascade detections (s2m.cascade) in the detection-cache format, for the usual pipeline.

The threshold tau = 0.2 is the largest on the grid of scripts/select_cascade.py whose label misses on the
development scene apt_0 are no larger than those of the plain union of the three detectors; relabelling is
off (it multiplies the false area of the TV stand and the bike on apt_0).

Example: python scripts/make_cascade.py data/replica_cad/*
"""

import argparse
from pathlib import Path

from s2m.cascade import cascade, load_verified, verifier_prompts
from s2m.data import load_scene
from s2m.perception import load_detections, save_detections

DETECTORS = ("data/cache/detections", "data/cache/detections_x", "data/cache/detections_yoloe")
TAU = 0.2

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dirs", type=Path, nargs="+")
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--clip", type=Path, default=Path("data/cache/clip"))
    ap.add_argument("--out", type=Path, default=Path("data/cache/detections_cascade"))
    args = ap.parse_args()
    for scene_dir in args.scene_dirs:
        scene = load_scene(scene_dir)
        sets = [load_detections(Path(d) / f"{scene.name}.npz") for d in DETECTORS]
        dets = cascade(sets, load_verified(args.clip / f"{scene.name}.npz"), verifier_prompts(scene.classes)[0], args.tau)
        save_detections(args.out / f"{scene.name}.npz", dets)
        print(f"{scene.name}: {sum(len(d.scores) for d in dets.values())} of "
              f"{sum(len(d.scores) for s in sets for d in s.values())} detections kept")
