"""CLIP probabilities for every cached detection of several detectors (stage 2 of s2m.cascade).

Writes data/cache/clip/<scene>.npz (resumable). Needs models/clip/ViT-B-16.pt (OpenAI CLIP, 335 MB,
downloaded by the clip package on first use).

Example: python scripts/run_verifier.py data/replica_cad/*
"""

import argparse
import time
from pathlib import Path

from s2m.cascade import ClipVerifier, save_verified, verifier_prompts, verify_scene
from s2m.data import load_scene
from s2m.perception import load_detections

DETECTORS = ("data/cache/detections", "data/cache/detections_x", "data/cache/detections_yoloe")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dirs", type=Path, nargs="+")
    ap.add_argument("--detections", type=Path, nargs="+", default=[Path(d) for d in DETECTORS])
    ap.add_argument("--out", type=Path, default=Path("data/cache/clip"))
    args = ap.parse_args()
    verifier = None
    for scene_dir in args.scene_dirs:
        out = args.out / f"{scene_dir.name}.npz"
        if out.exists():
            continue
        t = time.time()
        scene = load_scene(scene_dir)
        if verifier is None:
            verifier = ClipVerifier(verifier_prompts(scene.classes)[1])
        sets = [load_detections(d / f"{scene.name}.npz") for d in args.detections]
        data = verify_scene(scene, sets, verifier)
        save_verified(out, data)
        n = sum(len(v) for k, v in data.items() if k.startswith("cluster_"))
        print(f"{scene.name}: {n} detections, {len(data['probs'])} crops, {time.time() - t:.0f} s", flush=True)
