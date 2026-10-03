"""Cache MobileSAM instance masks for the cached YOLO-World boxes (resumable, one file per scene).

Masks are stored on the stride-4 pixel grid that the map back-projects, so the "sam" perception
variant needs about 1/16 of the full-resolution storage and is exact for mapping.

Example: python scripts/run_segmenter.py data/replica_cad/*
"""

import argparse
import time
from pathlib import Path

from s2m.data import load_scene
from s2m.perception import BoxSegmenter, load_detections, save_masks

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dirs", type=Path, nargs="+")
    ap.add_argument("--detections", type=Path, default=Path("data/cache/detections"))
    ap.add_argument("--out", type=Path, default=Path("data/cache/masks_sam"))
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()

    seg = None
    for scene_dir in args.scene_dirs:
        out = args.out / f"{scene_dir.name}.npz"
        if out.exists():
            continue
        seg = seg or BoxSegmenter()
        t = time.time()
        scene = load_scene(scene_dir)
        dets = load_detections(args.detections / f"{scene.name}.npz")
        save_masks(out, {i: seg(scene.rgb(i), dets[i]) for i in scene.frame_ids()}, args.stride)
        print(f"{scene.name}: {len(dets)} frames, {time.time() - t:.0f} s", flush=True)
