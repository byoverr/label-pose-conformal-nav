"""Run the open-vocabulary detector on all downloaded frames of the given scenes and cache results.

Example: python scripts/run_detector.py data/replica_cad/apt_0 --limit 50   # speed check
"""

import argparse
import time
from pathlib import Path

from s2m.data import load_scene
from s2m.perception import OpenVocabDetector, object_vocabulary, save_detections

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dirs", type=Path, nargs="+")
    ap.add_argument("--cache", type=Path, default=Path("data/cache/detections"))
    ap.add_argument("--limit", type=int, default=None, help="only the first N frames (speed check)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    detector = None
    for scene_dir in args.scene_dirs:
        scene = load_scene(scene_dir)
        out = args.cache / f"{scene.name}.npz"
        if out.exists() and not args.overwrite and args.limit is None:
            print(f"{scene.name}: cached")
            continue
        if detector is None:  # same 102-class vocabulary in every ReplicaCAD scene
            detector = OpenVocabDetector(object_vocabulary(scene.classes))
            print(f"detector on {detector.device}, {len(detector.ids)} prompts")
        ids = scene.frame_ids()[: args.limit]
        t = time.time()
        dets = {i: detector(scene.rgb(i)) for i in ids}
        dt = time.time() - t
        n = sum(len(d.scores) for d in dets.values())
        print(f"{scene.name}: {len(ids)} frames in {dt:.1f} s ({1000 * dt / len(ids):.0f} ms/frame), "
              f"{n / len(ids):.1f} detections/frame")
        if args.limit is None:
            save_detections(out, dets)
