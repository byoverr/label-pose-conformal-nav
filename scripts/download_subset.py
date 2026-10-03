"""Download metadata and every k-th frame of OSMa-Bench ReplicaCAD scenes.

Example: python scripts/download_subset.py apt_0 --step 8
         python scripts/download_subset.py apt_0 --step 2 --rgbd-only --out data/replica_cad_vo
"""

import argparse
from pathlib import Path

from s2m.data import FRAME_KINDS, download_scene

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--step", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("data/replica_cad"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--root", default="data/replica_cad", help="dataset folder on HuggingFace (data/hm3d for HM3D)")
    ap.add_argument("--rgbd-only", action="store_true", help="skip semantic frames (for odometry)")
    args = ap.parse_args()
    kinds = FRAME_KINDS[:2] if args.rgbd_only else FRAME_KINDS
    for scene in args.scenes:
        path = download_scene(scene, args.out, step=args.step, workers=args.workers, kinds=kinds, root=args.root)
        print(f"{scene}: {path}")
