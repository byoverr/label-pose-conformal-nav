"""Download metadata and every k-th frame of OSMa-Bench ReplicaCAD scenes.

Example: python scripts/download_subset.py apt_0 --step 8
"""

import argparse
from pathlib import Path

from s2m.data import download_scene

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--step", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("data/replica_cad"))
    args = ap.parse_args()
    for scene in args.scenes:
        path = download_scene(scene, args.out, step=args.step)
        print(f"{scene}: {path}")
