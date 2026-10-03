"""Build and plot the ground-truth 2D semantic grid of one scene."""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from s2m.data import load_scene
from s2m.grid import build_gt_grid

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--res", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=Path("results/figures"))
    args = ap.parse_args()

    scene = load_scene(args.scene_dir)
    t = time.time()
    grid = build_gt_grid(scene, res=args.res)
    occ, free, lab = grid.occupied(), grid.free(), grid.label()
    print(f"{scene.name}: grid {grid.spec.shape} at {args.res} m, built in {time.time() - t:.1f} s")
    print(f"  free {free.mean():.1%}, occupied {occ.mean():.1%}, unknown {1 - free.mean() - occ.mean():.1%}")
    present = np.unique(lab[lab >= 0])
    print(f"  {len(present)} classes in occupied cells:",
          ", ".join(scene.classes[k] for k in present[:25]), "..." if len(present) > 25 else "")

    img = np.ones((*grid.spec.shape, 3))
    img[free] = 0.85
    cmap = plt.get_cmap("tab20")
    img[occ] = cmap((lab[occ] % 20) / 19)[:, :3]
    x0, z0, res = grid.spec.x0, grid.spec.z0, grid.spec.res
    extent = [x0, x0 + grid.spec.shape[1] * res, z0, z0 + grid.spec.shape[0] * res]

    args.out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.imshow(img, origin="lower", extent=extent, interpolation="nearest")
    cam = scene.poses[:, :3, 3]
    ax.plot(cam[:, 0], cam[:, 2], "k-", lw=0.8, label="trajectory")
    ax.set_xlabel("x, m")
    ax.set_ylabel("z, m")
    ax.set_title(f"{scene.name}: GT grid (grey = free, white = unknown, colour = class)")
    ax.legend(loc="upper right")
    fig.savefig(args.out / f"{scene.name}_gt_grid.png", dpi=150, bbox_inches="tight")
    print(f"  saved {args.out / (scene.name + '_gt_grid.png')}")
