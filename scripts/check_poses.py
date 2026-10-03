"""Which pose convention does traj.txt use? Fuse overlapping frames and see which one aligns.

For pairs of nearby frames we back-project depth to 3D, move both clouds to the world frame
under each candidate convention and measure the median nearest-neighbour distance between
them. The right convention makes the two views of the same surfaces coincide (~cm).
Also saves a top-down view of many fused frames as a visual sanity check.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from s2m.data import backproject, load_scene, transform

OPENGL_FROM_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])  # x right, y up, z backward

CONVENTIONS = {
    "cam2world, OpenCV camera": lambda T: T,
    "cam2world, OpenGL camera": lambda T: T @ OPENGL_FROM_OPENCV,
    "world2cam (inverse)": lambda T: np.linalg.inv(T),
}


def pair_error(scene, i, j, to_world, stride=4):
    clouds = []
    for k in (i, j):
        pts, _ = backproject(scene.depth(k), scene.intrinsics, stride)
        clouds.append(transform(pts, to_world(scene.poses[k])))
    d, _ = cKDTree(clouds[1]).query(clouds[0], k=1)
    return float(np.median(d))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results/figures"))
    args = ap.parse_args()

    scene = load_scene(args.scene_dir)
    ids = scene.frame_ids()
    pairs = [(ids[k], ids[k + 1]) for k in range(0, len(ids) - 1, max(1, len(ids) // 8))]

    print(f"{scene.name}: {len(ids)} frames on disk, {len(scene.poses)} poses")
    for name, conv in CONVENTIONS.items():
        errs = [pair_error(scene, i, j, conv) for i, j in pairs]
        print(f"  {name:28s} median NN distance between neighbouring frames: "
              f"{np.median(errs) * 100:6.1f} cm  (pairs: {len(pairs)})")

    # Top-down view (Habitat world: y is up, floor is the x-z plane) under the first convention.
    pts_all, cls_all = [], []
    for i in ids[:: max(1, len(ids) // 60)]:
        pts, pix = backproject(scene.depth(i), scene.intrinsics, stride=8)
        sem = scene.semantic(i).ravel()[pix]
        pts_all.append(transform(pts, scene.poses[i]))
        cls_all.append(sem)
    P, C = np.concatenate(pts_all), np.concatenate(cls_all)
    cam = scene.poses[:, :3, 3]

    args.out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(P[:, 0], P[:, 2], c=C % 20, cmap="tab20", s=0.3)
    ax.plot(cam[:, 0], cam[:, 2], "k-", lw=1, label="camera trajectory")
    ax.set_aspect("equal")
    ax.set_xlabel("x, m")
    ax.set_ylabel("z, m")
    ax.set_title(f"{scene.name}: fused frames, top-down (colour = GT class)")
    ax.legend(loc="upper right")
    fig.savefig(args.out / f"{scene.name}_topdown.png", dpi=150, bbox_inches="tight")
    print(f"  height of points (y): p1={np.percentile(P[:, 1], 1):.2f} m, "
          f"p99={np.percentile(P[:, 1], 99):.2f} m; camera y={cam[:, 1].mean():.2f} m")
    print(f"  trajectory length: {np.linalg.norm(np.diff(cam, axis=0), axis=1).sum():.1f} m")
    print(f"  saved {args.out / (scene.name + '_topdown.png')}")
