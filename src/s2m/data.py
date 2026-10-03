"""OSMa-Bench (ReplicaCAD part): download a frame subset and load scenes.

Dataset: https://huggingface.co/datasets/warmhammer/OSMa-Bench_dataset (CC BY 4.0).
Each scene folder contains traj.txt (one 4x4 camera pose per frame, row-major),
camera_params.yaml, embed_semseg_classes.json and results/ with
frameNNNNNN.jpg (RGB), depthNNNNNN.png (uint16) and semanticNNNNNN.png (uint16 class ids).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

REPO_ID = "warmhammer/OSMa-Bench_dataset"
SCENE_ROOT = "data/replica_cad"
META_FILES = ["traj.txt", "camera_params.yaml", "embed_semseg_classes.json", "log.txt"]


BASE_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main"


def download_scene(scene: str, out_dir: Path, step: int = 8, workers: int = 8) -> Path:
    """Download metadata and every `step`-th frame (RGB, depth, semantic) of one scene.

    Plain HTTPS GETs in a small thread pool: hf_hub_download makes several requests per
    file and was ~3.5 s/file. Unauthenticated HF limit is ~3000 file requests / 5 min.
    """
    from concurrent.futures import ThreadPoolExecutor

    scene_dir = Path(out_dir) / scene
    for name in META_FILES:
        _fetch(f"{scene}/{name}", out_dir)

    n_frames = len(np.loadtxt(scene_dir / "traj.txt", ndmin=2))
    rel_paths = [f"{scene}/results/{prefix}{i:06d}.{ext}"
                 for i in range(0, n_frames, step)
                 for prefix, ext in (("frame", "jpg"), ("depth", "png"), ("semantic", "png"))]
    with ThreadPoolExecutor(workers) as pool:
        ok = list(pool.map(lambda p: _fetch(p, out_dir), rel_paths))
    failed = [p for p, good in zip(rel_paths, ok) if not good]
    if failed:  # re-running the download skips existing files, so this is recoverable
        print(f"{scene}: {len(failed)} files failed, e.g. {failed[0]}")
    return scene_dir


def _fetch(rel_path: str, out_dir: Path, retries: int = 6) -> bool:
    """Download one file; returns False instead of raising after the last retry."""
    import time
    import urllib.error
    import urllib.request

    target = Path(out_dir) / rel_path
    if target.exists():
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{SCENE_ROOT}/{rel_path}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            tmp = target.with_suffix(target.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(target)  # never leave half-written files under the final name
            return True
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            time.sleep(min(60, 5 * 2 ** attempt))
    return False


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    depth_scale: float  # PNG value / depth_scale = metres


@dataclass
class Scene:
    name: str
    root: Path
    intrinsics: Intrinsics
    poses: np.ndarray  # (N, 4, 4), camera-to-world as stored in traj.txt
    classes: dict[int, str]

    def frame_ids(self) -> list[int]:
        """Frame indices that are present on disk (the downloaded subset)."""
        ids = [int(p.stem[len("depth"):]) for p in (self.root / "results").glob("depth*.png")]
        return sorted(ids)

    def depth(self, i: int) -> np.ndarray:
        raw = np.asarray(Image.open(self.root / "results" / f"depth{i:06d}.png"), dtype=np.float32)
        return raw / self.intrinsics.depth_scale  # metres, 0 = no data

    def semantic(self, i: int) -> np.ndarray:
        return np.asarray(Image.open(self.root / "results" / f"semantic{i:06d}.png")).astype(np.int32)

    def rgb(self, i: int) -> np.ndarray:
        return np.asarray(Image.open(self.root / "results" / f"frame{i:06d}.jpg").convert("RGB"))


def load_scene(scene_dir: Path) -> Scene:
    scene_dir = Path(scene_dir)
    cam = yaml.safe_load((scene_dir / "camera_params.yaml").read_text())["camera_params"]
    intr = Intrinsics(cam["fx"], cam["fy"], cam["cx"], cam["cy"],
                      int(cam["image_width"]), int(cam["image_height"]), float(cam["png_depth_scale"]))
    poses = np.loadtxt(scene_dir / "traj.txt", ndmin=2).reshape(-1, 4, 4)
    classes = _load_classes(scene_dir / "embed_semseg_classes.json")
    return Scene(scene_dir.name, scene_dir, intr, poses, classes)


def _load_classes(path: Path) -> dict[int, str]:
    raw = json.loads(path.read_text())
    items = raw if isinstance(raw, list) else raw.get("classes", raw)
    if isinstance(items, dict):
        return {int(k): str(v) for k, v in items.items()}
    return {int(c["id"]): str(c["name"]) for c in items}


def backproject(depth: np.ndarray, intr: Intrinsics, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Depth image -> 3D points in the camera frame (OpenCV axes: x right, y down, z forward).

    Returns (points (M, 3), flat pixel indices (M,)) for pixels with valid depth.
    """
    h, w = depth.shape
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[::stride, ::stride]
    valid = z > 0
    x = (u[valid] - intr.cx) * z[valid] / intr.fx
    y = (v[valid] - intr.cy) * z[valid] / intr.fy
    pix = (v[valid] * w + u[valid]).astype(np.int64)
    return np.stack([x, y, z[valid]], axis=1), pix


def transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    return points @ T[:3, :3].T + T[:3, 3]
