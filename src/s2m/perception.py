"""Open-vocabulary perception: YOLO-World detections turned into per-pixel labels.

Perception does not depend on the pose, so detections are computed once per frame and cached;
pose drift only changes where the labelled pixels land in the grid.

Box -> mask: a detection box also covers background. Two variants are supported:
  * "box": inside each box keep only pixels whose depth is close to the box's median depth
    (a cheap, depth-consistent instance mask; the default);
  * "sam": MobileSAM prompted with the box (ConceptGraphs-style instance masks). Masks are cached
    on the stride grid the map uses, so they are exact at every pixel that reaches the map.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Structural / background classes are not object prompts: they become unlabelled obstacles.
NON_OBJECT_CLASSES = ("other", "floor", "wall", "ceiling")


@dataclass
class Detections:
    boxes: np.ndarray  # (D, 4) xyxy in pixels
    class_ids: np.ndarray  # (D,) dataset class ids
    scores: np.ndarray  # (D,) detector confidence in [0, 1]

    @staticmethod
    def empty() -> "Detections":
        return Detections(np.zeros((0, 4), np.float32), np.zeros(0, np.int32), np.zeros(0, np.float32))


def object_vocabulary(classes: dict[int, str]) -> dict[int, str]:
    """Dataset class id -> text prompt, without structural classes."""
    return {k: v.replace("_", " ") for k, v in sorted(classes.items()) if v not in NON_OBJECT_CLASSES}


class OpenVocabDetector:
    """Thin wrapper around Ultralytics YOLO-World with a fixed text vocabulary."""

    def __init__(self, vocabulary: dict[int, str], weights: str = "models/yolov8s-worldv2.pt",
                 device: str | None = None, conf: float = 0.05):
        import torch
        from ultralytics import YOLOWorld

        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.conf = conf
        self.ids = np.array(list(vocabulary.keys()), dtype=np.int32)
        # set_classes needs CLIP (~340 MB) to embed the prompts. We embed once and save the model
        # with the vocabulary baked in, so later runs need neither CLIP nor the network.
        baked = Path(weights).with_name(Path(weights).stem + f"-vocab{len(vocabulary)}.pt")
        if baked.exists():
            self.model = YOLOWorld(str(baked))
        else:
            self.model = YOLOWorld(weights)
            self.model.set_classes(list(vocabulary.values()))
            self.model.model.clip_model = None  # keep the embeddings, drop the text encoder
            self.model.save(str(baked))

    def __call__(self, rgb: np.ndarray) -> Detections:
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])  # Ultralytics expects OpenCV-style BGR arrays
        res = self.model.predict(bgr, conf=self.conf, device=self.device, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return Detections.empty()
        b = res.boxes
        return Detections(b.xyxy.cpu().numpy().astype(np.float32),
                          self.ids[b.cls.cpu().numpy().astype(int)],
                          b.conf.cpu().numpy().astype(np.float32))


def save_detections(path: Path, dets: dict[int, Detections]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.array(sorted(dets), dtype=np.int32)
    counts = np.array([len(dets[f].scores) for f in frames], dtype=np.int32)
    cat = lambda attr: np.concatenate([getattr(dets[f], attr) for f in frames]) if len(frames) else []
    np.savez_compressed(path, frames=frames, counts=counts, boxes=cat("boxes"),
                        class_ids=cat("class_ids"), scores=cat("scores"))


def load_detections(path: Path) -> dict[int, Detections]:
    z = np.load(path)
    out, start = {}, 0
    for f, n in zip(z["frames"], z["counts"]):
        sl = slice(start, start + n)
        out[int(f)] = Detections(z["boxes"][sl], z["class_ids"][sl], z["scores"][sl])
        start += n
    return out


class BoxSegmenter:
    """MobileSAM prompted with detection boxes (Ultralytics SAM wrapper)."""

    def __init__(self, weights: str = "models/mobile_sam.pt", device: str | None = None):
        import torch
        from ultralytics import SAM

        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = SAM(weights)

    def __call__(self, rgb: np.ndarray, dets: Detections) -> np.ndarray:
        """(D, H, W) bool masks, one per detection box."""
        if len(dets.scores) == 0:
            return np.zeros((0, *rgb.shape[:2]), bool)
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        res = self.model.predict(bgr, bboxes=dets.boxes.tolist(), device=self.device, verbose=False)[0]
        return res.masks.data.cpu().numpy().astype(bool)


def save_masks(path: Path, masks: dict[int, np.ndarray], stride: int) -> None:
    """Masks subsampled on the stride grid, bit-packed; same frame/detection order as detections."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.array(sorted(masks), dtype=np.int32)
    sub = [masks[f][:, ::stride, ::stride] for f in frames]
    shape = sub[0].shape[1:]
    packed = np.concatenate([np.packbits(m.reshape(len(m), shape[0] * shape[1]), axis=1) for m in sub])
    np.savez_compressed(path, frames=frames, counts=[len(m) for m in sub], packed=packed,
                        shape=shape, stride=stride)


def load_masks(path: Path) -> tuple[dict[int, np.ndarray], int]:
    """Frame -> (D, h, w) bool masks on the stride grid, and the stride."""
    z = np.load(path)
    h, w = z["shape"]
    out, start = {}, 0
    for f, n in zip(z["frames"], z["counts"]):
        out[int(f)] = np.unpackbits(z["packed"][start:start + n], axis=1, count=h * w).reshape(n, h, w).astype(bool)
        start += n
    return out, int(z["stride"])


def upsample_masks(masks: np.ndarray, stride: int, shape: tuple[int, int]) -> np.ndarray:
    """Nearest upsampling to full resolution; exact on the stride grid the map samples."""
    return masks.repeat(stride, axis=1).repeat(stride, axis=2)[:, :shape[0], :shape[1]]


def pixel_labels(dets: Detections, depth: np.ndarray, depth_tol: float = 0.25,
                 min_score: float = 0.0, masks: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel (class id, score); -1 / 0 where no detection claims the pixel.

    Detections are painted from lowest to highest score, so overlapping pixels keep the most
    confident detection. Without `masks`, only pixels inside the box within `depth_tol` metres of
    the box's median depth are kept; with `masks` (one per detection) the mask pixels with depth.
    """
    h, w = depth.shape
    labels = np.full((h, w), -1, dtype=np.int32)
    scores = np.zeros((h, w), dtype=np.float32)
    for j in np.argsort(dets.scores):
        if dets.scores[j] < min_score:
            continue
        if masks is not None:
            keep = masks[j] & (depth > 0)
            labels[keep] = dets.class_ids[j]
            scores[keep] = dets.scores[j]
            continue
        x0, y0, x1, y1 = np.clip(np.round(dets.boxes[j]).astype(int), 0, [w, h, w, h])
        if x1 <= x0 or y1 <= y0:
            continue
        d = depth[y0:y1, x0:x1]
        valid = d > 0
        if not valid.any():
            continue
        keep = valid & (np.abs(d - np.median(d[valid])) < depth_tol)
        labels[y0:y1, x0:x1][keep] = dets.class_ids[j]
        scores[y0:y1, x0:x1][keep] = dets.scores[j]
    return labels, scores
