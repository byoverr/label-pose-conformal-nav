"""Detection cascade: the union of several open-vocabulary detectors for recall, a CLIP check of every
box for precision.

Stage 1 takes every detection of every detector (confidence >= 0.05, as cached). Boxes of one frame that
overlap strongly (IoU >= 0.7) form a cluster and share one crop. Stage 2 classifies the crop with CLIP
(ViT-B/16) over the object vocabulary plus the structural classes wall / floor / ceiling, so a box on a bare
wall can lose to "wall". Stage 3 keeps a detection of class k if CLIP gives k at least `tau` of the
probability ("verify"), and optionally adds a detection of the CLIP class j when CLIP is confident in it
(p_j >= `tau_relabel`) although no detector proposed j for that box ("relabel": recovers objects the
detectors saw but named wrongly). Thresholds are chosen on the development scene apt_0 only.

Multi-frame voting needs no extra stage: the map keeps a cell's class only if the class gets at least
LAM0 of the cell's score-weighted observations over all frames.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from s2m.perception import NON_OBJECT_CLASSES, Detections

STRUCTURAL_PROMPTS = ("wall", "floor", "ceiling")
CLUSTER_IOU = 0.7


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of xyxy boxes (n, 4) x (m, 4)."""
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


def cluster_boxes(boxes: np.ndarray, scores: np.ndarray, thr: float = CLUSTER_IOU) -> tuple[np.ndarray, np.ndarray]:
    """Greedy clustering by score: (cluster id per box, representative box per cluster)."""
    cid = np.full(len(boxes), -1)
    reps = []
    for j in np.argsort(-scores):
        if reps:
            o = iou(boxes[j:j + 1], np.array(reps))[0]
            k = int(np.argmax(o))
            if o[k] >= thr:
                cid[j] = k
                continue
        cid[j] = len(reps)
        reps.append(boxes[j])
    return cid, np.array(reps).reshape(-1, 4)


def square_crop(rgb: np.ndarray, box: np.ndarray, pad: float = 0.1) -> np.ndarray:
    """Square crop around the box (side = longer box side + padding), clipped to the image."""
    h, w = rgb.shape[:2]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    half = max(box[2] - box[0], box[3] - box[1]) * (1 + pad) / 2 + 2
    x0, x1 = int(max(0, cx - half)), int(min(w, cx + half))
    y0, y1 = int(max(0, cy - half)), int(min(h, cy + half))
    return rgb[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]


class ClipVerifier:
    """Zero-shot CLIP classification of crops over a fixed list of class prompts."""

    def __init__(self, prompts: list[str], weights_dir: str = "models/clip", name: str = "ViT-B/16",
                 device: str | None = None):
        import clip
        import torch

        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model, self.preprocess = clip.load(name, device=self.device, download_root=weights_dir)
        self.model.eval()
        with torch.no_grad():
            tokens = clip.tokenize([f"a photo of a {p}." for p in prompts]).to(self.device)
            t = self.model.encode_text(tokens).float()
        self.text = t / t.norm(dim=-1, keepdim=True)

    def __call__(self, crops: list[np.ndarray], batch: int = 128) -> np.ndarray:
        from PIL import Image

        out = []
        for i in range(0, len(crops), batch):
            x = self.torch.stack([self.preprocess(Image.fromarray(c)) for c in crops[i:i + batch]]).to(self.device)
            with self.torch.no_grad():
                f = self.model.encode_image(x).float()
            f = f / f.norm(dim=-1, keepdim=True)
            out.append((100.0 * f @ self.text.T).softmax(dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.text)), np.float32)


def verifier_prompts(classes: dict[int, str]) -> tuple[np.ndarray, list[str]]:
    """Dataset class ids of the object prompts (structural prompts get id -1) and the prompt texts."""
    objects = {k: v.replace("_", " ") for k, v in sorted(classes.items()) if v not in NON_OBJECT_CLASSES}
    ids = np.array(list(objects) + [-1] * len(STRUCTURAL_PROMPTS), np.int32)
    return ids, list(objects.values()) + list(STRUCTURAL_PROMPTS)


def verify_scene(scene, detector_sets: list[dict[int, Detections]], verifier: ClipVerifier) -> dict:
    """CLIP probabilities for every detection of every detector, one crop per box cluster.
    Returns {"probs": (clusters, prompts) float16, "cluster_<i>": cluster id per detection of detector i
    in cache order (frames sorted)}."""
    frames = sorted(set().union(*[set(d) for d in detector_sets]))
    probs, cl = [], [[] for _ in detector_sets]
    base = 0
    for f in frames:
        per = [d.get(f, Detections.empty()) for d in detector_sets]
        boxes = np.concatenate([p.boxes for p in per]) if per else np.zeros((0, 4))
        scores = np.concatenate([p.scores for p in per]) if per else np.zeros(0)
        if len(boxes):
            cid, reps = cluster_boxes(boxes, scores)
            rgb = scene.rgb(f)
            probs.append(verifier([square_crop(rgb, b) for b in reps]))
        else:
            cid = np.zeros(0, int)
            reps = np.zeros((0, 4))
        start = 0
        for i, p in enumerate(per):
            cl[i].append(cid[start:start + len(p.scores)] + base)
            start += len(p.scores)
        base += len(reps)
    out = {"probs": (np.concatenate(probs) if probs else np.zeros((0, 1))).astype(np.float16)}
    for i, c in enumerate(cl):
        out[f"cluster_{i}"] = np.concatenate(c).astype(np.int32) if c else np.zeros(0, np.int32)
    return out


def cascade(detector_sets: list[dict[int, Detections]], verified: dict, prompt_ids: np.ndarray,
            tau: float, tau_relabel: float | None = None) -> dict[int, Detections]:
    """One detection set from several detectors and their CLIP probabilities (see the module docstring)."""
    col = {int(k): j for j, k in enumerate(prompt_ids) if k >= 0}
    probs = verified["probs"].astype(np.float32)
    out = {}
    offsets = [0] * len(detector_sets)
    frames = sorted(set().union(*[set(d) for d in detector_sets]))
    for f in frames:
        boxes, cls, sc, cids = [], [], [], []
        for i, d in enumerate(detector_sets):
            det = d.get(f, Detections.empty())
            n = len(det.scores)
            c = verified[f"cluster_{i}"][offsets[i]:offsets[i] + n]
            offsets[i] += n
            p = probs[c, [col[int(k)] for k in det.class_ids]] if n else np.zeros(0)
            keep = p >= tau
            boxes.append(det.boxes[keep]); cls.append(det.class_ids[keep]); sc.append(det.scores[keep])
            cids.append((c, det.class_ids, det.scores))
        if tau_relabel is not None:
            seen = {}
            for c, k, s in cids:
                for cj, kj, sj in zip(c, k, s):
                    seen.setdefault(int(cj), [set(), 0.0])
                    seen[int(cj)][0].add(int(kj))
                    seen[int(cj)][1] = max(seen[int(cj)][1], float(sj))
            for cj, (proposed, smax) in seen.items():
                j = int(np.argmax(probs[cj]))
                k = int(prompt_ids[j])
                if k >= 0 and k not in proposed and probs[cj, j] >= tau_relabel:
                    # the box of the cluster's most confident detection, with that detection's score
                    for (c, kk, s), det in zip(cids, [d.get(f, Detections.empty()) for d in detector_sets]):
                        hit = np.flatnonzero(c == cj)
                        if len(hit):
                            b = hit[np.argmax(s[hit])]
                            boxes.append(det.boxes[b:b + 1]); cls.append(np.array([k], np.int32))
                            sc.append(np.array([smax], np.float32))
                            break
        out[f] = Detections(np.concatenate(boxes).astype(np.float32).reshape(-1, 4),
                            np.concatenate(cls).astype(np.int32), np.concatenate(sc).astype(np.float32))
    return out


def save_verified(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def load_verified(path: Path) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}
