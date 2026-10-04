"""Calibration arms, evaluated per avoid class over random calibration/test splits of scenes.

Unit of calibration = one scene with one drift realization (scores are per scene, so cells,
frames and objects inside a scene never count as separate samples). For every split,
calibration scenes contribute one realization each (drawn at random); test scenes contribute
all their realizations. A guarantee is per class k: "every true footprint of class k lies
inside the map's class-k region (threshold lam) dilated by the arm's radius", in the planner
frame. An arm that cannot produce a finite radius ABSTAINS: its certificate is vacuous, which
is valid but useless. The conformal guarantee is P(abstain or covered) >= 1 - alpha, so
abstention counts as covered in `coverage`; coverage among certified splits and the
certification rate are reported next to it.

All arms except label_cell and geometry share the same region (threshold LAM0) and differ only
in what the radius is calibrated on:
  uncalibrated : radius 0 (the map is trusted as is)
  label_only   : semantic error only - miss distance of the predicted map at ground-truth poses
  pose_only    : pose error only - miss distance of a GT-label map built with drifted poses
  separate     : label_only radius + pose_only radius, each quantile at alpha (naive composition)
  joint        : one quantile of the miss distance of the drifted predicted map (this work)
  label_cell   : reference: per-cell label-space CP (Sundarsingh et al.-style), no inflation
  geometry     : no semantics: every occupied cell may be the class (region = all occupied cells,
                 lam = 0), radius calibrated on the drifted map (Perceive-with-Confidence-style).
                 It is also the fallback for a class whose own certificate abstains in missions.
Arms without a radius (uncalibrated, label_cell) count as covering a miss of up to one cell (TOL),
which favours them.
A Bonferroni split (alpha/2 + alpha/2) would need n_cal >= 19 at alpha = 0.1, more than the 13 available,
so no such arm is evaluated.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from s2m.conformal import UNCOVERABLE, conformal_quantile
from s2m.entities import AVOID_CLASSES
from s2m.experiment import LAMBDAS, R_MAX
from s2m.io import read_csv

TOL = 0.05  # one grid cell: "inside the region" for arms without inflation
LAM0 = 0.05  # region threshold shared by the arms, chosen on the dev scene
ARMS = ("uncalibrated", "label_cell", "geometry", "label_only", "pose_only", "separate", "joint")


@dataclass(frozen=True)
class ArmParams:
    lam: float  # map threshold defining the class region
    radius: float  # extra keep-out distance around the region (metres)


def load_scores(score_dir: Path, exclude=()) -> list[dict]:
    rows = []
    for p in sorted(Path(score_dir).glob("*.csv")):
        if p.stem in exclude:
            continue
        for r in read_csv(p):
            rows.append({k: (v if k in ("scene", "level") else float(v)) for k, v in r.items()})
    return rows


def miss_at(row: dict, cls: str, lam: float) -> float:
    """Miss distance of class `cls` at threshold lam (nearest stored lambda not above lam,
    i.e. never a smaller region than asked for)."""
    if lam > LAMBDAS[-1] + 1e-9:
        raise ValueError(f"lambda {lam} is above the largest stored threshold {LAMBDAS[-1]}")
    i = np.searchsorted(LAMBDAS, lam + 1e-9) - 1
    return row[f"miss_pred_{cls}_{LAMBDAS[max(i, 0)]:.2f}"]


def _radius(scores, alpha: float) -> float | None:
    q = conformal_quantile(scores, alpha)
    return None if not math.isfinite(q) or q >= R_MAX else q


def calibrate_class(cal: list[dict], cal_L0: list[dict], cls: str, alpha: float,
                    lam0: float = LAM0) -> dict[str, ArmParams | None]:
    """Parameters of every arm for one class (None = abstains)."""
    out: dict[str, ArmParams | None] = {"uncalibrated": ArmParams(lam0, 0.0)}

    q = conformal_quantile([r[f"label_score_{cls}"] for r in cal_L0], alpha)
    out["label_cell"] = None if q >= UNCOVERABLE - 1e-9 else ArmParams(max(0.0, 1.0 - q), 0.0)

    r_geo = _radius([miss_at(r, cls, 0.0) for r in cal], alpha)
    out["geometry"] = None if r_geo is None else ArmParams(0.0, r_geo)

    r_label = _radius([miss_at(r, cls, lam0) for r in cal_L0], alpha)
    r_pose = _radius([r[f"miss_gtlab_{cls}"] for r in cal], alpha)
    r_joint = _radius([miss_at(r, cls, lam0) for r in cal], alpha)
    out["label_only"] = None if r_label is None else ArmParams(lam0, r_label)
    out["pose_only"] = None if r_pose is None else ArmParams(lam0, r_pose)
    out["separate"] = None if r_label is None or r_pose is None else ArmParams(lam0, r_label + r_pose)
    out["joint"] = None if r_joint is None else ArmParams(lam0, r_joint)
    return out


def calibrated_params(rows: list[dict], cal_scenes, levels, alpha: float, rng: np.random.Generator,
                      cls_ids: dict[str, int]):
    """Parameters of every arm for planning in a scene that is NOT among `cal_scenes`.

    Returns params[level][arm][class_id] (ArmParams or None = abstains) and fallback[level][class_id],
    the geometry-only radius used when a class abstains (None if that abstains too). Each calibration
    scene contributes one random drift realization per level.
    """
    by: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        by.setdefault((r["scene"], r["level"]), []).append(r)
    params, fallback = {}, {}
    for lv in levels:
        cal = [by[(s, lv)][rng.integers(len(by[(s, lv)]))] for s in cal_scenes]
        cal_L0 = [by[(s, "L0")][0] for s in cal_scenes]
        per_class = {cid: calibrate_class(cal, cal_L0, name, alpha) for name, cid in cls_ids.items()}
        params[lv] = {arm: {cid: per_class[cid][arm] for cid in per_class} for arm in ARMS}
        fallback[lv] = {cid: None if per_class[cid]["geometry"] is None else per_class[cid]["geometry"].radius
                        for cid in per_class}
    return params, fallback


def read_params(path, level: str, class_ids: dict[str, int]):
    """Inverse of the params_<scene>.csv files written by scripts/run_missions.py."""
    params, fallback = {}, {}
    for r in read_csv(path):
        if r["level"] != level or r["class"] not in class_ids:
            continue
        cid = class_ids[r["class"]]
        params.setdefault(r["arm"], {})[cid] = None if r["lam"] == "" else ArmParams(float(r["lam"]), float(r["radius"]))
        fallback[cid] = None if r["fallback_radius"] == "" else float(r["fallback_radius"])
    return params, fallback


def covers(row: dict, cls: str, p: ArmParams | None) -> bool:
    """Abstention (p is None) is a vacuous, hence valid, certificate."""
    return True if p is None else miss_at(row, cls, p.lam) <= max(p.radius, TOL) + 1e-9


MAX_EXACT_SPLITS = 2000  # enumerate every calibration set when there are at most this many


def split_sets(scenes: list[str], n_cal: int, n_splits: int, rng: np.random.Generator) -> list[list[str]]:
    """Calibration scene sets: all of them if few enough, else `n_splits` random ones."""
    if math.comb(len(scenes), n_cal) <= MAX_EXACT_SPLITS:
        return [list(c) for c in itertools.combinations(scenes, n_cal)]
    return [list(rng.permutation(scenes)[:n_cal]) for _ in range(n_splits)]


def run_splits(rows: list[dict], level: str, alpha: float, n_cal: int, n_splits: int,
               classes=AVOID_CLASSES, lam0: float = LAM0, seed: int = 0) -> dict:
    """stats[cls][arm] -> lists over splits of coverage (abstention = covered), certified coverage
    (NaN when the arm abstains), radius and abstained."""
    rng = np.random.default_rng(seed)
    by_scene: dict[str, dict[str, list]] = {}
    for r in rows:
        by_scene.setdefault(r["scene"], {}).setdefault(r["level"], []).append(r)
    scenes = sorted(by_scene)
    stats = {c: {a: {"coverage": [], "coverage_cert": [], "radius": [], "abstain": []} for a in ARMS}
             for c in classes}
    for cal_s in split_sets(scenes, n_cal, n_splits, rng):
        test_s = [s for s in scenes if s not in cal_s]
        pick = lambda s, lv: by_scene[s][lv][rng.integers(len(by_scene[s][lv]))]
        cal = [pick(s, level) for s in cal_s]
        cal_L0 = [by_scene[s]["L0"][0] for s in cal_s]
        test = [r for s in test_s for r in by_scene[s][level]]
        for cls in classes:
            for arm, p in calibrate_class(cal, cal_L0, cls, alpha, lam0).items():
                st = stats[cls][arm]
                cov = float(np.mean([covers(t, cls, p) for t in test]))
                st["abstain"].append(float(p is None))
                st["coverage"].append(cov)
                st["coverage_cert"].append(np.nan if p is None else cov)
                st["radius"].append(np.nan if p is None else p.radius)
    return stats
