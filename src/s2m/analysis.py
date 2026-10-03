"""Calibration arms, evaluated per avoid class over random calibration/test splits of scenes.

Unit of calibration = one scene with one drift realization (scores are per scene, so cells,
frames and objects inside a scene never count as separate samples). For every split,
calibration scenes contribute one realization each (drawn at random); test scenes contribute
all their realizations. A guarantee is per class k: "every true footprint of class k lies
inside the map's class-k region (threshold lam) dilated by the arm's radius", in the planner
frame. An arm that cannot produce a finite radius ABSTAINS: its certificate is vacuous, which
is valid but useless, so abstention is reported separately and counted as covered.

All arms except label_cell share the same region (threshold LAM0) and differ only in what the
radius is calibrated on:
  uncalibrated : radius 0 (the map is trusted as is)
  label_only   : semantic error only - miss distance of the predicted map at ground-truth poses
  pose_only    : pose error only - miss distance of a GT-label map built with drifted poses
  separate     : label_only radius + pose_only radius, each quantile at alpha (naive composition)
  joint        : one quantile of the miss distance of the drifted predicted map (this work)
  label_cell   : reference: per-cell label-space CP (Sundarsingh et al.-style), no inflation
A Bonferroni split (alpha/2 + alpha/2) needs n_cal >= 19 at alpha = 0.1 and is reported as such.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from s2m.conformal import UNCOVERABLE, conformal_quantile
from s2m.entities import AVOID_CLASSES
from s2m.experiment import LAMBDAS, R_MAX

TOL = 0.05  # one grid cell: "inside the region" for arms without inflation
LAM0 = 0.05  # region threshold shared by the arms, chosen on the dev scene
ARMS = ("uncalibrated", "label_cell", "label_only", "pose_only", "separate", "joint")


@dataclass(frozen=True)
class ArmParams:
    lam: float  # map threshold defining the class region
    radius: float  # extra keep-out distance around the region (metres)


def load_scores(score_dir: Path, exclude=()) -> list[dict]:
    rows = []
    for p in sorted(Path(score_dir).glob("*.csv")):
        if p.stem in exclude:
            continue
        for r in csv.DictReader(open(p)):
            rows.append({k: (v if k in ("scene", "level") else float(v)) for k, v in r.items()})
    return rows


def miss_at(row: dict, cls: str, lam: float) -> float:
    """Miss distance of class `cls` at threshold lam (nearest stored lambda not above lam,
    i.e. never a smaller region than asked for)."""
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

    r_label = _radius([miss_at(r, cls, lam0) for r in cal_L0], alpha)
    r_pose = _radius([r[f"miss_gtlab_{cls}"] for r in cal], alpha)
    r_joint = _radius([miss_at(r, cls, lam0) for r in cal], alpha)
    out["label_only"] = None if r_label is None else ArmParams(lam0, r_label)
    out["pose_only"] = None if r_pose is None else ArmParams(lam0, r_pose)
    out["separate"] = None if r_label is None or r_pose is None else ArmParams(lam0, r_label + r_pose)
    out["joint"] = None if r_joint is None else ArmParams(lam0, r_joint)
    return out


def covers(row: dict, cls: str, p: ArmParams | None) -> bool:
    """Abstention (p is None) is a vacuous, hence valid, certificate."""
    return True if p is None else miss_at(row, cls, p.lam) <= max(p.radius, TOL) + 1e-9


def run_splits(rows: list[dict], level: str, alpha: float, n_cal: int, n_splits: int,
               classes=AVOID_CLASSES, lam0: float = LAM0, seed: int = 0) -> dict:
    """stats[cls][arm] -> lists over splits of coverage, radius, abstained."""
    rng = np.random.default_rng(seed)
    by_scene: dict[str, dict[str, list]] = {}
    for r in rows:
        by_scene.setdefault(r["scene"], {}).setdefault(r["level"], []).append(r)
    scenes = sorted(by_scene)
    stats = {c: {a: {"coverage": [], "radius": [], "abstain": []} for a in ARMS} for c in classes}
    for _ in range(n_splits):
        perm = rng.permutation(scenes)
        cal_s, test_s = perm[:n_cal], perm[n_cal:]
        pick = lambda s, lv: by_scene[s][lv][rng.integers(len(by_scene[s][lv]))]
        cal = [pick(s, level) for s in cal_s]
        cal_L0 = [by_scene[s]["L0"][0] for s in cal_s]
        test = [r for s in test_s for r in by_scene[s][level]]
        for cls in classes:
            for arm, p in calibrate_class(cal, cal_L0, cls, alpha, lam0).items():
                st = stats[cls][arm]
                st["abstain"].append(float(p is None))
                st["coverage"].append(float(np.mean([covers(t, cls, p) for t in test])))
                st["radius"].append(np.nan if p is None else p.radius)
    return stats
