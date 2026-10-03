"""Small shared helpers for result files: CSV in/out, mission rows, drift-level labels."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

DEV_SCENES = ("apt_0",)  # every design choice was made on this scene; it is never calibrated or tested on
LEVELS = ("L0", "L1", "L2", "L3", "L4", "L5")


def write_csv(path, rows: list[dict], fields: list[str] | None = None) -> None:
    """Write dict rows; a header-only file when `rows` is empty (resumable scripts treat it as done)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def read_csv(path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def mission_rows(directory) -> list[dict]:
    """All mission outcome rows of a results/missions*/ directory (parameter files skipped)."""
    rows = []
    for p in sorted(Path(directory).glob("*.csv")):
        if not p.name.startswith("params_"):
            rows += read_csv(p)
    return rows


def rate(rows: list[dict], key: str) -> float:
    """Fraction of rows whose boolean column `key` is true (nan for no rows)."""
    return float(np.mean([r[key] == "True" for r in rows])) if rows else float("nan")


def level_labels(score_rows: list[dict]) -> dict[str, str]:
    """Drift level -> median ATE over the scenes it was applied to, e.g. {"L2": "2 cm"}.

    Labels come from the realized trajectories in the score files, not from the calibration
    targets on the dev scene, so they describe the error the calibration actually saw.
    """
    out = {"L0": "GT pose"}
    for lv in sorted({r["level"] for r in score_rows} - {"L0"}):
        ate = float(np.median([float(r["ate"]) for r in score_rows if r["level"] == lv]))
        out[lv] = f"{ate * 100:.1g} cm" if ate < 0.01 else f"{ate * 100:.0f} cm" if ate < 1 else f"{ate:.1f} m"
    return out
