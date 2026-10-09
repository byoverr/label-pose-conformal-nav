"""Could a conditional margin be smaller where the error is small? The most accessible feature for the robot is
the length of the path it has travelled while mapping. Spearman correlation between that length and the joint
miss distance (region at LAM0) over the 21 evaluation scenes and all drift realizations, per level and class.
Writes results/tables/conditional_margin.csv.

Example: python scripts/check_conditional.py
"""

from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from s2m.analysis import LAM0, load_scores, miss_at
from s2m.data import load_scene
from s2m.drift import path_length
from s2m.io import DEV_SCENES, write_csv

if __name__ == "__main__":
    rows = load_scores(Path("results/scores"), exclude=DEV_SCENES)
    length = {s: path_length(load_scene(Path("data/replica_cad") / s).poses) for s in sorted({r["scene"] for r in rows})}
    out = []
    for lv in ("L2", "L3", "L4"):
        for cls in ("indoor_plant", "bike"):
            rr = [r for r in rows if r["level"] == lv]
            rho, p = spearmanr([length[r["scene"]] for r in rr], [miss_at(r, cls, LAM0) for r in rr])
            out.append({"level": lv, "class": cls, "n": len(rr), "spearman": float(rho), "p_value": float(p)})
            print(f"{lv} {cls}: rho = {rho:.2f} (p = {p:.2f}, n = {len(rr)})")
    write_csv("results/tables/conditional_margin.csv", out)
