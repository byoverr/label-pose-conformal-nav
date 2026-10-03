"""LaTeX tables for the report, generated from the result CSVs (no hand-typed numbers).

Writes report/tables/*.tex.
"""

import csv
from pathlib import Path

OUT = Path("report/tables")
ARM_RU = {
    "uncalibrated": "без калибровки",
    "label_cell": "покадровая CP меток",
    "label_only": "только метки",
    "pose_only": "только поза",
    "separate": "раздельно (сумма)",
    "joint": "\\textbf{совместно}",
    "oracle": "оракул",
}
CLASS_RU = {"indoor_plant": "растение", "tv_stand": "тумба под ТВ", "bike": "велосипед"}


def fmt(v, digits=2):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "---"
    return "---" if x != x else f"{x:.{digits}f}".replace(".", "{,}")


def coverage_table():
    rows = list(csv.DictReader(open("results/tables/coverage.csv")))
    levels = ("L0", "L2", "L3", "L4")
    arms = ("uncalibrated", "label_cell", "label_only", "pose_only", "separate", "joint")
    lines = ["\\begin{tabular}{ll" + "cc" * len(levels) + "}", "\\toprule",
             "Класс & Ветка & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{lv}}}" for lv in levels) + " \\\\",
             " & & " + " & ".join(["покр. & $\\hat q$, м"] * len(levels)) + " \\\\", "\\midrule"]
    for cls in ("bike", "indoor_plant", "tv_stand"):
        for i, arm in enumerate(arms):
            cells = []
            for lv in levels:
                r = next(x for x in rows if x["level"] == lv and x["class"] == cls and x["arm"] == arm)
                abst = float(r["abstain_rate"])
                cov = fmt(r["coverage_mean"])
                rad = fmt(r["radius_median"]) if arm in ("label_only", "pose_only", "separate", "joint") else "0"
                if abst >= 0.5:
                    rad = f"отк.\\,{fmt(abst, 2)}"
                cells += [cov, rad]
            name = CLASS_RU[cls] if i == 0 else ""
            lines.append(f"{name} & {ARM_RU[arm]} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule" if cls != "tv_stand" else "\\bottomrule")
    lines.append("\\end{tabular}")
    (OUT / "coverage.tex").write_text("\n".join(lines) + "\n")


def missions_table(tag=""):
    path = Path(f"results/tables/missions{tag}.csv")
    if not path.exists():
        return
    rows = list(csv.DictReader(open(path)))
    levels = ("L0", "L2", "L4")
    arms = ("uncalibrated", "label_cell", "label_only", "pose_only", "separate", "joint", "oracle")
    lines = ["\\begin{tabular}{l" + "ccc" * len(levels) + "}", "\\toprule",
             "Ветка & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{lv}}}" for lv in levels) + " \\\\",
             " & " + " & ".join(["путь & наруш. & успех"] * len(levels)) + " \\\\", "\\midrule"]
    for arm in arms:
        cells = []
        for lv in levels:
            r = next(x for x in rows if x["level"] == lv and x["arm"] == arm)
            cells += [fmt(r["planned"]), fmt(r["violation_given_planned"]), fmt(r["success"])]
        lines.append(f"{ARM_RU[arm]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / f"missions{tag}.tex").write_text("\n".join(lines) + "\n")


def bridge_table(tag=""):
    path = Path(f"results/tables/planner_bridge{tag}_summary.csv")
    if not path.exists():
        return
    rows = list(csv.DictReader(open(path)))
    pls = ("rrt_connect", "angle_zone", "angle_zone_adaptive")
    lines = ["\\begin{tabular}{ccc|ccc|ccc|ccc}", "\\toprule",
             "$m$ & своб. & решаемо & \\multicolumn{3}{c|}{решено за бюджет} & "
             "\\multicolumn{3}{c|}{итераций (медиана)} & \\multicolumn{3}{c}{длина / кратчайший} \\\\",
             " & & & Р & У & А & Р & У & А & Р & У & А \\\\", "\\midrule"]
    for r in rows:
        cells = [fmt(r["scale"], 1), fmt(r["free_frac"]), fmt(r["solvable_frac"])]
        cells += [fmt(r[f"{pl}_solved"]) for pl in pls]
        cells += [fmt(r[f"{pl}_iters_median"], 0) for pl in pls]
        cells += [fmt(r[f"{pl}_length_ratio_median"]) for pl in pls]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / f"bridge{tag}.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    coverage_table()
    for tag in ("", "_pb"):
        missions_table(tag)
        bridge_table(tag)
    print("tables:", sorted(p.name for p in OUT.glob("*.tex")))
