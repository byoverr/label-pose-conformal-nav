"""LaTeX tables for the report and the paper, generated from the result CSVs (no hand-typed numbers).

Writes report/tables/*.tex (Russian, decimal comma) and paper/tables/*.tex (English).
"""

from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from s2m.io import read_csv

OUT = Path("report/tables")
ARM_RU = {
    "uncalibrated": "без калибровки",
    "label_cell": "множества меток",
    "geometry": "только геометрия",
    "label_only": "только метки",
    "pose_only": "только поза",
    "separate": "сумма",
    "joint": "\\textbf{совместный}",
    "oracle": "оракул",
}
SHORT_RU = {"label_only": "метки", "separate": "сумма", "joint": "\\textbf{совм.}"}
CLASS_RU = {"indoor_plant": "растение", "tv_stand": "тумба под ТВ", "bike": "велосипед"}
RADIUS_ARMS = ("geometry", "label_only", "pose_only", "separate", "joint")
DAGGER = "$^\\dagger$"


def fmt(v, digits=2, comma=True):
    """Half-up rounding (0.945 -> 0.95, as in the text), decimal comma unless comma=False."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "---"
    if x != x:
        return "---"
    q = Decimal(repr(x)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    return f"{q:.{digits}f}".replace(".", "{,}" if comma else ".")


def radius_cell(r) -> str:
    """Median radius; dagger if the arm abstains in at least half of the splits (median of the rest)."""
    if r["arm"] not in RADIUS_ARMS:
        return "0"
    return fmt(r["radius_median"]) + (DAGGER if float(r["abstain_rate"]) >= 0.5 else "")


def coverage_table():
    rows = read_csv("results/tables/coverage.csv")
    levels = ("L0", "L2", "L3", "L4")
    arms = ("uncalibrated", "label_cell", "geometry", "label_only", "pose_only", "separate", "joint")
    lines = ["\\begin{tabular}{ll" + "cc" * len(levels) + "}", "\\toprule",
             "Класс & Калибровка & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{lv}}}" for lv in levels) + " \\\\",
             " & & " + " & ".join(["покр. & $\\hat r$, м"] * len(levels)) + " \\\\", "\\midrule"]
    for cls in ("bike", "indoor_plant"):
        for i, arm in enumerate(arms):
            cells = []
            for lv in levels:
                r = next(x for x in rows if x["level"] == lv and x["class"] == cls and x["arm"] == arm)
                cells += [fmt(r["coverage_mean"]), radius_cell(r)]
            name = CLASS_RU[cls] if i == 0 else ""
            lines.append(f"{name} & {ARM_RU[arm]} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule" if cls != "indoor_plant" else "\\bottomrule")
    lines.append("\\end{tabular}")
    (OUT / "coverage.tex").write_text("\n".join(lines) + "\n")



def missions_table(tag="_pb", levels=("L0", "L2", "L3")):
    path = Path(f"results/tables/missions{tag}.csv")
    if not path.exists():
        return
    rows = read_csv(path)
    arms = ("uncalibrated", "label_cell", "geometry", "label_only", "pose_only", "separate", "joint", "oracle")
    lines = ["\\begin{tabular}{l" + "ccc" * len(levels) + "}", "\\toprule",
             "Калибровка & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{lv}}}" for lv in levels) + " \\\\",
             " & " + " & ".join(["путь & опасн. & успех"] * len(levels)) + " \\\\", "\\midrule"]
    for arm in arms:
        cells = []
        for lv in levels:
            r = next(x for x in rows if x["level"] == lv and x["arm"] == arm)
            cells += [fmt(r["planned"]), f"{int(float(r['n_violation']))}/{int(float(r['n_planned']))}", fmt(r["success"])]
        lines.append(f"{ARM_RU[arm]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / f"missions{tag}.tex").write_text("\n".join(lines) + "\n")



def alpha_table():
    path = Path("results/tables/alpha_sweep.csv")
    if not path.exists():
        return
    rows = [r for r in read_csv(path)]
    get = lambda kind, a, arm, key: next((r[key] for r in rows if r["tasks"] == kind and r["alpha"] == a
                                          and r["arm"] == arm), "nan")
    lines = ["\\begin{tabular}{c|cc|cc|c|cc|cc}", "\\toprule",
             "$\\alpha$ & \\multicolumn{2}{c|}{покрытие} & \\multicolumn{2}{c|}{$\\hat r$, м} & отказ & "
             "\\multicolumn{2}{c|}{трудные: путь} & \\multicolumn{2}{c}{случайные: успех} \\\\",
             " & раст. & велос. & раст. & велос. & раст. & совм. & оракул & совм. & оракул \\\\", "\\midrule"]
    for a in sorted({r["alpha"] for r in rows}, key=float):
        cells = [fmt(a, 2), fmt(get("pass_by", a, "joint", "coverage_mean_indoor_plant")),
                 fmt(get("pass_by", a, "joint", "coverage_mean_bike")),
                 fmt(get("pass_by", a, "joint", "radius_median_indoor_plant")),
                 fmt(get("pass_by", a, "joint", "radius_median_bike")),
                 fmt(get("pass_by", a, "joint", "abstain_rate_indoor_plant")),
                 fmt(get("pass_by", a, "joint", "planned")), fmt(get("pass_by", a, "oracle", "planned")),
                 fmt(get("random", a, "joint", "success")), fmt(get("random", a, "oracle", "success"))]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "alpha.tex").write_text("\n".join(lines) + "\n")


def variants_table():
    """One row per (variant, level), joint / separate / labels-only: coverage and radius, bike and plant."""
    arms = ("label_only", "separate", "joint")
    specs = []
    sam = Path("results/tables/variants_sam.csv")
    if sam.exists():
        rows = read_csv(sam)
        for m, name in (("box", "рамка + глубина"), ("sam", "MobileSAM")):
            for lv in ("L0", "L4"):
                specs.append((f"{name}, {lv}", [r for r in rows if r["masks"] == m and r["level"] == lv]))
    vo = Path("results/tables/variants_vo.csv")
    if vo.exists():
        rows = read_csv(vo)
        for lv in sorted({r["level"] for r in rows}):
            specs.append((f"одометрия, каждый {lv[2:]}-й кадр", [r for r in rows if r["level"] == lv]))
    if not specs:
        return
    lines = ["\\begin{tabular}{l|" + "cc" * len(arms) + "|" + "cc" * len(arms) + "}", "\\toprule",
             "Вариант & \\multicolumn{6}{c|}{велосипед} & \\multicolumn{6}{c}{растение} \\\\",
             " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{SHORT_RU[a]}}}" for a in arms * 2) + " \\\\",
             " & " + " & ".join(["покр. & $\\hat r$"] * len(arms) * 2) + " \\\\", "\\midrule"]
    for name, rs in specs:
        cells = []
        for cls in ("bike", "indoor_plant"):
            for arm in arms:
                r = next((x for x in rs if x["class"] == cls and x["arm"] == arm), None)
                cells += ["---", "---"] if r is None else [fmt(r["coverage_mean"]), radius_cell(r)]
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "variants.tex").write_text("\n".join(lines) + "\n")


def comparison_table():
    """2 x 2 design: vocabulary x risk level, random tasks; mission success per arm."""
    path = Path("results/tables/comparison.csv")
    if not path.exists():
        return
    rows = read_csv(path)
    arms = ("label_cell", "geometry", "joint", "uncalibrated", "oracle")
    lines = ["\\begin{tabular}{llc|ccccc}", "\\toprule",
             "Словарь & $1 - \\alpha$ & Дрейф & " + " & ".join(ARM_RU[a] for a in arms) + " \\\\", "\\midrule"]
    settings = [("closed", "0.05"), ("closed", "0.1"), ("open", "0.05"), ("open", "0.1")]
    for s, (vocab, alpha) in enumerate(settings):
        for i, lv in enumerate(("L0", "L2", "L3")):
            g = lambda arm, key: next(r[key] for r in rows if r["vocabulary"] == vocab and r["alpha"] == alpha
                                      and r["level"] == lv and r["arm"] == arm)
            cells = []
            for arm in arms:
                c = fmt(g(arm, "success"))
                if arm != "oracle" and float(g(arm, "violation")) > 0:
                    c += "$^\\ast$"
                cells.append(c)
            head = ["закрытый" if vocab == "closed" else "открытый", fmt(1 - float(alpha))] if i == 0 else ["", ""]
            lines.append(" & ".join(head + ["известна" if lv == "L0" else lv] + cells) + " \\\\")
        lines.append("\\midrule" if s < len(settings) - 1 else "\\bottomrule")
    lines.append("\\end{tabular}")
    (OUT / "comparison.tex").write_text("\n".join(lines) + "\n")


def detector_table():
    """Detectors (YOLO-World v2-s main, v2-x, YOLOE-v8-S): joint margin and abstention per class, missions."""
    tags = [(tag, name) for tag, name in (("", "YOLO-World v2-s"), ("_x", "YOLO-World v2-x"), ("_yoloe", "YOLOE-v8-S"))
            if Path(f"results/tables/coverage{tag}.csv").exists()]
    if len(tags) < 2:
        return
    g = lambda rows, **kw: next(r for r in rows if all(r[k] == v for k, v in kw.items()))
    lines = ["\\begin{tabular}{l|cc|cc|cc|cc}", "\\toprule",
             "Детектор & \\multicolumn{2}{c|}{$\\hat r$ / отказ, L0} & \\multicolumn{2}{c|}{$\\hat r$ / отказ, L4} & "
             "\\multicolumn{2}{c|}{трудные: путь} & \\multicolumn{2}{c}{случайные: успех} \\\\",
             " & растение & велосипед & растение & велосипед & L0 & L3 & L0 & L3 \\\\", "\\midrule"]
    for tag, name in tags:
        c = read_csv(f"results/tables/coverage{tag}.csv")
        cell = lambda lv, cls: (f"{fmt(g(c, level=lv, arm='joint', **{'class': cls})['radius_median'])} / "
                                f"{fmt(g(c, level=lv, arm='joint', **{'class': cls})['abstain_rate'])}")
        cells = [cell(lv, cls) for lv in ("L0", "L4") for cls in ("indoor_plant", "bike")]
        for kind, key in (("pb", "planned"), ("rand", "success")):
            path = Path(f"results/tables/missions_{kind}{tag}.csv")
            m = read_csv(path) if path.exists() else None
            cells += [fmt(g(m, level=lv, arm="joint")[key]) if m else "---" for lv in ("L0", "L3")]
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "detector.tex").write_text("\n".join(lines) + "\n")

PAPER = Path("paper/tables")
ARM_EN = {"uncalibrated": "uncalibrated", "label_cell": "label sets~\\cite{sundarsingh}", "geometry": "geometry only",
          "label_only": "labels only", "pose_only": "pose only", "separate": "sum", "joint": "\\textbf{joint (ours)}",
          "oracle": "oracle"}


def paper_coverage_table():
    rows = read_csv("results/tables/coverage.csv")
    f = lambda v: fmt(v, comma=False)
    lines = ["\\begin{tabular}{l|ccc|c|ccc|c}", "\\toprule",
             " & \\multicolumn{4}{c|}{bike} & \\multicolumn{4}{c}{indoor plant} \\\\",
             "calibration & L0 & L2 & L4 & $\\hat r$ L4 & L0 & L2 & L4 & $\\hat r$ L4 \\\\", "\\midrule"]
    for arm in ("uncalibrated", "label_cell", "geometry", "label_only", "pose_only", "separate", "joint"):
        cells = []
        for cls in ("bike", "indoor_plant"):
            g = lambda lv: next(x for x in rows if x["level"] == lv and x["class"] == cls and x["arm"] == arm)
            cells += [f(g(lv)["coverage_mean"]) for lv in ("L0", "L2", "L4")]
            r = g("L4")
            cells.append("0" if arm not in RADIUS_ARMS else f(r["radius_median"]) + (DAGGER if float(r["abstain_rate"]) >= 0.5 else ""))
        lines.append(f"{ARM_EN[arm]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (PAPER / "coverage.tex").write_text("\n".join(lines) + "\n")


def paper_missions_table():
    """Hard and random tasks at L0 and L3: path found / unsafe among found / success."""
    lines = ["\\begin{tabular}{l|cc|cc|cc}", "\\toprule",
             " & \\multicolumn{2}{c|}{hard: path} & \\multicolumn{2}{c|}{hard: unsafe} & \\multicolumn{2}{c}{random: success} \\\\",
             "calibration & L0 & L3 & L0 & L3 & L0 & L3 \\\\", "\\midrule"]
    pb = read_csv("results/tables/missions_pb.csv")
    rnd = read_csv("results/tables/missions_rand.csv")
    f = lambda v: fmt(v, comma=False)
    for arm in ("uncalibrated", "label_cell", "geometry", "joint", "oracle"):
        g = lambda rows, lv: next(x for x in rows if x["level"] == lv and x["arm"] == arm)
        cells = [f(g(pb, lv)["planned"]) for lv in ("L0", "L3")]
        cells += [f(g(pb, lv)["violation_given_planned"]) for lv in ("L0", "L3")]
        cells += [f(g(rnd, lv)["success"]) for lv in ("L0", "L3")]
        lines.append(f"{ARM_EN[arm]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (PAPER / "missions.tex").write_text("\n".join(lines) + "\n")


def paper_comparison_table():
    path = Path("results/tables/comparison.csv")
    if not path.exists():
        return
    rows = read_csv(path)
    f = lambda v: fmt(v, comma=False)
    arms = ("label_cell", "geometry", "joint", "uncalibrated")
    lines = ["\\begin{tabular}{llc|cccc}", "\\toprule",
             "vocab. & $1-\\alpha$ & drift & label sets & geometry & \\textbf{joint} & uncal. \\\\", "\\midrule"]
    settings = [("closed", "0.05"), ("closed", "0.1"), ("open", "0.05"), ("open", "0.1")]
    for s, (vocab, alpha) in enumerate(settings):
        for i, lv in enumerate(("L0", "L2", "L3")):
            g = lambda arm, key: next(r[key] for r in rows if r["vocabulary"] == vocab and r["alpha"] == alpha
                                      and r["level"] == lv and r["arm"] == arm)
            cells = [f(g(a, "success")) + ("$^\\ast$" if float(g(a, "violation")) > 0 else "") for a in arms]
            head = [vocab, f(1 - float(alpha))] if i == 0 else ["", ""]
            lines.append(" & ".join(head + ["none" if lv == "L0" else lv] + cells) + " \\\\")
        lines.append("\\midrule" if s < len(settings) - 1 else "\\bottomrule")
    lines.append("\\end{tabular}")
    (PAPER / "comparison.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    coverage_table()
    alpha_table()
    comparison_table()
    variants_table()
    detector_table()
    missions_table("_pb")
    PAPER.mkdir(parents=True, exist_ok=True)
    paper_coverage_table()
    paper_missions_table()
    paper_comparison_table()
    print("tables:", sorted(p.name for p in OUT.glob("*.tex")), sorted(p.name for p in PAPER.glob("*.tex")))
