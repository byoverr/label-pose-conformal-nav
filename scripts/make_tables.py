"""LaTeX tables for the Russian article and the English paper, generated from the result CSVs (no hand-typed numbers).

Writes paper/tables_ru/*.tex (Russian, decimal comma) and paper/tables/*.tex (English).
"""

from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from s2m.io import read_csv

OUT = Path("paper/tables_ru")
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


def comparison_table():
    """2 x 2 design: vocabulary x risk level, random tasks; mission success per arm."""
    path = Path("results/tables/comparison.csv")
    if not path.exists():
        return
    rows = read_csv(path)
    arms = ("label_cell", "geometry", "joint", "uncalibrated", "oracle")
    lines = ["\\begin{tabular}{llc|ccccc}", "\\toprule",
             "Словарь & $1 - \\alpha$ & Поза & " + " & ".join(ARM_RU[a] for a in arms) + " \\\\", "\\midrule"]
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
            lines.append(" & ".join(head + ["точная" if lv == "L0" else lv] + cells) + " \\\\")
        lines.append("\\midrule" if s < len(settings) - 1 else "\\bottomrule")
    lines.append("\\end{tabular}")
    (OUT / "comparison.tex").write_text("\n".join(lines) + "\n")


def detector_table():
    """Detectors (YOLO-World v2-s main, v2-x, YOLOE-v8-S): joint margin and abstention per class, missions."""
    tags = [(tag, name) for tag, name in (("", "YOLO-World v2-s"), ("_x", "YOLO-World v2-x"), ("_yoloe", "YOLOE-v8-S"),
                                         ("_ens_max", "ансамбль, максимум"), ("_ens_mean", "ансамбль, среднее"),
                                         ("_ens_vote2", "ансамбль, согласие двух"), ("_cascade", "каскад с CLIP"))
            if Path(f"results/tables/coverage{tag}.csv").exists()]
    if len(tags) < 2:
        return
    g = lambda rows, **kw: next(r for r in rows if all(r[k] == v for k, v in kw.items()))
    lines = ["\\begin{tabular}{l|ccc|cc|cc|cc}", "\\toprule",
             "Детектор & \\multicolumn{3}{c|}{$\\hat r$ / отказ, L0} & \\multicolumn{2}{c|}{$\\hat r$ / отказ, L4} & "
             "\\multicolumn{2}{c|}{трудные: путь} & \\multicolumn{2}{c}{случайные: успех} \\\\",
             " & растение & велосипед & тумба & растение & велосипед & L0 & L3 & L0 & L3 \\\\", "\\midrule"]
    for tag, name in tags:
        c = read_csv(f"results/tables/coverage{tag}.csv")
        cell = lambda lv, cls: (f"{fmt(g(c, level=lv, arm='joint', **{'class': cls})['radius_median'])} / "
                                f"{fmt(g(c, level=lv, arm='joint', **{'class': cls})['abstain_rate'])}")
        cells = [cell("L0", cls) for cls in ("indoor_plant", "bike", "tv_stand")]
        cells += [cell("L4", cls) for cls in ("indoor_plant", "bike")]
        for kind, key in (("pb", "planned"), ("rand", "success")):
            path = Path(f"results/tables/missions_{kind}{tag}.csv")
            m = read_csv(path) if path.exists() else None
            cells += [fmt(g(m, level=lv, arm="joint")[key]) if m else "---" for lv in ("L0", "L3")]
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "detector.tex").write_text("\n".join(lines) + "\n")

def scene_unsafe(directory: str, arm: str, level: str) -> tuple[float, float]:
    """Mean over scenes and worst scene of the share of unsafe missions (violations among all task realizations)."""
    shares = []
    for f in sorted(Path(directory).glob("*.csv")):
        if f.stem.startswith("params_"):
            continue
        rows = [r for r in read_csv(f) if r["arm"] == arm and r["level"] == level]
        if rows:
            shares.append(sum(r["violation"] == "True" for r in rows) / len(rows))
    return (sum(shares) / len(shares), max(shares)) if shares else (float("nan"), float("nan"))


def crc_table():
    """Guarantees of different strength on the same tasks: map-level joint margin, per-path risk, planner-level
    conformal risk control (plan and execution). Success: share of task realizations with a path that is neither
    unsafe nor hits an obstacle. Unsafe: share of unsafe missions per scene, mean over scenes and worst scene, the
    largest over the levels shown (the mission-level guarantee bounds the mean)."""
    if not Path("results/tables/crc_risk.csv").exists():
        return
    src = {name: read_csv(f"results/tables/{name}.csv") for name in ("crc_risk", "crc_exec_execution", "crc_risk_ens")
           if Path(f"results/tables/{name}.csv").exists()}
    mis = {k: read_csv(f"results/tables/missions_{k}.csv") for k in ("rand", "pb")}
    levels = ("L0", "L2", "L3")
    get = lambda rows, **kw: next((r for r in rows if all(str(r[k]) == str(v) for k, v in kw.items())), None)

    def mission_rows(arm):
        return lambda kind, lv: get(mis[kind], level=lv, arm=arm), \
               lambda kind, lv: scene_unsafe(f"results/missions_{kind}", arm, lv)

    def crc_rows(name, target, rule):
        row = lambda kind, lv: get(src.get(name, []), tasks=kind, level=lv, target=target, rule=rule)
        return row, lambda kind, lv: ((float(row(kind, lv)["unsafe_scene_mean"]), float(row(kind, lv)["unsafe_scene_max"]))
                                      if row(kind, lv) else (float("nan"), float("nan")))

    specs = [("без калибровки", "---", *mission_rows("uncalibrated")),
             ("совместный запас", "$\\alpha = 0{,}1$ на класс", *mission_rows("joint")),
             ("риск пути", "$a = 0{,}2$", *crc_rows("crc_risk", "0.2", "path")),
             ("риск планировщика", "$a = 0{,}05$", *crc_rows("crc_risk", "0.05", "crc")),
             ("риск планировщика", "$a = 0{,}1$", *crc_rows("crc_risk", "0.1", "crc")),
             ("то же, по исполнению", "$a = 0{,}05$", *crc_rows("crc_exec_execution", "0.05", "crc")),
             ("то же, по исполнению", "$a = 0{,}1$", *crc_rows("crc_exec_execution", "0.1", "crc")),
             ("риск планировщика, ансамбль", "$a = 0{,}05$", *crc_rows("crc_risk_ens", "0.05", "crc")),
             ("риск планировщика, ансамбль", "$a = 0{,}1$", *crc_rows("crc_risk_ens", "0.1", "crc"))]
    lines = ["\\begin{tabular}{ll|ccc|c|ccc|c}", "\\toprule",
             "Правило & Уровень & \\multicolumn{4}{c|}{случайные задачи} & \\multicolumn{4}{c}{трудные задачи} \\\\",
             " & & L0 & L2 & L3 & опасн. & L0 & L2 & L3 & опасн. \\\\", "\\midrule"]
    for name, level, row, unsafe in specs:
        cells = []
        for kind in ("rand", "pb"):
            rows = [row(kind, lv) for lv in levels]
            cells += [fmt(r["success"]) if r else "---" for r in rows]
            u = [unsafe(kind, lv) for lv, r in zip(levels, rows) if r]
            cells.append(f"{fmt(max(x[0] for x in u), 3)} / {fmt(max(x[1] for x in u))}" if u else "---")
        lines.append(f"{name} & {level} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "crc.tex").write_text("\n".join(lines) + "\n")


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
    comparison_table()
    detector_table()
    crc_table()
    missions_table("_pb")
    PAPER.mkdir(parents=True, exist_ok=True)
    paper_coverage_table()
    paper_missions_table()
    paper_comparison_table()
    print("tables:", sorted(p.name for p in OUT.glob("*.tex")), sorted(p.name for p in PAPER.glob("*.tex")))
