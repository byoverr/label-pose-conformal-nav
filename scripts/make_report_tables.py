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
        for lv in sorted({r["level"] for r in rows}, key=lambda lv: (lv.startswith("SLAM"), lv)):
            name = (f"одометрия, каждый {lv[2:]}-й кадр" if lv.startswith("VO") else
                    f"SLAM с замыканиями, каждый {lv[4:]}-й кадр")
            specs.append((name, [r for r in rows if r["level"] == lv]))
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
    tags = [(tag, name) for tag, name in (("", "YOLO-World v2-s"), ("_x", "YOLO-World v2-x"), ("_yoloe", "YOLOE-v8-S"),
                                         ("_ens_max", "ансамбль, максимум"), ("_ens_mean", "ансамбль, среднее"),
                                         ("_ens_vote2", "ансамбль, второе"), ("_cascade", "каскад с CLIP"))
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

def crc_table():
    """Guarantees of different strength on the same tasks: map-level joint margin, per-path risk, planner-level
    conformal risk control (plan and execution). Success and unsafe share of all task realizations."""
    if not Path("results/tables/crc_risk.csv").exists():
        return
    crc = read_csv("results/tables/crc_risk.csv")
    exe = read_csv("results/tables/crc_exec_execution.csv") if Path("results/tables/crc_exec_execution.csv").exists() else []
    ens = read_csv("results/tables/crc_risk_ens.csv") if Path("results/tables/crc_risk_ens.csv").exists() else []
    mis = {k: read_csv(f"results/tables/missions_{k}.csv") for k in ("rand", "pb")}
    levels = ("L0", "L2", "L3")
    get = lambda rows, **kw: next((r for r in rows if all(str(r[k]) == str(v) for k, v in kw.items())), None)
    specs = [("без калибровки", "---", lambda kind, lv: get(mis[kind], level=lv, arm="uncalibrated"), "violation"),
             ("совместный запас", "$\\alpha = 0{,}1$ на класс", lambda kind, lv: get(mis[kind], level=lv, arm="joint"), "violation"),
             ("риск пути", "$a = 0{,}2$", lambda kind, lv: get(crc, tasks=kind, level=lv, target="0.2", rule="path"), "unsafe"),
             ("риск планировщика", "$a = 0{,}05$", lambda kind, lv: get(crc, tasks=kind, level=lv, target="0.05", rule="crc"), "unsafe"),
             ("риск планировщика", "$a = 0{,}1$", lambda kind, lv: get(crc, tasks=kind, level=lv, target="0.1", rule="crc"), "unsafe"),
             ("то же, по исполнению", "$a = 0{,}1$", lambda kind, lv: get(exe, tasks=kind, level=lv, target="0.1", rule="crc"), "unsafe"),
             ("риск планировщика, ансамбль", "$a = 0{,}05$", lambda kind, lv: get(ens, tasks=kind, level=lv, target="0.05", rule="crc"), "unsafe"),
             ("риск планировщика, ансамбль", "$a = 0{,}1$", lambda kind, lv: get(ens, tasks=kind, level=lv, target="0.1", rule="crc"), "unsafe")]
    lines = ["\\begin{tabular}{ll|ccc|c|ccc|c}", "\\toprule",
             "Правило & Уровень & \\multicolumn{4}{c|}{случайные задачи} & \\multicolumn{4}{c}{трудные задачи} \\\\",
             " & & L0 & L2 & L3 & опасн. & L0 & L2 & L3 & опасн. \\\\", "\\midrule"]
    for name, level, src, unsafe_key in specs:
        cells = []
        for kind in ("rand", "pb"):
            rows = [src(kind, lv) for lv in levels]
            cells += [fmt(r["success"]) if r else "---" for r in rows]
            u = [float(r[unsafe_key]) for r in rows if r]
            cells.append(fmt(max(u), 3) if u else "---")
        lines.append(f"{name} & {level} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "crc.tex").write_text("\n".join(lines) + "\n")


def execution_table():
    """Plan-certified paths executed under continued drift (hard tasks), plan rule vs execution-aware rule, a = 0.2."""
    path = Path("results/tables/execution.csv")
    if not path.exists():
        return
    rows = read_csv(path)
    lines = ["\\begin{tabular}{l|cc|ccc|ccc}", "\\toprule",
             "Дрейф & \\multicolumn{2}{c|}{отклонение, \\% пути} & \\multicolumn{3}{c|}{сертификат по плану} & "
             "\\multicolumn{3}{c}{с запасом на исполнение} \\\\",
             " & медиана & 95\\,\\% & путь & опасн. & столкн. & путь & опасн. & столкн. \\\\", "\\midrule"]
    import numpy as np
    for kind, title in (("pb", "трудные"), ("rand", "случайные")):
        for lv in ("L2", "L3", "L4"):
            g = lambda rule: next((r for r in rows if r["tasks"] == kind and r["level"] == lv and r["rule"] == rule
                                   and r["target_risk"] == "0.2"), None)
            p, e = g("plan"), g("execution")
            if p is None:
                continue
            cells = [fmt(100 * float(p["u_median"]), 1), fmt(100 * float(p["u_q95"]), 1)]
            for r in (p, e):
                cells += [fmt(r["certified"]), fmt(r["exec_unsafe_given_cert"], 3), fmt(r["exec_collision_given_cert"])]
            lines.append(f"{lv}, {title} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "execution.tex").write_text("\n".join(lines) + "\n")


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
    crc_table()
    execution_table()
    missions_table("_pb")
    PAPER.mkdir(parents=True, exist_ok=True)
    paper_coverage_table()
    paper_missions_table()
    paper_comparison_table()
    print("tables:", sorted(p.name for p in OUT.glob("*.tex")), sorted(p.name for p in PAPER.glob("*.tex")))
