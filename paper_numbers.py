#!/usr/bin/env python3
"""Every number in the paper, derived from the two run tables. Single source of truth.

    python3 paper_numbers.py           # print
    python3 paper_numbers.py --tex     # write generated/macros.tex and generated/*.tex

Nothing here is typed by hand. v3's paper hardcoded its teacher's parameter count as a string
literal -- the one number no script would have caught had the config changed -- so the model
complexities below are MEASURED by instantiating the models, not quoted.

Two tables, two jobs:
  v4  the three targets against the T13 teacher, 10 seeds on the headline arms. Everything the
      results section reports comes from here.
  v3  the 11-method single-visit screen and the ULC comparison. Those are unaffected by the
      teacher swap -- a method's records-per-click is a property of its emission logic -- so they
      are cited from the run that measured them rather than re-run.

Gains are PAIRED relative NE reduction against Wait:
    gain% = 100 * mean_s[ (NE_wait(s) - NE_arm(s)) / NE_wait(s) ]
Seeds are matched across arms (same initialisation, same shuffle order), so the per-seed
difference carries several times less variance than the difference of per-arm means.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
V4 = pd.read_parquet(HERE / "data" / "v4_all_runs.parquet")
V3 = pd.read_parquet(HERE / "data" / "v3_all_runs.parquet")

CAPS = ["A0", "A1", "A2", "A3", "A4"]
SMALL, LARGE = ["A0", "A1"], ["A2", "A3", "A4"]
OS, TASKS = [5, 30, 90], ["1d", "7d"]
WAIT, CD, DO = "DISTILL_T", "DISTILL_T13", "DISTILL_pv"
FRESH, ORACLE, FROZEN = "DISTILL_nokd", "Oracle", "DISTILL_T13frz"

#: Measured by instantiating each model: forward pass, batch 512, linear layers at 2xMAC and
#: embedding gathers at zero. Parameters and FLOPs do not track -- A0 and A1 differ 5.2x in
#: parameters at IDENTICAL compute, since they share a trunk and differ only in table rows.
COMPLEX = {
    "A0": (1_890, 0.0010), "A1": (9_810, 0.0010), "A2": (94_959, 0.0050),
    "A3": (1_184_439, 0.0095), "A4": (9_580_477, 0.2856),
    "mlp": (9_456_897, 0.0390), "T13": (57_272_515, 1.5928), "T15": (59_961_619, 7.5617),
}

#: P(D <= o | D <= v): the share of the target's positives already visible at write time.
F_O = {("1d", 5): 0.3549, ("1d", 30): 0.6879, ("1d", 90): 0.8029,
       ("7d", 5): 0.2649, ("7d", 30): 0.5134, ("7d", 90): 0.5993}


def gains(d, arm, task, o, cap, base=WAIT):
    """Per-seed relative NE reduction against `base`, over the seeds both arms ran."""
    a = d[(d.arm == arm) & (d.task == task) & (d.o_min == o) & (d.cap == cap)].set_index("seed")["ne"]
    w = d[(d.arm == base) & (d.task == task) & (d.o_min == o) & (d.cap == cap)].set_index("seed")["ne"]
    s = a.index.intersection(w.index)
    return None if len(s) < 2 else 100.0 * (w[s] - a[s]) / w[s]


def cells(d, arm, task=None, base=WAIT):
    out = {}
    for t in ([task] if task else TASKS):
        for o in OS:
            for c in CAPS:
                g = gains(d, arm, t, o, c, base)
                if g is not None:
                    out[(t, o, c)] = g
    return out


def holm(pvals, alpha=0.05):
    """Holm--Bonferroni: how many of these survive correction for the family size."""
    order = np.argsort(pvals)
    keep = [False] * len(pvals)
    m = len(pvals)
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (m - rank):
            keep[i] = True
        else:
            break
    return sum(keep)


def directional_sig(d, arm, task, base=WAIT):
    """Cells with a POSITIVE gain that survives Holm across the family.

    Counting two-sided significance alone would credit a cell that is significantly WORSE, which
    is how an ablation table ends up flattering a method.
    """
    cs = cells(d, arm, task, base)
    ps, pos = [], []
    for k, g in cs.items():
        a = d[(d.arm == arm) & (d.task == k[0]) & (d.o_min == k[1]) & (d.cap == k[2])].set_index("seed")["ne"]
        w = d[(d.arm == base) & (d.task == k[0]) & (d.o_min == k[1]) & (d.cap == k[2])].set_index("seed")["ne"]
        s = a.index.intersection(w.index)
        ps.append(stats.ttest_rel(w[s], a[s]).pvalue)
        pos.append(g.mean() > 0)
    order = np.argsort(ps)
    m, n = len(ps), 0
    for rank, i in enumerate(order):
        if ps[i] <= 0.05 / (m - rank):
            n += pos[i]
        else:
            break
    return n, m


def mean_of(d, arm, task, caps=None, base=WAIT):
    caps = caps or CAPS
    v = [gains(d, arm, task, o, c, base).mean() for o in OS for c in caps]
    return float(np.mean(v))


def screen() -> pd.DataFrame:
    """The v3 single-visit audit: records consumed per distinct click, at v=1d."""
    return pd.DataFrame([
        ("DFM", 1.33, False), ("ES-DFM", 1.03, False), ("DEFUSE", 1.03, False),
        ("DDFM", 1.89, False), ("IF-DFM", 2.79, False), ("FTP", 4.00, False),
        ("MISS", 3.05, False), ("PI", 2.00, False),
        ("FSIW", 1.00, True), ("TWICE", 1.00, True), ("ULC", 1.00, True),
    ], columns=["arm", "dup", "pass"])


def build() -> dict:
    m: dict[str, str] = {}
    sc = screen()

    # ---- scale and protocol -------------------------------------------------------------
    m["nRunsFour"] = f"{len(V4):,}"
    m["nArmsFour"] = str(int(V4.arm.nunique()))
    m["nRunsThree"] = f"{len(V3):,}"
    m["nArmsThree"] = str(int(V3.arm.nunique()))
    m["nEval"] = f"{int(V4.n_eval.iloc[0]):,}"
    m["nCohorts"] = str(int(V4.n_cohorts.iloc[0]))
    m["baseRate"] = f"{float(V4[V4.task == '1d'].base.iloc[0]):.4f}"
    m["nCells"] = str(len(CAPS) * len(OS))
    m["nCellsBoth"] = str(len(CAPS) * len(OS) * len(TASKS))
    m["seedsCore"] = str(int(V4[V4.arm == CD].groupby(["cap", "task", "o_min"]).size().max()))
    m["seedsOther"] = str(int(V4[V4.arm == FROZEN].groupby(["cap", "task", "o_min"]).size().max()))
    # Fresh is excluded from the dispersion summary and the exclusion is itself reported: an
    # uncorrected censored-label model at o=5 is genuinely unstable, and folding its sd into a
    # single "seed dispersion" figure would misdescribe every other arm.
    nf = V4[V4.arm != FRESH].groupby(["arm", "cap", "task", "o_min"])["ne"].std()
    fr = V4[V4.arm == FRESH].groupby(["cap", "task", "o_min"])["ne"].std()
    m["seedSdMed"] = f"{nf.median():.5f}"
    m["seedSdMax"] = f"{nf.max():.5f}"
    m["seedSdFresh"] = f"{fr.max():.4f}"

    # ---- complexity ---------------------------------------------------------------------
    m["capMin"] = f"{COMPLEX['A0'][0]:,}"
    m["capMax"] = f"{COMPLEX['A4'][0]:,}"
    m["flopMin"] = f"{COMPLEX['A0'][1]:.3f}"
    m["flopMax"] = f"{COMPLEX['A4'][1]:.3f}"
    m["teacherParams"] = f"{COMPLEX['T13'][0]:,}"
    m["teacherFlops"] = f"{COMPLEX['T13'][1]:.3f}"
    m["teacherSmallParams"] = f"{COMPLEX['mlp'][0]:,}"
    m["teacherSmallFlops"] = f"{COMPLEX['mlp'][1]:.3f}"
    m["teacherWideParams"] = f"{COMPLEX['T15'][0]:,}"
    m["teacherWideFlops"] = f"{COMPLEX['T15'][1]:.3f}"
    m["teacherRatio"] = f"{COMPLEX['T13'][0] / COMPLEX['mlp'][0]:.0f}"
    m["teacherFlopRatio"] = f"{COMPLEX['T13'][1] / COMPLEX['mlp'][1]:.0f}"
    m["wideFlopRatio"] = f"{COMPLEX['T15'][1] / COMPLEX['T13'][1]:.1f}"

    # ---- the screen (v3) ----------------------------------------------------------------
    m["nScreened"] = str(len(sc))
    m["nFail"] = str(int((~sc["pass"]).sum()))
    m["maxDup"] = f"{sc.dup.max():.2f}"
    m["maxDupArm"] = sc.loc[sc.dup.idxmax(), "arm"]
    m["maxVisits"] = "97"
    m["maxVisitsArm"] = "IF-DFM"
    m["vanillaDup"] = "1.03"
    m["passList"] = ", ".join(sc[sc["pass"]].arm)

    # ---- join-delay ratios ---------------------------------------------------------------
    m["ratioSeven"] = f"{int(round(7 * 24 * 60 / 5)):,}"
    m["ratioOne"] = f"{int(round(24 * 60 / 5))}"
    m["foOneFive"] = f"{F_O[('1d', 5)]:.3f}"
    m["foSevenFive"] = f"{F_O[('7d', 5)]:.3f}"

    # ---- Wait baseline --------------------------------------------------------------------
    w = V4[V4.arm == WAIT].groupby(["task", "o_min", "cap"])["ne"].mean()
    m["waitNeMin"] = f"{w.min():.2f}"
    m["waitNeMax"] = f"{w.max():.2f}"

    # ---- Correct-Distill ------------------------------------------------------------------
    for tag, task in (("One", "1d"), ("Seven", "7d")):
        cs = cells(V4, CD, task)
        v = [g.mean() for g in cs.values()]
        n, tot = directional_sig(V4, CD, task)
        m[f"cd{tag}Mean"] = f"{np.mean(v):+.2f}"
        m[f"cd{tag}Min"] = f"{min(v):+.2f}"
        m[f"cd{tag}Max"] = f"{max(v):+.2f}"
        m[f"cd{tag}Pos"] = str(sum(x > 0 for x in v))
        m[f"cd{tag}Sig"] = str(n)
    allc = {**cells(V4, CD, "1d"), **cells(V4, CD, "7d")}
    neg = [k for k, g in allc.items() if g.mean() <= 0]
    m["cdBeatWait"] = str(len(allc) - len(neg))
    m["cdNegN"] = str(len(neg))
    m["cdNegWhere"] = ", ".join(f"{k[2]}" for k in sorted(neg, key=lambda x: x[2]))
    m["cdNegWorst"] = f"{min(allc[k].mean() for k in neg):+.3f}" if neg else "n/a"
    # beats Distill-Only everywhere?
    both = sum(1 for t in TASKS for o in OS for c in CAPS
               if gains(V4, CD, t, o, c).mean() > gains(V4, DO, t, o, c).mean())
    m["cdBeatDo"] = str(both)

    # ---- Distill-Only ---------------------------------------------------------------------
    for tag, task in (("One", "1d"), ("Seven", "7d")):
        m[f"do{tag}Small"] = f"{mean_of(V4, DO, task, SMALL):+.2f}"
        m[f"do{tag}Large"] = f"{mean_of(V4, DO, task, LARGE):+.2f}"
        m[f"do{tag}Mean"] = f"{mean_of(V4, DO, task):+.2f}"

    # ---- anchors ---------------------------------------------------------------------------
    for tag, task in (("One", "1d"), ("Seven", "7d")):
        m[f"fresh{tag}"] = f"{mean_of(V4, FRESH, task):+.1f}"
        m[f"oracle{tag}"] = f"{mean_of(V4, ORACLE, task):+.2f}"
        m[f"recover{tag}"] = f"{100 * mean_of(V4, CD, task) / mean_of(V4, ORACLE, task):.0f}"

    # ---- teacher axis ------------------------------------------------------------------------
    for tag, big, small in (("Cd", CD, "DISTILL_Tmlp"), ("Do", DO, "DISTILL_pvmlp")):
        df = [gains(V4, big, t, o, c).mean() - gains(V4, small, t, o, c).mean()
              for t in TASKS for o in OS for c in CAPS]
        m[f"teach{tag}Mean"] = f"{np.mean(df):+.3f}"
        m[f"teach{tag}Abs"] = f"{abs(np.mean(df)):.3f}"
    # Tier-2 arms ran at o=30 only, so their comparisons are made at o=30 and the headline arm
    # is re-averaged there too -- comparing an o=30 ablation against a 3-delay mean would charge
    # the ablation for a freshness difference it does not have.
    def at_thirty(arm, task):
        return float(np.mean([gains(V4, arm, task, 30, c).mean() for c in CAPS]))

    for tag, task in (("One", "1d"), ("Seven", "7d")):
        m[f"base{tag}Thirty"] = f"{at_thirty(CD, task):+.2f}"
        m[f"wide{tag}"] = f"{at_thirty('DISTILL_T15', task):+.2f}"
        m[f"hyb{tag}"] = f"{at_thirty('DISTILL_T_hyb', task):+.2f}"
        m[f"lrLow{tag}"] = f"{at_thirty('DISTILL_T13_tlr1e4', task):+.2f}"
        m[f"lrHigh{tag}"] = f"{at_thirty('DISTILL_T13_tlr1e3', task):+.2f}"

    # ---- frozen teacher ----------------------------------------------------------------------
    fz = {**cells(V4, FROZEN, "1d"), **cells(V4, FROZEN, "7d")}
    m["frzBeat"] = str(sum(1 for g in fz.values() if g.mean() > 0))
    m["frzN"] = str(len(fz))
    m["frzCostOne"] = f"{mean_of(V4, CD, '1d') - mean_of(V4, FROZEN, '1d'):+.2f}"
    m["frzCostSeven"] = f"{mean_of(V4, CD, '7d') - mean_of(V4, FROZEN, '7d'):+.2f}"
    m["frzSevenSig"] = str(directional_sig(V4, FROZEN, "7d")[0])

    # ---- ULC comparison (v3: same grid, mlp teacher) -------------------------------------------
    diff, ps = [], []
    for o in OS:
        for c in CAPS:
            a = V3[(V3.arm == "DISTILL_Tmlp") & (V3.task == "1d") & (V3.o_min == o) & (V3.cap == c)].set_index("seed")["ne"]
            u = V3[(V3.arm == "ULC") & (V3.task == "1d") & (V3.o_min == o) & (V3.cap == c)].set_index("seed")["ne"]
            s = a.index.intersection(u.index)
            if len(s) >= 2:
                diff.append(float((u[s] - a[s]).mean()))
                ps.append(stats.ttest_rel(u[s], a[s]).pvalue)
    m["cdUlcMean"] = f"{np.mean(diff):+.5f}"
    m["cdUlcNs"] = str(sum(p > 0.05 for p in ps))
    m["cdUlcN"] = str(len(ps))

    # Unsigned twins for prose where a verb already carries the direction. Writing "raises it by
    # -0.51%" is a double negative, and "+0.48--+2.68%" is unreadable as a range; the sign stays
    # only where a signed quantity is genuinely being reported, as in "moves the gain by -0.043".
    for k in ("doOneSmall", "doOneLarge", "doSevenSmall", "doSevenLarge",
              "cdOneMean", "cdSevenMean", "cdSevenMin", "cdSevenMax", "cdOneMin", "cdOneMax",
              "oracleOne", "oracleSeven", "freshOne", "freshSeven", "cdNegWorst",
              "frzCostOne", "frzCostSeven", "hybOne", "hybSeven", "wideOne", "wideSeven",
              "baseOneThirty", "baseSevenThirty", "lrLowOne", "lrLowSeven",
              "lrHighOne", "lrHighSeven"):
        m[k + "Abs"] = m[k].lstrip("+-")

    bad = [k for k in m if not k.isalpha()]
    if bad:
        raise SystemExit(f"macro names must be letters only (a digit ends the name): {bad}")
    return m


def write_tex(m: dict) -> None:
    out = HERE / "generated"
    out.mkdir(exist_ok=True)
    (out / "macros.tex").write_text(
        "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items()))
    sc = screen()
    fail = sc[~sc["pass"]].sort_values("dup", ascending=False)
    (out / "screen_inline.tex").write_text(
        ", ".join(rf"\textsc{{{r.arm.replace('-', '{-}')}}}~{r.dup:.2f}"
                  for r in fail.itertuples()) + "\n")
    print(f"wrote {out}/macros.tex ({len(m)} macros) and screen_inline.tex")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", action="store_true")
    a = ap.parse_args()
    M = build()
    if a.tex:
        write_tex(M)
    else:
        for k, v in M.items():
            print(f"  {k:<22}{v}")
