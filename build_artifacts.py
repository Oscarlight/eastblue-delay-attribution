#!/usr/bin/env python3
"""Regenerate every table and data figure in the paper FROM the run CSVs.

Nothing in the manuscript is transcribed by hand: `main.tex` \\input{}s the files this writes, so a
number can only change by re-running an experiment. Run from Publication/ecir2027/:

    python build_artifacts.py

Springer LNCS compliance (Instructions for Authors, Sec. 4.5): greyscale only, distinct markers and
hatching so the figures survive black-and-white printing, vector PDF, drawn at the true 122 mm text
width so no label falls below 6 pt.
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

LNCS_W = 4.80                                   # 122 mm text width, in inches
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 6.6, "axes.titlesize": 8,
                     "pdf.fonttype": 42, "axes.linewidth": 0.6})

RUNS = pathlib.Path("../../Experiment/analysis/twice")
OUT_T, OUT_F, OUT_D = pathlib.Path("tables"), pathlib.Path("figs"), pathlib.Path("data")
for p in (OUT_T, OUT_F, OUT_D):
    p.mkdir(exist_ok=True)

LADDER = ["S0", "S1", "S2", "S3", "S4"]
#: categorical hash rows per rung and the resulting parameter count. The ladder varies table rows
#: at a fixed embedding dimension of 8, so it is a pure serving-capacity axis.
ROWS = {"S0": 36, "S1": 64, "S2": 160, "S3": 896, "S4": 131072}
PARAMS = {"S0": "26K", "S1": "46K", "S2": "115K", "S3": "645K", "S4": "9.6M"}

#: arm -> (paper name, family). Names, never letters, and never the registry key.
NAME = {
    "Oracle":        ("Oracle",       "ceiling"),
    "Vanilla_fresh": ("Fresh = Distil-NoKD", "compliant"),
    "DISTILL_T":     ("Wait",         "compliant"),
    "FSIW":          ("Reweight",     "compliant"),
    "ULC":           ("Correct",      "compliant"),
    "TWICE":         ("Twice",        "compliant"),
    "DISTILL_nokd":  ("Distil-NoKD",  "compliant"),
    "DISTILL_Tmlp":  ("Distil",       "ours"),
    "DISTILL":       ("Distil-T57M",  "ablation"),
    "Vanilla":       ("Vanilla",      "inadmissible"),
    "ES-DFM":        ("ES-DFM",       "inadmissible"),
    "DEFUSE":        ("DEFUSE",       "inadmissible"),
    "DDFM":          ("DDFM",         "inadmissible"),
}
COMPLIANT = ["Vanilla_fresh", "DISTILL_T", "FSIW", "TWICE", "ULC", "DISTILL_Tmlp"]
#: `Distil-NoKD` is the same algorithm as `Fresh` (one record, observed label) and produces
#: identical numbers, so the ladder table shows one merged row and says so in the caption.
MERGED_FLOOR = "Vanilla_fresh"
INADM = ["Vanilla", "ES-DFM", "DEFUSE", "DDFM"]

#: run files per capacity, in PRECEDENCE order -- earlier wins on a duplicate (method, cap, seed).
#: `kd5_`/`f5_`/`inadm_` are the 5-seed runs and supersede the 3-seed exploratory ones.
#: `cap7d_*_kd` is EXCLUDED entirely: it was run with the teacher's learning rate coupled to the
#: student's sweep, which tunes two models with one scalar, and is void.
PREC = ["pv{t}_{c}", "abl5_{c}", "fair_{c}", "auxlr_{c}", "kd5_{c}", "f5_{c}",
        "inadm_{c}", "kd2_{c}", "cap{t}_{c}_fast", "tsweep_S2", "wonly_S2"]


def load_capacity(task: str = "7d"):
    frames = []
    for cap in LADDER:
        for pat in PREC:
            if pat.endswith("_S2") and cap != "S2":
                continue
            f = RUNS / f"{pat.format(c=cap, t=task)}_runs.csv"
            if f.exists():
                d = pd.read_csv(f)
                d["cap"] = cap
                frames.append(d)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True).drop_duplicates(["method", "cap", "seed"],
                                                                keep="first")


def agg(d):
    return d.groupby(["method", "cap"]).agg(
        auc=("auc", "mean"), sd=("auc", "std"), pr=("pr_auc", "mean"),
        ll=("ll", "mean"), ne=("ne", "mean"), n=("auc", "size")).reset_index()


def paired(d, a, b, cap, metric="auc"):
    x = d[(d.method == b) & (d.cap == cap)].sort_values("seed")[metric].values
    y = d[(d.method == a) & (d.cap == cap)].sort_values("seed")[metric].values
    n = min(len(x), len(y))
    if n < 2:
        return float("nan"), float("nan")
    return y[:n].mean() - x[:n].mean(), float(stats.ttest_rel(y[:n], x[:n]).pvalue)


def num(x, sig=4, signed=False):
    """LaTeX-safe number: real minus signs, not hyphens."""
    if x != x:
        return "--"
    t = f"{x:+.{sig}f}" if signed else f"{x:.{sig}f}"
    return "$" + t.replace("-", "-") + "$" if signed else f"{x:.{sig}f}"


def stars(p):
    return "" if p != p else (r"$^{\ast\ast}$" if p < 0.01 else (r"$^{\ast}$" if p < 0.05 else ""))


def get(a, m, c, col="auc"):
    r = a[(a.method == m) & (a.cap == c)]
    return float(r.iloc[0][col]) if len(r) else float("nan")


# ======================================================================================
def tab_compliance():
    d = pd.read_csv(OUT_D / "compliance_audit.csv")
    # paper names throughout -- registry keys must never reach the manuscript
    pretty = {"Vanilla_fresh": "Fresh", "DISTILL_T": "Wait / teacher", "TWICE": "Twice",
              "FSIW": "Reweight", "ULC": "Correct", "DFM": "DFM", "Vanilla": "Vanilla"}
    order = [("admissible", ["Vanilla_fresh", "DISTILL_T", "FSIW", "ULC", "TWICE"]),
             ("inadmissible", ["Vanilla", "DFM", "ES-DFM", "DEFUSE", "DDFM",
                               "PI", "MISS", "IF-DFM", "FTP"])]
    L = [r"\begin{tabular}{l r@{\hskip 1.1em} r@{\hskip 1.4em} c@{\hskip 1.0em} c}",
         r"\toprule", r"method & records/click & max & R1 & R2\\"]
    for grp, arms in order:
        L.append(r"\midrule")
        L.append(r"\multicolumn{5}{l}{\emph{" + grp + r"}}\\")
        for arm in arms:
            r = d[d.arm == arm]
            if r.empty:
                continue
            r = r.iloc[0]
            ok1 = r.R1 in ("OK", "n/a frozen")
            c1 = r"\checkmark" if ok1 else r"$\times$"
            c2 = {"OK": r"\checkmark", "OK*": r"\checkmark$^{\dag}$"}.get(r.R2, r"$\times$")
            L.append(f"\\quad {pretty.get(arm, arm)} & {r.visits_per_click:.2f} & "
                     f"{r.max_visits} & {c1} & {c2}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT_T / "tab_compliance.tex").write_text("\n".join(L) + "\n")
    print("tables/tab_compliance.tex")


def tab_ladder(task="7d"):
    d = load_capacity(task)
    a = agg(d)
    caps = [c for c in LADDER if not a[a.cap == c].empty]
    L = [r"\begin{tabular}{l" + "r" * len(caps) + "}", r"\toprule",
         r"student & " + " & ".join(caps) + r"\\",
         r"parameters & " + " & ".join(PARAMS[c] for c in caps) + r"\\", r"\midrule",
         r"\multicolumn{%d}{l}{\emph{admissible: one record per click}}\\" % (len(caps) + 1)]
    for arm in COMPLIANT:
        v = [get(a, arm, c) for c in caps]
        if all(x != x for x in v):
            continue
        if NAME[arm][1] == "ours":
            row = []
            for c, x in zip(caps, v):
                _dd, p = paired(d, arm, "ULC", c)
                row.append(f"\\textbf{{{x:.4f}}}{stars(p)}" if x == x else "--")
            L.append(r"\quad \textbf{" + NAME[arm][0] + "} & " + " & ".join(row) + r"\\")
        else:
            L.append(r"\quad " + NAME[arm][0] + " & " +
                     " & ".join(f"{x:.4f}" if x == x else "--" for x in v) + r"\\")
    if any(not a[a.method == m].empty for m in INADM):
        L.append(r"\midrule")
        L.append(r"\multicolumn{%d}{l}{\emph{inadmissible: $>1$ record per click}}\\"
                 % (len(caps) + 1))
        for arm in INADM:
            v = [get(a, arm, c) for c in caps]
            if all(x != x for x in v):
                continue
            L.append(r"\quad " + NAME[arm][0] + " & " +
                     " & ".join(f"{x:.4f}" if x == x else "--" for x in v) + r"\\")
    L.append(r"\midrule")
    L.append(r"\quad Oracle \emph{(not causal)} & " +
             " & ".join(f"{get(a,'Oracle',c):.4f}" for c in caps) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT_T / f"tab_ladder_{task}.tex").write_text("\n".join(L) + "\n")
    print(f"tables/tab_ladder_{task}.tex")


def tab_matched(task="7d", cap="S4"):
    d = load_capacity(task)
    a = agg(d)
    L = [r"\begin{tabular}{lrrrl}", r"\toprule",
         r"method & AUC & PR-AUC & LL & $\Delta$AUC vs Correct\\", r"\midrule"]
    for arm in ["ULC", "DISTILL_Tmlp", "Oracle"]:
        if a[(a.method == arm) & (a.cap == cap)].empty:
            continue
        dd, p = paired(d, arm, "ULC", cap)
        cmp_ = "--" if arm == "ULC" else (f"{num(dd,4,True)}{stars(p)}" if dd == dd else "--")
        nm = NAME[arm][0]
        if NAME[arm][1] == "ours":
            nm = r"\textbf{" + nm + "}"
        if arm == "Oracle":
            L.append(r"\midrule")
            nm += r" \emph{(not causal)}"
        L.append(f"{nm} & {get(a,arm,cap):.4f} & {get(a,arm,cap,'pr'):.4f} & "
                 f"{get(a,arm,cap,'ll'):.4f} & {cmp_}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT_T / f"tab_matched_{task}.tex").write_text("\n".join(L) + "\n")
    print(f"tables/tab_matched_{task}.tex")


def tab_ablation(task="7d", cap="S2"):
    d = load_capacity(task)
    a = agg(d)
    rows = [(r"\textsc{Distil}, teacher 9.5\,M (default)", "DISTILL_Tmlp"),
            (r"\quad teacher 57\,M", "DISTILL_T13"),
            (r"\quad teacher 60\,M", "DISTILL_T15"),
            (r"\quad teacher trains only $w$", "DISTILL_wonly_mlp"),
            (r"\quad teacher label unused (NoKD)", "DISTILL_nokd")]
    L = [r"\begin{tabular}{l r@{\hskip 1.4em} l}", r"\toprule",
         r"variant & AUC & $\Delta$ vs default\\", r"\midrule"]
    for lab, arm in rows:
        v = get(a, arm, cap)
        if v != v:
            continue
        dd, p = paired(d, arm, "DISTILL_Tmlp", cap)
        L.append(f"{lab} & {v:.4f} & " +
                 ("--" if arm == "DISTILL_Tmlp" else f"{num(dd,4,True)}{stars(p)}") + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT_T / f"tab_ablation_{task}.tex").write_text("\n".join(L) + "\n")
    print(f"tables/tab_ablation_{task}.tex")


# ======================================================================================
def fig_audit():
    d = pd.read_csv(OUT_D / "compliance_audit.csv")
    d = d[~d.arm.isin(["Pretrain", "DISTILL_T_hyb", "Oracle"])].copy()
    d["nm"] = d.arm.replace({"Vanilla_fresh": "Fresh", "DISTILL_T": "Wait", "TWICE": "Twice"})
    d = d.sort_values("visits_per_click")
    fig, ax = plt.subplots(figsize=(LNCS_W, 1.85))
    ok = (d.visits_per_click <= 1.01).values
    x = np.arange(len(d))
    ax.bar(x[ok], d.visits_per_click[ok], color="0.82", edgecolor="0.0", lw=0.6, width=0.72)
    ax.bar(x[~ok], d.visits_per_click[~ok], color="0.42", edgecolor="0.0", lw=0.6,
           width=0.72, hatch="////")
    ax.axhline(1.0, color="0.0", ls=(0, (4, 2)), lw=1.0)
    ax.annotate("one record per click", (len(d) - 0.45, 1.06), ha="right", va="bottom",
                fontsize=6.8)
    for xx, vv in zip(x, d.visits_per_click):
        if vv > 1.01:
            ax.annotate(f"{vv:.2f}", (xx, vv), ha="center", va="bottom", fontsize=6.0)
    ax.set_xticks(x)
    ax.set_xticklabels(d.nm, rotation=38, ha="right")
    ax.set_ylabel("records per click")
    ax.set_ylim(0, 3.45)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.25)
    fig.savefig(OUT_F / "fig_audit.pdf")
    plt.close(fig)
    print("figs/fig_audit.pdf")


def fig_capacity(task="7d"):
    d = load_capacity(task)
    a = agg(d)
    caps = [c for c in LADDER if not a[a.cap == c].empty]
    x = np.arange(len(caps))
    fig, ax = plt.subplots(1, 2, figsize=(LNCS_W, 2.15))

    style = [("DISTILL_Tmlp", "o-", "0.0", 1.7), ("ULC", "s--", "0.0", 1.0),
             ("FSIW", "^:", "0.45", 1.0), ("DISTILL_T", "v-.", "0.45", 1.0),
             ("Vanilla_fresh", "d-", "0.68", 0.9)]
    ys = [get(a, "Oracle", c) for c in caps]
    ax[0].plot(x, ys, "-", color="0.0", lw=0.8, label="Oracle")
    for arm, mk, col, lw in style:
        # the table's merged "Fresh = Distil-NoKD" label is too wide for the key
        ax[0].plot(x, [get(a, arm, c) for c in caps], mk, color=col, lw=lw, ms=4.0, mfc="white",
                   mew=1.0, label="Fresh (= NoKD)" if arm == "Vanilla_fresh" else NAME[arm][0])
    ax[0].set_xticks(x); ax[0].set_xticklabels(caps)
    ax[0].set_xlabel("student capacity"); ax[0].set_ylabel("AUC")
    # The curves fill the panel, so no in-data corner is free: open a band above the Oracle
    # line and put a flat 3-column key in it.
    lo_y, hi_y = ax[0].get_ylim()
    ax[0].set_ylim(lo_y, hi_y + 0.42 * (hi_y - lo_y))
    h, l = ax[0].get_legend_handles_labels()
    order = [1, 2, 3, 4, 5, 0]          # Oracle plotted first, but belongs last in the key
    ax[0].legend([h[i] for i in order], [l[i] for i in order], frameon=False, loc="upper left",
                 ncol=3, handlelength=1.6, handletextpad=0.4, columnspacing=0.75,
                 labelspacing=0.2, borderaxespad=0.0, fontsize=6.0)

    for arm, ref, mk, col, lab in (("DISTILL_Tmlp", "ULC", "o-", "0.0", r"Distil $-$ Correct"),
                                   ("ULC", "Oracle", "s--", "0.45", r"Correct $-$ Oracle")):
        g, lo, hi = [], [], []
        for c in caps:
            xs = d[(d.method == ref) & (d.cap == c)].sort_values("seed").auc.values
            yy = d[(d.method == arm) & (d.cap == c)].sort_values("seed").auc.values
            n = min(len(xs), len(yy))
            if n < 2:
                g.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
            dif = yy[:n] - xs[:n]
            se = dif.std(ddof=1) / np.sqrt(n)
            g.append(dif.mean()); lo.append(dif.mean() - 1.96 * se); hi.append(dif.mean() + 1.96 * se)
        g = np.array(g)
        ax[1].errorbar(x, g, yerr=[g - np.array(lo), np.array(hi) - g], fmt=mk, color=col,
                       lw=1.4, ms=4.0, mfc="white", mew=1.0, capsize=2.0, label=lab)
    ax[1].axhline(0.0, color="0.0", lw=0.8)
    ax[1].set_xticks(x); ax[1].set_xticklabels(caps)
    ax[1].set_xlabel("student capacity"); ax[1].set_ylabel(r"$\Delta$ AUC")
    ax[1].legend(frameon=False, loc="lower left", handlelength=2.1)
    for A in ax:
        for s in ("top", "right"):
            A.spines[s].set_visible(False)
    fig.tight_layout(pad=0.25, w_pad=1.3)
    fig.savefig(OUT_F / f"fig_capacity_{task}.pdf")
    plt.close(fig)
    print(f"figs/fig_capacity_{task}.pdf")


def facts(task="7d"):
    """Every number quoted in the prose, dumped so the text can be checked against the data."""
    d = load_capacity(task)
    a = agg(d)
    out = []
    for c in LADDER:
        dd, p = paired(d, "DISTILL_Tmlp", "ULC", c)
        d2, p2 = paired(d, "DISTILL_Tmlp", "DISTILL_nokd", c)
        d3, p3 = paired(d, "ULC", "Oracle", c)
        out.append(f"{c}: Distil-Correct {dd:+.5f} (p={p:.3g}) | Distil-NoKD {d2:+.5f} "
                   f"(p={p2:.3g}) | Correct-Oracle {d3:+.5f} (p={p3:.3g})")
    for c in LADDER:
        cand = [(get(a, m, c), NAME[m][0]) for m in INADM if get(a, m, c) == get(a, m, c)]
        if not cand:
            continue
        best = max(cand)
        out.append(f"{c}: best inadmissible {best[1]} {best[0]:.4f} vs Distil "
                   f"{get(a,'DISTILL_Tmlp',c):.4f}  gap {get(a,'DISTILL_Tmlp',c)-best[0]:+.4f}")
    (OUT_D / f"facts_{task}.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))



# ======================================================================================
# Target-hardness axis (Sect. 5.4). Source is pv7d_{cap}_runs.csv, which carries EVERY arm of
# the axis plus both anchors in ONE file per rung -- so the comparison is origin-matched by
# construction and the paired tests never cross a run boundary.
AXIS = [("hard: target is a realised label", [
            (r"\quad \textsc{Fresh}\ ($\tilde y = \Yo$)", "Vanilla_fresh"),
            (r"\quad \textsc{Reweight}\ (loss-weighted $\Yo$)", "FSIW"),
            (r"\quad \textsc{Wait}\ ($\tilde y = \Yv$, stale by $v$)", "DISTILL_T")]),
        ("hybrid: realised label $+$ learned correction", [
            (r"\quad \textsc{Twice}", "TWICE"),
            (r"\quad \textsc{Correct}", "ULC_aux5e4"),
            (r"\quad \textsc{Distil}", "DISTILL_Tmlp")]),
        ("soft: model estimate alone, no realised label", [
            (r"\quad \textsc{Soft}\ ($\tilde y = \hat p_v$)", "DISTILL_pvmlp")])]

INADM_ROWS = [(r"\quad \textsc{Vanilla}", "Vanilla"), (r"\quad \textsc{Es-Dfm}", "ES-DFM"),
              (r"\quad \textsc{Defuse}", "DEFUSE"),   (r"\quad \textsc{Ddfm}", "DDFM")]


def _axis_frames(task="7d"):
    out = {}
    for c in LADDER:
        f = RUNS / f"pv{task}_{c}_runs.csv"
        if f.exists():
            out[c] = pd.read_csv(f).drop_duplicates(["method", "seed"])
    return out


def _axis_pair(D, c, a, b):
    d = D[c]
    x = d[d.method == a].sort_values("seed").auc.values
    y = d[d.method == b].sort_values("seed").auc.values
    n = min(len(x), len(y))
    if n < 2:
        return float("nan"), float("nan")
    return x[:n].mean() - y[:n].mean(), stats.ttest_rel(x[:n], y[:n])[1]


def tab_hardness(task="7d"):
    """The paper's main table: every arm, grouped by its point on the hardness axis.

    Bold marks the best ADMISSIBLE arm in each column -- the Oracle is not causal and the
    inadmissible block cannot be run at all, so neither competes for the bold.
    """
    d = load_capacity(task)
    a = agg(d)
    caps = [c for c in LADDER if not a[a.cap == c].empty]
    adm = [k for _, rows in AXIS for _, k in rows]
    best = {}
    for c in caps:
        vals = [(get(a, k, c), k) for k in adm if get(a, k, c) == get(a, k, c)]
        best[c] = max(vals)[1] if vals else None
    L = [r"\begin{tabular}{l" + "r" * len(caps) + "}", r"\toprule",
         r"training target & " + " & ".join(caps) + r"\\",
         r"served parameters & " + " & ".join(PARAMS[c] for c in caps) + r"\\", r"\midrule"]
    for grp, rows in AXIS:
        L.append(r"\multicolumn{%d}{l}{\emph{%s}}\\" % (len(caps) + 1, grp))
        for lab, arm in rows:
            v = [get(a, arm, c) for c in caps]
            if all(x != x for x in v):
                continue
            L.append(f"{lab} & " + " & ".join(
                "--" if x != x else (f"\\textbf{{{x:.4f}}}" if best[c] == arm else f"{x:.4f}")
                for c, x in zip(caps, v)) + r"\\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{%d}{l}{\emph{inadmissible: $>1$ record per click}}\\" % (len(caps) + 1))
    for lab, arm in INADM_ROWS:
        v = [get(a, arm, c) for c in caps]
        if all(x != x for x in v):
            continue
        L.append(f"{lab} & " + " & ".join(f"{x:.4f}" if x == x else "--" for x in v) + r"\\")
    L.append(r"\midrule")
    L.append(r"\quad Oracle \emph{(not causal)} & " +
             " & ".join(f"{get(a,'Oracle',c):.4f}" for c in caps) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT_T / f"tab_hardness_{task}.tex").write_text("\n".join(L) + "\n")
    print(f"tables/tab_hardness_{task}.tex")



def fig_hardness(task="7d"):
    """Both crossovers in one panel.

    Everything is plotted as a margin against the non-causal Oracle, so the Oracle is the zero
    line. Two things are then visible at once: the soft and hybrid curves crossing EACH OTHER
    between S0 and S1, and both crossing ZERO between S1 and S2. Greyscale, distinct markers.
    """
    D = _axis_frames(task)
    caps = [c for c in LADDER if c in D]
    x = np.arange(len(caps))
    fig, ax = plt.subplots(figsize=(LNCS_W, 2.05))

    style = [("DISTILL_pvmlp", "o-", "0.0", 1.7, "Soft  ($\\hat p_v$, no realised label)"),
             ("DISTILL_Tmlp", "s--", "0.0", 1.1, "Hybrid ($Y^{(o)}\\!+\\!(1\\!-\\!Y^{(o)})w$)"),
             ("Vanilla_fresh", "d-.", "0.55", 1.0, "Hard  ($Y^{(o)}$, censored)")]
    for arm, mk, col, lw, lab in style:
        g, lo, hi = [], [], []
        for c in caps:
            d = D[c]
            y = d[d.method == arm].sort_values("seed").auc.values
            o = d[d.method == "Oracle"].sort_values("seed").auc.values
            n = min(len(y), len(o))
            dif = y[:n] - o[:n]
            se = dif.std(ddof=1) / np.sqrt(n)
            g.append(dif.mean()); lo.append(1.96 * se); hi.append(1.96 * se)
        ax.errorbar(x, g, yerr=[lo, hi], fmt=mk, color=col, lw=lw, ms=4.2, mfc="white",
                    mew=1.0, capsize=2.0, label=lab)
    ax.axhline(0.0, color="0.0", lw=0.9)
    ax.annotate("Oracle (not causal)", (len(caps) - 1, 0.0), fontsize=6.3, va="bottom",
                ha="right", xytext=(0, 2.5), textcoords="offset points")
    ax.set_xticks(x); ax.set_xticklabels(caps)
    ax.set_xlabel("served model capacity"); ax.set_ylabel(r"AUC $-$ Oracle AUC")
    lo_y, hi_y = ax.get_ylim()
    ax.set_ylim(lo_y, hi_y + 0.34 * (hi_y - lo_y))
    ax.legend(frameon=False, loc="upper right", handlelength=2.0, fontsize=6.3,
              labelspacing=0.25, borderaxespad=0.15)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT_F / f"fig_hardness_{task}.pdf")
    plt.close(fig)
    print(f"figs/fig_hardness_{task}.pdf")


def facts_axis(task="7d"):
    """Every crossover number the prose quotes, with its paired p-value."""
    D = _axis_frames(task)
    caps = [c for c in LADDER if c in D]
    out = [f"target-hardness axis, v={task}, 5 seeds, origin-matched within each rung"]
    for a, b, lab in (("DISTILL_pvmlp", "DISTILL_Tmlp", "soft   - hybrid"),
                      ("DISTILL_Tmlp", "Oracle",        "hybrid - Oracle"),
                      ("DISTILL_pvmlp", "Oracle",       "soft   - Oracle"),
                      ("DISTILL_Tmlp", "ULC_aux5e4",    "Distil - Correct"),
                      ("DISTILL_Tmlp", "Vanilla_fresh", "hybrid - Fresh ")):
        row = []
        for c in caps:
            g, pv = _axis_pair(D, c, a, b)
            row.append(f"{c} {g:+.4f} (p={pv:.2g})")
        out.append(f"  {lab}: " + "  ".join(row))
    for c in caps:
        arms = [k for _, rows in AXIS for _, k in rows]
        mm = {k: D[c][D[c].method == k].auc.mean() for k in arms if k in set(D[c].method)}
        out.append(f"  best at {c}: " + max(mm.items(), key=lambda kv: kv[1])[0])
    (OUT_D / f"facts_axis_{task}.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    tab_compliance()
    fig_audit()
    tab_ladder("7d"); tab_matched("7d"); tab_ablation("7d"); fig_capacity("7d"); facts("7d")
    tab_hardness("7d"); fig_hardness("7d"); facts_axis("7d")
