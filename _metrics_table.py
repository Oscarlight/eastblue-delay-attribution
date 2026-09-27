"""Multi-metric tables in the shape of TWICE's Table 2, with serving capacity in place of dataset.

Metric definitions follow TWICE Sect. 5.1 where they overlap: ROC-AUC, average precision as PR-AUC,
and RI per their Eq. (23), computed per seed from matched anchor and Oracle runs and then averaged.
NE and calibration are not reported by TWICE and are defined in the paper.

RI ANCHORS. Eq. (23) divides by the Vanilla-to-Oracle gap. Our Vanilla duplicates (1.08
records/click), leaving it only 0.0031-0.0053 AUC below the Oracle, so the quotient amplifies every
absolute difference 4-5x and leaves [0,100] in 34 of 50 cells. We therefore report RI against the
two admissible endpoints instead, which answer different questions:

    Oracle - Fresh   value of LABEL TRUTH at fixed Delta = o.  Same record, same wait, only the
                     target differs, so this is exactly the span of the hardness axis.
    Oracle - Wait    value of FRESHNESS at fixed label truth.  Wait emits at Delta = v, so this
                     mixes a change of wait with a change of target, but it is the admissible
                     production status quo.

Values above 100 are the Oracle inversion, not a defect; Fresh's negative RI under the Wait anchor
is likewise real -- Fresh genuinely loses to Wait.
"""
import pandas as pd

# (paper label, registry key) grouped by position on the hardness axis of Sect. 3.4
GROUPS = [
    (r"\emph{hard: target is a realised label}", [
        (r"\quad \textsc{Fresh}", "Vanilla_fresh"),
        (r"\quad \textsc{Reweight}", "FSIW"),
        (r"\quad \textsc{Wait}", "DISTILL_T")]),
    (r"\emph{hybrid: realised label $+$ learned correction}", [
        (r"\quad \textsc{Twice}", "TWICE"),
        (r"\quad \textsc{Correct}", "ULC_aux5e4"),
        (r"\quad \textsc{Distil}", "DISTILL_Tmlp")]),
    (r"\emph{soft: model estimate alone}", [
        (r"\quad \textsc{Soft}", "DISTILL_pvmlp")]),
    (r"\emph{inadmissible: $>1$ record per click}", [
        (r"\quad \textsc{Vanilla}", "Vanilla"),
        (r"\quad \textsc{Es-Dfm}", "ES-DFM"),
        (r"\quad \textsc{Defuse}", "DEFUSE"),
        (r"\quad \textsc{Ddfm}", "DDFM")]),
]
ADMISSIBLE = [k for _, rows in GROUPS[:3] for _, k in rows]
ALL_ARMS = [k for _, rows in GROUPS for _, k in rows]


def _strip0(v, nd=4):
    """.8273 rather than 0.8273: two characters per cell, twenty cells per row."""
    s = f"{v:.{nd}f}"
    return s[1:] if s.startswith("0.") else s


def _num(v, nd):
    """RI is an integer percentage with a real minus sign; the rest are stripped decimals."""
    if nd == 0:
        return (r"$-$%.0f" % abs(v)) if v < 0 else "%.0f" % v
    return _strip0(v, nd)


def ri(d, arm, cap, col, anchor):
    """TWICE Eq. (23): per seed from matched anchor/Oracle runs, then averaged."""
    def s(m):
        return d[(d.method == m) & (d.cap == cap)].set_index("seed")[col]
    a, o, v = s(anchor), s("Oracle"), s(arm)
    idx = a.index.intersection(o.index).intersection(v.index)
    if len(idx) < 2:
        return float("nan")
    return float(((v[idx] - a[idx]) / (o[idx] - a[idx])).mean() * 100)


def _cell(d, arm, cap, col, anchor):
    """One value: an RI if an anchor is given, otherwise the raw metric."""
    if anchor is None:
        return d[(d.method == arm) & (d.cap == cap)][col].mean()
    return ri(d, arm, cap, col, anchor)


def _emit(B, name, task, caps, d, spec):
    """spec: [(header, column, anchor_or_None, ndigits, 'min'|'max'|None)]."""
    ncol = 1 + len(spec) * len(caps)
    L = [r"\begin{tabular}{l" + ("r" * len(caps)) * len(spec) + "}", r"\toprule",
         "method" + "".join(r" & \multicolumn{%d}{c}{%s}" % (len(caps), h)
                            for h, _, _, _, _ in spec) + r"\\",
         "".join(r"\cmidrule(lr){%d-%d}" % (2 + i * len(caps), 1 + (i + 1) * len(caps))
                 for i in range(len(spec))),
         " & " + " & ".join(c for _ in spec for c in caps) + r"\\", r"\midrule"]

    # best admissible per (metric, capacity); an anchor arm never competes in its own column
    mark = {}
    for si, (_, col, anc, _, best) in enumerate(spec):
        if best is None:
            continue
        for c in caps:
            vals = [(_cell(d, k, c, col, anc), k) for k in ADMISSIBLE if k != anc]
            vals = [(v, k) for v, k in vals if v == v]
            if vals:
                mark[(si, c)] = (max if best == "max" else min)(vals)[1]

    for gname, rows in GROUPS:
        L.append(r"\multicolumn{%d}{l}{%s}\\" % (ncol, gname))
        for lab, arm in rows:
            cells = []
            for si, (_, col, anc, nd, _) in enumerate(spec):
                for c in caps:
                    if anc is not None and arm == anc:
                        cells.append(r"\emph{0}")          # anchor: zero by construction
                        continue
                    v = _cell(d, arm, c, col, anc)
                    if v != v:
                        cells.append("--")
                        continue
                    t = _num(v, nd)
                    cells.append(r"\textbf{%s}" % t if mark.get((si, c)) == arm else t)
            L.append(f"{lab} & " + " & ".join(cells) + r"\\")

    L.append(r"\midrule")
    cells = []
    for _, col, anc, nd, _ in spec:
        for c in caps:
            cells.append("--" if anc is not None
                         else _strip0(d[(d.method == "Oracle") & (d.cap == c)][col].mean(), nd))
    L.append(r"\quad Oracle \emph{(not causal)} & " + " & ".join(cells) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (B.OUT_T / f"{name}_{task}.tex").write_text("\n".join(L) + "\n")
    print(f"tables/{name}_{task}.tex")


def _anchor_diag(B, d, caps, task):
    """Why the anchor matters: headroom and RI spread for each candidate denominator."""
    cand = [(r"\textsc{Vanilla}, duplicating --- \emph{inadmissible}", "Vanilla"),
            (r"\textsc{Fresh}, $\Delta{=}o$, censored label", "Vanilla_fresh"),
            (r"\textsc{Wait}, $\Delta{=}v$, true label", "DISTILL_T")]
    L = [r"\begin{tabular}{l" + "r" * len(caps) + "cc}", r"\toprule",
         r"anchor & \multicolumn{%d}{c}{Oracle $-$ anchor (AUC)} & RI range & outside\\" % len(caps),
         r"\cmidrule(lr){2-%d}" % (1 + len(caps)),
         " & " + " & ".join(caps) + r" & \% & $[0,100]$\\", r"\midrule"]
    for lab, anc in cand:
        gaps = [d[(d.method == "Oracle") & (d.cap == c)]["auc"].mean()
                - d[(d.method == anc) & (d.cap == c)]["auc"].mean() for c in caps]
        vals = [ri(d, a, c, "auc", anc) for a in ALL_ARMS if a != anc for c in caps]
        vals = [v for v in vals if v == v]
        out = sum(1 for v in vals if v < 0 or v > 100)
        L.append(f"{lab} & " + " & ".join(_strip0(g, 4) for g in gaps) +
                 f" & $[{min(vals):.0f}, {max(vals):.0f}]$ & {out}/{len(vals)}" + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (B.OUT_T / f"tab_anchor_{task}.tex").write_text("\n".join(L) + "\n")
    print(f"tables/tab_anchor_{task}.tex")


def build(B, task="7d"):
    d = B.load_capacity(task)
    d["cal"] = d["pred_mean"] / d["base"]
    caps = [c for c in B.LADDER if not d[d.cap == c].empty]

    _emit(B, "tab_metrics", task, caps, d, [
        (r"NE $\downarrow$", "ne", None, 4, "min"),
        (r"AUC $\uparrow$", "auc", None, 4, "max"),
        (r"PR-AUC $\uparrow$", "pr_auc", None, 4, "max"),
        (r"Cal.\ ($\to 1$)", "cal", None, 3, None)])

    _emit(B, "tab_ri", task, caps, d, [
        (r"RI$^{\textsc{wait}}_{\text{AUC}}$", "auc", "DISTILL_T", 0, "max"),
        (r"RI$^{\textsc{wait}}_{\text{NE}}$", "ne", "DISTILL_T", 0, "max"),
        (r"RI$^{\textsc{fresh}}_{\text{AUC}}$", "auc", "Vanilla_fresh", 0, "max"),
        (r"RI$^{\textsc{fresh}}_{\text{NE}}$", "ne", "Vanilla_fresh", 0, "max")])

    _anchor_diag(B, d, caps, task)
