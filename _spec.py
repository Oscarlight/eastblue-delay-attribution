"""Generate the review tables for appendix/spec.tex.

Everything here is either measured (compliance audit, label statistics, introspected architecture)
or a declaration of what the production run will execute (the matrix). No hand-typed results.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "Experiment"
sys.path[:0] = [str(ROOT / "twice_repro"), str(ROOT / "core_code")]

import pandas as pd  # noqa: E402
import eb_data  # noqa: E402
import tr_backbone as TB  # noqa: E402

OUT = HERE / "appendix"
OUT.mkdir(exist_ok=True)

# paper name, registry key, target in math, rule-1 status, rule-2 status
METHODS = [
    ("Fresh", "Vanilla_fresh", r"$\tilde y=\Yo$", "yes", "yes"),
    ("Reweight", "FSIW", r"$\tilde y=\Yo$, loss $\times\,w_{\text{iw}}(x)$", "yes", "yes"),
    ("Wait", "DISTILL_T", r"$\tilde y=\Yv$ at $\Delta{=}v$", "yes", "yes"),
    ("Twice", "TWICE", r"$\tilde y=\Yo$ + delay head", "yes", "yes"),
    ("Correct", "ULC_aux5e4", r"$\tilde y=\Yo+(1-\Yo)\,w(x)$", "yes", "yes"),
    ("Distil", "DISTILL_Tmlp", r"$\tilde y=\Yo+(1-\Yo)\,w^{T}(x)$", "1b", "yes"),
    ("Soft", "DISTILL_pvmlp", r"$\tilde y=\hat p_v(x)$", "1b", "yes"),
    ("Vanilla", "Vanilla", r"$\Yo$ plus a late positive", "no", "no"),
    ("ES-DFM", "ES-DFM", r"duplicated delayed positive", "no", "no"),
    ("DEFUSE", "DEFUSE", r"duplicated stream + correction", "no", "no"),
    ("DDFM", "DDFM", r"both streams together", "no", "no"),
    ("Oracle", "Oracle", r"$\tilde y=\Yv$ at $\Delta{=}o$", "no", "yes"),
]
MARK = {"yes": r"\checkmark", "no": r"$\times$", "1b": r"\checkmark\,\textsuperscript{1b}"}


def compliance():
    f = HERE / "data" / "compliance_audit.csv"
    a = pd.read_csv(f).set_index("arm") if f.exists() else pd.DataFrame()
    L = [r"\begin{tabular}{l l c c r}", r"\toprule",
         r"method & target $\tilde y$ & Rule 1 & Rule 2 & rec./click\\", r"\midrule"]
    for nm, key, math, r1, r2 in METHODS:
        v = a.visits_per_click.get(key, float("nan")) if len(a) else float("nan")
        cell = "--" if v != v else f"{v:.2f}"
        L.append(f"\\textsc{{{nm}}} & {math} & {MARK[r1]} & {MARK[r2]} & {cell}" + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_spec_compliance.tex").write_text("\n".join(L) + "\n")
    print("appendix/tab_spec_compliance.tex")


def ladder():
    c = eb_data.load("cpu")
    L = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"rung & frontier node & emb.\ dim & table rows & tables & trunk & total\\", r"\midrule"]
    for k in ("A0", "A1", "A2", "A3", "A4"):
        m = TB.build(k, c, "cpu", 0)
        tbl = sum(p.numel() for n, p in m.named_parameters() if "sparse_arch.tables" in n)
        tot = sum(p.numel() for p in m.parameters())
        node, rows, emb = TB.AGGRESSIVE[k]
        nd = "--" if node is None else node
        L.append(f"{k} & {nd} & {emb} & {rows:,} & {tbl:,} & {tot-tbl:,} & {tot:,}"
                 .replace(",", "{,}") + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_spec_ladder.tex").write_text("\n".join(L) + "\n")
    print("appendix/tab_spec_ladder.tex")


def labels():
    src = HERE / "tables" / "tab_labelstats.tex"
    (OUT / "tab_spec_labels.tex").write_text(src.read_text())
    print("appendix/tab_spec_labels.tex")


def matrix():
    arms = len(METHODS)
    L = [r"\begin{tabular}{l l r}", r"\toprule",
         r"axis & levels & count\\", r"\midrule",
         rf"method & the {arms} arms of Table~\ref{{tab:spec-compliance}} & {arms}\\",
         r"capacity & A0, A1, A2, A3, A4 & 5\\",
         r"window & $v=1$\,d, $v=7$\,d & 2\\",
         r"seed & $0\ldots4$ (init and shuffle only; data identical) & 5\\",
         r"\midrule",
         rf"\multicolumn{{2}}{{l}}{{streaming runs}} & {arms*5*2*5:,}\\".replace(",", "{,}"),
         r"\multicolumn{2}{l}{learning-rate selections (7 rates, per arm/rung/window)} & "
         rf"{arms*5*2*7:,}\\".replace(",", "{,}"),
         r"\multicolumn{2}{l}{NE curves recorded (seed 0)} & "
         rf"{arms*5*2:,}\\".replace(",", "{,}"),
         r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_spec_matrix.tex").write_text("\n".join(L) + "\n")
    print("appendix/tab_spec_matrix.tex")


def timeline():
    """Hand-maintained: the generated version overlapped. Kept in appendix/fig_timeline.tikz."""
    print("appendix/fig_timeline.tikz (hand-maintained, not regenerated)")
    return


def _timeline_unused():
    tz = r"""% Record timing for one click, greyscale, LNCS text width.
\begin{figure}[t]
\centering
\begin{tikzpicture}[font=\scriptsize, x=1.05cm, y=1cm]
  \draw[-{Latex[length=1.3mm]}, line width=.5pt] (-0.3,0) -- (10.6,0);
  \foreach \x/\l in {0/{$C_i$ \\ features known}, 1.1/{$C_i{+}o$ \\ fresh record},
                     4.2/{$V_i$ \\ conversion}, 8.4/{$C_i{+}v$ \\ matured record}} {
    \draw[line width=.5pt] (\x,.12) -- (\x,-.12);
    \node[align=center, below=1pt, inner sep=1pt] at (\x,-.12) {\l};
  }
  \node[anchor=south east] at (10.6,.05) {time};
  % what each pipeline may read
  \draw[line width=1.1pt] (1.1,1.05) -- (1.1,1.05) node[circle,fill,inner sep=1.3pt]{};
  \node[anchor=west] at (1.25,1.05) {$\mathcal{D}_{\text{fresh}}$: one record, label $\Yo=\ind[D_i\le o]$, frozen};
  \draw[line width=1.1pt] (8.4,1.75) node[circle,fill,inner sep=1.3pt]{};
  \node[anchor=west] at (8.55,1.75) {};
  \node[anchor=west] at (1.25,1.75) {$\mathcal{D}_{\text{mat}}$: one record, label $\Yv=\ind[D_i\le v]$, frozen};
  \draw[dashed, line width=.4pt] (8.4,1.75) -- (1.25,1.75);
  % teacher read-out: trained on matured, evaluated at the fresh release (Rule 1b)
  \draw[-{Latex[length=1.2mm]}, line width=.7pt] (8.4,.45) .. controls (5,.9) .. (1.15,.45);
  \node[anchor=south] at (4.8,.62) {teacher trained on $\mathcal{D}_{\text{mat}}$, \emph{scored} at $C_i{+}o$ (Rule 1b)};
  % evaluation
  \draw[line width=.5pt, dotted] (1.1,-1.15) -- (2.2,-1.15);
  \node[anchor=west, align=left] at (2.3,-1.15)
    {scored once in the next hour $(T_n,T_{n+1}]$, retrospectively against $\Yv$};
\end{tikzpicture}
\caption{Timing for a single click. Features are known at $C_i$. The fresh pipeline writes one
record at $C_i{+}o$ carrying $\Yo$; the matured pipeline writes one at $C_i{+}v$ carrying $\Yv$.
Rule~2 permits the served model only one of them. Under Rule~1b the teacher learns from the
matured pipeline but is \emph{evaluated} at the fresh release, so the student's record carries a
soft target and no future hard label}
\label{fig:spec-timeline}
\end{figure}
"""
    (OUT / "fig_timeline.tikz").write_text(tz)
    print("appendix/fig_timeline.tikz")


if __name__ == "__main__":
    compliance(); ladder(); labels(); matrix(); timeline()
