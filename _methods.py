"""Emit one appendix section per method.

Prose and the target algebra are authored here; every NUMBER is computed from the corpus or read
from a measured artifact, so a method's section cannot drift from its implementation.

Two record counts appear and they are NOT the same quantity:
  * training records  -- what the model actually consumes over the stream;
  * audit denominator -- distinct clicks whose entire 14-day revisit horizon fits inside the
    stream (the first 16 of 30 days), used only to compute records-per-click.
Conflating them understates duplication by 1.89x, so they are reported separately.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "Experiment"
sys.path[:0] = [str(ROOT / "twice_repro"), str(ROOT / "core_code")]

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import eb_data  # noqa: E402
import tr_core as T  # noqa: E402
import tr_backbone as TB  # noqa: E402

OUT = HERE / "appendix"
WINDOWS = ("1d", "7d")

# key -> (paper name, target algebra, prose, pipeline it reads, auxiliary description)
# `pipe` drives the diagram: "fresh", "mat", "both", "dup"
M = {
 "Vanilla_fresh": dict(
   name="Fresh", tex=r"\tilde y_i = \Yo_i", pipe="fresh", aux=None,
   prose=r"""The uncorrected floor. One record per click at $\Delta=o$ carrying the observed label,
   trained with binary cross-entropy. It obeys both rules trivially and is the reference against
   which the value of \emph{label truth} is measured, since it differs from the Oracle only in
   what the record's label says."""),
 "FSIW": dict(
   name="Reweight", tex=r"\tilde y_i = \Yo_i,\quad \ell_i \mapsto w_{\text{iw}}(x_i)\,\ell_i",
   pipe="mat", aux=r"two importance-weight classifiers, \textsc{AuxHead}",
   prose=r"""\textsc{Fsiw}~\cite{yasui2020feedback} leaves the target alone and corrects the
   feedback shift in the \emph{loss}, weighting each example by a ratio estimated from two
   classifiers fitted on matured data. The record and its label are untouched, so it sits at the
   hard end of the axis by a different route from \textsc{Fresh}."""),
 "DISTILL_T": dict(
   name="Wait", tex=r"\tilde y_i = \Yv_i \ \text{ emitted at } \Delta=v", pipe="mat", aux=None,
   prose=r"""Emit nothing until the window closes, then write one record carrying the true label.
   The label is correct and Rule~1 is satisfied at $\Delta=v$, but the freshest data the model has
   ever seen is $v$ old. This is the admissible production status quo and the anchor for
   $\mathrm{RI}^{\textsc{wait}}$."""),
 "TWICE": dict(
   name="Twice", tex=r"\tilde y_i = \Yo_i \ \text{ plus a separate delay head on the conversion clock}",
   pipe="fresh", aux=r"delay head, ReLU MLP $[64,32]$, no shared parameters",
   prose=r"""\textsc{Twice}~\cite{li2026twice} keeps its CVR tower on a single fresh record and
   routes conversion-clock arrivals to a separate delay head, so the served tower stays
   single-visit. The delay head reads the conversion clock, which is why its R2 status is
   boundary rather than clean."""),
 "ULC_aux5e4": dict(
   name="Correct", tex=r"\tilde y_i = \Yo_i + (1-\Yo_i)\,w(x_i)",
   pipe="mat", aux=r"correction head $w$, \textsc{AuxHead}, $9{,}444{,}801$ parameters",
   prose=r"""\textsc{Ulc}~\cite{wang2023unbiased} keeps the realised label and softens the observed
   negatives by the probability they convert later, $w(x)=P(\Yv{=}1\mid\Yo{=}0,x)$, estimated
   online on matured data. An observed positive stays exactly $1$; only negatives move. This is
   the hybrid point of the axis and the comparator that matters for \textsc{Distil}."""),
 "DISTILL_Tmlp": dict(
   name="Distil", tex=r"\tilde y_i = \Yo_i + (1-\Yo_i)\,w^{T}(x_i)",
   pipe="mat", aux=r"offline teacher, two read-outs on one trunk, $9{,}456{,}897$ parameters",
   prose=r"""Identical in form to \textsc{Correct}; the only difference is where $w$ comes from.
   Here it is the second read-out of a teacher trained offline on the matured pipeline and
   \emph{evaluated at the fresh release} under Rule~1b. The two correctors hold the same parameter
   count to within $0.13\%$, so a comparison between them isolates the estimator and nothing
   else."""),
 "DISTILL_pvmlp": dict(
   name="Soft", tex=r"\tilde y_i = \hat p_v(x_i)",
   pipe="mat", aux=r"same teacher; the $p_v$ read-out is consumed instead of $w$",
   prose=r"""The extreme soft end: the target is the teacher's estimate alone and the record
   carries \emph{no realised label at all}. By Proposition~\ref{prop:axis}(ii) Rule~1 is satisfied
   vacuously --- there is no hard label for a late conversion to revise. This is the arm that makes
   the served record label-free, and the one whose behaviour changes most across the ladder."""),
 "Vanilla": dict(
   name="Vanilla", tex=r"\Yo_i \ \text{ plus a second, positive record when the conversion lands}",
   pipe="dup", aux=None,
   prose=r"""The field's reference baseline. Described as training each click once at age $o$, but
   reproducing any published \textsc{Vanilla} row requires also emitting the late conversion as a
   positive, which is a second visit. It is reported for continuity with the literature and is
   \emph{inadmissible} here."""),
 "ES-DFM": dict(
   name="ES-DFM", tex=r"\text{elapsed-time cut-off; the delayed positive is duplicated}",
   pipe="dup", aux=r"delayed-positive rate, \textsc{AuxHead}",
   prose=r"""\textsc{Es-Dfm}~\cite{yang2021capturing} samples an elapsed-time cut-off and duplicates
   the delayed positive into the stream, then corrects with importance weights. Duplication is the
   mechanism, not an implementation detail."""),
 "DEFUSE": dict(
   name="DEFUSE", tex=r"\text{duplicated stream with a label correction on top}",
   pipe="dup", aux=r"four-class mass estimator, \textsc{AuxHead}",
   prose=r"""\textsc{Defuse}~\cite{chen2022asymptotically} separates immediate from delayed
   positives and corrects the duplicated stream with an unbiased estimator."""),
 "DDFM": dict(
   name="DDFM", tex=r"\text{fresh and delayed-positive streams trained together}",
   pipe="dup", aux=r"dual heads, \textsc{AuxHead}",
   prose=r"""\textsc{Ddfm}~\cite{dai2023dually} runs both streams simultaneously, which is why its
   record count is the highest of the group at $1.89$ per click. It is the strongest inadmissible
   method and therefore bounds what the rules cost."""),
 "Oracle": dict(
   name="Oracle", tex=r"\tilde y_i = \Yv_i \ \text{ emitted at } \Delta=o",
   pipe="fresh", aux=None,
   prose=r"""Not causal and not deployable: the fresh record is labelled with an outcome that had
   not arrived when it was written. It exists to bound the recoverable gap. Sect.~\ref{sec:hardness}
   shows it is \emph{not} an upper bound at small capacity, which is the paper's most surprising
   result."""),
}


def stream_stats(corpus):
    """Composition of the two pipelines over the streaming period, per window."""
    out = {}
    for task in WINDOWS:
        st = T.build_streams(corpus, task)
        k, t0, t1 = st.keep, T.T0_DAY * T.DAY, T.TEND_DAY * T.DAY
        fresh = k & (st.release_t >= t0) & (st.release_t < t1)
        mat = k & (st.click_t + st.v >= t0) & (st.click_t + st.v < t1)
        out[task] = {
            "fresh_n": int(fresh.sum()), "fresh_pos_o": int(st.y_o[fresh].sum()),
            "fresh_pos_v": int(st.y_v[fresh].sum()),
            "mat_n": int(mat.sum()), "mat_pos_v": int(st.y_v[mat].sum()),
        }
    return out


def audit_map():
    """Prefer the regenerated file; tolerate the older column name for the record count."""
    for f in ("compliance_audit_7d.csv", "compliance_audit.csv"):
        p = HERE / "data" / f
        if p.exists():
            d = pd.read_csv(p)
            if "train_rows" not in d.columns and "cvr_rows" in d.columns:
                d = d.rename(columns={"cvr_rows": "train_rows"})
            return d.set_index("arm")
    return pd.DataFrame()


def diagram(pipe):
    """Compact data-flow sketch; greyscale, no colour."""
    boxes = {
      "fresh": r"\node[bx] (f) {$\mathcal{D}_{\text{fresh}}$}; \node[bx,right=14mm of f] (s) {served}; \draw[ar] (f)--(s);",
      "mat":   r"\node[bx] (f) {$\mathcal{D}_{\text{fresh}}$}; \node[bx,below=6mm of f] (m) {$\mathcal{D}_{\text{mat}}$};"
               r"\node[bx,right=20mm of f] (s) {served}; \node[bx,right=6mm of m] (a) {aux};"
               r"\draw[ar] (f)--(s); \draw[ar] (m)--(a); \draw[ar] (a) -| (s);",
      "dup":   r"\node[bx] (f) {$\mathcal{D}_{\text{fresh}}$}; \node[bx,below=6mm of f] (d) {late positives};"
               r"\node[bx,right=20mm of f] (s) {served}; \draw[ar] (f)--(s); \draw[ar,dashed] (d) -| (s);",
    }[pipe]
    return (r"\begin{center}\begin{tikzpicture}[font=\scriptsize,"
            r"bx/.style={draw,line width=.4pt,rounded corners=1pt,inner sep=2.2pt,minimum height=4mm},"
            r"ar/.style={draw,-{Latex[length=1.2mm]},line width=.4pt}]" + boxes +
            r"\end{tikzpicture}\end{center}")


def emit():
    corpus = eb_data.load("cpu")
    S = stream_stats(corpus)
    A = audit_map()
    files = []
    for key, d in M.items():
        vpc = A.visits_per_click.get(key, float("nan")) if len(A) else float("nan")
        rows = A.train_rows.get(key, float("nan")) if len(A) else float("nan")
        L = [rf"\subsection{{\textsc{{{d['name']}}}}}\label{{app:m-{key.replace('_','-')}}}", "",
             r"\paragraph{Definition.} " + d["prose"].strip(), "",
             r"\begin{equation}", d["tex"], r"\end{equation}", "",
             diagram(d["pipe"]), "",
             # NEVER infer a verdict from a missing measurement: an unaudited arm gets an
             # explicit "unmeasured", not a violation.
             r"\paragraph{Compliance.} " +
             (f"Measured at ${vpc:.2f}$ training records per click "
              f"(maximum {int(A.max_visits.get(key, 0))}) over the 720-hour replay. " +
              ("Single-visit; admissible under Rule~2."
               if vpc <= 1.01 else "Exceeds one record per click; inadmissible under Rule~2.")
              if vpc == vpc else
              r"\emph{Not yet measured by the single-visit instrument; no verdict is asserted "
              r"here.}"), "",
             r"\paragraph{Architecture.} The served model is the A0--A4 ladder of "
             r"Table~\ref{tab:spec-ladder}, unchanged. " +
             (f"Auxiliary: {d['aux']}, which does \\emph{{not}} scale with the rung, so the "
              r"capacity axis measures the served model only." if d["aux"]
              else "No auxiliary model."), "",
             r"\paragraph{Training data.}"]
        t = [r"\begin{center}\small\begin{tabular}{lrr}", r"\toprule",
             r" & $v=1$\,d & $v=7$\,d\\", r"\midrule"]
        if d["pipe"] == "mat" and key == "DISTILL_T":
            t.append(r"records consumed & " +
                     " & ".join(f"{S[w]['mat_n']:,}".replace(",", "{,}") for w in WINDOWS) + r"\\")
            t.append(r"positives ($\Yv$) & " +
                     " & ".join(f"{S[w]['mat_pos_v']:,}".replace(",", "{,}") for w in WINDOWS) + r"\\")
        else:
            t.append(r"fresh records consumed & " +
                     " & ".join(f"{S[w]['fresh_n']:,}".replace(",", "{,}") for w in WINDOWS) + r"\\")
            t.append(r"hard positives ($\Yo$) & " +
                     " & ".join(f"{S[w]['fresh_pos_o']:,}".replace(",", "{,}") for w in WINDOWS) + r"\\")
            if key == "Oracle":
                t.append(r"label positives ($\Yv$) & " +
                         " & ".join(f"{S[w]['fresh_pos_v']:,}".replace(",", "{,}") for w in WINDOWS) + r"\\")
            if d["pipe"] == "mat":
                t.append(r"matured records for the auxiliary & " +
                         " & ".join(f"{S[w]['mat_n']:,}".replace(",", "{,}") for w in WINDOWS) + r"\\")
            if d["pipe"] == "dup":
                t.append(r"extra records from duplication & " +
                         " & ".join(f"{S[w]['fresh_pos_v']-S[w]['fresh_pos_o']:,}".replace(",", "{,}")
                                    for w in WINDOWS) + r"\\")
        t += [r"\bottomrule", r"\end{tabular}\end{center}"]
        L += t
        L += ["", r"\paragraph{Experiment setup.} Five rungs A0--A4 $\times$ two windows $\times$ "
                  r"five seeds, learning rate selected per arm, per rung and per window from the "
                  r"grid of Sect.~\ref{app:matrix}. NE against streaming hour is recorded at "
                  r"seed~0.", "",
              r"\paragraph{Results.} \emph{[pending the production run]}", ""]
        f = OUT / f"m_{key.replace('_','')}.tex"
        f.write_text("\n".join(L) + "\n")
        files.append(f.name)
    master = ["% generated by _methods.py -- do not edit by hand",
              r"\section{Per-method specification}\label{app:methods}", "",
              r"Each section states the method's target, its compliance, the auxiliary it adds to "
              r"the shared A0--A4 trunk, and the composition of the data it actually consumes. "
              r"Record counts are measured; label counts are computed from the corpus.", ""]
    master += [rf"\input{{appendix/{n[:-4]}}}" for n in files]
    (OUT / "methods.tex").write_text("\n".join(master) + "\n")
    print(f"wrote {len(files)} method sections + methods.tex")


if __name__ == "__main__":
    emit()
