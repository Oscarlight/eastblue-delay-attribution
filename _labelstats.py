"""Label and training-data statistics for the appendix, computed from the corpus.

Gate G5: positives + negatives = N at every window, and the observed-positive rate equals
base x F(o). Both are asserted here rather than left to the reader.

Nothing in this file depends on a trained model, so it runs on CPU in seconds and its numbers are
fixed properties of the corpus and the protocol.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "Experiment"
sys.path[:0] = [str(ROOT / "twice_repro"), str(ROOT / "core_code")]

import torch  # noqa: E402
import eb_data  # noqa: E402
import tr_core as T  # noqa: E402

WINDOWS = ("1d", "7d")


def stats(corpus, task):
    st = T.build_streams(corpus, task)
    k = st.keep                                   # clicks inside [0, TEND)
    yv, yo, d = st.y_v[k], st.y_o[k], st.delay[k]
    n = int(k.sum())
    pos_v, pos_o = int(yv.sum()), int(yo.sum())
    # a click that converts inside v but not by o: the censored positive the fresh label misses
    late = int(((yv > 0.5) & (yo < 0.5)).sum())
    fin = d[torch.isfinite(d)]
    within = d[(yv > 0.5)]
    q = [float(torch.quantile(within.float(), p)) / 3600.0 for p in (.5, .9, .99)]
    out = {
        "task": task, "v_hours": st.v / 3600, "o_hours": st.o / 3600, "n_clicks": n,
        "pos_v": pos_v, "neg_v": n - pos_v, "rate_v": pos_v / n,
        "pos_o": pos_o, "neg_o": n - pos_o, "rate_o": pos_o / n,
        "late_positives": late, "F_o": pos_o / pos_v,
        "ever_converts": int(torch.isfinite(d).sum()),
        "delay_med_h": q[0], "delay_p90_h": q[1], "delay_p99_h": q[2],
        "delay_max_d": float(fin.max()) / 86400.0,
    }
    # G5: the arithmetic must close
    assert out["pos_v"] + out["neg_v"] == n
    assert out["pos_o"] + out["neg_o"] == n
    assert out["pos_o"] + out["late_positives"] == out["pos_v"], "Y^o positives are a subset of Y^v"
    assert abs(out["rate_o"] - out["rate_v"] * out["F_o"]) < 1e-9
    return out


def emit(out_dir):
    corpus = eb_data.load("cpu")
    rows = [stats(corpus, t) for t in WINDOWS]

    def f(x, nd=4):
        return f"{x:,.{nd}f}" if isinstance(x, float) else f"{x:,}"

    L = [r"\begin{tabular}{lrr}", r"\toprule",
         r"quantity & $v=1$\,d & $v=7$\,d\\", r"\midrule",
         r"\multicolumn{3}{l}{\emph{clicks in the replay $[0,60)$\,d}}\\"]
    def row(lab, key, nd=0):
        L.append(f"\\quad {lab} & " + " & ".join(
            (f"{r[key]:,}" if nd == 0 else f"{r[key]:,.{nd}f}") for r in rows) + r"\\")
    row("clicks $N$", "n_clicks")
    row("ever converts (any horizon)", "ever_converts")
    L.append(r"\midrule\multicolumn{3}{l}{\emph{served target $\Yv=\ind[D\le v]$}}\\")
    row("positives", "pos_v"); row("negatives", "neg_v"); row("positive rate", "rate_v", 4)
    L.append(r"\midrule\multicolumn{3}{l}{\emph{fresh label $\Yo=\ind[D\le o]$, $o=1$\,h}}\\")
    row("positives", "pos_o"); row("negatives", "neg_o"); row("positive rate", "rate_o", 4)
    row("censored positives ($\\Yv{=}1,\\Yo{=}0$)", "late_positives")
    row("$F(o)=P(D\\le o\\mid D\\le v)$", "F_o", 4)
    L.append(r"\midrule\multicolumn{3}{l}{\emph{delay of within-window converters (hours)}}\\")
    row("median", "delay_med_h", 2); row("p90", "delay_p90_h", 2); row("p99", "delay_p99_h", 2)
    L += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "tab_labelstats.tex").write_text("\n".join(L) + "\n")

    txt = "\n".join(f"{r['task']}: " + "  ".join(f"{k}={v}" for k, v in r.items()) for r in rows)
    (out_dir.parent / "data" / "labelstats.txt").write_text(txt + "\n")
    print(txt)
    return rows


if __name__ == "__main__":
    emit(Path(__file__).resolve().parent / "tables")
