"""Architecture facts for the appendix, introspected from the instantiated models.

Gate G6: nothing here is arithmetic done in prose. Every shape and parameter count is read off a
real `tr_backbone.build(...)` module, so the appendix cannot drift from the code.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "Experiment"
sys.path[:0] = [str(ROOT / "twice_repro"), str(ROOT / "core_code")]

import torch  # noqa: E402
import eb_data  # noqa: E402
import tr_backbone as TB  # noqa: E402

STUDENTS = ["S0", "S1", "S2", "S3", "S4"]
TEACHERS = ["mlp", "T13", "T15"]


#: L0Model keeps its tables as nn.Parameter under `sparse_arch.tables.*`; CriteoMLP uses
#: nn.Embedding under `emb.*`. Classify by name so both are handled.
def _is_table(name):
    return name.startswith("sparse_arch.tables") or name.startswith("emb.")


def _split(model):
    """Parameters in the categorical tables vs everything else (the trunk)."""
    emb = sum(p.numel() for n, p in model.named_parameters() if _is_table(n))
    tot = sum(p.numel() for p in model.parameters())
    return emb, tot - emb, tot


def collect(corpus, device="cpu"):
    out = {}
    for cap in STUDENTS + TEACHERS:
        try:
            m = TB.build(cap, corpus, device, seed=0)
        except Exception as e:                      # a teacher spec may be unavailable
            out[cap] = {"error": str(e)}
            continue
        emb, trunk, tot = _split(m)
        lin = [(tuple(p.shape)) for n, p in m.named_parameters()
               if p.dim() == 2 and "emb" not in n]
        out[cap] = {"rows": TB.CAPACITY_ROWS.get(cap), "emb_dim": TB.EMB_DIM,
                    "n_cat": TB.N_CAT, "emb": emb, "trunk": trunk, "total": tot,
                    "dense_layers": lin}
    return out


def _fmt(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}\\,M"
    if n >= 1_000:
        return f"{n/1_000:.0f}\\,K"
    return str(n)


def emit(out_dir):
    corpus = eb_data.load("cpu")
    a = collect(corpus)
    L = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"model & table rows & emb.\ params & trunk params & total & dense stack\\",
         r"\midrule",
         r"\multicolumn{6}{l}{\emph{served model (the S0--S4 ladder)}}\\"]
    for cap in STUDENTS:
        r = a[cap]
        if "error" in r:
            continue
        stack = r"$\to$".join(str(s[0]) for s in r["dense_layers"]) or "--"
        L.append(f"\\quad {cap} & {r['rows']:,} & {_fmt(r['emb'])} & {r['trunk']:,} & "
                 f"{_fmt(r['total'])} & {stack}\\\\".replace(",", "{,}"))
    L.append(r"\midrule")
    L.append(r"\multicolumn{6}{l}{\emph{matured-data estimator (training only)}}\\")
    for cap in TEACHERS:
        r = a[cap]
        if "error" in r:
            continue
        stack = r"$\to$".join(str(s[0]) for s in r["dense_layers"]) or "--"
        L.append(f"\\quad {cap} & {r['rows']:,} & {_fmt(r['emb'])} & {r['trunk']:,} & "
                 f"{_fmt(r['total'])} & {stack}\\\\".replace(",", "{,}"))
    L += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "tab_arch.tex").write_text("\n".join(L) + "\n")

    facts = [f"{k}: rows={v.get('rows')} emb={v.get('emb')} trunk={v.get('trunk')} "
             f"total={v.get('total')}" for k, v in a.items()]
    (out_dir / "arch_facts.txt").write_text("\n".join(facts) + "\n")
    print("\n".join(facts))
    return a


def params_map():
    """{rung: '146K'} introspected, for build_artifacts. Replaces a hardcoded literal."""
    a = collect(eb_data.load("cpu"))
    return {c: _fmt(a[c]["total"]).replace("\\,", "") for c in STUDENTS if "error" not in a[c]}


if __name__ == "__main__":
    emit(Path(__file__).resolve().parent / "tables")
    print("\nPARAMS =", params_map())
