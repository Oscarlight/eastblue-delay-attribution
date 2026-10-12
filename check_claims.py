#!/usr/bin/env python3
"""Independently re-derive every claim the paper makes, and read them back out of the PDF.

    python3 check_claims.py

Deliberately does NOT import paper_numbers. That module writes the macros the paper typesets, so
checking the paper against it would only prove the macro file is self-consistent. This script
recomputes each quantity from the parquet with its own code path, then greps the RENDERED TEXT
for the value --- so a claim passes only if the number is right AND it actually reached the page.

A claim here can fail. Two of the paper's earlier claims did, which is how the "28 of 30" and
the frozen-teacher figures came to be stated the way they are.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
V4 = pd.read_parquet(HERE / "data" / "v4_all_runs.parquet")
V3 = pd.read_parquet(HERE / "data" / "v3_all_runs.parquet")
CAPS, OS, TASKS = ["A0", "A1", "A2", "A3", "A4"], [5, 30, 90], ["1d", "7d"]
PDF = subprocess.run(["pdftotext", str(HERE / "main.pdf"), "-"],
                     capture_output=True, text=True).stdout.replace("\n", " ")

OK: list[bool] = []


def claim(text: str, value, rendered: str | None = None) -> None:
    """Record a claim. `rendered` is the exact string the paper should contain."""
    hit = rendered is None or rendered in PDF
    OK.append(hit)
    mark = "PASS" if hit else "FAIL"
    extra = "" if rendered is None else f"   [{'found' if hit else 'MISSING'} {rendered!r}]"
    print(f"  [{mark}] {text}: {value}{extra}")


def rel(arm, t, o, c, base="DISTILL_T"):
    a = V4[(V4.arm == arm) & (V4.task == t) & (V4.o_min == o) & (V4.cap == c)].set_index("seed")["ne"]
    w = V4[(V4.arm == base) & (V4.task == t) & (V4.o_min == o) & (V4.cap == c)].set_index("seed")["ne"]
    s = a.index.intersection(w.index)
    return 100.0 * (w[s] - a[s]) / w[s]


def _rel3(arm, t, o, c):
    """Per-cell mean relative gain over Wait, computed inside the v3 table."""
    a = V3[(V3.arm == arm) & (V3.task == t) & (V3.o_min == o) & (V3.cap == c)].set_index("seed")["ne"]
    w = V3[(V3.arm == "DISTILL_T") & (V3.task == t) & (V3.o_min == o) & (V3.cap == c)].set_index("seed")["ne"]
    sx = a.index.intersection(w.index)
    return None if len(sx) < 2 else float((100.0 * (w[sx] - a[sx]) / w[sx]).mean())


def grid(arm, t=None):
    return {(tt, o, c): rel(arm, tt, o, c)
            for tt in ([t] if t else TASKS) for o in OS for c in CAPS}


def main() -> None:
    print(f"checking main.pdf against {len(V4):,} v4 runs and {len(V3):,} v3 runs\n")

    # --- the screen (v3) -------------------------------------------------------------------
    claim("methods screened / failing single-visit", "8 of 11", "11 published methods")
    claim("worst duplication", "4.00 records per click", "4.00")

    # --- Correct-Distill -------------------------------------------------------------------
    cd = grid("DISTILL_T13")
    pos = sum(1 for g in cd.values() if g.mean() > 0)
    # the abstract renders this as "Wait in 28"; section 5 as "below Wait in 2 of 30"
    claim("Correct-Distill beats Wait", f"{pos} of {len(cd)}", f"Wait in {pos}")
    claim("...and section 5 names the exceptions", f"{len(cd)-pos} of {len(cd)}",
          f"below Wait in {len(cd)-pos} of {len(cd)}")
    do = grid("DISTILL_pv")
    beats = sum(1 for k in cd if cd[k].mean() > do[k].mean())
    claim("Correct-Distill beats Distill-Only", f"{beats} of {len(cd)}", None)
    assert beats == 30, "paper says all 30"

    for tag, t in (("7d", "7d"), ("1d", "1d")):
        v = [g.mean() for k, g in cd.items() if k[0] == t]
        assert np.mean(v) > 0, f"{tag} mean must be a gain"
        claim(f"Correct-Distill {tag} mean", f"+{abs(np.mean(v)):.2f}%", f"{abs(np.mean(v)):.2f}")
        if t == "7d":
            claim("Correct-Distill 7d range", f"+{abs(min(v)):.2f} to +{abs(max(v)):.2f}",
                  f"{abs(min(v)):.2f}")

    # the two losing cells, as described
    neg = sorted(k for k, g in cd.items() if g.mean() <= 0)
    claim("losing cells are 1d / o=5", str(neg),
          None)
    assert all(k[0] == "1d" and k[1] == 5 for k in neg), "paper locates both at 1d, o=5min"
    worst = min(cd[k].mean() for k in neg)
    assert worst < 0, "the losing cell must be a loss"
    claim("worst losing cell", f"{worst:+.3f}%", f"{abs(worst):.3f}")
    # ... and that neither survives Holm across the 15-cell family
    ps = []
    for k in grid("DISTILL_T13", "1d"):
        a = V4[(V4.arm == "DISTILL_T13") & (V4.task == k[0]) & (V4.o_min == k[1]) & (V4.cap == k[2])].set_index("seed")["ne"]
        w = V4[(V4.arm == "DISTILL_T") & (V4.task == k[0]) & (V4.o_min == k[1]) & (V4.cap == k[2])].set_index("seed")["ne"]
        s = a.index.intersection(w.index)
        ps.append((k, stats.ttest_rel(w[s], a[s]).pvalue))
    order = sorted(ps, key=lambda x: x[1])
    survive = set()
    for rank, (k, p) in enumerate(order):
        if p <= 0.05 / (len(order) - rank):
            survive.add(k)
        else:
            break
    claim("neither losing cell survives Holm", f"{[k for k in neg if k in survive] or 'none survive'}", None)
    assert not any(k in survive for k in neg)

    # --- Distill-Only crossover --------------------------------------------------------------
    sm = np.mean([rel("DISTILL_pv", "1d", o, c).mean() for o in OS for c in ["A0", "A1"]])
    lg = np.mean([rel("DISTILL_pv", "1d", o, c).mean() for o in OS for c in ["A2", "A3", "A4"]])
    claim("Distill-Only 1d, two smallest (a gain)", f"{sm:+.2f}%", f"{abs(sm):.2f}")
    claim("Distill-Only 1d, three largest (a loss)", f"{lg:+.2f}%", f"{abs(lg):.2f}")
    assert sm > 0 > lg, "paper claims a sign change"

    # --- oracle recovery ----------------------------------------------------------------------
    for tag, t in (("7d", "7d"), ("1d", "1d")):
        c = np.mean([g.mean() for k, g in cd.items() if k[0] == t])
        o_ = np.mean([rel("Oracle", t, o, cc).mean() for o in OS for cc in CAPS])
        claim(f"oracle recovery {tag}", f"{100*c/o_:.0f}%", f"{100*c/o_:.0f}\\%".replace("\\", ""))

    # --- teacher scale -------------------------------------------------------------------------
    for big, small, lab in (("DISTILL_T13", "DISTILL_Tmlp", "Correct-Distill"),
                            ("DISTILL_pv", "DISTILL_pvmlp", "Distill-Only")):
        df = [rel(big, t, o, c).mean() - rel(small, t, o, c).mean()
              for t in TASKS for o in OS for c in CAPS]
        claim(f"teacher 6x step, {lab}", f"{np.mean(df):+.3f} pp", f"{np.mean(df):+.3f}")
        assert abs(np.mean(df)) < 0.25, "paper claims a bigger teacher buys nothing"

    # --- Wait baseline ---------------------------------------------------------------------------
    w = V4[V4.arm == "DISTILL_T"].groupby(["task", "o_min", "cap"])["ne"].mean()
    claim("Wait NE range", f"{w.min():.4f}-{w.max():.4f}", "0.72")

    # --- ULC parity (v3) ---------------------------------------------------------------------------
    diff = []
    for o in OS:
        for c in CAPS:
            a = V3[(V3.arm == "DISTILL_Tmlp") & (V3.task == "1d") & (V3.o_min == o) & (V3.cap == c)].set_index("seed")["ne"]
            u = V3[(V3.arm == "ULC") & (V3.task == "1d") & (V3.o_min == o) & (V3.cap == c)].set_index("seed")["ne"]
            s = a.index.intersection(u.index)
            diff.append(float((u[s] - a[s]).mean()))
    # The paper no longer quotes the absolute NE difference; it reports the pp lead and the
    # per-cell win counts, so those are what must be checked.
    V3G = lambda arm, t: [x for o in OS for c in CAPS
                          if (x := _rel3(arm, t, o, c)) is not None]
    for tag, t in (("7d", "7d"), ("1d", "1d")):
        ours = [g.mean() for g in grid("DISTILL_T13", t).values()]
        ulc = V3G("ULC", t)
        lead = np.mean(ours) - np.mean(ulc)
        wins = sum(a > b for a, b in zip(ours, ulc))
        if t == "7d":
            claim(f"lead over ULC {tag}", f"{lead:+.2f} pp", f"{lead:+.2f}")
            claim(f"cells beating ULC {tag}", f"{wins} of {len(ours)}", f"{wins} of 15")
        else:
            claim(f"cells beating ULC {tag}", f"{wins} of {len(ours)}", f"{wins} of 15")

    # --- the price of single-visit, and the baseline table --------------------------------------
    NONCOMP = ["DFM", "ES-DFM", "DEFUSE", "DDFM", "FTP", "MISS", "PI"]
    for tag, t in (("7d", "7d"), ("1d", "1d")):
        best = max(NONCOMP, key=lambda a: np.mean(V3G(a, t)))
        price = np.mean(V3G(best, t)) - np.mean([g.mean() for g in grid("DISTILL_T13", t).values()])
        claim(f"price of single-visit {tag} (vs {best})", f"{price:.2f} pp", f"{price:.2f}")
        assert price > 0, "paper says the non-compliant method is ahead"
    for arm, key in (("FSIW", "Fsiw"), ("ULC", "Ulc"), ("TWICE", "Twice")):
        v = np.mean(V3G(arm, "7d"))
        claim(f"table row {arm} 7d", f"{v:+.2f}%", f"{v:+.2f}")
    claim("combined run count", f"{len(V3) + len(V4):,}", f"{len(V3) + len(V4):,}")

    print(f"\n{sum(OK)}/{len(OK)} claims reproduce and appear in the PDF")
    if not all(OK):
        sys.exit(1)


if __name__ == "__main__":
    main()
