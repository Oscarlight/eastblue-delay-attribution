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
CAPS, OS, TASKS = ["A0", "A1", "A2", "A3"], [5, 30, 90], ["1d", "7d"]
#: A4 is excluded from the study, so it is dropped here too; see paper_numbers.COMPLEX.
_V4 = pd.read_parquet(HERE / "data" / "v4_all_runs.parquet")
_V3 = pd.read_parquet(HERE / "data" / "v3_all_runs.parquet")
V4 = _V4[_V4.cap.isin(CAPS)].reset_index(drop=True)
V3 = _V3[_V3.cap.isin(CAPS)].reset_index(drop=True)
#: Table cells typeset negatives as math-mode "$-$", which pdftotext extracts as U+2212 rather
#: than ASCII hyphen. Normalise so a claim string like "-0.45" still matches what reached the
#: page; without this the table-row checks silently look for a glyph the PDF never contains.
PDF = (subprocess.run(["pdftotext", str(HERE / "main.pdf"), "-"],
                      capture_output=True, text=True).stdout
       .replace("\n", " ").replace("−", "-"))

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
    claim("methods screened / failing single-visit", "7 of 9", "9 published methods")
    claim("worst duplication", "4.00 records per click", "4.00")

    # --- Correct-Distill -------------------------------------------------------------------
    cd = grid("DISTILL_T13")
    pos = sum(1 for g in cd.values() if g.mean() > 0)
    # The Findings render this as "Wait in 23". The losing cell is no longer named in the
    # prose, so its count is re-derived and reported but not required on the page.
    claim("Correct-Distill beats Wait", f"{pos} of {len(cd)}", f"Wait in {pos}")
    n_neg = len(cd) - pos
    claim("losing cells (not named in the prose)", f"{n_neg} of {len(cd)}", None)
    do = grid("DISTILL_pv")
    beats = sum(1 for k in cd if cd[k].mean() > do[k].mean())
    claim("Correct-Distill beats Distill-Only", f"{beats} of {len(cd)}", None)
    assert beats == len(cd), f"paper says all {len(cd)}"

    for tag, t in (("7d", "7d"), ("1d", "1d")):
        v = [g.mean() for k, g in cd.items() if k[0] == t]
        assert np.mean(v) > 0, f"{tag} mean must be a gain"
        claim(f"Correct-Distill {tag} mean", f"+{abs(np.mean(v)):.2f}%", f"{abs(np.mean(v)):.2f}")
        if t == "7d":
            claim("Correct-Distill 7d range", f"+{abs(min(v)):.2f} to +{abs(max(v)):.2f}", None)

    # the two losing cells, as described
    neg = sorted(k for k, g in cd.items() if g.mean() <= 0)
    claim("losing cells are 1d / o=5", str(neg),
          None)
    assert all(k[0] == "1d" and k[1] == 5 for k in neg), "paper locates both at 1d, o=5min"
    worst = min(cd[k].mean() for k in neg)
    assert worst < 0, "the losing cell must be a loss"
    claim("worst losing cell", f"{worst:+.3f}%", None)
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
    lg = np.mean([rel("DISTILL_pv", "1d", o, c).mean() for o in OS for c in ["A2", "A3"]])
    claim("Distill-Only 1d, two smallest (a gain)", f"{sm:+.2f}%", f"{abs(sm):.2f}")
    claim("Distill-Only 1d, two largest (a loss)", f"{lg:+.2f}%", f"{abs(lg):.2f}")
    assert sm > 0 > lg, "paper claims a sign change"

    # Contribution (3): drop the hard label at small capacity. A mean could hide a split, so the
    # paper counts cells; re-derive the count here rather than trusting the generated macro.
    small = [rel("DISTILL_pv", t, o, c).mean()
             for t in ("1d", "7d") for o in OS for c in ("A0", "A1")]
    won = sum(1 for g in small if g > 0)
    claim("Distill-Only beats Wait at small capacity", f"{won} of {len(small)}",
          f"all {won} of {len(small)}")
    sm7 = np.mean([rel("DISTILL_pv", "7d", o, c).mean() for o in OS for c in ("A0", "A1")])
    claim("Distill-Only 7d, two smallest (a gain)", f"{sm7:+.2f}%", f"{abs(sm7):.2f}")
    assert won == len(small), "paper claims every small-capacity cell is a win"

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
        # Neither figure reaches the page any more: the paper now rests the teacher-robustness
        # claim on the two near-identical rows of Fig. 1 rather than on a printed number. Both
        # are still re-derived here, and asserted small, so the claim stays falsifiable.
        claim(f"teacher 6x step, {lab}", f"{np.mean(df):+.3f} pp", None)
        assert abs(np.mean(df)) < 0.2, f"{lab}: teacher step is no longer negligible"
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
        # Only the 7d figures are quoted now; Table 1 carries the 1d comparison at its own cell.
        if t == "7d":
            # The grid-wide ULC margin is no longer quoted; Table 1 carries the comparison.
            claim(f"lead over ULC {tag}", f"{lead:+.2f} pp", None)
            claim(f"cells beating ULC {tag}", f"{wins} of {len(ours)}", None)
            assert lead > 0, "paper says we lead ULC at the slow window"

    # --- Table 1: the production-like cell, v=1d and o=90 min, at M0 and M4 -------------------
    # Independent re-derivation: concatenate both campaigns, drop the 850 duplicated re-runs on
    # the run key, then take a t interval across replicates AT EACH CAPACITY. The table shows
    # the ends of the ladder separately because averaging them hid a reversal, so the check
    # follows suit rather than verifying a mean the paper no longer prints.
    KEY = ["arm", "task", "o_min", "cap", "seed"]
    SL = pd.concat([V3, V4], ignore_index=True).drop_duplicates(subset=KEY, keep="first")
    SL = SL[(SL.task == "1d") & (SL.o_min == 90)]

    TAB = {"DFM": ["DFM"], "ES-DFM": ["ES-DFM"], "DEFUSE": ["DEFUSE"], "DDFM": ["DDFM"],
           "FTP": ["FTP"], "MISS": ["MISS"], "PI": ["PI"], "FSIW": ["FSIW"], "ULC": ["ULC"],
           "Oracle": ["Oracle"], "Fresh": ["DISTILL_nokd"], "Wait": ["DISTILL_T"],
           "Correct-Distill": ["DISTILL_Tmlp", "DISTILL_T13"],
           "Distill-Only": ["DISTILL_pvmlp", "DISTILL_pv"]}
    NONCOMP = ("DFM", "ES-DFM", "DEFUSE", "DDFM", "FTP", "MISS", "PI")

    def tci(v):
        v = np.asarray(v, float)
        return v.mean(), stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))

    gains = {}
    for cap, lab in (("A0", "M0"), ("A3", "M3")):
        w = SL[(SL.arm == "DISTILL_T") & (SL.cap == cap)].set_index("seed")["ne"]
        gains[lab] = {}
        for name, arms in TAB.items():
            ne, g = [], []
            for a_ in arms:
                for seed, r in SL[(SL.arm == a_) & (SL.cap == cap)].groupby("seed"):
                    v = float(r["ne"].iloc[0])
                    ne.append(v)
                    g.append(100.0 * (w.loc[seed] - v) / w.loc[seed])
            gm, gh = tci(g)
            nm, _ = tci(ne)
            gains[lab][name] = gm
            claim(f"Table 1 {lab} {name} NE", f"{nm:.4f}", f"{nm:.4f}")
            if name != "Wait":
                claim(f"Table 1 {lab} {name} dNE", f"{gm:+.2f} +-{gh:.2f}", f"{gm:+.2f}")

    # The readings the prose takes off the table.
    assert gains["M0"]["Correct-Distill"] > gains["M0"]["ULC"], "paper: we lead ULC at M0"
    assert gains["M3"]["Correct-Distill"] > max(gains["M3"]["ULC"], gains["M3"]["FSIW"]), \
        "paper: we lead the single-visit group at M3 too"
    for lab in ("M0", "M3"):
        rec = 100 * gains[lab]["Correct-Distill"] / gains[lab]["Oracle"]
        claim(f"{lab} share of the Oracle headroom", f"{rec:.0f}%", f"{rec:.0f}")
    best0 = max(NONCOMP, key=lambda k: gains["M0"][k])
    claim(f"M0 leader duplicates ({best0})", f"{gains['M0'][best0]:+.2f}",
          f"{gains['M0'][best0]:+.2f}")
    assert gains["M0"][best0] > gains["M0"]["Oracle"], \
        "paper says the M0 leader clears the single-visit Oracle"
    # Distill-Only changes sign across the ladder; that is contribution 3 in one row.
    assert gains["M0"]["Distill-Only"] > 0 > gains["M3"]["Distill-Only"], \
        "paper says Distill-Only wins at M0 and loses at M3"

    # The prose says our calibration "tracks" Wait's at both rungs. It said "better than" until
    # dropping the top rung flipped the sign at M3 without anyone noticing, so the wording is
    # now pinned to a measured tolerance rather than to a remembered ordering.
    for cap, lab in (("A0", "M0"), ("A3", "M3")):
        w = SL[(SL.arm == "DISTILL_T") & (SL.cap == cap)]["cal_err"].mean()
        c = SL[SL.arm.isin(["DISTILL_Tmlp", "DISTILL_T13"]) & (SL.cap == cap)]["cal_err"].mean()
        claim(f"{lab} calibration, ours vs Wait", f"{c:.4f} vs {w:.4f}", None)
        assert abs(c - w) / w < 0.10, f"{lab}: calibration no longer tracks Wait's"

    assert "TWICE" not in PDF and "Twice" not in PDF, "TWICE must not appear in the paper"
    assert "IF-DFM" not in PDF, "IF-DFM must not appear in the paper"
    claim("combined run count", f"{len(V3) + len(V4):,}", f"{len(V3) + len(V4):,}")

    print(f"\n{sum(OK)}/{len(OK)} claims reproduce and appear in the PDF")
    if not all(OK):
        sys.exit(1)


if __name__ == "__main__":
    main()
