#!/usr/bin/env python3
"""Figure 1: relative NE gain over Wait against served capacity, one panel per join delay.

    python3 make_fig.py        # -> figs/gain.pdf and .png

Why relative gain and not absolute NE: the arms differ by ~1-2% of a quantity that itself moves
12% across the capacity ladder, so on an absolute axis every line is a near-copy of the Wait
curve and the comparison is invisible. Plotting the paired difference puts the whole figure on
the quantity the paper argues about and pins Wait at zero.

Gains are paired per seed before averaging. Seeds are matched across arms, so the error bar is
the sd of the per-seed differences -- several times tighter than the sd of either arm alone, and
the correct bar for a paired comparison.

Oracle is drawn because it bounds the panel: without it a reader cannot tell whether a 2% gain
is most of what was available or a tenth of it. Fresh is NOT drawn -- it sits 12-20% below the
axis and would flatten everything else into a band.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

HERE = Path(__file__).resolve().parent
CAPS = ["A0", "A1", "A2", "A3", "A4"]
PARAMS = {"A0": 1_890, "A1": 9_810, "A2": 94_959, "A3": 1_184_439, "A4": 9_580_477}
OS = [5, 30, 90]
WAIT = "DISTILL_T"

INK, MUTED, LINE = "#222222", "#6b7a7d", "#c9d4cf"
TEAL, CORAL, GREY = "#12807a", "#e0674f", "#9aa7a9"

#: (arm, window, label, colour, dash, marker)
SERIES = [
    ("Oracle",      "7d", "Oracle, $v{=}7$d (not causal)", GREY,  (0, (1, 1.6)), None),
    ("DISTILL_T13", "7d", "Correct-Distill, $v{=}7$d", TEAL,  "-",  "o"),
    ("DISTILL_T13", "1d", "Correct-Distill, $v{=}1$d", TEAL,  "--", "o"),
    ("DISTILL_pv",  "7d", "Distill-Only, $v{=}7$d",    CORAL, "-",  "s"),
    ("DISTILL_pv",  "1d", "Distill-Only, $v{=}1$d",    CORAL, "--", "s"),
]


def paired(d, arm, o, task):
    """Mean and sd of the per-seed relative gain over Wait, per capacity."""
    out = []
    for cap in CAPS:
        sel = (d.o_min == o) & (d.task == task) & (d.cap == cap)
        a = d[sel & (d.arm == arm)].set_index("seed")["ne"]
        w = d[sel & (d.arm == WAIT)].set_index("seed")["ne"]
        s = a.index.intersection(w.index)
        if len(s) < 2:
            continue
        g = 100.0 * (w[s] - a[s]) / w[s]
        out.append(dict(x=PARAMS[cap], mean=g.mean(), sd=g.std(ddof=1)))
    return pd.DataFrame(out)


def main() -> None:
    d = pd.read_parquet(HERE / "data" / "v4_all_runs.parquet")
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 1.72), sharey=True)
    for ax, o in zip(axes, OS):
        ax.axhline(0, color=INK, lw=1.0, zorder=2)
        for arm, task, lab, col, ls, mk in SERIES:
            p = paired(d, arm, o, task)
            if p.empty:
                continue
            ax.errorbar(p.x, p["mean"], yerr=p.sd, color=col, ls=ls, marker=mk, ms=2.7,
                        lw=1.5, capsize=1.5, elinewidth=.7, label=lab, zorder=3)
        ax.set_xscale("log")
        ax.set_xticks([PARAMS[c] for c in CAPS])
        ax.set_xticklabels(CAPS, fontsize=6.3)
        ax.set_xticks([], minor=True)
        ax.tick_params(labelsize=6.3, colors=MUTED, length=2)
        ax.set_title(rf"$o={o}$ min", fontsize=7.5, color=INK, pad=2)
        ax.grid(alpha=.3, lw=.4, color=LINE)
        for sp in ax.spines.values():
            sp.set_color(LINE)
            sp.set_linewidth(.6)
        ax.set_xlabel("served capacity", fontsize=6.8, color=MUTED)
    axes[0].set_ylabel("NE gain over Wait (%)", fontsize=7.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=6.0, frameon=False,
               bbox_to_anchor=(.5, -.015), handlelength=2.8, columnspacing=.85)
    fig.tight_layout(rect=(0, .085, 1, 1), w_pad=.8)
    out = HERE / "figs" / "gain.pdf"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
