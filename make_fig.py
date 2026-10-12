#!/usr/bin/env python3
"""Figure 1: relative NE gain over Wait against served capacity, one panel per join delay.

    python3 make_fig.py        # -> figs/gain.pdf and .png

Built to the publication standards in the scientific-figures skill: Wong/Okabe-Ito
colourblind-safe palette, sans-serif 8-9pt type, L-shaped axes with outward ticks, no chart
junk, bold uppercase panel labels, vector PDF straight into LaTeX.

Why relative gain and not absolute NE: the arms differ by ~1-2% of a quantity that itself moves
12% across the capacity ladder, so on an absolute axis every line is a near-copy of the Wait
curve and the comparison is invisible. Plotting the paired difference puts the whole figure on
the quantity the paper argues about and pins Wait at zero.

Gains are paired per seed before averaging. Seeds are matched across arms, so the error band is
the sd of the per-seed differences, several times tighter than the sd of either arm alone, and
the correct interval for a paired comparison.

Oracle is drawn because it bounds the panel: without it a reader cannot tell whether a 2% gain
is most of what was available or a tenth of it. Fresh is NOT drawn; it sits 12-20% below the
axis and would flatten everything else into a band.

The A0-A1 span is shaded because the paper's third contribution lives there: it is the regime
where Distill-Only, which carries no hard label at all, still beats Wait.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
CAPS = ["A0", "A1", "A2", "A3", "A4"]
PARAMS = {"A0": 1_890, "A1": 9_810, "A2": 94_959, "A3": 1_184_439, "A4": 9_580_477}
OS = [5, 30, 90]
WAIT = "DISTILL_T"

#: Wong/Okabe-Ito, the Nature-standard colourblind-safe qualitative palette. Blue and vermilion
#: are the two maximally separable hues in it and carry the two targets; the oracle bound is
#: grey so it reads as scaffolding rather than as a fourth method.
BLUE, VERM, GREY = "#0072B2", "#D55E00", "#7F7F7F"
INK, MUTED, BAND = "#1A1A1A", "#4A4A4A", "#EFEFEF"

#: Colour encodes the target, line style encodes the attribution window. Never colour alone.
SERIES = [
    ("Oracle",      "7d", "Oracle, $v{=}7$d (not causal)", GREY, (0, (1, 1.8)), None),
    ("DISTILL_T13", "7d", "Correct-Distill, $v{=}7$d",     BLUE, "-",            "o"),
    ("DISTILL_T13", "1d", "Correct-Distill, $v{=}1$d",     BLUE, (0, (4, 1.6)),  "o"),
    ("DISTILL_pv",  "7d", "Distill-Only, $v{=}7$d",        VERM, "-",            "s"),
    ("DISTILL_pv",  "1d", "Distill-Only, $v{=}1$d",        VERM, (0, (4, 1.6)),  "s"),
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})


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
    fig, axes = plt.subplots(1, 3, figsize=(4.80, 2.12), sharey=True)

    for i, (ax, o) in enumerate(zip(axes, OS)):
        # The regime contribution (3) is about, shaded before the data so it sits behind it.
        ax.axvspan(PARAMS["A0"] / 1.9, PARAMS["A1"] * 1.9, color=BAND, lw=0, zorder=0)
        ax.axhline(0, color=INK, lw=0.8, zorder=2)

        for arm, task, lab, col, ls, mk in SERIES:
            p = paired(d, arm, o, task)
            if p.empty:
                continue
            ax.fill_between(p.x, p["mean"] - p.sd, p["mean"] + p.sd,
                            color=col, alpha=0.13, lw=0, zorder=2)
            ax.plot(p.x, p["mean"], color=col, ls=ls, marker=mk, ms=3.0, lw=1.3,
                    markeredgecolor="white", markeredgewidth=0.45, label=lab, zorder=3)

        ax.set_xscale("log")
        ax.set_xticks([PARAMS[c] for c in CAPS])
        ax.set_xticklabels(CAPS)
        ax.set_xticks([], minor=True)
        ax.tick_params(colors=MUTED, pad=1.5)
        ax.set_title(rf"$o = {o}$ min", fontsize=8, color=INK, pad=3)
        ax.text(0.0, 1.20, "ABC"[i], transform=ax.transAxes,
                fontsize=9.5, fontweight="bold", va="top", ha="left", color=INK)
        for sp in ax.spines.values():
            sp.set_color(MUTED)

    # One shared axis label rather than three copies: at this width the repetition costs more
    # vertical space than it conveys. The y-label is kept short because a rotated label longer
    # than the axes height gets clipped; the caption carries the full definition.
    fig.supxlabel("Served capacity", fontsize=7.5, color=MUTED, y=0.185)
    axes[0].set_ylabel("NE gain (%)", fontsize=7.5, color=INK, labelpad=2)
    # Headroom so the Oracle band never touches the top spine.
    lo, hi = axes[0].get_ylim()
    axes[0].set_ylim(lo, hi + 0.12 * (hi - lo))

    # Two legends rather than five entries. Colour and line style are independent encodings, so
    # spelling them out separately halves the width and teaches the reader the scheme; a single
    # five-entry row does not fit the LNCS text block without shrinking the type below spec.
    # Both anchored by their top edge at the same y so the two blocks align, whatever their
    # differing entry counts. "\," is a LaTeX control sequence matplotlib does not parse outside
    # mathtext, so the unit space here is a literal space.
    line = lambda **kw: Line2D([], [], lw=1.3, **kw)
    common = dict(loc="upper left", ncol=1, frameon=False, fontsize=6.8,
                  handlelength=2.2, handletextpad=0.5, labelspacing=0.32,
                  borderpad=0, borderaxespad=0)
    fig.legend(handles=[line(color=BLUE, marker="o", ms=3.0, label="Correct-Distill"),
                        line(color=VERM, marker="s", ms=3.0, label="Distill-Only"),
                        line(color=GREY, ls=(0, (1, 1.8)), label="Oracle (not causal)")],
               bbox_to_anchor=(0.095, 0.145), **common)
    fig.legend(handles=[line(color=INK, label="$v = 7$ d"),
                        line(color=INK, ls=(0, (4, 1.6)), label="$v = 1$ d")],
               bbox_to_anchor=(0.605, 0.145), **common)
    # Explicit geometry rather than tight_layout: the two detached legends and the shared
    # x-label are figure-level artists that tight_layout cannot see, so it leaves a void.
    fig.subplots_adjust(left=0.098, right=0.995, top=0.855, bottom=0.315, wspace=0.14)

    out = HERE / "figs" / "gain.pdf"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
