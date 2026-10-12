#!/usr/bin/env python3
"""Figure 1: relative NE gain over Wait, one column per join delay and one row per teacher.

    python3 make_fig.py        # -> figs/gain.pdf and .png

Row 1 (A1, B1, C1) is the v3 campaign, in which the two targets ran against the MLP teacher.
Row 2 (A2, B2, C2) is the v4 campaign, in which they ran against the deep-trunk teacher. Both
campaigns cover the full 30-cell grid, so the rows are a like-for-like comparison and the point
of the figure is how little the teacher matters. Arm names differ across the campaigns and the
naming is a trap: DISTILL_pv carries no teacher suffix but Distill.teacher_capacity defaults to
"T13", so it is the deep-trunk arm, not an MLP or teacher-agnostic one. See ROWS below.

Built to the publication standards in the scientific-figures skill: Wong/Okabe-Ito
colourblind-safe palette, sans-serif 7-8pt type, L-shaped axes with outward ticks, no chart
junk, bold panel labels, vector PDF straight into LaTeX at the measured LNCS text width.

Why relative gain and not absolute NE: the arms differ by ~1-2% of a quantity that itself moves
12% across the capacity ladder, so on an absolute axis every line is a near-copy of the Wait
curve and the comparison is invisible. Plotting the paired difference puts the whole figure on
the quantity the paper argues about and pins Wait at zero.

Gains are paired per seed before averaging, and each row is read against the Wait arm of its own
campaign, which is what places the two rows on one axis. Seeds are matched across arms, so the
band is the sd of the per-seed differences, several times tighter than the sd of either arm
alone, and the correct interval for a paired comparison.

Oracle is drawn because it bounds each panel: without it a reader cannot tell whether a 2% gain
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
from matplotlib.lines import Line2D      # noqa: E402

HERE = Path(__file__).resolve().parent
CAPS = ["A0", "A1", "A2", "A3", "A4"]
PARAMS = {"A0": 1_890, "A1": 9_810, "A2": 94_959, "A3": 1_184_439, "A4": 9_580_477}
OS = [5, 30, 90]
WAIT = "DISTILL_T"                        # teacher-as-predictor: trains only on closed windows

#: Wong/Okabe-Ito, the Nature-standard colourblind-safe qualitative palette. Blue and vermilion
#: are the two maximally separable hues in it and carry the two targets; the oracle bound is
#: grey so it reads as scaffolding rather than as a fourth method.
BLUE, VERM, GREY = "#0072B2", "#D55E00", "#7F7F7F"
INK, MUTED, BAND = "#1A1A1A", "#4A4A4A", "#EFEFEF"

#: (row tag, parquet, row label, Correct-Distill arm, Distill-Only arm). Same two methods in
#: both rows under the campaign's own arm names.
ROWS = [
    ("1", "v3_all_runs.parquet", "Teacher 1", "DISTILL_Tmlp", "DISTILL_pvmlp"),
    ("2", "v4_all_runs.parquet", "Teacher 2", "DISTILL_T13",  "DISTILL_pv"),
]

#: Data keys are historical; the paper calls the served-capacity rungs M0..M4 so they are never
#: confused with the figure's A/B/C panel labels.
MDISP = [f"M{i}" for i in range(len(CAPS))]

#: Colour encodes the target, line style the attribution window. Never colour alone.
#: `cd`/`do` are resolved per row from ROWS.
SERIES = [
    ("Oracle", "7d", GREY, (0, (1, 1.8)), None),
    ("cd",     "7d", BLUE, "-",           "o"),
    ("cd",     "1d", BLUE, (0, (4, 1.6)), "o"),
    ("do",     "7d", VERM, "-",           "s"),
    ("do",     "1d", VERM, (0, (4, 1.6)), "s"),
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
    """Mean and sd of the per-seed relative gain over this campaign's own Wait, per capacity."""
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
    fig, axes = plt.subplots(2, 3, figsize=(4.80, 3.35), sharey=True, sharex=True)

    for r, (tag, fname, rowlab, cd, do) in enumerate(ROWS):
        d = pd.read_parquet(HERE / "data" / fname)
        arm_of = {"cd": cd, "do": do, "Oracle": "Oracle"}
        for c, o in enumerate(OS):
            ax = axes[r, c]
            # The regime contribution (3) is about, drawn before the data so it sits behind.
            ax.axvspan(PARAMS["A0"] / 1.9, PARAMS["A1"] * 1.9, color=BAND, lw=0, zorder=0)
            ax.axhline(0, color=INK, lw=0.8, zorder=2)

            for key, task, col, ls, mk in SERIES:
                p = paired(d, arm_of[key], o, task)
                if p.empty:
                    raise SystemExit(f"no data: {arm_of[key]} o={o} {task} in {fname}")
                ax.fill_between(p.x, p["mean"] - p.sd, p["mean"] + p.sd,
                                color=col, alpha=0.13, lw=0, zorder=2)
                ax.plot(p.x, p["mean"], color=col, ls=ls, marker=mk, ms=2.8, lw=1.2,
                        markeredgecolor="white", markeredgewidth=0.4, zorder=3)

            ax.set_xscale("log")
            ax.set_xticks([PARAMS[c_] for c_ in CAPS])
            ax.set_xticklabels(MDISP)
            ax.set_xticks([], minor=True)
            ax.tick_params(colors=MUTED, pad=1.5, labelbottom=True)
            ax.text(0.0, 1.17, f"{'ABC'[c]}$_{tag}$", transform=ax.transAxes,
                    fontsize=9, fontweight="bold", va="top", ha="left", color=INK)
            ax.set_title(rf"$o = {o}$ min", fontsize=8, color=INK, pad=3)
            for sp in ax.spines.values():
                sp.set_color(MUTED)

        axes[r, 0].set_ylabel("NE gain (%)", fontsize=7.5, color=INK, labelpad=2)

        # Row identity on the right edge, where it cannot collide with the y-label.
        axes[r, 2].text(1.04, 0.5, rowlab, transform=axes[r, 2].transAxes, rotation=270,
                        fontsize=7.5, color=INK, va="center", ha="left")

    # Headroom so the Oracle band never touches the top spine.
    lo, hi = axes[0, 0].get_ylim()
    axes[0, 0].set_ylim(lo, hi + 0.10 * (hi - lo))

    # One shared label per axis rather than six copies. The y-label is kept short because a
    # rotated label longer than the axes height gets clipped; the caption carries the detail.
    fig.supxlabel("Served capacity", fontsize=7.5, color=MUTED, y=0.158)

    # Two legends rather than five entries. Colour and line style are independent encodings, so
    # spelling them out separately halves the width and teaches the reader the scheme; a single
    # five-entry row does not fit the LNCS text block without shrinking the type below spec.
    line = lambda **kw: Line2D([], [], lw=1.2, **kw)
    common = dict(loc="upper left", ncol=1, frameon=False, fontsize=6.8,
                  handlelength=2.2, handletextpad=0.5, labelspacing=0.32,
                  borderpad=0, borderaxespad=0)
    fig.legend(handles=[line(color=BLUE, marker="o", ms=2.8, label="Correct-Distill"),
                        line(color=VERM, marker="s", ms=2.8, label="Distill-Only"),
                        line(color=GREY, ls=(0, (1, 1.8)), label="Oracle (not causal)")],
               bbox_to_anchor=(0.095, 0.108), **common)
    fig.legend(handles=[line(color=INK, label="$v = 7$ d"),
                        line(color=INK, ls=(0, (4, 1.6)), label="$v = 1$ d")],
               bbox_to_anchor=(0.605, 0.108), **common)

    # Explicit geometry rather than tight_layout: the detached legends and the shared axis
    # labels are figure-level artists that tight_layout cannot see, so it leaves a void.
    fig.subplots_adjust(left=0.105, right=0.955, top=0.915, bottom=0.255,
                        wspace=0.14, hspace=0.38)

    out = HERE / "figs" / "gain.pdf"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
