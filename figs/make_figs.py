"""Figures for the ECIR short paper.  Run from Publication/ecir2027/:  python figs/make_figs.py

Springer LNCS compliance (Instructions for Authors, Sec. 4.5):
  * no reliance on colour -- greyscale fills plus distinct markers and line styles, so the
    figures are legible when the volume is printed in black and white;
  * drawn at the true LNCS text width (122 mm) and included at width=\\textwidth, so nothing is
    rescaled and no label ends up below the 6 pt floor;
  * vector PDF output.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd, numpy as np

LNCS_W = 4.80                      # 122 mm text width, in inches
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 6.8, "axes.titlesize": 8,
                     "pdf.fonttype": 42, "axes.linewidth": 0.6})

import os
# vendored copy keeps this repository self-contained; falls back to the experiment tree
A = "data/" if os.path.exists("data/manuscript_metrics.csv") else "../../Experiment/analysis/manuscript/"
M = pd.read_csv(A + "manuscript_metrics.csv")
P = pd.read_csv(A + "theory_prediction.csv")
S = ["S0", "S1", "S2", "S3", "S4"]

def piv(task, s):
    return M[(M["task"] == task) & (M["student"] == s)].pivot_table(
        index="origin", columns="arm", values="ne")

def recov(task, s, arm):
    p = piv(task, s)
    return (p["A"] - p[arm]).mean() / (p["A"] - p["oracle"]).mean()

# --- Fig 1: the two ceilings -----------------------------------------------------------------
# Correct is pinned under the Oracle line at 1.0 and tracks the Prop. 1 ceiling; Distil is not.
fig, ax = plt.subplots(1, 2, figsize=(LNCS_W, 2.05), sharey=True)
for k, task in enumerate(("1d", "7d")):
    pred = P[P.task == task]["pred"].mean()
    ax[k].axhline(1.0, color="0.0", ls="-", lw=0.8)
    ax[k].annotate("Oracle", (0.98, 1.0), xycoords=("axes fraction", "data"),
                   va="bottom", ha="right", fontsize=6.8)
    ax[k].axhline(pred, color="0.0", ls=(0, (4, 2)), lw=1.0)
    ax[k].annotate(f"correction ceiling {pred:.3f}", (0.02, pred),
                   xycoords=("axes fraction", "data"), va="top", fontsize=6.8)
    ax[k].plot(S, [recov(task, s, "C") for s in S], "o-", color="0.0", lw=1.3, ms=4.5,
               mfc="white", mew=1.1, label="Correct")
    ax[k].plot(S, [recov(task, s, "B") for s in S], "s:", color="0.45", lw=1.3, ms=4.0,
               label="Distil")
    ax[k].axhline(0, color="0.0", lw=0.5)
    ax[k].set_title(rf"$\Delta$ = {task}"); ax[k].set_xlabel("student capacity")
    ax[k].grid(alpha=.25, lw=.4)
ax[0].set_ylabel("fraction of penalty\nrecovered")
ax[0].legend(loc="lower left", frameon=False, handlelength=2.6)
plt.tight_layout(pad=0.3); plt.savefig("figs/fig1_recovery.pdf", bbox_inches="tight"); plt.close()

# --- Fig 2: what Distil's gain is bought with ------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(LNCS_W, 2.05))

# (a) teacher headroom (the purchasable quantity) against the gain it buys
for task, mk, fc in (("1d", "o", "white"), ("7d", "s", "0.45")):
    hx = [(piv(task, s)["A"] - piv(task, s)["B_teacher"]).mean() for s in S]
    gy = [(piv(task, s)["A"] - piv(task, s)["B"]).mean() for s in S]
    ax[0].plot(hx, gy, mk, color="0.0", mfc=fc, mew=1.0, ms=4.5, ls=":", lw=0.9,
               label=rf"$\Delta$={task}")
    for s, x, y in zip(S, hx, gy):
        ax[0].annotate(s, (x, y), textcoords="offset points", xytext=(3, 3), fontsize=6.2)
ax[0].axhline(0, color="0.0", lw=.5)
ax[0].set_xscale("log"); ax[0].grid(alpha=.25, lw=.4)
ax[0].set_xlabel(r"teacher headroom ($\Delta$NE vs. Wait)")
ax[0].set_ylabel(r"Distil gain ($\Delta$NE vs. Wait)")
ax[0].set_title("(a) the gain is bought with headroom")
ax[0].legend(frameon=False, loc="lower right")

# (b) Distil decomposed against the Distil-Stale control
cap, fre = [], []
for s in S:
    p = piv("7d", s)
    cap.append((p["A"] - p["B_nofresh"]).mean()); fre.append((p["B_nofresh"] - p["B"]).mean())
ax[1].bar(S, cap, .55, label="capacity transfer", color="0.75", edgecolor="black", lw=.5)
ax[1].bar(S, fre, .55, bottom=cap, label="fresher covariates", color="0.25",
          edgecolor="black", lw=.5, hatch="///")
ax[1].axhline(0, color="0.0", lw=.5); ax[1].grid(alpha=.25, axis="y", lw=.4)
ax[1].set_ylabel(r"$\Delta$NE vs. Wait"); ax[1].set_title(r"(b) Distil decomposed, $\Delta$=7d")
ax[1].legend(frameon=False, loc="upper left")
plt.tight_layout(pad=0.3); plt.savefig("figs/fig2_penalty_decomp.pdf", bbox_inches="tight")
plt.close()
print("wrote greyscale-safe figs at LNCS text width")
