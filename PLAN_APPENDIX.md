# Plan: per-method technical appendix + three-axis main paper

Status written 2026-09-25. Supersedes nothing; the 11-page main paper stands and gains a window
axis plus an appendix.

## 0. Premise correction

The request said the paper "only reports 1d". It is the other way round: every table and figure in
`main.tex` is `_7d`, and 1d appears only in a one-sentence *Second window* note. The ask — make both
windows first class — is right either way, and is Phase 6 below.

## 1. What the data audit found

| | status |
|---|---|
| **7d ladder** | complete: 14 arms x 5 rungs x 5 seeds |
| **1d ladder** | S1 and S3 missing for every arm; `ES-DFM`/`DEFUSE`/`DDFM` missing at all rungs; `Soft` at 4 of 5; `Correct(tuned)` at S0 |
| **NE curves** | **do not exist anywhere.** `stream_one` did `rows, _ = run_stream(m)` then `aggregate(rows)` and discarded the per-cohort rows |
| **Architecture** | introspectable: `CAPACITY_ROWS = {S0:36, S1:64, S2:160, S3:896, S4:131072}`, emb_dim 8, capacity varied only by categorical table rows |

Consequences: NE curves need a code change (done, Phase 1) **plus** re-runs; the 1d ladder needs a
full re-run, not a patch.

## 2. Why per-method reports are not 12x duplication

The served trunk is shared, but three things genuinely differ per method and are the substance of
sections 1-2 of each report:

* **heads / auxiliaries** — DFM a hazard head, FSIW two IW classifiers, TWICE a delay head, DISTILL
  a two-head teacher, Soft a single distilled read-out;
* **which records the method writes** — the audit already measured 1.00 (Fresh, Wait, FSIW, ULC,
  TWICE), 1.08 (Vanilla, ES-DFM, DEFUSE), 1.89 (DDFM), 2.00 (PI), 3.00 (FTP), 2.90 (IF-DFM). The
  training set is therefore a different object per method;
* **what label each record carries** — Fresh's positive rate is 0.107 against Oracle's 0.185 at
  v=7d, so the label statistics differ even when the record count does not.

Shared material (corpus, streaming protocol, S0-S4 trunk, timeline conventions) is written **once**
in Appendix A and cross-referenced, rather than repeated twelve times.

## 3. Deliverables

```
appendix/common.tex        A  corpus, protocol, S0-S4 trunk + computation graph,
                              record-timing diagram, label statistics at 1d and 7d
appendix/m_<method>.tex    B  one per method, four sections each:
                              1 architecture delta + computation graph + param counts S0-S4
                              2 training data: records emitted, timestamps, label definitions,
                                label statistics (true/false counts and rates) at 1d and 7d,
                                timeline diagram for train and eval
                              3 experiment setups: LR grid, selected LR per rung per window,
                                method-specific hyperparameters, NE curves per rung per window
                              4 results: full metric table S0-S4 at both windows
```

Methods: Fresh, Reweight, Wait, Twice, Correct, Distil, Soft, Vanilla, ES-DFM, DEFUSE, DDFM, Oracle.
DISTILL additionally documents the teacher backbone.

## 4. Phases

| phase | work | cost |
|---|---|---|
| 1 | persist per-cohort metrics (`--curves` -> `{out}_curve.csv`) | **done, verified** |
| 2 | full 1d ladder: 12 arms + 2 anchors x 5 rungs x 5 seeds, curves on | **running**, ~1-2 h |
| 3 | 7d curve runs, seed 0, LRs pinned from existing runs so no LR sweep | ~30 min |
| 4 | label + data statistics and timeline diagrams, from the corpus (no GPU) | cheap |
| 5 | architecture introspection -> per-capacity shape/param tables + TikZ graphs | cheap |
| 6 | main paper: three-axis presentation (method x capacity x window) | writing |
| 7 | write the twelve method reports + Appendix A | writing |
| 8 | quality gate below | automated + read |

## 5. Quality gate

Every item is checkable, and the build fails loudly rather than silently degrading.

| id | gate |
|---|---|
| G1 | **Provenance.** Every number in paper and appendix is emitted by `build_artifacts.py` from a run CSV or a corpus computation. Numeric literals in prose are cross-checked against a generated facts file. |
| G2 | **Seeds.** Headline results are 5 seeds; any cell with fewer is either re-run or explicitly labelled. Curves are declared seed 0. |
| G3 | **Origin matching.** Arms compared in one table come from runs whose Oracle agrees to <1e-4 at that (task, rung). Checked programmatically. |
| G4 | **Record counts.** Per-method record counts in the appendix equal the audit table in the main paper. |
| G5 | **Label arithmetic.** positives + negatives = N; the observed-positive rate equals base x F(o) within tolerance at each window. |
| G6 | **Architecture.** Reported parameter counts equal `sum(p.numel())` on the instantiated model, not arithmetic in prose. |
| G7 | **Completeness.** Both windows present for every arm x rung in the headline tables; no `--` in a headline cell. |
| G8 | **Build.** 0 LaTeX errors, 0 undefined references, 0 overfull boxes >5pt, every `\include`d file present. |
| G9 | **Integrity.** No fabricated citations; IP boundary clean (no internal system names). |
| G10 | **Statistics.** Every significance claim carries a paired test over matched seeds with its p-value. |

## 6. Known open items

* `abl5_S2` (five-seed estimator ablation) still running; the Sect. 4 estimator claims rest on
  three seeds until it lands.
* Two Tier-2 references have incomplete author lists and cannot be completed without network.
