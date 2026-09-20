# Decoupling the Attribution Window

ECIR 2027 short-paper draft. Springer LNCS format, prepared with the official template
(`llncs.cls` v2.25, 2026/09/03) supplied with the call.

**Logic chain.** Ads retrieval scores O(1e6) candidates in O(10) ms, so one small model must serve
every advertiser objective. Objectives carry different conversion attribution windows, and a hard
label `1[d<=D]` is knowable only after its window closes — so a consolidated model's shared
training window is forced to `T - max_k D_k`, pinning every fast objective to the slowest one's
clock. A teacher's soft label has no attribution window. Supervising with a foundation model
therefore *decouples the training window from the attribution window* rather than compensating for
the coupling. Ranking is by price (paced bid x predicted rate), so calibration is part of the
objective, not a diagnostic.

**What the paper tests.** The delayed-label half, on Criteo-DFD: K=1, two attribution windows,
five serving-scale capacities, twenty rolling origins, five strategies, 1,600 runs. Two ceilings —
correction is capped by Fisher information surviving in a censored label (derived, validated to
0.02); distillation is not, and its gain is governed by teacher headroom. Plus: calibration
transfers from teacher to student faithfully, instability included.

**Scope.** Multi-objective consolidation and sparsity imbalance are *motivation*, not experiments —
Criteo has one conversion event. Stated plainly in Sec. 8.

**IP boundary.** The industrial motivation rests only on the public GEM engineering post and
textbook auction mechanics. No internal system names, roadmaps, architectures or numbers. Verified
absent by grep; keep it that way.

## Strategy names

The paper uses names, never letters. `A`–`E` are **internal** experiment-tree identifiers and must
not appear in the manuscript.

| paper name | internal | training set | target |
|---|---|---|---|
| Oracle | `oracle` | all rows | true label (infeasible at the origin) |
| Wait | `A` | mature rows | true label |
| Distil | `B` | all rows | teacher's prediction |
| Distil-Stale | `B_nofresh` | mature rows | teacher's prediction (capacity-only control) |
| Correct | `C` | all rows | observed label, censoring-corrected loss |
| Band | `D` | nested windows | disjoint delay bands, summed |
| Reweight | `E` | mature rows | true label, density-ratio weighted |

## Build

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Requires `newtxtext` / `newtxmath`, which the supplied template loads. Overleaf and a full TeX Live
both have them. If they are missing, comment those two lines for a CMR fallback build —
`preview-cmr-fallback.pdf` was produced that way and is **not** the submission PDF.

## Layout

| path | contents |
|---|---|
| `main.tex` | the paper |
| `refs.bib` | bibliography — **six of nine entries are unverified, see below** |
| `figs/make_figs.py` | regenerates both figures from `data/` |
| `figs/*.pdf` | vector, greyscale, drawn at the true LNCS text width |
| `data/` | the 1,600-run result table and the theory prediction, vendored so the repo stands alone |
| `llncs.cls`, `splncs04.bst` | Springer template files, unmodified |
| `samplepaper.tex`, `readme.txt`, `history.txt`, `fig1.eps` | Springer's own sample, kept for reference |

## Status

- Section 7 (Related Work) is a deliberate stub, marked in the source. See the length note.
- Authors, affiliation, ORCID and acknowledgements are placeholders.

### Length

Springer puts short papers at **6–11 pages**.

| build | pages | note |
|---|---|---|
| CMR fallback (this repo's preview) | 12 | CMR sets looser than the specified Times-based newtx |
| mathptmx probe, `VERIFY` notes stripped | 11 | submission *shape*; references begin on p. 10 |

**Related Work (Sec. 7) has roughly 0.4 page of room, not more.** The submission-shape build is
already at the 11-page limit with that section still a stub. Write it as three compact
sentence-groups over the existing eight citations; anything longer needs a compensating cut
elsewhere. Re-measure on a real newtx build before adding.

## Springer compliance (checked against *Instructions for Authors*)

| requirement | status |
|---|---|
| §4.2 short paper 6–11 pages | see Length above |
| §4.3 template fonts | `newtxtext`/`newtxmath` as shipped |
| §4.5 no colour in text, tables, equations | verified absent |
| §4.5 figures legible in black and white | greyscale, distinct markers and hatching |
| §4.5 figures vector, labels ≥ 6 pt | vector PDF at true text width; 0 non-grey colour operators in either figure, verified by decompressing the PDF streams |
| §4.5 tables editable, captions above | real `tabular`, captions above |
| §4.5 figure captions below, no terminal period | as required |
| §4.1 only two heading levels numbered | LNCS default |
| §4.1 propositions consecutive, no section counter | class `proposition`, no `\newtheorem` |
| §4 avoid self-defined environments | none defined |
| §4.6 equations numbered consecutively | 4 numbered equations |
| §4.9 citations by number | `splncs04` |
| abstract 150–250 words | 222 |

Five overfull hboxes remain in the CMR build. They are font-metric artefacts of the fallback and
should be re-checked, not fixed, against newtx.

Not yet done: **alternative text for both figures** (§3, EU Accessibility Act). Springer collects
this from the PC chairs or at proofing stage.

## ⚠ Citations need verification before submission

`arXiv` and CrossRef access were blocked on the machine this draft was prepared on, so **no BibTeX
entry was machine-fetched**. Entries were transcribed from this project's own reference bank
(`Survey/reports/delayed_label_vs_distillation_lit_review.md` §10.1), which records a verification
status per work:

- **Full text read** (safe to paraphrase): Ktena et al. 2019; Zhao et al. 2023.
- **Title and venue confirmed by search, content NOT verified** — Chapelle 2014; Yasui et al. 2020;
  Yang et al. 2021; Gu et al. 2021; Chen et al. 2022; Yang et al. 2022.
- **Meta GEM post — now fully verified.** The authors supplied the page HTML in-project; it was
  read in full on 2026-09-20, and title/venue/date were separately confirmed by web search. Every
  fact the paper draws from it (LLM-scale FM on thousands of GPUs; post-training transfer to
  hundreds of vertical models via distillation, representation learning, parameter sharing; the
  "stale supervision" problem and the teacher-refinement component; latency-sensitive vertical
  models; 5%/3% conversion lifts) appears verbatim on the page. No `VERIFY` note; safe to
  paraphrase.

Every unverified entry carries a `note = {VERIFY: ...}` field. Re-fetch each from its DOI/arXiv ID
and drop the `note` fields before submitting — note that the notes currently render in the
bibliography and inflate it by roughly a third of a page.
