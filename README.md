# Beyond the Delayed-Feedback Ceiling

ECIR 2027 short-paper draft. Springer LNCS format, prepared with the official template
(`llncs.cls` v2.25, 2026/09/03) supplied with the call.

**Thesis.** Censoring correction is bounded by the information left in a partly observed label —
a property of the delay distribution, which is fixed. Distillation is bounded by the teacher — a
training-time asset, which can be bought. We derive both ceilings, show correction already sits at
its own, show distillation is not subject to it, and show distillation's gain is bought with
teacher headroom. That makes offline teacher scale the one lever whose supply is not fixed by the
corpus, and positions LLM-scale teachers as the continuation of this baseline.

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
| `refs.bib` | bibliography — **seven of nine entries are unverified, see below** |
| `figs/make_figs.py` | regenerates both figures from `data/` |
| `figs/*.pdf` | vector, greyscale, drawn at the true LNCS text width |
| `data/` | the 1,600-run result table and the theory prediction, vendored so the repo stands alone |
| `llncs.cls`, `splncs04.bst` | Springer template files, unmodified |
| `samplepaper.tex`, `readme.txt`, `history.txt`, `fig1.eps` | Springer's own sample, kept for reference |

## Status

- Section 6 (Related Work) is a deliberate stub, marked in the source and budgeted at ≤ 1 page.
- Authors, affiliation, ORCID and acknowledgements are placeholders.

### Length

Springer puts short papers at **6–11 pages**.

| build | pages | note |
|---|---|---|
| CMR fallback (this repo's preview) | 11 | CMR sets looser than the specified Times-based newtx |
| mathptmx probe, `VERIFY` notes stripped | 10 | submission *shape*; body ends p. 9 |

The second row is the honest estimate for the real `newtx` build, leaving roughly one page for
Section 6. **Re-measure on a real newtx build before adding anything further** — the margin is
about one page, not several.

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
| abstract 150–250 words | 237 |

Two overfull hboxes remain in the CMR build (7.7 pt in Sec. 5.1, 5.3 pt in the Sec. 6 stub). Both
are font-metric artefacts of the fallback and should be re-checked, not fixed, against newtx.

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
- **Meta GEM blog post** — URL supplied by the authors; title, venue and date independently
  confirmed by web search on 2026-09-20, but `engineering.fb.com` is unreachable from the drafting
  host so **the page content was not read**. It is cited in Sec. 1 only as an existence pointer to
  large centrally trained industrial ads models, which is what its title supports. Do **not**
  paraphrase GEM's architecture, scale, or any knowledge-transfer mechanism until the page has
  been read.

Every unverified entry carries a `note = {VERIFY: ...}` field. Re-fetch each from its DOI/arXiv ID
and drop the `note` fields before submitting — note that the notes currently render in the
bibliography and inflate it by roughly a third of a page.
