# How Much of the Delayed-Feedback Penalty Is Recoverable?

ECIR 2027 short-paper draft. Springer LNCS format, prepared with the official template
(`llncs.cls` v2.25, 2026/09/03) supplied with the call.

## Build

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Requires `newtxtext` / `newtxmath`, which the supplied template loads. Overleaf and a full TeX
Live both have them. If they are missing, comment those two lines for a CMR fallback build —
`preview-cmr-fallback.pdf` in this repository was produced that way and is **not** the submission
PDF.

## Layout

| path | contents |
|---|---|
| `main.tex` | the paper |
| `refs.bib` | bibliography — **six of eight entries are unverified, see below** |
| `figs/make_figs.py` | regenerates both figures from `data/` |
| `figs/*.pdf` | vector, greyscale, drawn at the true LNCS text width |
| `data/` | the 1,600-run result table and the theory prediction, vendored so the repo stands alone |
| `llncs.cls`, `splncs04.bst` | Springer template files, unmodified |
| `samplepaper.tex`, `readme.txt`, `history.txt`, `fig1.eps` | Springer's own sample, kept for reference |

## Status

- Sections 1 (Introduction) and 6 (Related Work) are deliberate stubs, marked in the source.
- Currently **8 pages**. Springer's instructions put short papers at **6–11 pages**, so the two
  stub sections can be written without a length problem.
- Authors, affiliation, ORCID and acknowledgements are placeholders.

## Springer compliance (checked against *Instructions for Authors*)

| requirement | status |
|---|---|
| §4.2 short paper 6–11 pages | 8 pages |
| §4.3 template fonts | `newtxtext`/`newtxmath` as shipped |
| §4.5 no colour in text, tables, equations | verified absent |
| §4.5 figures legible in black and white | greyscale, distinct markers and hatching |
| §4.5 figures vector, labels ≥ 6 pt | vector PDF at true text width, smallest label 6.8 pt |
| §4.5 tables editable, captions above | real `tabular`, captions above |
| §4.5 figure captions below, no terminal period | as required |
| §4.1 only two heading levels numbered | LNCS default |
| §4.1 propositions consecutive, no section counter | class `proposition`, no `\newtheorem` |
| §4 avoid self-defined environments | none defined |
| §4.6 equations numbered consecutively | 4 numbered equations |
| §4.9 citations by number | `splncs04` |
| abstract 150–250 words | 192 |

Not yet done: **alternative text for both figures** (§3, EU Accessibility Act). Springer collects
this from the PC chairs or at proofing stage.

## ⚠ Citations need verification before submission

`arXiv`, CrossRef and general web access were blocked on the machine this draft was prepared on,
so **no BibTeX entry was machine-fetched**. Entries were transcribed from this project's own
reference bank (`Survey/reports/delayed_label_vs_distillation_lit_review.md` §10.1), which records
a verification status per work:

- **Full text read** (safe to paraphrase): Ktena et al. 2019; Zhao et al. 2023.
- **Title and venue confirmed by search, content NOT verified** — Chapelle 2014; Yasui et al. 2020;
  Yang et al. 2021; Gu et al. 2021; Chen et al. 2022; Yang et al. 2022.

Every unverified entry carries a `note = {VERIFY: ...}` field. The paper cites them but never
paraphrases their mechanisms as established. Re-fetch each from its DOI/arXiv ID and drop the
`note` fields before submitting.
