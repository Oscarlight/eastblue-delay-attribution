# One Stream, Many Windows — ECIR 2027 short paper

**This directory is canonical.** It is also the repository that carries the GitHub remote
(`Oscarlight/eastblue-delay-attribution`), so this is where submission happens. Earlier drafts
live in `../archive/`; do not edit them.

Nothing in the paper is typed by hand: every number derives from the two run tables via
`paper_numbers.py`, and `check_claims.py` re-derives each claim independently — it does not
import `paper_numbers` — then greps the rendered PDF for the value, so a claim passes only if
the number is right *and* it reached the page.

## Build and verify

```bash
python3 paper_numbers.py --tex     # -> generated/macros.tex, screen_inline.tex, tab_baselines.tex
python3 make_fig.py                # -> figs/gain.pdf
pdflatex main && bibtex main && pdflatex main && pdflatex main
python3 check_claims.py            # must report 27/27
```

Current state: **8 pages** (body 6.53, references ~1.0), 0 LaTeX errors, 0 overfull boxes,
0 undefined references, **27/27 claims verified**.

## Evidence

| Source | Contributes |
|---|---|
| `data/v3_all_runs.parquet` — 4,850 runs, 37 arms | the 11-method single-visit screen (Sect. 2) and every published-baseline row in Table 1 |
| `data/v4_all_runs.parquet` — 2,050 runs, 14 arms | the three targets against the T13 teacher, 10 seeds on the headline arms (Sect. 5) |

**Why the two campaigns share an axis.** v4's harness is a frozen copy of v3's, and `Wait`
instantiates no teacher — so nothing v4 changed can reach it. It reproduces v3 to *zero*
difference across all 150 shared cells, under both source revisions. Every arm is reported as a
gain against `Wait` within its own run, so that bit-identical baseline is what licenses putting
both on one axis. This is stated in Sect. 4 and re-checked by `check_claims.py`.

## Two conventions before editing

**Macro names must be letters only.** `\cdOneMean7d` parses as `\cdOneMean` followed by `7` and
fails with a misleading "Missing \begin{document}". `paper_numbers.py` asserts this.

**Signed and unsigned twins.** Gains are stored signed (`cdSevenMean` = `+1.97`) and every one
has an unsigned twin (`cdSevenMeanAbs` = `1.97`). Use the unsigned form wherever a verb already
carries direction — "raises it by `\doOneLargeAbs`%" — because the signed form yields "raises it
by -0.51%". Use the signed form only for genuinely signed quantities, as in "moves the gain by
`\teachCdMean` pp".

## Open before submission

1. **`menon2021statistical` author list.** Still `and others`. Venue is verified (PMLR v139,
   ICML 2021, key `menon21a`); the preprint is arXiv 2005.10419 under a *different* title,
   "Why distillation helps: a statistical perspective". Not readable from the dev host — the
   Meta paper index lacks it, and WebFetch, the knowledge loader and external DNS are all
   closed. Open `https://proceedings.mlr.press/v139/menon21a.html` in a browser and paste its
   BibTeX block. **Do not invent the author list.**
2. **Author block.** `main.tex` is `Anonymous`; decide the author list, affiliations, and
   whether ECIR 2027 is double-blind.
3. **Page budget.** Body is 6.53 pp. If the limit is 6, this is now a content decision — the
   last several prose trims moved it 0.00 pp because page breaks are quantised. Cheapest cuts
   first: the `TWICE` table row and its sentence (~0.12 pp), the teacher learning-rate control
   (~0.10 pp, but it is what makes the no-larger-teacher claim falsifiable), the abstract from
   306 to ~250 words (~0.12 pp). **Confirm the actual ECIR body-page limit first.**
4. **Citation audit** (`inno-reference-audit`) once item 1 lands.

## Pushing

`git push` to github.com fails from inside the agent on this host: fwdproxy filters on agent
identity (`403 … not been allowlisted in filter {agent_id: agent:claude_code}`). Staging and
committing work; run the push from a plain terminal.
