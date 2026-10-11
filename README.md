# One Stream, Many Windows — ECIR short paper

Built from two experiment runs. Nothing in the paper is typed by hand: every number is derived
from the run tables by `paper_numbers.py` and independently re-derived by `check_claims.py`.

## Build

```bash
python3 paper_numbers.py --tex     # -> generated/macros.tex, generated/screen_inline.tex
python3 make_fig.py                # -> figs/gain.pdf
pdflatex main && bibtex main && pdflatex main && pdflatex main
python3 check_claims.py            # 19/19 must pass
```

Current state: **7 pages** (body 6.21, references 0.79), 0 LaTeX errors, 0 overfull boxes,
0 undefined references or citations, 19/19 claims verified.

## Where the data comes from

| Source | Used for |
|---|---|
| `data/v4_all_runs.parquet` (2,050 runs, 14 arms) | every result in Sect. 5 — the three targets against the T13 teacher, 10 seeds on the headline arms |
| `data/v3_all_runs.parquet` (4,850 runs, 37 arms) | the 11-method single-visit screen (Sect. 2) and the ULC comparison (Sect. 5) |

v3 is cited rather than re-run for those two because neither is affected by the teacher swap: a
method's records-per-click is a property of its emission logic, and v4 confirmed the teacher
choice is near-irrelevant to accuracy anyway.

## Two conventions worth knowing before editing

**Macro names must be letters only.** `\cdOneMean7d` parses as `\cdOneMean` followed by a `7`,
which fails with a misleading "Missing \begin{document}". `paper_numbers.py` asserts this.

**Signed and unsigned twins.** Gains are stored signed (`cdSevenMean` = `+1.97`), and every one
has an unsigned twin (`cdSevenMeanAbs` = `1.97`). Use the unsigned form wherever a verb already
carries the direction — "raises it by `\doOneLargeAbs`%" — because the signed form produces
"raises it by -0.51%", which is a double negative. Use the signed form only where a signed
quantity is genuinely being reported, as in "moves the gain by `\teachCdMean` pp".

## Open items before submission

1. **Two references carry placeholder author lists.** `menon2021statistical` and
   `xu2020privileged` render as "Menon, et al." and "Xu, et al."; `refs.bib` has
   `author = {X and others}`. The network is blocked on this host, so these were never
   completed. Fill them from the DOI/arXiv record — **do not invent author lists**. Note also
   that `xu2020privileged` renders a 2019 year against a 2020 key; check which is right.
2. **Author and affiliation** are "Anonymous" in `main.tex`.
3. **Body is 6.21 pages** against a 6-page target. The remaining 0.21 is page-break quantised —
   several rounds of word-shaving moved it only 0.03pp — so closing it means dropping a result
   rather than tightening prose. Candidates, least costly first: the Rule-2 relaxation clause at
   the end of Sect. 5 (~0.06pp), the teacher learning-rate control (~0.08pp, but it is what makes
   "a larger teacher buys nothing" falsifiable), the frozen-teacher sentence (~0.07pp).
4. **Title is provisional.** Alternatives considered: *Collapsing the Join Delay*,
   *The Slower the Funnel, the Less You Should Wait*.

## What the paper claims, and what the data says

`check_claims.py` recomputes each claim from the parquet with its own code path — it does not
import `paper_numbers` — and then greps the rendered PDF for the value, so a claim passes only
if the number is right *and* it reached the page. Claims here have failed before: "beats both in
all 30 configurations" and "the frozen teacher beats Wait everywhere" both came back false under
the T13 teacher, which is why the paper states 28 of 30 and names the two exceptions.
