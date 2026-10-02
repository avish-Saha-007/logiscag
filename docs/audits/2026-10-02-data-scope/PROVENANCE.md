# Provenance: data scope audit (B1b)

Audit artifact for issue "audit: data scope of generator training and
evaluation populations" (B1b, follow-up to the constraint provenance audit),
on branch `audit/data-scope-of-generator-training-and-evaluation-populations`.

This is a **static code-reading audit**, not a data run: no synthesizer was
trained and no CSV was processed to produce `REPORT.md`.

- **Command:** none (manual code review of `privacy_utility_sweep.py`,
  `pipeline/evaluation.py`, `pipeline/metrics.py`, `pipeline/pipeline.py`,
  `pipeline/data.py`, `pipeline/constraints.py`)
- **Git commit audited:** `e3ade5616b6eead1044160ae01a5c3ca2371322d` (tip of
  `main` when the parent audit branch was cut)
- **SDV version:** 1.37.2 (pinned in `pyproject.toml`)
- **Python:** 3.10.21
- **Seeds / config:** n/a (no stochastic run performed)

## Contents

- `REPORT.md` — the data-scope findings table (Q1–Q7) and the STOP-condition
  writeup regarding whether the real corpus schema has promise-date columns.

## Relationship to prior audit

Confirms and sharpens the constraint provenance audit's headline structural
finding (no train/test split exists before generation) by tracing its
concrete consequences through TSTR, DCR, and the MIA audit specifically. Adds
two things the first audit didn't cover: (1) the MIA member/non-member split
is genuinely correct and disjoint, so it is not a new leak; (2) a naming
contradiction bearing on whether KNOWN_ISSUES.md finding 10's real-world
applicability to the actual proprietary corpus is verified. Logged as
KNOWN_ISSUES.md finding 11 (new), finding 12 (new — carrier/service check
scope), and an update note on finding 10.

## Redactions applied to this document and to `REPORT.md`

This audit discusses a proprietary, non-redistributable corpus (see
`DATASHEET.md`). Relative to the original, unversioned copy of this report
(written to the gitignored `outputs/data_scope_audit_2026-10-02/`, which is
untouched and still present for internal reference), the following details
were generalized before committing this copy to version control:

- The corpus's approximate row count (as cited from `paper/paper.md:83`) is
  omitted; `REPORT.md` still points to `paper/paper.md:83` for the figure
  itself.
- `pipeline.data.load_real_dataset`'s literal default fallback filename is
  not quoted; `REPORT.md` instead paraphrases what the name implies and
  points to `pipeline/data.py:267` for the exact string.
- The schema variant lacking promise-date columns is referred to as "the
  no-promise-column schema" rather than by the code's internal name for it.
  (That internal name, `DISSERTATION`, is unchanged in the source — see
  `pipeline/data.py` — and is quoted once in `REPORT.md`, where it appears
  inside a direct quotation of an existing code comment, for traceability.)

No carrier names or data values from the proprietary corpus appear in this
report. The quantile constants `0.5442`/`0.9200` (`CANONICAL_LABEL_DISTRIBUTION`
in `pipeline/data.py`) were reviewed and kept as-is — they are code constants
used across both the public and proprietary data paths, not values derived
from the proprietary corpus itself.
