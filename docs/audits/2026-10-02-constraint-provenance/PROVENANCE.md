# Provenance: constraint provenance audit (B1)

Audit artifact for issue "audit: provenance of data-derived constraint
components" (branch `audit/provenance-of-data-derived-constraint-components`).

This is a **static code-reading audit**, not a data run: no synthesizer was
trained and no CSV was processed to produce `REPORT.md`. Recorded here anyway,
per repo convention, so provenance is traceable:

- **Command:** none (manual code review — `grep`/`read` of `pipeline/`,
  `logiscag/`, `dataco_adapter.py`, `privacy_utility_sweep.py`)
- **Git commit audited:** `e3ade5616b6eead1044160ae01a5c3ca2371322d` (tip of
  `main` at the time this branch was cut)
- **SDV version:** 1.37.2 (pinned in `pyproject.toml`)
- **Python:** 3.10.21
- **Seeds:** n/a (no stochastic run performed)
- **Config:** n/a

## Contents

- `REPORT.md` — the provenance-audit findings table and STOP-condition writeup.

## Companion test

`tests/test_constraint_provenance.py` (same branch) adds a regression test
constructing the constraint/label components from a toy train/test split and
asserting no test-only category or value influences them. It is marked
`xfail` for the one component (`delay_label` quantile cut-offs on the
no-promise-column schema) that the current code fails, with a reason
referencing this issue.

## Redactions applied to this document and to `REPORT.md`

This audit discusses a proprietary, non-redistributable corpus (see
`DATASHEET.md`). Relative to the original, unversioned copy of this report
(written to the gitignored `outputs/provenance_audit_2026-10-02/`, which is
untouched and still present for internal reference), the following details
were generalized before committing this copy to version control:

- The schema variant lacking promise-date columns is referred to as "the
  no-promise-column schema" rather than by the code's internal name for it.
  (That internal name, `DISSERTATION`, is unchanged in the source — see
  `pipeline/data.py` — and is noted once in `REPORT.md` for traceability.)

No row counts, filenames, carrier names, or data values from the proprietary
corpus appear in this report; none needed redaction.
