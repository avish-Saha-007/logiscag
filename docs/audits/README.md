# Audit records

This directory versions the project's internal audit reports, which were
previously written only to the gitignored `outputs/` directory (run outputs
are not versioned by convention, but these reports are not run outputs — they
are read-only code-provenance investigations with no stochastic or
data-dependent content, so they belong in source control). Each report here
is a point-in-time investigation: no audit in this directory changes
pipeline behavior. Each pair is `REPORT.md` (the findings) and
`PROVENANCE.md` (how/when it was produced — commit, environment, scope).

| Date | Title | Branch | Related KNOWN_ISSUES.md findings |
|---|---|---|---|
| 2026-10-02 | Constraint provenance audit (B1) | `audit/provenance-of-data-derived-constraint-components` | #10 (opened; status wording superseded by B1b, see #10's update note) |
| 2026-10-02 | Data scope audit (B1b) | `audit/data-scope-of-generator-training-and-evaluation-populations` (stacked on B1) | #10 (update note), #11, #12 |

See each audit's `REPORT.md` for the full findings table and STOP conditions,
and `PROVENANCE.md` for exact scope/commit/environment. Some details from the
proprietary corpus referenced by these audits (its exact row count, default
filename, and the code's internal schema-variant name) have been redacted
from both reports in favor of neutral wording; see each `PROVENANCE.md` for a
note on what was redacted and why.
