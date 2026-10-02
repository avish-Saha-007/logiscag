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
| 2026-10-02 | Constraint provenance audit (B1) | `audit/provenance-of-data-derived-constraint-components` | #10 |

See this audit's `REPORT.md` for the full findings table and STOP condition,
and `PROVENANCE.md` for exact scope/commit/environment. One detail from the
proprietary corpus referenced by this audit (the code's internal
schema-variant name) has been generalized in favor of neutral wording; see
`PROVENANCE.md` for what was changed and why.
