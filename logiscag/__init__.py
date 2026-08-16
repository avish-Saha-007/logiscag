"""
LogiSCAG: a constraint library, CAG (Constraint-Augmented Generation) wrapper,
four-pillar evaluation harness, and dataset adapters for operationally-valid
synthetic supply-chain data.

``logiscag`` is the stable public API surface for this project. The
implementation lives in ``pipeline/`` (constraints, metrics, evaluation,
generation), ``privacy_utility_sweep.py`` (the sweep harness), and
``dataco_adapter.py`` (the DataCo benchmark adapter). The split is deliberate:
the public API can stay stable while the internals evolve, and the declarative
catalog can describe rules without re-implementing them. This module
re-exports rather than duplicating logic -- see README.md ("Architecture") for
the rationale and the trade-offs.

    import logiscag
    real_df = logiscag.adapters.dataco.dataco_to_canonical(
        logiscag.adapters.dataco.load_dataco("DataCoSupplyChainDataset.csv"))
    synths = logiscag.cag.generate(real_df, level="strict+reject", architectures=["CopulaGAN"])

KNOWN CODE<->PAPER GAPS (flagged per the paper's own appendices; not silently
fixed -- see KNOWN_ISSUES.md for full detail and recommended resolutions):

0. RESOLVED 2026-08-16 (confirmed open 2026-07-27, now closed by implementing
   the missing check): the paper's Appendix C specifies eight rules (R1-R8) and
   `audit_constraints` now emits EIGHT executable checks. Previously it emitted
   seven, because the non-negativity half of the paper's R4
   (`promised_transit_days >= 0`) had no audit check. It now has its own check
   (audit key `RPT_non_negative_promised_transit`) and its own catalog entry
   (`R4_promised_transit_non_negative`). R4's ordering half remains folded into
   `R1_temporal_order`, so the paper's R4 is the one rule split across two
   checks.

   Note the 2026-07-27 wording of this item ("checked nowhere -- not in
   `audit_constraints`, not in `build_sdv_constraints`, not in
   `cag_rejection_filter`") was wrong about `build_sdv_constraints`: that
   function has always emitted `promised_transit_days >= 0` as a
   `ScalarInequality` at the `moderate`/`strict` tiers via `_NONNEG_SCALARS`.
   Per finding 6 below, those dict-style constraints are silently dropped by the
   installed SDV, so the rule was declared-but-inert there rather than absent.
   Genuinely absent from `audit_constraints` (now fixed) and from
   `cag_rejection_filter` (still absent, out of scope).

   Scope: audit layer only, matching the R3/R5 fixes below. The rule counts
   toward `total_hard_violations` but nothing rejects or constrains on it.
   Verified a no-op on DataCo by measurement, not assumption: 0 violations
   across all 180,519 canonical rows, for two independent reasons -- the
   adapter clips the scheduled-days offset at 0, and DataCo's raw
   scheduled-days column holds only {0, 1, 2, 4} to begin with. So no published
   number moves. Note 9,737 real rows sit exactly on the 0 boundary, so the
   `>= 0` predicate (per the paper) matters: a strict `> 0` would falsely flag
   all of them.

   Three numbering schemes remain in play (paper Appendix C, catalog ids, audit
   keys) and they still do not agree; README.md carries the mapping table, and
   it is the authoritative decoder. In particular `R4_referential_integrity`
   (audit key) is the paper's R5, and `R4b_...` is the paper's R4 -- not a
   sub-rule of `R4_referential_integrity`. See KNOWN_ISSUES.md finding 7.

1. Paper Appendices D and E (adapter spec/commands, metric definitions incl.
   the BUS formula) are listed in the paper's appendix index but their bodies
   are not yet written in the draft reviewed for this package. BUS was
   verified against the paper's prose (Sec 4.2, 3:1 minority-class false-
   negative weighting) and matches pipeline.metrics.business_utility_score's
   default fn_weight=3.0; the adapter mapping was verified against Sec 3.4 /
   the Datasheet's "Preprocessing" section and matches dataco_adapter.py.

2. Sec 4.2 cites "derivation in Appendix F" for BUS, but the paper's own
   appendix index lists BUS under "E. Metric definitions" (F is described as
   "per-strictness full tables") -- an internal citation inconsistency in the
   paper, independent of the code.

3. RESOLVED 2026-06-30 (was: Appendix C lists R3 as hard; the code reported
   it under an "_info"-suffixed key, excluded from `total_hard_violations`).
   `audit_constraints` now treats R3 as a genuine hard rule, with a named
   `CAPTURE_LATENCY_TOLERANCE_DAYS` constant (~1 second) absorbing the same
   OMS/DPE clock-skew noise R1 already excludes a sub-check for. Verified a
   no-op on DataCo by construction: `capture_latency_days` is identically 0
   for all 180,519 real rows. See KNOWN_ISSUES.md finding 4.

4. RESOLVED 2026-06-30 (was: Appendix C's R5 specifies true combination-
   membership referential integrity, `not((scac, svc) not in KnownCombos)`;
   the code only checked non-null presence). `audit_constraints` now accepts
   an optional `valid_combos` reference vocabulary (see
   `build_valid_carrier_combos(real_df)`) and checks genuine combination
   membership when supplied; falls back to the old presence-only check with
   a `UserWarning` when omitted, so the weaker path is never silent. Verified
   a structural (not just empirical) no-op on DataCo: `last_scac` (4 values)
   x `carrier_service_code` (3 values) is a full, saturated 12-combination
   cross-product in the real data, so any SDV-sampled synthetic pair is
   guaranteed to be a member. `cag_rejection_filter` and
   `build_sdv_constraints`'s SDV-native tiers are unchanged -- this was an
   audit/reporting-layer fix only. See KNOWN_ISSUES.md finding 4.

5. Sec 5.3 cites TVAE's DCR-vs-strictness significance as "p = 0.49, 0.84".
   Recomputing none-vs-strict+reject from the corrected (resample-to-target,
   not bootstrap-padded) data -- the methodology the paper itself claims in
   Sec 3.2 -- gives p = 0.98, not 0.84. The qualitative conclusion ("no
   significant response") is unchanged; the cited figure should be
   regenerated from the corrected pipeline before submission.

All other cross-checked figures (Table 1 headline, the CTGAN minority-rate
range, the rejection-overhead numbers, the label distribution, the ladder
collapse on DataCo, the label-SLA derivation formula) matched the code
exactly.

6. KNOWN ISSUE (see KNOWN_ISSUES.md for full detail): on the SDV version this
   package currently resolves to (unpinned -> 1.37.2 as of this writing), the
   middle ladder tiers (temporal/moderate/strict, all with `reject: False`)
   may not be enforcing anything at the SDV level. SDV deprecated dict-style
   constraints; `build_sdv_constraints()` still emits the pre-deprecation
   dict format, which SDV's `add_constraints()` silently filters to an empty
   list with a FutureWarning. The only constraint-enforcement mechanism
   confirmed to still work end-to-end is the post-hoc, non-SDV
   `cag_rejection_filter` used at the `strict+reject` level (which itself
   only checks carrier/service presence, not the combination-membership
   check described in finding 4 above). This is flagged, not fixed, here --
   see KNOWN_ISSUES.md for the verification steps run and the recommended
   follow-up.
"""

from . import constraints
from . import cag
from . import evaluation
from . import adapters

__version__ = "0.1.0"

__all__ = ["constraints", "cag", "evaluation", "adapters", "__version__"]
