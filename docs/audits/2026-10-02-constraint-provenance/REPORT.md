# Constraint provenance audit

**Issue:** audit: provenance of data-derived constraint components
**Branch:** `audit/provenance-of-data-derived-constraint-components`
**Date:** 2026-10-02
**Git commit audited:** `e3ade5616b6eead1044160ae01a5c3ca2371322d`
**SDV version:** 1.37.2 (pinned, per `pyproject.toml`)
**Python:** 3.10.21

## Scope and method

Grepped and read every place in `pipeline/`, `logiscag/`, `dataco_adapter.py`, and
`privacy_utility_sweep.py` that builds a constraint, enforcement predicate, or
label-derivation rule from a dataframe (as opposed to a fixed, hardcoded
constant), then traced each one back to its input dataframe and asked: does a
train/test split exist upstream of that dataframe, and if so, is the component
built from the train side only?

**Headline structural finding, which conditions every row below:** there is no
train/test split anywhere in the generation path. `pipeline.pipeline.run_pipeline`
loads one dataframe (`df`), derives features and labels on it, audits it, and
passes the *entire* thing to `train_sdv_models(df, ...)` — the SDV synthesizer is
fit on 100% of the available real rows. `pipeline/evaluation.py` only introduces
train/test splits (`_temporal_holdout_split`, `StratifiedKFold`, `TimeSeriesSplit`)
*after* that point, purely for scoring TRTR/TSTR/SMOTE/transfer-learning utility
downstream of generation. So for any component built from `df` at or before
`train_sdv_models`, "before/after split" literally means "before any split
exists" — there is no train-only dataframe to build it from in the current
architecture. This is a different, and separable, finding from the one genuine
label-cutoff leak below (#1), which doesn't depend on whether the generator
itself is later retrained on a proper split.

This matters for both code paths named in the issue: the public **DataCo**
benchmark (`dataco_adapter.py` → `pipeline.data.phase1_derived_features` →
`pipeline.pipeline.run_pipeline`) and the proprietary **corpus** path, which is
the same `pipeline/data.py` / `pipeline/pipeline.py` code running on a
no-promise-column schema CSV (no `date_promise_*` columns; the code's internal
name for this schema is `DISSERTATION` — see `pipeline/data.py`) via
`load_real_dataset` — there is no separate corpus-only module; the two
datasets diverge only in which branch of `_apply_latest_derived_features`'s
`_has_promise_cols` check fires.

## Findings table

| # | Component | Source dataframe | Built before/after split | Leak risk | Evidence |
|---|---|---|---|---|---|
| 1 | `delay_label` cut-offs (corpus / no-promise-column path only) — quantile thresholds `q1=P54.42`, `q2=P92.00` of `transit_duration_days` | Entire input `df`, every row, computed once at ingest | **Before** — no split exists yet; the same thresholds are then baked into `delay_label` for every row, including rows later assigned to CV test folds / the temporal holdout test tail | **HIGH** (corpus path only) | [`pipeline/data.py:125-134`](../../../pipeline/data.py) (`_derive_delay_labels_from_transit`), called from [`pipeline/data.py:204`](../../../pipeline/data.py) inside `_apply_latest_derived_features`, itself called by `generate_proxy_dataset` ([data.py:264](../../../pipeline/data.py)), `load_real_dataset` ([data.py:272](../../../pipeline/data.py)), and `phase1_derived_features` ([data.py:310](../../../pipeline/data.py)) — i.e. every entry point, always on the full frame |
| 2 | `delay_label` cut-offs (DataCo path) — SLA-breach rule (`sla>=0`→0, `-sla<=2`→1, else 2) | N/A — fixed 2-day business threshold, not fit from data | N/A | **None** | [`pipeline/data.py:137-149`](../../../pipeline/data.py) (`_derive_delay_labels_from_sla_buffer`); taken because `dataco_to_canonical` always derives `date_promise_delivery`/`date_promise_shipment` ([dataco_adapter.py:46-58](../../../dataco_adapter.py)), so `_has_promise_cols` is always `True` for DataCo ([data.py:178-182](../../../pipeline/data.py)) and the quantile branch (#1) is never reached on this dataset |
| 3 | Carrier/service `FixedCombinations` vocabulary (SDV-native `strict` tier) | `train_df = canonical_df[train_cols].copy()` — a *column* subset of the full dataset, not a *row* (train-split) subset | **Before** — there is no row-level split anywhere upstream; `train_df` here means "SDV training columns", not "training rows" | **Latent** — currently a no-op per KNOWN_ISSUES.md finding #1 (SDV silently drops these dict-style constraints), but the vocabulary itself is built from every row the model will ever be scored against, with no held-out set excluded. If finding #1's `sdv.cag` rewrite lands without also introducing a row split, this becomes a live, structural leak (every combo in the "test" evaluation folds is by definition already a seen category) | [`pipeline/pipeline.py:309`](../../../pipeline/pipeline.py), [`pipeline/pipeline.py:359`](../../../pipeline/pipeline.py), [`pipeline/constraints.py:374-381`](../../../pipeline/constraints.py) (`n_combos = train_df[["last_scac","carrier_service_code"]].drop_duplicates()...`) |
| 4 | `build_valid_carrier_combos(real_df)` — R4/R5 referential-integrity reference vocabulary | Would be `real_df`, whichever dataframe a caller supplies | **N/A — never called in any production code path** | **None in practice (dead code in prod)**, but see note below | Defined [`pipeline/constraints.py:27-57`](../../../pipeline/constraints.py); only ever invoked from `tests/test_constraint_catalog.py` (8 call sites) — not from `pipeline/pipeline.py`, `pipeline/tuning.py`, or `logiscag/constraints/engine.py`. `audit_constraints()` is called at [`pipeline/pipeline.py:563`](../../../pipeline/pipeline.py) with no `valid_combos` argument, so R4/R5 always runs the weaker presence-only fallback in real runs. **Note:** this means the "fix" KNOWN_ISSUES.md finding #4 describes is correct in isolation but is not wired into any path that would ever call it with data of any provenance, real or synthetic — it is untested-in-production dead code, not a leak |
| 5 | SCAC/service-code vocabularies (`SCAC_VOCABULARY`, `SERVICE_CODE_VOCABULARY`) | None — hardcoded Python literals | N/A | **None** | [`pipeline/constraints.py:8-9`](../../../pipeline/constraints.py) |
| 6 | Carrier non-delivery calendar (`SCAC_NON_DELIVERY_WEEKDAYS`, R6) | None — hardcoded Python literal | N/A | **None** | [`pipeline/constraints.py:10-17`](../../../pipeline/constraints.py) |
| 7 | Clock-skew tolerance (`CAPTURE_LATENCY_TOLERANCE_DAYS`) | None — hardcoded constant (~1 second) | N/A | **None** | [`pipeline/constraints.py:24`](../../../pipeline/constraints.py) |
| 8 | Non-negativity thresholds (`_NONNEG_SCALARS`) | None — hardcoded constants (all `0.0` / `0.01`) | N/A | **None** | [`pipeline/constraints.py:307-313`](../../../pipeline/constraints.py) |
| 9 | `qualitative_sanity_check`'s "known categorical value" vocabulary | `rdf` = whatever `real_df` the caller passes; in production, the full `df` (same no-split dataframe as everywhere else) | **Before** (same structural point as the headline finding) | **Low** — this is a QA/reporting metric (`unknown_categorical_value` count in a sanity-check dict), not an enforcement predicate; it never filters, rejects, or trains anything. Its effect is to make the synthetic-data QA report structurally unable to ever flag an "unseen category" problem, since the comparison vocabulary was built from the same full dataset the generator was trained on | [`pipeline/metrics.py:488`](../../../pipeline/metrics.py), [`pipeline/metrics.py:583-585`](../../../pipeline/metrics.py); called from [`pipeline/pipeline.py:724`](../../../pipeline/pipeline.py) as `qualitative_sanity_check(df, sdf, ...)` with `df` the full dataset |
| 10 | `qualitative_sanity_check`'s synthetic-side label/transit consistency quantiles (`q1=P54.42`, `q2=P92`) | `sdf` = the synthetic dataframe being checked, not a real/train dataframe | N/A — self-referential, checks synthetic data against its own distribution | **None** (not a train/test provenance issue; flagged here only because it duplicates finding #1's quantile values and could be mistaken for the same bug) | [`pipeline/metrics.py:565-570`](../../../pipeline/metrics.py), [`pipeline/metrics.py:622-627`](../../../pipeline/metrics.py) |
| 11 | `distance_km` median-fill for unresolvable ZIPs | Whatever dataframe is passed to `_compute_distance_km` — in production, the full `df` | Before (same structural point) | **Low** — this is feature imputation, not a constraint/enforcement predicate; flagged for completeness since it is a data-derived value computed over the whole frame, but it is out of the issue's explicit list (carrier-service sets, vocabularies, thresholds, calendars, clock-skew, label cut-offs) | [`pipeline/data.py:48-82`](../../../pipeline/data.py), specifically `median_dist = dist.median()` at [`pipeline/data.py:77`](../../../pipeline/data.py) |
| 12 | MIA audit member/non-member split (`_member_split`) | `real_df` | This *is* a genuine train/test-style split, done correctly for its purpose (trains the generator on `members` only, tests attack success against `non_members`) | **None** — included here only to document that one real split does exist in the codebase, scoped to the privacy-audit harness, and it is not what the other findings are about | [`privacy_utility_sweep.py:121-127`](../../../privacy_utility_sweep.py), [`privacy_utility_sweep.py:183`](../../../privacy_utility_sweep.py) |

## STOP condition triggered

Per the task instructions: **finding #1 uses information derived from what will
later be test data.** Specifically, on the corpus (proprietary, no-promise-column)
path only — not on the public DataCo benchmark, which uses a fixed 2-day
threshold instead (finding #2) — the three-class `delay_label` is assigned by
computing the 54.42th/92.00th percentile of `transit_duration_days` over *every*
row in the dataset, before any split exists, and those same thresholds then
determine the ground-truth label of every row that later lands in a CV test
fold or temporal holdout tail in `pipeline/evaluation.py`.

**Consequence, if this corpus path is used for a train/test evaluation claim:**
the label boundaries are fit on data that includes the evaluation set. This is
not a leaked *feature* (no leakage column crosses into `FEATURE_COLS`;
`run_preflight_validator`'s `leakage_columns` check, [`pipeline/data.py:495-514`](../../../pipeline/data.py),
guards against that separately and is unaffected) — it is a leaked *label
boundary*: the cut-points `q1`/`q2` are a function of the full-sample
distribution of `transit_duration_days`, including whichever rows end up in
whatever test fold is used to report TRTR/TSTR/SMOTE/transfer-learning numbers
downstream. A model's reported performance on "held-out" rows is therefore
partly validated against label thresholds that used those same rows' values
to decide where the class boundaries fall.

**I am stopping here, as instructed, without changing this behavior.** Two
things for you to verify by hand before any fix is written:

1. Whether this matters for any number currently reported anywhere (paper,
   README, CHANGES.md) — on the evidence read so far, **no published number in
   this repo's committed artifacts uses the corpus / no-promise-column
   branch**: DataCo (the only benchmark with committed, cited results) always
   takes the SLA-buffer fixed-threshold path (#2), which does not have this
   bug. The proprietary corpus is mentioned only in `paper/paper.md`'s
   aggregate-results sentence, with no committed run artifact in this repo to
   check (`DATASHEET.md:16` notes the corpus data itself is not
   redistributable). If the proprietary corpus run that produced that paper
   sentence was generated through this code path, its numbers may be
   optimistic in a way that is not quantified here — worth independently
   re-deriving label thresholds on a train-only split and re-running, if that
   corpus and its original run artifacts are still available to you.
2. Whether `build_valid_carrier_combos` (finding #4) was intended to be wired
   into `pipeline.pipeline.run_pipeline`'s `audit_constraints()` call and
   simply never was — right now it is fully-tested, documented as "the fix"
   in KNOWN_ISSUES.md finding #4's resolution, and never executed outside the
   test suite.

## What was NOT changed

No behavior, output, constraint, or label-derivation logic was modified. This
commit adds only this report and one new, independent test file (see below).
Existing results under `outputs/finding5_tvae_verify/` are untouched.
