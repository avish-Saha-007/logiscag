# Strictness-ladder + sweep harness — change log

## v0.2.0 — 2026-08-16

- Removed 6 files not meant for public release: `calibration_batch.py`,
  `config.yaml`, `dataco_combine_results.py`, `dataco_postprocess.sh`,
  `dataco_relaunch.sh` (orchestration/debugging scripts from producing paper
  results), and `artifact_design.md` (internal planning doc).
- Renamed the audit key `R4b_non_negative_promised_transit` ->
  `RPT_non_negative_promised_transit` to eliminate numeric-prefix confusion
  with `R4_referential_integrity`; verified zero stale references remain in
  code.
- Removed `viva_demo.py` (a dissertation-demo-only backward-compatibility
  shim, no longer needed); updated `tests/test_metrics.py` to import directly
  from the `pipeline` package facade instead.
- Converted `tests/test_constraint_ladder.py` from a standalone script with
  zero pytest-discoverable tests into 6 real `test_*` functions covering the
  strictness ladder's monotonic construction.
- Full test suite now passes end-to-end with zero exclusions: 57/57, for the
  first time in the project's test history.

## What changed and why
Your `train_sdv_models` built SDV constraints **inline**, and that block only
distinguished `none` vs `moderate=strict` — so the strictness sweep had no real
gradient to trace. These edits introduce a **single shared, graded, cumulative
constraint ladder** used by both the training path and the Optuna path.

## Files changed (drop-in replacements; diff against your originals with `git diff`)

1. **`pipeline/constraints.py`** — ADDED:
   - `build_sdv_constraints(train_df, constraint_level)` — the graded ladder.
   - `SDV_CONSTRAINT_LADDER = ["none", "temporal", "moderate", "strict"]`.
   - Cumulative tiers: `none`(0) -> `temporal`(R1 ordering via SDV `Inequality` +
     R2 positive transit) -> `moderate`(+ arithmetic non-negativity) ->
     `strict`(+ referential `FixedCombinations` on carrier/service, cardinality-guarded).
   - Every addition is column-presence guarded; works on production and proxy schemas.

2. **`pipeline/pipeline.py`** — in `train_sdv_models`:
   - imports `build_sdv_constraints`;
   - types the six date columns as `datetime` in metadata (required for R1 `Inequality`);
   - replaces the inline constraint block with `constraints = build_sdv_constraints(train_df, constraint_level)`.

3. **`pipeline/tuning.py`** — `_build_sdv_constraints` now delegates to the shared
   `build_sdv_constraints` (the Optuna path and training path can no longer drift apart).

## New files
- **`privacy_utility_sweep.py`** — the sweep harness; default ladder now
  `none, temporal, moderate, strict` (+ optional `strict+reject`).
- **`test_constraint_ladder.py`** — asserts the ladder is a strict monotonic gradient.

## Verify (no SDV needed)
```bash
python test_constraint_ladder.py        # asserts none<temporal<moderate<strict
python privacy_utility_sweep.py --smoke  # end-to-end flow on proxy data
```
Expected ladder sizes on the full schema: none=0, temporal=4, moderate=8, strict=9.

## Run for real (with SDV installed + your corpus)
```bash
python privacy_utility_sweep.py --real path/to/your.csv \
    --seeds 5 --n-synth 10000 --epochs 100 \
    --architectures TVAE CTGAN CopulaGAN \
    --levels none temporal moderate strict strict+reject \
    --out outputs/pu_sweep
```

## Verify on first REAL run (the one thing I could not test without SDV)
The R1 temporal `Inequality` constraints require SDV to accept the date columns as
`datetime`. If your SDV version raises a datetime_format error, either pass an explicit
`datetime_format` in the metadata update, OR rely on the `strict+reject` level, which
enforces the same temporal rules deterministically via post-hoc CAG rejection.
Sanity check: at `temporal`+ levels the emitted CVR should drop toward 0 and the
audit should report fewer R1 violations than at `none`.

---

## UPDATE: membership-inference (MIA) audit added to the harness

`privacy_utility_sweep.py` now includes a protocol-correct MIA audit that upgrades
the privacy pillar from screening proxy (DCR/NNAA) to a real audit — flipping your
gap-table privacy cell from `~` to `✓`.

### What it does
- `membership_inference_auc(members, non_members, synth)` — distance-to-nearest-
  synthetic attack; returns `mia_auc` and `mia_tpr_at_fpr` (TPR @ 1% FPR, the modern
  low-FPR metric). AUC ~ 0.50 = no leakage; -> 1.0 = leakage.
- `run_mia_audit(...)` — runs the **member-split protocol**: splits real into
  members/non-members (stratified by label), trains the generator on members only,
  then tests whether the attacker can distinguish them. This is why it is a separate
  pass (a generator trained on ALL data has no non-members to test against).

### Run it
```bash
python privacy_utility_sweep.py --real your.csv --mia \
    --seeds 5 --n-synth 10000 --epochs 100 \
    --architectures TVAE CTGAN CopulaGAN \
    --levels none temporal moderate strict strict+reject --out outputs/pu_sweep
```
Adds `mia_auc` / `mia_tpr_at_fpr` columns to the summary and writes
`mia_vs_strictness.png`. `--mia` is opt-in because it doubles training (member split).

### Smoke validation (no SDV)
The bundled mock bootstraps from members, so the attack correctly reports high
leakage (AUC ~0.70-0.81) — and MORE for the low-noise TVAE mock than the noisier
CTGAN mock, proving the attack discriminates. With real SDV, watch whether MIA-AUC
moves toward 0.5 as strictness rises (your hypothesised favourable privacy effect).

### Extension point
`membership_inference_auc()` implements the standard shadow-free distance attack;
a DOMIAS-style density-ratio variant can be dropped into that one function later
without touching the protocol or the harness.

---

## UPDATE: warning filter narrowed from a blanket suppression

`pipeline/pipeline.py` used to install a bare `warnings.filterwarnings("ignore")`
at import time -- no `category=`/`module=`/`message=` scoping, so it suppressed
*every* warning, from every module, for the rest of the process, the moment
anything imported the `pipeline` package. That included SDV's own `FutureWarning`
telling us our dict-style constraints are silently ignored -- the literal root
cause of KNOWN_ISSUES.md finding 1 (the SDV-native-constraint-no-op bug). The
blanket filter is the reason that warning went unnoticed for the whole project
until it was dug for explicitly.

### What changed
Replaced the blanket call with five scoped filters, each targeting a specific,
empirically-verified third-party noise source by `category=` + `module=` (or
`message=` where module-attribution doesn't work -- see below):

```python
warnings.filterwarnings("ignore", category=FutureWarning, module=r"sdv\.single_table\..*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"sdv\.single_table\..*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"sklearn\..*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"matplotlib\..*")
warnings.filterwarnings("ignore", category=FutureWarning,
                         message=r"DataFrameGroupBy\.apply operated on the grouping columns")
```

Deliberately **not** suppressed: `sdv.cag`'s dict-style-constraint `FutureWarning`.
Scoped to `sdv.single_table.*` specifically (not all of `sdv.*`) so it cannot
accidentally catch `sdv.cag.*` too -- confirmed these are genuinely different
module paths before writing the filters, not assumed.

The pandas `FutureWarning` (`DataFrameGroupBy.apply...`) is attributed by
pandas' own stacklevel convention to *our* calling module
(`privacy_utility_sweep`), not pandas' internal module -- so it's scoped by
`message=`, the only option that actually works for it.

### Why these five and not others
Verified empirically against the real `run_sweep`/`run_mia_audit` path (TVAE/
CTGAN/CopulaGAN, multiple levels including `strict+reject`'s oversample/reject
loop), with the filter neutralized and `simplefilter("always")` forcing every
distinct warning to surface. `ConvergenceWarning` is structurally impossible
(only `RandomForestClassifier`/`XGBClassifier` are used -- neither is an
iterative-convergence estimator). `UndefinedMetricWarning` was tested directly
against the exact degenerate condition seen in real runs (a synthetic training
set with zero severe-delay rows) -- confirmed `f1_score(..., average="weighted")`,
the only call signature used anywhere in this codebase, does not raise it on
sklearn 1.6.1, even though `precision_score(average=None)` would.

### Verification
On the real (non-mock) import path, with the new filters: SDV's dict-constraint
`FutureWarning` appears; the SDV metadata-deprecated/save_to_json, pandas
groupby.apply, and sklearn StandardScaler warnings do not. Regression-tested in
`tests/test_warning_hygiene.py`, which runs each check in a genuinely separate
subprocess (not in-process) because pytest's own warnings plugin overrides the
ambient filter state for the duration of every test, which would otherwise mask
exactly the regression this test exists to catch. Confirmed the test fails
against the old blanket filter (reverted it temporarily, ran the test, restored
the fix) before considering it a real regression guard.

No check logic, audit keys, or results changed by this update -- warning
filters only affect console output, not computed values.
