# Known Issues

## 1. SDV-native constraint tiers may not be enforcing anything (critical, tracked)

**Status:** documented, not fixed. Do not regenerate Table 1 from this finding alone.

**What's wrong:** `pipeline/constraints.py`'s `build_sdv_constraints()` returns
constraints as plain Python dicts (`{"constraint_class": "Inequality",
"constraint_parameters": {...}}`). This was SDV's constraint format prior to
the `sdv.cag` object API. The SDV version this package currently resolves to
via an unpinned `pip install sdv` (1.37.2 as of 2026-06) deprecated the dict
format: `sdv.cag._utils._filter_old_style_constraints()` strips every
dict-style constraint out of the list passed to `synth.add_constraints()` and
emits `FutureWarning: ... constraints will be ignored`, before any training
happens.

**Verified consequence:** I confirmed, on this machine, that:
- The warning fires on every real (non-mock) sweep run at every level except
  `none` (which never calls `add_constraints` at all, since
  `build_sdv_constraints("none")` returns `[]`).
- Calling `synth.add_constraints([...])` with a constraint that gets fully
  filtered to an empty list still produces *different* sampled output than
  never calling `add_constraints` at all, at the same global seed (verified
  with a controlled same-seed A/B run: sum of abs numeric differences ≈ 829
  on a small proxy-data test, output not bitwise-identical). This is because
  `BaseSingleTableSynthesizer.add_constraints()` rebuilds `self._data_processor`
  as a side effect regardless of whether any constraint survives filtering --
  so calling it perturbs the deterministic RNG-consumption path without
  enforcing any constraint.

**Why this matters for the paper's headline result:** in
`outputs/copulagan_final/sweep_summary_combined.csv` /
`sweep_significance_combined.csv`, CopulaGAN's `temporal` level (which has
`reject: False` in `STRICTNESS_LADDER` -- no post-hoc filtering either) shows
DCR statistically indistinguishable from `strict+reject` (0.5457 vs 0.5455,
p=0.99) and statistically very different from `none` (0.3775, p<0.0001 vs
both). Given the above, the only thing that differs between `none` and
`temporal` in the actual generation code is this no-op `add_constraints()`
call -- not real constraint enforcement. The data is real and not fabricated,
but the *causal story* ("a graded SDV-native ladder progressively improves the
trade-off") is not what's demonstrated. The only mechanism in the whole
pipeline confirmed to genuinely enforce anything is the post-hoc
`cag_rejection_filter`, exercised only at `strict+reject`.

This is also consistent with finding 4 in `logiscag/__init__.py`'s docstring:
`build_sdv_constraints`'s `strict`-tier `FixedCombinations` referential check
is dict-style too, so it has also never been enforced by SDV; the only
referential check that has ever actually run is `cag_rejection_filter`'s
weaker null/presence check.

Why `temporal` nonetheless tracks `strict+reject` so closely is not fully
explained by this investigation and is worth understanding before relying on
it as evidence of anything -- it may be a property of how the no-op
`add_constraints` call's deterministic RNG perturbation happens to interact
with this particular dataset/seed combination, rather than a generalizable
result.

**Recommended follow-up (tracked, not done here):** on a separate branch,
rewrite `build_sdv_constraints()` to return `sdv.cag` constraint objects
(`sdv.cag.Inequality`, `sdv.cag.ScalarInequality`, `sdv.cag.FixedCombinations`)
instead of dicts, confirm via the same A/B same-seed test that constrained and
unconstrained runs now diverge *and* that the synthetic output actually
satisfies the constraint (e.g. zero `R2_positive_transit` violations post-hoc
for any row that should be constrained), then re-run the full sweep and
re-verify Table 1's numbers before the corpus run / camera-ready. Pin the SDV
version in `pyproject.toml` once that rewrite is verified, so this doesn't
silently regress again on a future `pip install --upgrade sdv`.

## 2. numpy 2.x / torch ABI mismatch on a fresh install (fixed)

**Status:** fixed by pinning `numpy<2` in `pyproject.toml`.

Installing `logiscag[sdv]` into a genuinely clean venv resolved `numpy==2.0.2`
by default, but SDV's `torch==2.2.2` wheel was compiled against the numpy 1.x
ABI, producing `UserWarning: Failed to initialize NumPy: _ARRAY_API not
found` on `import sdv` and (per torch/numpy's own warning text) a risk of
crashing rather than just warning. The known-working development venv used
for every real result in this project has `numpy==1.26.4`, which does not hit
this. Pinned `numpy<2` accordingly; re-verified `pip install -e ".[sdv]"` in a
fresh venv resolves `numpy==1.26.4` and imports cleanly with no warning.

## 3. `test_metrics.py` depended on a missing `viva_demo.py` module

**Status:** resolved via a one-line compatibility shim (`viva_demo.py` at the
repo root, `from pipeline import *`), not a logic fix. `pipeline/__init__.py`'s
own docstring already documented its intent to serve as "a stable import
facade for `viva_demo.py` and tests" -- the shim file itself was simply
missing from this bundle. All 12 names `test_metrics.py` imports from `viva_demo`
were already present in `pipeline/__init__.py`'s facade.

**Update (v0.2.0, 2026-08-16):** superseded by a cleaner fix. `viva_demo.py`
was a dissertation-demo-only backward-compatibility shim, no longer needed, and
has been removed entirely. `tests/test_metrics.py` now imports the same 12
names directly from the `pipeline` package facade instead of going through the
shim.

## 4. R3 (capture latency) and R5 (referential carrier integrity) audit-level code<->paper mismatches

**Status:** both resolved in `audit_constraints()` (2026-06-30), ahead of the
v0.1.0 GitHub/Zenodo release.

- **R3** was previously informational-only (key suffixed `_info`, excluded
  from `total_hard_violations`), contradicting Appendix C's "hard" framing.
  It is now a genuine hard rule, with a named `CAPTURE_LATENCY_TOLERANCE_DAYS`
  constant (~1 second, mirroring R1's documented OMS/DPE clock-skew
  rationale) so the same clock-skew noise that motivated excluding R1's
  created<=capture sub-check doesn't get misread as a violation here either.
- **R5** previously only checked that `last_scac`/`carrier_service_code` were
  non-null (presence), not that the pair was a real observed combination,
  contradicting Appendix C's `KnownCombos` membership specification.
  `audit_constraints()` now accepts an optional `valid_combos` parameter (see
  `build_valid_carrier_combos(real_df)`); when supplied, R5 is checked as true
  combination-membership against a reference vocabulary derived from
  real/training data. When omitted, it falls back to the old presence-only
  check and raises a `UserWarning` every call -- documented degradation, not
  silent.
- **Sub-bug found and fixed during review:** the fallback warning was
  initially a plain `warnings.warn()` call, which Python's default
  dedup-by-location semantics would only show once per process even under
  normal conditions -- and `pipeline/pipeline.py:49` separately installs a
  blanket `warnings.filterwarnings("ignore")` at import time (pre-existing,
  unrelated, out of scope to remove) that transitively suppresses *all*
  warnings, including this one, for the rest of the process once anything
  imports the `pipeline` package -- i.e. effectively always, in real (non-test)
  usage. Confirmed via direct testing: `from pipeline.constraints import
  audit_constraints` (the normal import path) printed zero warnings across
  three calls; loading `constraints.py` standalone (bypassing
  `pipeline/__init__.py`) printed all three. `pytest.warns()` did not catch
  this, because it forces `simplefilter("always")` for its own block,
  masking the ambient suppression -- the original test suite's "passing"
  warning tests were not exercising real-world behavior. Fixed by wrapping
  just this warn call in a local `with warnings.catch_warnings():
  simplefilter("always", UserWarning)` block, which is immune to the ambient
  filter without touching or weakening it. Regression-tested in
  `tests/test_constraint_catalog.py::test_r5_fallback_warns_on_every_call_despite_ambient_ignore_filter`,
  which explicitly reproduces the hostile ignore-filter condition (rather
  than relying on `pytest.warns()`, which would mask the regression again).

**Verified no-op on DataCo, by structural guarantee, not just empirical
observation:** `dataco_canonical.csv`'s `capture_latency_days` is `[0.]` for
all 180,519 rows (the adapter sets order-creation and order-capture to the
same timestamp), so R3 cannot fire on this benchmark regardless of tolerance.
`last_scac` (4 values: shipping mode) x `carrier_service_code` (3 values:
customer segment) is a **full, saturated 4x3=12 cross-product** in the real
data -- not a sparse subset. Since SDV always samples a categorical column
from its training-observed vocabulary, any synthetic (scac, svc) pair is
mathematically guaranteed to land in one of the 12 valid combinations; R5's
membership check cannot fire on DataCo synthetic output either, independent
of whether per-column categorical correlation is modeled well.

**Scope of the fix:** `audit_constraints()` only. `cag_rejection_filter()`
(the actual post-hoc enforcement mechanism for `strict+reject`) and
`build_sdv_constraints()`'s SDV-native tiers are unchanged -- they still do
presence-only checking (the former) or are not currently enforced at all by
the installed SDV version (the latter, see finding 1). This was an
audit/reporting-layer fix only, by design (see the catalog's updated `notes`
fields and `CONTRIBUTING.md`).

**Headline Table 1 unaffected, by construction:** the `cvr` column reported
in `sweep_summary_combined.csv` is computed by `integrity_check_synthetic()`,
a separate, narrower function (checks only transit positivity, capture
latency, and label cardinality -- it has no R4/R5 referential check at all).
It was not modified by this fix, so the headline CVR numbers
(4.31% / 2.96% / 0.00% for none / temporal / strict+reject) are unchanged by
construction, not merely re-verified empirically. `integrity_check_synthetic`
not implementing R1/R4/R5/R3b despite its docstring claiming to was already a
known, separate inconsistency at packaging time and remains untouched here.

## 5. `pipeline/pipeline.py`'s blanket warning suppression (narrowed)

**Status:** narrowed 2026-06-30. Was: a bare `warnings.filterwarnings("ignore")`
at import time -- no `category=`/`module=`/`message=` scoping, suppressing
every warning from every module process-wide, the moment anything imports the
`pipeline` package. Discovered as the root cause of finding 4's "warns on
every call" claim being false in practice (see that finding's "sub-bug" note);
investigated further on its own, since it was also independently suppressing
the `sdv.cag` `FutureWarning` that is the literal root-cause signal for
finding 1 (the SDV-native-constraint-no-op bug).

**No git history or code comment documented its original intent**; this repo
has no `.git` anywhere in its tree, and no doc (`CHANGES.md`, `RUN_PLAYBOOK.md`,
`artifact_design.md`, `DATASHEET.md`) mentioned it before this entry. Its
position -- immediately after the full import block, immediately before
`np.random.seed(42)` -- reads like a generic console-quieting convenience, not
a targeted fix.

**Empirically catalogued** (neutralizing the filter and running the real
`run_sweep`/`run_mia_audit` path across TVAE/CTGAN/CopulaGAN and every level
including `strict+reject`) what it was actually suppressing: SDV's
`SingleTableMetadata`-deprecated `FutureWarning` and `save_to_json`
`UserWarning` (both `sdv.single_table.base`), SDV's dict-style-constraint
`FutureWarning` (`sdv.cag._utils` -- the one that matters), pandas'
`DataFrameGroupBy.apply` `FutureWarning`, and sklearn's `StandardScaler`
feature-names `UserWarning`. Specifically checked and ruled out: sklearn
`ConvergenceWarning` (structurally impossible -- only `RandomForestClassifier`/
`XGBClassifier` are used, neither iterative-convergence) and
`UndefinedMetricWarning` (tested directly against the exact degenerate
zero-predicted-class condition seen in real `strict+reject` runs; the specific
call signature used here, `f1_score(..., average="weighted")`, does not raise
it on the installed sklearn 1.6.1).

**Fix:** replaced the blanket call with five scoped filters (see `CHANGES.md`
for the exact code) targeting each catalogued source by `category=` + `module=`
(or `message=` for the pandas warning, which pandas attributes to *our* calling
module via its own stacklevel convention, not to a pandas-internal module --
`module=` scoping doesn't work for it). The `sdv.cag` warning is deliberately
left unsuppressed, scoped via `module=r"sdv\.single_table\..*"` rather than a
broader `sdv\..*` so it structurally cannot catch `sdv.cag.*` too.

**Verified** on the real (non-mock, non-pytest) import path: the `sdv.cag`
warning appears; the other four do not. Regression-tested in
`tests/test_warning_hygiene.py`, which runs each check in a genuinely separate
subprocess rather than in-process, because pytest's own warnings plugin
overrides the ambient filter state for the duration of every test (needed for
its end-of-session summary), which would otherwise mask this exact regression
the way it masked finding 4's sub-bug originally. Confirmed the test fails
against the old blanket filter (temporarily reverted, ran, restored) before
relying on it as a real guard.

**Scope:** console output only. No check logic, audit keys, CVR, or any
other result changed.

## 6. `build_valid_carrier_combos` admitted null carriers on pandas >= 3 (fixed)

**Symptom.** `tests/test_constraint_catalog.py::test_build_valid_carrier_combos_derives_from_real_data_only`
failed on a modern dependency stack: a row with a null `last_scac` produced a
`(nan, "XE")` entry in the reference vocabulary instead of being excluded.

**Cause.** The exclusion filter stringified first and then compared against the
sentinel set `{"", "nan", "None", "NaN"}`. Under pandas < 3, `astype(str)`
rendered `None`/`NaN` as the literal string `"nan"`, so the sentinels caught
them. Under pandas >= 3 the new string dtype propagates NA through
`astype(str)` instead, so nothing matched and the null pair survived.

**Impact.** Non-trivial, and in the weakening direction: `valid_combos` is the
reference vocabulary for the R4 referential-integrity membership check. A null
pair inside it means a synthetic row with a *missing* carrier or service code
would be scored as referentially valid — exactly the class of row the rule
exists to reject. Affects any run on pandas >= 3; runs on pandas < 3 (including
the published benchmark numbers) are unaffected, since the sentinel filter
worked there.

**Fix.** Nullity is now tested on the raw series (`.isna()`) *before*
stringification, with the sentinel-string check retained for genuinely empty
strings and the `<NA>`/`NaT` renderings added. The presence-only fallback path
inside `audit_constraints` already used `.isna() |` and was never affected.

**Verified** by the existing regression test, which now passes; the full suite
is green on pandas 3.0.2 / numpy 1.26.4.

## 7. Paper's R4 non-negativity half was not checked anywhere (fixed)

**Status.** Resolved 2026-08-16 via option 1 below — the check is implemented,
so the paper's "eight rules" claim now holds. Audit layer only.

**What was wrong.** The paper's Appendix C specifies eight rules;
`audit_constraints()` emitted seven. The ordering half of the paper's R4
(promise-ship precedes promise-delivery) is enforced inside `R1_temporal_order`.
Its **non-negativity half — `promised_transit_days >= 0` — had no audit check**,
so it was invisible to every CVR figure and to the catalog.

**Correction to this finding's earlier wording.** The 2026-07-27 entry claimed
the rule was checked "NOWHERE -- not in `audit_constraints`, not in
`build_sdv_constraints`, not in `cag_rejection_filter`." The
`build_sdv_constraints` part of that claim is **false**, verified 2026-08-16 by
calling it directly: `_NONNEG_SCALARS` has always contained
`("promised_transit_days", 0.0)`, and `build_sdv_constraints` emits it as a
`ScalarInequality` at the `moderate` and `strict` tiers (constraint counts 8 and
9 respectively on DataCo; absent at `none`/`temporal`, correctly, since that tier
predates the arithmetic block). The accurate statement of the original gap is:

- **`audit_constraints`** — genuinely absent. This was the real coverage gap, and
  is what the fix closes.
- **`cag_rejection_filter`** — genuinely absent. Still absent; out of scope here.
- **`build_sdv_constraints`** — *present* since before packaging, but per
  finding 1 its dict-style constraints are silently filtered out by the
  installed SDV, so the rule was declared-but-not-enforced rather than missing.

The practical upshot is unchanged — nothing was *effectively* enforcing the rule,
and nothing was reporting on it — but "not expressed in the generation path" was
the wrong diagnosis, and anyone rewriting `build_sdv_constraints` to the
`sdv.cag` object API (finding 1's recommended follow-up) will find this rule
already there and should not add it twice.

**Fix.** `audit_constraints()` now emits an eighth check under the audit key
`RPT_non_negative_promised_transit`, with a matching catalog entry
`R4_promised_transit_non_negative` (which fills the R4 slot the catalog-id
namespace had left vacant, bringing the catalog to eight entries against the
paper's eight rules). Presence-guarded, so the promise-column-less DISSERTATION
schema is unaffected; NaN is not counted as a violation, matching R3's treatment
of unmeasurable values. R4's ordering half remains in `R1_temporal_order`, so
the paper's R4 is the one rule deliberately split across two checks.

**Key naming.** `RPT_`, not `R4_`, because the audit-key namespace already
spends `R4_` on referential integrity — the rule the paper and catalog both
number R5. The clear separation eliminates any confusion about hierarchical
relationship. Verified by test that the two keys carry distinct counts and that
no two catalog entries share an `audit_key`.

**Scope.** Audit layer only, matching the finding-4 fix. The rule counts toward
`total_hard_violations`, but nothing drops or constrains a row on its account:
`cag_rejection_filter()` does not check it, and `build_sdv_constraints()`'s
declaration of it is inert on the installed SDV (finding 1). The catalog entry
therefore declares `type: hard` with `on_violation: flag`, and a test pins that
scope so wiring it into the rejection path later forces these docs to be updated
in the same commit. `build_sdv_constraints` was left untouched deliberately —
the rule is already declared there, and making that declaration actually bite
requires finding 1's `sdv.cag` rewrite, not a change to this rule.

**Verified a no-op on real data by measurement, not assumption.** Run against
the authoritative upstream CSV (Mendeley Data DOI
[10.17632/8gx2fvg2k6.3](https://data.mendeley.com/datasets/8gx2fvg2k6/3),
SHA-256 `fa6d022e…80aa6` verified against the published hash) through
`dataco_to_canonical`: **0 violations across all 180,519 canonical rows.** The
zero holds for two independent reasons — (a) the adapter derives
`date_promise_delivery` as `date_promise_shipment` plus a `clip(lower=0)`
scheduled-days offset, making a negative window structurally unreachable, and
(b) DataCo's raw `Days for shipment (scheduled)` column holds only `{0, 1, 2, 4}`
with no nulls, so the clip in (a) never has anything to absorb here. No
published number moves: `total_hard_violations` on real canonical data stays at
5,080 and `violation_rate_pct` at 2.8141% (both entirely R2's zero-transit
same-day rows, pre-`deterministic_repair`), and Table 1's `cvr` comes from the
separate, narrower `integrity_check_synthetic()`, which this fix does not touch.

**Boundary note for anyone porting the rule.** 9,737 of the 180,519 real rows
sit exactly on `promised_transit_days == 0` — a legitimate same-day promise. The
predicate is `>= 0`, per the paper. Writing it as `> 0` by false analogy with
R2 (whose predicate genuinely is strict) would flag all 9,737 valid rows and
inflate the real-data hard CVR by ~5.4 points. Regression-tested.

**Still open, separately:** the three numbering schemes — the paper's Appendix C
(R1–R8), the catalog's ids, and `audit_constraints`'s output keys — still do not
agree, and `R4b` adds one more asymmetry to decode. `README.md`'s mapping table
is the authoritative decoder and now lists all three columns per rule. Genuinely
unifying the schemes would be a breaking change to every `audit_key` and is not
attempted here.

**Update (v0.2.0, 2026-08-16):** the audit key itself has been renamed,
`R4b_non_negative_promised_transit` -> `RPT_non_negative_promised_transit`, to
remove the numeric-prefix confusion with `R4_referential_integrity` specifically
called out above. Verified zero stale references to the old key remain in
code. This does not resolve the broader three-scheme mismatch noted above,
which remains open.

## 8. Middle strictness tiers show identical metrics on the verification path

**Status.** Open — expected to be a symptom of finding 1, not a separate bug.

`python -m logiscag.reproduce --verify` reports identical fidelity, utility,
and privacy figures across `none`, `temporal`, `moderate`, and `strict` for a
given architecture. This is consistent with finding 1 (SDV silently dropping
dict-style constraints, so the middle tiers enforce nothing), and with the mock
synthesiser ignoring constraints by construction on the no-SDV path.

Worth confirming which of those two causes is operative before the ladder is
described as validated anywhere. The `strict+reject` tier is the only one whose
enforcement is confirmed end-to-end, via the post-hoc `cag_rejection_filter`.
