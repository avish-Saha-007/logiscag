# Contributing to LogiSCAG

This is a research-benchmark toolkit; contributions should keep the four
pillars (integrity, fidelity, utility, privacy) and the constraint/audit-key
spec in `logiscag/__init__.py` / `KNOWN_ISSUES.md` consistent. If you change
a code path the paper describes, update the paper or flag the mismatch --
don't let prose and code drift apart again.

## Adding a denial constraint

Constraints live in `pipeline/constraints.py` and are consumed by two
independent paths that must stay in sync (see `KNOWN_ISSUES.md` finding 4 for
what happens when they don't):

1. **Audit** (`audit_constraints`) -- a CVR-reporting check, run against any
   dataframe (real or synthetic). Add a new branch that computes a boolean
   violation mask and folds it into the returned dict under a stable
   `R{n}_{name}` key. If the rule is genuinely hard, make sure your key is
   *not* excluded by the `"soft" not in k and "_info" not in k` filter that
   computes `total_hard_violations` -- excluding it silently makes the rule
   informational, regardless of what you intended.
2. **Generation** (`build_sdv_constraints` + optionally `cag_rejection_filter`)
   -- the same rule expressed as something SDV (or the post-hoc filter) can
   actually enforce at sample time. Add it at the appropriate tier in the
   cumulative ladder (`none < temporal < moderate < strict`), guarded by
   column presence so it doesn't crash on a schema that lacks the relevant
   column. **Read `KNOWN_ISSUES.md` finding 1 first**: as of the SDV version
   this package currently resolves to, `build_sdv_constraints`'s dict-style
   output is silently dropped by SDV's `add_constraints`. Either rewrite to
   the `sdv.cag` object API (preferred, tracked in `KNOWN_ISSUES.md`) or rely
   on `cag_rejection_filter` for genuine enforcement until that's done.

Add the new rule to `test_constraint_ladder.py` (assert it changes the
constraint count at the correct tier) and to the audit-key table in this
repo's copy of the paper's Appendix C if the rule is meant to be part of the
public spec.

3. **Declarative catalog entry** (`logiscag/constraints/catalog/seed_constraints.yaml`)
   -- if the rule is meant to be governance-readable (auditable by a
   compliance reviewer without reading Python), add an entry with an
   `audit_key` pointing at the key you added in step 1. The catalog engine
   (`logiscag/constraints/engine.py`) resolves violation counts by looking
   that key up in `audit_constraints()`'s output -- it never re-implements a
   check, so there is exactly one place the rule's logic lives. If the rule's
   actual code behavior differs from how you'd like to describe it (as
   happened with R3 and R5 here -- see the `notes` fields in the catalog and
   `KNOWN_ISSUES.md`), describe what the code does and say so explicitly;
   don't write a prettier predicate than what's actually checked.

## Adding a one-off custom constraint without touching the catalog

For a constraint that doesn't belong in the public seed catalog (e.g. a
partner-specific rule), use the `@constraint` decorator instead of editing
`pipeline/constraints.py`:

```python
from logiscag.constraints import constraint

@constraint(id="custom_cod_limit", category="arithmetic", type="hard",
            rationale="Cash-on-delivery amount cannot be negative.")
def cod_amount_nonnegative(row):
    return row["cash_on_delivery"] >= 0
```

`logiscag.constraints.catalog_audit_df(df)` then reports it alongside the
seed eight, evaluated row-wise directly against the predicate (there's no
existing Python check to delegate to for a custom rule). This only audits --
it does not yet wire into `build_sdv_constraints`'s generation-time ladder;
treat custom constraints as audit/report-card-only until that's extended.

## Adding a generator (architecture)

`pipeline/pipeline.py`'s `train_sdv_models` takes an `architectures` list and
a `configs` table of `(name, synth_class, kwargs)`. To add one:

1. Add `(name, YourSynthesizerClass, your_kwargs)` to `configs`.
2. Make sure your synthesizer accepts the same `metadata` object (typed
   columns, including the datetime columns needed for `Inequality`
   constraints) and supports `.add_constraints()` / `.fit()` / `.sample()`.
3. If it can't take SDV-native constraints at all, it still benefits from
   `post_filter=cag_rejection_filter` (the resample-to-target oversample
   loop in `_sample_until_valid` works for any synthesizer that supports
   repeated `.sample(num_rows=...)` calls).
4. Add it to `privacy_utility_sweep.py`'s `--architectures` choices and to
   `tests/test_smoke.py` if you want it covered by CI.

## Adding a dataset adapter

Adapters map a raw/public schema onto the canonical logistics schema
(`date_order_capture`, `date_ship_last_shipment`, `date_delivery_last_shipment`,
`date_promise_shipment`, `date_promise_delivery`, `last_scac`,
`carrier_service_code`, `delay_label`, ... -- see `dataco_adapter.py` for the
full target schema and `pipeline/data.py`'s `FEATURE_COLS` for the
leakage-safe feature subset).

1. Create `your_adapter.py` at the repo root with `load_xxx(path) -> raw_df`
   and `xxx_to_canonical(raw_df) -> canonical_df`, following
   `dataco_adapter.py`'s structure.
2. **Leakage check, mandatory**: verify no canonical column is an exact (or
   near-exact) arithmetic function of another canonical column that the
   label is also derived from. The original leakage bug this project fixed
   was exactly this: `sla_buffer = shipment_buffer_days + promised_transit_days
   - transit_duration_days`, an exact identity, with all three terms in the
   feature set used to predict a label derived from `sla_buffer`. Check this
   with the same kind of correlation/exact-reconstruction test before adding
   any new derived feature to `FEATURE_COLS`.
3. Add a thin wrapper in `logiscag/adapters/your_adapter.py` that re-exports
   your two functions (see `logiscag/adapters/dataco.py`), import it in
   `logiscag/adapters/__init__.py`, and add it to `logiscag.reproduce` if it
   should be reachable from the one-command path.
4. Document the exact column mapping in `DATASHEET.md` or a parallel
   datasheet for your dataset.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

`tests/test_constraint_ladder.py` and `tests/test_smoke.py` must pass without
SDV installed (they exercise the mock-synthesiser fallback). Tests that need
real SDV behavior should be skippable via `pytest.importorskip("sdv")` at the
top of the test, so CI's no-SDV job keeps working.

## Reporting issues

Please open an issue at
<https://github.com/avish-Saha-007/logiscag/issues>. Before filing, check
[`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) — several behaviours that look like bugs
are tracked there deliberately, with the verification steps already run.

A useful bug report includes:

- what you ran (the exact command, or a minimal snippet);
- the Python version, and the output of `pip show logiscag pandas numpy sdv`;
- whether you are on the no-SDV verification path or the SDV real path;
- the full traceback, and any `UserWarning` text — the warnings in this package
  are deliberate signals (for example, the `valid_combos` fallback warning
  means you are getting a strictly weaker referential-integrity check).

If the issue concerns a divergence between what the code does and what the
paper claims, say so explicitly — those are treated as higher priority than
ordinary bugs, and are the reason `KNOWN_ISSUES.md` exists.

## Getting support

- **Questions about using the toolkit** — open a
  [GitHub Discussion](https://github.com/avish-Saha-007/logiscag/discussions)
  or an issue labelled `question`.
- **Reproducing the benchmark** — start with [`RUN_PLAYBOOK.md`](RUN_PLAYBOOK.md),
  which covers the phased run sequence and the diagnostics for each phase.
- **Interpreting the four pillars or the constraint catalog** —
  [`DATASHEET.md`](DATASHEET.md) documents the benchmark dataset, and
  `logiscag/__init__.py`'s docstring lists the known code↔paper gaps.

Maintainer response times are best-effort; this is volunteer-maintained
research software.

## Pull requests

1. Fork, branch from `main`, and keep the change focused.
2. Add or update tests — `pytest` must pass, and CI runs it on Python
   3.10–3.12.
3. Run `python -m logiscag.reproduce --verify` before pushing; it exercises the
   constraint ladder and a smoke sweep in seconds and catches most wiring
   regressions.
4. If your change alters anything the paper describes, update
   [`CHANGES.md`](CHANGES.md) and say so in the PR description.

## Code of conduct

Participation in this project is governed by the
[Contributor Covenant](CODE_OF_CONDUCT.md). By taking part, you agree to
uphold it.
