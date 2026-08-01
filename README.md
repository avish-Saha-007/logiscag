# LogiSCAG

[![tests](https://github.com/avish-Saha-007/logiscag/actions/workflows/tests.yml/badge.svg)](https://github.com/avish-Saha-007/logiscag/actions/workflows/tests.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
<!-- TODO: paste the Zenodo DOI badge here after the first GitHub release -->

A constraint-validated benchmark and toolkit for operationally-valid synthetic
supply-chain data: a declarative constraint library of last-mile denial
constraints, a generator-agnostic Constraint-Augmented Generation (CAG)
wrapper, a four-pillar evaluation harness (integrity, fidelity, utility,
privacy), and a reproducible public benchmark on the DataCo Smart Supply Chain
dataset.

## Statement of need

Operational supply-chain records are among the most useful and least shareable
data in industry: commercially sensitive, privacy-governed, and often
contractually restricted. Synthetic data offers a bridge, but general-purpose
tabular generators optimise statistical resemblance with no notion of the
physical and logical rules of logistics, so they readily emit impossible
records — deliveries that precede shipments, negative transit times, carrier
lanes that do not exist.

Two gaps recur in practice. First, **operational validity is not measured**:
widely used evaluation suites score fidelity, utility, and privacy, but none
treats violation of domain rules as a first-class evaluation axis. Second,
**evaluation itself fails silently** — arithmetic label leakage that yields
degenerate utility ceilings, constraint specifications a generator quietly
ignores, feature-pipeline collapse that reduces a utility model to a single
predictor. LogiSCAG packages both the missing axis and the guards against those
failure modes: a preflight leakage validator, a warned (never silent) fallback
when a referential vocabulary is absent, regression-tested feature-width
checks, and a membership-inference audit that is power-validated against a
deliberately leaky baseline before its verdicts are trusted.

Intended users are (a) industry data teams releasing or exchanging
operationally plausible synthetic logistics data under governance constraints,
and (b) researchers benchmarking tabular generators on supply-chain tasks who
need integrity, leakage-safe utility, and audited privacy measured
reproducibly.

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                 # core dependencies, no SDV
```

Add the generator backend when you need it:

```bash
pip install -e ".[sdv]"          # adds SDV (pinned; see KNOWN_ISSUES.md)
pip install -e ".[dev]"          # SDV + pytest
```

Requires Python 3.10 or newer. Dependencies are declared in `pyproject.toml`
and installed automatically; `numpy<2` is pinned deliberately (SDV's torch
wheel is built against the numpy 1.x ABI).

## Quickstart

```bash
python -m logiscag.reproduce --verify
```

This is the no-SDV verification path. It runs the constraint-ladder test and a
smoke sweep on proxy data with a mock synthesiser, takes seconds, and proves
the wiring is correct without needing SDV installed or any real data.

```bash
python -m logiscag.reproduce --real DataCoSupplyChainDataset.csv
```

The real path requires SDV and the public
[DataCo Smart Supply Chain dataset](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)
CSV, which you download separately under DataCo's own terms (see
[`DATASHEET.md`](DATASHEET.md)). It defaults to a fast reduced-fidelity demo;
pass `--full` to match the paper's exact protocol (5 seeds, 100 epochs,
~2–3 hours on CPU).

## Example usage

Audit any dataframe — real or synthetic — against the constraint catalog, and
get one row per rule with its declarative metadata and a live violation rate:

```python
import logiscag

report = logiscag.constraints.catalog_audit_df(my_dataframe)
print(report[["id", "type", "severity", "violations", "violation_rate_pct"]])
```

For referential-integrity rules, pass a reference vocabulary derived from the
**real** training data — never from the synthetic data being audited, which
would make every combination trivially valid:

```python
combos = logiscag.constraints.build_valid_carrier_combos(real_df)
report = logiscag.constraints.catalog_audit_df(synth_df, valid_combos=combos)
```

Omitting `valid_combos` falls back to a strictly weaker presence-only check and
raises a `UserWarning` on every call — the weaker path is documented, never
silent.

Generate under a named strictness level and evaluate all four pillars:

```python
real_df = logiscag.adapters.dataco.dataco_to_canonical(
    logiscag.adapters.dataco.load_dataco("DataCoSupplyChainDataset.csv"))

synths = logiscag.cag.generate(real_df, level="strict+reject",
                               architectures=["CopulaGAN"])
```

Register your own row-level rule without touching the seed catalog:

```python
from logiscag.constraints import constraint

@constraint(id="custom_cod_limit", name="Non-negative COD amount",
            category="arithmetic", type="hard")
def cod_amount_nonnegative(row):
    return row["cash_on_delivery"] >= 0
```

## What's in the box

| Component | Module | Paper section |
|---|---|---|
| Constraint library | `logiscag.constraints` | §3.1, Appendix C |
| Declarative catalog + `@constraint` extension API | `logiscag.constraints.catalog` / `.engine` / `.api` | §3.1, Appendix C |
| CAG wrapper | `logiscag.cag` | §3.2 |
| Four-pillar evaluation harness | `logiscag.evaluation` | §3.3 |
| DataCo benchmark adapter | `logiscag.adapters.dataco` | §3.4 |

### Architecture

The `logiscag` namespace is a stable public API surface; the implementation
lives in `pipeline/` (constraints, metrics, evaluation, generation), the sweep
harness `privacy_utility_sweep.py`, and the adapter `dataco_adapter.py`. This
split is deliberate: the public API can stay stable while the internals evolve,
and the declarative catalog can describe rules without re-implementing them.

That last point matters more than it sounds. The catalog resolves each entry's
`audit_key` against `audit_constraints()`'s output rather than evaluating its
own copy of the predicate, so a catalog description cannot silently drift from
the executable check it claims to describe. Custom constraints registered via
`@constraint` are the exception — they are evaluated row-wise, because there is
no existing implementation to delegate to.

### Constraint coverage — read this before citing rule counts

`audit_constraints()` emits **seven** executable checks:

| Audit key | Catalog id | Type |
|---|---|---|
| `R1_temporal_order` | `R1_temporal_ordering` | hard |
| `R2_positive_transit` | `R2_positive_transit` | hard |
| `R3_non_negative_capture_latency` | `R3_non_negative_capture_latency` | hard |
| `R3b_label_sla_consistency` | `R7_label_sla_consistency` | hard |
| `R4_referential_integrity` | `R5_referential_carrier_integrity` | hard |
| `R5_label_cardinality` | `R6_label_cardinality` | hard |
| `R6_scac_calendar_soft` | `R8_carrier_calendar_plausibility` | soft |

The paper's Appendix C specifies eight rules (R1–R8). Seven map to executable
checks as above. The ordering half of the paper's R4 (promise-ship precedes
promise-delivery) is folded into `R1_temporal_order`; its non-negativity half
(`promised_transit_days >= 0`) is **not currently checked anywhere**. The
catalog ids and the audit keys also use different numbering, which is why the
table above exists. See [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).

## Tests

```bash
pip install -e ".[test]"
pytest
```

CI runs the suite on Python 3.10–3.12 on every push and pull request, plus the
no-SDV verification path. The SDV-dependent job runs separately and
non-blocking, because SDV pulls in torch and periodically breaks against new
numpy/pandas releases.

## Documentation

- [`DATASHEET.md`](DATASHEET.md) — datasheet for the DataCo benchmark (Gebru et al. format)
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — extension points, reporting issues, getting support
- [`RUN_PLAYBOOK.md`](RUN_PLAYBOOK.md) — the full phased run sequence and diagnostics
- [`CHANGES.md`](CHANGES.md) — strictness-ladder and sweep-harness change log
- [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) — tracked issues, including a constraint-enforcement
  caveat on current SDV versions that affects how to read the middle strictness tiers
- [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)

## Support

- Bugs and code↔paper divergences → [open an issue](https://github.com/avish-Saha-007/logiscag/issues)
- Usage questions → [Discussions](https://github.com/avish-Saha-007/logiscag/discussions) or an issue labelled `question`
- Reproducing the benchmark → start with [`RUN_PLAYBOOK.md`](RUN_PLAYBOOK.md)

## Citing LogiSCAG

If you use this software, please cite it. Machine-readable metadata is in
[`CITATION.cff`](CITATION.cff); GitHub renders a "Cite this repository" button
from it.

<!-- TODO: once the first release is archived, add the Zenodo DOI here.
     Once a JOSS paper is accepted, cite that in preference. -->

## License

Apache-2.0 (code) — see [`LICENSE`](LICENSE). The DataCo CSV is fetched
separately by the user under DataCo's own terms; see [`DATASHEET.md`](DATASHEET.md).
