# Datasheet for the LogiSCAG DataCo Benchmark

*Following Gebru et al., "Datasheets for Datasets." This datasheet documents
the **benchmark** LogiSCAG releases -- the canonical adapted form plus
evaluation protocol layered on the public DataCo Smart Supply Chain dataset.
Where original DataCo provenance is not authoritatively known to us, we say so
and point to the source distributor rather than fabricate it. This file is
transcribed from the paper draft's Appendix A; placeholders and `%VERIFY`
markers are preserved as-is rather than filled in with invented values.*

## Motivation

- **Why was it created?** To provide a reproducible, **public** benchmark for
  evaluating operationally-valid synthetic supply-chain data across four
  pillars (integrity, fidelity, utility, privacy) without requiring privileged
  access to proprietary logistics data.
- **What is "the benchmark"?** The released artifact is the **adapter +
  canonical schema + SLA-breach label + leakage-safe split + four-pillar
  evaluation protocol**, not the raw DataCo records (which are independently
  public). LogiSCAG performs no primary data collection.
- **Who created/funded it?** [Authors / institutions -- placeholder, to be
  filled in before submission.]

## Composition

- **Instances.** Order-level supply-chain records, one row per order line;
  **180,519** instances after adapter mapping (all DataCo rows with parseable
  dates and finite durations; rows failing these are dropped).
  **Verified during packaging (2026-06):** running `dataco_adapter.py`'s
  `dataco_to_canonical()` against the full `DataCoSupplyChainDataset.csv`
  produces exactly 180,519 canonical rows from 180,519 raw rows -- **zero
  rows dropped** for this dataset. This replaces the paper draft's "report
  exact dropped count at camera-ready" placeholder with a verified figure;
  re-verify if the adapter or source CSV changes.
- **What each instance represents.** A customer order with order/shipping
  timestamps, adapter-derived delivery and promise timestamps, scheduled and
  actual transit durations, a derived 3-class delay label, and categorical
  attributes (shipping mode, customer segment, market, region, product
  category) mapped onto canonical carrier/service/node slots.
- **Label.** `delay_label ∈ {0 = on-time, 1 = moderate (≤2 days late), 2 =
  severe (>2 days late)}`, derived from `sla_buffer = scheduled − actual
  transit days`; empirical distribution **42.7% / 49.5% / 7.8%**. The 7.8%
  severe-delay minority is the operationally critical, imbalanced class.
- **Sensitive data / PII.** The fields used are order-level operational and
  categorical attributes, not direct personal identifiers. DataCo is publicly
  distributed for analytics/ML; users must consult DataCo's own terms for the
  authoritative statement of its contents. **[%VERIFY: DataCo license and PII
  status -- confirm and cite before submission.]**
- **Self-contained?** The benchmark *code/adapter* is self-contained; it
  operates on the public DataCo CSV, which the user obtains from the source.

## Collection process (honest provenance statement)

The underlying records were collected and distributed by DataCo's original
publishers. LogiSCAG does **not** perform primary collection and therefore
cannot authoritatively document the original collection mechanism,
timeframe, sampling frame, or consent basis; we refer users to the source
dataset's documentation **[DataCo, %VERIFY]**. What LogiSCAG fully specifies
and controls is the deterministic **transformation** from the public CSV to
the canonical benchmark, which is the reproducible object of this paper (see
`dataco_adapter.py` / `logiscag.adapters.dataco` for the exact mapping).

## Preprocessing / cleaning / labeling (fully specified -- this is the contribution)

- The adapter maps DataCo columns to the canonical schema; derives `delivery =
  ship + actual_transit` and `promise_delivery = ship + scheduled_transit` (so
  transit duration and SLA buffer are exact identities, not estimates);
  derives the SLA-breach label; and constructs the **leakage-safe feature
  set** by excluding every arithmetic component of the label's source
  quantity (`transit_duration_days`, `shipment_buffer_days`), retaining the
  legitimately-predictive scheduled duration (`promised_transit_days`). This
  exclusion is what `pipeline/data.py`'s `FEATURE_COLS` encodes; see
  `CHANGES.md` for the history of this fix.
- The raw public CSV is never modified; all transformation happens at load
  time and is versioned with the code.

## Uses

- **Intended.** Benchmarking constraint-validated synthetic-data methods;
  developing and comparing constraints, generators, and evaluation metrics;
  reproducing the paper's §5 study.
- **Out of scope / cautions.** Treating the screening-grade privacy metrics
  (DCR, an underpowered-on-this-schema MIA) as formal guarantees; deploying
  synthetic data operationally without organization-specific validation;
  fairness-critical use without an added fairness audit (the four pillars do
  not audit fairness). See `KNOWN_ISSUES.md` for a caveat on what the middle
  strictness tiers (`temporal`/`moderate`/`strict`) currently do and do not
  enforce at the SDV level on this dataset.

## Distribution

- The adapter, constraint library, CAG wrapper, and harness are released
  under **Apache-2.0**. The DataCo CSV itself is fetched by the user from its
  public source under DataCo's terms; LogiSCAG redistributes the raw data
  only if DataCo's license permits **[%VERIFY and state explicitly]**.
  Archival DOI via Zenodo **[placeholder -- not yet minted; see the manual
  publication steps in the Phase-0 packaging report]**.

## Maintenance

- Maintainer and contact **[placeholder]**; semantic versioning (starting at
  `0.1.0`); public issue tracker; stated support window; contributions of new
  dataset adapters and constraint sets explicitly invited (see
  `CONTRIBUTING.md`). Errata and version history published in the repository.
