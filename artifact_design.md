# Artifact Design & Release Plan
### A reusable supply-chain synthetic-data toolkit (working name: **`logiscag`** — rename as you like)

> **Goal restated.** Not "a paper with code attached," but an **adoptable artifact** that supply-chain practitioners reuse, with a paper that documents it. Success is measured in *adoption* (installs, forks, integrations, citations of the tool), not venue prestige alone. The paper is the announcement; the artifact is the contribution.

---

## 1. What it is and who it's for

**One-line value proposition:** *"Generate synthetic last-mile logistics data that is guaranteed operationally valid, with a one-page report card your governance team can read — composing with the synthesizers you already use."*

**Three target users, in priority order:**
1. **Supply-chain ML practitioners** who need shareable synthetic data and currently get operationally-impossible rows from off-the-shelf SDV. They want: drop-in validity + a trustworthy evaluation.
2. **Data-governance / compliance teams** who must sign off on data sharing. They want: a *human-readable* constraint catalog and a pass/fail report card — not a black box.
3. **Researchers** in the supply-chain-AI community (the SynDelay/Brintrup ecosystem). They want: a reproducible benchmark and evaluation harness to compare methods.

The middle user is your differentiator. Most synthetic-data tools speak only to ML engineers; yours speaks to the governance team too, because your constraints are *auditable denial constraints*, not learned weights. That's the adoption wedge.

---

## 2. The five components

### Component A — The constraint library (the durable IP; build this first)
A documented, extensible **catalog of supply-chain denial constraints**, each declarative and human-readable so a compliance officer can author or audit one. This is the part with the longest reuse life — generators change yearly; "delivery cannot precede shipment" does not.

Categories: **temporal** (ordering, positive durations), **referential** (carrier/service vocabulary, ship-node keys), **arithmetic** (SLA-buffer maths, non-negativity, distance sanity), **calendar** (carrier operational days — soft), **cardinality** (label/category domains).

A catalog entry should be declarative — readable without code:

```yaml
- id: R1_temporal_ordering
  name: Temporal ordering
  category: temporal
  type: hard                 # hard | soft
  predicate: "order_ts <= ship_ts <= delivery_ts"
  forbids: "a shipment before its order, or a delivery before its shipment"
  severity: high
  on_violation: reject       # reject | repair | flag
  tolerance_days: 0
  rationale: "Physical event chain; an impossible ordering can never occur in reality."
```

…with a thin Python layer compiling each entry into a checkable predicate and an **extension API** so users register their own:

```python
@constraint(id="custom_cod_limit", category="arithmetic", type="hard")
def cod_amount_nonnegative(row):
    return row["cash_on_delivery"] >= 0
```

Ship it with your eight rules as the seed catalog, generalized so other companies' schemas can adopt them. **This component alone is a useful, citable release** even before the rest exists.

### Component B — The CAG wrapper
Composes with any SDV synthesizer (TVAE/CTGAN/CopulaGAN) + a constraint set → constraint-valid synthetic data. Houses the rejection-sampling engine, the 50-attempt cap, the 80%-rejection convergence guard, and the soft-penalty fallback. Key API shape: `cag.wrap(synthesizer, constraints).sample(n)`. ★ Expose the **repair-vs-reject** switch (`on_violation`) so the future HoloClean-style repair path has a home.

### Component C — The evaluation harness → the "report card"
The four-pillar (utility/fidelity/privacy/integrity) + BUS evaluation, runnable on any logistics table, emitting a **standardized one-page report card**: each pillar, each gate, pass/fail, CVR, and the MIA-audit AUC. This is the governance-facing adoption hook — a practitioner runs it and hands the card to their compliance team. Make the report card a first-class output (HTML + JSON), not buried logs.

### Component D — The benchmark
Public-data adapters (**SynDelay**, **DataCo**) with pre-mapped constraint sets, baseline results, and one-command reproduction. Because these are *existing* public datasets only lightly processed, you reference them by source URL + ship preprocessing code — no dataset hosting/Croissant burden. This is the E&D-track-facing contribution: a reproducible substrate others compare against.

### Component E — The patterns (the reusable industry observations)
The privacy–utility-under-constraint-strictness findings, as **reproducible notebooks** plus a short written report: which features/carriers drive the trade-off, how strictness changes safe-sharing behavior *for logistics data specifically*. Not a general law (that's the deferred theory paper) — concrete, validated, domain patterns a practitioner can act on. This is what makes the work *reusable in the industry* rather than just runnable.

---

## 3. Repository structure

```
logiscag/
├── README.md                  # value prop + 5-minute quickstart on DataCo
├── LICENSE                     # Apache-2.0 (see §6 on the patent tension)
├── pyproject.toml              # pip-installable, minimal deps
├── logiscag/
│   ├── constraints/
│   │   ├── catalog/            # YAML catalog (the seed 8 + categories)
│   │   ├── engine.py           # compile + check + audit
│   │   └── api.py              # @constraint extension decorator
│   ├── cag/
│   │   └── wrapper.py          # rejection engine + guards + repair hook
│   ├── eval/
│   │   ├── pillars.py          # KS/JS, TSTR/BUS, DCR/NNAA, CVR
│   │   ├── mia.py              # membership-inference audit (DOMIAS/TAPAS hook)
│   │   └── report_card.py      # HTML + JSON one-pager
│   ├── benchmark/
│   │   ├── syndelay.py         # adapter + constraint mapping
│   │   └── dataco.py           # adapter (+ real SLA-breach label)
│   └── schema_mapping.py       # map any logistics schema → canonical fields
├── patterns/                   # reproducible notebooks + findings report
├── examples/
│   └── quickstart_dataco.ipynb # end-to-end in minutes
├── docs/                       # constraint-authoring guide, API, governance guide
├── tests/
└── CITATION.cff + croissant.json (only if you release a processed dataset)
```

---

## 4. Design principles that actually drive adoption

1. **Declarative-first.** Constraints are config a non-ML person can read and write. This is the single biggest adoption lever — it opens the tool to governance teams, not just engineers.
2. **A schema-mapping layer.** A small mapping config (`schema_mapping.py`) lets the same constraints apply across *different* companies' logistics schemas. Cross-schema portability is what turns "Avish's Adidas tool" into "the industry's logistics-synthesis tool." Demonstrate it working on three schemas (your corpus + SynDelay + DataCo).
3. **The report card as the product.** Practitioners adopt what produces something they can *show someone*. A clean pass/fail one-pager is more adoptable than any API.
4. **Compose, don't replace.** Wrapping SDV (not forcing a new generator) drops the switching cost to near zero.
5. **Five-minute quickstart on public data.** If `pip install` → first report card takes more than a coffee, adoption dies. Optimize that path ruthlessly.
6. **Reproducible by construction.** Pinned versions, fixed seeds, MLflow logging — so the benchmark numbers are trusted and re-runnable.

---

## 5. Release & adoption plan

**Code hosting:** GitHub, documented and executable (E&D-track-compliant). Tag a versioned release; mint a **Zenodo DOI** so the artifact itself is citable independent of the paper.
**Data:** reference SynDelay/DataCo by source URL + ship preprocessing code; no Croissant needed unless you publish a *new* processed dataset (then host on Hugging Face/Dataverse for auto-Croissant, and include Responsible-AI metadata — a 2026 E&D requirement).
**Docs:** README quickstart, a **constraint-authoring guide** (for governance users), an API reference, and a **contribution guide** (so the community adds constraints — community-authored rules are how the catalog compounds).
**Community engagement:** contribute your harness as an *evaluation layer on SynDelay*, and open an integration PR / issue with the supply-chain-AI group. Being inside that ecosystem's citation graph is worth more than a standalone release.
**Announcement:** the paper is the announcement; pair it with a short blog post and a runnable Colab. Practitioners find tools through Colabs and blog posts, not OpenReview.

**E&D-track compliance checklist (if you target NeurIPS E&D):**
- [ ] Code hosted, documented, executable, anonymized at submission.
- [ ] If a new dataset is released: Croissant + Responsible-AI metadata, on a preferred host.
- [ ] If only existing public data: source URLs + preprocessing code (no hosting/Croissant).
- [ ] 9-page main-text limit (same as main track); appendix unlimited.
- [ ] Single-blind option available via the `preprint` style option.

---

## 6. The decision you must make consciously: open-source vs. patent

Your PhD proposal floated patenting the constraint-audit methodology. **Be clear-eyed: open-sourcing for adoption and patenting are largely mutually exclusive** — a public release is prior disclosure that generally forecloses patentability. You can't maximize industry reuse *and* hold defensive IP on the same core. Given your stated goal (reusable in the industry), **open-source wins** — adoption is the impact, and a widely-used Apache-2.0 tool is more durable career capital for an industry practitioner than an unenforced patent. If you want *some* protection, the licensable know-how (the cost-elicitation protocol, proprietary constraint sets for specific partners) can stay closed while the core engine is open. Decide this *before* the first public commit, because it's irreversible.

(Apache-2.0 recommended over MIT: it includes an explicit patent grant, which enterprises' legal teams prefer — lowering corporate adoption friction.)

---

## 7. Phased build order (realistic for a solo practitioner)

- **Phase 0 — MVP (the releasable core):** Component A (constraint library + catalog + extension API) + Component B (CAG wrapper) + Component C (report card), running on DataCo via one adapter. *This is already a useful, citable open-source release.* Ship it early; iterate in public.
- **Phase 1 — Benchmark:** add the SynDelay adapter, baseline results, the full four-pillar harness, and the MIA audit. Now it's an E&D-track-grade benchmark.
- **Phase 2 — Patterns:** the privacy–utility sweep notebooks + findings report (Component E) — the reusable industry observations.
- **Phase 3 — Polish & publish:** docs, schema-mapping demo across three schemas, Zenodo DOI, Croissant/RAI if needed, and the paper.

Phase 0 is weeks, not months, and it's the part that creates adoption momentum. Don't wait for the full thing to release the constraint library.

---

## 8. One honest caveat

Even the E&D track has reviewers who, per the 2025 author survey, over-weight *methodological novelty over real-world impact*. So frame the contribution as a genuine *evaluation and tooling advance* (the auditable constraint catalog, the governance-facing report card, the cross-schema portability, the documented patterns) — not as "we wrapped SDV with rejection sampling," which invites the novelty objection we already diagnosed. The artifact's value is real; the framing has to make the *reusability and rigor* the headline, since that's what this track is meant to reward.

---
*Companion documents: `manuscript_skeleton.md` (the paper), `cag_study_handbook.md` (the concepts), `cag_references.bib` (citations).*
