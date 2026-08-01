---
title: 'LogiSCAG: operational-validity enforcement and four-pillar evaluation for synthetic supply-chain data'
tags:
  - Python
  - synthetic data
  - tabular data
  - supply chain
  - logistics
  - data privacy
  - constraint validation
  - evaluation
authors:
  - name: Avish Saha
    orcid: 0009-0001-1124-7119
    affiliation: 1
# TODO: JOSS's "collaborative effort" criterion treats co-authors (advisor,
# domain collaborators) as evidence of collaborative context for an otherwise
# single-author project. Discuss with your BITS supervisor before submitting.
affiliations:
  - name: "[Affiliation — Independent Researcher]"
    index: 1
date: "[submission date]"
bibliography: paper.bib
---

# Summary

Operational supply-chain records — orders, shipments, deliveries, delays — are among the most useful and least shareable data in industry: commercially sensitive, privacy-governed, and often contractually restricted. Synthetic data offers a bridge, but general-purpose tabular generators optimize statistical resemblance with no notion of the *physical and logical rules* of logistics, so they readily emit impossible records: deliveries that precede shipments, negative transit times, carrier/service lanes that do not exist.

**LogiSCAG** is an open-source Python toolkit that makes operational validity a first-class, auditable property of synthetic logistics data. It provides (i) a **declarative constraint catalog** of last-mile rules — temporal ordering, positive transit, referential carrier integrity, label–SLA consistency — expressed as single-tuple denial constraints in YAML and extensible via a `@constraint` decorator; (ii) a generator-agnostic **rejection wrapper** enforcing any subset of those rules on any fitted synthesizer's output, with resample-to-target sampling and first-class reporting of rejection overhead; (iii) a **four-pillar evaluation harness** scoring *integrity* (constraint-violation rate), *fidelity* (distributional distances), *utility* (train-on-synthetic-test-on-real with a minority-weighted Business Utility Score and a leakage-safe feature protocol), and *privacy* (distance-to-closest-record plus a membership-inference audit whose attack power is validated before any null result is reported); and (iv) a documented **adapter and reproducible benchmark** on the public DataCo dataset [@dataco] (180,519 orders), so the pipeline runs end-to-end without privileged data access.

LogiSCAG is implemented in Python (≥3.10), released under the Apache-2.0 license, installable via `pip`, and ships with a test suite, continuous integration, a datasheet in the format of @gebru2021datasheets, and single-command reproduction of its benchmark results.

# Statement of need

Two gaps recur for anyone generating synthetic tabular logistics data. First, *operational validity is not measured*: widely used evaluation suites score fidelity, utility, and privacy, but none treats violation of domain rules — records that could not physically occur — as a first-class axis, even though such records silently undermine the operational tasks synthetic data is meant to support. Second, *evaluation itself fails silently*: developing LogiSCAG surfaced several failure modes that inflate results without raising errors — arithmetic label leakage yielding degenerate utility ceilings, constraint specifications a generator quietly ignores, and feature-pipeline collapse reducing a utility model to a single predictor. LogiSCAG packages both the missing axis and the guards: a preflight leakage validator, a warned (never silent) fallback when a referential vocabulary is absent, regression-tested feature-width checks, and a membership-inference audit power-validated against a deliberately leaky baseline before its verdicts are trusted.

The intended users are (a) industry data teams who must release or exchange operationally plausible synthetic logistics data under governance constraints, and (b) researchers benchmarking tabular generators on supply-chain tasks who need integrity, leakage-safe utility, and audited privacy measured reproducibly.

# State of the field

The Synthetic Data Vault [@patki2016sdv] and its CTGAN/TVAE synthesizers [@xu2019ctgan] are the de-facto generation stack, and SDV ships its own constraint feature for generation-time enforcement [@sdvcag]. LogiSCAG is complementary: an *evaluation-and-audit layer* that is generator-agnostic, applying to the output of any synthesizer, with a catalog that is declarative and independently inspectable rather than embedded in one generator's API. SynthEval [@syntheval] evaluates utility and privacy for tabular synthetic data but has no operational-integrity pillar and performs no enforcement; TAPAS [@tapas] provides adversarial privacy auditing only. TabKG [@tabkg2026] addresses the adjacent problem of *generating* logically consistent supply-chain data via LLM-driven knowledge-graph reasoning; LogiSCAG differs in role — it audits validity on any generator's output across all four pillars — and can serve as an independent harness for methods such as TabKG. HoloClean [@rekatsinas2017holoclean] pioneered constraint-based repair of existing data; LogiSCAG addresses the inverse problem in generated data.

Contributing these capabilities upstream to SDV was considered and rejected. SDV's constraint feature is necessarily coupled to SDV's own fitting and sampling loop, whereas the central claim here is that operational validity must be auditable on *any* generator's output — including generators that offer no constraint API at all, and including outputs whose provenance is unknown to the auditor. An audit layer that lives inside one generator cannot make that claim.

# Software design

Three design decisions shape the toolkit, each trading something away.

**Delegation over re-implementation in the catalog.** The YAML catalog exists so a compliance reviewer can audit each rule without reading Python. The obvious implementation — a predicate expression per entry, evaluated by the engine — was rejected: it creates two implementations of every rule that can silently diverge, precisely the failure mode the catalog exists to expose. Instead each entry carries an `audit_key` naming a key in the executable audit's output, and the engine resolves counts through it. The cost is that the catalog cannot describe a rule the code does not implement; the benefit is that a description structurally cannot drift from the check it describes. Custom `@constraint` registrations are the deliberate exception, evaluated row-wise because no prior implementation exists to delegate to.

**Post-hoc rejection as the enforcement mechanism of record.** Generation-time constraints are more efficient, but they are generator-specific and — as this project found the hard way — can be silently dropped by a dependency's API change without any error surfacing. Post-hoc rejection filtering is slower and discards samples, but it is generator-agnostic, independently verifiable, and its cost is observable: rejection overhead is reported as a first-class result rather than hidden. The sampling loop resamples to the target row count rather than bootstrap-padding a surviving subset, since padding would inflate apparent fidelity by duplicating exactly the rows that passed.

**Warned degradation rather than silent fallback or hard failure.** Referential-integrity checking needs a vocabulary of valid carrier/service combinations derived from real training data. Failing hard when a caller omits it would make the audit unusable where the training set is unavailable; falling back quietly would report a weaker presence-only check under the same rule name. The library degrades to the weaker check and warns on every call — raised inside a local `catch_warnings` block so an ambient filter elsewhere in the package cannot suppress it. Where a guarantee cannot be met, the library narrows it loudly.

These choices matter because the output is an *audit* — a claim about what a dataset does and does not violate. An audit that can silently weaken itself is worse than none, because it is trusted.

# Research impact

<!-- TODO — this section is REQUIRED and JOSS is explicit that the evidence must
     be "compelling and specific, not aspirational". Statements about potential
     future use are marked "not acceptable". Fill in with what is concrete:

     - The master's dissertation this work was developed for: add a full
       citation (paper.bib has a stub, `saha2026dissertation`) including the
       institutional repository URL/handle once deposited. An examined,
       deposited dissertation is realized impact and the strongest single piece
       of evidence available here.
     - The cross-dataset study: state the headline benchmark result concretely
       (Table 1 numbers) rather than describing it in the abstract.
     - Any presentation, viva, internal adidas use, or request from another
       group — each is a "community-readiness signal" JOSS names explicitly.
     - Do NOT cite the p-value from Sec 5.3 until it has been regenerated from
       the corrected pipeline (0.98 vs the 0.84 currently written). -->

LogiSCAG was developed for, and is demonstrated by, a cross-dataset study of constraint-validated synthetic delivery-delay data: the full pipeline runs reproducibly on the public DataCo benchmark [@dataco] and was additionally applied (aggregate results only) to a proprietary ~230k-record last-mile corpus. The study surfaced the silent evaluation failures now guarded against in the toolkit and found that a distance-based privacy effect significant on the public benchmark did not replicate on the proprietary corpus — evidence of the dataset-dependence of common privacy proxies and of the need for exactly this kind of audited, cross-dataset evaluation.

The benchmark is reproducible from a single command on data anyone can obtain, and the repository ships the datasheet, run playbook, and tracked known-issues register needed to interpret the results.

# AI usage disclosure

<!-- TODO — this section is REQUIRED and must be complete and accurate. JOSS
     states that failure to provide a complete and accurate disclosure "may be
     considered an ethical breach", with consequences up to desk rejection and
     notification of institutions and funders. Do not under-report. The
     required elements are:

       1. TOOLS: which tools/models, and their versions, and WHERE each was
          used — code, paper text, documentation. Be specific (e.g. "Claude
          Code (Claude Opus 4.x) for X; ChatGPT for Y").
       2. NATURE AND SCOPE: code generation, refactoring, test scaffolding,
          copy-editing, drafting — say which, for which parts.
       3. CONFIRMATION OF REVIEW: an explicit assertion that you reviewed,
          edited, and validated all AI-assisted output and made the core
          design decisions yourself.

     Draft below is a SKELETON with placeholders. Replace every [bracketed]
     span with what actually happened. Delete this comment before submitting. -->

Generative AI tools were used in the preparation of this software and paper, as follows.

**Tools and where they were used.** [TOOL AND VERSION] was used for [WHICH PARTS OF THE CODE — e.g. packaging the public API surface, scaffolding tests, refactoring]. [TOOL AND VERSION] was used for [WHICH DOCUMENTATION — e.g. drafting README/CONTRIBUTING prose]. [TOOL AND VERSION] was used for [PAPER TEXT — e.g. drafting and copy-editing sections]. [State explicitly any part where no AI tools were used — for example, if the research design, the constraint specification, or the experimental protocol were developed without AI assistance, say so.]

**Nature and scope of assistance.** [Describe concretely: e.g. "code generation for the public-API facade modules; test scaffolding for the constraint-catalog suite; refactoring of the warning-filter handling; drafting and copy-editing of documentation and of this paper. The research question, the eight-rule constraint specification, the four-pillar evaluation design, the experimental protocol, and all interpretation of results are the author's own."]

**Verification.** All AI-assisted output was reviewed, edited, and validated by the author. [Describe how — e.g. "Generated code was verified against the test suite, which runs in CI on Python 3.10–3.12; the constraint catalog was cross-checked entry-by-entry against the executable audit; known divergences between code and paper were tracked in `KNOWN_ISSUES.md` rather than silently resolved."] The core design decisions described in the Software design section are the author's own.

# Acknowledgements

[TODO: acknowledge as appropriate — colleagues, supervisor, institutional support. JOSS also requires a statement of financial support; if this work was supported by adidas India or BITS Pilani, state it here.]

# References
