# Run Playbook — from code to confirmed result

Execute in order. Each phase is cheap-to-expensive and de-risks the next. **Do not
write the discussion until Phase 4** — the diagnostics decide what you can claim.

---

## Phase 0 — Verify the code (no SDV, ~2 min)

Drop the three modified files into your repo (`pipeline/constraints.py`,
`pipeline/pipeline.py`, `pipeline/tuning.py`), and `privacy_utility_sweep.py` +
`tests/test_constraint_ladder.py`. Then, equivalently:

```bash
python tests/test_constraint_ladder.py    # expect: none=0 < temporal=4 < moderate=8 < strict=9
python privacy_utility_sweep.py --smoke --mia --seeds 2   # full flow on proxy data
# or, as of the logiscag packaging: python -m logiscag.reproduce --verify
```

**Gate:** the ladder test must pass and the smoke run must write 5 files +
2 PNGs to `outputs/pu_sweep/`. If not, stop — the wiring is wrong, not the science.

---

## Phase 1 — Install SDV + tiny real pilot (the critical de-risk, ~10 min)

```bash
pip install sdv
python privacy_utility_sweep.py --real path/to/your.csv \
    --architectures CopulaGAN --levels none temporal strict \
    --seeds 1 --n-synth 2000 --epochs 20 --out outputs/pilot_tiny
```

**This phase exists to catch ONE thing:** whether SDV accepts the date columns as
`datetime` for the R1 temporal `Inequality` constraints (the only piece I could not
test without SDV).

- **If it runs and `outputs/pilot_tiny/sweep_summary.csv` shows CVR falling from
  `none` -> `temporal` -> `strict`** → the SDV-native ladder works. Proceed.
- **If SDV raises a `datetime_format` / constraint error** → either add an explicit
  `datetime_format` in the metadata update in `train_sdv_models`, OR switch your
  levels to use the rejection path: `--levels none temporal+... ` won't exist, so
  instead run with `--levels none strict strict+reject` and rely on `strict+reject`
  (deterministic post-hoc enforcement) for the full-strictness end. Either way you
  get a working gradient.

---

## Phase 2 — Pilot sweep: is the finding even there? (~30–90 min)

One architecture, real strictness gradient, enough seeds for a trend.

```bash
python privacy_utility_sweep.py --real path/to/your.csv \
    --architectures CopulaGAN \
    --levels none temporal moderate strict strict+reject \
    --seeds 3 --n-synth 10000 --epochs 100 --mia --out outputs/pilot
```

Open `outputs/pilot/privacy_utility_frontier.png` and `mia_vs_strictness.png` and
run the **Phase 4 diagnostics** on this single architecture. If the curve is flat
or wrong here, the full run won't save it — decide before spending the compute.

---

## Phase 3 — Full sweep + data-scaling run (hours; GPU helps; CTGAN is slowest)

**3a. The headline sweep (fills Table 3, Table 4, Figs 1–2):**
```bash
python privacy_utility_sweep.py --real path/to/your.csv \
    --architectures TVAE CTGAN CopulaGAN \
    --levels none temporal moderate strict strict+reject \
    --seeds 5 --n-synth 10000 --epochs 100 --mia --out outputs/full
```

**3b. The data-scaling run (resolves the 0.837-vs-0.9347 `[CONFIRM REGIME]`):**
Run the sweep at two training sizes — your conservative regime and full data — by
varying the data you pass in (e.g. a 10k-row sample vs the full corpus), same flags,
`--out outputs/scale_10k` and `outputs/scale_full`. Compare BUS at `strict` across the two.

> Compute budget: 3 arch × 5 levels × 5 seeds = 75 sweep runs, + 75 for `--mia` = 150
> SDV trainings. At ~1–4 min each that's a few hours. Drop CTGAN or seeds to 3 if tight.

---

## Output → manuscript map

| File | Feeds |
|---|---|
| `sweep_summary.csv` | **Table 3** (four-pillar) and **Table 4** (privacy–utility × strictness), §8–9 |
| `sweep_long.csv` | **Appendix G** (per-strictness full metrics with CIs) |
| `sweep_significance.csv` | the significance claims in **§9** (paired tests between levels) |
| `trtr_baseline.json` | the honest **0.718 ceiling**, §3.4 / §7 |
| `privacy_utility_frontier.png` | **Figure 1** (the headline) |
| `mia_vs_strictness.png` | the **privacy-audit figure**, §7.4 / §9 |
| `scale_10k` vs `scale_full` BUS | **Figure 2** / §8.2, resolves `[CONFIRM REGIME]` |

---

## Phase 4 — Diagnostics: is the result real? (run BEFORE writing §9–10)

Read these off `sweep_summary.csv`. Each has an explicit pass criterion.

**D1 — Ladder is working (prerequisite).**
CVR (mean) decreases monotonically: `none ≥ temporal ≥ moderate ≥ strict`, and
`strict+reject` ≈ 0%. *Fail* → the strictness isn't biting; fix Phase 1 before
interpreting anything else.

**D2 — The privacy effect (your headline).**
As strictness rises: `dcr_mean` falls **and/or** `mia_auc` trends toward 0.50.
*This is the finding.* Pass → constraint enforcement measurably improves privacy.
The MIA trend is the stronger evidence; lead with it.

**D3 — Utility holds or improves (the trade-off must be favorable).**
As strictness rises: `business_utility` rises or holds and `f1_gap` shrinks or holds.
The claim is "privacy improves *while* utility improves" — if utility collapses as
constraints tighten, you have a *trade-off*, not a free lunch, and must reframe.

**D4 — It's significant, not noise.**
In `sweep_significance.csv`, the BUS and DCR (and MIA) differences between `none` and
`strict` have p < 0.05 across seeds. *Fail* → report as directional/underpowered,
not as an effect; consider more seeds.

**D5 — Data-scaling resolves the regime question.**
BUS at full data > BUS at 10k at `strict`, ideally clearing 0.90. Pass → you write the
clean "gates clear on scaling" story and the two documents (paper + proposal) reconcile.

---

## Decision tree — what to write

- **D1–D4 all pass** → you have a main-finding paper. Write §9 around the curve, lead
  with the MIA-audited privacy improvement, claim the favorable trade-off honestly.
- **D2 passes, D3 fails (utility drops)** → reframe as a genuine privacy↔utility
  *trade-off* under constraints (still publishable, less splashy). Do not claim a free lunch.
- **D2 fails (curve flat)** → honest **null result**: operational-validity enforcement
  does not materially move the frontier in this domain. Still reportable (the E&D track
  welcomes negative results), but reconsider whether the privacy angle is the headline
  or whether the auditable-constraint *tooling* becomes the contribution instead.
- **D1 fails** → don't interpret D2–D5 yet; the strictness lever isn't engaged.

**Discipline:** write the claims to match D1–D5, never the reverse. The fastest
desk-reject is a discussion that asserts a favorable trade-off the tables don't show.
Your honest-evaluation framing is an asset — keep it.

---

## One-line readiness check
You are ready to draft §9–10 when you can fill this sentence from the data:
*"As constraints tighten from none to full, MIA-AUC moved from ___ to ___ (p=___),
BUS moved from ___ to ___, and CVR fell from ___% to ___% — [a favorable / a trade-off /
no] privacy–utility relationship."*
