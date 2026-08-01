"""
Public four-pillar evaluation API: integrity, fidelity, utility, privacy.

Wraps privacy_utility_sweep.py (the sweep harness) and pipeline.metrics /
pipeline.evaluation (the per-metric implementations) without duplicating any
metric logic.

Pillar -> source mapping:
  integrity -> integrity_check_synthetic, audit_constraints (logiscag.constraints)
  fidelity  -> compute_fidelity (KS / JS)
  utility   -> trtr_baseline, tstr_evaluate, business_utility_score (BUS)
  privacy   -> compute_dcr_nnaa (DCR), membership_inference_auc (MIA)

run_sweep / run_mia_audit orchestrate all four pillars across architectures,
strictness levels, and seeds, and significance_between_levels runs the paired
significance tests reported in the paper's tables.
"""

from privacy_utility_sweep import (
    evaluate_synth,
    run_sweep,
    run_mia_audit,
    aggregate,
    significance_between_levels,
    membership_inference_auc,
    plot_frontier,
    plot_mia,
)
from pipeline.evaluation import trtr_baseline, tstr_evaluate
from pipeline.metrics import (
    compute_fidelity,
    compute_dcr_nnaa,
    business_utility_score,
    expected_calibration_error,
    significance_test,
    PRIVACY_FIDELITY_NUM_COLS,
)
from pipeline.constraints import integrity_check_synthetic

__all__ = [
    "evaluate_synth",
    "run_sweep",
    "run_mia_audit",
    "aggregate",
    "significance_between_levels",
    "membership_inference_auc",
    "plot_frontier",
    "plot_mia",
    "trtr_baseline",
    "tstr_evaluate",
    "compute_fidelity",
    "compute_dcr_nnaa",
    "business_utility_score",
    "expected_calibration_error",
    "significance_test",
    "integrity_check_synthetic",
    "PRIVACY_FIDELITY_NUM_COLS",
]
