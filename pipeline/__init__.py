"""
Presents a stable import facade for `viva_demo.py` and tests.
Re-exports constants and functions from internal modules to minimize import churn and preserve backward compatibility.
"""

from .data import (
    FEATURE_COLS,
    CANONICAL_LABEL_DISTRIBUTION,
    PROXY_LABEL_DISTRIBUTION,
    generate_proxy_dataset,
    load_real_dataset,
    phase1_derived_features,
    encode_features,
    get_XY,
    rank_features_by_shap,
    select_top_features,
    run_preflight_validator,
)
from .constraints import audit_constraints, deterministic_repair, integrity_check_synthetic
from .metrics import (
    ks_complement,
    wasserstein_metric,
    js_divergence_numeric,
    tvd_complement,
    compute_fidelity,
    compute_dcr_nnaa,
    business_utility_score,
    expected_calibration_error,
    significance_test,
)
from .evaluation import (
    trtr_baseline,
    tstr_evaluate,
    smote_baseline,
    transfer_learning_curve,
    xgboost_tstr,
    compute_feature_importance,
)
from .tuning import tune_tvae_optuna, tune_sdv_optuna
from .pipeline import train_sdv_models, run_pipeline
from .sdmetrics_eval import evaluate_sdmetrics_pair, evaluate_sdmetrics_all
from .sdmetrics_reporting import export_sdmetrics_report
