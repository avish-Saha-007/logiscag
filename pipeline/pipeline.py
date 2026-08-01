"""
Implements the main Viva pipeline, orchestrating data loading, preprocessing, synthetic data generation, evaluation, 
tuning, and reporting.
This module defines the `run_pipeline` function, which serves as the central entry point for executing the entire workflow. 
It also includes a `train_sdv_models` function that encapsulates the logic for training synthetic data generation models 
using the SDV library, with support for TVAE, CTGAN, and CopulaGAN architectures. 
The pipeline is designed to be flexible and configurable, allowing users to specify various parameters and options to 
tailor the execution to their needs. Key features include:
- Data loading and preprocessing, including feature engineering and validation.
- Synthetic data generation with configurable model architectures and training parameters.
- Evaluation of synthetic data quality using fidelity, privacy, and utility metrics.
- Hyperparameter tuning using Optuna to optimize synthetic data generation for downstream predictive performance.
- Comprehensive reporting and visualization of results.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd

from .data import (
    generate_proxy_dataset,
    load_real_dataset,
    phase1_derived_features,
    run_preflight_validator,
    CANONICAL_LABEL_DISTRIBUTION,
    PROXY_LABEL_DISTRIBUTION,
    FEATURE_COLS,
    RAW_CATEGORICAL_COLS,
    select_top_features,
)
from .constraints import audit_constraints, deterministic_repair, build_sdv_constraints
from .metrics import compute_fidelity, compute_dcr_nnaa, significance_test, qualitative_sanity_check
from .evaluation import (
    trtr_baseline,
    tstr_evaluate,
    smote_baseline,
    transfer_learning_curve,
    xgboost_tstr,
    compute_feature_importance,
)
from .tuning import tune_tvae_optuna, tune_sdv_optuna
from .reporting import print_final_report, export_report
from .sdmetrics_eval import evaluate_sdmetrics_all
from .sdmetrics_reporting import export_sdmetrics_report
from .viz import plot_results

# Scoped warning suppression (narrowed 2026-06-30 from a blanket
# warnings.filterwarnings("ignore") -- see KNOWN_ISSUES.md and CHANGES.md for
# why). Each rule targets a specific, already-catalogued third-party noise
# source by module and/or message, verified empirically against a real run.
#
# Deliberately NOT suppressed: sdv.cag's FutureWarning about dict-style
# constraints being silently ignored. That warning is a correctness signal
# (see KNOWN_ISSUES.md finding 1 -- it's the root-cause warning behind the
# SDV-native-constraint-no-op bug), not cosmetic noise, and must keep
# surfacing during normal runs. Scoped to sdv.single_table.* specifically
# (not all of sdv.*) so it cannot accidentally catch sdv.cag.* too.
warnings.filterwarnings("ignore", category=FutureWarning, module=r"sdv\.single_table\..*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"sdv\.single_table\..*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"sklearn\..*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"matplotlib\..*")
warnings.filterwarnings(
    "ignore", category=FutureWarning,
    message=r"DataFrameGroupBy\.apply operated on the grouping columns",
)
np.random.seed(42)

# MPS patch for Rosetta x86_64 processes on Apple Silicon.
# ctgan's _set_device gates MPS on platform.machine() == 'arm64', which is
# False for Python running under Rosetta. MPS is accessible to Rosetta
# processes via Metal regardless of process ISA, so the guard is overly
# conservative. Patch it out when MPS is confirmed available.
try:
    import platform as _platform
    import torch as _torch
    if (
        _platform.system() == "Darwin"
        and _platform.machine() != "arm64"
        and getattr(_torch.backends, "mps", None)
        and _torch.backends.mps.is_available()
    ):
        from ctgan.synthesizers import _utils as _ctgan_utils

        def _mps_force_device(enable_gpu, device=None):
            if device:
                return _torch.device(device)
            return _torch.device("mps") if enable_gpu else _torch.device("cpu")

        _ctgan_utils._set_device = _mps_force_device

        # gumbel_softmax NaN fallback: MPS occasionally produces all-NaN results
        # under GPU memory pressure (all 10 built-in retries fail). Fall back to
        # CPU for that one call and move the result back to the MPS device.
        import ctgan.synthesizers.ctgan as _ctgan_mod
        import torch.nn.functional as _F
        import torch as _t

        @staticmethod
        def _gumbel_softmax_mps_safe(logits, tau=1, hard=False, eps=1e-10, dim=-1):
            for _ in range(10):
                t = _F.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
                if not _t.isnan(t).any():
                    return t
            # All MPS attempts produced NaN — retry on CPU, return to original device.
            cpu = _F.gumbel_softmax(
                logits.cpu().float(), tau=tau, hard=hard, eps=eps, dim=dim)
            return cpu.to(logits.device)

        _ctgan_mod.CTGAN._gumbel_softmax = _gumbel_softmax_mps_safe
        print("[pipeline] MPS device patch applied (Rosetta x86_64 + Apple Silicon)"
              " + gumbel_softmax NaN fallback")
except Exception:
    pass  # never crash the import; CPU fallback is fine

SDV_TRAIN_COLS = [
    # date_order_created_omni removed: known 1-sec OMS/DPE clock-skew makes it
    # unreliable for synthesis; date_order_capture is the actionable event timestamp.
    "date_order_capture",
    "order_no",
    "dpe_id",
    "item_id",
    "line_type",
    "dpe_shipnode",
    "last_shipnode",
    "last_scac",
    "last_shipment_carrier_service",
    "date_ship_last_shipment",
    "date_delivery_last_shipment",
    "date_promise_delivery",
    "date_promise_shipment",
    "carrier_service_code",
    "order_type",
    # capture_latency_days removed: derived from date_order_created_omni (dropped above).
    "distance_km",          # Haversine km from DC to customer zip; 0.0 when zip absent
    "transit_duration_days",
    "promised_transit_days",
    "shipment_buffer_days",
    "sla_buffer",
    "delay_label",
]


def _sample_until_valid(draw_fn, n_synth, post_filter, max_oversample_factor=8, label="",
                         label_col="delay_label", label_value=2):
    """Keep drawing fresh batches from draw_fn (e.g. a fitted synthesizer's .sample, or a
    mock bootstrap-and-noise callable) and running them through post_filter, accumulating
    valid rows until n_synth have survived -- instead of drawing once and bootstrap-padding
    a too-small survivor set, which only duplicates whatever rows happened to pass rather
    than adding genuine new diversity (this matters most for under-represented classes).

    draw_fn(num_rows) -> DataFrame of num_rows freshly generated rows.
    Stops at max_oversample_factor * n_synth total draws (graceful fallback: returns
    however many valid rows were collected, with a warning, rather than looping forever).

    Records survival-rate and label_value (minority-class) stability stats on the
    returned DataFrame's .attrs['oversample_stats'] -- a side channel that doesn't
    change the DataFrame's shape/columns, so callers that don't care can ignore it.
    """
    collected = []
    survived = 0
    drawn = 0
    raw_label2_count = 0
    cap = n_synth * max_oversample_factor
    draw_size = n_synth
    last_batch = None
    while survived < n_synth and drawn < cap:
        draw_size = min(draw_size, cap - drawn)
        if draw_size <= 0:
            break
        batch = draw_fn(draw_size)
        last_batch = batch
        drawn += draw_size
        if label_col in batch.columns:
            raw_label2_count += int((pd.to_numeric(batch[label_col], errors="coerce") == label_value).sum())
        valid = post_filter(batch)
        if len(valid):
            collected.append(valid)
            survived += len(valid)
        remaining = n_synth - survived
        if remaining <= 0:
            break
        survival_rate = max(survived / drawn, 0.02)
        draw_size = int(remaining / survival_rate) + 1

    result = pd.concat(collected, ignore_index=True) if collected else last_batch.iloc[0:0].copy()
    cap_hit = survived < n_synth
    if cap_hit:
        print(f"  [warn] {label}: oversample cap reached, kept {survived}/{n_synth} "
              f"valid rows after {drawn} draws")
        result = result.reset_index(drop=True)
    else:
        result = result.iloc[:n_synth].reset_index(drop=True)

    stats = {
        "n_synth": n_synth, "drawn": drawn, "survived": survived,
        "survival_rate": round(survived / drawn, 4) if drawn else float("nan"),
        "cap_hit": cap_hit,
        "raw_label2_rate": round(raw_label2_count / drawn, 4) if drawn else float("nan"),
        "kept_label2_n": int((pd.to_numeric(result[label_col], errors="coerce") == label_value).sum())
                         if label_col in result.columns else None,
    }
    result.attrs["oversample_stats"] = stats
    return result


def _mock_synthesise(real_df, n, architectures=None, post_filter=None):
    """Generates mock synthetic datasets by adding noise to real data. Used when SDV is not available.

	- **Functional**: Provides approximate synthetic alternatives when SDV is not installed.
	- **Technical**: Bootstraps real rows with architecture-specific numeric noise patterns to mimic multiple generator behaviors.


    - **Functional**: Creates synthetic datasets by sampling from the real data and adding Gaussian noise to numeric features.
    - **Technical**: For each architecture (TVAE, CopulaGAN, CTGAN), samples with replacement from the real dataframe, adds normally distributed noise scaled to the feature's standard deviation, and applies specific clipping/rounding for the 'sla_buffer' feature in the CTGAN mock.
    Args:
        real_df (pd.DataFrame): The real dataset to base the synthetic data on.
        n (int): The number of synthetic samples to generate for each architecture.
        post_filter: optional callable(df) -> df; when given, draws are oversampled
            (see _sample_until_valid) until n rows survive the filter.
    Returns:
        dict: A dictionary containing mock synthetic datasets for each architecture.
    """
    requested = architectures or ["TVAE", "CopulaGAN", "CTGAN"]
    synths = {}
    rng = np.random.RandomState(42)
    for arch, noise in [("TVAE", 0.03), ("CopulaGAN", 0.02), ("CTGAN", 0.15)]:
        if arch not in requested:
            continue

        def draw(num_rows, noise=noise, arch=arch):
            mock = real_df.sample(n=num_rows, replace=True,
                                   random_state=rng.randint(0, 2**31 - 1)).copy()
            mock.index = range(num_rows)
            for col in ["capture_latency_days", "transit_duration_days", "promised_transit_days", "shipment_buffer_days", "sla_buffer"]:
                if col in mock.columns:
                    std = real_df[col].std()
                    mock[col] = mock[col] + rng.normal(0, noise * std, num_rows)
            if arch == "CTGAN":
                if "capture_latency_days" in mock.columns:
                    mock["capture_latency_days"] = mock["capture_latency_days"].clip(lower=0)
                if "transit_duration_days" in mock.columns:
                    mock["transit_duration_days"] = mock["transit_duration_days"].clip(lower=0.01)
                if "promised_transit_days" in mock.columns:
                    mock["promised_transit_days"] = mock["promised_transit_days"].clip(lower=0.01)
            return mock

        if post_filter is None:
            synths[arch] = draw(n)
        else:
            synths[arch] = _sample_until_valid(draw, n, post_filter, label=arch)
    return synths


def train_sdv_models(
    canonical_df,
    epochs=100,
    n_synth=10000,
    batch_size=500,
    tvae_l2scale=1e-5,
    constraint_level="strict",
    tvae_params=None,
    ctgan_params=None,
    copulagan_params=None,
    architectures=None,
    post_filter=None,
    max_oversample_factor=8,
):
    """Trains SDV synthesizer models (TVAE, CTGAN, CopulaGAN) on the provided canonical dataframe with specified hyperparameters and constraints.

	- **Functional**: Trains SDV synthesizers and returns generated datasets for downstream evaluation.
	- **Technical**: Detects metadata, enforces integer/categorical typing, configures constraint sets, builds per-architecture synthesizers (TVAE/CopulaGAN/CTGAN), trains/samples each model, and falls back to mock mode if SDV import fails.

    Args:
        canonical_df (pd.DataFrame): The canonical dataframe to train the synthesizers on.
        epochs (int): Number of training epochs.
        n_synth (int): Number of synthetic samples to generate.
        batch_size (int): Batch size for training.
        tvae_l2scale (float): L2 regularization scale for TVAE.
        constraint_level (str): Level of constraints to apply ("none", "moderate", "strict").
        tvae_params (dict): Additional parameters for TVAE.
        ctgan_params (dict): Additional parameters for CTGAN.
        copulagan_params (dict): Additional parameters for CopulaGAN.
        post_filter: optional callable(df) -> df (e.g. cag_rejection_filter). When given,
            each architecture's sample is drawn from the fitted synthesizer in a loop
            (see _sample_until_valid) until n_synth rows pass the filter, rather than
            drawing once and padding a too-small survivor set by bootstrap duplication.
        max_oversample_factor (int): cap on total rows drawn per architecture, as a
            multiple of n_synth, before giving up and returning a short result.
    Returns:
        dict: A dictionary containing trained synthesizers and their synthetic samples.
    """
    try:
        from sdv.single_table import TVAESynthesizer, CTGANSynthesizer, CopulaGANSynthesizer
        from sdv.metadata import SingleTableMetadata
        sdv_available = True
    except ImportError:
        sdv_available = False

    requested_arches = architectures or ["TVAE", "CopulaGAN", "CTGAN"]

    if not sdv_available:
        print("SDV not installed; using mock synthesiser.")
        return _mock_synthesise(canonical_df, n_synth, architectures=requested_arches,
                                 post_filter=post_filter)

    train_cols = [c for c in SDV_TRAIN_COLS if c in canonical_df.columns]
    train_df = canonical_df[train_cols].copy()
    # Drop columns that are entirely NaN — SDV's transformer cannot handle them
    # and would crash. This happens for promise-date derived columns when the dataset
    # does not contain date_promise_delivery / date_promise_shipment.
    all_null_cols = [c for c in train_df.columns if train_df[c].isna().all()]
    if all_null_cols:
        train_df = train_df.drop(columns=all_null_cols)

    # Cap SDV training data at 200 k rows to bound memory usage of CTGAN's internal
    # parallel VGM transformer. Stratified by delay_label to preserve class proportions.
    _MAX_SDV_ROWS = 200_000
    if len(train_df) > _MAX_SDV_ROWS:
        if "delay_label" in train_df.columns:
            train_df = (
                train_df.groupby("delay_label", group_keys=False)
                .apply(lambda g: g.sample(
                    n=int(_MAX_SDV_ROWS * len(g) / len(train_df)),
                    random_state=42,
                ))
                .reset_index(drop=True)
            )
        else:
            train_df = train_df.sample(n=_MAX_SDV_ROWS, random_state=42).reset_index(drop=True)

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_df)

    for col in ["transit_duration_days", "promised_transit_days", "shipment_buffer_days", "sla_buffer", "distance_km"]:
        if col in train_df.columns:
            metadata.update_column(col, sdtype="numerical")
    if "delay_label" in train_df.columns:
        metadata.update_column("delay_label", sdtype="categorical")
    # SDV's auto-detector flags any "*_id"-named column as sdtype="id" by name alone,
    # regardless of content -- it misreads dpe_id/item_id (real-world category labels,
    # not unique identifiers) and then crashes trying to average them as numerics.
    # Force the actual categoricals back to "categorical" to override that heuristic.
    for col in RAW_CATEGORICAL_COLS:
        if col in train_df.columns:
            metadata.update_column(col, sdtype="categorical")
    # Temporal columns must be typed datetime for the R1 ordering (Inequality)
    # constraints in the strictness ladder to apply.
    for col in ["date_order_capture", "date_ship_last_shipment", "date_delivery_last_shipment",
                "date_promise_shipment", "date_promise_delivery", "date_order_created_omni"]:
        if col in train_df.columns:
            try:
                metadata.update_column(col, sdtype="datetime")
            except Exception:
                pass  # keep detected type if SDV needs an explicit datetime_format

    # Graded, cumulative constraint ladder (none -> temporal -> moderate -> strict).
    constraints = build_sdv_constraints(train_df, constraint_level)

    tvae_kwargs = dict(
        epochs=epochs,
        batch_size=batch_size,
        embedding_dim=128,
        compress_dims=(256, 256),
        decompress_dims=(256, 256),
        l2scale=tvae_l2scale,
        enforce_min_max_values=True,
    )
    if tvae_params:
        tvae_kwargs.update(tvae_params)
        tvae_kwargs["enforce_min_max_values"] = True

    ctgan_kwargs = dict(epochs=epochs, batch_size=batch_size, enforce_min_max_values=True)
    if ctgan_params:
        ctgan_kwargs.update(ctgan_params)
        ctgan_kwargs["enforce_min_max_values"] = True

    copulagan_kwargs = dict(epochs=epochs, batch_size=batch_size, enforce_min_max_values=True)
    if copulagan_params:
        copulagan_kwargs.update(copulagan_params)
        copulagan_kwargs["enforce_min_max_values"] = True

    synths = {}
    configs = [
        ("TVAE", TVAESynthesizer, tvae_kwargs),
        ("CopulaGAN", CopulaGANSynthesizer, copulagan_kwargs),
        ("CTGAN", CTGANSynthesizer, ctgan_kwargs),
    ]

    for name, synth_class, kwargs in configs:
        if name not in requested_arches:
            continue
        print(f"Training {name}...")
        t0 = time.time()
        synth = synth_class(metadata, **kwargs)
        if constraints:
            synth.add_constraints(constraints=constraints)
        synth.fit(train_df)
        if post_filter is None:
            synths[name] = synth.sample(num_rows=n_synth, batch_size=1000)
        else:
            draw = lambda num_rows: synth.sample(num_rows=num_rows, batch_size=1000)
            synths[name] = _sample_until_valid(draw, n_synth, post_filter,
                                               max_oversample_factor=max_oversample_factor,
                                               label=name)
        print(f"{name} done in {time.time() - t0:.1f}s")

    return synths


def run_pipeline(
    phases="all",
    demo_mode=False,
    real_data_path=None,
    export_report_dir=None,
    skip_preflight=False,
    tuning_profile="balanced",
    epochs_override=None,
    n_synth_override=None,
    sample_parity=False,
    constraint_level=None,
    tvae_l2scale=None,
    batch_size=500,
    feature_selection_method="none",
    feature_top_k=None,
    feature_min_score=None,
    validation_strategy="temporal",
    optuna_tune_tvae=False,
    optuna_tune_all_sdv=False,
    optuna_trials=20,
    optuna_timeout_sec=0,
    optuna_weight_fidelity=0.25,
    optuna_weight_utility=0.25,
    optuna_weight_privacy=0.25,
    optuna_weight_integrity=0.25,
    optuna_min_tstr_f1=None,
    optuna_min_business_utility=None,
    optuna_min_ks_mean=None,
    optuna_max_js_mean=None,
    optuna_min_dcr_mean=None,
    optuna_search_space="standard",
    preflight_ratio_tolerance=None,
    random_state=42,
    enable_sdmetrics=False,
    sdmetrics_sample_size=None,
    sdmetrics_include_column_pairs=None,
    config_overrides=None,
    sdv_architectures=None,
):
    """Runs the full Viva pipeline with configurable options for each phase, including data loading, preprocessing, synthetic data generation, evaluation, tuning, and reporting.   

	- **Functional**: Trains SDV synthesizers and returns generated datasets for downstream evaluation.
	- **Technical**: Detects metadata, enforces integer/categorical typing, configures constraint sets, builds per-architecture synthesizers (TVAE/CopulaGAN/CTGAN), trains/samples each model, and falls back to mock mode if SDV import fails.

    Args:
        phases (str): Which phases to run ("all" or comma-separated list of phase numbers).
        demo_mode (bool): Whether to run in demo mode with smaller datasets and fewer epochs.
        real_data_path (str): Path to real dataset CSV file. If None, uses proxy    dataset.
        export_report_dir (str): Directory to export the final report. If None, does not export.
        skip_preflight (bool): Whether to skip the preflight validation phase.
        tuning_profile (str): Tuning profile to use ("balanced", "realism", "privacy").
        epochs_override (int): Override number of training epochs for SDV models.
        n_synth_override (int): Override number of synthetic samples to generate.
        sample_parity (bool): Whether to enforce sample parity between real and synthetic datasets.
        constraint_level (str): Level of constraints to apply during SDV training ("none", "moderate", "strict").
        tvae_l2scale (float): L2 regularization scale for TVAE training.
        batch_size (int): Batch size for SDV model training.
        feature_selection_method (str): Method for feature selection ("none", "shap").
        feature_top_k (int): If using SHAP feature selection, the top-k features to select.
        feature_min_score (float): If using SHAP feature selection, the minimum importance score threshold to select features.
        validation_strategy (str): Strategy for validation split ("temporal", "stratified").
        optuna_tune_tvae (bool): Whether to perform Optuna tuning for TVAE hyperparameters.
        optuna_tune_all_sdv (bool): Whether to perform Optuna tuning for all SDV architectures (TVAE, CTGAN, CopulaGAN).
        optuna_trials (int): Number of Optuna trials to run for tuning.
        optuna_timeout_sec (int): Optional timeout in seconds for Optuna tuning. 0 means no timeout.
        optuna_weight_fidelity (float): Weight for the fidelity component in the Optuna composite score.
        optuna_weight_utility (float): Weight for the utility component in the Optuna composite score.
        optuna_weight_privacy (float): Weight for the privacy component in the Optuna composite score.
        optuna_weight_integrity (float): Weight for the integrity component in the Optuna composite score.
        optuna_min_tstr_f1 (float): Minimum TSTR weighted F1 score for a trial to be considered feasible during Optuna tuning.
        optuna_min_business_utility (float): Minimum business utility score for a trial to be considered feasible during Optuna tuning.
        optuna_min_ks_mean (float): Minimum KS mean score for a trial to be considered feasible during Optuna tuning.
        optuna_max_js_mean (float): Maximum JS mean score for a trial to be considered feasible during Optuna tuning.
        optuna_min_dcr_mean (float): Minimum DCR mean score for a trial to be considered feasible during Optuna tuning.
        optuna_search_space (str): Search space profile for Optuna tuning ("standard", "wide").
        preflight_ratio_tolerance (float): Override for class ratio tolerance in preflight validation.
        enable_sdmetrics (bool): Whether to run SDMetrics secondary validation layer.
        sdmetrics_sample_size (int): Optional row cap per dataset for SDMetrics evaluation.
        sdmetrics_include_column_pairs (bool): Whether to compute SDMetrics column-pair trend details.
        config_overrides (dict): Optional dictionary to override configuration values from YAML profiles.
    Returns:
        dict: A dictionary containing results from the executed phases of the pipeline.
    """
    np.random.seed(int(random_state))

    n = 2000 if demo_mode else 10000
    profile_defaults = {
        "balanced": {"epochs": 20 if demo_mode else 100, "constraint_level": "strict", "tvae_l2scale": 1e-5},
        "realism": {"epochs": 40 if demo_mode else 200, "constraint_level": "moderate", "tvae_l2scale": 1e-6},
        "privacy": {"epochs": 40 if demo_mode else 200, "constraint_level": "strict", "tvae_l2scale": 5e-5},
    }
    defaults = profile_defaults.get(tuning_profile, profile_defaults["balanced"]).copy()

    cfg = config_overrides if isinstance(config_overrides, dict) else {}
    cfg_profiles = cfg.get("profiles", {}) if isinstance(cfg.get("profiles", {}), dict) else {}
    cfg_profile = cfg_profiles.get(tuning_profile, {}) if isinstance(cfg_profiles.get(tuning_profile, {}), dict) else {}
    cfg_model_params = cfg_profile.get("model_params", {}) if isinstance(cfg_profile.get("model_params", {}), dict) else {}

    if "epochs" in cfg_model_params:
        defaults["epochs"] = int(cfg_model_params["epochs"])
    if "constraint_level" in cfg_profile:
        defaults["constraint_level"] = str(cfg_profile["constraint_level"])
    if "tvae_l2scale" in cfg_model_params:
        defaults["tvae_l2scale"] = float(cfg_model_params["tvae_l2scale"])

    epochs = defaults["epochs"] if epochs_override is None else int(epochs_override)
    constraint_level_effective = defaults["constraint_level"] if constraint_level is None else constraint_level
    tvae_l2scale_effective = defaults["tvae_l2scale"] if tvae_l2scale is None else tvae_l2scale

    sample_parity_effective = bool(sample_parity)
    if not sample_parity_effective and "sample_parity" in cfg_profile:
        sample_parity_effective = bool(cfg_profile.get("sample_parity"))

    cfg_preflight = cfg.get("preflight", {}) if isinstance(cfg.get("preflight", {}), dict) else {}
    cfg_sdmetrics = cfg.get("sdmetrics", {}) if isinstance(cfg.get("sdmetrics", {}), dict) else {}
    ratio_tolerance = (
        float(preflight_ratio_tolerance)
        if preflight_ratio_tolerance is not None
        else float(cfg_preflight.get("drift_tolerance", 0.03))
    )

    enable_sdmetrics_effective = bool(enable_sdmetrics) or bool(cfg_sdmetrics.get("enabled", False))
    sdmetrics_sample_size_effective = int(sdmetrics_sample_size) if sdmetrics_sample_size is not None else int(cfg_sdmetrics.get("sample_size", 2000))
    sdmetrics_include_pairs_effective = (
        bool(sdmetrics_include_column_pairs)
        if sdmetrics_include_column_pairs is not None
        else bool(cfg_sdmetrics.get("include_column_pairs", True))
    )

    n_synth = 2000 if demo_mode else 10000
    if n_synth_override is not None:
        n_synth = int(n_synth_override)
    max_phase = 4 if (phases == "all" or phases == 0) else int(phases)

    if real_data_path:
        df = load_real_dataset(real_data_path)
    else:
        df = generate_proxy_dataset(n=n)

    df = phase1_derived_features(df)

    if not skip_preflight and not demo_mode:
        expected_dist = CANONICAL_LABEL_DISTRIBUTION if real_data_path else PROXY_LABEL_DISTRIBUTION
        run_preflight_validator(df, expected_distribution=expected_dist, ratio_tolerance=ratio_tolerance)

    if sample_parity_effective:
        n_synth = len(df)

    if max_phase < 2:
        return {"real_df": df}

    constraint_report = audit_constraints(df, verbose=True)
    df = deterministic_repair(df)

    if max_phase < 3:
        return {"real_df": df, "constraint_report": constraint_report}

    selected_feature_cols = [c for c in FEATURE_COLS if c in df.columns or c.endswith("_enc")]
    feature_selection = {
        "method": "none",
        "selected_features": selected_feature_cols,
        "ranking": {},
        "available_features": selected_feature_cols,
        "error": None,
    }
    if feature_selection_method and str(feature_selection_method).lower() != "none":
        feature_selection = select_top_features(
            df,
            top_k=feature_top_k,
            min_score=feature_min_score,
            feature_cols=FEATURE_COLS,
        )
        if feature_selection.get("selected_features"):
            selected_feature_cols = feature_selection["selected_features"]

    print(
        f"Feature selection: method={feature_selection.get('method')} "
        f"selected={len(selected_feature_cols)}/{len(feature_selection.get('available_features', selected_feature_cols))}"
    )

    trtr_for_tuning = trtr_baseline(df, feature_cols=selected_feature_cols, cv_strategy=validation_strategy)

    optuna_tuning = {
        "status": "not_run",
        "best_params": None,
        "best_score": None,
    }
    tuned_tvae_params = None
    tuned_ctgan_params = None
    tuned_copulagan_params = None
    if bool(optuna_tune_tvae) or bool(optuna_tune_all_sdv):
        try:
            from sdv.metadata import SingleTableMetadata

            train_cols = [c for c in SDV_TRAIN_COLS if c in df.columns]
            train_df = df[train_cols].copy()
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(train_df)
            for col in ["capture_latency_days", "transit_duration_days", "promised_transit_days", "shipment_buffer_days", "sla_buffer"]:
                if col in train_df.columns:
                    metadata.update_column(col, sdtype="numerical")
            if "delay_label" in train_df.columns:
                metadata.update_column("delay_label", sdtype="categorical")

            metric_weights = {
                "fidelity": float(optuna_weight_fidelity),
                "utility": float(optuna_weight_utility),
                "privacy": float(optuna_weight_privacy),
                "integrity": float(optuna_weight_integrity),
            }

            if bool(optuna_tune_all_sdv):
                optuna_tuning = {
                    "status": "completed_multi",
                    "architectures": {},
                }
                for arch in ["TVAE", "CTGAN", "CopulaGAN"]:
                    print(f"Running Optuna {arch} tuning: trials={optuna_trials}, timeout_sec={optuna_timeout_sec}...")
                    arch_result = tune_sdv_optuna(
                        architecture=arch,
                        real_df=df,
                        train_df=train_df,
                        metadata=metadata,
                        trtr_baseline_result=trtr_for_tuning,
                        feature_cols=selected_feature_cols,
                        validation_strategy=validation_strategy,
                        n_trials=int(optuna_trials),
                        timeout_sec=int(optuna_timeout_sec),
                        n_synth=min(n_synth, len(df)) if len(df) > 0 else n_synth,
                        base_epochs=epochs,
                        constraint_level=constraint_level_effective,
                        metric_weights=metric_weights,
                        min_tstr_f1=optuna_min_tstr_f1,
                        min_business_utility=optuna_min_business_utility,
                        min_ks_mean=optuna_min_ks_mean,
                        max_js_mean=optuna_max_js_mean,
                        min_dcr_mean=optuna_min_dcr_mean,
                        search_space=optuna_search_space,
                        random_state=int(random_state),
                    )
                    optuna_tuning["architectures"][arch] = arch_result

                tuned_tvae_params = optuna_tuning["architectures"].get("TVAE", {}).get("best_params")
                tuned_ctgan_params = optuna_tuning["architectures"].get("CTGAN", {}).get("best_params")
                tuned_copulagan_params = optuna_tuning["architectures"].get("CopulaGAN", {}).get("best_params")
                print(
                    "Optuna multi-arch summary: "
                    f"TVAE={optuna_tuning['architectures'].get('TVAE', {}).get('best_score')} "
                    f"CTGAN={optuna_tuning['architectures'].get('CTGAN', {}).get('best_score')} "
                    f"CopulaGAN={optuna_tuning['architectures'].get('CopulaGAN', {}).get('best_score')}"
                )
            else:
                print(f"Running Optuna TVAE tuning: trials={optuna_trials}, timeout_sec={optuna_timeout_sec}...")
                optuna_tuning = tune_tvae_optuna(
                    real_df=df,
                    train_df=train_df,
                    metadata=metadata,
                    trtr_baseline_result=trtr_for_tuning,
                    feature_cols=selected_feature_cols,
                    validation_strategy=validation_strategy,
                    n_trials=int(optuna_trials),
                    timeout_sec=int(optuna_timeout_sec),
                    n_synth=min(n_synth, len(df)) if len(df) > 0 else n_synth,
                    base_epochs=epochs,
                    constraint_level=constraint_level_effective,
                    metric_weights=metric_weights,
                    min_tstr_f1=optuna_min_tstr_f1,
                    min_business_utility=optuna_min_business_utility,
                    min_ks_mean=optuna_min_ks_mean,
                    max_js_mean=optuna_max_js_mean,
                    min_dcr_mean=optuna_min_dcr_mean,
                    search_space=optuna_search_space,
                    random_state=int(random_state),
                )
                tuned_tvae_params = optuna_tuning.get("best_params") if optuna_tuning.get("status") == "completed" else None
                if tuned_tvae_params:
                    print(f"Optuna best TVAE score={optuna_tuning.get('best_score')} params={tuned_tvae_params}")
                elif optuna_tuning.get("status") == "no_feasible_trials":
                    print("Optuna tuning found no feasible trials under current hard constraints; using profile TVAE params.")
        except Exception as exc:
            optuna_tuning = {
                "status": "failed",
                "reason": str(exc),
                "best_params": None,
                "best_score": None,
            }
            print(f"Optuna tuning failed; falling back to profile params. Reason: {exc}")

    print(
        f"Training config: profile={tuning_profile}, epochs={epochs}, n_synth={n_synth}, "
        f"constraint_level={constraint_level_effective}, tvae_l2scale={tvae_l2scale_effective}, batch_size={batch_size}"
    )
    requested_arches = sdv_architectures or ["TVAE", "CopulaGAN", "CTGAN"]

    synths = train_sdv_models(
        df,
        epochs=epochs,
        n_synth=n_synth,
        batch_size=batch_size,
        tvae_l2scale=tvae_l2scale_effective,
        constraint_level=constraint_level_effective,
        tvae_params=tuned_tvae_params,
        ctgan_params=tuned_ctgan_params,
        copulagan_params=tuned_copulagan_params,
        architectures=requested_arches,
    )

    if max_phase < 4:
        return {"real_df": df, "constraint_report": constraint_report, "synths": synths}

    fidelity_all = {arch: compute_fidelity(df, sdf) for arch, sdf in synths.items()}
    privacy_all = {arch: compute_dcr_nnaa(df, sdf) for arch, sdf in synths.items()}
    qualitative_all = {arch: qualitative_sanity_check(df, sdf, sample_size=25) for arch, sdf in synths.items()}
    sdmetrics_secondary = {
        "status": "not_run",
        "architectures": {},
    }
    if enable_sdmetrics_effective:
        sdmetrics_secondary = evaluate_sdmetrics_all(
            real_df=df,
            synths=synths,
            sample_size=sdmetrics_sample_size_effective,
            include_column_pairs=sdmetrics_include_pairs_effective,
            seed=int(random_state),
        )

    trtr = trtr_for_tuning
    tstr_all = {}
    for arch, sdf in synths.items():
        tstr_all[arch] = tstr_evaluate(
            df,
            sdf,
            trtr["weighted_f1_mean"],
            trtr["business_utility_mean"],
            trtr["label2_recall_mean"],
            feature_cols=selected_feature_cols,
            cv_strategy=validation_strategy,
        )

    xgb_results = xgboost_tstr(
        df,
        synths.get("TVAE", list(synths.values())[0]),
        trtr["weighted_f1_mean"],
        feature_cols=selected_feature_cols,
        cv_strategy=validation_strategy,
    )
    smote_res = smote_baseline(
        df,
        trtr["weighted_f1_mean"],
        trtr["business_utility_mean"],
        feature_cols=selected_feature_cols,
        split_strategy=validation_strategy,
    )
    tl_results_all = {}
    for arch, sdf in synths.items():
        tl_results_all[arch] = transfer_learning_curve(
            sdf,
            df,
            feature_cols=selected_feature_cols,
            split_strategy=validation_strategy,
        )
    tl_results = tl_results_all.get("TVAE", next(iter(tl_results_all.values())))
    # Use TVAE fold F1s for significance test if available, otherwise fall back to
    # the first architecture that was actually run (supports single-arch runs like CTGAN-only).
    _primary_arch = "TVAE" if "TVAE" in tstr_all else next(iter(tstr_all), None)
    _primary_fold_f1s = tstr_all.get(_primary_arch, {}).get("fold_f1s", []) if _primary_arch else []
    sig_results = significance_test(trtr["fold_f1s"], _primary_fold_f1s)
    feat_imp = compute_feature_importance(df, feature_cols=selected_feature_cols)

    print_final_report(
        trtr,
        tstr_all,
        smote_res,
        tl_results,
        constraint_report,
        fidelity_all,
        privacy_all,
        sig_results=sig_results,
        xgb_results=xgb_results,
        feat_imp=feat_imp,
        feature_selection=feature_selection,
        optuna_tuning=optuna_tuning,
        qualitative_all=qualitative_all,
        tl_results_all=tl_results_all,
        sdmetrics_secondary=sdmetrics_secondary,
    )

    plot_results(
        df,
        synths,
        tl_results,
        trtr,
        tstr_all,
        smote_res,
        {},
        fidelity_all=fidelity_all,
        privacy_all=privacy_all,
        feat_imp=feat_imp,
        selected_feature_cols=selected_feature_cols,
    )

    if export_report_dir:
        export_report(
            export_report_dir,
            trtr,
            tstr_all,
            smote_res,
            tl_results,
            constraint_report,
            fidelity_all,
            privacy_all,
            sig_results=sig_results,
            xgb_results=xgb_results,
            feat_imp=feat_imp,
            feature_selection=feature_selection,
            validation_strategy=validation_strategy,
            optuna_tuning=optuna_tuning,
            qualitative_all=qualitative_all,
            tl_results_all=tl_results_all,
            sdmetrics_secondary=sdmetrics_secondary,
        )
        export_sdmetrics_report(export_report_dir, sdmetrics_secondary)

    return {
        "real_df": df,
        "synths": synths,
        "trtr": trtr,
        "tstr_all": tstr_all,
        "smote": smote_res,
        "transfer_learning": tl_results,
        "transfer_learning_all_arch": tl_results_all,
        "fidelity": fidelity_all,
        "privacy": privacy_all,
        "constraint_report": constraint_report,
        "significance": sig_results,
        "xgboost": xgb_results,
        "feature_importance": feat_imp,
        "feature_selection": feature_selection,
        "selected_feature_cols": selected_feature_cols,
        "validation_strategy": validation_strategy,
        "optuna_tuning": optuna_tuning,
        "qualitative_analysis": qualitative_all,
        "sdmetrics_secondary": sdmetrics_secondary,
    }
