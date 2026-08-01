"""
Provides tuning utilities for synthetic data generation using SDV and Optuna.
Includes an Optuna-based hyperparameter tuning function for SDV synthesizers that optimizes a composite score based on fidelity, utility, privacy, and integrity metrics.
Designed to be flexible and extensible for different SDV architectures (TVAE, CTGAN, CopulaGAN) and customizable search spaces and constraints.
"""

import time
import numpy as np
import pandas as pd

from .constraints import integrity_check_synthetic, build_sdv_constraints
from .data import FEATURE_COLS
from .evaluation import tstr_evaluate
from .metrics import compute_fidelity, compute_dcr_nnaa


def _build_sdv_constraints(train_df: pd.DataFrame, constraint_level: str):
    """Builds a list of SDV constraints based on the specified strictness level.

	- **Functional**: Creates SDV constraint policy matching selected realism strictness.
	- **Technical**: Builds scalar inequality constraint dictionaries for positivity/non-negativity and conditionally includes delay-component rules.    

    Args:
        train_df: The training DataFrame used for synthetic data generation, which may be needed to determine applicable constraints.
        constraint_level: The desired strictness level of constraints ("strict", "moderate", "relaxed").
    Returns:        
        A list of constraint dictionaries to be passed to SDV synthesizers.
    """
    constraints = build_sdv_constraints(train_df, constraint_level)
    return constraints


def _score_fidelity(fidelity: dict) -> float:
    """Computes a composite fidelity score based on KS complement, TVD complement, and JS divergence metrics.    
    
    - **Functional**: Converts fidelity metric bundle into single optimization-friendly score.
	- **Technical**: Weighted combination of KS, TVD, and inverted JS with clipping to `[0,1]`.

    The score is a weighted combination of:
        - KS complement (higher is better)
        - TVD complement (higher is better)
        - JS divergence complement (higher is better, with a cap at 1.0)
    Args:
        fidelity: A dictionary containing fidelity metrics, expected to have keys "ks_mean", "tvd_mean", and "js_mean".
    Returns:
        A float score between 0.0 and 1.0 representing the overall fidelity of the synthetic data compared to the real data.
    """
    ks = float(fidelity.get("ks_mean", 0.0))
    tvd = float(fidelity.get("tvd_mean", 0.0))
    js = float(fidelity.get("js_mean", 1.0))
    score = 0.5 * ks + 0.3 * tvd + 0.2 * max(0.0, 1.0 - min(js, 1.0))
    return float(np.clip(score, 0.0, 1.0))


def _score_utility(tstr: dict) -> float:
    """Computes a composite utility score based on TSTR evaluation metrics.

	- **Functional**: Converts utility outputs into a scalar objective component.
	- **Technical**: Weighted blend of weighted F1, business utility, and class-2 recall with clipping.    

	- **Functional**: Converts TSTR metric bundle into single optimization-friendly score.
	- **Technical**: Weighted combination of F1, business utility, and label2 recall with clipping to `[0,1]`.

    The score is a weighted combination of:
        - F1 score (higher is better)
        - Business utility (higher is better)
        - Label2 recall (higher is better)
    Args:
        tstr: A dictionary containing TSTR metrics, expected to have keys "weighted_f1", "business_utility", and "label2_recall".
    Returns:
        A float score between 0.0 and 1.0 representing the overall utility of the synthetic data compared to the real data.
    """
    f1 = float(tstr.get("weighted_f1", 0.0))
    bus = float(tstr.get("business_utility", 0.0))
    l2 = float(tstr.get("label2_recall", 0.0))
    score = 0.45 * f1 + 0.45 * bus + 0.10 * l2
    return float(np.clip(score, 0.0, 1.0))


def _score_privacy(privacy: dict) -> float:
    """Computes a composite privacy score based on DCR mean and NNAa metrics.

	- **Functional**: Converts privacy diagnostics into optimization score preferring healthy distance and balanced NNAA.
	- **Technical**: Uses saturating transform for DCR and target-distance scoring around NNAA ≈ 0.60.

    - **Functional**: Converts privacy metrics into a scalar score reflecting privacy risk.
    - **Technical**: Weighted combination of DCR-based score and NNAa-based score with clipping.
    The score is a weighted combination of:
        - DCR mean (lower is better, transformed to a score where higher is better)
        - NNAa (closer to 0.60 is better, transformed to a score where higher is better)
   
    Args:        
        privacy: A dictionary containing privacy metrics, expected to have keys "dcr_mean" and "nnaa".
    Returns:
        A float score between 0.0 and 1.0 representing the overall privacy of the synthetic data compared to the real data.
    """
    dcr_mean = float(privacy.get("dcr_mean", 0.0))
    nnaa = float(privacy.get("nnaa", 0.0))
    dcr_score = dcr_mean / (1.0 + dcr_mean)
    nnaa_target = 0.60
    nnaa_score = max(0.0, 1.0 - abs(nnaa - nnaa_target) / 0.25)
    score = 0.5 * dcr_score + 0.5 * nnaa_score
    return float(np.clip(score, 0.0, 1.0))


def _score_integrity(integrity_violation_pct: float) -> float:
    """Computes an integrity score based on the percentage of integrity violations in the synthetic data.

	- **Functional**: Rewards low synthetic rule violations.
	- **Technical**: Computes linear inversion `1 - violation_pct/100` and clips to valid range.

    - **Functional**: Transforms integrity violation percentage into a score where lower violations yield higher scores.
    - **Technical**: Uses a linear transform with clipping to convert violation percentage into a score between 0 and 1.
    The score is calculated as `1.0 - (violation_pct / 100)`, clipped to the range [0.0, 1.0], where:
        - 0% violations → score of 1.0 (best)
        - 100% violations → score of 0.0 (worst)

    Args:
        integrity_violation_pct: A float representing the percentage of integrity violations detected in the synthetic data.
    Returns:         
        A float score between 0.0 and 1.0 representing the integrity of the synthetic data, where higher is better.
    """
    return float(np.clip(1.0 - float(integrity_violation_pct) / 100.0, 0.0, 1.0))


def _aggregate_score(fidelity_score: float, utility_score: float, privacy_score: float, integrity_score: float) -> float:
    """Aggregates individual metric scores into a single composite score.

	- **Functional**: Produces overall pillar-balanced optimization score.
	- **Technical**: Returns arithmetic mean of four scalar pillar scores.

    - **Functional**: Combines multiple metric scores into a single optimization target.
    - **Technical**: Simple average of the four component scores, assuming they are all in the range [0,1].
    The aggregated score is calculated as the average of the four component scores:
        - fidelity_score: A float between 0.0 and 1.0 representing fidelity.
        - utility_score: A float between 0.0 and 1.0 representing utility.
        - privacy_score: A float between 0.0 and 1.0 representing privacy.
        - integrity_score: A float between 0.0 and 1.0 representing integrity.
    Returns:
        A float score between 0.0 and 1.0 representing the overall performance of the synthetic data across all evaluated dimensions.
    """
    return float((fidelity_score + utility_score + privacy_score + integrity_score) / 4.0)


def tune_sdv_optuna(
    architecture: str,
    real_df: pd.DataFrame,
    train_df: pd.DataFrame,
    metadata,
    trtr_baseline_result: dict,
    feature_cols: list = None,
    validation_strategy: str = "temporal",
    n_trials: int = 20,
    timeout_sec: int = 0,
    n_synth: int = 2000,
    base_epochs: int = 20,
    constraint_level: str = "strict",
    metric_weights: dict = None,
    min_tstr_f1: float = None,
    min_business_utility: float = None,
    min_ks_mean: float = None,
    max_js_mean: float = None,
    min_dcr_mean: float = None,
    search_space: str = "standard",
    random_state: int = 42,
):
    """Tunes SDV synthesizer hyperparameters using Optuna to optimize a composite score based on fidelity, utility, privacy, and integrity metrics. 

	- **Functional**: Searches TVAE hyperparameters to maximize weighted multi-pillar objective with optional hard thresholds.
	- **Technical**: Creates Optuna TPE study, samples search space (standard/wide), trains TVAE each trial, computes fidelity/utility/privacy/integrity, applies feasibility penalties, stores detailed user attributes, and returns best feasible configuration plus full trial logs.

    - **Functional**: Searches for the best hyperparameters to maximize a composite score.
    - **Technical**: Uses Optuna for hyperparameter optimization and evaluates a composite score based on multiple metrics.
    Args:
        architecture: The SDV synthesizer architecture to tune ("TVAE", "CTGAN", "CopulaGAN").
        real_df: The real dataset DataFrame for evaluation.
        train_df: The training dataset DataFrame for fitting the synthesizer.
        metadata: The SDV metadata object describing the dataset schema.
        trtr_baseline_result: A dictionary containing baseline TSTR results for the real data, used for comparison during tuning.
        feature_cols: Optional list of feature columns to use for evaluation and tuning. If None, defaults to all feature columns defined in `FEATURE_COLS`.
        validation_strategy: The cross-validation strategy to use during TSTR evaluation (e.g., "temporal", "random").
        n_trials: The number of Optuna trials to perform during tuning.
        timeout_sec: Optional timeout in seconds for the entire tuning process. If 0 or negative, no timeout is applied.
        n_synth: The number of synthetic samples to generate for each trial's evaluation.
        base_epochs: The base number of epochs to use as a reference point for scaling epoch-related hyperparameters in the search space.
        constraint_level: The strictness level of constraints to apply during synthesis ("strict", "moderate", "relaxed").
        metric_weights: Optional dictionary specifying weights for each metric pillar ("fidelity", "utility", "privacy", "integrity") when calculating the composite score. Defaults to equal weights if None.
        min_tstr_f1: Optional minimum acceptable weighted F1 score from TSTR evaluation. Trials that do not meet this threshold will be considered infeasible and penalized in the objective function.
        min_business_utility: Optional minimum acceptable business utility score from TSTR evaluation. Trials that do not meet this threshold will be considered infeasible and penalized in the objective function.
        min_ks_mean: Optional minimum acceptable KS mean from fidelity evaluation. Trials that do not meet this threshold will be considered infeasible and penalized in the objective function.
        max_js_mean: Optional maximum acceptable JS mean from fidelity evaluation. Trials that exceed this threshold will be considered infeasible and penalized in the objective function.
        min_dcr_mean: Optional minimum acceptable DCR mean from privacy evaluation. Trials that do not meet this threshold will be considered infeasible and penalized in the objective function.
        search_space: The search space profile to use for hyperparameter sampling ("standard" or "wide"), which determines the range
            and distribution of hyperparameters sampled during tuning.
    Returns:
        A dictionary containing the results of the tuning process, including the best hyperparameters found, the best composite score achieved, and details about all trials conducted. The structure of the returned dictionary includes:
        - status: A string indicating the outcome of the tuning process ("completed", "skipped", "no_feasible_trials").
        - architecture: The SDV architecture that was tuned.
        - n_trials: The total number of Optuna trials conducted.
        - best_score: The best composite score achieved during tuning (or None if no feasible trials were found).
        - best_params: A dictionary of the best hyperparameters found for the specified architecture (or None if no feasible trials were found).
        - best_raw_params: The raw hyperparameters from the best trial before any architecture-specific processing (or None if no feasible trials were found).
        - best_attrs: The user attributes from the best trial, which may include detailed metric scores and feasibility information (or None if no feasible trials were found).
        - n_feasible_trials: The number of trials that were considered feasible based on the specified metric thresholds.
        - constraints: A dictionary summarizing the metric thresholds used for determining trial feasibility.
        - metric_weights: The weights used for each metric pillar in the composite score calculation.
        - search_space: The search space profile used during tuning.
        - trials: A list of dictionaries containing details about each trial conducted, including trial number, value, parameters, and user attributes. 

    """
    arch = str(architecture).strip()
    supported_arches = {"TVAE", "CTGAN", "CopulaGAN"}
    if arch not in supported_arches:
        return {
            "status": "skipped",
            "reason": f"unsupported_architecture: {arch}",
            "architecture": arch,
            "best_params": None,
            "best_score": None,
            "trials": [],
        }

    try:
        import optuna
        from sdv.single_table import TVAESynthesizer, CTGANSynthesizer, CopulaGANSynthesizer
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": f"optuna_or_sdv_unavailable: {exc}",
            "architecture": arch,
            "best_params": None,
            "best_score": None,
            "trials": [],
        }

    np.random.seed(int(random_state))
    cols = FEATURE_COLS if feature_cols is None else feature_cols

    default_weights = {
        "fidelity": 0.25,
        "utility": 0.25,
        "privacy": 0.25,
        "integrity": 0.25,
    }
    w = default_weights if metric_weights is None else {**default_weights, **metric_weights}
    w_sum = max(1e-9, float(w["fidelity"] + w["utility"] + w["privacy"] + w["integrity"]))

    trtr_f1 = trtr_baseline_result["weighted_f1_mean"]
    trtr_bus = trtr_baseline_result["business_utility_mean"]
    trtr_l2 = trtr_baseline_result["label2_recall_mean"]

    constraints = _build_sdv_constraints(train_df, constraint_level)

    synth_cls = {
        "TVAE": TVAESynthesizer,
        "CTGAN": CTGANSynthesizer,
        "CopulaGAN": CopulaGANSynthesizer,
    }[arch]

    def _build_trial_kwargs(trial):
        if arch == "TVAE":
            if search_space == "wide":
                epochs = trial.suggest_int("epochs", max(8, int(base_epochs * 0.4)), max(20, int(base_epochs * 4.0)))
                embedding_dim = trial.suggest_categorical("embedding_dim", [32, 64, 128, 256, 512])
                compress_width = trial.suggest_categorical("compress_width", [64, 128, 256, 512, 768])
                compress_depth = trial.suggest_int("compress_depth", 1, 4)
                l2scale = trial.suggest_float("l2scale", 1e-8, 1e-2, log=True)
                batch_size = trial.suggest_categorical("batch_size", [128, 256, 500, 1024])
            else:
                epochs = trial.suggest_int("epochs", max(5, int(base_epochs * 0.5)), max(10, int(base_epochs * 2.0)))
                embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256])
                compress_width = trial.suggest_categorical("compress_width", [128, 256, 512])
                compress_depth = trial.suggest_int("compress_depth", 1, 3)
                l2scale = trial.suggest_float("l2scale", 1e-7, 1e-3, log=True)
                batch_size = trial.suggest_categorical("batch_size", [256, 500, 1024])

            compress_dims = tuple([compress_width] * compress_depth)
            decompress_dims = tuple([compress_width] * compress_depth)
            return {
                "epochs": epochs,
                "batch_size": batch_size,
                "embedding_dim": embedding_dim,
                "compress_dims": compress_dims,
                "decompress_dims": decompress_dims,
                "l2scale": l2scale,
                "enforce_min_max_values": True,
            }

        if search_space == "wide":
            epochs = trial.suggest_int("epochs", max(8, int(base_epochs * 0.4)), max(20, int(base_epochs * 4.0)))
            batch_size = trial.suggest_categorical("batch_size", [250, 500, 1000])
        else:
            epochs = trial.suggest_int("epochs", max(5, int(base_epochs * 0.5)), max(10, int(base_epochs * 2.0)))
            batch_size = trial.suggest_categorical("batch_size", [250, 500, 1000])

        # Keep search stable/safe across GAN variants by tuning shared knobs only.
        return {
            "epochs": epochs,
            "batch_size": batch_size,
            "enforce_min_max_values": True,
        }

    def objective(trial):
        synth_kwargs = _build_trial_kwargs(trial)
        synth = synth_cls(metadata, **synth_kwargs)

        if constraints:
            synth.add_constraints(constraints=constraints)

        t0 = time.time()
        synth.fit(train_df)
        synth_df = synth.sample(num_rows=n_synth, batch_size=1000)

        fidelity = compute_fidelity(real_df, synth_df)
        privacy = compute_dcr_nnaa(real_df, synth_df)
        tstr = tstr_evaluate(
            real_df,
            synth_df,
            trtr_f1,
            trtr_bus,
            trtr_l2,
            feature_cols=cols,
            cv_strategy=validation_strategy,
        )
        integrity_pct = integrity_check_synthetic(synth_df)

        fidelity_score = _score_fidelity(fidelity)
        utility_score = _score_utility(tstr)
        privacy_score = _score_privacy(privacy)
        integrity_score = _score_integrity(integrity_pct)
        weighted_score = (
            w["fidelity"] * fidelity_score
            + w["utility"] * utility_score
            + w["privacy"] * privacy_score
            + w["integrity"] * integrity_score
        ) / w_sum

        feasible = True
        penalties = []
        tstr_f1 = float(tstr.get("weighted_f1", 0.0))
        tstr_bus = float(tstr.get("business_utility", 0.0))
        ks_mean = float(fidelity.get("ks_mean", 0.0))
        js_mean = float(fidelity.get("js_mean", 1.0))
        dcr_mean = float(privacy.get("dcr_mean", 0.0))

        if min_tstr_f1 is not None and tstr_f1 < float(min_tstr_f1):
            feasible = False
            penalties.append("tstr_f1")
        if min_business_utility is not None and tstr_bus < float(min_business_utility):
            feasible = False
            penalties.append("business_utility")
        if min_ks_mean is not None and ks_mean < float(min_ks_mean):
            feasible = False
            penalties.append("ks_mean")
        if max_js_mean is not None and js_mean > float(max_js_mean):
            feasible = False
            penalties.append("js_mean")
        if min_dcr_mean is not None and dcr_mean < float(min_dcr_mean):
            feasible = False
            penalties.append("dcr_mean")

        final_score = float(weighted_score if feasible else weighted_score * 0.1)

        trial.set_user_attr("elapsed_sec", round(float(time.time() - t0), 2))
        trial.set_user_attr("fidelity_score", round(fidelity_score, 6))
        trial.set_user_attr("utility_score", round(utility_score, 6))
        trial.set_user_attr("privacy_score", round(privacy_score, 6))
        trial.set_user_attr("integrity_score", round(integrity_score, 6))
        trial.set_user_attr("final_score", round(final_score, 6))
        trial.set_user_attr("weighted_score", round(float(weighted_score), 6))
        trial.set_user_attr("feasible", bool(feasible))
        trial.set_user_attr("violations", penalties)
        trial.set_user_attr("tstr_weighted_f1", round(float(tstr.get("weighted_f1", 0.0)), 6))
        trial.set_user_attr("tstr_business_utility", round(float(tstr.get("business_utility", 0.0)), 6))
        trial.set_user_attr("privacy_nnaa", round(float(privacy.get("nnaa", 0.0)), 6))
        trial.set_user_attr("fidelity_ks_mean", round(float(fidelity.get("ks_mean", 0.0)), 6))
        trial.set_user_attr("integrity_violation_pct", round(float(integrity_pct), 6))

        return final_score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=int(random_state)))
    study.optimize(
        objective,
        n_trials=int(n_trials),
        timeout=(None if int(timeout_sec) <= 0 else int(timeout_sec)),
        show_progress_bar=False,
        catch=(Exception,),
    )

    feasible_trials = [t for t in study.trials if bool(t.user_attrs.get("feasible", True)) and t.value is not None]
    if not feasible_trials:
        trials_out = []
        for t in study.trials:
            trials_out.append({
                "trial": t.number,
                "value": float(t.value) if t.value is not None else None,
                "params": t.params,
                "attrs": t.user_attrs,
            })
        return {
            "status": "no_feasible_trials",
            "architecture": arch,
            "n_trials": len(study.trials),
            "best_score": None,
            "best_params": None,
            "best_raw_params": None,
            "best_attrs": None,
            "n_feasible_trials": 0,
            "constraints": {
                "min_tstr_f1": min_tstr_f1,
                "min_business_utility": min_business_utility,
                "min_ks_mean": min_ks_mean,
                "max_js_mean": max_js_mean,
                "min_dcr_mean": min_dcr_mean,
            },
            "metric_weights": w,
            "trials": trials_out,
        }

    best = max(feasible_trials, key=lambda t: t.value)
    best_params = dict(best.params)

    if arch == "TVAE":
        best_arch_params = {
            "epochs": int(best_params["epochs"]),
            "embedding_dim": int(best_params["embedding_dim"]),
            "compress_dims": tuple([int(best_params["compress_width"])] * int(best_params["compress_depth"])),
            "decompress_dims": tuple([int(best_params["compress_width"])] * int(best_params["compress_depth"])),
            "l2scale": float(best_params["l2scale"]),
            "batch_size": int(best_params["batch_size"]),
        }
    else:
        best_arch_params = {
            "epochs": int(best_params["epochs"]),
            "batch_size": int(best_params["batch_size"]),
        }

    trials_out = []
    for t in study.trials:
        trials_out.append({
            "trial": t.number,
            "value": float(t.value) if t.value is not None else None,
            "params": t.params,
            "attrs": t.user_attrs,
        })

    return {
        "status": "completed",
        "architecture": arch,
        "n_trials": len(study.trials),
        "best_score": round(float(best.value), 6),
        "best_params": best_arch_params,
        "best_raw_params": best_params,
        "best_attrs": best.user_attrs,
        "n_feasible_trials": len(feasible_trials),
        "constraints": {
            "min_tstr_f1": min_tstr_f1,
            "min_business_utility": min_business_utility,
            "min_ks_mean": min_ks_mean,
            "max_js_mean": max_js_mean,
            "min_dcr_mean": min_dcr_mean,
        },
        "metric_weights": w,
        "search_space": search_space,
        "trials": trials_out,
    }


def tune_tvae_optuna(
    real_df: pd.DataFrame,
    train_df: pd.DataFrame,
    metadata,
    trtr_baseline_result: dict,
    feature_cols: list = None,
    validation_strategy: str = "temporal",
    n_trials: int = 20,
    timeout_sec: int = 0,
    n_synth: int = 2000,
    base_epochs: int = 20,
    constraint_level: str = "strict",
    metric_weights: dict = None,
    min_tstr_f1: float = None,
    min_business_utility: float = None,
    min_ks_mean: float = None,
    max_js_mean: float = None,
    min_dcr_mean: float = None,
    search_space: str = "standard",
    random_state: int = 42,
):
    # Backward-compatible wrapper for existing callers.
    return tune_sdv_optuna(
        architecture="TVAE",
        real_df=real_df,
        train_df=train_df,
        metadata=metadata,
        trtr_baseline_result=trtr_baseline_result,
        feature_cols=feature_cols,
        validation_strategy=validation_strategy,
        n_trials=n_trials,
        timeout_sec=timeout_sec,
        n_synth=n_synth,
        base_epochs=base_epochs,
        constraint_level=constraint_level,
        metric_weights=metric_weights,
        min_tstr_f1=min_tstr_f1,
        min_business_utility=min_business_utility,
        min_ks_mean=min_ks_mean,
        max_js_mean=max_js_mean,
        min_dcr_mean=min_dcr_mean,
        search_space=search_space,
        random_state=random_state,
    )
