"""
Evaluation utilities for the VIVA pipeline, including baseline models, transfer learning evaluation, and feature importance computation.
This module provides functions to evaluate synthetic data quality and model performance using various metrics and strategies.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, brier_score_loss
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from .data import get_XY, rank_features_by_shap
from .metrics import business_utility_score, expected_calibration_error

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    class SMOTE:
        def __init__(self, **kwargs):
            pass
        def fit_resample(self, X, y):
            return X, y

try:
    from xgboost import XGBClassifier
    XGBOOST_OK = True
except ImportError:
    XGBOOST_OK = False


def _sort_df_temporally(df: pd.DataFrame) -> pd.DataFrame:
    """Sorts the DataFrame by the earliest available timestamp column to facilitate temporal splitting.
    If no timestamp columns are found, returns the DataFrame as-is.

	- **Functional**: Establishes chronological ordering for leakage-aware evaluation.
	- **Technical**: Sorts by first available timestamp among preferred columns using stable sort and reset index.

    Args:        
        df: Input DataFrame to sort.
    Returns:        
        A DataFrame sorted by the earliest timestamp column if available, otherwise the original DataFrame.
    """
    for time_col in [
        "date_order_created_omni",
        "date_order_capture",
        "date_ship_last_shipment",
        "date_delivery_last_shipment",
        "order_timestamp",
        "shipping_date",
        "delivery_date_actual",
    ]:
        if time_col in df.columns:
            sdf = df.copy()
            sdf[time_col] = pd.to_datetime(sdf[time_col], errors="coerce")
            return sdf.sort_values(time_col, kind="stable").reset_index(drop=True)
    return df.reset_index(drop=True)


def _generate_cv_splits(df: pd.DataFrame, y: np.ndarray, n_splits: int, strategy: str):
    """Generates cross-validation splits based on the specified strategy.

	- **Functional**: Creates fold definitions for temporal or class-stratified validation.
	- **Technical**: Uses `TimeSeriesSplit` for temporal mode and `StratifiedKFold` for stratified mode, returning both normalized dataframe and split indices.


    - **Functional**: Creates train-test splits for model evaluation, supporting both temporal and stratified strategies.
    - **Technical**: Uses TimeSeriesSplit for temporal strategy and StratifiedKFold for stratified strategy, ensuring appropriate handling of class distribution and temporal order.
    Args:
        df: Input DataFrame for which to generate splits.
        y: Target labels corresponding to the DataFrame rows.
        n_splits: Number of splits/folds to generate.
        strategy: Splitting strategy to use ("temporal" or "stratified").
    Returns:
        A tuple containing the (potentially sorted) DataFrame and a list of train-test index splits.
    """
    if strategy == "temporal":
        sdf = _sort_df_temporally(df)
        if len(sdf) <= n_splits:
            raise ValueError(f"Not enough rows ({len(sdf)}) for TimeSeriesSplit with n_splits={n_splits}")
        splitter = TimeSeriesSplit(n_splits=n_splits)
        return sdf, list(splitter.split(np.arange(len(sdf))))

    sdf = df.reset_index(drop=True)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return sdf, list(splitter.split(np.arange(len(sdf)), y))


def _temporal_holdout_split(df: pd.DataFrame, test_size: float = 0.2):
    """Splits the DataFrame into training and testing sets based on temporal order.

	- **Functional**: Creates an operationally realistic train/test split by time.
	- **Technical**: Computes tail-size holdout with edge-case guards to ensure non-empty train and test partitions.
    
    - **Functional**: Creates a single train-test split that respects temporal ordering to prevent leakage.
    - **Technical**: Sorts the DataFrame by timestamp and splits based on the specified test size, ensuring at least one test sample.
    Args:
        df: Input DataFrame to split.
        test_size: Proportion of the dataset to include in the test split (between 0 and 1).
    Returns:
        A tuple containing the training and testing DataFrames.
    """
    sdf = _sort_df_temporally(df)
    n = len(sdf)
    n_test = max(1, int(round(n * float(test_size))))
    n_test = min(n_test, n - 1)
    split_idx = n - n_test
    return sdf.iloc[:split_idx].reset_index(drop=True), sdf.iloc[split_idx:].reset_index(drop=True)


def trtr_baseline(real_df: pd.DataFrame, n_estimators: int = 200, feature_cols: list = None, cv_strategy: str = "temporal") -> dict:
    """Train-Test-Retrain (TRTR) baseline evaluation using a Random Forest classifier.

	- **Functional**: Establishes real-data upper-reference utility under chosen validation strategy.
	- **Technical**: Trains balanced random forest across folds and aggregates weighted F1, business utility, and class-2 recall with fold trace retention.

    - **Functional**: Establishes a realistic upper bound on performance by training and testing on real data with proper validation.
    - **Technical**: Performs cross-validation with specified strategy, returning mean and std of weighted F1, business utility, and label-2 recall.    

    Args:
        real_df: Input DataFrame containing the real data.
        n_estimators: Number of trees in the Random Forest.
        feature_cols: List of feature column names to use for training.
        cv_strategy: Cross-validation strategy to use ("temporal" or "stratified").
    Returns:
        A dictionary containing evaluation metrics including weighted F1 mean and std, business utility mean, label-2 recall mean, and fold-wise F1 scores.
    """
    _, y_tmp = get_XY(real_df, feature_cols=feature_cols)
    eval_df, splits = _generate_cv_splits(real_df, y_tmp, n_splits=5, strategy=cv_strategy)
    X, y = get_XY(eval_df, feature_cols=feature_cols)
    clf = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced", random_state=42, n_jobs=-1)

    f1s, bus_scores, l2_recalls = [], [], []
    for tr, te in splits:
        clf.fit(X[tr], y[tr])
        preds = clf.predict(X[te])
        f1s.append(f1_score(y[te], preds, average="weighted"))
        bus_scores.append(business_utility_score(y[te], preds))
        cm = confusion_matrix(y[te], preds, labels=[0, 1, 2])
        l2_recall = cm[2, 2] / cm[2, :].sum() if cm[2, :].sum() > 0 else 0
        l2_recalls.append(l2_recall)

    return {
        "weighted_f1_mean": round(np.mean(f1s), 4),
        "weighted_f1_std": round(np.std(f1s), 4),
        "business_utility_mean": round(np.mean(bus_scores), 4),
        "label2_recall_mean": round(np.mean(l2_recalls), 4),
        "fold_f1s": f1s,
    }


def tstr_evaluate(real_df: pd.DataFrame, synth_df: pd.DataFrame, trtr_f1: float, trtr_bus: float, trtr_l2_recall: float, n_estimators: int = 200, feature_cols: list = None, cv_strategy: str = "temporal") -> dict:
    """Train on synthetic, test on real (TSTR) evaluation using a Random Forest classifier.

	- **Functional**: Evaluates how well synthetic-trained models transfer to real-world holdout folds.
	- **Technical**: Fits once on synthetic data, scores across real folds, computes utility/calibration/brier/class-2 preservation and confusion matrix with baseline-relative ratios.
    
    - **Functional**: Evaluates the utility of synthetic data by training on it and testing on real data.
    - **Technical**: Trains a balanced random forest on synthetic data and evaluates on real data across folds, returning weighted F1, business utility, label-2 recall, ECE, Brier score, and confusion matrix.

    Args:
        real_df: Input DataFrame containing the real data.
        synth_df: Input DataFrame containing the synthetic data.
        trtr_f1: Weighted F1 score from TRTR baseline.
        trtr_bus: Business utility score from TRTR baseline.
        trtr_l2_recall: Label-2 recall from TRTR baseline.
        n_estimators: Number of trees in the Random Forest.
        feature_cols: List of feature column names to use for training.
        cv_strategy: Cross-validation strategy to use ("temporal" or "stratified").
    Returns:
        A dictionary containing evaluation metrics including weighted F1, business utility, label-2 recall, ECE, Brier score, and confusion matrix.
    """
    X_synth, y_synth = get_XY(synth_df, feature_cols=feature_cols)
    _, y_tmp = get_XY(real_df, feature_cols=feature_cols)
    eval_df, splits = _generate_cv_splits(real_df, y_tmp, n_splits=5, strategy=cv_strategy)
    X_real, y_real = get_XY(eval_df, feature_cols=feature_cols)

    clf = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(X_synth, y_synth)

    f1s, bus_scores, l2_recalls, eces = [], [], [], []
    briers_all = []
    all_y_true, all_y_pred = [], []

    for _, te_idx in splits:
        X_te, y_te = X_real[te_idx], y_real[te_idx]
        preds = clf.predict(X_te)
        proba = clf.predict_proba(X_te)

        f1s.append(f1_score(y_te, preds, average="weighted"))
        bus_scores.append(business_utility_score(y_te, preds))
        eces.append(expected_calibration_error(y_te, proba))

        fold_brier = []
        for cls in range(proba.shape[1]):
            fold_brier.append(brier_score_loss((y_te == cls).astype(int), proba[:, cls]))
        briers_all.append(np.mean(fold_brier))

        cm = confusion_matrix(y_te, preds, labels=[0, 1, 2])
        l2_recall = cm[2, 2] / cm[2, :].sum() if cm[2, :].sum() > 0 else 0
        l2_recalls.append(l2_recall)

        all_y_true.extend(y_te.tolist())
        all_y_pred.extend(preds.tolist())

    weighted_f1 = np.mean(f1s)
    bus = np.mean(bus_scores)
    l2_recall = np.mean(l2_recalls)

    return {
        "weighted_f1": round(weighted_f1, 4),
        "weighted_f1_std": round(np.std(f1s), 4),
        "f1_gap_vs_trtr": round(trtr_f1 - weighted_f1, 4),
        "business_utility": round(bus, 4),
        "bus_ratio_vs_trtr": round(bus / trtr_bus if trtr_bus > 0 else 0, 4),
        "label2_recall": round(l2_recall, 4),
        "l2_recall_pres_ratio": round(l2_recall / trtr_l2_recall if trtr_l2_recall > 0 else 0, 4),
        "ece": round(np.mean(eces), 4),
        "brier_score_mean": round(np.mean(briers_all), 4),
        "confusion_matrix": confusion_matrix(all_y_true, all_y_pred, labels=[0, 1, 2]),
        "fold_f1s": f1s,
    }


def smote_baseline(real_df: pd.DataFrame, trtr_f1: float, trtr_bus: float, n_estimators: int = 200, feature_cols: list = None, split_strategy: str = "temporal") -> dict:
    """SMOTE baseline evaluation using a Random Forest classifier.
    
	- **Functional**: Provides a non-generative imbalance baseline for fair comparison.
	- **Technical**: Applies SMOTE on train split (or no-op fallback), trains random forest, and reports core utility plus baseline-relative gaps/ratios.

    - **Functional**: Evaluates the utility of SMOTE-augmented training data by training on it and testing on real data.
    - **Technical**: Applies SMOTE to the training set, trains a balanced random forest, and evaluates on the test set, returning weighted F1, business utility, and label-2 recall.

    Args:
        real_df: Input DataFrame containing the real data.
        trtr_f1: Weighted F1 score from TRTR baseline.
        trtr_bus: Business utility score from TRTR baseline.
        n_estimators: Number of trees in the Random Forest.
        feature_cols: List of feature column names to use for training.
        split_strategy: Strategy for splitting the data ("temporal" or "stratified").
    Returns:
        A dictionary containing evaluation metrics including weighted F1, business utility, and label-2 recall.
    """
    if split_strategy == "temporal":
        tr_df, te_df = _temporal_holdout_split(real_df, test_size=0.2)
        X_tr, y_tr = get_XY(tr_df, feature_cols=feature_cols)
        X_te, y_te = get_XY(te_df, feature_cols=feature_cols)
    else:
        X_real, y_real = get_XY(real_df, feature_cols=feature_cols)
        split = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        tr_idx, te_idx = next(iter(split.split(X_real, y_real)))
        X_tr, X_te = X_real[tr_idx], X_real[te_idx]
        y_tr, y_te = y_real[tr_idx], y_real[te_idx]

    # Cap SMOTE input to 20k rows to avoid O(n log n) BallTree explosion on large datasets.
    smote_cap = 20000
    if len(X_tr) > smote_cap:
        rng_sm = np.random.RandomState(42)
        sm_idx = rng_sm.choice(len(X_tr), smote_cap, replace=False)
        X_tr_sm, y_tr_sm = X_tr[sm_idx], y_tr[sm_idx]
    else:
        X_tr_sm, y_tr_sm = X_tr, y_tr

    sm = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X_tr_sm, y_tr_sm)

    clf = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(X_res, y_res)
    preds = clf.predict(X_te)

    weighted_f1 = f1_score(y_te, preds, average="weighted")
    bus = business_utility_score(y_te, preds)
    cm = confusion_matrix(y_te, preds, labels=[0, 1, 2])
    l2_recall = cm[2, 2] / cm[2, :].sum() if cm[2, :].sum() > 0 else 0

    return {
        "weighted_f1": round(weighted_f1, 4),
        "f1_gap_vs_trtr": round(trtr_f1 - weighted_f1, 4),
        "business_utility": round(bus, 4),
        "bus_ratio": round(bus / trtr_bus if trtr_bus > 0 else 0, 4),
        "label2_recall": round(l2_recall, 4),
    }


def transfer_learning_curve(synth_df: pd.DataFrame, real_df: pd.DataFrame, fractions: list = None, n_estimators: int = 200, feature_cols: list = None, split_strategy: str = "temporal") -> dict:
    """Transfer learning curve evaluation by training on combined synthetic and varying fractions of real data.
    
	- **Functional**: Quantifies sample-efficiency gains from synthetic pretraining.
	- **Technical**: For each fraction, compares synthetic+fractional-real fine-tuning against real-only model trained on identical fractional sample counts.

    - **Functional**: Assesses the impact of incorporating varying amounts of real data into a synthetic pretraining setup.
    - **Technical**: Trains a Random Forest on synthetic data combined with different fractions of real data, evaluates on a holdout set, and reports weighted F1 and business utility.

    Args:
        synth_df: Input DataFrame containing the synthetic data.
        real_df: Input DataFrame containing the real data.
        fractions: List of fractions of real data to include in training.
        n_estimators: Number of trees in the Random Forest.
        feature_cols: List of feature column names to use for training.
        split_strategy: Strategy for splitting the data ("temporal" or "stratified").
    Returns:
        A dictionary containing evaluation metrics for each fraction of real data included in training.
    """
    if fractions is None:
        fractions = [0.0, 0.25, 1.0]  # Reduced from 6 to 3 points; covers synth-only, quarter-real, full-real.

    X_synth, y_synth = get_XY(synth_df, feature_cols=feature_cols)
    if split_strategy == "temporal":
        tr_df, te_df = _temporal_holdout_split(real_df, test_size=0.2)
        X_tr, y_tr = get_XY(tr_df, feature_cols=feature_cols)
        X_te, y_te = get_XY(te_df, feature_cols=feature_cols)
    else:
        X_real, y_real = get_XY(real_df, feature_cols=feature_cols)
        split = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        tr_idx, te_idx = next(iter(split.split(X_real, y_real)))
        X_tr, X_te = X_real[tr_idx], X_real[te_idx]
        y_tr, y_te = y_real[tr_idx], y_real[te_idx]

    synth_results, real_only_results = {}, {}
    rng = np.random.RandomState(42)

    for frac in fractions:
        clf = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced", random_state=42, n_jobs=-1)
        if frac == 0.0:
            clf.fit(X_synth, y_synth)
        else:
            n_real = max(1, int(len(X_tr) * frac))
            idx = rng.choice(len(X_tr), n_real, replace=False)
            X_combined = np.vstack([X_synth, X_tr[idx]])
            y_combined = np.hstack([y_synth, y_tr[idx]])
            clf.fit(X_combined, y_combined)

        preds = clf.predict(X_te)
        synth_results[frac] = {
            "weighted_f1": round(f1_score(y_te, preds, average="weighted"), 4),
            "business_utility": round(business_utility_score(y_te, preds), 4),
        }

        if frac > 0:
            n_real = max(1, int(len(X_tr) * frac))
            idx_r = rng.choice(len(X_tr), n_real, replace=False)
            clf_r = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced", random_state=42, n_jobs=-1)
            clf_r.fit(X_tr[idx_r], y_tr[idx_r])
            preds_r = clf_r.predict(X_te)
            real_only_results[frac] = {
                "weighted_f1": round(f1_score(y_te, preds_r, average="weighted"), 4),
                "business_utility": round(business_utility_score(y_te, preds_r), 4),
            }

    return {"synth_pretrain": synth_results, "real_only": real_only_results}


def xgboost_tstr(real_df: pd.DataFrame, synth_df: pd.DataFrame, trtr_f1: float, feature_cols: list = None, cv_strategy: str = "temporal") -> dict:
    """TSTR evaluation using an XGBoost classifier as a stronger baseline.
    
	- **Functional**: Adds strong boosting baseline to test architecture sensitivity of TSTR results.
	- **Technical**: Trains `XGBClassifier` on synthetic features and evaluates weighted F1 over real-data folds with graceful skip path when XGBoost is unavailable.

    - **Functional**: Provides a non-generative imbalance baseline for fair comparison.
    - **Technical**: Trains an XGBoost classifier on synthetic data and evaluates on real data using cross-validation.

     Args:
        real_df: Input DataFrame containing the real data.
        synth_df: Input DataFrame containing the synthetic data.
        trtr_f1: Weighted F1 score from TRTR baseline.
        feature_cols: List of feature column names to use for training.
        cv_strategy: Cross-validation strategy to use ("temporal" or "stratified").
    Returns:
        A dictionary containing evaluation metrics including weighted F1 and its gap vs TRTR baseline.
    """
    if not XGBOOST_OK:
        return {"status": "xgboost not installed"}

    X_synth, y_synth = get_XY(synth_df, feature_cols=feature_cols)
    _, y_tmp = get_XY(real_df, feature_cols=feature_cols)
    eval_df, splits = _generate_cv_splits(real_df, y_tmp, n_splits=5, strategy=cv_strategy)
    X_real, y_real = get_XY(eval_df, feature_cols=feature_cols)

    clf = XGBClassifier(  # reduced n_estimators/depth for evaluation speed
        n_estimators=50, max_depth=4, learning_rate=0.1,
        use_label_encoder=False, eval_metric="mlogloss", random_state=42, verbosity=0,
    )
    clf.fit(X_synth, y_synth)

    f1s = []
    for _, te_idx in splits:
        preds = clf.predict(X_real[te_idx])
        f1s.append(f1_score(y_real[te_idx], preds, average="weighted"))

    return {
        "weighted_f1": round(np.mean(f1s), 4),
        "f1_gap_vs_trtr": round(trtr_f1 - np.mean(f1s), 4),
        "fold_f1s": f1s,
    }


def compute_feature_importance(real_df: pd.DataFrame, n_estimators: int = 200, feature_cols: list = None) -> dict:
    """Computes feature importance scores using SHAP values from a Random Forest classifier trained on real data.

    - **Functional**: Exposes explainability-oriented feature ranking for reporting and feature selection.
	- **Technical**: Delegates to SHAP/fallback ranking routine and rounds importance scores for stable presentation.

    - **Functional**: Identifies key drivers of delivery delay in the real dataset for interpretability and feature selection.
    - **Technical**: Trains a Random Forest on the real data and uses SHAP values to rank feature importance, returning a dictionary of feature names and their corresponding importance scores.

    Args:
        real_df: Input DataFrame containing the real data.
        n_estimators: Number of trees in the Random Forest.
        feature_cols: List of feature column names to use for training.

    Returns:
        A dictionary containing feature importance scores.
    """
    res = rank_features_by_shap(real_df, feature_cols=feature_cols, n_estimators=n_estimators)
    ranking = res.get("ranking") or {}
    return {k: round(float(v), 4) for k, v in ranking.items()}
