"""
This module implements the core evaluation metrics for synthetic data fidelity, utility, and sanity checks. 
It includes statistical distance measures (KS complement, TVD complement, Wasserstein distance, JS divergence) for
comparing real and synthetic distributions, as well as downstream utility metrics like business utility score and 
expected calibration error. 
Additionally, it provides a qualitative sanity check function to identify common data issues in the synthetic dataset.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# Numeric columns that carry genuine operational signal for fidelity/privacy distance
# metrics. Deliberately excludes order_no (a row identifier, not a feature -- including
# it would let a privacy attack key off literal record IDs rather than attribute
# similarity), delay_label (the target, not an input), and transit_duration (a duplicate
# alias of transit_duration_days kept only for backward-compat references).
PRIVACY_FIDELITY_NUM_COLS = [
    "capture_latency_days",
    "transit_duration_days",
    "promised_transit_days",
    "shipment_buffer_days",
    "sla_buffer",
]


def ks_complement(real_series: pd.Series, synth_series: pd.Series) -> float:
    """
    - **Functional**: Measures numeric distribution similarity where larger values indicate closer alignment.
	- **Technical**: Executes two-sample KS test on non-null values and returns `1 - ks_stat`.
    - **Intuition**: A KS complement of 0.95 suggests the synthetic distribution closely matches the real distribution, while a value of 0.5 indicates significant divergence.
    - **Use Cases**: Ideal for evaluating fidelity of continuous features like `distance_km` and `transit_duration` in the delivery dataset.
    - **Limitations**: Sensitive to outliers and may not capture differences in distribution tails effectively. Should be used alongside other metrics for a comprehensive assessment.
    - **Best Practices**: Interpret in context of feature importance and consider domain knowledge when evaluating results. A high KS complement is desirable but not sufficient alone to confirm synthetic data quality.
    - **Example**: If the KS complement for `distance_km` is 0.92, it indicates that the synthetic data's distance distribution closely resembles that of the real data, which is a positive sign for fidelity in that feature.
    
    Computes the complement of the Kolmogorov-Smirnov statistic between two numeric series.

    Args:
        real_series: A pandas Series containing real data values.
        synth_series: A pandas Series containing synthetic data values.
    Returns:
        A float in [0, 1] where higher values indicate more similar distributions.
    
    """
    real_vals = pd.to_numeric(pd.Series(real_series), errors="coerce").dropna().values
    synth_vals = pd.to_numeric(pd.Series(synth_series), errors="coerce").dropna().values
    if real_vals.size == 0 or synth_vals.size == 0:
        return 0.0
    ks_stat, _ = stats.ks_2samp(real_vals, synth_vals)
    return round(1 - ks_stat, 4)


def tvd_complement(real_series: pd.Series, synth_series: pd.Series) -> float:
    """    
    Computes the complement of the total variation distance between two categorical series.
    
    - **Functional**: Measures categorical distribution overlap as a similarity score.
	- **Technical**: Computes union of categories, compares normalized frequencies, calculates TVD, then returns `1 - tvd`.
    
    - **Functional**: Measures categorical distribution similarity where larger values indicate closer alignment.
    - **Technical**: Computes total variation distance (TVD) between value counts of two categorical series and returns `1 - TVD`.
    - **Intuition**: A TVD complement of 0.98 for the `carrier` feature suggests that the synthetic data has a very similar distribution of carriers as the real data, while a value of 0.6 would indicate a significant mismatch in carrier representation.
    - **Use Cases**: Best suited for evaluating fidelity of categorical features such as `carrier`, `service_level`, and `delay_label` in the delivery dataset. 
    - **Limitations**: Does not account for semantic similarity between categories (e.g., two different carriers may be treated as completely different). Should be used in conjunction with domain knowledge to assess the impact of distribution differences.
    - **Best Practices**: A high TVD complement is desirable, but also consider the importance of the feature in downstream tasks. For critical categorical features, aim for a TVD complement above 0.9 to ensure good fidelity.
    - **Example**: If the TVD complement for `service_level` is 0.85, it indicates that the synthetic data's distribution of service levels is fairly similar to the real data, which is a positive sign for fidelity in that feature, though there may be some discrepancies worth investigating further.
    
    Args:
        real_series: A pandas Series containing real data categorical values.
        synth_series: A pandas Series containing synthetic data categorical values.
    Returns:
        A float in [0, 1] where higher values indicate more similar distributions.
    """
    cats = set(real_series.unique()) | set(synth_series.unique())
    real_dist = real_series.value_counts(normalize=True)
    synth_dist = synth_series.value_counts(normalize=True)
    tvd = 0.5 * sum(abs(real_dist.get(c, 0) - synth_dist.get(c, 0)) for c in cats)
    return round(1 - tvd, 4)


def wasserstein_metric(real_series: pd.Series, synth_series: pd.Series) -> float:
    """
    Computes the Wasserstein distance between two numeric series.
    
	- **Functional**: Quantifies distribution shift magnitude for numeric variables.
	- **Technical**: Converts both vectors to float arrays and uses SciPy Wasserstein distance.    
    
    - **Functional**: Measures the "cost" of transforming one distribution into another, with lower values indicating closer alignment.
    - **Technical**: Uses `scipy.stats.wasserstein_distance` on non-null float values from both series.
    - **Intuition**: A Wasserstein distance of 0.5 for the `transit_duration` feature suggests that, on average, the synthetic data's transit durations are 0.5 units (e.g., hours) away from the real data's transit durations, indicating good fidelity. A distance of 5.0 would indicate a significant mismatch in transit duration distributions.
    - **Use Cases**: Effective for evaluating fidelity of continuous features where the magnitude of differences matters, such as `transit_duration` and `distance_km` in the delivery dataset.
    - **Limitations**: Can be sensitive to outliers and may not capture differences in distribution shape as effectively as other metrics. Should be interpreted in the context of the feature's scale and importance.
    - **Best Practices**: A lower Wasserstein distance is desirable, but also consider the feature's role in downstream tasks. For critical numeric features, aim for a Wasserstein distance that is a small fraction of the feature's typical range to ensure good fidelity.
    - **Example**: If the Wasserstein distance for `sla_buffer` is 0.2, it indicates that the synthetic data's SLA buffer values are, on average, 0.2 units away from the real data's SLA buffer values, which suggests a good level of fidelity for that feature.  
    
    Args:
        real_series: A pandas Series containing real data numeric values.
        synth_series: A pandas Series containing synthetic data numeric values.
    Returns:
        A float representing the Wasserstein distance between the two series.
    """

    real_vals = pd.to_numeric(pd.Series(real_series), errors="coerce").dropna().values
    synth_vals = pd.to_numeric(pd.Series(synth_series), errors="coerce").dropna().values
    if real_vals.size == 0 or synth_vals.size == 0:
        return 0.0
    return round(float(stats.wasserstein_distance(real_vals, synth_vals)), 4)


def js_divergence_numeric(real_series: pd.Series, synth_series: pd.Series, n_bins: int = 30) -> float:
    """
    Computes the Jensen-Shannon divergence between two numeric series by binning values into histograms.

	- **Functional**: Captures probabilistic divergence between numeric marginals.
	- **Technical**: Builds common histogram bins over combined range, adds epsilon smoothing, normalizes to PMFs, and computes Jensen–Shannon divergence via entropy.

    
    - **Functional**: Measures the similarity between two probability distributions, with lower values indicating closer alignment.
    - **Technical**: Bins the numeric values into histograms and computes the Jensen-Shannon divergence using the resulting probability distributions.
    - **Intuition**: A Jensen-Shannon divergence of 0.1 for the `transit_duration` feature suggests that the synthetic data's transit durations are very similar to the real data's transit durations, indicating good fidelity. A divergence of 0.5 would indicate a significant mismatch in transit duration distributions.
    - **Use Cases**: Effective for evaluating fidelity of continuous features where the shape of the distribution matters, such as `transit_duration` and `distance_km` in the delivery dataset.
    - **Limitations**: Can be sensitive to the choice of binning and may not capture differences in distribution tails effectively. Should be interpreted in the context of the feature's scale and importance.
    - **Best Practices**: A lower Jensen-Shannon divergence is desirable, but also consider the feature's role in downstream tasks. For critical numeric features, aim for a divergence that is a small fraction of the feature's typical range to ensure good fidelity.
    - **Example**: If the Jensen-Shannon divergence for `sla_buffer` is 0.2, it indicates that the synthetic data's SLA buffer values are, on average, 0.2 units away from the real data's SLA buffer values, which suggests a good level of fidelity for that feature.  
    
    Args:
        real_series: A pandas Series containing real data numeric values.
        synth_series: A pandas Series containing synthetic data numeric values.
        n_bins: The number of bins to use for histogramming the data.
    Returns:
        A float representing the Jensen-Shannon divergence between the two series.
    """
    real_vals = pd.to_numeric(pd.Series(real_series), errors="coerce").dropna().values
    synth_vals = pd.to_numeric(pd.Series(synth_series), errors="coerce").dropna().values

    combined = np.concatenate([real_vals, synth_vals])
    if combined.size == 0:
        return 0.0

    cmin, cmax = float(np.min(combined)), float(np.max(combined))
    if cmin == cmax:
        return 0.0

    bins = np.linspace(cmin, cmax, n_bins + 1)
    p, _ = np.histogram(real_vals, bins=bins, density=False)
    q, _ = np.histogram(synth_vals, bins=bins, density=False)

    p = p.astype(float) + 1e-12
    q = q.astype(float) + 1e-12
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    js = 0.5 * (stats.entropy(p, m) + stats.entropy(q, m))
    return round(float(js), 4)


def compute_fidelity(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> dict:
    """
    Computes a suite of fidelity metrics comparing real and synthetic datasets, including KS complement, Wasserstein distance, JS divergence for numeric features, and TVD complement for categorical features.  Returns a dictionary with per-field scores and overall means for each metric.   

	- **Functional**: Produces a unified fidelity report across numeric and categorical fields.
	- **Technical**: Iterates fixed column families, computes per-field metrics (KS/Wasserstein/JS/TVD), and returns per-field dictionaries plus means.

     - **Functional**: Provides a comprehensive set of metrics to evaluate how closely the synthetic data matches the real data across both numeric and categorical features.
     - **Technical**: Iterates over predefined numeric and categorical columns, computes appropriate metrics for each, and aggregates results into a structured dictionary.
     - **Intuition**: High KS complement and TVD complement values (close to 1) indicate that the synthetic data closely matches the real data distributions for numeric and categorical features, respectively. 
        Low Wasserstein distance and JS divergence values indicate that the synthetic data's numeric distributions are similar to the real data's distributions.
     - **Use Cases**: Essential for assessing the fidelity of synthetic data in the delivery dataset, particularly for features like `distance_km`, `transit_duration`, `carrier`, and `service_level`.
     - **Limitations**: These metrics provide a statistical comparison but do not capture all aspects of data quality (e.g., multivariate relationships, temporal consistency). Should be used in conjunction with other evaluation methods for a comprehensive assessment. 
     
     Args:
        real_df: A pandas DataFrame containing the real dataset.
        synth_df: A pandas DataFrame containing the synthetic dataset.
    Returns:
        A dictionary containing per-field and mean scores for KS complement, Wasserstein distance, JS divergence, and TVD complement.
    """
    num_cols = PRIVACY_FIDELITY_NUM_COLS
    cat_cols = [
        "dpe_id",
        "line_type",
        "dpe_shipnode",
        "last_shipnode",
        "last_scac",
        "last_shipment_carrier_service",
        "carrier_service_code",
        "order_type",
        "delay_label",
    ]

    ks_scores = {}
    wasserstein_scores = {}
    js_scores = {}
    tvd_scores = {}

    for col in num_cols:
        if col in real_df.columns and col in synth_df.columns:
            ks_scores[col] = ks_complement(real_df[col], synth_df[col])
            wasserstein_scores[col] = wasserstein_metric(real_df[col], synth_df[col])
            js_scores[col] = js_divergence_numeric(real_df[col], synth_df[col])

    for col in cat_cols:
        if col in real_df.columns and col in synth_df.columns:
            tvd_scores[col] = tvd_complement(real_df[col].astype(str), synth_df[col].astype(str))

    return {
        "ks_per_field": ks_scores,
        "ks_mean": round(np.mean(list(ks_scores.values())), 4) if ks_scores else 0,
        "wasserstein_per_field": wasserstein_scores,
        "wasserstein_mean": round(np.mean(list(wasserstein_scores.values())), 4) if wasserstein_scores else 0,
        "js_per_field": js_scores,
        "js_mean": round(np.mean(list(js_scores.values())), 4) if js_scores else 0,
        "tvd_per_field": tvd_scores,
        "tvd_mean": round(np.mean(list(tvd_scores.values())), 4) if tvd_scores else 0,
    }


def compute_dcr_nnaa(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> dict:
    """
    Computes Distance to Closest Record (DCR) and Nearest Neighbor Adversarial Accuracy (NNAA) metrics to evaluate synthetic data privacy and uniqueness.

	- **Functional**: Estimates privacy risk through nearest-neighbor distance behavior.
	- **Technical**: Standardizes numeric space, samples capped subsets for efficiency, computes DCR stats and NNAA ratio, and emits full distribution summaries plus sampled raw values.    

    - **Functional**: Quantifies privacy risk and overfitting in synthetic data.
    - **Technical**: Scales numeric features, samples records, computes DCR as distance to nearest real record, and NNAA by comparing distances to real vs synthetic neighbors.

     - **Functional**: Provides insights into the privacy and uniqueness of the synthetic data by measuring how closely synthetic records resemble real records (DCR) and whether synthetic records are more similar to real data than to other synthetic data (NNAA).
     - **Technical**: Uses Euclidean distance on scaled numeric features to compute DCR for synthetic records against real records, and NNAA by comparing distances to nearest real and synthetic neighbors.
     - **Intuition**: A low mean DCR suggests that synthetic records are very close to real records, which may indicate a privacy risk. An NNAA value close to 0.5 indicates that synthetic records are equally similar to real and synthetic neighbors, which is desirable for privacy. Values significantly above 0.5 may indicate overfitting to the training data.
     - **Use Cases**: Important for assessing the privacy implications of the synthetic data in the delivery dataset, particularly for features that could be sensitive or identifying.
     - **Limitations**: These metrics focus on numeric features and may not capture privacy risks associated with categorical features or complex multivariate relationships. Should be used alongside other privacy evaluation methods for a comprehensive assessment.

    Args:
        real_df: A pandas DataFrame containing the real dataset.
        synth_df: A pandas DataFrame containing the synthetic dataset.
    Returns:
        A dictionary containing DCR and NNAA metrics, including mean, min, percentiles, and distributions.
    """

    num_cols = PRIVACY_FIDELITY_NUM_COLS
    available = [c for c in num_cols if c in real_df.columns and c in synth_df.columns]

    if not available or len(real_df) == 0 or len(synth_df) == 0:
        empty = np.asarray([], dtype=float)
        return {
            "dcr_mean": 0.0,
            "dcr_min": 0.0,
            "dcr_p5": 0.0,
            "nnaa": 0.0,
            "dcr_distribution": _distance_distribution_summary(empty),
            "nn_to_real_distribution": _distance_distribution_summary(empty),
            "nn_to_synth_distribution": _distance_distribution_summary(empty),
            "dcr_values_sample": [],
            "nn_to_real_values_sample": [],
            "nn_to_synth_values_sample": [],
        }

    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_df[available].fillna(0))
    synth_scaled = scaler.transform(synth_df[available].fillna(0))

    rng = np.random.RandomState(42)
    n_sample = min(1000, len(synth_scaled), len(real_scaled))
    if n_sample == 0:
        empty = np.asarray([], dtype=float)
        return {
            "dcr_mean": 0.0,
            "dcr_min": 0.0,
            "dcr_p5": 0.0,
            "nnaa": 0.0,
            "dcr_distribution": _distance_distribution_summary(empty),
            "nn_to_real_distribution": _distance_distribution_summary(empty),
            "nn_to_synth_distribution": _distance_distribution_summary(empty),
            "dcr_values_sample": [],
            "nn_to_real_values_sample": [],
            "nn_to_synth_values_sample": [],
        }
    synth_idx = rng.choice(len(synth_scaled), n_sample, replace=False)
    real_idx = rng.choice(len(real_scaled), n_sample, replace=False)

    dists_sr = cdist(synth_scaled[synth_idx], real_scaled[real_idx], metric="euclidean")
    dcr_per_record = dists_sr.min(axis=1)

    synth_perm = rng.permutation(len(synth_scaled))
    split = int(0.8 * len(synth_perm))
    synth_train = synth_scaled[synth_perm[:split]]
    synth_test = synth_scaled[synth_perm[split:]]

    n_nnaa = min(1000, len(synth_test))
    n_ref = min(1000, len(real_scaled), len(synth_train))
    if n_nnaa == 0 or n_ref == 0:
        dcr_mean = float(np.mean(dcr_per_record)) if dcr_per_record.size > 0 else 0.0
        dcr_min = float(np.min(dcr_per_record)) if dcr_per_record.size > 0 else 0.0
        dcr_p5 = float(np.percentile(dcr_per_record, 5)) if dcr_per_record.size > 0 else 0.0
        empty = np.asarray([], dtype=float)
        return {
            "dcr_mean": round(dcr_mean, 4),
            "dcr_min": round(dcr_min, 4),
            "dcr_p5": round(dcr_p5, 4),
            "nnaa": 0.0,
            "dcr_distribution": _distance_distribution_summary(dcr_per_record),
            "nn_to_real_distribution": _distance_distribution_summary(empty),
            "nn_to_synth_distribution": _distance_distribution_summary(empty),
            "dcr_values_sample": np.round(dcr_per_record, 6).tolist(),
            "nn_to_real_values_sample": [],
            "nn_to_synth_values_sample": [],
        }
    test_sub = synth_test[:n_nnaa]
    real_sub = real_scaled[rng.choice(len(real_scaled), n_ref, replace=False)]
    train_sub = synth_train[rng.choice(len(synth_train), n_ref, replace=False)]

    d_to_real = cdist(test_sub, real_sub).min(axis=1)
    d_to_synth = cdist(test_sub, train_sub).min(axis=1)
    nnaa = float(np.mean(d_to_real < d_to_synth))

    return {
        "dcr_mean": round(float(np.mean(dcr_per_record)), 4),
        "dcr_min": round(float(np.min(dcr_per_record)), 4),
        "dcr_p5": round(float(np.percentile(dcr_per_record, 5)), 4),
        "nnaa": round(nnaa, 4),
        "dcr_distribution": _distance_distribution_summary(dcr_per_record),
        "nn_to_real_distribution": _distance_distribution_summary(d_to_real),
        "nn_to_synth_distribution": _distance_distribution_summary(d_to_synth),
        "dcr_values_sample": np.round(dcr_per_record, 6).tolist(),
        "nn_to_real_values_sample": np.round(d_to_real, 6).tolist(),
        "nn_to_synth_values_sample": np.round(d_to_synth, 6).tolist(),
    }


def _distance_distribution_summary(values: np.ndarray, n_bins: int = 20) -> dict:
    """
    Helper function to summarize distance distributions for DCR and NNAA metrics.

	- **Functional**: Compresses distance vectors into robust descriptive statistics for reporting.
	- **Technical**: Returns count, mean, quantiles, min/max, and histogram counts/edges with empty-array safeguards.

     - **Functional**: Summarizes distance distributions with stats and histograms.
     - **Technical**: Converts values to array, computes count, mean, min, percentiles, max, and histogram counts/edges.
     
     Args:
         values: A numpy array of distance values.
         n_bins: Number of bins for the histogram.
     Returns:
         A dictionary containing summary statistics and histogram data.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "p5": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "max": None,
            "hist_counts": [],
            "hist_edges": [],
        }

    hist_counts, hist_edges = np.histogram(arr, bins=n_bins)
    return {
        "count": int(arr.size),
        "mean": round(float(np.mean(arr)), 6),
        "min": round(float(np.min(arr)), 6),
        "p5": round(float(np.percentile(arr, 5)), 6),
        "p25": round(float(np.percentile(arr, 25)), 6),
        "p50": round(float(np.percentile(arr, 50)), 6),
        "p75": round(float(np.percentile(arr, 75)), 6),
        "p95": round(float(np.percentile(arr, 95)), 6),
        "max": round(float(np.max(arr)), 6),
        "hist_counts": hist_counts.tolist(),
        "hist_edges": np.round(hist_edges, 6).tolist(),
    }


def business_utility_score(y_true, y_pred, fn_weight: float = 3.0) -> float:
    """
    Computes a business utility score based on a confusion matrix, applying a higher penalty for false negatives of the minority class (label 2) to reflect their greater business impact.
    
	- **Functional**: Scores business impact with stronger penalty on missed severe delays.
	- **Technical**: Uses confusion matrix, computes weighted misclassification cost versus max possible cost, and returns normalized utility in `[0,1]`.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        fn_weight: Weight for false negatives of the minority class.
    
    Returns:
        A float representing the business utility score.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    n = len(y_true)
    std_errors = n - np.trace(cm)
    label2_total = cm[2, :].sum()
    label2_fn = label2_total - cm[2, 2]
    penalty = (fn_weight - 1) * label2_fn
    max_cost = n + (fn_weight - 1) * label2_total
    actual_cost = std_errors + penalty
    return round(1 - actual_cost / max_cost, 4)


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """
    Computes the Expected Calibration Error (ECE) for probabilistic predictions, measuring how well predicted probabilities align with actual outcomes. 

	- **Functional**: Measures how prediction confidence aligns with actual accuracy.
	- **Technical**: Buckets max-class confidence into bins, computes weighted absolute accuracy-confidence gap per bin, and sums to ECE.


    - **Functional**: Evaluates probability calibration quality.
    - **Technical**: Bins predictions by confidence, computes accuracy and average confidence per bin, and aggregates weighted absolute differences to produce ECE.
    - **Functional**: Provides a single metric to assess how well the predicted probabilities reflect true likelihoods, with lower values indicating better calibration.
    - **Technical**: Bins predictions into confidence intervals, calculates accuracy and average confidence for each bin, and computes a weighted average of the absolute differences to yield the ECE.
    - **Intuition**: An ECE of 0.05 suggests that, on average, the predicted probabilities are off by 5 percentage points from the true outcomes, indicating good calibration. An ECE of 0.2 would indicate poor calibration, where predicted probabilities are significantly misaligned with actual outcomes.
    - **Use Cases**: Important for evaluating the reliability of probabilistic predictions in downstream tasks, such as predicting delivery delays in the delivery dataset.
    - **Limitations**: ECE can be sensitive to the choice of bins and may not capture all aspects of calibration (e.g., it does not account for class imbalance). Should be interpreted in conjunction with other evaluation metrics for a comprehensive assessment.
    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities for the positive class.
        n_bins: Number of bins to use for calibration.  
    Returns:
        A float representing the Expected Calibration Error.
    """
    y_true = np.asarray(y_true)
    y_prob_max = y_prob.max(axis=1)
    y_pred = y_prob.argmax(axis=1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        if i == n_bins - 1:
            mask = (y_prob_max >= lo) & (y_prob_max <= hi)
        else:
            mask = (y_prob_max >= lo) & (y_prob_max < hi)
        if mask.sum() == 0:
            continue
        acc = (y_pred[mask] == y_true[mask]).mean()
        conf = y_prob_max[mask].mean()
        ece += abs(acc - conf) * mask.sum() / n

    return round(ece, 4)


def significance_test(trtr_fold_f1s: list, tstr_fold_f1s: list) -> dict:
    """
    Performs paired t-test and Wilcoxon signed-rank test on fold-level F1 scores from training-training (trtr) and training-test (tstr) evaluations to assess statistical significance of performance differences.

	- **Functional**: Tests whether TRTR and TSTR fold results differ statistically.
	- **Technical**: Runs paired t-test and Wilcoxon signed-rank on per-fold differences and returns test statistics, p-values, and significance boolean.
    - **Functional**: Provides statistical evidence on whether the performance differences between training-training and training-test evaluations are significant.
    - **Technical**: Uses paired t-test to compare fold-level F1 scores and Wilcoxon signed-rank test as a non-parametric alternative, returning test statistics, p-values, and a boolean indicating significance at the 0.05 level.
    - **Intuition**: A significant result (p-value < 0.05) would suggest that there is a statistically meaningful difference in performance between the two evaluation setups, which could indicate issues with generalization or overfitting in the synthetic data.
    - **Use Cases**: Important for validating whether observed differences in model performance when trained on synthetic data versus real data are statistically significant, particularly in the context of evaluating synthetic data utility for downstream tasks.
    - **Limitations**: The tests assume that the fold-level F1 scores are independent and identically distributed, which may not always hold true. Additionally, the Wilcoxon test may not be valid if there are many tied ranks (i.e., identical F1 scores across folds). Should be interpreted in conjunction with effect size measures and domain knowledge for a comprehensive assessment.

    Args:
        trtr_fold_f1s: List of F1 scores from training-training evaluations.
        tstr_fold_f1s: List of F1 scores from training-test evaluations.
    Returns:
        A dictionary containing the test statistics and p-values for both tests, as well as a boolean indicating significance at the 0.05 level.

    """
    from scipy.stats import ttest_rel, wilcoxon

    t_stat, t_pval = ttest_rel(trtr_fold_f1s, tstr_fold_f1s)
    try:
        w_stat, w_pval = wilcoxon(np.array(trtr_fold_f1s) - np.array(tstr_fold_f1s))
    except ValueError:
        w_stat, w_pval = float("nan"), float("nan")

    return {
        "paired_t_stat": round(float(t_stat), 4),
        "paired_t_pvalue": round(float(t_pval), 4),
        "wilcoxon_stat": round(float(w_stat), 4) if not np.isnan(w_stat) else None,
        "wilcoxon_pvalue": round(float(w_pval), 4) if not np.isnan(w_pval) else None,
        "significant_005": float(t_pval) < 0.05,
    }


def qualitative_sanity_check(real_df: pd.DataFrame, synth_df: pd.DataFrame, sample_size: int = 25, seed: int = 42) -> dict:
    """
    Performs a qualitative sanity check on the synthetic dataset by identifying common data issues such as temporal inconsistencies, negative or zero values in numeric fields, label-SLA inconsistencies, unknown categorical values, and extreme outliers based on real data distributions. 
    Returns a summary of checks performed, counts of flagged rows, and sample examples.    

	- **Functional**: Produces human-auditable anomaly evidence beyond aggregate metrics.
	- **Technical**: Validates temporal/logical constraints, unknown categories, and numeric outliers against real reference quantiles, then returns check counts, sample rows, and flagged examples.

    - **Functional**: Identifies common data quality issues in synthetic dataset.
    - **Technical**: Checks temporal order, numeric validity, label-SLA consistency, categorical value presence, and outliers; flags rows; samples examples for review.
    - **Functional**: Provides a comprehensive sanity check to identify common data quality issues in the synthetic dataset, such as temporal inconsistencies, invalid numeric values, label-SLA mismatches, unknown categorical values, and extreme outliers.
    - **Technical**: Implements a series of checks on the synthetic dataset, comparing against the real dataset where applicable. Flags rows that violate any of the checks and provides a summary of the issues found, including counts and sample examples for further review.
    - **Intuition**: A high number of flagged rows or certain types of issues (e.g., temporal order violations) may indicate fundamental problems with the synthetic data generation process that could impact downstream utility and fidelity. Reviewing sample flagged examples can provide insights into specific areas where the synthetic data may be failing to capture important real-world patterns or constraints.
    - **Use Cases**: Essential for performing an initial quality assessment of the synthetic dataset before conducting more formal evaluations of fidelity and utility. Helps identify glaring issues that could undermine the usefulness of the synthetic data for analysis or modeling tasks.
    - **Limitations**: This sanity check focuses on univariate issues and may not capture multivariate inconsistencies or more subtle data quality problems. Should be used as a preliminary step in conjunction with more comprehensive evaluation methods for a thorough assessment of synthetic data quality.

    Args:        
        real_df: A pandas DataFrame containing the real dataset.
        synth_df: A pandas DataFrame containing the synthetic dataset.
        sample_size: The number of sample rows to return for review.
        seed: Random seed for reproducibility when sampling examples.
    Returns:        
        A dictionary containing the results of the sanity checks, including counts of flagged rows, types of issues found, and sample examples for review.
    """
    sdf = synth_df.copy()
    rdf = real_df.copy()

    for c in [
        "date_order_created_omni",
        "date_order_capture",
        "date_ship_last_shipment",
        "date_delivery_last_shipment",
        "order_timestamp",
        "shipping_date",
        "delivery_date_actual",
        "promised_date",
    ]:
        if c in sdf.columns:
            sdf[c] = pd.to_datetime(sdf[c], errors="coerce")
        if c in rdf.columns:
            rdf[c] = pd.to_datetime(rdf[c], errors="coerce")

    n = len(sdf)
    if n == 0:
        return {
            "row_count": 0,
            "checks": {},
            "total_flagged_rows": 0,
            "flagged_row_rate_pct": 0.0,
            "sample_rows": [],
            "flagged_examples": [],
        }

    checks = {}

    if {"date_order_created_omni", "date_order_capture", "date_ship_last_shipment", "date_delivery_last_shipment"}.issubset(sdf.columns):
        temporal_bad = ~(
            (sdf["date_order_created_omni"] <= sdf["date_order_capture"]) &
            (sdf["date_order_capture"] <= sdf["date_ship_last_shipment"]) &
            (sdf["date_ship_last_shipment"] <= sdf["date_delivery_last_shipment"])
        )
        checks["temporal_order_violation"] = int(temporal_bad.fillna(False).sum())

    if "transit_duration_days" in sdf.columns:
        checks["non_positive_transit_duration"] = int((pd.to_numeric(sdf["transit_duration_days"], errors="coerce") <= 0).fillna(False).sum())
    if "capture_latency_days" in sdf.columns:
        checks["negative_capture_latency"] = int((pd.to_numeric(sdf["capture_latency_days"], errors="coerce") < 0).fillna(False).sum())

    derived_transit = pd.to_numeric(sdf.get("transit_duration_days"), errors="coerce")
    if {"date_promise_delivery", "date_delivery_last_shipment", "delay_label"}.issubset(sdf.columns):
        sla = (
            pd.to_datetime(sdf["date_promise_delivery"], errors="coerce") -
            pd.to_datetime(sdf["date_delivery_last_shipment"], errors="coerce")
        ).dt.total_seconds() / 86400.0
        expected = np.where(sla >= 0, 0, np.where(-sla <= 2.0, 1, 2))
        lbl = pd.to_numeric(sdf["delay_label"], errors="coerce").fillna(-1).astype(int)
        checks["label_sla_inconsistency"] = int((lbl != expected).sum())
    elif "delay_label" in sdf.columns and derived_transit.notna().sum() > 0:
        q1 = float(derived_transit.quantile(0.5442))
        q2 = float(derived_transit.quantile(0.92))
        expected = np.where(derived_transit <= q1, 0, np.where(derived_transit <= q2, 1, 2))
        lbl = pd.to_numeric(sdf["delay_label"], errors="coerce").fillna(-1).astype(int)
        checks["label_transit_inconsistency"] = int((lbl != expected).sum())

    cat_unknown = pd.Series(False, index=sdf.index)
    for col in [
        "dpe_id",
        "line_type",
        "dpe_shipnode",
        "last_shipnode",
        "last_scac",
        "last_shipment_carrier_service",
        "carrier_service_code",
        "order_type",
    ]:
        if col in sdf.columns and col in rdf.columns:
            known = set(rdf[col].dropna().astype(str).unique())
            cat_unknown |= ~sdf[col].astype(str).isin(known)
    checks["unknown_categorical_value"] = int(cat_unknown.sum())

    numeric_cols = [
        "capture_latency_days",
        "transit_duration_days",
    ]
    outlier_mask = pd.Series(False, index=sdf.index)
    for col in numeric_cols:
        if col in sdf.columns and col in rdf.columns:
            r = pd.to_numeric(rdf[col], errors="coerce").dropna()
            s = pd.to_numeric(sdf[col], errors="coerce")
            if len(r) >= 20:
                lo = float(np.percentile(r, 0.5))
                hi = float(np.percentile(r, 99.5))
                outlier_mask |= ((s < lo) | (s > hi)).fillna(False)
    checks["extreme_numeric_outlier"] = int(outlier_mask.sum())

    any_flag = pd.Series(False, index=sdf.index)
    if "temporal_order_violation" in checks and {"date_order_created_omni", "date_order_capture", "date_ship_last_shipment", "date_delivery_last_shipment"}.issubset(sdf.columns):
        any_flag |= ~(
            (sdf["date_order_created_omni"] <= sdf["date_order_capture"]) &
            (sdf["date_order_capture"] <= sdf["date_ship_last_shipment"]) &
            (sdf["date_ship_last_shipment"] <= sdf["date_delivery_last_shipment"])
        ).fillna(False)
    if "non_positive_transit_duration" in checks and "transit_duration_days" in sdf.columns:
        any_flag |= (pd.to_numeric(sdf["transit_duration_days"], errors="coerce") <= 0).fillna(False)
    if "negative_capture_latency" in checks and "capture_latency_days" in sdf.columns:
        any_flag |= (pd.to_numeric(sdf["capture_latency_days"], errors="coerce") < 0).fillna(False)
    if "label_sla_inconsistency" in checks and {"date_promise_delivery", "date_delivery_last_shipment", "delay_label"}.issubset(sdf.columns):
        sla = (
            pd.to_datetime(sdf["date_promise_delivery"], errors="coerce") -
            pd.to_datetime(sdf["date_delivery_last_shipment"], errors="coerce")
        ).dt.total_seconds() / 86400.0
        expected = np.where(sla >= 0, 0, np.where(-sla <= 2.0, 1, 2))
        label = pd.to_numeric(sdf["delay_label"], errors="coerce").fillna(-1).astype(int)
        any_flag |= (label != expected)
    if "label_transit_inconsistency" in checks and "delay_label" in sdf.columns:
        q1 = float(derived_transit.quantile(0.5442))
        q2 = float(derived_transit.quantile(0.92))
        expected = np.where(derived_transit <= q1, 0, np.where(derived_transit <= q2, 1, 2))
        label = pd.to_numeric(sdf["delay_label"], errors="coerce").fillna(-1).astype(int)
        any_flag |= (label != expected)
    any_flag |= cat_unknown
    any_flag |= outlier_mask

    sample_n = min(int(sample_size), n)
    sample_rows = sdf.sample(n=sample_n, random_state=seed).copy() if sample_n > 0 else sdf.iloc[:0].copy()

    display_cols = [
        "date_order_created_omni",
        "date_order_capture",
        "date_ship_last_shipment",
        "date_delivery_last_shipment",
        "date_promise_shipment",
        "date_promise_delivery",
        "dpe_id",
        "line_type",
        "dpe_shipnode",
        "last_shipnode",
        "last_scac",
        "last_shipment_carrier_service",
        "carrier_service_code",
        "order_type",
        "capture_latency_days",
        "transit_duration_days",
        "promised_transit_days",
        "shipment_buffer_days",
        "sla_buffer",
        "delay_label",
    ]
    display_cols = [c for c in display_cols if c in sample_rows.columns]
    sample_rows = sample_rows[display_cols]

    flagged_examples = sdf[any_flag].head(10).copy()
    flagged_examples = flagged_examples[[c for c in display_cols if c in flagged_examples.columns]]

    return {
        "row_count": int(n),
        "checks": checks,
        "total_flagged_rows": int(any_flag.sum()),
        "flagged_row_rate_pct": round(float(any_flag.mean() * 100), 4),
        "sample_rows": _records_for_json(sample_rows),
        "flagged_examples": _records_for_json(flagged_examples),
    }


def _records_for_json(df: pd.DataFrame) -> list:
    """Helper function to convert a DataFrame to a list of dictionaries suitable for JSON serialization, with special handling for datetime columns.    

	- **Functional**: Converts dataframes into API/report-friendly record lists.
	- **Technical**: Casts datetime columns to fixed string format and outputs `orient='records'` dictionaries.

    - **Functional**: Transforms a DataFrame into a list of dictionaries for easy JSON serialization, ensuring that datetime columns are formatted as strings.
    - **Technical**: Iterates through DataFrame columns, identifies datetime types, converts them to a standardized string format (`YYYY-MM-DD HH:MM:SS`), and then uses `to_dict(orient='records')` to produce a list of dictionaries representing each row.

    Args:
        df (pd.DataFrame): The DataFrame to convert.

    Returns:
        list: A list of dictionaries representing the DataFrame rows, with datetime columns formatted as strings.

    """
    if df.empty:
        return []
    out = df.copy()
    for c in out.columns:
        if np.issubdtype(out[c].dtype, np.datetime64):
            out[c] = out[c].astype("datetime64[ns]").dt.strftime("%Y-%m-%d %H:%M:%S")
    return out.to_dict(orient="records")
