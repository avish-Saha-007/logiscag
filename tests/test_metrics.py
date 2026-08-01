"""
Unit tests for viva_demo metric functions.

Run with:
    python -m pytest test_metrics.py -v
    # or
    python test_metrics.py
"""

import numpy as np
import sys
import os

# Ensure viva_demo is importable from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import viva_demo as v


# ─────────────────────────────────────────────────────────────────────────────
# business_utility_score
# ─────────────────────────────────────────────────────────────────────────────

def test_bus_perfect_predictions():
    """Perfect predictions should yield BUS = 1.0."""
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]
    assert v.business_utility_score(y_true, y_pred) == 1.0


def test_bus_all_wrong():
    """All wrong predictions should yield BUS < 0.5."""
    y_true = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    y_pred = [1, 2, 1, 0, 2, 0, 0, 1, 0]
    score = v.business_utility_score(y_true, y_pred)
    assert score < 0.5, f"BUS should be < 0.5, got {score}"


def test_bus_label2_fn_penalty():
    """Misclassifying Label 2 should be penalised more heavily."""
    y_true = [0, 1, 2, 2, 0, 1, 2, 0]
    y_pred = [0, 1, 0, 2, 0, 1, 1, 0]   # Two Label 2 FN
    score = v.business_utility_score(y_true, y_pred)
    assert 0 < score < 1.0, f"BUS should be between 0 and 1, got {score}"


def test_bus_no_label2_records():
    """BUS should degrade gracefully when no Label 2 records exist."""
    y_true = [0, 0, 1, 1, 0, 1]
    y_pred = [0, 1, 1, 0, 0, 1]
    score = v.business_utility_score(y_true, y_pred)
    assert 0 <= score <= 1.0


def test_bus_weight_increases_penalty():
    """Higher fn_weight should lower BUS when Label 2 FN exist."""
    y_true = [0, 1, 2, 2]
    y_pred = [0, 1, 0, 1]  # Both Label 2 are FN
    bus_3 = v.business_utility_score(y_true, y_pred, fn_weight=3.0)
    bus_5 = v.business_utility_score(y_true, y_pred, fn_weight=5.0)
    assert bus_5 < bus_3, "Higher weight should produce lower BUS"


# ─────────────────────────────────────────────────────────────────────────────
# ks_complement
# ─────────────────────────────────────────────────────────────────────────────

def test_ks_identical_distributions():
    """Identical distributions should have KS complement ≈ 1.0."""
    import pandas as pd
    data = pd.Series(np.random.normal(0, 1, 1000))
    ks = v.ks_complement(data, data)
    assert ks >= 0.99, f"KS complement should be ≈1.0, got {ks}"


def test_ks_very_different_distributions():
    """Very different distributions should have low KS complement."""
    import pandas as pd
    a = pd.Series(np.zeros(500))
    b = pd.Series(np.ones(500))
    ks = v.ks_complement(a, b)
    assert ks < 0.1, f"KS complement should be low, got {ks}"


# ─────────────────────────────────────────────────────────────────────────────
# tvd_complement
# ─────────────────────────────────────────────────────────────────────────────

def test_tvd_identical_distributions():
    """Identical categorical distributions should have TVD complement = 1.0."""
    import pandas as pd
    data = pd.Series(["A", "B", "C"] * 100)
    tvd = v.tvd_complement(data, data)
    assert tvd == 1.0, f"TVD complement should be 1.0, got {tvd}"


def test_tvd_completely_different():
    """Completely non-overlapping distributions should have TVD complement = 0.0."""
    import pandas as pd
    a = pd.Series(["A"] * 100)
    b = pd.Series(["B"] * 100)
    tvd = v.tvd_complement(a, b)
    assert tvd == 0.0, f"TVD complement should be 0.0, got {tvd}"


# ─────────────────────────────────────────────────────────────────────────────
# expected_calibration_error
# ─────────────────────────────────────────────────────────────────────────────

def test_ece_perfect_calibration():
    """A perfectly calibrated model should have ECE ≈ 0."""
    # Simulate 3-class, 90 samples: model predicts perfect probabilities
    y_true = np.array([0]*30 + [1]*30 + [2]*30)
    y_prob = np.zeros((90, 3))
    for i in range(90):
        y_prob[i, y_true[i]] = 0.95
        y_prob[i, (y_true[i] + 1) % 3] = 0.03
        y_prob[i, (y_true[i] + 2) % 3] = 0.02
    ece = v.expected_calibration_error(y_true, y_prob)
    assert ece < 0.1, f"ECE should be low for good calibration, got {ece}"


def test_ece_handles_confidence_1():
    """ECE should handle predictions with confidence = 1.0 (last bin edge)."""
    y_true = np.array([0, 1, 2])
    y_prob = np.array([[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]])
    ece = v.expected_calibration_error(y_true, y_prob)
    assert ece == 0.0, f"Perfect confidence should give ECE = 0, got {ece}"


# ─────────────────────────────────────────────────────────────────────────────
# audit_constraints
# ─────────────────────────────────────────────────────────────────────────────

def test_audit_proxy_zero_hard_violations():
    """Proxy dataset from generate_proxy_dataset should have 0 hard violations."""
    df = v.generate_proxy_dataset(n=500, seed=42)
    df = v.phase1_derived_features(df)
    report = v.audit_constraints(df, verbose=False)
    assert report["total_hard_violations"] == 0, \
        f"Expected 0 hard violations, got {report['total_hard_violations']}"


# ─────────────────────────────────────────────────────────────────────────────
# integrity_check_synthetic
# ─────────────────────────────────────────────────────────────────────────────

def test_integrity_proxy_clean():
    """Proxy dataset should pass integrity check with 0% violations."""
    df = v.generate_proxy_dataset(n=500, seed=42)
    df = v.phase1_derived_features(df)
    vr = v.integrity_check_synthetic(df)
    assert vr == 0.0, f"Expected 0% violations, got {vr}%"


# ─────────────────────────────────────────────────────────────────────────────
# encode_features / get_XY
# ─────────────────────────────────────────────────────────────────────────────

def test_get_xy_shape():
    """get_XY should return expected shapes."""
    df = v.generate_proxy_dataset(n=200, seed=42)
    df = v.phase1_derived_features(df)
    X, y = v.get_XY(df)
    assert X.shape[0] == 200, f"Expected 200 rows, got {X.shape[0]}"
    assert X.shape[1] == len(v.FEATURE_COLS), \
        f"Expected {len(v.FEATURE_COLS)} cols, got {X.shape[1]}"
    assert len(y) == 200


# ─────────────────────────────────────────────────────────────────────────────
# load_real_dataset
# ─────────────────────────────────────────────────────────────────────────────

def test_load_real_dataset():
    """load_real_dataset should produce expected columns and label distribution."""
    csv_path = os.path.join(os.path.dirname(__file__),
                            "SCM_Delivery_Promise_Dataset.csv")
    if not os.path.exists(csv_path):
        print("  SKIP: SCM_Delivery_Promise_Dataset.csv not found")
        return
    df = v.load_real_dataset(csv_path)
    assert "delay_label" in df.columns, "delay_label column missing"
    assert "transit_duration_days" in df.columns, "transit_duration_days column missing"
    assert "capture_latency_days" in df.columns, "capture_latency_days column missing"
    assert "date_order_created_omni" in df.columns, "date_order_created_omni missing"
    assert "date_ship_last_shipment" in df.columns, "date_ship_last_shipment missing"
    assert set(df["delay_label"].unique()).issubset({0, 1, 2}), \
        f"Unexpected labels: {df['delay_label'].unique()}"
    assert len(df) > 0, "Expected non-empty dataset"


# ─────────────────────────────────────────────────────────────────────────────
# significance_test
# ─────────────────────────────────────────────────────────────────────────────

def test_significance_identical_scores():
    """Identical fold scores should produce non-significant result."""
    scores = [0.85, 0.86, 0.84, 0.87, 0.85]
    result = v.significance_test(scores, scores)
    # t-stat should be 0 or NaN, p-value should be 1 or NaN
    assert result["paired_t_stat"] == 0.0 or np.isnan(result["paired_t_stat"])


def test_significance_different_scores():
    """Very different fold scores should produce significant result."""
    trtr_f1s = [0.90, 0.91, 0.89, 0.92, 0.90]
    tstr_f1s = [0.50, 0.52, 0.48, 0.51, 0.49]
    result = v.significance_test(trtr_f1s, tstr_f1s)
    assert result["significant_005"] is True, \
        f"Expected significant, got p={result['paired_t_pvalue']}"


# ─────────────────────────────────────────────────────────────────────────────
# Run as script
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_funcs = [v for k, v in globals().items() if k.startswith("test_")]
    passed = failed = 0
    for fn in test_funcs:
        try:
            fn()
            print(f"  ✓ {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ {fn.__name__}: UNEXPECTED ERROR: {e}")
            failed += 1

    print(f"\n  Results: {passed} passed, {failed} failed, "
          f"{passed + failed} total")
    sys.exit(1 if failed else 0)
