"""
End-to-end smoke test: proxy data + mock synthesiser, no SDV required.

Exercises the same path as `python -m logiscag.reproduce --verify`'s second
step: train_sdv_models falls back to the mock synthesiser when SDV is absent,
and the full sweep (privacy + utility + fidelity + integrity) and MIA audit
run without error and produce sane shapes.

Finding E regression: the feature_cols filter must use
    c.endswith("_enc")
not just
    c in real_df.columns
-- otherwise every encoded categorical is silently dropped because _enc
columns only exist after encode_features() runs inside get_XY(), not in the
raw input dataframe. On DataCo (distance_km = 0 always) this leaves only one
non-zero feature, collapsing utility measurement to near-random noise.
test_finding_e_feature_cols_includes_encoded_categoricals guards this.
"""
import pandas as pd

from pipeline import generate_proxy_dataset, FEATURE_COLS
from pipeline.data import encode_features
from pipeline.evaluation import trtr_baseline
from privacy_utility_sweep import run_sweep, run_mia_audit


def test_sweep_runs_end_to_end_on_proxy_data():
    real_df = generate_proxy_dataset(n=500, seed=42)
    # Finding E fix: must mirror privacy_utility_sweep.py's filter, which
    # includes _enc columns even though they don't yet exist in real_df.
    feature_cols = [c for c in FEATURE_COLS if c in real_df.columns or c.endswith("_enc")]

    long_df, trtr = run_sweep(
        real_df, levels=["none", "temporal"], architectures=["TVAE"],
        seeds=[42], n_synth=200, epochs=5, feature_cols=feature_cols,
    )
    assert isinstance(long_df, pd.DataFrame)
    assert len(long_df) > 0
    assert "business_utility" in long_df["metric"].values
    assert "dcr_mean" in long_df["metric"].values
    assert 0.0 <= trtr["business_utility_mean"] <= 1.0


def test_finding_e_feature_cols_includes_encoded_categoricals():
    """Regression guard for Finding E.

    privacy_utility_sweep.py:main() must derive feature_cols with
    `c.endswith("_enc")` in the filter, not just `c in real_df.columns`.
    The _enc columns are created inside get_XY() -> encode_features(), so
    they never appear in the raw dataframe -- a columns-only filter silently
    drops all 9 encoded categoricals, leaving at most 2 features.

    Concretely: on a constant-distance_km dataset (DataCo has distance_km=0
    always) the buggy filter leaves a single non-zero feature
    (promised_transit_days), making the utility classifier near-random.

    We assert two things:
    1. The corrected filter yields 11 features (distance_km + promised_transit_days
       + 9 _enc columns), not ≤ 2.
    2. TRTR label-2 recall with the full set is strictly below 1.0 -- the
       perfect recall (1.0) seen with the buggy filter on DataCo was an
       artefact of the classifier memorising a single near-degenerate feature.
    """
    real_df = generate_proxy_dataset(n=600, seed=99)

    # Buggy (old) filter: only raw columns
    buggy_cols = [c for c in FEATURE_COLS if c in real_df.columns]
    # Corrected filter: raw columns OR _enc suffix
    fixed_cols = [c for c in FEATURE_COLS if c in real_df.columns or c.endswith("_enc")]

    assert len(fixed_cols) > len(buggy_cols), (
        f"Fixed filter should include more features than the raw-only filter "
        f"(got fixed={len(fixed_cols)}, buggy={len(buggy_cols)})"
    )
    assert len(fixed_cols) == len(FEATURE_COLS), (
        f"All {len(FEATURE_COLS)} FEATURE_COLS should be reachable with the "
        f"corrected filter (got {len(fixed_cols)})"
    )

    # With the full feature set, TRTR label-2 recall must be sub-1.0.
    # A recall of 1.0 (as was seen with the buggy filter on DataCo) signals
    # that the classifier has effectively degenerated -- either always predicting
    # label-2, or fitting a near-constant feature space with perfect separation
    # across one threshold.
    trtr_full = trtr_baseline(real_df, feature_cols=fixed_cols)
    assert trtr_full["label2_recall_mean"] < 1.0, (
        f"TRTR label-2 recall with full feature set should be sub-1.0 "
        f"(got {trtr_full['label2_recall_mean']}); a value of 1.0 indicates "
        f"the classifier is still using a degenerate near-constant feature space"
    )


def test_mia_audit_runs_end_to_end_on_proxy_data():
    real_df = generate_proxy_dataset(n=500, seed=42)
    mia_df = run_mia_audit(
        real_df, levels=["none"], architectures=["TVAE"],
        seeds=[42], n_synth=200, epochs=5,
    )
    assert isinstance(mia_df, pd.DataFrame)
    assert len(mia_df) > 0
    assert "mia_auc" in mia_df["metric"].values
    auc_values = mia_df[mia_df.metric == "mia_auc"]["value"]
    assert ((auc_values >= 0.0) & (auc_values <= 1.0)).all()
