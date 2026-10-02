"""Regression tests for KNOWN_ISSUES provenance audit (issue: "audit:
provenance of data-derived constraint components").

See outputs/provenance_audit_2026-10-02/REPORT.md for the full findings
table. These tests construct a toy train/test split and assert that
data-derived constraint/label components built from "whatever dataframe you
hand the function" are not, in fact, influenced by test-only rows -- i.e.
that train-only construction holds in practice, not just by convention.
"""
import numpy as np
import pandas as pd
import pytest

from pipeline.data import _apply_latest_derived_features


def _toy_transit_df(ship_ts, transit_days):
    """Minimal no-promise-column (corpus/DISSERTATION schema) frame: only the
    two date columns _apply_latest_derived_features needs to compute
    transit_duration_days and, in the absence of promise columns, delay_label
    via the quantile-cutoff path."""
    ship = pd.to_datetime(ship_ts)
    delivery = ship + pd.to_timedelta(np.asarray(transit_days, dtype=float), unit="D")
    return pd.DataFrame({
        "date_ship_last_shipment": ship,
        "date_delivery_last_shipment": delivery,
    })


def _toy_promise_df(ship_ts, transit_days, scheduled_days):
    """Minimal DataCo-style frame: promise columns present, so delay_label is
    derived via the fixed 2-day SLA-buffer threshold instead of a quantile fit."""
    ship = pd.to_datetime(ship_ts)
    delivery = ship + pd.to_timedelta(np.asarray(transit_days, dtype=float), unit="D")
    promise_delivery = ship + pd.to_timedelta(np.asarray(scheduled_days, dtype=float), unit="D")
    return pd.DataFrame({
        "date_ship_last_shipment": ship,
        "date_delivery_last_shipment": delivery,
        "date_promise_shipment": ship,
        "date_promise_delivery": promise_delivery,
    })


@pytest.mark.xfail(
    reason=(
        "KNOWN_ISSUES provenance audit (issue: 'audit: provenance of "
        "data-derived constraint components'): _derive_delay_labels_from_"
        "transit fits its quantile cut-offs (q1, q2) over whatever dataframe "
        "it is given, with no train/test split applied anywhere upstream in "
        "pipeline.data or pipeline.pipeline. On the no-promise-column "
        "(corpus/DISSERTATION) schema, this means delay_label's class "
        "boundaries are influenced by rows that later become the evaluation "
        "test fold. See outputs/provenance_audit_2026-10-02/REPORT.md finding #1. "
        "Does not affect the DataCo benchmark path, which uses a fixed "
        "threshold instead -- see the passing counterpart test below."
    ),
    strict=True,
)
def test_transit_quantile_label_cutoffs_are_not_influenced_by_test_only_rows():
    rng = np.random.RandomState(0)
    n_train = 400
    train_transit = rng.gamma(shape=2.3, scale=0.8, size=n_train)
    train_ship = pd.date_range("2025-01-01", periods=n_train, freq="h")
    train_df = _toy_transit_df(train_ship, train_transit)

    # Test-only rows: a cluster of extreme, never-seen-in-train transit times.
    # If label cut-offs were fit train-only, these could never move the
    # thresholds applied to the train rows above.
    n_test = 50
    test_transit = np.full(n_test, 500.0)
    test_ship = pd.date_range("2026-01-01", periods=n_test, freq="h")
    test_df = _toy_transit_df(test_ship, test_transit)

    # Labels the train rows WOULD get if cut-offs were fit on the train split alone.
    train_alone = _apply_latest_derived_features(train_df)
    train_only_labels = train_alone["delay_label"].to_numpy()

    # Labels the same train rows actually get in this codebase: thresholds are
    # fit on train+test concatenated, since no split happens anywhere before
    # this point in the real pipeline (pipeline.pipeline.run_pipeline passes
    # the entire loaded dataframe straight through phase1_derived_features).
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined = _apply_latest_derived_features(combined_df)
    train_labels_within_combined = combined["delay_label"].to_numpy()[:n_train]

    assert np.array_equal(train_only_labels, train_labels_within_combined), (
        "train-row delay_label assignments changed when test-only rows were "
        "present in the dataframe passed through _apply_latest_derived_features "
        "-- the quantile cut-offs are not train-only; see REPORT.md finding #1"
    )


def test_sla_fixed_threshold_label_cutoffs_are_not_influenced_by_test_only_rows():
    """Counterpart to the xfail above: the DataCo-style label rule
    (_derive_delay_labels_from_sla_buffer) uses a fixed 2-day business
    threshold, not a data-fitted quantile, so it is not susceptible to the
    provenance bug in finding #1. Pins that the DataCo benchmark path stays
    safe even though the corpus/DISSERTATION path (tested above) is not."""
    rng = np.random.RandomState(1)
    n_train = 400
    train_transit = rng.gamma(shape=2.3, scale=0.8, size=n_train)
    train_scheduled = np.clip(train_transit + rng.uniform(-0.5, 0.5, n_train), 0, None)
    train_ship = pd.date_range("2025-01-01", periods=n_train, freq="h")
    train_df = _toy_promise_df(train_ship, train_transit, train_scheduled)

    n_test = 50
    test_transit = np.full(n_test, 500.0)
    test_scheduled = np.full(n_test, 1.0)
    test_ship = pd.date_range("2026-01-01", periods=n_test, freq="h")
    test_df = _toy_promise_df(test_ship, test_transit, test_scheduled)

    train_alone = _apply_latest_derived_features(train_df)
    train_only_labels = train_alone["delay_label"].to_numpy()

    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined = _apply_latest_derived_features(combined_df)
    train_labels_within_combined = combined["delay_label"].to_numpy()[:n_train]

    assert np.array_equal(train_only_labels, train_labels_within_combined)
