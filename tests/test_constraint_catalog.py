"""Tests for the declarative constraint catalog, engine, and @constraint API."""
import pandas as pd
import pytest

import logiscag
from pipeline import generate_proxy_dataset
from pipeline.constraints import (
    audit_constraints,
    build_valid_carrier_combos,
    CAPTURE_LATENCY_TOLERANCE_DAYS,
)


def test_catalog_loads_all_eight_seed_entries():
    catalog = logiscag.constraints.load_catalog()
    ids = {c["id"] for c in catalog}
    assert len(catalog) == 7  # R1+R4 grouped under one audit key, per Appendix C's own note
    assert "R1_temporal_ordering" in ids
    assert "R8_carrier_calendar_plausibility" in ids
    for entry in catalog:
        assert entry["audit_key"], f"{entry['id']} must declare an audit_key"


def test_catalog_audit_delegates_to_audit_constraints_not_reimplements():
    df = generate_proxy_dataset(n=300, seed=7)
    report = logiscag.constraints.run_catalog_audit(df)
    from pipeline.constraints import audit_constraints
    raw = audit_constraints(df, verbose=False)
    for row in report:
        if row["source"] == "seed":
            assert row["violations"] == raw[row["audit_key"]]


def _baseline_df(n=1, **overrides):
    """A minimally-valid audit_constraints input: audit_constraints accesses
    several columns unconditionally (R1's date chain, R2's transit, R4's
    carrier fields, R5's label), so every test needs a full row, not just the
    column(s) it's actually exercising. Dates are chosen to also avoid
    tripping the soft R6 carrier-calendar check, to keep these fixtures
    unambiguous across every rule.
    """
    base = {
        "date_order_capture": ["2026-01-01"] * n,
        "date_ship_last_shipment": ["2026-01-02"] * n,
        "date_delivery_last_shipment": ["2026-01-05"] * n,  # Monday: not in DHL's non-delivery set
        "transit_duration_days": [2.0] * n,
        "delay_label": [0] * n,
        "last_scac": ["DHL"] * n,
        "carrier_service_code": ["XP"] * n,
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_catalog_r3_and_r5_now_hard_and_resolved_in_notes():
    """R3/R5 were code<->paper mismatches at packaging time; both are now
    fixed in code, and the catalog should describe the current (fixed)
    behavior rather than the old gap."""
    catalog = logiscag.constraints.load_catalog()
    by_id = {c["id"]: c for c in catalog}
    assert by_id["R3_non_negative_capture_latency"]["type"] == "hard"
    assert by_id["R3_non_negative_capture_latency"]["audit_key"] == "R3_non_negative_capture_latency"
    assert "RESOLVED" in by_id["R3_non_negative_capture_latency"]["notes"]
    assert by_id["R5_referential_carrier_integrity"]["type"] == "hard"
    assert "UPGRADED" in by_id["R5_referential_carrier_integrity"]["notes"]
    assert "KnownCombos" in by_id["R5_referential_carrier_integrity"]["predicate"]


# ---------------------------------------------------------------------------
# Fix 1: R3 hard, with a clock-skew tolerance
# ---------------------------------------------------------------------------

def test_r3_within_tolerance_does_not_violate():
    df = _baseline_df(n=2, capture_latency_days=[0.0, -CAPTURE_LATENCY_TOLERANCE_DAYS / 2])
    with pytest.warns(UserWarning, match="valid_combos"):  # R5 fallback, irrelevant to this test
        report = audit_constraints(df, verbose=False)
    assert report["R3_non_negative_capture_latency"] == 0


def test_r3_beyond_tolerance_violates_and_counts_as_hard():
    df = _baseline_df(n=2, capture_latency_days=[0.0, -1.0])  # one clean row, one genuine violation (1 full day early)
    with pytest.warns(UserWarning, match="valid_combos"):
        report = audit_constraints(df, verbose=False)
    assert report["R3_non_negative_capture_latency"] == 1
    # R3 is no longer "_info"-suffixed/excluded -- it must now count toward
    # total_hard_violations (this is the entire point of Fix 1).
    assert report["total_hard_violations"] >= 1


def test_r3_dataco_capture_latency_identically_zero_is_a_noop():
    """DataCo's adapter sets order-creation == order-capture, so
    capture_latency_days is identically 0 -- the fix must be a no-op there."""
    df = _baseline_df(n=1000, capture_latency_days=[0.0] * 1000)
    with pytest.warns(UserWarning, match="valid_combos"):
        report = audit_constraints(df, verbose=False)
    assert report["R3_non_negative_capture_latency"] == 0


def test_r5_fallback_warns_on_every_call_despite_ambient_ignore_filter():
    """Regression test: pipeline.pipeline installs a blanket
    warnings.filterwarnings("ignore") at import time, which transitively
    suppresses any warning -- including this one -- for the rest of the
    process once anything imports the pipeline package. pytest.warns() alone
    doesn't catch this regression, because it forces simplefilter("always")
    for its own block regardless of ambient state, masking the bug. This test
    instead explicitly re-installs the same hostile "ignore everything"
    filter pipeline.pipeline sets, then confirms the warning still gets
    through on every one of several calls -- the way a real (non-pytest)
    caller actually experiences it."""
    df = _baseline_df(n=1)
    import warnings as _warnings
    with _warnings.catch_warnings(record=True) as record:
        _warnings.filterwarnings("ignore")  # reproduce pipeline.pipeline's blanket suppression
        audit_constraints(df, verbose=False)
        audit_constraints(df, verbose=False)
        audit_constraints(df, verbose=False)
    fallback_warnings = [w for w in record if "valid_combos" in str(w.message)]
    assert len(fallback_warnings) == 3, (
        f"expected the fallback warning on all 3 calls despite the ambient ignore "
        f"filter, got {len(fallback_warnings)} -- if this regresses, the warning is "
        "being silently suppressed again, exactly as it originally was"
    )


# ---------------------------------------------------------------------------
# Fix 2: R5 combination-membership referential integrity
# ---------------------------------------------------------------------------

def test_build_valid_carrier_combos_derives_from_real_data_only():
    real_df = pd.DataFrame({
        "last_scac": ["DHL", "UPS", "DHL", None],
        "carrier_service_code": ["XP", "XG", "XP", "XE"],
    })
    combos = build_valid_carrier_combos(real_df)
    assert combos == {("DHL", "XP"), ("UPS", "XG")}  # the null-scac row is excluded


def test_r5_fallback_with_no_valid_combos_warns_and_uses_presence_check():
    df = _baseline_df(n=2, last_scac=["DHL", ""], carrier_service_code=["XP", "XG"])
    with pytest.warns(UserWarning, match="valid_combos"):
        report = audit_constraints(df, verbose=False)
    assert report["R4_referential_integrity"] == 1  # only the empty-scac row, presence-only


def test_r5_membership_check_fires_on_unseen_combination(recwarn):
    real_df = pd.DataFrame({
        "last_scac": ["DHL", "UPS"],
        "carrier_service_code": ["XP", "XG"],
    })
    valid_combos = build_valid_carrier_combos(real_df)

    synth_df = _baseline_df(
        n=2,
        last_scac=["DHL", "DHL"],
        carrier_service_code=["XP", "XG"],  # (DHL, XG) was never seen in real_df
    )
    report = audit_constraints(synth_df, verbose=False, valid_combos=valid_combos)
    # No fallback warning should fire when a valid_combos reference is given.
    assert not any(issubclass(w.category, UserWarning) for w in recwarn.list)
    assert report["R4_referential_integrity"] == 1


def test_r5_membership_check_passes_when_combos_match_reference():
    real_df = pd.DataFrame({"last_scac": ["DHL", "UPS"], "carrier_service_code": ["XP", "XG"]})
    valid_combos = build_valid_carrier_combos(real_df)
    synth_df = _baseline_df(n=2, last_scac=["DHL", "UPS"], carrier_service_code=["XP", "XG"])
    report = audit_constraints(synth_df, verbose=False, valid_combos=valid_combos)
    assert report["R4_referential_integrity"] == 0


def test_r5_deriving_valid_combos_from_synthetic_data_is_not_the_documented_usage():
    """Sanity check the documented anti-pattern the docstring warns against:
    deriving valid_combos from the synthetic data itself makes every
    combination trivially valid, which is exactly why the contract requires
    real/training data."""
    synth_df = _baseline_df(n=1, last_scac=["ZZZ"], carrier_service_code=["???"])
    circular_combos = build_valid_carrier_combos(synth_df)  # derived from synth itself -- wrong usage
    report = audit_constraints(synth_df, verbose=False, valid_combos=circular_combos)
    assert report["R4_referential_integrity"] == 0  # trivially "valid" -- this is the failure mode to avoid


def test_catalog_threads_valid_combos_through_to_audit_constraints():
    real_df = pd.DataFrame({"last_scac": ["DHL"], "carrier_service_code": ["XP"]})
    valid_combos = build_valid_carrier_combos(real_df)
    synth_df = _baseline_df(n=2, last_scac=["DHL", "UPS"], carrier_service_code=["XP", "XG"])

    report = logiscag.constraints.run_catalog_audit(synth_df, valid_combos=valid_combos)
    r5_row = next(r for r in report if r["id"] == "R5_referential_carrier_integrity")
    assert r5_row["violations"] == 1  # the (UPS, XG) row is unseen in valid_combos


def test_custom_constraint_decorator_registers_and_audits():
    logiscag.constraints.clear_custom_constraints()

    @logiscag.constraints.constraint(id="test_custom_nonneg", category="arithmetic", type="hard")
    def nonneg_distance(row):
        return row.get("distance_km", 0) >= 0

    try:
        df = generate_proxy_dataset(n=100, seed=3)
        df.loc[0:9, "distance_km"] = -1
        report = logiscag.constraints.catalog_audit_df(df)
        custom_row = report[report["id"] == "test_custom_nonneg"].iloc[0]
        assert custom_row["violations"] == 10
        assert custom_row["source"] == "custom"
    finally:
        logiscag.constraints.clear_custom_constraints()


def test_duplicate_custom_constraint_id_raises():
    logiscag.constraints.clear_custom_constraints()

    @logiscag.constraints.constraint(id="dup_id")
    def f1(row):
        return True

    with pytest.raises(ValueError):
        @logiscag.constraints.constraint(id="dup_id")
        def f2(row):
            return True

    logiscag.constraints.clear_custom_constraints()
