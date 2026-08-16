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
    """Eight entries, one per rule in the paper's Appendix C. This was 7 until
    2026-08-16, when the non-negativity half of the paper's R4 got its first
    executable check (KNOWN_ISSUES.md finding 7). If this count drops back to
    7, the catalog has silently stopped enumerating a rule the paper claims."""
    catalog = logiscag.constraints.load_catalog()
    ids = {c["id"] for c in catalog}
    assert len(catalog) == 8
    assert "R1_temporal_ordering" in ids
    assert "R4_promised_transit_non_negative" in ids
    assert "R8_carrier_calendar_plausibility" in ids
    # The paper numbers its rules R1..R8; the catalog ids must cover that range
    # exactly once each, with no gap of the kind R4 used to be.
    assert sorted(i.split("_")[0] for i in ids) == [f"R{n}" for n in range(1, 9)]
    for entry in catalog:
        assert entry["audit_key"], f"{entry['id']} must declare an audit_key"


def test_catalog_audit_keys_are_unique_and_all_resolve():
    """Guards the numbering hazard that made R4b's key awkward in the first
    place: three disagreeing numbering schemes (paper ids, catalog ids, audit
    keys). Two entries sharing an audit_key would silently report the same
    violation count twice; an audit_key that no longer exists in
    audit_constraints()'s output would silently report zero forever."""
    catalog = logiscag.constraints.load_catalog()
    keys = [c["audit_key"] for c in catalog]
    assert len(keys) == len(set(keys)), f"duplicate audit_key in catalog: {keys}"

    df = generate_proxy_dataset(n=50, seed=11)
    with pytest.warns(UserWarning, match="valid_combos"):
        report = audit_constraints(df, verbose=False)
    missing = [k for k in keys if k not in report]
    assert not missing, f"catalog audit_keys absent from audit_constraints output: {missing}"


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


# ---------------------------------------------------------------------------
# Fix 3: R4b non-negative promised transit (KNOWN_ISSUES.md finding 7)
# ---------------------------------------------------------------------------

R4B_KEY = "RPT_non_negative_promised_transit"  # Promised Transit non-negativity


def test_r4b_negative_promised_transit_violates_and_counts_as_hard():
    """The check finding 7 says existed nowhere. One clean row, one row whose
    promised delivery precedes its own promised shipment."""
    df = _baseline_df(n=2, promised_transit_days=[3.0, -1.5])
    with pytest.warns(UserWarning, match="valid_combos"):  # R5 fallback, irrelevant here
        report = audit_constraints(df, verbose=False)
    assert report[R4B_KEY] == 1
    # Must be HARD, not informational: the key deliberately contains neither
    # "soft" nor "_info", which is what folds it into total_hard_violations.
    # CONTRIBUTING.md calls out getting this wrong as a silent failure mode.
    assert report["total_hard_violations"] >= 1


def test_r4b_zero_and_positive_promised_transit_do_not_violate():
    """Boundary: the predicate is >= 0 per the paper, so a same-day promise
    (0.0) is VALID. Only a strictly negative window is incoherent.

    This is not a pedantic edge case: 9,737 of the 180,519 real DataCo rows sit
    exactly on 0.0. Tightening this to > 0 -- by false analogy with R2, whose
    predicate genuinely is strict -- would report all 9,737 as violations and
    inflate the real-data hard CVR by ~5.4 points."""
    df = _baseline_df(n=3, promised_transit_days=[0.0, 0.5, 10.0])
    with pytest.warns(UserWarning, match="valid_combos"):
        report = audit_constraints(df, verbose=False)
    assert report[R4B_KEY] == 0


def test_r4b_treats_unknown_promised_transit_as_non_violating():
    """NaN/unparseable is not a violation, matching R3's treatment of unknowns:
    an unmeasurable promise window is not evidence of a bad one. Guards against
    a future refactor that inverts the mask (`~(x >= 0)` would count NaN)."""
    df = _baseline_df(n=3, promised_transit_days=[float("nan"), None, "not-a-number"])
    with pytest.warns(UserWarning, match="valid_combos"):
        report = audit_constraints(df, verbose=False)
    assert report[R4B_KEY] == 0


def test_r4b_absent_column_omits_the_key_entirely():
    """The DISSERTATION schema has no promise columns. The check is
    presence-guarded, so the key must be absent rather than a misleading 0 --
    and audit_constraints must not raise."""
    df = _baseline_df(n=1)
    assert "promised_transit_days" not in df.columns
    with pytest.warns(UserWarning, match="valid_combos"):
        report = audit_constraints(df, verbose=False)
    assert R4B_KEY not in report


def test_r4b_key_does_not_collide_with_referential_integrity_key():
    """The audit-key namespace already spends `R4_` on referential integrity
    (the rule the PAPER numbers R5). R4b must be a distinct key carrying a
    distinct count, not an overwrite of it."""
    df = _baseline_df(n=2, promised_transit_days=[-1.0, -2.0], last_scac=["DHL", ""])
    with pytest.warns(UserWarning, match="valid_combos"):
        report = audit_constraints(df, verbose=False)
    assert report[R4B_KEY] == 2
    assert report["R4_referential_integrity"] == 1


def test_r4b_catalog_entry_delegates_and_declares_audit_only_enforcement():
    """The catalog entry must describe what the code actually does: a hard rule
    that is counted but never enforced (no cag_rejection_filter / SDV wiring).
    Describing it as `reject` would be the exact catalog-vs-code drift the
    catalog exists to prevent."""
    catalog = logiscag.constraints.load_catalog()
    entry = next(c for c in catalog if c["id"] == "R4_promised_transit_non_negative")
    assert entry["audit_key"] == R4B_KEY
    assert entry["type"] == "hard"
    assert entry["on_violation"] == "flag"
    assert "AUDIT-LAYER ONLY" in entry["notes"]

    synth_df = _baseline_df(n=4, promised_transit_days=[1.0, -1.0, -2.0, 0.0])
    with pytest.warns(UserWarning, match="valid_combos"):
        report = logiscag.constraints.run_catalog_audit(synth_df)
    row = next(r for r in report if r["id"] == "R4_promised_transit_non_negative")
    assert row["violations"] == 2
    assert row["violation_rate_pct"] == 50.0


def test_r4b_not_wired_into_generation_path_as_documented():
    """Pins the documented SCOPE of this fix. If someone later wires the rule
    into cag_rejection_filter or build_sdv_constraints, that is a welcome
    change -- but the catalog note and KNOWN_ISSUES.md finding 7 claim it is
    audit-only, and those claims must be updated in the same commit."""
    from pipeline.constraints import cag_rejection_filter

    df = _baseline_df(n=2, promised_transit_days=[1.0, -5.0])
    kept = cag_rejection_filter(df)
    assert len(kept) == 2, (
        "cag_rejection_filter now drops negative-promised-transit rows -- update "
        "the R4_promised_transit_non_negative catalog note (on_violation, and the "
        "AUDIT-LAYER ONLY paragraph) and KNOWN_ISSUES.md finding 7 to match"
    )


def test_r4b_dataco_promise_derivation_makes_it_a_noop_by_construction():
    """Documents WHY the real-data count is 0 on the DataCo benchmark, so the
    zero reads as a property of the adapter rather than as a toothless rule.
    dataco_to_canonical sets date_promise_shipment = shipping date and
    date_promise_delivery = shipping date + scheduled_days.clip(lower=0), so
    promised_transit_days is a clipped non-negative offset by construction.
    This reproduces that derivation and asserts the rule cannot fire on it,
    including for the negative raw scheduled-days input the clip absorbs.

    Measured on the real 180,519-row canonical dataset: 0 violations. Note the
    clip is belt-and-braces there -- DataCo's raw scheduled-days column only
    ever holds {0, 1, 2, 4}, so it has no negatives for the clip to absorb. The
    -3.0 input below is therefore synthetic, exercising the clip on input the
    real benchmark never supplies."""
    ship = pd.to_datetime(["2026-01-02"] * 4)
    raw_scheduled_days = pd.Series([0.0, 2.0, 6.0, -3.0])  # last one: clip absorbs it
    promise_delivery = ship + pd.to_timedelta(raw_scheduled_days.clip(lower=0), unit="D")
    promised_transit = (promise_delivery - ship).dt.total_seconds() / 86400.0

    df = _baseline_df(n=4, promised_transit_days=list(promised_transit))
    with pytest.warns(UserWarning, match="valid_combos"):
        report = audit_constraints(df, verbose=False)
    assert report[R4B_KEY] == 0


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
