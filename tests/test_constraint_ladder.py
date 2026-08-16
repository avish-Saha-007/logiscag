"""
Verify the SDV-native strictness ladder produces a genuine monotonic
gradient in constraint *specification* (not enforcement -- see below).

History: this file used to be a standalone script (module-level code with
bare `assert` statements and a final `print("ALL ASSERTIONS PASSED...")`),
meant to be run directly via `python tests/test_constraint_ladder.py`. It
lived in `tests/` and matched the `test_*.py` naming pattern, so pytest
imported it during collection -- but since it defined zero `test_*`
functions, pytest silently collected nothing from it. It has therefore
never actually been part of the automated suite; running it required
someone to remember to invoke it directly. Converted here so it runs in
CI like everything else.

Scope note: these tests confirm that `build_sdv_constraints()` *returns* a
strictly-growing set of constraint specs as strictness increases. They do
**not** confirm SDV actually *enforces* those specs during generation --
that is a separate, already-documented issue (KNOWN_ISSUES.md finding #1:
the dict-style constraint spec is silently ignored by newer SDV versions,
which expect `sdv.cag` objects instead). A pass here means "the ladder is
specified correctly," not "the ladder is enforced correctly."
"""
from collections import Counter

import pandas as pd

from pipeline import generate_proxy_dataset
from pipeline.constraints import SDV_CONSTRAINT_LADDER, build_sdv_constraints


def _full_schema_df():
    """A frame with the complete production schema so every tier can activate."""
    return pd.DataFrame({
        "date_order_capture":          pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "date_ship_last_shipment":     pd.to_datetime(["2026-01-02", "2026-01-03"]),
        "date_delivery_last_shipment": pd.to_datetime(["2026-01-04", "2026-01-05"]),
        "date_promise_shipment":       pd.to_datetime(["2026-01-02", "2026-01-03"]),
        "date_promise_delivery":       pd.to_datetime(["2026-01-05", "2026-01-06"]),
        "transit_duration_days":       [2.0, 2.0],
        "capture_latency_days":        [0.1, 0.2],
        "promised_transit_days":       [3.0, 3.0],
        "shipment_buffer_days":        [1.0, 1.0],
        "distance_km":                 [10.0, 20.0],
        "last_scac":                   ["DHL", "UPS"],
        "carrier_service_code":        ["XP", "XG"],
        "delay_label":                 [0, 1],
    })


def _ladder_counts(df):
    """{level: number of constraint specs built at that level}, in one pass."""
    return {level: len(build_sdv_constraints(df, level)) for level in SDV_CONSTRAINT_LADDER}


def _classes(constraints):
    return dict(Counter(c["constraint_class"] for c in constraints))


def test_none_level_is_unconstrained():
    counts = _ladder_counts(_full_schema_df())
    assert counts["none"] == 0, f"'none' must build zero constraints; got {counts}"


def test_ladder_counts_strictly_increase_through_moderate():
    df = _full_schema_df()
    counts = _ladder_counts(df)
    assert counts["temporal"] > counts["none"], (
        f"'temporal' must add constraints over 'none'; got counts={counts}"
    )
    assert counts["moderate"] > counts["temporal"], (
        f"'moderate' must add constraints over 'temporal'; got counts={counts}"
    )


def test_strict_is_at_least_as_strict_as_moderate():
    # Not necessarily a strict ">" -- 'strict' may add referential constraints
    # that don't increase the raw count on every schema, so '>=' is the
    # correct (weaker) claim here, matching the original script's intent.
    counts = _ladder_counts(_full_schema_df())
    assert counts["strict"] >= counts["moderate"], (
        f"'strict' must not build fewer constraints than 'moderate'; got counts={counts}"
    )


def test_temporal_tier_includes_ordering_constraint():
    df = _full_schema_df()
    temporal_cs = build_sdv_constraints(df, "temporal")
    classes = _classes(temporal_cs)
    assert any(c["constraint_class"] == "Inequality" for c in temporal_cs), (
        f"'temporal' must include an Inequality (ordering) constraint; got classes={classes}"
    )


def test_strict_tier_includes_referential_constraint():
    df = _full_schema_df()
    strict_cs = build_sdv_constraints(df, "strict")
    classes = _classes(strict_cs)
    assert any(c["constraint_class"] == "FixedCombinations" for c in strict_cs), (
        f"'strict' must include a FixedCombinations (referential) constraint; got classes={classes}"
    )


def test_ladder_construction_is_robust_to_proxy_schema():
    """The proxy schema (used by smoke tests) may lack date/carrier columns
    entirely -- building constraints at every level against it must not
    raise, even though most tiers will legitimately build zero constraints
    on a schema with nothing to constrain."""
    proxy = generate_proxy_dataset(n=200, seed=1)
    for level in SDV_CONSTRAINT_LADDER:
        build_sdv_constraints(proxy, level)  # must not raise
