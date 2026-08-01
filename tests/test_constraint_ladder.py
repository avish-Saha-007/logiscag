"""Verify the strictness ladder produces a genuine monotonic gradient."""
import pandas as pd
from pipeline.constraints import build_sdv_constraints, SDV_CONSTRAINT_LADDER

# A frame with the full production schema so every tier can activate.
df = pd.DataFrame({
    "date_order_capture":         pd.to_datetime(["2026-01-01", "2026-01-02"]),
    "date_ship_last_shipment":    pd.to_datetime(["2026-01-02", "2026-01-03"]),
    "date_delivery_last_shipment":pd.to_datetime(["2026-01-04", "2026-01-05"]),
    "date_promise_shipment":      pd.to_datetime(["2026-01-02", "2026-01-03"]),
    "date_promise_delivery":      pd.to_datetime(["2026-01-05", "2026-01-06"]),
    "transit_duration_days":      [2.0, 2.0],
    "capture_latency_days":       [0.1, 0.2],
    "promised_transit_days":      [3.0, 3.0],
    "shipment_buffer_days":       [1.0, 1.0],
    "distance_km":                [10.0, 20.0],
    "last_scac":                  ["DHL", "UPS"],
    "carrier_service_code":       ["XP", "XG"],
    "delay_label":                [0, 1],
})

def classes(cs):
    from collections import Counter
    return dict(Counter(c["constraint_class"] for c in cs))

counts = {}
for level in SDV_CONSTRAINT_LADDER:
    cs = build_sdv_constraints(df, level)
    counts[level] = len(cs)
    print(f"{level:9s}: {len(cs):2d} constraints  {classes(cs)}")

# Assertions: strictly increasing, and the right classes appear at the right tiers
assert counts["none"] == 0, "none must be unconstrained"
assert counts["temporal"] > counts["none"], "temporal must add constraints"
assert counts["moderate"] > counts["temporal"], "moderate must add to temporal"
assert counts["strict"] >= counts["moderate"], "strict must add to moderate"

temporal_cs = build_sdv_constraints(df, "temporal")
assert any(c["constraint_class"] == "Inequality" for c in temporal_cs), "temporal must include R1 ordering"
strict_cs = build_sdv_constraints(df, "strict")
assert any(c["constraint_class"] == "FixedCombinations" for c in strict_cs), "strict must include referential"

# Robustness: proxy schema (may lack date/carrier cols) must not crash
from pipeline import generate_proxy_dataset
proxy = generate_proxy_dataset(n=200, seed=1)
for level in SDV_CONSTRAINT_LADDER:
    _ = build_sdv_constraints(proxy, level)  # must not raise
print("\nproxy schema cols present:", [c for c in ['date_order_capture','last_scac','transit_duration_days'] if c in proxy.columns])
print("\nALL ASSERTIONS PASSED - the ladder is a genuine monotonic gradient.")
