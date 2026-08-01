"""
dataco_adapter.py
=================
Map the public DataCo Smart Supply Chain dataset onto the canonical raw schema
consumed by pipeline.data.phase1_derived_features, so the sweep harness runs on
it unmodified. This is the cross-schema reproducibility port (manuscript Sec 8.4).

Mapping (DataCo -> canonical):
  order date (DateOrders)            -> date_order_capture, date_order_created_omni
  shipping date (DateOrders)         -> date_ship_last_shipment, date_promise_shipment
  shipping date + Days real          -> date_delivery_last_shipment   (transit = real days)
  shipping date + Days scheduled     -> date_promise_delivery         (sla_buffer = scheduled - real)
  Shipping Mode                      -> last_scac, last_shipment_carrier_service
  Customer Segment                   -> carrier_service_code, order_type
  Type                               -> line_type
  Department Name / Category Name    -> dpe_id / item_id
  Order Region / Market              -> dpe_shipnode / last_shipnode

Leakage-safe: Delivery Status, Late_delivery_risk, and the real transit are NEVER
mapped into predictor categoricals. transit_duration_days (= real days) is the
label source and must be excluded from the model feature set (the harness does this).
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from pipeline.data import phase1_derived_features

_DATACO_CSV = "DataCoSupplyChainDataset.csv"


def load_dataco(path=_DATACO_CSV) -> pd.DataFrame:
    return pd.read_csv(path, encoding="latin-1", low_memory=False)


def dataco_to_canonical(dc: pd.DataFrame) -> pd.DataFrame:
    """Return a canonical-schema dataframe with derived features + delay_label."""
    order = pd.to_datetime(dc["order date (DateOrders)"], errors="coerce")
    ship = pd.to_datetime(dc["shipping date (DateOrders)"], errors="coerce")
    real_days = pd.to_numeric(dc["Days for shipping (real)"], errors="coerce")
    sched_days = pd.to_numeric(dc["Days for shipment (scheduled)"], errors="coerce")

    # Derive delivery and promise-delivery from durations so the temporal chain is
    # clean (ship <= delivery always) and transit == real days exactly.
    delivery = ship + pd.to_timedelta(real_days.clip(lower=0), unit="D")
    promise_delivery = ship + pd.to_timedelta(sched_days.clip(lower=0), unit="D")

    def s(col):  # safe string categorical
        return dc[col].astype(str) if col in dc.columns else "UNK"

    canon = pd.DataFrame({
        "date_order_created_omni":       order,
        "date_order_capture":            order,
        "date_ship_last_shipment":       ship,
        "date_delivery_last_shipment":   delivery,
        "date_promise_shipment":         ship,
        "date_promise_delivery":         promise_delivery,
        "order_no":                      s("Order Id"),
        # categoricals (non-leaky)
        "dpe_id":                        s("Department Name"),
        "item_id":                       s("Category Name"),
        "line_type":                     s("Type"),
        "dpe_shipnode":                  s("Order Region"),
        "last_shipnode":                 s("Market"),
        "last_scac":                     s("Shipping Mode"),
        "last_shipment_carrier_service": s("Shipping Mode"),
        "carrier_service_code":          s("Customer Segment"),
        "order_type":                    s("Customer Segment"),
    })

    # Drop rows with unusable core fields, then compute derived features + label.
    canon = canon[order.notna() & ship.notna() & real_days.notna() & sched_days.notna()].copy()
    canon = phase1_derived_features(canon)
    # Keep only valid 3-class labels and finite transit
    canon = canon[pd.to_numeric(canon["delay_label"], errors="coerce").isin([0, 1, 2])]
    canon = canon[np.isfinite(pd.to_numeric(canon["transit_duration_days"], errors="coerce"))]
    return canon.reset_index(drop=True)


if __name__ == "__main__":
    dc = load_dataco(sys.argv[1] if len(sys.argv) > 1 else _DATACO_CSV)
    print(f"DataCo rows: {len(dc):,}")
    canon = dataco_to_canonical(dc)
    print(f"Canonical rows: {len(canon):,}")
    print("\nLabel distribution (0=on-time, 1=late<=2d, 2=late>2d):")
    print((canon["delay_label"].value_counts(normalize=True).sort_index() * 100).round(2))
    print("\nDerived-feature summary:")
    print(canon[["transit_duration_days", "promised_transit_days", "sla_buffer"]].describe().round(3))
