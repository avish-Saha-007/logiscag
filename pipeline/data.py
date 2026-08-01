"""Data loading, feature engineering, and preflight checks for VIVA pipeline."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

RAW_DATE_COLS = [
    "date_order_created_omni",
    "date_order_capture",
    "date_ship_last_shipment",
    "date_delivery_last_shipment",
    "date_promise_delivery",
    "date_promise_shipment",
]

# Minimum date columns required for the raw operational schema.
# Promise-date columns are optional (absent in the DISSERTATION dataset).
_RAW_DATE_COLS_REQUIRED = [
    "date_order_capture",
    "date_ship_last_shipment",
    "date_delivery_last_shipment",
]

# Distribution-centre (ECDTC warehouse, Wilkes-Barre PA 18706).
_DC_LAT: float = 41.2044
_DC_LON: float = -75.9113


def _haversine_km(
    lat1: np.ndarray,
    lon1: float,
    lat2: float,
    lon2: float,
) -> np.ndarray:
    """Vectorised Haversine distance (km) between arrays of points and a fixed DC."""
    R = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _compute_distance_km(zip_series: pd.Series) -> pd.Series:
    """Return distance in km from each row's zip_code to the DC (zip 18706).

    Uses pgeocode for zip → lat/lon lookup. Rows with unresolvable zips get NaN,
    which is then filled with the dataset median so downstream models are unaffected.
    """
    try:
        import pgeocode  # lightweight, bundled GeoNames DB – no network call
        nomi = pgeocode.Nominatim("us")

        # Extract leading 5 digits to handle ZIP+4 format (e.g. '75103-8335').
        zip5 = zip_series.astype(str).str.extract(r"^(\d{5})", expand=False)

        unique_zips = zip5.dropna().unique()
        geo = nomi.query_postal_code(unique_zips.tolist())
        geo.index = unique_zips

        lats = zip5.map(geo["latitude"].to_dict())
        lons = zip5.map(geo["longitude"].to_dict())

        valid = lats.notna() & lons.notna()
        dist = pd.Series(np.nan, index=zip_series.index)
        dist[valid] = _haversine_km(
            lats[valid].values,
            _DC_LON,
            _DC_LAT,
            _DC_LON,
        )
        # For invalid zips fill with median distance.
        median_dist = dist.median()
        dist = dist.fillna(median_dist if not np.isnan(median_dist) else 0.0)
        return dist
    except Exception:
        # If pgeocode is unavailable, return zeros so pipeline still runs.
        return pd.Series(0.0, index=zip_series.index)

RAW_CATEGORICAL_COLS = [
    "dpe_id",
    "item_id",
    "line_type",
    "dpe_shipnode",
    "last_shipnode",
    "last_scac",
    "last_shipment_carrier_service",
    "carrier_service_code",
    "order_type",
]

ENCODED_FEATURE_COLS = [f"{c}_enc" for c in RAW_CATEGORICAL_COLS]

FEATURE_COLS = [
    # capture_latency_days removed: derived from date_order_created_omni which
    # exhibits a known 1-second OMS/DPE clock-skew artefact making it unreliable.
    # date_order_capture is retained in SDV training and used directly.
    # transit_duration_days and shipment_buffer_days removed: together with
    # promised_transit_days they sum to an exact reconstruction of sla_buffer
    # (sla_buffer = shipment_buffer_days + promised_transit_days - transit_duration_days),
    # which delay_label is a deterministic threshold of -- keeping all three let
    # the classifier solve for the label by arithmetic instead of learning real signal.
    "distance_km",          # DC-to-customer Haversine distance (km); 0.0 when zip unavailable
    "promised_transit_days",
    *ENCODED_FEATURE_COLS,
]

CANONICAL_LABEL_DISTRIBUTION = {
    0: 0.5442,
    1: 0.3758,
    2: 0.0800,
}

PROXY_LABEL_DISTRIBUTION = {
    0: 0.5442,
    1: 0.3758,
    2: 0.0800,
}


def _derive_delay_labels_from_transit(transit_days: pd.Series) -> pd.Series:
    clean = pd.to_numeric(transit_days, errors="coerce")
    if clean.notna().sum() == 0:
        return pd.Series(np.zeros(len(transit_days), dtype=int), index=transit_days.index)

    q1 = float(clean.quantile(CANONICAL_LABEL_DISTRIBUTION[0]))
    q2 = float(clean.quantile(CANONICAL_LABEL_DISTRIBUTION[0] + CANONICAL_LABEL_DISTRIBUTION[1]))

    labels = np.where(clean <= q1, 0, np.where(clean <= q2, 1, 2))
    return pd.Series(labels, index=transit_days.index).astype(int)


def _derive_delay_labels_from_sla_buffer(sla_buffer_days: pd.Series) -> pd.Series:
    """Derive labels from SLA adherence when promised delivery is available.

    0: on-time/early (buffer >= 0)
    1: short delay (0 < delay <= 2 days)
    2: long delay (delay > 2 days)
    """
    sla = pd.to_numeric(sla_buffer_days, errors="coerce")
    delay_days = -sla
    labels = np.where(sla >= 0, 0, np.where(delay_days <= 2.0, 1, 2))
    out = pd.Series(labels, index=sla_buffer_days.index)
    out[sla.isna()] = _derive_delay_labels_from_transit(pd.Series(np.nan, index=sla_buffer_days.index))[sla.isna()]
    return out.fillna(0).astype(int)


def _apply_latest_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in RAW_DATE_COLS:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    # capture_latency_days is retained for constraint-audit transparency only;
    # it is NOT a model feature (OMS/DPE clock-skew makes it unreliable as a predictor).
    if "date_order_created_omni" in out.columns:
        out["capture_latency_days"] = (
            out["date_order_capture"] - out["date_order_created_omni"]
        ).dt.total_seconds() / 86400.0
    else:
        out["capture_latency_days"] = np.nan

    # distance_km: Haversine distance from DC (ECDTC, Wilkes-Barre PA 18706) to customer zip.
    if "zip_code" in out.columns:
        out["distance_km"] = _compute_distance_km(out["zip_code"])
    else:
        out["distance_km"] = 0.0

    out["transit_duration_days"] = (
        out["date_delivery_last_shipment"] - out["date_ship_last_shipment"]
    ).dt.total_seconds() / 86400.0

    # Promise-date derived columns are optional (absent in DISSERTATION schema).
    _has_promise_cols = (
        "date_promise_delivery" in out.columns
        and "date_promise_shipment" in out.columns
    )
    if _has_promise_cols:
        out["promised_transit_days"] = (
            out["date_promise_delivery"] - out["date_promise_shipment"]
        ).dt.total_seconds() / 86400.0
        out["shipment_buffer_days"] = (
            out["date_promise_shipment"] - out["date_ship_last_shipment"]
        ).dt.total_seconds() / 86400.0
        out["sla_buffer"] = (
            out["date_promise_delivery"] - out["date_delivery_last_shipment"]
        ).dt.total_seconds() / 86400.0
    else:
        out["promised_transit_days"] = np.nan
        out["shipment_buffer_days"] = np.nan
        out["sla_buffer"] = np.nan

    # Transitional alias keeps older metric/test references valid during migration.
    out["transit_duration"] = out["transit_duration_days"]

    has_promise = _has_promise_cols and out["sla_buffer"].notna().any()
    if has_promise:
        out["delay_label"] = _derive_delay_labels_from_sla_buffer(out["sla_buffer"])
    else:
        out["delay_label"] = _derive_delay_labels_from_transit(out["transit_duration_days"])
    return out


def _is_latest_raw_schema(df: pd.DataFrame) -> bool:
    """Return True when df contains at minimum the required operational date columns
    plus all categorical columns. Promise-date columns are optional."""
    required = set(_RAW_DATE_COLS_REQUIRED + RAW_CATEGORICAL_COLS)
    return required.issubset(df.columns)


def generate_proxy_dataset(n: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Create proxy dataset using the latest raw operational schema."""
    rng = np.random.RandomState(seed)

    dpe_ids = [f"DPE_{i:03d}" for i in range(1, 51)]
    item_ids = [f"ITEM_{i:05d}" for i in range(1, 251)]
    line_types = ["NORMAL", "BACKORDER", "PREORDER"]
    shipnodes = [f"SN_{i:03d}" for i in range(1, 71)]
    scacs = ["DHL", "UPS", "FDX", "DPD", "USPS"]
    svc = ["EXPRESS", "PRIORITY", "GROUND", "ECONOMY", "SAME_DAY"]
    carrier_codes = ["XS", "XP", "XG", "XE", "XD"]
    order_types = ["B2C", "B2B", "MARKETPLACE"]

    base = datetime(2025, 1, 1)
    created_ts = [
        base + timedelta(days=int(rng.uniform(0, 180)), hours=int(rng.uniform(0, 24)))
        for _ in range(n)
    ]
    capture_lag = rng.uniform(0.01, 0.8, n)
    ship_lag = rng.uniform(0.05, 1.2, n)
    transit = np.maximum(0.08, rng.gamma(shape=2.3, scale=0.8, size=n))

    capture_ts = [created_ts[i] + timedelta(days=float(capture_lag[i])) for i in range(n)]
    ship_ts = [capture_ts[i] + timedelta(days=float(ship_lag[i])) for i in range(n)]
    delivery_ts = [ship_ts[i] + timedelta(days=float(transit[i])) for i in range(n)]
    promise_ship_ts = [capture_ts[i] + timedelta(days=float(rng.uniform(0.0, 0.6))) for i in range(n)]
    promised_transit = np.maximum(0.25, transit - rng.uniform(-0.25, 0.75, n))
    promise_deliv_ts = [promise_ship_ts[i] + timedelta(days=float(promised_transit[i])) for i in range(n)]

    df = pd.DataFrame(
        {
            "date_order_created_omni": created_ts,
            "date_order_capture": capture_ts,
            "order_no": [f"ORD_{seed}_{i:07d}" for i in range(n)],
            "dpe_id": rng.choice(dpe_ids, size=n),
            "item_id": rng.choice(item_ids, size=n),
            "line_type": rng.choice(line_types, size=n, p=[0.86, 0.09, 0.05]),
            "dpe_shipnode": rng.choice(shipnodes, size=n),
            "last_shipnode": rng.choice(shipnodes, size=n),
            "last_scac": rng.choice(scacs, size=n),
            "last_shipment_carrier_service": rng.choice(svc, size=n),
            "date_ship_last_shipment": ship_ts,
            "date_delivery_last_shipment": delivery_ts,
            "date_promise_delivery": promise_deliv_ts,
            "date_promise_shipment": promise_ship_ts,
            "carrier_service_code": rng.choice(carrier_codes, size=n),
            "order_type": rng.choice(order_types, size=n, p=[0.72, 0.08, 0.20]),
        }
    )
    return _apply_latest_derived_features(df)


def load_real_dataset(csv_path: str = "SCM_Delivery_Promise_Dataset_LATEST.csv") -> pd.DataFrame:
    """Load either latest raw schema or legacy schema and return latest-schema dataframe."""
    df = pd.read_csv(csv_path)

    if _is_latest_raw_schema(df):
        return _apply_latest_derived_features(df)

    # Legacy fallback: map old dataset into latest logical fields.
    old_required = {
        "order_timestamp",
        "shipping_date",
        "delivery_date_actual",
        "fc_id",
        "carrier",
        "service_level",
    }
    if old_required.issubset(df.columns):
        out = pd.DataFrame()
        out["date_order_created_omni"] = pd.to_datetime(df["order_timestamp"], errors="coerce")
        out["date_order_capture"] = out["date_order_created_omni"] + pd.to_timedelta(2, unit="h")
        out["order_no"] = df.get("order_id", df.index.to_series().map(lambda i: f"ORD_LEG_{i:07d}"))
        out["dpe_id"] = df.get("region", "UNKNOWN").astype(str)
        out["item_id"] = "ITEM_UNKNOWN"
        out["line_type"] = "NORMAL"
        out["dpe_shipnode"] = df.get("fc_id", "SN_UNKNOWN").astype(str)
        out["last_shipnode"] = df.get("fc_city", "SN_UNKNOWN").astype(str)
        out["last_scac"] = df.get("carrier", "SCAC_UNKNOWN").astype(str)
        out["last_shipment_carrier_service"] = df.get("service_level", "SERVICE_UNKNOWN").astype(str)
        out["date_ship_last_shipment"] = pd.to_datetime(df["shipping_date"], errors="coerce")
        out["date_delivery_last_shipment"] = pd.to_datetime(df["delivery_date_actual"], errors="coerce")
        out["date_promise_delivery"] = pd.to_datetime(df.get("promised_date"), errors="coerce")
        out["date_promise_shipment"] = out["date_order_capture"] + pd.to_timedelta(6, unit="h")
        out["carrier_service_code"] = df.get("service_level", "CODE_UNKNOWN").astype(str)
        out["order_type"] = "B2C"
        return _apply_latest_derived_features(out)

    raise ValueError("Unsupported dataset schema: expected latest raw schema or legacy SCM schema")


def phase1_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute derived features for latest raw schema."""
    if not _is_latest_raw_schema(df):
        return df.copy()
    return _apply_latest_derived_features(df)


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode raw categorical columns into *_enc columns."""
    out = df.copy()
    for col in RAW_CATEGORICAL_COLS:
        if col in out.columns:
            le = LabelEncoder()
            out[f"{col}_enc"] = le.fit_transform(out[col].astype(str))
    return out


def get_XY(df: pd.DataFrame, feature_cols: list = None):
    """Build model matrix X and target y."""
    df_enc = encode_features(df)
    cols = FEATURE_COLS if feature_cols is None else feature_cols
    available = [c for c in cols if c in df_enc.columns]
    X = df_enc[available].values.astype(float)
    y = df_enc["delay_label"].astype(int).values
    return X, y


def _mean_abs_shap_per_feature(shap_values, feature_names):
    values = shap_values
    if isinstance(values, list):
        if len(values) == 0:
            return {name: 0.0 for name in feature_names}
        stacked = np.stack([np.asarray(v) for v in values], axis=0)
        mean_abs = np.mean(np.abs(stacked), axis=(0, 1))
    else:
        arr = np.asarray(values)
        if arr.ndim == 3:
            mean_abs = np.mean(np.abs(arr), axis=(0, 2))
        elif arr.ndim == 2:
            mean_abs = np.mean(np.abs(arr), axis=0)
        else:
            mean_abs = np.zeros(len(feature_names), dtype=float)
    return {name: float(score) for name, score in zip(feature_names, mean_abs)}


def rank_features_by_shap(
    df: pd.DataFrame,
    feature_cols: list = None,
    n_estimators: int = 300,
    random_state: int = 42,
) -> dict:
    df_enc = encode_features(df)
    cols = FEATURE_COLS if feature_cols is None else feature_cols
    available = [c for c in cols if c in df_enc.columns]
    if not available:
        return {
            "method": "none",
            "available_features": [],
            "ranking": {},
            "error": "No available features for SHAP ranking",
        }

    X = df_enc[available].values.astype(float)
    y = df_enc["delay_label"].astype(int).values

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)

    ranking = None
    method = "model_importance_fallback"
    error = None
    try:
        import shap

        # Cap SHAP sample to 5k rows to avoid O(n) explosion on large datasets.
        shap_sample_size = min(5000, len(X))
        rng_shap = np.random.RandomState(random_state)
        shap_idx = rng_shap.choice(len(X), shap_sample_size, replace=False)
        X_shap = X[shap_idx]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)
        shap_scores = _mean_abs_shap_per_feature(shap_values, available)  # uses X_shap subset
        ranking = dict(sorted(shap_scores.items(), key=lambda kv: kv[1], reverse=True))
        method = "shap_tree_explainer"
    except Exception as exc:
        error = str(exc)
        importances = model.feature_importances_
        ranking = {
            available[i]: float(importances[i])
            for i in np.argsort(importances)[::-1]
        }

    return {
        "method": method,
        "available_features": available,
        "ranking": ranking,
        "error": error,
    }


def select_top_features(
    df: pd.DataFrame,
    top_k: int = None,
    min_score: float = None,
    feature_cols: list = None,
) -> dict:
    rank_res = rank_features_by_shap(df, feature_cols=feature_cols)
    ranking = rank_res.get("ranking", {})
    ordered = list(ranking.keys())

    selected = ordered
    if min_score is not None:
        selected = [f for f in selected if ranking.get(f, 0.0) >= float(min_score)]
    if top_k is not None:
        selected = selected[: max(1, int(top_k))]

    if not selected:
        fallback = rank_res.get("available_features", [])
        selected = fallback[:1] if fallback else []

    return {
        "method": rank_res.get("method"),
        "selected_features": selected,
        "ranking": ranking,
        "available_features": rank_res.get("available_features", []),
        "error": rank_res.get("error"),
    }


def run_preflight_validator(
    df: pd.DataFrame,
    expected_distribution: dict = None,
    ratio_tolerance: float = 0.03,
):
    """Fail-fast data governance checks for latest raw schema."""
    if expected_distribution is None:
        expected_distribution = CANONICAL_LABEL_DISTRIBUTION

    required = [
        "date_order_created_omni",
        "date_order_capture",
        "date_ship_last_shipment",
        "date_delivery_last_shipment",
        "date_promise_delivery",
        "date_promise_shipment",
        "transit_duration_days",
        "capture_latency_days",
        "delay_label",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Preflight failed: missing required columns {missing}")

    d_created = pd.to_datetime(df["date_order_created_omni"], errors="coerce")
    d_capture = pd.to_datetime(df["date_order_capture"], errors="coerce")
    d_ship = pd.to_datetime(df["date_ship_last_shipment"], errors="coerce")
    d_deliv = pd.to_datetime(df["date_delivery_last_shipment"], errors="coerce")
    d_p_ship = pd.to_datetime(df["date_promise_shipment"], errors="coerce")
    d_p_deliv = pd.to_datetime(df["date_promise_delivery"], errors="coerce")

    dt_fail_1 = int((d_capture < d_created).fillna(True).sum())
    dt_fail_2 = int((d_ship < d_capture).fillna(True).sum())
    dt_fail_3 = int((d_deliv < d_ship).fillna(True).sum())
    dt_fail_4 = int((d_p_deliv < d_p_ship).fillna(True).sum())

    transit = pd.to_numeric(df["transit_duration_days"], errors="coerce")
    sla = (
        d_p_deliv - d_deliv
    ).dt.total_seconds() / 86400.0
    if pd.Series(sla).notna().any():
        derived = _derive_delay_labels_from_sla_buffer(pd.Series(sla, index=df.index))
    else:
        derived = _derive_delay_labels_from_transit(transit)
    label = pd.to_numeric(df["delay_label"], errors="coerce").fillna(-1).astype(int)
    consistency = float((label == derived).mean()) if len(df) > 0 else 1.0

    observed = label.value_counts(normalize=True).sort_index().to_dict()
    ratio_failures = {}
    for cls, expected in expected_distribution.items():
        obs = observed.get(cls, 0.0)
        if abs(obs - expected) > ratio_tolerance:
            ratio_failures[cls] = {"observed": round(float(obs), 4), "expected": expected}

    leakage_columns = {
        "promised_date",
        "delivery_date_actual",
        "on_time_exact",
        "transit_delay_days",
        "delay_source",
    }
    active_leakage = [c for c in leakage_columns if c in FEATURE_COLS]

    failures = []
    if dt_fail_1 > 0 or dt_fail_2 > 0 or dt_fail_3 > 0 or dt_fail_4 > 0:
        failures.append(
            f"Temporal monotonicity failed: capture<created={dt_fail_1}, ship<capture={dt_fail_2}, delivery<ship={dt_fail_3}, promise_delivery<promise_shipment={dt_fail_4}"
        )
    if consistency < 0.95:
        failures.append(f"Label consistency failed: derived-label alignment={consistency:.4f} (<0.95)")
    if ratio_failures:
        failures.append(f"Class ratio failed: {ratio_failures}")
    if active_leakage:
        failures.append(f"Target leakage failed: forbidden features in training columns={active_leakage}")

    if failures:
        raise ValueError("Preflight failed:\n- " + "\n- ".join(failures))

    return {
        "status": "pass",
        "consistency": round(float(consistency), 4),
        "distribution": {int(k): round(float(v), 4) for k, v in observed.items()},
    }


def confidence_interval(scores, z: float = 1.96):
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return {"mean": None, "std": None, "ci95_low": None, "ci95_high": None}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    margin = z * (std / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "ci95_low": round(mean - margin, 4),
        "ci95_high": round(mean + margin, 4),
    }