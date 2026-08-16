"""Constraint checks and deterministic repairs for latest raw delivery schema."""

import warnings

import numpy as np
import pandas as pd

SCAC_VOCABULARY = {"DHL", "UPS", "FDX", "DPD", "USPS"}
SERVICE_CODE_VOCABULARY = {"XS", "XP", "XG", "XE", "XD"}
SCAC_NON_DELIVERY_WEEKDAYS = {
    "DHL": {6},
    "UPS": {6},
    "FDX": {6},
    "DPD": {5, 6},
    "USPS": {6},
    "default": {6},
}

# ~1-second OMS/DPE clock-skew tolerance, mirroring the same documented root
# cause as R1's excluded created<=capture sub-check below: the two systems'
# clocks disagree by about a second, which would otherwise register as a
# spurious capture-latency violation. Values more negative than this are
# genuine integrity violations; values within it are skew noise.
CAPTURE_LATENCY_TOLERANCE_DAYS = 1.0 / 86400  # 1 second, expressed in days


def build_valid_carrier_combos(real_df: pd.DataFrame) -> set:
    """
    Derive the reference vocabulary of valid (last_scac, carrier_service_code)
    combinations from REAL/training data, for R4/R5 referential-integrity
    membership checking (see audit_constraints' valid_combos parameter).

    Always derive this from real data, never from the synthetic data being
    audited -- deriving it from the synthetic data itself would make every
    synthetic combination trivially "valid" by construction.
    """
    if not {"last_scac", "carrier_service_code"}.issubset(real_df.columns):
        return set()
    scac_raw = real_df["last_scac"]
    svc_raw = real_df["carrier_service_code"]
    scac = scac_raw.astype(str)
    svc = svc_raw.astype(str)
    # Nullity is tested on the RAW series, before stringification. Under
    # pandas < 3, `astype(str)` rendered None/NaN as the literal string "nan",
    # so the sentinel-string filter below caught them; under pandas >= 3 the
    # new string dtype propagates NA through astype(str) instead, and a null
    # carrier leaked into the reference vocabulary -- which would then make a
    # synthetic row with a missing carrier pass the R4 membership check.
    # Regression-tested in tests/test_constraint_catalog.py.
    absent = {"", "nan", "None", "NaN", "<NA>", "NaT"}
    present = (
        ~scac_raw.isna()
        & ~svc_raw.isna()
        & ~scac.isin(absent)
        & ~svc.isin(absent)
    )
    return set(zip(scac[present], svc[present]))


def audit_constraints(df: pd.DataFrame, verbose: bool = True, valid_combos: set = None) -> dict:
    """
    Audits hard and soft business constraints and quantifies violations.

    Args:
        valid_combos: optional reference vocabulary of valid (last_scac,
            carrier_service_code) tuples, as returned by
            build_valid_carrier_combos(real_df). When supplied, R4/R5
            referential integrity is checked as true combination-membership
            against this vocabulary. When omitted (None, the default), R4/R5
            falls back to a presence-only check (non-null/non-empty) and
            raises a UserWarning each call, since that fallback is strictly
            weaker than membership checking and should not pass silently.
    """
    report = {}
    n = len(df)

    if n == 0:
        report.update(
            {
                "R1_temporal_order": 0,
                "R2_positive_transit": 0,
                "RPT_non_negative_promised_transit": 0,
                "R3_non_negative_capture_latency": 0,
                "R4_referential_integrity": 0,
                "R5_label_cardinality": 0,
                "R6_scac_calendar_soft": 0,
                "total_hard_violations": 0,
                "violation_rate_pct": 0.0,
            }
        )
        if verbose:
            print("\nConstraint audit: empty dataframe (all counts set to 0)")
        return report

    # R1: temporal ordering of the operational event chain.
    # Sub-condition (A) date_order_created_omni <= date_order_capture is intentionally
    # excluded: a 1-second OMS/DPE clock-skew artefact causes ~29% of rows to fail
    # this check spuriously. Keeping it would misrepresent 60 k rows as violations
    # when the root cause is a system write-ordering issue, not bad data.
    # The remaining chain (capture → ship → deliver, promise_ship → promise_deliver)
    # captures genuine sequencing violations only.
    _capture   = pd.to_datetime(df["date_order_capture"],           errors="coerce")
    _ship      = pd.to_datetime(df["date_ship_last_shipment"],      errors="coerce")
    _deliver   = pd.to_datetime(df["date_delivery_last_shipment"],  errors="coerce")

    _chain_ok = (_capture <= _ship) & (_ship <= _deliver)

    # Only enforce promise-date ordering when those columns are present and populated.
    if "date_promise_shipment" in df.columns and "date_promise_delivery" in df.columns:
        _p_ship    = pd.to_datetime(df["date_promise_shipment"], errors="coerce")
        _p_deliver = pd.to_datetime(df["date_promise_delivery"], errors="coerce")
        _promise_ok = _p_ship.isna() | _p_deliver.isna() | (_p_ship <= _p_deliver)
        _chain_ok = _chain_ok & _promise_ok

    r1 = ~_chain_ok
    report["R1_temporal_order"] = int(r1.sum())

    r2 = pd.to_numeric(df["transit_duration_days"], errors="coerce") <= 0
    report["R2_positive_transit"] = int(r2.sum())

    # RPT (Promised Transit non-negativity): non-negative promised transit -- the
    # non-negativity half of the paper's Appendix C R4. R4's other half
    # (promise_ship <= promise_delivery) is folded into R1's chain above; this
    # half had no AUDIT check until 2026-08-16 (KNOWN_ISSUES.md finding 7). It
    # is declared in build_sdv_constraints' _NONNEG_SCALARS at the moderate/strict
    # tiers, but inertly -- finding 1. cag_rejection_filter still does not check
    # it, so this rule is counted but never enforced; the catalog entry says so
    # explicitly.
    #
    # Key naming: RPT avoids any numeric prefix collision. The paper's R4 is
    # split between R1_temporal_order (ordering half) and here (non-negativity
    # half). The mapping table in README.md is the authoritative decoder for all
    # three numbering schemes (paper R1–R8, catalog ids, audit keys).
    #
    # Presence-guarded because the DISSERTATION schema has no promise columns
    # (pipeline.data sets this to NaN there). NaN is not counted as a violation
    # -- `NaN < 0` is False -- matching R3's treatment of unknowns: an
    # unmeasurable promise window is not evidence of a bad one.
    if "promised_transit_days" in df.columns:
        r4b = pd.to_numeric(df["promised_transit_days"], errors="coerce") < 0
        report["RPT_non_negative_promised_transit"] = int(r4b.sum())

    # R3: hard, with a clock-skew tolerance (CAPTURE_LATENCY_TOLERANCE_DAYS).
    # capture_latency_days is no longer a model feature, but it is a genuine
    # integrity rule: a row cannot be captured a meaningful amount of time
    # before it was created.
    if "capture_latency_days" in df.columns:
        r3 = pd.to_numeric(df["capture_latency_days"], errors="coerce") < -CAPTURE_LATENCY_TOLERANCE_DAYS
        report["R3_non_negative_capture_latency"] = int(r3.sum())

    if {"date_promise_delivery", "date_delivery_last_shipment"}.issubset(df.columns):
        sla = (
            pd.to_datetime(df["date_promise_delivery"], errors="coerce") -
            pd.to_datetime(df["date_delivery_last_shipment"], errors="coerce")
        ).dt.total_seconds() / 86400.0
        label = pd.to_numeric(df["delay_label"], errors="coerce")
        derived = pd.Series(
            np.where(sla >= 0, 0, np.where(-sla <= 2.0, 1, 2)),
            index=df.index,
        )
        report["R3b_label_sla_consistency"] = int((label != derived).fillna(False).sum())

    # R4: referential carrier/service integrity.
    if valid_combos is not None:
        # Combination-membership check against a reference vocabulary derived
        # from REAL/training data (build_valid_carrier_combos). A row is a
        # violation if its (scac, svc) pair was never observed in the
        # reference -- this also naturally catches missing/null values, since
        # a null pair is never a member of a real-data-derived vocabulary.
        _combo = pd.Series(
            list(zip(df["last_scac"].astype(str), df["carrier_service_code"].astype(str))),
            index=df.index,
        )
        r4 = ~_combo.isin(valid_combos)
    else:
        # Fallback: no reference vocabulary supplied. Degrades to the
        # pre-upgrade presence-only check (non-null/non-empty), which is
        # strictly weaker than true referential integrity -- it does not
        # verify combination membership. Not silent: always warns, so callers
        # notice they're getting the weaker check rather than assuming full
        # coverage.
        #
        # pipeline.pipeline installs a blanket `warnings.filterwarnings("ignore")`
        # at import time (unrelated to this rule, predates this fix, out of
        # scope to remove here), which transitively suppresses this warning
        # for the rest of the process once anything imports the pipeline
        # package -- i.e. always, in practice. simplefilter("always") inside a
        # local catch_warnings() block makes this specific warning immune to
        # that ambient filter (and to caller-side filters generally) without
        # touching or weakening the blanket filter itself.
        with warnings.catch_warnings():
            warnings.simplefilter("always", UserWarning)
            warnings.warn(
                "audit_constraints: no valid_combos supplied for R4/R5 referential "
                "integrity; falling back to a presence-only check (last_scac / "
                "carrier_service_code non-null), which does not verify combination "
                "membership against any reference vocabulary. Pass valid_combos "
                "(e.g. build_valid_carrier_combos(real_df)) for the full check.",
                UserWarning,
                stacklevel=2,
            )
        _scac_absent = df["last_scac"].isna() | df["last_scac"].astype(str).isin({"", "nan", "None", "NaN"})
        _svc_absent = df["carrier_service_code"].isna() | df["carrier_service_code"].astype(str).isin({"", "nan", "None", "NaN"})
        r4 = _scac_absent | _svc_absent
    report["R4_referential_integrity"] = int(r4.sum())

    r5 = ~pd.to_numeric(df["delay_label"], errors="coerce").isin([0, 1, 2])
    report["R5_label_cardinality"] = int(r5.sum())

    delivery_weekday = pd.to_datetime(df["date_delivery_last_shipment"], errors="coerce").dt.weekday
    r6 = pd.Series(False, index=df.index)
    for carrier_name, non_days in SCAC_NON_DELIVERY_WEEKDAYS.items():
        if carrier_name == "default":
            continue
        mask = (df["last_scac"].astype(str) == carrier_name)
        r6 |= mask & delivery_weekday.isin(non_days)
    known = set(SCAC_NON_DELIVERY_WEEKDAYS.keys()) - {"default"}
    default_mask = ~df["last_scac"].astype(str).isin(known)
    r6 |= default_mask & delivery_weekday.isin(SCAC_NON_DELIVERY_WEEKDAYS["default"])
    report["R6_scac_calendar_soft"] = int(r6.sum())

    hard_violations = sum(v for k, v in report.items() if "soft" not in k and "_info" not in k)
    report["total_hard_violations"] = hard_violations
    report["violation_rate_pct"] = round(hard_violations / n * 100, 4)

    if verbose:
        print("\nConstraint audit:")
        for rule, count in report.items():
            print(f"  {rule}: {count}")
    return report


def deterministic_repair(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies deterministic repairs for latest raw schema.
    """
    out = df.copy()
    c_created = pd.to_datetime(out["date_order_created_omni"], errors="coerce")
    c_capture = pd.to_datetime(out["date_order_capture"], errors="coerce")
    c_ship = pd.to_datetime(out["date_ship_last_shipment"], errors="coerce")
    c_deliv = pd.to_datetime(out["date_delivery_last_shipment"], errors="coerce")

    bad_capture = c_capture < c_created
    out.loc[bad_capture, "date_order_capture"] = out.loc[bad_capture, "date_order_created_omni"]

    c_capture = pd.to_datetime(out["date_order_capture"], errors="coerce")
    bad_ship = c_ship < c_capture
    out.loc[bad_ship, "date_ship_last_shipment"] = out.loc[bad_ship, "date_order_capture"]

    c_ship = pd.to_datetime(out["date_ship_last_shipment"], errors="coerce")
    bad_deliv = c_deliv < c_ship
    out.loc[bad_deliv, "date_delivery_last_shipment"] = out.loc[bad_deliv, "date_ship_last_shipment"]

    out["capture_latency_days"] = pd.to_numeric(out.get("capture_latency_days", 0), errors="coerce").fillna(0).clip(lower=0)
    out["transit_duration_days"] = pd.to_numeric(out.get("transit_duration_days", 0), errors="coerce").fillna(0.01).clip(lower=0.01)
    if "promised_transit_days" in out.columns:
        out["promised_transit_days"] = pd.to_numeric(out["promised_transit_days"], errors="coerce").fillna(0).clip(lower=0)
    if "shipment_buffer_days" in out.columns:
        out["shipment_buffer_days"] = pd.to_numeric(out["shipment_buffer_days"], errors="coerce").fillna(0)
    if "sla_buffer" in out.columns:
        out["sla_buffer"] = pd.to_numeric(out["sla_buffer"], errors="coerce").fillna(0)
    out = out[pd.to_numeric(out["delay_label"], errors="coerce").isin([0, 1, 2])].copy()
    return out


def integrity_check_synthetic(synth_df: pd.DataFrame) -> float:
    """
    Returns a single interpretable integrity-violation percentage for synthetic datasets.
	- **Technical**: Counts violations across positivity, non-negativity, and label-domain rules and computes `violations / n * 100` with rounding.

    Checks the synthetic DataFrame for violations of key integrity constraints and returns the percentage of records that violate any hard constraint.
    - **Technical**: Evaluates the synthetic dataset against critical integrity constraints (R2, R4, R5, R7) and computes an overall violation rate to assess the realism and quality of the generated data.
    - **Business**: Ensures that the synthetic data adheres to essential operational rules, such as positive transit durations and valid labels, to maintain its utility for downstream applications and decision-making.
    Args:
        synth_df (pd.DataFrame): The synthetic DataFrame to check.
    Returns:
        float: The percentage of records that violate any hard constraint.
    """
    n = len(synth_df)
    if n == 0:
        return 0.0
    violations = 0

    if "transit_duration_days" in synth_df.columns:
        violations += (pd.to_numeric(synth_df["transit_duration_days"], errors="coerce") <= 0).sum()
    if "capture_latency_days" in synth_df.columns:
        violations += (pd.to_numeric(synth_df["capture_latency_days"], errors="coerce") < 0).sum()
    if "delay_label" in synth_df.columns:
        violations += (~pd.to_numeric(synth_df["delay_label"], errors="coerce").isin([0, 1, 2])).sum()

    return round(violations / n * 100, 4)


# Ordered strictness tiers for the SDV-native constraint ladder. Each level is
# CUMULATIVE: it includes every constraint of the levels before it, so the
# feasible region contracts monotonically as strictness rises. This is what
# gives the privacy-utility sweep a genuine gradient to trace.
SDV_CONSTRAINT_LADDER = ["none", "temporal", "moderate", "strict"]

# Temporal event-chain ordering (R1): low_column must precede high_column.
_TEMPORAL_PAIRS = [
    ("date_order_capture",      "date_ship_last_shipment"),
    ("date_ship_last_shipment", "date_delivery_last_shipment"),
    ("date_promise_shipment",   "date_promise_delivery"),
]
# Arithmetic non-negativity (R2/R4-arith): column >= value.
_NONNEG_SCALARS = [
    ("transit_duration_days", 0.01),
    ("capture_latency_days",  0.0),
    ("promised_transit_days", 0.0),
    ("shipment_buffer_days",  0.0),
    ("distance_km",           0.0),
]


def build_sdv_constraints(train_df, constraint_level: str) -> list:
    """
    Build the SDV-native constraint list for a given strictness level.

    Cumulative ladder (each level adds to the previous):
      none      -> [] (unconstrained)
      temporal  -> R1 temporal ordering (Inequality) + R2 positive transit
      moderate  -> + arithmetic non-negativity (ScalarInequality)
      strict    -> + referential integrity (FixedCombinations carrier/service)

    All additions are guarded by column presence so the same function works on
    the production schema and the proxy schema. Temporal Inequality constraints
    require the date columns to be typed as `datetime` in the SDV metadata (the
    caller sets this); if your SDV version rejects them, fall back to the
    post-hoc CAG rejection path which enforces the same rules deterministically.
    """
    level = (constraint_level or "none").lower()
    # Backward-compat: legacy callers passed "relaxed" to mean "no constraints".
    if level == "relaxed":
        level = "none"
    cols = set(train_df.columns)

    def _has(c):
        return c in cols and train_df[c].notna().any()

    constraints = []
    if level == "none":
        return constraints

    # Tier 1 - temporal ordering (R1) + positive transit (R2)
    if level in {"temporal", "moderate", "strict"}:
        if _has("transit_duration_days"):
            constraints.append({
                "constraint_class": "ScalarInequality",
                "constraint_parameters": {"column_name": "transit_duration_days", "relation": ">=", "value": 0.01},
            })
        for lo, hi in _TEMPORAL_PAIRS:
            if _has(lo) and _has(hi):
                constraints.append({
                    "constraint_class": "Inequality",
                    "constraint_parameters": {
                        "low_column_name": lo, "high_column_name": hi, "strict_boundaries": False,
                    },
                })

    # Tier 2 - arithmetic non-negativity
    if level in {"moderate", "strict"}:
        for col, val in _NONNEG_SCALARS:
            if col == "transit_duration_days":
                continue  # already added in Tier 1
            if _has(col):
                constraints.append({
                    "constraint_class": "ScalarInequality",
                    "constraint_parameters": {"column_name": col, "relation": ">=", "value": val},
                })

    # Tier 3 - referential integrity via fixed (carrier, service) combinations.
    # Guarded against cardinality blow-up: FixedCombinations enumerates combos.
    if level == "strict":
        if _has("last_scac") and _has("carrier_service_code"):
            n_combos = train_df[["last_scac", "carrier_service_code"]].drop_duplicates().shape[0]
            if n_combos <= 200:
                constraints.append({
                    "constraint_class": "FixedCombinations",
                    "constraint_parameters": {"column_names": ["last_scac", "carrier_service_code"]},
                })

    return constraints


def cag_rejection_filter(synth_df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-hoc CAG rejection: drop rows that violate the HARD constraints,
    using the same logic as audit_constraints. Each check is guarded by
    column presence so it also runs on the proxy schema.

    Lives here (not in the sweep harness) so pipeline.pipeline can call it
    as a post_filter while oversampling from the fitted synthesizer -- see
    train_sdv_models' post_filter parameter.
    """
    df = synth_df
    mask = pd.Series(True, index=df.index)

    if "transit_duration_days" in df:
        mask &= pd.to_numeric(df["transit_duration_days"], errors="coerce") > 0
    if "delay_label" in df:
        mask &= pd.to_numeric(df["delay_label"], errors="coerce").isin([0, 1, 2])
    for col in ("last_scac", "carrier_service_code"):
        if col in df:
            present = ~(df[col].isna() | df[col].astype(str).isin({"", "nan", "None", "NaN"}))
            mask &= present
    tcols = {"date_order_capture", "date_ship_last_shipment", "date_delivery_last_shipment"}
    if tcols.issubset(df.columns):
        cap = pd.to_datetime(df["date_order_capture"], errors="coerce")
        shp = pd.to_datetime(df["date_ship_last_shipment"], errors="coerce")
        dlv = pd.to_datetime(df["date_delivery_last_shipment"], errors="coerce")
        mask &= (cap <= shp) & (shp <= dlv)

    return df[mask.fillna(False)].copy()
