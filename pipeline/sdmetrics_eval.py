"""SDMetrics-based secondary validation layer.

This module computes standardized SDMetrics quality/diagnostic reports as
supplementary evidence without changing primary pillar verdict logic.
"""

from __future__ import annotations

from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd


def _import_reports() -> Tuple[Any, Any, Any]:
    """Returns SDMetrics report classes and SDV metadata helper if available."""
    from sdmetrics.reports.single_table import QualityReport, DiagnosticReport
    from sdv.metadata import SingleTableMetadata

    return QualityReport, DiagnosticReport, SingleTableMetadata


def _records(df: pd.DataFrame) -> list:
    if df is None or df.empty:
        return []
    return df.replace({np.nan: None}).to_dict(orient="records")


def _sample_aligned(real_df: pd.DataFrame, synth_df: pd.DataFrame, sample_size: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    common_cols = [c for c in real_df.columns if c in synth_df.columns]
    if not common_cols:
        return pd.DataFrame(), pd.DataFrame(), []

    rr = real_df[common_cols].copy()
    ss = synth_df[common_cols].copy()

    n = min(len(rr), len(ss), int(sample_size))
    if n <= 0:
        return pd.DataFrame(columns=common_cols), pd.DataFrame(columns=common_cols), common_cols

    rr = rr.sample(n=n, random_state=seed, replace=False).reset_index(drop=True)
    ss = ss.sample(n=n, random_state=seed, replace=False).reset_index(drop=True)
    return rr, ss, common_cols


def evaluate_sdmetrics_pair(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    sample_size: int = 2000,
    include_column_pairs: bool = True,
    seed: int = 42,
) -> dict:
    """Computes SDMetrics quality and diagnostic summaries for one synthetic dataset."""
    result = {
        "status": "not_run",
        "error": None,
        "n_real": int(len(real_df)),
        "n_synth": int(len(synth_df)),
        "n_common_columns": 0,
        "common_columns": [],
        "sample_size_used": 0,
        "quality_overall": None,
        "quality_properties": [],
        "quality_details": {
            "column_shapes": [],
            "column_pair_trends": [],
        },
        "diagnostic_overall": None,
        "diagnostic_properties": [],
    }

    rr, ss, common_cols = _sample_aligned(real_df, synth_df, sample_size=sample_size, seed=seed)
    result["n_common_columns"] = int(len(common_cols))
    result["common_columns"] = common_cols
    result["sample_size_used"] = int(len(rr))

    if len(common_cols) == 0:
        result["status"] = "skipped"
        result["error"] = "No common columns between real and synthetic data"
        return result

    if len(rr) == 0 or len(ss) == 0:
        result["status"] = "skipped"
        result["error"] = "Insufficient rows for SDMetrics evaluation"
        return result

    try:
        QualityReport, DiagnosticReport, SingleTableMetadata = _import_reports()
    except Exception as exc:
        result["status"] = "skipped"
        result["error"] = f"sdmetrics_unavailable: {exc}"
        return result

    try:
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(rr)
        metadata_dict = metadata.to_dict()

        quality = QualityReport()
        quality.generate(real_data=rr, synthetic_data=ss, metadata=metadata_dict)

        result["quality_overall"] = float(quality.get_score())
        q_props = quality.get_properties()
        if isinstance(q_props, pd.DataFrame):
            result["quality_properties"] = _records(q_props)

        # Property names are API-defined; guard each call individually.
        try:
            col_shapes = quality.get_details(property_name="Column Shapes")
            if isinstance(col_shapes, pd.DataFrame):
                result["quality_details"]["column_shapes"] = _records(col_shapes)
        except Exception:
            pass

        if include_column_pairs:
            try:
                pair_trends = quality.get_details(property_name="Column Pair Trends")
                if isinstance(pair_trends, pd.DataFrame):
                    result["quality_details"]["column_pair_trends"] = _records(pair_trends)
            except Exception:
                pass

        diagnostic = DiagnosticReport()
        diagnostic.generate(real_data=rr, synthetic_data=ss, metadata=metadata_dict)

        result["diagnostic_overall"] = float(diagnostic.get_score())
        d_props = diagnostic.get_properties()
        if isinstance(d_props, pd.DataFrame):
            result["diagnostic_properties"] = _records(d_props)

        result["status"] = "completed"
        return result
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        return result


def evaluate_sdmetrics_all(
    real_df: pd.DataFrame,
    synths: Dict[str, pd.DataFrame],
    sample_size: int = 2000,
    include_column_pairs: bool = True,
    seed: int = 42,
) -> dict:
    """Computes SDMetrics secondary report for every architecture in `synths`."""
    out = {
        "status": "completed",
        "sample_size": int(sample_size),
        "include_column_pairs": bool(include_column_pairs),
        "architectures": {},
    }

    for arch, sdf in (synths or {}).items():
        out["architectures"][arch] = evaluate_sdmetrics_pair(
            real_df=real_df,
            synth_df=sdf,
            sample_size=sample_size,
            include_column_pairs=include_column_pairs,
            seed=seed,
        )

    return out
