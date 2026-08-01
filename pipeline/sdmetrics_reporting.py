"""Exports SDMetrics secondary validation artifacts into a dedicated report folder."""

from __future__ import annotations

import json
import os
from datetime import datetime

import pandas as pd


def export_sdmetrics_report(export_dir: str, sdmetrics_results: dict | None) -> str | None:
    """Writes SDMetrics report files under `<export_dir>/sdmetrics_report`.

    Returns the report directory path when written, else None.
    """
    if not export_dir:
        return None

    report_dir = os.path.join(export_dir, "sdmetrics_report")
    os.makedirs(report_dir, exist_ok=True)

    payload = sdmetrics_results or {"status": "not_run", "architectures": {}}

    with open(os.path.join(report_dir, "sdmetrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    arch_rows = []
    property_rows = []
    column_shape_rows = []
    pair_trend_rows = []

    for arch, res in (payload.get("architectures", {}) or {}).items():
        arch_rows.append(
            {
                "model": arch,
                "status": res.get("status"),
                "error": res.get("error"),
                "n_real": res.get("n_real"),
                "n_synth": res.get("n_synth"),
                "n_common_columns": res.get("n_common_columns"),
                "sample_size_used": res.get("sample_size_used"),
                "quality_overall": res.get("quality_overall"),
                "diagnostic_overall": res.get("diagnostic_overall"),
            }
        )

        for row in res.get("quality_properties", []) or []:
            property_rows.append(
                {
                    "model": arch,
                    "report": "quality",
                    "property": row.get("Property"),
                    "score": row.get("Score"),
                }
            )

        for row in res.get("diagnostic_properties", []) or []:
            property_rows.append(
                {
                    "model": arch,
                    "report": "diagnostic",
                    "property": row.get("Property"),
                    "score": row.get("Score"),
                }
            )

        for row in (res.get("quality_details", {}).get("column_shapes", []) or []):
            row = dict(row)
            row["model"] = arch
            column_shape_rows.append(row)

        for row in (res.get("quality_details", {}).get("column_pair_trends", []) or []):
            row = dict(row)
            row["model"] = arch
            pair_trend_rows.append(row)

    pd.DataFrame(arch_rows).to_csv(os.path.join(report_dir, "sdmetrics_summary.csv"), index=False)
    pd.DataFrame(property_rows).to_csv(os.path.join(report_dir, "sdmetrics_properties.csv"), index=False)
    pd.DataFrame(column_shape_rows).to_csv(os.path.join(report_dir, "sdmetrics_column_shapes.csv"), index=False)
    pd.DataFrame(pair_trend_rows).to_csv(os.path.join(report_dir, "sdmetrics_column_pair_trends.csv"), index=False)

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# SDMetrics Secondary Validation Report",
        "",
        f"Generated: {generated_at}",
        "",
        "This folder contains **secondary** standardized SDMetrics evidence.",
        "Primary pipeline verdict logic remains in the main report files.",
        "",
        "Files:",
        "- sdmetrics_summary.json: Full nested SDMetrics payload.",
        "- sdmetrics_summary.csv: One-row-per-model SDMetrics overview.",
        "- sdmetrics_properties.csv: Quality and diagnostic property scores.",
        "- sdmetrics_column_shapes.csv: Column-level shape similarity details.",
        "- sdmetrics_column_pair_trends.csv: Pairwise trend similarity details.",
    ]
    with open(os.path.join(report_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return report_dir
