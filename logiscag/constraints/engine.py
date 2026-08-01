"""
Catalog engine: load the declarative YAML catalog, and audit a dataframe
against it.

For the seed eight constraints, this engine does NOT re-implement any check
-- it loads each catalog entry's `audit_key` and looks the violation count up
in pipeline.constraints.audit_constraints()'s output, which is the single
source of truth for what the code actually checks. This avoids the exact
failure mode the catalog exists to make visible: a declarative description
that quietly drifts from the executable check it claims to describe.

For custom constraints registered via the @constraint decorator (api.py),
the engine evaluates the predicate function row-wise, since there is no
existing Python implementation to delegate to.
"""

import os
import yaml
import pandas as pd

from pipeline.constraints import audit_constraints
from .api import CUSTOM_CONSTRAINTS

_DEFAULT_CATALOG_PATH = os.path.join(os.path.dirname(__file__), "catalog", "seed_constraints.yaml")


def load_catalog(path=None):
    """Load the declarative constraint catalog (default: the seed eight)."""
    path = path or _DEFAULT_CATALOG_PATH
    with open(path) as f:
        return yaml.safe_load(f)


def run_catalog_audit(df, catalog=None, include_custom=True, valid_combos=None):
    """Audit `df` against every entry in the catalog, returning one report row
    per constraint: its declarative metadata plus a live violation count.

    Seed-catalog entries are resolved via pipeline.constraints.audit_constraints
    (delegation, not reimplementation). Custom constraints registered via
    @constraint are evaluated directly against `df`, row-wise.

    Args:
        valid_combos: optional reference vocabulary for R4/R5 referential
            integrity, forwarded as-is to audit_constraints -- see
            pipeline.constraints.build_valid_carrier_combos(real_df). When
            auditing synthetic data, pass the combos built from the real
            training data, not from `df` itself.

    Returns a list of dicts, one per constraint, each with at least:
    id, name, category, type, severity, on_violation, rationale,
    violations, n, violation_rate_pct, source ("seed" | "custom").
    """
    catalog = catalog if catalog is not None else load_catalog()
    n = len(df)
    report = audit_constraints(df, verbose=False, valid_combos=valid_combos) if n else {}

    rows = []
    for entry in catalog:
        audit_key = entry.get("audit_key")
        violations = report.get(audit_key, 0) if audit_key else None
        rows.append({
            "id": entry["id"],
            "name": entry.get("name", entry["id"]),
            "category": entry.get("category"),
            "type": entry.get("type"),
            "severity": entry.get("severity"),
            "on_violation": entry.get("on_violation"),
            "rationale": entry.get("rationale"),
            "notes": entry.get("notes"),
            "audit_key": audit_key,
            "violations": violations,
            "n": n,
            "violation_rate_pct": round(100.0 * violations / n, 4) if (violations is not None and n) else None,
            "source": "seed",
        })

    if include_custom:
        for cid, spec in CUSTOM_CONSTRAINTS.items():
            if n:
                valid_mask = df.apply(spec["predicate_fn"], axis=1)
                violations = int((~valid_mask.astype(bool)).sum())
            else:
                violations = 0
            rows.append({
                "id": spec["id"],
                "name": spec["name"],
                "category": spec["category"],
                "type": spec["type"],
                "severity": spec["severity"],
                "on_violation": spec["on_violation"],
                "rationale": spec["rationale"],
                "notes": None,
                "audit_key": None,
                "violations": violations,
                "n": n,
                "violation_rate_pct": round(100.0 * violations / n, 4) if n else None,
                "source": "custom",
            })

    return rows


def catalog_audit_df(df, catalog=None, include_custom=True, valid_combos=None):
    """Same as run_catalog_audit, as a pandas DataFrame -- convenient for a report card."""
    return pd.DataFrame(run_catalog_audit(df, catalog=catalog, include_custom=include_custom,
                                           valid_combos=valid_combos))
