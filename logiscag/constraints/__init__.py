"""
Public constraint-library API.

Wraps pipeline.constraints (the eight last-mile denial constraints, R1-R8 in
the paper's Appendix C) without re-implementing any logic. See NOTES on known
code<->paper terminology gaps in the top-level package docstring
(`logiscag.__doc__`) before relying on exact rule semantics.

Also exposes a declarative YAML catalog (catalog/seed_constraints.yaml) that
describes the same eight rules in a governance-readable form, plus an
extension API (the @constraint decorator) for registering custom row-level
constraints. The catalog audits by delegating to audit_constraints for the
seed eight -- it is a documentation/audit-reporting layer, not a second
implementation of the rules.
"""

from pipeline.constraints import (
    audit_constraints,
    build_sdv_constraints,
    build_valid_carrier_combos,
    cag_rejection_filter,
    deterministic_repair,
    integrity_check_synthetic,
    SDV_CONSTRAINT_LADDER,
    SCAC_VOCABULARY,
    SERVICE_CODE_VOCABULARY,
    SCAC_NON_DELIVERY_WEEKDAYS,
    CAPTURE_LATENCY_TOLERANCE_DAYS,
)
from .api import constraint, CUSTOM_CONSTRAINTS, clear_custom_constraints
from .engine import load_catalog, run_catalog_audit, catalog_audit_df

__all__ = [
    "audit_constraints",
    "build_sdv_constraints",
    "build_valid_carrier_combos",
    "cag_rejection_filter",
    "deterministic_repair",
    "integrity_check_synthetic",
    "SDV_CONSTRAINT_LADDER",
    "SCAC_VOCABULARY",
    "SERVICE_CODE_VOCABULARY",
    "SCAC_NON_DELIVERY_WEEKDAYS",
    "CAPTURE_LATENCY_TOLERANCE_DAYS",
    "constraint",
    "CUSTOM_CONSTRAINTS",
    "clear_custom_constraints",
    "load_catalog",
    "run_catalog_audit",
    "catalog_audit_df",
]
