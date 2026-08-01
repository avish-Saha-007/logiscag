"""
DataCo Smart Supply Chain dataset adapter (the paper's public benchmark adapter).

Wraps dataco_adapter.py (kept at the repo root and shipped as a top-level module
-- see pyproject.toml's py-modules) without modifying its mapping logic. See
load_dataco / dataco_to_canonical docstrings there for the exact column mapping
(order/shipping timestamps, the SLA-breach label derivation, and the
carrier/service categorical mapping).
"""

from dataco_adapter import load_dataco, dataco_to_canonical, _DATACO_CSV

__all__ = ["load_dataco", "dataco_to_canonical", "DEFAULT_CSV_NAME"]

DEFAULT_CSV_NAME = _DATACO_CSV
