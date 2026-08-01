"""
Regression test for the scoped warning filters in pipeline/pipeline.py.

pipeline.pipeline used to install a blanket `warnings.filterwarnings("ignore")`
at import time, suppressing every warning process-wide -- including SDV's own
FutureWarning that dict-style constraints are silently ignored (the root
cause of KNOWN_ISSUES.md finding 1). That warning is a correctness signal,
not cosmetic noise, and must keep surfacing. This test asserts the narrowed
filter set lets it through while still muting the four already-catalogued
benign third-party warnings (see KNOWN_ISSUES.md / CHANGES.md for the full
investigation and the exact filter rules).

Runs each check in a genuinely separate subprocess rather than in-process.
This is deliberate, not incidental: pytest's own warnings plugin overrides
the ambient warnings.filters state for the duration of every test (it needs
to, to build its end-of-session "warnings summary"), so warnings.catch_warnings()
called from inside a pytest test cannot observe what the application's own
filters actually suppress -- it sees pytest's "always" override instead. A
subprocess sees exactly what a real, non-pytest user running this code would
see: the actual filters pipeline.pipeline installs at import time, nothing
more.

Requires sdv (the warning only exists in the real, non-mock path) -- skipped
when sdv is unavailable, consistent with CONTRIBUTING.md's stated convention.
"""
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("sdv")

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_script(script: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True, text=True, cwd=str(_REPO_ROOT), timeout=120,
    )
    assert result.returncode == 0, f"subprocess failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    return result.stderr


def test_sdv_cag_warning_surfaces_while_benign_sdv_warnings_are_muted():
    stderr = _run_script("""
        import pipeline
        from pipeline import generate_proxy_dataset
        from pipeline.pipeline import train_sdv_models
        real_df = generate_proxy_dataset(n=300, seed=11)
        # constraint_level="strict" (the default) exercises sdv.cag's
        # add_constraints path, which is what emits the dict-style-constraint
        # FutureWarning this test is protecting.
        train_sdv_models(real_df, epochs=2, n_synth=50, architectures=["TVAE"])
    """)
    assert "no longer supports constraints using the older" in stderr, (
        "sdv.cag's dict-style-constraint FutureWarning did not surface in a "
        "real (non-pytest) run -- if this regresses, a warning filter is "
        "silently swallowing the correctness signal behind KNOWN_ISSUES.md "
        f"finding 1 again. Captured stderr:\n{stderr}"
    )
    assert "SingleTableMetadata' is deprecated" not in stderr
    assert "save_to_json" not in stderr


def test_pandas_and_sklearn_warnings_are_muted_in_a_real_run():
    """The pandas FutureWarning is attributed (by pandas' own stacklevel
    convention) to OUR calling module, not pandas' internal module -- so it
    can only be scoped by message, not by module=. Exercised via
    run_mia_audit, which groups by delay_label internally and fits a
    StandardScaler for the MIA distance computation."""
    stderr = _run_script("""
        import pipeline
        import privacy_utility_sweep as pus
        from pipeline import generate_proxy_dataset
        real_df = generate_proxy_dataset(n=300, seed=11)
        pus.run_mia_audit(real_df, levels=["none"], architectures=["TVAE"],
                           seeds=[1], n_synth=50, epochs=2)
    """)
    assert "DataFrameGroupBy.apply operated on the grouping columns" not in stderr
    assert "valid feature names" not in stderr  # sklearn StandardScaler UserWarning
