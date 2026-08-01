"""Verify the public logiscag facade resolves to the same underlying objects
as the internal modules it wraps (i.e. it re-exports, not reimplements)."""
import logiscag


def test_constraints_facade_matches_internals():
    from pipeline.constraints import audit_constraints, build_sdv_constraints
    assert logiscag.constraints.audit_constraints is audit_constraints
    assert logiscag.constraints.build_sdv_constraints is build_sdv_constraints


def test_cag_facade_matches_internals():
    from pipeline.pipeline import train_sdv_models
    assert logiscag.cag.train_sdv_models is train_sdv_models
    assert "strict+reject" in logiscag.cag.STRICTNESS_LADDER


def test_evaluation_facade_matches_internals():
    from privacy_utility_sweep import run_sweep, evaluate_synth
    assert logiscag.evaluation.run_sweep is run_sweep
    assert logiscag.evaluation.evaluate_synth is evaluate_synth


def test_dataco_adapter_facade_matches_internals():
    from dataco_adapter import load_dataco, dataco_to_canonical
    assert logiscag.adapters.dataco.load_dataco is load_dataco
    assert logiscag.adapters.dataco.dataco_to_canonical is dataco_to_canonical


def test_cag_generate_convenience_wrapper_uses_mock_without_sdv():
    from pipeline import generate_proxy_dataset
    real_df = generate_proxy_dataset(n=200, seed=1)
    synths = logiscag.cag.generate(real_df, level="none", architectures=["TVAE"],
                                    n_synth=50, epochs=5, seed=1)
    assert "TVAE" in synths
    assert len(synths["TVAE"]) == 50
