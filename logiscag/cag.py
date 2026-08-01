"""
Public CAG (Constraint-Augmented Generation) wrapper API.

Wraps pipeline.pipeline.train_sdv_models -- the generator-agnostic fit/sample
wrapper that enforces a chosen constraint set via SDV-native constraints and/or
post-hoc rejection sampling. The resample-to-target oversample loop
(_sample_until_valid) draws fresh batches from the fitted model until enough
valid rows accumulate, with a capped, graceful fallback -- it does not
bootstrap-pad a too-small survivor set.

STRICTNESS_LADDER (from the sweep harness) is re-exported here because it is
the canonical mapping from a named strictness level ("none", "temporal",
"moderate", "strict", "strict+reject") to the (constraint_level, reject) pair
that train_sdv_models / cag_rejection_filter consume.
"""

from pipeline.pipeline import train_sdv_models
from privacy_utility_sweep import STRICTNESS_LADDER, DEFAULT_LEVELS, set_global_seed

__all__ = [
    "train_sdv_models",
    "STRICTNESS_LADDER",
    "DEFAULT_LEVELS",
    "set_global_seed",
]


def generate(real_df, level="strict+reject", architectures=None, n_synth=10000,
             epochs=100, seed=42, **kwargs):
    """Convenience entry point: fit + sample under a named strictness level.

    Thin orchestration only -- looks up `level` in STRICTNESS_LADDER and calls
    train_sdv_models with the resulting constraint_level/post_filter; does not
    duplicate any constraint or sampling logic.
    """
    from pipeline.constraints import cag_rejection_filter

    spec = STRICTNESS_LADDER[level]
    set_global_seed(seed)
    return train_sdv_models(
        real_df, epochs=epochs, n_synth=n_synth,
        constraint_level=spec["constraint_level"],
        architectures=architectures,
        post_filter=cag_rejection_filter if spec["reject"] else None,
        **kwargs,
    )
