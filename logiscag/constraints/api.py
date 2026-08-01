"""
Extension API for custom constraints.

Lets a user register their own row-level predicate without editing the seed
catalog or pipeline.constraints. A custom constraint is a plain function
`(row) -> bool` that returns True when the row is VALID (matching the seed
catalog's predicate convention), decorated with @constraint(...) so the
catalog engine (engine.py) can find and audit it alongside the seed eight.

    from logiscag.constraints import constraint

    @constraint(id="custom_cod_limit", name="Non-negative COD amount",
                category="arithmetic", type="hard")
    def cod_amount_nonnegative(row):
        return row["cash_on_delivery"] >= 0
"""

CUSTOM_CONSTRAINTS = {}


def constraint(id, name=None, category="custom", type="hard", severity="medium",
                on_violation="flag", rationale=""):
    """Register a row-level predicate function as a named, catalog-visible constraint.

    The decorated function must take a single row (a pandas Series, as from
    `df.apply(fn, axis=1)`) and return True if the row satisfies the
    constraint, False if it violates it.
    """
    if id in CUSTOM_CONSTRAINTS:
        raise ValueError(f"A custom constraint with id '{id}' is already registered.")

    def decorator(fn):
        CUSTOM_CONSTRAINTS[id] = {
            "id": id,
            "name": name or id,
            "category": category,
            "type": type,
            "severity": severity,
            "on_violation": on_violation,
            "rationale": rationale,
            "predicate_fn": fn,
        }
        return fn

    return decorator


def clear_custom_constraints():
    """Mainly for tests: reset the registry between test cases."""
    CUSTOM_CONSTRAINTS.clear()
