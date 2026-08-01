"""
Backward-compatibility shim for the original monolithic `viva_demo` module.

The functions and constants that used to live in a single `viva_demo.py` now
live in the `pipeline/` package (see pipeline/__init__.py's facade docstring,
which already documents this module as its intended compatibility target).
This file restores `import viva_demo as v` for callers/tests written against
the original name, without duplicating any logic.
"""

from pipeline import *  # noqa: F401,F403
