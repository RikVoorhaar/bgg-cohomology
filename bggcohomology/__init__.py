import os

from . import (
    bggcomplex,
    compute_maps,
    compute_signs,
    la_modules,
    pbw,
    quantum_center,
    weight_set,
)

# ReadTheDocs cannot import cython modules.
on_rtd = os.environ.get("READTHEDOCS") == "True"
if not on_rtd:
    from . import cohomology

__version__ = "1.6"
