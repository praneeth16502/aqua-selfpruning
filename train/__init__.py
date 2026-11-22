"""Training loops and experiment drivers. Step 2+.

Reproducibility: BLAS thread count changes the order of float sums inside matmul, and over ~6000 training steps that
is enough to land on a different final model (measured: Fisher 90% seed 0 gives 98.13% single-threaded and 98.08%
with the default Accelerate threading). So every `python -m train.*` entry point pins BLAS to 1 thread unless the
caller already set it.  This runs before numpy is imported (`train/__init__` is loaded first by `python -m`).
Numbers are bit-reproducible on the same BLAS build; a different BLAS (OpenBLAS vs Accelerate vs MKL) can still
differ in the last digits, which is well inside the seed-to-seed spread.
"""
import os as _os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")
