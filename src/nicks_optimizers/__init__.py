"""nicks_optimizers: matrix-free optimization methods.

Modules, in dependency order:

  spaces                    how points, vectors and covectors are represented
                            (VectorSpace records; numpy by default)
  metrics                   inner products and norms (the geometry)
  dense                     small dense linear algebra in coordinates
  krylov                    Krylov processes (preconditioned Lanczos)
  criteria                  tests shared by inexact Newton methods
  trust_region_subproblem   the trust-region subproblem over a Krylov space
  trust_region_diagnostics  records, resumption state and text output
  trust_region              trust_region_newton

spaces, metrics, dense, krylov and criteria do not depend on the trust-region
modules and are meant to be reused by other optimizers. The names below are
the public interface; lower-level pieces are available from their modules.
"""

from .criteria import forcing_term, gradient_converged, reduction_ratio
from .dense import pencil_ritz, solve_sphere_trs, solve_sphere_trs_eig, tridiagonal
from .krylov import LanczosState, lanczos_advance, lanczos_start, lanczos_tridiagonal
from .metrics import (
    DualInnerProduct,
    FrozenDualInnerProduct,
    FrozenInnerProduct,
    InnerProduct,
    covector_norm,
    euclidean_dual_inner,
    euclidean_inner,
    vector_norm,
)
from .spaces import (
    Array,
    Coords,
    Covector,
    CovectorBlock,
    CovectorToVector,
    Gradient,
    HessianAction,
    NumpySpace,
    Objective,
    Point,
    PreconditionerSolve,
    Retraction,
    Vector,
    VectorBlock,
    VectorSpace,
    VectorToCovector,
    check_space,
    numpy_space,
    numpy_space_with_reductions,
    tuple_block_space,
)
from .trust_region import trust_region_newton
from .trust_region_diagnostics import (
    IterationInfo,
    IterationRecord,
    ResumeState,
    RunReason,
    RunSummary,
    format_header,
    format_lanczos_trace,
    format_row,
    format_summary,
    history_array,
    merge_summaries,
)
from .trust_region_subproblem import StopReason, SubproblemStats, solve_tr_subproblem

__all__ = [
    # spaces
    "Array",
    "Coords",
    "Covector",
    "CovectorBlock",
    "CovectorToVector",
    "Gradient",
    "HessianAction",
    "NumpySpace",
    "Objective",
    "Point",
    "PreconditionerSolve",
    "Retraction",
    "Vector",
    "VectorBlock",
    "VectorSpace",
    "VectorToCovector",
    "check_space",
    "numpy_space",
    "numpy_space_with_reductions",
    "tuple_block_space",
    # metrics
    "DualInnerProduct",
    "FrozenDualInnerProduct",
    "FrozenInnerProduct",
    "InnerProduct",
    "covector_norm",
    "euclidean_dual_inner",
    "euclidean_inner",
    "vector_norm",
    # dense
    "pencil_ritz",
    "solve_sphere_trs",
    "solve_sphere_trs_eig",
    "tridiagonal",
    # krylov
    "LanczosState",
    "lanczos_advance",
    "lanczos_start",
    "lanczos_tridiagonal",
    # criteria
    "forcing_term",
    "gradient_converged",
    "reduction_ratio",
    # trust region
    "IterationInfo",
    "IterationRecord",
    "ResumeState",
    "RunReason",
    "RunSummary",
    "StopReason",
    "SubproblemStats",
    "format_header",
    "format_lanczos_trace",
    "format_row",
    "format_summary",
    "history_array",
    "merge_summaries",
    "solve_tr_subproblem",
    "trust_region_newton",
]

__version__ = "0.1.0"
