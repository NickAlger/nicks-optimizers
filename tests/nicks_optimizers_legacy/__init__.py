"""nicks_optimizers: matrix-free optimization methods.

(Test copy of the package before the backend refactor, see
tests/test_legacy.py. Changed since it was vendored in one respect, to
follow a deliberate rounding-level change of the refactor so that the
comparison stays bit for bit: numpy blocks store their members along the
leading axis, as they do since phase 7, which changes the BLAS calls of the
block products.)

Modules, in dependency order:

  trees                     tuple trees (nested tuples) and maps over them
  spaces                    the vector space structure: how vectors and
                            covectors are represented (VectorSpace records)
  metrics                   the Hilbert structure: the metric M, given by
                            operators, and norms
  dense                     small dense linear algebra in coordinates
  krylov                    Krylov processes (preconditioned Lanczos and CG)
  criteria                  tests shared by inexact Newton methods
  trust_region_subproblem   the trust-region subproblem over a Krylov space
                            (Lanczos, CG, and a flexible solver for
                            unconventional preconditioners)
  trust_region_diagnostics  records, resumption state and text output
  trust_region              trust_region_newton

trees, spaces, metrics, dense, krylov and criteria do not depend on the trust-region
modules and are meant to be reused by other optimizers. The names below are
the public interface; lower-level pieces are available from their modules.
"""

from .criteria import forcing_term, gradient_converged, reduction_ratio
from .dense import pencil_ritz, solve_sphere_trs, solve_sphere_trs_eig, tridiagonal
from .krylov import (
    CGState,
    LanczosState,
    cg_advance,
    cg_curvature,
    cg_start,
    lanczos_advance,
    lanczos_start,
    lanczos_tridiagonal,
)
from .metrics import (
    MetricApply,
    MetricSolve,
    covector_norm,
    euclidean_metric_apply,
    direct_sum,
    euclidean_metric_solve,
    extend_metric_gram,
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
    LeafBlock,
    NumpySpace,
    Objective,
    Point,
    PreconditionerSolve,
    Retraction,
    TreeNumpySpace,
    Vector,
    VectorBlock,
    VectorSpace,
    VectorToCovector,
    check_space,
    flat_retraction,
    numpy_space,
    numpy_space_with_reductions,
    tuple_block_space,
    tuple_tree_space,
)
from .trees import Tree, as_tuple_tree, tree_leaves, tree_map, tree_map2
from .trust_region import PointOperators, trust_region_newton
from .trust_region_diagnostics import (
    IterationInfo,
    IterationRecord,
    ResumeState,
    RunReason,
    RunSummary,
    format_header,
    format_krylov_trace,
    format_row,
    format_summary,
    history_array,
    merge_summaries,
)
from .trust_region_subproblem import (
    KrylovMethod,
    StopReason,
    SubproblemStats,
    solve_tr_subproblem,
    solve_tr_subproblem_cg,
    solve_tr_subproblem_flexible,
)

__all__ = [
    # spaces
    "Array",
    "Coords",
    "Covector",
    "CovectorBlock",
    "CovectorToVector",
    "Gradient",
    "HessianAction",
    "LeafBlock",
    "NumpySpace",
    "Objective",
    "Point",
    "PreconditionerSolve",
    "Retraction",
    "TreeNumpySpace",
    "Vector",
    "VectorBlock",
    "VectorSpace",
    "VectorToCovector",
    "check_space",
    "flat_retraction",
    "numpy_space",
    "numpy_space_with_reductions",
    "tuple_block_space",
    "tuple_tree_space",
    # trees
    "Tree",
    "as_tuple_tree",
    "tree_leaves",
    "tree_map",
    "tree_map2",
    # metrics
    "MetricApply",
    "MetricSolve",
    "covector_norm",
    "direct_sum",
    "euclidean_metric_apply",
    "euclidean_metric_solve",
    "extend_metric_gram",
    "vector_norm",
    # dense
    "pencil_ritz",
    "solve_sphere_trs",
    "solve_sphere_trs_eig",
    "tridiagonal",
    # krylov
    "CGState",
    "LanczosState",
    "cg_advance",
    "cg_curvature",
    "cg_start",
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
    "KrylovMethod",
    "PointOperators",
    "StopReason",
    "SubproblemStats",
    "format_header",
    "format_krylov_trace",
    "format_row",
    "format_summary",
    "history_array",
    "merge_summaries",
    "solve_tr_subproblem",
    "solve_tr_subproblem_cg",
    "solve_tr_subproblem_flexible",
    "trust_region_newton",
]

__version__ = "0.1.0"
