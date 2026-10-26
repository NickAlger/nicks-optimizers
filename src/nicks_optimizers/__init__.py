"""nicks_optimizers: matrix-free optimization methods.

Modules, in dependency order:

  backend                   how loops, branches and arrays are computed
                            (Backend records; numpy by default)
  trees                     tuple trees (nested tuples) and maps over them
  spaces                    the vector space structure: how vectors and
                            covectors are represented (VectorSpace records)
  metrics                   the Hilbert structure: the metric M, given by
                            operators, and norms
  dense                     small dense linear algebra in coordinates
  krylov                    Krylov processes (preconditioned Lanczos and CG)
  criteria                  tests shared by inexact Newton methods
  krylov_solvers            Krylov solvers for Newton steps (Lanczos, CG, and
                            a flexible solver for unconventional
                            preconditioners), as pieces a backend can compile
  trust_region_subproblem   the trust-region subproblem over a Krylov space
  newton_core               the shared parts of an inexact Newton iteration
                            (the state at a point, the step solve) as pure
                            pieces
  newton_driver             what the optimizers' drivers share
  trust_region_diagnostics  records, resumption state and text output
  trust_region_core         the trust-region update as a pure piece
  trust_region              trust_region_newton, the driver around them

backend, trees, spaces, metrics, dense, krylov and criteria do not depend on
the optimizers and are meant to be reused by other methods; krylov_solvers,
newton_core and newton_driver are shared by the Newton optimizers. The names
below are the public interface; lower-level pieces are available from their
modules.
"""

from .backend import Backend, loop_from_scan, loop_from_while, numpy_backend
from .criteria import forcing_term, gradient_converged, reduction_ratio
from .dense import pencil_ritz, solve_projected_trs, solve_sphere_trs, solve_sphere_trs_eig
from .krylov import (
    CGState,
    LanczosState,
    cg_advance,
    cg_curvature,
    cg_start,
    lanczos_advance,
    lanczos_start,
    lanczos_tridiagonal,
    tridiagonal,
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
    NumpyBlock,
    NumpySpace,
    Objective,
    Point,
    PreconditionerSolve,
    Retraction,
    TreeNumpySpace,
    TupleBlock,
    Vector,
    VectorBlock,
    VectorSpace,
    VectorToCovector,
    array_space,
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
    # backend
    "Backend",
    "loop_from_scan",
    "loop_from_while",
    "numpy_backend",
    # spaces
    "Array",
    "Coords",
    "Covector",
    "CovectorBlock",
    "CovectorToVector",
    "Gradient",
    "HessianAction",
    "NumpyBlock",
    "NumpySpace",
    "Objective",
    "Point",
    "PreconditionerSolve",
    "Retraction",
    "TreeNumpySpace",
    "TupleBlock",
    "Vector",
    "VectorBlock",
    "VectorSpace",
    "VectorToCovector",
    "array_space",
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
    "solve_projected_trs",
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
