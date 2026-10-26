"""nicks_optimizers: matrix-free optimization methods.

Modules, in dependency order:

  backend                   how loops, branches and arrays are computed
                            (Backend records; numpy by default)
  trees                     tuple trees (nested tuples) and maps over them
  spaces                    the vector space structure: how vectors and
                            covectors are represented (VectorSpace records)
  metrics                   the Hilbert structure: the metric M, given by
                            operators, and norms
  transport                 vector transports: how vectors and covectors
                            move between tangent spaces
  dense                     small dense linear algebra in coordinates
  krylov                    Krylov processes (preconditioned Lanczos and CG)
  criteria                  tests shared by inexact Newton methods
  line_search               backtracking with the Armijo condition, as a
                            pure function a backend can compile
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
  newton_line_search_diagnostics  records, resumption state and text output
                            of the line search
  newton_line_search_core   the line-search part of an iteration as a pure
                            piece
  newton_line_search        line_search_newton, the driver, and Newton
                            directions for a line search
  lbfgs_memory              the memory of lbfgs and the two-loop recursion
                            in coordinates, as pure functions

backend, trees, spaces, metrics, transport, dense, krylov, criteria and
line_search do not depend on the optimizers and are meant to be reused by other methods; krylov_solvers,
newton_core and newton_driver are shared by the Newton optimizers. The names
below are the public interface; lower-level pieces are available from their
modules.
"""

from .backend import Backend, loop_from_scan, loop_from_while, numpy_backend
from .criteria import forcing_term, gradient_converged, reduction_ratio
from .dense import (
    pencil_ritz,
    solve_projected_newton,
    solve_projected_trs,
    solve_sphere_trs,
    solve_sphere_trs_eig,
)
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
from .line_search import LineSearchResult, backtracking
from .newton_line_search import (
    line_search_newton,
    solve_newton_direction,
    solve_newton_direction_cg,
    solve_newton_direction_flexible,
)
from .newton_line_search_diagnostics import (
    LineSearchInfo,
    LineSearchReason,
    LineSearchRecord,
    LineSearchResume,
    LineSearchSummary,
    format_ls_header,
    format_ls_krylov_trace,
    format_ls_row,
    format_ls_summary,
    format_ls_trials,
    ls_history_array,
    merge_ls_summaries,
)
from .transport import VectorTransport, identity_transport, riesz_covector_transport
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
    # transport
    "VectorTransport",
    "identity_transport",
    "riesz_covector_transport",
    # dense
    "pencil_ritz",
    "solve_projected_newton",
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
    # line search
    "LineSearchInfo",
    "LineSearchReason",
    "LineSearchRecord",
    "LineSearchResult",
    "LineSearchResume",
    "LineSearchSummary",
    "backtracking",
    "format_ls_header",
    "format_ls_krylov_trace",
    "format_ls_row",
    "format_ls_summary",
    "format_ls_trials",
    "line_search_newton",
    "ls_history_array",
    "merge_ls_summaries",
    "solve_newton_direction",
    "solve_newton_direction_cg",
    "solve_newton_direction_flexible",
]

__version__ = "0.1.0"
