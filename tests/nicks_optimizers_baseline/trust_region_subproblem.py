"""The trust-region subproblem, solved over a Krylov space.

    min_s  <g, s> + 0.5 <H_model s, s>    s.t.  ||s||_M <= radius,

where M is the metric (the geometry, see metrics) and H_model is H, or |H|_M
for the saddle-free model. A preconditioner P only chooses the subspace;
by default P = M. Three solvers, the solvers of krylov_solvers run to the
end:

  solve_tr_subproblem           Lanczos with full reorthogonalization over
                                K_k(P^{-1} H, P^{-1} g): any symmetric
                                positive definite P, and the saddle-free
                                model.
  solve_tr_subproblem_cg        Steihaug-Toint truncated conjugate gradients
                                for P = M and the plain model: five vectors,
                                two reductions per step.
  solve_tr_subproblem_flexible  a subspace grown by applying P to the KKT
                                residual, for preconditioners that are
                                nonsymmetric, nonlinear (inexact inner
                                solves) or stochastic; needs M and M^{-1}.

Lanczos and the flexible solver solve the small problem in M-orthonormal
coordinates z, where the ball is round,

    min_z  b^T z + 0.5 z^T A_model z    s.t.  ||z|| <= radius,

exactly, including the hard case (dense.solve_projected_trs; see
krylov_solvers for the coordinates).

Caveat when P differs from M: if the trust region is active, the exact step
solves (H + lam M) s = -g, which K_k(P^{-1} H, P^{-1} g) approximates slowly
unless M is close to a multiple of P. Steps always beat the Cauchy point
along -P^{-1} g, so global convergence is unaffected, but boundary solves stop
when the model stops improving (stall_tol) rather than on the residual.

The pieces, statistics and status codes live in krylov_solvers and are
re-exported here.
"""

from typing import Any, Callable, Optional

from .backend import Backend
from .dense import solve_projected_trs
from .krylov_solvers import (
    BOUNDARY,
    BREAKDOWN,
    DEFAULT_CAPACITY,
    FULL,
    INVALID,
    INVALID_START,
    MAX,
    METHODS,
    NEGATIVE,
    RUNNING,
    STALL,
    STOP_REASONS,
    TOL,
    KrylovMethod,
    KrylovSolver,
    RawSubproblem,
    StopReason,
    SubproblemOperators,
    SubproblemStats,
    as_integer,
    as_number,
    boundary_fraction,
    boundary_stalled,
    cg_solver,
    flexible_solver,
    initial_capacity,
    krylov_residual_norm,
    lanczos_solver,
    leading,
    solve_with,
    subproblem_stats,
)
from .spaces import CovectorBlockT, CovectorT, VectorBlockT, VectorSpace, VectorT

__all__ = [
    "solve_tr_subproblem", "solve_tr_subproblem_cg", "solve_tr_subproblem_flexible",
    "truncated_cg",
    # re-exported from krylov_solvers and dense
    "BOUNDARY", "BREAKDOWN", "DEFAULT_CAPACITY", "FULL", "INVALID", "INVALID_START", "MAX",
    "METHODS", "NEGATIVE", "RUNNING", "STALL", "STOP_REASONS", "TOL", "KrylovMethod",
    "KrylovSolver", "RawSubproblem", "StopReason", "SubproblemOperators", "SubproblemStats",
    "as_integer", "as_number", "boundary_fraction", "boundary_stalled", "cg_solver",
    "flexible_solver", "initial_capacity", "krylov_residual_norm", "lanczos_solver", "leading",
    "solve_projected_trs", "solve_with", "subproblem_stats",
]


def solve_tr_subproblem(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                        g: CovectorT, hessvec: Callable[[VectorT], CovectorT],
                        metric_solve: Callable[[CovectorT], VectorT],
                        radius: float, resid_tol: float, max_iter: int, *,
                        precond_solve: Optional[Callable[[CovectorT], VectorT]] = None,
                        metric_apply: Optional[Callable[[VectorT], CovectorT]] = None,
                        stall_tol: float = 1e-3, saddle_free: bool = False,
                        breakdown_tol: float = 1e-12,
                        project_covector: Optional[Callable[[CovectorT], CovectorT]] = None,
                        start_vector: Optional[VectorT] = None,
                        backend: Optional[Backend] = None, capacity: Optional[int] = None
                        ) -> tuple[VectorT, SubproblemStats]:
    """Approximately solve  min <g, s> + 0.5 <H_model s, s>  s.t.  ||s||_M <= radius
    over the growing Krylov spaces K_k(P^{-1} H, P^{-1} g) by Lanczos, where
    H_model is H, or |H|_M (via M-Ritz pairs) when saddle_free.

    The operators are frozen at the current point. M is given by
    metric_solve (M^{-1}); P = M unless precond_solve (P^{-1}) is given, and
    then metric_apply (M) is needed too, consistent with metric_solve.
    start_vector = P^{-1} g, if already computed, saves one solve.
    project_covector, for representations with redundant components, maps
    covectors to canonical representatives; it is applied to every Krylov
    covector (see krylov).

    Stops at the first of:
      * ||R||_{M^{-1}} <= resid_tol, with R the (surrogate) KKT residual
        (see krylov_residual_norm),
      * the trust region is active and the model stopped changing
        (stall_tol; see boundary_stalled),
      * Lanczos breakdown (an invariant subspace: the subspace solution is
        exact),
      * max_iter Lanczos steps (capped at the dimension when the space
        provides one).
    Every k >= 1 gives at least the model decrease of the Cauchy point along
    -P^{-1} g, since that point lies in K_1.

    backend (numpy by default) computes it; capacity, the initial room in
    the basis (None: DEFAULT_CAPACITY), doubles when the basis fills up. Raises
    ValueError if <g, P^{-1} g> is not positive. Returns (s, stats): the step
    and its SubproblemStats.
    """
    if precond_solve is not None and metric_apply is None:
        raise ValueError("a separate preconditioner needs metric_apply (M) for the Gram matrix")
    ops = SubproblemOperators(hessvec=hessvec, metric_solve=metric_solve,
                              precond_solve=precond_solve, metric_apply=metric_apply,
                              project_covector=project_covector)
    solver = lanczos_solver(max_iter, saddle_free=saddle_free, stall_tol=stall_tol,
                            breakdown_tol=breakdown_tol)
    return solve_with(solver, backend, space, ops, g, radius, resid_tol, start_vector, capacity)


def truncated_cg(backend: Backend, space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                 g: CovectorT, hessvec: Callable[[VectorT], CovectorT],
                 metric_solve: Callable[[CovectorT], VectorT], radius: Any, resid_tol: Any,
                 max_iter: int, *, project_covector: Optional[Callable[[CovectorT], CovectorT]] = None,
                 start_vector: Optional[VectorT] = None) -> tuple[VectorT, RawSubproblem]:
    """solve_tr_subproblem_cg as one pure function of the backend, returning
    the step and RawSubproblem statistics (status INVALID when
    <g, M^{-1} g> is not positive)."""
    solver = cg_solver(max_iter)
    ops = SubproblemOperators(hessvec=hessvec, metric_solve=metric_solve,
                              project_covector=project_covector)
    carry = solver.begin(backend, space, ops, g, radius, resid_tol, start_vector, None)
    s, raw = solver.result(backend, space, ops, solver.run(backend, space, ops, g, carry))
    return s, raw


def solve_tr_subproblem_cg(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                           g: CovectorT, hessvec: Callable[[VectorT], CovectorT],
                           metric_solve: Callable[[CovectorT], VectorT],
                           radius: float, resid_tol: float, max_iter: int, *,
                           project_covector: Optional[Callable[[CovectorT], CovectorT]] = None,
                           start_vector: Optional[VectorT] = None,
                           backend: Optional[Backend] = None
                           ) -> tuple[VectorT, SubproblemStats]:
    """Steihaug-Toint truncated CG for  min <g, s> + 0.5 <H s, s>
    s.t.  ||s||_M <= radius, preconditioned by M itself (P = M).

    Runs CG on H s = -g from s = 0, preconditioned with metric_solve, and
    stops at the first of:
      * ||r||_{M^{-1}} <= resid_tol, r = g + H s the model gradient ("tol");
      * the next CG iterate would leave the ball: the step stops where the
        segment to it crosses the boundary ("boundary");
      * negative curvature, <H p, p> <= 0: the step follows p to the boundary
        ("negative");
      * max_iter steps ("max"). Unlike Lanczos, CG is not capped at the
        dimension: without reorthogonalization it can need more than n steps
        in floating point.
    The iterates' M-norms increase (Steihaug), so the first crossing is the
    right place to stop, and the first step is the Cauchy point along
    -M^{-1} g: every step gives at least the Cauchy decrease. Unlike the
    Lanczos solver it does not continue on the boundary, keeps no basis
    (five vectors), and needs two reductions per step.

    project_covector maps each residual to its canonical representative,
    for redundant representations (see krylov). start_vector = M^{-1} g, if
    already computed, saves one solve. Raises ValueError if <g, M^{-1} g> is
    not positive. backend (numpy by default) computes it. Returns
    (s, stats).
    """
    ops = SubproblemOperators(hessvec=hessvec, metric_solve=metric_solve,
                              project_covector=project_covector)
    return solve_with(cg_solver(max_iter), backend, space, ops, g, radius, resid_tol, start_vector)


def solve_tr_subproblem_flexible(
        space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
        g: CovectorT, hessvec: Callable[[VectorT], CovectorT],
        metric_solve: Callable[[CovectorT], VectorT],
        metric_apply: Callable[[VectorT], CovectorT],
        precond_solve: Callable[[CovectorT], VectorT],
        radius: float, resid_tol: float, max_iter: int, *,
        saddle_free: bool = False, breakdown_tol: float = 1e-10,
        project_covector: Optional[Callable[[CovectorT], CovectorT]] = None,
        start_vector: Optional[VectorT] = None,
        backend: Optional[Backend] = None, capacity: Optional[int] = None
        ) -> tuple[VectorT, SubproblemStats]:
    """Approximately solve  min <g, s> + 0.5 <H_model s, s>  s.t.  ||s||_M <= radius
    over a subspace grown with a preconditioner that need not be symmetric,
    linear or deterministic: P^{-1} is any map from covectors to vectors
    (an inexact inner solve, a block triangular solve, sweeps of a randomized
    row-action method, a few steps of stochastic gradient descent, ...).

    P only proposes directions; nothing relies on its properties:
      * the basis starts with M^{-1} g, so every step gives at least the
        decrease of the Cauchy point along -M^{-1} g;
      * each new direction is P^{-1} R, R = H s + lam M s + g the KKT
        residual of the current step, made M-orthonormal to the basis by
        two Gram-Schmidt passes; if it is not finite or adds nothing (norm
        below breakdown_tol times its norm before orthogonalization),
        M^{-1} R is used instead ("fallback");
      * the projected Hessian T = Q^T H Q is formed from stored products H Q,
        and the small problem is solved exactly (as by solve_tr_subproblem,
        with G = I), so the step is the model's minimizer over the subspace
        and its model value is the true one;
      * the stopping test uses the explicit residual norm ||R||_{M^{-1}}.
    Applying P to the residual targets the system (H + lam M) s = -g that the
    step must solve, so a separate P that approximates H (or H + lam M) is
    effective on the boundary too, where Lanczos over K_k(P^{-1} H, P^{-1} g)
    stalls. With P = M (fixed) the subspace is the Krylov space of the
    Lanczos solver. With saddle_free, H_model is |H|_M (via M-Ritz pairs) and
    R the surrogate residual of solve_tr_subproblem.

    Each step costs one product with H, one application of P, one product
    with M (two with a fallback) and one solve with M^{-1}; the blocks Q,
    M Q and H Q are stored. start_vector = M^{-1} g, if already computed,
    saves one solve. project_covector (for redundant representations) is
    applied to every stored covector and residual; P's outputs must be valid
    vectors of the representation.

    Stops at the first of ||R||_{M^{-1}} <= resid_tol ("tol"), a breakdown
    ("breakdown", not even M^{-1} R adds a direction), or max_iter steps
    (capped at the dimension when the space provides one, "max"). backend
    (numpy by default) computes it; capacity, the initial room in the basis
    (None: DEFAULT_CAPACITY), doubles when the basis fills up. Raises ValueError if
    g = 0. Returns (s, stats).
    """
    ops = SubproblemOperators(hessvec=hessvec, metric_solve=metric_solve,
                              precond_solve=precond_solve, metric_apply=metric_apply,
                              project_covector=project_covector)
    solver = flexible_solver(max_iter, saddle_free=saddle_free, breakdown_tol=breakdown_tol)
    return solve_with(solver, backend, space, ops, g, radius, resid_tol, start_vector, capacity)
