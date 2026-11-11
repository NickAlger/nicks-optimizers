"""The line-search Newton-Krylov optimizer, line_search_newton, and the
Newton directions it computes over a Krylov space.

line_search_newton is the inexact Newton method of trust_region_newton
globalized by a line search: at each iterate x it solves H d = -g
approximately over a Krylov space (the solvers of krylov_solvers in
line-search mode), then backtracks along retract(x, alpha d) from alpha = 1
(capped by step_max, if given) until the Armijo condition holds
(line_search.backtracking), and moves there. The problem pieces, operators,
spaces, backends, callback and resuming are those of trust_region_newton;
the records and display show the direction and the line search instead of
a radius.

The direction minimizes the model

    <g, d> + 0.5 <H_model d, d>

without a bound, until ||H d + g||_{M^{-1}} <= resid_tol (the surrogate
residual for the saddle-free model). Negative curvature is handled in one of
two ways:

  truncation     (default) the solve stops as soon as the Krylov space shows
                 nonpositive curvature, and returns the last direction found
                 with positive curvature, or, when it appears at the first
                 step, the preconditioned steepest descent direction
                 (-P^{-1} g; -M^{-1} g for the flexible solver). Every such
                 direction is a descent direction: <g, d> < 0.
  saddle-free    (saddle_free=True, Lanczos and flexible) H_model is |H|_M on
                 the Krylov space, with the M-Ritz values floored at
                 curvature_floor times the largest |theta|; always a descent
                 direction.

  solve_newton_direction           Lanczos over K_k(P^{-1} H, P^{-1} g): any
                                   symmetric positive definite P, and the
                                   saddle-free model. With a separate P and
                                   truncation it keeps no M-Gram matrix
                                   (the Newton system on the Krylov space
                                   does not involve M), so each step costs
                                   H, P^{-1} and one M^{-1} solve (the
                                   residual norm), and ||d||_M one product
                                   with M at the end.
  solve_newton_direction_cg        Newton-CG for P = M: five vectors, two
                                   reductions per step.
  solve_newton_direction_flexible  the flexible solver, for unconventional
                                   preconditioners; needs M and M^{-1}.

Each returns (d, SubproblemStats) with lam = 0 and stop reason "negative"
when it truncated.
"""

import time
from functools import partial
from typing import Any, Callable, Literal, Optional, SupportsFloat

import numpy as np

from .backend import Backend, compile_function, numpy_backend
from .criteria import gradient_converged
from .krylov_solvers import (
    SubproblemOperators,
    SubproblemStats,
    cg_solver,
    flexible_solver,
    lanczos_solver,
    solve_with,
    subproblem_stats,
)
from .line_search import ARMIJO
from .newton_core import OuterState, PointOperators
from .newton_driver import check_arguments, check_resumed_f, make_problem, number, step_solve
from .newton_line_search_core import DIRECTIONS, LineSearchOptions, line_search_finish
from .newton_line_search_diagnostics import (
    LineSearchInfo,
    LineSearchReason,
    LineSearchRecord,
    LineSearchResume,
    LineSearchSummary,
    display_ls_iteration,
    format_ls_summary,
)
from .spaces import CovectorBlockT, CovectorT, PointT, VectorBlockT, VectorSpace, VectorT

UNBOUNDED = float("inf")              # the radius the solvers carry in line-search mode (unused)


def solve_newton_direction(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                           g: CovectorT, hessvec: Callable[[VectorT], CovectorT],
                           metric_solve: Callable[[CovectorT], VectorT], resid_tol: float,
                           max_iter: int, *,
                           precond_solve: Optional[Callable[[CovectorT], VectorT]] = None,
                           metric_apply: Optional[Callable[[VectorT], CovectorT]] = None,
                           saddle_free: bool = False, curvature_floor: float = 1e-6,
                           breakdown_tol: float = 1e-12,
                           project_covector: Optional[Callable[[CovectorT], CovectorT]] = None,
                           start_vector: Optional[VectorT] = None,
                           backend: Optional[Backend] = None, capacity: Optional[int] = None
                           ) -> tuple[VectorT, SubproblemStats]:
    """The Newton direction by Lanczos over K_k(P^{-1} H, P^{-1} g) (see the
    module docstring). M is given by metric_solve (M^{-1}); P = M unless
    precond_solve (P^{-1}) is given, and then metric_apply (M) is needed too.
    start_vector = P^{-1} g, if already computed, saves one solve.

    Stops at the first of: ||R||_{M^{-1}} <= resid_tol ("tol"), nonpositive
    curvature without saddle_free ("negative", see the module docstring),
    breakdown (the Krylov space is invariant: the direction is exact), or
    max_iter steps (capped at the dimension when the space provides one,
    "max"). capacity, the initial room in the basis (None:
    DEFAULT_CAPACITY), doubles when the basis fills up. Raises ValueError if
    <g, P^{-1} g> is not positive. Returns (d, stats).
    """
    if precond_solve is not None and metric_apply is None:
        raise ValueError("a separate preconditioner needs metric_apply (M)")
    ops = SubproblemOperators(hessvec=hessvec, metric_solve=metric_solve,
                              precond_solve=precond_solve, metric_apply=metric_apply,
                              project_covector=project_covector)
    solver = lanczos_solver(max_iter, saddle_free=saddle_free, breakdown_tol=breakdown_tol,
                            trust_region=False, curvature_floor=curvature_floor)
    return solve_with(solver, backend, space, ops, g, UNBOUNDED, resid_tol, start_vector, capacity)


def solve_newton_direction_cg(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                              g: CovectorT, hessvec: Callable[[VectorT], CovectorT],
                              metric_solve: Callable[[CovectorT], VectorT], resid_tol: float,
                              max_iter: int, *,
                              project_covector: Optional[Callable[[CovectorT], CovectorT]] = None,
                              start_vector: Optional[VectorT] = None,
                              backend: Optional[Backend] = None) -> tuple[VectorT, SubproblemStats]:
    """The Newton direction by Newton-CG, preconditioned by M (P = M): CG on
    H d = -g from d = 0, stopping at ||r||_{M^{-1}} <= resid_tol ("tol"), at
    nonpositive curvature <H p, p> <= 0 ("negative": the current iterate, or
    p_0 = -M^{-1} g at the first step) or after max_iter steps ("max").
    start_vector = M^{-1} g, if already computed, saves one solve. Raises
    ValueError if <g, M^{-1} g> is not positive. Returns (d, stats).
    """
    ops = SubproblemOperators(hessvec=hessvec, metric_solve=metric_solve,
                              project_covector=project_covector)
    return solve_with(cg_solver(max_iter, trust_region=False), backend, space, ops, g, UNBOUNDED,
                      resid_tol, start_vector)


def solve_newton_direction_flexible(
        space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
        g: CovectorT, hessvec: Callable[[VectorT], CovectorT],
        metric_solve: Callable[[CovectorT], VectorT],
        metric_apply: Callable[[VectorT], CovectorT],
        precond_solve: Callable[[CovectorT], VectorT], resid_tol: float, max_iter: int, *,
        saddle_free: bool = False, curvature_floor: float = 1e-6, breakdown_tol: float = 1e-10,
        project_covector: Optional[Callable[[CovectorT], CovectorT]] = None,
        start_vector: Optional[VectorT] = None,
        backend: Optional[Backend] = None, capacity: Optional[int] = None
        ) -> tuple[VectorT, SubproblemStats]:
    """The Newton direction by the flexible solver (see
    solve_tr_subproblem_flexible for how the subspace grows, and the module
    docstring for negative curvature). Its first basis vector is M^{-1} g /
    ||g||_{M^{-1}}, so the truncation fallback is -M^{-1} g (P may not be
    symmetric, so -P^{-1} g need not be a descent direction).
    start_vector = M^{-1} g, if already computed, saves one solve. Raises
    ValueError if g = 0. Returns (d, stats).
    """
    ops = SubproblemOperators(hessvec=hessvec, metric_solve=metric_solve,
                              precond_solve=precond_solve, metric_apply=metric_apply,
                              project_covector=project_covector)
    solver = flexible_solver(max_iter, saddle_free=saddle_free, breakdown_tol=breakdown_tol,
                             trust_region=False, curvature_floor=curvature_floor)
    return solve_with(solver, backend, space, ops, g, UNBOUNDED, resid_tol, start_vector, capacity)


def line_search_newton(
        fun: Callable[[PointT], SupportsFloat],
        grad: Callable[[PointT], CovectorT],
        hessvec: Optional[Callable[[PointT, VectorT], CovectorT]],
        x0: PointT,
        *,
        space: Optional[VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT]] = None,
        retract: Optional[Callable[[PointT, VectorT], PointT]] = None,
        metric_solve: Optional[Callable[[PointT, CovectorT], VectorT]] = None,
        metric_apply: Optional[Callable[[PointT, VectorT], CovectorT]] = None,
        precond_solve: Optional[Callable[[PointT, CovectorT], VectorT]] = None,
        operators_at: Optional[Callable[[PointT, Any], PointOperators[VectorT, CovectorT]]] = None,
        point_data: Optional[Callable[[PointT], Any]] = None,
        project_covector: Optional[Callable[[PointT, CovectorT], CovectorT]] = None,
        saddle_free: bool = False,
        krylov: Literal["auto", "cg", "lanczos", "flexible"] = "auto",
        convergence_norm: Literal["metric", "preconditioner"] = "metric",
        gtol: float = 0.0, rtol: float = 1e-8, gnorm0: Optional[float] = None,
        forcing_max: float = 0.5, max_iter: int = 500, max_krylov: int = 50,
        c1: float = 1e-4, max_backtracks: int = 30, step_max: Optional[float] = None,
        curvature_floor: float = 1e-6,
        display: int = 0, print_fn: Callable[[str], None] = print,
        callback: Optional[Callable[[PointT, LineSearchInfo[VectorT, CovectorT]], Optional[bool]]] = None,
        resume: Optional[LineSearchResume] = None,
        g0: Optional[CovectorT] = None,
        backend: Optional[Backend] = None,
        krylov_capacity: Optional[int] = None,
        ) -> tuple[PointT, LineSearchSummary]:
    """Inexact Newton method with a backtracking line search.

    The problem pieces (fun, grad, hessvec or operators_at with point_data,
    space, retract, metric_solve, metric_apply, precond_solve,
    project_covector), the Krylov solver choice (krylov: "auto" is CG for
    P = M without saddle_free, else Lanczos), convergence_norm, gtol, rtol,
    gnorm0, forcing_max, max_iter, max_krylov, display, print_fn, callback,
    g0, backend and krylov_capacity are those of trust_region_newton.

    Each iteration computes a direction d by a Krylov solve of H d = -g to
    the relative tolerance min(forcing_max, sqrt(||g|| / ||g0||)), with
    nonpositive curvature truncated (or the saddle-free model |H|_M with its
    M-Ritz values floored at curvature_floor times the largest, see the
    module docstring), then backtracks along retract(x, alpha d) from
    alpha = 1, or from step_max / ||d||_M if that is smaller (no cap by
    default), until

        f(retract(x, alpha d)) <= f(x) + c1 alpha <g, d> + 10 eps max(1, |f(x)|),

    with safeguarded quadratic and cubic interpolation, at most
    max_backtracks evaluations of f. If no trial is accepted the run stops
    with reason "line_search" at the last accepted iterate.

    With a compiling backend the line search is compiled with the rest of
    the iteration; the functions given must be traceable (see
    trust_region_newton). The display (display=1) shows one row per
    iteration with the direction and the line search (see
    format_ls_row); display=2 adds the Krylov trace and the trials.

    Resuming: save x, info.resume (a LineSearchResume) and info.g in the
    callback; line_search_newton(..., x, resume=state, g0=g) continues
    exactly as the run would have.

    Returns (x, summary): the final iterate and a LineSearchSummary.
    """
    check_arguments(hessvec=hessvec, space=space, retract=retract, metric_solve=metric_solve,
                    metric_apply=metric_apply, precond_solve=precond_solve,
                    operators_at=operators_at, point_data=point_data, saddle_free=saddle_free,
                    krylov=krylov, convergence_norm=convergence_norm,
                    krylov_capacity=krylov_capacity)
    if not 0.0 < c1 < 1.0:
        raise ValueError(f"c1 must lie in (0, 1), not {c1}")
    if max_backtracks < 1:
        raise ValueError(f"max_backtracks must be positive, not {max_backtracks}")
    if step_max is not None and not step_max > 0.0:
        raise ValueError(f"step_max must be positive (or None for no cap), not {step_max}")
    if not 0.0 <= curvature_floor < 1.0:
        raise ValueError(f"curvature_floor must lie in [0, 1), not {curvature_floor}")

    bk = numpy_backend if backend is None else backend
    xp = bk.xp
    problem = make_problem(
        fun=fun, grad=grad, hessvec=hessvec, space=space, retract=retract,
        metric_solve=metric_solve, metric_apply=metric_apply, precond_solve=precond_solve,
        operators_at=operators_at, point_data=point_data, project_covector=project_covector,
        saddle_free=saddle_free, krylov=krylov, convergence_norm=convergence_norm,
        forcing_max=forcing_max, stall_tol=0.0, max_krylov=max_krylov, trust_region=False,
        curvature_floor=curvature_floor)
    pieces = step_solve(problem, bk, krylov_capacity)
    options = LineSearchOptions(c1=c1, max_backtracks=max_backtracks, step_max=step_max)
    finish_piece = compile_function(bk, partial(line_search_finish, problem, options, bk))
    unbounded = xp.asarray(np.inf)                  # the solvers carry a radius; unused here

    start = time.perf_counter()
    state: OuterState = pieces.start(x0, g0)
    fx = number(state.f)
    fun_calls, grad_calls, trials, growths = 1, (0 if g0 is not None else 1), 0, 0
    it = 0
    if resume is not None:
        check_resumed_f(resume.f, fx)
        gnorm0, it = resume.gnorm0, resume.next_iteration
    gnorm = number(state.gnorm)
    metric_solves, precond_solves = int(np.asarray(state.metric_solves)), int(np.asarray(state.precond_solves))
    metric_applies = 0
    if gnorm0 is None:
        gnorm0 = gnorm
    records: list[LineSearchRecord] = []
    while True:
        if gradient_converged(gnorm, gnorm0, gtol, rtol):
            reason: LineSearchReason = "converged"
            break
        if it >= max_iter:
            reason = "max_iter"
            break
        gnorm0_x = xp.asarray(gnorm0)
        carry, grown = pieces.solve(state, unbounded, gnorm0_x)
        growths += grown
        new_state, raw = finish_piece(state, carry, gnorm0_x)
        stats = subproblem_stats(raw.subproblem)
        n_trials = int(np.asarray(raw.trials))
        accepted = bool(np.asarray(raw.status) == ARMIJO)
        fun_calls += n_trials
        trials += n_trials
        precond_solves += int(np.asarray(raw.precond_solves))
        metric_applies += int(np.asarray(raw.metric_applies))
        record = LineSearchRecord(
            iteration=it, f=fx, gnorm=gnorm, gnorm_ratio=gnorm / gnorm0 if gnorm0 > 0.0 else 0.0,
            forcing=number(raw.forcing), direction=DIRECTIONS[int(np.asarray(raw.direction))],
            step_norm=number(raw.step_norm), slope=number(raw.slope), alpha=number(raw.alpha),
            trials=n_trials, predicted=number(raw.predicted), actual=number(raw.actual),
            accepted=accepted, elapsed=time.perf_counter() - start, subproblem=stats,
            trace_alpha=tuple(np.asarray(raw.trace_alpha)[:n_trials].tolist()),
            trace_f=tuple(np.asarray(raw.trace_f)[:n_trials].tolist()))
        records.append(record)
        state = new_state
        if accepted:
            fx, gnorm = number(state.f), number(state.gnorm)
            grad_calls += 1
            metric_solves += int(np.asarray(state.metric_solves))
            precond_solves += int(np.asarray(state.precond_solves))
        it += 1
        if display >= 1:
            display_ls_iteration(record, display, print_fn)
        stop = False
        if callback is not None:
            resume_state = LineSearchResume(gnorm0=gnorm0, next_iteration=it, f=fx)
            info: LineSearchInfo[VectorT, CovectorT] = LineSearchInfo(
                record=record, f=fx, g=state.g, step=raw.step, resume=resume_state)
            stop = bool(callback(state.x, info))
        if not accepted:
            reason = "line_search"
            break
        if stop:
            reason = "callback"
            break

    subs = [r.subproblem for r in records]
    summary = LineSearchSummary(
        reason=reason, iterations=tuple(records), f=fx, gnorm=gnorm, gnorm0=gnorm0,
        fun_calls=fun_calls, grad_calls=grad_calls,
        hessvec_calls=sum(s.hessvec_calls for s in subs),
        metric_solve_calls=metric_solves + sum(s.metric_solve_calls for s in subs),
        precond_solve_calls=precond_solves + sum(s.precond_solve_calls for s in subs),
        metric_apply_calls=metric_applies + sum(s.metric_apply_calls for s in subs),
        trials=trials, elapsed=time.perf_counter() - start,
        resume=LineSearchResume(gnorm0=gnorm0, next_iteration=it, f=fx),
        capacity_growths=growths)
    if display >= 1:
        print_fn(format_ls_summary(summary))
    x: PointT = state.x
    return x, summary
