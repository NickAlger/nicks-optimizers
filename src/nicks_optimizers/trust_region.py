"""Trust-region Newton method with an ellipsoidal trust region, an optional
preconditioner and an optional saddle-free model.

Three structures, given separately:

  space     the vector space structure (see spaces): adding, scaling and
            pairing vectors and covectors.
  metric    the Hilbert structure (see metrics), a symmetric positive
            definite M(x). It defines the trust region { s : ||s||_M <=
            radius }, the gradient norm ||g||_{M^{-1}} of the stopping test,
            the M-Ritz values of the saddle-free model and the Krylov
            residual norms.
  retract   the point structure: a step s moves x to retract(x, s).

A preconditioner P(x) is secondary to the metric: it only shapes the Krylov
space, and by default P = M.

Problem pieces
--------------
  fun(x) -> float                      objective
  grad(x) -> Covector                  differential df(x)
  hessvec(x, v) -> Covector            H(x) v, the Hessian of the pullback
                                       s -> f(retract(x, s)) at s = 0
  x0                                   starting point

Keyword arguments:
  space                                a VectorSpace (default: numpy arrays)
  retract(x, s) -> Point               the step update; default x + s for
                                       numpy arrays, required with a space
                                       (flat_retraction(space) gives x + s)
  metric_solve(x, w) -> Vector         M(x)^{-1} w; default M = I for numpy
                                       arrays, required with a space
                                       (euclidean_metric_solve gives M = I)
  metric_apply(x, v) -> Covector       M(x) v; needed only with
                                       precond_solve, and only with
                                       metric_solve (it must be its inverse)
  precond_solve(x, w) -> Vector        P(x)^{-1} w, if P should differ from M
                                       (e.g. a Hessian preconditioner inside a
                                       fixed metric)

With P = M (no precond_solve) only M^{-1} is ever applied: any symmetric
positive definite operator passed as metric_solve, a multigrid cycle for
instance, defines M as its inverse. Without saddle_free the subproblem is then
solved by truncated conjugate gradients (five vectors, two reductions per
step) unless krylov="lanczos" asks for the reorthogonalized Lanczos solver
(exact subspace solutions on the boundary, more robust in floating point).
With a separate P, Lanczos builds the M-Gram matrix with metric_apply and
measures residuals with metric_solve. krylov="flexible" (with a separate P)
grows the subspace by applying P to the KKT residual and forms the projected
Hessian explicitly, so P may be nonsymmetric, nonlinear or stochastic (see
solve_tr_subproblem_flexible).

Per-point setup, instead of the operator arguments:
  operators_at(x, data) -> PointOperators
                                       all the operators at x (hessvec,
                                       metric_solve, and optionally
                                       precond_solve and metric_apply). Pass
                                       hessvec=None and none of metric_solve,
                                       metric_apply, precond_solve with it.
  point_data(x) -> data                optional: the setup at x (assembled
                                       matrices, a multigrid hierarchy, ...),
                                       computed once per new iterate and
                                       passed to operators_at, which may be
                                       called several times per point and
                                       should only wrap data in functions.
                                       Without point_data, data is None.
                                       With the numpy backend data may be
                                       any object, e.g. the PointOperators
                                       themselves (point_data=setup,
                                       operators_at=lambda x, ops: ops); with
                                       a compiling backend it must be arrays
                                       in tuples or dicts.

Optional for representations with redundant components (e.g. gauged
tangent vectors of a low-rank manifold):
  project_covector(x, w) -> Covector   the canonical representative of w,
                                       dropping components that pair to zero
                                       with every vector at x. Applied to
                                       gradients and to every Krylov covector,
                                       where the Krylov recurrences would
                                       otherwise amplify rounding errors in
                                       the invisible components (see krylov).

The radius carries over from one point to the next, so the theory wants
M(x) uniformly equivalent to a fixed norm over the iterates; an x-dependent
M also moves the norm of the stopping test.

Each iteration computes a step with solve_tr_subproblem,
solve_tr_subproblem_cg or solve_tr_subproblem_flexible (see
trust_region_subproblem) and records what happened (see
trust_region_diagnostics).

Backends: the iterations are pure pieces (trust_region_core) that a
Backend (see backend) may compile; this function is the Python driver
around them, and keeps the stopping tests, records, display, callback and
resuming. The default numpy backend runs the pieces as plain Python. With a
compiling backend, every function given (fun, grad, the operators,
retract, project_covector, point_data, operators_at) must be traceable by
it: no Python control flow on values, and arrays out (fun returns a 0-d
array rather than a float).
"""

import time
from functools import partial
from typing import Any, Callable, Literal, Optional, SupportsFloat

import numpy as np

from .backend import Backend, compile_function, numpy_backend
from .criteria import gradient_converged
from .newton_driver import check_arguments, check_resumed_f, make_problem, number, step_solve
from .spaces import (
    CovectorBlockT,
    CovectorT,
    PointT,
    VectorBlockT,
    VectorSpace,
    VectorT,
)
from .trust_region_core import OuterState, PointOperators, TrustRegionOptions, step_finish
from .trust_region_diagnostics import (
    IterationInfo,
    IterationRecord,
    ResumeState,
    RunReason,
    RunSummary,
    build_summary,
    display_iteration,
    format_summary,
)
from .krylov_solvers import subproblem_stats

__all__ = ["PointOperators", "trust_region_newton", "update_radius", "check_resume_matches"]


def update_radius(radius: float, rho: float, step_norm: float, radius_max: float) -> float:
    """Standard radius update, measuring the step in the M-norm."""
    if rho < 0.25:
        return 0.25 * step_norm
    if rho > 0.75 and step_norm >= 0.99 * radius:
        return min(2.0 * radius, radius_max)
    return radius


def check_resume_matches(state: ResumeState, f_x0: float, rtol: float = 1e-10) -> None:
    """Raise if the ResumeState was saved at a different point than x0.

    f(x0) is evaluated at startup anyway, so comparing it with the f stored in
    the state costs nothing. It catches an x and a state (or gradient) saved at
    different accepted iterates; it cannot tell apart states saved before and
    after a rejected step, since x and f do not change there.
    """
    check_resumed_f(state.f, f_x0, rtol)


def trust_region_newton(
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
        radius: float = 1.0, radius_max: float = 1e4,
        radius_min: float = 1e-14, eta: float = 1e-4,
        gtol: float = 0.0, rtol: float = 1e-8,
        gnorm0: Optional[float] = None, forcing_max: float = 0.5,
        stall_tol: float = 1e-3, max_iter: int = 500, max_krylov: int = 50,
        display: int = 0, print_fn: Callable[[str], None] = print,
        callback: Optional[Callable[[PointT, IterationInfo[VectorT, CovectorT]], Optional[bool]]] = None,
        resume: Optional[ResumeState] = None,
        g0: Optional[CovectorT] = None,
        backend: Optional[Backend] = None,
        krylov_capacity: Optional[int] = None,
        ) -> tuple[PointT, RunSummary]:
    """Inexact Newton trust-region method with an ellipsoidal trust region.

    The problem pieces and operators are described in the module docstring.
    All operators are evaluated at the current iterate, and the radius is
    measured in the M(x)-norm of that iterate.

    Without space, x0 and the vectors are numpy arrays, the metric is M = I
    unless metric_solve is given, and retract is x + s unless given. With a
    space, metric_solve and retract must be given (they are separate
    structures, see the module docstring). Objects are used as they are, not
    copied or converted. On a manifold, hessvec is the Hessian of the
    pullback s -> f(retract(x, s)), which for a second-order retraction is
    the Riemannian Hessian. If the representation of vectors and covectors
    at x is redundant (a subspace or quotient of the stored components),
    pass project_covector (see the module docstring).

    The initial radius is an absolute length in the M-norm (default 1.0);
    scale it to the problem when a sensible first step size is known, since
    each factor of 4 too large costs a rejected step.

    Subproblem solver (krylov): "auto" uses truncated CG when P = M and
    saddle_free is off, and Lanczos otherwise; "lanczos" forces Lanczos; "cg"
    forces CG and requires P = M and no saddle_free; "flexible" uses
    solve_tr_subproblem_flexible and requires a separate P (any map from
    covectors to vectors) and metric_apply. At most max_krylov steps per
    solve. With saddle_free, the model Hessian is |H|_{M}, approximated
    through the M-Ritz pairs of the Krylov space, so saddle points repel the
    iterates instead of attracting them.

    The Krylov solve stops when the (surrogate) residual satisfies
    ||R||_{M^{-1}} <= forcing * ||g||_{M^{-1}}, with
    forcing = forcing_term(...) computed from the convergence norm. Lanczos
    also stops when the trust region is active and the model stops changing
    (stall_tol); CG stops when its next iterate would leave the trust region
    or it meets negative curvature, and then steps to the boundary; the
    flexible solver stops on the residual (or a breakdown) only.

    Convergence: stops when ||g|| <= max(gtol, rtol * gnorm0); with the
    default gtol = 0, when ||g|| <= rtol ||g(x0)||. The norm is
    ||g||_{M(x)^{-1}} ("metric", the default; the Euclidean norm when M = I)
    or ||g||_{P(x)^{-1}}
    ("preconditioner"; the same unless precond_solve is given, and then one
    more P solve per new iterate). The recorded and displayed ||g|| is this
    norm. gnorm0 defaults to ||g(x0)|| and can be passed explicitly, e.g.
    when restarting, so that rtol and the forcing term still refer to the
    original start. Also stops when the radius falls below radius_min, after
    max_iter iterations, or when the callback returns True.

    Diagnostics: display = 0 prints nothing, 1 prints one row per iteration
    and a summary, 2 also prints each subproblem's Krylov trace. Output goes
    through print_fn (e.g. print only on one MPI rank). callback(x, info), if
    given, is called after every iteration with the current iterate and an
    IterationInfo; returning True stops the run.

    Resuming: resume=state, with a ResumeState saved together with x0 (from
    info.resume in the callback, or summary.resume), restores the radius,
    gnorm0 and the iteration count, and overrides the radius and gnorm0
    arguments (use dataclasses.replace on the state to change them). With the
    same options and deterministic functions the run continues exactly as if
    it had not stopped; max_iter remains the budget for the whole run. The
    state is checked against f(x0) (see check_resume_matches). g0, the
    gradient at x0 (e.g. info.g saved with x), skips the initial gradient
    evaluation; it must belong to x0.

    Backend: backend=None runs on numpy; another Backend compiles the
    iterations (see the module docstring and backend). Each Lanczos or
    flexible solve starts with room for krylov_capacity basis vectors
    (default min(32, max_krylov)) and doubles it, up to max_krylov, whenever
    the basis fills, continuing the solve without recomputing anything
    (summary.capacity_growths counts the doublings). The room is allocated
    in full, and with static shapes the block products and small dense
    problems run over all of it, so restarting every solve at
    krylov_capacity keeps the work and memory within a factor of two of what
    the solve uses. A compiling backend compiles the pieces once per
    capacity, and keeps them for the run.

    Returns (x, summary): the final iterate and a RunSummary, which holds the
    stopping reason, one IterationRecord per iteration, the work done and the
    ResumeState at x. history_array(summary) gives the iterations as a numeric
    array; merge_summaries joins the summaries of a stopped and resumed run.
    """
    check_arguments(hessvec=hessvec, space=space, retract=retract, metric_solve=metric_solve,
                    metric_apply=metric_apply, precond_solve=precond_solve,
                    operators_at=operators_at, point_data=point_data, saddle_free=saddle_free,
                    krylov=krylov, convergence_norm=convergence_norm,
                    krylov_capacity=krylov_capacity)
    bk = numpy_backend if backend is None else backend
    xp = bk.xp
    problem = make_problem(
        fun=fun, grad=grad, hessvec=hessvec, space=space, retract=retract,
        metric_solve=metric_solve, metric_apply=metric_apply, precond_solve=precond_solve,
        operators_at=operators_at, point_data=point_data, project_covector=project_covector,
        saddle_free=saddle_free, krylov=krylov, convergence_norm=convergence_norm,
        forcing_max=forcing_max, stall_tol=stall_tol, max_krylov=max_krylov)
    pieces = step_solve(problem, bk, krylov_capacity)
    finish_piece = compile_function(bk, partial(step_finish, problem,
                                                TrustRegionOptions(eta=eta, radius_max=radius_max), bk))
    growths = 0

    start = time.perf_counter()
    state: OuterState = pieces.start(x0, g0)
    fx = number(state.f)                        # records hold Python floats
    fun_calls, grad_calls = 1, (0 if g0 is not None else 1)
    it = 0
    if resume is not None:
        check_resume_matches(resume, fx)
        radius, gnorm0, it = resume.radius, resume.gnorm0, resume.next_iteration
    gnorm = number(state.gnorm)
    metric_solves, precond_solves = int(np.asarray(state.metric_solves)), int(np.asarray(state.precond_solves))
    if gnorm0 is None:
        gnorm0 = gnorm
    records: list[IterationRecord] = []
    while True:
        if gradient_converged(gnorm, gnorm0, gtol, rtol):
            reason: RunReason = "converged"
            break
        if radius < radius_min:
            reason = "radius"
            break
        if it >= max_iter:
            reason = "max_iter"
            break
        radius_x, gnorm0_x = xp.asarray(radius), xp.asarray(gnorm0)
        carry, grown = pieces.solve(state, radius_x, gnorm0_x)
        growths += grown
        new_state, raw = finish_piece(state, carry, radius_x, gnorm0_x)
        stats = subproblem_stats(raw.subproblem)
        fun_calls += 1
        accepted = bool(np.asarray(raw.accepted))
        new_radius = number(raw.new_radius)
        record = IterationRecord(
            iteration=it, f=fx, gnorm=gnorm,
            gnorm_ratio=gnorm / gnorm0 if gnorm0 > 0.0 else 0.0,
            radius=radius, new_radius=new_radius, forcing=number(raw.forcing),
            predicted=number(raw.predicted), actual=number(raw.actual), rho=number(raw.rho),
            accepted=accepted, elapsed=time.perf_counter() - start, subproblem=stats)
        records.append(record)
        state = new_state
        if accepted:
            fx, gnorm = number(state.f), number(state.gnorm)
            grad_calls += 1
            metric_solves += int(np.asarray(state.metric_solves))
            precond_solves += int(np.asarray(state.precond_solves))
        radius = new_radius
        it += 1
        if display >= 1:
            display_iteration(record, display, print_fn)
        if callback is not None:
            resume_state = ResumeState(radius=radius, gnorm0=gnorm0, next_iteration=it, f=fx)
            info: IterationInfo[VectorT, CovectorT] = IterationInfo(
                record=record, f=fx, g=state.g, step=raw.step, resume=resume_state)
            if callback(state.x, info):
                reason = "callback"
                break

    final_state = ResumeState(radius=radius, gnorm0=gnorm0, next_iteration=it, f=fx)
    summary = build_summary(reason, records, fx, gnorm, gnorm0, fun_calls, grad_calls,
                            metric_solves, precond_solves, time.perf_counter() - start,
                            final_state, capacity_growths=growths)
    if display >= 1:
        print_fn(format_summary(summary))
    x: PointT = state.x
    return x, summary
