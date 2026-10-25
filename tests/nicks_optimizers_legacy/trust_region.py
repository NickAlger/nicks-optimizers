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
  operators_at(x) -> PointOperators    all the operators at x, frozen
                                       (hessvec, metric_solve, and optionally
                                       precond_solve and metric_apply).
                                       Called once per new iterate, so
                                       expensive setup (assembling a Hessian,
                                       building a multigrid hierarchy)
                                       happens once per point and can be
                                       shared. Pass hessvec=None and none of
                                       metric_solve, metric_apply,
                                       precond_solve with it.

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
"""

import time
from dataclasses import dataclass
from typing import Any, Callable, Generic, Literal, Optional, cast

import numpy as np

from .criteria import forcing_term, gradient_converged, reduction_ratio
from .spaces import (
    CovectorBlockT,
    CovectorT,
    PointT,
    VectorBlockT,
    VectorSpace,
    VectorT,
    numpy_space,
)
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
from .trust_region_subproblem import (
    solve_tr_subproblem,
    solve_tr_subproblem_cg,
    solve_tr_subproblem_flexible,
)


@dataclass(frozen=True, kw_only=True)
class PointOperators(Generic[VectorT, CovectorT]):
    """The linear operators at one point, frozen: what operators_at(x)
    returns, everything the method uses at that point.

      hessvec(v) -> Covector         H v
      metric_solve(w) -> Vector      M^{-1} w (None: M = I, for numpy arrays
                                     without a space only)
      precond_solve(w) -> Vector     P^{-1} w (None: P = M)
      metric_apply(v) -> Covector    M v, the inverse of metric_solve (needed
                                     only with precond_solve)
    """
    hessvec: Callable[[VectorT], CovectorT]
    metric_solve: Optional[Callable[[CovectorT], VectorT]] = None
    precond_solve: Optional[Callable[[CovectorT], VectorT]] = None
    metric_apply: Optional[Callable[[VectorT], CovectorT]] = None


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
    if abs(f_x0 - state.f) > rtol * max(1.0, abs(state.f)):
        raise ValueError(
            f"resume state does not belong to x0: f(x0) = {f_x0!r}, but the state was saved at "
            f"f = {state.f!r}. Save x, the state (and g) from the same callback call.")


def trust_region_newton(
        fun: Callable[[PointT], float],
        grad: Callable[[PointT], CovectorT],
        hessvec: Optional[Callable[[PointT, VectorT], CovectorT]],
        x0: PointT,
        *,
        space: Optional[VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT]] = None,
        retract: Optional[Callable[[PointT, VectorT], PointT]] = None,
        metric_solve: Optional[Callable[[PointT, CovectorT], VectorT]] = None,
        metric_apply: Optional[Callable[[PointT, VectorT], CovectorT]] = None,
        precond_solve: Optional[Callable[[PointT, CovectorT], VectorT]] = None,
        operators_at: Optional[Callable[[PointT], PointOperators[VectorT, CovectorT]]] = None,
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

    Returns (x, summary): the final iterate and a RunSummary, which holds the
    stopping reason, one IterationRecord per iteration, the work done and the
    ResumeState at x. history_array(summary) gives the iterations as a numeric
    array; merge_summaries joins the summaries of a stopped and resumed run.
    """
    if operators_at is not None:
        if any(f is not None for f in (hessvec, metric_solve, metric_apply, precond_solve)):
            raise ValueError("operators_at gives all the operators at a point: pass hessvec=None "
                             "and no metric_solve, metric_apply or precond_solve")
    elif hessvec is None:
        raise ValueError("pass hessvec, or operators_at")
    elif metric_apply is not None and metric_solve is None:
        raise ValueError("metric_apply needs metric_solve (its inverse)")
    numpy_default = space is None
    if not numpy_default and retract is None:
        raise ValueError("with a space, pass retract (flat_retraction(space) gives x + s)")
    if not numpy_default and operators_at is None and metric_solve is None:
        raise ValueError("with a space, pass metric_solve (euclidean_metric_solve gives M = I)")
    if krylov not in ("auto", "cg", "lanczos", "flexible"):
        raise ValueError(f"krylov must be 'auto', 'cg', 'lanczos' or 'flexible', not {krylov!r}")
    if krylov == "flexible" and operators_at is None and (precond_solve is None or metric_apply is None):
        raise ValueError("krylov='flexible' needs a separate precond_solve, and metric_apply")
    if krylov == "cg" and (saddle_free or precond_solve is not None):
        raise ValueError("krylov='cg' needs P = M (no precond_solve) and saddle_free=False")
    if convergence_norm not in ("metric", "preconditioner"):
        raise ValueError(f"convergence_norm must be 'metric' or 'preconditioner', "
                         f"not {convergence_norm!r}")

    sp = space if space is not None else cast(
        VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT], numpy_space)

    def identity_solve(w: CovectorT) -> VectorT:
        """M = I for numpy arrays: the same array, as a vector."""
        return cast(VectorT, w)

    def identity_apply(v: VectorT) -> CovectorT:
        return cast(CovectorT, v)

    def operators(x: PointT) -> PointOperators[VectorT, CovectorT]:
        """All the operators at x, from operators_at or the keyword arguments,
        with the numpy default M = I filled in and the combination checked."""
        if operators_at is not None:
            given = operators_at(x)
        else:
            assert hessvec is not None
            hv, ms, ma, ps = hessvec, metric_solve, metric_apply, precond_solve
            given = PointOperators(
                hessvec=lambda v: hv(x, v),
                metric_solve=None if ms is None else (lambda w: ms(x, w)),
                metric_apply=None if ma is None else (lambda v: ma(x, v)),
                precond_solve=None if ps is None else (lambda w: ps(x, w)))
        Ms, Ma = given.metric_solve, given.metric_apply
        if Ms is None:
            if Ma is not None:
                raise ValueError("metric_apply needs metric_solve (its inverse)")
            if not numpy_default:
                raise ValueError("with a space, give metric_solve")
            Ms, Ma = identity_solve, identity_apply
        if given.precond_solve is not None and Ma is None:
            raise ValueError("a separate preconditioner needs metric_apply (M)")
        if krylov == "cg" and given.precond_solve is not None:
            raise ValueError("krylov='cg' needs P = M (no precond_solve)")
        if krylov == "flexible" and given.precond_solve is None:
            raise ValueError("krylov='flexible' needs a separate precond_solve, and metric_apply")
        return PointOperators(hessvec=given.hessvec, metric_solve=Ms,
                              precond_solve=given.precond_solve, metric_apply=Ma)

    def gradient_norms(g: CovectorT, ops: PointOperators[VectorT, CovectorT]
                       ) -> tuple[float, float, Optional[VectorT], int, int]:
        """(convergence norm, ||g||_{M^{-1}}, start vector, M^{-1} solves,
        P^{-1} solves). The start vector is what the subproblem solver starts
        from, when it was computed anyway: M^{-1} g for CG and the flexible
        solver, P^{-1} g for Lanczos (M^{-1} g when P = M)."""
        assert ops.metric_solve is not None
        Mg = ops.metric_solve(g)
        m_norm = float(np.sqrt(max(sp.pair(g, Mg), 0.0)))
        if ops.precond_solve is None or krylov == "flexible":
            if convergence_norm == "preconditioner" and ops.precond_solve is not None:
                Pg = ops.precond_solve(g)
                return float(np.sqrt(max(sp.pair(g, Pg), 0.0))), m_norm, Mg, 1, 1
            return m_norm, m_norm, Mg, 1, 0
        if convergence_norm == "preconditioner":
            Pg = ops.precond_solve(g)
            return float(np.sqrt(max(sp.pair(g, Pg), 0.0))), m_norm, Pg, 1, 1
        return m_norm, m_norm, None, 1, 0

    start = time.perf_counter()
    retraction: Callable[[PointT, VectorT], PointT] = (
        retract if retract is not None else (lambda x, s: cast(PointT, cast(Any, x) + s)))
    x = x0
    fx = float(fun(x))                          # records hold Python floats
    fun_calls = 1
    it = 0
    if resume is not None:
        check_resume_matches(resume, fx)
        radius, gnorm0, it = resume.radius, resume.gnorm0, resume.next_iteration

    def canonical(x: PointT, w: CovectorT) -> CovectorT:
        return w if project_covector is None else project_covector(x, w)

    if g0 is not None:
        g, grad_calls = canonical(x, g0), 0
    else:
        g, grad_calls = canonical(x, grad(x)), 1
    ops = operators(x)
    gnorm, metric_gnorm, start_vector, metric_solves, precond_solves = gradient_norms(g, ops)
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
        forcing = forcing_term(gnorm, gnorm0, forcing_max)
        project = None if project_covector is None else (lambda w: project_covector(x, w))
        assert ops.metric_solve is not None
        use_cg = krylov == "cg" or (krylov == "auto" and ops.precond_solve is None
                                    and not saddle_free)
        if use_cg:
            s, stats = solve_tr_subproblem_cg(
                sp, g, ops.hessvec, ops.metric_solve, radius, forcing * metric_gnorm, max_krylov,
                project_covector=project, start_vector=start_vector)
        elif krylov == "flexible":
            assert ops.precond_solve is not None and ops.metric_apply is not None
            s, stats = solve_tr_subproblem_flexible(
                sp, g, ops.hessvec, ops.metric_solve, ops.metric_apply, ops.precond_solve,
                radius, forcing * metric_gnorm, max_krylov, saddle_free=saddle_free,
                project_covector=project, start_vector=start_vector)
        else:
            s, stats = solve_tr_subproblem(
                sp, g, ops.hessvec, ops.metric_solve, radius, forcing * metric_gnorm, max_krylov,
                precond_solve=ops.precond_solve, metric_apply=ops.metric_apply,
                stall_tol=stall_tol, saddle_free=saddle_free, project_covector=project,
                start_vector=start_vector)

        x_trial = retraction(x, s)
        f_trial = float(fun(x_trial))
        fun_calls += 1
        predicted = -stats.model_value
        rho = reduction_ratio(fx, f_trial, predicted)
        accepted = bool(rho > eta)
        new_radius = update_radius(radius, rho, stats.step_norm, radius_max)
        record = IterationRecord(
            iteration=it, f=fx, gnorm=gnorm,
            gnorm_ratio=gnorm / gnorm0 if gnorm0 > 0.0 else 0.0,
            radius=radius, new_radius=new_radius, forcing=forcing,
            predicted=predicted, actual=fx - f_trial, rho=rho, accepted=accepted,
            elapsed=time.perf_counter() - start, subproblem=stats)
        records.append(record)
        if accepted:
            x, fx = x_trial, f_trial
            g = canonical(x, grad(x))
            grad_calls += 1
            ops = operators(x)
            gnorm, metric_gnorm, start_vector, a, b = gradient_norms(g, ops)
            metric_solves, precond_solves = metric_solves + a, precond_solves + b
        radius = new_radius
        it += 1
        if display >= 1:
            display_iteration(record, display, print_fn)
        if callback is not None:
            state = ResumeState(radius=radius, gnorm0=gnorm0, next_iteration=it, f=fx)
            if callback(x, IterationInfo(record=record, f=fx, g=g, step=s, resume=state)):
                reason = "callback"
                break

    final_state = ResumeState(radius=radius, gnorm0=gnorm0, next_iteration=it, f=fx)
    summary = build_summary(reason, records, fx, gnorm, gnorm0, fun_calls, grad_calls,
                            metric_solves, precond_solves, time.perf_counter() - start,
                            final_state)
    if display >= 1:
        print_fn(format_summary(summary))
    return x, summary
