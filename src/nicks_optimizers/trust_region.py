"""Trust-region Newton method with an ellipsoidal trust region, an optional
preconditioner and an optional saddle-free model.

The geometry comes first. A symmetric positive definite metric M(x) defines
the trust region { s : ||s||_M <= radius }, the gradient norm ||g||_{M^{-1}}
of the stopping test, the M-Ritz values of the saddle-free model and the
Krylov residual norms. A preconditioner P(x) is secondary: it only shapes
the Krylov space, and by default P = M.

Problem pieces
--------------
  fun(x) -> float                      objective
  grad(x) -> Covector                  differential df(x)
  hessvec(x, v) -> Covector            H(x) v, the Hessian of the pullback
                                       s -> f(retract(x, s)) at s = 0
  x0                                   starting point

Geometry and preconditioning (keyword arguments):
  metric_solve(x, w) -> Vector         M(x)^{-1} w. Defaults to M = I for
                                       plain numpy arrays (numpy_space, the
                                       default space); any other
                                       representation must pass it.
  precond_solve(x, w) -> Vector        P(x)^{-1} w, if P should differ from M
                                       (e.g. a Hessian preconditioner inside a
                                       fixed metric). Optional.
  metric_apply(x, v) -> Covector       M(x) v; needed only with precond_solve,
                                       and must be consistent with
                                       metric_solve.

With P = M (no precond_solve) only M^{-1} is ever applied: any symmetric
positive definite operator passed as metric_solve, a multigrid cycle for
instance, defines M as its inverse. Without saddle_free the subproblem is then
solved by truncated conjugate gradients (five vectors, two reductions per
step) unless krylov="lanczos" asks for the reorthogonalized Lanczos solver
(exact subspace solutions on the boundary, more robust in floating point).
With a separate P, Lanczos builds the M-Gram matrix with metric_apply and
measures residuals with metric_solve.

Optional per-point setup:
  operators_at(x) -> PointOperators    the operators frozen at x (hessvec,
                                       and optionally metric_solve,
                                       precond_solve, metric_apply). Called
                                       once per new iterate, so expensive
                                       setup (assembling a Hessian, building
                                       a multigrid hierarchy) happens once per
                                       point and can be shared. Fields left
                                       None fall back to the keyword
                                       arguments. Pass hessvec=None with it.

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

Each iteration computes a step with solve_tr_subproblem or
solve_tr_subproblem_cg (see trust_region_subproblem) and records what
happened (see trust_region_diagnostics).
"""

import time
from dataclasses import dataclass
from typing import Callable, Generic, Literal, Optional, cast

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
from .trust_region_subproblem import solve_tr_subproblem, solve_tr_subproblem_cg


@dataclass(frozen=True, kw_only=True)
class PointOperators(Generic[VectorT, CovectorT]):
    """The linear operators at one point, frozen: what operators_at(x)
    returns. Fields left None fall back to the keyword arguments of
    trust_region_newton (and metric_solve to M = I for plain numpy arrays).

      hessvec(v) -> Covector         H v
      metric_solve(w) -> Vector      M^{-1} w
      precond_solve(w) -> Vector     P^{-1} w (None: P = M)
      metric_apply(v) -> Covector    M v (needed only with a separate P)
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
        metric_solve: Optional[Callable[[PointT, CovectorT], VectorT]] = None,
        precond_solve: Optional[Callable[[PointT, CovectorT], VectorT]] = None,
        metric_apply: Optional[Callable[[PointT, VectorT], CovectorT]] = None,
        operators_at: Optional[Callable[[PointT], PointOperators[VectorT, CovectorT]]] = None,
        space: Optional[VectorSpace[PointT, VectorT, CovectorT, VectorBlockT, CovectorBlockT]] = None,
        retract: Optional[Callable[[PointT, VectorT], PointT]] = None,
        project_covector: Optional[Callable[[PointT, CovectorT], CovectorT]] = None,
        saddle_free: bool = False,
        krylov: Literal["auto", "cg", "lanczos"] = "auto",
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

    space says how points, vectors and covectors are represented (default
    numpy_space), and x0 must already be a point in that representation; it
    is not copied or converted. retract defaults to space.translate; a space
    without one (a manifold) needs retract. On a manifold, hessvec is the
    Hessian of the pullback s -> f(retract(x, s)), which for a second-order
    retraction is the Riemannian Hessian. If the representation of vectors
    and covectors at x is redundant (a subspace or quotient of the stored
    components), pass project_covector (see the module docstring).

    The initial radius is an absolute length in the M-norm (default 1.0);
    scale it to the problem when a sensible first step size is known, since
    each factor of 4 too large costs a rejected step.

    Subproblem solver (krylov): "auto" uses truncated CG when P = M and
    saddle_free is off, and Lanczos otherwise; "lanczos" forces Lanczos; "cg"
    forces CG and requires P = M and no saddle_free. At most max_krylov steps
    per solve. With saddle_free, the model Hessian is |H|_{M}, approximated
    through the M-Ritz pairs of the Krylov space, so saddle points repel the
    iterates instead of attracting them.

    The Krylov solve stops when the (surrogate) residual satisfies
    ||R||_{M^{-1}} <= forcing * ||g||_{M^{-1}}, with
    forcing = forcing_term(...) computed from the convergence norm. Lanczos
    also stops when the trust region is active and the model stops changing
    (stall_tol); CG stops when its next iterate would leave the trust region
    or it meets negative curvature, and then steps to the boundary.

    Convergence: stops when ||g|| <= max(gtol, rtol * gnorm0); with the
    default gtol = 0, when ||g|| <= rtol ||g(x0)||. The norm is
    ||g||_{M(x)^{-1}} ("metric", the default; the Euclidean norm for plain
    numpy arrays without metric_solve) or ||g||_{P(x)^{-1}}
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
    if (hessvec is None) == (operators_at is None):
        raise ValueError("pass hessvec, or operators_at (with hessvec=None), but not both")
    euclidean_default = space is None or space is numpy_space
    if metric_solve is None and operators_at is None and not euclidean_default:
        raise ValueError("metric_solve is required with a custom space "
                         "(the default, M = I, is only for plain numpy arrays)")
    if krylov not in ("auto", "cg", "lanczos"):
        raise ValueError(f"krylov must be 'auto', 'cg' or 'lanczos', not {krylov!r}")
    if krylov == "cg" and (saddle_free or precond_solve is not None):
        raise ValueError("krylov='cg' needs P = M (no precond_solve) and saddle_free=False")
    if convergence_norm not in ("metric", "preconditioner"):
        raise ValueError(f"convergence_norm must be 'metric' or 'preconditioner', "
                         f"not {convergence_norm!r}")

    sp = space if space is not None else cast(
        VectorSpace[PointT, VectorT, CovectorT, VectorBlockT, CovectorBlockT], numpy_space)

    def default_metric_solve(w: CovectorT) -> VectorT:
        """M = I for plain numpy arrays: the same array, as a vector."""
        return cast(VectorT, w)

    def operators(x: PointT) -> PointOperators[VectorT, CovectorT]:
        """The operators at x, with fallbacks resolved and checked."""
        given = operators_at(x) if operators_at is not None else None
        if given is not None:
            H = given.hessvec
        else:
            assert hessvec is not None
            hv = hessvec
            H = lambda v: hv(x, v)                                      # noqa: E731
        Ms = given.metric_solve if given is not None else None
        if Ms is None:
            if metric_solve is not None:
                ms = metric_solve
                Ms = lambda w: ms(x, w)                                 # noqa: E731
            elif euclidean_default:
                Ms = default_metric_solve
            else:
                raise ValueError("metric_solve is required with a custom space")
        Ps = given.precond_solve if given is not None else None
        if Ps is None and precond_solve is not None:
            ps = precond_solve
            Ps = lambda w: ps(x, w)                                     # noqa: E731
        Ma = given.metric_apply if given is not None else None
        if Ma is None and metric_apply is not None:
            ma = metric_apply
            Ma = lambda v: ma(x, v)                                     # noqa: E731
        if Ps is not None and Ma is None:
            raise ValueError("a separate preconditioner needs metric_apply (M)")
        if krylov == "cg" and Ps is not None:
            raise ValueError("krylov='cg' needs P = M (no precond_solve)")
        return PointOperators(hessvec=H, metric_solve=Ms, precond_solve=Ps, metric_apply=Ma)

    def gradient_norms(g: CovectorT, ops: PointOperators[VectorT, CovectorT]
                       ) -> tuple[float, float, Optional[VectorT], int, int]:
        """(convergence norm, ||g||_{M^{-1}}, start vector, M^{-1} solves,
        P^{-1} solves). The start vector is P^{-1} g when it was computed
        anyway (M^{-1} g when P = M), for the subproblem to reuse."""
        assert ops.metric_solve is not None
        Mg = ops.metric_solve(g)
        m_norm = float(np.sqrt(max(sp.pair(g, Mg), 0.0)))
        if ops.precond_solve is None:
            return m_norm, m_norm, Mg, 1, 0
        if convergence_norm == "preconditioner":
            Pg = ops.precond_solve(g)
            return float(np.sqrt(max(sp.pair(g, Pg), 0.0))), m_norm, Pg, 1, 1
        return m_norm, m_norm, None, 1, 0

    start = time.perf_counter()
    retraction = retract if retract is not None else sp.translate
    if retraction is None:
        raise ValueError("the space has no translate (e.g. a manifold): pass retract")
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
