"""Trust-region Newton method with a Hessian preconditioner, an ellipsoidal
trust region and an optional saddle-free model.

Problem pieces
--------------
  fun(x) -> float                      objective
  grad(x) -> Covector                  differential df(x)
  hessvec(x, v) -> Covector            H(x) v, the Hessian of the pullback
                                       s -> f(retract(x, s)) at s = 0
  precond_solve(x, w) -> Vector        P(x)^{-1} w, P(x) symmetric positive
                                       definite; only shapes the Krylov space
  ball_inner(x, U, v) -> Coords        the M(x) inner products (u_i, v)_M of
                                       each vector u_i in the block U with v.
                                       The trust region at x is
                                       { s : (s, s)_M <= radius^2 }. For an
                                       ellipsoid with covariance B(x),
                                       (u, v)_M = <B(x)^{-1} v, u>.
  ball_dual_inner(x, Psi, phi) -> Coords  the dual inner products
                                       (psi_i, phi)_{M^{-1}} = <psi_i, B(x) phi>;
                                       used for dual norms (stopping tests).

Optional per-point setup, in place of hessvec and precond_solve (pass None
for both):
  operators_at(x) -> (hessvec_x, precond_solve_x)
                                       the operators frozen at x, v -> H(x) v
                                       and w -> P(x)^{-1} w. Called once per
                                       new iterate, and reused after rejected
                                       steps, so expensive setup (assembling a
                                       Hessian, building a multigrid
                                       hierarchy) happens once per point and
                                       can be shared by both operators.

Optional for representations with redundant components (e.g. gauged
tangent vectors of a low-rank manifold):
  project_covector(x, w) -> Covector   the canonical representative of w,
                                       dropping components that pair to zero
                                       with every vector at x. Applied to
                                       gradients and to every Krylov covector,
                                       where the Lanczos recurrence would
                                       otherwise amplify rounding errors in
                                       the invisible components (see krylov).

H and P^{-1} are the only operators the method applies; the geometry enters
only through the two inner products (see metrics), which must be exact
duals. The radius carries over from one point to the next, so the theory
wants M(x) uniformly equivalent to a fixed norm over the iterates.

Each iteration computes a step with solve_tr_subproblem (see
trust_region_subproblem) and records what happened (see
trust_region_diagnostics).
"""

import time
from typing import Callable, Optional, cast

from .criteria import forcing_term, gradient_converged, reduction_ratio
from .metrics import covector_norm
from .spaces import (
    Coords,
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
from .trust_region_subproblem import solve_tr_subproblem


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
        precond_solve: Optional[Callable[[PointT, CovectorT], VectorT]],
        ball_inner: Callable[[PointT, VectorBlockT, VectorT], Coords],
        ball_dual_inner: Callable[[PointT, CovectorBlockT, CovectorT], Coords],
        x0: PointT,
        space: Optional[VectorSpace[PointT, VectorT, CovectorT, VectorBlockT, CovectorBlockT]] = None,
        retract: Optional[Callable[[PointT, VectorT], PointT]] = None,
        project_covector: Optional[Callable[[PointT, CovectorT], CovectorT]] = None,
        saddle_free: bool = False,
        radius: float = 1.0, radius_max: float = 1e4,
        radius_min: float = 1e-14, eta: float = 1e-4,
        gtol: float = 1e-8, rtol: float = 0.0,
        gnorm0: Optional[float] = None, forcing_max: float = 0.5,
        stall_tol: float = 1e-3, max_iter: int = 500, max_lanczos: int = 50,
        display: int = 0, print_fn: Callable[[str], None] = print,
        callback: Optional[Callable[[PointT, IterationInfo[VectorT, CovectorT]], Optional[bool]]] = None,
        resume: Optional[ResumeState] = None,
        g0: Optional[CovectorT] = None,
        operators_at: Optional[Callable[[PointT], tuple[Callable[[VectorT], CovectorT],
                                                        Callable[[CovectorT], VectorT]]]] = None,
        ) -> tuple[PointT, RunSummary]:
    """Inexact Newton trust-region method with an ellipsoidal trust region.

    The operators and inner products are described in the module docstring.
    All of them are evaluated at the current iterate, and the radius is
    measured in the M(x)-norm of that iterate. There is no default for the
    inner products: they must be exact duals, and a default dual that ignored
    ball_inner would silently break that. For numpy_space and M = I, pass
    euclidean_inner and euclidean_dual_inner.

    Operators: either hessvec(x, v) and precond_solve(x, w), or
    operators_at(x) returning both frozen at x (then pass None for
    hessvec and precond_solve). operators_at is called once per iterate at
    which a subproblem is solved: setup shared by H(x) and P(x) happens once
    per point, and rejected steps reuse it.

    space says how points, vectors and covectors are represented (default
    numpy_space), and x0 must already be a point in that representation; it
    is not copied or converted. retract defaults to space.translate; a space
    without one (a manifold) needs retract. On a manifold, hessvec is the
    Hessian of the pullback s -> f(retract(x, s)), which for a second-order
    retraction is the Riemannian Hessian. If the representation of vectors
    and covectors at x is redundant (a subspace or quotient of the stored
    components), pass project_covector (see the module docstring).

    Gradient norms are dual norms, ||g|| = ||g||_{M(x)^{-1}}. With saddle_free, the model
    Hessian is |H|_{M}, approximated through the M-Ritz pairs of the Krylov
    space, so saddle points repel the iterates instead of attracting them.

    The Krylov solve stops when the (surrogate) residual satisfies
    ||R|| <= forcing * ||g||, with forcing = forcing_term(...), or, while the
    trust region is active, when the model stops changing (stall_tol).

    Convergence: stops when ||g|| <= max(gtol, rtol * gnorm0). gnorm0 defaults
    to ||g(x0)|| and can be passed explicitly, e.g. when restarting, so that
    rtol and the forcing term still refer to the original start. Also stops
    when the radius falls below radius_min, after max_iter iterations, or
    when the callback returns True.

    Diagnostics: display = 0 prints nothing, 1 prints one row per iteration
    and a summary, 2 also prints each subproblem's Lanczos trace. Output goes
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
    if operators_at is None and (hessvec is None or precond_solve is None):
        raise ValueError("pass hessvec and precond_solve, or operators_at")
    if operators_at is not None and (hessvec is not None or precond_solve is not None):
        raise ValueError("pass either hessvec and precond_solve, or operators_at, not both")

    def frozen_operators(x: PointT) -> tuple[Callable[[VectorT], CovectorT],
                                             Callable[[CovectorT], VectorT]]:
        if operators_at is not None:
            return operators_at(x)
        assert hessvec is not None and precond_solve is not None
        return (lambda v: hessvec(x, v)), (lambda w: precond_solve(x, w))

    start = time.perf_counter()
    sp = space if space is not None else cast(
        VectorSpace[PointT, VectorT, CovectorT, VectorBlockT, CovectorBlockT], numpy_space)
    retraction = retract if retract is not None else sp.translate
    if retraction is None:
        raise ValueError("the space has no translate (e.g. a manifold): pass retract")
    x = x0
    fx = fun(x)
    fun_calls, outer_dual_calls = 1, 1
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
    gnorm = covector_norm(sp, lambda Psi, phi: ball_dual_inner(x, Psi, phi), g)
    if gnorm0 is None:
        gnorm0 = gnorm
    records: list[IterationRecord] = []
    operators: Optional[tuple[Callable[[VectorT], CovectorT], Callable[[CovectorT], VectorT]]] = None
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
        if operators is None:                   # first iteration at this point
            operators = frozen_operators(x)
        s, stats = solve_tr_subproblem(
            sp, g, operators[0], operators[1],
            lambda U, v: ball_inner(x, U, v),
            lambda Psi, phi: ball_dual_inner(x, Psi, phi),
            radius, forcing * gnorm, max_lanczos,
            g_dual_norm=gnorm, stall_tol=stall_tol, saddle_free=saddle_free,
            project_covector=None if project_covector is None else (
                lambda w: project_covector(x, w)))

        x_trial = retraction(x, s)
        f_trial = fun(x_trial)
        fun_calls += 1
        predicted = -stats.model_value
        rho = reduction_ratio(fx, f_trial, predicted)
        accepted = rho > eta
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
            operators = None
            g = canonical(x, grad(x))
            grad_calls += 1
            gnorm = covector_norm(sp, lambda Psi, phi: ball_dual_inner(x, Psi, phi), g)
            outer_dual_calls += 1
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
                            outer_dual_calls, time.perf_counter() - start, final_state)
    if display >= 1:
        print_fn(format_summary(summary))
    return x, summary
