"""One outer iteration of trust_region_newton, as pure pieces a backend can
compile (see backend and dev/jit_backends_plan.md).

  CoreProblem    the problem and the options of a run (fixed, closed over)
  OuterState     what is known at the current point: x, f, the gradient,
                 the point data, the gradient norms and the Krylov start
                 vector
  RawIteration   what one iteration reports, as arrays; the driver turns it
                 into an IterationRecord
  init(problem, backend, x0, g0) -> OuterState
  iterate(problem, backend, state, radius, gnorm0) -> (OuterState, RawIteration)

iterate computes the step (the subproblem solve), the trial point, the
reduction ratio and the new radius, and, when the step is accepted, the
gradient, the point data and the gradient norms at the new point (under
backend.cond). Nothing here branches in Python on computed values, and
nothing imports numpy: arrays go through backend.xp. The stopping tests,
records, display, callback and resuming stay in the driver
(trust_region.trust_region_newton).

The Lanczos and flexible solvers are not converted yet: they run inside
iterate with the numpy backend only.
"""

from dataclasses import dataclass
from typing import Any, Callable, Generic, Literal, Optional

from .backend import Backend, backend_state, run_cond
from .spaces import CovectorT, VectorSpace, VectorT
from .trust_region_subproblem import (
    solve_tr_subproblem,
    solve_tr_subproblem_flexible,
    truncated_cg,
)


@dataclass(frozen=True, kw_only=True)
class PointOperators(Generic[VectorT, CovectorT]):
    """The linear operators at one point, frozen: what operators_at(x, data)
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


@dataclass(frozen=True, kw_only=True)
class CoreProblem:
    """The problem pieces and the options of a run (see trust_region_newton),
    with defaults filled in. Fixed during the run: compiled pieces close
    over it, so none of it is traced."""
    fun: Callable[[Any], Any]
    grad: Callable[[Any], Any]
    hessvec: Optional[Callable[[Any, Any], Any]]
    space: VectorSpace[Any, Any, Any, Any]
    numpy_default: bool                      # no space given: M = I is the default
    retract: Callable[[Any, Any], Any]
    metric_solve: Optional[Callable[[Any, Any], Any]]
    metric_apply: Optional[Callable[[Any, Any], Any]]
    precond_solve: Optional[Callable[[Any, Any], Any]]
    operators_at: Optional[Callable[[Any, Any], PointOperators[Any, Any]]]
    point_data: Optional[Callable[[Any], Any]]
    project_covector: Optional[Callable[[Any, Any], Any]]
    saddle_free: bool
    krylov: Literal["auto", "cg", "lanczos", "flexible"]
    convergence_norm: Literal["metric", "preconditioner"]
    eta: float
    radius_max: float
    forcing_max: float
    stall_tol: float
    max_krylov: int


@backend_state
@dataclass(frozen=True, kw_only=True)
class OuterState:
    """What is known at the current point.

      x, f, g          the point, f(x) and the (canonical) gradient
      data             point_data(x), or None
      gnorm            ||g|| in the convergence norm
      metric_gnorm     ||g||_{M^{-1}}
      start            the Krylov start vector computed with the norms
                       (M^{-1} g or P^{-1} g), or None
      metric_solves, precond_solves
                       the solves the norms took at this point
    """
    x: Any
    f: Any
    g: Any
    data: Any
    gnorm: Any
    metric_gnorm: Any
    start: Any
    metric_solves: Any
    precond_solves: Any


@backend_state
@dataclass(frozen=True, kw_only=True)
class RawIteration:
    """What one iteration reports (arrays): the forcing term, the predicted
    and actual reductions, rho, acceptance, the new radius, the subproblem's
    statistics (RawSubproblem; SubproblemStats from the solvers not yet
    converted) and the step tried."""
    forcing: Any
    predicted: Any
    actual: Any
    rho: Any
    accepted: Any
    new_radius: Any
    subproblem: Any
    step: Any


# --- the tests of criteria.py and the radius update, in where form -------------------------


def forcing_term(xp: Any, gnorm: Any, gnorm0: Any, forcing_max: Any) -> Any:
    """criteria.forcing_term: min(forcing_max, sqrt(||g|| / ||g0||))."""
    positive = gnorm0 > 0.0
    ratio = gnorm / xp.where(positive, gnorm0, 1.0)
    return xp.where(positive, xp.minimum(forcing_max, xp.sqrt(ratio)), forcing_max)


def reduction_ratio(xp: Any, f_old: Any, f_trial: Any, predicted: Any) -> Any:
    """criteria.reduction_ratio: actual over predicted reduction, both
    shifted by 10 eps max(1, |f_old|); -inf when the predicted reduction is
    not positive or f_trial is not finite."""
    valid = (predicted > 0.0) & xp.isfinite(f_trial)
    eps = xp.finfo(xp.asarray(f_old).dtype).eps
    delta = 10.0 * eps * xp.maximum(1.0, xp.abs(f_old))
    rho = ((f_old - f_trial) + delta) / xp.where(valid, predicted + delta, 1.0)
    return xp.where(valid, rho, -xp.inf)


def update_radius(xp: Any, radius: Any, rho: Any, step_norm: Any, radius_max: Any) -> Any:
    """The standard radius update, measuring the step in the M-norm: shrink
    to a quarter of the step when rho < 1/4, double (up to radius_max) when
    rho > 3/4 and the step reached the boundary."""
    expand = (rho > 0.75) & (step_norm >= 0.99 * radius)
    return xp.where(rho < 0.25, 0.25 * step_norm,
                    xp.where(expand, xp.minimum(2.0 * radius, radius_max), radius))


# --- the pieces ---------------------------------------------------------------------------


def _identity(w: Any) -> Any:
    """M = I for numpy arrays: the same array, as a vector or covector."""
    return w


def operators(problem: CoreProblem, x: Any, data: Any) -> PointOperators[Any, Any]:
    """All the operators at x, from operators_at or the keyword arguments,
    with the numpy default M = I filled in and the combination checked."""
    if problem.operators_at is not None:
        given = problem.operators_at(x, data)
    else:
        hv, ms, ma, ps = problem.hessvec, problem.metric_solve, problem.metric_apply, problem.precond_solve
        assert hv is not None
        given = PointOperators(
            hessvec=lambda v: hv(x, v),
            metric_solve=None if ms is None else (lambda w: ms(x, w)),
            metric_apply=None if ma is None else (lambda v: ma(x, v)),
            precond_solve=None if ps is None else (lambda w: ps(x, w)))
    Ms, Ma = given.metric_solve, given.metric_apply
    if Ms is None:
        if Ma is not None:
            raise ValueError("metric_apply needs metric_solve (its inverse)")
        if not problem.numpy_default:
            raise ValueError("with a space, give metric_solve")
        Ms, Ma = _identity, _identity
    if given.precond_solve is not None and Ma is None:
        raise ValueError("a separate preconditioner needs metric_apply (M)")
    if problem.krylov == "cg" and given.precond_solve is not None:
        raise ValueError("krylov='cg' needs P = M (no precond_solve)")
    if problem.krylov == "flexible" and given.precond_solve is None:
        raise ValueError("krylov='flexible' needs a separate precond_solve, and metric_apply")
    return PointOperators(hessvec=given.hessvec, metric_solve=Ms,
                          precond_solve=given.precond_solve, metric_apply=Ma)


def canonical(problem: CoreProblem, x: Any, w: Any) -> Any:
    return w if problem.project_covector is None else problem.project_covector(x, w)


def gradient_norms(problem: CoreProblem, xp: Any, g: Any, ops: PointOperators[Any, Any]
                   ) -> tuple[Any, Any, Any, int, int]:
    """(convergence norm, ||g||_{M^{-1}}, start vector, M^{-1} solves,
    P^{-1} solves). The start vector is what the subproblem solver starts
    from, when it was computed anyway: M^{-1} g for CG and the flexible
    solver, P^{-1} g for Lanczos (M^{-1} g when P = M)."""
    sp = problem.space
    assert ops.metric_solve is not None
    Mg = ops.metric_solve(g)
    m_norm = xp.sqrt(xp.maximum(sp.pair(g, Mg), 0.0))
    preconditioner_norm = problem.convergence_norm == "preconditioner"
    if ops.precond_solve is None or problem.krylov == "flexible":
        if preconditioner_norm and ops.precond_solve is not None:
            Pg = ops.precond_solve(g)
            return xp.sqrt(xp.maximum(sp.pair(g, Pg), 0.0)), m_norm, Mg, 1, 1
        return m_norm, m_norm, Mg, 1, 0
    if preconditioner_norm:
        Pg = ops.precond_solve(g)
        return xp.sqrt(xp.maximum(sp.pair(g, Pg), 0.0)), m_norm, Pg, 1, 1
    return m_norm, m_norm, None, 1, 0


def state_at(problem: CoreProblem, backend: Backend, x: Any, f: Any, g: Any) -> OuterState:
    """The OuterState at x, given f(x) and the canonical gradient g."""
    xp = backend.xp
    data = None if problem.point_data is None else problem.point_data(x)
    gnorm, metric_gnorm, start, metric_solves, precond_solves = gradient_norms(
        problem, xp, g, operators(problem, x, data))
    return OuterState(x=x, f=f, g=g, data=data, gnorm=gnorm, metric_gnorm=metric_gnorm,
                      start=start, metric_solves=xp.asarray(metric_solves),
                      precond_solves=xp.asarray(precond_solves))


def init(problem: CoreProblem, backend: Backend, x0: Any, g0: Any) -> OuterState:
    """The state at x0; g0, if not None, is the gradient at x0."""
    g = problem.grad(x0) if g0 is None else g0
    return state_at(problem, backend, x0, problem.fun(x0), canonical(problem, x0, g))


def uses_cg(problem: CoreProblem, ops: PointOperators[Any, Any]) -> bool:
    return problem.krylov == "cg" or (problem.krylov == "auto" and ops.precond_solve is None
                                      and not problem.saddle_free)


def iterate(problem: CoreProblem, backend: Backend, state: OuterState, radius: Any, gnorm0: Any
            ) -> tuple[OuterState, RawIteration]:
    """One iteration from state with the given radius: the step, the trial
    point, rho, the new radius, and the state at the next point (the trial
    point if accepted, else the same)."""
    xp = backend.xp
    x = state.x
    forcing = forcing_term(xp, state.gnorm, gnorm0, problem.forcing_max)
    resid_tol = forcing * state.metric_gnorm
    ops = operators(problem, x, state.data)
    pc = problem.project_covector
    project = None if pc is None else (lambda w: pc(x, w))
    assert ops.metric_solve is not None
    sub: Any                    # RawSubproblem; SubproblemStats from the solvers not yet converted
    if uses_cg(problem, ops):
        s, sub = truncated_cg(backend, problem.space, state.g, ops.hessvec, ops.metric_solve,
                              radius, resid_tol, problem.max_krylov, project_covector=project,
                              start_vector=state.start)
    elif backend.static_shapes:
        raise NotImplementedError("the Lanczos and flexible solvers run with the numpy backend "
                                  "only, for now (dev/jit_backends_plan.md, phase 6)")
    elif problem.krylov == "flexible":
        assert ops.precond_solve is not None and ops.metric_apply is not None
        # Not converted yet: Python numbers in, SubproblemStats out.
        s, sub = solve_tr_subproblem_flexible(
            problem.space, state.g, ops.hessvec, ops.metric_solve, ops.metric_apply,
            ops.precond_solve, float(radius), float(resid_tol), problem.max_krylov,
            saddle_free=problem.saddle_free, project_covector=project, start_vector=state.start)
    else:
        s, sub = solve_tr_subproblem(
            problem.space, state.g, ops.hessvec, ops.metric_solve, float(radius),
            float(resid_tol), problem.max_krylov, precond_solve=ops.precond_solve,
            metric_apply=ops.metric_apply, stall_tol=problem.stall_tol,
            saddle_free=problem.saddle_free, project_covector=project, start_vector=state.start)

    x_trial = problem.retract(x, s)
    f_trial = problem.fun(x_trial)
    predicted = -sub.model_value
    rho = reduction_ratio(xp, state.f, f_trial, predicted)
    accepted = rho > problem.eta
    new_radius = update_radius(xp, radius, rho, sub.step_norm, problem.radius_max)

    def move(_: Any) -> OuterState:
        g = canonical(problem, x_trial, problem.grad(x_trial))
        return state_at(problem, backend, x_trial, f_trial, g)

    def stay(_: Any) -> OuterState:
        return state

    new_state = run_cond(backend, accepted, move, stay, None)
    return new_state, RawIteration(forcing=forcing, predicted=predicted, actual=state.f - f_trial,
                                   rho=rho, accepted=accepted, new_radius=new_radius,
                                   subproblem=sub, step=s)
