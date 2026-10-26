"""The parts of an inexact Newton iteration that the optimizers share, as
pure pieces a backend can compile (see backend and
dev/jit_backends_plan.md): what is known at a point, and the Krylov solve
of the step.

  PointOperators the linear operators at one point (what operators_at
                 returns)
  CoreProblem    the problem and the options of the step computation
                 (fixed, closed over)
  OuterState     what is known at the current point: x, f, the gradient,
                 the point data, the gradient norms and the Krylov start
                 vector
  init(problem, backend, x0, g0) -> OuterState
  state_at(problem, backend, x, f, g) -> OuterState

and the solve of the step, which a driver calls in turn:

  step_start(problem, backend, state, radius, gnorm0, capacity=c) -> carry
                 start the Krylov solve and run it until it stops or its
                 basis (room for c vectors) is full
  step_grow(problem, backend, state, carry, capacity=c2) -> carry
  step_continue(problem, backend, state, carry) -> carry
                 when the basis was full: copy the solve into a larger basis
                 and continue where it stopped (nothing is recomputed)

The globalization finishes the iteration with its own piece (the
trust-region update in trust_region_core). Each piece rebuilds the
operators from (x, point data), so no function crosses the boundary of a
compiled piece. Nothing here branches in Python on computed values, and
nothing imports numpy: arrays go through backend.xp.
"""

from dataclasses import dataclass
from typing import Any, Callable, Generic, Literal, Optional

from .backend import Backend, backend_state
from .krylov_solvers import (
    KrylovSolver,
    SubproblemOperators,
    cg_solver,
    flexible_solver,
    lanczos_solver,
)
from .spaces import CovectorT, VectorSpace, VectorT


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
    """The problem pieces and the options of the step computation shared by
    the optimizers (see trust_region_newton), with defaults filled in. Fixed
    during the run: compiled pieces close over it, so none of it is traced.
    The options of the globalization (the radius update, the line search)
    are passed to its own pieces."""
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


def forcing_term(xp: Any, gnorm: Any, gnorm0: Any, forcing_max: Any) -> Any:
    """criteria.forcing_term: min(forcing_max, sqrt(||g|| / ||g0||))."""
    positive = gnorm0 > 0.0
    ratio = gnorm / xp.where(positive, gnorm0, 1.0)
    return xp.where(positive, xp.minimum(forcing_max, xp.sqrt(ratio)), forcing_max)


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


def subproblem_at(problem: CoreProblem, state: OuterState
                  ) -> tuple[KrylovSolver, SubproblemOperators[Any, Any]]:
    """The subproblem solver for this run (chosen from the configuration and
    whether a separate preconditioner is given, so fixed when a piece is
    traced) and its operators at the current point."""
    ops = operators(problem, state.x, state.data)
    pc, x = problem.project_covector, state.x
    assert ops.metric_solve is not None
    sub_ops = SubproblemOperators(
        hessvec=ops.hessvec, metric_solve=ops.metric_solve, precond_solve=ops.precond_solve,
        metric_apply=ops.metric_apply,
        project_covector=None if pc is None else (lambda w: pc(x, w)))
    if uses_cg(problem, ops):
        return cg_solver(problem.max_krylov), sub_ops
    if problem.krylov == "flexible":
        return flexible_solver(problem.max_krylov, saddle_free=problem.saddle_free), sub_ops
    return lanczos_solver(problem.max_krylov, saddle_free=problem.saddle_free,
                          stall_tol=problem.stall_tol), sub_ops


def step_start(problem: CoreProblem, backend: Backend, state: OuterState, radius: Any,
               gnorm0: Any, *, capacity: int) -> Any:
    """Start the subproblem solve at state with the given radius, and run it
    until it stops or its basis (of the given capacity)
    is full. Returns the solver's carry."""
    solver, ops = subproblem_at(problem, state)
    forcing = forcing_term(backend.xp, state.gnorm, gnorm0, problem.forcing_max)
    carry = solver.begin(backend, problem.space, ops, state.g, radius, forcing * state.metric_gnorm,
                         state.start, capacity)
    return solver.run(backend, problem.space, ops, state.g, carry)


def step_continue(problem: CoreProblem, backend: Backend, state: OuterState, carry: Any) -> Any:
    """Continue a solve after its basis has grown."""
    solver, ops = subproblem_at(problem, state)
    return solver.run(backend, problem.space, ops, state.g, carry)


def step_grow(problem: CoreProblem, backend: Backend, state: OuterState, carry: Any, *,
              capacity: int) -> Any:
    """The same solve with room for capacity basis vectors (copies only)."""
    solver, _ = subproblem_at(problem, state)
    return solver.grow(backend, problem.space, carry, capacity)
