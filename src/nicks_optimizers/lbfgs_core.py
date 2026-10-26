"""An iteration of lbfgs as pure pieces a backend can compile (see
dev/lbfgs_plan.md).

  LbfgsProblem    the problem pieces and the options, with defaults filled
                  in (fixed during a run: compiled pieces close over it)
  LbfgsState      what is known at the current point: x, f, the canonical
                  gradient g, the point data, M^{-1} g and the gradient
                  norms, the memory, gamma (the scaling of the base
                  gamma M^{-1}) and the stamp of the next stored pair
  init(problem, backend, x0, g0) -> LbfgsState
  lbfgs_step(problem, backend, state) -> (LbfgsState, RawLbfgsIteration)
                  the direction d = -H g (the two-loop recursion of
                  lbfgs_memory), its descent and angle tests, the line
                  search along retract(x, alpha d), and, when it succeeds,
                  the state at the new point: the gradient and its norms, the
                  memory transported there, the new pair (s, y) tested and
                  stored, gamma refitted
  restart(problem, backend, state) -> LbfgsState
                  the same state with the memory cleared

Everything geometric is measured with the metric M, given by both of its
operators (M^{-1}, the Riesz map, and M): the base gamma M^{-1}, the scaling
fits, the gradient norm, ||d||_M and the angle between -g and d. Nothing here
branches in Python on computed values, and nothing imports numpy.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

from .backend import Backend, backend_state, run_cond
from .lbfgs_memory import (
    Memory,
    memory_clear,
    memory_count,
    memory_empty,
    memory_layout,
    memory_recompute,
    memory_store,
    memory_transport,
    two_loop_finish,
    two_loop_project,
)
from .line_search import ARMIJO, backtracking
from .newton_core import PointOperators
from .spaces import VectorSpace
from .transport import VectorTransport, identity_transport

Scaling = Literal["y", "s", "geometric", "none"]
Pairings = Literal["stored", "recomputed"]
CurvatureTest = Literal["relative", "positive", "dennis-schnabel", "cautious"]
Direction = Literal["quasi-newton", "steepest", "fallback"]
Update = Literal["-", "stored", "skipped"]

QUASI_NEWTON, STEEPEST, FALLBACK = range(3)
DIRECTIONS: tuple[Direction, ...] = ("quasi-newton", "steepest", "fallback")
NO_UPDATE, STORED, SKIPPED = range(3)
UPDATES: tuple[Update, ...] = ("-", "stored", "skipped")
SCALINGS: tuple[Scaling, ...] = ("y", "s", "geometric", "none")
PAIRINGS: tuple[Pairings, ...] = ("stored", "recomputed")
CURVATURE_TESTS: tuple[CurvatureTest, ...] = ("relative", "positive", "dennis-schnabel", "cautious")
# The default tolerance of each curvature test (see LbfgsProblem.curvature_tol).
CURVATURE_TOL: dict[str, float] = {"relative": 1e-8, "positive": 0.0,
                                   "dennis-schnabel": 2.0 ** -26, "cautious": 1e-6}


@dataclass(frozen=True, kw_only=True)
class LbfgsProblem:
    """The problem pieces and options of lbfgs (see its docstring).

      memory           m, the number of pairs kept
      scaling          the fit of gamma after each stored pair: "y"
                       (<y, s> / <y, M^{-1} y>), "s" (<M s, s> / <y, s>),
                       "geometric" (||s||_M / ||y||_{M^{-1}}) or "none"
      gamma0           gamma before the first stored pair (the first step is
                       -gamma0 M^{-1} g), unless initial_step is given
      initial_step     if given, gamma0 = initial_step / ||g0||_{M^{-1}}: a
                       first step of that M-length
      pairings         "stored" (the pairings computed when pairs were
                       stored, kept under transport) or "recomputed" (those of
                       the transported pairs; pairs whose curvature falls to
                       their floor are dropped)
      curvature_test   when a new pair is stored: <y, s> > floor with floor
                       curvature_tol alpha |<g, d>| ("relative"), 0
                       ("positive"), curvature_tol ||s||_M ||y||_{M^{-1}}
                       ("dennis-schnabel") or curvature_tol ||g||_{M^{-1}}
                       ||s||_M^2 ("cautious", Li and Fukushima, with g at the
                       previous point)
      angle_tol        a direction is used only if <-g, d> >= angle_tol
                       ||g||_{M^{-1}} ||d||_M (else -gamma M^{-1} g, and the
                       memory is cleared)
      c1, max_backtracks, step_max   the line search (line_search.backtracking)
    """
    fun: Callable[[Any], Any]
    grad: Callable[[Any], Any]
    space: VectorSpace[Any, Any, Any, Any]
    numpy_default: bool
    retract: Callable[[Any, Any], Any]
    transport: VectorTransport[Any, Any, Any, Any, Any]
    metric_solve: Optional[Callable[[Any, Any], Any]]
    metric_apply: Optional[Callable[[Any, Any], Any]]
    operators_at: Optional[Callable[[Any, Any], PointOperators[Any, Any]]]
    point_data: Optional[Callable[[Any], Any]]
    project_covector: Optional[Callable[[Any, Any], Any]]
    memory: int
    scaling: Scaling
    gamma0: float
    initial_step: Optional[float]
    pairings: Pairings
    curvature_test: CurvatureTest
    curvature_tol: float
    angle_tol: float
    c1: float
    max_backtracks: int
    step_max: Optional[float]


@backend_state
@dataclass(frozen=True, kw_only=True)
class LbfgsState:
    """What is known at the current point.

      x, f, g          the point, f(x) and the canonical gradient
      data             point_data(x), or None
      Mg               M^{-1} g (computed for the norm; the base applied to g)
      gnorm            ||g|| in the convergence norm
      metric_gnorm     ||g||_{M^{-1}}
      memory           the stored pairs, at x
      gamma            the scaling of the base gamma M^{-1}
      stamp            the stamp the next stored pair gets
    """
    x: Any
    f: Any
    g: Any
    data: Any
    Mg: Any
    gnorm: Any
    metric_gnorm: Any
    memory: Memory[Any, Any]
    gamma: Any
    stamp: Any


@backend_state
@dataclass(frozen=True, kw_only=True)
class RawLbfgsIteration:
    """What one iteration reports (arrays).

      direction          QUASI_NEWTON, STEEPEST (empty memory) or FALLBACK
                         (the memory's direction failed the descent or angle
                         test: -gamma M^{-1} g instead, memory cleared)
      pairs, gamma       the pairs and the scaling the direction used
      step_norm, cosine  ||d||_M and <-g, d> / (||g||_{M^{-1}} ||d||_M)
      slope              <g, d>
      alpha, trials, status    the line search (line_search statuses)
      actual             f(x) - f at the last trial
      ys, curvature      <y, s> at the new point, and <y, s> / (alpha |<g, d>|)
      update             NO_UPDATE (search failed), STORED or SKIPPED
      dropped            pairs dropped by recomputed pairings
      transport_defect   <Tb g, T s> / <g, s> - 1 (0 for the identity)
      step               alpha d, at x
      trace_alpha, trace_f      the trials
      metric_solves, metric_applies, vector_transports, covector_transports,
      block_transports   applications of the operators and calls of the
                         transport maps in this iteration
    """
    direction: Any
    pairs: Any
    gamma: Any
    step_norm: Any
    cosine: Any
    slope: Any
    alpha: Any
    trials: Any
    status: Any
    actual: Any
    ys: Any
    curvature: Any
    update: Any
    dropped: Any
    transport_defect: Any
    step: Any
    trace_alpha: Any
    trace_f: Any
    metric_solves: Any
    metric_applies: Any
    vector_transports: Any
    covector_transports: Any
    block_transports: Any


def _identity(w: Any) -> Any:
    return w


def point_operators(problem: LbfgsProblem, x: Any, data: Any) -> PointOperators[Any, Any]:
    """M^{-1} and M at x, from operators_at or the keyword operators (the
    identity for numpy arrays without a space when neither is given)."""
    if problem.operators_at is not None:
        given = problem.operators_at(x, data)
        solve, apply = given.metric_solve, given.metric_apply
    else:
        ms, ma = problem.metric_solve, problem.metric_apply
        solve = None if ms is None else (lambda w: ms(x, w))
        apply = None if ma is None else (lambda v: ma(x, v))
    if solve is None and apply is None and problem.numpy_default:
        solve, apply = _identity, _identity
    if solve is None or apply is None:
        raise ValueError("lbfgs needs both metric_solve (M^{-1}) and metric_apply (M)")
    return PointOperators(metric_solve=solve, metric_apply=apply)


def canonical(problem: LbfgsProblem, x: Any, w: Any) -> Any:
    return w if problem.project_covector is None else problem.project_covector(x, w)


def _norm(xp: Any, value: Any) -> Any:
    return xp.sqrt(xp.maximum(value, 0.0))


def _at(problem: LbfgsProblem, backend: Backend, x: Any, g: Any
        ) -> tuple[Any, PointOperators[Any, Any], Any, Any]:
    """(point data, operators, M^{-1} g, ||g||_{M^{-1}}) at x."""
    data = None if problem.point_data is None else problem.point_data(x)
    ops = point_operators(problem, x, data)
    assert ops.metric_solve is not None
    Mg = ops.metric_solve(g)
    return data, ops, Mg, _norm(backend.xp, problem.space.pair(g, Mg))


def init(problem: LbfgsProblem, backend: Backend, x0: Any, g0: Any) -> LbfgsState:
    """The state at x0 with an empty memory; g0, if not None, is the
    gradient at x0."""
    xp = backend.xp
    g = canonical(problem, x0, problem.grad(x0) if g0 is None else g0)
    data, _, Mg, gnorm = _at(problem, backend, x0, g)
    if problem.initial_step is None:
        gamma = xp.asarray(problem.gamma0)
    else:
        gamma = xp.where(gnorm > 0.0, problem.initial_step / xp.where(gnorm > 0.0, gnorm, 1.0), 1.0)
    return LbfgsState(x=x0, f=problem.fun(x0), g=g, data=data, Mg=Mg, gnorm=gnorm, metric_gnorm=gnorm,
                      memory=memory_empty(problem.space, backend, Mg, g, problem.memory),
                      gamma=gamma, stamp=xp.asarray(0))


def restart(problem: LbfgsProblem, backend: Backend, state: LbfgsState) -> LbfgsState:
    """The state with the memory cleared (gamma is kept)."""
    return dataclasses.replace(state, memory=memory_clear(problem.space, backend, state.memory))


def _floor(problem: LbfgsProblem, xp: Any, gs: Any, ss: Any, yy: Any, metric_gnorm: Any) -> Any:
    """The threshold <y, s> must exceed for the pair to be stored."""
    tol, test = problem.curvature_tol, problem.curvature_test
    if test == "relative":
        return tol * xp.abs(gs)
    if test == "positive":
        return xp.asarray(0.0)
    if test == "dennis-schnabel":
        return tol * _norm(xp, ss) * _norm(xp, yy)
    return tol * metric_gnorm * ss                     # cautious


def _fit(problem: LbfgsProblem, xp: Any, ys: Any, ss: Any, yy: Any, gamma: Any) -> Any:
    if problem.scaling == "y":
        return ys / yy
    if problem.scaling == "s":
        return ss / ys
    if problem.scaling == "geometric":
        return _norm(xp, ss / yy)
    return gamma


def lbfgs_step(problem: LbfgsProblem, backend: Backend, state: LbfgsState
               ) -> tuple[LbfgsState, RawLbfgsIteration]:
    """One iteration (see the module docstring). When the line search fails,
    the state is returned unchanged (the driver decides on a restart)."""
    xp, space = backend.xp, problem.space
    ops = point_operators(problem, state.x, state.data)
    assert ops.metric_solve is not None and ops.metric_apply is not None
    solve, apply = ops.metric_solve, ops.metric_apply
    memory = state.memory
    pairs = memory_count(backend, memory)

    # The direction: H g with the base gamma M^{-1}; with an empty memory the
    # base acts on g itself, and M^{-1} g is known.
    layout = memory_layout(backend, memory)
    q, a = two_loop_project(space, backend, memory, layout, state.g)
    u, base_solves = run_cond(backend, pairs > 0, lambda q: (solve(q), xp.asarray(1)),
                              lambda q: (state.Mg, xp.asarray(0)), q)
    d = space.scale_vector(-1.0, two_loop_finish(space, backend, memory, layout, a,
                                                 space.scale_vector(state.gamma, u)))
    slope = space.pair(state.g, d)
    step_norm = _norm(xp, space.pair(apply(d), d))
    denominator = state.metric_gnorm * step_norm
    cosine = -slope / xp.where(denominator > 0.0, denominator, 1.0)
    good = (xp.isfinite(slope) & xp.isfinite(step_norm) & (slope < 0.0)
            & (cosine >= problem.angle_tol))

    def keep(_: Any) -> tuple[Any, Any, Any, Any]:
        return d, slope, step_norm, cosine

    def steepest(_: Any) -> tuple[Any, Any, Any, Any]:
        """-gamma M^{-1} g: its slope and M-norm follow from ||g||_{M^{-1}}."""
        gamma, gnorm = state.gamma, state.metric_gnorm
        return (space.scale_vector(-gamma, state.Mg), -gamma * gnorm ** 2, gamma * gnorm,
                xp.asarray(1.0))

    d, slope, step_norm, cosine = run_cond(backend, good, keep, steepest, None)
    direction = xp.where(pairs == 0, STEEPEST, xp.where(good, QUASI_NEWTON, FALLBACK))
    memory = run_cond(backend, direction == FALLBACK, lambda m: memory_clear(space, backend, m),
                      lambda m: m, memory)
    used = memory_count(backend, memory)

    alpha0 = xp.asarray(1.0)
    if problem.step_max is not None:
        alpha0 = xp.minimum(1.0, problem.step_max / xp.where(step_norm > 0.0, step_norm, 1.0))

    def phi(alpha: Any) -> tuple[Any, Any]:
        x = problem.retract(state.x, space.scale_vector(alpha, d))
        return problem.fun(x), x

    found = backtracking(backend, phi, state.x, state.f, slope, alpha0, c1=problem.c1,
                         max_trials=problem.max_backtracks)
    accepted = found.status == ARMIJO
    s = space.scale_vector(found.alpha, d)
    gs = found.alpha * slope                                # <g, s>
    transport = problem.transport
    identity = transport is identity_transport
    need_ss = problem.scaling in ("s", "geometric") or problem.curvature_test in ("dennis-schnabel",
                                                                                  "cautious")
    need_yy = problem.scaling in ("y", "geometric") or problem.curvature_test == "dennis-schnabel"
    zero, nan = xp.asarray(0), xp.asarray(xp.nan)

    def move(_: Any) -> tuple[LbfgsState, tuple[Any, ...]]:
        x1, f1 = found.point, found.f
        g1 = canonical(problem, x1, problem.grad(x1))
        data1, ops1, Mg1, gnorm1 = _at(problem, backend, x1, g1)
        assert ops1.metric_solve is not None and ops1.metric_apply is not None
        moved = memory_transport(space, backend, transport, state.x, x1, memory)
        if identity:
            s1, gk = s, state.g
        else:
            s1 = transport.vector(state.x, x1, s)
            gk = canonical(problem, x1, transport.covector(state.x, x1, state.g))
        y = space.axpy_covector(-1.0, gk, g1)
        ys = space.pair(y, s1)
        defect = (xp.asarray(0.0) if identity
                  else space.pair(gk, s1) / xp.where(gs != 0.0, gs, 1.0) - 1.0)
        ss = space.pair(ops1.metric_apply(s1), s1) if need_ss else nan
        yy = space.pair(y, ops1.metric_solve(y)) if need_yy else nan
        floor = _floor(problem, xp, gs, ss, yy, state.metric_gnorm)
        ok = xp.isfinite(ys) & (ys > floor)
        dropped = zero
        if problem.pairings == "recomputed" and not identity:
            moved, dropped = memory_recompute(space, backend, moved)
        stored = run_cond(backend, ok, lambda m: memory_store(space, backend, m, s1, y, state.stamp, floor),
                          lambda m: m, moved)
        fit = _fit(problem, xp, ys, ss, yy, state.gamma)
        gamma = xp.where(ok & xp.isfinite(fit) & (fit > 0.0), fit, state.gamma)
        new = LbfgsState(x=x1, f=f1, g=g1, data=data1, Mg=Mg1, gnorm=gnorm1, metric_gnorm=gnorm1,
                         memory=stored, gamma=gamma, stamp=state.stamp + xp.where(ok, 1, 0))
        solves = xp.asarray(1 + int(need_yy))
        applies = xp.asarray(int(need_ss))
        if identity:
            vectors = covectors = blocks = zero
        else:
            blocks = xp.asarray(int(transport.vector_block is not None)
                                + int(transport.covector_block is not None))
            vectors = 1 + (zero if transport.vector_block is not None else used)
            covectors = 1 + (zero if transport.covector_block is not None else used)
        return new, (ys, ys / xp.where(gs != 0.0, xp.abs(gs), 1.0), xp.where(ok, STORED, SKIPPED),
                     dropped, defect, solves, applies, vectors, covectors, blocks)

    def stay(_: Any) -> tuple[LbfgsState, tuple[Any, ...]]:
        return state, (nan, nan, xp.asarray(NO_UPDATE), zero, nan, zero, zero, zero, zero, zero)

    new_state, (ys, curvature, update, dropped, defect, solves, applies, vectors, covectors,
                blocks) = run_cond(backend, accepted, move, stay, None)
    return new_state, RawLbfgsIteration(
        direction=direction, pairs=pairs, gamma=state.gamma, step_norm=step_norm, cosine=cosine,
        slope=slope, alpha=found.alpha, trials=found.trials, status=found.status,
        actual=state.f - found.f, ys=ys, curvature=curvature, update=update, dropped=dropped,
        transport_defect=defect, step=s, trace_alpha=found.trace_alpha, trace_f=found.trace_f,
        metric_solves=base_solves + solves, metric_applies=1 + applies, vector_transports=vectors,
        covector_transports=covectors, block_transports=blocks)
