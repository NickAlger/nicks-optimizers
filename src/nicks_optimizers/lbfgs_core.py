"""An iteration of lbfgs as pure pieces a backend can compile (see
dev/lbfgs_plan.md).

  LbfgsProblem    the problem pieces and the options, with defaults filled
                  in (fixed during a run: compiled pieces close over it)
  LbfgsState      what is known at the current point: x, f, the canonical
                  gradient g, the point data, M^{-1} g (and P^{-1} g when
                  needed), the gradient norms, the memory, the scalings
                  gamma (of the base gamma M^{-1}) and gamma_P (of the
                  preconditioner's base gamma_P P^{-1}) and the stamp of the
                  next stored pair
  init(problem, backend, x0, g0) -> LbfgsState
  lbfgs_step(problem, backend, state) -> (LbfgsState, RawLbfgsIteration)
                  the direction d = -H g (the two-loop recursion of
                  lbfgs_memory), its descent and angle tests, the line
                  search along retract(x, alpha d), and, when it succeeds,
                  the state at the new point: the gradient and its norms, the
                  memory transported there, the new pair (s, y) tested and
                  stored, the scalings refitted
  restart(problem, backend, state) -> LbfgsState
                  the same state with the memory cleared

Everything geometric is measured with the metric M, given by both of its
operators (M^{-1}, the Riesz map, and M): the base gamma M^{-1}, the scaling
fits, the gradient norm, ||d||_M and the angle between -g and d. A
preconditioner P (precond_solve, P^{-1}) is an accelerant only: the same
recursion with the base gamma_P P^{-1} proposes a direction, used if it
passes the angle test in M ("safeguarded") or the descent test ("trusted"),
else the M direction is used. P may be nonsymmetric, nonlinear or random:
it is applied once per iteration as a black box (twice with the scale fit),
and the stored pairs, being data from f, never depend on it. Nothing here
branches in Python on computed values, and nothing imports numpy.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

from .backend import Backend, backend_state, run_cond
from .lbfgs_memory import (
    Memory,
    block_base,
    finish_coefficients,
    memory_clear,
    memory_count,
    memory_empty,
    memory_layout,
    memory_recompute,
    memory_store,
    memory_transport,
    two_loop_project,
)
from .line_search import ARMIJO, backtracking
from .newton_core import PointOperators
from .spaces import VectorSpace
from .transport import VectorTransport, identity_transport

Scaling = Literal["y", "s", "geometric", "none"]
ScalingP = Literal["fit", "none"]
Pairings = Literal["stored", "recomputed"]
CurvatureTest = Literal["relative", "positive", "dennis-schnabel", "cautious"]
PrecondMode = Literal["safeguarded", "trusted"]
ConvergenceNorm = Literal["metric", "preconditioner"]
Direction = Literal["quasi-newton", "steepest", "fallback"]
Update = Literal["-", "stored", "skipped"]
Base = Literal["M", "P"]

QUASI_NEWTON, STEEPEST, FALLBACK = range(3)
DIRECTIONS: tuple[Direction, ...] = ("quasi-newton", "steepest", "fallback")
NO_UPDATE, STORED, SKIPPED = range(3)
UPDATES: tuple[Update, ...] = ("-", "stored", "skipped")
BASE_M, BASE_P = range(2)
BASES: tuple[Base, ...] = ("M", "P")
SCALINGS: tuple[Scaling, ...] = ("y", "s", "geometric", "none")
SCALINGS_P: tuple[ScalingP, ...] = ("fit", "none")
PAIRINGS: tuple[Pairings, ...] = ("stored", "recomputed")
CURVATURE_TESTS: tuple[CurvatureTest, ...] = ("relative", "positive", "dennis-schnabel", "cautious")
PRECOND_MODES: tuple[PrecondMode, ...] = ("safeguarded", "trusted")
CONVERGENCE_NORMS: tuple[ConvergenceNorm, ...] = ("metric", "preconditioner")
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
      gamma0           gamma (and gamma_P) before the first stored pair (the
                       first step is -gamma0 M^{-1} g, or -gamma0 P^{-1} g),
                       unless initial_step is given
      initial_step     if given, gamma0 = initial_step / ||g0||_{M^{-1}} (and
                       gamma_P = initial_step / ||P^{-1} g0||_M): a first step
                       of that M-length
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
                       ||g||_{M^{-1}} ||d||_M (else the M direction for P's,
                       and -gamma M^{-1} g with the memory cleared for M's)
      c1, max_backtracks, step_max   the line search (line_search.backtracking)
      fixed_metric     M does not depend on x (and the transport is the
                       identity): M^{-1} y and M s are kept with each pair, so
                       the M direction and its fits need no operator
      precond_solve    P^{-1}(x, w), the accelerant (or from operators_at)
      precond_mode     "safeguarded" (P's direction must pass the angle test)
                       or "trusted" (the descent test only)
      scaling_P        "fit" (gamma_P = <M u, s> / <M u, u> with u = P^{-1} y:
                       the fit of gamma_P P^{-1} y = s in the M-norm) or
                       "none" (gamma_P stays gamma0)
      convergence_norm "metric" (||g||_{M^{-1}}) or "preconditioner"
                       (||g||_{P^{-1}}, for a symmetric positive definite P)
      fixed_preconditioner   P is linear and does not depend on x (and the
                       transport is the identity): P^{-1} y is kept with each
                       pair and P^{-1} g computed at each point, so P's base
                       and its fit need no further P^{-1}
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
    fixed_metric: bool = False
    precond_solve: Optional[Callable[[Any, Any], Any]] = None
    precond_mode: PrecondMode = "safeguarded"
    scaling_P: ScalingP = "fit"
    convergence_norm: ConvergenceNorm = "metric"
    fixed_preconditioner: bool = False


@backend_state
@dataclass(frozen=True, kw_only=True)
class LbfgsState:
    """What is known at the current point.

      x, f, g          the point, f(x) and the canonical gradient
      data             point_data(x), or None
      Mg               M^{-1} g (computed for the norm; the base applied to g)
      Pg               P^{-1} g when the convergence norm or a fixed
                       preconditioner needs it, else None
      gnorm            ||g|| in the convergence norm
      metric_gnorm     ||g||_{M^{-1}}
      memory           the stored pairs, at x
      gamma, gamma_P   the scalings of the bases gamma M^{-1} and gamma_P P^{-1}
      stamp            the stamp the next stored pair gets
      metric_solves, precond_solves, metric_applies
                       the applications made to form this state (counted by
                       the driver for the initial state; the step's counts
                       include them for the others)
    """
    x: Any
    f: Any
    g: Any
    data: Any
    Mg: Any
    Pg: Any
    gnorm: Any
    metric_gnorm: Any
    memory: Memory[Any, Any]
    gamma: Any
    gamma_P: Any
    stamp: Any
    metric_solves: Any
    precond_solves: Any
    metric_applies: Any


@backend_state
@dataclass(frozen=True, kw_only=True)
class RawLbfgsIteration:
    """What one iteration reports (arrays).

      direction          QUASI_NEWTON, STEEPEST (empty memory) or FALLBACK
                         (the memory's direction failed the descent or angle
                         test: -gamma M^{-1} g instead, memory cleared)
      base               BASE_M or BASE_P: the base of the direction taken
      pairs, gamma       the pairs and the scaling of the base the direction
                         used
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
      metric_solves, precond_solves, metric_applies, vector_transports,
      covector_transports, block_transports
                         applications of the operators and calls of the
                         transport maps in this iteration
    """
    direction: Any
    base: Any
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
    precond_solves: Any
    metric_applies: Any
    vector_transports: Any
    covector_transports: Any
    block_transports: Any


def _identity(w: Any) -> Any:
    return w


def point_operators(problem: LbfgsProblem, x: Any, data: Any) -> PointOperators[Any, Any]:
    """M^{-1}, M and P^{-1} (or None) at x, from operators_at or the keyword
    operators (M = I for numpy arrays without a space when neither M operator
    is given)."""
    if problem.operators_at is not None:
        given = problem.operators_at(x, data)
        solve, apply, precond = given.metric_solve, given.metric_apply, given.precond_solve
    else:
        ms, ma, ps = problem.metric_solve, problem.metric_apply, problem.precond_solve
        solve = None if ms is None else (lambda w: ms(x, w))
        apply = None if ma is None else (lambda v: ma(x, v))
        precond = None if ps is None else (lambda w: ps(x, w))
    if solve is None and apply is None and problem.numpy_default:
        solve, apply = _identity, _identity
    if solve is None or apply is None:
        raise ValueError("lbfgs needs both metric_solve (M^{-1}) and metric_apply (M)")
    if precond is None and (problem.convergence_norm == "preconditioner" or problem.fixed_preconditioner):
        raise ValueError("convergence_norm='preconditioner' and fixed_preconditioner need precond_solve")
    return PointOperators(metric_solve=solve, metric_apply=apply, precond_solve=precond)


def canonical(problem: LbfgsProblem, x: Any, w: Any) -> Any:
    return w if problem.project_covector is None else problem.project_covector(x, w)


def _norm(xp: Any, value: Any) -> Any:
    return xp.sqrt(xp.maximum(value, 0.0))


def _needs_Pg(problem: LbfgsProblem) -> bool:
    return problem.convergence_norm == "preconditioner" or problem.fixed_preconditioner


def _at(problem: LbfgsProblem, backend: Backend, x: Any, g: Any
        ) -> tuple[Any, PointOperators[Any, Any], Any, Any, Any, Any]:
    """(point data, operators, M^{-1} g, P^{-1} g or None, the convergence
    norm, ||g||_{M^{-1}}) at x."""
    xp, space = backend.xp, problem.space
    data = None if problem.point_data is None else problem.point_data(x)
    ops = point_operators(problem, x, data)
    assert ops.metric_solve is not None
    Mg = ops.metric_solve(g)
    metric_gnorm = _norm(xp, space.pair(g, Mg))
    Pg = None
    if _needs_Pg(problem):
        assert ops.precond_solve is not None
        Pg = ops.precond_solve(g)
    gnorm = _norm(xp, space.pair(g, Pg)) if problem.convergence_norm == "preconditioner" else metric_gnorm
    return data, ops, Mg, Pg, gnorm, metric_gnorm


def init(problem: LbfgsProblem, backend: Backend, x0: Any, g0: Any) -> LbfgsState:
    """The state at x0 with an empty memory; g0, if not None, is the
    gradient at x0."""
    xp, space = backend.xp, problem.space
    g = canonical(problem, x0, problem.grad(x0) if g0 is None else g0)
    data, ops, Mg, Pg, gnorm, metric_gnorm = _at(problem, backend, x0, g)
    has_P = ops.precond_solve is not None
    precond_solves, applies = int(Pg is not None), 0
    gamma = gamma_P = xp.asarray(problem.gamma0)
    if problem.initial_step is not None:
        step = problem.initial_step
        gamma = xp.where(metric_gnorm > 0.0, step / xp.where(metric_gnorm > 0.0, metric_gnorm, 1.0), 1.0)
        if has_P:
            assert ops.precond_solve is not None and ops.metric_apply is not None
            u = Pg if Pg is not None else ops.precond_solve(g)
            unorm = _norm(xp, space.pair(ops.metric_apply(u), u))
            gamma_P = xp.where(unorm > 0.0, step / xp.where(unorm > 0.0, unorm, 1.0), 1.0)
            precond_solves, applies = 1, 1
    return LbfgsState(x=x0, f=problem.fun(x0), g=g, data=data, Mg=Mg, Pg=Pg, gnorm=gnorm,
                      metric_gnorm=metric_gnorm,
                      memory=memory_empty(space, backend, Mg, g, problem.memory,
                                          metric_blocks=problem.fixed_metric,
                                          precond_block=problem.fixed_preconditioner),
                      gamma=gamma, gamma_P=gamma_P, stamp=xp.asarray(0), metric_solves=xp.asarray(1),
                      precond_solves=xp.asarray(precond_solves), metric_applies=xp.asarray(applies))


def restart(problem: LbfgsProblem, backend: Backend, state: LbfgsState) -> LbfgsState:
    """The state with the memory cleared (the scalings are kept)."""
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
    """The scaling of the M base fitted to a pair, or gamma if not valid
    (no division by a nonpositive number is evaluated)."""
    def ratio(a: Any, b: Any) -> Any:
        return xp.where(b > 0.0, a / xp.where(b > 0.0, b, 1.0), xp.nan)
    if problem.scaling == "y":
        fit = ratio(ys, yy)
    elif problem.scaling == "s":
        fit = ratio(ss, ys)
    elif problem.scaling == "geometric":
        fit = _norm(xp, ratio(ss, yy))
    else:
        return gamma
    return xp.where(xp.isfinite(fit) & (fit > 0.0), fit, gamma)


def lbfgs_step(problem: LbfgsProblem, backend: Backend, state: LbfgsState
               ) -> tuple[LbfgsState, RawLbfgsIteration]:
    """One iteration (see the module docstring). When the line search fails,
    the state is returned unchanged (the driver decides on a restart)."""
    xp, space = backend.xp, problem.space
    ops = point_operators(problem, state.x, state.data)
    assert ops.metric_solve is not None and ops.metric_apply is not None
    solve, apply, precond = ops.metric_solve, ops.metric_apply, ops.precond_solve
    has_P = precond is not None
    memory = state.memory
    pairs = memory_count(backend, memory)
    m = problem.memory
    zero, one, nan = xp.asarray(0), xp.asarray(1), xp.asarray(xp.nan)
    need_ss = problem.scaling in ("s", "geometric") or problem.curvature_test in ("dennis-schnabel",
                                                                                  "cautious")
    need_yy = problem.scaling in ("y", "geometric") or problem.curvature_test == "dennis-schnabel"

    layout = memory_layout(backend, memory)
    q, a = two_loop_project(space, backend, memory, layout, state.g)

    def finish(r0: Any) -> tuple[Any, Any]:
        """(d = -H g, the coefficients a - b) for the base applied, r0."""
        coefficients = finish_coefficients(space, backend, memory, layout, a, r0)
        Hg = space.axpy_vector(1.0, space.combine_vectors(memory.S, coefficients, m), r0)
        return space.scale_vector(-1.0, Hg), coefficients

    def m_direction(gamma: Any) -> tuple[Any, Any, Any, Any]:
        """(d, M d, solves, products) with the base gamma M^{-1}: free from the
        metric blocks for a constant metric; otherwise a solve (none with an
        empty memory, M^{-1} g being known) and a product with M."""
        if problem.fixed_metric:
            assert memory.Z is not None and memory.W is not None
            d, coefficients = finish(block_base(space, backend, memory, layout, a, memory.Z, state.Mg, gamma))
            MHg = space.axpy_covector(1.0, space.combine_covectors(memory.W, coefficients, m),
                                      space.scale_covector(gamma, q))
            return d, space.scale_covector(-1.0, MHg), zero, zero
        u, solves = run_cond(backend, pairs > 0, lambda q: (solve(q), one), lambda q: (state.Mg, zero), q)
        d, _ = finish(space.scale_vector(gamma, u))
        return d, apply(d), solves, one

    def tested(d: Any, Md: Any, tol: float) -> tuple[Any, Any, Any, Any]:
        """(slope, ||d||_M, cosine, whether d passes the descent and angle tests)."""
        slope = space.pair(state.g, d)
        step_norm = _norm(xp, space.pair(Md, d))
        denominator = state.metric_gnorm * step_norm
        cosine = -slope / xp.where(denominator > 0.0, denominator, 1.0)
        good = xp.isfinite(slope) & xp.isfinite(step_norm) & (slope < 0.0) & (cosine >= tol)
        return slope, step_norm, cosine, good

    def refitted_gamma() -> tuple[Any, Any, Any]:
        """(gamma, solves, products): gamma of the M base refitted to the
        newest stored pair, at this point (with a preconditioner the M base
        is a fallback, so its fit is made only when it is used); gamma as it
        is with an empty memory, or when it is kept up to date at storage."""
        if not has_P or problem.fixed_metric or problem.scaling == "none":
            return state.gamma, zero, zero
        uses_yy, uses_ss = problem.scaling in ("y", "geometric"), problem.scaling in ("s", "geometric")

        def refit(_: Any) -> tuple[Any, Any, Any]:
            j = xp.take(xp.argsort(memory.stamp), m - 1)          # the newest slot
            s_j, y_j = space.vector_member(memory.S, j), space.covector_member(memory.Y, j)
            yy = space.pair(y_j, solve(y_j)) if uses_yy else nan
            ss = space.pair(apply(s_j), s_j) if uses_ss else nan
            return (_fit(problem, xp, space.pair(y_j, s_j), ss, yy, state.gamma),
                    xp.asarray(int(uses_yy)), xp.asarray(int(uses_ss)))

        return run_cond(backend, pairs > 0, refit, lambda _: (state.gamma, zero, zero), None)

    if not has_P:
        d, Md, base_solves, base_applies = m_direction(state.gamma)
        slope, step_norm, cosine, good = tested(d, Md, problem.angle_tol)
        base, gamma_used, gamma_M, precond_solves = xp.asarray(BASE_M), state.gamma, state.gamma, zero
    else:
        # The preconditioner's proposal, tested in the geometry of M.
        assert precond is not None
        if problem.fixed_preconditioner:
            assert memory.ZP is not None
            r0 = block_base(space, backend, memory, layout, a, memory.ZP, state.Pg, state.gamma_P)
            precond_solves = zero
        else:
            if state.Pg is not None:
                u, precond_solves = run_cond(backend, pairs > 0, lambda q: (precond(q), one),
                                             lambda q: (state.Pg, zero), q)
            else:
                u, precond_solves = precond(q), one
            r0 = space.scale_vector(state.gamma_P, u)
        dP, _ = finish(r0)
        MdP = apply(dP)
        tol_P = problem.angle_tol if problem.precond_mode == "safeguarded" else 0.0
        slope_P, norm_P, cosine_P, good_P = tested(dP, MdP, tol_P)

        def take_P(_: Any) -> tuple[Any, ...]:
            return (dP, MdP, slope_P, norm_P, cosine_P, xp.asarray(True), xp.asarray(BASE_P), zero,
                    zero, state.gamma_P, state.gamma)

        def take_M(_: Any) -> tuple[Any, ...]:
            gamma, fit_solves, fit_applies = refitted_gamma()
            d, Md, solves, applies = m_direction(gamma)
            slope, step_norm, cosine, good = tested(d, Md, problem.angle_tol)
            return (d, Md, slope, step_norm, cosine, good, xp.asarray(BASE_M), solves + fit_solves,
                    applies + fit_applies, gamma, gamma)

        (d, Md, slope, step_norm, cosine, good, base, base_solves, base_applies, gamma_used,
         gamma_M) = run_cond(backend, good_P, take_P, take_M, None)
        base_applies = base_applies + 1                       # M d_P

    def keep(_: Any) -> tuple[Any, ...]:
        return d, Md, slope, step_norm, cosine, base, gamma_used

    def steepest(_: Any) -> tuple[Any, ...]:
        """-gamma M^{-1} g: M d = -gamma g, and its slope and M-norm follow
        from ||g||_{M^{-1}}."""
        gnorm = state.metric_gnorm
        return (space.scale_vector(-gamma_M, state.Mg), space.scale_covector(-gamma_M, state.g),
                -gamma_M * gnorm ** 2, gamma_M * gnorm, xp.asarray(1.0), xp.asarray(BASE_M), gamma_M)

    d, Md, slope, step_norm, cosine, base, gamma_used = run_cond(backend, good, keep, steepest, None)
    direction = xp.where(pairs == 0, STEEPEST, xp.where(good, QUASI_NEWTON, FALLBACK))
    memory = run_cond(backend, direction == FALLBACK, lambda m_: memory_clear(space, backend, m_),
                      lambda m_: m_, memory)
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
    fixed = problem.fixed_metric

    def move(_: Any) -> tuple[LbfgsState, tuple[Any, ...]]:
        x1, f1 = found.point, found.f
        g1 = canonical(problem, x1, problem.grad(x1))
        data1, ops1, Mg1, Pg1, gnorm1, metric_gnorm1 = _at(problem, backend, x1, g1)
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
        solves, applies, psolves = one, zero, xp.asarray(int(Pg1 is not None))
        if fixed:             # M^{-1} y and M s by linearity, from what is known
            metric_pair = (space.axpy_vector(-1.0, state.Mg, Mg1), space.scale_covector(found.alpha, Md))
            ss, yy = space.pair(metric_pair[1], s1), space.pair(y, metric_pair[0])
        else:
            metric_pair = None
            # with a preconditioner the M base's fit is made when it is used
            fit_M = problem.scaling != "none" and not has_P
            uses_ss = need_ss and (fit_M or problem.curvature_test in ("dennis-schnabel", "cautious"))
            uses_yy = need_yy and (fit_M or problem.curvature_test == "dennis-schnabel")
            ss = space.pair(ops1.metric_apply(s1), s1) if uses_ss else nan
            yy = space.pair(y, ops1.metric_solve(y)) if uses_yy else nan
            solves, applies = solves + int(uses_yy), applies + int(uses_ss)
        floor = _floor(problem, xp, gs, ss, yy, state.metric_gnorm)
        ok = xp.isfinite(ys) & (ys > floor)
        precond_y = None
        gamma_P = state.gamma_P
        if has_P:
            assert ops1.precond_solve is not None
            if problem.fixed_preconditioner:
                precond_y = space.axpy_vector(-1.0, state.Pg, Pg1)
            if problem.scaling_P == "fit":
                u = precond_y if precond_y is not None else ops1.precond_solve(y)
                Mu = ops1.metric_apply(u)
                fit_P = space.pair(Mu, s1) / space.pair(Mu, u)
                gamma_P = xp.where(ok & xp.isfinite(fit_P) & (fit_P > 0.0), fit_P, state.gamma_P)
                psolves = psolves + int(precond_y is None)
                applies = applies + 1
        dropped = zero
        if problem.pairings == "recomputed" and not identity:
            moved, dropped = memory_recompute(space, backend, moved)
        stored = run_cond(backend, ok,
                          lambda m_: memory_store(space, backend, m_, s1, y, state.stamp, floor,
                                                  metric_pair=metric_pair, precond_y=precond_y),
                          lambda m_: m_, moved)
        gamma = xp.where(ok, _fit(problem, xp, ys, ss, yy, gamma_M), gamma_M) if (fixed or not has_P) \
            else gamma_M
        new = LbfgsState(x=x1, f=f1, g=g1, data=data1, Mg=Mg1, Pg=Pg1, gnorm=gnorm1,
                         metric_gnorm=metric_gnorm1, memory=stored, gamma=gamma, gamma_P=gamma_P,
                         stamp=state.stamp + xp.where(ok, 1, 0), metric_solves=solves,
                         precond_solves=psolves, metric_applies=applies)
        if identity:
            vectors = covectors = blocks = zero
        else:
            blocks = xp.asarray(int(transport.vector_block is not None)
                                + int(transport.covector_block is not None))
            vectors = 1 + (zero if transport.vector_block is not None else used)
            covectors = 1 + (zero if transport.covector_block is not None else used)
        return new, (ys, ys / xp.where(gs != 0.0, xp.abs(gs), 1.0), xp.where(ok, STORED, SKIPPED),
                     dropped, defect, solves, psolves, applies, vectors, covectors, blocks)

    def stay(_: Any) -> tuple[LbfgsState, tuple[Any, ...]]:
        kept = dataclasses.replace(state, gamma=gamma_M)
        return kept, (nan, nan, xp.asarray(NO_UPDATE), zero, nan, zero, zero, zero, zero, zero, zero)

    new_state, (ys, curvature, update, dropped, defect, solves, psolves, applies, vectors, covectors,
                blocks) = run_cond(backend, accepted, move, stay, None)
    return new_state, RawLbfgsIteration(
        direction=direction, base=base, pairs=pairs, gamma=gamma_used, step_norm=step_norm,
        cosine=cosine, slope=slope, alpha=found.alpha, trials=found.trials, status=found.status,
        actual=state.f - found.f, ys=ys, curvature=curvature, update=update, dropped=dropped,
        transport_defect=defect, step=s, trace_alpha=found.trace_alpha, trace_f=found.trace_f,
        metric_solves=base_solves + solves, precond_solves=precond_solves + psolves,
        metric_applies=base_applies + applies, vector_transports=vectors,
        covector_transports=covectors, block_transports=blocks)
