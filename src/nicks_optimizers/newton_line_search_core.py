"""The line-search part of an iteration of line_search_newton, as a pure
piece a backend can compile (see backend and dev/line_search_plan.md).

The direction is computed by the shared pieces of newton_core (step_start,
step_grow, step_continue) with the solvers in line-search mode; then

  line_search_finish(problem, options, backend, state, carry, gnorm0)
                 -> (OuterState, RawLineSearchIteration)
                 the direction, the slope <g, d> and ||d||_M, the line
                 search along retract(x, alpha d), and, when it succeeds,
                 the gradient, the point data and the gradient norms at the
                 new point (under backend.cond)

  LineSearchOptions       c1, max_backtracks and step_max
  RawLineSearchIteration  what one iteration reports, as arrays; the driver
                          turns it into a LineSearchRecord

A direction that is not a descent direction (rounding, or a non-finite
direction) is replaced by the preconditioned steepest descent direction
under backend.cond, so the replacement costs operator applications only
when it happens. Nothing here branches in Python on computed values, and
nothing imports numpy.
"""

from dataclasses import dataclass
from typing import Any, Literal, Optional

from .backend import Backend, backend_state, run_cond
from .krylov_solvers import NEGATIVE
from .line_search import ARMIJO, backtracking
from .newton_core import CoreProblem, OuterState, canonical, forcing_term, state_at, subproblem_at

Direction = Literal["newton", "truncated", "steepest", "fallback"]
NEWTON, TRUNCATED, STEEPEST, FALLBACK = range(4)
DIRECTIONS: tuple[Direction, ...] = ("newton", "truncated", "steepest", "fallback")


@dataclass(frozen=True, kw_only=True)
class LineSearchOptions:
    """The options of the line search: the Armijo constant c1, the most
    evaluations of f per iteration, and the cap on ||alpha d||_M (None: no
    cap)."""
    c1: float
    max_backtracks: int
    step_max: Optional[float]


@backend_state
@dataclass(frozen=True, kw_only=True)
class RawLineSearchIteration:
    """What one iteration reports (arrays).

      forcing             the relative Krylov tolerance used
      direction           NEWTON, TRUNCATED, STEEPEST or FALLBACK (see
                          DIRECTIONS and LineSearchRecord)
      slope, step_norm    <g, d> and ||d||_M
      alpha, trials       the step length taken (the last one tried if the
                          search failed) and the evaluations of f
      status              the line search's status (line_search.ARMIJO, ...)
      predicted, actual   the decrease of the quadratic model along alpha d
                          (nan for a fallback direction) and f(x) - f there
      subproblem          the solver's RawSubproblem
      step                alpha d
      trace_alpha, trace_f    the trials
      precond_solves, metric_applies   applications made for a fallback
    """
    forcing: Any
    direction: Any
    slope: Any
    step_norm: Any
    alpha: Any
    trials: Any
    status: Any
    predicted: Any
    actual: Any
    subproblem: Any
    step: Any
    trace_alpha: Any
    trace_f: Any
    precond_solves: Any
    metric_applies: Any


def line_search_finish(problem: CoreProblem, options: LineSearchOptions, backend: Backend,
                       state: OuterState, carry: Any, gnorm0: Any
                       ) -> tuple[OuterState, RawLineSearchIteration]:
    """The rest of the iteration once the solve has stopped: the direction,
    the line search, and the state at the next point (the accepted point,
    or the same point when the search failed)."""
    xp, space = backend.xp, problem.space
    solver, ops = subproblem_at(problem, state)
    d, sub = solver.result(backend, space, ops, carry)
    slope = space.pair(state.g, d)
    descent = xp.isfinite(slope) & (slope < 0.0) & xp.isfinite(sub.step_norm)

    def keep(_: Any) -> tuple[Any, Any, Any, Any, Any]:
        return d, slope, sub.step_norm, xp.asarray(0), xp.asarray(0)

    def steepest(_: Any) -> tuple[Any, Any, Any, Any, Any]:
        """-P^{-1} g (-M^{-1} g for the flexible solver), from the start
        vector when the gradient norms computed it (see
        newton_core.gradient_norms)."""
        solves, applies = xp.asarray(0), xp.asarray(0)
        if ops.precond_solve is None or problem.krylov == "flexible":
            v = state.start                   # M^{-1} g: ||v||_M^2 = <g, v>
            gv = space.pair(state.g, v)
            norm2 = gv
        else:                                 # Lanczos with a separate P: v = P^{-1} g
            assert ops.metric_apply is not None
            if state.start is not None:       # (computed for the preconditioner norm)
                v = state.start
            else:
                v, solves = ops.precond_solve(state.g), xp.asarray(1)
            gv = space.pair(state.g, v)
            norm2, applies = space.pair(ops.metric_apply(v), v), xp.asarray(1)
        return (space.scale_vector(-1.0, v), -gv, xp.sqrt(xp.maximum(norm2, 0.0)), solves,
                applies)

    d, slope, step_norm, precond_solves, metric_applies = run_cond(backend, descent, keep, steepest,
                                                                   None)
    alpha0 = xp.asarray(1.0)
    if options.step_max is not None:
        alpha0 = xp.minimum(1.0, options.step_max / xp.where(step_norm > 0.0, step_norm, 1.0))

    def phi(alpha: Any) -> tuple[Any, Any]:
        x = problem.retract(state.x, space.scale_vector(alpha, d))
        return problem.fun(x), x

    found = backtracking(backend, phi, state.x, state.f, slope, alpha0, c1=options.c1,
                         max_trials=options.max_backtracks)
    accepted = found.status == ARMIJO

    def move(_: Any) -> OuterState:
        g = canonical(problem, found.point, problem.grad(found.point))
        return state_at(problem, backend, found.point, found.f, g)

    def stay(_: Any) -> OuterState:
        return state

    new_state = run_cond(backend, accepted, move, stay, None)
    curvature = 2.0 * (sub.true_model_value - slope)          # <H d, d>, for the solver's d
    alpha = found.alpha
    predicted = xp.where(descent, -(alpha * slope + 0.5 * alpha ** 2 * curvature), xp.nan)
    direction = xp.where(~descent, FALLBACK, xp.where(
        sub.status == NEGATIVE, xp.where(sub.krylov_steps == 1, STEEPEST, TRUNCATED), NEWTON))
    return new_state, RawLineSearchIteration(
        forcing=forcing_term(xp, state.gnorm, gnorm0, problem.forcing_max), direction=direction,
        slope=slope, step_norm=step_norm, alpha=alpha, trials=found.trials, status=found.status,
        predicted=predicted, actual=state.f - found.f, subproblem=sub,
        step=space.scale_vector(alpha, d), trace_alpha=found.trace_alpha, trace_f=found.trace_f,
        precond_solves=precond_solves, metric_applies=metric_applies)
