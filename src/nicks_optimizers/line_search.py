"""Backtracking line search with the Armijo condition, as a pure function a
backend can compile (see backend): reusable by any optimizer that has a
point, a descent direction and the slope along it.

    backtracking(backend, phi, x0, f0, slope, alpha0, c1=1e-4, max_trials=30,
                 shrink=(0.1, 0.5)) -> LineSearchResult

phi(alpha) returns (f, point) for the step alpha along the direction, e.g.
(f(retract(x, alpha d)), retract(x, alpha d)); f0 = phi(0) and slope =
phi'(0) = <g, d> < 0. The search tries alpha0 first and accepts the first
trial with

    phi(alpha) <= f0 + c1 alpha slope + delta,   delta = 10 eps max(1, |f0|),

where delta absorbs rounding near convergence (the safeguard of the
trust-region ratio rho, see criteria.reduction_ratio). After a rejected
trial the next step minimizes the quadratic through f0, slope and the trial,
then the cubic through f0, slope and the last two trials (Dennis and
Schnabel, Numerical Methods for Unconstrained Optimization, A6.3.1),
clipped to [shrink[0], shrink[1]] times the rejected step; a non-finite
phi(alpha) gives shrink[0] alpha. It is a bounded loop of at most
max_trials evaluations of phi, with no Python control flow on values, so a
compiled loop built from scan skips the trials after acceptance without
evaluating phi.
"""

from dataclasses import dataclass
from typing import Any, Callable

from .backend import Backend, backend_state, run_loop

SEARCHING, ARMIJO, FAILED, NOT_DESCENT = range(4)
LINE_SEARCH_STATUS = {ARMIJO: "armijo", FAILED: "failed", NOT_DESCENT: "not descent"}


@backend_state
@dataclass(frozen=True, kw_only=True)
class LineSearchResult:
    """The outcome of backtracking, as arrays of the backend.

      alpha       the last step tried (the accepted one if status is ARMIJO)
      f, point    phi at alpha (x0 and f0 if nothing was tried)
      trials      evaluations of phi
      status      ARMIJO (accepted), FAILED (max_trials rejected trials) or
                  NOT_DESCENT (slope not negative or not finite: nothing
                  tried); see LINE_SEARCH_STATUS
      trace_alpha, trace_f
                  the steps tried and phi there, arrays of length max_trials
                  whose first `trials` entries are set
    """
    alpha: Any
    f: Any
    point: Any
    trials: Any
    status: Any
    trace_alpha: Any
    trace_f: Any


@backend_state
@dataclass(frozen=True, kw_only=True)
class _Search:
    """The carry: the result so far, and the last finite trial before the
    current one (step and value) for the cubic interpolation (alpha_prev = 0
    while there is none)."""
    result: LineSearchResult
    alpha_prev: Any
    f_prev: Any


def armijo_allowance(xp: Any, f0: Any) -> Any:
    """delta = 10 eps max(1, |f0|), the rounding allowance of the test."""
    eps = xp.finfo(xp.asarray(f0).dtype).eps
    return 10.0 * eps * xp.maximum(1.0, xp.abs(f0))


def next_step(xp: Any, f0: Any, slope: Any, alpha: Any, f_alpha: Any, alpha_prev: Any,
              f_prev: Any, shrink: tuple[float, float]) -> Any:
    """The step after rejecting alpha: the minimizer of the quadratic model
    (no previous trial) or of the cubic model (alpha_prev > 0) of phi,
    clipped to [shrink[0], shrink[1]] alpha; shrink[0] alpha when phi(alpha)
    is not finite."""
    lo, hi = shrink[0] * alpha, shrink[1] * alpha
    finite = xp.isfinite(f_alpha)
    # > 0 after a rejected finite trial; a placeholder otherwise (the result is
    # lo then), so that no invalid value is computed
    excess = xp.where(finite, f_alpha - f0 - slope * alpha, 1.0)
    quadratic = -slope * alpha ** 2 / (2.0 * xp.where(excess > 0.0, excess, 1.0))
    excess_prev = f_prev - f0 - slope * alpha_prev
    has_prev = alpha_prev > 0.0
    ap = xp.where(has_prev, alpha_prev, 1.0)
    gap = xp.where(has_prev & (alpha != ap), alpha - ap, 1.0)
    a = (excess / alpha ** 2 - excess_prev / ap ** 2) / gap
    b = (-ap * excess / alpha ** 2 + alpha * excess_prev / ap ** 2) / gap
    disc = xp.maximum(b ** 2 - 3.0 * a * slope, 0.0)
    cubic_a = (-b + xp.sqrt(disc)) / (3.0 * xp.where(a != 0.0, a, 1.0))
    cubic_b = -slope / (2.0 * xp.where(b != 0.0, b, 1.0))
    cubic = xp.where(a != 0.0, cubic_a, cubic_b)
    candidate = xp.where(has_prev, cubic, quadratic)
    candidate = xp.where(xp.isfinite(candidate), candidate, hi)
    step = xp.minimum(xp.maximum(candidate, lo), hi)
    return xp.where(finite, step, lo)


def backtracking(backend: Backend, phi: Callable[[Any], tuple[Any, Any]], x0: Any, f0: Any,
                 slope: Any, alpha0: Any, *, c1: float = 1e-4, max_trials: int = 30,
                 shrink: tuple[float, float] = (0.1, 0.5)) -> LineSearchResult:
    """Backtrack from alpha0 until the Armijo condition holds (see the module
    docstring). x0 is the point at alpha = 0, which the result holds when
    nothing was tried."""
    xp, put = backend.xp, backend.index_update
    f0, slope, alpha0 = xp.asarray(f0), xp.asarray(slope), xp.asarray(alpha0)
    delta = armijo_allowance(xp, f0)
    descent = xp.isfinite(slope) & (slope < 0.0)

    def trial(s: _Search) -> _Search:
        r = s.result
        f_alpha, point = phi(r.alpha)
        finite = xp.isfinite(f_alpha)
        accept = finite & (f_alpha <= f0 + c1 * r.alpha * slope + delta)
        trials = r.trials + 1
        status = xp.where(accept, ARMIJO, xp.where(trials >= max_trials, FAILED, SEARCHING))
        following = next_step(xp, f0, slope, r.alpha, f_alpha, s.alpha_prev, s.f_prev, shrink)
        return _Search(
            result=LineSearchResult(
                alpha=xp.where(status == SEARCHING, following, r.alpha), f=f_alpha, point=point,
                trials=trials, status=status, trace_alpha=put(r.trace_alpha, r.trials, r.alpha),
                trace_f=put(r.trace_f, r.trials, f_alpha)),
            # the cubic uses finite trials only
            alpha_prev=xp.where(finite, r.alpha, s.alpha_prev), f_prev=xp.where(finite, f_alpha, s.f_prev))

    start = _Search(
        result=LineSearchResult(
            alpha=alpha0, f=f0, point=x0, trials=xp.asarray(0),
            status=xp.where(descent, SEARCHING, NOT_DESCENT),
            trace_alpha=xp.zeros(max_trials), trace_f=xp.zeros(max_trials)),
        alpha_prev=xp.asarray(0.0), f_prev=f0)
    final: _Search = run_loop(backend, lambda s: s.result.status != SEARCHING, trial, start,
                              max_trials)
    return final.result
