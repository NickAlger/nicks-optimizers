"""Diagnostics of line_search_newton: records, resumption state, and text.

The optimizer only produces records; formatting and printing are separate.

  SubproblemStats      (krylov_solvers) one per direction computation
  LineSearchRecord     one per iteration: f, ||g||, the direction, the step
                       length and the trials of the line search, predicted
                       and actual decrease, and its SubproblemStats
  LineSearchSummary    returned at the end: stopping reason, all records,
                       final values, total work, and the final
                       LineSearchResume
  LineSearchInfo       what callback(x, info) receives after each iteration
  LineSearchResume     the state carried between iterations besides x
                       (gnorm0, the iteration count, and f for a consistency
                       check): save it with x in the callback to resume a run
                       exactly with resume=state (and g0=g to skip one
                       gradient)

format_ls_header, format_ls_row, format_ls_krylov_trace,
format_ls_trials and format_ls_summary turn records into text;
display_ls_iteration prints them.
"""

from dataclasses import dataclass
from typing import Callable, Generic, Literal

import numpy as np

from .krylov_solvers import SubproblemStats
from .newton_line_search_core import Direction
from .spaces import Array, CovectorT, VectorT

LineSearchReason = Literal["converged", "max_iter", "callback", "line_search"]


@dataclass(frozen=True, kw_only=True)
class LineSearchRecord:
    """What happened in one iteration, which started from the iterate x_k.

      iteration          k
      f, gnorm           f(x_k) and ||g(x_k)|| in the convergence norm
      gnorm_ratio        ||g(x_k)|| / gnorm0
      forcing            relative Krylov tolerance used
                         (resid_tol = forcing * ||g(x_k)||_{M^{-1}})
      direction          "newton" (the solve met its tolerance, or its step
                         limit), "truncated" (stopped at nonpositive
                         curvature after the first step: the last direction
                         with positive curvature), "steepest" (nonpositive
                         curvature at the first step: the preconditioned
                         steepest descent direction), or "fallback" (the
                         solver's direction was not a descent direction, from
                         rounding or non-finite values, and the preconditioned
                         steepest descent direction was used instead)
      step_norm          ||d||_M
      slope              <g(x_k), d>, negative
      alpha, trials      the step length taken (the last one tried when the
                         search failed) and the evaluations of f it cost
      predicted          the decrease of the quadratic model along alpha d
                         (nan for a fallback direction)
      actual             f(x_k) - f(retract(x_k, alpha d))
      accepted           whether the line search succeeded (x_{k+1} =
                         retract(x_k, alpha d)); if not, the run stops
      elapsed            seconds since the start of the run, at the end of
                         this iteration
      subproblem         SubproblemStats of the direction computation (lam = 0)
      trace_alpha, trace_f   the steps tried and f there
    """
    iteration: int
    f: float
    gnorm: float
    gnorm_ratio: float
    forcing: float
    direction: Direction
    step_norm: float
    slope: float
    alpha: float
    trials: int
    predicted: float
    actual: float
    accepted: bool
    elapsed: float
    subproblem: SubproblemStats
    trace_alpha: tuple[float, ...]
    trace_f: tuple[float, ...]


@dataclass(frozen=True, kw_only=True)
class LineSearchResume:
    """Everything, besides the iterate x, that line_search_newton carries
    from one iteration to the next: with x, the same options and
    deterministic functions, line_search_newton(..., x, resume=state)
    continues exactly as the original run would have.

      gnorm0          the reference gradient norm (rtol test, forcing term)
      next_iteration  the number of the next iteration (max_iter stays a
                      budget for the whole run)
      f               f(x) at the checkpoint, to detect a state that does not
                      belong to the x it is resumed from

    Plain numbers only, so it round-trips through JSON exactly.
    """
    gnorm0: float
    next_iteration: int
    f: float


@dataclass(frozen=True, kw_only=True)
class LineSearchInfo(Generic[VectorT, CovectorT]):
    """What the callback receives besides the iterate x after an iteration:
    the record of the iteration just finished, f and g at x, the step taken
    (alpha d; the last trial's when the search failed) and the
    LineSearchResume at x."""
    record: LineSearchRecord
    f: float
    g: CovectorT
    step: VectorT
    resume: LineSearchResume


@dataclass(frozen=True, kw_only=True)
class LineSearchSummary:
    """The outcome of a run.

      reason        "converged" (||g|| <= max(gtol, rtol gnorm0)),
                    "max_iter", "callback", or "line_search" (the line search
                    failed: no step satisfied the Armijo condition within
                    max_backtracks trials; x is the last accepted iterate)
      iterations    one LineSearchRecord per iteration
      f, gnorm      at the returned point; gnorm0 is the reference norm
      *_calls       work over the whole run: calls of fun (including every
                    line-search trial), grad, and each linear operator
      trials        line-search evaluations of f over the run
      elapsed       seconds
      resume        the LineSearchResume at the returned point
      capacity_growths  how often a Krylov basis doubled its capacity
    """
    reason: LineSearchReason
    iterations: tuple[LineSearchRecord, ...]
    f: float
    gnorm: float
    gnorm0: float
    fun_calls: int
    grad_calls: int
    hessvec_calls: int
    metric_solve_calls: int
    precond_solve_calls: int
    metric_apply_calls: int
    trials: int
    elapsed: float
    resume: LineSearchResume
    capacity_growths: int = 0


def merge_ls_summaries(first: LineSearchSummary, second: LineSearchSummary) -> LineSearchSummary:
    """Combine the summaries of a run that was stopped and then resumed, where
    second started from first's final point and state."""
    return LineSearchSummary(
        reason=second.reason, iterations=first.iterations + second.iterations,
        f=second.f, gnorm=second.gnorm, gnorm0=first.gnorm0,
        fun_calls=first.fun_calls + second.fun_calls,
        grad_calls=first.grad_calls + second.grad_calls,
        hessvec_calls=first.hessvec_calls + second.hessvec_calls,
        metric_solve_calls=first.metric_solve_calls + second.metric_solve_calls,
        precond_solve_calls=first.precond_solve_calls + second.precond_solve_calls,
        metric_apply_calls=first.metric_apply_calls + second.metric_apply_calls,
        trials=first.trials + second.trials, elapsed=first.elapsed + second.elapsed,
        resume=second.resume, capacity_growths=first.capacity_growths + second.capacity_growths)


def ls_history_array(summary: LineSearchSummary) -> Array:
    """The iterations as a numeric array, one row per iteration, with columns
    (iteration, f, ||g||, alpha, trials, krylov_steps, n_negative, step_norm)."""
    rows = [(r.iteration, r.f, r.gnorm, r.alpha, r.trials, r.subproblem.krylov_steps,
             r.subproblem.n_negative, r.step_norm) for r in summary.iterations]
    history: Array = np.array(rows, dtype=float).reshape(-1, 8)
    return history


_ROW = ("{it:>4}  {f:>13}  {g:>8}  {gr:>8}  {eta:>7}  {k:>3}  {stop:>8}  {neg:>3}  "
        "{dir:>9}  {d:>8}  {slope:>9}  {alpha:>8}  {ls:>3}  {ratio:>8}")

_REASONS: dict[str, str] = {
    "converged": "converged, ||g|| <= max(gtol, rtol * gnorm0)",
    "max_iter": "reached max_iter",
    "callback": "stopped by the callback",
    "line_search": "the line search found no acceptable step",
}


def format_ls_header() -> str:
    """Column headings for format_ls_row."""
    return _ROW.format(it="it", f="f", g="|g|", gr="|g|/|g0|", eta="eta", k="k", stop="stop",
                       neg="neg", dir="direction", d="|d|", slope="slope", alpha="alpha",
                       ls="ls", ratio="act/pred")


def format_ls_row(record: LineSearchRecord) -> str:
    """One line per iteration: k, stop and neg describe the Krylov solve; |d|
    and slope the direction (||d||_M, <g, d>); alpha and ls the step length
    and the evaluations of f; act/pred the actual over the predicted
    decrease at the step taken (near 1 when the quadratic model is good)."""
    s = record.subproblem
    ratio = (record.actual / record.predicted
             if np.isfinite(record.predicted) and record.predicted > 0.0 else float("nan"))
    return _ROW.format(
        it=record.iteration, f=f"{record.f:.6e}", g=f"{record.gnorm:.1e}",
        gr=f"{record.gnorm_ratio:.1e}", eta=f"{record.forcing:.1e}", k=s.krylov_steps,
        stop=s.stop_reason, neg=s.n_negative, dir=record.direction, d=f"{record.step_norm:.2e}",
        slope=f"{record.slope:.2e}", alpha=f"{record.alpha:.2e}",
        ls=record.trials if record.accepted else f"{record.trials}!", ratio=f"{ratio:.3f}")


def format_ls_krylov_trace(stats: SubproblemStats) -> str:
    """The per-step history of one direction solve, indented under its row."""
    pad = " " * 8
    lines = [f"{pad}{'step':>4}  {'model':>11}  {'resid':>9}  {'neg':>3}"]
    for j, (m, r, neg) in enumerate(zip(stats.trace_model, stats.trace_resid,
                                        stats.trace_negative), start=1):
        lines.append(f"{pad}{j:>4}  {m:11.4e}  {r:9.2e}  {neg:3d}")
    smallest = ("smallest curvature <Hp, p> / ||p||_M^2" if stats.krylov == "cg"
                else "smallest M-Ritz value")
    last = (f"{pad}stop: {stats.stop_reason} ({stats.krylov}, resid tol {stats.resid_tol:.2e}), "
            f"{smallest} {stats.theta_min:.2e}")
    if stats.true_model_value != stats.model_value:
        last += f", true-model value {stats.true_model_value:.4e}"
    if stats.fallbacks:
        last += f", {stats.fallbacks} preconditioner fallbacks"
    lines.append(last)
    return "\n".join(lines)


def format_trials(f0: float, slope: float, accepted: bool, trace_alpha: tuple[float, ...],
                  trace_f: tuple[float, ...]) -> str:
    """The trials of a line search from f0 with the slope <g, d>, indented
    under an iteration's row: the step, f there, and the Armijo ratio
    (f - f0) / (alpha slope), at least c1 for an accepted step (up to the
    rounding allowance); the last trial is marked if it was accepted."""
    pad = " " * 8
    lines = [f"{pad}{'trial':>5}  {'alpha':>9}  {'f':>13}  {'(f-f0)/(alpha slope)':>20}"]
    for j, (a, f) in enumerate(zip(trace_alpha, trace_f), start=1):
        armijo = (f - f0) / (a * slope) if a * slope != 0.0 else float("nan")
        mark = "  accepted" if accepted and j == len(trace_alpha) else ""
        lines.append(f"{pad}{j:>5}  {a:9.2e}  {f:13.6e}  {armijo:20.3f}{mark}")
    return "\n".join(lines)


def format_ls_trials(record: LineSearchRecord) -> str:
    """The trials of the line search of an iteration (see format_trials)."""
    return format_trials(record.f, record.slope, record.accepted, record.trace_alpha, record.trace_f)


def format_ls_summary(summary: LineSearchSummary) -> str:
    """The end-of-run report: stopping reason, final values and total work."""
    ratio = summary.gnorm / summary.gnorm0 if summary.gnorm0 > 0.0 else 0.0
    return "\n".join([
        f"Stopped: {_REASONS[summary.reason]}",
        f"  iterations {len(summary.iterations)}, {summary.trials} line-search trials, "
        f"{summary.elapsed:.2f} s",
        f"  f = {summary.f:.6e}   |g| = {summary.gnorm:.2e}   |g|/|g0| = {ratio:.2e}",
        f"  work: {summary.hessvec_calls} Hessian-vector products, "
        f"{summary.metric_solve_calls} metric solves, {summary.precond_solve_calls} "
        f"preconditioner solves, {summary.metric_apply_calls} metric products, "
        f"{summary.fun_calls} objective and {summary.grad_calls} gradient evaluations",
    ])


def display_ls_iteration(record: LineSearchRecord, level: int, print_fn: Callable[[str], None],
                         header_every: int = 20) -> None:
    """Print one iteration: the header every header_every rows, the row, and
    at level 2 the Krylov trace and the line-search trials."""
    if record.iteration % header_every == 0:
        print_fn(format_ls_header())
    print_fn(format_ls_row(record))
    if level >= 2:
        print_fn(format_ls_krylov_trace(record.subproblem))
        print_fn(format_ls_trials(record))
