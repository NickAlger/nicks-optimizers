"""Diagnostics of lbfgs: records, resumption state, and text.

The optimizer only produces records; formatting and printing are separate.

  LbfgsRecord     one per iteration: f, ||g||, the direction and its tests,
                  the line search, the curvature of the new pair and whether
                  it was stored, the transport's defect
  LbfgsSummary    returned at the end: stopping reason, all records, final
                  values, total work, restarts and skipped pairs, and the
                  final LbfgsResume
  LbfgsInfo       what callback(x, info) receives after each iteration
  LbfgsResume     everything carried between iterations besides x: gnorm0,
                  the iteration count, f (a consistency check), gamma and the
                  memory (the stored pairs, as the user's vectors and
                  covectors, with their pairings): save it with x in the
                  callback to resume a run exactly

format_lbfgs_header, format_lbfgs_row, format_lbfgs_summary turn records
into text (the line-search trials use newton_line_search_diagnostics.format_trials);
display_lbfgs_iteration prints them.
"""

from dataclasses import dataclass
from typing import Callable, Generic, Literal, Optional

import numpy as np

from .lbfgs_core import Base, Direction, Update
from .newton_line_search_diagnostics import format_trials
from .spaces import Array, CovectorT, VectorT

LbfgsReason = Literal["converged", "max_iter", "callback", "line_search"]


@dataclass(frozen=True, kw_only=True)
class LbfgsRecord:
    """What happened in one iteration, which started from the iterate x_k.

      iteration          k
      f, gnorm           f(x_k) and ||g(x_k)|| in the convergence norm
      gnorm_ratio        ||g(x_k)|| / gnorm0
      direction          "quasi-newton" (from the memory), "steepest" (empty
                         memory: -gamma M^{-1} g), or "fallback" (the
                         memory's direction failed the descent or angle test;
                         -gamma M^{-1} g instead, and the memory was cleared)
      base               "M" (the base gamma M^{-1}) or "P" (the preconditioner's
                         base gamma_P P^{-1}): the base of the direction taken
      pairs, gamma       the pairs in the memory and the scaling the
                         direction used
      step_norm          ||d||_M
      cosine             <-g, d> / (||g||_{M^{-1}} ||d||_M), the angle test's
                         value
      slope              <g, d>, negative
      alpha, trials      the step length taken (the last one tried when the
                         search failed) and the evaluations of f it cost
      actual             f(x_k) - f at the step taken
      accepted           whether the line search succeeded
      curvature          <y, s> / (alpha |<g, d>|) for the new pair: in a
                         flat space 1 - phi'(alpha) / phi'(0), so 1 when the
                         step reached the one-dimensional minimizer (nan if
                         the search failed)
      update             "stored", "skipped" (the curvature test failed) or
                         "-" (the search failed)
      dropped            pairs dropped by recomputed pairings
      transport_defect   <Tb g, T s> / <g, s> - 1: how far the transport is
                         from preserving pairings on this step (0 for the
                         identity)
      restart            the search failed and the memory was cleared for the
                         next iteration
      elapsed            seconds since the start of the run, at the end of
                         this iteration
      trace_alpha, trace_f   the steps tried and f there
    """
    iteration: int
    f: float
    gnorm: float
    gnorm_ratio: float
    direction: Direction
    base: Base
    pairs: int
    gamma: float
    step_norm: float
    cosine: float
    slope: float
    alpha: float
    trials: int
    actual: float
    accepted: bool
    curvature: float
    update: Update
    dropped: int
    transport_defect: float
    restart: bool
    elapsed: float
    trace_alpha: tuple[float, ...]
    trace_f: tuple[float, ...]


@dataclass(frozen=True, kw_only=True)
class LbfgsResume(Generic[VectorT, CovectorT]):
    """Everything, besides the iterate x, that lbfgs carries from one
    iteration to the next: with x, the same options and deterministic
    functions, lbfgs(..., x, resume=state) continues exactly as the original
    run would have.

      gnorm0          the reference gradient norm (rtol test)
      next_iteration  the number of the next iteration (max_iter stays a
                      budget for the whole run)
      f               f(x) at the checkpoint, to detect a state that does not
                      belong to the x it is resumed from
      gamma           the scaling of the base
      pairs           the stored pairs (s_i, y_i) at x, oldest first
      pairings        pairings[j][i] = <y_j, s_i> as stored (same order)
      floors          the curvature threshold each pair passed when stored
      slots           the slot of each pair in the memory's blocks (kept on
                      resuming, so the block operations add in the same order)
      metric_pairs    with fixed_metric, (M^{-1} y_i, M s_i) for each pair
                      (empty otherwise; when resuming with fixed_metric
                      without them, they are recomputed: one solve and one
                      product per pair)
      gamma_P         the scaling of the preconditioner's base
      precond_vectors with fixed_preconditioner, P^{-1} y_i for each pair
                      (empty otherwise; recomputed when missing)

    The pairs are the user's vectors and covectors: save them like x and g.
    """
    gnorm0: float
    next_iteration: int
    f: float
    gamma: float
    pairs: tuple[tuple[VectorT, CovectorT], ...]
    pairings: tuple[tuple[float, ...], ...]
    floors: tuple[float, ...]
    slots: tuple[int, ...]
    metric_pairs: tuple[tuple[VectorT, CovectorT], ...] = ()
    gamma_P: Optional[float] = None
    precond_vectors: tuple[VectorT, ...] = ()


@dataclass(frozen=True, kw_only=True)
class LbfgsInfo(Generic[VectorT, CovectorT]):
    """What the callback receives besides the iterate x after an iteration:
    the record of the iteration just finished, f and g at x, the step taken
    (alpha d at the previous point; the last trial's when the search failed)
    and the LbfgsResume at x."""
    record: LbfgsRecord
    f: float
    g: CovectorT
    step: VectorT
    resume: LbfgsResume[VectorT, CovectorT]


@dataclass(frozen=True, kw_only=True)
class LbfgsSummary(Generic[VectorT, CovectorT]):
    """The outcome of a run.

      reason        "converged" (||g|| <= max(gtol, rtol gnorm0)),
                    "max_iter", "callback", or "line_search" (the line search
                    failed with an empty memory; x is the last accepted
                    iterate)
      iterations    one LbfgsRecord per iteration
      f, gnorm      at the returned point; gnorm0 is the reference norm
      *_calls       work over the whole run: calls of fun (including every
                    line-search trial), grad, M^{-1}, M, P^{-1}, and of the
                    transport's maps (single vectors, single covectors,
                    blocks)
      precond_directions   iterations whose direction came from the
                    preconditioner's base
      trials        line-search evaluations of f over the run
      restarts      memories cleared after a failed line search
      skipped       pairs not stored (curvature test)
      dropped       pairs dropped by recomputed pairings
      elapsed       seconds
      resume        the LbfgsResume at the returned point
    """
    reason: LbfgsReason
    iterations: tuple[LbfgsRecord, ...]
    f: float
    gnorm: float
    gnorm0: float
    fun_calls: int
    grad_calls: int
    metric_solve_calls: int
    metric_apply_calls: int
    vector_transport_calls: int
    covector_transport_calls: int
    block_transport_calls: int
    precond_solve_calls: int
    precond_directions: int
    trials: int
    restarts: int
    skipped: int
    dropped: int
    elapsed: float
    resume: LbfgsResume[VectorT, CovectorT]


def merge_lbfgs_summaries(first: LbfgsSummary[VectorT, CovectorT],
                          second: LbfgsSummary[VectorT, CovectorT]) -> LbfgsSummary[VectorT, CovectorT]:
    """Combine the summaries of a run that was stopped and then resumed, where
    second started from first's final point and state."""
    return LbfgsSummary(
        reason=second.reason, iterations=first.iterations + second.iterations, f=second.f,
        gnorm=second.gnorm, gnorm0=first.gnorm0, fun_calls=first.fun_calls + second.fun_calls,
        grad_calls=first.grad_calls + second.grad_calls,
        metric_solve_calls=first.metric_solve_calls + second.metric_solve_calls,
        metric_apply_calls=first.metric_apply_calls + second.metric_apply_calls,
        vector_transport_calls=first.vector_transport_calls + second.vector_transport_calls,
        covector_transport_calls=first.covector_transport_calls + second.covector_transport_calls,
        block_transport_calls=first.block_transport_calls + second.block_transport_calls,
        precond_solve_calls=first.precond_solve_calls + second.precond_solve_calls,
        precond_directions=first.precond_directions + second.precond_directions,
        trials=first.trials + second.trials, restarts=first.restarts + second.restarts,
        skipped=first.skipped + second.skipped, dropped=first.dropped + second.dropped,
        elapsed=first.elapsed + second.elapsed, resume=second.resume)


def lbfgs_history_array(summary: LbfgsSummary[VectorT, CovectorT]) -> Array:
    """The iterations as a numeric array, one row per iteration, with columns
    (iteration, f, ||g||, alpha, trials, pairs, gamma, step_norm, curvature)."""
    rows = [(r.iteration, r.f, r.gnorm, r.alpha, r.trials, r.pairs, r.gamma, r.step_norm, r.curvature)
            for r in summary.iterations]
    history: Array = np.array(rows, dtype=float).reshape(-1, 9)
    return history


_ROW = ("{it:>4}  {f:>13}  {g:>8}  {gr:>8}  {mem:>3}  {dir:>12}  {base:>4}  {d:>8}  {cos:>7}  "
        "{slope:>9}  {alpha:>8}  {ls:>3}  {gamma:>8}  {curv:>6}  {tdef:>8}  {upd:>7}")

_REASONS: dict[str, str] = {
    "converged": "converged, ||g|| <= max(gtol, rtol * gnorm0)",
    "max_iter": "reached max_iter",
    "callback": "stopped by the callback",
    "line_search": "the line search found no acceptable step from an empty memory",
}


def format_lbfgs_header() -> str:
    """Column headings for format_lbfgs_row."""
    return _ROW.format(it="it", f="f", g="|g|", gr="|g|/|g0|", mem="mem", dir="direction",
                       base="base", d="|d|", cos="cos", slope="slope", alpha="alpha", ls="ls",
                       gamma="gamma", curv="curv", tdef="tdef", upd="update")


def format_lbfgs_row(record: LbfgsRecord) -> str:
    """One line per iteration: mem, direction and base describe the
    direction (pairs used, its kind, the base); |d|, cos and slope its
    M-norm, the angle test's cosine and <g, d>; alpha and ls the step length
    and the evaluations of f ("!" when the search failed, "r" when the memory
    was then cleared); gamma the base's scaling; curv the new pair's
    <y, s> / (alpha |<g, d>|); tdef the transport defect; update whether the
    pair was stored."""
    ls = str(record.trials) + ("" if record.accepted else ("r" if record.restart else "!"))
    curv = f"{record.curvature:.2f}" if np.isfinite(record.curvature) else "-"
    tdef = f"{record.transport_defect:.1e}" if np.isfinite(record.transport_defect) else "-"
    upd = record.update + (f" -{record.dropped}" if record.dropped else "")
    return _ROW.format(
        it=record.iteration, f=f"{record.f:.6e}", g=f"{record.gnorm:.1e}",
        gr=f"{record.gnorm_ratio:.1e}", mem=record.pairs, dir=record.direction, base=record.base,
        d=f"{record.step_norm:.2e}", cos=f"{record.cosine:.1e}", slope=f"{record.slope:.2e}",
        alpha=f"{record.alpha:.2e}", ls=ls, gamma=f"{record.gamma:.2e}", curv=curv, tdef=tdef, upd=upd)


def format_lbfgs_summary(summary: LbfgsSummary[VectorT, CovectorT]) -> str:
    """The end-of-run report: stopping reason, final values and total work."""
    ratio = summary.gnorm / summary.gnorm0 if summary.gnorm0 > 0.0 else 0.0
    transports = summary.vector_transport_calls + summary.covector_transport_calls
    lines = [
        f"Stopped: {_REASONS[summary.reason]}",
        f"  iterations {len(summary.iterations)}, {summary.trials} line-search trials, "
        f"{summary.restarts} restarts, {summary.skipped} pairs skipped, {summary.elapsed:.2f} s",
        f"  f = {summary.f:.6e}   |g| = {summary.gnorm:.2e}   |g|/|g0| = {ratio:.2e}",
        f"  work: {summary.fun_calls} objective and {summary.grad_calls} gradient evaluations, "
        f"{summary.metric_solve_calls} metric solves, {summary.metric_apply_calls} metric products"
        + (f", {summary.precond_solve_calls} preconditioner solves" if summary.precond_solve_calls else ""),
    ]
    if summary.precond_solve_calls:
        lines.append(f"  preconditioner: its direction taken in {summary.precond_directions} of "
                     f"{len(summary.iterations)} iterations")
    if transports or summary.block_transport_calls:
        lines.append(f"  transports: {summary.vector_transport_calls} vectors, "
                     f"{summary.covector_transport_calls} covectors, "
                     f"{summary.block_transport_calls} blocks"
                     + (f", {summary.dropped} pairs dropped" if summary.dropped else ""))
    return "\n".join(lines)


def display_lbfgs_iteration(record: LbfgsRecord, level: int, print_fn: Callable[[str], None],
                            header_every: int = 20) -> None:
    """Print one iteration: the header every header_every rows, the row, and
    at level 2 the line-search trials."""
    if record.iteration % header_every == 0:
        print_fn(format_lbfgs_header())
    print_fn(format_lbfgs_row(record))
    if level >= 2:
        print_fn(format_trials(record.f, record.slope, record.accepted, record.trace_alpha,
                               record.trace_f))
