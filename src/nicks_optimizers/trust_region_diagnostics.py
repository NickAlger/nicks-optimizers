"""Diagnostics of trust_region_newton: records, resumption state, and text.

The optimizer only produces records; formatting and printing are separate.

  SubproblemStats   (trust_region_subproblem) one per step computation
  IterationRecord   one per outer iteration: f, ||g||, radius, predicted and
                    actual reduction, rho, acceptance, and its SubproblemStats
  RunSummary        returned at the end: stopping reason, all IterationRecords,
                    final values, total work, and the final ResumeState
  IterationInfo     what callback(x, info) receives after each iteration:
                    the IterationRecord plus f, g, the step s and the
                    ResumeState
  ResumeState       the state carried between iterations besides x (radius,
                    gnorm0, iteration count, and f for a consistency check):
                    save it with x in the callback to resume a run exactly
                    with resume=state (and g0=g to skip one gradient)

Records hold only numbers and tuples, so keeping them all is cheap; vectors
go only to the callback. format_header, format_row, format_lanczos_trace and
format_summary turn records into text; display_iteration prints them.
"""

from dataclasses import dataclass
from typing import Callable, Generic, Literal

import numpy as np

from .spaces import Array, CovectorT, VectorT
from .trust_region_subproblem import SubproblemStats


RunReason = Literal["converged", "radius", "max_iter", "callback"]


@dataclass(frozen=True, kw_only=True)
class IterationRecord:
    """What happened in one outer iteration, which started from the iterate x_k.

      iteration          k
      f, gnorm           f(x_k) and ||g(x_k)||_{M^{-1}}
      gnorm_ratio        ||g(x_k)|| / gnorm0
      radius, new_radius trust-region radius used for this step, and after the update
      forcing            relative Krylov tolerance used (resid_tol = forcing * gnorm)
      predicted          model decrease m(0) - m(s) for the model that was minimized
      actual             f(x_k) - f(retract(x_k, s))
      rho                reduction ratio (with the rounding safeguard)
      accepted           whether x_{k+1} = retract(x_k, s)
      elapsed            seconds since the start of the run, at the end of this iteration
      subproblem         SubproblemStats of the step computation
    """
    iteration: int
    f: float
    gnorm: float
    gnorm_ratio: float
    radius: float
    new_radius: float
    forcing: float
    predicted: float
    actual: float
    rho: float
    accepted: bool
    elapsed: float
    subproblem: SubproblemStats


@dataclass(frozen=True, kw_only=True)
class ResumeState:
    """Everything, besides the iterate x, that the method carries from one
    iteration to the next. With x, the same options and deterministic
    functions, trust_region_newton(..., x, resume=state) continues exactly as
    the original run would have.

      radius          the trust-region radius for the next iteration
      gnorm0          the reference gradient norm (rtol test, forcing term)
      next_iteration  the number of the next iteration; max_iter stays a
                      budget for the whole run, and records keep their numbering
      f               f(x) at the checkpoint, used to detect a state that does
                      not belong to the x it is resumed from

    Plain numbers only, so dataclasses.asdict / ResumeState(**d) round-trip
    through JSON or any other format exactly. P, M, f and g are recomputed
    from x, and the Lanczos basis is rebuilt every iteration, so nothing else
    needs saving; pass g0 to skip recomputing the gradient.
    """
    radius: float
    gnorm0: float
    next_iteration: int
    f: float


@dataclass(frozen=True, kw_only=True)
class IterationInfo(Generic[VectorT, CovectorT]):
    """What the callback receives besides the iterate x after an iteration.

      record   the IterationRecord of the iteration just finished (the same
               object that is kept in the RunSummary)
      f, g     objective and gradient at x, already computed by the method
      step     the step s that was tried in this iteration, accepted or not
      resume   the ResumeState at x: save it with x (and g, to pass as g0) to
               be able to continue the run later

    x, f and g describe the point the next iteration starts from; record
    describes the iteration that led there. The method never modifies objects
    in place, so a callback may keep references without copying.
    """
    record: IterationRecord
    f: float
    g: CovectorT
    step: VectorT
    resume: ResumeState


@dataclass(frozen=True, kw_only=True)
class RunSummary:
    """The outcome of a run.

      reason        "converged" (||g|| <= max(gtol, rtol gnorm0)), "radius"
                    (radius below radius_min), "max_iter", or "callback"
      iterations    one IterationRecord per iteration
      f, gnorm      at the returned point; gnorm0 is the reference norm
      accepted_steps
      *_calls       work done over the whole run
      elapsed       seconds
      resume        the ResumeState at the returned point, for continuing the
                    run (e.g. after max_iter) with resume=summary.resume
    """
    reason: RunReason
    iterations: tuple[IterationRecord, ...]
    f: float
    gnorm: float
    gnorm0: float
    accepted_steps: int
    fun_calls: int
    grad_calls: int
    hessvec_calls: int
    precond_calls: int
    inner_calls: int
    dual_inner_calls: int
    elapsed: float
    resume: ResumeState


def merge_summaries(first: RunSummary, second: RunSummary) -> RunSummary:
    """Combine the summaries of a run that was stopped and then resumed, where
    second started from first's final point and state. Records are joined,
    work and time are added, and the outcome is second's. The work includes
    the restart's extra objective (and gradient, without g0) evaluation."""
    return RunSummary(
        reason=second.reason, iterations=first.iterations + second.iterations,
        f=second.f, gnorm=second.gnorm, gnorm0=first.gnorm0,
        accepted_steps=first.accepted_steps + second.accepted_steps,
        fun_calls=first.fun_calls + second.fun_calls,
        grad_calls=first.grad_calls + second.grad_calls,
        hessvec_calls=first.hessvec_calls + second.hessvec_calls,
        precond_calls=first.precond_calls + second.precond_calls,
        inner_calls=first.inner_calls + second.inner_calls,
        dual_inner_calls=first.dual_inner_calls + second.dual_inner_calls,
        elapsed=first.elapsed + second.elapsed, resume=second.resume)


def history_array(summary: RunSummary) -> Array:
    """The iterations as a numeric array, one row per iteration, with columns
    (iteration, f, ||g||, radius, rho, lanczos_steps, accepted, n_negative)."""
    rows = [(r.iteration, r.f, r.gnorm, r.radius, r.rho, r.subproblem.lanczos_steps,
             float(r.accepted), r.subproblem.n_negative) for r in summary.iterations]
    history: Array = np.array(rows, dtype=float).reshape(-1, 8)
    return history


def build_summary(reason: RunReason, records: list[IterationRecord], f: float, gnorm: float,
                  gnorm0: float, fun_calls: int, grad_calls: int, outer_dual_calls: int,
                  elapsed: float, resume: ResumeState) -> RunSummary:
    """Assemble the RunSummary, adding up the work recorded per iteration."""
    subs = [r.subproblem for r in records]
    return RunSummary(
        reason=reason, iterations=tuple(records), f=f, gnorm=gnorm, gnorm0=gnorm0,
        accepted_steps=sum(r.accepted for r in records),
        fun_calls=fun_calls, grad_calls=grad_calls,
        hessvec_calls=sum(s.hessvec_calls for s in subs),
        precond_calls=sum(s.precond_calls for s in subs),
        inner_calls=sum(s.inner_calls for s in subs),
        dual_inner_calls=outer_dual_calls + sum(s.dual_inner_calls for s in subs),
        elapsed=elapsed, resume=resume)


_ROW = ("{it:>4}  {f:>13}  {g:>8}  {gr:>8}  {r:>8}  {sr:>5}  {pred:>9}  {ared:>9}  "
        "{rho:>7}  {acc:>3}  {k:>3}  {stop:>9}  {neg:>3}")


_RUN_REASONS: dict[str, str] = {
    "converged": "converged, ||g|| <= max(gtol, rtol * gnorm0)",
    "radius": "trust-region radius fell below radius_min",
    "max_iter": "reached max_iter",
    "callback": "stopped by the callback",
}


def format_header() -> str:
    """Column headings for format_row."""
    return _ROW.format(it="it", f="f", g="|g|", gr="|g|/|g0|", r="radius", sr="|s|/r",
                       pred="pred", ared="actual", rho="rho", acc="acc", k="k",
                       stop="stop", neg="neg")


def format_row(record: IterationRecord) -> str:
    """One line per iteration. |s|/r = 1.00 means the step reached the
    trust-region boundary; pred and actual are the model and true decreases;
    k and stop describe the Lanczos solve; neg counts negative M-Ritz values."""
    s = record.subproblem
    return _ROW.format(
        it=record.iteration, f=f"{record.f:.6e}", g=f"{record.gnorm:.1e}",
        gr=f"{record.gnorm_ratio:.1e}", r=f"{record.radius:.2e}",
        sr=f"{s.step_norm / record.radius:.2f}", pred=f"{record.predicted:.2e}",
        ared=f"{record.actual:.2e}", rho=f"{record.rho:+.3f}",
        acc="yes" if record.accepted else "no", k=s.lanczos_steps, stop=s.stop_reason,
        neg=s.n_negative)


def format_lanczos_trace(stats: SubproblemStats) -> str:
    """The per-step history of one subproblem solve, indented under its row."""
    pad = " " * 8
    lines = [f"{pad}{'step':>4}  {'model':>11}  {'resid':>9}  {'lam':>9}  {'neg':>3}"]
    for j, (m, r, lam, neg) in enumerate(zip(stats.trace_model, stats.trace_resid,
                                             stats.trace_lam, stats.trace_negative), start=1):
        lines.append(f"{pad}{j:>4}  {m:11.4e}  {r:9.2e}  {lam:9.2e}  {neg:3d}")
    last = (f"{pad}stop: {stats.stop_reason} (resid tol {stats.resid_tol:.2e}), "
            f"smallest M-Ritz value {stats.theta_min:.2e}")
    if stats.true_model_value != stats.model_value:
        last += f", true-model value {stats.true_model_value:.4e}"
    lines.append(last)
    return "\n".join(lines)


def format_summary(summary: RunSummary) -> str:
    """The end-of-run report: stopping reason, final values and total work."""
    ratio = summary.gnorm / summary.gnorm0 if summary.gnorm0 > 0.0 else 0.0
    return "\n".join([
        f"Stopped: {_RUN_REASONS[summary.reason]}",
        f"  iterations {len(summary.iterations)} ({summary.accepted_steps} accepted), "
        f"{summary.elapsed:.2f} s",
        f"  f = {summary.f:.6e}   |g| = {summary.gnorm:.2e}   |g|/|g0| = {ratio:.2e}",
        f"  work: {summary.hessvec_calls} Hessian-vector products, "
        f"{summary.precond_calls} preconditioner solves, {summary.inner_calls} ball_inner "
        f"and {summary.dual_inner_calls} ball_dual_inner calls, {summary.fun_calls} objective "
        f"and {summary.grad_calls} gradient evaluations",
    ])


def display_iteration(record: IterationRecord, level: int, print_fn: Callable[[str], None],
                      header_every: int = 20) -> None:
    """Print one iteration: the header every header_every rows, the row, and
    at level 2 the Lanczos trace."""
    if record.iteration % header_every == 0:
        print_fn(format_header())
    print_fn(format_row(record))
    if level >= 2:
        print_fn(format_lanczos_trace(record.subproblem))
