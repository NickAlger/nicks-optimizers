"""lbfgs: limited-memory BFGS in the geometry of a metric M (see
dev/lbfgs_plan.md).

At each iterate x the direction is d = -H g, where H, the inverse Hessian
approximation, is built from the base gamma M^{-1} (so B0 = M / gamma) and
the last m pairs (s_i, y_i): the steps s_i are vectors and the gradient
changes y_i covectors, and they enter only through pairings <y, s>. Then a
backtracking line search along retract(x, alpha d) from alpha = 1 (Armijo,
line_search.backtracking), and at the accepted point the memory is carried
there by the vector transport, the new pair

    s = T(alpha d),   y = g(x_new) - Tb(g(x))

is stored if it passes the curvature test, and gamma is refitted to it.
Everything geometric is measured with M: the gradient norm in the stopping
test ||g||_{M^{-1}}, step lengths ||d||_M, the angle test between -g and d,
and the scaling fits. M is given by both of its operators, metric_solve
(M^{-1}) and metric_apply (M).

The update itself is affine covariant (it uses pairings only), so with a
constant metric M = L L^T the iterates are those of textbook L-BFGS on
f(L^{-T} z), mapped back. The transport carries the memory between tangent
spaces; on a flat space it is the identity, even when M changes from point
to point (the pairs are data about the Hessian, whatever M is).
"""

import time
from functools import partial
from typing import Any, Callable, Optional, SupportsFloat

import numpy as np

from .backend import Backend, compile_function, numpy_backend
from .criteria import gradient_converged
from .lbfgs_core import (
    CURVATURE_TESTS,
    CURVATURE_TOL,
    DIRECTIONS,
    PAIRINGS,
    SCALINGS,
    UPDATES,
    CurvatureTest,
    LbfgsProblem,
    LbfgsState,
    Pairings,
    Scaling,
    init,
    lbfgs_step,
    restart,
)
from .lbfgs_diagnostics import (
    LbfgsInfo,
    LbfgsReason,
    LbfgsRecord,
    LbfgsResume,
    LbfgsSummary,
    display_lbfgs_iteration,
    format_lbfgs_summary,
)
from .lbfgs_memory import memory_from_pairs
from .line_search import ARMIJO
from .newton_core import PointOperators
from .newton_driver import check_resumed_f, number
from .spaces import CovectorBlockT, CovectorT, PointT, VectorBlockT, VectorSpace, VectorT, numpy_space
from .transport import VectorTransport, identity_transport


def _check(*, space: Any, retract: Any, transport: Any, metric_solve: Any, metric_apply: Any,
           operators_at: Any, point_data: Any, memory: int, scaling: str, gamma0: float,
           initial_step: Optional[float], pairings: str, curvature_test: str,
           curvature_tol: Optional[float], angle_tol: float, c1: float, max_backtracks: int,
           step_max: Optional[float]) -> None:
    """Raise ValueError for an inconsistent combination of arguments."""
    if operators_at is not None and (metric_solve is not None or metric_apply is not None):
        raise ValueError("operators_at gives the operators at a point: pass no metric_solve or "
                         "metric_apply")
    if point_data is not None and operators_at is None:
        raise ValueError("point_data is passed to operators_at: give operators_at too")
    if operators_at is None and (metric_solve is None) != (metric_apply is None):
        raise ValueError("lbfgs needs both metric_solve (M^{-1}) and metric_apply (M)")
    if space is not None:
        if retract is None:
            raise ValueError("with a space, pass retract (flat_retraction(space) gives x + s)")
        if transport is None:
            raise ValueError("with a space, pass transport (identity_transport for a flat space)")
        if operators_at is None and metric_solve is None:
            raise ValueError("with a space, pass metric_solve and metric_apply "
                             "(euclidean_metric_solve and euclidean_metric_apply give M = I)")
    if memory < 1:
        raise ValueError(f"memory must be positive, not {memory}")
    if scaling not in SCALINGS:
        raise ValueError(f"scaling must be one of {SCALINGS}, not {scaling!r}")
    if pairings not in PAIRINGS:
        raise ValueError(f"pairings must be one of {PAIRINGS}, not {pairings!r}")
    if curvature_test not in CURVATURE_TESTS:
        raise ValueError(f"curvature_test must be one of {CURVATURE_TESTS}, not {curvature_test!r}")
    if curvature_tol is not None and not curvature_tol >= 0.0:
        raise ValueError(f"curvature_tol must be nonnegative, not {curvature_tol}")
    if not 0.0 <= angle_tol < 1.0:
        raise ValueError(f"angle_tol must lie in [0, 1), not {angle_tol}")
    if not gamma0 > 0.0:
        raise ValueError(f"gamma0 must be positive, not {gamma0}")
    if initial_step is not None and not initial_step > 0.0:
        raise ValueError(f"initial_step must be positive (or None), not {initial_step}")
    if not 0.0 < c1 < 1.0:
        raise ValueError(f"c1 must lie in (0, 1), not {c1}")
    if max_backtracks < 1:
        raise ValueError(f"max_backtracks must be positive, not {max_backtracks}")
    if step_max is not None and not step_max > 0.0:
        raise ValueError(f"step_max must be positive (or None for no cap), not {step_max}")


def _resume_of(space: VectorSpace[Any, Any, Any, Any], state: LbfgsState, gnorm0: float, it: int,
               f: float) -> LbfgsResume[Any, Any]:
    """The LbfgsResume at the state: the stored pairs, oldest first."""
    memory = state.memory
    stamp = np.asarray(memory.stamp)
    order = [int(j) for j in np.argsort(stamp, kind="stable") if stamp[j] >= 0]
    P = np.asarray(memory.pairings)[np.ix_(order, order)]
    return LbfgsResume(
        gnorm0=gnorm0, next_iteration=it, f=f, gamma=number(state.gamma),
        pairs=tuple((space.vector_member(memory.S, j), space.covector_member(memory.Y, j)) for j in order),
        pairings=tuple(tuple(float(v) for v in row) for row in P),
        floors=tuple(float(np.asarray(memory.floor)[j]) for j in order), slots=tuple(order))


def lbfgs(
        fun: Callable[[PointT], SupportsFloat],
        grad: Callable[[PointT], CovectorT],
        x0: PointT,
        *,
        space: Optional[VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT]] = None,
        retract: Optional[Callable[[PointT, VectorT], PointT]] = None,
        transport: Optional[VectorTransport[PointT, VectorT, CovectorT, VectorBlockT, CovectorBlockT]] = None,
        metric_solve: Optional[Callable[[PointT, CovectorT], VectorT]] = None,
        metric_apply: Optional[Callable[[PointT, VectorT], CovectorT]] = None,
        operators_at: Optional[Callable[[PointT, Any], PointOperators[VectorT, CovectorT]]] = None,
        point_data: Optional[Callable[[PointT], Any]] = None,
        project_covector: Optional[Callable[[PointT, CovectorT], CovectorT]] = None,
        memory: int = 10,
        scaling: Scaling = "y",
        gamma0: float = 1.0,
        initial_step: Optional[float] = None,
        pairings: Pairings = "stored",
        curvature_test: CurvatureTest = "relative",
        curvature_tol: Optional[float] = None,
        angle_tol: float = 1e-8,
        gtol: float = 0.0, rtol: float = 1e-8, gnorm0: Optional[float] = None,
        max_iter: int = 1000,
        c1: float = 1e-4, max_backtracks: int = 30, step_max: Optional[float] = None,
        display: int = 0, print_fn: Callable[[str], None] = print,
        callback: Optional[Callable[[PointT, LbfgsInfo[VectorT, CovectorT]], Optional[bool]]] = None,
        resume: Optional[LbfgsResume[VectorT, CovectorT]] = None,
        g0: Optional[CovectorT] = None,
        backend: Optional[Backend] = None,
        ) -> tuple[PointT, LbfgsSummary[VectorT, CovectorT]]:
    """Limited-memory BFGS in the geometry of the metric M (see the module
    docstring).

    Problem pieces:
      fun(x), grad(x)        f and its gradient, a covector (the derivative of
                             the pullback s -> f(retract(x, s)) at s = 0)
      x0                     the starting point
      space, retract         the vector space structure and the retraction,
                             as for trust_region_newton (defaults for plain
                             numpy arrays: numpy_space and x + s)
      transport              a VectorTransport carrying vectors and covectors
                             from one point to the next (default for plain
                             numpy arrays: identity_transport; required with a
                             space)
      metric_solve(x, w)     M^{-1} w, and metric_apply(x, v), M v: both
                             required (default for plain numpy arrays: M = I)
      operators_at(x, data), point_data(x)
                             alternatively, the operators at a point from
                             operators_at (a PointOperators; hessvec is not
                             used), with data = point_data(x) computed once
                             per point and shared by them
      project_covector(x, w) the canonical representative of a covector, for
                             redundant representations (applied to gradients
                             and transported gradients)

    Options:
      memory                 m, the number of pairs kept (default 10)
      scaling                gamma after each stored pair: "y" (default,
                             <y, s> / <y, M^{-1} y>), "s" (<M s, s> / <y, s>),
                             "geometric" (||s||_M / ||y||_{M^{-1}}), "none"
      gamma0, initial_step   gamma until the first pair: gamma0 (default 1:
                             B0 = M, the first step is -M^{-1} g), or a first
                             step of M-length initial_step
      pairings               "stored" (default: the pairings <y_j, s_i>
                             computed when pairs were stored are kept when
                             the memory is transported) or "recomputed"
                             (those of the transported pairs; pairs whose
                             curvature falls to the floor they passed are
                             dropped)
      curvature_test         a pair is stored only if <y, s> exceeds:
                             curvature_tol alpha |<g, d>| ("relative",
                             default; curvature_tol 1e-8), 0 ("positive"),
                             curvature_tol ||s||_M ||y||_{M^{-1}}
                             ("dennis-schnabel", 2^-26), curvature_tol
                             ||g||_{M^{-1}} ||s||_M^2 ("cautious", 1e-6)
      angle_tol              a direction from the memory is used only if
                             <-g, d> >= angle_tol ||g||_{M^{-1}} ||d||_M;
                             else -gamma M^{-1} g, and the memory is cleared
                             (default 1e-8)
      gtol, rtol, gnorm0, max_iter   stopping: ||g||_{M^{-1}} <= max(gtol,
                             rtol gnorm0), gnorm0 = ||g(x0)|| unless given
      c1, max_backtracks, step_max   the line search: Armijo constant, most
                             evaluations of f per iteration, cap on
                             ||alpha d||_M (None: no cap)
      display, print_fn      0 silent, 1 one row per iteration
                             (format_lbfgs_row), 2 also the trials
      callback(x, info)      after each iteration, with an LbfgsInfo; return
                             True to stop
      resume, g0             continue a run exactly from x with the
                             info.resume saved in the callback (and g0 =
                             info.g to skip one gradient)
      backend                how the iteration is computed (numpy by default;
                             with a compiling backend every iteration is one
                             compiled piece, and the functions given must be
                             traceable, see trust_region_newton)

    If the line search fails, the memory is cleared and the next iteration
    starts from -gamma M^{-1} g (a restart); if it fails with an empty
    memory, the run stops with reason "line_search" at the last accepted
    iterate.

    Returns (x, summary): the final iterate and an LbfgsSummary.
    """
    _check(space=space, retract=retract, transport=transport, metric_solve=metric_solve,
           metric_apply=metric_apply, operators_at=operators_at, point_data=point_data,
           memory=memory, scaling=scaling, gamma0=gamma0, initial_step=initial_step,
           pairings=pairings, curvature_test=curvature_test, curvature_tol=curvature_tol,
           angle_tol=angle_tol, c1=c1, max_backtracks=max_backtracks, step_max=step_max)
    bk = numpy_backend if backend is None else backend
    xp = bk.xp
    sp: VectorSpace[Any, Any, Any, Any] = numpy_space if space is None else space
    problem = LbfgsProblem(
        fun=fun, grad=grad, space=sp, numpy_default=space is None,
        retract=retract if retract is not None else (lambda x, s: x + s),
        transport=transport if transport is not None else identity_transport,
        metric_solve=metric_solve, metric_apply=metric_apply, operators_at=operators_at,
        point_data=point_data, project_covector=project_covector, memory=memory, scaling=scaling,
        gamma0=gamma0, initial_step=initial_step, pairings=pairings, curvature_test=curvature_test,
        curvature_tol=CURVATURE_TOL[curvature_test] if curvature_tol is None else curvature_tol,
        angle_tol=angle_tol, c1=c1, max_backtracks=max_backtracks, step_max=step_max)
    start_piece = compile_function(bk, partial(init, problem, bk))
    step_piece = compile_function(bk, partial(lbfgs_step, problem, bk))
    restart_piece = compile_function(bk, partial(restart, problem, bk))

    start = time.perf_counter()
    state: LbfgsState = start_piece(x0, g0)
    fx = number(state.f)
    fun_calls, grad_calls = 1, (0 if g0 is not None else 1)
    metric_solves, metric_applies = 1, 0
    vector_transports = covector_transports = block_transports = 0
    trials = restarts = skipped = dropped = 0
    it = 0
    if resume is not None:
        check_resumed_f(resume.f, fx)
        gnorm0, it = resume.gnorm0, resume.next_iteration
        k = min(len(resume.pairs), memory)          # the newest, if memory shrank
        kept = resume.slots[len(resume.slots) - k:]
        slots = kept if max(kept, default=-1) < memory else tuple(range(k))
        state = LbfgsState(
            x=state.x, f=state.f, g=state.g, data=state.data, Mg=state.Mg, gnorm=state.gnorm,
            metric_gnorm=state.metric_gnorm, gamma=xp.asarray(resume.gamma), stamp=xp.asarray(k),
            memory=memory_from_pairs(sp, bk, resume.pairs[len(resume.pairs) - k:],
                                     [row[len(row) - k:] for row in resume.pairings[len(resume.pairings) - k:]],
                                     resume.floors[len(resume.floors) - k:], slots, state.Mg, state.g,
                                     memory))
    gnorm = number(state.gnorm)
    if gnorm0 is None:
        gnorm0 = gnorm
    records: list[LbfgsRecord] = []
    while True:
        if gradient_converged(gnorm, gnorm0, gtol, rtol):
            reason: LbfgsReason = "converged"
            break
        if it >= max_iter:
            reason = "max_iter"
            break
        new_state, raw = step_piece(state)
        n_trials = int(np.asarray(raw.trials))
        accepted = bool(np.asarray(raw.status) == ARMIJO)
        direction = DIRECTIONS[int(np.asarray(raw.direction))]
        n_pairs = int(np.asarray(raw.pairs))
        cleared = not accepted and direction != "fallback" and n_pairs > 0
        if cleared:
            new_state = restart_piece(state)
            restarts += 1
        update = UPDATES[int(np.asarray(raw.update))]
        n_dropped = int(np.asarray(raw.dropped))
        fun_calls += n_trials
        trials += n_trials
        skipped += update == "skipped"
        dropped += n_dropped
        metric_solves += int(np.asarray(raw.metric_solves))
        metric_applies += int(np.asarray(raw.metric_applies))
        vector_transports += int(np.asarray(raw.vector_transports))
        covector_transports += int(np.asarray(raw.covector_transports))
        block_transports += int(np.asarray(raw.block_transports))
        record = LbfgsRecord(
            iteration=it, f=fx, gnorm=gnorm, gnorm_ratio=gnorm / gnorm0 if gnorm0 > 0.0 else 0.0,
            direction=direction, base="M", pairs=n_pairs, gamma=number(raw.gamma),
            step_norm=number(raw.step_norm), cosine=number(raw.cosine), slope=number(raw.slope),
            alpha=number(raw.alpha), trials=n_trials, actual=number(raw.actual), accepted=accepted,
            curvature=number(raw.curvature), update=update, dropped=n_dropped,
            transport_defect=number(raw.transport_defect), restart=cleared,
            elapsed=time.perf_counter() - start,
            trace_alpha=tuple(np.asarray(raw.trace_alpha)[:n_trials].tolist()),
            trace_f=tuple(np.asarray(raw.trace_f)[:n_trials].tolist()))
        records.append(record)
        state = new_state
        if accepted:
            fx, gnorm = number(state.f), number(state.gnorm)
            grad_calls += 1
        it += 1
        if display >= 1:
            display_lbfgs_iteration(record, display, print_fn)
        stop = False
        if callback is not None:
            info: LbfgsInfo[VectorT, CovectorT] = LbfgsInfo(
                record=record, f=fx, g=state.g, step=raw.step,
                resume=_resume_of(sp, state, gnorm0, it, fx))
            stop = bool(callback(state.x, info))
        if not accepted and not cleared:
            reason = "line_search"
            break
        if stop:
            reason = "callback"
            break

    summary: LbfgsSummary[VectorT, CovectorT] = LbfgsSummary(
        reason=reason, iterations=tuple(records), f=fx, gnorm=gnorm, gnorm0=gnorm0,
        fun_calls=fun_calls, grad_calls=grad_calls, metric_solve_calls=metric_solves,
        metric_apply_calls=metric_applies, vector_transport_calls=vector_transports,
        covector_transport_calls=covector_transports, block_transport_calls=block_transports,
        trials=trials, restarts=restarts, skipped=skipped, dropped=dropped,
        elapsed=time.perf_counter() - start, resume=_resume_of(sp, state, gnorm0, it, fx))
    if display >= 1:
        print_fn(format_lbfgs_summary(summary))
    x: PointT = state.x
    return x, summary
