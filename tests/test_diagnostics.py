"""Diagnostics and resumption: work counts, stopping reasons, the callback,
display output, and exact continuation from a ResumeState."""

import dataclasses
import json
from typing import Callable, cast

import numpy as np

from nicks_optimizers import (
    Coords,
    Covector,
    CovectorBlock,
    IterationInfo,
    Point,
    ResumeState,
    RunSummary,
    Vector,
    VectorBlock,
    euclidean_dual_inner,
    euclidean_inner,
    merge_summaries,
    numpy_space,
    trust_region_newton,
)
from helpers import (
    expected_display,
    report,
    without_times,
)
from problems import (
    jacobi_precond_solve,
    rosen_fun,
    rosen_grad,
    rosen_hessvec,
    smoothing_dual_inner,
    smoothing_inner,
    weighted_dual_inner,
    weighted_inner,
)


def check_work_counts() -> bool:
    print("RunSummary work counts equal the calls actually made (counting wrappers), saddle-free run")
    counts = {"fun": 0, "grad": 0, "hessvec": 0, "precond": 0, "inner": 0, "dual": 0}

    def fun(x: Point) -> float:
        counts["fun"] += 1
        return rosen_fun(x)

    def grad(x: Point) -> Covector:
        counts["grad"] += 1
        return rosen_grad(x)

    def hessvec(x: Point, v: Vector) -> Covector:
        counts["hessvec"] += 1
        return rosen_hessvec(x, v)

    def precond_solve(x: Point, w: Covector) -> Vector:
        counts["precond"] += 1
        return jacobi_precond_solve(x, w)

    def inner(x: Point, U: VectorBlock, v: Vector) -> Coords:
        counts["inner"] += 1
        return smoothing_inner(x, U, v)

    def dual_inner(x: Point, Psi: CovectorBlock, phi: Covector) -> Coords:
        counts["dual"] += 1
        return smoothing_dual_inner(x, Psi, phi)

    x, summary = trust_region_newton(fun, grad, hessvec, precond_solve, inner, dual_inner,
                                     Point(np.tile([-1.2, 1.0], 10)), saddle_free=True, max_iter=1000)
    reported = {"fun": summary.fun_calls, "grad": summary.grad_calls, "hessvec": summary.hessvec_calls,
                "precond": summary.precond_calls, "inner": summary.inner_calls,
                "dual": summary.dual_inner_calls}
    print(f"  actual calls:   {counts}")
    print(f"  reported calls: {reported}")
    ok = report("counters that disagree", float(sum(reported[key] != counts[key] for key in counts)), 0.0)
    ok &= report("accepted_steps agrees with the records",
                 float(abs(summary.accepted_steps - sum(r.accepted for r in summary.iterations))), 0.0)
    return ok


def check_stopping_reasons() -> bool:
    print("RunSummary.reason for each way a run can end")
    x0 = Point(np.tile([-1.2, 1.0], 10))

    def run(max_iter: int = 500, radius_min: float = 1e-14) -> RunSummary:
        return trust_region_newton(rosen_fun, rosen_grad, rosen_hessvec, jacobi_precond_solve,
                                   euclidean_inner, euclidean_dual_inner, x0,
                                   max_iter=max_iter, radius_min=radius_min)[1]

    cases = [("converged", run(), None), ("max_iter", run(max_iter=3), 3),
             ("radius", run(radius_min=10.0), 0)]
    ok = True
    for expected, summary, n_its in cases:
        right = summary.reason == expected and (n_its is None or len(summary.iterations) == n_its)
        print(f"  [{'PASS' if right else 'FAIL'}] {expected}: reason = {summary.reason!r}, "
              f"{len(summary.iterations)} iterations")
        ok &= right
    return ok


def check_callback() -> bool:
    print("callback(x, info): the iterate, the record, f, g and the step")
    x0 = Point(np.tile([-1.2, 1.0], 10))
    seen: list[tuple[Point, IterationInfo[Vector, Covector]]] = []

    def keep(x: Point, info: IterationInfo[Vector, Covector]) -> None:
        seen.append((x, info))

    x, summary = trust_region_newton(rosen_fun, rosen_grad, rosen_hessvec, jacobi_precond_solve,
                                     euclidean_inner, euclidean_dual_inner, x0, callback=keep)
    ok = report("one call per iteration (count difference)",
                float(abs(len(seen) - len(summary.iterations))), 0.0)
    ok &= report("last x passed is the returned x (0 if identical object)",
                 0.0 if seen[-1][0] is x else 1.0, 0.0)
    ok &= report("info.record is the record kept in the summary",
                 0.0 if all(info.record is rec for (_, info), rec in zip(seen, summary.iterations)) else 1.0, 0.0)
    f_err = max(abs(info.f - rosen_fun(xk)) for xk, info in seen)
    g_err = max(float(np.max(np.abs(info.g - rosen_grad(xk)))) for xk, info in seen)
    ok &= report("info.f and info.g are f and g at x", max(f_err, g_err), 0.0)
    prev, step_err = x0, 0.0
    for xk, info in seen:
        expected = Point(prev + info.step) if info.record.accepted else prev
        step_err = max(step_err, float(np.max(np.abs(xk - expected))))
        prev = xk
    ok &= report("x = retract(previous x, step) when accepted, unchanged when rejected", step_err, 0.0)

    _, stopped = trust_region_newton(rosen_fun, rosen_grad, rosen_hessvec, jacobi_precond_solve,
                                     euclidean_inner, euclidean_dual_inner, x0,
                                     callback=lambda x, info: info.record.iteration >= 3)
    early = stopped.reason == "callback" and len(stopped.iterations) == 4
    print(f"  [{'PASS' if early else 'FAIL'}] returning True stops the run: reason = "
          f"{stopped.reason!r} after {len(stopped.iterations)} iterations")
    return ok and early


def check_display_output() -> bool:
    print("display output is exactly the formatting functions applied to the returned records")
    x0 = Point(np.tile([-1.2, 1.0], 25))
    ok = True
    for level in (0, 1, 2):
        lines: list[str] = []
        _, summary = trust_region_newton(rosen_fun, rosen_grad, rosen_hessvec, jacobi_precond_solve,
                                         euclidean_inner, euclidean_dual_inner, x0, display=level,
                                         print_fn=lines.append, max_iter=45)
        expected = expected_display(summary, level) if level > 0 else []
        same = lines == expected
        print(f"  [{'PASS' if same else 'FAIL'}] display={level}: {len(lines)} printed blocks "
              f"({len(summary.iterations)} iterations)")
        ok &= same
    return ok


def check_resume() -> bool:
    print("Resuming: a run stopped mid-way and resumed from x + ResumeState (+ g0) continues exactly")
    print("  (saddle-free, x-dependent ball; comparison is exact, every record field but the wall clock)")
    args = (rosen_fun, rosen_grad, rosen_hessvec, jacobi_precond_solve,
            weighted_inner, weighted_dual_inner)
    x0 = Point(np.tile([-1.2, 1.0], 25))

    def run(x_start: Point, max_iter: int = 500,
            callback: Callable[[Point, IterationInfo[Vector, Covector]], bool] | None = None,
            resume: ResumeState | None = None, g0: Covector | None = None) -> tuple[Point, RunSummary]:
        return trust_region_newton(*args, x_start, saddle_free=True, max_lanczos=100,
                                   max_iter=max_iter, callback=callback, resume=resume, g0=g0)

    x_full, full = run(x0)

    # 1. "Crash" after iteration 36: the callback saves x, g and the state, as a checkpoint would,
    #    with the state round-tripped through JSON.
    saved: dict[str, object] = {}

    def checkpoint(x: Point, info: IterationInfo[Vector, Covector]) -> bool:
        saved.update(x=x, g=info.g, state=json.dumps(dataclasses.asdict(info.resume)))
        return info.record.iteration == 36

    _, first = run(x0, callback=checkpoint)
    state = ResumeState(**json.loads(str(saved["state"])))
    x_res, second = run(cast(Point, saved["x"]), resume=state, g0=cast(Covector, saved["g"]))
    merged = merge_summaries(first, second)
    ok = report("records of first + resumed run vs uninterrupted run (0 if identical)",
                0.0 if without_times(merged.iterations) == without_times(full.iterations) else 1.0, 0.0)
    ok &= report("final x vs uninterrupted run", float(np.max(np.abs(x_res - x_full))), 0.0)
    ok &= report("merged reason and counts consistent (0 if so)",
                 0.0 if (merged.reason == full.reason
                         and merged.hessvec_calls == full.hessvec_calls) else 1.0, 0.0)
    ok &= report("g0 skipped the restart gradient (extra gradient calls)",
                 float(second.grad_calls - second.accepted_steps), 0.0)
    print(f"         (uninterrupted: {len(full.iterations)} iterations; stopped after "
          f"{len(first.iterations)}, resumed for {len(second.iterations)}; "
          f"resumed run numbered from {second.iterations[0].iteration})")

    # 2. Running in chunks with summary.resume (max_iter is the budget for the whole run).
    x_a, part = run(x0, max_iter=50)
    x_b, rest = run(x_a, resume=part.resume)
    ok &= report("two chunks via summary.resume vs uninterrupted (records, 0 if identical)",
                 0.0 if without_times(part.iterations + rest.iterations)
                 == without_times(full.iterations) else 1.0, 0.0)

    # 3. A state saved at another point than x is rejected.
    try:
        run(cast(Point, saved["x"]), resume=part.resume)
        rejected = False
    except ValueError:
        rejected = True
    print(f"  [{'PASS' if rejected else 'FAIL'}] a state that does not belong to x0 is rejected")
    return ok and rejected


def test_work_counts() -> None:
    assert check_work_counts()


def test_stopping_reasons() -> None:
    assert check_stopping_reasons()


def test_callback() -> None:
    assert check_callback()


def test_display_output() -> None:
    assert check_display_output()


def test_resume() -> None:
    assert check_resume()
