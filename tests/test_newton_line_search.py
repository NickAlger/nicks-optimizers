"""line_search_newton (dev/line_search_plan.md, phase 5): convergence, the
line search in the records, resuming, display, operator calls, other
representations, and the strict backend."""

import dataclasses
import json
from typing import Any, Callable, Optional, cast

import numpy as np
import pytest

import matrix_completion as mc
from nicks_optimizers import (
    Covector,
    Point,
    Vector,
    array_space,
    euclidean_metric_apply,
    euclidean_metric_solve,
    flat_retraction,
    numpy_space_with_reductions,
    trust_region_newton,
    tuple_tree_space,
)
from nicks_optimizers.newton_line_search import line_search_newton
from nicks_optimizers.newton_line_search_diagnostics import (
    LineSearchResume,
    display_ls_iteration,
    format_ls_summary,
    ls_history_array,
    merge_ls_summaries,
)
from problems import (
    jacobi_precond_solve,
    pa_rosen_fun,
    pa_rosen_grad,
    pa_rosen_hessvec,
    rosen_fun,
    rosen_grad,
    rosen_hessvec,
    smoothing_metric_apply,
    smoothing_metric_solve,
    split_pair,
    well_fun,
    well_grad,
    well_hessvec,
    well_precond_solve,
)
from strict_backend import strict_backend

X0 = Point(np.tile([-1.2, 1.0], 10))

CONFIGS: dict[str, dict[str, Any]] = {
    "cg": dict(metric_solve=jacobi_precond_solve),
    "lanczos": dict(metric_solve=jacobi_precond_solve, krylov="lanczos"),
    "lanczos, separate P": dict(metric_solve=smoothing_metric_solve, metric_apply=smoothing_metric_apply,
                                precond_solve=jacobi_precond_solve),
    "flexible": dict(metric_solve=smoothing_metric_solve, metric_apply=smoothing_metric_apply,
                     precond_solve=jacobi_precond_solve, krylov="flexible"),
    "saddle-free": dict(metric_solve=jacobi_precond_solve, saddle_free=True),
}


def without_times(records: Any) -> list[str]:
    return [repr(dataclasses.replace(r, elapsed=0.0)) for r in records]


def is_local_minimizer(x: Any) -> bool:
    """Small gradient and a positive definite Hessian (chained Rosenbrock in
    20 dimensions has a second local minimizer near (-1, 1, ..., 1), which
    the line search may reach from X0)."""
    H = np.column_stack([rosen_hessvec(x, Vector(e)) for e in np.eye(len(x))])
    return bool(np.linalg.norm(rosen_grad(x)) <= 1e-4 and np.linalg.eigvalsh(0.5 * (H + H.T))[0] > 0.0)


def test_a_convex_quadratic_takes_one_unit_newton_step() -> None:
    """With an exact solve (forcing_max tiny) the first step is the Newton
    step, the line search accepts alpha = 1 at once, and the run converges."""
    rng = np.random.default_rng(0)
    Q = np.linalg.qr(rng.standard_normal((15, 15)))[0]
    A = (Q * np.logspace(0, 2, 15)) @ Q.T
    b = rng.standard_normal(15)
    x, s = line_search_newton(lambda x: 0.5 * x @ A @ x - b @ x, lambda x: Covector(A @ x - b),
                              lambda x, v: Covector(A @ v), Point(np.zeros(15)), forcing_max=1e-12)
    r = s.iterations[0]
    assert s.reason == "converged" and len(s.iterations) == 1
    assert (r.alpha, r.trials, r.direction, r.accepted) == (1.0, 1, "newton", True)
    assert np.linalg.norm(x - np.linalg.solve(A, b)) <= 1e-10 * np.linalg.norm(np.linalg.solve(A, b))
    assert abs(r.actual - r.predicted) <= 1e-10 * abs(r.predicted) and r.slope < 0.0


@pytest.mark.parametrize("name", list(CONFIGS))
def test_rosenbrock_converges(name: str) -> None:
    x, s = line_search_newton(rosen_fun, rosen_grad, rosen_hessvec, X0, **CONFIGS[name])
    assert s.reason == "converged" and is_local_minimizer(x)
    for r in s.iterations:
        assert r.accepted and r.slope < 0.0 and 0.0 < r.alpha <= 1.0
        assert r.trials == len(r.trace_alpha) and r.trace_alpha[-1] == r.alpha
        assert r.trace_f[-1] <= r.f + 1e-4 * r.alpha * r.slope + 1e-12 * max(1.0, abs(r.f))
        assert r.subproblem.lam == 0.0
    assert s.fun_calls == 1 + s.trials and s.trials == sum(r.trials for r in s.iterations)
    assert s.grad_calls == 1 + len(s.iterations)


@pytest.mark.parametrize("saddle_free", [False, True])
def test_double_well(saddle_free: bool) -> None:
    """A nonconvex problem: truncated directions or the saddle-free model,
    then a minimizer (positive definite Hessian)."""
    x0 = Point(np.random.default_rng(3).uniform(-0.3, 0.3, 30))
    options: dict[str, Any] = dict(saddle_free=True, precond_solve=well_precond_solve,
                                   metric_solve=euclidean_metric_solve,
                                   metric_apply=euclidean_metric_apply) if saddle_free else {}
    x, s = line_search_newton(well_fun, well_grad, well_hessvec, x0, **options)
    assert s.reason == "converged"
    H = np.column_stack([well_hessvec(x, Vector(e)) for e in np.eye(30)])
    assert np.linalg.eigvalsh(0.5 * (H + H.T))[0] > 0.0
    kinds = {r.direction for r in s.iterations}
    assert ("truncated" in kinds or "steepest" in kinds) != saddle_free


def test_steepest_descent_at_negative_curvature() -> None:
    """At a saddle-like start the first direction is -M^{-1} g."""
    def fun(x: Any) -> Any:
        return -0.5 * x[0] ** 2 + 0.25 * x[0] ** 4 + 0.5 * x[1] ** 2

    def grad(x: Any) -> Any:
        return Covector(np.array([-x[0] + x[0] ** 3, x[1]]))

    def hessvec(x: Any, v: Any) -> Any:
        return Covector(np.array([(-1.0 + 3.0 * x[0] ** 2) * v[0], v[1]]))
    x, s = line_search_newton(fun, grad, hessvec, Point(np.array([0.1, 0.0])))
    assert s.iterations[0].direction == "steepest" and s.reason == "converged"
    assert abs(abs(x[0]) - 1.0) <= 1e-8


def test_step_max_caps_every_first_trial() -> None:
    x, s = line_search_newton(rosen_fun, rosen_grad, rosen_hessvec, X0, step_max=0.05, max_iter=2000)
    assert s.reason == "converged" and is_local_minimizer(x)
    for r in s.iterations:
        assert r.trace_alpha[0] * r.step_norm <= 0.05 * (1.0 + 1e-12)


def test_a_failed_line_search_stops_the_run() -> None:
    """A gradient of the wrong sign: every direction claims descent but f
    increases, so no trial is accepted and the run stops where it started."""
    x, s = line_search_newton(rosen_fun, lambda x: Covector(-rosen_grad(x)),
                              lambda x, v: Covector(rosen_hessvec(x, v) + 1e3 * v), X0,
                              max_backtracks=5)
    assert s.reason == "line_search" and len(s.iterations) == 1
    r = s.iterations[0]
    assert not r.accepted and r.trials == 5 and np.array_equal(x, X0) and s.f == rosen_fun(X0)


def test_resuming_reproduces_the_run() -> None:
    _, whole = line_search_newton(rosen_fun, rosen_grad, rosen_hessvec, X0)
    saved: dict[str, Any] = {}

    def callback(x: Any, info: Any) -> Optional[bool]:
        if info.record.iteration == 9:
            saved.update(x=x.copy(), g=info.g.copy(), state=json.dumps(dataclasses.asdict(info.resume)))
            return True
        return None
    _, first = line_search_newton(rosen_fun, rosen_grad, rosen_hessvec, X0, callback=callback)
    assert first.reason == "callback" and len(first.iterations) == 10
    state = LineSearchResume(**json.loads(saved["state"]))
    _, second = line_search_newton(rosen_fun, rosen_grad, rosen_hessvec, saved["x"], resume=state,
                                   g0=saved["g"])
    merged = merge_ls_summaries(first, second)
    assert without_times(merged.iterations) == without_times(whole.iterations)
    assert (merged.f, merged.gnorm, merged.reason) == (whole.f, whole.gnorm, whole.reason)
    with pytest.raises(ValueError):
        line_search_newton(rosen_fun, rosen_grad, rosen_hessvec, X0, resume=state)
    assert ls_history_array(whole).shape == (len(whole.iterations), 8)


def test_display() -> None:
    lines: list[str] = []
    _, s = line_search_newton(rosen_fun, rosen_grad, rosen_hessvec, X0, display=2,
                              print_fn=lines.append, max_iter=25)
    expected: list[str] = []
    for r in s.iterations:
        display_ls_iteration(r, 2, expected.append)
    assert lines == expected + [format_ls_summary(s)]
    assert lines[0].split()[:3] == ["it", "f", "|g|"] and "direction" in lines[0]
    assert any("accepted" in line for line in lines) and any("trial" in line for line in lines)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_operator_calls(name: str) -> None:
    """The user's functions, instrumented: the summary counts are the calls
    made, and a basis that grows from room for one vector makes the same."""
    calls: dict[int, dict[str, int]] = {}
    for capacity in (1, 50):
        count: dict[str, int] = {}

        def counted(key: str, f: Callable[..., Any]) -> Callable[..., Any]:
            def wrapper(*args: Any) -> Any:
                count[key] = count.get(key, 0) + 1
                return f(*args)
            return wrapper
        options: dict[str, Any] = {k: counted(k, f) if callable(f) else f
                                   for k, f in CONFIGS[name].items()}
        _, s = line_search_newton(counted("fun", rosen_fun), counted("grad", rosen_grad),
                                  counted("hessvec", rosen_hessvec), X0, krylov_capacity=capacity,
                                  **options)
        assert (s.fun_calls, s.grad_calls, s.hessvec_calls, s.metric_solve_calls, s.metric_apply_calls,
                s.precond_solve_calls) == tuple(count.get(k, 0) for k in (
                    "fun", "grad", "hessvec", "metric_solve", "metric_apply", "precond_solve"))
        calls[capacity] = count
    assert calls[1] == calls[50]


def test_other_representations() -> None:
    """Tuple trees, a distributed numpy space, and the fixed-rank manifold."""
    space: Any = tuple_tree_space()
    x, s = line_search_newton(cast(Any, pa_rosen_fun), cast(Any, pa_rosen_grad),
                              cast(Any, pa_rosen_hessvec), split_pair(X0), space=space,
                              retract=flat_retraction(space), metric_solve=euclidean_metric_solve)
    assert s.reason == "converged"
    dist = numpy_space_with_reductions(lambda a: a)
    _, t = line_search_newton(rosen_fun, rosen_grad, rosen_hessvec, X0, space=dist,
                              retract=flat_retraction(dist), metric_solve=euclidean_metric_solve)
    _, ref = line_search_newton(rosen_fun, rosen_grad, rosen_hessvec, X0,
                                metric_solve=euclidean_metric_solve)
    assert without_times(t.iterations) == without_times(ref.iterations)
    rng = np.random.default_rng(0)
    shape, r = (60, 50), 3
    a = mc.random_low_rank(shape, 0.5 ** np.arange(8, dtype=float), rng)
    samples = mc.sample_uniformly(a, 4 * (shape[0] + shape[1] - r) * r, rng)
    tree = dataclasses.replace(cast(Any, tuple_tree_space()), dimension=mc.tangent_dimension)
    problem = (lambda x: mc.objective(samples, x), lambda x: mc.gradient(samples, x),
               lambda x, v: mc.hessian_action(samples, x, v), mc.random_point(shape, r, rng))
    options: dict[str, Any] = dict(space=tree, retract=mc.retract, metric_solve=euclidean_metric_solve,
                                   project_covector=mc.gauge_projection, max_iter=200)
    _, m = line_search_newton(*problem, **options)
    _, tr = trust_region_newton(*problem, **options)          # the best rank-3 fit of a rank-8 matrix
    assert m.reason == tr.reason == "converged" and abs(m.f - tr.f) <= 1e-8 * tr.f


@pytest.mark.parametrize("name", ["cg", "lanczos", "saddle-free"])
def test_strict_backend(name: str) -> None:
    """The code path of a compiling backend (the line search included):
    the numpy run's iterations and directions, values to rounding."""
    def strict_options(options: dict[str, Any]) -> dict[str, Any]:
        out = dict(options)
        out["metric_solve"] = lambda x, w: w / (200.0 + 1200.0 * x ** 2)     # traceable Jacobi
        return out
    options = strict_options(CONFIGS[name])
    _, ref = line_search_newton(rosen_fun, rosen_grad, rosen_hessvec, X0, max_iter=30, **options)
    for capacity in (1, 3):
        _, run = line_search_newton(
            lambda x: 100.0 * np.sum((x[1:] - x[:-1] ** 2) ** 2) + np.sum((1.0 - x[:-1]) ** 2),
            rosen_grad, rosen_hessvec, X0, space=array_space(strict_backend),
            retract=lambda x, s: x + s, backend=strict_backend, krylov_capacity=capacity,
            max_iter=30, **options)
        assert [(r.direction, r.trials, r.subproblem.krylov_steps) for r in run.iterations] == \
            [(r.direction, r.trials, r.subproblem.krylov_steps) for r in ref.iterations]
        for a, b in zip(run.iterations, ref.iterations):
            assert abs(a.f - b.f) <= 1e-10 * max(1.0, abs(b.f)) and abs(a.alpha - b.alpha) <= 1e-8


def test_arguments_are_checked() -> None:
    for bad in (dict(c1=0.0), dict(c1=1.0), dict(max_backtracks=0), dict(step_max=0.0),
                dict(curvature_floor=1.0)):
        with pytest.raises(ValueError):
            line_search_newton(rosen_fun, rosen_grad, rosen_hessvec, X0, **bad)
