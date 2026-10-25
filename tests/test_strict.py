"""The method on the strict backend (tests/strict_backend.py): the code path a
compiling backend takes, checked without a framework. Every run must give
the records of the same run on the numpy backend.

The problem functions here are traceable (no float() of values, no Python
control flow on data), as they must be for a compiling backend.
"""

import dataclasses
from typing import Any, Optional

import numpy as np
import pytest

import matrix_completion as mc
from helpers import without_times
from nicks_optimizers import (
    array_space,
    euclidean_metric_apply,
    euclidean_metric_solve,
    solve_tr_subproblem,
    solve_tr_subproblem_flexible,
    flat_retraction,
    numpy_backend,
    numpy_space,
    solve_tr_subproblem_cg,
    trust_region_newton,
    tuple_tree_space,
)
from problems import (
    rosen_hess_diag,
    pa_rosen_grad,
    pa_rosen_hessvec,
    rosen_grad,
    rosen_hessvec,
    split_pair,
    well_grad,
    well_hessvec,
)
from strict_backend import plain, strict, strict_backend


def rosen_fun(x: Any) -> Any:
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def well_fun(x: Any) -> Any:
    return np.sum(0.25 * x**4 - 0.5 * x**2) + 0.05 * np.sum((x[1:] - x[:-1]) ** 2)


def pa_rosen_fun(x: Any) -> Any:
    return rosen_fun(np.concatenate(x))


SMOOTHING = 10.0


def laplacian(n: int) -> Any:
    """The dense 1-D Laplacian with Neumann ends of problems.laplacian_apply."""
    L = 2.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    L[0, 0] = L[-1, -1] = 1.0
    return L


def smoothing_metric_solve(x: Any, w: Any) -> Any:
    """problems.smoothing_metric_solve by a dense solve (the Thomas algorithm
    there writes computed values into array elements, which no compiler
    traces)."""
    n = w.shape[0]
    return np.linalg.solve(np.eye(n) + SMOOTHING * laplacian(n), w)


def weighted_metric_solve(x: Any, w: Any) -> Any:
    n = w.shape[0]
    return np.linalg.solve(np.diag(1.0 + x**2) + SMOOTHING * laplacian(n), w)


TRACEABLE_METRICS = [("identity", euclidean_metric_solve), ("smoothing", smoothing_metric_solve),
                     ("x-dependent", weighted_metric_solve)]


def both(run: Any, exact: bool = True) -> None:
    """The same run on the numpy and strict backends gives the same records:
    bit for bit, or (exact=False) with the same counts and stop reasons and
    values equal to rounding (numpy scalars on one side and 0-d arrays on the
    other may round differently, e.g. in x ** 2)."""
    x_np, s_np = run(numpy_backend)
    x_st, s_st = run(strict_backend)
    assert s_np.reason == "converged" or s_np.reason == "max_iter"
    if not exact:
        assert [(r.accepted, r.subproblem.krylov_steps, r.subproblem.stop_reason)
                for r in s_st.iterations] == [(r.accepted, r.subproblem.krylov_steps,
                                              r.subproblem.stop_reason) for r in s_np.iterations]
        assert all(abs(a.f - b.f) <= 1e-12 * max(abs(b.f), 1e-300) + 1e-300
                   for a, b in zip(s_st.iterations, s_np.iterations))
        return
    assert without_times(s_st.iterations) == without_times(s_np.iterations)
    assert repr(dataclasses.replace(s_st, iterations=(), elapsed=0.0)) == \
        repr(dataclasses.replace(s_np, iterations=(), elapsed=0.0))


@pytest.mark.parametrize("metric", TRACEABLE_METRICS, ids=[m[0] for m in TRACEABLE_METRICS])
def test_rosenbrock_cg(metric: Any) -> None:
    _, solve = metric
    x0: Any = np.tile([-1.2, 1.0], 10)
    both(lambda bk: trust_region_newton(rosen_fun, rosen_grad, rosen_hessvec, x0,
                                        metric_solve=solve, backend=bk))


def test_double_well_cg_with_rejections() -> None:
    x0: Any = np.random.default_rng(3).uniform(-0.3, 0.3, 30)
    both(lambda bk: trust_region_newton(well_fun, well_grad, well_hessvec, x0, radius=10.0,
                                        backend=bk))


def test_tuple_trees_and_few_krylov_steps() -> None:
    space: Any = tuple_tree_space()
    x0: Any = split_pair(np.tile([-1.2, 1.0], 10))
    both(lambda bk: trust_region_newton(pa_rosen_fun, pa_rosen_grad, pa_rosen_hessvec, x0,
                                        space=space, retract=flat_retraction(space),
                                        metric_solve=euclidean_metric_solve, max_krylov=3,
                                        max_iter=40, backend=bk))


def test_array_space_cg() -> None:
    """array_space, the space for compiling backends, on both backends."""
    x0: Any = np.tile([-1.2, 1.0], 10)
    both(lambda bk: trust_region_newton(rosen_fun, rosen_grad, rosen_hessvec, x0,
                                        space=array_space(bk), retract=lambda x, s: x + s,
                                        metric_solve=euclidean_metric_solve, backend=bk),
         exact=False)


def test_matrix_completion_cg() -> None:
    """The fixed-rank manifold: tuple trees, the gauge projection, a retraction
    by QR and SVD."""
    rng = np.random.default_rng(0)
    shape, r = (40, 30), 3
    a = mc.random_low_rank(shape, 0.5 ** np.arange(8, dtype=float), rng)
    samples = mc.sample_uniformly(a, 4 * (shape[0] + shape[1] - r) * r, rng)
    x0 = mc.random_point(shape, r, rng)

    def objective(x: Any) -> Any:
        res = mc.residual(samples, x)
        return 0.5 * (res @ res)
    both(lambda bk: trust_region_newton(
        objective, lambda x: mc.gradient(samples, x), lambda x, v: mc.hessian_action(samples, x, v),
        x0, space=mc.fixed_rank_space(), retract=mc.retract, metric_solve=euclidean_metric_solve,
        project_covector=mc.gauge_projection, max_iter=40, backend=bk))


def test_callback_resume_and_point_data() -> None:
    """Per-point data, the callback and resuming work the same way."""
    x0: Any = np.tile([-1.2, 1.0], 10)

    def run(bk: Any) -> Any:
        seen: list[int] = []

        def callback(x: Any, info: Any) -> bool:
            seen.append(info.record.iteration)
            return False
        x, first = trust_region_newton(
            rosen_fun, rosen_grad, None, x0, max_iter=12, backend=bk,
            point_data=lambda x: np.maximum(np.abs(x), 1.0),
            operators_at=lambda x, d: _ops(x, d), callback=callback)
        x, second = trust_region_newton(rosen_fun, rosen_grad, None, x, resume=first.resume,
                                        point_data=lambda x: np.maximum(np.abs(x), 1.0),
                                        operators_at=lambda x, d: _ops(x, d), backend=bk)
        assert seen == list(range(12))
        return x, dataclasses.replace(second, iterations=first.iterations + second.iterations)
    both(run)


def _ops(x: Any, d: Any) -> Any:
    from nicks_optimizers import PointOperators
    return PointOperators(hessvec=lambda v: rosen_hessvec(x, v), metric_solve=lambda w: w / d)


@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
def test_errors_raise_after_the_compiled_piece() -> None:
    """A metric that is not positive definite is reported as before."""
    g = np.ones(6)
    for bk in (numpy_backend, strict_backend):
        with pytest.raises(ValueError, match="must be positive"):
            solve_tr_subproblem_cg(numpy_space, strict(g), lambda v: 2.0 * v, lambda w: -w, 1.0,
                                   1e-8, 10, backend=bk)


def jacobi(x: Any, w: Any) -> Any:
    """P(x) = diag(max(|H_ii(x)|, 1)) of problems.jacobi_precond_solve."""
    return w / np.maximum(np.abs(rosen_hess_diag(x)), 1.0)


def agree(reference: Any, other: Any, first: int = 15) -> None:
    """Runs that differ only by rounding (padded against cut-down projected
    problems): the same counts, acceptances and stop reasons over the first
    iterations, objective values equal to rounding there, and the same
    outcome."""
    a, b = reference.iterations[:first], other.iterations[:first]
    assert [(r.accepted, r.subproblem.krylov_steps, r.subproblem.stop_reason) for r in a] == \
        [(r.accepted, r.subproblem.krylov_steps, r.subproblem.stop_reason) for r in b]
    assert all(abs(u.f - v.f) <= 1e-9 * max(abs(v.f), 1e-12) for u, v in zip(a, b))
    assert other.reason == reference.reason == "converged"


SOLVER_CASES: dict[str, dict[str, Any]] = {
    "lanczos": dict(krylov="lanczos"),
    "lanczos, separate P": dict(krylov="lanczos", precond_solve=jacobi,
                                metric_apply=euclidean_metric_apply),
    "lanczos, saddle-free": dict(saddle_free=True),
    "flexible": dict(krylov="flexible", precond_solve=jacobi, metric_apply=euclidean_metric_apply),
}


@pytest.mark.parametrize("name", list(SOLVER_CASES))
def test_lanczos_and_flexible_on_the_strict_backend(name: str) -> None:
    """The Lanczos and flexible solvers on the code path of a compiling
    backend: array_space, padded projected problems, and a basis that starts
    with room for 1, 2 or 3 vectors and doubles whenever a solve fills it."""
    options = SOLVER_CASES[name]
    x0: Any = np.tile([-1.2, 1.0], 10)
    _, reference = trust_region_newton(rosen_fun, rosen_grad, rosen_hessvec, x0,
                                       metric_solve=euclidean_metric_solve, **options)
    assert reference.capacity_growths == 0
    for capacity in (1, 2, 3, 50):
        space = array_space(strict_backend)
        _, run = trust_region_newton(rosen_fun, rosen_grad, rosen_hessvec, x0, space=space,
                                     retract=lambda x, s: x + s, metric_solve=euclidean_metric_solve,
                                     backend=strict_backend, krylov_capacity=capacity, **options)
        agree(reference, run)
        most = max(r.subproblem.krylov_steps for r in run.iterations)
        assert (run.capacity_growths > 0) == (capacity < min(most, 50))


def test_subproblem_solvers_grow_their_basis() -> None:
    """The subproblem solvers alone: a basis that starts with room for one
    vector and doubles gives the solution of an unlimited basis."""
    rng = np.random.default_rng(4)
    n = 25
    Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
    H = (Q * np.linspace(-1.0, 8.0, n)) @ Q.T
    g = rng.standard_normal(n)
    P = np.diag(np.abs(np.diag(H)) + 1.0)
    def lanczos(bk: Any, cap: Optional[int], radius: float) -> Any:
        return solve_tr_subproblem(array_space(bk), strict(g) if bk is strict_backend else g,
                                   lambda v: H @ v, lambda w: w, radius, 1e-10, 40, backend=bk,
                                   capacity=cap)

    def flexible(bk: Any, cap: Optional[int], radius: float) -> Any:
        return solve_tr_subproblem_flexible(
            array_space(bk), strict(g) if bk is strict_backend else g, lambda v: H @ v,
            lambda w: w, lambda v: v, lambda w: np.linalg.solve(P, w), radius, 1e-10, 40,
            backend=bk, capacity=cap)
    for radius in (0.3, 100.0):
        for solve in (lanczos, flexible):
            s_ref, st_ref = solve(numpy_backend, None, radius)
            for cap in (1, 2, 5):
                s, st = solve(strict_backend, cap, radius)
                assert st.krylov_steps == st_ref.krylov_steps and st.stop_reason == st_ref.stop_reason
                assert np.max(np.abs(np.asarray(plain(s)) - np.asarray(s_ref))) <= 1e-9 * np.max(np.abs(s_ref))
