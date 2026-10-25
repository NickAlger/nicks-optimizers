"""The method on the strict backend (tests/strict_backend.py): the code path a
compiling backend takes, checked without a framework. Every run must give
the records of the same run on the numpy backend.

The problem functions here are traceable (no float() of values, no Python
control flow on data), as they must be for a compiling backend.
"""

import dataclasses
from typing import Any

import numpy as np
import pytest

import matrix_completion as mc
from helpers import without_times
from nicks_optimizers import (
    euclidean_metric_solve,
    flat_retraction,
    numpy_backend,
    numpy_space,
    solve_tr_subproblem_cg,
    trust_region_newton,
    tuple_tree_space,
)
from problems import (
    pa_rosen_grad,
    pa_rosen_hessvec,
    rosen_grad,
    rosen_hessvec,
    split_pair,
    well_grad,
    well_hessvec,
)
from strict_backend import strict, strict_backend


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


def both(run: Any) -> None:
    """The same run on the numpy and strict backends gives the same records."""
    x_np, s_np = run(numpy_backend)
    x_st, s_st = run(strict_backend)
    assert s_np.reason == "converged" or s_np.reason == "max_iter"
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
    """A metric that is not positive definite is reported as before, and the
    solvers not converted yet refuse the strict backend."""
    g = np.ones(6)
    for bk in (numpy_backend, strict_backend):
        with pytest.raises(ValueError, match="must be positive"):
            solve_tr_subproblem_cg(numpy_space, strict(g), lambda v: 2.0 * v, lambda w: -w, 1.0,
                                   1e-8, 10, backend=bk)
    with pytest.raises(NotImplementedError):
        x0: Any = np.tile([-1.2, 1.0], 5)
        trust_region_newton(rosen_fun, rosen_grad, rosen_hessvec, x0, krylov="lanczos",
                            backend=strict_backend)
