"""The backtracking line search (src/nicks_optimizers/line_search.py,
dev/line_search_plan.md phase 4) on one-dimensional functions."""

from typing import Any, Callable

import numpy as np
import pytest

from nicks_optimizers import numpy_backend
from nicks_optimizers.backend import Backend, loop_from_scan, python_cond
from nicks_optimizers.line_search import (
    ARMIJO,
    FAILED,
    LINE_SEARCH_STATUS,
    NOT_DESCENT,
    backtracking,
    next_step,
)
from strict_backend import plain, strict, strict_backend

Phi = Callable[[Any], tuple[Any, Any]]


def along(f: Callable[[Any], Any]) -> Phi:
    """phi for the one-dimensional function f: the point is alpha itself."""
    return lambda alpha: (f(alpha), alpha)


def search(f: Callable[[Any], Any], slope: float, alpha0: float = 1.0, backend: Backend = numpy_backend,
           **options: Any) -> Any:
    f0 = f(np.asarray(0.0))
    return backtracking(backend, along(f), np.asarray(0.0), f0, slope, alpha0, **options)


def test_a_good_first_step_is_taken() -> None:
    """A quadratic whose minimizer is at alpha = 1 (a Newton step): one trial."""
    r = search(lambda a: 3.0 - 2.0 * a + a ** 2, slope=-2.0)
    assert (int(r.status), int(r.trials), float(r.alpha), float(r.f)) == (ARMIJO, 1, 1.0, 2.0)
    assert r.trace_alpha.tolist()[:1] == [1.0] and r.trace_f.tolist()[:1] == [2.0]


def test_quadratic_interpolation_finds_a_quadratic_minimizer() -> None:
    """The minimizer at alpha = 0.3: the first backtrack interpolates exactly
    (within the safeguards [0.1, 0.5] of the rejected step)."""
    def f(a: Any) -> Any:
        return 1.0 - 0.6 * a + a ** 2
    r = search(f, slope=-0.6)
    assert int(r.status) == ARMIJO and int(r.trials) == 2
    assert abs(float(r.alpha) - 0.3) <= 1e-15
    assert float(r.f) <= 1.0 + 1e-4 * float(r.alpha) * -0.6


def test_the_safeguards_bound_each_backtrack() -> None:
    """A minimizer far below the first step: every backtrack cuts by at most
    10 and at least 2, and the result satisfies Armijo."""
    def f(a: Any) -> Any:
        return -1e-3 * a + 0.5 * 1e4 * a ** 2           # minimizer 1e-7
    r = search(f, slope=-1e-3)
    tried = r.trace_alpha[:int(r.trials)]
    ratios = tried[1:] / tried[:-1]
    assert int(r.status) == ARMIJO and np.all(ratios >= 0.1 - 1e-15) and np.all(ratios <= 0.5 + 1e-15)
    assert float(r.f) <= 1e-4 * float(r.alpha) * -1e-3 + 1e-15


def test_cubic_interpolation_after_the_second_trial() -> None:
    """The step after two rejected trials minimizes the cubic through f0, the
    slope and both trials (checked against the cubic itself)."""
    f0, slope, a1, f1, a2, f2 = 1.0, -1.0, 1.0, 3.0, 0.4, 1.2
    step = float(next_step(np, f0, slope, a2, f2, a1, f1, (0.1, 0.5)))
    A = np.array([[a2 ** 3, a2 ** 2], [a1 ** 3, a1 ** 2]])
    a, b = np.linalg.solve(A, [f2 - f0 - slope * a2, f1 - f0 - slope * a1])
    minimizer = (-b + np.sqrt(b ** 2 - 3 * a * slope)) / (3 * a)
    assert abs(step - min(max(minimizer, 0.1 * a2), 0.5 * a2)) <= 1e-14


def test_non_finite_values_shrink_the_step() -> None:
    """f is infinite beyond 0.05: steps shrink by 10 until it is finite."""
    def f(a: Any) -> Any:
        return np.where(a > 0.05, np.inf, 1.0 - a + a ** 2)
    r = search(f, slope=-1.0)
    assert int(r.status) == ARMIJO and float(r.alpha) <= 0.05
    assert np.isinf(r.trace_f[0]) and abs(r.trace_alpha[1] - 0.1) <= 1e-15


def test_failure_after_max_trials() -> None:
    """A slope that lies (f increases): FAILED after max_trials, alpha the
    last step tried, f its value."""
    r = search(lambda a: 1.0 + a, slope=-1.0, max_trials=7)
    assert int(r.status) == FAILED and int(r.trials) == 7
    assert float(r.alpha) == r.trace_alpha[6] and float(r.f) == r.trace_f[6]
    assert LINE_SEARCH_STATUS[int(r.status)] == "failed"


@pytest.mark.parametrize("slope", [0.0, 1.0, np.nan, np.inf])
def test_a_direction_that_does_not_descend_is_not_searched(slope: float) -> None:
    r = search(lambda a: 1.0 - a, slope=slope)
    assert int(r.status) == NOT_DESCENT and int(r.trials) == 0
    assert float(r.alpha) == 1.0 and float(r.f) == 1.0 and float(r.point) == 0.0


def test_rounding_allowance() -> None:
    """Near convergence f changes by rounding only: a unit step that raises f
    by less than 10 eps max(1, |f0|) is accepted; by more, it is rejected,
    and a shorter step within the allowance is accepted."""
    eps = np.finfo(float).eps
    small = search(lambda a: 1.0 + 5.0 * eps * a, slope=-1e-20)
    large = search(lambda a: 1.0 + 20.0 * eps * a, slope=-1e-20)
    assert int(small.status) == ARMIJO and int(small.trials) == 1
    assert int(large.status) == ARMIJO and int(large.trials) == 2 and float(large.alpha) < 0.5


def test_the_step_cap_is_the_first_trial() -> None:
    r = search(lambda a: 3.0 - 2.0 * a + a ** 2, slope=-2.0, alpha0=0.25)
    assert (int(r.trials), float(r.alpha)) == (1, 0.25)


def test_strict_backend_agrees() -> None:
    """On the code path of a compiling backend (where-selects, a scan-built
    loop, cond evaluating both branches), the same search."""
    cases: list[tuple[Callable[[Any], Any], float]] = [
        (lambda a: 1.0 - 0.6 * a + a ** 2, -0.6),
        (lambda a: -1e-3 * a + 0.5 * 1e4 * a ** 2, -1e-3),
        (lambda a: 1.0 + a, -1.0)]
    for f, slope in cases:
        ref = search(f, slope)
        f0 = f(np.asarray(0.0))
        r = backtracking(strict_backend, along(f), strict(0.0), strict(f0), strict(slope), strict(1.0))
        assert int(plain(r.status)) == int(ref.status) and int(plain(r.trials)) == int(ref.trials)
        assert np.array_equal(np.asarray(plain(r.trace_alpha)), ref.trace_alpha)
        assert float(plain(r.alpha)) == float(ref.alpha)


def test_a_scan_built_loop_evaluates_no_skipped_trial() -> None:
    """With a loop built from scan (and a cond that runs one branch, as under
    compilation), phi is evaluated once per trial, not max_trials times."""
    def scan(step: Any, carry: Any, xs: Any, length: int) -> Any:
        for _ in range(length):
            carry, _ = step(carry, None)
        return carry, None
    backend = Backend(jit=lambda f: f, loop=loop_from_scan(scan, python_cond), cond=python_cond,
                      index_update=numpy_backend.index_update, xp=np, static_shapes=False)
    calls = [0]

    def f(a: Any) -> Any:
        calls[0] += 1
        return 1.0 - 0.6 * a + a ** 2
    r = backtracking(backend, along(f), np.asarray(0.0), 1.0, -0.6, 1.0, max_trials=30)
    assert int(r.trials) == 2 and calls[0] == 2
