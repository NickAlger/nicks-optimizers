"""The forcing power: the Krylov tolerance min(forcing_max, (||g|| /
||g0||)^forcing_power) of both Newton optimizers (dev/archive/gauss_newton_switch_plan.md,
phase 1)."""

import dataclasses
from typing import Any

import numpy as np
import pytest

from helpers import without_times
from nicks_optimizers import forcing_term, line_search_newton, trust_region_newton
from problems import rosen_fun, rosen_grad, rosen_hessvec
from strict_backend import strict_backend

OPTIMIZERS: dict[str, Any] = {"trust region": trust_region_newton, "line search": line_search_newton}


def rosenbrock_x0() -> Any:
    return np.tile([-1.2, 1.0], 10)


def test_criteria_forcing_term() -> None:
    assert forcing_term(1.0, 4.0, 0.9) == 0.5                       # the square root by default
    assert forcing_term(1.0, 4.0, 0.9, power=0.5) == 0.5
    assert forcing_term(1.0, 4.0, 0.9, power=1.0) == 0.25
    assert forcing_term(1.0, 4.0, 0.9, power=2.0) == 0.0625
    assert forcing_term(3.0, 4.0, 0.5, power=1.0) == 0.5            # capped at forcing_max
    assert forcing_term(1.0, 0.0, 0.5, power=1.0) == 0.5            # gnorm0 = 0


@pytest.mark.parametrize("method", OPTIMIZERS)
@pytest.mark.parametrize("power", [0.75, 1.0, 2.0])
def test_recorded_forcing_follows_the_power(method: str, power: float) -> None:
    _, summary = OPTIMIZERS[method](rosen_fun, rosen_grad, rosen_hessvec, rosenbrock_x0(),
                                    forcing_power=power)
    assert summary.reason == "converged"
    for r in summary.iterations:
        assert r.forcing == pytest.approx(min(0.5, r.gnorm_ratio ** power), rel=1e-14)
        assert r.subproblem.resid_tol == pytest.approx(r.forcing * r.gnorm, rel=1e-14)


@pytest.mark.parametrize("method", OPTIMIZERS)
def test_explicit_square_root_is_the_default(method: str) -> None:
    opt = OPTIMIZERS[method]
    default = opt(rosen_fun, rosen_grad, rosen_hessvec, rosenbrock_x0())
    explicit = opt(rosen_fun, rosen_grad, rosen_hessvec, rosenbrock_x0(), forcing_power=0.5)
    assert without_times(default[1].iterations) == without_times(explicit[1].iterations)
    assert default[0].tobytes() == explicit[0].tobytes()


@pytest.mark.parametrize("method", OPTIMIZERS)
def test_larger_power_solves_more_tightly(method: str) -> None:
    """From near the solution, power 1 asks more of each Krylov solve: more
    Hessian products, fewer outer iterations. (From far away the paths
    differ and may end in different local minima of chained Rosenbrock.)"""
    opt = OPTIMIZERS[method]
    x0 = np.ones(20) + 0.05 * np.random.default_rng(0).standard_normal(20)
    _, half = opt(rosen_fun, rosen_grad, rosen_hessvec, x0, rtol=1e-12)
    _, one = opt(rosen_fun, rosen_grad, rosen_hessvec, x0, rtol=1e-12, forcing_power=1.0)
    assert half.reason == one.reason == "converged"
    assert len(one.iterations) < len(half.iterations)
    assert one.hessvec_calls > half.hessvec_calls


@pytest.mark.parametrize("method", OPTIMIZERS)
@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_power_raises(method: str, bad: float) -> None:
    with pytest.raises(ValueError, match="forcing_power"):
        OPTIMIZERS[method](rosen_fun, rosen_grad, rosen_hessvec, rosenbrock_x0(), forcing_power=bad)


@pytest.mark.parametrize("method", OPTIMIZERS)
def test_strict_backend(method: str) -> None:
    """The power through the backend: the same records on the strict backend
    (whose arrays reject Python control flow on values)."""
    def strict_rosen_fun(x: Any) -> Any:
        return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)

    opt = OPTIMIZERS[method]
    _, a = opt(rosen_fun, rosen_grad, rosen_hessvec, rosenbrock_x0(), forcing_power=1.0)
    _, b = opt(strict_rosen_fun, rosen_grad, rosen_hessvec, rosenbrock_x0(), forcing_power=1.0,
               backend=strict_backend)
    assert [(r.accepted, r.subproblem.krylov_steps) for r in a.iterations] == \
        [(r.accepted, r.subproblem.krylov_steps) for r in b.iterations]
    assert all(r.forcing == pytest.approx(s.forcing, rel=1e-12)
               for r, s in zip(a.iterations, b.iterations))
    assert dataclasses.replace(a, iterations=(), elapsed=0.0).reason == b.reason
