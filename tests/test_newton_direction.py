"""Newton directions for the line search (dev/line_search_plan.md, phase 3):
the Krylov solvers in line-search mode, against dense references."""

from typing import Any, Callable, Optional

import numpy as np
import pytest

from helpers import (
    dense_abs_m,
    dense_apply,
    dense_solve,
    dense_vec_to_covec,
    matrix_with_spectrum,
    random_spd,
)
from nicks_optimizers import Covector, array_space, numpy_backend, numpy_space
from nicks_optimizers import (
    solve_newton_direction,
    solve_newton_direction_cg,
    solve_newton_direction_flexible,
)
from nicks_optimizers.trust_region_subproblem import solve_tr_subproblem
from strict_backend import plain, strict, strict_backend

SOLVERS = ["lanczos", "lanczos, separate P", "cg", "flexible"]
SADDLE_FREE_SOLVERS = ["lanczos", "lanczos, separate P", "flexible"]


def direction(solver: str, H: Any, P: Any, M: Any, g: Any, *, resid_tol: float = 0.0,
              max_iter: Optional[int] = None, saddle_free: bool = False, backend: Any = None,
              capacity: Optional[int] = None, count: Optional[dict[str, int]] = None) -> Any:
    """A direction from one of the solvers; the operators are traceable
    (products with precomputed inverses) so the strict backend can run them."""
    n = len(g)
    calls = count if count is not None else {}

    def op(name: str, A: Any) -> Callable[[Any], Any]:
        def apply(v: Any) -> Any:
            calls[name] = calls.get(name, 0) + 1
            return A @ v
        return apply
    hv, Ms, Ma, Ps = op("hessvec", H), op("metric_solve", np.linalg.inv(M)), op("metric_apply", M), \
        op("precond_solve", np.linalg.inv(P))
    bk = numpy_backend if backend is None else backend
    space = numpy_space if backend is None else array_space(bk)
    gg = strict(g) if bk is strict_backend else g
    k = n if max_iter is None else max_iter
    if solver == "cg":
        return solve_newton_direction_cg(space, gg, hv, Ms, resid_tol, k, backend=bk)
    if solver == "flexible":
        return solve_newton_direction_flexible(space, gg, hv, Ms, Ma, Ps, resid_tol, k,
                                               saddle_free=saddle_free, backend=bk, capacity=capacity)
    separate = dict(precond_solve=Ps, metric_apply=Ma) if solver == "lanczos, separate P" else {}
    return solve_newton_direction(space, gg, hv, Ms, resid_tol, k, saddle_free=saddle_free,
                                  backend=bk, capacity=capacity, **separate)  # type: ignore[arg-type]


def operators(solver: str, n: int, rng: np.random.Generator) -> tuple[Any, Any]:
    """(P, M): P = M for CG and plain Lanczos, separate SPD matrices otherwise."""
    M = random_spd(n, 10.0, rng)
    return (M, M) if solver in ("cg", "lanczos") else (random_spd(n, 30.0, rng), M)


@pytest.mark.parametrize("solver", SOLVERS)
def test_positive_definite_problems_give_the_newton_step(solver: str) -> None:
    rng = np.random.default_rng(1)
    n = 30
    H = random_spd(n, 1e3, rng)
    P, M = operators(solver, n, rng)
    g = rng.standard_normal(n)
    # (CG has no reorthogonalization, so in floating point it may need more than n steps)
    d, stats = direction(solver, H, P, M, g, resid_tol=1e-12 * np.linalg.norm(g),
                         max_iter=10 * n if solver == "cg" else n)
    exact = -np.linalg.solve(H, g)
    assert np.linalg.norm(d - exact) <= 1e-8 * np.linalg.norm(exact)
    assert stats.stop_reason in ("tol", "breakdown") and stats.lam == 0.0 and g @ d < 0.0
    assert abs(stats.step_norm - np.sqrt(d @ M @ d)) <= 1e-10 * stats.step_norm
    assert abs(stats.model_value - (g @ d + 0.5 * d @ H @ d)) <= 1e-10 * abs(stats.model_value)


def indefinite_problem(n: int, rng: np.random.Generator) -> tuple[Any, Any]:
    """Mostly positive curvature with a few negative eigenvalues: a solve sees
    positive curvature first and truncates after some steps."""
    H = matrix_with_spectrum(np.concatenate([np.linspace(1.0, 50.0, n - 3), [-0.5, -2.0, -4.0]]), rng)
    return H, rng.standard_normal(n)


@pytest.mark.parametrize("solver", SOLVERS)
def test_truncation_returns_the_last_positive_curvature_direction(solver: str) -> None:
    """The direction of a solve that truncates at step K is the one a solve
    limited to K - 1 steps returns."""
    rng = np.random.default_rng(2)
    n = 40
    H, g = indefinite_problem(n, rng)
    P, M = operators(solver, n, rng)
    d, stats = direction(solver, H, P, M, g)
    K = stats.krylov_steps
    assert stats.stop_reason == "negative" and K > 1 and stats.n_negative >= 1
    shorter, short_stats = direction(solver, H, P, M, g, max_iter=K - 1)
    assert short_stats.stop_reason == "max"
    assert np.linalg.norm(d - shorter) <= 1e-10 * np.linalg.norm(shorter)
    assert abs(stats.model_value - short_stats.model_value) <= 1e-10 * abs(short_stats.model_value)
    assert abs(stats.resid - short_stats.resid) <= 1e-8 * short_stats.resid and g @ d < 0.0


@pytest.mark.parametrize("solver", SOLVERS)
def test_negative_curvature_at_the_first_step_gives_steepest_descent(solver: str) -> None:
    n = 6
    H = np.diag([-2.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    P = np.diag([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]) if solver != "cg" and solver != "lanczos" else np.eye(n)
    M = np.eye(n) if solver != "flexible" else np.diag(np.linspace(1.0, 2.0, n))
    g = np.eye(n)[0]                                  # P^{-1} g and M^{-1} g along e_0: curvature < 0
    d, stats = direction(solver, H, P, M, g)
    start = np.linalg.solve(M if solver == "flexible" else P, g)
    assert np.allclose(d, -start, rtol=1e-14, atol=0.0)
    assert stats.stop_reason == "negative" and stats.krylov_steps == 1 and np.isnan(stats.resid)
    assert abs(stats.model_value - (g @ d + 0.5 * d @ H @ d)) <= 1e-14
    assert abs(stats.step_norm - np.sqrt(d @ M @ d)) <= 1e-14


@pytest.mark.parametrize("solver", SADDLE_FREE_SOLVERS)
def test_saddle_free_directions(solver: str) -> None:
    """Over the whole space, the saddle-free direction is -|H|_M^{-1} g (the
    floor does not bind here); a descent direction, no truncation."""
    rng = np.random.default_rng(3)
    n = 25
    H, g = indefinite_problem(n, rng)
    P, M = operators(solver, n, rng)
    d, stats = direction(solver, H, P, M, g, resid_tol=1e-11 * np.linalg.norm(g), saddle_free=True)
    exact = -np.linalg.solve(dense_abs_m(H, M), g)
    assert stats.stop_reason in ("tol", "breakdown", "max") and g @ d < 0.0
    assert np.linalg.norm(d - exact) <= 1e-7 * np.linalg.norm(exact)
    assert stats.n_negative == 3 and stats.model_value > stats.true_model_value    # |theta| >= theta


@pytest.mark.parametrize("seed", range(6))
def test_every_direction_descends(seed: int) -> None:
    rng = np.random.default_rng(10 + seed)
    n = 20
    H = matrix_with_spectrum(rng.uniform(-5.0, 10.0, n), rng)
    g = rng.standard_normal(n)
    for solver in SOLVERS:
        P, M = operators(solver, n, rng)
        for saddle_free in ((False, True) if solver in SADDLE_FREE_SOLVERS else (False,)):
            for resid_tol in (0.0, 0.3 * np.linalg.norm(g)):
                d, stats = direction(solver, H, P, M, g, resid_tol=resid_tol, saddle_free=saddle_free)
                assert g @ d < 0.0, (solver, saddle_free, stats.stop_reason)
                assert stats.model_value < 0.0


@pytest.mark.parametrize("solver", ["lanczos", "lanczos, separate P", "flexible"])
@pytest.mark.parametrize("saddle_free", [False, True])
def test_capacity_changes_nothing(solver: str, saddle_free: bool) -> None:
    rng = np.random.default_rng(4)
    n = 30
    H, g = indefinite_problem(n, rng)
    P, M = operators(solver, n, rng)
    ref, ref_stats = direction(solver, H, P, M, g, saddle_free=saddle_free)
    for capacity in (1, 2, 3):
        d, stats = direction(solver, H, P, M, g, saddle_free=saddle_free, capacity=capacity)
        assert d.tobytes() == ref.tobytes() and repr(stats) == repr(ref_stats)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("saddle_free", [False, True])
def test_strict_backend_agrees(solver: str, saddle_free: bool) -> None:
    """The code path of a compiling backend: array_space, padded projected
    problems, bases that grow from room for one vector. (The problems are
    solved in few steps: CG without reorthogonalization amplifies rounding
    differences by about ten per step once it loses orthogonality.)"""
    if saddle_free and solver not in SADDLE_FREE_SOLVERS:
        pytest.skip("CG has no saddle-free model")
    rng = np.random.default_rng(5)
    n = 20
    for H, g in (indefinite_problem(n, rng), (random_spd(n, 10.0, rng), rng.standard_normal(n))):
        P, M = operators(solver, n, rng)
        ref, ref_stats = direction(solver, H, P, M, g, saddle_free=saddle_free,
                                   resid_tol=1e-6 * np.linalg.norm(g))
        for capacity in (1, 2, 50):
            d, stats = direction(solver, H, P, M, g, saddle_free=saddle_free, backend=strict_backend,
                                 capacity=capacity, resid_tol=1e-6 * np.linalg.norm(g))
            assert (stats.krylov_steps, stats.stop_reason) == (ref_stats.krylov_steps,
                                                               ref_stats.stop_reason)
            tol = 1e-6 if solver == "cg" else 1e-9
            assert np.linalg.norm(np.asarray(plain(d)) - ref) <= tol * np.linalg.norm(ref)


@pytest.mark.parametrize("solver", SOLVERS)
def test_operator_calls(solver: str) -> None:
    """The calls made are the statistics' counts; Lanczos with a separate P
    and truncation keeps no Gram matrix: one product with M in all (for
    ||d||_M), where the trust-region solver makes one per step."""
    rng = np.random.default_rng(6)
    n = 30
    H, g = indefinite_problem(n, rng)
    P, M = operators(solver, n, rng)
    count: dict[str, int] = {}
    _, stats = direction(solver, H, P, M, g, count=count)
    assert (count.get("hessvec", 0), count.get("metric_solve", 0), count.get("metric_apply", 0),
            count.get("precond_solve", 0)) == (stats.hessvec_calls, stats.metric_solve_calls,
                                               stats.metric_apply_calls, stats.precond_solve_calls)
    if solver == "lanczos, separate P":
        assert stats.metric_apply_calls == 1
        # the residual is measured at every step but the truncating one
        assert stats.metric_solve_calls == stats.krylov_steps - 1
        _, tr = solve_tr_subproblem(numpy_space, Covector(g), dense_vec_to_covec(H), dense_solve(M),
                                    1e3, 0.0, n, precond_solve=dense_solve(P),
                                    metric_apply=dense_apply(M))
        assert tr.metric_apply_calls >= tr.krylov_steps
