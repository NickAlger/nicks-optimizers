"""The flexible subproblem solver: checked against dense linear algebra and
the Lanczos solver, and with preconditioners that are nonsymmetric,
nonlinear, stochastic or broken."""

from typing import Callable

import numpy as np
import pytest

from nicks_optimizers import (
    Array,
    Covector,
    Point,
    SubproblemStats,
    Vector,
    numpy_space,
    solve_sphere_trs,
    solve_tr_subproblem,
    solve_tr_subproblem_flexible,
    trust_region_newton,
)
from helpers import (
    dense_abs_m,
    dense_apply,
    dense_dual_norm,
    dense_solve,
    dense_vec_to_covec,
    full_space_kkt_error,
    matrix_with_spectrum,
    random_spd,
)
from problems import (
    jacobi_precond_solve,
    rosen_fun,
    rosen_grad,
    rosen_hessvec,
    smoothing_metric_apply,
    smoothing_metric_solve,
)


Precond = Callable[[Covector], Vector]


def problem(seed: int = 0, n: int = 120) -> tuple[Array, Array, Array, Array]:
    """(H indefinite, M SPD diagonal, g, Hs = H + 3 I) for the robustness tests."""
    rng = np.random.default_rng(seed)
    H = matrix_with_spectrum(np.concatenate([[-2.0, -0.5, -0.1], np.logspace(-1, 3, n - 3)]), rng)
    return H, np.diag(np.linspace(0.5, 2.0, n)), rng.standard_normal(n), H + 3.0 * np.eye(n)


def exact_model(H: Array, M: Array, g: Array, radius: float) -> float:
    Li = np.linalg.inv(np.linalg.cholesky(M))
    z, _ = solve_sphere_trs(Li @ H @ Li.T, Li @ g, radius)
    s = Li.T @ z
    return float(g @ s + 0.5 * s @ H @ s)


def flexible(H: Array, M: Array, g: Array, precond: Precond, radius: float, max_iter: int,
             resid_tol: float = 0.0, saddle_free: bool = False) -> tuple[Vector, SubproblemStats]:
    return solve_tr_subproblem_flexible(
        numpy_space, Covector(g), dense_vec_to_covec(H), dense_solve(M), dense_apply(M), precond,
        radius, resid_tol, max_iter, saddle_free=saddle_free)


def unconventional_preconditioners(Hs: Array, seed: int) -> dict[str, Precond]:
    """Maps covector -> vector that a Krylov method would normally reject."""
    n = Hs.shape[0]
    rng = np.random.default_rng(seed)
    tri = Hs.copy()
    tri[: n // 2, n // 2:] = 0.0                     # block lower triangular: nonsymmetric

    def inner_cg(r: Covector) -> Vector:             # CG on Hs z = r to 30%: nonlinear in r
        z, res = np.zeros(n), np.array(r, dtype=float)
        p, rr = res.copy(), float(res @ res)
        r0 = np.sqrt(rr)
        while np.sqrt(rr) > 0.3 * r0:
            Ap = Hs @ p
            a = rr / float(p @ Ap)
            z, res = z + a * p, res - a * Ap
            rr_new = float(res @ res)
            p, rr = res + (rr_new / rr) * p, rr_new
        return Vector(z)

    def noisy(r: Covector) -> Vector:                # exact solve with 30% random noise: stochastic
        z = np.linalg.solve(Hs, r)
        return Vector(z + 0.3 * np.linalg.norm(z) * rng.standard_normal(n) / np.sqrt(n))

    return {"block triangular": lambda r: Vector(np.linalg.solve(tri, r)),
            "inexact inner CG": inner_cg, "noisy": noisy}


def test_reproduces_lanczos_when_the_preconditioner_is_the_metric() -> None:
    """With P = M the residual-expansion subspace is the Lanczos space, so
    the model values agree step by step (plain and saddle-free, boundary
    active and inactive)."""
    H, M, g, _ = problem(1, 80)
    for radius in (0.3, 1e4):
        for saddle_free in (False, True):
            for k in (2, 6, 15, 30):
                _, lz = solve_tr_subproblem(numpy_space, Covector(g), dense_vec_to_covec(H),
                                            dense_solve(M), radius, 0.0, k, stall_tol=0.0,
                                            saddle_free=saddle_free)
                _, fx = flexible(H, M, g, dense_solve(M), radius, k, saddle_free=saddle_free)
                assert fx.krylov_steps == lz.krylov_steps == k
                assert abs(fx.model_value - lz.model_value) <= 1e-9 * abs(lz.model_value)


def test_full_space_solution() -> None:
    """Run to k = n with a separate (fixed SPD) P, it solves the trust-region
    problem: KKT and second-order conditions, for the plain and the
    saddle-free model; the reported model value and residual are the true
    ones."""
    rng = np.random.default_rng(3)
    n = 40
    H = matrix_with_spectrum(np.linspace(-2.0, 10.0, n), rng)
    P, M = random_spd(n, 1e2, rng), random_spd(n, 1e3, rng)
    g = rng.standard_normal(n)
    gn = dense_dual_norm(g, M)
    for radius in (0.05, 2.0, 1e4):
        for saddle_free in (False, True):
            Hm = dense_abs_m(H, M) if saddle_free else H
            s, st = flexible(H, M, g, dense_solve(P), radius, n, 1e-11 * gn, saddle_free)
            assert full_space_kkt_error(Hm, M, g, radius, s, st.lam) <= 1e-8
            model = float(g @ s + 0.5 * s @ Hm @ s)
            assert abs(st.model_value - model) <= 1e-10 * abs(model)
            actual = dense_dual_norm(g + Hm @ s + st.lam * M @ s, M)
            assert abs(st.resid - actual) <= 1e-10 * gn


def test_robust_to_unconventional_preconditioners() -> None:
    """Nonsymmetric, nonlinear and stochastic preconditioners: every step
    stays in the ball, beats the Cauchy point along -M^{-1} g, and reports
    its true model value and residual; within 40 steps the model is within
    1e-6 of the optimum (the Lanczos solver, by contrast, returns steps that
    can increase the model, or fails)."""
    H, M, g, Hs = problem(2)
    Minv_g = np.linalg.solve(M, g)
    for radius in (0.3, 30.0):
        m_star = exact_model(H, M, g, radius)
        d = -Minv_g / np.sqrt(g @ Minv_g)                 # Cauchy direction, unit M-norm
        slope, curv = float(g @ d), float(d @ H @ d)
        t = radius if curv <= 0.0 else min(-slope / curv, radius)
        cauchy = slope * t + 0.5 * curv * t ** 2
        for name, precond in unconventional_preconditioners(Hs, seed=4).items():
            for k in (1, 5, 40):
                s, st = flexible(H, M, g, precond, radius, k)
                model = float(g @ s + 0.5 * s @ H @ s)
                assert float(np.sqrt(s @ M @ s)) <= radius * (1.0 + 1e-10), name
                assert model <= cauchy + 1e-12 * abs(cauchy), name
                assert abs(st.model_value - model) <= 1e-9 * abs(model), name
                actual = dense_dual_norm(g + H @ s + st.lam * M @ s, M)
                assert abs(st.resid - actual) <= 1e-9 * max(actual, 1e-3), name
                if k == 40:
                    assert (model - m_star) / abs(m_star) <= 1e-6, (name, radius)


def test_falls_back_when_the_preconditioner_fails() -> None:
    """A preconditioner that returns NaN, zero, or a direction already in the
    subspace is replaced by M^{-1} R, the unpreconditioned direction, so the
    solve continues as Lanczos with P = M would; every replacement is
    counted."""
    H, M, g, _ = problem(5, 60)
    first_basis_vector = np.linalg.solve(M, g)
    broken: dict[str, Precond] = {
        "nan": lambda r: Vector(np.full_like(r, np.nan)),
        "zero": lambda r: Vector(np.zeros_like(r)),
        "stuck": lambda r: Vector(first_basis_vector.copy()),
    }
    for radius in (0.3, 1e4):
        for k in (5, 20):
            _, lz = solve_tr_subproblem(numpy_space, Covector(g), dense_vec_to_covec(H),
                                        dense_solve(M), radius, 0.0, k, stall_tol=0.0)
            for name, precond in broken.items():
                _, st = flexible(H, M, g, precond, radius, k)
                assert st.fallbacks == k - 1, name
                assert abs(st.model_value - lz.model_value) <= 1e-9 * abs(lz.model_value), name


def test_flexible_in_the_outer_loop() -> None:
    """krylov="flexible" with a separate Jacobi preconditioner and a
    smoothing metric converges on Rosenbrock, plain and saddle-free, and the
    work counts are the calls made; it needs a separate preconditioner and
    metric_apply."""
    x0 = Point(np.tile([-1.2, 1.0], 25))
    for saddle_free in (False, True):
        counts = {"P": 0}

        def precond(x: Point, w: Covector) -> Vector:
            counts["P"] += 1
            return jacobi_precond_solve(x, w)

        x, summary = trust_region_newton(
            rosen_fun, rosen_grad, rosen_hessvec, x0, metric_solve=smoothing_metric_solve,
            metric_apply=smoothing_metric_apply, precond_solve=precond, krylov="flexible",
            saddle_free=saddle_free, max_krylov=100, max_iter=1000)
        assert summary.reason == "converged"
        assert {r.subproblem.krylov for r in summary.iterations} == {"flexible"}
        assert summary.precond_solve_calls == counts["P"]
    with pytest.raises(ValueError):
        trust_region_newton(rosen_fun, rosen_grad, rosen_hessvec, x0, krylov="flexible")
    with pytest.raises(ValueError):
        trust_region_newton(rosen_fun, rosen_grad, rosen_hessvec, x0, krylov="flexible",
                            metric_solve=smoothing_metric_solve, precond_solve=jacobi_precond_solve)
