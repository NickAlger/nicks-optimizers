"""The outer trust-region loop: operators, point-dependent metrics, the
relative convergence test, and convergence on test problems."""

from typing import Callable

import numpy as np

from nicks_optimizers import (
    Array,
    Covector,
    CovectorBlock,
    DualInnerProduct,
    InnerProduct,
    Point,
    Vector,
    VectorBlock,
    covector_norm,
    euclidean_dual_inner,
    euclidean_inner,
    forcing_term,
    history_array,
    numpy_space,
    reduction_ratio,
    solve_tr_subproblem,
    trust_region_newton,
)
from nicks_optimizers.trust_region import update_radius
from helpers import (
    Rng,
    dense_from_operator,
    report,
    without_times,
)
from problems import (
    METRICS,
    jacobi_precond_solve,
    rosen_fun,
    rosen_grad,
    rosen_hessvec,
    smoothing_dual_inner,
    smoothing_inner,
    weighted_dual_inner,
    weighted_inner,
    well_fun,
    well_grad,
    well_hessvec,
    well_precond_solve,
)


def check_problem_operators(rng: Rng) -> bool:
    print("Problem operators: finite differences, and duality of the inner-product pairs")
    n = 30
    x = Point(rng.standard_normal(n))
    v = Vector(rng.standard_normal(n))
    eps = 1e-6
    ok = True
    for name, f, gr, hv in (("Rosenbrock", rosen_fun, rosen_grad, rosen_hessvec),
                            ("double well", well_fun, well_grad, well_hessvec)):
        fd_g = (f(Point(x + eps * v)) - f(Point(x - eps * v))) / (2 * eps)
        fd_h = (gr(Point(x + eps * v)) - gr(Point(x - eps * v))) / (2 * eps)
        ok &= report(f"{name} gradient", abs(fd_g - numpy_space.pair(gr(x), v)) / abs(fd_g), 1e-6)
        ok &= report(f"{name} Hessian-vector product",
                     float(np.linalg.norm(fd_h - hv(x, v)) / np.linalg.norm(fd_h)), 1e-6)
    # The two inner products of each metric must be exact duals: their Gram
    # matrices on the coordinate basis are M and M^{-1}, so they multiply to I.
    I = np.eye(n)
    for name, inner, dual_inner in METRICS:
        GM = np.column_stack([inner(x, VectorBlock(I), Vector(I[:, j])) for j in range(n)])
        GD = np.column_stack([dual_inner(x, CovectorBlock(I), Covector(I[:, j])) for j in range(n)])
        ok &= report(f"{name}: inner products are exact duals (Gram_M Gram_M^-1 = I)",
                     float(np.linalg.norm(GM @ GD - I)), 1e-12)
    return ok


def reference_iterations(x0: Array, inner: InnerProduct, dual_inner: DualInnerProduct,
                         n_iter: int, radius: float = 1.0) -> Point:
    """Outer iterations written out by hand, with every operator and inner
    product evaluated at the current point; same defaults as trust_region_newton."""
    sp = numpy_space
    x = Point(np.array(x0, dtype=float))
    fx, g = rosen_fun(x), rosen_grad(x)
    gnorm = gnorm0 = covector_norm(sp, lambda P_, p: dual_inner(x, P_, p), g)
    for _ in range(n_iter):
        s, st = solve_tr_subproblem(
            sp, g,
            lambda v: rosen_hessvec(x, v),
            lambda w: jacobi_precond_solve(x, w),
            lambda U, v: inner(x, U, v),
            lambda P_, p: dual_inner(x, P_, p),
            radius, forcing_term(gnorm, gnorm0, 0.5) * gnorm, 50, g_dual_norm=gnorm)
        x_trial = Point(x + s)
        f_trial = rosen_fun(x_trial)
        rho = reduction_ratio(fx, f_trial, -st.model_value)
        if rho > 1e-4:
            x, fx, g = x_trial, f_trial, rosen_grad(x_trial)
            gnorm = covector_norm(sp, lambda P_, p: dual_inner(x, P_, p), g)
        radius = update_radius(radius, rho, st.step_norm, 1e4)
    return x


def check_metric_uses_current_point() -> bool:
    print("Point-dependent inner products are evaluated at the current iterate")
    n, n_iter = 20, 8
    x0 = Point(np.tile([-1.2, 1.0], n // 2))
    x_method, _ = trust_region_newton(
        rosen_fun, rosen_grad, rosen_hessvec, jacobi_precond_solve,
        weighted_inner, weighted_dual_inner, x0, gtol=0.0, max_iter=n_iter)
    x_ref = reference_iterations(x0, weighted_inner, weighted_dual_inner, n_iter)
    x_frozen, _ = trust_region_newton(
        rosen_fun, rosen_grad, rosen_hessvec, jacobi_precond_solve,
        lambda x, U, v: weighted_inner(x0, U, v),
        lambda x, P_, p: weighted_dual_inner(x0, P_, p),
        x0, gtol=0.0, max_iter=n_iter)
    ok = report("matches hand-written loop with M(x_k)", float(np.max(np.abs(x_method - x_ref))), 1e-12)
    diff = float(np.max(np.abs(x_method - x_frozen)))
    differs = diff > 1e-6
    print(f"  [{'PASS' if differs else 'FAIL'}] a run with M frozen at x0 differs "
          f"(so the dependence is exercised): {diff:.2e}")
    return ok and differs


def check_gnorm0_and_rtol() -> bool:
    print("Relative convergence test ||g|| <= max(gtol, rtol * gnorm0)  (identity metric)")
    n, rtol = 20, 1e-4
    x0 = Point(np.tile([-1.2, 1.0], n // 2))
    g0 = float(np.linalg.norm(rosen_grad(x0)))
    thr = rtol * g0

    def run(x_start: Point, gnorm0: float | None = None) -> tuple[Point, Array]:
        x, summary = trust_region_newton(
            rosen_fun, rosen_grad, rosen_hessvec, jacobi_precond_solve,
            euclidean_inner, euclidean_dual_inner,
            x_start, gtol=0.0, rtol=rtol, gnorm0=gnorm0, max_iter=1000)
        return x, history_array(summary)

    def stopped_at_first_below(x: Point, hist: Array) -> bool:
        return bool(np.linalg.norm(rosen_grad(x)) <= thr and np.all(hist[:, 2] > thr))

    x_def, h_def = run(x0)
    x_exp, h_exp = run(x0, gnorm0=g0)
    ok = report("default gnorm0: stops at first iterate with ||g|| <= rtol ||g(x0)||",
                0.0 if stopped_at_first_below(x_def, h_def) else 1.0, 0.0)
    ok &= report("explicit gnorm0 = ||g(x0)|| reproduces the default run",
                 float(np.max(np.abs(x_def - x_exp))), 0.0)
    x_mid, _ = trust_region_newton(
        rosen_fun, rosen_grad, rosen_hessvec, jacobi_precond_solve,
        euclidean_inner, euclidean_dual_inner, x0, gtol=0.0, max_iter=40)
    x_restart, h_restart = run(x_mid, gnorm0=g0)
    _, h_naive = run(x_mid)
    ok &= report("restart with gnorm0 = ||g(x0)|| stops at the original threshold",
                 0.0 if stopped_at_first_below(x_restart, h_restart) else 1.0, 0.0)
    print(f"         (restart: {len(h_restart)} iterations with gnorm0 = ||g(x0)||, "
          f"{len(h_naive)} if gnorm0 defaults to ||g(x_restart)|| = "
          f"{np.linalg.norm(rosen_grad(x_mid)):.1e})")
    return ok


def run_and_report(label: str, fun: Callable[[Point], float], grad: Callable[[Point], Covector],
                   hessvec: Callable[[Point, Vector], Covector],
                   precond_solve: Callable[[Point, Covector], Vector],
                   inner: InnerProduct, dual_inner: DualInnerProduct,
                   x0: Point, saddle_free: bool) -> bool:
    x, summary = trust_region_newton(
        fun, grad, hessvec, precond_solve, inner, dual_inner, x0,
        saddle_free=saddle_free, gtol=1e-8, max_iter=1000, max_lanczos=100)
    hist = history_array(summary)
    gfinal = covector_norm(numpy_space, lambda P_, p: dual_inner(x, P_, p), grad(x))
    lam_min = float(np.linalg.eigvalsh(dense_from_operator(lambda v: hessvec(x, v), x.shape[0]))[0])
    tail = ", ".join(f"{gn:.0e}" for gn in hist[-4:, 2])
    print(f"  {label}: {len(hist)} its, {int(hist[:, 5].sum())} Hv, f = {fun(x):.4f}, "
          f"max neg. Ritz = {int(hist[:, 7].max())}, |g| tail: {tail}")
    ok = report(f"{label}: final |g|_M^-1", gfinal, 1e-8)
    ok &= report(f"{label}: Hessian positive definite (-lam_min)", -lam_min, 0.0)
    return ok


def demo_rosenbrock() -> bool:
    print("Demo: chained Rosenbrock, n=50, Jacobi preconditioner, plain and saddle-free")
    print("  (converges to the known local minimizer near x = (-0.99, 1, ...), f = 3.99)")
    x0 = Point(np.tile([-1.2, 1.0], 25))
    ok = True
    for name, inner, dual_inner in METRICS:
        for saddle_free in (False, True):
            label = f"{name}, {'saddle-free' if saddle_free else 'plain'}"
            ok &= run_and_report(label, rosen_fun, rosen_grad, rosen_hessvec, jacobi_precond_solve,
                                 inner, dual_inner, x0, saddle_free)
    return ok


def demo_double_well(rng: Rng) -> bool:
    print("Demo: chain of double wells, n=50, started next to the local maximum at 0, smoothing ball")
    x0 = Point(0.01 * rng.standard_normal(50))
    ok = True
    for saddle_free in (False, True):
        label = "saddle-free" if saddle_free else "plain"
        ok &= run_and_report(label, well_fun, well_grad, well_hessvec, well_precond_solve,
                             smoothing_inner, smoothing_dual_inner, x0, saddle_free)
    return ok


def test_problem_operators() -> None:
    assert check_problem_operators(np.random.default_rng(5))


def test_metric_uses_current_point() -> None:
    assert check_metric_uses_current_point()


def test_gnorm0_and_rtol() -> None:
    assert check_gnorm0_and_rtol()


def test_rosenbrock_converges() -> None:
    assert demo_rosenbrock()


def test_double_well_converges() -> None:
    assert demo_double_well(np.random.default_rng(6))


def test_operators_at_matches_and_runs_once_per_point() -> None:
    """operators_at(x) gives the same run as hessvec/precond_solve, and is
    called exactly once per iterate at which a subproblem is solved (rejected
    steps reuse the operators)."""
    x0 = Point(np.tile([-1.2, 1.0], 25))
    _, reference = trust_region_newton(rosen_fun, rosen_grad, rosen_hessvec, jacobi_precond_solve,
                                       euclidean_inner, euclidean_dual_inner, x0)
    points: list[Point] = []

    def operators_at(x: Point) -> tuple[Callable[[Vector], Covector], Callable[[Covector], Vector]]:
        points.append(x)
        return (lambda v: rosen_hessvec(x, v)), (lambda w: jacobi_precond_solve(x, w))

    x, summary = trust_region_newton(rosen_fun, rosen_grad, None, None, euclidean_inner,
                                     euclidean_dual_inner, x0, operators_at=operators_at)
    assert without_times(summary.iterations) == without_times(reference.iterations)
    records = summary.iterations
    new_points = 1 + sum(records[i - 1].accepted for i in range(1, len(records)))
    assert any(not r.accepted for r in records)             # rejections did happen
    assert len(points) == new_points
    assert len({id(p) for p in points}) == len(points)


def test_operator_arguments_are_checked() -> None:
    x0 = Point(np.zeros(4))
    for hv, ps, ops in ((None, jacobi_precond_solve, None), (rosen_hessvec, None, None),
                        (rosen_hessvec, jacobi_precond_solve,
                         lambda x: ((lambda v: rosen_hessvec(x, v)), (lambda w: w)))):
        try:
            trust_region_newton(rosen_fun, rosen_grad, hv, ps, euclidean_inner,
                                euclidean_dual_inner, x0, operators_at=ops)
        except ValueError:
            continue
        raise AssertionError("invalid operator arguments were accepted")


def test_trust_region_in_the_preconditioner_norm() -> None:
    """Passing no inner products means M = P. With the diagonal (Jacobi)
    preconditioner of the Rosenbrock problem, (u, v)_P can also be written
    out explicitly: both runs take the same steps, and the M = P run makes no
    inner-product calls."""
    from problems import rosen_hess_diag
    x0 = Point(np.tile([-1.2, 1.0], 25))

    def jacobi_diagonal(x: Point) -> Array:
        """The P(x) of jacobi_precond_solve."""
        diagonal: Array = np.maximum(np.abs(rosen_hess_diag(x)), 1.0)
        return diagonal

    def p_inner(x: Point, U: VectorBlock, v: Vector) -> Array:
        return U.T @ (jacobi_diagonal(x) * v)

    def p_dual_inner(x: Point, Psi: CovectorBlock, phi: Covector) -> Array:
        return Psi.T @ (phi / jacobi_diagonal(x))

    _, explicit = trust_region_newton(rosen_fun, rosen_grad, rosen_hessvec, jacobi_precond_solve,
                                      p_inner, p_dual_inner, x0)
    _, matched = trust_region_newton(rosen_fun, rosen_grad, rosen_hessvec, jacobi_precond_solve,
                                     None, None, x0)
    a, b = explicit.iterations, matched.iterations
    assert len(a) == len(b) and matched.reason == explicit.reason == "converged"
    assert [(r.accepted, r.subproblem.lanczos_steps) for r in a] == \
        [(r.accepted, r.subproblem.lanczos_steps) for r in b]
    assert max(abs(r.f - s.f) / max(1.0, abs(r.f)) for r, s in zip(a, b)) < 1e-10
    assert max(abs(r.gnorm - s.gnorm) / r.gnorm for r, s in zip(a, b)) < 1e-6
    assert matched.inner_calls == 0 and matched.dual_inner_calls == 0
