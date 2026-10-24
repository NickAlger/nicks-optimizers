"""The outer trust-region loop: operators, point-dependent metrics, the
relative convergence test and its norm, and convergence on test problems."""

from typing import Callable, Literal

import numpy as np

from nicks_optimizers import (
    Array,
    Covector,
    IterationInfo,
    MetricApply,
    MetricSolve,
    Point,
    PointOperators,
    Vector,
    covector_norm,
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
    rosen_hess_diag,
    rosen_hessvec,
    smoothing_metric_apply,
    smoothing_metric_solve,
    weighted_metric_apply,
    weighted_metric_solve,
    well_fun,
    well_grad,
    well_hessvec,
    well_precond_solve,
)


def check_problem_operators(rng: Rng) -> bool:
    print("Problem operators: finite differences, and consistency of the metric pairs")
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
    # metric_apply and metric_solve of each metric must be inverse to each other.
    I = np.eye(n)
    for name, metric_solve, metric_apply in METRICS:
        Mdense = dense_from_operator(lambda v: metric_apply(x, v), n)
        Minv = np.column_stack([metric_solve(x, Covector(I[:, j])) for j in range(n)])
        ok &= report(f"{name}: metric_apply and metric_solve are inverses",
                     float(np.linalg.norm(Mdense @ Minv - I)), 1e-12)
        ok &= report(f"{name}: M symmetric", float(np.linalg.norm(Mdense - Mdense.T)), 1e-12)
    return ok


def reference_iterations(x0: Array, metric_solve: MetricSolve, metric_apply: MetricApply,
                         n_iter: int, radius: float = 1.0) -> Point:
    """Outer iterations written out by hand, with every operator evaluated at
    the current point and the Jacobi preconditioner separate from M; same
    defaults as trust_region_newton."""
    sp = numpy_space
    x = Point(np.array(x0, dtype=float))
    fx, g = rosen_fun(x), rosen_grad(x)
    gnorm = gnorm0 = covector_norm(sp, lambda w: metric_solve(x, w), g)
    for _ in range(n_iter):
        s, st = solve_tr_subproblem(
            sp, g, lambda v: rosen_hessvec(x, v), lambda w: metric_solve(x, w),
            radius, forcing_term(gnorm, gnorm0, 0.5) * gnorm, 50,
            precond_solve=lambda w: jacobi_precond_solve(x, w),
            metric_apply=lambda v: metric_apply(x, v))
        x_trial = Point(x + s)
        f_trial = rosen_fun(x_trial)
        rho = reduction_ratio(fx, f_trial, -st.model_value)
        if rho > 1e-4:
            x, fx, g = x_trial, f_trial, rosen_grad(x_trial)
            gnorm = covector_norm(sp, lambda w: metric_solve(x, w), g)
        radius = update_radius(radius, rho, st.step_norm, 1e4)
    return x


def check_metric_uses_current_point() -> bool:
    print("A point-dependent metric is evaluated at the current iterate")
    n, n_iter = 20, 8
    x0 = Point(np.tile([-1.2, 1.0], n // 2))
    x_method, _ = trust_region_newton(
        rosen_fun, rosen_grad, rosen_hessvec, x0, metric_solve=weighted_metric_solve,
        metric_apply=weighted_metric_apply, precond_solve=jacobi_precond_solve, max_iter=n_iter)
    x_ref = reference_iterations(x0, weighted_metric_solve, weighted_metric_apply, n_iter)
    x_frozen, _ = trust_region_newton(
        rosen_fun, rosen_grad, rosen_hessvec, x0,
        metric_solve=lambda x, w: weighted_metric_solve(x0, w),
        metric_apply=lambda x, v: weighted_metric_apply(x0, v),
        precond_solve=jacobi_precond_solve, max_iter=n_iter)
    ok = report("matches hand-written loop with M(x_k)", float(np.max(np.abs(x_method - x_ref))), 1e-12)
    diff = float(np.max(np.abs(x_method - x_frozen)))
    differs = diff > 1e-6
    print(f"  [{'PASS' if differs else 'FAIL'}] a run with M frozen at x0 differs "
          f"(so the dependence is exercised): {diff:.2e}")
    return ok and differs


def check_gnorm0_and_rtol() -> bool:
    print("Relative convergence test ||g|| <= max(gtol, rtol * gnorm0)  (default metric M = I)")
    n, rtol = 20, 1e-4
    x0 = Point(np.tile([-1.2, 1.0], n // 2))
    g0 = float(np.linalg.norm(rosen_grad(x0)))
    thr = rtol * g0

    def run(x_start: Point, gnorm0: float | None = None) -> tuple[Point, Array]:
        x, summary = trust_region_newton(
            rosen_fun, rosen_grad, rosen_hessvec, x_start, precond_solve=jacobi_precond_solve,
            rtol=rtol, gnorm0=gnorm0, max_iter=1000)
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
        rosen_fun, rosen_grad, rosen_hessvec, x0, precond_solve=jacobi_precond_solve,
        max_iter=40)
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
                   metric_solve: MetricSolve, metric_apply: MetricApply,
                   x0: Point, saddle_free: bool) -> bool:
    x, summary = trust_region_newton(
        fun, grad, hessvec, x0, metric_solve=metric_solve, metric_apply=metric_apply,
        precond_solve=precond_solve, saddle_free=saddle_free, gtol=1e-8, rtol=0.0,
        max_iter=1000, max_krylov=100)
    hist = history_array(summary)
    gfinal = covector_norm(numpy_space, lambda w: metric_solve(x, w), grad(x))
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
    for name, metric_solve, metric_apply in METRICS:
        for saddle_free in (False, True):
            label = f"{name}, {'saddle-free' if saddle_free else 'plain'}"
            ok &= run_and_report(label, rosen_fun, rosen_grad, rosen_hessvec, jacobi_precond_solve,
                                 metric_solve, metric_apply, x0, saddle_free)
    return ok


def demo_double_well(rng: Rng) -> bool:
    print("Demo: chain of double wells, n=50, started next to the local maximum at 0, smoothing ball")
    x0 = Point(0.01 * rng.standard_normal(50))
    ok = True
    for saddle_free in (False, True):
        label = "saddle-free" if saddle_free else "plain"
        ok &= run_and_report(label, well_fun, well_grad, well_hessvec, well_precond_solve,
                             smoothing_metric_solve, smoothing_metric_apply, x0, saddle_free)
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


def jacobi_diagonal(x: Point) -> Array:
    """The P(x) of jacobi_precond_solve, also used as a metric below."""
    diagonal: Array = np.maximum(np.abs(rosen_hess_diag(x)), 1.0)
    return diagonal


def jacobi_metric_apply(x: Point, v: Vector) -> Covector:
    return Covector(jacobi_diagonal(x) * v)


def test_operators_at_matches_and_runs_once_per_point() -> None:
    """operators_at(x) gives the same run as the keyword operators, and is
    called exactly once per new iterate (rejected steps reuse the
    operators), for P = M and for a separate preconditioner."""
    x0 = Point(np.tile([-1.2, 1.0], 25))
    for separate in (False, True):
        if separate:
            _, reference = trust_region_newton(rosen_fun, rosen_grad, rosen_hessvec, x0,
                                               precond_solve=jacobi_precond_solve)
        else:
            _, reference = trust_region_newton(rosen_fun, rosen_grad, rosen_hessvec, x0,
                                               metric_solve=jacobi_precond_solve)
        points: list[Point] = []

        def operators_at(x: Point) -> PointOperators[Vector, Covector]:
            points.append(x)
            P_solve: Callable[[Covector], Vector] = lambda w: jacobi_precond_solve(x, w)  # noqa: E731
            H: Callable[[Vector], Covector] = lambda v: rosen_hessvec(x, v)               # noqa: E731
            if separate:
                return PointOperators(hessvec=H, precond_solve=P_solve)
            return PointOperators(hessvec=H, metric_solve=P_solve)

        # With a separate P, the metric (M = I) is the space's default.
        x, summary = trust_region_newton(rosen_fun, rosen_grad, None, x0, operators_at=operators_at)
        assert without_times(summary.iterations) == without_times(reference.iterations)
        records = summary.iterations
        assert any(not r.accepted for r in records)             # rejections did happen
        assert len(points) == 1 + summary.accepted_steps
        assert len({id(p) for p in points}) == len(points)


def test_operator_arguments_are_checked() -> None:
    x0 = Point(np.zeros(4))

    def hv_ops(x: Point) -> PointOperators[Vector, Covector]:
        return PointOperators(hessvec=lambda v: rosen_hessvec(x, v))

    def p_ops(x: Point) -> PointOperators[Vector, Covector]:
        """A user metric and a separate preconditioner, but no metric_apply
        (the space's M must not be paired with a user M^{-1})."""
        return PointOperators(hessvec=lambda v: rosen_hessvec(x, v),
                              metric_solve=lambda w: smoothing_metric_solve(x, w),
                              precond_solve=lambda w: jacobi_precond_solve(x, w))

    def ma_ops(x: Point) -> PointOperators[Vector, Covector]:
        """metric_apply without metric_solve."""
        return PointOperators(hessvec=lambda v: rosen_hessvec(x, v),
                              metric_apply=lambda v: smoothing_metric_apply(x, v))
    for hv, ops in ((None, None), (rosen_hessvec, hv_ops), (None, p_ops), (None, ma_ops)):
        try:
            trust_region_newton(rosen_fun, rosen_grad, hv, x0, operators_at=ops)
        except ValueError:
            continue
        raise AssertionError("invalid operator arguments were accepted")


def test_matched_path_equals_explicit_metric() -> None:
    """With P = M (no precond_solve) only M^{-1} is applied. Passing the same
    M as a separate preconditioner, with metric_apply, takes the same
    steps with the Lanczos solver; the P = M run applies neither M nor a
    separate preconditioner."""
    x0 = Point(np.tile([-1.2, 1.0], 25))
    _, explicit = trust_region_newton(rosen_fun, rosen_grad, rosen_hessvec, x0,
                                      metric_solve=jacobi_precond_solve,
                                      precond_solve=jacobi_precond_solve,
                                      metric_apply=jacobi_metric_apply)
    _, matched = trust_region_newton(rosen_fun, rosen_grad, rosen_hessvec, x0,
                                     metric_solve=jacobi_precond_solve, krylov="lanczos")
    a, b = explicit.iterations, matched.iterations
    assert len(a) == len(b) and matched.reason == explicit.reason == "converged"
    assert [(r.accepted, r.subproblem.krylov_steps) for r in a] == \
        [(r.accepted, r.subproblem.krylov_steps) for r in b]
    assert max(abs(r.f - s.f) / max(1.0, abs(r.f)) for r, s in zip(a, b)) < 1e-10
    assert max(abs(r.gnorm - s.gnorm) / r.gnorm for r, s in zip(a, b)) < 1e-10
    assert matched.metric_apply_calls == 0 and matched.precond_solve_calls == 0


def test_krylov_selection() -> None:
    """krylov="auto" uses CG when P = M without saddle_free, and Lanczos
    otherwise; every combination converges on Rosenbrock."""
    x0 = Point(np.tile([-1.2, 1.0], 25))
    pieces = (rosen_fun, rosen_grad, rosen_hessvec, x0)
    runs = {
        "auto, M = P = I": trust_region_newton(*pieces, max_krylov=200),
        "auto, M = P = Jacobi": trust_region_newton(*pieces, metric_solve=jacobi_precond_solve),
        "lanczos, M = P": trust_region_newton(*pieces, metric_solve=jacobi_precond_solve,
                                              krylov="lanczos"),
        "auto, saddle-free": trust_region_newton(*pieces, metric_solve=jacobi_precond_solve,
                                                 saddle_free=True),
        "auto, M = I, P = Jacobi": trust_region_newton(*pieces, precond_solve=jacobi_precond_solve),
    }
    expected = {"auto, M = P = I": "cg", "auto, M = P = Jacobi": "cg", "lanczos, M = P": "lanczos",
                "auto, saddle-free": "lanczos", "auto, M = I, P = Jacobi": "lanczos"}
    for label, (_, summary) in runs.items():
        assert summary.reason == "converged", label
        assert {r.subproblem.krylov for r in summary.iterations} == {expected[label]}, label


def test_cg_reuses_the_gradient_solve() -> None:
    """With P = M the outer loop computes M^{-1} g for the gradient norm, and
    the CG solve starts from it: one M^{-1} solve per CG step, plus one per
    new iterate, and nothing else."""
    x0 = Point(np.tile([-1.2, 1.0], 25))
    _, summary = trust_region_newton(rosen_fun, rosen_grad, rosen_hessvec, x0,
                                     metric_solve=jacobi_precond_solve)
    advanced = sum(r.subproblem.krylov_steps - (r.subproblem.stop_reason in ("boundary", "negative"))
                   for r in summary.iterations)
    new_points = 1 + summary.accepted_steps
    assert summary.metric_solve_calls == advanced + new_points
    assert summary.precond_solve_calls == 0 and summary.metric_apply_calls == 0


def test_convergence_norm() -> None:
    """The recorded ||g|| and the stopping test use ||g||_{M^{-1}} by default
    (the Euclidean norm when no metric is given), and ||g||_{P^{-1}} with
    convergence_norm="preconditioner"; the default test is
    ||g|| <= rtol ||g(x0)||. The preconditioner norm costs one P solve per
    new iterate, which the subproblem reuses as its start vector."""
    x0 = Point(np.tile([-1.2, 1.0], 25))
    rtol = 1e-6
    pieces = (rosen_fun, rosen_grad, rosen_hessvec, x0)
    solves: dict[Literal["metric", "preconditioner"], MetricSolve] = {
        "metric": smoothing_metric_solve, "preconditioner": jacobi_precond_solve}
    for which, solve in solves.items():
        def norm(x: Point, g: Covector) -> float:
            return covector_norm(numpy_space, lambda w: solve(x, w), g)

        points: list[Point] = []

        def keep(x: Point, info: IterationInfo[Vector, Covector]) -> None:
            points.append(x)

        x, summary = trust_region_newton(
            *pieces, metric_solve=smoothing_metric_solve, metric_apply=smoothing_metric_apply,
            precond_solve=jacobi_precond_solve, convergence_norm=which, rtol=rtol, max_iter=1000,
            callback=keep)
        records = summary.iterations
        assert summary.reason == "converged", which
        xs = [x0] + points[:-1]                       # the iterate each record started from
        for r, xk in zip(records, xs):
            assert abs(r.gnorm - norm(xk, rosen_grad(xk))) <= 1e-12 * r.gnorm, which
        assert summary.gnorm <= rtol * summary.gnorm0 < min(r.gnorm for r in records), which
        assert abs(summary.gnorm - norm(x, rosen_grad(x))) <= 1e-12 * summary.gnorm, which
        new_points = 1 + summary.accepted_steps
        k_total = sum(r.subproblem.krylov_steps for r in records)
        if which == "metric":
            assert summary.precond_solve_calls == k_total + len(records)
        else:
            assert summary.precond_solve_calls == k_total + new_points
    _, euclidean = trust_region_newton(*pieces, max_krylov=200)
    g0 = float(np.linalg.norm(rosen_grad(x0)))
    assert abs(euclidean.gnorm0 - g0) <= 1e-14 * g0
    assert euclidean.gnorm <= 1e-8 * g0
