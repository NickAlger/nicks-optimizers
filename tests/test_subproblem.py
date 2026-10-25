"""Lanczos, the small trust-region solvers and the subproblem solver, checked
against dense linear algebra."""

import numpy as np
import pytest

from nicks_optimizers import (
    Array,
    Covector,
    Point,
    PointOperators,
    Vector,
    euclidean_metric_solve,
    flat_retraction,
    numpy_space,
    numpy_space_with_reductions,
    solve_sphere_trs,
    solve_tr_subproblem,
    solve_tr_subproblem_cg,
    trust_region_newton,
)
from nicks_optimizers.trust_region_subproblem import krylov_residual_norm, solve_projected_trs
from helpers import (
    Rng,
    dense_abs_m,
    dense_apply,
    dense_dual_norm,
    dense_solve,
    dense_vec_to_covec,
    full_space_kkt_error,
    matrix_with_spectrum,
    random_orthogonal,
    random_spd,
    report,
    run_dense_subproblem,
    run_lanczos,
    sphere_kkt_error,
)
from problems import (
    jacobi_precond_solve,
    rosen_fun,
    rosen_grad,
    rosen_hessvec,
)


def check_lanczos_relations(rng: Rng) -> bool:
    print("Preconditioned Lanczos identities (n=80, k=60, H indefinite with outliers)")
    n, k = 80, 60
    evals = np.concatenate([np.linspace(-1.0, 1.0, n - 4), [-50.0, 30.0, 200.0, 1000.0]])
    H = matrix_with_spectrum(evals, rng)
    P = random_spd(n, 1e2, rng)
    M = np.eye(n)
    g = rng.standard_normal(n)
    ok = True
    for passes in (2, 0):
        state, T, G = run_lanczos(H, P, M, g, k, passes)
        Q, V, gamma0, gamma_next, v_next = state.Q, state.V, state.gamma0, state.gamma_next, state.v_next
        e1 = np.zeros(k); e1[0] = 1.0
        ek = np.zeros(k); ek[-1] = 1.0
        orth = float(np.linalg.norm(Q.T @ P @ Q - np.eye(k)))
        proj = float(np.linalg.norm(Q.T @ H @ Q - T) / np.linalg.norm(T))
        grad = float(np.linalg.norm(Q.T @ g - gamma0 * e1) / gamma0)
        recur = float(np.linalg.norm(H @ Q - V @ T - gamma_next * np.outer(v_next, ek))
                      / np.linalg.norm(H @ Q))
        if passes == 2:
            print(" with full reorthogonalization:")
            ok &= report("||Q^T P Q - I||", orth, 1e-10)
            ok &= report("||Q^T H Q - T|| / ||T||", proj, 1e-10)
            ok &= report("||Q^T g - gamma0 e1|| / gamma0", grad, 1e-10)
            ok &= report("||H Q - V T - gamma v e_k^T|| / ||H Q||", recur, 1e-10)
        else:
            print(" without reorthogonalization (for comparison, not checked):")
            print(f"         ||Q^T P Q - I|| = {orth:.2e},  ||Q^T H Q - T||/||T|| = {proj:.2e},"
                  f"  recurrence residual = {recur:.2e}")
    return ok


def check_sphere_trs(rng: Rng) -> bool:
    print("Small spherical TRS: KKT + second-order conditions (certify global optimality)")
    k = 12
    A_pd = random_spd(k, 1e3, rng)
    A_ind = matrix_with_spectrum(np.linspace(-3.0, 5.0, k), rng)
    b = rng.standard_normal(k)
    U = random_orthogonal(k, rng)
    A_hard = (U * np.linspace(-1.0, 4.0, k)) @ U.T
    b_hard = U[:, 1:] @ (0.01 * rng.standard_normal(k - 1))   # orthogonal to bottom eigenvector
    b_near = b_hard + 1e-9 * U[:, 0]
    cases = [
        ("PD, interior", A_pd, b, 1e3),
        ("PD, boundary", A_pd, b, 0.05),
        ("indefinite, small radius", A_ind, b, 0.1),
        ("indefinite, large radius", A_ind, b, 10.0),
        ("hard case", A_hard, b_hard, 1.0),
        ("near hard case", A_hard, b_near, 1.0),
    ]
    ok = True
    for name, A, bb, radius in cases:
        z, lam = solve_sphere_trs(A, bb, radius)
        ok &= report(name, sphere_kkt_error(A, bb, radius, z, lam), 1e-8)
    return ok


def check_explicit_residual(rng: Rng) -> bool:
    print("Explicit residual vs dense ||H s_flip + lam M s + g||_{M^-1} (n=60, k=25, P != M)")
    n, k = 60, 25
    H = matrix_with_spectrum(np.concatenate([np.linspace(0.5, 20.0, n - 5),
                                             [-3.0, -2.0, -1.0, -0.5, -0.2]]), rng)
    P, M = random_spd(n, 30.0, rng), random_spd(n, 30.0, rng)
    g = rng.standard_normal(n)
    state, T, G = run_lanczos(H, P, M, g, k)
    Q = state.Q
    ok = True
    for radius in (0.3, 1e4):
        for saddle_free in (False, True):
            y, y_flip, lam, mval, _, theta = solve_projected_trs(T, G, state.gamma0, radius, saddle_free)
            resid, applies, solves = krylov_residual_norm(
                numpy_space, state, y, float(y_flip[-1]), lam, G, dense_apply(M), dense_solve(M))
            s, s_flip = Q @ y, Q @ y_flip
            dense = dense_dual_norm(H @ s_flip + lam * M @ s + g, M)
            model = float(g @ s + 0.5 * s @ H @ s_flip)   # = <g,s> + 0.5 <|H| s, s> on the subspace
            label = f"{'saddle-free' if saddle_free else 'plain'}, radius {radius:g}"
            print(f"  {label}: {int(np.sum(theta < 0))} negative M-Ritz values, lam = {lam:.3f}")
            ok &= report(f"{label}: explicit residual vs dense", abs(resid - dense) / dense, 1e-10)
            ok &= report(f"{label}: one M product iff lam > 0, one M^-1 solve",
                         float(applies != int(lam > 0.0) or solves != 1), 0.0)
            ok &= report(f"{label}: model value vs <g,s> + 0.5 <H s_flip, s>",
                         abs(mval - model) / abs(model), 1e-12)
    return ok


def check_matched_residual(rng: Rng) -> bool:
    print("P = M: the residual norm is gamma_next |y_k| with no operator applied (n=60, k=25)")
    n, k = 60, 25
    H = matrix_with_spectrum(np.concatenate([np.linspace(0.5, 20.0, n - 3), [-3.0, -1.0, -0.2]]), rng)
    M = random_spd(n, 30.0, rng)
    g = rng.standard_normal(n)
    gn = dense_dual_norm(g, M)
    state, T, G = run_lanczos(H, M, M, g, k)
    ok = report("||G - I|| (the basis is M-orthonormal)", float(np.linalg.norm(G - np.eye(k))), 1e-10)
    for saddle_free in (False, True):
        y, y_flip, lam, _, _, _ = solve_projected_trs(T, np.eye(k), state.gamma0, 0.3, saddle_free)
        resid, applies, solves = krylov_residual_norm(
            numpy_space, state, y, float(y_flip[-1]), lam, np.eye(k), None, dense_solve(M))
        s, s_flip = state.Q @ y, state.Q @ y_flip
        dense = dense_dual_norm(H @ s_flip + lam * M @ s + g, M)
        label = "saddle-free" if saddle_free else "plain"
        ok &= report(f"{label}: |gamma_next |y_k| - dense| / ||g||", abs(resid - dense) / gn, 1e-12)
        ok &= report(f"{label}: no operator applied", float(applies + solves), 0.0)
    return ok


def check_subproblem_full_space(rng: Rng) -> bool:
    print("Subproblem run to k = n reproduces the global solution (n=40, P != M)")
    n = 40
    H_ind = matrix_with_spectrum(np.linspace(-2.0, 10.0, n), rng)
    H_pd = random_spd(n, 1e2, rng)
    P, M = random_spd(n, 1e2, rng), random_spd(n, 1e3, rng)
    g = rng.standard_normal(n)
    gn = dense_dual_norm(g, M)
    cases = [
        ("plain, indefinite H, radius 0.05", H_ind, 0.05, False),
        ("plain, indefinite H, radius 2", H_ind, 2.0, False),
        ("plain, PD H, radius 0.05", H_pd, 0.05, False),
        ("plain, PD H, radius 1e4 (interior)", H_pd, 1e4, False),
        ("saddle-free, indefinite H, radius 0.05", H_ind, 0.05, True),
        ("saddle-free, indefinite H, radius 2", H_ind, 2.0, True),
        ("saddle-free, indefinite H, radius 1e4 (interior)", H_ind, 1e4, True),
    ]
    ok = True
    for name, H, radius, saddle_free in cases:
        Hm = dense_abs_m(H, M) if saddle_free else H
        s, mval, step_norm, lam, k, resid, n_neg = run_dense_subproblem(
            H, P, M, g, radius, n, resid_tol=1e-10 * gn, saddle_free=saddle_free)
        model = float(g @ s + 0.5 * s @ Hm @ s)
        ok &= report(f"{name}: KKT", full_space_kkt_error(Hm, M, g, radius, s, lam), 1e-8)
        ok &= report(f"{name}: model value", abs(mval - model) / abs(model), 1e-10)
        # The explicit residual is accurate on the boundary too.
        where = "boundary" if lam > 0.0 else "interior"
        ok &= report(f"{name}: residual vs dense ({where})",
                     abs(resid - dense_dual_norm(g + Hm @ s + lam * M @ s, M)) / gn, 1e-10)
    # Interior saddle-free step is exactly -|H|_M^{-1} g, with |.| taken in the M metric.
    s_sf = run_dense_subproblem(H_ind, P, M, g, 1e4, n, resid_tol=1e-10 * gn, saddle_free=True)[0]
    exact = -np.linalg.solve(dense_abs_m(H_ind, M), g)
    ok &= report("interior saddle-free step vs -|H|_M^{-1} g",
                 float(np.linalg.norm(s_sf - exact) / np.linalg.norm(exact)), 1e-8)
    wrong_metric = -np.linalg.solve(dense_abs_m(H_ind, P), g)
    print(f"         (for contrast, -|H|_P^{{-1}} g differs by "
          f"{np.linalg.norm(wrong_metric - exact) / np.linalg.norm(exact):.1e})")
    return ok


def check_saddle_step_direction() -> bool:
    print("Saddle f = x1^2/2 - x2^2/2 + x2^4/4 at x = (0.5, 0.01): Newton moves into the saddle,")
    print("the saddle-free step moves away from it")
    x = np.array([0.5, 0.01])
    H = np.diag([1.0, -1.0 + 3.0 * x[1] ** 2])
    g = np.array([x[0], -x[1] + x[1] ** 3])
    s = run_dense_subproblem(H, np.eye(2), np.eye(2), g, 10.0, 2, resid_tol=1e-14, saddle_free=True)[0]
    newton = -np.linalg.solve(H, g)
    print(f"  x2 after Newton step: {x[1] + newton[1]:+.2e},  after saddle-free step: {x[1] + s[1]:+.2e}")
    ok = report("saddle-free step vs -|H|^{-1} g",
                float(np.linalg.norm(s + np.linalg.solve(np.abs(H), g))), 1e-12)
    ok &= report("saddle-free step increases |x2| (value is -|x2_new| + |x2|)",
                 -abs(x[1] + s[1]) + abs(x[1]), 0.0)
    return ok


def check_metric_requirements(rng: Rng) -> bool:
    print("A separate preconditioner needs M itself; reported boundary residuals are actual ones")
    n, k, radius = 60, 15, 0.05
    H = matrix_with_spectrum(np.linspace(-2.0, 10.0, n), rng)
    P, M = random_spd(n, 30.0, rng), random_spd(n, 30.0, rng)
    g = rng.standard_normal(n)
    ok = True
    try:
        solve_tr_subproblem(numpy_space, Covector(g), dense_vec_to_covec(H), dense_solve(M),
                            radius, 0.0, k, precond_solve=dense_solve(P))
        refused = False
    except ValueError:
        refused = True
    ok &= report("precond_solve without metric_apply is refused", 0.0 if refused else 1.0, 0.0)
    s, st = solve_tr_subproblem(numpy_space, Covector(g), dense_vec_to_covec(H), dense_solve(M),
                                radius, 0.0, k, precond_solve=dense_solve(P),
                                metric_apply=dense_apply(M), stall_tol=0.0)
    actual = dense_dual_norm(g + H @ s + st.lam * M @ s, M)
    ok &= report("boundary active", 0.0 if st.lam > 0.0 else 1.0, 0.0)
    ok &= report("reported boundary residual = actual", abs(st.resid - actual) / actual, 1e-10)
    # M products: k for the Gram matrix (q_0 .. q_{k-1}), one per residual with lam > 0.
    boundary_steps = sum(1 for lam in st.trace_lam if lam > 0.0)
    ok &= report("calls: k Hessian products, k + 1 P solves, k + (boundary steps) M products, "
                 "k M^-1 solves", float((st.hessvec_calls, st.precond_solve_calls, st.metric_apply_calls,
                                         st.metric_solve_calls) != (k, k + 1, k + boundary_steps, k)), 0.0)
    return ok


def check_mismatch_example() -> bool:
    print("2-D example from the discussion: H=diag(1,100), g=(1,1), M=diag(0.01,100), radius 0.17, P=I")
    H, M, g = np.diag([1.0, 100.0]), np.diag([0.01, 100.0]), np.array([1.0, 1.0])
    s, mval, step_norm, lam, k, resid, n_neg = run_dense_subproblem(
        H, np.eye(2), M, g, 0.17, 2, resid_tol=1e-12)
    print(f"  step = {s},  model value = {mval:.4f},  ||s||_M = {step_norm:.4f},  lam = {lam:.1e}")
    ok = report("distance to Newton step (-1, -0.01)",
                float(np.linalg.norm(s - np.array([-1.0, -0.01]))), 1e-10)
    ok &= report("model value vs -0.505", abs(mval + 0.505), 1e-10)
    return ok


def test_lanczos_relations() -> None:
    assert check_lanczos_relations(np.random.default_rng(0))


def test_sphere_trs() -> None:
    assert check_sphere_trs(np.random.default_rng(1))


def test_explicit_residual() -> None:
    assert check_explicit_residual(np.random.default_rng(2))


def test_matched_residual() -> None:
    assert check_matched_residual(np.random.default_rng(5))


def test_subproblem_full_space() -> None:
    assert check_subproblem_full_space(np.random.default_rng(3))


def test_saddle_step_direction() -> None:
    assert check_saddle_step_direction()


def test_metric_requirements() -> None:
    assert check_metric_requirements(np.random.default_rng(4))


def test_outer_loop_checks_the_structure_arguments() -> None:
    """trust_region_newton takes the vector space, the metric and the
    retraction separately. With numpy defaults (no space) M = I and x + s,
    also with a separate preconditioner. It refuses: a space without
    metric_solve or without retract; metric_apply without metric_solve; a
    user metric_solve with a separate preconditioner but no metric_apply;
    operators_at together with operator arguments; point_data without
    operators_at; krylov='cg' with a
    separate preconditioner or saddle_free; unknown options."""
    x0 = Point(np.ones(4))
    pieces = (rosen_fun, rosen_grad, rosen_hessvec, x0)
    reductions = numpy_space_with_reductions(lambda c: c)
    retract = flat_retraction(reductions)
    with pytest.raises(ValueError):
        trust_region_newton(*pieces, space=numpy_space, retract=retract)
    with pytest.raises(ValueError):
        trust_region_newton(*pieces, space=numpy_space, metric_solve=euclidean_metric_solve)
    with pytest.raises(ValueError):
        trust_region_newton(*pieces, metric_apply=lambda x, v: Covector(0.5 * v))
    with pytest.raises(ValueError):
        trust_region_newton(*pieces, metric_solve=lambda x, w: Vector(2.0 * w),
                            precond_solve=jacobi_precond_solve)
    with pytest.raises(ValueError):
        trust_region_newton(rosen_fun, rosen_grad, None, x0, precond_solve=jacobi_precond_solve,
                            operators_at=lambda x, _: PointOperators(hessvec=lambda v: rosen_hessvec(x, v)))
    with pytest.raises(ValueError):
        trust_region_newton(*pieces, point_data=lambda x: x)
    with pytest.raises(ValueError):
        trust_region_newton(*pieces, krylov="cg", precond_solve=jacobi_precond_solve)
    with pytest.raises(ValueError):
        trust_region_newton(*pieces, krylov="cg", saddle_free=True)
    with pytest.raises(ValueError):
        trust_region_newton(*pieces, krylov="steihaug")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        trust_region_newton(*pieces, convergence_norm="euclidean")  # type: ignore[arg-type]
    trust_region_newton(*pieces, max_iter=2)
    trust_region_newton(*pieces, precond_solve=jacobi_precond_solve, max_iter=2)
    trust_region_newton(*pieces, space=reductions, retract=retract,
                        metric_solve=euclidean_metric_solve, max_iter=2)


def test_mismatch_example() -> None:
    assert check_mismatch_example()


def test_matched_path_equals_explicit_metric() -> None:
    """P = M (no precond_solve) gives the same steps as passing P = M
    explicitly (precond_solve = metric_solve, with metric_apply), plain and
    saddle-free, with the trust region inactive, active, and near the hard
    case, without applying M or a separate preconditioner."""
    rng = np.random.default_rng(11)
    n = 60
    H = matrix_with_spectrum(np.concatenate([[-3.0, -1.0], np.linspace(0.5, 20.0, n - 2)]), rng)
    M = random_spd(n, 50.0, rng)
    g = Covector(rng.standard_normal(n))
    for radius in (1e-3, 0.5, 1e3):
        for saddle_free in (False, True):
            s1, a = solve_tr_subproblem(numpy_space, g, dense_vec_to_covec(H), dense_solve(M),
                                        radius, 1e-10, 40, precond_solve=dense_solve(M),
                                        metric_apply=dense_apply(M), saddle_free=saddle_free)
            s2, b = solve_tr_subproblem(numpy_space, g, dense_vec_to_covec(H), dense_solve(M),
                                        radius, 1e-10, 40, saddle_free=saddle_free)
            assert a.krylov_steps == b.krylov_steps and a.stop_reason == b.stop_reason
            assert np.linalg.norm(s1 - s2) <= 1e-8 * np.linalg.norm(s1)
            assert abs(a.lam - b.lam) <= 1e-8 * max(1.0, a.lam)
            assert abs(a.model_value - b.model_value) <= 1e-10 * abs(a.model_value)
            assert abs(a.resid - b.resid) <= 1e-6 * max(a.resid, 1e-10 * float(np.sqrt(g @ g)))
            assert b.metric_apply_calls == 0 and b.precond_solve_calls == 0
            assert b.metric_solve_calls == b.krylov_steps + 1


def test_steihaug_cg_subproblem() -> None:
    """Truncated CG for the trust region of the M-norm (P = M).

    Inside the ball it computes the same step as the Lanczos solver with
    M = P (same Krylov space, same stopping test), at tolerances reached
    before CG's loss of conjugacy matters; to tight tolerances Lanczos needs
    fewer steps. When the next iterate
    would leave the ball, it stops on the boundary (||s||_P = radius) with
    at least the Cauchy decrease. On negative curvature it follows the
    direction to the boundary. Statistics report the method and calls."""
    rng = np.random.default_rng(13)
    n = 50
    P = random_spd(n, 5.0, rng)
    hessvec_spd = dense_vec_to_covec(matrix_with_spectrum(np.linspace(1.0, 10.0, n), rng))
    H_ind = matrix_with_spectrum(np.concatenate([[-5.0, -0.5], np.linspace(0.3, 30.0, n - 2)]), rng)
    g = Covector(rng.standard_normal(n))
    precond = dense_solve(P)

    def cauchy_model(H: Array, radius: float) -> float:
        d = -np.linalg.solve(P, g)
        dnorm = float(np.sqrt(d @ P @ d))
        curvature, slope = float(d @ H @ d), float(g @ d)
        t = radius / dnorm if curvature <= 0 else min(-slope / curvature, radius / dnorm)
        return slope * t + 0.5 * curvature * t ** 2

    # Interior: same step as Lanczos with M = P.
    tol = 1e-4 * float(np.sqrt(g @ np.linalg.solve(P, g)))
    s_cg, st_cg = solve_tr_subproblem_cg(numpy_space, g, hessvec_spd, precond, 1e6, tol, 60)
    s_lz, st_lz = solve_tr_subproblem(numpy_space, g, hessvec_spd, precond, 1e6, tol, 60)
    assert st_cg.krylov == "cg" and st_cg.stop_reason == "tol" and st_cg.lam == 0.0
    assert st_cg.krylov_steps == st_lz.krylov_steps
    assert np.linalg.norm(s_cg - s_lz) <= 1e-6 * np.linalg.norm(s_lz)
    assert abs(st_cg.model_value - st_lz.model_value) <= 1e-10 * abs(st_lz.model_value)
    assert st_cg.metric_apply_calls == 0 and st_cg.precond_solve_calls == 0
    assert st_cg.metric_solve_calls == st_cg.krylov_steps + 1

    # Boundary and negative curvature.
    H_spd = np.column_stack([hessvec_spd(Vector(np.eye(n)[:, j])) for j in range(n)])
    for H, radius, reason in ((H_spd, 0.05, "boundary"), (H_ind, 1e3, "negative")):
        s, st = solve_tr_subproblem_cg(numpy_space, g, dense_vec_to_covec(H), precond, radius, 1e-12, 60)
        assert st.stop_reason == reason and np.isnan(st.lam)
        assert abs(float(np.sqrt(s @ P @ s)) - radius) <= 1e-10 * radius
        assert abs(st.model_value - float(g @ s + 0.5 * s @ H @ s)) <= 1e-9 * abs(st.model_value)
        assert st.model_value <= cauchy_model(H, radius) + 1e-12
        assert st.n_negative == (1 if reason == "negative" else 0)

    # The step cap, and M^{-1} g supplied by the caller.
    _, st = solve_tr_subproblem_cg(numpy_space, g, hessvec_spd, precond, 1e6, 0.0, 3,
                                   start_vector=precond(g))
    assert st.stop_reason == "max" and st.krylov_steps == 3 and st.metric_solve_calls == 3



def test_lanczos_is_more_robust_than_cg_to_tight_tolerances() -> None:
    """Why the reorthogonalized Lanczos solver stays available for M = P: on
    an ill-conditioned pencil (cond(P^{-1} H) about 10^3) and a tight
    tolerance, Lanczos converges within n steps, while CG loses conjugacy in
    floating point and needs more."""
    rng = np.random.default_rng(13)
    n = 50
    P = random_spd(n, 30.0, rng)
    hessvec = dense_vec_to_covec(matrix_with_spectrum(np.linspace(0.5, 40.0, n), rng))
    g = Covector(rng.standard_normal(n))
    tol = 1e-10 * float(np.sqrt(g @ np.linalg.solve(P, g)))
    _, lanczos = solve_tr_subproblem(numpy_space, g, hessvec, dense_solve(P), 1e6, tol, 200)
    _, cg = solve_tr_subproblem_cg(numpy_space, g, hessvec, dense_solve(P), 1e6, tol, 200)
    assert lanczos.krylov_steps <= n and lanczos.stop_reason in ("tol", "breakdown")
    assert cg.stop_reason == "tol" and cg.krylov_steps > lanczos.krylov_steps
