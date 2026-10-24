"""Lanczos, the small trust-region solvers and the subproblem solver, checked
against dense linear algebra."""


import numpy as np

from nicks_optimizers import (
    Array,
    Covector,
    FrozenDualInnerProduct,
    Point,
    euclidean_inner,
    numpy_space,
    solve_sphere_trs,
    solve_tr_subproblem,
    trust_region_newton,
)
from nicks_optimizers.trust_region_subproblem import kkt_residual_dual_norm, solve_projected_trs
from helpers import (
    Rng,
    dense_abs_m,
    dense_dual_inner,
    dense_dual_norm,
    dense_inner,
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
        Q, V, T, G, W, gamma0, gamma_next, v_next = run_lanczos(H, P, M, g, k, passes)
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


def check_surrogate_residual(rng: Rng) -> bool:
    print("Residual from Gram matrices vs dense ||H s_flip + lam M s + g||_{M^-1} (n=60, k=25)")
    n, k = 60, 25
    H = matrix_with_spectrum(np.concatenate([np.linspace(0.5, 20.0, n - 5),
                                             [-3.0, -2.0, -1.0, -0.5, -0.2]]), rng)
    P, M = random_spd(n, 30.0, rng), random_spd(n, 30.0, rng)
    g = rng.standard_normal(n)
    Q, V, T, G, W, gamma0, gamma_next, v_next = run_lanczos(H, P, M, g, k)
    ok = True
    for saddle_free in (False, True):
        y, y_flip, lam, mval, _, theta = solve_projected_trs(T, G, gamma0, 0.3, saddle_free)
        resid = kkt_residual_dual_norm(y, float(y_flip[-1]), lam, G, W, gamma_next)
        s, s_flip = Q @ y, Q @ y_flip
        dense = dense_dual_norm(H @ s_flip + lam * M @ s + g, M)
        model = float(g @ s + 0.5 * s @ H @ s_flip)       # = <g,s> + 0.5 <|H| s, s> on the subspace
        label = "saddle-free" if saddle_free else "plain"
        print(f"  {label}: {int(np.sum(theta < 0))} negative M-Ritz values, lam = {lam:.3f}")
        ok &= report(f"{label}: Gram-matrix residual vs dense", abs(resid - dense) / dense, 1e-10)
        ok &= report(f"{label}: model value vs <g,s> + 0.5 <H s_flip, s>", abs(mval - model) / abs(model), 1e-12)
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
        # Gram-matrix residual: exact for interior steps, sqrt(eps)-type floor when lam > 0.
        tol, where = (1e-6, "boundary, floor") if lam > 0.0 else (1e-10, "interior")
        ok &= report(f"{name}: residual vs dense ({where})",
                     abs(resid - dense_dual_norm(g + Hm @ s + lam * M @ s, M)) / gn, tol)
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


def check_dual_requirement(rng: Rng) -> bool:
    print("Regression: the dual inner product is required and must be the exact dual of ball_inner")
    ok = True
    try:
        trust_region_newton(rosen_fun, rosen_grad, rosen_hessvec, jacobi_precond_solve,  # type: ignore[call-arg]
                            ball_inner=euclidean_inner, x0=Point(np.ones(4)))
        refused = False
    except TypeError:
        refused = True
    ok &= report("calling without ball_dual_inner is refused (no silent Euclidean fallback)",
                 0.0 if refused else 1.0, 0.0)

    # Boundary residual with a consistent pair vs. the old failure mode (M inner, Euclidean dual).
    n, k, radius = 60, 15, 0.05
    H = matrix_with_spectrum(np.linspace(-2.0, 10.0, n), rng)
    P, M = random_spd(n, 30.0, rng), random_spd(n, 30.0, rng)
    g = rng.standard_normal(n)
    pairs: list[tuple[str, FrozenDualInnerProduct, Array]] = [
        ("exact dual (M^-1)", dense_dual_inner(M), M),
        ("Euclidean dual with M != I", lambda Psi, phi: Psi.T @ phi, np.eye(n)),
    ]
    for name, dual_inner, Mdual in pairs:
        s, st = solve_tr_subproblem(
            numpy_space, Covector(g), dense_vec_to_covec(H), dense_solve(P), dense_inner(M),
            dual_inner, radius, 0.0, k, stall_tol=0.0)
        resid = st.resid
        actual = dense_dual_norm(g + H @ s + st.lam * M @ s, Mdual)
        if name.startswith("exact"):
            ok &= report(f"{name}: reported boundary residual = actual", abs(resid - actual) / actual, 1e-10)
        else:
            print(f"         ({name}: reports {resid:.3e}, actual {actual:.3e}; "
                  f"the boundary formula needs exact duality)")
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


def test_surrogate_residual() -> None:
    assert check_surrogate_residual(np.random.default_rng(2))


def test_subproblem_full_space() -> None:
    assert check_subproblem_full_space(np.random.default_rng(3))


def test_saddle_step_direction() -> None:
    assert check_saddle_step_direction()


def test_dual_requirement() -> None:
    assert check_dual_requirement(np.random.default_rng(4))


def test_mismatch_example() -> None:
    assert check_mismatch_example()
