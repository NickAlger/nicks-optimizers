"""The Lanczos process used on its own, without any trust region."""

from typing import Callable, Optional

import numpy as np
import pytest

from nicks_optimizers import (
    Covector,
    CovectorBlock,
    LanczosState,
    Vector,
    VectorBlock,
    lanczos_advance,
    lanczos_start,
    lanczos_tridiagonal,
    numpy_space,
)
from helpers import dense_solve, dense_vec_to_covec, matrix_with_spectrum, random_spd


def test_lanczos_as_eigensolver() -> None:
    """Run the process to breakdown on an n-dimensional problem: it stops after
    exactly n steps, its Ritz values (eigenvalues of T) are the eigenvalues of
    the pencil H u = theta P u, and a broken-down process does not advance
    (the state comes back unchanged). Earlier states are left untouched by
    later steps."""
    rng = np.random.default_rng(9)
    n = 30
    H = matrix_with_spectrum(np.linspace(-3.0, 7.0, n), rng)
    P = random_spd(n, 10.0, rng)
    hessvec, precond_solve = dense_vec_to_covec(H), dense_solve(P)

    state = lanczos_start(numpy_space, Covector(rng.standard_normal(n)), precond_solve,
                          capacity=n + 2)
    history = [state]
    while not state.breakdown and state.steps < n + 2:
        state = lanczos_advance(numpy_space, state, hessvec, precond_solve)
        history.append(state)
    assert state.steps == n

    Linv = np.linalg.inv(np.linalg.cholesky(P))
    exact = np.linalg.eigvalsh(Linv @ H @ Linv.T)
    ritz = np.linalg.eigvalsh(lanczos_tridiagonal(state))
    assert np.max(np.abs(ritz - exact)) <= 1e-9 * np.max(np.abs(exact))

    assert lanczos_advance(numpy_space, state, hessvec, precond_solve) is state

    # Frozen states: each one still describes its own step count.
    for k, earlier in enumerate(history):
        assert earlier.steps == k
        assert np.count_nonzero(earlier.deltas) == k and lanczos_tridiagonal(earlier).shape == (k, k)
        assert earlier.Q.count == max(k, 1)


def test_extreme_ritz_values_converge_early() -> None:
    """The typical eigensolver use: a few steps already resolve the extreme
    eigenvalues of a pencil with well-separated ends of its spectrum."""
    rng = np.random.default_rng(10)
    n = 200
    evals = np.concatenate([[-50.0, -20.0], np.linspace(1.0, 2.0, n - 4), [40.0, 100.0]])
    H = matrix_with_spectrum(evals, rng)
    state = lanczos_start(numpy_space, Covector(rng.standard_normal(n)), lambda w: Vector(w),
                          capacity=26)
    for _ in range(25):
        state = lanczos_advance(numpy_space, state, dense_vec_to_covec(H), lambda w: Vector(w))
    ritz = np.linalg.eigvalsh(lanczos_tridiagonal(state))
    assert abs(ritz[0] - (-50.0)) <= 1e-8 * 50.0
    assert abs(ritz[-1] - 100.0) <= 1e-8 * 100.0


def test_redundant_representation_needs_projection() -> None:
    """Vectors and covectors stored with redundant components: the space is
    {v : B^T v = 0} inside R^n, and H = Pi A Pi with Pi = I - B B^T. The
    components along B pair to zero with every vector, so reorthogonalization
    cannot see them, and the recurrence amplifies their rounding errors.
    Without project_covector they grow to O(1), a spurious zero Ritz value
    appears and the process runs past the dimension n - m; with it the Ritz
    values are those of A on the subspace and the process breaks down after
    exactly n - m steps."""
    rng = np.random.default_rng(3)
    n, m = 120, 10
    B = np.linalg.qr(rng.standard_normal((n, m)))[0]

    def project(w: Covector) -> Covector:
        return Covector(w - B @ (B.T @ w))

    A = matrix_with_spectrum(np.linspace(1.0, 10.0, n), rng)
    hessvec = lambda v: project(Covector(A @ project(Covector(v))))
    basis = np.linalg.svd(np.eye(n) - B @ B.T)[0][:, :n - m]
    exact = np.linalg.eigvalsh(basis.T @ A @ basis)
    g = project(Covector(rng.standard_normal(n)))

    def run(project_covector: Optional[Callable[[Covector], Covector]]
            ) -> LanczosState[Vector, Covector, VectorBlock, CovectorBlock]:
        state = lanczos_start(numpy_space, g, lambda w: Vector(w), capacity=n)
        while not state.breakdown and state.steps < n - m:
            state = lanczos_advance(numpy_space, state, hessvec, lambda w: Vector(w),
                                    project_covector=project_covector)
        return state

    plain, projected = run(None), run(project)
    assert np.abs(B.T @ plain.v).max() > 1e-3
    assert np.linalg.eigvalsh(lanczos_tridiagonal(plain))[0] < 1e-6 * exact[0]
    assert not plain.breakdown
    assert np.abs(B.T @ projected.v).max() < 1e-14
    ritz = np.linalg.eigvalsh(lanczos_tridiagonal(projected))
    assert projected.breakdown and projected.steps == n - m
    assert np.max(np.abs(ritz - exact)) <= 1e-9 * exact[-1]


def test_cg_process_against_dense_algebra() -> None:
    """Preconditioned CG on an SPD H: at every step the residual is g + H s,
    z = P^{-1} r, the recurrences for ||s||_P^2, (s, p)_P, ||p||_P^2 and the
    model value match direct computation (the (s, p)_P recurrence to the
    accuracy the boundary test needs), ||s||_P increases (Steihaug), and the
    process solves H s = -g."""
    from nicks_optimizers import cg_advance, cg_curvature, cg_start
    rng = np.random.default_rng(12)
    n = 40
    H = matrix_with_spectrum(np.linspace(1.0, 10.0, n), rng)
    P = random_spd(n, 5.0, rng)
    g = Covector(rng.standard_normal(n))
    hessvec, precond = dense_vec_to_covec(H), dense_solve(P)
    state = cg_start(numpy_space, g, precond)
    rz0 = state.rz
    norms = [0.0]
    while state.rz > 1e-24 * rz0:
        Hp, curvature = cg_curvature(numpy_space, state, hessvec)
        state = cg_advance(numpy_space, state, Hp, curvature, precond)
        s, p = state.s, state.p
        scale = float(np.linalg.norm(g))
        assert np.max(np.abs(state.r - (g + H @ s))) < 1e-10 * scale
        assert np.max(np.abs(P @ state.z - state.r)) < 1e-10 * scale
        sPs, pPp = float(s @ P @ s), float(p @ P @ p)
        model = float(g @ s + 0.5 * s @ H @ s)
        assert abs(state.s_norm2 - sPs) <= 1e-8 * sPs
        assert abs(state.p_norm2 - pPp) <= 1e-8 * pPp
        # (s, p)_P only enters ||s + alpha p||_P^2 in the boundary test, so its
        # drift (it loses relative accuracy as conjugacy degrades) is
        # measured against ||s||_P^2.
        assert abs(state.sp - float(s @ P @ p)) <= 1e-10 * sPs
        assert abs(state.model - model) <= 1e-9 * abs(model)
        norms.append(state.s_norm2)
    assert all(b >= a * (1.0 - 1e-14) for a, b in zip(norms, norms[1:]))   # up to rounding
    assert np.linalg.norm(H @ state.s + g) < 1e-10 * np.linalg.norm(g)
