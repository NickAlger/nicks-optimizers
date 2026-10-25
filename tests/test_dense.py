"""The small dense kernels with padding (dev/jit_backends_plan.md, phase 5):
the projected trust-region problem padded to a larger capacity, on the
strict backend (static shapes), has the solution of the unpadded problem;
on numpy the padded problem is cut down to the active block and gives the
unpadded result exactly."""

from typing import Any

import numpy as np
import pytest

from nicks_optimizers import numpy_backend, solve_projected_trs, solve_sphere_trs_eig
from strict_backend import plain, strict, strict_backend


def projected_problem(k: int, seed: int, hard: bool = False) -> tuple[Any, Any]:
    """T and G (k x k) of a Krylov method: G symmetric positive definite, T
    indefinite. hard=True makes the hard case: b = gamma0 L^{-1} e_1 is
    orthogonal to the bottom M-Ritz vector (G = I, and T has a bottom
    eigenvector with zero first component)."""
    rng = np.random.default_rng(seed)
    if hard:
        Q = np.linalg.qr(rng.standard_normal((k, k)))[0]
        v = Q[:, 0] - Q[0, 0] * np.eye(k)[0]          # zero first component
        v /= np.linalg.norm(v)
        others = np.linalg.qr(np.column_stack([v, rng.standard_normal((k, k - 1))]))[0][:, 1:]
        T = -5.0 * np.outer(v, v) + others @ np.diag(np.linspace(1.0, 4.0, k - 1)) @ others.T
        return T, np.eye(k)
    A = rng.standard_normal((k, k))
    S = rng.standard_normal((k, k))
    return A + A.T - 1.0 * np.eye(k), S @ S.T / k + np.eye(k)


def pad(T: Any, G: Any, capacity: int, garbage: float = 0.0) -> tuple[Any, Any]:
    """Embed in capacity x capacity arrays; the inactive part holds garbage,
    which the kernels must ignore."""
    k = T.shape[0]
    Tp = np.full((capacity, capacity), garbage)
    Gp = np.full((capacity, capacity), garbage)
    Tp[:k, :k], Gp[:k, :k] = T, G
    return Tp, Gp


CASES = [(k, seed, hard) for k in (1, 2, 5, 9) for seed in (0, 1) for hard in (False, True)
         if not (hard and k < 3)]


@pytest.mark.parametrize("k,seed,hard", CASES)
@pytest.mark.parametrize("saddle_free", [False, True])
def test_padding_does_not_change_the_solution(k: int, seed: int, hard: bool, saddle_free: bool) -> None:
    T, G = projected_problem(k, seed, hard)
    for radius in (1e-2, 1.0, 1e3):
        ref = solve_projected_trs(T, G, 1.3, radius, saddle_free)
        for capacity in (k, k + 1, 2 * k + 3):
            for garbage in (0.0, 7.0):
                Tp, Gp = pad(T, G, capacity, garbage)
                out = solve_projected_trs(strict(Tp), strict(Gp), strict(1.3), strict(radius),
                                          saddle_free, backend=strict_backend, k=strict(k))
                y, y_flip, lam, model, true_model, theta = (np.asarray(plain(v)) for v in out)
                scale = max(1.0, float(np.max(np.abs(ref[0]))))
                assert np.max(np.abs(y[:k] - ref[0])) <= 1e-9 * scale
                assert np.max(np.abs(y_flip[:k] - ref[1])) <= 1e-9 * scale
                assert np.all(y[k:] == 0.0) or np.max(np.abs(y[k:])) <= 1e-12 * scale
                assert abs(float(lam) - float(ref[2])) <= 1e-9 * max(1.0, abs(float(ref[2])))
                assert abs(float(model) - float(ref[3])) <= 1e-9 * max(1.0, abs(float(ref[3])))
                assert abs(float(true_model) - float(ref[4])) <= 1e-9 * max(1.0, abs(float(ref[4])))
                assert np.max(np.abs(theta[:k] - ref[5])) <= 1e-9 * max(1.0, float(np.max(np.abs(ref[5]))))
                if capacity > k:
                    assert np.min(theta[k:]) > np.max(np.abs(theta[:k]))     # pads last, and not negative


@pytest.mark.parametrize("k,seed,hard", CASES)
def test_numpy_cuts_the_padded_problem_down_exactly(k: int, seed: int, hard: bool) -> None:
    T, G = projected_problem(k, seed, hard)
    Tp, Gp = pad(T, G, 2 * k + 1, garbage=3.0)
    for saddle_free in (False, True):
        ref = solve_projected_trs(T, G, 1.3, 0.7, saddle_free)
        out = solve_projected_trs(Tp, Gp, 1.3, 0.7, saddle_free, backend=numpy_backend, k=k)
        assert out[0][:k].tobytes() == ref[0].tobytes() and np.all(out[0][k:] == 0.0)
        assert out[1][:k].tobytes() == ref[1].tobytes()
        assert [float(v) for v in out[2:5]] == [float(v) for v in ref[2:5]]
        assert out[5][:k].tobytes() == ref[5].tobytes() and np.all(np.isinf(out[5][k:]))


def test_ball_solver_cases_on_the_strict_backend() -> None:
    """Interior, boundary and hard case, traced the way a compiler would."""
    rng = np.random.default_rng(3)
    U = np.linalg.qr(rng.standard_normal((5, 5)))[0]
    b = rng.standard_normal(5)
    hard_b = b - (U[:, 0] @ b) * U[:, 0]
    for evals, rhs, radius, expect_lam_zero in (
            (np.linspace(1.0, 3.0, 5), b, 100.0, True),            # interior
            (np.linspace(-1.0, 3.0, 5), b, 0.5, False),            # boundary
            (np.linspace(-1.0, 3.0, 5), hard_b, 100.0, False)):    # hard case
        z_ref, lam_ref = solve_sphere_trs_eig(evals, U, rhs, radius)
        z, lam = solve_sphere_trs_eig(strict(evals), strict(U), strict(rhs), strict(radius),
                                      backend=strict_backend)
        assert np.max(np.abs(np.asarray(plain(z)) - z_ref)) <= 1e-12 * max(1.0, radius)
        assert abs(float(np.asarray(plain(lam))) - float(lam_ref)) <= 1e-12
        assert (float(lam_ref) == 0.0) == expect_lam_zero
