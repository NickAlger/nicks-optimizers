"""Dense reference algebra and small utilities shared by the tests."""

import dataclasses
from typing import Callable

import numpy as np

from nicks_optimizers import (
    Array,
    Coords,
    Covector,
    CovectorBlock,
    CovectorToVector,
    IterationRecord,
    LanczosState,
    RunSummary,
    Vector,
    VectorBlock,
    VectorToCovector,
    format_header,
    format_krylov_trace,
    format_row,
    format_summary,
    lanczos_advance,
    lanczos_start,
    lanczos_tridiagonal,
    numpy_space,
    solve_tr_subproblem,
)
from nicks_optimizers.metrics import extend_metric_gram


Rng = np.random.Generator


def random_orthogonal(n: int, rng: Rng) -> Array:
    Q, R = np.linalg.qr(rng.standard_normal((n, n)))
    signed: Array = Q * np.sign(np.diag(R))
    return signed


def matrix_with_spectrum(evals: Array, rng: Rng) -> Array:
    U = random_orthogonal(len(evals), rng)
    A: Array = (U * evals) @ U.T
    return A


def random_spd(n: int, cond: float, rng: Rng) -> Array:
    return matrix_with_spectrum(np.logspace(0.0, np.log10(cond), n), rng)


def dense_vec_to_covec(A: Array) -> VectorToCovector:
    """A dense matrix as a Vector -> Covector operator (the Hessian)."""
    return lambda v: Covector(A @ v)


def dense_solve(A: Array) -> CovectorToVector:
    """A dense matrix solve as a Covector -> Vector operator (M^{-1}, P^{-1})."""
    return lambda w: Vector(np.linalg.solve(A, w))


def dense_apply(M: Array) -> VectorToCovector:
    """A dense metric as a Vector -> Covector operator (M)."""
    return lambda v: Covector(M @ v)


def dense_abs_m(H: Array, M: Array) -> Array:
    """|H|_M = M U |Lambda| U^T M from the generalized eigenproblem H u = lambda M u."""
    Linv = np.linalg.inv(np.linalg.cholesky(M))
    lam, Y = np.linalg.eigh(Linv @ H @ Linv.T)
    U = Linv.T @ Y                               # M-orthonormal eigenvectors
    absH: Array = M @ (U * np.abs(lam)) @ U.T @ M
    return absH


def dense_dual_norm(r: Array, M: Array) -> float:
    return float(np.sqrt(r @ np.linalg.solve(M, r)))


def report(name: str, value: float, tol: float) -> bool:
    ok = value <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {value:.2e} (tol {tol:.0e})")
    return ok


def run_lanczos(H: Array, P: Array, M: Array, g: Array, k: int, reorth_passes: int = 2
                ) -> tuple[LanczosState[Vector, Covector, VectorBlock, CovectorBlock], Coords, Coords]:
    """k steps of the Lanczos process with dense H, P, M wrapped as operators,
    keeping the M-Gram matrix the way the trust-region solver does.

    Returns (state, T, G), where G = Q^T M Q.
    """
    sp = numpy_space
    hessvec, precond_solve, metric_apply = dense_vec_to_covec(H), dense_solve(P), dense_apply(M)
    state = lanczos_start(sp, Covector(g), precond_solve)
    G = sp.pair_block(state.Q, metric_apply(state.q)).reshape(1, 1)
    for j in range(k):
        state = lanczos_advance(sp, state, hessvec, precond_solve, reorth_passes=reorth_passes)
        if j < k - 1:
            G = extend_metric_gram(sp, G, state.Q, state.q_next, metric_apply)
    return state, lanczos_tridiagonal(state), G


def sphere_kkt_error(A: Array, b: Array, radius: float, z: Array, lam: float) -> float:
    k = len(b)
    znorm = float(np.linalg.norm(z))
    scale = max(1.0, float(np.linalg.norm(A, 2)))
    stationarity = float(np.linalg.norm((A + lam * np.eye(k)) @ z + b)) / max(1.0, float(np.linalg.norm(b)))
    feasibility = max(0.0, znorm - radius) / radius
    complementarity = lam * abs(znorm - radius) / (scale * radius)
    curvature = max(0.0, -float(np.linalg.eigvalsh(A + lam * np.eye(k))[0])) / scale
    return max(stationarity, feasibility, complementarity, curvature, max(0.0, -lam))


def full_space_kkt_error(Hm: Array, M: Array, g: Array, radius: float, s: Array, lam: float) -> float:
    """KKT + second-order conditions for min <g,s> + 0.5 <Hm s, s> s.t. ||s||_M <= radius."""
    snorm = float(np.sqrt(s @ M @ s))
    Linv = np.linalg.inv(np.linalg.cholesky(M))
    scale = max(1.0, float(np.linalg.norm(Linv @ Hm @ Linv.T, 2)))
    stationarity = dense_dual_norm(g + Hm @ s + lam * M @ s, M) / dense_dual_norm(g, M)
    feasibility = max(0.0, snorm - radius) / radius
    complementarity = lam * abs(snorm - radius) / (scale * radius)
    curvature = max(0.0, -float(np.linalg.eigvalsh(Linv @ (Hm + lam * M) @ Linv.T)[0])) / scale
    return max(stationarity, feasibility, complementarity, curvature, max(0.0, -lam))


def run_dense_subproblem(H: Array, P: Array, M: Array, g: Array, radius: float, max_iter: int,
                         resid_tol: float = 0.0, stall_tol: float = 0.0, saddle_free: bool = False
                         ) -> tuple[Vector, float, float, float, int, float, int]:
    """Dense Lanczos subproblem with metric M and preconditioner P (passed
    separately, even when equal); returns
    (s, model_value, step_norm, lam, k, resid, n_negative)."""
    s, st = solve_tr_subproblem(
        numpy_space, Covector(g), dense_vec_to_covec(H), dense_solve(M), radius, resid_tol,
        max_iter, precond_solve=dense_solve(P), metric_apply=dense_apply(M),
        stall_tol=stall_tol, saddle_free=saddle_free)
    return s, st.model_value, st.step_norm, st.lam, st.krylov_steps, st.resid, st.n_negative


def dense_from_operator(apply: Callable[[Vector], Covector], n: int) -> Array:
    """Dense matrix of a linear operator, for checks only."""
    I = np.eye(n)
    dense: Array = np.column_stack([apply(Vector(I[:, j])) for j in range(n)])
    return dense


def without_times(records: tuple[IterationRecord, ...]) -> list[str]:
    """Records with the wall-clock field cleared, as text for exact
    comparisons: repr gives every float exactly, and nan fields (CG steps on
    the boundary have no multiplier) compare equal."""
    return [repr(dataclasses.replace(r, elapsed=0.0)) for r in records]


def expected_display(summary: RunSummary, level: int) -> list[str]:
    """What display should have printed, rebuilt from the returned records."""
    lines: list[str] = []
    for record in summary.iterations:
        if record.iteration % 20 == 0:
            lines.append(format_header())
        lines.append(format_row(record))
        if level >= 2:
            lines.append(format_krylov_trace(record.subproblem))
    lines.append(format_summary(summary))
    return lines
