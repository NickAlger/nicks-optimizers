"""Matrix-free test problems, written the way user code would be: plain numpy
inside, wrapped in Vector(...) / Covector(...) on the way out."""

import threading
from typing import Callable

import numpy as np

from nicks_optimizers import (
    Array,
    BallDualInner,
    BallInner,
    Coords,
    Covector,
    CovectorBlock,
    Point,
    Vector,
    VectorBlock,
    euclidean_dual_inner,
    euclidean_inner,
    tuple_block_space,
)


SMOOTHING = 10.0


COUPLING = 0.1


def laplacian_apply(v: Array) -> Array:
    """1-D Laplacian with Neumann ends."""
    dv = np.diff(v)
    lv = np.zeros_like(v)
    lv[:-1] -= dv
    lv[1:] += dv
    return lv


def laplacian_diagonal(n: int) -> Array:
    d = 2.0 * np.ones(n)
    d[0] = d[-1] = 1.0
    return d


def solve_symmetric_tridiagonal(d: Array, e: float, r: Array) -> Array:
    """Solve (diag(d) + e (sub- and superdiagonal of ones)) x = r (Thomas algorithm)."""
    n = d.shape[0]
    c, y = np.zeros(n), np.zeros(n)
    c[0], y[0] = e / d[0], r[0] / d[0]
    for i in range(1, n):
        denom = d[i] - e * c[i - 1]
        c[i] = e / denom
        y[i] = (r[i] - e * y[i - 1]) / denom
    x = np.zeros(n)
    x[-1] = y[-1]
    for i in range(n - 2, -1, -1):
        x[i] = y[i] - c[i] * x[i + 1]
    return x


def smoothing_inner(x: Point, U: VectorBlock, v: Vector) -> Coords:
    """(u, v)_M with M = B^{-1} = I + SMOOTHING L, the precision of a smoothing
    covariance B. M is applied once per call, to v."""
    return U.T @ (v + SMOOTHING * laplacian_apply(v))


def smoothing_dual_inner(x: Point, Psi: CovectorBlock, phi: Covector) -> Coords:
    """(psi, phi)_{M^{-1}} = <psi, B phi>, one tridiagonal solve per call."""
    d = 1.0 + SMOOTHING * laplacian_diagonal(phi.shape[0])
    return Psi.T @ solve_symmetric_tridiagonal(d, -SMOOTHING, phi)


def weighted_inner(x: Point, U: VectorBlock, v: Vector) -> Coords:
    """(u, v)_M(x) with M(x) = diag(1 + x^2) + SMOOTHING L, which depends on x."""
    return U.T @ ((1.0 + x**2) * v + SMOOTHING * laplacian_apply(v))


def weighted_dual_inner(x: Point, Psi: CovectorBlock, phi: Covector) -> Coords:
    d = 1.0 + x**2 + SMOOTHING * laplacian_diagonal(phi.shape[0])
    return Psi.T @ solve_symmetric_tridiagonal(d, -SMOOTHING, phi)


METRICS: list[tuple[str, BallInner, BallDualInner]] = [
    ("identity ball", euclidean_inner, euclidean_dual_inner),
    ("smoothing ball, M = I + 10 L", smoothing_inner, smoothing_dual_inner),
    ("x-dependent ball, M(x) = diag(1+x^2) + 10 L", weighted_inner, weighted_dual_inner),
]


def rosen_fun(x: Point) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def rosen_grad(x: Point) -> Covector:
    d = x[1:] - x[:-1] ** 2
    g = np.zeros_like(x)
    g[:-1] = -400.0 * x[:-1] * d - 2.0 * (1.0 - x[:-1])
    g[1:] += 200.0 * d
    return Covector(g)


def rosen_hess_diag(x: Array) -> Array:
    h = np.zeros_like(x)
    h[:-1] = 1200.0 * x[:-1] ** 2 - 400.0 * x[1:] + 2.0
    h[1:] += 200.0
    return h


def rosen_hessvec(x: Point, v: Vector) -> Covector:
    off = -400.0 * x[:-1]
    hv = rosen_hess_diag(x) * v
    hv[:-1] += off * v[1:]
    hv[1:] += off * v[:-1]
    return Covector(hv)


def jacobi_precond_solve(x: Point, w: Covector) -> Vector:
    """P(x) = diag(max(|H_ii(x)|, 1)): a Hessian preconditioner that changes with x."""
    return Vector(w / np.maximum(np.abs(rosen_hess_diag(x)), 1.0))


def well_fun(x: Point) -> float:
    """Chain of double wells: sum(x^4/4 - x^2/2) + COUPLING/2 sum((x_{i+1} - x_i)^2).
    The origin is a local maximum; there are many saddles and minimizers."""
    return float(np.sum(0.25 * x**4 - 0.5 * x**2) + 0.5 * COUPLING * np.sum(np.diff(x) ** 2))


def well_grad(x: Point) -> Covector:
    return Covector(x**3 - x + COUPLING * laplacian_apply(x))


def well_hessvec(x: Point, v: Vector) -> Covector:
    return Covector((3.0 * x**2 - 1.0) * v + COUPLING * laplacian_apply(v))


def well_precond_solve(x: Point, w: Covector) -> Vector:
    diag = 3.0 * x**2 - 1.0 + COUPLING * laplacian_diagonal(x.shape[0])
    return Vector(w / np.maximum(np.abs(diag), 0.1))


PairArray = tuple[Array, Array]


def split_pair(x: Array) -> PairArray:
    h = x.shape[0] // 2
    return (x[:h].copy(), x[h:].copy())


def join_pair(x: PairArray) -> Array:
    return np.concatenate(x)


def pa_scale(a: float, v: PairArray) -> PairArray:
    return (a * v[0], a * v[1])


def pa_axpy(a: float, v: PairArray, w: PairArray) -> PairArray:
    return (a * v[0] + w[0], a * v[1] + w[1])


def pa_pair(w: PairArray, v: PairArray) -> float:
    return float(w[0] @ v[0] + w[1] @ v[1])


def pa_dimension(v: PairArray) -> int:
    return int(v[0].shape[0] + v[1].shape[0])


def pa_translate(x: PairArray, s: PairArray) -> PairArray:
    return (x[0] + s[0], x[1] + s[1])


PAIR_SPACE = tuple_block_space(
    scale_vector=pa_scale, scale_covector=pa_scale, axpy_vector=pa_axpy,
    axpy_covector=pa_axpy, pair=pa_pair, translate=pa_translate, dimension=pa_dimension)


def pa_rosen_fun(x: PairArray) -> float:
    return rosen_fun(Point(join_pair(x)))


def pa_rosen_grad(x: PairArray) -> PairArray:
    return split_pair(rosen_grad(Point(join_pair(x))))


def pa_rosen_hessvec(x: PairArray, v: PairArray) -> PairArray:
    return split_pair(rosen_hessvec(Point(join_pair(x)), Vector(join_pair(v))))


def pa_jacobi_precond_solve(x: PairArray, w: PairArray) -> PairArray:
    return split_pair(jacobi_precond_solve(Point(join_pair(x)), Covector(join_pair(w))))


def pa_smoothing_inner(x: PairArray, U: tuple[PairArray, ...], v: PairArray) -> Coords:
    """The smoothing-ball inner products, with blocks as tuples (M applied once)."""
    vv = join_pair(v)
    Mv = vv + SMOOTHING * laplacian_apply(vv)
    return np.array([join_pair(u) @ Mv for u in U], dtype=float)


def pa_smoothing_dual_inner(x: PairArray, Psi: tuple[PairArray, ...], phi: PairArray) -> Coords:
    pp = join_pair(phi)
    Bphi = solve_symmetric_tridiagonal(1.0 + SMOOTHING * laplacian_diagonal(pp.shape[0]), -SMOOTHING, pp)
    return np.array([join_pair(p) @ Bphi for p in Psi], dtype=float)


COUPLED = 0.5


CoupledProblem = tuple[
    Callable[[Point], float], Callable[[Point], Covector], Callable[[Point, Vector], Covector],
    Callable[[Point, Covector], Vector], BallInner, BallDualInner]


def coupled_problem(b: Array, d: Array, allreduce: Callable[[Coords], Coords]) -> CoupledProblem:
    """f(x) = sum(x^4/4 + x^2/2 - b x) + COUPLED/2 (sum x)^2 on one process's
    share of x, with the trust region M = diag(d). Every global quantity goes
    through allreduce, as it would with MPI."""
    def total(value: float) -> float:
        return float(allreduce(np.array([value]))[0])

    def fun(x: Point) -> float:
        return total(float(np.sum(0.25 * x**4 + 0.5 * x**2 - b * x))) + 0.5 * COUPLED * total(float(np.sum(x))) ** 2

    def grad(x: Point) -> Covector:
        return Covector(x**3 + x - b + COUPLED * total(float(np.sum(x))))

    def hessvec(x: Point, v: Vector) -> Covector:
        return Covector((3.0 * x**2 + 1.0) * v + COUPLED * total(float(np.sum(v))))

    def precond_solve(x: Point, w: Covector) -> Vector:
        return Vector(w / (3.0 * x**2 + 1.0 + COUPLED))

    def inner(x: Point, U: VectorBlock, v: Vector) -> Coords:
        return allreduce(U.T @ (d * v))

    def dual_inner(x: Point, Psi: CovectorBlock, phi: Covector) -> Coords:
        return allreduce(Psi.T @ (phi / d))

    return fun, grad, hessvec, precond_solve, inner, dual_inner


def make_thread_allreduce(n_ranks: int, timeout: float = 30.0) -> Callable[[int], Callable[[Coords], Coords]]:
    """An allreduce for threads standing in for MPI ranks: every rank gets the
    same sum, added in the same order, so the results are bitwise identical."""
    barrier = threading.Barrier(n_ranks, timeout=timeout)
    slots: list[Coords] = [np.zeros(0) for _ in range(n_ranks)]

    def for_rank(rank: int) -> Callable[[Coords], Coords]:
        def allreduce(local: Coords) -> Coords:
            slots[rank] = np.array(local, dtype=float)
            barrier.wait()
            total: Coords = slots[0].copy()
            for r in range(1, n_ranks):
                total = total + slots[r]
            barrier.wait()                 # nobody overwrites a slot before all have read
            return total
        return allreduce
    return for_rank
