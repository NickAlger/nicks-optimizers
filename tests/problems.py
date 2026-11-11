"""Matrix-free test problems, written the way user code would be: plain numpy
inside, wrapped in Vector(...) / Covector(...) on the way out."""

import threading
from typing import Callable, NamedTuple

import numpy as np

from nicks_optimizers import (
    Array,
    Coords,
    Covector,
    MetricApply,
    MetricSolve,
    Point,
    Vector,
    euclidean_metric_apply,
    euclidean_metric_solve,
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


def smoothing_metric_apply(x: Point, v: Vector) -> Covector:
    """M v with M = B^{-1} = I + SMOOTHING L, the precision of a smoothing
    covariance B."""
    return Covector(v + SMOOTHING * laplacian_apply(v))


def smoothing_metric_solve(x: Point, w: Covector) -> Vector:
    """M^{-1} w = B w, one tridiagonal solve."""
    d = 1.0 + SMOOTHING * laplacian_diagonal(w.shape[0])
    return Vector(solve_symmetric_tridiagonal(d, -SMOOTHING, w))


def weighted_metric_apply(x: Point, v: Vector) -> Covector:
    """M(x) v with M(x) = diag(1 + x^2) + SMOOTHING L, which depends on x."""
    return Covector((1.0 + x**2) * v + SMOOTHING * laplacian_apply(v))


def weighted_metric_solve(x: Point, w: Covector) -> Vector:
    d = 1.0 + x**2 + SMOOTHING * laplacian_diagonal(w.shape[0])
    return Vector(solve_symmetric_tridiagonal(d, -SMOOTHING, w))


METRICS: list[tuple[str, MetricSolve, MetricApply]] = [
    ("identity ball", euclidean_metric_solve, euclidean_metric_apply),
    ("smoothing ball, M = I + 10 L", smoothing_metric_solve, smoothing_metric_apply),
    ("x-dependent ball, M(x) = diag(1+x^2) + 10 L", weighted_metric_solve, weighted_metric_apply),
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


def pa_retract(x: PairArray, s: PairArray) -> PairArray:
    return (x[0] + s[0], x[1] + s[1])


PAIR_SPACE = tuple_block_space(
    scale_vector=pa_scale, scale_covector=pa_scale, axpy_vector=pa_axpy,
    axpy_covector=pa_axpy, pair=pa_pair, dimension=pa_dimension)


def pa_rosen_fun(x: PairArray) -> float:
    return rosen_fun(Point(join_pair(x)))


def pa_rosen_grad(x: PairArray) -> PairArray:
    return split_pair(rosen_grad(Point(join_pair(x))))


def pa_rosen_hessvec(x: PairArray, v: PairArray) -> PairArray:
    return split_pair(rosen_hessvec(Point(join_pair(x)), Vector(join_pair(v))))


def pa_jacobi_precond_solve(x: PairArray, w: PairArray) -> PairArray:
    return split_pair(jacobi_precond_solve(Point(join_pair(x)), Covector(join_pair(w))))


def pa_smoothing_metric_apply(x: PairArray, v: PairArray) -> PairArray:
    """The smoothing metric on pairs of arrays."""
    return split_pair(smoothing_metric_apply(Point(join_pair(x)), Vector(join_pair(v))))


def pa_smoothing_metric_solve(x: PairArray, w: PairArray) -> PairArray:
    return split_pair(smoothing_metric_solve(Point(join_pair(x)), Covector(join_pair(w))))


COUPLED = 0.5


CoupledProblem = tuple[
    Callable[[Point], float], Callable[[Point], Covector], Callable[[Point, Vector], Covector],
    Callable[[Point, Covector], Vector], MetricSolve, MetricApply]


def coupled_problem(b: Array, d: Array, allreduce: Callable[[Coords], Coords]) -> CoupledProblem:
    """f(x) = sum(x^4/4 + x^2/2 - b x) + COUPLED/2 (sum x)^2 on one process's
    share of x, with the metric M = diag(d) (local operators: the space's
    pairing does the reductions). Every global quantity goes
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

    def metric_solve(x: Point, w: Covector) -> Vector:
        return Vector(w / d)

    def metric_apply(x: Point, v: Vector) -> Covector:
        return Covector(d * v)

    return fun, grad, hessvec, precond_solve, metric_solve, metric_apply


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


# --- least squares with a Gauss-Newton and a full Hessian (the switch) ----------------------


class LeastSquares(NamedTuple):
    """f = 1/2 ||r(x)||^2 with its gradient J^T r, the Gauss-Newton Hessian
    J^T J and the full Hessian J^T J + S, S = sum_k r_k Hess(r_k), and a
    start. The functions are traceable (no float() of values, no Python
    control flow on data), so they run on the strict backend too."""
    fun: Callable[[Point], float]
    grad: Callable[[Point], Covector]
    gn_hessvec: Callable[[Point, Vector], Covector]
    hessvec: Callable[[Point, Vector], Covector]
    x0: Point


def regularized_rosenbrock(n: int = 20, alpha: float = 0.1, cscale: float = 0.5,
                           seed: int = 1, a: float = 10.0) -> LeastSquares:
    """The regularized extended Rosenbrock problem of the switching study:
    for each pair (u, w) = (x[2k], x[2k+1]) the residuals a (w - u^2) and
    1 - u, plus sqrt(alpha) (x - c) with c = 1 + cscale N(0, 1). With
    cscale > 0 the residual at the solution is nonzero and Gauss-Newton
    converges linearly; with cscale = 0 the solution x = 1 has residual
    zero. S is diagonal on the u components: -2 a times the valley residual.
    Start (-1.2, 1, -1.2, 1, ...)."""
    c = 1.0 + cscale * np.random.default_rng(seed).standard_normal(n)
    root = float(np.sqrt(alpha))

    def parts(x: Point) -> tuple[Array, Array, Array, Array, Array]:
        u, w = x[0::2], x[1::2]
        return u, w, a * (w - u**2), 1.0 - u, root * (x - c)

    def interleave(du: Array, dw: Array) -> Array:
        out: Array = np.stack([du, dw], axis=1).reshape(-1)
        return out

    def fun(x: Point) -> float:
        _, _, rv, rl, rr = parts(x)
        value: float = 0.5 * (np.sum(rv**2) + np.sum(rl**2) + np.sum(rr**2))
        return value

    def grad(x: Point) -> Covector:
        u, _, rv, rl, rr = parts(x)
        return Covector(interleave(-2.0 * a * u * rv - rl, a * rv) + root * rr)

    def gn_hessvec(x: Point, v: Vector) -> Covector:
        u = x[0::2]
        vu, vw = v[0::2], v[1::2]
        jv = a * (vw - 2.0 * u * vu)
        return Covector(interleave(-2.0 * a * u * jv + vu, a * jv) + alpha * v)

    def hessvec(x: Point, v: Vector) -> Covector:
        _, _, rv, _, _ = parts(x)
        s = interleave(-2.0 * a * rv * v[0::2], 0.0 * v[1::2])
        return Covector(gn_hessvec(x, v) + s)

    return LeastSquares(fun, grad, gn_hessvec, hessvec, Point(np.tile([-1.2, 1.0], n // 2)))


def linear_least_squares(m: int = 30, n: int = 12, seed: int = 0) -> LeastSquares:
    """f = 1/2 ||A x - b||^2: S = 0, so the Gauss-Newton and the full Hessian
    are the same operator (computed by separate functions)."""
    rng = np.random.default_rng(seed)
    A, b = rng.standard_normal((m, n)), rng.standard_normal(m)

    def fun(x: Point) -> float:
        value: float = 0.5 * np.sum((A @ x - b) ** 2)
        return value

    def grad(x: Point) -> Covector:
        return Covector(A.T @ (A @ x - b))

    def gn_hessvec(x: Point, v: Vector) -> Covector:
        return Covector(A.T @ (A @ v))

    def hessvec(x: Point, v: Vector) -> Covector:
        return Covector((A.T @ A) @ v)

    return LeastSquares(fun, grad, gn_hessvec, hessvec, Point(np.zeros(n)))
