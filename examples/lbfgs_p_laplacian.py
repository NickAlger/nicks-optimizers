"""L-BFGS in the geometry of a metric: the p-Laplacian energy with constant
and point-dependent metrics, and a preconditioner.

lbfgs takes the metric M by its two operators, M^{-1} (metric_solve) and M
(metric_apply), and builds its quasi-Newton approximation on B0 = M / gamma;
every norm and fit is taken in M. This example shows the ways to give M and
P, on the convex energy

    E(u) = h^2 [ sum over edges of (eps^2 + |du/h|^2)^(p/2) / p - sum of u ]

with N x N interior nodes, p = 4, eps = 1e-2, from u = 0 (scipy sparse, the
gradient by hand):

  Euclidean     M = I, the default for numpy arrays
  H1            the discrete H^1 inner product, constant: fixed_metric=True
                keeps M^{-1} y and M s with each stored pair, so an
                iteration applies M^{-1} once (for the gradient norm) and M
                never, with the iterates of the general path
  Kacanov       M(u) = Dx^T S(u) Dx + Dy^T S(u) Dy, S = h^2 (eps^2 + |grad u|^2)^((p-2)/2),
                the Hessian without its rank-structured part, so
                M(u) <= H(u) <= (p - 1) M(u). A new matrix at every point:
                operators_at and point_data factorize it once per point. With
                memory=1: on this problem older pairs, formed under a
                different metric, conflict with the new one
  H1, P Kacanov M = H1 for the geometry (norms, angles, the M direction), and
                P = M(u) as the accelerant: its direction is taken when it
                passes the angle test in H1 (precond_mode="safeguarded")

Printed per run: iterations, gradients, solves with M and with P, products
with M, factorizations, seconds. experiments/lbfgs_metrics/ studies these
choices further, and says what one problem does not show: other problems,
or multigrid instead of sparse LU, can favor other choices.

Run:  python examples/lbfgs_p_laplacian.py [--N 31 63 127]
"""

import argparse
import time
from typing import Any, Callable, NamedTuple

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from nicks_optimizers import Covector, LbfgsSummary, Point, PointOperators, Vector, lbfgs


class Row(NamedTuple):
    N: int
    config: str
    reason: str
    iterations: int
    gradients: int
    metric_solves: int
    precond_solves: int
    metric_products: int
    factorizations: int
    precond_directions: int
    seconds: float
    f: float


class Problem(NamedTuple):
    fun: Callable[[Point], float]
    grad: Callable[[Point], Covector]
    x0: Point
    configs: dict[str, dict[str, Any]]
    factorizations: list[int]


def p_laplacian(N: int, p: float = 4.0, eps: float = 1e-2) -> Problem:
    """The energy, its gradient, and the options of each configuration."""
    h = 1.0 / (N + 1)
    n = N * N
    D = sps.diags([-np.ones(N - 1), np.ones(N)], [-1, 0], shape=(N, N)) / h
    eye = sps.identity(N)
    Dx, Dy = sps.kron(eye, D).tocsr(), sps.kron(D, eye).tocsr()
    load = np.full(n, h * h)
    count = [0]

    def factor(A: Any) -> Any:
        count[0] += 1
        return spla.splu(A.tocsc())

    def weights(u: Point) -> tuple[Any, Any, Any]:
        gx, gy = Dx @ u, Dy @ u
        return gx, gy, h * h * (gx ** 2 + gy ** 2 + eps ** 2) ** (p / 2 - 1)

    def fun(u: Point) -> float:
        gx, gy = Dx @ u, Dy @ u
        return float(h * h * np.sum((gx ** 2 + gy ** 2 + eps ** 2) ** (p / 2)) / p - load @ u)

    def grad(u: Point) -> Covector:
        gx, gy, s = weights(u)
        return Covector(Dx.T @ (s * gx) + Dy.T @ (s * gy) - load)

    H1 = ((Dx.T @ Dx + Dy.T @ Dy) * h * h + h * h * sps.identity(n)).tocsc()
    h1: dict[str, Any] = {}

    def h1_solve(u: Point, w: Covector) -> Vector:
        if "lu" not in h1:                        # factorized once, at the first solve
            h1["lu"] = factor(H1)
        return Vector(h1["lu"].solve(w))

    def h1_apply(u: Point, v: Vector) -> Covector:
        return Covector(H1 @ v)

    def kacanov(u: Point) -> tuple[Any, Any]:
        """point_data: M(u) and its factorization, once per point."""
        _, _, s = weights(u)
        S = sps.diags(s)
        A = (Dx.T @ S @ Dx + Dy.T @ S @ Dy).tocsc()
        return A, factor(A)

    def kacanov_operators(u: Point, data: Any) -> PointOperators[Vector, Covector]:
        A, lu = data
        return PointOperators(metric_solve=lambda w: Vector(lu.solve(w)), metric_apply=lambda v: Covector(A @ v))

    def h1_with_kacanov(u: Point, data: Any) -> PointOperators[Vector, Covector]:
        _, lu = data
        return PointOperators(metric_solve=lambda w: h1_solve(u, w), metric_apply=lambda v: h1_apply(u, v),
                              precond_solve=lambda w: Vector(lu.solve(w)))

    configs: dict[str, dict[str, Any]] = {
        "Euclidean": dict(),
        "H1": dict(metric_solve=h1_solve, metric_apply=h1_apply, fixed_metric=True),
        "Kacanov": dict(operators_at=kacanov_operators, point_data=kacanov, memory=1),
        "H1, P Kacanov": dict(operators_at=h1_with_kacanov, point_data=kacanov, fixed_metric=True,
                              memory=1),
    }
    return Problem(fun=fun, grad=grad, x0=Point(np.zeros(n)), configs=configs, factorizations=count)


def run(N: int, config: str, max_iter: int = 5000) -> tuple[Row, LbfgsSummary[Any, Any]]:
    problem = p_laplacian(N)
    t = time.perf_counter()
    _, s = lbfgs(problem.fun, problem.grad, problem.x0, max_iter=max_iter, **problem.configs[config])
    return Row(N=N, config=config, reason=s.reason, iterations=len(s.iterations), gradients=s.grad_calls,
               metric_solves=s.metric_solve_calls, precond_solves=s.precond_solve_calls,
               metric_products=s.metric_apply_calls, factorizations=problem.factorizations[0],
               precond_directions=s.precond_directions, seconds=time.perf_counter() - t, f=s.f), s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--N", type=int, nargs="+", default=[31, 63, 127])
    args = parser.parse_args()
    print(f"{'N':>4s} {'configuration':14s} {'its':>5s} {'grad':>5s} {'M^-1':>5s} {'P^-1':>5s} {'M':>5s} "
          f"{'LU':>4s} {'P dirs':>6s} {'time':>7s}  f")
    for N in args.N:
        for config in ("Euclidean", "H1", "Kacanov", "H1, P Kacanov"):
            row, _ = run(N, config)
            print(f"{N:4d} {config:14s} {row.iterations:5d} {row.gradients:5d} {row.metric_solves:5d} "
                  f"{row.precond_solves:5d} {row.metric_products:5d} {row.factorizations:4d} "
                  f"{row.precond_directions:6d} {row.seconds:6.2f}s  {row.f:.10e}", flush=True)


if __name__ == "__main__":
    main()
