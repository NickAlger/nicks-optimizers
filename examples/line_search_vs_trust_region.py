"""Line search or trust region? line_search_newton and trust_region_newton
on the same problems, with the same operators and options.

Both are inexact Newton methods with the same Krylov solvers, metric M and
preconditioner P. They differ in how they make a step safe: the trust
region bounds the step in the M-norm and adapts the bound; the line search
takes the Newton direction and shortens it until f decreases enough. Which
spends less depends on the problem:

  rosenbrock     chained Rosenbrock (n = 20): a curved valley, where the
                 Newton direction overshoots and the trust region's radius
                 remembers the right length
  p-laplacian    a convex energy, -div(|grad u|^2 grad u) = 1 on the unit
                 square by finite differences, with three choices of M and P:
                   H^1         M = P = the discrete H^1 inner product
                   L^2 / H^1   M = h^2 I, P = H^1: a separate preconditioner,
                               which slows trust-region steps on the
                               boundary; the line search has no boundary
                   Jacobi      M = P = diag(H^1): a weak preconditioner

Printed per run: iterations, evaluations of f and of the gradient,
Hessian-vector products, solves with M or P, products with M.

Run:  python examples/line_search_vs_trust_region.py [--N 63]
"""

import argparse
from typing import Any, Callable, NamedTuple

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from nicks_optimizers import (
    Covector,
    Point,
    Vector,
    line_search_newton,
    trust_region_newton,
)


class Row(NamedTuple):
    problem: str
    method: str
    reason: str
    iterations: int
    f: float
    fun_calls: int
    grad_calls: int
    hessvec_calls: int
    solves: int
    metric_applies: int


def rosenbrock(n: int = 20) -> dict[str, Any]:
    def fun(x: Point) -> float:
        return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))

    def grad(x: Point) -> Covector:
        g = np.zeros_like(x)
        g[:-1] = -400.0 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2.0 * (1.0 - x[:-1])
        g[1:] += 200.0 * (x[1:] - x[:-1] ** 2)
        return Covector(g)

    def hessvec(x: Point, v: Vector) -> Covector:
        d = np.zeros_like(x)
        d[:-1] = 1200.0 * x[:-1] ** 2 - 400.0 * x[1:] + 2.0
        d[1:] += 200.0
        off = -400.0 * x[:-1]
        hv = d * v
        hv[:-1] += off * v[1:]
        hv[1:] += off * v[:-1]
        return Covector(hv)

    return dict(fun=fun, grad=grad, hessvec=hessvec, x0=Point(np.tile([-1.2, 1.0], n // 2)))


def p_laplacian(N: int, p: float = 4.0, eps: float = 1e-2) -> dict[str, Any]:
    """The energy h^2 [sum over edges of (eps^2 + |du/h|^2)^(p/2) / p - sum u]
    with N x N interior nodes, and the metrics."""
    h = 1.0 / (N + 1)
    n = N * N
    D = sps.diags([-np.ones(N - 1), np.ones(N)], [-1, 0], shape=(N, N)) / h
    eye = sps.identity(N)
    Dx, Dy = sps.kron(eye, D).tocsr(), sps.kron(D, eye).tocsr()
    H1 = ((Dx.T @ Dx + Dy.T @ Dy) * h * h + h * h * sps.identity(n)).tocsc()
    lu = spla.splu(H1)
    diagonal = H1.diagonal()
    f = np.full(n, h * h)

    def fun(u: Point) -> float:
        gx, gy = Dx @ u, Dy @ u
        return float(h * h * np.sum((gx ** 2 + gy ** 2 + eps ** 2) ** (p / 2)) / p - f @ u)

    def grad(u: Point) -> Covector:
        gx, gy = Dx @ u, Dy @ u
        s = h * h * (gx ** 2 + gy ** 2 + eps ** 2) ** (p / 2 - 1)
        return Covector(Dx.T @ (s * gx) + Dy.T @ (s * gy) - f)

    def hessvec(u: Point, v: Vector) -> Covector:
        gx, gy = Dx @ u, Dy @ u
        r = gx ** 2 + gy ** 2 + eps ** 2
        s, t = h * h * r ** (p / 2 - 1), h * h * (p - 2) * r ** (p / 2 - 2)
        vx, vy = Dx @ v, Dy @ v
        d = gx * vx + gy * vy
        return Covector(Dx.T @ (s * vx + t * d * gx) + Dy.T @ (s * vy + t * d * gy))

    h2 = h * h
    return dict(fun=fun, grad=grad, hessvec=hessvec, x0=Point(np.zeros(n)),
                h1_solve=lambda u, w: Vector(lu.solve(w)), jacobi_solve=lambda u, w: Vector(w / diagonal),
                l2_solve=lambda u, w: Vector(w / h2), l2_apply=lambda u, v: Covector(h2 * v))


def cases(N: int) -> list[tuple[str, dict[str, Any]]]:
    rosen = rosenbrock()
    pl = p_laplacian(N)
    base = {key: pl[key] for key in ("fun", "grad", "hessvec", "x0")}
    return [
        ("rosenbrock", rosen),
        ("p-laplacian, H^1", dict(base, metric_solve=pl["h1_solve"])),
        ("p-laplacian, L^2 / H^1", dict(base, metric_solve=pl["l2_solve"], metric_apply=pl["l2_apply"],
                                        precond_solve=pl["h1_solve"])),
        ("p-laplacian, Jacobi", dict(base, metric_solve=pl["jacobi_solve"])),
    ]


def compare(N: int = 63) -> list[Row]:
    rows: list[Row] = []
    methods: list[tuple[str, Callable[..., Any]]] = [("trust region", trust_region_newton),
                                                     ("line search", line_search_newton)]
    for name, case in cases(N):
        options = dict(case)
        fun, grad, hessvec, x0 = (options.pop(key) for key in ("fun", "grad", "hessvec", "x0"))
        for label, method in methods:
            _, s = method(fun, grad, hessvec, x0, max_iter=2000, **options)
            rows.append(Row(name, label, s.reason, len(s.iterations), s.f, s.fun_calls, s.grad_calls,
                            s.hessvec_calls, s.metric_solve_calls + s.precond_solve_calls,
                            s.metric_apply_calls))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--N", type=int, default=63, help="interior nodes per side of the p-Laplacian grid")
    args = parser.parse_args()
    print(f"{'problem':24s} {'method':13s} {'its':>5s} {'f evals':>8s} {'g evals':>8s} {'Hv':>6s} "
          f"{'solves':>7s} {'M products':>10s}  f")
    for r in compare(args.N):
        print(f"{r.problem:24s} {r.method:13s} {r.iterations:5d} {r.fun_calls:8d} {r.grad_calls:8d} "
              f"{r.hessvec_calls:6d} {r.solves:7d} {r.metric_applies:10d}  {r.f:.6e}")


if __name__ == "__main__":
    main()
