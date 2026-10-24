"""Quickstart: chained Rosenbrock, three ways.

  1. Defaults: M = P = I (Euclidean trust region, unpreconditioned CG).
  2. The usual setup: the Jacobi preconditioner as the metric, M = P. The
     trust region is measured in the preconditioner's norm, and only P^{-1}
     is ever applied (truncated CG).
  3. A metric chosen for its geometry, here an ellipsoid with a diagonal
     covariance B (M = B^{-1}), with the Jacobi preconditioner kept as a
     separate accelerant. Lanczos is used then, plain or saddle-free; it
     needs M itself as well as M^{-1}.

The stopping test is ||g||_{M^{-1}} <= 1e-8 ||g(x0)||_{M^{-1}} in each case.

Run:  python examples/quickstart.py
"""

import numpy as np

from nicks_optimizers import Covector, Point, RunSummary, Vector, trust_region_newton

N = 50
B = np.linspace(0.5, 2.0, N)        # the covariance of the ellipsoid in setup 3


def fun(x: Point) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def grad(x: Point) -> Covector:
    d = x[1:] - x[:-1] ** 2
    g = np.zeros_like(x)
    g[:-1] = -400.0 * x[:-1] * d - 2.0 * (1.0 - x[:-1])
    g[1:] += 200.0 * d
    return Covector(g)


def hessian_diagonal(x: Point) -> np.ndarray:
    h = np.zeros_like(x)
    h[:-1] = 1200.0 * x[:-1] ** 2 - 400.0 * x[1:] + 2.0
    h[1:] += 200.0
    return h


def hessvec(x: Point, v: Vector) -> Covector:
    off = -400.0 * x[:-1]
    hv = hessian_diagonal(x) * v
    hv[:-1] += off * v[1:]
    hv[1:] += off * v[:-1]
    return Covector(hv)


def jacobi_solve(x: Point, w: Covector) -> Vector:
    """P(x)^{-1} w with P(x) = diag(max(|H_ii(x)|, 1))."""
    return Vector(w / np.maximum(np.abs(hessian_diagonal(x)), 1.0))


def ellipsoid_metric_solve(x: Point, w: Covector) -> Vector:
    """M^{-1} w = B w: the trust region is an ellipsoid with covariance B."""
    return Vector(B * w)


def ellipsoid_metric_apply(x: Point, v: Vector) -> Covector:
    """M v = B^{-1} v, consistent with ellipsoid_metric_solve."""
    return Covector(v / B)


def describe(label: str, summary: RunSummary) -> None:
    krylov = sum(r.subproblem.krylov_steps for r in summary.iterations)
    solver = "/".join(sorted({r.subproblem.krylov for r in summary.iterations}))
    print(f"{label:44s} {summary.reason:9s} {len(summary.iterations):4d} iterations "
          f"{krylov:5d} {solver} steps   f = {summary.f:.6f}")


if __name__ == "__main__":
    x0 = Point(np.tile([-1.2, 1.0], N // 2))
    describe("1. M = P = I", trust_region_newton(fun, grad, hessvec, x0, max_krylov=200)[1])
    describe("2. M = P = Jacobi", trust_region_newton(fun, grad, hessvec, x0,
                                                      metric_solve=jacobi_solve)[1])
    for saddle_free in (False, True):
        _, summary = trust_region_newton(
            fun, grad, hessvec, x0, metric_solve=ellipsoid_metric_solve,
            metric_apply=ellipsoid_metric_apply, precond_solve=jacobi_solve,
            saddle_free=saddle_free)
        describe(f"3. M = B^-1, P = Jacobi{', saddle-free' if saddle_free else ''}", summary)
    print("\nSetup 2 again, its first 10 iterations in detail (display=1):")
    trust_region_newton(fun, grad, hessvec, x0, metric_solve=jacobi_solve, display=1, max_iter=10)
