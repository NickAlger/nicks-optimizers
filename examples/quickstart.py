"""Quickstart: chained Rosenbrock with a Jacobi preconditioner and an
ellipsoidal trust region given by a diagonal covariance.

Run:  python examples/quickstart.py
"""

import numpy as np

from nicks_optimizers import (
    Coords, Covector, CovectorBlock, Point, Vector, VectorBlock, trust_region_newton,
)

N = 50
B = np.linspace(0.5, 2.0, N)        # the ball's covariance: diagonal, B = diag(b)


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


def precond_solve(x: Point, w: Covector) -> Vector:
    """P(x) = diag(max(|H_ii(x)|, 1)); it only shapes the Krylov space."""
    return Vector(w / np.maximum(np.abs(hessian_diagonal(x)), 1.0))


def ball_inner(x: Point, U: VectorBlock, v: Vector) -> Coords:
    """(u_i, v)_M with M = B^{-1}: the trust region is an ellipsoid with covariance B."""
    return U.T @ (v / B)


def ball_dual_inner(x: Point, Psi: CovectorBlock, phi: Covector) -> Coords:
    """(psi_i, phi)_{M^{-1}} = <psi_i, B phi>, the exact dual of ball_inner."""
    return Psi.T @ (B * phi)


if __name__ == "__main__":
    x0 = Point(np.tile([-1.2, 1.0], N // 2))
    for saddle_free in (False, True):
        print(f"\n=== saddle_free={saddle_free} ===")
        x, summary = trust_region_newton(fun, grad, hessvec, precond_solve, ball_inner,
                                         ball_dual_inner, x0, saddle_free=saddle_free, display=1)
