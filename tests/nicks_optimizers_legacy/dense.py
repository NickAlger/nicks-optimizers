"""Small dense linear algebra in coordinates.

Everything here acts on small numpy arrays (projected matrices, coefficient
vectors), never on the optimizer's vectors, so it is independent of any
VectorSpace: tridiagonal and Gram-matrix assembly, Rayleigh-Ritz for a
projected pencil, and an exact solver for the trust-region problem on a ball
(including the hard case).
"""

import numpy as np

from .spaces import Coords


def tridiagonal(diag: Coords, offdiag: Coords) -> Coords:
    """Dense symmetric tridiagonal matrix. Uses the first len(diag)-1 off-diagonals."""
    k = len(diag)
    T: Coords = np.diag(np.asarray(diag, dtype=float))
    if k > 1:
        off = np.asarray(offdiag[: k - 1], dtype=float)
        T = T + np.diag(off, 1) + np.diag(off, -1)
    return T


def grow_gram(G: Coords, col: Coords) -> Coords:
    """Add one row and column to the symmetric matrix G.

    col has length G.shape[0] + 1; its last entry is the new diagonal entry.
    """
    off = col[:-1, None]
    grown: Coords = np.block([[G, off], [off.T, col[-1:, None]]])
    return grown


def pencil_ritz(T: Coords, G: Coords) -> tuple[Coords, Coords, Coords]:
    """Rayleigh-Ritz for the pencil H u = theta M u on the Krylov space.

    With G = L L^T, Q L^{-T} is an M-orthonormal basis of the space and
    A = L^{-1} T L^{-T} is H in that basis. Returns (L, theta, Z): the
    M-Ritz values theta (ascending) and the M-Ritz vectors Z in z-coordinates.
    """
    L = np.linalg.cholesky(G)
    A = np.linalg.solve(L, np.linalg.solve(L, T).T)
    theta, Z = np.linalg.eigh(0.5 * (A + A.T))
    return L, theta, Z


def solve_sphere_trs_eig(evals: Coords, U: Coords, b: Coords, radius: float,
                         rtol: float = 1e-12, max_iter: int = 200,
                         hard_tol: float = 1e-10) -> tuple[Coords, float]:
    """Solve  min_z b^T z + 0.5 z^T A z  s.t.  ||z|| <= radius,  given the
    eigendecomposition A = U diag(evals) U^T with evals in ascending order.

    Safeguarded Newton on the secular equation. Two details keep it accurate:
      * it solves the unit-ball problem with b / radius and rescales, so very
        small or large radii do not underflow or overflow;
      * it iterates on the shifted multiplier mu = lam + evals[0], so the
        bottom term b_0 / mu keeps full relative accuracy when the root sits
        very close to the pole (the near-hard case).
    Returns (z, lam) with (A + lam I) z = -b, lam >= 0, A + lam I positive
    semidefinite and lam (||z|| - radius) = 0, including the hard case.
    """
    lam_min = float(evals[0])
    gaps = evals - lam_min                  # >= 0, and gaps[0] == 0
    bh = (U.T @ b) / radius                 # unit-ball problem, z = radius * u
    bnorm = float(np.linalg.norm(bh))

    # Interior: A positive definite and the Newton step lies inside the ball.
    if lam_min > 0.0:
        u = -bh / evals
        if np.linalg.norm(u) <= 1.0:
            return radius * (U @ u), 0.0

    # Hard case: b (numerically) orthogonal to the bottom eigenspace, and the
    # solution at lam = -lam_min is too short to reach the boundary.
    if lam_min <= 0.0:
        scale_ = max(1.0, float(np.max(np.abs(evals))))
        bottom = gaps <= hard_tol * scale_
        if np.linalg.norm(bh[bottom]) <= hard_tol * bnorm:
            u = np.zeros_like(bh)
            rest = ~bottom
            u[rest] = -bh[rest] / gaps[rest]
            short = float(np.linalg.norm(u))
            if short < 1.0:
                tau = float(np.sqrt(1.0 - short**2))
                u[0] = -tau if bh[0] > 0.0 else tau  # sign that lowers b^T z
                return radius * (U @ u), max(0.0, -lam_min)

    # Boundary: find mu in (lo, hi] with ||u(mu)|| = 1, u(mu) = -bh / (gaps + mu).
    # lam >= 0 and lam >= -lam_min mean mu >= max(0, lam_min); and
    # ||u(mu)|| <= ||bh|| / mu, so mu = ||bh|| is on the feasible side.
    lo = max(0.0, lam_min)
    hi = max(lo, bnorm)
    mu = hi
    for _ in range(max_iter):
        w = bh / (gaps + mu)
        unorm = float(np.linalg.norm(w))
        if abs(unorm - 1.0) <= rtol:
            break
        if unorm > 1.0:
            lo = mu
        else:
            hi = mu
        if hi - lo <= 1e-15 * hi:
            break
        phi = 1.0 / unorm - 1.0
        dphi = float(np.sum(w**2 / (gaps + mu))) / unorm**3
        mu_new = mu - phi / dphi
        if not (lo < mu_new < hi):
            mu_new = 0.5 * (lo + hi)
        mu = mu_new
    u = -bh / (gaps + mu)
    return radius * (U @ u), max(0.0, mu - lam_min)


def solve_sphere_trs(A: Coords, b: Coords, radius: float) -> tuple[Coords, float]:
    """Solve  min_z b^T z + 0.5 z^T A z  s.t.  ||z|| <= radius  (small, dense)."""
    evals, U = np.linalg.eigh(A)
    return solve_sphere_trs_eig(evals, U, b, radius)
