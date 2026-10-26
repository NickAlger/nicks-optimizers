"""Small dense linear algebra in coordinates.

Everything here acts on small arrays (projected matrices, coefficient
vectors), never on the optimizer's vectors, so it is independent of any
VectorSpace: Rayleigh-Ritz for a projected pencil, an exact solver for the
trust-region problem on a ball (including the hard case), and the projected
trust-region problem of a Krylov method.

The functions are written for a backend's array namespace (see backend;
numpy by default) without Python control flow on values, so they run inside
compiled code. Under a backend with static shapes, the projected matrices
are padded to the capacity of the Krylov basis: with k active coordinates
(the leading k x k block), pass k, and the inactive coordinates are decoupled
so that they do not change the solution (see pencil_ritz). With numpy, the
problem is cut down to the active block instead, so nothing changes in what
is computed.
"""

from dataclasses import dataclass
from typing import Any, Optional

from .backend import Backend, backend_state, numpy_backend, run_loop
from .spaces import Coords


def pencil_ritz(T: Coords, G: Optional[Coords], *, backend: Backend = numpy_backend,
                k: Optional[Any] = None) -> tuple[Optional[Coords], Coords, Coords]:
    """Rayleigh-Ritz for the pencil H u = theta M u on the Krylov space.

    With G = L L^T, Q L^{-T} is an M-orthonormal basis of the space and
    A = L^{-1} T L^{-T} is H in that basis. Returns (L, theta, Z): the
    M-Ritz values theta (ascending) and the M-Ritz vectors Z in z-coordinates.
    G = None stands for the identity (an M-orthonormal basis, as with P = M):
    nothing is factored, A = T, and L is None.

    Padding (k given, static shapes): only the leading k x k blocks of T and
    G are used. The inactive part becomes G = I and A = pad I, with pad above
    every active |theta| (1 + twice the largest absolute row sum of A), so
    the inactive pairs come last in theta, also when sorted by |theta|, and
    their vectors live on the inactive coordinates.
    """
    xp = backend.xp
    if k is None or not backend.static_shapes:
        L = None if G is None else xp.linalg.cholesky(G)
        A = T if L is None else xp.linalg.solve(L, xp.linalg.solve(L, T).T)
        theta, Z = xp.linalg.eigh(0.5 * (A + A.T))
        return L, theta, Z
    m = T.shape[0]
    index = xp.arange(m)
    active = index < k
    both = active[:, None] & active[None, :]
    diagonal = index[:, None] == index[None, :]
    A = xp.where(both, T, 0.0)
    L = None
    if G is not None:
        L = xp.linalg.cholesky(xp.where(both, G, xp.where(diagonal, 1.0, 0.0)))
        A = xp.linalg.solve(L, xp.linalg.solve(L, A).T)
    A = 0.5 * (A + A.T)
    pad = 1.0 + 2.0 * xp.max(xp.sum(xp.abs(A), axis=1))
    A = A + xp.where(diagonal & ~active[None, :], pad, 0.0)
    theta, Z = xp.linalg.eigh(A)
    return L, theta, Z


@backend_state
@dataclass(frozen=True, kw_only=True)
class _Secular:
    """The carry of the safeguarded Newton iteration on the secular equation:
    the shifted multiplier mu, its bracket, and whether it has stopped."""
    mu: Any
    lo: Any
    hi: Any
    stop: Any


def solve_sphere_trs_eig(evals: Coords, U: Coords, b: Coords, radius: Any,
                         rtol: float = 1e-12, max_iter: int = 60, hard_tol: float = 1e-10, *,
                         backend: Backend = numpy_backend, k: Optional[Any] = None
                         ) -> tuple[Coords, Any]:
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

    The interior, hard-case and boundary solutions are all formed and the
    right one selected, without Python branches on values. max_iter bounds
    the Newton iteration (3 to 5 iterations are typical, 10 the most seen on
    the tests' problems; bisection alone would need about 50). For padded
    problems, k is the number of active eigenvalues, the first k (padded
    ones are larger than the rest, and b and U vanish on them).
    """
    xp = backend.xp
    m = evals.shape[0]
    active = xp.arange(m) < (m if k is None else k)
    lam_min = evals[0]
    gaps = evals - lam_min                  # >= 0, and gaps[0] == 0
    bh = (U.T @ b) / radius                 # unit-ball problem, z = radius * u
    bnorm = xp.sqrt(bh @ bh)

    # Interior: A positive definite and the Newton step lies inside the ball.
    u_interior = -bh / xp.where(evals != 0.0, evals, 1.0)
    interior = (lam_min > 0.0) & (xp.sqrt(u_interior @ u_interior) <= 1.0)

    # Hard case: b (numerically) orthogonal to the bottom eigenspace, and the
    # solution at lam = -lam_min is too short to reach the boundary.
    scale_ = xp.maximum(1.0, xp.max(xp.where(active, xp.abs(evals), 0.0)))
    bottom = gaps <= hard_tol * scale_
    rest = ~bottom
    u_hard = xp.where(rest, -bh / xp.where(rest, gaps, 1.0), 0.0)
    short = xp.sqrt(u_hard @ u_hard)
    tau = xp.sqrt(xp.maximum(1.0 - short ** 2, 0.0))
    u_hard = backend.index_update(u_hard, 0, xp.where(bh[0] > 0.0, -tau, tau))  # sign lowers b^T z
    bottom_b = xp.sqrt(xp.sum(xp.where(bottom, bh ** 2, 0.0)))
    hard = ~interior & (lam_min <= 0.0) & (bottom_b <= hard_tol * bnorm) & (short < 1.0)

    # Boundary: find mu in (lo, hi] with ||u(mu)|| = 1, u(mu) = -bh / (gaps + mu).
    # lam >= 0 and lam >= -lam_min mean mu >= max(0, lam_min); and
    # ||u(mu)|| <= ||bh|| / mu, so mu = ||bh|| is on the feasible side.
    lo = xp.maximum(0.0, lam_min)
    hi = xp.maximum(lo, bnorm)

    def newton(c: _Secular) -> _Secular:
        w = bh / (gaps + c.mu)
        unorm = xp.sqrt(w @ w)
        converged = xp.abs(unorm - 1.0) <= rtol
        lo, hi = xp.where(unorm > 1.0, c.mu, c.lo), xp.where(unorm > 1.0, c.hi, c.mu)
        narrow = hi - lo <= 1e-15 * hi
        phi = 1.0 / unorm - 1.0
        dphi = xp.sum(w ** 2 / (gaps + c.mu)) / unorm ** 3
        mu_new = c.mu - phi / dphi
        mu_new = xp.where((lo < mu_new) & (mu_new < hi), mu_new, 0.5 * (lo + hi))
        stop = converged | narrow
        return _Secular(mu=xp.where(stop, c.mu, mu_new), lo=lo, hi=hi, stop=stop)

    final = run_loop(backend, lambda c: c.stop, newton,
                     _Secular(mu=hi, lo=lo, hi=hi, stop=interior | hard), max_iter)
    u_boundary = -bh / (gaps + final.mu)

    u = xp.where(interior, u_interior, xp.where(hard, u_hard, u_boundary))
    lam = xp.where(interior, 0.0, xp.where(hard, xp.maximum(0.0, -lam_min),
                                           xp.maximum(0.0, final.mu - lam_min)))
    return radius * (U @ u), lam


def solve_sphere_trs(A: Coords, b: Coords, radius: Any, *, backend: Backend = numpy_backend
                     ) -> tuple[Coords, Any]:
    """Solve  min_z b^T z + 0.5 z^T A z  s.t.  ||z|| <= radius  (small, dense)."""
    evals, U = backend.xp.linalg.eigh(A)
    return solve_sphere_trs_eig(evals, U, b, radius, backend=backend)


def solve_projected_trs(T: Coords, G: Optional[Coords], gamma0: Any, radius: Any,
                        saddle_free: bool = False,
                        *, backend: Backend = numpy_backend, k: Optional[Any] = None
                        ) -> tuple[Coords, Coords, Any, Any, Any, Coords]:
    """Solve the small trust-region problem of a Krylov method in
    M-orthonormal coordinates: min b^T z + 0.5 z^T A_model z, ||z|| <= radius,
    with A = L^{-1} T L^{-T}, G = L L^T, b = gamma0 L^{-1} e_1.

    The model Hessian has the M-Ritz vectors as eigenvectors and the M-Ritz
    values theta as eigenvalues, or |theta| for the saddle-free model.

    Returns (y, y_flip, lam, model_value, true_model_value, theta):
      y            coefficients of the step, s = Q y
      y_flip       coefficients of the flipped step s_flip = Q y_flip, whose
                   components along negative M-Ritz vectors are negated. If
                   the Ritz vectors were exact eigenvectors, |H|_M s would
                   equal H s_flip; y_flip = y unless saddle_free.
      lam          trust-region multiplier, in both coordinate systems
      model_value  g^T s + 0.5 s^T H_model s, the model that was minimized
      true_model_value  g^T s + 0.5 s^T H s with the true H on the subspace
                   (equal to model_value unless saddle_free)
      theta        M-Ritz values, ascending

    G = None stands for the identity (an M-orthonormal basis): no
    factorization, z = y. T and G may be padded (see the module docstring): with k active
    coordinates, y and y_flip vanish beyond k, and theta holds the k M-Ritz
    values first, then values that are neither negative nor smallest (pad
    values, or +inf when the problem was cut down to the active block).
    """
    xp = backend.xp
    m = T.shape[0]
    if k is not None and not backend.static_shapes and int(k) < m:
        kk = int(k)
        y, y_flip, lam, model_value, true_model_value, theta = solve_projected_trs(
            T[:kk, :kk], None if G is None else G[:kk, :kk], gamma0, radius, saddle_free,
            backend=backend)
        return (_pad(backend, y, m), _pad(backend, y_flip, m), lam, model_value,
                true_model_value, _pad(backend, theta, m, xp.inf))
    L, theta, Z = pencil_ritz(T, G, backend=backend, k=k)
    curvature = xp.abs(theta) if saddle_free else theta
    order = xp.argsort(curvature)
    e1 = xp.where(xp.arange(m) == 0, 1.0, 0.0)
    b = gamma0 * (e1 if L is None else xp.linalg.solve(L, e1))
    # The Ritz vectors in curvature order (as rows of Z^T, which keeps numpy's
    # memory layout, and with it the rounding, of Z[:, order]).
    z, lam = solve_sphere_trs_eig(xp.take(curvature, order), xp.take(Z.T, order, axis=0).T, b,
                                  radius, backend=backend, k=k)
    zh = Z.T @ z                                     # M-Ritz coordinates
    model_value = b @ z + 0.5 * xp.sum(curvature * zh ** 2)
    true_model_value = b @ z + 0.5 * xp.sum(theta * zh ** 2)
    y = z if L is None else xp.linalg.solve(L.T, z)
    if saddle_free:
        signs = xp.where(theta < 0.0, -1.0, 1.0)
        flipped = Z @ (signs * zh)
        y_flip = flipped if L is None else xp.linalg.solve(L.T, flipped)
    else:
        y_flip = y
    return y, y_flip, lam, model_value, true_model_value, theta


def _pad(backend: Backend, v: Coords, m: int, fill: float = 0.0) -> Coords:
    """v extended to length m with fill (dynamic shapes only)."""
    padded: Coords = backend.index_update(backend.xp.full(m, fill), slice(0, v.shape[0]), v)
    return padded
