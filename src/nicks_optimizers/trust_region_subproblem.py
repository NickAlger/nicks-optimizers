"""The trust-region subproblem, solved over a Lanczos Krylov space.

    min_s  <g, s> + 0.5 <H_model s, s>    s.t.  ||s||_M <= radius,

over K_k(P^{-1} H, P^{-1} g), where H_model is H, or |H|_M for the
saddle-free model. P only chooses the space; the geometry comes from M. With
T = <H Q, Q> and G = [(q_i, q_j)_M] = L L^T, the matrix A = L^{-1} T L^{-T}
is H in the M-orthonormal basis Q L^{-T}, so its eigenpairs are the M-Ritz
pairs (Rayleigh-Ritz for the pencil H u = theta M u). In those coordinates z
the ball is round and the small problem is

    min_z  b^T z + 0.5 z^T A_model z    s.t.  ||z|| <= radius,

with b = gamma0 L^{-1} e_1, and A_model = A, or |A| (eigenvalues replaced by
their absolute values) for the saddle-free model. The step is s = Q L^{-T} z.

Each Lanczos step costs one Hessian-vector product, one preconditioner solve,
one call to each inner product (with a block of all earlier basis vectors),
and O(k) vector operations for reorthogonalization. Storage is the two blocks
Q and V = P Q.

Caveat when M differs from P: if the trust region is active, the exact step
solves (H + lam M) s = -g, which K_k(P^{-1} H, P^{-1} g) approximates slowly
unless M is close to a multiple of P. Steps always beat the Cauchy point
along -P^{-1} g, so global convergence is unaffected, but boundary solves stop
when the model stops improving (stall_tol) rather than on the residual.
"""

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np

from .dense import grow_gram, pencil_ritz, solve_sphere_trs_eig
from .krylov import lanczos_advance, lanczos_start, lanczos_tridiagonal
from .metrics import covector_norm, extend_dual_gram, extend_metric_gram
from .spaces import (
    Coords,
    CovectorBlockT,
    CovectorT,
    PointT,
    VectorBlockT,
    VectorSpace,
    VectorT,
)


StopReason = Literal["tol", "stall", "breakdown", "max"]


@dataclass(frozen=True, kw_only=True)
class SubproblemStats:
    """Diagnostics of one trust-region subproblem solve: scalars and tuples
    only, so records stay small however long the run.

      lanczos_steps     number of Lanczos steps k
      stop_reason       "tol" (residual met resid_tol), "stall" (boundary
                        active and the model stopped changing), "breakdown"
                        (invariant subspace: the subspace solution is exact) or
                        "max" (max_lanczos or the dimension)
      step_norm, lam    ||s||_M and the multiplier (> 0 when the boundary is active)
      model_value       <g,s> + 0.5 <H_model s, s>, the model that was minimized
      true_model_value  <g,s> + 0.5 <H s, s> (differs only with saddle_free)
      resid, resid_tol  final ||R||_{M^{-1}} and the tolerance it was tested against
      n_negative, theta_min  number of negative M-Ritz values, smallest one
      *_calls           operator and inner-product calls made by this solve
      trace_*           per Lanczos step: model value, residual, multiplier,
                        number of negative M-Ritz values
    """
    lanczos_steps: int
    stop_reason: StopReason
    step_norm: float
    lam: float
    model_value: float
    true_model_value: float
    resid: float
    resid_tol: float
    n_negative: int
    theta_min: float
    hessvec_calls: int
    precond_calls: int
    inner_calls: int
    dual_inner_calls: int
    trace_model: tuple[float, ...]
    trace_resid: tuple[float, ...]
    trace_lam: tuple[float, ...]
    trace_negative: tuple[int, ...]


def solve_projected_trs(T: Coords, G: Coords, gamma0: float, radius: float,
                        saddle_free: bool = False
                        ) -> tuple[Coords, Coords, float, float, float, Coords]:
    """Solve the small trust-region problem in M-orthonormal coordinates.

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
      theta        M-Ritz values
    """
    k = T.shape[0]
    L, theta, Z = pencil_ritz(T, G)
    curvature = np.abs(theta) if saddle_free else theta
    order = np.argsort(curvature)
    e1 = np.zeros(k)
    e1[0] = 1.0
    b = gamma0 * np.linalg.solve(L, e1)
    z, lam = solve_sphere_trs_eig(curvature[order], Z[:, order], b, radius)
    zh = Z.T @ z                                     # M-Ritz coordinates
    model_value = float(b @ z + 0.5 * np.sum(curvature * zh**2))
    true_model_value = float(b @ z + 0.5 * np.sum(theta * zh**2))
    y = np.linalg.solve(L.T, z)
    if saddle_free:
        signs = np.where(theta < 0.0, -1.0, 1.0)
        y_flip = np.linalg.solve(L.T, Z @ (signs * zh))
    else:
        y_flip = y
    return y, y_flip, lam, model_value, true_model_value, theta


def kkt_residual_dual_norm(y: Coords, y_flip_last: float, lam: float, G: Coords,
                           W: Coords, gamma_next: float) -> float:
    """M^{-1}-norm of the residual R = H s_flip + lam M s + g, from Gram
    matrices only.

    The Lanczos relation H Q = V T + gamma_next v_next e_k^T and the projected
    equations give
        R = lam (M Q y - V G y) + gamma_next y_flip[-1] v_next.
    For the plain model s_flip = s, and this is the exact KKT residual; for
    the saddle-free model it is the surrogate for |H|_M s + lam M s + g. The
    covector lam M Q y is never formed: with W = [(v_i, v_j)_{M^{-1}}] over
    V_{k+1}, <V, Q> = I and <v_next, Q> = 0, the duality identities
    (M u, M w)_{M^{-1}} = (u, w)_M and (M u, psi)_{M^{-1}} = <psi, u> give
        ||R||^2 = beta^T W beta - lam^2 y^T G y,  beta = (-lam G y, gamma_next y_flip[-1]).
    This is why the two inner products must be exact duals. For lam = 0 only
    W enters, and any dual inner product gives a valid (if different) norm.

    Accuracy: for lam = 0 (interior step) there is no subtraction and the
    value is accurate to rounding. For lam > 0 the subtraction gives a floor
    of roughly sqrt(eps * ||G W G||) * lam * ||y|| (about 1e-7 relative to
    ||g|| in the tests, with cond(M) = 1e3). Tight tolerances arise near
    convergence, where the trust region is inactive, so the floor rarely
    binds; if it does, store M^{-1} V and form M^{-1} R explicitly.
    """
    beta = np.append(-lam * (G @ y), gamma_next * y_flip_last)
    squared = float(beta @ W @ beta - lam**2 * (y @ G @ y))
    return float(np.sqrt(max(squared, 0.0)))


def boundary_stalled(lam: float, model_value: float, prev_model_value: Optional[float],
                     stall_tol: float) -> bool:
    """True when the trust region is active and the last Lanczos step changed
    the model value by less than stall_tol (relative).

    With M != P the residual on the boundary contains lam M Q y - lam V G y,
    which the Krylov space of P^{-1} H only reduces slowly, so boundary solves
    stop on stagnation of the model value instead. The change is taken in
    absolute value because the saddle-free model itself changes with k.
    """
    if lam <= 0.0 or prev_model_value is None:
        return False
    return abs(prev_model_value - model_value) <= stall_tol * abs(model_value)


def solve_tr_subproblem(space: VectorSpace[PointT, VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                        g: CovectorT, hessvec: Callable[[VectorT], CovectorT],
                        precond_solve: Callable[[CovectorT], VectorT],
                        ball_inner: Callable[[VectorBlockT, VectorT], Coords],
                        ball_dual_inner: Callable[[CovectorBlockT, CovectorT], Coords],
                        radius: float, resid_tol: float, max_iter: int,
                        g_dual_norm: Optional[float] = None, stall_tol: float = 1e-3,
                        saddle_free: bool = False, breakdown_tol: float = 1e-12
                        ) -> tuple[VectorT, SubproblemStats]:
    """Approximately solve  min <g, s> + 0.5 <H_model s, s>  s.t.  ||s||_M <= radius
    over the growing Krylov spaces K_k(P^{-1} H, P^{-1} g), where H_model is
    H, or |H|_M (via M-Ritz pairs) when saddle_free.

    The operators and inner products are frozen at the current point; the
    inner products take a block as their first argument (see the module
    docstring) and must be exact duals. g_dual_norm = ||g||_{M^{-1}} may be
    passed in to save one call.

    Stops at the first of:
      * ||R||_{M^{-1}} <= resid_tol, with R the (surrogate) KKT residual
        (see kkt_residual_dual_norm),
      * the trust region is active and the model stopped changing
        (stall_tol; see boundary_stalled),
      * Lanczos breakdown (an invariant subspace: the subspace solution is
        exact),
      * max_iter Lanczos steps (capped at the dimension when the space
        provides one).
    Every k >= 1 gives at least the model decrease of the Cauchy point along
    -P^{-1} g, since that point lies in K_1.

    Returns (s, stats): the step and its SubproblemStats.
    """
    dual_calls = 0
    if g_dual_norm is None:
        g_dual_norm = covector_norm(space, ball_dual_inner, g)
        dual_calls += 1
    state = lanczos_start(space, g, precond_solve)
    if space.dimension is not None:
        max_iter = min(max_iter, space.dimension(state.q))
    G = ball_inner(state.Q, state.q).reshape(1, 1)              # (q0, q0)_M
    W = np.array([[(g_dual_norm / state.gamma0) ** 2]])         # (v0, v0)_{M^{-1}}
    prev_model_value: Optional[float] = None
    trace_model: list[float] = []
    trace_resid: list[float] = []
    trace_lam: list[float] = []
    trace_negative: list[int] = []

    for k in range(1, max_iter + 1):
        state = lanczos_advance(space, state, hessvec, precond_solve, breakdown_tol)
        # W covers the basis V plus the pending covector v_next.
        if state.breakdown:
            W_next = grow_gram(W, np.zeros(k + 1))
        else:
            W_next = extend_dual_gram(space, W, state.V, state.v_next, ball_dual_inner)
            dual_calls += 1

        y, y_flip, lam, model_value, true_model_value, theta = solve_projected_trs(
            lanczos_tridiagonal(state), G, state.gamma0, radius, saddle_free)
        resid = kkt_residual_dual_norm(y, float(y_flip[-1]), lam, G, W_next, state.gamma_next)
        n_negative = int(np.sum(theta < 0.0))
        trace_model.append(model_value)
        trace_resid.append(resid)
        trace_lam.append(lam)
        trace_negative.append(n_negative)
        stalled = boundary_stalled(lam, model_value, prev_model_value, stall_tol)
        if resid <= resid_tol or stalled or state.breakdown or k == max_iter:
            reason: StopReason = ("breakdown" if state.breakdown else "tol" if resid <= resid_tol
                                  else "stall" if stalled else "max")
            stats = SubproblemStats(
                lanczos_steps=k, stop_reason=reason,
                step_norm=float(np.sqrt(max(float(y @ G @ y), 0.0))), lam=lam,
                model_value=model_value, true_model_value=true_model_value,
                resid=resid, resid_tol=resid_tol, n_negative=n_negative,
                theta_min=float(theta[0]),
                hessvec_calls=k, precond_calls=k + 1, inner_calls=k, dual_inner_calls=dual_calls,
                trace_model=tuple(trace_model), trace_resid=tuple(trace_resid),
                trace_lam=tuple(trace_lam), trace_negative=tuple(trace_negative))
            return space.combine_vectors(state.Q, y), stats

        prev_model_value = model_value
        # The pending vector joins the basis at the next lanczos_advance.
        G = extend_metric_gram(space, G, state.Q, state.q_next, ball_inner)
        W = W_next
    raise AssertionError("unreachable: the loop returns at k == max_iter")
