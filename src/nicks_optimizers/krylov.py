"""Krylov processes over any VectorSpace.

P-preconditioned Lanczos with full reorthogonalization, as an immutable
process: lanczos_start makes a LanczosState from a covector g, and
lanczos_advance takes one more step (one product with H, one solve with P).
After k steps the state holds a P-orthonormal basis Q of the Krylov space
K_k(P^{-1} H, P^{-1} g), the covectors V = P Q, and the tridiagonal
T = <H Q, Q> (lanczos_tridiagonal), with the Lanczos relation

    H Q = V T + gamma_next v_next e_k^T,

where (gamma_next, v_next, q_next) is the pending next basis direction. The
process knows nothing about inner products or trust regions, so it serves
any consumer: trust-region and line-search Newton steps, Ritz values and
low-rank approximations of the pencil (H, P), and so on. CG-type solvers
belong here as well.
"""

from dataclasses import dataclass
from typing import Callable, Generic, Optional

import numpy as np

from .dense import tridiagonal
from .spaces import (
    Coords,
    CovectorBlockT,
    CovectorT,
    PointT,
    VectorBlockT,
    VectorSpace,
    VectorT,
)


def lanczos_initial_pair(space: VectorSpace[PointT, VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                  g: CovectorT, precond_solve: Callable[[CovectorT], VectorT]
                  ) -> tuple[CovectorT, VectorT, float]:
    """The first Lanczos pair, from g.

    Returns (v0, q0, gamma0) with g = gamma0 v0, q0 = P^{-1} v0 and
    <v0, q0> = <P q0, q0> = 1. gamma0 = ||g||_{P^{-1}}.
    """
    y = precond_solve(g)
    gPg = space.pair(g, y)
    if not gPg > 0.0:
        raise ValueError("<g, P^{-1} g> must be positive; is P positive definite and g nonzero?")
    gamma0 = float(np.sqrt(gPg))
    return space.scale_covector(1.0 / gamma0, g), space.scale_vector(1.0 / gamma0, y), gamma0


def reorthogonalize(space: VectorSpace[PointT, VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                    t: CovectorT, Q: VectorBlockT, V: CovectorBlockT, passes: int = 2) -> CovectorT:
    """Make P^{-1} t P-orthogonal to the vectors in Q.

    The pairs (q_i, v_i = P q_i) satisfy <v_i, q_j> = delta_ij, and
    <P q_i, P^{-1} t> = <t, q_i>, so it suffices to drive the pairings
    <t, q_i> to zero, which needs no preconditioner solves. Two passes of
    classical Gram-Schmidt ("twice is enough").
    """
    for _ in range(passes):
        t = space.axpy_covector(-1.0, space.combine_covectors(V, space.pair_block(Q, t)), t)
    return t


def lanczos_step(space: VectorSpace[PointT, VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                 hessvec: Callable[[VectorT], CovectorT],
                 precond_solve: Callable[[CovectorT], VectorT],
                 Q: VectorBlockT, V: CovectorBlockT, q: VectorT, v: CovectorT,
                 v_prev: Optional[CovectorT], gamma_prev: float,
                 reorth_passes: int = 2) -> tuple[float, float, CovectorT, VectorT]:
    """One step of P-preconditioned Lanczos with full reorthogonalization.

    q, v are the newest pair (the last members of Q, V) and v_prev the one
    before it (None at the first step). Forms the covector
        t = H q - delta v - gamma_prev v_prev,   delta = <H q, q>,
    reorthogonalizes it, and returns (delta, gamma_next, v_next, q_next) with
    gamma_next = ||t||_{P^{-1}}, v_next = t / gamma_next and
    q_next = P^{-1} t / gamma_next. On exact breakdown the vectors are zero.
    Costs one Hessian-vector product and one preconditioner solve.
    """
    h = hessvec(q)
    delta = space.pair(h, q)
    t = space.axpy_covector(-delta, v, h)
    if v_prev is not None:
        t = space.axpy_covector(-gamma_prev, v_prev, t)
    t = reorthogonalize(space, t, Q, V, reorth_passes)
    y = precond_solve(t)
    gamma_next = float(np.sqrt(max(space.pair(t, y), 0.0)))
    if gamma_next == 0.0:
        return delta, 0.0, space.scale_covector(0.0, t), space.scale_vector(0.0, y)
    return (delta, gamma_next, space.scale_covector(1.0 / gamma_next, t),
            space.scale_vector(1.0 / gamma_next, y))


@dataclass(frozen=True, kw_only=True)
class LanczosState(Generic[VectorT, CovectorT, VectorBlockT, CovectorBlockT]):
    """A P-preconditioned Lanczos process after `steps` steps.

      steps          number of steps taken (= products with H)
      gamma0         ||g||_{P^{-1}}, with g = gamma0 v_0
      Q, V           the basis q_0, ..., q_{m-1} (P-orthonormal) and v_i = P q_i,
                     with m = max(steps, 1)
      q, v           the newest basis members q_{m-1}, v_{m-1}
      deltas         diagonal of T (length steps)
      gammas         off-diagonal of T within the basis (length steps - 1)
      gamma_next, v_next, q_next
                     the pending next direction: H Q = V T + gamma_next
                     v_next e_k^T. Zero before the first step and on breakdown.
      breakdown      an invariant subspace was found: the Krylov space contains
                     the exact solution and cannot grow further
      v_prev, hscale internal (the previous v, and the scale of T for the
                     breakdown test)

    Frozen; lanczos_advance returns a new state and never modifies vectors.
    """
    steps: int
    gamma0: float
    Q: VectorBlockT
    V: CovectorBlockT
    q: VectorT
    v: CovectorT
    deltas: tuple[float, ...]
    gammas: tuple[float, ...]
    gamma_next: float
    v_next: CovectorT
    q_next: VectorT
    breakdown: bool
    v_prev: Optional[CovectorT]
    hscale: float


def lanczos_start(space: VectorSpace[PointT, VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                  g: CovectorT, precond_solve: Callable[[CovectorT], VectorT]
                  ) -> LanczosState[VectorT, CovectorT, VectorBlockT, CovectorBlockT]:
    """The process before its first step: the basis is q_0 = P^{-1} g / gamma0.
    Costs one preconditioner solve."""
    v0, q0, gamma0 = lanczos_initial_pair(space, g, precond_solve)
    return LanczosState(
        steps=0, gamma0=gamma0, Q=space.vector_block(q0), V=space.covector_block(v0),
        q=q0, v=v0, deltas=(), gammas=(), gamma_next=0.0,
        v_next=space.scale_covector(0.0, v0), q_next=space.scale_vector(0.0, q0),
        breakdown=False, v_prev=None, hscale=0.0)


def lanczos_advance(space: VectorSpace[PointT, VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                    state: LanczosState[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                    hessvec: Callable[[VectorT], CovectorT],
                    precond_solve: Callable[[CovectorT], VectorT],
                    breakdown_tol: float = 1e-12, reorth_passes: int = 2
                    ) -> LanczosState[VectorT, CovectorT, VectorBlockT, CovectorBlockT]:
    """One more Lanczos step.

    After the first step, the pending direction from the previous step joins
    the basis; then H is applied to the newest basis vector, the result is
    reorthogonalized, and the new pending direction is formed. Breakdown is
    declared when gamma_next <= breakdown_tol times the largest entry of T
    seen so far; a broken-down process cannot be advanced. Costs one product
    with H and one solve with P.
    """
    if state.breakdown:
        raise ValueError("the Lanczos process has broken down (invariant subspace); it cannot grow")
    if state.steps > 0:
        Q = space.append_vector(state.Q, state.q_next)
        V = space.append_covector(state.V, state.v_next)
        gammas = state.gammas + (state.gamma_next,)
        v_prev: Optional[CovectorT] = state.v
        q, v = state.q_next, state.v_next
    else:
        Q, V, gammas, v_prev, q, v = state.Q, state.V, state.gammas, state.v_prev, state.q, state.v
    gamma_prev = gammas[-1] if gammas else 0.0
    delta, gamma_next, v_next, q_next = lanczos_step(
        space, hessvec, precond_solve, Q, V, q, v, v_prev, gamma_prev, reorth_passes)
    hscale = max(state.hscale, abs(delta), gamma_prev)
    breakdown = gamma_next <= breakdown_tol * hscale
    return LanczosState(
        steps=state.steps + 1, gamma0=state.gamma0, Q=Q, V=V, q=q, v=v,
        deltas=state.deltas + (delta,), gammas=gammas,
        gamma_next=0.0 if breakdown else gamma_next, v_next=v_next, q_next=q_next,
        breakdown=breakdown, v_prev=v_prev, hscale=hscale)


def lanczos_tridiagonal(state: LanczosState[VectorT, CovectorT, VectorBlockT, CovectorBlockT]) -> Coords:
    """T = <H Q, Q> for the current basis (k x k after k steps)."""
    return tridiagonal(np.array(state.deltas, dtype=float), np.array(state.gammas, dtype=float))
