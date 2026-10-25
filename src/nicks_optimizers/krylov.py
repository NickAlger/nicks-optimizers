"""Krylov processes over any VectorSpace.

P-preconditioned Lanczos with full reorthogonalization, as an immutable
process: lanczos_start makes a LanczosState from a covector g, and
lanczos_advance takes one more step (one product with H, one solve with P).
After k steps the state holds a P-orthonormal basis Q of the Krylov space
K_k(P^{-1} H, P^{-1} g), the covectors V = P Q, and the tridiagonal
T = <H Q, Q> (lanczos_tridiagonal), with the Lanczos relation

    H Q = V T + gamma_next v_next e_k^T,

where (gamma_next, v_next, q_next) is the pending next basis direction. The
process knows nothing about the metric or trust regions, so it serves
any consumer: trust-region and line-search Newton steps, Ritz values and
low-rank approximations of the pencil (H, P), and so on.

Preconditioned conjugate gradients for H s = -g is the same Krylov space
built with short recurrences: cg_start makes a CGState, cg_curvature
applies H to the search direction, and cg_advance takes the step. It keeps
five vectors instead of a basis and does two reductions per step, but without
reorthogonalization the directions lose conjugacy in floating point (which
delays convergence, without breaking it). The P-norms of the iterates, which
Steihaug's trust-region method needs, follow from recurrences without
products with P (Conn, Gould & Toint, Trust-Region Methods, 2000, Sec. 7.5.1).

Redundant representations. Sometimes the stored components of a covector
have more freedom than the covector itself, e.g. tangent vectors of a
low-rank manifold stored as gauged factor variations: a component that pairs
to zero with every vector is invisible, the zero functional. Linear
combinations of canonical covectors stay canonical only up to rounding, and
the three-term recurrence amplifies the invisible rounding errors (like a
Lanczos polynomial evaluated at an eigenvalue outside the spectrum), while
no pairing, and hence no reorthogonalization, can see them. Within tens of
steps they swamp the visible part, the basis leaves the space and T stops
being <H Q, Q>. Pass project_covector, the projection onto canonical
representatives, to remove them at every step.
"""

from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional

import numpy as np

from .backend import Backend, backend_state, numpy_backend
from .dense import tridiagonal
from .spaces import (
    Coords,
    CovectorBlockT,
    CovectorT,
    VectorBlockT,
    VectorSpace,
    VectorT,
)


def lanczos_initial_pair(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                  g: CovectorT, precond_solve: Callable[[CovectorT], VectorT],
                  precond_g: Optional[VectorT] = None) -> tuple[CovectorT, VectorT, float]:
    """The first Lanczos pair, from g (and P^{-1} g, if already computed).

    Returns (v0, q0, gamma0) with g = gamma0 v0, q0 = P^{-1} v0 and
    <v0, q0> = <P q0, q0> = 1. gamma0 = ||g||_{P^{-1}}.
    """
    y = precond_solve(g) if precond_g is None else precond_g
    gPg = space.pair(g, y)
    if not gPg > 0.0:
        raise ValueError("<g, P^{-1} g> must be positive; is P positive definite and g nonzero?")
    gamma0 = float(np.sqrt(gPg))
    return space.scale_covector(1.0 / gamma0, g), space.scale_vector(1.0 / gamma0, y), gamma0


def reorthogonalize(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                    t: CovectorT, Q: VectorBlockT, V: CovectorBlockT, k: Any, passes: int = 2
                    ) -> CovectorT:
    """Make P^{-1} t P-orthogonal to the first k vectors in Q.

    The pairs (q_i, v_i = P q_i) satisfy <v_i, q_j> = delta_ij, and
    <P q_i, P^{-1} t> = <t, q_i>, so it suffices to drive the pairings
    <t, q_i> to zero, which needs no preconditioner solves. Two passes of
    classical Gram-Schmidt ("twice is enough").
    """
    for _ in range(passes):
        t = space.axpy_covector(-1.0, space.combine_covectors(V, space.pair_block(Q, t, k), k), t)
    return t


def lanczos_step(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                 hessvec: Callable[[VectorT], CovectorT],
                 precond_solve: Callable[[CovectorT], VectorT],
                 Q: VectorBlockT, V: CovectorBlockT, k: Any, q: VectorT, v: CovectorT,
                 v_prev: Optional[CovectorT], gamma_prev: float,
                 reorth_passes: int = 2,
                 project_covector: Optional[Callable[[CovectorT], CovectorT]] = None
                 ) -> tuple[float, float, CovectorT, VectorT]:
    """One step of P-preconditioned Lanczos with full reorthogonalization.

    Q, V hold k members; q, v are the newest pair (the last members) and
    v_prev the one before it (None at the first step). Forms the covector
        t = H q - delta v - gamma_prev v_prev,   delta = <H q, q>,
    reorthogonalizes it, maps it to its canonical representative with
    project_covector if given (see the module docstring), and returns
    (delta, gamma_next, v_next, q_next) with
    gamma_next = ||t||_{P^{-1}}, v_next = t / gamma_next and
    q_next = P^{-1} t / gamma_next. On exact breakdown the vectors are zero.
    Costs one Hessian-vector product and one preconditioner solve.
    """
    h = hessvec(q)
    delta = space.pair(h, q)
    t = space.axpy_covector(-delta, v, h)
    if v_prev is not None:
        t = space.axpy_covector(-gamma_prev, v_prev, t)
    t = reorthogonalize(space, t, Q, V, k, reorth_passes)
    if project_covector is not None:
        t = project_covector(t)
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


def lanczos_start(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                  g: CovectorT, precond_solve: Callable[[CovectorT], VectorT],
                  precond_g: Optional[VectorT] = None, *, capacity: int
                  ) -> LanczosState[VectorT, CovectorT, VectorBlockT, CovectorBlockT]:
    """The process before its first step: the basis is q_0 = P^{-1} g / gamma0.
    Costs one preconditioner solve, unless precond_g = P^{-1} g is given. The
    blocks Q and V have room for capacity members: the process can take
    capacity steps."""
    v0, q0, gamma0 = lanczos_initial_pair(space, g, precond_solve, precond_g)
    return LanczosState(
        steps=0, gamma0=gamma0, Q=space.vector_block(q0, capacity),
        V=space.covector_block(v0, capacity),
        q=q0, v=v0, deltas=(), gammas=(), gamma_next=0.0,
        v_next=space.scale_covector(0.0, v0), q_next=space.scale_vector(0.0, q0),
        breakdown=False, v_prev=None, hscale=0.0)


def lanczos_advance(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                    state: LanczosState[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                    hessvec: Callable[[VectorT], CovectorT],
                    precond_solve: Callable[[CovectorT], VectorT],
                    breakdown_tol: float = 1e-12, reorth_passes: int = 2,
                    project_covector: Optional[Callable[[CovectorT], CovectorT]] = None
                    ) -> LanczosState[VectorT, CovectorT, VectorBlockT, CovectorBlockT]:
    """One more Lanczos step.

    After the first step, the pending direction from the previous step joins
    the basis; then H is applied to the newest basis vector, the result is
    reorthogonalized, and the new pending direction is formed. Breakdown is
    declared when gamma_next <= breakdown_tol times the largest entry of T
    seen so far; a broken-down process cannot be advanced. Costs one product
    with H and one solve with P. project_covector, for redundant
    representations, is applied to each new covector (see the module
    docstring).
    """
    if state.breakdown:
        raise ValueError("the Lanczos process has broken down (invariant subspace); it cannot grow")
    if state.steps > 0:
        Q = space.set_vector(state.Q, state.steps, state.q_next)
        V = space.set_covector(state.V, state.steps, state.v_next)
        gammas = state.gammas + (state.gamma_next,)
        v_prev: Optional[CovectorT] = state.v
        q, v = state.q_next, state.v_next
    else:
        Q, V, gammas, v_prev, q, v = state.Q, state.V, state.gammas, state.v_prev, state.q, state.v
    gamma_prev = gammas[-1] if gammas else 0.0
    delta, gamma_next, v_next, q_next = lanczos_step(
        space, hessvec, precond_solve, Q, V, state.steps + 1, q, v, v_prev, gamma_prev,
        reorth_passes, project_covector)
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


# --- conjugate gradients ---------------------------------------------------------


@backend_state
@dataclass(frozen=True, kw_only=True)
class CGState(Generic[VectorT, CovectorT]):
    """Preconditioned conjugate gradients for H s = -g after `steps` steps.

      steps          number of steps taken (= products with H)
      s              the iterate s_k, starting from s_0 = 0
      r              the residual covector r_k = g + H s_k (the model gradient)
      z              P^{-1} r_k
      p              the search direction p_k
      rz             <r_k, z_k> = ||r_k||_{P^{-1}}^2
      s_norm2, sp, p_norm2
                     ||s_k||_P^2, (s_k, p_k)_P and ||p_k||_P^2, by recurrences
      model          <g, s_k> + 0.5 <H s_k, s_k>, by recurrence

    In exact arithmetic ||s_k||_P increases with k (Steihaug), which is what
    makes stopping on the trust-region boundary meaningful. Frozen;
    cg_advance returns a new state. The numbers are 0-d arrays of the
    backend (numpy scalars for numpy), so the state can be carried through a
    compiled loop (see backend).
    """
    steps: Any
    s: VectorT
    r: CovectorT
    z: VectorT
    p: VectorT
    rz: Any
    s_norm2: Any
    sp: Any
    p_norm2: Any
    model: Any


def cg_start(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
             g: CovectorT, precond_solve: Callable[[CovectorT], VectorT],
             precond_g: Optional[VectorT] = None, *, backend: Backend = numpy_backend
             ) -> CGState[VectorT, CovectorT]:
    """CG before its first step: s_0 = 0, r_0 = g, p_0 = -P^{-1} g. Costs one
    preconditioner solve, unless precond_g = P^{-1} g is given. CG needs
    rz = <g, P^{-1} g> > 0 (P positive definite, g nonzero); the state does
    not check it, so callers test state.rz (solve_tr_subproblem_cg raises)."""
    xp = backend.xp
    z = precond_solve(g) if precond_g is None else precond_g
    rz = space.pair(g, z)
    zero = xp.asarray(0.0)
    return CGState(steps=xp.asarray(0), s=space.scale_vector(0.0, z), r=g, z=z,
                   p=space.scale_vector(-1.0, z), rz=rz, s_norm2=zero, sp=zero, p_norm2=rz,
                   model=zero)


def cg_curvature(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                 state: CGState[VectorT, CovectorT], hessvec: Callable[[VectorT], CovectorT]
                 ) -> tuple[CovectorT, float]:
    """(H p_k, <H p_k, p_k>): the curvature along the search direction, which
    decides whether (and how far) to step. One product with H."""
    Hp = hessvec(state.p)
    return Hp, space.pair(Hp, state.p)


def cg_advance(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
               state: CGState[VectorT, CovectorT], Hp: CovectorT, curvature: float,
               precond_solve: Callable[[CovectorT], VectorT],
               project_covector: Optional[Callable[[CovectorT], CovectorT]] = None
               ) -> CGState[VectorT, CovectorT]:
    """The CG step s_{k+1} = s_k + alpha p_k, alpha = rz_k / <H p_k, p_k>, given
    (H p_k, <H p_k, p_k>) from cg_curvature; the curvature must be positive
    (not checked: with a compiled backend nothing can raise on values).
    One preconditioner solve.

    The P-norm recurrences use <r_{k+1}, p_j> = 0 for j <= k (so
    (s_{k+1}, P^{-1} r_{k+1}) = 0):
        ||s_{k+1}||_P^2 = ||s_k||_P^2 + 2 alpha (s_k, p_k)_P + alpha^2 ||p_k||_P^2
        (s_{k+1}, p_{k+1})_P = beta ((s_k, p_k)_P + alpha ||p_k||_P^2)
        ||p_{k+1}||_P^2 = rz_{k+1} + beta^2 ||p_k||_P^2
    and the model decreases by alpha rz_k / 2. project_covector, for
    redundant representations, is applied to the new residual (see the
    module docstring).
    """
    alpha = state.rz / curvature
    s = space.axpy_vector(alpha, state.p, state.s)
    r = space.axpy_covector(alpha, Hp, state.r)
    if project_covector is not None:
        r = project_covector(r)
    z = precond_solve(r)
    rz = space.pair(r, z)
    beta = rz / state.rz
    p = space.axpy_vector(beta, state.p, space.scale_vector(-1.0, z))
    return CGState(
        steps=state.steps + 1, s=s, r=r, z=z, p=p, rz=rz,
        s_norm2=state.s_norm2 + 2.0 * alpha * state.sp + alpha ** 2 * state.p_norm2,
        sp=beta * (state.sp + alpha * state.p_norm2),
        p_norm2=rz + beta ** 2 * state.p_norm2,
        model=state.model - 0.5 * alpha * state.rz)
