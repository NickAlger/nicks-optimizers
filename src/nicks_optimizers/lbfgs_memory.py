"""The memory of lbfgs and the two-loop recursion in coordinates (see
dev/lbfgs_plan.md), as pure functions a backend can compile.

The memory holds up to m pairs (s_i, y_i) in the slots of two blocks of
capacity m: the steps s_i are vectors (a VectorBlock S), the gradient
changes y_i are covectors (a CovectorBlock Y). Free slots hold zero. Next to
them, in coordinates:

  pairings   pairings[j, i] = <y_j, s_i>, by slot, as computed when the
             pairs were stored (the diagonal is 1/rho)
  stamp      when each pair was stored (its age), EMPTY for a free slot
  floor      the threshold each pair's curvature <y_i, s_i> had to exceed
             when it was stored (for memory_recompute)

The inverse Hessian approximation H applied to a covector g, for a base H0
(a map from covectors to vectors, e.g. gamma M^{-1}), is the textbook
two-loop recursion with its 2m sequential pairings replaced by batched ones
(the active pairs in logical order, oldest first, R = diag(1/rho) + U with
U[i, j] = <y_j, s_i> for i older than j):

  two_loop_project:   c = S^T g, a = R^{-1} c, q = g - Y a
  (the caller):       r0 = H0 q
  two_loop_finish:    e = Y^T r0, b = R^{-T} (e + U^T a), H g = r0 + S (a - b)

For any positive rho and any U, with a symmetric positive definite base,
H = W* H0 W + S R^{-T} D R^{-1} S^T (W = I - Y R^{-1} S^T, D = diag(1/rho))
is symmetric positive definite and satisfies the newest secant equation
H y = s exactly; with the pairings of the current vectors it is the textbook
L-BFGS operator. So the stored pairings may be kept when the pairs are
transported (they are the values T(B)(T u, T v) = B(u, v) prescribes), or
recomputed from the transported pairs (memory_recompute).

  memory_empty, memory_from_pairs, memory_clear, memory_store, memory_count
  memory_transport            carry S and Y to a new point
  memory_recompute            the pairings of the current vectors; pairs
                              whose curvature no longer exceeds their floor
                              are dropped
  memory_correct              the transported covectors corrected (least
                              change in the M^{-1}-norm) so that they pair with
                              the transported steps as the stored pairings say
  memory_layout, two_loop_project, two_loop_finish, apply_inverse
  apply_inverse_fixed         H g and M H g for a constant metric, from the
                              blocks Z = M^{-1} Y and W = M S kept next to
                              S and Y (memory_empty(..., metric_blocks=True)):
                              no operator applications

Nothing here branches in Python on computed values, and nothing imports
numpy. The functions are generic in the vector types, so a static checker
verifies that steps stay vectors and gradient changes covectors.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, Sequence

from .backend import Backend, backend_state, run_cond, run_loop
from .spaces import CovectorBlockT, CovectorT, PointT, VectorBlockT, VectorSpace, VectorT
from .transport import VectorTransport, identity_transport

EMPTY = -1                       # the stamp of a free slot


@backend_state
@dataclass(frozen=True, kw_only=True)
class Memory(Generic[VectorBlockT, CovectorBlockT]):
    """The stored pairs and their pairings (see the module docstring); for a
    constant metric the blocks Z = M^{-1} Y (vectors) and W = M S
    (covectors), and for a constant preconditioner ZP = P^{-1} Y, slot for
    slot (None otherwise)."""
    S: VectorBlockT
    Y: CovectorBlockT
    pairings: Any
    stamp: Any
    floor: Any
    Z: Optional[VectorBlockT] = None
    W: Optional[CovectorBlockT] = None
    ZP: Optional[VectorBlockT] = None


@backend_state
@dataclass(frozen=True, kw_only=True)
class Layout:
    """The active pairs in logical order, oldest first (free slots before
    them, decoupled): the permutation from logical order to slots and back,
    which logical coordinates are active, and U and R = diag(1/rho) + U
    (identity rows for free slots)."""
    perm: Any
    inverse: Any
    active: Any
    U: Any
    R: Any


def capacity(memory: Memory[Any, Any]) -> int:
    return int(memory.stamp.shape[0])


def memory_empty(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                 backend: Backend, v: VectorT, w: CovectorT, m: int, *, metric_blocks: bool = False,
                 precond_block: bool = False) -> Memory[VectorBlockT, CovectorBlockT]:
    """A memory with room for m pairs, none stored; v and w are a sample
    vector and covector (for the shapes). With metric_blocks, also the
    blocks Z = M^{-1} Y and W = M S (for a constant metric); with
    precond_block, ZP = P^{-1} Y (for a constant preconditioner)."""
    xp = backend.xp
    zero_v, zero_w = space.scale_vector(0.0, v), space.scale_covector(0.0, w)
    return Memory(S=space.vector_block(zero_v, m), Y=space.covector_block(zero_w, m),
                  pairings=xp.zeros((m, m)), stamp=xp.full(m, EMPTY), floor=xp.zeros(m),
                  Z=space.vector_block(zero_v, m) if metric_blocks else None,
                  W=space.covector_block(zero_w, m) if metric_blocks else None,
                  ZP=space.vector_block(zero_v, m) if precond_block else None)


def memory_from_pairs(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                      backend: Backend, pairs: Sequence[tuple[VectorT, CovectorT]],
                      pairings: Sequence[Sequence[float]], floors: Sequence[float], slots: Sequence[int],
                      v: VectorT, w: CovectorT, m: int,
                      metric_pairs: Optional[Sequence[tuple[VectorT, CovectorT]]] = None,
                      precond_vectors: Optional[Sequence[VectorT]] = None
                      ) -> Memory[VectorBlockT, CovectorBlockT]:
    """A memory with room for m pairs holding the given pairs (oldest first)
    in the given slots, with their stored pairings (pairings[j][i] =
    <y_j, s_i>, in the same order) and floors, and with metric_pairs
    (M^{-1} y_i, M s_i) the metric blocks too, with precond_vectors
    P^{-1} y_i the preconditioner block: for resuming a run. Keeping the
    slots keeps the arithmetic of the block operations unchanged."""
    xp, put = backend.xp, backend.index_update
    k = len(pairs)
    if k > m or len(set(slots)) != k or any(not 0 <= j < m for j in slots):
        raise ValueError(f"{k} pairs in slots {tuple(slots)} do not fit a memory of {m}")
    if metric_pairs is not None and len(metric_pairs) != k:
        raise ValueError(f"{len(metric_pairs)} metric pairs for {k} pairs")
    if precond_vectors is not None and len(precond_vectors) != k:
        raise ValueError(f"{len(precond_vectors)} preconditioner vectors for {k} pairs")
    memory = memory_empty(space, backend, v, w, m, metric_blocks=metric_pairs is not None,
                          precond_block=precond_vectors is not None)
    S, Y, Z, W, ZP = memory.S, memory.Y, memory.Z, memory.W, memory.ZP
    for j, (s, y) in sorted(zip(slots, pairs), key=lambda item: item[0]):
        S, Y = space.set_vector(S, j, s), space.set_covector(Y, j, y)
    if metric_pairs is not None and Z is not None and W is not None:
        for j, (z, Ms) in sorted(zip(slots, metric_pairs), key=lambda item: item[0]):
            Z, W = space.set_vector(Z, j, z), space.set_covector(W, j, Ms)
    if precond_vectors is not None and ZP is not None:
        for j, zp in sorted(zip(slots, precond_vectors), key=lambda item: item[0]):
            ZP = space.set_vector(ZP, j, zp)
    P, stamp, floor = memory.pairings, memory.stamp, memory.floor
    for a, ja in enumerate(slots):
        stamp, floor = put(stamp, ja, a), put(floor, ja, floors[a])
        for b, jb in enumerate(slots):
            P = put(P, (ja, jb), pairings[a][b])
    return Memory(S=S, Y=Y, pairings=P, stamp=stamp, floor=floor, Z=Z, W=W, ZP=ZP)


def memory_clear(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                 backend: Backend, memory: Memory[VectorBlockT, CovectorBlockT]
                 ) -> Memory[VectorBlockT, CovectorBlockT]:
    """The same memory with no pairs (zero blocks of the same capacity)."""
    return memory_empty(space, backend, space.vector_member(memory.S, 0),
                        space.covector_member(memory.Y, 0), capacity(memory),
                        metric_blocks=memory.Z is not None, precond_block=memory.ZP is not None)


def memory_count(backend: Backend, memory: Memory[Any, Any]) -> Any:
    """The number of pairs stored."""
    xp = backend.xp
    return xp.sum(xp.where(memory.stamp >= 0, 1, 0))


def memory_store(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                 backend: Backend, memory: Memory[VectorBlockT, CovectorBlockT], s: VectorT,
                 y: CovectorT, stamp: Any, floor: Any, *, full_pairings: bool = False,
                 metric_pair: Optional[tuple[VectorT, CovectorT]] = None,
                 precond_y: Optional[VectorT] = None) -> Memory[VectorBlockT, CovectorBlockT]:
    """Store the pair (s, y) in a free slot, or in the oldest one when the
    memory is full; with metric blocks, metric_pair = (M^{-1} y, M s) goes
    into the same slot of Z and W, and with a preconditioner block,
    precond_y = P^{-1} y into ZP. One block pairing gives the pairings of y
    with every stored step (its row, 1/rho included); with full_pairings, a
    second one gives those of every stored y with s (its column), else the
    column is zero (it is not needed when the stored pairings are kept)."""
    xp, put = backend.xp, backend.index_update
    m = capacity(memory)
    index = xp.arange(m)
    # the lowest free slot (numpy blocks then append in place), else the oldest
    key = xp.where(memory.stamp >= 0, memory.stamp * m + index, index - m * m)
    slot = xp.take(xp.argsort(key), 0)
    S = space.set_vector(memory.S, slot, s)
    Y = space.set_covector(memory.Y, slot, y)
    row = space.pair_block(S, y, m)
    column = space.pair_covector_block(Y, s, m) if full_pairings else xp.zeros(m)
    pairings =xp.where(index[None, :] == slot, column[:, None], memory.pairings)
    pairings = xp.where(index[:, None] == slot, row[None, :], pairings)
    Z, W = memory.Z, memory.W
    if Z is not None and W is not None:
        if metric_pair is None:
            raise ValueError("a memory with metric blocks needs metric_pair = (M^{-1} y, M s)")
        Z, W = space.set_vector(Z, slot, metric_pair[0]), space.set_covector(W, slot, metric_pair[1])
    ZP = memory.ZP
    if ZP is not None:
        if precond_y is None:
            raise ValueError("a memory with a preconditioner block needs precond_y = P^{-1} y")
        ZP = space.set_vector(ZP, slot, precond_y)
    return Memory(S=S, Y=Y, pairings=pairings, stamp=put(memory.stamp, slot, stamp),
                  floor=put(memory.floor, slot, floor), Z=Z, W=W, ZP=ZP)


def _transport_members(backend: Backend, block: Any, member: Callable[[Any, Any], Any],
                       make_block: Callable[[Any, int], Any], set_member: Callable[[Any, Any, Any], Any],
                       move: Callable[[Any], Any], active: Any, m: int) -> Any:
    """A fresh block whose member j is move(member j) for the active slots
    (built by appending, so numpy blocks are written in place)."""
    xp = backend.xp

    def moved(j: Any) -> Any:
        return run_cond(backend, xp.take(active, j), move, lambda v: v, member(block, j))

    def step(c: tuple[Any, Any]) -> tuple[Any, Any]:
        j, out = c
        return j + 1, set_member(out, j, moved(j))

    start = (xp.asarray(1), make_block(moved(xp.asarray(0)), m))
    return run_loop(backend, lambda c: c[0] >= m, step, start, m - 1)[1]


def memory_transport(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                     backend: Backend,
                     transport: VectorTransport[PointT, VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                     x: PointT, x_new: PointT, memory: Memory[VectorBlockT, CovectorBlockT]
                     ) -> Memory[VectorBlockT, CovectorBlockT]:
    """The memory with S and Y carried from x to x_new, by the transport's
    block maps when it has them, else member by member (active slots only).
    The pairings are left as they are (see memory_recompute)."""
    if transport is identity_transport:
        return memory
    if memory.Z is not None or memory.ZP is not None:
        raise ValueError("the blocks of a constant metric or preconditioner need the identity transport")
    m, active = capacity(memory), memory.stamp >= 0
    if transport.vector_block is not None:
        S = transport.vector_block(x, x_new, memory.S)
    else:
        vector = transport.vector
        S = _transport_members(backend, memory.S, space.vector_member, space.vector_block,
                               space.set_vector, lambda v: vector(x, x_new, v), active, m)
    if transport.covector_block is not None:
        Y = transport.covector_block(x, x_new, memory.Y)
    else:
        covector = transport.covector
        Y = _transport_members(backend, memory.Y, space.covector_member, space.covector_block,
                               space.set_covector, lambda w: covector(x, x_new, w), active, m)
    return dataclasses.replace(memory, S=S, Y=Y)


def memory_recompute(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                     backend: Backend, memory: Memory[VectorBlockT, CovectorBlockT]
                     ) -> tuple[Memory[VectorBlockT, CovectorBlockT], Any]:
    """The pairings of the current vectors, <y_j, s_i> for all slots (one
    block pairing per slot), and the pairs whose curvature <y_i, s_i> no
    longer exceeds the floor they passed when stored dropped (their slots
    zeroed). Returns (memory, the number dropped)."""
    if memory.Z is not None or memory.ZP is not None:
        raise ValueError("with a constant metric or preconditioner the stored pairings stay current")
    xp, put = backend.xp, backend.index_update
    m = capacity(memory)
    index = xp.arange(m)

    def row(c: tuple[Any, Any]) -> tuple[Any, Any]:
        j, P = c
        paired = space.pair_block(memory.S, space.covector_member(memory.Y, j), m)
        return j + 1, xp.where(index[:, None] == j, paired[None, :], P)

    pairings = run_loop(backend, lambda c: c[0] >= m, row, (xp.asarray(0), xp.zeros((m, m))), m)[1]
    curvature = xp.sum(pairings * xp.eye(m), axis=1)
    active = memory.stamp >= 0
    drop = active & ~(xp.isfinite(curvature) & (curvature > memory.floor))

    def zero(blocks: tuple[Any, Any, Any]) -> tuple[Any, Any, Any]:
        S, Y, j = blocks
        return (space.set_vector(S, j, space.scale_vector(0.0, space.vector_member(S, j))),
                space.set_covector(Y, j, space.scale_covector(0.0, space.covector_member(Y, j))), j)

    def slot(c: tuple[Any, Any, Any]) -> tuple[Any, Any, Any]:
        S, Y, j = run_cond(backend, xp.take(drop, c[2]), zero, lambda b: b, c)
        return S, Y, j + 1

    S, Y, _ = run_loop(backend, lambda c: c[2] >= m, slot, (memory.S, memory.Y, xp.asarray(0)), m)
    kept = ~drop
    pairings = xp.where(kept[:, None] & kept[None, :], pairings, 0.0)
    dropped = xp.sum(xp.where(drop, 1, 0))
    return Memory(S=S, Y=Y, pairings=pairings, stamp=xp.where(drop, EMPTY, memory.stamp),
                  floor=xp.where(drop, 0.0, memory.floor)), dropped


def memory_correct(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                   backend: Backend, memory: Memory[VectorBlockT, CovectorBlockT],
                   metric_apply: Callable[[VectorT], CovectorT], regularization: float = 1e-12
                   ) -> tuple[Memory[VectorBlockT, CovectorBlockT], Any]:
    """The transport of a bilinear form B moves the pairs consistently when
    T(B)(T u, T v) = B(u, v); the stored pairings are the values that
    identity keeps. After a transport that does not preserve pairings,
    replace each covector y_j (already transported) by the nearest covector
    in the M^{-1}-norm, M = metric_apply at the new point, that pairs with
    the transported steps s_i as stored, <y_j', s_i> = pairings[j, i]:

        y_j' = y_j + sum_l C_jl M s_l,   C = (Pi - Y^T S) G^{-1},  G = S^T M S

    (one product with M per pair, one block pairing per pair for G and one
    for Y^T S, one block combination per pair; G is regularized by
    regularization times its mean diagonal). Needs the full stored
    pairings (memory_store(..., full_pairings=True)). Afterwards the stored
    and the recomputed pairings agree. Returns (memory, products with M)."""
    if memory.Z is not None or memory.ZP is not None:
        raise ValueError("with a constant metric or preconditioner the stored pairings stay current")
    xp = backend.xp
    m = capacity(memory)
    active = memory.stamp >= 0
    index = xp.arange(m)
    zero_w = space.scale_covector(0.0, space.covector_member(memory.Y, 0))

    def apply_member(j: Any) -> Any:
        return run_cond(backend, xp.take(active, j), lambda v: metric_apply(v), lambda v: zero_w,
                        space.vector_member(memory.S, j))

    def step(c: tuple[Any, Any]) -> tuple[Any, Any]:
        j, MS = c
        return j + 1, space.set_covector(MS, j, apply_member(j))

    MS = run_loop(backend, lambda c: c[0] >= m, step,
                  (xp.asarray(1), space.covector_block(apply_member(xp.asarray(0)), m)), m - 1)[1]

    def rows(c: tuple[Any, Any, Any]) -> tuple[Any, Any, Any]:
        j, G, P = c
        g_row = space.pair_block(memory.S, space.covector_member(MS, j), m)       # <M s_j, s_i>
        p_row = space.pair_block(memory.S, space.covector_member(memory.Y, j), m)  # <y_j, s_i>
        here = index[:, None] == j
        return j + 1, xp.where(here, g_row[None, :], G), xp.where(here, p_row[None, :], P)

    _, G, P = run_loop(backend, lambda c: c[0] >= m, rows,
                       (xp.asarray(0), xp.zeros((m, m)), xp.zeros((m, m))), m)
    both = active[:, None] & active[None, :]
    eye = xp.eye(m)
    G = xp.where(both, G, eye)
    scale = xp.sum(G * eye) / m
    G = G + regularization * scale * eye
    C = xp.linalg.solve(G, xp.where(both, memory.pairings - P, 0.0).T).T

    def corrected(j: Any) -> Any:
        y = space.covector_member(memory.Y, j)
        return run_cond(backend, xp.take(active, j),
                        lambda y: space.axpy_covector(1.0, space.combine_covectors(MS, xp.take(C, j, axis=0), m), y),
                        lambda y: y, y)

    def store(c: tuple[Any, Any]) -> tuple[Any, Any]:
        j, Y = c
        return j + 1, space.set_covector(Y, j, corrected(j))

    Y = run_loop(backend, lambda c: c[0] >= m, store,
                 (xp.asarray(1), space.covector_block(corrected(xp.asarray(0)), m)), m - 1)[1]
    return dataclasses.replace(memory, Y=Y), xp.sum(xp.where(active, 1, 0))


def memory_layout(backend: Backend, memory: Memory[Any, Any]) -> Layout:
    """The coordinates of the two-loop recursion (see Layout)."""
    xp = backend.xp
    m = capacity(memory)
    perm = xp.argsort(memory.stamp)
    active = xp.take(memory.stamp, perm) >= 0
    P = xp.take(xp.take(memory.pairings, perm, axis=0), perm, axis=1)     # P[j, i] = <y_j, s_i>
    index = xp.arange(m)
    eye = xp.eye(m)
    U = xp.where(active[:, None] & active[None, :] & (index[:, None] < index[None, :]), P.T, 0.0)
    d = xp.where(active, xp.sum(P * eye, axis=1), 1.0)
    return Layout(perm=perm, inverse=xp.argsort(perm), active=active, U=U, R=U + eye * d[:, None])


def two_loop_project(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                     backend: Backend, memory: Memory[VectorBlockT, CovectorBlockT], layout: Layout,
                     g: CovectorT) -> tuple[CovectorT, Any]:
    """The first half of the recursion: (q, a) with q = g - Y a."""
    xp = backend.xp
    m = capacity(memory)
    c = xp.where(layout.active, xp.take(space.pair_block(memory.S, g, m), layout.perm), 0.0)
    a = xp.linalg.solve(layout.R, c)
    q = space.axpy_covector(-1.0, space.combine_covectors(memory.Y, xp.take(a, layout.inverse), m), g)
    return q, a


def finish_coefficients(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                         backend: Backend, memory: Memory[VectorBlockT, CovectorBlockT], layout: Layout,
                         a: Any, r0: VectorT) -> Any:
    """a - b by slot, for r0 = H0 q."""
    xp = backend.xp
    m = capacity(memory)
    e = xp.where(layout.active, xp.take(space.pair_covector_block(memory.Y, r0, m), layout.perm), 0.0)
    b = xp.linalg.solve(layout.R.T, e + layout.U.T @ a)
    return xp.take(a - b, layout.inverse)


def two_loop_finish(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                    backend: Backend, memory: Memory[VectorBlockT, CovectorBlockT], layout: Layout,
                    a: Any, r0: VectorT) -> VectorT:
    """The second half: H g = r0 + S (a - b), for r0 = H0 q."""
    coefficients = finish_coefficients(space, backend, memory, layout, a, r0)
    return space.axpy_vector(1.0, space.combine_vectors(memory.S, coefficients, capacity(memory)), r0)


def block_base(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT], backend: Backend,
               memory: Memory[VectorBlockT, CovectorBlockT], layout: Layout, a: Any, block: VectorBlockT,
               known: VectorT, gamma: Any) -> VectorT:
    """The base gamma A q = gamma (A g - (A Y) a) of a constant linear
    operator A (M^{-1} or P^{-1}), from known = A g and block = A Y: no
    application of A."""
    combined = space.combine_vectors(block, backend.xp.take(a, layout.inverse), capacity(memory))
    return space.scale_vector(gamma, space.axpy_vector(-1.0, combined, known))


def apply_inverse(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                  backend: Backend, memory: Memory[VectorBlockT, CovectorBlockT], g: CovectorT,
                  base: Callable[[CovectorT], VectorT]) -> VectorT:
    """H g for the base H0 = base (one application of it)."""
    layout = memory_layout(backend, memory)
    q, a = two_loop_project(space, backend, memory, layout, g)
    return two_loop_finish(space, backend, memory, layout, a, base(q))


def apply_inverse_fixed(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                        backend: Backend, memory: Memory[VectorBlockT, CovectorBlockT], g: CovectorT,
                        Mg: VectorT, gamma: Any) -> tuple[VectorT, CovectorT]:
    """(H g, M H g) for the base gamma M^{-1} of a constant metric, from
    M^{-1} g and the metric blocks, with no operator applications: the base
    is r0 = gamma M^{-1} q = gamma (M^{-1} g - Z a) by linearity, and
    M H g = gamma q + W (a - b)."""
    Z, W = memory.Z, memory.W
    if Z is None or W is None:
        raise ValueError("apply_inverse_fixed needs a memory with metric blocks")
    m = capacity(memory)
    layout = memory_layout(backend, memory)
    q, a = two_loop_project(space, backend, memory, layout, g)
    r0 = block_base(space, backend, memory, layout, a, Z, Mg, gamma)
    coefficients = finish_coefficients(space, backend, memory, layout, a, r0)
    Hg = space.axpy_vector(1.0, space.combine_vectors(memory.S, coefficients, m), r0)
    MHg = space.axpy_covector(1.0, space.combine_covectors(W, coefficients, m), space.scale_covector(gamma, q))
    return Hg, MHg
