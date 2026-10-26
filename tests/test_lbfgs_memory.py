"""The memory of lbfgs and the two-loop recursion in coordinates
(lbfgs_memory), against dense references: the textbook sequential two-loop
recursion and the BFGS recursion as matrices; stale pairings after
transports that do not preserve pairings; recomputed pairings and dropped
pairs; transports member by member and by block; the strict backend."""

from typing import Any, Callable

import numpy as np
import pytest

from nicks_optimizers import (
    Array,
    Covector,
    Vector,
    VectorTransport,
    array_space,
    identity_transport,
    numpy_backend,
    numpy_space,
    riesz_covector_transport,
)
from nicks_optimizers.lbfgs_memory import (
    Memory,
    apply_inverse,
    memory_clear,
    memory_count,
    memory_empty,
    memory_recompute,
    memory_store,
    memory_transport,
)
from strict_backend import plain, strict_backend

N = 12
Mem = Memory[Any, Any]


def spd(rng: np.random.Generator, n: int, shift: float = 1.0) -> Array:
    a = rng.standard_normal((n, n))
    out: Array = a @ a.T + shift * np.eye(n)
    return out


def pairs(rng: np.random.Generator, A: Array, k: int) -> list[tuple[Vector, Covector]]:
    """k random steps and their exact gradient changes y = A s."""
    out = []
    for _ in range(k):
        s = Vector(rng.standard_normal(int(A.shape[0])))
        out.append((s, Covector(A @ s)))
    return out


def stored(steps: list[tuple[Vector, Covector]], m: int, *, full: bool = False) -> Mem:
    memory: Mem = memory_empty(numpy_space, numpy_backend, steps[0][0], steps[0][1], m)
    for k, (s, y) in enumerate(steps):
        memory = memory_store(numpy_space, numpy_backend, memory, s, y, np.asarray(k),
                              np.asarray(0.0), full_pairings=full)
    return memory


def operator(memory: Mem, base: Array, backend: Any = numpy_backend, space: Any = numpy_space) -> Array:
    """H as a matrix: H applied to the unit covectors."""
    Hinv = np.linalg.inv(base)                       # base is B0; H0 = B0^{-1}
    cols = [apply_inverse(space, backend, memory, Covector(e), lambda q: Vector(np.asarray(Hinv @ q)))
            for e in np.eye(base.shape[0])]
    return np.column_stack([np.asarray(plain(c)) for c in cols])


def sequential(steps: list[tuple[Vector, Covector]], H0: Array, g: Array) -> Array:
    """The textbook two-loop recursion (Nocedal and Wright, Algorithm 7.4)."""
    rho = [1.0 / float(y @ s) for s, y in steps]
    q, a = g.copy(), [0.0] * len(steps)
    for i in reversed(range(len(steps))):
        a[i] = rho[i] * float(steps[i][0] @ q)
        q = q - a[i] * steps[i][1]
    r: Array = H0 @ q
    for i in range(len(steps)):
        b = rho[i] * float(steps[i][1] @ r)
        r = r + (a[i] - b) * steps[i][0]
    return r


def dense_bfgs(steps: list[tuple[Vector, Covector]], H0: Array) -> Array:
    """The BFGS recursion H+ = (I - rho s y^T) H (I - rho y s^T) + rho s s^T."""
    H = H0.copy()
    I = np.eye(H0.shape[0])
    for s, y in steps:
        rho = 1.0 / float(y @ s)
        H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
    return H


def rel(a: Array, b: Array) -> float:
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


@pytest.mark.parametrize("m", [1, 2, 5, N])
def test_two_loop_matches_the_textbook_recursions(m: int) -> None:
    """With the pairings of the current vectors, the coordinate recursion is
    the sequential two-loop recursion and the dense BFGS recursion over the
    last m pairs, also after the memory has wrapped around."""
    rng = np.random.default_rng(m)
    A, B0 = spd(rng, N, float(N)), spd(rng, N)
    steps = pairs(rng, A, m + 3)
    memory = stored(steps, m)
    assert int(memory_count(numpy_backend, memory)) == m
    H = operator(memory, B0)
    H0 = np.linalg.inv(B0)
    assert rel(H, dense_bfgs(steps[-m:], H0)) < 1e-12
    g = rng.standard_normal(N)
    assert rel(H @ g, sequential(steps[-m:], H0, g)) < 1e-12
    s, y = steps[-1]
    assert np.linalg.norm(H @ y - s) < 1e-10 * np.linalg.norm(s)


def test_partial_and_cleared_memories() -> None:
    """Fewer pairs than slots use only those; a cleared memory gives H = H0."""
    rng = np.random.default_rng(1)
    A, B0 = spd(rng, N, float(N)), spd(rng, N)
    steps = pairs(rng, A, 3)
    memory = stored(steps, 6)
    H0 = np.linalg.inv(B0)
    assert rel(operator(memory, B0), dense_bfgs(steps, H0)) < 1e-12
    cleared = memory_clear(numpy_space, numpy_backend, memory)
    assert int(memory_count(numpy_backend, cleared)) == 0
    assert rel(operator(cleared, B0), H0) < 1e-14


def matrix_transport(T: Array, Tb: Array, blocks: bool = False) -> VectorTransport[Any, Any, Any, Any, Any]:
    vector: Callable[[Any, Any, Any], Any] = lambda x, x2, v: Vector(T @ v)
    covector: Callable[[Any, Any, Any], Any] = lambda x, x2, w: Covector(Tb @ w)
    if not blocks:
        return VectorTransport(vector=vector, covector=covector)

    def block_map(Mat: Array) -> Callable[[Any, Any, Any], Any]:
        def moved(x: Any, x2: Any, B: Any) -> Any:
            out = numpy_space.vector_block(Vector(Mat @ numpy_space.vector_member(B, 0)), B.capacity)
            for j in range(1, B.count):
                out = numpy_space.set_vector(out, j, Vector(Mat @ numpy_space.vector_member(B, j)))
            return out
        return moved
    return VectorTransport(vector=vector, covector=covector, vector_block=block_map(T),
                           covector_block=block_map(Tb))


def test_stale_pairings_keep_the_operator_positive_definite() -> None:
    """After transports that do not preserve pairings, the kept pairings give
    a symmetric positive definite operator, and a pair stored at the new
    point satisfies its secant equation exactly."""
    rng = np.random.default_rng(2)
    A = spd(rng, N, float(N))
    worst = np.inf
    for trial in range(200):
        memory = stored(pairs(rng, A, 5), 5)
        for _ in range(3):
            T = np.eye(N) + 0.4 * rng.standard_normal((N, N))
            Tb = np.eye(N) + 0.4 * rng.standard_normal((N, N))
            memory = memory_transport(numpy_space, numpy_backend, matrix_transport(T, Tb), None, None, memory)
        s, y = pairs(rng, A, 1)[0]
        memory = memory_store(numpy_space, numpy_backend, memory, s, y, np.asarray(10), np.asarray(0.0))
        B0 = spd(rng, N)
        H = operator(memory, B0)
        assert rel(H, H.T) < 1e-10
        worst = min(worst, float(np.linalg.eigvalsh(0.5 * (H + H.T)).min() / np.abs(H).max()))
        assert np.linalg.norm(H @ y - s) < 1e-8 * np.linalg.norm(s)
    assert worst > 0.0


def test_pairing_preserving_transports_keep_the_pairings_current() -> None:
    """With Tb = T^{-T}, the kept pairings are the pairings of the
    transported vectors, and recomputing changes nothing."""
    rng = np.random.default_rng(3)
    A = spd(rng, N, float(N))
    memory = stored(pairs(rng, A, 4), 4, full=True)
    T = np.eye(N) + 0.3 * rng.standard_normal((N, N))
    moved = memory_transport(numpy_space, numpy_backend, matrix_transport(T, np.linalg.inv(T).T),
                             None, None, memory)
    again, dropped = memory_recompute(numpy_space, numpy_backend, moved)
    assert int(dropped) == 0
    assert rel(np.asarray(again.pairings), np.asarray(memory.pairings)) < 1e-12


def test_recomputed_pairings_are_the_textbook_operator_of_the_transported_pairs() -> None:
    """memory_recompute gives the pairings of the current vectors, so the
    operator is the BFGS operator of the transported pairs; a pair whose
    transported curvature fell to its floor or below is dropped."""
    rng = np.random.default_rng(4)
    A, B0 = spd(rng, N, float(N)), spd(rng, N)
    steps = pairs(rng, A, 4)
    memory = stored(steps, 4)
    T = np.eye(N) + 0.1 * rng.standard_normal((N, N))
    Tb = np.eye(N) + 0.1 * rng.standard_normal((N, N))
    moved = memory_transport(numpy_space, numpy_backend, matrix_transport(T, Tb), None, None, memory)
    again, dropped = memory_recompute(numpy_space, numpy_backend, moved)
    assert int(dropped) == 0
    transported = [(Vector(T @ s), Covector(Tb @ y)) for s, y in steps]
    assert rel(operator(again, B0), dense_bfgs(transported, np.linalg.inv(B0))) < 1e-10

    # a covector transport that sends y_0 to -y_0 and fixes the other y_j
    Ys = np.column_stack([y for _, y in steps])
    z = np.linalg.lstsq(Ys.T, np.eye(4)[0], rcond=None)[0]           # z.y_0 = 1, z.y_j = 0
    flip = np.eye(N) - 2.0 * np.outer(steps[0][1], z)
    worse = memory_transport(numpy_space, numpy_backend, matrix_transport(np.eye(N), flip), None, None, memory)
    kept, dropped = memory_recompute(numpy_space, numpy_backend, worse)
    assert int(dropped) == 1 and int(memory_count(numpy_backend, kept)) == 3
    assert np.all(numpy_space.vector_member(kept.S, 0) == 0.0)        # its slot is zeroed
    assert rel(operator(kept, B0), dense_bfgs(steps[1:], np.linalg.inv(B0))) < 1e-10


def test_transport_by_member_and_by_block_agree() -> None:
    rng = np.random.default_rng(5)
    A = spd(rng, N, float(N))
    memory = stored(pairs(rng, A, 3), 5)
    T, Tb = rng.standard_normal((N, N)), rng.standard_normal((N, N))
    by_member = memory_transport(numpy_space, numpy_backend, matrix_transport(T, Tb), None, None, memory)
    by_block = memory_transport(numpy_space, numpy_backend, matrix_transport(T, Tb, blocks=True),
                                None, None, memory)
    for j in range(5):
        assert np.allclose(numpy_space.vector_member(by_member.S, j), numpy_space.vector_member(by_block.S, j))
        assert np.allclose(numpy_space.covector_member(by_member.Y, j),
                           numpy_space.covector_member(by_block.Y, j))
    assert np.all(numpy_space.vector_member(by_member.S, 4) == 0.0)     # free slots stay zero
    assert memory_transport(numpy_space, numpy_backend, identity_transport, None, None, memory) is memory


def test_riesz_covector_transport_preserves_pairings_for_isometries() -> None:
    """Tb = M' T M^{-1} preserves pairings when T is isometric from M to M'."""
    rng = np.random.default_rng(6)
    M0, M1 = spd(rng, N), spd(rng, N)
    L0, L1 = np.linalg.cholesky(M0), np.linalg.cholesky(M1)
    Q = np.linalg.qr(rng.standard_normal((N, N)))[0]
    T = np.linalg.solve(L1.T, Q @ L0.T)              # T^T M1 T = M0
    assert rel(T.T @ M1 @ T, M0) < 1e-12
    points = {0: M0, 1: M1}
    transport: VectorTransport[int, Array, Array, Any, Any] = riesz_covector_transport(
        lambda x, x2, v: T @ v,
                                         lambda x, w: np.linalg.solve(points[x], w),
                                         lambda x, v: points[x] @ v)
    v, w = rng.standard_normal(N), rng.standard_normal(N)
    paired = transport.covector(0, 1, w) @ transport.vector(0, 1, v)
    assert abs(paired - w @ v) < 1e-9 * np.linalg.norm(w) * np.linalg.norm(v)


@pytest.mark.parametrize("count", [0, 2, 4, 7])
def test_strict_backend_agrees_with_numpy(count: int) -> None:
    """The code path of a compiling backend (array_space, fixed shapes, no
    Python control flow on values) gives the numpy operator, for memories
    empty, partly filled, full and wrapped around, and member-by-member
    transports work there too."""
    rng = np.random.default_rng(10 + count)
    A, B0 = spd(rng, N, float(N)), spd(rng, N)
    steps = pairs(rng, A, count)
    m = 4
    space = array_space(strict_backend)
    sample = np.zeros(N)
    memory: Mem = memory_empty(space, strict_backend, sample, sample, m)
    reference: Mem = memory_empty(numpy_space, numpy_backend, Vector(sample), Covector(sample), m)
    for k, (s, y) in enumerate(steps):
        memory = memory_store(space, strict_backend, memory, s, y, np.asarray(k), np.asarray(0.0))
        reference = memory_store(numpy_space, numpy_backend, reference, s, y, np.asarray(k), np.asarray(0.0))
    assert rel(operator(memory, B0, strict_backend, space), operator(reference, B0)) < 1e-12
    T, Tb = rng.standard_normal((N, N)), rng.standard_normal((N, N))
    moved = memory_transport(space, strict_backend, matrix_transport(T, Tb), None, None, memory)
    moved_ref = memory_transport(numpy_space, numpy_backend, matrix_transport(T, Tb), None, None, reference)
    assert np.allclose(np.asarray(plain(moved.S)), np.stack([numpy_space.vector_member(moved_ref.S, j)
                                                             for j in range(m)]))
    again, _ = memory_recompute(space, strict_backend, moved)
    again_ref, _ = memory_recompute(numpy_space, numpy_backend, moved_ref)
    assert np.allclose(np.asarray(plain(again.pairings)), np.asarray(again_ref.pairings))
