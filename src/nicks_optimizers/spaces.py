"""Representation of points, vectors and covectors.

Optimizers in this package never touch the storage of their objects. They
work through a VectorSpace record: a frozen dataclass of functions (no
methods) saying how vectors are scaled, paired, collected into blocks and
combined. numpy_space, the default, uses plain numpy arrays; other
representations (distributed objects, collections of arrays, GPU arrays, ...)
supply their own record:

  numpy_space                          one numpy array per vector
  numpy_space_with_reductions(allreduce)  one local numpy array per process
                                       (e.g. MPI ranks); only the reductions
                                       change
  tuple_block_space(...)               any vector type, from five functions;
                                       blocks are tuples of vectors
  dataclasses.replace(space, ...)      override individual entries

check_space(space, ...) verifies a record's internal consistency on sample
vectors. Vector and covector operations are separate fields on purpose, so a
space spells out which kind of object every function acts on. A space says
nothing about geometry; inner products live in metrics.

Types
-----
  Point      an iterate x. Points are never added; a step s moves x to
             retract(x, s). On a flat space the default retraction is
             space.translate(x, s) = x + s; a manifold has no translate and
             supplies retract instead.
  Vector     a tangent (primal) vector at x: steps, Krylov vectors q_j.
  Covector   a cotangent (dual) vector at x: gradients, H v, residuals.
  blocks     collections of vectors or covectors (Krylov bases).
  Coords     small coordinate arrays: always numpy.

Algorithms are generic over these types (type variables PointT, VectorT, ...).
For numpy_space they are the NewTypes Point, Vector, Covector, VectorBlock and
CovectorBlock, erased at run time: plain arrays work unchanged, and a static
checker (mypy, pyright) can catch primal/dual mix-ups if you wrap the arrays
your functions return (Covector(...) and so on). The same works for your own
representation if you define NewTypes over it.
"""

import dataclasses
from dataclasses import dataclass
from typing import Callable, Generic, NewType, Optional, TypeVar

import numpy as np
import numpy.typing as npt


Array = npt.NDArray[np.float64]
Coords = Array
PointT = TypeVar("PointT")
VectorT = TypeVar("VectorT")
CovectorT = TypeVar("CovectorT")
VectorBlockT = TypeVar("VectorBlockT")
CovectorBlockT = TypeVar("CovectorBlockT")


@dataclass(frozen=True, kw_only=True)
class VectorSpace(Generic[PointT, VectorT, CovectorT, VectorBlockT, CovectorBlockT]):
    """A record of the functions the method uses to handle its objects.

    A frozen, keyword-only dataclass used purely as a composite value: no
    methods, no inheritance. Build one with a constructor (numpy_space,
    numpy_space_with_reductions, tuple_block_space) and change entries with
    dataclasses.replace. Every entry is required except translate and
    dimension.

      scale_vector(a, v) -> Vector           a v
      scale_covector(a, w) -> Covector       a w
      axpy_vector(a, v, z) -> Vector         a v + z
      axpy_covector(a, w, z) -> Covector     a w + z
      pair(w, v) -> float                    the duality pairing <w, v>
      pair_block(B, w) -> Coords             <w, b_i> for each vector b_i in B
      vector_block(v) -> VectorBlock         the one-vector block [v]
      covector_block(w) -> CovectorBlock     the one-covector block [w]
      append_vector(B, v) -> VectorBlock     a new block [B, v]
      append_covector(C, w) -> CovectorBlock a new block [C, w]
      combine_vectors(B, c) -> Vector        sum_i c_i b_i
      combine_covectors(C, c) -> Covector    sum_i c_i w_i
      translate(x, s) -> Point               optional: x + s, the point update of
                                             a flat space (the default
                                             retraction); None on a manifold
      dimension(v) -> int                    optional: the dimension n, used to
                                             cap the number of Lanczos steps

    The method never modifies an object in place, and it only calls these
    functions; pairings are the only places where numbers come out of vectors.
    """
    scale_vector: Callable[[float, VectorT], VectorT]
    scale_covector: Callable[[float, CovectorT], CovectorT]
    axpy_vector: Callable[[float, VectorT, VectorT], VectorT]
    axpy_covector: Callable[[float, CovectorT, CovectorT], CovectorT]
    pair: Callable[[CovectorT, VectorT], float]
    pair_block: Callable[[VectorBlockT, CovectorT], Coords]
    vector_block: Callable[[VectorT], VectorBlockT]
    covector_block: Callable[[CovectorT], CovectorBlockT]
    append_vector: Callable[[VectorBlockT, VectorT], VectorBlockT]
    append_covector: Callable[[CovectorBlockT, CovectorT], CovectorBlockT]
    combine_vectors: Callable[[VectorBlockT, Coords], VectorT]
    combine_covectors: Callable[[CovectorBlockT, Coords], CovectorT]
    translate: Optional[Callable[[PointT, VectorT], PointT]] = None
    dimension: Optional[Callable[[VectorT], int]] = None


# --- numpy representation ---------------------------------------------------


Point = NewType("Point", Array)
Vector = NewType("Vector", Array)
Covector = NewType("Covector", Array)
VectorBlock = NewType("VectorBlock", Array)
CovectorBlock = NewType("CovectorBlock", Array)
NumpySpace = VectorSpace[Point, Vector, Covector, VectorBlock, CovectorBlock]


# Signatures of problem pieces for numpy_space (documentation aliases).


Objective = Callable[[Point], float]
Gradient = Callable[[Point], Covector]
HessianAction = Callable[[Point, Vector], Covector]
PreconditionerSolve = Callable[[Point, Covector], Vector]
Retraction = Callable[[Point, Vector], Point]
VectorToCovector = Callable[[Vector], Covector]
CovectorToVector = Callable[[Covector], Vector]


def _np_scale_vector(a: float, v: Vector) -> Vector:
    return Vector(a * v)


def _np_scale_covector(a: float, w: Covector) -> Covector:
    return Covector(a * w)


def _np_axpy_vector(a: float, v: Vector, z: Vector) -> Vector:
    return Vector(a * v + z)


def _np_axpy_covector(a: float, w: Covector, z: Covector) -> Covector:
    return Covector(a * w + z)


def _np_pair(w: Covector, v: Vector) -> float:
    return float(w @ v)


def _np_pair_block(B: VectorBlock, w: Covector) -> Coords:
    return B.T @ w


def _np_vector_block(v: Vector) -> VectorBlock:
    return VectorBlock(v[:, None])


def _np_covector_block(w: Covector) -> CovectorBlock:
    return CovectorBlock(w[:, None])


def _np_append_vector(B: VectorBlock, v: Vector) -> VectorBlock:
    return VectorBlock(np.column_stack((B, v)))


def _np_append_covector(C: CovectorBlock, w: Covector) -> CovectorBlock:
    return CovectorBlock(np.column_stack((C, w)))


def _np_combine_vectors(B: VectorBlock, c: Coords) -> Vector:
    return Vector(B @ c)


def _np_combine_covectors(C: CovectorBlock, c: Coords) -> Covector:
    return Covector(C @ c)


def _np_translate(x: Point, s: Vector) -> Point:
    return Point(x + s)


def _np_dimension(v: Vector) -> int:
    return int(v.shape[0])


numpy_space: NumpySpace = VectorSpace(
    scale_vector=_np_scale_vector, scale_covector=_np_scale_covector,
    axpy_vector=_np_axpy_vector, axpy_covector=_np_axpy_covector,
    pair=_np_pair, pair_block=_np_pair_block,
    vector_block=_np_vector_block, covector_block=_np_covector_block,
    append_vector=_np_append_vector, append_covector=_np_append_covector,
    combine_vectors=_np_combine_vectors, combine_covectors=_np_combine_covectors,
    translate=_np_translate, dimension=_np_dimension)


def numpy_space_with_reductions(allreduce: Callable[[Coords], Coords]) -> NumpySpace:
    """numpy_space for data stored as one local numpy array per process
    (e.g. MPI ranks).

    Local operations stay numpy; the reductions (pair, pair_block, dimension)
    are completed with allreduce, which must return the elementwise sum of its
    argument over all processes, identical on every process. Every decision
    the method makes comes from reduced numbers, so the processes stay in
    lockstep. Your inner products must do their own reduction.
    """
    def pair(w: Covector, v: Vector) -> float:
        return float(allreduce(np.array([float(w @ v)]))[0])

    def pair_block(B: VectorBlock, w: Covector) -> Coords:
        return allreduce(B.T @ w)

    def dimension(v: Vector) -> int:
        return int(round(float(allreduce(np.array([float(v.shape[0])]))[0])))

    return dataclasses.replace(numpy_space, pair=pair, pair_block=pair_block, dimension=dimension)


def tuple_block_space(*, scale_vector: Callable[[float, VectorT], VectorT],
                      scale_covector: Callable[[float, CovectorT], CovectorT],
                      axpy_vector: Callable[[float, VectorT, VectorT], VectorT],
                      axpy_covector: Callable[[float, CovectorT, CovectorT], CovectorT],
                      pair: Callable[[CovectorT, VectorT], float],
                      translate: Optional[Callable[[PointT, VectorT], PointT]] = None,
                      dimension: Optional[Callable[[VectorT], int]] = None
                      ) -> VectorSpace[PointT, VectorT, CovectorT,
                                       tuple[VectorT, ...], tuple[CovectorT, ...]]:
    """A complete space for any vector type, from five functions (plus
    translate for a flat space, and optionally dimension).

    Blocks are tuples of vectors (or covectors), and the block entries are
    built from the functions given: pair_block pairs one vector at a time and
    combine_* accumulate with scale and axpy. That is correct for any
    representation; for distributed objects it costs one reduction per block
    member, so override pair_block (and combine_*) with dataclasses.replace
    when a batched version is available. Your inner products then receive
    tuples as their block argument.
    """
    def pair_block(B: tuple[VectorT, ...], w: CovectorT) -> Coords:
        return np.array([pair(w, b) for b in B], dtype=float)

    def combine_vectors(B: tuple[VectorT, ...], c: Coords) -> VectorT:
        acc = scale_vector(float(c[0]), B[0])
        for ci, bi in zip(c[1:], B[1:]):
            acc = axpy_vector(float(ci), bi, acc)
        return acc

    def combine_covectors(C: tuple[CovectorT, ...], c: Coords) -> CovectorT:
        acc = scale_covector(float(c[0]), C[0])
        for ci, wi in zip(c[1:], C[1:]):
            acc = axpy_covector(float(ci), wi, acc)
        return acc

    def vector_block(v: VectorT) -> tuple[VectorT, ...]:
        return (v,)

    def covector_block(w: CovectorT) -> tuple[CovectorT, ...]:
        return (w,)

    def append_vector(B: tuple[VectorT, ...], v: VectorT) -> tuple[VectorT, ...]:
        return B + (v,)

    def append_covector(C: tuple[CovectorT, ...], w: CovectorT) -> tuple[CovectorT, ...]:
        return C + (w,)

    return VectorSpace(
        scale_vector=scale_vector, scale_covector=scale_covector,
        axpy_vector=axpy_vector, axpy_covector=axpy_covector, pair=pair, pair_block=pair_block,
        vector_block=vector_block, covector_block=covector_block,
        append_vector=append_vector, append_covector=append_covector,
        combine_vectors=combine_vectors, combine_covectors=combine_covectors,
        translate=translate, dimension=dimension)


def check_space(space: VectorSpace[PointT, VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                v1: VectorT, v2: VectorT, w1: CovectorT, w2: CovectorT) -> float:
    """Largest relative violation of the identities a VectorSpace must satisfy,
    tested through pairings on two sample vectors and two sample covectors
    (so it works for any representation). Expect rounding level, ~1e-15."""
    a, b = 0.7, -1.3
    c = np.array([a, b])
    p11, p12, p21, p22 = (space.pair(w1, v1), space.pair(w1, v2),
                          space.pair(w2, v1), space.pair(w2, v2))
    VB = space.append_vector(space.vector_block(v1), v2)
    CB = space.append_covector(space.covector_block(w1), w2)
    errors = [
        float(np.max(np.abs(space.pair_block(VB, w1) - np.array([p11, p12])))),
        abs(float(space.pair_block(space.vector_block(v2), w2)[0]) - p22),
        abs(space.pair(w1, space.combine_vectors(VB, c)) - (a * p11 + b * p12)),
        abs(space.pair(space.combine_covectors(CB, c), v1) - (a * p11 + b * p21)),
        abs(space.pair(w1, space.scale_vector(a, v1)) - a * p11),
        abs(space.pair(space.scale_covector(a, w1), v1) - a * p11),
        abs(space.pair(space.axpy_covector(a, w1, w2), v1) - (a * p11 + p21)),
        abs(space.pair(w1, space.axpy_vector(a, v1, v2)) - (a * p11 + p12)),
    ]
    scale = max(abs(p11), abs(p12), abs(p21), abs(p22), 1e-300)
    return max(errors) / scale
