"""The vector space structure: how vectors and covectors are represented.

Optimizers in this package never touch the storage of their objects. They
work through a VectorSpace record: a frozen dataclass of functions (no
methods) saying how vectors and covectors are scaled and added, how they
pair, and how they are collected into blocks and combined. That is all a
VectorSpace holds. The Hilbert structure (a metric M) is given separately,
as operators (see metrics), and so is the point structure (the retraction
x, s -> retract(x, s)).

  numpy_space                          one numpy array (of any shape) per
                                       vector; the default
  numpy_space_with_reductions(allreduce)  one local numpy array per process
                                       (e.g. MPI ranks); only the reductions
                                       change
  tuple_tree_space(leaf_space)         tuple trees whose leaves are vectors
                                       of another space (numpy arrays by
                                       default), with leafwise operations
  tuple_block_space(...)               any vector type, from five functions;
                                       blocks are tuples of vectors
  dataclasses.replace(space, ...)      override individual entries

flat_retraction(space) is the retraction x + s for points stored like
vectors. check_space(space, ...) verifies a record's internal consistency on
sample vectors. Vector and covector operations are separate fields on
purpose, so a space spells out which kind of object every function acts on.

Types
-----
  Point      an iterate x. Points are not part of the vector space: a step s
             moves x to retract(x, s).
  Vector     a tangent (primal) vector at x: steps, Krylov vectors q_j.
  Covector   a cotangent (dual) vector at x: gradients, H v, residuals.
  blocks     collections of vectors or covectors (Krylov bases).
  Coords     small coordinate arrays: always numpy.

Algorithms are generic over these types (type variables PointT, VectorT, ...).
For numpy arrays they are the NewTypes Point, Vector, Covector, VectorBlock
and CovectorBlock, erased at run time: plain arrays work unchanged, and a
static checker (mypy, pyright) can catch primal/dual mix-ups if you wrap the
arrays your functions return (Covector(...) and so on). The same works for
your own representation if you define NewTypes over it.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Generic, NewType, Optional, TypeVar, cast, overload

import numpy as np
import numpy.typing as npt

from .trees import Tree, tree_leaf_pairs, tree_leaves, tree_map, tree_map2


Array = npt.NDArray[np.float64]
Coords = Array
PointT = TypeVar("PointT")
VectorT = TypeVar("VectorT")
CovectorT = TypeVar("CovectorT")
VectorBlockT = TypeVar("VectorBlockT")
CovectorBlockT = TypeVar("CovectorBlockT")


@dataclass(frozen=True, kw_only=True)
class VectorSpace(Generic[VectorT, CovectorT, VectorBlockT, CovectorBlockT]):
    """A record of the functions the method uses to handle vectors and
    covectors.

    A frozen, keyword-only dataclass used purely as a composite value: no
    methods, no inheritance. Build one with a constructor (numpy_space,
    numpy_space_with_reductions, tuple_tree_space, tuple_block_space) and
    change entries with dataclasses.replace. Every entry is required except
    dimension.

      scale_vector(a, v) -> Vector           a v
      scale_covector(a, w) -> Covector       a w
      axpy_vector(a, v, z) -> Vector         a v + z
      axpy_covector(a, w, z) -> Covector     a w + z
      pair(w, v) -> Scalar                   the duality pairing <w, v>: a 0-d
                                             array (a numpy scalar for numpy)
      pair_block(B, w) -> Coords             <w, b_i> for each vector b_i in B
      vector_block(v) -> VectorBlock         the one-vector block [v]
      covector_block(w) -> CovectorBlock     the one-covector block [w]
      append_vector(B, v) -> VectorBlock     a new block [B, v]
      append_covector(C, w) -> CovectorBlock a new block [C, w]
      combine_vectors(B, c) -> Vector        sum_i c_i b_i
      combine_covectors(C, c) -> Covector    sum_i c_i w_i
      dimension(v) -> int                    optional: the dimension n, used to
                                             cap the number of Lanczos steps

    The method never modifies an object in place, and it only calls these
    functions; pairings are the only places where numbers come out of vectors.
    They return arrays rather than Python floats, so that a backend can
    compile the method (see backend); a space that returns floats still
    works with the numpy backend.
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
    dimension: Optional[Callable[[VectorT], int]] = None


# --- numpy representation ---------------------------------------------------


Point = NewType("Point", Array)
Vector = NewType("Vector", Array)
Covector = NewType("Covector", Array)
VectorBlock = NewType("VectorBlock", Array)
CovectorBlock = NewType("CovectorBlock", Array)
NumpySpace = VectorSpace[Vector, Covector, VectorBlock, CovectorBlock]


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


# Vectors may have any shape; a block of k of them has the shape of a vector
# plus a trailing axis of length k (a matrix of columns for 1-D vectors).
# Products flatten the vector axes, so an array of any shape computes exactly
# what its flattened 1-D version would.


def _np_pair(w: Covector, v: Vector) -> float:
    return cast(float, w.ravel() @ v.ravel())        # a numpy scalar, or a 0-d array


def _np_pair_block(B: VectorBlock, w: Covector) -> Coords:
    return B.reshape(-1, B.shape[-1]).T @ w.ravel()


def _np_vector_block(v: Vector) -> VectorBlock:
    return VectorBlock(v[..., None])


def _np_covector_block(w: Covector) -> CovectorBlock:
    return CovectorBlock(w[..., None])


def _np_append_vector(B: VectorBlock, v: Vector) -> VectorBlock:
    return VectorBlock(np.concatenate((B, v[..., None]), axis=-1))


def _np_append_covector(C: CovectorBlock, w: Covector) -> CovectorBlock:
    return CovectorBlock(np.concatenate((C, w[..., None]), axis=-1))


def _np_combine(B: Array, c: Coords) -> Array:
    combined: Array = (B.reshape(-1, B.shape[-1]) @ c).reshape(B.shape[:-1])
    return combined


def _np_combine_vectors(B: VectorBlock, c: Coords) -> Vector:
    return Vector(_np_combine(B, c))


def _np_combine_covectors(C: CovectorBlock, c: Coords) -> Covector:
    return Covector(_np_combine(C, c))


def _np_dimension(v: Vector) -> int:
    return int(v.size)


numpy_space: NumpySpace = VectorSpace(
    scale_vector=_np_scale_vector, scale_covector=_np_scale_covector,
    axpy_vector=_np_axpy_vector, axpy_covector=_np_axpy_covector,
    pair=_np_pair, pair_block=_np_pair_block,
    vector_block=_np_vector_block, covector_block=_np_covector_block,
    append_vector=_np_append_vector, append_covector=_np_append_covector,
    combine_vectors=_np_combine_vectors, combine_covectors=_np_combine_covectors,
    dimension=_np_dimension)


def numpy_space_with_reductions(allreduce: Callable[[Coords], Coords]) -> NumpySpace:
    """numpy_space for data stored as one local numpy array per process
    (e.g. MPI ranks).

    Local operations stay numpy; the reductions (pair, pair_block, dimension)
    are completed with allreduce, which must return the elementwise sum of its
    argument over all processes, identical on every process. Every decision
    the method makes comes from reduced numbers, so the processes stay in
    lockstep. Operators (metric_solve, precond_solve, hessvec) act on the
    local arrays, with whatever neighbor communication they need; the
    pairings that turn their results into numbers are reduced here.
    """
    def pair(w: Covector, v: Vector) -> float:
        return cast(float, allreduce(np.array([_np_pair(w, v)]))[0])

    def pair_block(B: VectorBlock, w: Covector) -> Coords:
        return allreduce(_np_pair_block(B, w))

    def dimension(v: Vector) -> int:
        return int(round(float(allreduce(np.array([float(v.size)]))[0])))

    return dataclasses.replace(numpy_space, pair=pair, pair_block=pair_block, dimension=dimension)


# --- tuple trees ---------------------------------------------------------------


BlockT = TypeVar("BlockT")


@dataclass(frozen=True)
class LeafBlock(Generic[BlockT]):
    """A leaf of a block of tuple trees: a block of the leaf space, wrapped so
    that a leaf space whose blocks are tuples (tuple_block_space) cannot be
    mistaken for more tree."""
    block: BlockT


TreeNumpySpace = VectorSpace[Tree[Vector], Tree[Covector],
                             Tree[LeafBlock[VectorBlock]], Tree[LeafBlock[CovectorBlock]]]


@overload
def tuple_tree_space() -> TreeNumpySpace: ...


@overload
def tuple_tree_space(leaf_space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT]
                     ) -> VectorSpace[Tree[VectorT], Tree[CovectorT], Tree[LeafBlock[VectorBlockT]],
                                      Tree[LeafBlock[CovectorBlockT]]]: ...


def tuple_tree_space(leaf_space: Any = None) -> Any:
    """Vectors and covectors as tuple trees (see trees) whose leaves belong
    to leaf_space (default numpy_space, so leaves are numpy arrays of any
    shape), e.g. ((a, b), c) with arrays a, b, c of different shapes. The
    vectors and covectors combined in one operation must share a structure.

    The operations work leaf by leaf with the leaf space's functions:
    scaling and axpy leafwise, pairings as the sum of the leaves' pairings,
    and a block of vectors as a tree of the same structure whose leaves are
    leaf-space blocks (each wrapped in a LeafBlock), so the leaf space's
    batched block operations are used (e.g. one matrix product per numpy
    leaf). dimension, when the leaf space has one, is the sum over the
    leaves. With leaf_space = numpy_space_with_reductions(allreduce) the
    trees are distributed (one reduction per leaf per pairing).

    Trees are tuples only: a list raises (convert with as_tuple_tree).
    """
    leaf = numpy_space if leaf_space is None else leaf_space

    def pair(w: Any, v: Any) -> float:
        pairs = [leaf.pair(wl, vl) for wl, vl in tree_leaf_pairs(w, v)]
        total = pairs[0]
        for p in pairs[1:]:
            total = total + p
        return total

    def pair_block(B: Any, w: Any) -> Coords:
        total: Optional[Coords] = None
        for Bl, wl in tree_leaf_pairs(B, w):
            p = leaf.pair_block(Bl.block, wl)
            total = p if total is None else total + p
        if total is None:
            raise ValueError("a tuple tree without leaves")
        return total

    def combine_vectors(B: Any, c: Coords) -> Any:
        return tree_map(lambda Bl: leaf.combine_vectors(Bl.block, c), B)

    def combine_covectors(C: Any, c: Coords) -> Any:
        return tree_map(lambda Cl: leaf.combine_covectors(Cl.block, c), C)

    dimension = None
    if leaf.dimension is not None:
        leaf_dimension = leaf.dimension

        def dimension(v: Any) -> int:
            return sum(leaf_dimension(vl) for vl in tree_leaves(v))

    return VectorSpace(
        scale_vector=lambda a, v: tree_map(lambda vl: leaf.scale_vector(a, vl), v),
        scale_covector=lambda a, w: tree_map(lambda wl: leaf.scale_covector(a, wl), w),
        axpy_vector=lambda a, v, z: tree_map2(lambda vl, zl: leaf.axpy_vector(a, vl, zl), v, z),
        axpy_covector=lambda a, w, z: tree_map2(lambda wl, zl: leaf.axpy_covector(a, wl, zl), w, z),
        pair=pair, pair_block=pair_block,
        vector_block=lambda v: tree_map(lambda vl: LeafBlock(leaf.vector_block(vl)), v),
        covector_block=lambda w: tree_map(lambda wl: LeafBlock(leaf.covector_block(wl)), w),
        append_vector=lambda B, v: tree_map2(
            lambda Bl, vl: LeafBlock(leaf.append_vector(Bl.block, vl)), B, v),
        append_covector=lambda C, w: tree_map2(
            lambda Cl, wl: LeafBlock(leaf.append_covector(Cl.block, wl)), C, w),
        combine_vectors=combine_vectors, combine_covectors=combine_covectors,
        dimension=dimension)


def tuple_block_space(*, scale_vector: Callable[[float, VectorT], VectorT],
                      scale_covector: Callable[[float, CovectorT], CovectorT],
                      axpy_vector: Callable[[float, VectorT, VectorT], VectorT],
                      axpy_covector: Callable[[float, CovectorT, CovectorT], CovectorT],
                      pair: Callable[[CovectorT, VectorT], float],
                      dimension: Optional[Callable[[VectorT], int]] = None
                      ) -> VectorSpace[VectorT, CovectorT, tuple[VectorT, ...], tuple[CovectorT, ...]]:
    """A complete space for any vector type, from five functions (and
    optionally dimension).

    Blocks are tuples of vectors (or covectors), and the block entries are
    built from the functions given: pair_block pairs one vector at a time and
    combine_* accumulate with scale and axpy. That is correct for any
    representation; for distributed objects it costs one reduction per block
    member, so override pair_block (and combine_*) with dataclasses.replace
    when a batched version is available.
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
        dimension=dimension)


# --- points ---------------------------------------------------------------------


def flat_retraction(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT]
                    ) -> Callable[[Any, VectorT], Any]:
    """The retraction x, s -> x + s of a flat space, for points stored like
    vectors (numpy arrays, tuple trees of them, distributed arrays): computed
    with the space's axpy_vector."""
    return lambda x, s: space.axpy_vector(1.0, s, x)


def check_space(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
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
