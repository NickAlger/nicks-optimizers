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
                                       blocks are tuples of vectors (numpy
                                       backend only)
  array_space(backend)                 one array per vector, for any backend
                                       (see backend): blocks are arrays of
                                       fixed capacity, as compiled code needs
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
  blocks     collections of vectors or covectors (Krylov bases), with room
             for a fixed number of members, their capacity: members not set
             yet are zero.
  Coords     small coordinate arrays (numpy, or the backend's arrays).

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
    numpy_space_with_reductions, tuple_tree_space, tuple_block_space,
    array_space) and change entries with dataclasses.replace. Every entry is
    required except dimension.

      scale_vector(a, v) -> Vector           a v
      scale_covector(a, w) -> Covector       a w
      axpy_vector(a, v, z) -> Vector         a v + z
      axpy_covector(a, w, z) -> Covector     a w + z
      pair(w, v) -> Scalar                   the duality pairing <w, v>: a 0-d
                                             array (a numpy scalar for numpy)

    Blocks (Krylov bases) have room for a fixed number of members, their
    capacity, and members not set yet are zero. k, the number of members
    in use (the first k), lets an implementation skip the rest; j and k may
    be computed values under a compiling backend.

      vector_block(v, capacity) -> VectorBlock       v, then zeros
      covector_block(w, capacity) -> CovectorBlock
      set_vector(B, j, v) -> VectorBlock             B with member j replaced by v
      set_covector(C, j, w) -> CovectorBlock
      pair_block(B, w, k) -> Coords                  (<w, b_i>)_i, of length
                                                     capacity, zero for i >= k
      combine_vectors(B, c, k) -> Vector             sum_{i < k} c_i b_i (c of length
                                                     capacity, or shorter, >= k)
      combine_covectors(C, c, k) -> Covector         sum_{i < k} c_i w_i
      grow_vector_block(B, capacity) -> VectorBlock  the same members, more room
      grow_covector_block(C, capacity) -> CovectorBlock
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
    vector_block: Callable[[VectorT, int], VectorBlockT]
    covector_block: Callable[[CovectorT, int], CovectorBlockT]
    set_vector: Callable[[VectorBlockT, Any, VectorT], VectorBlockT]
    set_covector: Callable[[CovectorBlockT, Any, CovectorT], CovectorBlockT]
    pair_block: Callable[[VectorBlockT, CovectorT, Any], Coords]
    combine_vectors: Callable[[VectorBlockT, Coords, Any], VectorT]
    combine_covectors: Callable[[CovectorBlockT, Coords, Any], CovectorT]
    grow_vector_block: Callable[[VectorBlockT, int], VectorBlockT]
    grow_covector_block: Callable[[CovectorBlockT, int], CovectorBlockT]
    dimension: Optional[Callable[[VectorT], int]] = None


# --- numpy representation ---------------------------------------------------


class _Rows:
    """The storage of NumpyBlocks: an array of shape (capacity,) + the shape
    of a vector, and how many of its rows have been written (the high-water
    mark), so that blocks sharing it can tell whether a row is free."""
    __slots__ = ("array", "filled")

    def __init__(self, array: Array, filled: int) -> None:
        self.array = array
        self.filled = filled


@dataclass(frozen=True, eq=False)
class NumpyBlock:
    """A block of numpy_space: its first `count` members, along the leading
    axis of a preallocated array (room for the capacity), the rest zero.

    Setting the next member writes it in place when no other block on the
    same storage has written that row, so building a basis copies nothing;
    otherwise the members are copied first, and the block it came from is
    unchanged either way (blocks are values). Memory is that of the capacity,
    committed by the operating system as rows are written (np.zeros), and the
    method keeps the capacity within a factor of two of the members in use
    (see trust_region_newton's krylov_capacity)."""
    storage: _Rows
    count: int

    @property
    def capacity(self) -> int:
        return int(self.storage.array.shape[0])

    @property
    def members(self) -> Array:
        """The members set, stacked along the leading axis (a view)."""
        return self.storage.array[:self.count]


Point = NewType("Point", Array)
Vector = NewType("Vector", Array)
Covector = NewType("Covector", Array)
VectorBlock = NewType("VectorBlock", NumpyBlock)
CovectorBlock = NewType("CovectorBlock", NumpyBlock)
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


# Vectors may have any shape; a block stores its members along a leading
# axis. Products flatten the vector axes, so an array of any shape computes
# exactly what its flattened 1-D version would.


def _np_pair(w: Covector, v: Vector) -> float:
    return cast(float, w.ravel() @ v.ravel())        # a numpy scalar, or a 0-d array


def _np_block(v: Array, capacity: int) -> NumpyBlock:
    dtype = v.dtype if np.issubdtype(v.dtype, np.inexact) else np.float64
    array = np.zeros((capacity,) + v.shape, dtype=dtype)
    array[0] = v
    return NumpyBlock(_Rows(array, 1), 1)


def _np_set(B: NumpyBlock, j: Any, v: Array) -> NumpyBlock:
    j = int(j)
    if not 0 <= j < B.capacity:
        raise IndexError(f"member {j} of a block of capacity {B.capacity}")
    rows = B.storage
    if j >= B.count and rows.filled <= B.count:     # rows from B.count on are free (zero)
        rows.array[j] = v
        rows.filled = j + 1
        return NumpyBlock(rows, j + 1)
    array = np.zeros_like(rows.array)               # in use by another block: copy
    array[:B.count] = rows.array[:B.count]
    array[j] = v
    count = max(B.count, j + 1)
    return NumpyBlock(_Rows(array, count), count)


def _np_active(B: NumpyBlock, k: Any) -> Array:
    """The first k members (k is at most the capacity; members beyond those
    set are zero, so they are left out)."""
    m = min(int(k), B.count)
    active: Array = B.storage.array[:m]
    return active.reshape(m, -1)


def _np_pair_block(B: NumpyBlock, w: Covector, k: Any) -> Coords:
    active = _np_active(B, k)
    out: Coords = np.zeros(B.capacity)
    out[:active.shape[0]] = active @ w.ravel()
    return out


def _np_combine(B: NumpyBlock, c: Coords, k: Any) -> Array:
    active = _np_active(B, k)
    combined: Array = (c[:active.shape[0]] @ active).reshape(B.storage.array.shape[1:])
    return combined


def _np_grow(B: NumpyBlock, capacity: int) -> NumpyBlock:
    if capacity == B.capacity:
        return B
    array = np.zeros((capacity,) + B.storage.array.shape[1:], dtype=B.storage.array.dtype)
    array[:B.count] = B.storage.array[:B.count]
    return NumpyBlock(_Rows(array, B.count), B.count)


def _np_dimension(v: Vector) -> int:
    return int(v.size)


numpy_space: NumpySpace = VectorSpace(
    scale_vector=_np_scale_vector, scale_covector=_np_scale_covector,
    axpy_vector=_np_axpy_vector, axpy_covector=_np_axpy_covector, pair=_np_pair,
    vector_block=cast(Any, _np_block), covector_block=cast(Any, _np_block),
    set_vector=cast(Any, _np_set), set_covector=cast(Any, _np_set),
    pair_block=cast(Any, _np_pair_block),
    combine_vectors=cast(Any, _np_combine), combine_covectors=cast(Any, _np_combine),
    grow_vector_block=cast(Any, _np_grow), grow_covector_block=cast(Any, _np_grow),
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

    def pair_block(B: VectorBlock, w: Covector, k: Any) -> Coords:
        return allreduce(_np_pair_block(B, w, k))

    def dimension(v: Vector) -> int:
        return int(round(float(allreduce(np.array([float(v.size)]))[0])))

    return dataclasses.replace(numpy_space, pair=pair, pair_block=pair_block, dimension=dimension)


# --- tuple trees ---------------------------------------------------------------


TreeNumpySpace = VectorSpace[Tree[Vector], Tree[Covector], Tree[NumpyBlock], Tree[NumpyBlock]]


@overload
def tuple_tree_space() -> TreeNumpySpace: ...


@overload
def tuple_tree_space(leaf_space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT]
                     ) -> VectorSpace[Tree[VectorT], Tree[CovectorT], Tree[VectorBlockT],
                                      Tree[CovectorBlockT]]: ...


def tuple_tree_space(leaf_space: Any = None) -> Any:
    """Vectors and covectors as tuple trees (see trees) whose leaves belong
    to leaf_space (default numpy_space, so leaves are numpy arrays of any
    shape), e.g. ((a, b), c) with arrays a, b, c of different shapes. The
    vectors and covectors combined in one operation must share a structure.

    The operations work leaf by leaf with the leaf space's functions:
    scaling and axpy leafwise, pairings as the sum of the leaves' pairings,
    and a block of vectors as a tree of the same structure whose leaves are
    leaf-space blocks, so the leaf space's batched block operations are used
    (e.g. one matrix product per numpy leaf). Leaf-space blocks must not be
    tuples, so that they cannot be mistaken for more tree (the blocks of
    numpy_space, array_space and tuple_block_space are not). dimension, when
    the leaf space has one, is the sum over the leaves. With leaf_space =
    numpy_space_with_reductions(allreduce) the trees are distributed (one
    reduction per leaf per pairing); with leaf_space = array_space(backend)
    they work with a compiling backend.

    Trees are tuples only: a list raises (convert with as_tuple_tree).
    """
    leaf = numpy_space if leaf_space is None else leaf_space

    def pair(w: Any, v: Any) -> float:
        pairs = [leaf.pair(wl, vl) for wl, vl in tree_leaf_pairs(w, v)]
        total = pairs[0]
        for p in pairs[1:]:
            total = total + p
        return total

    def pair_block(B: Any, w: Any, k: Any) -> Coords:
        total: Optional[Coords] = None
        for Bl, wl in tree_leaf_pairs(B, w):
            p = leaf.pair_block(Bl, wl, k)
            total = p if total is None else total + p
        if total is None:
            raise ValueError("a tuple tree without leaves")
        return total

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
        pair=pair,
        vector_block=lambda v, cap: tree_map(lambda vl: leaf.vector_block(vl, cap), v),
        covector_block=lambda w, cap: tree_map(lambda wl: leaf.covector_block(wl, cap), w),
        set_vector=lambda B, j, v: tree_map2(lambda Bl, vl: leaf.set_vector(Bl, j, vl), B, v),
        set_covector=lambda C, j, w: tree_map2(lambda Cl, wl: leaf.set_covector(Cl, j, wl), C, w),
        pair_block=pair_block,
        combine_vectors=lambda B, c, k: tree_map(lambda Bl: leaf.combine_vectors(Bl, c, k), B),
        combine_covectors=lambda C, c, k: tree_map(lambda Cl: leaf.combine_covectors(Cl, c, k), C),
        grow_vector_block=lambda B, cap: tree_map(lambda Bl: leaf.grow_vector_block(Bl, cap), B),
        grow_covector_block=lambda C, cap: tree_map(lambda Cl: leaf.grow_covector_block(Cl, cap), C),
        dimension=dimension)


MemberT = TypeVar("MemberT")


@dataclass(frozen=True)
class TupleBlock(Generic[MemberT]):
    """A block of tuple_block_space: the members set so far, as a tuple, and
    the capacity. Python-only: it cannot be indexed with a computed j, so
    tuple_block_space works with the numpy backend only."""
    members: tuple[MemberT, ...]
    capacity: int


def tuple_block_space(*, scale_vector: Callable[[float, VectorT], VectorT],
                      scale_covector: Callable[[float, CovectorT], CovectorT],
                      axpy_vector: Callable[[float, VectorT, VectorT], VectorT],
                      axpy_covector: Callable[[float, CovectorT, CovectorT], CovectorT],
                      pair: Callable[[CovectorT, VectorT], float],
                      dimension: Optional[Callable[[VectorT], int]] = None
                      ) -> VectorSpace[VectorT, CovectorT, TupleBlock[VectorT], TupleBlock[CovectorT]]:
    """A complete space for any vector type, from five functions (and
    optionally dimension).

    Blocks are TupleBlocks (tuples of the members set so far), and the block
    entries are built from the functions given: pair_block pairs one vector
    at a time and combine_* accumulate with scale and axpy. That is correct
    for any representation; for distributed objects it costs one reduction
    per block member, so override pair_block (and combine_*) with
    dataclasses.replace when a batched version is available. For the numpy
    backend only (see TupleBlock).
    """
    def block(v: Any, capacity: int) -> TupleBlock[Any]:
        return TupleBlock((v,), capacity)

    def set_member(B: TupleBlock[Any], j: Any, v: Any) -> TupleBlock[Any]:
        j = int(j)
        m = len(B.members)
        if not 0 <= j <= m or j >= B.capacity:
            raise IndexError(f"member {j} of a block with {m} members and capacity {B.capacity}")
        return TupleBlock(B.members[:j] + (v,) + B.members[j + 1:], B.capacity)

    def pair_block(B: TupleBlock[VectorT], w: CovectorT, k: Any) -> Coords:
        out: Coords = np.zeros(B.capacity)
        out[:int(k)] = np.array([pair(w, b) for b in B.members[:int(k)]], dtype=float)
        return out

    def combine(scale: Callable[[float, Any], Any], axpy: Callable[[float, Any, Any], Any]
                ) -> Callable[[TupleBlock[Any], Coords, Any], Any]:
        def combined(B: TupleBlock[Any], c: Coords, k: Any) -> Any:
            members = B.members[:int(k)]
            acc = scale(float(c[0]), members[0])
            for ci, bi in zip(c[1:], members[1:]):
                acc = axpy(float(ci), bi, acc)
            return acc
        return combined

    def grow(B: TupleBlock[Any], capacity: int) -> TupleBlock[Any]:
        return TupleBlock(B.members, capacity)

    return VectorSpace(
        scale_vector=scale_vector, scale_covector=scale_covector,
        axpy_vector=axpy_vector, axpy_covector=axpy_covector, pair=pair,
        vector_block=block, covector_block=block, set_vector=set_member, set_covector=set_member,
        pair_block=pair_block, combine_vectors=combine(scale_vector, axpy_vector),
        combine_covectors=combine(scale_covector, axpy_covector),
        grow_vector_block=grow, grow_covector_block=grow, dimension=dimension)


def array_space(backend: Any) -> VectorSpace[Any, Any, Any, Any]:
    """One array (of any shape) per vector and covector, computed through
    backend.xp: the space for compiling backends (see backend), and for the
    numpy backend too.

    A block is one array of shape (capacity,) + the shape of a vector,
    allocated in full (compiled code has fixed shapes): memory is the
    capacity, not the members in use. Block products run over all the
    members, with the coefficients of members at and beyond k masked to zero.
    """
    xp, put = backend.xp, backend.index_update

    def pair(w: Any, v: Any) -> Any:
        return xp.sum(w * v)

    def block(v: Any, capacity: int) -> Any:
        return put(xp.zeros((capacity,) + tuple(v.shape), dtype=v.dtype), 0, v)

    def set_member(B: Any, j: Any, v: Any) -> Any:
        return put(B, j, v)

    def rows(B: Any) -> Any:
        return xp.reshape(B, (B.shape[0], -1))

    def active(c: Any, k: Any) -> Any:
        return xp.where(xp.arange(c.shape[0]) < k, c, 0.0)

    def pair_block(B: Any, w: Any, k: Any) -> Any:
        return active(xp.matmul(rows(B), xp.reshape(w, (-1,))), k)

    def combine(B: Any, c: Any, k: Any) -> Any:
        n = c.shape[0]                    # c may be shorter than the capacity (static)
        return xp.reshape(xp.matmul(active(c, k), rows(B)[:n]), tuple(B.shape[1:]))

    def grow(B: Any, capacity: int) -> Any:
        return put(xp.zeros((capacity,) + tuple(B.shape[1:]), dtype=B.dtype),
                   slice(0, B.shape[0]), B)

    return VectorSpace(
        scale_vector=lambda a, v: a * v, scale_covector=lambda a, w: a * w,
        axpy_vector=lambda a, v, z: a * v + z, axpy_covector=lambda a, w, z: a * w + z,
        pair=pair, vector_block=block, covector_block=block, set_vector=set_member,
        set_covector=set_member, pair_block=pair_block, combine_vectors=combine,
        combine_covectors=combine, grow_vector_block=grow, grow_covector_block=grow,
        dimension=lambda v: int(v.size))


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
    (so it works for any representation), including the block entries (a
    block of capacity 3 with two members, grown to 4, and members beyond k
    ignored). Expect rounding level, ~1e-15."""
    def pair(w: CovectorT, v: VectorT) -> float:
        return float(np.asarray(space.pair(w, v)))

    a, b = 0.7, -1.3
    c = np.array([a, b])
    p11, p12, p21, p22 = pair(w1, v1), pair(w1, v2), pair(w2, v1), pair(w2, v2)
    VB = space.set_vector(space.vector_block(v1, 3), 1, v2)
    CB = space.set_covector(space.covector_block(w1, 3), 1, w2)
    grown = space.grow_vector_block(VB, 4)
    errors = [
        float(np.max(np.abs(np.asarray(space.pair_block(VB, w1, 2)) - np.array([p11, p12, 0.0])))),
        float(np.max(np.abs(np.asarray(space.pair_block(grown, w1, 2)) - np.array([p11, p12, 0.0, 0.0])))),
        abs(float(np.asarray(space.pair_block(space.vector_block(v2, 1), w2, 1))[0]) - p22),
        abs(float(np.asarray(space.pair_block(VB, w1, 1))[1])),
        abs(pair(w1, space.combine_vectors(VB, c, 2)) - (a * p11 + b * p12)),
        abs(pair(w1, space.combine_vectors(VB, np.array([a, b, 5.0]), 2)) - (a * p11 + b * p12)),
        abs(pair(space.combine_covectors(CB, c, 2), v1) - (a * p11 + b * p21)),
        abs(pair(w1, space.scale_vector(a, v1)) - a * p11),
        abs(pair(space.scale_covector(a, w1), v1) - a * p11),
        abs(pair(space.axpy_covector(a, w1, w2), v1) - (a * p11 + p21)),
        abs(pair(w1, space.axpy_vector(a, v1, v2)) - (a * p11 + p12)),
    ]
    scale = max(abs(p11), abs(p12), abs(p21), abs(p22), 1e-300)
    return max(errors) / scale
