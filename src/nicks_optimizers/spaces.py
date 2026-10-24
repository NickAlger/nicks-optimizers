"""Representation of points, vectors and covectors.

Optimizers in this package never touch the storage of their objects. They
work through a VectorSpace record: a frozen dataclass of functions (no
methods) saying how vectors are scaled, paired, collected into blocks and
combined. numpy_space, the default, uses plain numpy arrays; other
representations (distributed objects, collections of arrays, GPU arrays, ...)
supply their own record:

  numpy_space                          one numpy array (of any shape) per
                                       vector
  numpy_space_with_reductions(allreduce)  one local numpy array per process
                                       (e.g. MPI ranks); only the reductions
                                       change
  tuple_tree_space(leaf_space)         nested tuples whose leaves are vectors
                                       of another space (numpy arrays by
                                       default); lists are accepted on entry
  tuple_block_space(...)               any vector type, from five functions;
                                       blocks are tuples of vectors
  dataclasses.replace(space, ...)      override individual entries

check_space(space, ...) verifies a record's internal consistency on sample
vectors. Vector and covector operations are separate fields on purpose, so a
space spells out which kind of object every function acts on.

Geometry. A space may carry a default metric, the fixed Hilbert structure of
its representation: metric_solve (M^{-1}) and metric_apply (M), used when the
optimizer is given no metric of its own. numpy_space has M = I; a tuple tree
has the direct sum of its leaves' metrics, which direct_sum_operator lets
you change leaf by leaf; tuple_block_space has one only if you give it. A
metric passed to the optimizer (point-dependent, see metrics) always takes
precedence.

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
from typing import Any, Callable, Generic, Iterator, NewType, Optional, TypeVar, Union, overload

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
    numpy_space_with_reductions, tuple_tree_space, tuple_block_space) and
    change entries with dataclasses.replace. Every entry is required except
    translate, dimension, the default metric and the normalize_* entries.

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
      metric_solve(w) -> Vector              optional: the default metric, M^{-1} w
      metric_apply(v) -> Covector            and M v, of the representation (fixed,
                                             not point-dependent), used when the
                                             optimizer is given no metric; the
                                             two must be inverse to each other
      normalize_point(x) -> Point            optional: the space's own form of a
      normalize_vector(v) -> Vector          point, vector or covector supplied
      normalize_covector(w) -> Covector      by the user (x0, g0, and the results
                                             of the user's functions), applied
                                             once where they enter the
                                             optimizer; e.g. lists to tuples for
                                             tuple_tree_space. None: used as is

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
    metric_solve: Optional[Callable[[CovectorT], VectorT]] = None
    metric_apply: Optional[Callable[[VectorT], CovectorT]] = None
    normalize_point: Optional[Callable[[PointT], PointT]] = None
    normalize_vector: Optional[Callable[[VectorT], VectorT]] = None
    normalize_covector: Optional[Callable[[CovectorT], CovectorT]] = None


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


# Vectors may have any shape; a block of k of them has the shape of a vector
# plus a trailing axis of length k (a matrix of columns for 1-D vectors).
# Products flatten the vector axes, so an array of any shape computes exactly
# what its flattened 1-D version would.


def _np_pair(w: Covector, v: Vector) -> float:
    return float(w.ravel() @ v.ravel())


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


def _np_translate(x: Point, s: Vector) -> Point:
    return Point(x + s)


def _np_dimension(v: Vector) -> int:
    return int(v.size)


def _np_metric_solve(w: Covector) -> Vector:
    """M = I: the same array, as a vector."""
    return Vector(w)


def _np_metric_apply(v: Vector) -> Covector:
    return Covector(v)


numpy_space: NumpySpace = VectorSpace(
    scale_vector=_np_scale_vector, scale_covector=_np_scale_covector,
    axpy_vector=_np_axpy_vector, axpy_covector=_np_axpy_covector,
    pair=_np_pair, pair_block=_np_pair_block,
    vector_block=_np_vector_block, covector_block=_np_covector_block,
    append_vector=_np_append_vector, append_covector=_np_append_covector,
    combine_vectors=_np_combine_vectors, combine_covectors=_np_combine_covectors,
    translate=_np_translate, dimension=_np_dimension,
    metric_solve=_np_metric_solve, metric_apply=_np_metric_apply)


def numpy_space_with_reductions(allreduce: Callable[[Coords], Coords]) -> NumpySpace:
    """numpy_space for data stored as one local numpy array per process
    (e.g. MPI ranks).

    Local operations stay numpy; the reductions (pair, pair_block, dimension)
    are completed with allreduce, which must return the elementwise sum of its
    argument over all processes, identical on every process. Every decision
    the method makes comes from reduced numbers, so the processes stay in
    lockstep. Operators (metric_solve, precond_solve, hessvec) act on the
    local arrays, with whatever neighbor communication they need; the
    pairings that turn their results into numbers are reduced here. The
    default metric is M = I, as for numpy_space (it acts locally).
    """
    def pair(w: Covector, v: Vector) -> float:
        return float(allreduce(np.array([_np_pair(w, v)]))[0])

    def pair_block(B: VectorBlock, w: Covector) -> Coords:
        return allreduce(_np_pair_block(B, w))

    def dimension(v: Vector) -> int:
        return int(round(float(allreduce(np.array([float(v.size)]))[0])))

    return dataclasses.replace(numpy_space, pair=pair, pair_block=pair_block, dimension=dimension)


# --- tuple trees ---------------------------------------------------------------


LeafT = TypeVar("LeafT")
Tree = Union[LeafT, tuple["Tree[LeafT]", ...]]
"""A tuple tree: a leaf, or a tuple of tuple trees. Anything that is not a
tuple is a leaf, so leaves must not be tuples or lists themselves."""


def as_tuple_tree(tree: Any) -> Any:
    """The same tree with every list turned into a tuple (recursively).
    Leaves are returned as they are, not copied. Named tuples become plain
    tuples."""
    if isinstance(tree, (tuple, list)):
        return tuple(as_tuple_tree(child) for child in tree)
    return tree


def _not_a_list(tree: Any) -> None:
    if isinstance(tree, list):
        raise TypeError("a tuple tree contains a list; convert user input with as_tuple_tree "
                        "(trust_region_newton does this for the objects it is given)")


def _tree_map(f: Callable[[Any], Any], tree: Any) -> Any:
    """f applied to every leaf; the result has the same tuple structure."""
    if isinstance(tree, tuple):
        return tuple(_tree_map(f, child) for child in tree)
    _not_a_list(tree)
    return f(tree)


def _leaf_pairs(a: Any, b: Any) -> Iterator[tuple[Any, Any]]:
    """The corresponding leaves of two trees, in order. Raises if the trees
    differ in structure."""
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b):
        for x, y in zip(a, b):
            yield from _leaf_pairs(x, y)
        return
    _not_a_list(a)
    _not_a_list(b)
    if isinstance(a, tuple) or isinstance(b, tuple):
        raise ValueError("tuple trees with different structures")
    yield a, b


def _tree_map2(f: Callable[[Any, Any], Any], a: Any, b: Any) -> Any:
    """f applied to corresponding leaves; the result has the structure of a."""
    if isinstance(a, tuple):
        if not (isinstance(b, tuple) and len(a) == len(b)):
            _not_a_list(b)
            raise ValueError("tuple trees with different structures")
        return tuple(_tree_map2(f, x, y) for x, y in zip(a, b))
    _not_a_list(a)
    _not_a_list(b)
    if isinstance(b, tuple):
        raise ValueError("tuple trees with different structures")
    return f(a, b)


def _tree_leaves(tree: Any) -> Iterator[Any]:
    if isinstance(tree, tuple):
        for child in tree:
            yield from _tree_leaves(child)
        return
    _not_a_list(tree)
    yield tree


BlockT = TypeVar("BlockT")


@dataclass(frozen=True)
class LeafBlock(Generic[BlockT]):
    """A leaf of a block of tuple trees: a block of the leaf space, wrapped so
    that a leaf space whose blocks are tuples (tuple_block_space) cannot be
    mistaken for more tree."""
    block: BlockT


def direct_sum_operator(operators: Any) -> Callable[[Any], Any]:
    """The operator on tuple trees that applies each leaf of `operators` (a
    tuple tree of functions, lists allowed) to the corresponding leaf of its
    argument: the direct sum of the leaf operators. For example, a metric
    that weights two leaves differently:

        space = dataclasses.replace(
            tuple_tree_space(),
            metric_solve=direct_sum_operator((lambda w: w / m1, lambda w: w / m2)),
            metric_apply=direct_sum_operator((lambda v: m1 * v, lambda v: m2 * v)))
    """
    ops = as_tuple_tree(operators)
    return lambda tree: _tree_map2(lambda op, leaf: op(leaf), ops, tree)


TreeNumpySpace = VectorSpace[Tree[Point], Tree[Vector], Tree[Covector],
                             Tree[LeafBlock[VectorBlock]], Tree[LeafBlock[CovectorBlock]]]


@overload
def tuple_tree_space() -> TreeNumpySpace: ...


@overload
def tuple_tree_space(leaf_space: VectorSpace[PointT, VectorT, CovectorT, VectorBlockT, CovectorBlockT]
                     ) -> VectorSpace[Tree[PointT], Tree[VectorT], Tree[CovectorT],
                                      Tree[LeafBlock[VectorBlockT]],
                                      Tree[LeafBlock[CovectorBlockT]]]: ...


def tuple_tree_space(leaf_space: Any = None) -> Any:
    """Points, vectors and covectors as tuple trees: nested tuples whose
    leaves belong to leaf_space (default numpy_space, so leaves are numpy
    arrays of any shape). For example ((a, b), c) with arrays a, b, c of
    different shapes. Vectors and covectors share one structure, and points
    have it too when they are translated (on a manifold, with retract,
    points can be any tree).

    Every operation works leaf by leaf with the leaf space's functions:
    pairings add up the leaves' pairings, and a block of vectors is a tree of
    the same structure whose leaves are leaf-space blocks (each wrapped in a
    LeafBlock), so the leaf space's batched block operations are used (e.g.
    one matrix product per numpy leaf). translate and dimension exist when
    the leaf space has them. With leaf_space =
    numpy_space_with_reductions(allreduce) the trees are distributed (one
    reduction per leaf per pairing).

    The default metric is the direct sum of the leaves' Hilbert spaces: M
    applies the leaf space's metric to every leaf (M = I for numpy leaves),
    when the leaf space has one. Override it as a whole, or leaf by leaf
    with direct_sum_operator, through dataclasses.replace(space,
    metric_solve=..., metric_apply=...); a metric passed to the optimizer
    overrides it too.

    Lists are accepted where user objects enter the optimizer: x0, g0 and the
    results of the user's functions go through as_tuple_tree (the
    normalize_* entries), so the method works with, and returns, tuples
    only. Inside, a list is an error. Leaves must not be tuples or lists:
    anything that is a tuple is taken for an inner node of the tree.
    """
    leaf = numpy_space if leaf_space is None else leaf_space

    def pair(w: Any, v: Any) -> float:
        return float(sum(leaf.pair(wl, vl) for wl, vl in _leaf_pairs(w, v)))

    def pair_block(B: Any, w: Any) -> Coords:
        total: Optional[Coords] = None
        for Bl, wl in _leaf_pairs(B, w):
            p = leaf.pair_block(Bl.block, wl)
            total = p if total is None else total + p
        if total is None:
            raise ValueError("a tuple tree without leaves")
        return total

    def combine_vectors(B: Any, c: Coords) -> Any:
        return _tree_map(lambda Bl: leaf.combine_vectors(Bl.block, c), B)

    def combine_covectors(C: Any, c: Coords) -> Any:
        return _tree_map(lambda Cl: leaf.combine_covectors(Cl.block, c), C)

    translate = None
    if leaf.translate is not None:
        leaf_translate = leaf.translate

        def translate(x: Any, s: Any) -> Any:
            return _tree_map2(leaf_translate, x, s)

    dimension = None
    if leaf.dimension is not None:
        leaf_dimension = leaf.dimension

        def dimension(v: Any) -> int:
            return sum(leaf_dimension(vl) for vl in _tree_leaves(v))

    metric_solve = None
    if leaf.metric_solve is not None:
        leaf_metric_solve = leaf.metric_solve

        def metric_solve(w: Any) -> Any:
            return _tree_map(leaf_metric_solve, w)

    metric_apply = None
    if leaf.metric_apply is not None:
        leaf_metric_apply = leaf.metric_apply

        def metric_apply(v: Any) -> Any:
            return _tree_map(leaf_metric_apply, v)

    return VectorSpace(
        scale_vector=lambda a, v: _tree_map(lambda vl: leaf.scale_vector(a, vl), v),
        scale_covector=lambda a, w: _tree_map(lambda wl: leaf.scale_covector(a, wl), w),
        axpy_vector=lambda a, v, z: _tree_map2(lambda vl, zl: leaf.axpy_vector(a, vl, zl), v, z),
        axpy_covector=lambda a, w, z: _tree_map2(lambda wl, zl: leaf.axpy_covector(a, wl, zl), w, z),
        pair=pair, pair_block=pair_block,
        vector_block=lambda v: _tree_map(lambda vl: LeafBlock(leaf.vector_block(vl)), v),
        covector_block=lambda w: _tree_map(lambda wl: LeafBlock(leaf.covector_block(wl)), w),
        append_vector=lambda B, v: _tree_map2(
            lambda Bl, vl: LeafBlock(leaf.append_vector(Bl.block, vl)), B, v),
        append_covector=lambda C, w: _tree_map2(
            lambda Cl, wl: LeafBlock(leaf.append_covector(Cl.block, wl)), C, w),
        combine_vectors=combine_vectors, combine_covectors=combine_covectors,
        translate=translate, dimension=dimension,
        metric_solve=metric_solve, metric_apply=metric_apply,
        normalize_point=as_tuple_tree, normalize_vector=as_tuple_tree,
        normalize_covector=as_tuple_tree)


def tuple_block_space(*, scale_vector: Callable[[float, VectorT], VectorT],
                      scale_covector: Callable[[float, CovectorT], CovectorT],
                      axpy_vector: Callable[[float, VectorT, VectorT], VectorT],
                      axpy_covector: Callable[[float, CovectorT, CovectorT], CovectorT],
                      pair: Callable[[CovectorT, VectorT], float],
                      translate: Optional[Callable[[PointT, VectorT], PointT]] = None,
                      dimension: Optional[Callable[[VectorT], int]] = None,
                      metric_solve: Optional[Callable[[CovectorT], VectorT]] = None,
                      metric_apply: Optional[Callable[[VectorT], CovectorT]] = None
                      ) -> VectorSpace[PointT, VectorT, CovectorT,
                                       tuple[VectorT, ...], tuple[CovectorT, ...]]:
    """A complete space for any vector type, from five functions (plus
    translate for a flat space, and optionally dimension and a default
    metric).

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
        translate=translate, dimension=dimension, metric_solve=metric_solve,
        metric_apply=metric_apply)


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
