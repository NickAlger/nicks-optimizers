"""Inner products and norms: the geometry, kept separate from representation.

A metric M is given by its inner product on vectors, (u, v)_M, and the dual
inner product on covectors, (psi, phi)_{M^{-1}}; the operators M and M^{-1}
are never needed as maps. Inner products take a block first and return one
number per block member, so an implementation can apply M once per call
(e.g. lambda x, U, v: U.T @ (M @ v)).

The two inner products of a metric must be exact duals,
(M u, M v)_{M^{-1}} = (u, v)_M. Gradient norms are dual norms,
||g||_{M^{-1}}, which makes stopping tests invariant under changes of
variables. euclidean_inner and euclidean_dual_inner are such a pair for
numpy_space and M = I.
"""

from typing import Callable

import numpy as np

from .dense import grow_gram
from .spaces import (
    Coords,
    Covector,
    CovectorBlock,
    CovectorBlockT,
    CovectorT,
    Point,
    PointT,
    Vector,
    VectorBlock,
    VectorBlockT,
    VectorSpace,
    VectorT,
)


# Inner products for numpy_space: point-dependent, and frozen at one point.
InnerProduct = Callable[[Point, VectorBlock, Vector], Coords]
DualInnerProduct = Callable[[Point, CovectorBlock, Covector], Coords]
FrozenInnerProduct = Callable[[VectorBlock, Vector], Coords]
FrozenDualInnerProduct = Callable[[CovectorBlock, Covector], Coords]


def euclidean_inner(x: Point, U: VectorBlock, v: Vector) -> Coords:
    """(u, v)_M with M = I, for numpy_space. Pass it together with
    euclidean_dual_inner, its exact dual, to get the round trust region."""
    return U.T @ v


def euclidean_dual_inner(x: Point, Psi: CovectorBlock, phi: Covector) -> Coords:
    """(psi, phi)_{M^{-1}} with M = I: the exact dual of euclidean_inner."""
    return Psi.T @ phi


def vector_norm(space: VectorSpace[PointT, VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                inner: Callable[[VectorBlockT, VectorT], Coords], v: VectorT) -> float:
    """||v||_M = sqrt((v, v)_M)."""
    return float(np.sqrt(max(float(inner(space.vector_block(v), v)[0]), 0.0)))


def covector_norm(space: VectorSpace[PointT, VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                  dual_inner: Callable[[CovectorBlockT, CovectorT], Coords], w: CovectorT) -> float:
    """||w||_{M^{-1}} = sqrt((w, w)_{M^{-1}}). This is the dual norm of the
    M-norm: the largest <w, s> over steps with ||s||_M <= 1."""
    return float(np.sqrt(max(float(dual_inner(space.covector_block(w), w)[0]), 0.0)))


def extend_metric_gram(space: VectorSpace[PointT, VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                       G: Coords, Q: VectorBlockT, q_new: VectorT,
                       inner: Callable[[VectorBlockT, VectorT], Coords]) -> Coords:
    """Grow G = [(q_i, q_j)_M] by the row and column for q_new (one call)."""
    return grow_gram(G, inner(space.append_vector(Q, q_new), q_new))


def extend_dual_gram(space: VectorSpace[PointT, VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                     W: Coords, V: CovectorBlockT, v_new: CovectorT,
                     dual_inner: Callable[[CovectorBlockT, CovectorT], Coords]) -> Coords:
    """Grow W = [(v_i, v_j)_{M^{-1}}] by the row and column for v_new (one call)."""
    return grow_gram(W, dual_inner(space.append_covector(V, v_new), v_new))
