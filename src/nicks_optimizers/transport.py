"""Vector transports: how vectors and covectors move from the tangent space
at one point to the tangent space at another.

A transport is part of the point structure, given separately like the
retraction (see spaces). Methods that carry vectors from one iterate to the
next (lbfgs, whose memory of steps and gradient changes must live in the
current tangent space) need one; the Newton methods never move a vector
between points and take none.

  VectorTransport             a frozen record of the maps
      vector(x, x_new, v) -> Vector         T_{x -> x_new} v
      covector(x, x_new, w) -> Covector     Tb_{x -> x_new} w
      vector_block(x, x_new, B) -> VectorBlock        optional: every member
      covector_block(x, x_new, C) -> CovectorBlock    of a block at once
  identity_transport          the identity, for flat spaces (points and
                              vectors in one vector space): nothing is called
                              and nothing is copied
  riesz_covector_transport(vector, metric_solve, metric_apply)
                              a transport whose covector map goes through the
                              metric, Tb = M(x_new) T M(x)^{-1}

Both maps must be linear. The ideal pair preserves pairings,
<Tb w, T v> = <w, v>: then a bilinear form B moves exactly, T(B)(T u, T v) =
B(u, v). An M-isometric T with Tb = M(x_new) T M(x)^{-1} is one such pair.
Projections onto the new tangent space (the usual transport on embedded
manifolds, with the same formula for canonical covector representatives)
preserve pairings only approximately, to second order in the distance. On a
flat space the identity is right even when the metric M(x) changes: the
stored pairs are data about the Hessian, which do not depend on M.

Without the block maps, a block is transported member by member (through the
space's vector_member and set_vector). The block maps act on the space's own
block representation, e.g. one batched projection of an array_space block;
members that are not set are zero, and stay zero under a linear map.
"""

from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional

from .spaces import CovectorBlockT, CovectorT, PointT, VectorBlockT, VectorT


@dataclass(frozen=True, kw_only=True)
class VectorTransport(Generic[PointT, VectorT, CovectorT, VectorBlockT, CovectorBlockT]):
    """The maps of a vector transport (see the module docstring)."""
    vector: Callable[[PointT, PointT, VectorT], VectorT]
    covector: Callable[[PointT, PointT, CovectorT], CovectorT]
    vector_block: Optional[Callable[[PointT, PointT, VectorBlockT], VectorBlockT]] = None
    covector_block: Optional[Callable[[PointT, PointT, CovectorBlockT], CovectorBlockT]] = None


def _same(x: Any, x_new: Any, v: Any) -> Any:
    return v


identity_transport: VectorTransport[Any, Any, Any, Any, Any] = VectorTransport(
    vector=_same, covector=_same, vector_block=_same, covector_block=_same)


def riesz_covector_transport(
        vector: Callable[[PointT, PointT, VectorT], VectorT],
        metric_solve: Callable[[PointT, CovectorT], VectorT],
        metric_apply: Callable[[PointT, VectorT], CovectorT],
        vector_block: Optional[Callable[[PointT, PointT, VectorBlockT], VectorBlockT]] = None,
        ) -> VectorTransport[PointT, VectorT, CovectorT, VectorBlockT, CovectorBlockT]:
    """The transport with vector map T and covector map
    Tb w = M(x_new) T M(x)^{-1} w: the Riesz representative of w, transported
    as a vector, and turned back into a covector. It preserves pairings when
    T is isometric from M(x) to M(x_new). Each covector costs a solve with M
    and a product with M (covectors are transported member by member)."""
    def covector(x: PointT, x_new: PointT, w: CovectorT) -> CovectorT:
        return metric_apply(x_new, vector(x, x_new, metric_solve(x, w)))
    return VectorTransport(vector=vector, covector=covector, vector_block=vector_block)
