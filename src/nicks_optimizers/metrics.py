"""The geometry: a metric M, given by operators.

Everything geometric is measured with a symmetric positive definite metric
M(x): the trust region { s : ||s||_M <= radius }, gradient norms
||g||_{M^{-1}} (the stopping test), curvature (M-Ritz values, for the
saddle-free model) and Krylov residuals. M is given by operators:

  metric_solve(w) -> Vector     M^{-1} w, the Riesz map; always used
  metric_apply(v) -> Covector   M v; needed only when a separate
                                preconditioner P is used

and the norms follow from the pairing: ||v||_M = sqrt(<M v, v>) and
||w||_{M^{-1}} = sqrt(<w, M^{-1} w>). When M doubles as the preconditioner
(P = M, the usual case), only M^{-1} is ever applied: any symmetric positive
definite operator passed as metric_solve, a multigrid cycle for instance,
defines M as its inverse, with nothing to keep consistent. With a separate
preconditioner, metric_apply and metric_solve must be inverse to each other.

Without a metric of its own, trust_region_newton uses the space's default
metric (VectorSpace.metric_solve and metric_apply, fixed rather than
point-dependent): M = I for numpy arrays, the direct sum of the leaves'
metrics for tuple trees. A metric passed to the optimizer overrides it, and
comes as a pair: metric_solve with metric_apply, never mixed with the
space's. euclidean_metric_solve and euclidean_metric_apply are M = I for
numpy arrays (also distributed ones: they act locally), written as the
point-dependent operators the optimizer takes.
"""

from typing import Callable

import numpy as np

from .dense import grow_gram
from .spaces import (
    Coords,
    Covector,
    CovectorBlockT,
    CovectorT,
    Point,
    PointT,
    Vector,
    VectorBlockT,
    VectorSpace,
    VectorT,
)


# Metric operators for numpy_space: point-dependent.
MetricSolve = Callable[[Point, Covector], Vector]
MetricApply = Callable[[Point, Vector], Covector]


def euclidean_metric_solve(x: Point, w: Covector) -> Vector:
    """M^{-1} w with M = I, for numpy arrays: the same array, as a vector."""
    return Vector(w)


def euclidean_metric_apply(x: Point, v: Vector) -> Covector:
    """M v with M = I, for numpy arrays: the same array, as a covector."""
    return Covector(v)


def vector_norm(space: VectorSpace[PointT, VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                metric_apply: Callable[[VectorT], CovectorT], v: VectorT) -> float:
    """||v||_M = sqrt(<M v, v>)."""
    return float(np.sqrt(max(space.pair(metric_apply(v), v), 0.0)))


def covector_norm(space: VectorSpace[PointT, VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                  metric_solve: Callable[[CovectorT], VectorT], w: CovectorT) -> float:
    """||w||_{M^{-1}} = sqrt(<w, M^{-1} w>), the dual norm of the M-norm: the
    largest <w, s> over steps with ||s||_M <= 1."""
    return float(np.sqrt(max(space.pair(w, metric_solve(w)), 0.0)))


def extend_metric_gram(space: VectorSpace[PointT, VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                       G: Coords, Q: VectorBlockT, q_new: VectorT,
                       metric_apply: Callable[[VectorT], CovectorT]) -> Coords:
    """Grow G = [(q_i, q_j)_M] by the row and column for q_new: one product
    with M and one block pairing."""
    return grow_gram(G, space.pair_block(space.append_vector(Q, q_new), metric_apply(q_new)))
