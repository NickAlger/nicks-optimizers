"""The Hilbert structure: a metric M, given by operators.

Everything geometric is measured with a symmetric positive definite metric
M(x), a structure on top of the vector space (see spaces): the trust region
{ s : ||s||_M <= radius }, gradient norms ||g||_{M^{-1}} (the stopping test),
curvature (M-Ritz values, for the saddle-free model) and Krylov residuals.
M is given by two separate operators, each needed only where it is used:

  metric_solve(x, w) -> Vector     M^{-1} w, the Riesz map: always needed
  metric_apply(x, v) -> Covector   M v: needed only with a separate
                                   preconditioner P

and the inner products and norms follow from the pairing:
(u, v)_M = <M u, v> and (phi, psi)_{M^{-1}} = <phi, M^{-1} psi>. Operators
rather than inner products, because the method needs vectors out of M^{-1}
(when P = M it is the preconditioner) and covectors out of M (the explicit
Krylov residual), and inner products alone would not give them. When P = M,
only M^{-1} is ever applied: any symmetric positive definite operator, a
multigrid cycle for instance, defines M as its inverse, with nothing to keep
consistent. When both are given they must be inverse to each other.

Helpers for the simple cases:

  euclidean_metric_solve, euclidean_metric_apply
        M = I: the identity on components, for numpy arrays (also
        distributed: it acts locally) and for tuple trees of them, where it
        is the direct sum of the leaves' Euclidean inner products. Written as
        the point-dependent operators trust_region_newton takes.
  direct_sum(leaf_operators)
        for tuple trees: the operator that applies each leaf of a tree of
        operators to the corresponding leaf, e.g. a different metric on each
        leaf (for M and M^{-1} separately).
  vector_norm, covector_norm, extend_metric_gram
        norms and the M-Gram matrix of a Krylov basis.
"""

from typing import Any, Callable

import numpy as np

from .dense import grow_gram
from .spaces import (
    Coords,
    Covector,
    CovectorBlockT,
    CovectorT,
    Point,
    Vector,
    VectorBlockT,
    VectorSpace,
    VectorT,
)
from .trees import as_tuple_tree, tree_map2


# Metric operators for numpy arrays (documentation aliases): point-dependent.
MetricSolve = Callable[[Point, Covector], Vector]
MetricApply = Callable[[Point, Vector], Covector]


def euclidean_metric_solve(x: Any, w: Any) -> Any:
    """M^{-1} w with M = I: the covector's components, as a vector. For numpy
    arrays and tuple trees of them (the direct sum of Euclidean inner
    products)."""
    return w


def euclidean_metric_apply(x: Any, v: Any) -> Any:
    """M v with M = I: the vector's components, as a covector."""
    return v


def direct_sum(operators: Any) -> Callable[[Any], Any]:
    """The direct sum of leaf operators, for tuple trees: given a tuple tree
    of functions (lists allowed), the operator that applies each of them to
    the corresponding leaf of its argument. For example, a metric that
    weights the leaves of (a, b) by m_a and m_b:

        M_inv = direct_sum((lambda w: w / m_a, lambda w: w / m_b))
        M = direct_sum((lambda v: m_a * v, lambda v: m_b * v))
        trust_region_newton(..., metric_solve=lambda x, w: M_inv(w),
                            metric_apply=lambda x, v: M(v))
    """
    ops = as_tuple_tree(operators)
    return lambda tree: tree_map2(lambda op, leaf: op(leaf), ops, tree)


def vector_norm(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                metric_apply: Callable[[VectorT], CovectorT], v: VectorT) -> float:
    """||v||_M = sqrt(<M v, v>)."""
    return float(np.sqrt(max(space.pair(metric_apply(v), v), 0.0)))


def covector_norm(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                  metric_solve: Callable[[CovectorT], VectorT], w: CovectorT) -> float:
    """||w||_{M^{-1}} = sqrt(<w, M^{-1} w>), the dual norm of the M-norm: the
    largest <w, s> over steps with ||s||_M <= 1."""
    return float(np.sqrt(max(space.pair(w, metric_solve(w)), 0.0)))


def extend_metric_gram(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                       G: Coords, Q: VectorBlockT, q_new: VectorT,
                       metric_apply: Callable[[VectorT], CovectorT]) -> Coords:
    """Grow G = [(q_i, q_j)_M] by the row and column for q_new: one product
    with M and one block pairing."""
    Mq = metric_apply(q_new)
    return grow_gram(G, np.append(space.pair_block(Q, Mq), space.pair(Mq, q_new)))
