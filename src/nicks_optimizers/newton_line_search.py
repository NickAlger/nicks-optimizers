"""Newton directions for a line search, computed over a Krylov space.

The direction approximately solves H d = -g: the solvers of krylov_solvers
in their line-search mode minimize the model

    <g, d> + 0.5 <H_model d, d>

without a bound, until ||H d + g||_{M^{-1}} <= resid_tol (the surrogate
residual for the saddle-free model). Negative curvature is handled in one of
two ways:

  truncation     (default) the solve stops as soon as the Krylov space shows
                 nonpositive curvature, and returns the last direction found
                 with positive curvature, or, when it appears at the first
                 step, the preconditioned steepest descent direction
                 (-P^{-1} g; -M^{-1} g for the flexible solver). Every such
                 direction is a descent direction: <g, d> < 0.
  saddle-free    (saddle_free=True, Lanczos and flexible) H_model is |H|_M on
                 the Krylov space, with the M-Ritz values floored at
                 curvature_floor times the largest |theta|; always a descent
                 direction.

  solve_newton_direction           Lanczos over K_k(P^{-1} H, P^{-1} g): any
                                   symmetric positive definite P, and the
                                   saddle-free model. With a separate P and
                                   truncation it keeps no M-Gram matrix
                                   (the Newton system on the Krylov space
                                   does not involve M), so each step costs
                                   H, P^{-1} and one M^{-1} solve (the
                                   residual norm), and ||d||_M one product
                                   with M at the end.
  solve_newton_direction_cg        Newton-CG for P = M: five vectors, two
                                   reductions per step.
  solve_newton_direction_flexible  the flexible solver, for unconventional
                                   preconditioners; needs M and M^{-1}.

Each returns (d, SubproblemStats) with lam = 0 and stop reason "negative"
when it truncated.
"""

from typing import Callable, Optional

from .backend import Backend
from .krylov_solvers import (
    SubproblemOperators,
    SubproblemStats,
    cg_solver,
    flexible_solver,
    lanczos_solver,
    solve_with,
)
from .spaces import CovectorBlockT, CovectorT, VectorBlockT, VectorSpace, VectorT

UNBOUNDED = float("inf")              # the radius the solvers carry in line-search mode (unused)


def solve_newton_direction(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                           g: CovectorT, hessvec: Callable[[VectorT], CovectorT],
                           metric_solve: Callable[[CovectorT], VectorT], resid_tol: float,
                           max_iter: int, *,
                           precond_solve: Optional[Callable[[CovectorT], VectorT]] = None,
                           metric_apply: Optional[Callable[[VectorT], CovectorT]] = None,
                           saddle_free: bool = False, curvature_floor: float = 1e-6,
                           breakdown_tol: float = 1e-12,
                           project_covector: Optional[Callable[[CovectorT], CovectorT]] = None,
                           start_vector: Optional[VectorT] = None,
                           backend: Optional[Backend] = None, capacity: Optional[int] = None
                           ) -> tuple[VectorT, SubproblemStats]:
    """The Newton direction by Lanczos over K_k(P^{-1} H, P^{-1} g) (see the
    module docstring). M is given by metric_solve (M^{-1}); P = M unless
    precond_solve (P^{-1}) is given, and then metric_apply (M) is needed too.
    start_vector = P^{-1} g, if already computed, saves one solve.

    Stops at the first of: ||R||_{M^{-1}} <= resid_tol ("tol"), nonpositive
    curvature without saddle_free ("negative", see the module docstring),
    breakdown (the Krylov space is invariant: the direction is exact), or
    max_iter steps (capped at the dimension when the space provides one,
    "max"). capacity, the initial room in the basis (None:
    DEFAULT_CAPACITY), doubles when the basis fills up. Raises ValueError if
    <g, P^{-1} g> is not positive. Returns (d, stats).
    """
    if precond_solve is not None and metric_apply is None:
        raise ValueError("a separate preconditioner needs metric_apply (M)")
    ops = SubproblemOperators(hessvec=hessvec, metric_solve=metric_solve,
                              precond_solve=precond_solve, metric_apply=metric_apply,
                              project_covector=project_covector)
    solver = lanczos_solver(max_iter, saddle_free=saddle_free, breakdown_tol=breakdown_tol,
                            trust_region=False, curvature_floor=curvature_floor)
    return solve_with(solver, backend, space, ops, g, UNBOUNDED, resid_tol, start_vector, capacity)


def solve_newton_direction_cg(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                              g: CovectorT, hessvec: Callable[[VectorT], CovectorT],
                              metric_solve: Callable[[CovectorT], VectorT], resid_tol: float,
                              max_iter: int, *,
                              project_covector: Optional[Callable[[CovectorT], CovectorT]] = None,
                              start_vector: Optional[VectorT] = None,
                              backend: Optional[Backend] = None) -> tuple[VectorT, SubproblemStats]:
    """The Newton direction by Newton-CG, preconditioned by M (P = M): CG on
    H d = -g from d = 0, stopping at ||r||_{M^{-1}} <= resid_tol ("tol"), at
    nonpositive curvature <H p, p> <= 0 ("negative": the current iterate, or
    p_0 = -M^{-1} g at the first step) or after max_iter steps ("max").
    start_vector = M^{-1} g, if already computed, saves one solve. Raises
    ValueError if <g, M^{-1} g> is not positive. Returns (d, stats).
    """
    ops = SubproblemOperators(hessvec=hessvec, metric_solve=metric_solve,
                              project_covector=project_covector)
    return solve_with(cg_solver(max_iter, trust_region=False), backend, space, ops, g, UNBOUNDED,
                      resid_tol, start_vector)


def solve_newton_direction_flexible(
        space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
        g: CovectorT, hessvec: Callable[[VectorT], CovectorT],
        metric_solve: Callable[[CovectorT], VectorT],
        metric_apply: Callable[[VectorT], CovectorT],
        precond_solve: Callable[[CovectorT], VectorT], resid_tol: float, max_iter: int, *,
        saddle_free: bool = False, curvature_floor: float = 1e-6, breakdown_tol: float = 1e-10,
        project_covector: Optional[Callable[[CovectorT], CovectorT]] = None,
        start_vector: Optional[VectorT] = None,
        backend: Optional[Backend] = None, capacity: Optional[int] = None
        ) -> tuple[VectorT, SubproblemStats]:
    """The Newton direction by the flexible solver (see
    solve_tr_subproblem_flexible for how the subspace grows, and the module
    docstring for negative curvature). Its first basis vector is M^{-1} g /
    ||g||_{M^{-1}}, so the truncation fallback is -M^{-1} g (P may not be
    symmetric, so -P^{-1} g need not be a descent direction).
    start_vector = M^{-1} g, if already computed, saves one solve. Raises
    ValueError if g = 0. Returns (d, stats).
    """
    ops = SubproblemOperators(hessvec=hessvec, metric_solve=metric_solve,
                              precond_solve=precond_solve, metric_apply=metric_apply,
                              project_covector=project_covector)
    solver = flexible_solver(max_iter, saddle_free=saddle_free, breakdown_tol=breakdown_tol,
                             trust_region=False, curvature_floor=curvature_floor)
    return solve_with(solver, backend, space, ops, g, UNBOUNDED, resid_tol, start_vector, capacity)
