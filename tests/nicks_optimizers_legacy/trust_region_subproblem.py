"""The trust-region subproblem, solved over a Krylov space.

    min_s  <g, s> + 0.5 <H_model s, s>    s.t.  ||s||_M <= radius,

where M is the metric (the geometry, see metrics) and H_model is H, or |H|_M
for the saddle-free model. A preconditioner P only chooses the subspace;
by default P = M. Three solvers:

  solve_tr_subproblem           Lanczos with full reorthogonalization over
                                K_k(P^{-1} H, P^{-1} g): any symmetric
                                positive definite P, and the saddle-free
                                model.
  solve_tr_subproblem_cg        Steihaug-Toint truncated conjugate gradients
                                for P = M and the plain model: five vectors,
                                two reductions per step.
  solve_tr_subproblem_flexible  a subspace grown by applying P to the KKT
                                residual, for preconditioners that are
                                nonsymmetric, nonlinear (inexact inner
                                solves) or stochastic; needs M and M^{-1}.

Lanczos. With T = <H Q, Q> and G = [(q_i, q_j)_M] = L L^T, the matrix
A = L^{-1} T L^{-T} is H in the M-orthonormal basis Q L^{-T}, so its
eigenpairs are the M-Ritz pairs (Rayleigh-Ritz for the pencil
H u = theta M u). In those coordinates z the ball is round and the small
problem is

    min_z  b^T z + 0.5 z^T A_model z    s.t.  ||z|| <= radius,

with b = gamma0 L^{-1} e_1, and A_model = A, or |A| (eigenvalues replaced by
their absolute values) for the saddle-free model. The step is s = Q L^{-T} z.

When P = M the basis is M-orthonormal, so G = I and A = T: only M^{-1} is
ever applied, and the KKT residual norm is gamma_next |y_k|, exactly. With a
separate P, each step also costs one product with M (to grow G) and one
solve with M^{-1} (to measure the residual, formed explicitly, see
krylov_residual_norm), on top of the Hessian product and the P solve, plus
O(k) vector operations for reorthogonalization. Storage is the blocks Q and
V = P Q.

Caveat when P differs from M: if the trust region is active, the exact step
solves (H + lam M) s = -g, which K_k(P^{-1} H, P^{-1} g) approximates slowly
unless M is close to a multiple of P. Steps always beat the Cauchy point
along -P^{-1} g, so global convergence is unaffected, but boundary solves stop
when the model stops improving (stall_tol) rather than on the residual.
"""

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np

from .dense import grow_gram, pencil_ritz, solve_sphere_trs_eig
from .krylov import (
    CGState,
    LanczosState,
    cg_advance,
    cg_curvature,
    cg_start,
    lanczos_advance,
    lanczos_start,
    lanczos_tridiagonal,
)
from .metrics import extend_metric_gram
from .spaces import (
    Coords,
    CovectorBlockT,
    CovectorT,
    VectorBlockT,
    VectorSpace,
    VectorT,
)


StopReason = Literal["tol", "stall", "breakdown", "max", "boundary", "negative"]
KrylovMethod = Literal["lanczos", "cg", "flexible"]


@dataclass(frozen=True, kw_only=True)
class SubproblemStats:
    """Diagnostics of one trust-region subproblem solve: scalars and tuples
    only, so records stay small however long the run.

      krylov            "lanczos", "cg" or "flexible", the solver used
      krylov_steps      number of Krylov steps k (= Hessian-vector products)
      stop_reason       "tol" (residual met resid_tol), "stall" (Lanczos:
                        boundary active and the model stopped changing),
                        "breakdown" (Lanczos: invariant subspace, the subspace
                        solution is exact; flexible: no new direction, not
                        even M^{-1} R), "max" (max_krylov or, for Lanczos
                        and flexible, the dimension), "boundary" (CG: the next
                        iterate would leave the trust region; the step stops
                        on it) or "negative" (CG: negative curvature along the
                        search direction; the step follows it to the boundary)
      step_norm, lam    ||s||_M and the multiplier (> 0 when the boundary is
                        active). CG does not compute the multiplier: lam is 0
                        for interior steps and nan on the boundary.
      model_value       <g,s> + 0.5 <H_model s, s>, the model that was minimized
      true_model_value  <g,s> + 0.5 <H s, s> (differs only with saddle_free)
      resid, resid_tol  final ||R||_{M^{-1}} and the tolerance it was tested
                        against (CG: nan when the step stopped on the boundary)
      n_negative, theta_min  Lanczos and flexible: number of negative M-Ritz
                        values and the smallest one. CG: 1 if negative
                        curvature was found, and the smallest
                        <H p, p> / ||p||_M^2 seen.
      *_calls           calls of each operator made by this solve
                        (metric_solve also counts its use as the
                        preconditioner when P = M)
      trace_*           per Krylov step: model value, residual, multiplier,
                        number of negative curvature values
      fallbacks         flexible: how often the preconditioner's direction was
                        unusable (not finite, or nothing new) and M^{-1} R
                        was used instead; 0 for the other solvers
    """
    krylov: KrylovMethod
    krylov_steps: int
    stop_reason: StopReason
    step_norm: float
    lam: float
    model_value: float
    true_model_value: float
    resid: float
    resid_tol: float
    n_negative: int
    theta_min: float
    hessvec_calls: int
    metric_solve_calls: int
    precond_solve_calls: int
    metric_apply_calls: int
    trace_model: tuple[float, ...]
    trace_resid: tuple[float, ...]
    trace_lam: tuple[float, ...]
    trace_negative: tuple[int, ...]
    fallbacks: int = 0


def solve_projected_trs(T: Coords, G: Coords, gamma0: float, radius: float,
                        saddle_free: bool = False
                        ) -> tuple[Coords, Coords, float, float, float, Coords]:
    """Solve the small trust-region problem in M-orthonormal coordinates.

    The model Hessian has the M-Ritz vectors as eigenvectors and the M-Ritz
    values theta as eigenvalues, or |theta| for the saddle-free model.

    Returns (y, y_flip, lam, model_value, true_model_value, theta):
      y            coefficients of the step, s = Q y
      y_flip       coefficients of the flipped step s_flip = Q y_flip, whose
                   components along negative M-Ritz vectors are negated. If
                   the Ritz vectors were exact eigenvectors, |H|_M s would
                   equal H s_flip; y_flip = y unless saddle_free.
      lam          trust-region multiplier, in both coordinate systems
      model_value  g^T s + 0.5 s^T H_model s, the model that was minimized
      true_model_value  g^T s + 0.5 s^T H s with the true H on the subspace
                   (equal to model_value unless saddle_free)
      theta        M-Ritz values
    """
    k = T.shape[0]
    L, theta, Z = pencil_ritz(T, G)
    curvature = np.abs(theta) if saddle_free else theta
    order = np.argsort(curvature)
    e1 = np.zeros(k)
    e1[0] = 1.0
    b = gamma0 * np.linalg.solve(L, e1)
    z, lam = solve_sphere_trs_eig(curvature[order], Z[:, order], b, radius)
    zh = Z.T @ z                                     # M-Ritz coordinates
    model_value = float(b @ z + 0.5 * np.sum(curvature * zh**2))
    true_model_value = float(b @ z + 0.5 * np.sum(theta * zh**2))
    y = np.linalg.solve(L.T, z)
    if saddle_free:
        signs = np.where(theta < 0.0, -1.0, 1.0)
        y_flip = np.linalg.solve(L.T, Z @ (signs * zh))
    else:
        y_flip = y
    return y, y_flip, lam, model_value, true_model_value, theta


def krylov_residual_norm(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                         state: LanczosState[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                         y: Coords, y_flip_last: float, lam: float, G: Coords,
                         metric_apply: Optional[Callable[[VectorT], CovectorT]],
                         metric_solve: Callable[[CovectorT], VectorT]) -> tuple[float, int, int]:
    """||R||_{M^{-1}} for the residual R = H s_flip + lam M s + g of the step
    s = Q y. Returns (norm, products with M, solves with M^{-1}).

    The Lanczos relation H Q = V T + gamma_next v_next e_k^T and the projected
    equations give

        R = lam (M Q y - V G y) + gamma_next y_flip[-1] v_next.

    For the plain model s_flip = s, and this is the exact KKT residual; for
    the saddle-free model it is the surrogate for |H|_M s + lam M s + g.

    With metric_apply None (P = M), V = M Q and G = I, so R is
    gamma_next y_flip[-1] v_next and its norm is gamma_next |y_flip[-1]|,
    exactly (<v_next, M^{-1} v_next> = 1): no operator is applied. With a
    separate P, R is formed explicitly, which costs one product with M when
    lam > 0, and measured with one solve with M^{-1}.
    """
    if metric_apply is None:
        return state.gamma_next * abs(y_flip_last), 0, 0
    R = space.scale_covector(state.gamma_next * y_flip_last, state.v_next)
    applies = 0
    if lam > 0.0:
        Ms = metric_apply(space.combine_vectors(state.Q, y))
        applies = 1
        R = space.axpy_covector(lam, Ms, R)
        R = space.axpy_covector(-lam, space.combine_covectors(state.V, G @ y), R)
    return float(np.sqrt(max(space.pair(R, metric_solve(R)), 0.0))), applies, 1


def boundary_stalled(lam: float, model_value: float, prev_model_value: Optional[float],
                     stall_tol: float) -> bool:
    """True when the trust region is active and the last Lanczos step changed
    the model value by less than stall_tol (relative).

    With P != M the residual on the boundary contains lam M Q y - lam V G y,
    which the Krylov space of P^{-1} H only reduces slowly, so boundary solves
    stop on stagnation of the model value instead. The change is taken in
    absolute value because the saddle-free model itself changes with k.
    """
    if lam <= 0.0 or prev_model_value is None:
        return False
    return abs(prev_model_value - model_value) <= stall_tol * abs(model_value)


def solve_tr_subproblem(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                        g: CovectorT, hessvec: Callable[[VectorT], CovectorT],
                        metric_solve: Callable[[CovectorT], VectorT],
                        radius: float, resid_tol: float, max_iter: int, *,
                        precond_solve: Optional[Callable[[CovectorT], VectorT]] = None,
                        metric_apply: Optional[Callable[[VectorT], CovectorT]] = None,
                        stall_tol: float = 1e-3, saddle_free: bool = False,
                        breakdown_tol: float = 1e-12,
                        project_covector: Optional[Callable[[CovectorT], CovectorT]] = None,
                        start_vector: Optional[VectorT] = None
                        ) -> tuple[VectorT, SubproblemStats]:
    """Approximately solve  min <g, s> + 0.5 <H_model s, s>  s.t.  ||s||_M <= radius
    over the growing Krylov spaces K_k(P^{-1} H, P^{-1} g) by Lanczos, where
    H_model is H, or |H|_M (via M-Ritz pairs) when saddle_free.

    The operators are frozen at the current point. M is given by
    metric_solve (M^{-1}); P = M unless precond_solve (P^{-1}) is given, and
    then metric_apply (M) is needed too, consistent with metric_solve.
    start_vector = P^{-1} g, if already computed, saves one solve.
    project_covector, for representations with redundant components, maps
    covectors to canonical representatives; it is applied to every Krylov
    covector (see krylov).

    Stops at the first of:
      * ||R||_{M^{-1}} <= resid_tol, with R the (surrogate) KKT residual
        (see krylov_residual_norm),
      * the trust region is active and the model stopped changing
        (stall_tol; see boundary_stalled),
      * Lanczos breakdown (an invariant subspace: the subspace solution is
        exact),
      * max_iter Lanczos steps (capped at the dimension when the space
        provides one).
    Every k >= 1 gives at least the model decrease of the Cauchy point along
    -P^{-1} g, since that point lies in K_1.

    Returns (s, stats): the step and its SubproblemStats.
    """
    if precond_solve is not None and metric_apply is None:
        raise ValueError("a separate preconditioner needs metric_apply (M) for the Gram matrix")
    matched = precond_solve is None
    precond = metric_solve if precond_solve is None else precond_solve
    applies, solves = 0, 0
    state = lanczos_start(space, g, precond, start_vector)
    if space.dimension is not None:
        max_iter = min(max_iter, space.dimension(state.q))
    if metric_apply is None or matched:
        G = np.ones((1, 1))                                     # M-orthonormal basis
    else:
        G = np.array([[space.pair(metric_apply(state.q), state.q)]])   # (q0, q0)_M
        applies += 1
    prev_model_value: Optional[float] = None
    trace_model: list[float] = []
    trace_resid: list[float] = []
    trace_lam: list[float] = []
    trace_negative: list[int] = []

    for k in range(1, max_iter + 1):
        state = lanczos_advance(space, state, hessvec, precond, breakdown_tol,
                                project_covector=project_covector)
        y, y_flip, lam, model_value, true_model_value, theta = solve_projected_trs(
            lanczos_tridiagonal(state), G, state.gamma0, radius, saddle_free)
        resid, a, b = krylov_residual_norm(space, state, y, float(y_flip[-1]), lam, G,
                                           None if matched else metric_apply, metric_solve)
        applies, solves = applies + a, solves + b
        n_negative = int(np.sum(theta < 0.0))
        trace_model.append(model_value)
        trace_resid.append(resid)
        trace_lam.append(lam)
        trace_negative.append(n_negative)
        stalled = boundary_stalled(lam, model_value, prev_model_value, stall_tol)
        if resid <= resid_tol or stalled or state.breakdown or k == max_iter:
            reason: StopReason = ("breakdown" if state.breakdown else "tol" if resid <= resid_tol
                                  else "stall" if stalled else "max")
            precond_calls = k + (1 if start_vector is None else 0)
            stats = SubproblemStats(
                krylov="lanczos", krylov_steps=k, stop_reason=reason,
                step_norm=float(np.sqrt(max(float(y @ G @ y), 0.0))), lam=lam,
                model_value=model_value, true_model_value=true_model_value,
                resid=resid, resid_tol=resid_tol, n_negative=n_negative,
                theta_min=float(theta[0]), hessvec_calls=k,
                metric_solve_calls=solves + (precond_calls if matched else 0),
                precond_solve_calls=0 if matched else precond_calls,
                metric_apply_calls=applies,
                trace_model=tuple(trace_model), trace_resid=tuple(trace_resid),
                trace_lam=tuple(trace_lam), trace_negative=tuple(trace_negative))
            return space.combine_vectors(state.Q, y), stats

        prev_model_value = model_value
        # The pending vector joins the basis at the next lanczos_advance.
        if matched or metric_apply is None:
            G = np.eye(k + 1)
        else:
            G = extend_metric_gram(space, G, state.Q, state.q_next, metric_apply)
            applies += 1
    raise AssertionError("unreachable: the loop returns at k == max_iter")


def boundary_fraction(s_norm2: float, sp: float, p_norm2: float, radius: float) -> float:
    """The tau >= 0 with ||s + tau p||_M = radius, from ||s||_M^2, (s, p)_M and
    ||p||_M^2 (s inside the ball). Written to avoid cancellation."""
    gap = max(radius ** 2 - s_norm2, 0.0)
    root = float(np.sqrt(sp ** 2 + p_norm2 * gap))
    return gap / (sp + root) if sp > 0.0 else (root - sp) / p_norm2


def solve_tr_subproblem_cg(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                           g: CovectorT, hessvec: Callable[[VectorT], CovectorT],
                           metric_solve: Callable[[CovectorT], VectorT],
                           radius: float, resid_tol: float, max_iter: int, *,
                           project_covector: Optional[Callable[[CovectorT], CovectorT]] = None,
                           start_vector: Optional[VectorT] = None
                           ) -> tuple[VectorT, SubproblemStats]:
    """Steihaug-Toint truncated CG for  min <g, s> + 0.5 <H s, s>
    s.t.  ||s||_M <= radius, preconditioned by M itself (P = M).

    Runs CG on H s = -g from s = 0, preconditioned with metric_solve, and
    stops at the first of:
      * ||r||_{M^{-1}} <= resid_tol, r = g + H s the model gradient ("tol");
      * the next CG iterate would leave the ball: the step stops where the
        segment to it crosses the boundary ("boundary");
      * negative curvature, <H p, p> <= 0: the step follows p to the boundary
        ("negative");
      * max_iter steps ("max"). Unlike Lanczos, CG is not capped at the
        dimension: without reorthogonalization it can need more than n steps
        in floating point.
    The iterates' M-norms increase (Steihaug), so the first crossing is the
    right place to stop, and the first step is the Cauchy point along
    -M^{-1} g: every step gives at least the Cauchy decrease. Unlike the
    Lanczos solver it does not continue on the boundary, keeps no basis
    (five vectors), and needs two reductions per step.

    project_covector maps each residual to its canonical representative,
    for redundant representations (see krylov). start_vector = M^{-1} g, if
    already computed, saves one solve. Returns (s, stats).
    """
    state: CGState[VectorT, CovectorT] = cg_start(space, g, metric_solve, start_vector)
    # No cap at the dimension: without reorthogonalization, CG can need more
    # than n steps in floating point.
    trace_model: list[float] = []
    trace_resid: list[float] = []
    trace_lam: list[float] = []
    trace_negative: list[int] = []
    theta_min = float("inf")
    s, model = state.s, 0.0
    reason: StopReason = "max"
    lam, resid, n_negative = 0.0, float(np.sqrt(state.rz)), 0

    for k in range(1, max_iter + 1):
        Hp, curvature = cg_curvature(space, state, hessvec)
        theta_min = min(theta_min, curvature / state.p_norm2)
        alpha = state.rz / curvature if curvature > 0.0 else float("inf")
        if curvature <= 0.0 or (state.s_norm2 + 2.0 * alpha * state.sp
                                + alpha ** 2 * state.p_norm2 >= radius ** 2):
            tau = boundary_fraction(state.s_norm2, state.sp, state.p_norm2, radius)
            s = space.axpy_vector(tau, state.p, state.s)
            model = state.model - tau * state.rz + 0.5 * tau ** 2 * curvature
            reason = "negative" if curvature <= 0.0 else "boundary"
            lam, resid, n_negative = float("nan"), float("nan"), int(curvature <= 0.0)
            trace_model.append(model)
            trace_resid.append(resid)
            trace_lam.append(lam)
            trace_negative.append(n_negative)
            break
        state = cg_advance(space, state, Hp, curvature, metric_solve, project_covector)
        s, model, resid = state.s, state.model, float(np.sqrt(max(state.rz, 0.0)))
        trace_model.append(model)
        trace_resid.append(resid)
        trace_lam.append(0.0)
        trace_negative.append(0)
        if resid <= resid_tol:
            reason = "tol"
            break
    k_done = len(trace_model)
    on_boundary = reason in ("boundary", "negative")
    step_norm = radius if on_boundary else float(np.sqrt(max(state.s_norm2, 0.0)))
    stats = SubproblemStats(
        krylov="cg", krylov_steps=k_done, stop_reason=reason, step_norm=step_norm, lam=lam,
        model_value=model, true_model_value=model, resid=resid, resid_tol=resid_tol,
        n_negative=n_negative, theta_min=theta_min, hessvec_calls=k_done,
        metric_solve_calls=state.steps + (1 if start_vector is None else 0),
        precond_solve_calls=0, metric_apply_calls=0,
        trace_model=tuple(trace_model), trace_resid=tuple(trace_resid),
        trace_lam=tuple(trace_lam), trace_negative=tuple(trace_negative))
    return s, stats


def solve_tr_subproblem_flexible(
        space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
        g: CovectorT, hessvec: Callable[[VectorT], CovectorT],
        metric_solve: Callable[[CovectorT], VectorT],
        metric_apply: Callable[[VectorT], CovectorT],
        precond_solve: Callable[[CovectorT], VectorT],
        radius: float, resid_tol: float, max_iter: int, *,
        saddle_free: bool = False, breakdown_tol: float = 1e-10,
        project_covector: Optional[Callable[[CovectorT], CovectorT]] = None,
        start_vector: Optional[VectorT] = None
        ) -> tuple[VectorT, SubproblemStats]:
    """Approximately solve  min <g, s> + 0.5 <H_model s, s>  s.t.  ||s||_M <= radius
    over a subspace grown with a preconditioner that need not be symmetric,
    linear or deterministic: P^{-1} is any map from covectors to vectors
    (an inexact inner solve, a block triangular solve, sweeps of a randomized
    row-action method, a few steps of stochastic gradient descent, ...).

    P only proposes directions; nothing relies on its properties:
      * the basis starts with M^{-1} g, so every step gives at least the
        decrease of the Cauchy point along -M^{-1} g;
      * each new direction is P^{-1} R, R = H s + lam M s + g the KKT
        residual of the current step, made M-orthonormal to the basis by
        two Gram-Schmidt passes; if it is not finite or adds nothing (norm
        below breakdown_tol times its norm before orthogonalization),
        M^{-1} R is used instead ("fallback");
      * the projected Hessian T = Q^T H Q is formed from stored products H Q,
        and the small problem is solved exactly (as by solve_tr_subproblem,
        with G = I), so the step is the model's minimizer over the subspace
        and its model value is the true one;
      * the stopping test uses the explicit residual norm ||R||_{M^{-1}}.
    Applying P to the residual targets the system (H + lam M) s = -g that the
    step must solve, so a separate P that approximates H (or H + lam M) is
    effective on the boundary too, where Lanczos over K_k(P^{-1} H, P^{-1} g)
    stalls. With P = M (fixed) the subspace is the Krylov space of the
    Lanczos solver. With saddle_free, H_model is |H|_M (via M-Ritz pairs) and
    R the surrogate residual of solve_tr_subproblem.

    Each step costs one product with H, one application of P, one product
    with M (two with a fallback) and one solve with M^{-1}; the blocks Q,
    M Q and H Q are stored. start_vector = M^{-1} g, if already computed,
    saves one solve. project_covector (for redundant representations) is
    applied to every stored covector and residual; P's outputs must be valid
    vectors of the representation.

    Stops at the first of ||R||_{M^{-1}} <= resid_tol ("tol"), a breakdown
    ("breakdown", not even M^{-1} R adds a direction), or max_iter steps
    (capped at the dimension when the space provides one, "max").
    Returns (s, stats).
    """
    proj = (lambda w: w) if project_covector is None else project_covector
    hessvec_calls = precond_calls = apply_calls = solve_calls = fallbacks = 0
    if start_vector is None:
        Mg = metric_solve(g)
        solve_calls += 1
    else:
        Mg = start_vector
    gamma0 = float(np.sqrt(max(space.pair(g, Mg), 0.0)))
    q = space.scale_vector(1.0 / gamma0, Mg)
    Hq = proj(hessvec(q))
    hessvec_calls += 1
    Q, MQ, HQ = (space.vector_block(q), space.covector_block(space.scale_covector(1.0 / gamma0, g)),
                 space.covector_block(Hq))
    T = np.array([[space.pair(Hq, q)]])
    if space.dimension is not None:
        max_iter = min(max_iter, space.dimension(q))
    trace_model: list[float] = []
    trace_resid: list[float] = []
    trace_lam: list[float] = []
    trace_negative: list[int] = []

    for k in range(1, max_iter + 1):
        y, y_flip, lam, model_value, true_model_value, theta = solve_projected_trs(
            T, np.eye(k), gamma0, radius, saddle_free)
        R = space.axpy_covector(1.0, g, space.combine_covectors(HQ, y_flip))
        if lam > 0.0:
            R = space.axpy_covector(lam, space.combine_covectors(MQ, y), R)
        R = proj(R)
        MinvR = metric_solve(R)
        solve_calls += 1
        resid = float(np.sqrt(max(space.pair(R, MinvR), 0.0)))
        n_negative = int(np.sum(theta < 0.0))
        trace_model.append(model_value)
        trace_resid.append(resid)
        trace_lam.append(lam)
        trace_negative.append(n_negative)
        reason: Optional[StopReason] = ("tol" if resid <= resid_tol
                                        else "max" if k == max_iter else None)
        if reason is None:
            z = precond_solve(R)
            precond_calls += 1
            q_new: Optional[VectorT] = None
            for candidate, from_precond in ((z, True), (MinvR, False)):
                Mz = metric_apply(candidate)
                apply_calls += 1
                before = float(np.sqrt(abs(space.pair(Mz, candidate))))
                if np.isfinite(before) and before > 0.0:
                    for _ in range(2):                  # Gram-Schmidt in the M inner product, twice
                        c = space.pair_block(Q, Mz)
                        candidate = space.axpy_vector(-1.0, space.combine_vectors(Q, c), candidate)
                        Mz = space.axpy_covector(-1.0, space.combine_covectors(MQ, c), Mz)
                    norm = float(np.sqrt(max(space.pair(Mz, candidate), 0.0)))
                    if np.isfinite(norm) and norm > breakdown_tol * before:
                        q_new = space.scale_vector(1.0 / norm, candidate)
                        Mq_new = proj(space.scale_covector(1.0 / norm, Mz))
                        break
                if from_precond:
                    fallbacks += 1
            if q_new is None:
                reason = "breakdown"
            else:
                Hq = proj(hessvec(q_new))
                hessvec_calls += 1
                Q = space.append_vector(Q, q_new)
                MQ = space.append_covector(MQ, Mq_new)
                HQ = space.append_covector(HQ, Hq)
                T = grow_gram(T, space.pair_block(Q, Hq))
        if reason is not None:
            stats = SubproblemStats(
                krylov="flexible", krylov_steps=k, stop_reason=reason,
                step_norm=float(np.linalg.norm(y)), lam=lam,
                model_value=model_value, true_model_value=true_model_value,
                resid=resid, resid_tol=resid_tol, n_negative=n_negative,
                theta_min=float(theta[0]), hessvec_calls=hessvec_calls,
                metric_solve_calls=solve_calls, precond_solve_calls=precond_calls,
                metric_apply_calls=apply_calls,
                trace_model=tuple(trace_model), trace_resid=tuple(trace_resid),
                trace_lam=tuple(trace_lam), trace_negative=tuple(trace_negative),
                fallbacks=fallbacks)
            return space.combine_vectors(Q, y), stats
    raise AssertionError("unreachable: the loop returns at k == max_iter")
