"""Krylov solvers for the steps of Newton methods, as pieces a backend can
compile.

The step minimizes the quadratic model

    m(s) = <g, s> + 0.5 <H_model s, s>

approximately, over a growing Krylov space, within a trust region
||s||_M <= radius (see trust_region_subproblem). M is the metric (the
geometry, see metrics) and H_model is H, or |H|_M for the saddle-free model.
A preconditioner P only chooses the subspace; by default P = M. Three
solvers:

  lanczos_solver    Lanczos with full reorthogonalization over
                    K_k(P^{-1} H, P^{-1} g): any symmetric positive definite
                    P, and the saddle-free model.
  cg_solver         truncated conjugate gradients for P = M and the plain
                    model: five vectors, two reductions per step.
  flexible_solver   a subspace grown by applying P to the KKT residual, for
                    preconditioners that are nonsymmetric, nonlinear
                    (inexact inner solves) or stochastic; needs M and M^{-1}.

Lanczos. With T = <H Q, Q> and G = [(q_i, q_j)_M] = L L^T, the matrix
A = L^{-1} T L^{-T} is H in the M-orthonormal basis Q L^{-T}, so its
eigenpairs are the M-Ritz pairs (Rayleigh-Ritz for the pencil
H u = theta M u). The small problem is solved in those coordinates z, with
b = gamma0 L^{-1} e_1 and A_model = A, or |A| (eigenvalues replaced by their
absolute values) for the saddle-free model; the step is s = Q L^{-T} z.

When P = M the basis is M-orthonormal, so G = I and A = T: only M^{-1} is
ever applied, and the KKT residual norm is gamma_next |y_k|, exactly. With a
separate P, each step also costs one product with M (to grow G) and one
solve with M^{-1} (to measure the residual, formed explicitly, see
krylov_residual_norm), on top of the Hessian product and the P solve, plus
O(k) vector operations for reorthogonalization. Storage is the blocks Q and
V = P Q.

Written for the backend
-----------------------
Each solver is a KrylovSolver: pure pieces without Python control flow on
values (see backend), which a backend can compile.

  begin(backend, space, ops, g, radius, resid_tol, start_vector, capacity)
                       the start, and the first step for Lanczos and the
                       flexible solver
  run(backend, space, ops, g, carry)
                       steps until the solve stops, or the basis is full
                       (status FULL)
  grow(backend, space, carry, capacity)
                       the same solve with room for more basis vectors: the
                       blocks and small matrices are copied, nothing is
                       recomputed, the status goes from FULL back to
                       RUNNING, and run continues where it stopped
  result(backend, space, ops, carry) -> (s, RawSubproblem)

capacity is the room in the basis at the start (None: DEFAULT_CAPACITY,
32, or max_iter if smaller). solve_with runs the pieces, doubling the
capacity when the basis fills up, and returns SubproblemStats; the
optimizers do the same for every solve. Statistics come out as
RawSubproblem (arrays, inside compiled code) and are converted to
SubproblemStats by subproblem_stats.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Generic, Literal, Optional

from .backend import Backend, backend_state, numpy_backend, run_cond, run_loop
from .dense import solve_projected_newton, solve_projected_trs
from .krylov import (
    CGState,
    LanczosState,
    cg_advance,
    cg_curvature,
    cg_start,
    lanczos_grow,
    lanczos_start,
    lanczos_step_state,
    tridiagonal,
)
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
                        on it) or "negative" (trust region, CG: negative
                        curvature along the search direction; the step
                        follows it to the boundary; line search, any solver:
                        nonpositive curvature appeared, and the direction is
                        the last one found before it, or the preconditioned
                        steepest descent direction when it appeared at the
                        first step)
      step_norm, lam    ||s||_M and the multiplier (> 0 when the boundary is
                        active). CG does not compute the multiplier: lam is 0
                        for interior steps and nan on the boundary. A line
                        search has no multiplier: lam = 0.
      model_value       <g,s> + 0.5 <H_model s, s>, the model that was minimized
      true_model_value  <g,s> + 0.5 <H s, s> (differs only with saddle_free)
      resid, resid_tol  final ||R||_{M^{-1}} and the tolerance it was tested
                        against (CG: nan when the step stopped on the boundary;
                        line search: nan for the steepest-descent fallback)
      n_negative, theta_min  Lanczos and flexible: number of negative M-Ritz
                        values and the smallest one (line search with a
                        separate P and no saddle-free model: eigenvalues of T,
                        which have the same signs). CG: 1 if negative
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


# Status codes of the solvers: RUNNING while the loop goes on, then the stop
# reason; FULL when the basis has no room for the next vector (grow and run
# again); INVALID when the solve could not start (<g, P^{-1} g> not positive).
RUNNING, TOL, STALL, BREAKDOWN, MAX, BOUNDARY, NEGATIVE, INVALID, FULL = range(9)
STOP_REASONS: dict[int, StopReason] = {TOL: "tol", STALL: "stall", BREAKDOWN: "breakdown",
                                       MAX: "max", BOUNDARY: "boundary", NEGATIVE: "negative"}
INVALID_START = "<g, P^{-1} g> must be positive; is P positive definite and g nonzero?"
METHODS: tuple[KrylovMethod, ...] = ("cg", "lanczos", "flexible")


@backend_state
@dataclass(frozen=True, kw_only=True)
class RawSubproblem:
    """SubproblemStats as the backend computes them: 0-d arrays, integer
    codes for the method (an index into METHODS) and the stop reason, and
    traces as arrays of fixed length whose first krylov_steps entries are
    filled. subproblem_stats converts them (outside compiled code)."""
    method: Any
    krylov_steps: Any
    status: Any
    step_norm: Any
    lam: Any
    model_value: Any
    true_model_value: Any
    resid: Any
    resid_tol: Any
    n_negative: Any
    theta_min: Any
    hessvec_calls: Any
    metric_solve_calls: Any
    precond_solve_calls: Any
    metric_apply_calls: Any
    trace_model: Any
    trace_resid: Any
    trace_lam: Any
    trace_negative: Any
    fallbacks: Any


def as_number(v: Any) -> float:
    """A 0-d array of any backend (or a number) as a Python float."""
    return float(v.item() if hasattr(v, "item") else v)


def as_integer(v: Any) -> int:
    return int(v.item() if hasattr(v, "item") else v)


def subproblem_stats(raw: RawSubproblem) -> SubproblemStats:
    """SubproblemStats (Python numbers and tuples) from the raw statistics.
    Raises ValueError for status INVALID."""
    status = as_integer(raw.status)
    if status == INVALID:
        raise ValueError(INVALID_START)
    k = as_integer(raw.krylov_steps)

    def trace(a: Any) -> list[Any]:
        values: list[Any] = a[:k].tolist()
        return values
    return SubproblemStats(
        krylov=METHODS[as_integer(raw.method)], krylov_steps=k, stop_reason=STOP_REASONS[status],
        step_norm=as_number(raw.step_norm), lam=as_number(raw.lam),
        model_value=as_number(raw.model_value), true_model_value=as_number(raw.true_model_value),
        resid=as_number(raw.resid), resid_tol=as_number(raw.resid_tol),
        n_negative=as_integer(raw.n_negative), theta_min=as_number(raw.theta_min),
        hessvec_calls=as_integer(raw.hessvec_calls),
        metric_solve_calls=as_integer(raw.metric_solve_calls),
        precond_solve_calls=as_integer(raw.precond_solve_calls),
        metric_apply_calls=as_integer(raw.metric_apply_calls),
        trace_model=tuple(float(v) for v in trace(raw.trace_model)),
        trace_resid=tuple(float(v) for v in trace(raw.trace_resid)),
        trace_lam=tuple(float(v) for v in trace(raw.trace_lam)),
        trace_negative=tuple(int(v) for v in trace(raw.trace_negative)),
        fallbacks=as_integer(raw.fallbacks))


@dataclass(frozen=True, kw_only=True)
class SubproblemOperators(Generic[VectorT, CovectorT]):
    """The operators a subproblem solver uses, frozen at the current point:
    hessvec (H), metric_solve (M^{-1}), precond_solve (P^{-1}, None for
    P = M), metric_apply (M, None when not needed) and project_covector
    (None, or the projection onto canonical covectors, see krylov)."""
    hessvec: Callable[[VectorT], CovectorT]
    metric_solve: Callable[[CovectorT], VectorT]
    precond_solve: Optional[Callable[[CovectorT], VectorT]] = None
    metric_apply: Optional[Callable[[VectorT], CovectorT]] = None
    project_covector: Optional[Callable[[CovectorT], CovectorT]] = None


@dataclass(frozen=True, kw_only=True)
class KrylovSolver:
    """The pieces of a subproblem solver (see the module docstring), and
    capacity(carry), the basis capacity of a carry (None: no basis)."""
    name: KrylovMethod
    begin: Callable[..., Any]
    run: Callable[..., Any]
    grow: Callable[..., Any]
    result: Callable[..., tuple[Any, RawSubproblem]]
    capacity: Callable[[Any], Optional[int]]


DEFAULT_CAPACITY = 32
"""The room in a Krylov basis at the start of a solve, unless given: it
doubles when the basis fills up."""


def initial_capacity(capacity: Optional[int], max_iter: int) -> int:
    """The room at the start of a solve: capacity (DEFAULT_CAPACITY if None),
    at most max_iter."""
    return min(DEFAULT_CAPACITY if capacity is None else capacity, max_iter)


def solve_with(solver: KrylovSolver, backend: Optional[Backend],
               space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT], ops: SubproblemOperators[VectorT, CovectorT],
               g: CovectorT, radius: Any, resid_tol: Any, start_vector: Optional[VectorT] = None,
               capacity: Optional[int] = None) -> tuple[VectorT, SubproblemStats]:
    """Run a solver's pieces (on numpy by default), doubling the capacity of
    the basis whenever it fills up, and return the step and its
    SubproblemStats."""
    bk = numpy_backend if backend is None else backend
    carry = solver.run(bk, space, ops, g,
                       solver.begin(bk, space, ops, g, radius, resid_tol, start_vector, capacity))
    while as_integer(carry.status) == FULL:
        current = solver.capacity(carry)
        assert current is not None
        carry = solver.run(bk, space, ops, g, solver.grow(bk, space, carry, 2 * current))
    s, raw = solver.result(bk, space, ops, carry)
    return s, subproblem_stats(raw)


def leading(backend: Backend, a: Any, k: Any) -> Any:
    """With numpy, the leading k entries (k x k block) of a padded array, so
    that small computations happen at the size they used to; with static
    shapes, a itself (the padding is masked or zero)."""
    if backend.static_shapes:
        return a
    n = int(k)
    return a[:n, :n] if a.ndim == 2 else a[:n]


def _next_status(backend: Backend, done: Any, k: Any, bound: int, capacity: int) -> Any:
    """MAX when k reached the bound on the steps, FULL when the basis is full
    below it, else RUNNING, unless done already holds a stop reason."""
    xp = backend.xp
    later = xp.where(k >= bound, MAX, xp.where(k >= capacity, FULL, RUNNING) if capacity < bound
                     else RUNNING)
    return xp.where(done != RUNNING, done, later)


# --- Lanczos ---------------------------------------------------------------------------------


def krylov_residual_norm(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                         state: LanczosState[VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                         y: Coords, y_flip_last: Any, lam: Any, G: Optional[Coords],
                         metric_apply: Optional[Callable[[VectorT], CovectorT]],
                         metric_solve: Callable[[CovectorT], VectorT], *,
                         backend: Backend = numpy_backend, k: Optional[Any] = None
                         ) -> tuple[Any, Any, Any]:
    """||R||_{M^{-1}} for the residual R = H s_flip + lam M s + g of the step
    s = Q y (y of length k, or padded with k given). Returns (norm, products
    with M, solves with M^{-1}).

    The Lanczos relation H Q = V T + gamma_next v_next e_k^T and the projected
    equations give

        R = lam (M Q y - V G y) + gamma_next y_flip[-1] v_next.

    For the plain model s_flip = s, and this is the exact KKT residual; for
    the saddle-free model it is the surrogate for |H|_M s + lam M s + g.

    With metric_apply None (P = M), V = M Q and G = I, so R is
    gamma_next y_flip[-1] v_next and its norm is gamma_next |y_flip[-1]|,
    exactly (<v_next, M^{-1} v_next> = 1): no operator is applied. With a
    separate P, R is formed explicitly, which costs one product with M when
    lam > 0, and measured with one solve with M^{-1}. lam None stands for
    no multiplier at all (a line search): R = gamma_next y_flip[-1] v_next,
    and G is not needed.
    """
    xp = backend.xp
    if metric_apply is None:
        return state.gamma_next * xp.abs(y_flip_last), xp.asarray(0), xp.asarray(0)
    R = space.scale_covector(state.gamma_next * y_flip_last, state.v_next)
    applies = xp.asarray(0)
    if lam is not None:
        assert G is not None
        m = y.shape[0] if k is None else k

        def with_metric(R: CovectorT) -> CovectorT:
            Ms = metric_apply(space.combine_vectors(state.Q, y, m))
            R = space.axpy_covector(lam, Ms, R)
            return space.axpy_covector(-lam, space.combine_covectors(state.V, G @ y, m), R)
        R = run_cond(backend, lam > 0.0, with_metric, lambda R: R, R)
        applies = xp.where(lam > 0.0, 1, 0)
    norm = xp.sqrt(xp.maximum(space.pair(R, metric_solve(R)), 0.0))
    return norm, applies, xp.asarray(1)


def _resized(backend: Backend, a: Any, n: int) -> Any:
    """a cut or zero-padded to length n (numpy coordinates grow with k)."""
    if a.shape[0] >= n:
        return a[:n]
    return backend.index_update(backend.xp.zeros(n), slice(0, a.shape[0]), a)


def _truncated(backend: Backend, truncate: Any, k: Any, gamma0: Any, y_new: Any, y_prev: Any,
               T: Any) -> tuple[Any, Any]:
    """The coefficients and the model value of the direction a line-search
    solve returns when it truncates at nonpositive curvature: the previous
    step's solution, or at the first step -gamma0 e_1 (minus the first basis
    vector times gamma0: the preconditioned steepest descent direction).
    y_new otherwise. The model value is the true quadratic model
    gamma0 y_0 + 0.5 y^T T y."""
    xp = backend.xp
    n = y_new.shape[0]
    steepest = xp.where(xp.arange(n) == 0, -gamma0, 0.0)
    y = xp.where(truncate, xp.where(k == 1, steepest, _resized(backend, y_prev, n)), y_new)
    return y, gamma0 * y[0] + 0.5 * (y @ (T @ y))


def boundary_stalled(lam: float, model_value: float, prev_model_value: Optional[float],
                     stall_tol: float) -> bool:
    """True when the trust region is active and the last Lanczos step changed
    the model value by less than stall_tol (relative).

    With P != M the residual on the boundary contains lam M Q y - lam V G y,
    which the Krylov space of P^{-1} H only reduces slowly, so boundary solves
    stop on stagnation of the model value instead. The change is taken in
    absolute value because the saddle-free model itself changes with k. (The
    solver computes the same test without branches.)
    """
    if lam <= 0.0 or prev_model_value is None:
        return False
    return abs(prev_model_value - model_value) <= stall_tol * abs(model_value)


@backend_state
@dataclass(frozen=True, kw_only=True)
class _LanczosCarry(Generic[VectorT, CovectorT, VectorBlockT, CovectorBlockT]):
    """The Lanczos solve: the process, the M-Gram matrix G of its basis
    (capacity x capacity, the identity when P = M), the problem's radius and
    tolerance, the status, the latest solution of the projected problem and
    its statistics, the work counters, and the traces (length: the bound on
    the steps)."""
    state: LanczosState[VectorT, CovectorT, VectorBlockT, CovectorBlockT]
    G: Any
    radius: Any
    resid_tol: Any
    status: Any
    prev_model: Any
    y: Any
    y_flip: Any
    lam: Any
    model_value: Any
    true_model_value: Any
    resid: Any
    n_negative: Any
    theta_min: Any
    metric_applies: Any
    metric_solves: Any
    precond_start: Any
    trace_model: Any
    trace_resid: Any
    trace_lam: Any
    trace_negative: Any


def lanczos_solver(max_iter: int, *, saddle_free: bool = False, stall_tol: float = 1e-3,
                   breakdown_tol: float = 1e-12, trust_region: bool = True,
                   curvature_floor: float = 1e-6) -> KrylovSolver:
    """The Lanczos solver as pieces: of the trust-region subproblem (see
    solve_tr_subproblem), or with trust_region=False of the Newton direction
    of a line search (no bound; truncated at nonpositive curvature, or the
    saddle-free model with curvature_floor; see solve_newton_direction).
    The M-Gram matrix is kept only when it is needed: with a separate P, for
    the trust region and for the saddle-free model."""

    def bound_of(c: _LanczosCarry[Any, Any, Any, Any]) -> int:
        return int(c.trace_model.shape[0])

    def gram(ops: SubproblemOperators[Any, Any]) -> bool:
        return ops.precond_solve is not None and (trust_region or saddle_free)

    def after_step(backend: Backend, space: VectorSpace[Any, Any, Any, Any],
                   ops: SubproblemOperators[Any, Any], c: _LanczosCarry[Any, Any, Any, Any],
                   state: LanczosState[Any, Any, Any, Any]) -> _LanczosCarry[Any, Any, Any, Any]:
        """The projected problem, its residual and the stopping tests after
        the process has taken its newest step."""
        xp, put = backend.xp, backend.index_update
        matched = ops.precond_solve is None
        k = state.steps
        T = tridiagonal(leading(backend, state.deltas, k), leading(backend, state.gammas, k),
                        backend=backend)
        G = leading(backend, c.G, k) if gram(ops) else None      # None: the identity, or not needed
        metric_apply = None if matched else ops.metric_apply
        if trust_region:
            y, y_flip, lam, model_value, true_model_value, theta = solve_projected_trs(
                T, G, state.gamma0, c.radius, saddle_free, backend=backend, k=k)
            resid, applies, solves = krylov_residual_norm(
                space, state, y, y_flip[k - 1], lam, G, metric_apply, ops.metric_solve,
                backend=backend, k=k)
            truncate = xp.asarray(False)
        else:
            y, y_flip, lam, model_value, true_model_value, theta = solve_projected_newton(
                T, G, state.gamma0, saddle_free, curvature_floor, backend=backend, k=k)

            def measure(_: Any) -> tuple[Any, Any, Any]:
                return krylov_residual_norm(space, state, y, y_flip[k - 1], None, None,
                                            metric_apply, ops.metric_solve, backend=backend, k=k)
            if saddle_free:
                truncate = xp.asarray(False)
                resid, applies, solves = measure(None)
            else:
                truncate = theta[0] <= 0.0
                # The truncated direction's residual is the previous step's (none
                # for the steepest-descent fallback); nothing new is measured.
                resid, applies, solves = run_cond(
                    backend, truncate,
                    lambda _: (xp.where(k == 1, xp.nan, c.resid), xp.asarray(0), xp.asarray(0)),
                    measure, None)
                y, model_value = _truncated(backend, truncate, k, state.gamma0, y, c.y, T)
                y_flip, true_model_value = y, model_value
        n_negative = xp.sum(theta < 0.0)
        stalled = ((lam > 0.0) & xp.isfinite(c.prev_model)
                   & (xp.abs(c.prev_model - model_value) <= stall_tol * xp.abs(model_value)))
        done = xp.where(truncate, NEGATIVE, xp.where(
            state.breakdown, BREAKDOWN,
            xp.where(resid <= c.resid_tol, TOL, xp.where(stalled, STALL, RUNNING))))
        status = _next_status(backend, done, k, bound_of(c), int(c.G.shape[0]))
        return dataclasses.replace(
            c, state=state, status=xp.where(state.gamma0 > 0.0, status, INVALID),
            prev_model=model_value, y=y, y_flip=y_flip, lam=lam, model_value=model_value,
            true_model_value=true_model_value, resid=resid, n_negative=n_negative,
            theta_min=theta[0], metric_applies=c.metric_applies + applies,
            metric_solves=c.metric_solves + solves,
            trace_model=put(c.trace_model, k - 1, model_value),
            trace_resid=put(c.trace_resid, k - 1, resid), trace_lam=put(c.trace_lam, k - 1, lam),
            trace_negative=put(c.trace_negative, k - 1, n_negative))

    def precond_of(ops: SubproblemOperators[Any, Any]) -> Callable[[Any], Any]:
        return ops.metric_solve if ops.precond_solve is None else ops.precond_solve

    def begin(backend: Backend, space: VectorSpace[Any, Any, Any, Any],
              ops: SubproblemOperators[Any, Any], g: Any, radius: Any, resid_tol: Any,
              start_vector: Any, capacity: Optional[int]) -> _LanczosCarry[Any, Any, Any, Any]:
        xp, put = backend.xp, backend.index_update
        if ops.precond_solve is not None and ops.metric_apply is None:
            raise ValueError("a separate preconditioner needs metric_apply (M) for the Gram matrix")
        precond = precond_of(ops)
        cap = initial_capacity(capacity, max_iter)
        state = lanczos_start(space, g, precond, start_vector, capacity=cap, backend=backend)
        bound = max_iter if space.dimension is None else min(max_iter, space.dimension(state.q))
        G = xp.eye(cap)
        applies = xp.asarray(0)
        if gram(ops):
            assert ops.metric_apply is not None
            G = put(G, (0, 0), space.pair_block(state.Q, ops.metric_apply(state.q), 1)[0])  # (q0, q0)_M
            applies = xp.asarray(1)
        zero = xp.asarray(0.0)
        c = _LanczosCarry(
            state=state, G=G, radius=xp.asarray(radius), resid_tol=xp.asarray(resid_tol),
            status=xp.asarray(RUNNING), prev_model=xp.asarray(xp.nan), y=xp.zeros(cap),
            y_flip=xp.zeros(cap), lam=zero, model_value=zero, true_model_value=zero, resid=zero,
            n_negative=xp.asarray(0), theta_min=zero, metric_applies=applies,
            metric_solves=xp.asarray(0), precond_start=xp.asarray(1 if start_vector is None else 0),
            trace_model=xp.zeros(bound), trace_resid=xp.zeros(bound), trace_lam=xp.zeros(bound),
            trace_negative=xp.zeros(bound))
        # The first step changes the shape of the process (v_prev appears), so
        # it is taken here rather than in the loop.
        first = lanczos_step_state(space, state, ops.hessvec, precond, breakdown_tol,
                                   project_covector=ops.project_covector, backend=backend)
        return after_step(backend, space, ops, c, first)

    def run(backend: Backend, space: VectorSpace[Any, Any, Any, Any],
            ops: SubproblemOperators[Any, Any], g: Any, c: _LanczosCarry[Any, Any, Any, Any]
            ) -> _LanczosCarry[Any, Any, Any, Any]:
        put = backend.index_update
        precond = precond_of(ops)

        def step(c: _LanczosCarry[Any, Any, Any, Any]) -> _LanczosCarry[Any, Any, Any, Any]:
            state, G, applies = c.state, c.G, c.metric_applies
            k = state.steps
            stepped = lanczos_step_state(space, state, ops.hessvec, precond, breakdown_tol,
                                         project_covector=ops.project_covector, backend=backend)
            if gram(ops):                          # the row of G of the vector that joined the basis
                assert ops.metric_apply is not None
                # Paired with the new basis (member k is that vector): reading the old
                # block too would keep it alive next to the updated one, and a compiler
                # would copy the basis to update it.
                column = space.pair_block(stepped.Q, ops.metric_apply(state.q_next), k + 1)
                G = put(put(G, (slice(None), k), column), k, column)
                applies = applies + 1
            return after_step(backend, space, ops,
                              dataclasses.replace(c, G=G, metric_applies=applies), stepped)

        # at most capacity steps before the basis is full (fewer skipped steps
        # for a loop built from scan)
        return run_loop(backend, lambda c: c.status != RUNNING, step, c,
                        min(bound_of(c), int(c.G.shape[0])))

    def grow(backend: Backend, space: VectorSpace[Any, Any, Any, Any],
             c: _LanczosCarry[Any, Any, Any, Any], capacity: int) -> _LanczosCarry[Any, Any, Any, Any]:
        xp, put = backend.xp, backend.index_update
        cap = min(capacity, bound_of(c))
        old = int(c.G.shape[0])
        return dataclasses.replace(
            c, status=xp.where(c.status == FULL, RUNNING, c.status),
            state=lanczos_grow(space, c.state, cap, backend=backend),
            G=put(xp.eye(cap), (slice(0, old), slice(0, old)), c.G),
            y=put(xp.zeros(cap), slice(0, c.y.shape[0]), c.y),
            y_flip=put(xp.zeros(cap), slice(0, c.y_flip.shape[0]), c.y_flip))

    def result(backend: Backend, space: VectorSpace[Any, Any, Any, Any],
               ops: SubproblemOperators[Any, Any], c: _LanczosCarry[Any, Any, Any, Any]
               ) -> tuple[Any, RawSubproblem]:
        xp = backend.xp
        matched = ops.precond_solve is None
        k = c.state.steps
        y, G = leading(backend, c.y, k), leading(backend, c.G, k)
        precond_calls = k + c.precond_start
        s = space.combine_vectors(c.state.Q, c.y, k)
        step_norm2, applies = y @ G @ y, c.metric_applies
        if not matched and not gram(ops):     # no Gram matrix (line search): one product with M
            assert ops.metric_apply is not None
            step_norm2, applies = space.pair(ops.metric_apply(s), s), applies + 1
        raw = RawSubproblem(
            method=xp.asarray(METHODS.index("lanczos")), krylov_steps=k, status=c.status,
            step_norm=xp.sqrt(xp.maximum(step_norm2, 0.0)), lam=c.lam, model_value=c.model_value,
            true_model_value=c.true_model_value, resid=c.resid, resid_tol=c.resid_tol,
            n_negative=c.n_negative, theta_min=c.theta_min, hessvec_calls=k,
            metric_solve_calls=c.metric_solves + precond_calls if matched else c.metric_solves,
            precond_solve_calls=xp.asarray(0) if matched else precond_calls,
            metric_apply_calls=applies, trace_model=c.trace_model,
            trace_resid=c.trace_resid, trace_lam=c.trace_lam, trace_negative=c.trace_negative,
            fallbacks=xp.asarray(0))
        return s, raw

    return KrylovSolver(name="lanczos", begin=begin, run=run, grow=grow, result=result,
                        capacity=lambda c: int(c.G.shape[0]))


# --- truncated conjugate gradients ----------------------------------------------------------


def boundary_fraction(xp: Any, s_norm2: Any, sp: Any, p_norm2: Any, radius: Any) -> Any:
    """The tau >= 0 with ||s + tau p||_M = radius, from ||s||_M^2, (s, p)_M and
    ||p||_M^2 (s inside the ball), in the array namespace xp. Written to
    avoid cancellation."""
    gap = xp.maximum(radius ** 2 - s_norm2, 0.0)
    root = xp.sqrt(sp ** 2 + p_norm2 * gap)
    positive = sp > 0.0
    return xp.where(positive, gap / xp.where(positive, sp + root, 1.0), (root - sp) / p_norm2)


@backend_state
@dataclass(frozen=True, kw_only=True)
class _CGCarry(Generic[VectorT, CovectorT]):
    """The truncated CG loop's carry: the CG state (whose s and model are the
    final step's once the step stops on the boundary), the radius and
    tolerance, the number of trace entries k, the status, and what the
    statistics need."""
    state: CGState[VectorT, CovectorT]
    radius: Any
    resid_tol: Any
    k: Any
    status: Any
    theta_min: Any
    lam: Any
    resid: Any
    n_negative: Any
    metric_start: Any
    trace_model: Any
    trace_resid: Any
    trace_lam: Any
    trace_negative: Any


def cg_solver(max_iter: int, *, trust_region: bool = True) -> KrylovSolver:
    """Truncated CG as pieces: Steihaug-Toint for the trust-region subproblem
    (see solve_tr_subproblem_cg), or with trust_region=False Newton-CG for
    the direction of a line search (no boundary; at nonpositive curvature it
    stops with the current iterate, or with p_0 = -P^{-1} g at the first
    step; see solve_newton_direction_cg). It keeps no basis, so it never
    fills up and grow is not needed."""

    def begin(backend: Backend, space: VectorSpace[Any, Any, Any, Any],
              ops: SubproblemOperators[Any, Any], g: Any, radius: Any, resid_tol: Any,
              start_vector: Any, capacity: Optional[int]) -> _CGCarry[Any, Any]:
        xp = backend.xp
        state: CGState[Any, Any] = cg_start(space, g, ops.metric_solve, start_vector, backend=backend)
        return _CGCarry(
            state=state, radius=xp.asarray(radius), resid_tol=xp.asarray(resid_tol),
            k=xp.asarray(0), status=xp.where(state.rz > 0.0, RUNNING, INVALID),
            theta_min=xp.asarray(xp.inf), lam=xp.asarray(0.0),
            resid=xp.sqrt(xp.maximum(state.rz, 0.0)), n_negative=xp.asarray(0),
            metric_start=xp.asarray(1 if start_vector is None else 0),
            trace_model=xp.zeros(max_iter), trace_resid=xp.zeros(max_iter),
            trace_lam=xp.zeros(max_iter), trace_negative=xp.zeros(max_iter))

    def run(backend: Backend, space: VectorSpace[Any, Any, Any, Any],
            ops: SubproblemOperators[Any, Any], g: Any, carry: _CGCarry[Any, Any]
            ) -> _CGCarry[Any, Any]:
        xp, put = backend.xp, backend.index_update
        nan, zero = xp.asarray(xp.nan), xp.asarray(0.0)

        def step(c: _CGCarry[Any, Any]) -> _CGCarry[Any, Any]:
            st, radius = c.state, c.radius
            Hp, curvature = cg_curvature(space, st, ops.hessvec)
            theta_min = xp.minimum(c.theta_min, curvature / st.p_norm2)
            positive = curvature > 0.0
            safe_curvature = xp.where(positive, curvature, 1.0)
            alpha = st.rz / safe_curvature
            leaves = (curvature <= 0.0) | (positive & (st.s_norm2 + 2.0 * alpha * st.sp
                                                        + alpha ** 2 * st.p_norm2 >= radius ** 2))
            first = c.k == 0

            def to_boundary(_: Any) -> _CGCarry[Any, Any]:
                """Stop where the segment crosses the boundary (or follow
                negative curvature to it)."""
                tau = boundary_fraction(xp, st.s_norm2, st.sp, st.p_norm2, radius)
                model = st.model - tau * st.rz + 0.5 * tau ** 2 * curvature
                negative = curvature <= 0.0
                n_negative = xp.where(negative, 1, 0)
                stopped = CGState(steps=st.steps, s=space.axpy_vector(tau, st.p, st.s), r=st.r,
                                  z=st.z, p=st.p, rz=st.rz, s_norm2=st.s_norm2, sp=st.sp,
                                  p_norm2=st.p_norm2, model=model)
                return dataclasses.replace(
                    c, state=stopped, k=c.k + 1, status=xp.where(negative, NEGATIVE, BOUNDARY),
                    theta_min=theta_min, lam=nan, resid=nan, n_negative=n_negative,
                    trace_model=put(c.trace_model, c.k, model),
                    trace_resid=put(c.trace_resid, c.k, nan), trace_lam=put(c.trace_lam, c.k, nan),
                    trace_negative=put(c.trace_negative, c.k, n_negative))

            def advance(_: Any) -> _CGCarry[Any, Any]:
                new = cg_advance(space, st, Hp, safe_curvature, ops.metric_solve,
                                 ops.project_covector)
                resid = xp.sqrt(xp.maximum(new.rz, 0.0))
                return dataclasses.replace(
                    c, state=new, k=c.k + 1, status=xp.where(resid <= c.resid_tol, TOL, RUNNING),
                    theta_min=theta_min, lam=zero, resid=resid, n_negative=xp.asarray(0),
                    trace_model=put(c.trace_model, c.k, new.model),
                    trace_resid=put(c.trace_resid, c.k, resid), trace_lam=put(c.trace_lam, c.k, zero),
                    trace_negative=put(c.trace_negative, c.k, zero))

            def truncate(_: Any) -> _CGCarry[Any, Any]:
                """Line search: stop at nonpositive curvature with the current
                iterate, or with p_0 = -P^{-1} g at the first step (whose
                residual is not measured)."""
                tau = xp.where(first, 1.0, 0.0)
                model = xp.where(first, -st.rz + 0.5 * curvature, st.model)
                stopped = CGState(steps=st.steps, s=space.axpy_vector(tau, st.p, st.s), r=st.r,
                                  z=st.z, p=st.p, rz=st.rz,
                                  s_norm2=xp.where(first, st.p_norm2, st.s_norm2), sp=st.sp,
                                  p_norm2=st.p_norm2, model=model)
                resid = xp.where(first, nan, c.resid)
                one = xp.asarray(1)
                return dataclasses.replace(
                    c, state=stopped, k=c.k + 1, status=xp.asarray(NEGATIVE), theta_min=theta_min,
                    lam=zero, resid=resid, n_negative=one,
                    trace_model=put(c.trace_model, c.k, model),
                    trace_resid=put(c.trace_resid, c.k, resid), trace_lam=put(c.trace_lam, c.k, zero),
                    trace_negative=put(c.trace_negative, c.k, one))

            if trust_region:
                return run_cond(backend, leaves, to_boundary, advance, None)
            return run_cond(backend, curvature <= 0.0, truncate, advance, None)

        return run_loop(backend, lambda c: c.status != RUNNING, step, carry, max_iter)

    def grow(backend: Backend, space: VectorSpace[Any, Any, Any, Any], c: _CGCarry[Any, Any],
             capacity: int) -> _CGCarry[Any, Any]:
        return c

    def result(backend: Backend, space: VectorSpace[Any, Any, Any, Any],
               ops: SubproblemOperators[Any, Any], c: _CGCarry[Any, Any]
               ) -> tuple[Any, RawSubproblem]:
        xp = backend.xp
        st = c.state
        on_boundary = ((c.status == BOUNDARY) | (c.status == NEGATIVE) if trust_region
                       else xp.asarray(False))
        raw = RawSubproblem(
            method=xp.asarray(METHODS.index("cg")), krylov_steps=c.k,
            status=xp.where(c.status == RUNNING, MAX, c.status),
            step_norm=xp.where(on_boundary, c.radius, xp.sqrt(xp.maximum(st.s_norm2, 0.0))),
            lam=c.lam, model_value=st.model, true_model_value=st.model, resid=c.resid,
            resid_tol=c.resid_tol, n_negative=c.n_negative, theta_min=c.theta_min,
            hessvec_calls=c.k, metric_solve_calls=st.steps + c.metric_start,
            precond_solve_calls=xp.asarray(0), metric_apply_calls=xp.asarray(0),
            trace_model=c.trace_model, trace_resid=c.trace_resid, trace_lam=c.trace_lam,
            trace_negative=c.trace_negative, fallbacks=xp.asarray(0))
        return st.s, raw

    return KrylovSolver(name="cg", begin=begin, run=run, grow=grow, result=result,
                        capacity=lambda c: None)


# --- the flexible solver --------------------------------------------------------------------


@backend_state
@dataclass(frozen=True, kw_only=True)
class _FlexibleCarry(Generic[VectorT, CovectorT, VectorBlockT, CovectorBlockT]):
    """The flexible solve: the blocks Q, M Q, H Q of the basis (k members),
    the projected Hessian T (capacity x capacity), the radius, tolerance and
    gamma0 = ||g||_{M^{-1}}, the status, the latest residual R and M^{-1} R,
    the latest solution of the projected problem and its statistics, the
    work counters, and the traces (length: the bound on the steps)."""
    Q: VectorBlockT
    MQ: CovectorBlockT
    HQ: CovectorBlockT
    T: Any
    k: Any
    gamma0: Any
    radius: Any
    resid_tol: Any
    status: Any
    R: CovectorT
    MinvR: VectorT
    y: Any
    y_flip: Any
    lam: Any
    model_value: Any
    true_model_value: Any
    resid: Any
    n_negative: Any
    theta_min: Any
    hessvec_calls: Any
    precond_calls: Any
    apply_calls: Any
    solve_calls: Any
    fallbacks: Any
    trace_model: Any
    trace_resid: Any
    trace_lam: Any
    trace_negative: Any


def flexible_solver(max_iter: int, *, saddle_free: bool = False, breakdown_tol: float = 1e-10,
                    trust_region: bool = True, curvature_floor: float = 1e-6) -> KrylovSolver:
    """The flexible solver as pieces: of the trust-region subproblem (see
    solve_tr_subproblem_flexible), or with trust_region=False of the Newton
    direction of a line search (no bound; truncated at nonpositive
    curvature, falling back to -M^{-1} g, its first basis vector, at the
    first step; or the saddle-free model with curvature_floor; see
    solve_newton_direction_flexible)."""

    def bound_of(c: _FlexibleCarry[Any, Any, Any, Any]) -> int:
        return int(c.trace_model.shape[0])

    def project(ops: SubproblemOperators[Any, Any], w: Any) -> Any:
        return w if ops.project_covector is None else ops.project_covector(w)

    def solve_projected(backend: Backend, space: VectorSpace[Any, Any, Any, Any],
                        ops: SubproblemOperators[Any, Any], g: Any,
                        c: _FlexibleCarry[Any, Any, Any, Any]) -> _FlexibleCarry[Any, Any, Any, Any]:
        """The projected problem on the current basis, its residual and the
        stopping tests."""
        xp, put = backend.xp, backend.index_update
        k = c.k
        cap = int(c.T.shape[0])
        T = leading(backend, c.T, k)              # the basis is M-orthonormal: G = I
        if trust_region:
            y, y_flip, lam, model_value, true_model_value, theta = solve_projected_trs(
                T, None, c.gamma0, c.radius, saddle_free, backend=backend, k=k)
        else:
            y, y_flip, lam, model_value, true_model_value, theta = solve_projected_newton(
                T, None, c.gamma0, saddle_free, curvature_floor, backend=backend, k=k)

        def measure(_: Any) -> tuple[Any, Any, Any, Any]:
            R = space.axpy_covector(1.0, g, space.combine_covectors(c.HQ, y_flip, k))
            if trust_region:
                R = run_cond(backend, lam > 0.0,
                             lambda R: space.axpy_covector(lam, space.combine_covectors(c.MQ, y, k), R),
                             lambda R: R, R)
            R = project(ops, R)
            MinvR = ops.metric_solve(R)
            return R, MinvR, xp.sqrt(xp.maximum(space.pair(R, MinvR), 0.0)), xp.asarray(1)
        if trust_region or saddle_free:
            truncate = xp.asarray(False)
            R, MinvR, resid, solves = measure(None)
        else:
            truncate = theta[0] <= 0.0
            # Truncated: the previous step's residual (none for the fallback);
            # nothing new is measured.
            R, MinvR, resid, solves = run_cond(
                backend, truncate,
                lambda _: (c.R, c.MinvR, xp.where(k == 1, xp.nan, c.resid), xp.asarray(0)),
                measure, None)
            y, model_value = _truncated(backend, truncate, k, c.gamma0, y, c.y, T)
            y_flip, true_model_value = y, model_value
        n_negative = xp.sum(theta < 0.0)
        done = xp.where(truncate, NEGATIVE, xp.where(resid <= c.resid_tol, TOL, RUNNING))
        status = _next_status(backend, done, k, bound_of(c), cap)
        return dataclasses.replace(
            c, status=xp.where(c.gamma0 > 0.0, status, INVALID), R=R, MinvR=MinvR, y=y,
            y_flip=y_flip, lam=lam, model_value=model_value, true_model_value=true_model_value,
            resid=resid, n_negative=n_negative, theta_min=theta[0], solve_calls=c.solve_calls + solves,
            trace_model=put(c.trace_model, k - 1, model_value),
            trace_resid=put(c.trace_resid, k - 1, resid), trace_lam=put(c.trace_lam, k - 1, lam),
            trace_negative=put(c.trace_negative, k - 1, n_negative))

    def begin(backend: Backend, space: VectorSpace[Any, Any, Any, Any],
              ops: SubproblemOperators[Any, Any], g: Any, radius: Any, resid_tol: Any,
              start_vector: Any, capacity: Optional[int]) -> _FlexibleCarry[Any, Any, Any, Any]:
        xp, put = backend.xp, backend.index_update
        assert ops.precond_solve is not None and ops.metric_apply is not None
        cap = initial_capacity(capacity, max_iter)
        Mg = ops.metric_solve(g) if start_vector is None else start_vector
        gamma0 = xp.sqrt(xp.maximum(space.pair(g, Mg), 0.0))
        inverse = 1.0 / xp.where(gamma0 > 0.0, gamma0, 1.0)
        q = space.scale_vector(inverse, Mg)
        Hq = project(ops, ops.hessvec(q))
        bound = max_iter if space.dimension is None else min(max_iter, space.dimension(q))
        zero = xp.asarray(0.0)
        c = _FlexibleCarry(
            Q=space.vector_block(q, cap),
            MQ=space.covector_block(space.scale_covector(inverse, g), cap),
            HQ=space.covector_block(Hq, cap), T=put(xp.zeros((cap, cap)), (0, 0), space.pair(Hq, q)),
            k=xp.asarray(1), gamma0=gamma0, radius=xp.asarray(radius),
            resid_tol=xp.asarray(resid_tol), status=xp.asarray(RUNNING),
            R=space.scale_covector(0.0, g), MinvR=space.scale_vector(0.0, q), y=xp.zeros(cap),
            y_flip=xp.zeros(cap), lam=zero, model_value=zero, true_model_value=zero, resid=zero,
            n_negative=xp.asarray(0), theta_min=zero, hessvec_calls=xp.asarray(1),
            precond_calls=xp.asarray(0), apply_calls=xp.asarray(0),
            solve_calls=xp.asarray(1 if start_vector is None else 0), fallbacks=xp.asarray(0),
            trace_model=xp.zeros(bound), trace_resid=xp.zeros(bound), trace_lam=xp.zeros(bound),
            trace_negative=xp.zeros(bound))
        return solve_projected(backend, space, ops, g, c)

    def run(backend: Backend, space: VectorSpace[Any, Any, Any, Any],
            ops: SubproblemOperators[Any, Any], g: Any, carry: _FlexibleCarry[Any, Any, Any, Any]
            ) -> _FlexibleCarry[Any, Any, Any, Any]:
        xp, put = backend.xp, backend.index_update
        assert ops.precond_solve is not None and ops.metric_apply is not None
        precond_solve, metric_apply = ops.precond_solve, ops.metric_apply

        def step(c: _FlexibleCarry[Any, Any, Any, Any]) -> _FlexibleCarry[Any, Any, Any, Any]:
            k = c.k

            def orthonormalize(candidate: Any, Mz: Any) -> tuple[Any, Any, Any]:
                """Two Gram-Schmidt passes in the M inner product; ok when the
                candidate was finite and kept more than breakdown_tol of its
                norm."""
                before = xp.sqrt(xp.abs(space.pair(Mz, candidate)))
                for _ in range(2):
                    coefficients = space.pair_block(c.Q, Mz, k)
                    candidate = space.axpy_vector(-1.0, space.combine_vectors(c.Q, coefficients, k),
                                                  candidate)
                    Mz = space.axpy_covector(-1.0, space.combine_covectors(c.MQ, coefficients, k), Mz)
                norm = xp.sqrt(xp.maximum(space.pair(Mz, candidate), 0.0))
                ok = (xp.isfinite(before) & (before > 0.0) & xp.isfinite(norm)
                      & (norm > breakdown_tol * before))
                inverse = 1.0 / xp.where(ok, norm, 1.0)
                return (space.scale_vector(inverse, candidate),
                        project(ops, space.scale_covector(inverse, Mz)), ok)

            z = precond_solve(c.R)
            first = orthonormalize(z, metric_apply(z))
            fallback = ~first[2]
            q_new, Mq_new, ok = run_cond(
                backend, fallback, lambda _: orthonormalize(c.MinvR, metric_apply(c.MinvR)),
                lambda _: first, None)
            counted = dataclasses.replace(
                c, precond_calls=c.precond_calls + 1,
                apply_calls=c.apply_calls + 1 + xp.where(fallback, 1, 0),
                fallbacks=c.fallbacks + xp.where(fallback, 1, 0))

            def add(_: Any) -> _FlexibleCarry[Any, Any, Any, Any]:
                Hq = project(ops, ops.hessvec(q_new))
                Q = space.set_vector(counted.Q, k, q_new)
                column = space.pair_block(Q, Hq, k + 1)
                return solve_projected(backend, space, ops, g, dataclasses.replace(
                    counted, Q=Q, MQ=space.set_covector(counted.MQ, k, Mq_new),
                    HQ=space.set_covector(counted.HQ, k, Hq),
                    T=put(put(counted.T, (slice(None), k), column), k, column), k=k + 1,
                    hessvec_calls=counted.hessvec_calls + 1))

            def broke_down(_: Any) -> _FlexibleCarry[Any, Any, Any, Any]:
                return dataclasses.replace(counted, status=xp.asarray(BREAKDOWN))

            return run_cond(backend, ok, add, broke_down, None)

        return run_loop(backend, lambda c: c.status != RUNNING, step, carry,
                        min(bound_of(carry), int(carry.T.shape[0])))

    def grow(backend: Backend, space: VectorSpace[Any, Any, Any, Any],
             c: _FlexibleCarry[Any, Any, Any, Any], capacity: int) -> _FlexibleCarry[Any, Any, Any, Any]:
        xp, put = backend.xp, backend.index_update
        cap = min(capacity, bound_of(c))
        old = int(c.T.shape[0])
        return dataclasses.replace(
            c, status=xp.where(c.status == FULL, RUNNING, c.status),
            Q=space.grow_vector_block(c.Q, cap), MQ=space.grow_covector_block(c.MQ, cap),
            HQ=space.grow_covector_block(c.HQ, cap),
            T=put(xp.zeros((cap, cap)), (slice(0, old), slice(0, old)), c.T),
            y=put(xp.zeros(cap), slice(0, c.y.shape[0]), c.y),
            y_flip=put(xp.zeros(cap), slice(0, c.y_flip.shape[0]), c.y_flip))

    def result(backend: Backend, space: VectorSpace[Any, Any, Any, Any],
               ops: SubproblemOperators[Any, Any], c: _FlexibleCarry[Any, Any, Any, Any]
               ) -> tuple[Any, RawSubproblem]:
        xp = backend.xp
        y = leading(backend, c.y, c.k)
        raw = RawSubproblem(
            method=xp.asarray(METHODS.index("flexible")), krylov_steps=c.k, status=c.status,
            step_norm=xp.sqrt(y @ y), lam=c.lam, model_value=c.model_value,
            true_model_value=c.true_model_value, resid=c.resid, resid_tol=c.resid_tol,
            n_negative=c.n_negative, theta_min=c.theta_min, hessvec_calls=c.hessvec_calls,
            metric_solve_calls=c.solve_calls, precond_solve_calls=c.precond_calls,
            metric_apply_calls=c.apply_calls, trace_model=c.trace_model,
            trace_resid=c.trace_resid, trace_lam=c.trace_lam, trace_negative=c.trace_negative,
            fallbacks=c.fallbacks)
        return space.combine_vectors(c.Q, c.y, c.k), raw

    return KrylovSolver(name="flexible", begin=begin, run=run, grow=grow, result=result,
                        capacity=lambda c: int(c.T.shape[0]))
