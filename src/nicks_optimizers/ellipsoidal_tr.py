"""
Trust-region Newton method with a Hessian preconditioner, a separate
ellipsoidal trust region and an optional saddle-free model, using an
explicitly reorthogonalized Lanczos basis.

Representation
--------------
The method never touches the storage of points, vectors or covectors itself.
It works through a VectorSpace record: a frozen dataclass of functions (no
methods) saying how vectors are scaled, paired, collected into blocks and
combined. numpy_space, the default, uses plain numpy arrays. Other
representations (distributed objects, collections of arrays, GPU arrays, ...)
supply their own record:

  numpy_space                          one numpy array per vector
  numpy_space_with_reductions(allreduce)  one local numpy array per process
                                       (e.g. MPI ranks); only the reductions
                                       change
  tuple_block_space(...)               any vector type, from six functions;
                                       blocks are tuples of vectors
  dataclasses.replace(space, ...)      override individual entries

check_space(space, ...) verifies a record's internal consistency on sample
vectors. Vector and covector operations are separate fields on purpose, so a
space spells out which kind of object every function acts on.

Types
-----
  Point      an iterate x. Points are never added; a step s moves x to
             retract(x, s), by default space.translate(x, s) = x + s.
  Vector     a tangent (primal) vector at x: steps, Krylov vectors q_j.
  Covector   a cotangent (dual) vector at x: gradients, H v, residuals, the
             Lanczos vectors v_j = P q_j.
  blocks     collections of vectors or covectors (the Lanczos bases).
  Coords     small coordinate arrays (y, z, T, G, W): always numpy.

The core is generic over these types (type variables PointT, VectorT, ...).
For numpy_space they are the NewTypes Point, Vector, Covector, VectorBlock and
CovectorBlock, erased at run time: plain arrays work unchanged, and a static
checker (mypy, pyright) can catch primal/dual mix-ups if you wrap the arrays
your functions return (Covector(...) and so on). The same works for your own
representation if you define NewTypes over it.

Problem pieces
--------------
  fun(x) -> float                      objective
  grad(x) -> Covector                  differential df(x)
  hessvec(x, v) -> Covector            H(x) v, the Hessian of the pullback
                                       s -> f(retract(x, s)) at s = 0
  precond_solve(x, w) -> Vector        P(x)^{-1} w, P(x) symmetric positive
                                       definite; only shapes the Krylov space
  ball_inner(x, U, v) -> Coords        the M(x) inner products (u_i, v)_M of
                                       each vector u_i in the block U with v.
                                       The trust region at x is
                                       { s : (s, s)_M <= radius^2 }. For an
                                       ellipsoid with covariance B(x),
                                       (u, v)_M = <B(x)^{-1} v, u>.
  ball_dual_inner(x, Psi, phi) -> Coords  the dual inner products
                                       (psi_i, phi)_{M^{-1}} = <psi_i, B(x) phi>;
                                       used for dual norms (stopping tests).

H and P^{-1} are the only operators the method applies; the geometry enters
only through the two inner products, which it uses to build Gram matrices.
Both inner products are required, and they must be exact duals of each other,
(M u, M v)_{M^{-1}} = (u, v)_M: the residual norm on the boundary of the trust
region relies on that identity. euclidean_inner and euclidean_dual_inner are
such a pair for numpy_space and M = I. Passing a block lets an implementation
apply M once per call, e.g. lambda x, U, v: U.T @ (M @ v). In a distributed
setting the inner products, like the space's reductions, must return the
global (reduced) values.

The radius carries over from one point to the next, so the theory wants M(x)
uniformly equivalent to a fixed norm over the iterates.

Subproblem
----------
Each outer iteration builds a P-orthonormal Lanczos basis Q of the Krylov
space K_k(P^{-1} H, P^{-1} g), with full reorthogonalization. P only chooses
the space; the geometry comes from M. With T = <H Q, Q> (tridiagonal) and
G = [(q_i, q_j)_M] = L L^T, the matrix A = L^{-1} T L^{-T} is H in the
M-orthonormal basis Q L^{-T}, so its eigenpairs are the M-Ritz pairs
(Rayleigh-Ritz for the pencil H u = theta M u). In those coordinates z the
ball is round and the small problem is

    min_z  b^T z + 0.5 z^T A_model z    s.t.  ||z|| <= radius,

with b = gamma0 L^{-1} e_1, and A_model = A, or |A| (eigenvalues replaced by
their absolute values) for the saddle-free model. The step is s = Q L^{-T} z.

Each Lanczos step costs one Hessian-vector product, one preconditioner solve,
one call to each inner product (with a block of all earlier basis vectors),
and O(k) vector operations for reorthogonalization. Storage is the two blocks
Q and V = P Q. Nothing else is applied to the small problem or the stopping
test.

Caveat when M differs from P: if the trust region is active, the exact step
solves (H + lam M) s = -g, which K_k(P^{-1} H, P^{-1} g) approximates slowly
unless M is close to a multiple of P. Steps always beat the Cauchy point
along -P^{-1} g, so global convergence is unaffected, but boundary solves stop
when the model stops improving (stall_tol) rather than on the residual.

Diagnostics
-----------
The solver only produces records; formatting and printing are separate.

  SubproblemStats   one per step computation: Lanczos steps, why Lanczos
                    stopped, ||s||_M, multiplier, model values, residual,
                    negative M-Ritz values, call counts, per-step trace
  IterationRecord   one per outer iteration: f, ||g||, radius, predicted and
                    actual reduction, rho, acceptance, and its SubproblemStats
  RunSummary        returned at the end: stopping reason, all IterationRecords,
                    final values, total work
  IterationInfo     what callback(x, info) receives after each iteration:
                    the IterationRecord plus f, g, the step s and the
                    ResumeState
  ResumeState       the state carried between iterations besides x (radius,
                    gnorm0, iteration count, and f for a consistency check):
                    save it with x in the callback to resume a run exactly
                    with resume=state (and g0=g to skip one gradient)

Records hold only numbers and tuples, so keeping them all is cheap; vectors
go only to the callback. format_header, format_row, format_lanczos_trace and
format_summary turn records into text; display=1 or 2 prints them as the run
goes, through print_fn.
"""

import dataclasses
import time
from dataclasses import dataclass
from typing import Callable, Generic, Literal, NewType, Optional, TypeVar, cast

import numpy as np
import numpy.typing as npt


# ---------------------------------------------------------------------------
# Vector spaces: how the objects are represented
# ---------------------------------------------------------------------------

Array = npt.NDArray[np.float64]
Coords = Array                                       # small coordinate arrays

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
    numpy_space_with_reductions, tuple_block_space) and change entries with
    dataclasses.replace. Every entry is required except dimension.

      scale_vector(a, v) -> Vector           a v
      scale_covector(a, w) -> Covector       a w
      axpy_covector(a, w, z) -> Covector     a w + z
      pair(w, v) -> float                    the duality pairing <w, v>
      pair_block(B, w) -> Coords             <w, b_i> for each vector b_i in B
      vector_block(v) -> VectorBlock         the one-vector block [v]
      covector_block(w) -> CovectorBlock     the one-covector block [w]
      append_vector(B, v) -> VectorBlock     a new block [B, v]
      append_covector(C, w) -> CovectorBlock a new block [C, w]
      combine_vectors(B, c) -> Vector        sum_i c_i b_i
      combine_covectors(C, c) -> Covector    sum_i c_i w_i
      translate(x, s) -> Point               x + s, the flat-space point update
                                             (the default retraction)
      dimension(v) -> int                    optional: the dimension n, used to
                                             cap the number of Lanczos steps

    The method never modifies an object in place, and it only calls these
    functions; pairings are the only places where numbers come out of vectors.
    """
    scale_vector: Callable[[float, VectorT], VectorT]
    scale_covector: Callable[[float, CovectorT], CovectorT]
    axpy_covector: Callable[[float, CovectorT, CovectorT], CovectorT]
    pair: Callable[[CovectorT, VectorT], float]
    pair_block: Callable[[VectorBlockT, CovectorT], Coords]
    vector_block: Callable[[VectorT], VectorBlockT]
    covector_block: Callable[[CovectorT], CovectorBlockT]
    append_vector: Callable[[VectorBlockT, VectorT], VectorBlockT]
    append_covector: Callable[[CovectorBlockT, CovectorT], CovectorBlockT]
    combine_vectors: Callable[[VectorBlockT, Coords], VectorT]
    combine_covectors: Callable[[CovectorBlockT, Coords], CovectorT]
    translate: Callable[[PointT, VectorT], PointT]
    dimension: Optional[Callable[[VectorT], int]] = None


# --- numpy representation --------------------------------------------------

Point = NewType("Point", Array)
Vector = NewType("Vector", Array)
Covector = NewType("Covector", Array)
VectorBlock = NewType("VectorBlock", Array)          # n x k, columns are Vectors
CovectorBlock = NewType("CovectorBlock", Array)      # n x k, columns are Covectors

NumpySpace = VectorSpace[Point, Vector, Covector, VectorBlock, CovectorBlock]

# Signatures of the problem pieces for numpy_space (documentation aliases).
Objective = Callable[[Point], float]
Gradient = Callable[[Point], Covector]
HessianAction = Callable[[Point, Vector], Covector]
PreconditionerSolve = Callable[[Point, Covector], Vector]
BallInner = Callable[[Point, VectorBlock, Vector], Coords]
BallDualInner = Callable[[Point, CovectorBlock, Covector], Coords]
Retraction = Callable[[Point, Vector], Point]
VectorToCovector = Callable[[Vector], Covector]
CovectorToVector = Callable[[Covector], Vector]
BlockInner = Callable[[VectorBlock, Vector], Coords]
BlockDualInner = Callable[[CovectorBlock, Covector], Coords]


def _np_scale_vector(a: float, v: Vector) -> Vector:
    return Vector(a * v)


def _np_scale_covector(a: float, w: Covector) -> Covector:
    return Covector(a * w)


def _np_axpy_covector(a: float, w: Covector, z: Covector) -> Covector:
    return Covector(a * w + z)


def _np_pair(w: Covector, v: Vector) -> float:
    return float(w @ v)


def _np_pair_block(B: VectorBlock, w: Covector) -> Coords:
    return B.T @ w


def _np_vector_block(v: Vector) -> VectorBlock:
    return VectorBlock(v[:, None])


def _np_covector_block(w: Covector) -> CovectorBlock:
    return CovectorBlock(w[:, None])


def _np_append_vector(B: VectorBlock, v: Vector) -> VectorBlock:
    return VectorBlock(np.column_stack((B, v)))


def _np_append_covector(C: CovectorBlock, w: Covector) -> CovectorBlock:
    return CovectorBlock(np.column_stack((C, w)))


def _np_combine_vectors(B: VectorBlock, c: Coords) -> Vector:
    return Vector(B @ c)


def _np_combine_covectors(C: CovectorBlock, c: Coords) -> Covector:
    return Covector(C @ c)


def _np_translate(x: Point, s: Vector) -> Point:
    return Point(x + s)


def _np_dimension(v: Vector) -> int:
    return int(v.shape[0])


numpy_space: NumpySpace = VectorSpace(
    scale_vector=_np_scale_vector, scale_covector=_np_scale_covector,
    axpy_covector=_np_axpy_covector, pair=_np_pair, pair_block=_np_pair_block,
    vector_block=_np_vector_block, covector_block=_np_covector_block,
    append_vector=_np_append_vector, append_covector=_np_append_covector,
    combine_vectors=_np_combine_vectors, combine_covectors=_np_combine_covectors,
    translate=_np_translate, dimension=_np_dimension)


def euclidean_inner(x: Point, U: VectorBlock, v: Vector) -> Coords:
    """(u, v)_M with M = I, for numpy_space. Pass it together with
    euclidean_dual_inner, its exact dual, to get the round trust region."""
    return U.T @ v


def euclidean_dual_inner(x: Point, Psi: CovectorBlock, phi: Covector) -> Coords:
    """(psi, phi)_{M^{-1}} with M = I: the exact dual of euclidean_inner."""
    return Psi.T @ phi


def numpy_space_with_reductions(allreduce: Callable[[Coords], Coords]) -> NumpySpace:
    """numpy_space for data stored as one local numpy array per process
    (e.g. MPI ranks).

    Local operations stay numpy; the reductions (pair, pair_block, dimension)
    are completed with allreduce, which must return the elementwise sum of its
    argument over all processes, identical on every process. Every decision
    the method makes comes from reduced numbers, so the processes stay in
    lockstep. Your inner products must do their own reduction.
    """
    def pair(w: Covector, v: Vector) -> float:
        return float(allreduce(np.array([float(w @ v)]))[0])

    def pair_block(B: VectorBlock, w: Covector) -> Coords:
        return allreduce(B.T @ w)

    def dimension(v: Vector) -> int:
        return int(round(float(allreduce(np.array([float(v.shape[0])]))[0])))

    return dataclasses.replace(numpy_space, pair=pair, pair_block=pair_block, dimension=dimension)


def tuple_block_space(*, scale_vector: Callable[[float, VectorT], VectorT],
                      scale_covector: Callable[[float, CovectorT], CovectorT],
                      axpy_vector: Callable[[float, VectorT, VectorT], VectorT],
                      axpy_covector: Callable[[float, CovectorT, CovectorT], CovectorT],
                      pair: Callable[[CovectorT, VectorT], float],
                      translate: Callable[[PointT, VectorT], PointT],
                      dimension: Optional[Callable[[VectorT], int]] = None
                      ) -> VectorSpace[PointT, VectorT, CovectorT,
                                       tuple[VectorT, ...], tuple[CovectorT, ...]]:
    """A complete space for any vector type, from six functions.

    Blocks are tuples of vectors (or covectors), and the block entries are
    built from the functions given: pair_block pairs one vector at a time and
    combine_* accumulate with scale and axpy. That is correct for any
    representation; for distributed objects it costs one reduction per block
    member, so override pair_block (and combine_*) with dataclasses.replace
    when a batched version is available. Your inner products then receive
    tuples as their block argument.
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
        axpy_covector=axpy_covector, pair=pair, pair_block=pair_block,
        vector_block=vector_block, covector_block=covector_block,
        append_vector=append_vector, append_covector=append_covector,
        combine_vectors=combine_vectors, combine_covectors=combine_covectors,
        translate=translate, dimension=dimension)


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
    ]
    scale = max(abs(p11), abs(p12), abs(p21), abs(p22), 1e-300)
    return max(errors) / scale


# ---------------------------------------------------------------------------
# Norms and Gram matrices from the inner products
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Preconditioned Lanczos
# ---------------------------------------------------------------------------

def lanczos_start(space: VectorSpace[PointT, VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                  g: CovectorT, precond_solve: Callable[[CovectorT], VectorT]
                  ) -> tuple[CovectorT, VectorT, float]:
    """First Lanczos pair.

    Returns (v0, q0, gamma0) with g = gamma0 v0, q0 = P^{-1} v0 and
    <v0, q0> = <P q0, q0> = 1. gamma0 = ||g||_{P^{-1}}.
    """
    y = precond_solve(g)
    gPg = space.pair(g, y)
    if not gPg > 0.0:
        raise ValueError("<g, P^{-1} g> must be positive; is P positive definite and g nonzero?")
    gamma0 = float(np.sqrt(gPg))
    return space.scale_covector(1.0 / gamma0, g), space.scale_vector(1.0 / gamma0, y), gamma0


def reorthogonalize(space: VectorSpace[PointT, VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                    t: CovectorT, Q: VectorBlockT, V: CovectorBlockT, passes: int = 2) -> CovectorT:
    """Make P^{-1} t P-orthogonal to the vectors in Q.

    The pairs (q_i, v_i = P q_i) satisfy <v_i, q_j> = delta_ij, and
    <P q_i, P^{-1} t> = <t, q_i>, so it suffices to drive the pairings
    <t, q_i> to zero, which needs no preconditioner solves. Two passes of
    classical Gram-Schmidt ("twice is enough").
    """
    for _ in range(passes):
        t = space.axpy_covector(-1.0, space.combine_covectors(V, space.pair_block(Q, t)), t)
    return t


def lanczos_step(space: VectorSpace[PointT, VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                 hessvec: Callable[[VectorT], CovectorT],
                 precond_solve: Callable[[CovectorT], VectorT],
                 Q: VectorBlockT, V: CovectorBlockT, q: VectorT, v: CovectorT,
                 v_prev: Optional[CovectorT], gamma_prev: float,
                 reorth_passes: int = 2) -> tuple[float, float, CovectorT, VectorT]:
    """One step of P-preconditioned Lanczos with full reorthogonalization.

    q, v are the newest pair (the last members of Q, V) and v_prev the one
    before it (None at the first step). Forms the covector
        t = H q - delta v - gamma_prev v_prev,   delta = <H q, q>,
    reorthogonalizes it, and returns (delta, gamma_next, v_next, q_next) with
    gamma_next = ||t||_{P^{-1}}, v_next = t / gamma_next and
    q_next = P^{-1} t / gamma_next. On exact breakdown the vectors are zero.
    Costs one Hessian-vector product and one preconditioner solve.
    """
    h = hessvec(q)
    delta = space.pair(h, q)
    t = space.axpy_covector(-delta, v, h)
    if v_prev is not None:
        t = space.axpy_covector(-gamma_prev, v_prev, t)
    t = reorthogonalize(space, t, Q, V, reorth_passes)
    y = precond_solve(t)
    gamma_next = float(np.sqrt(max(space.pair(t, y), 0.0)))
    if gamma_next == 0.0:
        return delta, 0.0, space.scale_covector(0.0, t), space.scale_vector(0.0, y)
    return (delta, gamma_next, space.scale_covector(1.0 / gamma_next, t),
            space.scale_vector(1.0 / gamma_next, y))


# ---------------------------------------------------------------------------
# Small dense helpers (coordinates only)
# ---------------------------------------------------------------------------

def tridiagonal(diag: Coords, offdiag: Coords) -> Coords:
    """Dense symmetric tridiagonal matrix. Uses the first len(diag)-1 off-diagonals."""
    k = len(diag)
    T: Coords = np.diag(np.asarray(diag, dtype=float))
    if k > 1:
        off = np.asarray(offdiag[: k - 1], dtype=float)
        T = T + np.diag(off, 1) + np.diag(off, -1)
    return T


def grow_gram(G: Coords, col: Coords) -> Coords:
    """Add one row and column to the symmetric matrix G.

    col has length G.shape[0] + 1; its last entry is the new diagonal entry.
    """
    off = col[:-1, None]
    grown: Coords = np.block([[G, off], [off.T, col[-1:, None]]])
    return grown


# ---------------------------------------------------------------------------
# Small trust-region problems (coordinates only)
# ---------------------------------------------------------------------------

def solve_sphere_trs_eig(evals: Coords, U: Coords, b: Coords, radius: float,
                         rtol: float = 1e-12, max_iter: int = 200,
                         hard_tol: float = 1e-10) -> tuple[Coords, float]:
    """Solve  min_z b^T z + 0.5 z^T A z  s.t.  ||z|| <= radius,  given the
    eigendecomposition A = U diag(evals) U^T with evals in ascending order.

    Safeguarded Newton on the secular equation. Two details keep it accurate:
      * it solves the unit-ball problem with b / radius and rescales, so very
        small or large radii do not underflow or overflow;
      * it iterates on the shifted multiplier mu = lam + evals[0], so the
        bottom term b_0 / mu keeps full relative accuracy when the root sits
        very close to the pole (the near-hard case).
    Returns (z, lam) with (A + lam I) z = -b, lam >= 0, A + lam I positive
    semidefinite and lam (||z|| - radius) = 0, including the hard case.
    """
    lam_min = float(evals[0])
    gaps = evals - lam_min                  # >= 0, and gaps[0] == 0
    bh = (U.T @ b) / radius                 # unit-ball problem, z = radius * u
    bnorm = float(np.linalg.norm(bh))

    # Interior: A positive definite and the Newton step lies inside the ball.
    if lam_min > 0.0:
        u = -bh / evals
        if np.linalg.norm(u) <= 1.0:
            return radius * (U @ u), 0.0

    # Hard case: b (numerically) orthogonal to the bottom eigenspace, and the
    # solution at lam = -lam_min is too short to reach the boundary.
    if lam_min <= 0.0:
        scale_ = max(1.0, float(np.max(np.abs(evals))))
        bottom = gaps <= hard_tol * scale_
        if np.linalg.norm(bh[bottom]) <= hard_tol * bnorm:
            u = np.zeros_like(bh)
            rest = ~bottom
            u[rest] = -bh[rest] / gaps[rest]
            short = float(np.linalg.norm(u))
            if short < 1.0:
                tau = float(np.sqrt(1.0 - short**2))
                u[0] = -tau if bh[0] > 0.0 else tau  # sign that lowers b^T z
                return radius * (U @ u), max(0.0, -lam_min)

    # Boundary: find mu in (lo, hi] with ||u(mu)|| = 1, u(mu) = -bh / (gaps + mu).
    # lam >= 0 and lam >= -lam_min mean mu >= max(0, lam_min); and
    # ||u(mu)|| <= ||bh|| / mu, so mu = ||bh|| is on the feasible side.
    lo = max(0.0, lam_min)
    hi = max(lo, bnorm)
    mu = hi
    for _ in range(max_iter):
        w = bh / (gaps + mu)
        unorm = float(np.linalg.norm(w))
        if abs(unorm - 1.0) <= rtol:
            break
        if unorm > 1.0:
            lo = mu
        else:
            hi = mu
        if hi - lo <= 1e-15 * hi:
            break
        phi = 1.0 / unorm - 1.0
        dphi = float(np.sum(w**2 / (gaps + mu))) / unorm**3
        mu_new = mu - phi / dphi
        if not (lo < mu_new < hi):
            mu_new = 0.5 * (lo + hi)
        mu = mu_new
    u = -bh / (gaps + mu)
    return radius * (U @ u), max(0.0, mu - lam_min)


def solve_sphere_trs(A: Coords, b: Coords, radius: float) -> tuple[Coords, float]:
    """Solve  min_z b^T z + 0.5 z^T A z  s.t.  ||z|| <= radius  (small, dense)."""
    evals, U = np.linalg.eigh(A)
    return solve_sphere_trs_eig(evals, U, b, radius)


def m_ritz(T: Coords, G: Coords) -> tuple[Coords, Coords, Coords]:
    """Rayleigh-Ritz for the pencil H u = theta M u on the Krylov space.

    With G = L L^T, Q L^{-T} is an M-orthonormal basis of the space and
    A = L^{-1} T L^{-T} is H in that basis. Returns (L, theta, Z): the
    M-Ritz values theta (ascending) and the M-Ritz vectors Z in z-coordinates.
    """
    L = np.linalg.cholesky(G)
    A = np.linalg.solve(L, np.linalg.solve(L, T).T)
    theta, Z = np.linalg.eigh(0.5 * (A + A.T))
    return L, theta, Z


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
    L, theta, Z = m_ritz(T, G)
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


def kkt_residual_dual_norm(y: Coords, y_flip_last: float, lam: float, G: Coords,
                           W: Coords, gamma_next: float) -> float:
    """M^{-1}-norm of the residual R = H s_flip + lam M s + g, from Gram
    matrices only.

    The Lanczos relation H Q = V T + gamma_next v_next e_k^T and the projected
    equations give
        R = lam (M Q y - V G y) + gamma_next y_flip[-1] v_next.
    For the plain model s_flip = s, and this is the exact KKT residual; for
    the saddle-free model it is the surrogate for |H|_M s + lam M s + g. The
    covector lam M Q y is never formed: with W = [(v_i, v_j)_{M^{-1}}] over
    V_{k+1}, <V, Q> = I and <v_next, Q> = 0, the duality identities
    (M u, M w)_{M^{-1}} = (u, w)_M and (M u, psi)_{M^{-1}} = <psi, u> give
        ||R||^2 = beta^T W beta - lam^2 y^T G y,  beta = (-lam G y, gamma_next y_flip[-1]).
    This is why the two inner products must be exact duals. For lam = 0 only
    W enters, and any dual inner product gives a valid (if different) norm.

    Accuracy: for lam = 0 (interior step) there is no subtraction and the
    value is accurate to rounding. For lam > 0 the subtraction gives a floor
    of roughly sqrt(eps * ||G W G||) * lam * ||y|| (about 1e-7 relative to
    ||g|| in the tests, with cond(M) = 1e3). Tight tolerances arise near
    convergence, where the trust region is inactive, so the floor rarely
    binds; if it does, store M^{-1} V and form M^{-1} R explicitly.
    """
    beta = np.append(-lam * (G @ y), gamma_next * y_flip_last)
    squared = float(beta @ W @ beta - lam**2 * (y @ G @ y))
    return float(np.sqrt(max(squared, 0.0)))


# ---------------------------------------------------------------------------
# Diagnostics records
# ---------------------------------------------------------------------------

StopReason = Literal["tol", "stall", "breakdown", "max"]
RunReason = Literal["converged", "radius", "max_iter", "callback"]


@dataclass(frozen=True, kw_only=True)
class SubproblemStats:
    """Diagnostics of one trust-region subproblem solve: scalars and tuples
    only, so records stay small however long the run.

      lanczos_steps     number of Lanczos steps k
      stop_reason       "tol" (residual met resid_tol), "stall" (boundary
                        active and the model stopped changing), "breakdown"
                        (invariant subspace: the subspace solution is exact) or
                        "max" (max_lanczos or the dimension)
      step_norm, lam    ||s||_M and the multiplier (> 0 when the boundary is active)
      model_value       <g,s> + 0.5 <H_model s, s>, the model that was minimized
      true_model_value  <g,s> + 0.5 <H s, s> (differs only with saddle_free)
      resid, resid_tol  final ||R||_{M^{-1}} and the tolerance it was tested against
      n_negative, theta_min  number of negative M-Ritz values, smallest one
      *_calls           operator and inner-product calls made by this solve
      trace_*           per Lanczos step: model value, residual, multiplier,
                        number of negative M-Ritz values
    """
    lanczos_steps: int
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
    precond_calls: int
    inner_calls: int
    dual_inner_calls: int
    trace_model: tuple[float, ...]
    trace_resid: tuple[float, ...]
    trace_lam: tuple[float, ...]
    trace_negative: tuple[int, ...]


@dataclass(frozen=True, kw_only=True)
class IterationRecord:
    """What happened in one outer iteration, which started from the iterate x_k.

      iteration          k
      f, gnorm           f(x_k) and ||g(x_k)||_{M^{-1}}
      gnorm_ratio        ||g(x_k)|| / gnorm0
      radius, new_radius trust-region radius used for this step, and after the update
      forcing            relative Krylov tolerance used (resid_tol = forcing * gnorm)
      predicted          model decrease m(0) - m(s) for the model that was minimized
      actual             f(x_k) - f(retract(x_k, s))
      rho                reduction ratio (with the rounding safeguard)
      accepted           whether x_{k+1} = retract(x_k, s)
      elapsed            seconds since the start of the run, at the end of this iteration
      subproblem         SubproblemStats of the step computation
    """
    iteration: int
    f: float
    gnorm: float
    gnorm_ratio: float
    radius: float
    new_radius: float
    forcing: float
    predicted: float
    actual: float
    rho: float
    accepted: bool
    elapsed: float
    subproblem: SubproblemStats


@dataclass(frozen=True, kw_only=True)
class ResumeState:
    """Everything, besides the iterate x, that the method carries from one
    iteration to the next. With x, the same options and deterministic
    functions, trust_region_newton(..., x, resume=state) continues exactly as
    the original run would have.

      radius          the trust-region radius for the next iteration
      gnorm0          the reference gradient norm (rtol test, forcing term)
      next_iteration  the number of the next iteration; max_iter stays a
                      budget for the whole run, and records keep their numbering
      f               f(x) at the checkpoint, used to detect a state that does
                      not belong to the x it is resumed from

    Plain numbers only, so dataclasses.asdict / ResumeState(**d) round-trip
    through JSON or any other format exactly. P, M, f and g are recomputed
    from x, and the Lanczos basis is rebuilt every iteration, so nothing else
    needs saving; pass g0 to skip recomputing the gradient.
    """
    radius: float
    gnorm0: float
    next_iteration: int
    f: float


@dataclass(frozen=True, kw_only=True)
class IterationInfo(Generic[VectorT, CovectorT]):
    """What the callback receives besides the iterate x after an iteration.

      record   the IterationRecord of the iteration just finished (the same
               object that is kept in the RunSummary)
      f, g     objective and gradient at x, already computed by the method
      step     the step s that was tried in this iteration, accepted or not
      resume   the ResumeState at x: save it with x (and g, to pass as g0) to
               be able to continue the run later

    x, f and g describe the point the next iteration starts from; record
    describes the iteration that led there. The method never modifies objects
    in place, so a callback may keep references without copying.
    """
    record: IterationRecord
    f: float
    g: CovectorT
    step: VectorT
    resume: ResumeState


@dataclass(frozen=True, kw_only=True)
class RunSummary:
    """The outcome of a run.

      reason        "converged" (||g|| <= max(gtol, rtol gnorm0)), "radius"
                    (radius below radius_min), "max_iter", or "callback"
      iterations    one IterationRecord per iteration
      f, gnorm      at the returned point; gnorm0 is the reference norm
      accepted_steps
      *_calls       work done over the whole run
      elapsed       seconds
      resume        the ResumeState at the returned point, for continuing the
                    run (e.g. after max_iter) with resume=summary.resume
    """
    reason: RunReason
    iterations: tuple[IterationRecord, ...]
    f: float
    gnorm: float
    gnorm0: float
    accepted_steps: int
    fun_calls: int
    grad_calls: int
    hessvec_calls: int
    precond_calls: int
    inner_calls: int
    dual_inner_calls: int
    elapsed: float
    resume: ResumeState


def merge_summaries(first: RunSummary, second: RunSummary) -> RunSummary:
    """Combine the summaries of a run that was stopped and then resumed, where
    second started from first's final point and state. Records are joined,
    work and time are added, and the outcome is second's. The work includes
    the restart's extra objective (and gradient, without g0) evaluation."""
    return RunSummary(
        reason=second.reason, iterations=first.iterations + second.iterations,
        f=second.f, gnorm=second.gnorm, gnorm0=first.gnorm0,
        accepted_steps=first.accepted_steps + second.accepted_steps,
        fun_calls=first.fun_calls + second.fun_calls,
        grad_calls=first.grad_calls + second.grad_calls,
        hessvec_calls=first.hessvec_calls + second.hessvec_calls,
        precond_calls=first.precond_calls + second.precond_calls,
        inner_calls=first.inner_calls + second.inner_calls,
        dual_inner_calls=first.dual_inner_calls + second.dual_inner_calls,
        elapsed=first.elapsed + second.elapsed, resume=second.resume)


def history_array(summary: RunSummary) -> Array:
    """The iterations as a numeric array, one row per iteration, with columns
    (iteration, f, ||g||, radius, rho, lanczos_steps, accepted, n_negative)."""
    rows = [(r.iteration, r.f, r.gnorm, r.radius, r.rho, r.subproblem.lanczos_steps,
             float(r.accepted), r.subproblem.n_negative) for r in summary.iterations]
    history: Array = np.array(rows, dtype=float).reshape(-1, 8)
    return history


# ---------------------------------------------------------------------------
# Subproblem driver
# ---------------------------------------------------------------------------

def boundary_stalled(lam: float, model_value: float, prev_model_value: Optional[float],
                     stall_tol: float) -> bool:
    """True when the trust region is active and the last Lanczos step changed
    the model value by less than stall_tol (relative).

    With M != P the residual on the boundary contains lam M Q y - lam V G y,
    which the Krylov space of P^{-1} H only reduces slowly, so boundary solves
    stop on stagnation of the model value instead. The change is taken in
    absolute value because the saddle-free model itself changes with k.
    """
    if lam <= 0.0 or prev_model_value is None:
        return False
    return abs(prev_model_value - model_value) <= stall_tol * abs(model_value)


def solve_tr_subproblem(space: VectorSpace[PointT, VectorT, CovectorT, VectorBlockT, CovectorBlockT],
                        g: CovectorT, hessvec: Callable[[VectorT], CovectorT],
                        precond_solve: Callable[[CovectorT], VectorT],
                        ball_inner: Callable[[VectorBlockT, VectorT], Coords],
                        ball_dual_inner: Callable[[CovectorBlockT, CovectorT], Coords],
                        radius: float, resid_tol: float, max_iter: int,
                        g_dual_norm: Optional[float] = None, stall_tol: float = 1e-3,
                        saddle_free: bool = False, breakdown_tol: float = 1e-12
                        ) -> tuple[VectorT, SubproblemStats]:
    """Approximately solve  min <g, s> + 0.5 <H_model s, s>  s.t.  ||s||_M <= radius
    over the growing Krylov spaces K_k(P^{-1} H, P^{-1} g), where H_model is
    H, or |H|_M (via M-Ritz pairs) when saddle_free.

    The operators and inner products are frozen at the current point; the
    inner products take a block as their first argument (see the module
    docstring) and must be exact duals. g_dual_norm = ||g||_{M^{-1}} may be
    passed in to save one call.

    Stops at the first of:
      * ||R||_{M^{-1}} <= resid_tol, with R the (surrogate) KKT residual
        (see kkt_residual_dual_norm),
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
    dual_calls = 0
    if g_dual_norm is None:
        g_dual_norm = covector_norm(space, ball_dual_inner, g)
        dual_calls += 1
    v, q, gamma0 = lanczos_start(space, g, precond_solve)
    if space.dimension is not None:
        max_iter = min(max_iter, space.dimension(q))
    Q, V = space.vector_block(q), space.covector_block(v)
    G = ball_inner(Q, q).reshape(1, 1)                   # (q0, q0)_M
    W = np.array([[(g_dual_norm / gamma0) ** 2]])        # (v0, v0)_{M^{-1}}
    v_prev: Optional[CovectorT] = None
    deltas, gammas = np.zeros(0), np.zeros(0)
    hscale = 0.0
    prev_model_value: Optional[float] = None
    trace_model: list[float] = []
    trace_resid: list[float] = []
    trace_lam: list[float] = []
    trace_negative: list[int] = []

    for k in range(1, max_iter + 1):
        gamma_prev = float(gammas[-1]) if gammas.size else 0.0
        delta, gamma_next, v_next, q_next = lanczos_step(
            space, hessvec, precond_solve, Q, V, q, v, v_prev, gamma_prev)
        deltas = np.append(deltas, delta)
        hscale = max(hscale, abs(delta), gamma_prev)
        breakdown = gamma_next <= breakdown_tol * hscale
        if breakdown:
            gamma_next = 0.0
            W_next = grow_gram(W, np.zeros(k + 1))
        else:
            W_next = extend_dual_gram(space, W, V, v_next, ball_dual_inner)
            dual_calls += 1

        T = tridiagonal(deltas, gammas)
        y, y_flip, lam, model_value, true_model_value, theta = solve_projected_trs(
            T, G, gamma0, radius, saddle_free)
        resid = kkt_residual_dual_norm(y, float(y_flip[-1]), lam, G, W_next, gamma_next)
        n_negative = int(np.sum(theta < 0.0))
        trace_model.append(model_value)
        trace_resid.append(resid)
        trace_lam.append(lam)
        trace_negative.append(n_negative)
        stalled = boundary_stalled(lam, model_value, prev_model_value, stall_tol)
        if resid <= resid_tol or stalled or breakdown or k == max_iter:
            reason: StopReason = ("breakdown" if breakdown else "tol" if resid <= resid_tol
                                  else "stall" if stalled else "max")
            stats = SubproblemStats(
                lanczos_steps=k, stop_reason=reason,
                step_norm=float(np.sqrt(max(float(y @ G @ y), 0.0))), lam=lam,
                model_value=model_value, true_model_value=true_model_value,
                resid=resid, resid_tol=resid_tol, n_negative=n_negative,
                theta_min=float(theta[0]),
                hessvec_calls=k, precond_calls=k + 1, inner_calls=k, dual_inner_calls=dual_calls,
                trace_model=tuple(trace_model), trace_resid=tuple(trace_resid),
                trace_lam=tuple(trace_lam), trace_negative=tuple(trace_negative))
            return space.combine_vectors(Q, y), stats

        prev_model_value = model_value
        gammas = np.append(gammas, gamma_next)
        G = extend_metric_gram(space, G, Q, q_next, ball_inner)
        Q = space.append_vector(Q, q_next)
        V = space.append_covector(V, v_next)
        W = W_next
        v_prev, v, q = v, v_next, q_next
    raise AssertionError("unreachable: the loop returns at k == max_iter")


# ---------------------------------------------------------------------------
# Outer trust-region loop
# ---------------------------------------------------------------------------

def reduction_ratio(f_old: float, f_trial: float, pred: float) -> float:
    """rho = actual reduction / predicted reduction, with a rounding safeguard.

    Near convergence both reductions fall to the level of rounding error in f,
    and the raw ratio becomes noise that can reject good steps and collapse
    the radius. The standard fix (see Conn, Gould & Toint, Trust-Region
    Methods) adds delta = 10 eps max(1, |f_old|) to both, so rho tends to 1
    in that regime and is unaffected otherwise.
    """
    if not (pred > 0.0 and np.isfinite(f_trial)):
        return -np.inf
    delta = 10.0 * float(np.finfo(float).eps) * max(1.0, abs(f_old))
    return ((f_old - f_trial) + delta) / (pred + delta)


def update_radius(radius: float, rho: float, step_norm: float, radius_max: float) -> float:
    """Standard radius update, measuring the step in the M-norm."""
    if rho < 0.25:
        return 0.25 * step_norm
    if rho > 0.75 and step_norm >= 0.99 * radius:
        return min(2.0 * radius, radius_max)
    return radius


def gradient_converged(gnorm: float, gnorm0: float, gtol: float, rtol: float) -> bool:
    """Newton convergence test:  ||g|| <= max(gtol, rtol * ||g0||)."""
    return gnorm <= max(gtol, rtol * gnorm0)


def forcing_term(gnorm: float, gnorm0: float, forcing_max: float) -> float:
    """Relative Krylov tolerance min(forcing_max, sqrt(||g|| / ||g0||)).

    Using the ratio makes it invariant to scaling f; it tends to zero as the
    gradient does, which gives superlinear local convergence.
    """
    if gnorm0 <= 0.0:
        return forcing_max
    return min(forcing_max, float(np.sqrt(gnorm / gnorm0)))


def build_summary(reason: RunReason, records: list[IterationRecord], f: float, gnorm: float,
                  gnorm0: float, fun_calls: int, grad_calls: int, outer_dual_calls: int,
                  elapsed: float, resume: ResumeState) -> RunSummary:
    """Assemble the RunSummary, adding up the work recorded per iteration."""
    subs = [r.subproblem for r in records]
    return RunSummary(
        reason=reason, iterations=tuple(records), f=f, gnorm=gnorm, gnorm0=gnorm0,
        accepted_steps=sum(r.accepted for r in records),
        fun_calls=fun_calls, grad_calls=grad_calls,
        hessvec_calls=sum(s.hessvec_calls for s in subs),
        precond_calls=sum(s.precond_calls for s in subs),
        inner_calls=sum(s.inner_calls for s in subs),
        dual_inner_calls=outer_dual_calls + sum(s.dual_inner_calls for s in subs),
        elapsed=elapsed, resume=resume)


def check_resume_matches(state: ResumeState, f_x0: float, rtol: float = 1e-10) -> None:
    """Raise if the ResumeState was saved at a different point than x0.

    f(x0) is evaluated at startup anyway, so comparing it with the f stored in
    the state costs nothing. It catches an x and a state (or gradient) saved at
    different accepted iterates; it cannot tell apart states saved before and
    after a rejected step, since x and f do not change there.
    """
    if abs(f_x0 - state.f) > rtol * max(1.0, abs(state.f)):
        raise ValueError(
            f"resume state does not belong to x0: f(x0) = {f_x0!r}, but the state was saved at "
            f"f = {state.f!r}. Save x, the state (and g) from the same callback call.")


def trust_region_newton(
        fun: Callable[[PointT], float],
        grad: Callable[[PointT], CovectorT],
        hessvec: Callable[[PointT, VectorT], CovectorT],
        precond_solve: Callable[[PointT, CovectorT], VectorT],
        ball_inner: Callable[[PointT, VectorBlockT, VectorT], Coords],
        ball_dual_inner: Callable[[PointT, CovectorBlockT, CovectorT], Coords],
        x0: PointT,
        space: Optional[VectorSpace[PointT, VectorT, CovectorT, VectorBlockT, CovectorBlockT]] = None,
        retract: Optional[Callable[[PointT, VectorT], PointT]] = None,
        saddle_free: bool = False,
        radius: float = 1.0, radius_max: float = 1e4,
        radius_min: float = 1e-14, eta: float = 1e-4,
        gtol: float = 1e-8, rtol: float = 0.0,
        gnorm0: Optional[float] = None, forcing_max: float = 0.5,
        stall_tol: float = 1e-3, max_iter: int = 500, max_lanczos: int = 50,
        display: int = 0, print_fn: Callable[[str], None] = print,
        callback: Optional[Callable[[PointT, IterationInfo[VectorT, CovectorT]], Optional[bool]]] = None,
        resume: Optional[ResumeState] = None,
        g0: Optional[CovectorT] = None,
        ) -> tuple[PointT, RunSummary]:
    """Inexact Newton trust-region method with an ellipsoidal trust region.

    The operators and inner products are described in the module docstring.
    All of them are evaluated at the current iterate, and the radius is
    measured in the M(x)-norm of that iterate. There is no default for the
    inner products: they must be exact duals, and a default dual that ignored
    ball_inner would silently break that. For numpy_space and M = I, pass
    euclidean_inner and euclidean_dual_inner.

    space says how points, vectors and covectors are represented (default
    numpy_space), and x0 must already be a point in that representation; it
    is not copied or converted. retract defaults to space.translate.

    Gradient norms are dual norms, ||g|| = ||g||_{M(x)^{-1}}. With saddle_free, the model
    Hessian is |H|_{M}, approximated through the M-Ritz pairs of the Krylov
    space, so saddle points repel the iterates instead of attracting them.

    The Krylov solve stops when the (surrogate) residual satisfies
    ||R|| <= forcing * ||g||, with forcing = forcing_term(...), or, while the
    trust region is active, when the model stops changing (stall_tol).

    Convergence: stops when ||g|| <= max(gtol, rtol * gnorm0). gnorm0 defaults
    to ||g(x0)|| and can be passed explicitly, e.g. when restarting, so that
    rtol and the forcing term still refer to the original start. Also stops
    when the radius falls below radius_min, after max_iter iterations, or
    when the callback returns True.

    Diagnostics: display = 0 prints nothing, 1 prints one row per iteration
    and a summary, 2 also prints each subproblem's Lanczos trace. Output goes
    through print_fn (e.g. print only on one MPI rank). callback(x, info), if
    given, is called after every iteration with the current iterate and an
    IterationInfo; returning True stops the run.

    Resuming: resume=state, with a ResumeState saved together with x0 (from
    info.resume in the callback, or summary.resume), restores the radius,
    gnorm0 and the iteration count, and overrides the radius and gnorm0
    arguments (use dataclasses.replace on the state to change them). With the
    same options and deterministic functions the run continues exactly as if
    it had not stopped; max_iter remains the budget for the whole run. The
    state is checked against f(x0) (see check_resume_matches). g0, the
    gradient at x0 (e.g. info.g saved with x), skips the initial gradient
    evaluation; it must belong to x0.

    Returns (x, summary): the final iterate and a RunSummary, which holds the
    stopping reason, one IterationRecord per iteration, the work done and the
    ResumeState at x. history_array(summary) gives the iterations as a numeric
    array; merge_summaries joins the summaries of a stopped and resumed run.
    """
    start = time.perf_counter()
    sp = space if space is not None else cast(
        VectorSpace[PointT, VectorT, CovectorT, VectorBlockT, CovectorBlockT], numpy_space)
    retraction = retract if retract is not None else sp.translate
    x = x0
    fx = fun(x)
    fun_calls, outer_dual_calls = 1, 1
    it = 0
    if resume is not None:
        check_resume_matches(resume, fx)
        radius, gnorm0, it = resume.radius, resume.gnorm0, resume.next_iteration
    if g0 is not None:
        g, grad_calls = g0, 0
    else:
        g, grad_calls = grad(x), 1
    gnorm = covector_norm(sp, lambda Psi, phi: ball_dual_inner(x, Psi, phi), g)
    if gnorm0 is None:
        gnorm0 = gnorm
    records: list[IterationRecord] = []
    while True:
        if gradient_converged(gnorm, gnorm0, gtol, rtol):
            reason: RunReason = "converged"
            break
        if radius < radius_min:
            reason = "radius"
            break
        if it >= max_iter:
            reason = "max_iter"
            break
        forcing = forcing_term(gnorm, gnorm0, forcing_max)
        s, stats = solve_tr_subproblem(
            sp, g,
            lambda v: hessvec(x, v),
            lambda w: precond_solve(x, w),
            lambda U, v: ball_inner(x, U, v),
            lambda Psi, phi: ball_dual_inner(x, Psi, phi),
            radius, forcing * gnorm, max_lanczos,
            g_dual_norm=gnorm, stall_tol=stall_tol, saddle_free=saddle_free)

        x_trial = retraction(x, s)
        f_trial = fun(x_trial)
        fun_calls += 1
        predicted = -stats.model_value
        rho = reduction_ratio(fx, f_trial, predicted)
        accepted = rho > eta
        new_radius = update_radius(radius, rho, stats.step_norm, radius_max)
        record = IterationRecord(
            iteration=it, f=fx, gnorm=gnorm,
            gnorm_ratio=gnorm / gnorm0 if gnorm0 > 0.0 else 0.0,
            radius=radius, new_radius=new_radius, forcing=forcing,
            predicted=predicted, actual=fx - f_trial, rho=rho, accepted=accepted,
            elapsed=time.perf_counter() - start, subproblem=stats)
        records.append(record)
        if accepted:
            x, fx = x_trial, f_trial
            g = grad(x)
            grad_calls += 1
            gnorm = covector_norm(sp, lambda Psi, phi: ball_dual_inner(x, Psi, phi), g)
            outer_dual_calls += 1
        radius = new_radius
        it += 1
        if display >= 1:
            display_iteration(record, display, print_fn)
        if callback is not None:
            state = ResumeState(radius=radius, gnorm0=gnorm0, next_iteration=it, f=fx)
            if callback(x, IterationInfo(record=record, f=fx, g=g, step=s, resume=state)):
                reason = "callback"
                break

    final_state = ResumeState(radius=radius, gnorm0=gnorm0, next_iteration=it, f=fx)
    summary = build_summary(reason, records, fx, gnorm, gnorm0, fun_calls, grad_calls,
                            outer_dual_calls, time.perf_counter() - start, final_state)
    if display >= 1:
        print_fn(format_summary(summary))
    return x, summary


# ---------------------------------------------------------------------------
# Formatting and display (pure functions of the records, plus one printer)
# ---------------------------------------------------------------------------

_ROW = ("{it:>4}  {f:>13}  {g:>8}  {gr:>8}  {r:>8}  {sr:>5}  {pred:>9}  {ared:>9}  "
        "{rho:>7}  {acc:>3}  {k:>3}  {stop:>9}  {neg:>3}")

_RUN_REASONS: dict[str, str] = {
    "converged": "converged, ||g|| <= max(gtol, rtol * gnorm0)",
    "radius": "trust-region radius fell below radius_min",
    "max_iter": "reached max_iter",
    "callback": "stopped by the callback",
}


def format_header() -> str:
    """Column headings for format_row."""
    return _ROW.format(it="it", f="f", g="|g|", gr="|g|/|g0|", r="radius", sr="|s|/r",
                       pred="pred", ared="actual", rho="rho", acc="acc", k="k",
                       stop="stop", neg="neg")


def format_row(record: IterationRecord) -> str:
    """One line per iteration. |s|/r = 1.00 means the step reached the
    trust-region boundary; pred and actual are the model and true decreases;
    k and stop describe the Lanczos solve; neg counts negative M-Ritz values."""
    s = record.subproblem
    return _ROW.format(
        it=record.iteration, f=f"{record.f:.6e}", g=f"{record.gnorm:.1e}",
        gr=f"{record.gnorm_ratio:.1e}", r=f"{record.radius:.2e}",
        sr=f"{s.step_norm / record.radius:.2f}", pred=f"{record.predicted:.2e}",
        ared=f"{record.actual:.2e}", rho=f"{record.rho:+.3f}",
        acc="yes" if record.accepted else "no", k=s.lanczos_steps, stop=s.stop_reason,
        neg=s.n_negative)


def format_lanczos_trace(stats: SubproblemStats) -> str:
    """The per-step history of one subproblem solve, indented under its row."""
    pad = " " * 8
    lines = [f"{pad}{'step':>4}  {'model':>11}  {'resid':>9}  {'lam':>9}  {'neg':>3}"]
    for j, (m, r, lam, neg) in enumerate(zip(stats.trace_model, stats.trace_resid,
                                             stats.trace_lam, stats.trace_negative), start=1):
        lines.append(f"{pad}{j:>4}  {m:11.4e}  {r:9.2e}  {lam:9.2e}  {neg:3d}")
    last = (f"{pad}stop: {stats.stop_reason} (resid tol {stats.resid_tol:.2e}), "
            f"smallest M-Ritz value {stats.theta_min:.2e}")
    if stats.true_model_value != stats.model_value:
        last += f", true-model value {stats.true_model_value:.4e}"
    lines.append(last)
    return "\n".join(lines)


def format_summary(summary: RunSummary) -> str:
    """The end-of-run report: stopping reason, final values and total work."""
    ratio = summary.gnorm / summary.gnorm0 if summary.gnorm0 > 0.0 else 0.0
    return "\n".join([
        f"Stopped: {_RUN_REASONS[summary.reason]}",
        f"  iterations {len(summary.iterations)} ({summary.accepted_steps} accepted), "
        f"{summary.elapsed:.2f} s",
        f"  f = {summary.f:.6e}   |g| = {summary.gnorm:.2e}   |g|/|g0| = {ratio:.2e}",
        f"  work: {summary.hessvec_calls} Hessian-vector products, "
        f"{summary.precond_calls} preconditioner solves, {summary.inner_calls} ball_inner "
        f"and {summary.dual_inner_calls} ball_dual_inner calls, {summary.fun_calls} objective "
        f"and {summary.grad_calls} gradient evaluations",
    ])


def display_iteration(record: IterationRecord, level: int, print_fn: Callable[[str], None],
                      header_every: int = 20) -> None:
    """Print one iteration: the header every header_every rows, the row, and
    at level 2 the Lanczos trace."""
    if record.iteration % header_every == 0:
        print_fn(format_header())
    print_fn(format_row(record))
    if level >= 2:
        print_fn(format_lanczos_trace(record.subproblem))
