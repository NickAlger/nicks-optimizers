"""Low-rank matrix completion: trust-region Newton on the manifold of
rank-r matrices.

Recover an N x M matrix A from K randomly observed entries by fitting a
rank-r matrix,

    min_X  f(X) = 1/2 sum_{(i,j) in Omega} (X_ij - A_ij)^2   over rank-r X.

The test matrix is A = U_A diag(d) V_A^T with random orthonormal U_A, V_A
and geometrically decaying d. Random singular vectors spread the energy over
all entries (an incoherent matrix), so uniformly sampled entries determine
it. A has rank n > r, so the residual at the solution is not zero and the
curvature of the manifold enters the Hessian.

The rank-r matrices form a smooth manifold of dimension (N + M - r) r. The
trust-region method never adds points; it needs a retraction and a
representation of tangent vectors and covectors at the current point, and
that is what this file supplies. Nothing N x M is ever formed: every
operation costs O((N + M) r^2 + K r).

Points. X = U diag(s) V^T, its thin SVD: U (N x r) and V (M x r) have
orthonormal columns; U is the left-orthogonal factor, V the right-orthogonal
one.

Tangent vectors. A pair (dU, dV) with the gauge U^T dU = 0, standing for

    xi = dU V^T + U dV^T.

This is the two-factor case of the tensor-train tangent space: each term
varies one factor while the other stays orthogonal. Because U and V are
orthonormal and the gauge makes the two terms orthogonal,

    <xi, eta>_F = <dU, eU> + <dV, eV>:

the Frobenius inner product of the ambient matrices is the plain dot product
of the components. Each pair with U^T dU = 0 is exactly one tangent vector.

Covectors. The same pairs (gU, gV), acting on tangent vectors by the same
dot product. An ambient functional xi -> <Z, xi>_F becomes
((I - U U^T) Z V, Z^T U). In these coordinates the Frobenius metric is M = I:
M^{-1} maps a canonical covector (below) to the tangent vector with the same
components: euclidean_metric_solve, the direct sum of the leaves' Euclidean
inner products. So the trust region is a Frobenius ball of matrix changes,
and with P = M the Krylov solver is unpreconditioned. (M^{-1} keeps the
gauge because the optimizer only passes it canonical covectors, through
project_covector.)

Redundancy. The components (N r + M r numbers) outnumber the dimension by
r^2: the block U^T gU of a covector pairs to zero with every tangent vector.
Canonical covectors have U^T gU = 0, and gauge_projection restores that. It
is passed to the solver as project_covector, because the Krylov recurrences
amplify rounding errors in the invisible block (exponentially, for Lanczos);
within a few dozen steps they swamp the covectors and the Krylov basis leaves
the tangent space (see nicks_optimizers.krylov). Any representation of the tangent space
with O((N + M) r) numbers is redundant in this way.

Retraction. X + xi = [U, dU] [[diag(s), I], [I, 0]] [V, dV]^T has rank at
most 2r. QR factorizations of the two outer factors and an SVD of the
2r x 2r core give its SVD; keeping the top r terms gives the closest rank-r
matrix (the metric projection), a second-order retraction.

Hessian. Because the retraction is second order, the Hessian of the pullback
s -> f(R_X(s)) at s = 0 is the Riemannian Hessian,

    <H xi, xi> = ||P_Omega(xi)||^2 + 2 <R, dU diag(s)^{-1} dV_perp^T>,

with R = P_Omega(X - A) the residual (sparse, N x M) and
dV_perp = (I - V V^T) dV. The first term is the Gauss-Newton Hessian; the
second comes from the curvature of the manifold: it is indefinite and grows
like ||R|| / s_r.

Starting point. A random start has two problems. Its r components start at
an arbitrary common size and must shrink to the scale of the data; on the
way the fit can localize on a few rows and columns (a coherent, "spiky"
iterate) and stall. And components that have not been found yet have tiny
singular values, where the curvature term makes the Hessian strongly
indefinite and badly conditioned. Starting from X = 0 (zero_point) avoids
the first: the gauged representation is valid even there, and components
grow only as far as the data ask. Fitting ranks 1, 2, ..., r in turn, each
from the previous fit plus a component of size zero, avoids the second
(complete_by_rank_continuation). On the problem below (rank 8, 4 samples
per degree of freedom), direct rank-8 fits from a random start or from
X = 0 had not converged after 150 iterations and some 6000 Lanczos steps,
with errors 20 to 40 times the best rank-8 error; rank continuation needs
about 120 iterations and 300 CG steps and comes within 15% of it. (The
Lanczos solver, krylov="lanczos", takes about 90 iterations for a similar
number of Krylov steps and reaches the same fit.)

Run:  python examples/matrix_completion.py
"""

import dataclasses
from dataclasses import dataclass
from typing import NewType, cast

import numpy as np
import numpy.typing as npt

from typing import Literal

from nicks_optimizers import (
    Array,
    RunSummary,
    CovectorBlock,
    LeafBlock,
    VectorBlock,
    VectorSpace,
    euclidean_metric_solve,
    trust_region_newton,
    tuple_tree_space,
)


# --- types -------------------------------------------------------------------

IntArray = npt.NDArray[np.intp]
LowRank = NewType("LowRank", tuple[Array, Array, Array])     # (U, s, V): U diag(s) V^T
Tangent = NewType("Tangent", tuple[Array, Array])            # (dU, dV), U^T dU = 0
Cotangent = NewType("Cotangent", tuple[Array, Array])        # (gU, gV), canonical if U^T gU = 0
TangentBlock = tuple[LeafBlock[VectorBlock], LeafBlock[VectorBlock]]          # blocks of dU, dV
CotangentBlock = tuple[LeafBlock[CovectorBlock], LeafBlock[CovectorBlock]]
FixedRankSpace = VectorSpace[Tangent, Cotangent, TangentBlock, CotangentBlock]


@dataclass(frozen=True, kw_only=True)
class Samples:
    """Observed entries of an N x M matrix: values[k] at (rows[k], cols[k])."""
    shape: tuple[int, int]
    rows: IntArray
    cols: IntArray
    values: Array


# --- sparse helpers ----------------------------------------------------------


def rowwise_dot(F: Array, G: Array) -> Array:
    """sum_a F[k, a] G[k, a] for each row k."""
    dots: Array = np.einsum("ka,ka->k", F, G)
    return dots


def sampled_entries(x: LowRank, rows: IntArray, cols: IntArray) -> Array:
    """The entries X[rows[k], cols[k]] of X = U diag(s) V^T, without forming X."""
    U, s, V = x
    return rowwise_dot(np.take(U, rows, axis=0) * s, np.take(V, cols, axis=0))


def scatter_rows(index: IntArray, W: Array, n_rows: int) -> Array:
    """The n_rows x r matrix whose row i is the sum of the rows W[k] with
    index[k] = i. With W[k] = values[k] B[cols[k]] this is S B for the sparse
    matrix S with S[index[k], cols[k]] = values[k]."""
    summed: Array = np.column_stack(
        [np.bincount(index, weights=W[:, a], minlength=n_rows) for a in range(W.shape[1])])
    return summed


def project_out(B: Array, Z: Array) -> Array:
    """(I - B B^T) Z for B with orthonormal columns."""
    projected: Array = Z - B @ (B.T @ Z)
    return projected


# --- data --------------------------------------------------------------------


def random_low_rank(shape: tuple[int, int], d: Array, rng: np.random.Generator) -> LowRank:
    """U_A diag(d) V_A^T with random orthonormal U_A and V_A."""
    N, M = shape
    U = np.linalg.qr(rng.standard_normal((N, d.shape[0])))[0]
    V = np.linalg.qr(rng.standard_normal((M, d.shape[0])))[0]
    return LowRank((U, d, V))


def sample_uniformly(a: LowRank, K: int, rng: np.random.Generator) -> Samples:
    """K distinct entries of a, uniformly at random (sorted by row, which makes
    gathers and scatters cache-friendlier)."""
    N, M = a[0].shape[0], a[2].shape[0]
    rows, cols = np.divmod(np.sort(rng.choice(N * M, size=K, replace=False)), M)
    return Samples(shape=(N, M), rows=rows, cols=cols, values=sampled_entries(a, rows, cols))


def relative_error(x: LowRank, a: LowRank) -> float:
    """||X - A||_F / ||A||_F from the factors (works for any ranks)."""
    (U, s, V), (Ua, sa, Va) = x, a
    cross = float(np.sum(s[:, None] * (U.T @ Ua) * (V.T @ Va) * sa[None, :]))
    squared = float(s @ s) + float(sa @ sa) - 2.0 * cross
    return float(np.sqrt(max(squared, 0.0) / float(sa @ sa)))


# --- the vector space and the metric ------------------------------------------


def tangent_dimension(v: Tangent) -> int:
    """(N + M - r) r: the components number (N + M) r, less r^2 for the gauge."""
    return int((v[0].shape[0] + v[1].shape[0] - v[0].shape[1]) * v[0].shape[1])


def fixed_rank_space() -> FixedRankSpace:
    """Tangent vectors and covectors as tuple trees (dU, dV) of numpy
    matrices (tuple_tree_space), with the dimension of the tangent space."""
    space = cast(FixedRankSpace, tuple_tree_space())
    return dataclasses.replace(space, dimension=tangent_dimension)


def gauge_projection(x: LowRank, w: Cotangent) -> Cotangent:
    """The canonical representative of a covector: drop U^T gU, the block that
    pairs to zero with every tangent vector."""
    return Cotangent((project_out(x[0], w[0]), w[1]))


# --- the manifold --------------------------------------------------------------


def retract(x: LowRank, v: Tangent) -> LowRank:
    """The truncated SVD of X + xi, from a rank-2r factorization."""
    U, s, V = x
    dU, dV = v
    r = s.shape[0]
    QU, RU = np.linalg.qr(np.hstack([U, dU]))
    QV, RV = np.linalg.qr(np.hstack([V, dV]))
    eye, zero = np.eye(r), np.zeros((r, r))
    core = RU @ np.block([[np.diag(s), eye], [eye, zero]]) @ RV.T
    Y, sigma, Zt = np.linalg.svd(core)
    return LowRank((QU @ Y[:, :r], sigma[:r], QV @ Zt[:r].T))


# --- the objective -----------------------------------------------------------------
#
# Products with the sparse residual are scatter_rows of gathered factor rows;
# each function gathers each factor once.


def residual(samples: Samples, x: LowRank) -> Array:
    """P_Omega(X - A), as the values on the observed entries."""
    r: Array = sampled_entries(x, samples.rows, samples.cols) - samples.values
    return r


def objective(samples: Samples, x: LowRank) -> float:
    r = residual(samples, x)
    return 0.5 * float(r @ r)


def gradient(samples: Samples, x: LowRank) -> Cotangent:
    """xi -> <R, xi>_F as a canonical covector: ((I - U U^T) R V, R^T U)."""
    U, s, V = x
    N, M = samples.shape
    Ur, Vc = np.take(U, samples.rows, axis=0), np.take(V, samples.cols, axis=0)
    R = rowwise_dot(Ur * s, Vc) - samples.values
    RV = scatter_rows(samples.rows, R[:, None] * Vc, N)
    RtU = scatter_rows(samples.cols, R[:, None] * Ur, M)
    return Cotangent((project_out(U, RV), RtU))


def hessian_action(samples: Samples, x: LowRank, v: Tangent, curvature: bool = True) -> Cotangent:
    """H xi for the pullback Hessian (see the module docstring), as a canonical
    covector; curvature=False gives the Gauss-Newton Hessian.

    With Z = P_Omega(xi), the Gauss-Newton part is ((I - U U^T) Z V, Z^T U);
    the curvature part adds ((I - U U^T) R dV_perp S^{-1}, (I - V V^T) R^T dU S^{-1}),
    S = diag(s). The gauge projection is applied to the input as well as the
    output, so the operator is symmetric on all component pairs, not only on
    gauged ones. At a rank-deficient point (some s_a = 0, e.g. the start
    X = 0) the curvature of the manifold is unbounded and its term is dropped
    for those components: the model there is Gauss-Newton.
    """
    U, s, V = x
    dU, dV = project_out(U, v[0]), v[1]
    N, M = samples.shape
    rows, cols = samples.rows, samples.cols
    Ur, Vc = np.take(U, rows, axis=0), np.take(V, cols, axis=0)
    dUr, dVc = np.take(dU, rows, axis=0), np.take(dV, cols, axis=0)
    Z = rowwise_dot(dUr, Vc) + rowwise_dot(Ur, dVc)
    wU = Z[:, None] * Vc
    hV = scatter_rows(cols, Z[:, None] * Ur, M)
    if curvature:
        R = rowwise_dot(Ur * s, Vc) - samples.values
        inv_s = np.divide(1.0, s, out=np.zeros_like(s), where=s > 0.0)
        dVc_perp = dVc - Vc @ (V.T @ dV)                    # rows of (I - V V^T) dV
        wU = wU + R[:, None] * (dVc_perp * inv_s)
        hV = hV + project_out(V, scatter_rows(cols, R[:, None] * (dUr * inv_s), M))
    return Cotangent((project_out(U, scatter_rows(rows, wU, N)), hV))


# --- checks ---------------------------------------------------------------------


def random_point(shape: tuple[int, int], r: int, rng: np.random.Generator) -> LowRank:
    """A random rank-r point with unit singular values."""
    return random_low_rank(shape, np.ones(r), rng)


def zero_point(shape: tuple[int, int], r: int, rng: np.random.Generator) -> LowRank:
    """X = 0, with arbitrary orthonormal factors and s = 0.

    Degenerate, but the gauged representation is still valid there:
    (dU, dV) -> dU V^T + U dV^T is injective and isometric for any orthonormal
    U, V, and the retraction is defined. The first step moves into the
    interior of the manifold, and the components then grow only as far as
    the data ask, instead of having to shrink from an arbitrary size.
    """
    return random_low_rank(shape, np.zeros(r), rng)


def random_tangent(x: LowRank, rng: np.random.Generator) -> Tangent:
    """A random unit tangent vector at x."""
    U, s, V = x
    dU, dV = project_out(U, rng.standard_normal(U.shape)), rng.standard_normal(V.shape)
    norm = np.sqrt(np.vdot(dU, dU) + np.vdot(dV, dV))
    return Tangent((dU / norm, dV / norm))


def derivative_check(samples: Samples, x: LowRank, rng: np.random.Generator,
                     t: float = 1e-4) -> tuple[float, float, float]:
    """Relative errors of the gradient and the Hessian against central
    differences of phi(t) = f(retract(x, t xi)), and the Hessian's asymmetry
    |<H xi, eta> - <H eta, xi>| / |<H xi, eta>|."""
    sp = fixed_rank_space()
    xi, eta = random_tangent(x, rng), random_tangent(x, rng)
    phi = [objective(samples, retract(x, sp.scale_vector(c * t, xi))) for c in (-1.0, 0.0, 1.0)]
    g = gradient(samples, x)
    Hxi, Heta = hessian_action(samples, x, xi), hessian_action(samples, x, eta)
    slope, curv = sp.pair(g, xi), sp.pair(Hxi, xi)
    fd_slope = (phi[2] - phi[0]) / (2.0 * t)
    fd_curv = (phi[2] - 2.0 * phi[1] + phi[0]) / t**2
    return (abs(fd_slope - slope) / abs(slope), abs(fd_curv - curv) / abs(curv),
            abs(sp.pair(Hxi, eta) - sp.pair(Heta, xi)) / abs(sp.pair(Hxi, eta)))


# --- driver ---------------------------------------------------------------------


def complete(samples: Samples, x0: LowRank, saddle_free: bool = False, curvature: bool = True,
             project: bool = True, radius: float = 1.0, display: int = 0,
             max_iter: int = 200, max_krylov: int = 50, rtol: float = 1e-10,
             krylov: Literal["auto", "cg", "lanczos"] = "auto") -> tuple[LowRank, RunSummary]:
    """Fit a matrix of the rank of x0 to the samples, starting from x0 with
    the given trust-region radius (a Frobenius norm). curvature=False uses
    the Gauss-Newton Hessian; project=False omits project_covector (to see
    what goes wrong without it). With the defaults the subproblems are
    solved by truncated CG (P = M = I); krylov="lanczos" uses the
    reorthogonalized Lanczos solver instead."""
    return trust_region_newton(
        lambda x: objective(samples, x),
        lambda x: gradient(samples, x),
        lambda x, v: hessian_action(samples, x, v, curvature),
        x0, space=fixed_rank_space(), retract=retract, metric_solve=euclidean_metric_solve,
        project_covector=gauge_projection if project else None,
        saddle_free=saddle_free, krylov=krylov, radius=radius, display=display,
        max_iter=max_iter, max_krylov=max_krylov, rtol=rtol)


def add_zero_component(x: LowRank, rng: np.random.Generator) -> LowRank:
    """The same matrix as a point of rank r + 1: one more component of size
    zero, with arbitrary unit vectors u (orthogonal to U) and v (orthogonal
    to V). This is the zero start for the new component."""
    U, s, V = x
    u = project_out(U, rng.standard_normal((U.shape[0], 1)))
    v = project_out(V, rng.standard_normal((V.shape[0], 1)))
    return LowRank((np.hstack([U, u / np.linalg.norm(u)]), np.append(s, 0.0),
                    np.hstack([V, v / np.linalg.norm(v)])))


def complete_by_rank_continuation(samples: Samples, r: int, rng: np.random.Generator,
                                  stage_rtol: float = 1e-4, rtol: float = 1e-10,
                                  display: int = 0) -> list[tuple[LowRank, RunSummary]]:
    """Fit ranks 1, 2, ..., r in turn, starting from X = 0 and adding one zero
    component between stages. Intermediate ranks are solved to stage_rtol,
    the final one to rtol (with the given display level). Returns the point
    and summary of each stage.

    Each stage after the first starts with the trust-region radius equal to
    the smallest singular value found so far: the new component is not
    expected to be larger, and while it is tiny the curvature term makes the
    model unreliable over larger steps.
    """
    stages: list[tuple[LowRank, RunSummary]] = []
    x, radius = zero_point(samples.shape, 1, rng), 1.0
    for k in range(1, r + 1):
        if k > 1:
            radius = float(x[1][-1])
            x = add_zero_component(x, rng)
        x, summary = complete(samples, x, radius=radius, rtol=rtol if k == r else stage_rtol,
                              display=display if k == r else 0)
        stages.append((x, summary))
    return stages


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N, M, n, r = 1000, 800, 30, 8
    d: Array = 0.5 ** np.arange(n, dtype=float)
    a = random_low_rank((N, M), d, rng)
    dof = (N + M - r) * r
    K = 4 * dof
    samples = sample_uniformly(a, K, rng)

    def best_error(k: int) -> float:
        return float(np.sqrt(np.sum(d[k:] ** 2) / np.sum(d ** 2)))

    print(f"A: {N} x {M} of rank {n}, singular values 2^-k, random singular vectors")
    print(f"observed: K = {K} entries ({K / (N * M):.1%}), 4 per degree of freedom of rank {r}")
    print("derivative check at a random point (gradient, Hessian, asymmetry): "
          + "  ".join(f"{e:.1e}" for e in derivative_check(samples, random_point((N, M), r, rng), rng)))

    print(f"\n=== rank continuation from X = 0 up to rank {r}; the last stage in full ===")
    stages = complete_by_rank_continuation(samples, r, rng, display=1)
    print("\nrank  iterations  Krylov steps  ||X - A|| / ||A||  best rank-k error")
    for k, (x, summary) in enumerate(stages, start=1):
        steps = sum(rec.subproblem.krylov_steps for rec in summary.iterations)
        print(f"{k:4d}  {len(summary.iterations):10d}  {steps:12d}  {relative_error(x, a):17.2e}"
              f"  {best_error(k):17.2e}")
