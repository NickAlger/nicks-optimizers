"""The trust-region method on a manifold: rank-r matrix completion, from
examples/matrix_completion.py."""

from typing import Callable, Literal, Optional

import numpy as np
import pytest

import matrix_completion as mc
from nicks_optimizers import (
    Array, euclidean_metric_solve, lanczos_advance, lanczos_start, lanczos_tridiagonal,
)


def small_problem(seed: int, shape: tuple[int, int], r: int, oversampling: float
                  ) -> tuple[mc.LowRank, mc.Samples, np.random.Generator]:
    """A rank-10 matrix with singular values 2^-k, and K = oversampling * dim
    uniformly sampled entries, dim = (N + M - r) r."""
    rng = np.random.default_rng(seed)
    a = mc.random_low_rank(shape, 0.5 ** np.arange(10, dtype=float), rng)
    K = int(oversampling * (shape[0] + shape[1] - r) * r)
    return a, mc.sample_uniformly(a, K, rng), rng


def test_derivatives_match_retraction_curves() -> None:
    """Gradient and Hessian against central differences of f(retract(x, t xi)).
    At a point with unit singular values the agreement is close to rounding.
    At a point with small singular values the curve bends sharply (third
    derivatives ~ 1/s^2), so the test is the O(t^2) rate of the difference
    errors. The Hessian is symmetric at both."""
    a, samples, rng = small_problem(0, (40, 30), 3, 4.0)
    x = mc.random_point((40, 30), 3, rng)
    g_err, h_err, asym = mc.derivative_check(samples, x, np.random.default_rng(6), t=1e-4)
    assert g_err < 1e-8 and h_err < 1e-5 and asym < 1e-12

    # Step sizes where truncation (not rounding) dominates, so the rate is
    # clean on any hardware.
    x = mc.random_low_rank((40, 30), np.array([1.0, 0.1, 0.01]), rng)
    coarse = mc.derivative_check(samples, x, np.random.default_rng(6), t=1e-3)
    fine = mc.derivative_check(samples, x, np.random.default_rng(6), t=1e-4)
    assert coarse[0] / fine[0] > 50.0 and fine[0] < 1e-4          # gradient: t^2
    assert coarse[1] / fine[1] > 50.0 and fine[1] < 1e-2          # Hessian: t^2
    assert fine[2] < 1e-12


def test_tangent_coordinates_are_isometric() -> None:
    """The component dot product equals the Frobenius inner product of the
    ambient matrices dU V^T + U dV^T, and pairing with the gradient is the
    ambient pairing <R, xi>."""
    a, samples, rng = small_problem(1, (40, 30), 3, 4.0)
    x = mc.random_point((40, 30), 3, rng)
    U, s, V = x
    xi, eta = mc.random_tangent(x, rng), mc.random_tangent(x, rng)

    def ambient(v: mc.Tangent) -> Array:
        matrix: Array = v[0] @ V.T + U @ v[1].T
        return matrix

    sp = mc.fixed_rank_space()
    assert abs(sp.pair(mc.Cotangent(xi), eta) - float(np.sum(ambient(xi) * ambient(eta)))) < 1e-14
    R = np.zeros((40, 30))
    R[samples.rows, samples.cols] = mc.residual(samples, x)
    g = mc.gradient(samples, x)
    assert abs(sp.pair(g, xi) - float(np.sum(R * ambient(xi)))) < 1e-13


def test_retraction_is_second_order() -> None:
    """R(x, 0) = X with orthonormal factors, and R(x, t xi) - X - t xi has a
    tangent component of order t^3 (the second-order term is normal)."""
    a, samples, rng = small_problem(2, (40, 30), 3, 4.0)
    x = mc.random_low_rank((40, 30), np.array([2.0, 1.0, 0.5]), rng)
    U, s, V = x
    X = (U * s) @ V.T
    xi = mc.random_tangent(x, rng)
    xi_matrix = xi[0] @ V.T + U @ xi[1].T
    y = mc.retract(x, mc.fixed_rank_space().scale_vector(0.0, xi))
    assert np.max(np.abs((y[0] * y[1]) @ y[2].T - X)) < 1e-14
    assert np.max(np.abs(y[0].T @ y[0] - np.eye(3))) < 1e-14

    def tangent_defect(t: float) -> float:
        yU, ys, yV = mc.retract(x, mc.Tangent((t * xi[0], t * xi[1])))
        D = (yU * ys) @ yV.T - X - t * xi_matrix
        tangent_part = U @ (U.T @ D) + D @ V @ V.T - U @ (U.T @ D @ V) @ V.T
        return float(np.linalg.norm(tangent_part))

    ratio = tangent_defect(1e-2) / tangent_defect(5e-3)
    assert 6.0 < ratio < 10.0                    # 8 for a t^3 defect


def test_gauge_stays_clean_only_with_projection() -> None:
    """Gauss-Newton Lanczos at a random point. Without project_covector the
    invisible block U^T gU of the Krylov covectors grows to O(1) and a
    spurious zero Ritz value appears (the true minimum is about 2e-2); with
    it the gauge stays at rounding level."""
    a, samples, rng = small_problem(1, (80, 60), 3, 3.0)
    x = mc.random_point((80, 60), 3, rng)
    sp = mc.fixed_rank_space()
    hessvec = lambda v: mc.hessian_action(samples, x, v, curvature=False)
    precond = lambda w: euclidean_metric_solve(x, w)   # the Frobenius metric, M = I

    def run(project: Optional[Callable[[mc.Cotangent], mc.Cotangent]]) -> tuple[float, float]:
        state = lanczos_start(sp, mc.gradient(samples, x), precond, capacity=201)
        for _ in range(200):
            state = lanczos_advance(sp, state, hessvec, precond, project_covector=project)
        gauge = float(np.max(np.abs(x[0].T @ state.v_next[0])))
        return gauge, float(np.linalg.eigvalsh(lanczos_tridiagonal(state))[0])

    gauge_plain, ritz_plain = run(None)
    gauge_projected, ritz_projected = run(lambda w: mc.gauge_projection(x, w))
    assert gauge_plain > 1e-3 and ritz_plain < 1e-8
    assert gauge_projected < 1e-14 and ritz_projected > 1e-2


def test_completion_from_zero() -> None:
    """From X = 0 the method converges to a fit within 20% of the best rank-r
    error, superlinearly at the end (the gradient contracts by a factor
    below 1e-2 in each of the last two steps, faster each time)."""
    a, samples, rng = small_problem(3, (120, 90), 4, 4.0)
    x, summary = mc.complete(samples, mc.zero_point((120, 90), 4, rng))
    assert summary.reason == "converged"
    d = a[1]
    best = float(np.sqrt(np.sum(d[4:] ** 2) / np.sum(d ** 2)))
    assert mc.relative_error(x, a) < 1.2 * best
    gnorms = [rec.gnorm for rec in summary.iterations] + [summary.gnorm]
    ratios = [gnorms[-2] / gnorms[-3], gnorms[-1] / gnorms[-2]]
    assert ratios[0] < 1e-2 and ratios[1] < ratios[0]


def test_rank_continuation_on_fast_decay() -> None:
    """Singular values 4^-k: the direct rank-4 fit is hard (the last
    components are tiny), but fitting ranks 1, ..., 4 in turn, each from the
    previous fit plus a zero component, tracks the best rank-k error at
    every stage."""
    rng = np.random.default_rng(4)
    d = 0.25 ** np.arange(10, dtype=float)
    a = mc.random_low_rank((150, 120), d, rng)
    samples = mc.sample_uniformly(a, 4 * (150 + 120 - 4) * 4, rng)
    stages = mc.complete_by_rank_continuation(samples, 4, rng)
    for k, (x, summary) in enumerate(stages, start=1):
        assert summary.reason == "converged"
        best = float(np.sqrt(np.sum(d[k:] ** 2) / np.sum(d ** 2)))
        assert mc.relative_error(x, a) < 1.3 * best


def test_cg_and_lanczos_on_the_manifold() -> None:
    """With M = P = I (the Frobenius metric) the default solver is truncated
    CG, with the gauge projection applied to its residuals; it reaches the
    same fit as the reorthogonalized Lanczos solver."""
    a, samples, rng = small_problem(3, (120, 90), 4, 4.0)
    x0 = mc.zero_point((120, 90), 4, rng)
    d = a[1]
    best = float(np.sqrt(np.sum(d[4:] ** 2) / np.sum(d ** 2)))
    fits = {}
    krylov: Literal["auto", "lanczos"]
    for krylov in ("auto", "lanczos"):
        x, summary = mc.complete(samples, x0, krylov=krylov)
        assert summary.reason == "converged"
        assert {r.subproblem.krylov for r in summary.iterations} == \
            {"cg" if krylov == "auto" else "lanczos"}
        assert mc.relative_error(x, a) < 1.2 * best
        fits[krylov] = mc.relative_error(x, a)
    assert abs(fits["auto"] - fits["lanczos"]) < 1e-6 * fits["lanczos"]


# --- L-BFGS on the manifold: the transport ---------------------------------------------


def ambient(x: mc.LowRank, v: tuple[Array, Array]) -> Array:
    """The N x M matrix dU V^T + U dV^T of a tangent vector (or of a canonical
    covector, the ambient functional it stands for)."""
    out: Array = v[0] @ x[2].T + x[0] @ v[1].T
    return out


def test_transport_is_the_projection_onto_the_new_tangent_space() -> None:
    """The transport of a tangent vector is the Frobenius projection of its
    matrix onto the tangent space at the new point, in gauge (U'^T dU' = 0);
    covectors move the same way. It preserves pairings and lengths to second
    order in the distance between the points."""
    rng = np.random.default_rng(5)
    shape, r = (14, 11), 3
    x = mc.random_point(shape, r, rng)
    v, w = mc.random_tangent(x, rng), mc.Cotangent(mc.random_tangent(x, rng))
    u = mc.random_tangent(x, rng)                                 # the step's direction
    defects = []
    for t in (1e-2, 1e-3):
        x1 = mc.retract(x, mc.Tangent((t * u[0], t * u[1])))
        U1, _, V1 = x1
        moved = mc.transport(x, x1, v)
        assert np.max(np.abs(U1.T @ moved[0])) < 1e-12
        Z = ambient(x, v)
        projected = Z - (np.eye(shape[0]) - U1 @ U1.T) @ Z @ (np.eye(shape[1]) - V1 @ V1.T)
        assert np.max(np.abs(ambient(x1, moved) - projected)) < 1e-12
        moved_w = mc.transport_covector(x, x1, w)
        assert np.max(np.abs(U1.T @ moved_w[0])) < 1e-12
        sp = mc.fixed_rank_space()
        pairing = sp.pair(w, v)
        defects.append(abs(sp.pair(moved_w, moved) - pairing) / abs(pairing))
        length = np.sqrt(sp.pair(mc.Cotangent(moved), moved))
        assert abs(length - 1.0) < 10 * t ** 2
    assert 0.005 < defects[1] / defects[0] < 0.02                 # second order: 100x smaller


@pytest.mark.parametrize("pairings", ["stored", "recomputed", "corrected"])
def test_lbfgs_reaches_the_trust_region_fit(pairings: Literal["stored", "recomputed", "corrected"]) -> None:
    """From X = 0, L-BFGS with the projection transport converges to the fit
    the trust region finds, with each treatment of the stored pairings."""
    a, samples, rng = small_problem(3, (120, 90), 4, 4.0)
    x0 = mc.zero_point((120, 90), 4, rng)
    x_tr, s_tr = mc.complete(samples, x0)
    x, s = mc.complete_lbfgs(samples, x0, pairings=pairings)
    assert s.reason == "converged" and s_tr.reason == "converged"
    assert abs(mc.relative_error(x, a) - mc.relative_error(x_tr, a)) < 1e-6 * mc.relative_error(x_tr, a)
    assert abs(s.f - s_tr.f) < 1e-8 * s_tr.f
    accepted = [r for r in s.iterations if r.accepted]
    assert all(np.isfinite(r.transport_defect) and np.isfinite(r.isometry_defect) for r in accepted)
    assert s.vector_transport_calls > len(accepted)


def test_lbfgs_rank_continuation_on_fast_decay() -> None:
    """Rank continuation with L-BFGS (the first step of each stage as long as
    the smallest singular value so far) tracks the best rank-k error at every
    stage, as the trust region does."""
    rng = np.random.default_rng(4)
    d = 0.25 ** np.arange(10, dtype=float)
    a = mc.random_low_rank((150, 120), d, rng)
    samples = mc.sample_uniformly(a, 4 * (150 + 120 - 4) * 4, rng)
    stages = mc.complete_by_rank_continuation(samples, 4, rng, method="lbfgs")
    for k, (x, summary) in enumerate(stages, start=1):
        assert summary.reason == "converged"
        best = float(np.sqrt(np.sum(d[k:] ** 2) / np.sum(d ** 2)))
        assert mc.relative_error(x, a) < 1.3 * best
