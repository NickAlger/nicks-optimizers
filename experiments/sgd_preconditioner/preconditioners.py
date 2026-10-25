"""Preconditioners for matrix completion (examples/matrix_completion.py) on
the fixed-rank manifold, as maps from canonical covectors to tangent vectors.

  cauchy_sgd     approximately minimizes 0.5 <H p, p> - <b, p> over tangent
                 vectors p by minibatch SGD with Cauchy steps: each step goes
                 along -g, g = H_B p - b the minibatch gradient, to the
                 minimizer of the minibatch quadratic along that ray. H_B is
                 the Gauss-Newton Hessian (curvature=False) or the full
                 Riemannian Hessian restricted to the batch B of sampled
                 entries, scaled by K / |B| so that it is unbiased. Batches
                 are a fixed partition visited in a fixed order ("cyclic", a
                 deterministic map) or a new random partition for every
                 epoch of every application ("reshuffled", from a seeded
                 generator). Scale-invariant step cap: alpha <= cap * alpha_0,
                 alpha_0 the Gauss-Newton Cauchy step of the first batch, so
                 the map stays homogeneous: P^{-1}(c b) = c P^{-1}(b).
                 Options against the noise floor of constant-step SGD:
                 averaging (return the mean of the last epoch's iterates)
                 and svrg (variance reduction: g = H_B p - H_B p_ref +
                 H p_ref - b with p_ref the iterate at the start of the
                 epoch, one full product per epoch).
  block_jacobi   the block diagonal of the Gauss-Newton Hessian in the
                 components: one r x r block per row of dU (the sum of V_j V_j^T
                 over the row's sampled columns) and per row of dV (the sum of
                 U_i U_i^T over the column's sampled rows), the normal equations
                 of alternating least squares. Linear; symmetric positive
                 definite on canonical covectors.

Both return valid tangent vectors (U^T dU = 0): the SGD gradients are
canonical covectors, and the block Jacobi output is projected.

Every preconditioner counts its work in passes over the sampled entries (a
full Hessian product is one pass) in counter["passes"], and its applications
in counter["applications"].
"""

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

import matrix_completion as mc

Map = Callable[[mc.Cotangent], mc.Tangent]
Counter = dict[str, float]


@dataclass(frozen=True)
class SGDConfig:
    epochs: int = 1
    batches: int = 10                                   # batches per epoch
    order: Literal["cyclic", "reshuffled"] = "cyclic"
    curvature: bool = False                             # False: Gauss-Newton
    cap: float = 10.0
    averaging: bool = False
    svrg: bool = False

    def label(self) -> str:
        kind = "full" if self.curvature else "GN"
        extra = (" avg" if self.averaging else "") + (" svrg" if self.svrg else "")
        return f"SGD {kind} {self.order} {self.epochs}x{self.batches}{extra}"


def batch_samples(samples: mc.Samples, index: np.ndarray) -> mc.Samples:
    return mc.Samples(shape=samples.shape, rows=samples.rows[index], cols=samples.cols[index],
                      values=samples.values[index])


def partition(K: int, batches: int, rng: np.random.Generator) -> list[np.ndarray]:
    return np.array_split(rng.permutation(K), batches)


def _dot(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> float:
    return float(np.vdot(a[0], b[0]) + np.vdot(a[1], b[1]))


def cauchy_sgd(samples: mc.Samples, x: mc.LowRank, cfg: SGDConfig, rng: np.random.Generator,
               counter: Counter, cyclic_batches: list[mc.Samples] | None = None) -> Map:
    """P^{-1} at the point x (see the module docstring). For cyclic order,
    pass the fixed batches (the same for every point of a run)."""
    K = samples.rows.shape[0]

    def batches_for_epoch() -> list[mc.Samples]:
        if cfg.order == "cyclic":
            assert cyclic_batches is not None
            return cyclic_batches
        return [batch_samples(samples, idx) for idx in partition(K, cfg.batches, rng)]

    def H(batch: mc.Samples, v: mc.Tangent, curvature: bool) -> tuple[np.ndarray, np.ndarray]:
        w = K / batch.rows.shape[0]
        hU, hV = mc.hessian_action(batch, x, v, curvature)
        return w * hU, w * hV

    def apply(b: mc.Cotangent) -> mc.Tangent:
        counter["applications"] += 1
        p = (np.zeros_like(b[0]), np.zeros_like(b[1]))
        alpha_ref: float | None = None
        for epoch in range(cfg.epochs):
            p_ref, full_ref = p, None
            if cfg.svrg and alpha_ref is not None:      # H p_ref on all samples
                full_ref = mc.hessian_action(samples, x, mc.Tangent(p_ref), cfg.curvature)
                counter["passes"] += 1.0
            last_epoch = epoch == cfg.epochs - 1
            p_sum = (np.zeros_like(p[0]), np.zeros_like(p[1]))
            n_sum = 0
            for batch in batches_for_epoch():
                frac = batch.rows.shape[0] / K
                if alpha_ref is None:                    # p = 0: the gradient is -b
                    g = (-b[0], -b[1])
                else:
                    Hp = H(batch, mc.Tangent(p), cfg.curvature)
                    counter["passes"] += frac
                    g = (Hp[0] - b[0], Hp[1] - b[1])
                    if full_ref is not None:
                        Href = H(batch, mc.Tangent(p_ref), cfg.curvature)
                        counter["passes"] += frac
                        g = (g[0] - Href[0] + full_ref[0], g[1] - Href[1] + full_ref[1])
                gg = _dot(g, g)
                if gg == 0.0:
                    return mc.Tangent(p)
                d = mc.Tangent((-g[0], -g[1]))
                Hd = H(batch, d, cfg.curvature)
                counter["passes"] += frac
                curv = _dot(Hd, d)
                if alpha_ref is None:
                    curv_gn = curv if not cfg.curvature else _dot(H(batch, d, False), d)
                    if cfg.curvature:
                        counter["passes"] += frac
                    alpha_ref = gg / curv_gn if curv_gn > 0.0 else 1.0
                alpha = gg / curv if curv > 0.0 else np.inf
                alpha = min(alpha, cfg.cap * alpha_ref)
                p = (p[0] + alpha * d[0], p[1] + alpha * d[1])
                if last_epoch:
                    p_sum, n_sum = (p_sum[0] + p[0], p_sum[1] + p[1]), n_sum + 1
        if cfg.averaging and n_sum > 0:
            return mc.Tangent((p_sum[0] / n_sum, p_sum[1] / n_sum))
        return mc.Tangent(p)

    return apply


def block_jacobi(samples: mc.Samples, x: mc.LowRank, counter: Counter, ridge: float = 1e-12) -> Map:
    """P^{-1} = projected block diagonal solve (see the module docstring).
    Building the blocks costs about one pass (counted once per point)."""
    U, s, V = x
    N, M = samples.shape
    r = U.shape[1]
    rows, cols = samples.rows, samples.cols
    Ur, Vc = U[rows], V[cols]
    BU = np.zeros((N, r, r))
    BV = np.zeros((M, r, r))
    np.add.at(BU, rows, Vc[:, :, None] * Vc[:, None, :])
    np.add.at(BV, cols, Ur[:, :, None] * Ur[:, None, :])
    eye = np.eye(r)
    BU += ridge * (np.trace(BU, axis1=1, axis2=2).mean() / r + 1.0) * eye
    BV += ridge * (np.trace(BV, axis1=1, axis2=2).mean() / r + 1.0) * eye
    counter["passes"] += 1.0

    def apply(b: mc.Cotangent) -> mc.Tangent:
        counter["applications"] += 1
        pU = np.linalg.solve(BU, b[0][:, :, None])[:, :, 0]
        pV = np.linalg.solve(BV, b[1][:, :, None])[:, :, 0]
        return mc.Tangent((mc.project_out(U, pU), pV))

    return apply
