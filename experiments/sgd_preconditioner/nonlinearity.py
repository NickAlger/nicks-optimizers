"""How nonlinear is the Cauchy-step SGD preconditioner?

At the fitted point of the matrix-completion example (rank continuation,
uniform sampling), for random canonical covectors b, b1, b2:

  homogeneity   ||P^{-1}(c b) - c P^{-1}(b)|| / ||c P^{-1}(b)||, c in {-2, 1e-3, 1e3}
                (reshuffled batches: the same random stream for both sides)
  additivity    ||P^{-1}(b1 + b2) - P^{-1}(b1) - P^{-1}(b2)|| / ||P^{-1}(b1 + b2)||
  symmetry      |<b1, P^{-1} b2> - <b2, P^{-1} b1>| / sqrt(<b1, P^{-1} b1> <b2, P^{-1} b2>)
  positivity    min over b of <b, P^{-1} b> / (||b|| ||P^{-1} b||): above 0 means
                the set {b : <b, P^{-1} b> <= 1} is star-shaped
  accuracy      ||H_GN P^{-1}(b) - b|| / ||b||, as an approximate inverse of the
                Gauss-Newton Hessian (1 = no better than zero)

Run:  python experiments/sgd_preconditioner/nonlinearity.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples"))
sys.path.insert(0, os.path.dirname(__file__))
import matrix_completion as mc  # noqa: E402
from preconditioners import SGDConfig, batch_samples, block_jacobi, cauchy_sgd, partition  # noqa: E402


def norm(a: tuple[np.ndarray, np.ndarray]) -> float:
    return float(np.sqrt(np.vdot(a[0], a[0]) + np.vdot(a[1], a[1])))


def dot(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> float:
    return float(np.vdot(a[0], b[0]) + np.vdot(a[1], b[1]))


def main() -> None:
    rng = np.random.default_rng(0)
    N, M, r = 1000, 800, 8
    d = 0.5 ** np.arange(30, dtype=float)
    a = mc.random_low_rank((N, M), d, rng)
    samples = mc.sample_uniformly(a, 4 * (N + M - r) * r, rng)
    x = mc.complete_by_rank_continuation(samples, r, rng)[-1][0]
    K = samples.rows.shape[0]

    def random_b(rng: np.random.Generator) -> mc.Cotangent:
        return mc.gauge_projection(x, mc.Cotangent((rng.standard_normal((N, r)),
                                                    rng.standard_normal((M, r)))))

    def make(kind: str, cfg: SGDConfig | None, seed: int):
        counter = {"passes": 0.0, "applications": 0.0}
        if kind == "block Jacobi":
            return block_jacobi(samples, x, counter)
        assert cfg is not None
        g = np.random.default_rng(seed)
        cyc = [batch_samples(samples, idx) for idx in partition(K, cfg.batches, g)]
        return cauchy_sgd(samples, x, cfg, g, counter, cyc)

    configs: list[tuple[str, SGDConfig | None]] = [("block Jacobi", None)]
    for order in ("cyclic", "reshuffled"):
        for epochs, batches in ((1, 10), (3, 10), (1, 50)):
            cfg = SGDConfig(epochs=epochs, batches=batches, order=order)  # type: ignore[arg-type]
            configs.append((cfg.label(), cfg))
    configs.append((SGDConfig(epochs=1, batches=10, order="reshuffled", curvature=True).label(),
                    SGDConfig(epochs=1, batches=10, order="reshuffled", curvature=True)))

    print(f"fitted point of the example (rank {r}, {K} samples); 20 random covectors per measurement\n")
    print(f"{'preconditioner':28s} {'homogeneity':>11s} {'additivity':>10s} {'symmetry':>9s} "
          f"{'positivity':>10s} {'accuracy':>9s}")
    for label, cfg in configs:
        kind = "block Jacobi" if cfg is None else "sgd"
        hom, add, sym, pos, acc = [], [], [], [], []
        for t in range(20):
            br = np.random.default_rng(100 + t)
            b, b1, b2 = random_b(br), random_b(br), random_b(br)
            for c in (-2.0, 1e-3, 1e3):
                P1, P2 = make(kind, cfg, 7 + t), make(kind, cfg, 7 + t)      # identical streams
                cb = mc.Cotangent((c * b[0], c * b[1]))
                lhs, rhs = P1(cb), P2(b)
                hom.append(norm((lhs[0] - c * rhs[0], lhs[1] - c * rhs[1])) / (abs(c) * norm(rhs)))
            P = make(kind, cfg, 7 + t)
            s12 = P(mc.Cotangent((b1[0] + b2[0], b1[1] + b2[1])))
            p1, p2 = P(b1), P(b2)
            add.append(norm((s12[0] - p1[0] - p2[0], s12[1] - p1[1] - p2[1])) / norm(s12))
            sym.append(abs(dot(b1, p2) - dot(b2, p1)) / np.sqrt(abs(dot(b1, p1) * dot(b2, p2))))
            pb = P(b)
            pos.append(dot(b, pb) / (norm(b) * norm(pb)))
            hp = mc.hessian_action(samples, x, pb, curvature=False)
            acc.append(norm((hp[0] - b[0], hp[1] - b[1])) / norm(b))
        print(f"{label:28s} {max(hom):11.1e} {np.median(add):10.1e} {np.median(sym):9.1e} "
              f"{min(pos):10.2f} {np.median(acc):9.2f}")
    print("\nhomogeneity: max; additivity, symmetry, accuracy: medians; positivity: min")


if __name__ == "__main__":
    main()
