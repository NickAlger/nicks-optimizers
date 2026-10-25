"""How well does each SGD variant solve the Gauss-Newton system H p = b?

At the fitted point of the matrix-completion example, the median over 10
random canonical covectors b of ||H_GN P^{-1}(b) - b|| / ||b|| (1: no better
than p = 0), and the cost of one application in passes over the samples (a
Hessian product is one pass). Block Jacobi for comparison.

Run:  python experiments/sgd_preconditioner/accuracy.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples"))
sys.path.insert(0, os.path.dirname(__file__))
import matrix_completion as mc  # noqa: E402
from preconditioners import SGDConfig, batch_samples, block_jacobi, cauchy_sgd, partition  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(0)
    N, M, r = 1000, 800, 8
    a = mc.random_low_rank((N, M), 0.5 ** np.arange(30, dtype=float), rng)
    samples = mc.sample_uniformly(a, 4 * (N + M - r) * r, rng)
    x = mc.complete_by_rank_continuation(samples, r, rng)[-1][0]
    K = samples.rows.shape[0]
    bs = [mc.gauge_projection(x, mc.Cotangent((g.standard_normal((N, r)), g.standard_normal((M, r)))))
          for g in (np.random.default_rng(100 + t) for t in range(10))]

    def residual(P) -> float:
        out = []
        for b in bs:
            p = P(b)
            h = mc.hessian_action(samples, x, p, curvature=False)
            out.append(np.sqrt(np.vdot(h[0] - b[0], h[0] - b[0]) + np.vdot(h[1] - b[1], h[1] - b[1]))
                       / np.sqrt(np.vdot(b[0], b[0]) + np.vdot(b[1], b[1])))
        return float(np.median(out))

    print(f"fitted point, {K} samples (4 per degree of freedom); one batch of b batches holds "
          f"{4 / 1:.0f}/b of the degrees of freedom\n")
    print(f"{'preconditioner':34s} {'residual':>8s} {'passes per application':>23s}")
    counter = {"passes": 0.0, "applications": 0.0}
    P = block_jacobi(samples, x, counter)
    print(f"{'block Jacobi':34s} {residual(P):8.2f} {'~0 (+1 per point)':>23s}")
    configs = []
    for batches in (2, 4, 10):
        for epochs in (1, 3, 10):
            configs.append(SGDConfig(epochs=epochs, batches=batches))
    configs += [SGDConfig(epochs=3, batches=10, averaging=True), SGDConfig(epochs=10, batches=10, averaging=True),
                SGDConfig(epochs=3, batches=10, svrg=True), SGDConfig(epochs=10, batches=10, svrg=True),
                SGDConfig(epochs=3, batches=4, svrg=True)]
    for cfg in configs:
        counter = {"passes": 0.0, "applications": 0.0}
        g = np.random.default_rng(7)
        cyc = [batch_samples(samples, idx) for idx in partition(K, cfg.batches, g)]
        P = cauchy_sgd(samples, x, cfg, g, counter, cyc)
        res = residual(P)
        print(f"{cfg.label():34s} {res:8.2f} {counter['passes'] / counter['applications']:23.1f}")


if __name__ == "__main__":
    main()
