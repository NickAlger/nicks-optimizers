"""What do lbfgs's memory and transport add over the metric alone?

memory=0 keeps no pairs: every direction is -gamma M^{-1} g with gamma
fitted to the newest step (steepest descent in M with Barzilai-Borwein
steps). Against it, memory 1, 3, 10 and 30 on every problem of the tests
and examples; on the manifolds also the identity "transport" (the stored
vectors reused as they are, and y = g_new - g by plain subtraction, as
textbook L-BFGS does) as a control for the projection transport.

Problems
  convex          the tests' convex quartic, n = 40, condition 100, M = I
  rosenbrock      chained Rosenbrock, n = 20: M = I, and the tests'
                  point-dependent Jacobi metric
  p-laplacian     N = 63 (the examples' energy): M = I, H1 (fixed), and the
                  point-dependent Kacanov metric
  sphere          min x^T A x / 2 on the unit sphere, n = 300, eigenvalues
                  geometrically from 1 to 1000 (the smallest eigenvector)
  completion      rank-5 completion of a 400 x 300 matrix (singular values
                  2^-k), 4 samples per degree of freedom: from X = 0 (a
                  degenerate start, where which critical point is found
                  depends on the path), and from the spectral start (the
                  rank-5 truncated SVD of the rescaled samples, from which
                  every run finds the same minimizer)

Run:  python experiments/lbfgs_memory/compare.py [--json results.json] [--only sphere ...]
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Callable

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "examples"))
import lbfgs_p_laplacian as plx                                      # noqa: E402
import matrix_completion as mc                                       # noqa: E402
from nicks_optimizers import (                                       # noqa: E402
    VectorTransport,
    euclidean_metric_apply,
    euclidean_metric_solve,
    identity_transport,
    lbfgs,
    numpy_space,
)
from problems import jacobi_precond_solve, rosen_fun, rosen_grad, rosen_hess_diag  # noqa: E402
from test_lbfgs import convex                                        # noqa: E402

MEMORIES = (0, 1, 3, 10, 30)


def problems() -> list[tuple[str, Callable[[], dict[str, Any]]]]:
    def convex_problem() -> dict[str, Any]:
        fun, grad, x0 = convex()
        return dict(fun=fun, grad=grad, x0=x0)

    def rosen(metric: bool) -> dict[str, Any]:
        out: dict[str, Any] = dict(fun=rosen_fun, grad=rosen_grad, x0=np.tile([-1.2, 1.0], 10))
        if metric:
            weights = lambda x: np.maximum(np.abs(rosen_hess_diag(x)), 1.0)
            out.update(metric_solve=jacobi_precond_solve, metric_apply=lambda x, v: weights(x) * v)
        return out

    def p_laplacian(config: str) -> dict[str, Any]:
        pb = plx.p_laplacian(63)
        options = {k: v for k, v in pb.configs[config].items() if k != "memory"}
        return dict(fun=pb.fun, grad=pb.grad, x0=pb.x0, **options)

    def sphere(transport: bool) -> dict[str, Any]:
        rng = np.random.default_rng(1)
        n = 300
        Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
        A = Q @ np.diag(np.geomspace(1.0, 1e3, n)) @ Q.T

        def project(x: Any, w: Any) -> Any:
            return w - (x @ w) * x
        move = lambda x, x1, v: project(x1, v)
        x0 = rng.standard_normal(n)
        return dict(fun=lambda x: 0.5 * x @ A @ x, grad=lambda x: project(x, A @ x), x0=x0 / np.linalg.norm(x0),
                    space=numpy_space, retract=lambda x, s: (x + s) / np.linalg.norm(x + s),
                    transport=VectorTransport(vector=move, covector=move) if transport else identity_transport,
                    metric_solve=euclidean_metric_solve, metric_apply=euclidean_metric_apply,
                    project_covector=project)

    def completion(transport: bool, spectral: bool = False) -> dict[str, Any]:
        rng = np.random.default_rng(0)
        shape, r = (400, 300), 5
        a = mc.random_low_rank(shape, 0.5 ** np.arange(30, dtype=float), rng)
        samples = mc.sample_uniformly(a, 4 * (shape[0] + shape[1] - r) * r, rng)
        x0 = mc.zero_point(shape, r, rng)
        if spectral:      # the rank-r truncated SVD of (NM / K) P_Omega(A)
            rescaled = sps.coo_matrix((samples.values * shape[0] * shape[1] / samples.values.size,
                                       (samples.rows, samples.cols)), shape=shape).tocsr()
            U, sv, Vt = spla.svds(rescaled, k=r, random_state=0)
            order = np.argsort(sv)[::-1]
            x0 = mc.LowRank((U[:, order], sv[order], Vt[order].T))
        return dict(fun=lambda x: mc.objective(samples, x), grad=lambda x: mc.gradient(samples, x),
                    x0=x0, space=mc.fixed_rank_space(), retract=mc.retract,
                    transport=mc.fixed_rank_transport if transport else identity_transport,
                    metric_solve=euclidean_metric_solve, metric_apply=euclidean_metric_apply,
                    project_covector=mc.gauge_projection, initial_step=None if spectral else 1.0)

    return [("convex, M = I", convex_problem),
            ("rosenbrock, M = I", lambda: rosen(False)),
            ("rosenbrock, Jacobi(x)", lambda: rosen(True)),
            ("p-laplacian, M = I", lambda: p_laplacian("Euclidean")),
            ("p-laplacian, H1", lambda: p_laplacian("H1")),
            ("p-laplacian, Kacanov(u)", lambda: p_laplacian("Kacanov")),
            ("sphere, projection", lambda: sphere(True)),
            ("sphere, no transport", lambda: sphere(False)),
            ("completion, projection", lambda: completion(True)),
            ("completion, no transport", lambda: completion(False)),
            ("completion, spectral, projection", lambda: completion(True, spectral=True)),
            ("completion, spectral, no transport", lambda: completion(False, spectral=True))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", help="write the results to this file (after every problem)")
    parser.add_argument("--only", nargs="+", help="run the problems whose names contain one of these")
    args = parser.parse_args()
    rows = []
    print(f"{'problem':34s} " + "  ".join(f"{'m=' + str(m):>16s}" for m in MEMORIES)
          + "    (iterations / f evaluations; * = not converged)")
    for name, make in problems():
        if args.only and not any(word in name for word in args.only):
            continue
        cells = []
        for m in MEMORIES:
            options = make()
            fun, grad, x0 = options.pop("fun"), options.pop("grad"), options.pop("x0")
            t = time.perf_counter()
            _, s = lbfgs(fun, grad, x0, memory=m, max_iter=20000, **options)
            row = dict(problem=name, memory=m, reason=s.reason, iterations=len(s.iterations),
                       grad=s.grad_calls, fun=s.fun_calls, f=float(s.f), gnorm_ratio=float(s.gnorm / s.gnorm0),
                       restarts=s.restarts, skipped=s.skipped, seconds=time.perf_counter() - t)
            rows.append(row)
            print(f"  {name}, memory {m}: {s.reason}, {row['iterations']} iterations, {row['seconds']:.1f} s",
                  file=sys.stderr, flush=True)
            mark = "" if s.reason == "converged" else "*"
            cells.append(f"{len(s.iterations):>7d}{mark:1s}/{s.fun_calls:<7d}")
        print(f"{name:34s} " + "  ".join(f"{c:>16s}" for c in cells), flush=True)
        if args.json:
            with open(args.json, "w") as fh:
                json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
