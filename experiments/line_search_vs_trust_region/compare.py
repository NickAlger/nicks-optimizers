"""line_search_newton against trust_region_newton (dev/line_search_plan.md,
phase 5 check-in): the same problems, operators and options, and the work
each method spends.

Problems
  rosenbrock      chained Rosenbrock, n = 20, from (-1.2, 1, ...): the tests'
                  problem (nonconvex: a curved valley, and a second local
                  minimizer near (-1, 1, ..., 1))
  double well     the tests' double well, n = 30 (nonconvex, saddles)
  p-laplacian N   a finite-difference p-Laplacian energy (p = 4, convex) on
                  the unit square with N x N interior nodes, scipy sparse:
                    h1      M = P = the discrete H^1 inner product (sparse LU)
                    l2/h1   M = h^2 I (L^2), P = H^1: Lanczos
                    jacobi  M = P = diag(H^1): a weak preconditioner
  matrix completion  the tests' rank-3 fit on the fixed-rank manifold

Run:  python experiments/line_search_vs_trust_region/compare.py [--json results.json]
"""

import argparse
import dataclasses
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
import matrix_completion as mc                    # noqa: E402
from nicks_optimizers import (                    # noqa: E402
    euclidean_metric_apply,
    euclidean_metric_solve,
    line_search_newton,
    trust_region_newton,
    tuple_tree_space,
)
from problems import (                            # noqa: E402
    jacobi_precond_solve,
    rosen_fun,
    rosen_grad,
    rosen_hessvec,
    smoothing_metric_apply,
    smoothing_metric_solve,
    well_fun,
    well_grad,
    well_hessvec,
    well_precond_solve,
)


def p_laplacian(N: int, p: float = 4.0, eps: float = 1e-2) -> dict[str, Any]:
    h = 1.0 / (N + 1)
    n = N * N
    D = sps.diags([-np.ones(N - 1), np.ones(N)], [-1, 0], shape=(N, N)) / h
    eye = sps.identity(N)
    Dx, Dy = sps.kron(eye, D).tocsr(), sps.kron(D, eye).tocsr()
    H1 = ((Dx.T @ Dx + Dy.T @ Dy) * h * h + h * h * sps.identity(n)).tocsc()
    lu = spla.splu(H1)
    diag = H1.diagonal()
    f = np.ones(n) * h * h

    def fun(u: Any) -> Any:
        gx, gy = Dx @ u, Dy @ u
        return h * h * np.sum((gx ** 2 + gy ** 2 + eps ** 2) ** (p / 2)) / p - f @ u

    def grad(u: Any) -> Any:
        gx, gy = Dx @ u, Dy @ u
        s = h * h * (gx ** 2 + gy ** 2 + eps ** 2) ** (p / 2 - 1)
        return Dx.T @ (s * gx) + Dy.T @ (s * gy) - f

    def hessvec(u: Any, v: Any) -> Any:
        gx, gy = Dx @ u, Dy @ u
        r = gx ** 2 + gy ** 2 + eps ** 2
        s, t = h * h * r ** (p / 2 - 1), h * h * (p - 2) * r ** (p / 2 - 2)
        vx, vy = Dx @ v, Dy @ v
        d = gx * vx + gy * vy
        return Dx.T @ (s * vx + t * d * gx) + Dy.T @ (s * vy + t * d * gy)

    return dict(fun=fun, grad=grad, hessvec=hessvec, x0=np.zeros(n), n=n, h=h,
                h1_solve=lambda u, w: lu.solve(w), jacobi_solve=lambda u, w: w / diag)


def cases() -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    rosen = dict(fun=rosen_fun, grad=rosen_grad, hessvec=rosen_hessvec, x0=np.tile([-1.2, 1.0], 10))
    out += [("rosenbrock, M = I (CG)", dict(rosen)),
            ("rosenbrock, Jacobi M = P (CG)", dict(rosen, metric_solve=jacobi_precond_solve)),
            ("rosenbrock, smoothing M, Jacobi P (Lanczos)",
             dict(rosen, metric_solve=smoothing_metric_solve, metric_apply=smoothing_metric_apply,
                  precond_solve=jacobi_precond_solve)),
            ("rosenbrock, saddle-free", dict(rosen, metric_solve=jacobi_precond_solve, saddle_free=True))]
    well = dict(fun=well_fun, grad=well_grad, hessvec=well_hessvec,
                x0=np.random.default_rng(3).uniform(-0.3, 0.3, 30))
    out += [("double well (CG)", dict(well)),
            ("double well, saddle-free", dict(well, saddle_free=True, precond_solve=well_precond_solve,
                                              metric_solve=euclidean_metric_solve,
                                              metric_apply=euclidean_metric_apply))]
    for N in (63, 127):
        pl = p_laplacian(N)
        base = dict(fun=pl["fun"], grad=pl["grad"], hessvec=pl["hessvec"], x0=pl["x0"])
        h2 = pl["h"] ** 2
        out += [(f"p-laplacian n={pl['n']}, h1 (CG)", dict(base, metric_solve=pl["h1_solve"])),
                (f"p-laplacian n={pl['n']}, l2/h1 (Lanczos)",
                 dict(base, metric_solve=lambda u, w, h2=h2: w / h2,
                      metric_apply=lambda u, v, h2=h2: h2 * v, precond_solve=pl["h1_solve"])),
                (f"p-laplacian n={pl['n']}, jacobi (CG)", dict(base, metric_solve=pl["jacobi_solve"]))]
    rng = np.random.default_rng(0)
    shape, r = (60, 50), 3
    a = mc.random_low_rank(shape, 0.5 ** np.arange(8, dtype=float), rng)
    samples = mc.sample_uniformly(a, 4 * (shape[0] + shape[1] - r) * r, rng)
    tree = dataclasses.replace(tuple_tree_space(), dimension=mc.tangent_dimension)
    out += [("matrix completion (CG)",
             dict(fun=lambda x: mc.objective(samples, x), grad=lambda x: mc.gradient(samples, x),
                  hessvec=lambda x, v: mc.hessian_action(samples, x, v),
                  x0=mc.random_point(shape, r, rng), space=tree, retract=mc.retract,
                  metric_solve=euclidean_metric_solve, project_covector=mc.gauge_projection))]
    return out


def run(method: Callable[..., Any], case: dict[str, Any]) -> dict[str, Any]:
    options = dict(case)
    fun, grad, hessvec, x0 = (options.pop(k) for k in ("fun", "grad", "hessvec", "x0"))
    t = time.perf_counter()
    _, s = method(fun, grad, hessvec, x0, max_iter=2000, **options)
    wall = time.perf_counter() - t
    return dict(reason=s.reason, iterations=len(s.iterations), f=s.f, gnorm_ratio=s.gnorm / s.gnorm0,
                fun=s.fun_calls, grad=s.grad_calls, hessvec=s.hessvec_calls,
                solves=s.metric_solve_calls + s.precond_solve_calls,
                applies=s.metric_apply_calls, seconds=wall)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", help="write the results to this file")
    args = parser.parse_args()
    rows = []
    print(f"{'problem':44s} {'method':6s} {'reason':10s} {'its':>5s} {'f':>13s} {'fun':>5s} {'grad':>5s} "
          f"{'Hv':>6s} {'solves':>6s} {'M':>5s} {'time':>7s}")
    for name, case in cases():
        for label, method in (("TR", trust_region_newton), ("LS", line_search_newton)):
            row = dict(problem=name, method=label, **run(method, case))
            rows.append(row)
            print(f"{name:44s} {label:6s} {row['reason']:10s} {row['iterations']:5d} {row['f']:13.6e} "
                  f"{row['fun']:5d} {row['grad']:5d} {row['hessvec']:6d} {row['solves']:6d} "
                  f"{row['applies']:5d} {row['seconds']:6.2f}s", flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
