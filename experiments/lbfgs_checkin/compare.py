"""lbfgs at the phase-2 check-in (dev/lbfgs_plan.md): against the Newton
methods, the metric's effect across meshes, skip, restart and fallback
rates, and the angle test.

Parts
  A  methods   trust_region_newton (TR), line_search_newton (LS) and lbfgs
               on the same problems, with the same metric where it applies
  B  meshes    lbfgs on the p-Laplacian over N = 15, 31, 63, 127 with the
               metrics I (Euclidean), L^2 (h^2 I), Jacobi (diag H^1) and H^1
  C  angle     angle_tol in {0, 1e-8, 1e-4, 1e-2, 1e-1}
  D  options   the scalings and the first step, on two problems

Problems
  rosenbrock      chained Rosenbrock, n = 20, from (-1.2, 1, ...) (the tests')
  double well     the tests' double well, n = 30, from a small random point
  p-laplacian N   the finite-difference p-Laplacian energy (p = 4, eps =
                  1e-2, convex, degenerate at u = 0) on the unit square with
                  N x N interior nodes, from u = 0

Run:  python experiments/lbfgs_checkin/compare.py [--json results.json] [--parts ABCD]
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
from nicks_optimizers import lbfgs, line_search_newton, trust_region_newton  # noqa: E402
from problems import (                                                       # noqa: E402
    jacobi_precond_solve,
    rosen_fun,
    rosen_grad,
    rosen_hess_diag,
    rosen_hessvec,
    well_fun,
    well_grad,
    well_hessvec,
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

    h2 = h * h
    metrics = {
        "I": dict(),
        "L2": dict(metric_solve=lambda u, w: w / h2, metric_apply=lambda u, v: h2 * v),
        "Jacobi": dict(metric_solve=lambda u, w: w / diag, metric_apply=lambda u, v: diag * v),
        "H1": dict(metric_solve=lambda u, w: lu.solve(w), metric_apply=lambda u, v: H1 @ v),
    }
    return dict(fun=fun, grad=grad, hessvec=hessvec, x0=np.zeros(n), n=n, metrics=metrics)


def rosenbrock() -> dict[str, Any]:
    weights = lambda x: np.maximum(np.abs(rosen_hess_diag(x)), 1.0)       # the tests' Jacobi P(x)
    return dict(fun=rosen_fun, grad=rosen_grad, hessvec=rosen_hessvec, x0=np.tile([-1.2, 1.0], 10),
                metrics={"I": dict(),
                         "Jacobi(x)": dict(metric_solve=jacobi_precond_solve,
                                           metric_apply=lambda x, v: weights(x) * v)})


def double_well() -> dict[str, Any]:
    return dict(fun=well_fun, grad=well_grad, hessvec=well_hessvec,
                x0=np.random.default_rng(3).uniform(-0.3, 0.3, 30), metrics={"I": dict()})


def summarize(s: Any, seconds: float) -> dict[str, Any]:
    row = dict(reason=s.reason, iterations=len(s.iterations), f=float(s.f),
               gnorm_ratio=float(s.gnorm / s.gnorm0), fun=s.fun_calls, grad=s.grad_calls,
               hessvec=getattr(s, "hessvec_calls", 0),
               solves=s.metric_solve_calls + getattr(s, "precond_solve_calls", 0),
               applies=s.metric_apply_calls, seconds=seconds)
    if hasattr(s, "restarts"):
        row.update(restarts=s.restarts, skipped=s.skipped,
                   fallbacks=sum(r.direction == "fallback" for r in s.iterations),
                   trials_per_it=s.trials / max(len(s.iterations), 1),
                   unit_steps=sum(r.accepted and r.alpha == 1.0 for r in s.iterations))
    return row


def newton(method: Callable[..., Any], problem: dict[str, Any], metric: dict[str, Any],
           max_iter: int = 2000) -> dict[str, Any]:
    options = dict(metric)
    options.pop("metric_apply", None)          # M = P: the Newton methods need only M^{-1}
    t = time.perf_counter()
    _, s = method(problem["fun"], problem["grad"], problem["hessvec"], problem["x0"], max_iter=max_iter,
                  **options)
    return summarize(s, time.perf_counter() - t)


def quasi(problem: dict[str, Any], metric: dict[str, Any], max_iter: int = 20000,
          **options: Any) -> dict[str, Any]:
    t = time.perf_counter()
    _, s = lbfgs(problem["fun"], problem["grad"], problem["x0"], max_iter=max_iter, **metric, **options)
    return summarize(s, time.perf_counter() - t)


HEADER = (f"{'problem':24s} {'method':16s} {'reason':10s} {'its':>6s} {'f':>13s} {'fun':>6s} {'grad':>6s} "
          f"{'Hv':>6s} {'solves':>6s} {'M':>6s} {'rst':>4s} {'skip':>4s} {'fb':>4s} {'time':>7s}")


def show(problem: str, method: str, row: dict[str, Any]) -> None:
    extra = (f"{row.get('restarts', ''):>4} {row.get('skipped', ''):>4} {row.get('fallbacks', ''):>4}")
    print(f"{problem:24s} {method:16s} {row['reason']:10s} {row['iterations']:6d} {row['f']:13.6e} "
          f"{row['fun']:6d} {row['grad']:6d} {row['hessvec']:6d} {row['solves']:6d} {row['applies']:6d} "
          f"{extra} {row['seconds']:6.2f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", help="write the results to this file")
    parser.add_argument("--parts", default="ABCD")
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []

    def record(part: str, problem: str, method: str, row: dict[str, Any]) -> None:
        rows.append(dict(part=part, problem=problem, method=method, **row))
        show(problem, method, row)

    if "A" in args.parts:
        print("A. methods\n" + HEADER)
        problems = [("rosenbrock", rosenbrock()), ("double well", double_well())]
        problems += [(f"p-laplacian N={N}", p_laplacian(N)) for N in (63, 127)]
        for name, problem in problems:
            for mname, metric in problem["metrics"].items():
                if name.startswith("p-laplacian") and mname not in ("H1", "Jacobi"):
                    continue
                record("A", name, f"TR {mname}", newton(trust_region_newton, problem, metric))
                record("A", name, f"LS {mname}", newton(line_search_newton, problem, metric))
                record("A", name, f"lbfgs {mname}", quasi(problem, metric))

    if "B" in args.parts:
        print("\nB. meshes\n" + HEADER)
        for N in (15, 31, 63, 127):
            problem = p_laplacian(N)
            for mname, metric in problem["metrics"].items():
                record("B", f"p-laplacian N={N}", f"lbfgs {mname}", quasi(problem, metric))

    if "C" in args.parts:
        print("\nC. angle_tol\n" + HEADER)
        cases = [("rosenbrock", rosenbrock(), "I"), ("p-laplacian N=63", p_laplacian(63), "H1"),
                 ("p-laplacian N=63", p_laplacian(63), "I")]
        for name, problem, mname in cases:
            for tol in (0.0, 1e-8, 1e-4, 1e-2, 1e-1):
                record("C", name, f"{mname} angle {tol:g}",
                       quasi(problem, problem["metrics"][mname], angle_tol=tol))

    if "D" in args.parts:
        print("\nD. scalings and the first step\n" + HEADER)
        cases = [("rosenbrock", rosenbrock(), "I"), ("p-laplacian N=63", p_laplacian(63), "H1"),
                 ("p-laplacian N=63", p_laplacian(63), "I")]
        for name, problem, mname in cases:
            for label, options in [("y", dict()), ("s", dict(scaling="s")),
                                   ("geometric", dict(scaling="geometric")),
                                   ("none", dict(scaling="none")),
                                   ("y, step 1", dict(initial_step=1.0)),
                                   ("y, mem 3", dict(memory=3)), ("y, mem 30", dict(memory=30))]:
                record("D", name, f"{mname} {label}", quasi(problem, problem["metrics"][mname], **options))

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
