"""lbfgs with a constant metric against a point-dependent one, on the
p-Laplacian (the note requested after the phase-2 check-in).

Metrics (all symmetric positive definite, solved by sparse LU):
  H1          the discrete H^1 inner product: constant, factorized once
  Kacanov     M(u) = Dx^T S(u) Dx + Dy^T S(u) Dy, S = h^2 (eps^2 + |grad u|^2)^((p-2)/2):
              the Hessian without its (p - 2) rank-structured part, so
              M(u) <= H(u) <= (p - 1) M(u): spectrally equivalent to the
              Hessian whatever the mesh and the degeneracy, but a new
              factorization at every point

Configurations
  lbfgs H1 general     the general path: three solves and one product per iteration
  lbfgs H1 fixed       fixed_metric=True: one solve per iteration, the same iterates
  lbfgs ... none       scaling="none" (gamma = 1, B0 = M)
  lbfgs ... m=k        memory k (default 10)
  lbfgs Kacanov        operators_at with point_data: one factorization per point
  lbfgs Kacanov kw     keyword operators: the factorization redone at every call
  lbfgs H1 fixed, P Kac   M = H1 on the fixed path, P = M(u) as the safeguarded
                       accelerant (phase 3), scale fitted ("fit") or not ("none")
  TR, LS               trust_region_newton and line_search_newton with M = P (CG)

Run:  python experiments/lbfgs_metrics/compare.py [--json results.json] [--N 31 63 127 255]
"""

import argparse
import json
import time
from typing import Any, Callable

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from nicks_optimizers import PointOperators, lbfgs, line_search_newton, trust_region_newton


class Count:
    def __init__(self) -> None:
        self.factorizations = 0


def p_laplacian(N: int, count: Count, p: float = 4.0, eps: float = 1e-2) -> dict[str, Any]:
    h = 1.0 / (N + 1)
    n = N * N
    D = sps.diags([-np.ones(N - 1), np.ones(N)], [-1, 0], shape=(N, N)) / h
    eye = sps.identity(N)
    Dx, Dy = sps.kron(eye, D).tocsr(), sps.kron(D, eye).tocsr()
    f = np.ones(n) * h * h

    def factor(A: Any) -> Any:
        count.factorizations += 1
        return spla.splu(A.tocsc())

    H1 = ((Dx.T @ Dx + Dy.T @ Dy) * h * h + h * h * sps.identity(n)).tocsc()
    cache: dict[str, Any] = {}

    def h1_solve(u: Any, w: Any) -> Any:
        """Factorized at the first call (counted and timed with the run)."""
        if "lu" not in cache:
            cache["lu"] = factor(H1)
        return cache["lu"].solve(w)

    def weights(u: Any) -> Any:
        gx, gy = Dx @ u, Dy @ u
        return gx, gy, h * h * (gx ** 2 + gy ** 2 + eps ** 2) ** (p / 2 - 1)

    def fun(u: Any) -> Any:
        gx, gy = Dx @ u, Dy @ u
        return h * h * np.sum((gx ** 2 + gy ** 2 + eps ** 2) ** (p / 2)) / p - f @ u

    def grad(u: Any) -> Any:
        gx, gy, s = weights(u)
        return Dx.T @ (s * gx) + Dy.T @ (s * gy) - f

    def hessvec(u: Any, v: Any) -> Any:
        gx, gy = Dx @ u, Dy @ u
        r = gx ** 2 + gy ** 2 + eps ** 2
        s, t = h * h * r ** (p / 2 - 1), h * h * (p - 2) * r ** (p / 2 - 2)
        vx, vy = Dx @ v, Dy @ v
        d = gx * vx + gy * vy
        return Dx.T @ (s * vx + t * d * gx) + Dy.T @ (s * vy + t * d * gy)

    def kacanov(u: Any) -> Any:
        _, _, s = weights(u)
        S = sps.diags(s)
        return (Dx.T @ S @ Dx + Dy.T @ S @ Dy).tocsc()

    def point_data(u: Any) -> Any:
        A = kacanov(u)
        return A, factor(A)

    def operators_at(u: Any, data: Any) -> PointOperators[Any, Any]:
        A, lu = data
        return PointOperators(metric_solve=lambda w: lu.solve(w), metric_apply=lambda v: A @ v)

    def h1_with_kacanov(u: Any, data: Any) -> PointOperators[Any, Any]:
        """M = H1 (constant), P = M(u) (the point's factorization)."""
        _, lu = data
        return PointOperators(metric_solve=lambda w: h1_solve(u, w), metric_apply=lambda v: H1 @ v,
                              precond_solve=lambda w: lu.solve(w))

    return dict(fun=fun, grad=grad, hessvec=hessvec, x0=np.zeros(n),
                h1_kacanov=dict(operators_at=h1_with_kacanov, point_data=point_data),
                h1=dict(metric_solve=h1_solve, metric_apply=lambda u, v: H1 @ v),
                kacanov=dict(operators_at=operators_at, point_data=point_data),
                kacanov_kw=dict(metric_solve=lambda u, w: factor(kacanov(u)).solve(w),
                                metric_apply=lambda u, v: kacanov(u) @ v))


def run(label: str, N: int, call: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
    count = Count()
    problem = p_laplacian(N, count)
    t = time.perf_counter()
    s = call(problem)
    seconds = time.perf_counter() - t
    return dict(N=N, method=label, reason=s.reason, iterations=len(s.iterations), f=float(s.f),
                fun=s.fun_calls, grad=s.grad_calls, hessvec=getattr(s, "hessvec_calls", 0),
                solves=s.metric_solve_calls + getattr(s, "precond_solve_calls", 0),
                applies=s.metric_apply_calls, factorizations=count.factorizations, seconds=seconds,
                skipped=getattr(s, "skipped", 0), restarts=getattr(s, "restarts", 0),
                precond_directions=getattr(s, "precond_directions", 0))


def configurations() -> list[tuple[str, Callable[[dict[str, Any]], Any]]]:
    def q(metric: str, **options: Any) -> Callable[[dict[str, Any]], Any]:
        return lambda pb: lbfgs(pb["fun"], pb["grad"], pb["x0"], max_iter=5000, **pb[metric], **options)[1]

    def newton(method: Callable[..., Any], metric: str) -> Callable[[dict[str, Any]], Any]:
        def go(pb: dict[str, Any]) -> Any:
            options = dict(pb[metric])
            options.pop("metric_apply", None)
            return method(pb["fun"], pb["grad"], pb["hessvec"], pb["x0"], **options)[1]
        return go

    return [("lbfgs H1 general", q("h1")), ("lbfgs H1 fixed", q("h1", fixed_metric=True)),
            ("lbfgs H1 fixed none", q("h1", fixed_metric=True, scaling="none")),
            ("lbfgs H1 fixed m=1", q("h1", fixed_metric=True, memory=1)),
            ("lbfgs Kacanov", q("kacanov")), ("lbfgs Kacanov none", q("kacanov", scaling="none")),
            ("lbfgs Kacanov m=1", q("kacanov", memory=1)), ("lbfgs Kacanov m=3", q("kacanov", memory=3)),
            ("lbfgs H1 fixed, P Kac", q("h1_kacanov", fixed_metric=True)),
            ("lbfgs H1 fixed, P Kac m=1", q("h1_kacanov", fixed_metric=True, memory=1)),
            ("lbfgs H1 fixed, P Kac none", q("h1_kacanov", fixed_metric=True, scaling_P="none")),
            ("TR H1", newton(trust_region_newton, "h1")), ("LS H1", newton(line_search_newton, "h1")),
            ("TR Kacanov", lambda pb: trust_region_newton(pb["fun"], pb["grad"], None, pb["x0"],
                                                          operators_at=lambda u, d: PointOperators(
                                                              hessvec=lambda v: pb["hessvec"](u, v),
                                                              metric_solve=pb["kacanov"]["operators_at"](u, d).metric_solve),
                                                          point_data=pb["kacanov"]["point_data"])[1])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", help="write the results to this file")
    parser.add_argument("--N", type=int, nargs="+", default=[31, 63, 127, 255])
    args = parser.parse_args()
    rows = []
    print(f"{'N':>4s} {'method':26s} {'reason':10s} {'its':>5s} {'f':>13s} {'fun':>5s} {'grad':>5s} "
          f"{'Hv':>5s} {'solves':>6s} {'M':>5s} {'LU':>4s} {'P':>4s} {'time':>8s}")
    for N in args.N:
        configs = configurations()
        if N == 63:
            configs.append(("lbfgs Kacanov kw", lambda pb: lbfgs(pb["fun"], pb["grad"], pb["x0"],
                                                                  **pb["kacanov_kw"])[1]))
        for label, call in configs:
            row = run(label, N, call)
            rows.append(row)
            print(f"{N:4d} {label:26s} {row['reason']:10s} {row['iterations']:5d} {row['f']:13.6e} "
                  f"{row['fun']:5d} {row['grad']:5d} {row['hessvec']:5d} {row['solves']:6d} "
                  f"{row['applies']:5d} {row['factorizations']:4d} {row['precond_directions']:4d} "
                  f"{row['seconds']:7.2f}s", flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
