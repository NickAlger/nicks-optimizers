"""Trust-region Newton on the distributed p-Laplacian (examples/p_laplacian.py)
with unconventional preconditioners, Lanczos against the flexible solver.

The metric is the L^2 metric of the lumped mass matrix in every run; the
preconditioner, separate from it, acts on each rank's owned block A of the
Hessian (block Jacobi over the ranks):

  amg           one AMG V-cycle for A                  fixed SPD (the example's)
  inner-pcg     AMG-preconditioned CG on A z = r,      nonlinear in r
                stopped at 30% relative residual
  gauss-seidel  one forward Gauss-Seidel sweep,        nonsymmetric, fixed
                z = tril(A)^{-1} r
  random-gs     Gauss-Seidel in a random order,        nonsymmetric, stochastic
                new for every application
  noisy-amg     AMG V-cycle, each entry multiplied     stochastic
                by (1 + 0.3 xi), xi standard normal

For reference, "amg as metric" runs the conventional trust region with the
AMG cycle as M = P (truncated CG).

Run:  mpiexec -n 2 python experiments/flexible_preconditioners/p_laplacian_study.py
      (options: --p, --rings, --precond, --solvers, --json)
"""
import argparse
import json
import os
import sys
import time
import warnings
from typing import Callable

import numpy as np
import scipy.sparse
from mpi4py import MPI
from scipy.sparse.linalg import spsolve_triangular

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples"))
import p_laplacian as pl  # noqa: E402
from nicks_optimizers import Covector, Point, PointOperators, Vector, trust_region_newton  # noqa: E402

Solve = Callable[[Covector], Vector]
warnings.simplefilter("ignore", RuntimeWarning)    # Lanczos overflows with the noisy preconditioner


def inner_pcg(A: scipy.sparse.csr_matrix, cycle: Solve, rtol: float = 0.3) -> Solve:
    def solve(r: Covector) -> Vector:
        z = np.zeros_like(r)
        res = np.array(r, dtype=float)
        w = cycle(Covector(res))
        p, rw = w.copy(), float(res @ w)
        r0 = float(np.linalg.norm(res))
        for _ in range(200):
            if np.linalg.norm(res) <= rtol * r0:
                break
            Ap = A @ p
            a = rw / float(p @ Ap)
            z, res = z + a * p, res - a * Ap
            w = cycle(Covector(res))
            rw_new = float(res @ w)
            p, rw = w + (rw_new / rw) * p, rw_new
        return Vector(z)
    return solve


def gauss_seidel(A: scipy.sparse.csr_matrix) -> Solve:
    L = scipy.sparse.tril(A, format="csr")
    return lambda r: Vector(spsolve_triangular(L, np.asarray(r), lower=True))


def random_gauss_seidel(A: scipy.sparse.csr_matrix, rng: np.random.Generator) -> Solve:
    n = A.shape[0]

    def solve(r: Covector) -> Vector:
        perm = rng.permutation(n)
        Ap = A[perm][:, perm]
        zp = spsolve_triangular(scipy.sparse.tril(Ap, format="csr"), np.asarray(r)[perm], lower=True)
        z = np.empty(n)
        z[perm] = zp
        return Vector(z)
    return solve


def noisy(cycle: Solve, rng: np.random.Generator, level: float = 0.3) -> Solve:
    return lambda r: Vector(cycle(r) * (1.0 + level * rng.standard_normal(r.shape[0])))


def operators_at(loc: pl.Local, e: pl.Energy, name: str, as_metric: bool, rng: np.random.Generator,
                 u: Point) -> PointOperators[Vector, Covector]:
    hessvec, g, a, c = pl.hessian_at(loc, e, u)
    A = pl.owned_block(loc, g, a, c)
    cycle = pl.multigrid_cycle(A)
    P = {"amg": lambda: cycle, "inner-pcg": lambda: inner_pcg(A, cycle),
         "gauss-seidel": lambda: gauss_seidel(A), "random-gs": lambda: random_gauss_seidel(A, rng),
         "noisy-amg": lambda: noisy(cycle, rng)}[name]()
    if as_metric:
        return PointOperators(hessvec=hessvec, metric_solve=P)
    return PointOperators(hessvec=hessvec, metric_solve=lambda w: pl.l2_metric_solve(loc, w),
                          metric_apply=lambda v: pl.l2_metric_apply(loc, v), precond_solve=P)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--p", type=float, nargs="+", default=[1.5, 3.0, 5.0])
    parser.add_argument("--rings", type=int, default=25)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--precond", nargs="+",
                        default=["amg", "inner-pcg", "gauss-seidel", "random-gs", "noisy-amg"])
    parser.add_argument("--solvers", nargs="+", default=["lanczos", "flexible"])
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--json", help="write the results here (rank 0)")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    root = comm.Get_rank() == 0
    loc = pl.partition(pl.graded_disk_mesh(args.rings), comm)
    space = pl.distributed_space(loc)
    if root:
        print(f"{loc.n_unknowns} unknowns on {comm.Get_size()} ranks; L^2 metric, separate P "
              f"(block Jacobi over ranks); max {args.max_iter} iterations, 100 Krylov steps each\n")
        print(f"{'p':>4}  {'preconditioner':14s} {'solver':9s} {'its':>4} {'acc':>4} {'Krylov':>6} "
              f"{'fallb.':>6}  {'error':>8}  {'time':>6}  reason")
    results = []
    runs = [(name, solver) for name in args.precond for solver in args.solvers]
    runs.append(("amg", "amg as metric"))
    for p in args.p:
        e, u_star = pl.manufactured_energy(loc, p, args.eps)
        for name, solver in runs:
            rng = np.random.default_rng([7, comm.Get_rank()])
            as_metric = solver == "amg as metric"
            start = time.perf_counter()
            try:
                u, summary = trust_region_newton(
                    lambda u: pl.objective(loc, e, u), lambda u: pl.gradient(loc, e, u), None,
                    Point(np.zeros(loc.n_owned)), space=space, retract=lambda u, s: Point(u + s),
                    operators_at=lambda u: operators_at(loc, e, name, as_metric, rng, u),
                    krylov="auto" if as_metric else solver,  # type: ignore[arg-type]
                    rtol=1e-10, max_iter=args.max_iter, max_krylov=100)
                error = pl.l2_error(loc, u, u_star)
                its, acc = len(summary.iterations), summary.accepted_steps
                krylov = sum(r.subproblem.krylov_steps for r in summary.iterations)
                fallbacks = sum(r.subproblem.fallbacks for r in summary.iterations)
                reason = summary.reason
            except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
                error, its, acc, krylov, fallbacks, reason = float("nan"), 0, 0, 0, 0, f"crash: {exc}"
            elapsed = comm.allreduce(time.perf_counter() - start, op=MPI.MAX)
            if root:
                print(f"{p:4.1f}  {name:14s} {solver:9s} {its:4d} {acc:4d} {krylov:6d} {fallbacks:6d}  "
                      f"{error:8.1e}  {elapsed:5.1f}s  {reason}")
            results.append({"p": p, "precond": name, "solver": solver, "iterations": its,
                            "accepted": acc, "krylov_steps": krylov, "fallbacks": fallbacks,
                            "error": error, "reason": reason, "time": elapsed})
        if root:
            print()
    if root and args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=1)


if __name__ == "__main__":
    main()
