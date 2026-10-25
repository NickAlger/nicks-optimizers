"""The Lanczos and flexible solvers compiled by JAX (dev/jit_backends_plan.md,
phase 6): fixed-capacity Krylov bases that double when a solve fills them,
against the numpy backend.

Problem: plaplace N of study.py (finite-difference p-Laplacian-type energy,
jax.grad and jax.jvp derivatives). Configurations:
  lanczos        Lanczos with P = M, the discrete H^1 metric
  lanczos P!=M   M = h^2 I (the L^2 metric), P = the H^1 metric: the Gram
                 matrix and the explicit residual are needed
  flexible       the flexible solver with the same M and P

Runs: numpy (numpy_space, every JAX function jit-compiled on its own) and
JAX (array_space, the iteration compiled in pieces) with the loop built from
scan, starting with room for 8 basis vectors (so the basis grows) or 32.

Reported per run: the median time of an iteration, and the rest of the
wall time ("compile": compilations, at the start and after each growth).

Run:  python experiments/jax_backend/solvers.py [--sizes 63 127] [--json solvers.json]
"""

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import jax                                   # noqa: E402
import jax.numpy as jnp                      # noqa: E402

import jax_backend                           # noqa: E402  (turns on float64)
from study import counted, numpy_wrapped, plaplace   # noqa: E402
from nicks_optimizers import RunSummary, array_space, trust_region_newton  # noqa: E402


def configurations(N: int, pieces: dict[str, Any]) -> dict[str, dict[str, Any]]:
    h2 = (1.0 / (N + 1)) ** 2
    H1 = pieces["metric_solve"]
    return {
        "lanczos": dict(krylov="lanczos", metric_solve=H1),
        "lanczos P!=M": dict(krylov="lanczos", metric_solve=lambda u, w: w / h2,
                             metric_apply=lambda u, v: h2 * v, precond_solve=H1),
        "flexible": dict(krylov="flexible", metric_solve=lambda u, w: w / h2,
                         metric_apply=lambda u, v: h2 * v, precond_solve=H1),
    }


def numpy_options(options: dict[str, Any]) -> dict[str, Any]:
    """The same operators, jit-compiled on their own, numpy arrays in and out."""
    out = dict(options)
    for key in ("metric_solve", "metric_apply", "precond_solve"):
        if key in out:
            f = jax.jit(out[key])
            out[key] = (lambda f: lambda u, w: np.asarray(f(u, w)))(f)
    return out


def per_iteration(summary: RunSummary) -> float:
    """The median time of an iteration (iterations that compile pieces, the
    first and those where the basis grows, stand out and do not count)."""
    e = [r.elapsed for r in summary.iterations]
    return float(np.median(np.diff(e)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[63, 127])
    parser.add_argument("--json", help="write the results to this file")
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    print(f"{'n':>6s} {'solver':13s} {'run':16s} {'its':>4s} {'Krylov':>6s} {'grow':>4s} "
          f"{'same':>5s} {'df/f':>8s} {'compile':>8s} {'per it':>9s} {'traces f/Hv':>11s}")
    for N in args.sizes:
        pieces = plaplace(N)
        np_pieces = numpy_wrapped(pieces)
        for solver, options in configurations(N, pieces).items():
            reference: RunSummary | None = None
            for run in ("numpy", "jax-scan cap 8", "jax-scan cap 32"):
                counter: dict[str, int] = {}
                if run == "numpy":
                    t = time.perf_counter()
                    _, summary = trust_region_newton(
                        counted(np_pieces["fun"], counter, "fun"), np_pieces["grad"],
                        counted(np_pieces["hessvec"], counter, "hessvec"), np.zeros(N * N),
                        max_iter=300, max_krylov=100, **numpy_options(options))
                else:
                    bk = jax_backend.backend("scan")
                    t = time.perf_counter()
                    _, summary = trust_region_newton(
                        counted(pieces["fun"], counter, "fun"), pieces["grad"],
                        counted(pieces["hessvec"], counter, "hessvec"), jnp.zeros(N * N),
                        space=array_space(bk), retract=lambda x, s: x + s, backend=bk,
                        krylov_capacity=int(run.split()[-1]), max_iter=300, max_krylov=100,
                        **options)
                wall = time.perf_counter() - t
                if reference is None:
                    reference = summary
                same = (len(summary.iterations) == len(reference.iterations)
                        and summary.hessvec_calls == reference.hessvec_calls)
                m = min(len(summary.iterations), len(reference.iterations), 20)
                df = max(abs(a.f - b.f) / max(abs(b.f), 1e-300)
                         for a, b in zip(summary.iterations[:m], reference.iterations[:m]))
                compiling = wall - len(summary.iterations) * per_iteration(summary)
                row = dict(n=N * N, solver=solver, run=run, reason=summary.reason, compiling=compiling,
                           iterations=len(summary.iterations), krylov=summary.hessvec_calls,
                           growths=summary.capacity_growths, same_counts=same, f_rel_diff=df,
                           first_iteration=summary.iterations[0].elapsed,
                           per_iteration=per_iteration(summary), wall=wall,
                           fun_traces=counter.get("fun", 0), hessvec_traces=counter.get("hessvec", 0))
                rows.append(row)
                print(f"{N * N:6d} {solver:13s} {run:16s} {row['iterations']:4d} {row['krylov']:6d} "
                      f"{row['growths']:4d} {str(same):>5s} {df:8.1e} {compiling:7.2f}s "
                      f"{row['per_iteration'] * 1e3:7.2f}ms {row['fun_traces']:5d}/{row['hessvec_traces']:<5d}",
                      flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
