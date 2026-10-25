"""Phase 7 of dev/jit_backends_plan.md: how the basis capacity and its
growth policy affect compiled Lanczos and flexible solves, against numpy.

Same problems and configurations as solvers.py. Runs per configuration:
  numpy          numpy_space, the reference
  scan 32        array_space, loop built from scan, room for 32 basis
                 vectors at the start of each solve, doubling when full
  while 32       the same with the loop built from while_loop
  scan 100       room for max_krylov = 100 vectors from the start (no growth)

The process starts with a throwaway JAX run: the first JAX run in a process
is slower per iteration (by about 35 ms per iteration at n = 3,969 on the
2-core machine used), whatever the loop, which confounded the scan / while
comparison of phase 6.

Reported per run: the median time of an iteration, the rest of the wall
time ("compile"), and the number of capacity growths.

Run:  python experiments/jax_backend/capacity.py [--sizes 63 127] [--json capacity.json]
"""

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import jax.numpy as jnp                      # noqa: E402

import jax_backend                           # noqa: E402  (turns on float64)
from solvers import configurations, numpy_options, per_iteration  # noqa: E402
from study import numpy_wrapped, plaplace    # noqa: E402
from nicks_optimizers import RunSummary, array_space, trust_region_newton  # noqa: E402

RUNS = {"numpy": None, "scan 32": ("scan", 32), "while 32": ("while", 32), "scan 100": ("scan", 100)}


def run(pieces: dict[str, Any], np_pieces: dict[str, Any], options: dict[str, Any], N: int,
        how: Any) -> tuple[RunSummary, float]:
    t = time.perf_counter()
    if how is None:
        _, summary = trust_region_newton(np_pieces["fun"], np_pieces["grad"], np_pieces["hessvec"],
                                         np.zeros(N * N), max_iter=300, max_krylov=100,
                                         **numpy_options(options))
    else:
        loop, cap = how
        bk = jax_backend.backend(loop)
        _, summary = trust_region_newton(
            pieces["fun"], pieces["grad"], pieces["hessvec"], jnp.zeros(N * N),
            space=array_space(bk), retract=lambda x, s: x + s, backend=bk, krylov_capacity=cap,
            max_iter=300, max_krylov=100, **options)
    return summary, time.perf_counter() - t


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[63, 127])
    parser.add_argument("--json", help="write the results to this file")
    args = parser.parse_args()

    warm = plaplace(15)                      # the throwaway run (see the module docstring)
    run(warm, numpy_wrapped(warm), configurations(15, warm)["lanczos"], 15, ("scan", 8))

    rows: list[dict[str, Any]] = []
    print(f"{'n':>6s} {'solver':13s} {'run':9s} {'its':>4s} {'Krylov':>6s} {'grow':>4s} "
          f"{'df/f':>8s} {'compile':>8s} {'per it':>9s}")
    for N in args.sizes:
        pieces = plaplace(N)
        np_pieces = numpy_wrapped(pieces)
        for solver, options in configurations(N, pieces).items():
            reference: RunSummary | None = None
            for name, how in RUNS.items():
                summary, wall = run(pieces, np_pieces, options, N, how)
                reference = summary if reference is None else reference
                m = min(len(summary.iterations), len(reference.iterations), 20)
                df = max(abs(a.f - b.f) / max(abs(b.f), 1e-300)
                         for a, b in zip(summary.iterations[:m], reference.iterations[:m]))
                per_it = per_iteration(summary)
                row = dict(n=N * N, solver=solver, run=name, iterations=len(summary.iterations),
                           krylov=summary.hessvec_calls, growths=summary.capacity_growths,
                           f_rel_diff=df, compiling=wall - len(summary.iterations) * per_it,
                           per_iteration=per_it, wall=wall,
                           same_counts=(len(summary.iterations) == len(reference.iterations)
                                        and summary.hessvec_calls == reference.hessvec_calls))
                rows.append(row)
                print(f"{N * N:6d} {solver:13s} {name:9s} {row['iterations']:4d} {row['krylov']:6d} "
                      f"{row['growths']:4d} {df:8.1e} {row['compiling']:7.2f}s "
                      f"{per_it * 1e3:7.2f}ms", flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
