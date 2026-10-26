"""The memory on the p-Laplacian, as the mesh is refined and as p grows.

compare.py runs the p-Laplacian at N = 63, p = 4. Here: memory 0, 1, 3, 10
and 30 for the Euclidean, H1 and Kacanov(u) configurations of
examples/lbfgs_p_laplacian.py, at N = 31, 63, 127 and 255 (the Euclidean
metric to N = 127), and at N = 63 for p = 3, 4 and 6 (p = 8 is left out: at
the start u = 0 the Kacanov metric is h^2 eps^(p-2) times a Laplacian, its
first step is too long for 30 backtracks, and the run stops at once).

Run:  python experiments/lbfgs_memory/p_laplacian.py [--json p_laplacian.json]
"""

import argparse
import json
import os
import sys
import time
from typing import Any

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "examples"))
import lbfgs_p_laplacian as plx                                      # noqa: E402
from nicks_optimizers import lbfgs                                   # noqa: E402

MEMORIES = (0, 1, 3, 10, 30)


def sweep(label: str, N: int, p: float, config: str, rows: list[dict[str, Any]]) -> None:
    cells = []
    for m in MEMORIES:
        problem = plx.p_laplacian(N, p=p)
        options = {k: v for k, v in problem.configs[config].items() if k != "memory"}
        t = time.perf_counter()
        _, s = lbfgs(problem.fun, problem.grad, problem.x0, memory=m, max_iter=20000, **options)
        rows.append(dict(sweep=label, N=N, p=p, config=config, memory=m, reason=s.reason,
                         iterations=len(s.iterations), fun=s.fun_calls, f=float(s.f),
                         seconds=time.perf_counter() - t))
        cells.append(f"{len(s.iterations)}{'' if s.reason == 'converged' else '*'}")
    print(f"{label:5s} N={N:<4d} p={p:<4g} {config:10s} " + " ".join(f"{c:>7s}" for c in cells), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", help="write the results to this file")
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    print(f"{'':30s}" + " ".join(f"{'m=' + str(m):>7s}" for m in MEMORIES) + "   (iterations)")
    for N in (31, 63, 127, 255):
        for config in ("H1", "Kacanov", "Euclidean"):
            if config != "Euclidean" or N <= 127:
                sweep("mesh", N, 4.0, config, rows)
    for p in (3.0, 4.0, 6.0):
        for config in ("H1", "Kacanov"):
            sweep("p", 63, p, config, rows)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
