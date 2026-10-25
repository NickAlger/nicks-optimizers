"""What finished steps cost in a scan-built bounded loop: per-iteration time
of the compiled CG path against max_krylov, for loops built from scan and
from while_loop (dev/jit_backends_plan.md, phase 3). The CG solves here use
about 12 Krylov steps per iteration, so a scan runs max_krylov - 12 skipped
steps per iteration, carrying five vectors of length n through each.

Run:  python experiments/jax_backend/skipped_steps.py [--N 127]
"""

import argparse
import os
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import jax_backend                     # noqa: E402
from study import plaplace             # noqa: E402
from nicks_optimizers import trust_region_newton  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--N", type=int, nargs="+", default=[63, 127])
    args = parser.parse_args()
    print(f"{'n':>6s} {'max_krylov':>10s} {'scan ms/it':>11s} {'while ms/it':>12s} {'skipped/it':>10s}")
    for N in args.N:
        p = plaplace(N)
        for max_krylov in (25, 100, 400):
            per_it = {}
            for loop in ("scan", "while"):
                _, s = trust_region_newton(p["fun"], p["grad"], p["hessvec"], jnp.zeros(N * N),
                                           metric_solve=p["metric_solve"], max_krylov=max_krylov,
                                           max_iter=300, backend=jax_backend.backend(loop))
                e = [r.elapsed for r in s.iterations]
                per_it[loop] = (e[-1] - e[0]) / (len(e) - 1)
                krylov_per_it = s.hessvec_calls / len(s.iterations)
            print(f"{N * N:6d} {max_krylov:10d} {per_it['scan'] * 1e3:10.2f} {per_it['while'] * 1e3:11.2f} "
                  f"{max_krylov - krylov_per_it:10.1f}", flush=True)


if __name__ == "__main__":
    main()
