"""The sphere rows of compare.py from 8 starts (x0, and x0 moved by 1e-12
and renormalized): single runs of a few thousand iterations spread by some
15% from rounding alone, so single-run differences of that size say
nothing. Medians and ranges of iterations, with the projection transport
and without a transport, for memory 0, 1, 3 and 10.

Run:  python experiments/lbfgs_memory/sphere_starts.py [--json sphere_starts.json]
"""

import argparse
import json

import numpy as np

from compare import problems
from nicks_optimizers import lbfgs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", help="write the results to this file")
    args = parser.parse_args()
    make = dict(problems())
    rows = []
    for name in ("sphere, projection", "sphere, no transport"):
        cells = []
        for m in (0, 1, 3, 10):
            counts = []
            for start in range(8):
                options = make[name]()
                fun, grad, x0 = options.pop("fun"), options.pop("grad"), options.pop("x0")
                if start:
                    x0 = x0 + 1e-12 * np.random.default_rng(start).standard_normal(x0.shape)
                    x0 = x0 / np.linalg.norm(x0)
                _, s = lbfgs(fun, grad, x0, memory=m, max_iter=20000, **options)
                rows.append(dict(problem=name, memory=m, start=start, reason=s.reason,
                                 iterations=len(s.iterations), fun=s.fun_calls))
                counts.append(len(s.iterations))
            cells.append(f"m={m}: {int(np.median(counts))} [{min(counts)}-{max(counts)}]")
        print(f"{name:22s} " + "   ".join(cells), flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
