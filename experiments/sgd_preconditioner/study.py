"""Trust-region Newton for matrix completion with SGD and block Jacobi
preconditioners: rank continuation against a cold start at full rank, for
uniform and power-law sampling.

The problem is the example's (1000 x 800, singular values 2^-k, rank-8 fit,
4 samples per degree of freedom). Power-law sampling draws rows and columns
with probability proportional to (index + 1)^-1, after giving every row and
column 8 uniform samples so that the fit stays identifiable.

Strategies:
  continuation  ranks 1, ..., 8 in turn from X = 0, as the example does
  cold          rank 8 directly from X = 0 (radius 1), stopped after 150
                iterations or the time budget

Solvers and preconditioners (metric: Frobenius, M = I in the components):
  CG            krylov="auto" (P = M = I), the example's setting
  flexible      krylov="flexible" with the named preconditioner:
                  identity, block Jacobi, SGD variants (see preconditioners.py)

Reported: iterations, Krylov steps, passes over the samples (objective,
gradient, Hessian products and preconditioner work; a Hessian product is one
pass), wall time, and the final error ||X - A|| / ||A|| against the best
rank-8 error.

Run:  python experiments/sgd_preconditioner/study.py --problem uniform --strategy continuation
"""
import argparse
import json
import os
import sys
import time
from typing import Any, Callable, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples"))
sys.path.insert(0, os.path.dirname(__file__))
import matrix_completion as mc  # noqa: E402
from nicks_optimizers import PointOperators, RunSummary, euclidean_metric_solve, trust_region_newton  # noqa: E402
from preconditioners import SGDConfig, batch_samples, block_jacobi, cauchy_sgd, partition  # noqa: E402

PRECONDITIONERS: dict[str, Optional[SGDConfig]] = {
    "CG": None, "identity": None, "block Jacobi": None,
    "SGD GN cyclic 1x10": SGDConfig(1, 10, "cyclic"),
    "SGD GN reshuffled 1x10": SGDConfig(1, 10, "reshuffled"),
    "SGD GN cyclic 3x10 avg": SGDConfig(3, 10, "cyclic", averaging=True),
    "SGD GN reshuffled 3x10 avg": SGDConfig(3, 10, "reshuffled", averaging=True),
    "SGD GN cyclic 3x4 svrg": SGDConfig(3, 4, "cyclic", svrg=True),
    "SGD full reshuffled 1x10": SGDConfig(1, 10, "reshuffled", curvature=True),
}


def sample_power_law(a: mc.LowRank, K: int, rng: np.random.Generator, exponent: float = 1.0,
                     per_line: int = 8) -> mc.Samples:
    N, M = a[0].shape[0], a[2].shape[0]
    chosen: set[tuple[int, int]] = set()
    for i in range(N):
        chosen.update((i, int(j)) for j in rng.choice(M, per_line, replace=False))
    for j in range(M):
        chosen.update((int(i), j) for i in rng.choice(N, per_line, replace=False))
    pr = (np.arange(N) + 1.0) ** -exponent
    pc = (np.arange(M) + 1.0) ** -exponent
    pr, pc = pr / pr.sum(), pc / pc.sum()
    while len(chosen) < K:
        need = K - len(chosen)
        chosen.update(zip(rng.choice(N, need, p=pr).tolist(), rng.choice(M, need, p=pc).tolist()))
    idx = np.array(sorted(chosen))[rng.permutation(len(chosen))[:K]]
    rows, cols = idx[:, 0].astype(np.intp), idx[:, 1].astype(np.intp)
    return mc.Samples(shape=(N, M), rows=rows, cols=cols, values=mc.sampled_entries(a, rows, cols))


def solve(samples: mc.Samples, x0: mc.LowRank, name: str, seed: int, radius: float, rtol: float,
          max_iter: int, deadline: float, counter: dict[str, float],
          cyclic_batches: dict[int, list[mc.Samples]]) -> tuple[mc.LowRank, RunSummary]:
    """One trust-region run with the named preconditioner."""
    cfg = PRECONDITIONERS[name]
    rng = np.random.default_rng([seed, 1])
    common: dict[str, Any] = dict(
        space=mc.fixed_rank_space(), retract=mc.retract, project_covector=mc.gauge_projection,
        radius=radius, rtol=rtol, max_iter=max_iter, max_krylov=50,
        callback=lambda x, info: time.perf_counter() > deadline)
    fun = lambda x: mc.objective(samples, x)                     # noqa: E731
    grad = lambda x: mc.gradient(samples, x)                     # noqa: E731
    if name == "CG":
        return trust_region_newton(fun, grad, lambda x, v: mc.hessian_action(samples, x, v), x0,
                                   metric_solve=euclidean_metric_solve, **common)

    def operators_at(x: mc.LowRank) -> PointOperators[mc.Tangent, mc.Cotangent]:
        if name == "identity":
            P: Callable[[mc.Cotangent], mc.Tangent] = lambda w: mc.Tangent(w)  # noqa: E731
        elif name == "block Jacobi":
            P = block_jacobi(samples, x, counter)
        else:
            assert cfg is not None
            if cfg.order == "cyclic" and cfg.batches not in cyclic_batches:
                cyclic_batches[cfg.batches] = [batch_samples(samples, idx) for idx in
                                               partition(samples.rows.shape[0], cfg.batches, rng)]
            P = cauchy_sgd(samples, x, cfg, rng, counter, cyclic_batches.get(cfg.batches))
        return PointOperators(hessvec=lambda v: mc.hessian_action(samples, x, v),
                              metric_solve=lambda w: mc.Tangent(w), metric_apply=lambda v: mc.Cotangent(v),
                              precond_solve=P)

    return trust_region_newton(fun, grad, None, x0, point_data=operators_at,
                               operators_at=lambda x, ops: ops, krylov="flexible", **common)


def run(samples: mc.Samples, a: mc.LowRank, best: float, strategy: str, name: str, seed: int,
        budget: float) -> dict[str, Any]:
    rng = np.random.default_rng([seed, 2])
    counter = {"passes": 0.0, "applications": 0.0}
    cyclic: dict[int, list[mc.Samples]] = {}
    start = time.perf_counter()
    deadline = start + budget
    summaries = []
    r = 8
    if strategy == "continuation":
        x, radius = mc.zero_point(samples.shape, 1, rng), 1.0
        for k in range(1, r + 1):
            if k > 1:
                radius = float(x[1][-1])
                x = mc.add_zero_component(x, rng)
            x, summary = solve(samples, x, name, seed, radius, 1e-10 if k == r else 1e-4, 200,
                               deadline, counter, cyclic)
            summaries.append(summary)
            if summary.reason == "callback":
                break
    else:
        x, summary = solve(samples, mc.zero_point(samples.shape, r, rng), name, seed, 1.0, 1e-10, 150,
                           deadline, counter, cyclic)
        summaries.append(summary)
    elapsed = time.perf_counter() - start
    passes = sum(s.fun_calls + s.grad_calls + s.hessvec_calls for s in summaries) + counter["passes"]
    return {"strategy": strategy, "preconditioner": name, "seed": seed,
            "iterations": sum(len(s.iterations) for s in summaries),
            "krylov_steps": sum(r.subproblem.krylov_steps for s in summaries for r in s.iterations),
            "fallbacks": sum(r.subproblem.fallbacks for s in summaries for r in s.iterations),
            "passes": passes, "time": elapsed, "error": mc.relative_error(x, a), "best": best,
            "reason": summaries[-1].reason, "final_rank": int(np.sum(x[1] > 0))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--problem", choices=["uniform", "powerlaw"], default="uniform")
    parser.add_argument("--strategy", choices=["continuation", "cold"], default="continuation")
    parser.add_argument("--precond", nargs="+", default=list(PRECONDITIONERS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--budget", type=float, default=300.0, help="seconds per run")
    parser.add_argument("--json", help="append the results to this JSON-lines file")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    N, M, r = 1000, 800, 8
    d = 0.5 ** np.arange(30, dtype=float)
    a = mc.random_low_rank((N, M), d, rng)
    K = 4 * (N + M - r) * r
    samples = (mc.sample_uniformly(a, K, rng) if args.problem == "uniform"
               else sample_power_law(a, K, rng))
    best = float(np.sqrt(np.sum(d[r:] ** 2) / np.sum(d ** 2)))
    counts = np.bincount(samples.rows, minlength=N)
    print(f"{args.problem} sampling ({K} samples, per row {counts.min()} to {counts.max()}), "
          f"{args.strategy}; best rank-{r} error {best:.2e}")
    print(f"{'preconditioner':28s} {'seed':>4s} {'its':>5s} {'Krylov':>6s} {'passes':>7s} "
          f"{'time':>7s} {'error':>9s} {'/best':>6s}  reason")
    for name in args.precond:
        seeds = args.seeds if (PRECONDITIONERS[name] is not None
                               and PRECONDITIONERS[name].order == "reshuffled") else args.seeds[:1]
        for seed in seeds:
            res = run(samples, a, best, args.strategy, name, seed, args.budget)
            res["problem"] = args.problem
            print(f"{name:28s} {seed:4d} {res['iterations']:5d} {res['krylov_steps']:6d} "
                  f"{res['passes']:7.0f} {res['time']:6.1f}s {res['error']:9.2e} "
                  f"{res['error'] / best:6.2f}  {res['reason']}", flush=True)
            if args.json:
                with open(args.json, "a") as fh:
                    fh.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
