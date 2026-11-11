"""hessian="switch" against pure Gauss-Newton and the full Hessian from the
start (dev/gauss_newton_switch_plan.md, phase 6), on the switching study's
three problems, with both Newton optimizers of the library.

Variants per case and method
  gn        hessvec = the Gauss-Newton Hessian (forcing power 0.5)
  switch    hessian="switch" (tau 0.25, k 2) with forcing powers 0.5
            before and 1.0 after the switch (the default when this ran)
  switch-p0.5  hessian="switch" with the power 0.5 in both phases (the
            default since the plan's D10)
  full      hessvec = the full Hessian, forcing power 0.5
  full-p1   the full Hessian, forcing power 1.0 (the switch run's power
            after the switch)

Every run: library defaults except rtol = 1e-9 (the study's stop, relative
to ||g(x0)||), max_iter = 500 and max_krylov = 200. Work: products with
each Hessian (the switch's tracking products included), PDE solves for the
diffusion problem (all of them: states, adjoints, products), wall time.

Run:  python compare.py [--only rosenbrock,diffusion,mc] [--jobs 8] [--out results.jsonl]
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, HERE)
from nicks_optimizers import line_search_newton, trust_region_newton  # noqa: E402
from problems import regularized_rosenbrock  # noqa: E402
from study_problems import DiffusionInversion, MatrixCompletion  # noqa: E402

VARIANTS = ("gn", "switch", "switch-p0.5", "full", "full-p1")
METHODS = {"TR": trust_region_newton, "LS": line_search_newton}
OPTIONS = dict(rtol=1e-9, max_iter=500, max_krylov=200)


def cases() -> list[tuple[str, str, dict[str, Any]]]:
    out: list[tuple[str, str, dict[str, Any]]] = []
    for n, alpha, cscale, seed in [(200, .1, .5, 1), (200, .1, .5, 2), (200, .1, .5, 3),
                                   (400, .1, .5, 5), (200, .03, .5, 1), (200, .1, 1., 1),
                                   (200, 1., .5, 1)]:
        out.append(("rosenbrock", f"n={n} alpha={alpha} c={cscale} seed={seed}",
                    dict(n=n, alpha=alpha, cscale=cscale, seed=seed)))
    for beta in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9):
        out.append(("diffusion", f"beta={beta:g}", dict(beta=beta)))
    mc: list[tuple[str, dict[str, Any]]] = [(f"lam={lam:g}", dict(lam=lam))
                                            for lam in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)]
    mc += [("noise 10%", dict(noise=0.1)), ("noise 0.1%", dict(noise=0.001)),
           ("rank 5", dict(r=5)), ("rank 20", dict(r=20)), ("15% observed", dict(obs_frac=0.15)),
           ("40% observed", dict(obs_frac=0.40)), ("seed 1", dict(seed=1)), ("seed 2", dict(seed=2)),
           ("seed 3", dict(seed=3)), ("start scale 1.0", dict(init_scale=1.0)),
           ("m = n = 400", dict(m=400, n=400))]
    out += [("mc", label, kw) for label, kw in mc]
    return out


def build(family: str, kw: dict[str, Any]) -> tuple[Any, Any, Any, Any, Any, Any]:
    """(fun, grad, gn_hessvec, hessvec, x0, problem object or None)."""
    if family == "rosenbrock":
        p = regularized_rosenbrock(**kw)
        return p.fun, p.grad, p.gn_hessvec, p.hessvec, p.x0, None
    if family == "diffusion":
        d = DiffusionInversion(**kw)
        return d.fun, d.grad, d.gn_hessvec, d.hessvec, d.x0(), d
    kw = dict(kw)
    scale = kw.pop("init_scale", 0.3)
    m = MatrixCompletion(**kw)
    return m.fun, m.grad, m.gn_hessvec, m.hessvec, m.x0(scale), m


def run(job: tuple[str, str, dict[str, Any], str, str]) -> dict[str, Any]:
    family, label, kw, method, variant = job
    fun, grad, gn, full, x0, obj = build(family, kw)
    opts: dict[str, Any] = dict(OPTIONS)
    if variant == "gn":
        hv = gn
    elif variant.startswith("switch"):
        hv, opts = full, dict(opts, hessian="switch", gn_hessvec=gn)
        opts["pre_switch_forcing_power"] = 0.5
        opts["forcing_power"] = 0.5 if variant == "switch-p0.5" else 1.0
    else:
        hv = full
        if variant == "full-p1":
            opts["forcing_power"] = 1.0
    start = time.perf_counter()
    _, s = METHODS[method](fun, grad, hv, x0, **opts)
    elapsed = time.perf_counter() - start
    gn_calls = (s.gn_hessvec_calls if variant.startswith("switch")
                else (s.hessvec_calls if variant == "gn" else 0))
    full_calls = s.hessvec_calls if variant != "gn" else 0
    return dict(family=family, case=label, method=method, variant=variant, reason=s.reason,
                iterations=len(s.iterations), gn_hv=gn_calls, full_hv=full_calls,
                pde_solves=None if family != "diffusion" else obj.nsolve,
                switch_iteration=s.switch_iteration, f=s.f, gnorm_ratio=s.gnorm / s.gnorm0,
                time=elapsed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="rosenbrock,diffusion,mc")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default=os.path.join(HERE, "results.jsonl"))
    args = ap.parse_args()
    families = args.only.split(",")
    done: set[tuple[str, str, str, str]] = set()
    if os.path.exists(args.out):                      # resume: skip the runs already recorded
        with open(args.out) as f:
            done = {(r["family"], r["case"], r["method"], r["variant"])
                    for r in map(json.loads, f)}
    jobs = [(f, label, kw, m, v) for f, label, kw in cases() if f in families
            for m in METHODS for v in VARIANTS if (f, label, m, v) not in done]
    with open(args.out, "a") as out, ProcessPoolExecutor(args.jobs) as pool:
        for result in pool.map(run, jobs):
            out.write(json.dumps(result) + "\n")
            out.flush()
            print(f"{result['family']:>10} {result['case']:>30} {result['method']} "
                  f"{result['variant']:>8} {result['reason']:>9} its {result['iterations']:>3} "
                  f"gn {result['gn_hv']:>6} full {result['full_hv']:>6} "
                  f"switch {result['switch_iteration']} {result['time']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
