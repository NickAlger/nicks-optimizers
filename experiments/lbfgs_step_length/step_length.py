"""Is memory 0's weakness its step length? Directions and step lengths of
memory 0 and memory 1, crossed.

lbfgs with memory 0 steps along -gamma M^{-1} g, refitting gamma to each pair
that passes the curvature test and keeping it otherwise. With memory 1 the
direction is -H1 g, H1 the BFGS update of gamma M^{-1} by the newest pair.
These arms separate the direction from its length (the trial step, alpha = 1):

  A    memory 0: -gamma M^{-1} g
  B1   memory 0's direction, with memory 1's length: memory 1's pair is kept
       and tested as memory 1 does, and the step is -t M^{-1} g with t the
       minimizer of memory 1's quadratic model along -M^{-1} g (the Cauchy
       step of B1 = H1^{-1}): t = ||g||^2_{M^{-1}} / <M^{-1} g, B1 M^{-1} g>
  B2   the same, with t = <g, H1 g> / ||g||^2_{M^{-1}}: the decrease <g, d>
       that memory 1's step predicts
  C1   memory 1's direction, with memory 0's length: c d1, d1 = -H1 g, with c
       the minimizer of memory 0's model (B0 = M / gamma) along d1:
       c = gamma <g, H1 g> / ||d1||_M^2
  C2   the same, with c = gamma ||g||^2_{M^{-1}} / <g, H1 g>: memory 0's
       predicted decrease
  D    memory 1: -H1 g
  E1   memory 0, and when a pair fails the curvature test gamma is refitted
       anyway as ||s||_M / ||y||_{M^{-1}} (positive whatever the sign of
       <y, s>); a pair that passes gets the usual fit <y, s> / <y, M^{-1} y>
  E2   memory 0, and when a pair fails the curvature test after a unit step,
       gamma doubles

With an empty memory every arm steps along -gamma M^{-1} g (both lengths of B
reduce to gamma there, and C to memory 1's own step). Everything else is
lbfgs's: the line search (line_search.backtracking from alpha = 1), the
curvature test (<y, s> > 1e-8 |<g, s>|), the angle test and fallback for
memory 1's direction, restarts after a failed line search with a pair kept,
gamma's fit, the stopping test (||g||_{M^{-1}} <= 1e-8 ||g0||_{M^{-1}}), the
transport of the pair and of g on the sphere. Arms A and D are checked
against lbfgs with memory 0 and 1 (--validate).

Problems: those of experiments/lbfgs_memory/compare.py without completion
(the convex quartic, Rosenbrock with M = I and the Jacobi metric, the
p-Laplacian at N = 63 with M = I, H1 and the Kacanov metric, the sphere).

Each arm runs from 8 starts (x0, and x0 moved by 1e-12): over thousands of
iterations rounding differences grow until single iteration counts spread
by some 15%, so medians and ranges are reported. lbfgs with memory 3 and 10
(and 1) is run from the same starts for reference.

Run:  python experiments/lbfgs_step_length/step_length.py [--validate] [--json results.json]
          [--starts 8] [--only sphere ...]
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lbfgs_memory"))
import compare as memory_compare                                     # noqa: E402
from nicks_optimizers import backtracking, identity_transport, lbfgs, numpy_backend  # noqa: E402
from nicks_optimizers.line_search import ARMIJO                      # noqa: E402

ARMS: dict[str, tuple[str, str]] = {          # arm: (direction, length)
    "A": ("m0", "m0"),
    "B1": ("m0", "m1-cauchy"),
    "B2": ("m0", "m1-slope"),
    "C1": ("m1", "m0-cauchy"),
    "C2": ("m1", "m0-slope"),
    "D": ("m1", "m1"),
    "E1": ("m0", "m0-geometric"),
    "E2": ("m0", "m0-double"),
}
PROBLEMS = ("convex, M = I", "rosenbrock, M = I", "rosenbrock, Jacobi(x)", "p-laplacian, M = I",
            "p-laplacian, H1", "p-laplacian, Kacanov(u)", "sphere, projection")


@dataclass
class Problem:
    fun: Callable[[Any], float]
    grad: Callable[[Any], Any]
    x0: Any
    ops: Callable[[Any], tuple[Callable[[Any], Any], Callable[[Any], Any]]]     # x -> (M^{-1}, M)
    retract: Callable[[Any, Any], Any]
    transport: Any
    canonical: Callable[[Any, Any], Any]
    options: dict[str, Any]                   # lbfgs's options, for --validate


def problem(name: str) -> Problem:
    options = dict(dict(memory_compare.problems())[name]())
    fun, grad, x0 = options.pop("fun"), options.pop("grad"), options.pop("x0")
    for key in ("fixed_metric", "space"):     # same iterates; this driver works on numpy arrays
        options.pop(key, None)

    def ops(x: Any) -> tuple[Callable[[Any], Any], Callable[[Any], Any]]:
        if "operators_at" in options:
            data = options["point_data"](x) if options.get("point_data") else None
            po = options["operators_at"](x, data)
            return po.metric_solve, po.metric_apply
        if "metric_solve" in options:
            ms, ma = options["metric_solve"], options["metric_apply"]
            return (lambda w: ms(x, w)), (lambda v: ma(x, v))
        return (lambda w: w), (lambda v: v)

    project = options.get("project_covector")
    return Problem(fun=fun, grad=grad, x0=np.asarray(x0, dtype=float), ops=ops,
                   retract=options.get("retract", lambda x, s: x + s),
                   transport=options.get("transport", identity_transport),
                   canonical=(lambda x, w: w) if project is None else project, options=options)


def run(pb: Problem, arm: str, max_iter: int = 20000, rtol: float = 1e-8, c1: float = 1e-4,
        max_backtracks: int = 30, angle_tol: float = 1e-8, curvature_tol: float = 1e-8,
        trace: bool = False) -> dict[str, Any]:
    direction_rule, length_rule = ARMS[arm]
    keeps_pair = direction_rule == "m1" or length_rule.startswith("m1")
    identity = pb.transport is identity_transport
    x = pb.x0
    f = pb.fun(x)
    g = pb.canonical(x, pb.grad(x))
    solve, apply = pb.ops(x)
    Mg = solve(g)
    G = float(g @ Mg)
    gnorm0 = np.sqrt(G)
    gamma = 1.0
    pair: Optional[tuple[Any, Any, float]] = None        # (s, y, rho) at x
    fun_calls, grad_calls, skipped, restarts, fallbacks, unit_steps, accepted = 1, 1, 0, 0, 0, 0, 0
    ratios: list[float] = []                  # trial step length / memory 0's (B and C)
    xs, gammas, alphas = [x], [], []
    reason = "max_iter"
    it = 0
    while it < max_iter:
        if np.sqrt(G) <= rtol * gnorm0:
            reason = "converged"
            break
        it += 1
        fallback = False
        Hg = None
        if pair is not None:                 # the two-loop recursion with H0 = gamma M^{-1}
            s_, y_, rho = pair
            a = rho * float(s_ @ g)
            r = gamma * solve(g - a * y_)
            Hg = r + (a - rho * float(y_ @ r)) * s_
        if direction_rule == "m0":
            t = gamma
            if pair is not None and length_rule == "m1-cauchy":
                s_, y_, rho = pair
                sBs = float(apply(s_) @ s_) / gamma
                vBv = G / gamma - (float(g @ s_) / gamma) ** 2 / sBs + float(y_ @ Mg) ** 2 / float(y_ @ s_)
                t = G / vBv
            elif pair is not None and length_rule == "m1-slope":
                assert Hg is not None
                t = float(g @ Hg) / G
            if pair is not None:
                ratios.append(t / gamma)
            d = -t * Mg
        elif pair is None:
            d = -gamma * Mg
        else:
            assert Hg is not None
            d1 = -Hg
            slope1 = float(g @ d1)
            norm1 = np.sqrt(max(float(apply(d1) @ d1), 0.0))
            cosine = -slope1 / (np.sqrt(G) * norm1) if norm1 > 0.0 else 0.0
            if not (np.isfinite(slope1) and slope1 < 0.0 and cosine >= angle_tol):
                fallback, pair = True, None
                fallbacks += 1
                d = -gamma * Mg
            else:
                c = 1.0
                if length_rule == "m0-cauchy":
                    c = gamma * (-slope1) / norm1 ** 2
                elif length_rule == "m0-slope":
                    c = gamma * G / (-slope1)
                if length_rule != "m1":
                    ratios.append(c * norm1 / (gamma * np.sqrt(G)))
                d = c * d1
        slope = float(g @ d)

        def phi(alpha: Any) -> tuple[Any, Any]:
            point = pb.retract(x, alpha * d)
            return pb.fun(point), point
        found = backtracking(numpy_backend, phi, x, f, slope, 1.0, c1=c1, max_trials=max_backtracks)
        trials = int(found.trials)
        fun_calls += trials
        if int(found.status) != ARMIJO:
            if pair is not None and not fallback:
                pair = None
                restarts += 1
                continue
            reason = "line_search"
            break
        alpha = float(found.alpha)
        accepted += 1
        unit_steps += alpha == 1.0
        x1, f1 = found.point, float(found.f)
        g1 = pb.canonical(x1, pb.grad(x1))
        grad_calls += 1
        solve1, apply1 = pb.ops(x1)
        s = alpha * d
        if identity:
            s1, gk = s, g
        else:
            s1 = pb.transport.vector(x, x1, s)
            gk = pb.canonical(x1, pb.transport.covector(x, x1, g))
            if pair is not None:
                pair = (pb.transport.vector(x, x1, pair[0]), pb.transport.covector(x, x1, pair[1]), pair[2])
        y = g1 - gk
        ys = float(y @ s1)
        ok = np.isfinite(ys) and ys > curvature_tol * abs(alpha * slope)
        if ok:
            yy = float(y @ solve1(y))
            fit = ys / yy if yy > 0.0 else np.nan
            gamma = fit if np.isfinite(fit) and fit > 0.0 else gamma
            if keeps_pair:
                pair = (s1, y, 1.0 / ys)
        else:
            skipped += 1
            if length_rule == "m0-geometric":
                fit = np.sqrt(float(apply1(s1) @ s1) / float(y @ solve1(y)))
                gamma = fit if np.isfinite(fit) and fit > 0.0 else gamma
            elif length_rule == "m0-double" and alpha == 1.0:
                gamma *= 2.0
        x, f, g, solve, apply = x1, f1, g1, solve1, apply1
        Mg = solve(g)
        G = float(g @ Mg)
        if trace:
            xs.append(np.array(x))
            gammas.append(gamma)
            alphas.append(alpha)
    out: dict[str, Any] = dict(
        arm=arm, reason=reason, iterations=it, fun=fun_calls, grad=grad_calls, f=f,
        gnorm_ratio=float(np.sqrt(G) / gnorm0), skipped=skipped, restarts=restarts, fallbacks=fallbacks,
        unit_steps=unit_steps / max(accepted, 1),
        ratio_median=float(np.median(ratios)) if ratios else None,
        ratio_quartiles=[float(q) for q in np.quantile(ratios, [0.25, 0.75])] if ratios else None)
    if trace:
        out.update(xs=xs, gammas=gammas, alphas=alphas)
    return out


def lbfgs_iterates(pb: Problem, memory: int, max_iter: int = 20000) -> tuple[list[Any], Any]:
    xs = [pb.x0]
    _, s = lbfgs(pb.fun, pb.grad, pb.x0, memory=memory, max_iter=max_iter,
                 callback=lambda x, info: xs.append(np.array(x)) if info.record.accepted else None,
                 **pb.options)
    return xs, s


def validate() -> None:
    """Arms A and D against lbfgs with memory 0 and memory 1."""
    print(f"{'problem':26s} {'arm':4s} {'driver':>8s} {'lbfgs':>8s}  max rel. difference of the first 40 iterates")
    for name in PROBLEMS:
        for arm, memory in (("A", 0), ("D", 1)):
            pb = problem(name)
            mine = run(pb, arm, trace=True)
            pb = problem(name)
            theirs, s = lbfgs_iterates(pb, memory)
            k = min(40, len(mine["xs"]), len(theirs))
            diff = max(float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))
                       for a, b in zip(mine["xs"][:k], theirs[:k]))
            print(f"{name:26s} {arm:4s} {mine['iterations']:8d} {len(s.iterations):8d}  {diff:.1e}", flush=True)


LBFGS_ARMS = {"m1": 1, "m3": 3, "m10": 10}   # lbfgs itself, for reference


def perturbed(pb: Problem, start: int) -> Problem:
    """Start 0 is the problem's x0; start k > 0 moves it by 1e-12 times a
    standard normal vector (seed k), back onto the sphere there. Long runs
    amplify rounding: iteration counts from such starts spread by some 15%,
    so each arm is run from several."""
    if start > 0:
        x0 = pb.x0 + 1e-12 * np.random.default_rng(start).standard_normal(pb.x0.shape)
        pb.x0 = x0 / np.linalg.norm(x0) if pb.transport is not identity_transport else x0
    return pb


def run_lbfgs(pb: Problem, memory: int, max_iter: int = 20000) -> dict[str, Any]:
    _, s = lbfgs(pb.fun, pb.grad, pb.x0, memory=memory, max_iter=max_iter, **pb.options)
    accepted = [r for r in s.iterations if r.accepted]
    return dict(reason=s.reason, iterations=len(s.iterations), fun=s.fun_calls, grad=s.grad_calls, f=float(s.f),
                gnorm_ratio=float(s.gnorm / s.gnorm0), skipped=s.skipped, restarts=s.restarts,
                fallbacks=sum(r.direction == "fallback" for r in s.iterations),
                unit_steps=sum(r.alpha == 1.0 for r in accepted) / max(len(accepted), 1),
                ratio_median=None, ratio_quartiles=None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--validate", action="store_true", help="check arms A and D against lbfgs")
    parser.add_argument("--json", help="write the results to this file (after every problem)")
    parser.add_argument("--starts", type=int, default=8, help="starts per arm (default 8)")
    parser.add_argument("--only", nargs="+", help="run the problems whose names contain one of these")
    parser.add_argument("--arms", nargs="+", help="run only these arms")
    args = parser.parse_args()
    if args.validate:
        validate()
        return
    rows: list[dict[str, Any]] = []
    arms = [arm for arm in list(ARMS) + list(LBFGS_ARMS) if not args.arms or arm in args.arms]
    print(f"{'problem':24s} " + " ".join(f"{arm:>16s}" for arm in arms)
          + "   (median iterations [min-max] over the starts; *k = k starts not converged)")
    for name in PROBLEMS:
        if args.only and not any(word in name for word in args.only):
            continue
        cells = []
        for arm in arms:
            counts, failures = [], 0
            for start in range(args.starts):
                pb = perturbed(problem(name), start)
                t0 = time.perf_counter()
                row = run(pb, arm) if arm in ARMS else run_lbfgs(pb, LBFGS_ARMS[arm])
                row.update(problem=name, arm=arm, start=start, seconds=time.perf_counter() - t0)
                rows.append(row)
                counts.append(row["iterations"])
                failures += row["reason"] != "converged"
            mark = f"*{failures}" if failures else ""
            cells.append(f"{int(np.median(counts))} [{min(counts)}-{max(counts)}]{mark}")
            print(f"  {name}, {arm}: {cells[-1]}", file=sys.stderr, flush=True)
        print(f"{name:24s} " + " ".join(f"{c:>16s}" for c in cells), flush=True)
        if args.json:
            with open(args.json, "w") as fh:
                json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
