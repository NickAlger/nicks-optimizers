"""JAX proof of concept for the backend design (dev/jit_backends_plan.md,
phase 3): trust_region_newton with each iteration compiled by JAX, on the CG
path, against the same problems on the numpy backend.

Problems
  rosenbrock    chained Rosenbrock, n = 20 (the tests' problem), identity
                metric
  plaplace N    a finite-difference p-Laplacian-type energy on the unit
                square, N x N interior nodes, zero boundary values:
                  E(u) = h^2 [sum_edges phi(du/h) - sum f u],
                  phi(t) = (eps^2 + t^2)^(p/2) / p, p = 4, eps = 1e-2, f = 1,
                with jax.grad for the gradient and jax.jvp of it for Hessian
                products, and the discrete H^1 metric
                  M = h^2 (I + T (x) I + I (x) T),
                T the 1-D Dirichlet Laplacian, solved exactly with the sine
                eigenbasis (two dense N x N products per side). P = M, so the
                subproblems are solved by truncated CG.

Runs
  numpy         the numpy backend; for the JAX problems, every function is
                jax.jit-compiled on its own and called from the Python loop
                (arrays converted at each call): what a JAX user gets
                without compiled iterations
  jax-scan      JAX backend, bounded loop from jax.lax.scan
  jax-while     JAX backend, bounded loop from jax.lax.while_loop

Reported: iterations, Krylov steps, final f, agreement with the numpy run,
time per iteration after compilation (from the records' elapsed times),
first-iteration time (which includes compiling), and how often fun and
hessvec were traced (for the compiled runs, a constant however many
iterations and Krylov steps run).

Run:  python experiments/jax_backend/study.py [--sizes 63 127 255] [--json results.json]
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Callable

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests"))
import jax                       # noqa: E402
import jax.numpy as jnp          # noqa: E402

import jax_backend               # noqa: E402  (turns on float64)
from nicks_optimizers import RunSummary, trust_region_newton  # noqa: E402
from problems import rosen_fun, rosen_grad, rosen_hessvec     # noqa: E402


def counted(f: Callable[..., Any], counter: dict[str, int], key: str) -> Callable[..., Any]:
    """f, counting its Python calls: for compiled code, the number of traces."""
    def wrapped(*args: Any) -> Any:
        counter[key] = counter.get(key, 0) + 1
        return f(*args)
    return wrapped


def rosenbrock_jax() -> dict[str, Any]:
    def fun(x: Any) -> Any:
        return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)
    grad = jax.grad(fun)
    return dict(fun=fun, grad=grad, hessvec=lambda x, v: jax.jvp(grad, (x,), (v,))[1],
                metric_solve=None)


def plaplace(N: int, p: float = 4.0, eps: float = 1e-2) -> dict[str, Any]:
    h = 1.0 / (N + 1)

    def energy(u_flat: Any) -> Any:
        up = jnp.pad(u_flat.reshape(N, N), 1)
        gx = (up[1:, 1:-1] - up[:-1, 1:-1]) / h
        gy = (up[1:-1, 1:] - up[1:-1, :-1]) / h
        phi = lambda t: (eps ** 2 + t ** 2) ** (p / 2) / p           # noqa: E731
        return h ** 2 * (jnp.sum(phi(gx)) + jnp.sum(phi(gy)) - jnp.sum(u_flat))

    grad = jax.grad(energy)
    k = jnp.arange(1, N + 1)
    V = jnp.sqrt(2.0 / (N + 1)) * jnp.sin(jnp.pi * jnp.outer(k, k) / (N + 1))   # symmetric, V V = I
    lam = (4.0 / h ** 2) * jnp.sin(k * jnp.pi / (2 * (N + 1))) ** 2
    denom = h ** 2 * (1.0 + lam[:, None] + lam[None, :])

    def metric_solve(u: Any, w: Any) -> Any:
        W = w.reshape(N, N)
        return (V @ ((V @ W @ V) / denom) @ V).reshape(-1)

    return dict(fun=energy, grad=grad, hessvec=lambda u, v: jax.jvp(grad, (u,), (v,))[1],
                metric_solve=metric_solve)


def numpy_wrapped(pieces: dict[str, Any]) -> dict[str, Any]:
    """Each JAX function jit-compiled on its own, with numpy arrays in and out."""
    fun, grad, hv = jax.jit(pieces["fun"]), jax.jit(pieces["grad"]), jax.jit(pieces["hessvec"])
    out = dict(fun=lambda x: float(fun(x)), grad=lambda x: np.asarray(grad(x)),
               hessvec=lambda x, v: np.asarray(hv(x, v)), metric_solve=None)
    if pieces["metric_solve"] is not None:
        ms = jax.jit(pieces["metric_solve"])
        out["metric_solve"] = lambda x, w: np.asarray(ms(x, w))
    return out


def solve(pieces: dict[str, Any], x0: Any, backend: Any, counter: dict[str, int]) -> RunSummary:
    kwargs: dict[str, Any] = dict(max_iter=300, max_krylov=100)
    if pieces["metric_solve"] is not None:
        kwargs["metric_solve"] = pieces["metric_solve"]
    if backend is not None:
        kwargs["backend"] = backend
    _, summary = trust_region_newton(counted(pieces["fun"], counter, "fun"), pieces["grad"],
                                     counted(pieces["hessvec"], counter, "hessvec"), x0, **kwargs)
    return summary


def describe(name: str, problem: str, summary: RunSummary, reference: RunSummary, wall: float,
             counter: dict[str, int]) -> dict[str, Any]:
    its = len(summary.iterations)
    elapsed = [r.elapsed for r in summary.iterations]
    steady = (elapsed[-1] - elapsed[0]) / (its - 1) if its > 1 else float("nan")
    m = min(its, len(reference.iterations), 20)
    f_diff = max(abs(a.f - b.f) / max(abs(b.f), 1e-300)
                 for a, b in zip(summary.iterations[:m], reference.iterations[:m]))
    return dict(problem=problem, run=name, reason=summary.reason, iterations=its,
                krylov=summary.hessvec_calls, f=summary.f, gnorm_ratio=summary.gnorm / summary.gnorm0,
                f_rel_diff_first_20=f_diff, same_counts=(its == len(reference.iterations)
                                                         and summary.hessvec_calls == reference.hessvec_calls),
                wall=wall, first_iteration=elapsed[0], per_iteration=steady,
                fun_calls_or_traces=counter.get("fun", 0),
                hessvec_calls_or_traces=counter.get("hessvec", 0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[63, 127, 255])
    parser.add_argument("--json", help="write the results to this file")
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []

    problems: list[tuple[str, dict[str, Any], dict[str, Any], Any]] = [
        ("rosenbrock n=20", dict(fun=lambda x: np.asarray(rosen_fun(x)), grad=rosen_grad,
                                 hessvec=rosen_hessvec, metric_solve=None),
         rosenbrock_jax(), np.tile([-1.2, 1.0], 10))]
    for N in args.sizes:
        pieces = plaplace(N)
        problems.append((f"plaplace N={N} (n={N * N})", numpy_wrapped(pieces), pieces, np.zeros(N * N)))

    print(f"{'problem':26s} {'run':10s} {'its':>4s} {'Krylov':>6s} {'f':>14s} {'same':>5s} "
          f"{'df/f':>8s} {'first it':>9s} {'per it':>9s} {'traces f/Hv':>12s}")
    for problem, np_pieces, jax_pieces, x0 in problems:
        reference: RunSummary | None = None
        for name, pieces, backend, start in (("numpy", np_pieces, None, x0),
                                             ("jax-scan", jax_pieces, jax_backend.backend("scan"),
                                              jnp.asarray(x0)),
                                             ("jax-while", jax_pieces, jax_backend.backend("while"),
                                              jnp.asarray(x0))):
            counter: dict[str, int] = {}
            t = time.perf_counter()
            summary = solve(pieces, start, backend, counter)
            wall = time.perf_counter() - t
            if reference is None:
                reference = summary
            row = describe(name, problem, summary, reference, wall, counter)
            rows.append(row)
            print(f"{problem:26s} {name:10s} {row['iterations']:4d} {row['krylov']:6d} "
                  f"{row['f']:14.7e} {str(row['same_counts']):>5s} {row['f_rel_diff_first_20']:8.1e} "
                  f"{row['first_iteration']:8.3f}s {row['per_iteration'] * 1e3:7.2f}ms "
                  f"{row['fun_calls_or_traces']:5d}/{row['hessvec_calls_or_traces']:<5d}", flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
