"""Compiled iterations with JAX.

trust_region_newton and line_search_newton compute through a Backend record
(see nicks_optimizers.backend); the default runs plain numpy. This example
defines a JAX backend in a few lines (the library imports no framework) and
compiles every iteration with jax.jit: the Krylov loops, the small dense
problems, the line search and the user's operators run as XLA code, and
only the stopping tests, the records and the display stay in Python.

Problem: a p-Laplacian-type energy on the unit square with N x N interior
nodes and zero boundary values (finite differences),

    E(u) = h^2 [ sum over edges of phi(du / h) - sum of u ],
    phi(t) = (eps^2 + t^2)^(p/2) / p,   p = 4, eps = 1e-2,

with the gradient from jax.grad and Hessian-vector products from jax.jvp
of it. The H^1 metric M = h^2 (I + T (x) I + I (x) T), T the 1-D Dirichlet
Laplacian, is solved exactly in the sine eigenbasis. Four solvers:

    cg              truncated CG, P = M = H^1 (the default)
    lanczos         Lanczos, P = M = H^1
    lanczos-sep     Lanczos with the L^2 metric M = h^2 I and the H^1
                    preconditioner P (the Gram matrix and explicit residual)
    flexible        the flexible solver with the same M and P

Each runs with both optimizers, on the numpy backend (the JAX functions
jit-compiled one by one and called from the Python loop) and compiled, with
the same iterates up to rounding. For a compiled run, the functions given to
the optimizer are traced, not called, so they must follow JAX's rules: no Python control
flow on values (no if, float() or bool() of arrays), arrays out (fun
returns a 0-d array), point_data (if used) returns arrays in tuples or
dicts, and 64-bit floats on (the method's tolerances assume float64).

Compiling takes a few seconds per run (more when a solve outgrows its
basis, once per new capacity), so compiled iterations pay off for longer
runs and larger problems; the first compiled run in a process is also
slower per iteration for a while.

Run:   python examples/jax_p_laplacian.py [--N 63] [--loop scan|while] [--solvers cg lanczos]
                                       [--methods trust-region line-search]
Needs: pip install jax
"""

import argparse
from typing import Any, Callable, NamedTuple, Union, cast

import jax
import jax.numpy as jnp
import numpy as np

from nicks_optimizers import (
    Backend,
    LineSearchSummary,
    RunSummary,
    array_space,
    line_search_newton,
    loop_from_scan,
    loop_from_while,
    trust_region_newton,
)

cast(Any, jax.config).update("jax_enable_x64", True)    # (untyped in some JAX versions)

SOLVERS = ("cg", "lanczos", "lanczos-sep", "flexible")
METHODS = ("trust-region", "line-search")
Summary = Union[RunSummary, LineSearchSummary]


def jax_backend(loop: str = "scan") -> Backend:
    """A Backend that compiles with jax.jit. The bounded loops are built from
    jax.lax.scan (a fixed trip count, finished steps skipped; works with
    reverse-mode differentiation) or jax.lax.while_loop (exits early)."""
    if loop == "scan":
        bounded = loop_from_scan(jax.lax.scan, jax.lax.cond)
    elif loop == "while":
        bounded = loop_from_while(jax.lax.while_loop)
    else:
        raise ValueError(f"loop must be 'scan' or 'while', not {loop!r}")
    return Backend(jit=jax.jit, loop=bounded, cond=jax.lax.cond,
                   index_update=lambda a, i, v: a.at[i].set(v), xp=jnp, static_shapes=True)


class Problem(NamedTuple):
    """The energy and its derivatives, and the metrics, as JAX functions of
    flattened N x N arrays."""
    N: int
    fun: Callable[[jax.Array], jax.Array]
    grad: Callable[[jax.Array], jax.Array]
    hessvec: Callable[[jax.Array, jax.Array], jax.Array]
    h1_solve: Callable[[jax.Array, jax.Array], jax.Array]
    l2_solve: Callable[[jax.Array, jax.Array], jax.Array]
    l2_apply: Callable[[jax.Array, jax.Array], jax.Array]


def p_laplacian(N: int, p: float = 4.0, eps: float = 1e-2) -> Problem:
    h = 1.0 / (N + 1)

    def energy(u: jax.Array) -> jax.Array:
        padded = jnp.pad(u.reshape(N, N), 1)
        gx = (padded[1:, 1:-1] - padded[:-1, 1:-1]) / h
        gy = (padded[1:-1, 1:] - padded[1:-1, :-1]) / h
        phi_x = (eps ** 2 + gx ** 2) ** (p / 2) / p
        phi_y = (eps ** 2 + gy ** 2) ** (p / 2) / p
        return h ** 2 * (jnp.sum(phi_x) + jnp.sum(phi_y) - jnp.sum(u))

    grad = jax.grad(energy)

    def hessvec(u: jax.Array, v: jax.Array) -> jax.Array:
        tangent: jax.Array = jax.jvp(grad, (u,), (v,))[1]
        return tangent

    k = jnp.arange(1, N + 1)
    sines = jnp.sqrt(2.0 / (N + 1)) * jnp.sin(jnp.pi * jnp.outer(k, k) / (N + 1))  # its own inverse
    eigenvalues = (4.0 / h ** 2) * jnp.sin(k * jnp.pi / (2 * (N + 1))) ** 2
    h1 = h ** 2 * (1.0 + eigenvalues[:, None] + eigenvalues[None, :])

    def h1_solve(u: jax.Array, w: jax.Array) -> jax.Array:
        return (sines @ ((sines @ w.reshape(N, N) @ sines) / h1) @ sines).reshape(-1)

    return Problem(N=N, fun=energy, grad=grad, hessvec=hessvec, h1_solve=h1_solve,
                   l2_solve=lambda u, w: w / h ** 2, l2_apply=lambda u, v: h ** 2 * v)


def operators(problem: Problem, solver: str) -> dict[str, Any]:
    """The metric and solver options of each configuration."""
    if solver in ("cg", "lanczos"):
        return dict(metric_solve=problem.h1_solve, krylov=solver)
    if solver in ("lanczos-sep", "flexible"):
        return dict(metric_solve=problem.l2_solve, metric_apply=problem.l2_apply,
                    precond_solve=problem.h1_solve, krylov=solver.split("-")[0])
    raise ValueError(f"unknown solver {solver!r}")


def optimize(method: str, *args: Any, **options: Any) -> tuple[Any, Summary]:
    """trust_region_newton or line_search_newton, by name."""
    if method == "trust-region":
        return trust_region_newton(*args, **options)
    if method == "line-search":
        return line_search_newton(*args, **options)
    raise ValueError(f"method must be one of {METHODS}, not {method!r}")


def solve_compiled(problem: Problem, solver: str, loop: str = "scan", method: str = "trust-region",
                   **options: Any) -> tuple[jax.Array, Summary]:
    """Every iteration compiled: array_space holds the vectors and Krylov
    bases as JAX arrays of fixed shapes."""
    backend = jax_backend(loop)
    x, summary = optimize(method, problem.fun, problem.grad, problem.hessvec,
                          jnp.zeros(problem.N ** 2), space=array_space(backend),
                          retract=lambda x, s: x + s, backend=backend,
                          **operators(problem, solver), **options)
    return x, summary


def solve_numpy(problem: Problem, solver: str, method: str = "trust-region",
                **options: Any) -> tuple[np.ndarray, Summary]:
    """The numpy backend: each JAX function compiled on its own, numpy arrays
    in and out, the iterations in Python."""
    def compiled(f: Callable[..., jax.Array]) -> Callable[..., np.ndarray]:
        g = jax.jit(f)
        return lambda *args: np.asarray(g(*args))
    ops: dict[str, Any] = {key: compiled(f) if callable(f) else f
                           for key, f in operators(problem, solver).items()}
    fun = jax.jit(problem.fun)
    x, summary = optimize(method, lambda x: float(fun(x)), compiled(problem.grad),
                          compiled(problem.hessvec), np.zeros(problem.N ** 2), **ops, **options)
    return x, summary


def per_iteration(summary: Summary) -> float:
    """Median seconds per iteration after the first (which includes compiling)."""
    elapsed = [r.elapsed for r in summary.iterations]
    return float(np.median(np.diff(elapsed))) if len(elapsed) > 2 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--N", type=int, default=63, help="interior nodes per side")
    parser.add_argument("--loop", choices=("scan", "while"), default="scan")
    parser.add_argument("--solvers", nargs="+", choices=SOLVERS, default=list(SOLVERS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args()

    solve_compiled(p_laplacian(7), "lanczos", args.loop)    # warm-up, for fair timings
    problem = p_laplacian(args.N)
    print(f"p-Laplacian energy, n = {args.N ** 2}, compiled loops from {args.loop}")
    print(f"{'solver':12s} {'method':13s} {'backend':8s} {'its':>4s} {'Krylov':>6s} {'f evals':>7s} "
          f"{'f':>22s} {'f vs numpy':>10s} {'first it':>9s} {'per it':>9s}")
    for solver in args.solvers:
        for method in args.methods:
            _, reference = solve_numpy(problem, solver, method, max_krylov=100)
            _, run = solve_compiled(problem, solver, args.loop, method, max_krylov=100)
            for name, summary in (("numpy", reference), ("compiled", run)):
                diff = abs(summary.f - reference.f) / abs(reference.f)
                print(f"{solver:12s} {method:13s} {name:8s} {len(summary.iterations):4d} "
                      f"{summary.hessvec_calls:6d} {summary.fun_calls:7d} {summary.f:22.15e} "
                      f"{diff:10.1e} {summary.iterations[0].elapsed:8.2f}s "
                      f"{per_iteration(summary) * 1e3:7.2f}ms")


if __name__ == "__main__":
    main()
