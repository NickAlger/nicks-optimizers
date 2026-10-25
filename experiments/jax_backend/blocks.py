"""Phase 7 of dev/jit_backends_plan.md: what fixed capacity costs in
compiled code, piece by piece, and whether block products can be made to
follow the members in use.

Measured, each jit-compiled on its own (median of repeated calls):
  products  two Gram-Schmidt passes (pair_block then combine, twice) over a
            block of `cap` members with k in use, done four ways:
              full     over all members, coefficients masked (array_space)
              chunks   a while loop over chunks of 16 members, read with
                       take (the contract) or lax.dynamic_slice, stopping
                       after member k - 1 (the plan's chunked array_space)
              classes  the leading 16, 32, 64, ... members, the smallest
                       class holding k chosen with nested conds
  dense     the padded projected trust-region problem (solve_projected_trs)
            and eigh alone, by capacity, with k = 15 active coordinates
  cond      whether a cond copies a large operand: read inside a branch, or
            returned unchanged by one branch and modified by the other

Run:  python experiments/jax_backend/blocks.py [--json blocks.json]
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Callable

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import jax                                   # noqa: E402
import jax.numpy as jnp                      # noqa: E402
from jax import lax                          # noqa: E402

import jax_backend                           # noqa: E402  (turns on float64)
from nicks_optimizers.dense import solve_projected_trs  # noqa: E402
from nicks_optimizers.krylov import tridiagonal         # noqa: E402

Products = tuple[Callable[..., Any], Callable[..., Any]]


def timed(f: Callable[..., Any], *args: Any, reps: int = 30) -> float:
    """Median seconds per call, after a first (compiling) call."""
    jax.block_until_ready(f(*args))
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(f(*args))
        times.append(time.perf_counter() - t)
    return float(np.median(times))


def full() -> Products:
    def pair(B: Any, w: Any, k: Any) -> Any:
        return jnp.where(jnp.arange(B.shape[0]) < k, B @ w, 0.0)

    def combine(B: Any, c: Any, k: Any) -> Any:
        return jnp.where(jnp.arange(B.shape[0]) < k, c, 0.0) @ B
    return pair, combine


def chunks(width: int, read: str) -> Products:
    def rows(B: Any, j: Any) -> tuple[Any, Any]:
        start = jnp.minimum(j * width, B.shape[0] - width)     # the last chunk overlaps
        if read == "take":
            return start, jnp.take(B, start + jnp.arange(width), axis=0)
        return start, lax.dynamic_slice_in_dim(B, start, width, axis=0)

    def pair(B: Any, w: Any, k: Any) -> Any:
        def body(carry: Any) -> Any:
            j, out = carry
            start, R = rows(B, j)
            index = start + jnp.arange(width)
            return j + 1, out.at[index].set(jnp.where(index < k, R @ w, 0.0))
        return lax.while_loop(lambda c: c[0] * width < k, body, (0, jnp.zeros(B.shape[0])))[1]

    def combine(B: Any, c: Any, k: Any) -> Any:
        def body(carry: Any) -> Any:
            j, acc = carry
            start, R = rows(B, j)
            index = start + jnp.arange(width)
            coefficients = jnp.where((index < k) & (index >= j * width), jnp.take(c, index), 0.0)
            return j + 1, acc + coefficients @ R
        return lax.while_loop(lambda s: s[0] * width < k, body, (0, jnp.zeros(B.shape[1])))[1]
    return pair, combine


def classes(base: int = 16) -> Products:
    def sizes(cap: int) -> list[int]:
        out, s = [], base
        while s < cap:
            out.append(s)
            s *= 2
        return out + [cap]

    def choose(k: Any, branches: list[Callable[..., Any]], *operands: Any) -> Any:
        def at(i: int) -> Callable[..., Any]:
            if i == len(branches) - 1:
                return branches[i]
            return lambda *o: lax.cond(k <= SIZES[i], branches[i], at(i + 1), *o)
        return at(0)(*operands)

    def pair(B: Any, w: Any, k: Any) -> Any:
        cap = B.shape[0]
        SIZES[:] = sizes(cap)
        branches = [lambda B, w, s=s: jnp.zeros(cap).at[:s].set(
            jnp.where(jnp.arange(s) < k, B[:s] @ w, 0.0)) for s in SIZES]
        return choose(k, branches, B, w)

    def combine(B: Any, c: Any, k: Any) -> Any:
        SIZES[:] = sizes(B.shape[0])
        branches = [lambda B, c, s=s: jnp.where(jnp.arange(s) < k, c[:s], 0.0) @ B[:s]
                    for s in SIZES]
        return choose(k, branches, B, c)
    return pair, combine


SIZES: list[int] = []


def products(rows: list[dict[str, Any]]) -> None:
    variants = {"full": full(), "chunks (take)": chunks(16, "take"),
                "chunks (dynamic_slice)": chunks(16, "slice"), "classes": classes()}
    rng = np.random.default_rng(0)
    for n in (16129, 262144):
        for cap, k in ((32, 15), (100, 15), (100, 60), (128, 100)):
            B = np.zeros((cap, n))
            B[:k] = rng.standard_normal((k, n)) / np.sqrt(n)
            Bj, w = jnp.asarray(B), jnp.asarray(rng.standard_normal(n))
            line, reference = f"n={n:7d} cap={cap:4d} k={k:3d}", None
            for name, (pair, combine) in variants.items():
                def passes(B: Any, w: Any, k: Any, pair: Any = pair, combine: Any = combine) -> Any:
                    for _ in range(2):
                        w = w - combine(B, pair(B, w, k), k)
                    return w
                f = jax.jit(passes)
                out = f(Bj, w, jnp.asarray(k))
                reference = out if reference is None else reference
                assert float(jnp.max(jnp.abs(out - reference))) < 1e-10, name
                t = timed(f, Bj, w, jnp.asarray(k))
                rows.append(dict(kind="products", variant=name, n=n, capacity=cap, k=k, seconds=t))
                line += f" | {name} {t * 1e3:6.2f} ms"
            print(line, flush=True)


def dense(rows: list[dict[str, Any]]) -> None:
    bk = jax_backend.backend("while")
    rng = np.random.default_rng(1)
    k = 15
    for cap in (16, 32, 64, 100, 128):
        d, g = np.zeros(cap), np.zeros(cap)
        d[:k], g[:k - 1] = rng.standard_normal(k) + 3.0, 0.5
        for gram in ("G = I", "G = None"):
            def solve(d: Any, g: Any, cap: int = cap, gram: str = gram) -> Any:
                G = jnp.eye(cap) if gram == "G = I" else None
                return solve_projected_trs(tridiagonal(d, g, backend=bk), G, 1.0, 0.3,
                                           backend=bk, k=jnp.asarray(k))[0]
            t = timed(jax.jit(solve), jnp.asarray(d), jnp.asarray(g))
            rows.append(dict(kind="dense", variant=gram, capacity=cap, k=k, seconds=t))
        A = rng.standard_normal((cap, cap))
        t_eigh = timed(jax.jit(jnp.linalg.eigh), jnp.asarray(A + A.T))
        rows.append(dict(kind="eigh", capacity=cap, seconds=t_eigh))
        print(f"cap={cap:4d}: projected problem {rows[-3]['seconds'] * 1e3:6.3f} ms with G = I, "
              f"{rows[-2]['seconds'] * 1e3:6.3f} ms with G = None; eigh {t_eigh * 1e3:6.3f} ms",
              flush=True)


def cond(rows: list[dict[str, Any]]) -> None:
    for cap, n in ((100, 16129), (100, 262144)):
        B = jnp.asarray(np.random.default_rng(2).standard_normal((cap, n)))
        w, k = jnp.ones(n), jnp.asarray(5)
        plain = timed(jax.jit(lambda B, w: B[:16] @ w), B, w)
        read = timed(jax.jit(lambda B, w, k: lax.cond(k < 10, lambda B, w: B[:16] @ w,
                                                      lambda B, w: B[16:32] @ w, B, w)), B, w, k)
        through = timed(jax.jit(lambda B, w, k: lax.cond(k < 10, lambda B: B, lambda B: 2.0 * B,
                                                         B)[:16] @ w), B, w, k)
        copy = timed(jax.jit(lambda B: B + 0.0), B)
        rows.append(dict(kind="cond", capacity=cap, n=n, plain=plain, read_in_branch=read,
                         returned=through, block_copy=copy))
        print(f"cap={cap} n={n}: B[:16] @ w {plain * 1e3:.2f} ms; read in a branch {read * 1e3:.2f} ms; "
              f"B returned by a cond {through * 1e3:.2f} ms (one copy of B: {copy * 1e3:.2f} ms)",
              flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", help="write the results to this file")
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    products(rows)
    dense(rows)
    cond(rows)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
