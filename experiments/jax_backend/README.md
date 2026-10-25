# JAX backend: proof of concept (CG path)

**Question.** Does the backend design of
[`dev/jit_backends_plan.md`](../../dev/jit_backends_plan.md) work with a real
compiling framework? Specifically: can `trust_region_newton` run with each
iteration compiled by JAX, through a `Backend` record written outside the
library, and with the same results as the numpy backend? And what does it
buy?

**Status.** Phase 3 of the plan: done for the CG path (P = M, no
saddle-free model), the only path converted so far. Positive: it works
unchanged, gives the same iterates, compiles once per run, and is up to
2.3 times faster per iteration on CPU.

## The backend

[`jax_backend.py`](jax_backend.py) is all the JAX-specific code:

```python
Backend(jit=jax.jit, loop=loop_from_scan(jax.lax.scan, jax.lax.cond),   # or loop_from_while(jax.lax.while_loop)
        cond=jax.lax.cond, index_update=lambda a, i, v: a.at[i].set(v),
        xp=jax.numpy, static_shapes=True)
```

plus `jax_enable_x64` (the method's tolerances assume float64). Nothing in
the library changed for JAX. Even `numpy_space` works on JAX arrays for the
CG path, since it only uses arithmetic operators and `@`.

## Scripts

    python experiments/jax_backend/study.py [--sizes 63 127 255] [--json results.json]
    python experiments/jax_backend/skipped_steps.py [--N 63 127]

Both need `pip install jax` (CPU here, JAX 0.12). `results.json` holds the
raw numbers of the study run below.

## Problems and runs (`study.py`)

- **rosenbrock**: the tests' chained Rosenbrock function, n = 20, M = I.
- **plaplace N**: a finite-difference energy on the unit square with N x N
  interior nodes and zero boundary values,
  E(u) = h^2 [sum over edges of phi(du/h) - sum of u], with
  phi(t) = (eps^2 + t^2)^(p/2) / p, p = 4, eps = 1e-2. The gradient comes from
  `jax.grad` and Hessian products from `jax.jvp` of it. The metric is the
  discrete H^1 inner product M = h^2 (I + T (x) I + I (x) T), with T the 1-D
  Dirichlet Laplacian, solved exactly in the sine eigenbasis by dense N x N
  products. P = M, so every subproblem is solved by truncated CG. The run
  starts from u = 0.

Each problem is solved three ways:

- **numpy**: the numpy backend. For the JAX problems, every function is
  `jax.jit`-compiled on its own and called from the Python loop, with arrays
  converted at each call. This is what a JAX user gets without compiled
  iterations.
- **jax-scan**: the JAX backend, with the bounded loop built from `scan`.
- **jax-while**: the JAX backend, with the loop built from `while_loop`.

## Results

CPU (this container), October 2026. "Per it" is the time per iteration
after compilation, from the records' elapsed times. "First it" includes
compiling `init` and `iterate` (for the numpy runs of the JAX problems, it
includes the separate jit compilations). "Traces" counts the Python calls
of `fun` and `hessvec`: for the compiled runs these are traces, made once.

| problem | run | its | Krylov | f vs numpy (first 20 its) | first it | per it | traces f / Hv |
|---|---|---|---|---|---|---|---|
| rosenbrock n = 20 | numpy | 54 | 409 | - | 0.01 s | 1.21 ms | 55 / 409 calls |
| | jax-scan | 54 | 409 | 2.6e-15 | 0.76 s | 0.69 ms | 2 / 1 |
| | jax-while | 54 | 409 | 2.6e-15 | 0.75 s | 0.70 ms | 2 / 1 |
| plaplace n = 3,969 | numpy | 16 | 141 | - | 0.65 s | 6.54 ms | 17 / 141 calls |
| | jax-scan | 16 | 141 | 1.8e-16 | 2.55 s | 3.20 ms | 2 / 1 |
| | jax-while | 16 | 141 | 1.8e-16 | 2.23 s | 2.87 ms | 2 / 1 |
| plaplace n = 16,129 | numpy | 17 | 207 | - | 0.62 s | 28.0 ms | 18 / 207 calls |
| | jax-scan | 17 | 207 | 3.7e-16 | 1.83 s | 17.3 ms | 2 / 1 |
| | jax-while | 17 | 207 | 3.7e-16 | 1.90 s | 12.2 ms | 2 / 1 |
| plaplace n = 65,025 | numpy | 16 | 331 | - | 0.50 s | 166 ms | 17 / 331 calls |
| | jax-scan | 16 | 331 | 3.7e-16 | 1.87 s | 148 ms | 2 / 1 |
| | jax-while | 16 | 331 | 3.7e-16 | 2.27 s | 137 ms | 2 / 1 |

Finished steps of a scan-built loop (`skipped_steps.py`): time per iteration
against `max_krylov`, for the plaplace problem. The solves use about 9 to 12
Krylov steps per iteration, so the scan skips the rest.

| n | max_krylov | scan ms / it | while ms / it |
|---|---|---|---|
| 3,969 | 25 | 3.10 | 2.68 |
| | 100 | 4.13 | 3.12 |
| | 400 | 3.84 | 2.72 |
| 16,129 | 25 | 13.2 | 13.0 |
| | 100 | 16.5 | 12.4 |
| | 400 | 16.2 | 12.0 |

## Findings

1. **The design works with JAX as planned.** A backend record of a few lines,
   written outside the library, compiles every iteration. Both loop helpers
   work. No JAX-specific change to the library was needed.
2. **Same algorithm.** Every run has the same number of iterations and Krylov
   steps as the numpy run. The objective values agree to rounding (relative
   differences 2e-16 to 3e-15, from XLA's different operation order).
3. **One compilation per run.** `fun` is traced twice (once in `init`, once in
   `iterate`) and `hessvec` once, however many iterations and Krylov steps
   follow. On the numpy backend they are called once per iteration and once
   per Krylov step.
4. **Faster per iteration** on CPU: 1.6 to 2.3 times at n = 4,000 to 16,000.
   At n = 65,000 the gain is 1.1 to 1.2 times, because the dense O(N^3) metric
   solve dominates there and is the same XLA computation in both runs.
   Compiling costs 0.75 to 2.5 s, once per `trust_region_newton` call, so
   compiled iterations pay off for runs longer than that.
5. **While-built loops were faster than scan-built ones**: by 2 to 40% per
   iteration on the plaplace problems, and equal on Rosenbrock. The skipped
   steps themselves are cheap:
   scan time does not grow from 100 to 400 steps. What remains is a roughly
   fixed extra cost once max_krylov is well above the steps used. Use
   `loop_from_while` for speed and `loop_from_scan` for a known trip count or
   reverse-mode differentiation.
6. **Requirements on user code, as expected**: functions must be traceable
   (no `float()` of values, no Python branches on data) and return arrays.
   x64 must be on.

## Open questions

- GPU: not measured. Per-iteration host transfers of the record are small,
  but conditionals and loop tests may need device-to-host syncs.
- Lanczos and the flexible solver (phases 4 to 6 of the plan): fixed-capacity
  bases with doubling and continuation.
- Compilation once per call: a rank continuation, for example, compiles each
  stage. Caching compiled pieces across calls (a solver factory) is left for
  later.
- Why the scan-built loop has a fixed extra cost (copies of the carry around
  the `cond`?): worth a look in the optimized HLO if scan is ever the
  preferred loop.
