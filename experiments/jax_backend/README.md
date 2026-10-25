# JAX backend: proof of concept (CG path)

**Question.** Does the backend design of
[`dev/jit_backends_plan.md`](../../dev/jit_backends_plan.md) work with a real
compiling framework? Specifically: can `trust_region_newton` run with each
iteration compiled by JAX, through a `Backend` record written outside the
library, and with the same results as the numpy backend? And what does it
buy?

**Status.** CG path (phase 3): positive. It works unchanged, gives the same
iterates, compiles once per run, and is up to 2.3 times faster per
iteration on CPU. Lanczos and the flexible solver (phases 6 and 7):
correct, and since phase 7, with each solve starting small and doubling its
basis, compiled iterations take 0.5 to 1.1 times as long as numpy's on CPU;
compiling takes 5 to 9 s per run. See the second and third parts below.

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
    python experiments/jax_backend/solvers.py [--sizes 63 127] [--json solvers.json]
    python experiments/jax_backend/capacity.py [--sizes 63 127] [--json capacity.json]
    python experiments/jax_backend/blocks.py [--json blocks.json]
    python experiments/jax_backend/calls.py

All need `pip install jax` (CPU here, JAX 0.12, a 2-core machine). The JSON
files hold the raw numbers of the runs below (`capacity_phase6.json`: the
capacity study on the phase 6 code).

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
   steps themselves are cheap: scan time does not grow from 100 to 400
   steps. What remains is a roughly fixed extra cost once max_krylov is well
   above the steps used. **Scan stays the default** (decision of 2026-10-25):
   it has fewer limitations with automatic differentiation (reverse mode
   works through it), the extra cost is moderate, and GPU timings are needed
   before recommending either; scan may do better there.
6. **Requirements on user code, as expected**: functions must be traceable
   (no `float()` of values, no Python branches on data) and return arrays.
   x64 must be on.

## Lanczos and the flexible solver (`solvers.py`, phase 6)

The same p-Laplacian problem with three configurations: Lanczos with
P = M (the H^1 metric); Lanczos with M = h^2 I (L^2) and P = the H^1 metric,
which needs the Gram matrix and the explicit residual; and the flexible
solver with the same M and P. The JAX runs use `array_space` (blocks of
fixed capacity) and start with room for 8 or 32 basis vectors, doubling
when a solve fills the basis. "Per it" is the median time of an iteration;
"compile" is the rest of the wall time (compilations at the start and after
each growth). `solvers.json` holds the numbers.

| n | solver | run | its | Krylov | growths | f vs numpy | compile | per it | traces f / Hv |
|---|---|---|---|---|---|---|---|---|---|
| 3,969 | lanczos | numpy | 16 | 200 | - | - | 0.9 s | 12.4 ms | 17 / 200 calls |
| | | jax, cap 8 | 16 | 200 | 3 | 4e-16 | 14.2 s | (48.7 ms) | 4 / 9 |
| | | jax, cap 32 | 16 | 200 | 1 | 4e-16 | 7.6 s | 9.2 ms | 3 / 5 |
| | lanczos P != M | numpy | 16 | 166 | - | - | 0.4 s | 17.9 ms | 17 / 166 calls |
| | | jax, cap 8 | 16 | 166 | 3 | 1e-14 | 12.7 s | 10.2 ms | 4 / 7 |
| | | jax, cap 32 | 16 | 166 | 1 | 8e-15 | 6.1 s | 9.5 ms | 3 / 3 |
| | flexible | numpy | 26 | 220 | - | - | 0.5 s | 11.2 ms | 27 / 220 calls |
| | | jax, cap 8 | 26 | 220 | 3 | 5e-15 | 16.1 s | 8.7 ms | 5 / 9 |
| | | jax, cap 32 | 26 | 220 | 1 | 5e-15 | 6.4 s | 6.3 ms | 3 / 3 |
| 16,129 | lanczos | numpy | 17 | 372 | - | - | 2.8 s | 47.9 ms | 18 / 372 calls |
| | | jax, cap 8 | 17 | 372 | 4 | 2e-16 | 15.0 s | (187 ms) | 5 / 10 |
| | | jax, cap 32 | 17 | 372 | 2 | 2e-16 | 10.7 s | 55.0 ms | 4 / 6 |
| | lanczos P != M | numpy | 16 | 235 | - | - | 1.5 s | 28.4 ms | 17 / 235 calls |
| | | jax, cap 8 | 16 | 235 | 3 | 2e-14 | 14.7 s | 20.0 ms | 4 / 9 |
| | | jax, cap 32 | 16 | 235 | 1 | 2e-14 | 8.7 s | 28.9 ms | 3 / 5 |
| | flexible | numpy | 28 | 419 | - | - | 3.4 s | 21.5 ms | 29 / 419 calls |
| | | jax, cap 8 | 28 | 419 | 4 | 5e-14 | 18.4 s | 24.5 ms | 5 / 10 |
| | | jax, cap 32 | 28 | 419 | 2 | 5e-14 | 13.6 s | 40.8 ms | 4 / 6 |

Medians in parentheses are unreliable: with four growths in 17 iterations,
half of the iterations compile something (the growth, then the start of the
next iteration at the new capacity).

With room for max_krylov = 100 vectors from the start (no growth), the
steady state is clean, and it shows what fixed capacity costs:

| n | solver | numpy | jax, scan loop | jax, while loop |
|---|---|---|---|---|
| 3,969 | lanczos | 11.7 ms / it | 76.5 ms / it | 25.4 ms / it |
| | flexible | 11.3 ms / it | 23.6 ms / it | 20.7 ms / it |
| 16,129 | lanczos | 45.1 ms / it | 106 ms / it | 87.5 ms / it |
| | flexible | 28.2 ms / it | 62.5 ms / it | 65.3 ms / it |

Findings:

7. **Correct, with growth as designed.** Every run has the numpy run's
   iteration and Krylov counts; objective values agree to 2e-16 to 5e-14.
   Bases that start with room for 8 vectors grow 3 or 4 times per run and
   give the same results as larger ones. The functions are traced a few
   times per run (once per piece and capacity), however many steps follow.
8. **Fixed capacity costs work.** With room for 100 vectors and 10 to 20 in
   use, the compiled iterations are slower than numpy: every block product
   (two Gram-Schmidt passes per step) and the padded small problems run
   over all 100 slots, where numpy works with the k in use. Starting small
   and doubling (cap 8 or 32) brings the steady state back to numpy's level
   (0.4 to 1.4 times). Phase 7 measured where the cost is and went further
   (third part).
9. **Compiling dominates short runs**: 4 to 6 s for the pieces at one
   capacity, and about as much again for each new capacity. A reusable
   compiled solver (pieces cached across calls) would amortize it.
10. ~~Scan-built loops pay for skipped steps when the carry holds a
   basis.~~ **Withdrawn in phase 7**: the gap was an artifact of run order.
   The first JAX run in a process is slower per iteration (by about 35 ms at
   n = 3,969 on this machine) whatever its loop, and in the fixed-capacity
   measurement above the scan runs came first. With a warm-up run, scan and
   while are within noise of each other, and the optimized HLO shows no
   copy of the basis in the loops (finding 15).

## Capacity, block products and copies (`capacity.py`, `blocks.py`, phase 7)

The plan's remedy for the cost of fixed capacity was chunked block products
(work proportional to the k members in use at any capacity). `blocks.py`
first measured where the cost is, each operation compiled on its own:

| capacity (k = 15) | 16 | 32 | 64 | 100 | 128 |
|---|---|---|---|---|---|
| eigh | 0.06 ms | 0.37 ms | 0.97 ms | 1.39 ms | 2.05 ms |
| padded projected problem, G = I | 0.30 ms | 0.24 ms | 0.91 ms | 1.90 ms | 3.04 ms |
| the same, G = None (phase 7) | 0.14 ms | 0.18 ms | 0.57 ms | 0.97 ms | 1.32 ms |

Two Gram-Schmidt passes (pair_block, then combine, twice) over a block with
k of its capacity in use, four ways: over all members with masked
coefficients (`full`, what `array_space` does); a while loop over chunks of
16 members read with `take` or `lax.dynamic_slice`, stopping after member
k - 1 (`chunks`, the plan's design); the smallest of the leading 16, 32,
64, ... members that holds k, chosen with nested conds (`classes`):

| n | capacity | k | full | chunks (take) | chunks (dynamic_slice) | classes |
|---|---|---|---|---|---|---|
| 16,129 | 32 | 15 | 1.00 ms | 2.26 ms | 1.19 ms | 1.09 ms |
| | 100 | 15 | 1.92 ms | 1.55 ms | 1.49 ms | 1.10 ms |
| | 100 | 60 | 2.31 ms | 4.64 ms | 4.16 ms | 6.11 ms |
| | 128 | 100 | 2.63 ms | 9.28 ms | 10.8 ms | 1.59 ms |
| 262,144 | 32 | 15 | 19.0 ms | 39.0 ms | 40.5 ms | 26.4 ms |
| | 100 | 15 | 57.1 ms | 37.5 ms | 37.4 ms | 39.9 ms |
| | 100 | 60 | 59.6 ms | 154 ms | 141 ms | 152 ms |
| | 128 | 100 | 76.5 ms | 255 ms | 261 ms | 73.9 ms |

Then the library changed (phase 7 of the plan): every Lanczos or flexible
solve starts with room for `krylov_capacity` = 32 vectors and doubles when
full, instead of keeping the run's largest capacity; G = None stands for an
identity Gram matrix (no Cholesky factor, no triangular solves); the loop of
a solve is bounded by its capacity. `capacity.py` runs the solvers with a
warm-up run first; "per it" is the median iteration (growths included),
"compile" the rest of the wall time, "growths" the doublings over all
solves (numpy grows the same way since phase 7):

| n | solver | run | its | Krylov | growths | f vs numpy | compile | per it | phase 6: per it |
|---|---|---|---|---|---|---|---|---|---|
| 3,969 | lanczos | numpy | 16 | 200 | 2 | - | 0.7 s | 12.1 ms | 13.9 ms |
| | | scan 32 | 16 | 200 | 2 | 2e-16 | 5.8 s | 9.8 ms | 18.2 ms |
| | | while 32 | 16 | 200 | 2 | 2e-16 | 5.6 s | 8.7 ms | 7.6 ms |
| | | scan 100 | 16 | 200 | 0 | 2e-16 | 3.4 s | 15.5 ms | 31.9 ms |
| | lanczos P != M | numpy | 16 | 166 | 1 | - | 0.2 s | 15.9 ms | 23.5 ms |
| | | scan 32 | 16 | 166 | 1 | 1e-14 | 6.2 s | 8.0 ms | 9.8 ms |
| | | while 32 | 16 | 166 | 1 | 1e-14 | 6.6 s | 9.5 ms | 12.0 ms |
| | | scan 100 | 16 | 166 | 0 | 2e-14 | 3.7 s | 32.6 ms | 28.6 ms |
| | flexible | numpy | 26 | 220 | 1 | - | 0.4 s | 9.0 ms | 9.5 ms |
| | | scan 32 | 26 | 220 | 1 | 3e-15 | 5.4 s | 8.4 ms | 11.4 ms |
| | | while 32 | 26 | 220 | 1 | 3e-15 | 5.9 s | 6.2 ms | 7.1 ms |
| | | scan 100 | 26 | 220 | 0 | 5e-15 | 3.6 s | 13.9 ms | 22.1 ms |
| 16,129 | lanczos | numpy | 17 | 372 | 6 | - | 1.5 s | 43.5 ms | 44.1 ms |
| | | scan 32 | 17 | 372 | 6 | 2e-16 | 7.1 s | 36.2 ms | 70.5 ms |
| | | while 32 | 17 | 372 | 6 | 2e-16 | 7.1 s | 34.1 ms | 41.5 ms |
| | | scan 100 | 17 | 372 | 0 | 2e-16 | 3.6 s | 62.9 ms | 83.3 ms |
| | lanczos P != M | numpy | 16 | 235 | 4 | - | 0.9 s | 26.6 ms | 32.9 ms |
| | | scan 32 | 16 | 235 | 4 | 4e-14 | 6.1 s | 23.9 ms | 31.5 ms |
| | | while 32 | 16 | 235 | 4 | 4e-14 | 6.3 s | 21.2 ms | 27.1 ms |
| | | scan 100 | 16 | 235 | 0 | 9e-15 | 3.8 s | 69.7 ms | 72.2 ms |
| | flexible | numpy | 28 | 419 | 7 | - | 2.0 s | 21.1 ms | 29.0 ms |
| | | scan 32 | 28 | 418 | 7 | 9e-14 | 8.7 s | 21.5 ms | 29.2 ms |
| | | while 32 | 28 | 418 | 7 | 9e-14 | 8.1 s | 23.1 ms | 26.3 ms |
| | | scan 100 | 28 | 419 | 0 | 5e-14 | 4.9 s | 50.9 ms | 51.3 ms |

Findings:

11. **The small dense problem, not the block products, dominated at fixed
    capacity for moderate n.** Its eigendecomposition costs 1.4 ms at
    capacity 100 against 0.06 ms at 16, at every Krylov step and whatever
    n. The block products grow with n and dominate for large n.
12. **Chunked block products do not pay on CPU.** XLA copies a sliced or
    gathered chunk before the matrix product (with `take` and with
    `dynamic_slice` alike, and also when the products are written as
    multiply-and-sum), so a chunk costs about three passes over its rows
    against one pass for the masked full product. Chunks win only when k
    is below about a third of the capacity; size classes behave the same,
    and their copies show at large n. So `array_space` stays unchunked;
    `blocks.py` keeps the variants for a GPU check, where XLA may fuse a
    slice into the matrix product.
13. **Capacity per solve instead.** Starting every solve at 32 and doubling
    keeps the block products and the small problem within a factor of two
    of the k in use, without chunks, at no compile cost (one version per
    capacity, kept for the run) and with copies only at growths. Compiled
    iterations now take 0.5 to 1.1 times numpy's (0.8 to 1.1 at n =
    16,129), against 0.4 to 1.6 times in phase 6 (numpy itself got faster
    in phase 7, see the plan's progress notes). Counts are the numpy
    run's except one Krylov step in 419 for the longest flexible run
    (rounding near a stopping test). A fixed room of 100 still costs 1.3 to
    2.6 times numpy's time, which is what `krylov_capacity = max_krylov`
    would buy.
    **Nothing is recomputed** when a basis grows: `calls.py` counts the
    executions of the user's operators inside the compiled code with host
    callbacks. For Lanczos (P = M and P != M) and the flexible solver, with
    loops from scan and from while, the counts equal the summary's and are
    the same whether the basis starts with room for 2 vectors (up to 28
    growths per run), 8 or 100; skipped scan steps run nothing. On numpy,
    `test_growing_the_basis_repeats_no_user_operation` checks the same for
    every solver with instrumented functions.
14. **G = None** (an M-orthonormal basis: P = M, and the flexible solver)
    halves the padded small problem at capacity 100.
15. **Copies.** The optimized HLO of the pieces has no copy of the basis
    inside the loops, for scan and while alike: block updates happen in
    place. Between pieces, the piece that continues a solve after a growth
    copies its input blocks once (its arguments are not donated), next to
    the growth's own copy: two copies of the basis per growth, small beside
    a solve's O(k^2 n) block work, so no donation hint was added. What does
    copy: anything that needs a block's old value and its updated value at
    once. A `cond` returning an argument of the compiled function unchanged
    from one branch copies it (`blocks.py`; inside the loops the operand is
    the carry, and XLA keeps it in place), and so does reading a block's
    old value next to its update. The latter slipped into phase 7 (the Gram
    column of Lanczos with a separate P paired the old basis) and made its
    iterations 2.5 times slower under scan until it was fixed.
16. **Scan and while** are within noise of each other here (finding 10 is
    withdrawn). The loop of a solve is now bounded by its capacity, so a
    scan skips fewer steps.
17. **Dispatch** costs about 0.5 ms per compiled piece call plus about 1 ms
    of driver work per iteration (tiny problem, n = 49): negligible beside
    the 10 to 60 ms iterations above, so the pieces need no further fusing.

## Open questions

- GPU: not measured. Per-iteration host transfers of the record are small,
  but conditionals and loop tests may need device-to-host syncs; chunked
  block products may behave differently there (`blocks.py`).
- Compilation once per call: a rank continuation, for example, compiles each
  stage. Caching compiled pieces across calls (a solver factory) is left for
  later.
