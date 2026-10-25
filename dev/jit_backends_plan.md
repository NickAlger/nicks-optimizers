# Plan: compiled iterations with a pluggable backend

October 2026. **Status: plan agreed (2026-10-25), not implemented yet.**

## Summary

Let a user compile the trust-region iterations with a framework of their
choice (JAX being the first target) by passing a `Backend` record, while
keeping numpy as the default and reference, and keeping the library free of
any dependency on, or drift toward, a particular framework.

What changes:

- The method is split into a **Python driver** and a **pure core**. The core
  computes one outer iteration, as a few pure pieces (start the Krylov
  solve, advance it, finish the iteration), without Python control flow on
  data, so a backend can compile each piece. The driver glues the pieces
  and keeps everything that talks to the outside world: stopping tests,
  records, display, the callback, resuming, timing, and raising errors.
- Loops on data become a **bounded loop** supplied by the backend (built
  from `scan` or from `while_loop`, the user's choice); branches on data
  become `xp.where` (on scalars) or `backend.cond` (when a branch applies an
  operator).
- Krylov bases get a **capacity**: a fixed shape within each compiled piece,
  so shapes are static. On static-shape backends the capacity starts small
  and **doubles when the basis fills up, continuing the solve** in the
  larger basis (only array copies, nothing recomputed). numpy grows its
  storage on demand, as today, and computes at the exact size k.
- Per-point setup becomes data (`point_data(x)`), because functions cannot
  cross the boundary of a compiled piece.
- States stay **frozen dataclasses**; they are flattened to plain tuples
  wherever the backend sees them.

What does not change: the arguments of `trust_region_newton` (apart from
the new `backend=`, `point_data=` and `krylov_capacity=` keywords and the
two-argument `operators_at`), the records, the display, the callback,
checkpointing and resuming, distributed numpy runs, and the results of
numpy runs (bit for bit within reason, see D7).

## Decisions

| | Decision | Outcome |
|---|---|---|
| A1 | How the user supplies the framework | a `Backend` record |
| A2 | What is compiled | the pieces of one iteration; the outer loop stays in Python. A fully fused run is out of scope for now |
| A3 | Frameworks | no numba; no framework imports in `src/`; guardrails against drifting toward JAX |
| A4 | Loops | a bounded loop `loop(done, body, carry, max_steps)` in the record, built from `scan` or `while_loop` by helpers in `src/` |
| A5 | Krylov basis size | fixed shape per compiled piece; on static-shape backends it starts at `krylov_capacity` and doubles when full, continuing the solve; block operations work in chunks up to the active size k |
| D1 | Per-point setup | `point_data(x) -> data` and `operators_at(x, data) -> PointOperators`, always two arguments |
| D2 | Active size in block operations | explicit hint k in `pair_block(B, w, k)` and `combine_*(B, c, k)` |
| D3 | Krylov process functions (`lanczos_start`, `cg_start`, ...) | stay public, with the new arguments |
| D4 | Where JAX code lives | `experiments/jax_backend/`, graduating to `examples/`; never `src/`. The framework-agnostic `array_space(backend)` lives in `src/` |
| D5 | Array namespace contract | names common to numpy >= 1.24, `jax.numpy` and the array API standard; `numpy>=1.24` stays |
| D6 | State containers | frozen dataclasses, flattened to tuples at the backend boundary (no named tuples, so no change of the Python floor) |
| D7 | Gate for the restructuring | bit for bit against the pre-refactor code, within reason: a natural, numerically sound change that alters rounding is accepted and noted, and that comparison falls back to tolerances with identical counts and stop reasons. The code is never contorted to keep bits identical |
| D8 | Check-in | after the vertical slice (phase 3: CG end to end, plus the JAX proof of concept) |

## Measurements behind the decisions

With JAX on CPU (October 2026, scripts not kept; easy to redo):

- **`cond` inside a loop does not compound.** A `cond` returns one value of a
  fixed shape, so the two paths rejoin after every step: a chain of
  diamonds, not a tree. Under `vmap` with a batched predicate the `cond`
  becomes a select (the program has `select_n` and no `cond`), both branches
  run, and the time of k steps grows linearly in k (0.16, 0.30, 0.64, 1.4,
  2.6 s for k = 2, 4, 8, 16, 32). Under plain `jit` an untaken expensive
  branch costs nothing.
- **XLA reserves memory for branches that never run.** A `cond` whose
  untaken branch holds a basis of capacity C reserved exactly C n 8 bytes
  (C = 32, 128, 512; n = 20000). So a basis cannot usefully grow inside one
  compiled piece: the largest stage would be reserved anyway.
- **Capacity costs time on every solve, not only memory.** Allocating,
  zero-filling and copying the basis: with n = 100,000, a 10-step loop took
  about 8 ms at capacity 32 and 173 ms at capacity 512, while a step cost
  about 1 ms.
- **Skipped `scan` steps are cheap on CPU**: about 0.2 to 10 µs each (`scan`
  over the full capacity with a `cond` skipping finished steps, against a
  `while_loop` that exits). GPU not measured.

## Architecture

### Layers

```
trust_region_newton (driver, plain Python)
  argument checks, stopping tests, capacity growth, records, display,
  callback, resume, timing, raising errors
      |
      |  compiled pieces: init, begin_c, advance_c, grow, finish_c
      |  (c = capacity; each piece compiled once per capacity, cached)
      v
core (pure, backend-generic)
  operators from (x, data); Krylov solves (CG, Lanczos, flexible) as
  bounded loops; retraction, f(trial), rho, radius update, cond(accepted:
  gradient, point data, gradient norms)
      |
      |  space.*, backend.loop / cond / index_update, xp.*
      v
Backend record (numpy default)          VectorSpace record
```

The driver only sees arrays coming out of the compiled pieces; it converts
them to Python floats for records and tests (one small host transfer per
iteration). User functions (fun, grad, hessvec, the metric and
preconditioner operators, retract, project_covector, point_data) run inside
the core, so with a compiled backend they must be traceable by it.

### The Backend record

```python
@dataclass(frozen=True, kw_only=True)
class Backend:
    jit: Callable[[F], F]       # compile a pure function (identity for numpy)
    loop: Callable              # loop(done, body, carry, max_steps) -> carry
    cond: Callable              # cond(pred, true_fun, false_fun, operand) -> result
    index_update: Callable      # index_update(a, index, value): a copy of a with a[index] = value
    xp: Any                     # array namespace, used only through the contract below
    static_shapes: bool         # True: shapes fixed at trace time (pad to capacity)
```

**The bounded loop.** `loop(done, body, carry, max_steps)` applies `body` at
most `max_steps` times (a Python int) and stops once `done(carry)` is true.
The contract: when `done(carry)` is true, `body(carry)` returns the carry
unchanged, so an implementation may stop early or keep going and the
result is the same. Two helpers in `src/` build it from either primitive:

```python
def loop_from_scan(scan, cond):
    def loop(done, body, carry, max_steps):
        step = lambda c, _: (cond(done(c), lambda c: c, body, c), None)
        return scan(step, carry, None, length=max_steps)[0]
    return loop

def loop_from_while(while_loop):
    def loop(done, body, carry, max_steps):
        go = lambda ic: (ic[0] < max_steps) & ~done(ic[1])
        return while_loop(go, lambda ic: (ic[0] + 1, body(ic[1])), (0, carry))[1]
    return loop
```

Built from `scan`: a known trip count, reverse-mode differentiable in JAX;
finished steps still run as cheap skips. Built from `while_loop`: early exit,
and under `vmap` the loop runs only until the slowest problem finishes
(with `scan`, the skip becomes a select under `vmap`, so every step does full
work). The core is written once against the contract.

The numpy default, in `src/`:

```python
def _python_loop(done, body, carry, max_steps):
    for _ in range(max_steps):
        if done(carry):
            break
        carry = body(carry)
    return carry

numpy_backend = Backend(jit=lambda f: f, loop=_python_loop,
                        cond=lambda pred, t, f, x: t(x) if pred else f(x),
                        index_update=_copy_and_set, xp=np, static_shapes=False)
```

The JAX one is just as thin (in the example, not in `src/`):

```python
jax_backend = Backend(jit=jax.jit, loop=loop_from_scan(jax.lax.scan, jax.lax.cond),
                      cond=jax.lax.cond, index_update=lambda a, i, v: a.at[i].set(v),
                      xp=jnp, static_shapes=True)
```

No host-callback primitive: display stays in the driver. A possible later
addition is an optional buffer-donation hint on `jit` (reuse input buffers
for outputs; numpy would ignore it), only if the measurements in phase 7
show the Krylov state being copied between pieces.

**The array namespace contract.** The core may use only these names from
`xp`, a subset common to numpy >= 1.24, `jax.numpy` and the array API
standard:

- creation: `asarray`, `zeros`, `ones`, `full`, `eye`, `arange`
- elementwise: `sqrt`, `abs`, `maximum`, `minimum`, `where`, `isfinite`
- reductions and sorting: `sum`, `max`, `min`, `argsort`, `take`
- linear algebra: `matmul` (and `@`), `linalg.eigh`, `linalg.cholesky`,
  `linalg.solve`
- dtypes: `finfo`, `.astype`, `.dtype`
- indexing: reads with integers (possibly traced) and slices; writes only
  through `backend.index_update`.

Adding a name is a design change, made in the contract list (a module-level
constant that the tests read), not in passing.

### States: frozen dataclasses, flattened at the boundary

States (the outer state, the Krylov carries, the raw records) are frozen
dataclasses, like everything else in the library. A framework can only
carry containers it understands, and JAX would need each dataclass
registered with it, a framework-specific requirement on our types. So the
backend never sees our dataclasses: small generic helpers in `src/` flatten
a state into a plain tuple (recursively through nested dataclasses of
ours; vectors and arrays are left as they are) and rebuild it from a
structure recorded on the Python side. The core uses them at every place the
backend touches data: the arguments and results of compiled pieces (the
structure of the results is recorded when the piece is traced), loop
carries, and `cond` operands and results. About 30 lines; the user-facing
records (`IterationRecord`, `SubproblemStats`, `RunSummary`) are built by the
driver and never cross the boundary.

Static configuration (the krylov method, saddle_free, capacities,
tolerances that select code paths) is closed over, never carried, so it
stays static under compilation.

### Rules for core code

1. **No Python control flow on data**: no `if`, `while`, `and`/`or`,
   `bool()`, `float()` or `int()` on values computed from the inputs.
   Control flow on configuration (krylov method, saddle_free, whether a
   preconditioner was given) is fine: it is fixed when a piece is traced.
2. **Choose the scalar, then do one vector operation.** Most branches in a
   Krylov step differ only in a coefficient (boundary step tau or interior
   step alpha, a done flag): pick it with `xp.where` and do a single axpy.
3. **Operator applications that are conditional go under `backend.cond`**
   (H, M, M^{-1}, P^{-1}, fun, grad, retract, point_data). Under plain
   compilation only the taken branch runs; under `vmap` both do, which costs
   the sum of the branches.
4. **Select with `where`, never multiply by a 0/1 mask**, and guard
   denominators (`a / where(b > 0, b, 1)`): the branch not taken still runs
   under `vmap`, on whatever data it gets, and 0 * inf = nan.
5. **Kernels never raise on bad data.** numpy raises (`LinAlgError`) where
   JAX returns nan, so factorizations get finite, well-posed inputs (padding
   by the identity, non-finite entries replaced before factorizing).
6. **Errors become status codes.** Today `lanczos_start` and `cg_start`
   raise when <g, P^{-1} g> <= 0. In the core this becomes a status the
   driver turns into the same `ValueError` after the piece returns.
7. **Loop bodies are the identity on finished carries** (the bounded-loop
   contract): every update is gated by the done flag.
8. **Peel the first step** of a loop when step 0 is special (Lanczos with
   no previous vector), instead of branching on the step number.
9. **No hard-coded dtype.** Coordinate arrays take the dtype of the first
   pairing, and `finfo` of that dtype replaces `np.finfo(float)`. (float32
   is not a goal; the tolerances assume float64, and the JAX example turns
   on `jax_enable_x64`.)

Small helpers keep the where-style code readable: `safe_div`,
`count_where(mask)`, `when(pred, fn, operand)` (a `cond` whose other branch
returns the operand).

### Blocks and capacity

Today blocks grow by appending, which needs dynamic shapes. The new
`VectorSpace` block entries:

```
vector_block(v, capacity) -> VectorBlock     block of `capacity` members: v, then zeros
covector_block(w, capacity) -> CovectorBlock
set_vector(B, j, v) -> VectorBlock           a copy with member j replaced (j may be traced)
set_covector(C, j, w) -> CovectorBlock
pair_block(B, w, k) -> Coords                (<w, b_i>)_i, length capacity, zeros for i >= k
combine_vectors(B, c, k) -> Vector           sum_{i < k} c_i b_i
combine_covectors(C, c, k) -> Covector
grow_vector_block(B, capacity) -> VectorBlock    the same members in a larger block
grow_covector_block(C, capacity) -> CovectorBlock
```

(replacing `append_*` and the one-argument `vector_block`). Members at and
beyond k are zero, so k is only a hint that lets an implementation skip work
(D2). CG uses no blocks at all (five vectors); Lanczos stores two blocks
(Q, V = P Q) and the flexible solver three (Q, M Q, H Q).

**Capacity.**

- numpy (`static_shapes=False`): the capacity is just the bound
  min(max_krylov, dimension). Storage grows on demand and operations use
  exactly the first k members, so the basis never fills early and memory
  stays proportional to k, as today.
- Static-shape backends: each compiled piece has one capacity c. Runs start
  at `krylov_capacity` (new option, default min(32, max_krylov), rounded to
  the chunk size). When a solve fills its basis before stopping, the driver
  grows the Krylov state to 2c (capped at min(max_krylov, dimension)) and
  continues the solve (next section). The capacity never shrinks during a
  run (a high-water mark, like an STL vector), so growth happens at most
  log2(max_krylov / krylov_capacity) times per run. Memory and the fixed
  per-solve cost follow the largest k seen, not max_krylov.

Doubling for storage, chunks for work: each distinct capacity costs a
compilation, so capacities double (four compilations from 32 to 512, where
growing by 16 would need up to 31), while operations run in chunks up to
the active size k.

Implementations:

- **numpy.** During the restructuring, storage stays as it is today (k
  columns, concatenated on `set_vector` at j = k, in a `NumpyBlock` wrapper
  that knows the capacity and pads coordinate results to it), so the
  arithmetic is unchanged. Phase 7 switches to preallocated storage with the
  members along the leading axis: `np.zeros` memory is committed lazily by
  the operating system, page by page as members are written, so memory
  stays proportional to k with no copies at all (to be verified).
- **`array_space(backend, chunk=None)`** (new, in `src/`, framework-agnostic):
  one array per block, of shape (capacity,) + vector shape. Without `chunk`,
  products run over all members (zeros contribute nothing). With `chunk=c`,
  a `backend.loop` over fixed-width chunks takes members `j*c + arange(c)`
  with `xp.take` and stops after the chunk containing member k - 1, so the
  work is that of k members rounded up to c. It is written against the
  contract, so it also runs on numpy with static-shape semantics, which is
  how the static path gets tested without JAX.
- **Tuple trees**: blocks are plain tuple trees of leaf blocks. `LeafBlock`
  goes away: leaf blocks are arrays or `NumpyBlock`s, never tuples, so they
  cannot be mistaken for tree nodes.
- **`tuple_block_space`**: blocks become a small `TupleBlock` dataclass. It
  cannot index a Python tuple with a traced j, so it is documented as
  numpy-backend only (it is meant for small custom representations anyway).

### Small dense algebra, padded

The projected problems (T, G, the M-Ritz pairs, the trust-region problem on
a ball) are c x c arrays with an active leading k x k block:

- Inactive part: G = I, T = pad * I, with pad = 1 + 2 ||A_active||_inf (a
  Gershgorin bound), so the padded eigenvalues sit above every active M-Ritz
  value (and above every |theta| for the saddle-free model). After sorting,
  the active eigenpairs are the first k, and b = gamma0 L^{-1} e_1 vanishes
  on the inactive coordinates, so they contribute nothing to the secular
  equation.
- The hard-case test uses a scale computed over the active eigenvalues only
  (the padded ones would inflate it).
- `solve_sphere_trs_eig` computes its three candidates (interior, hard case,
  boundary) and selects with `where`; the boundary Newton iteration becomes a
  bounded loop, done from the start when another case applies. Its bound
  (200 today) matters with a scan-built loop, where it is paid in skipped
  steps at every Krylov step: Newton on this secular equation from the safe
  side converges monotonically, so phase 5 sets the bound from measurements
  (with the bisection safeguard kept).
- **Dynamic-shape backends trim** to the leading k x k block at the entry of
  the kernel and pad the results back at the exit. The masks inside the
  kernel are then all true, so numpy computes what it computes today, at the
  same cost.

Tests: padding invariance (the same problem at capacities k, k + 1, 2k
gives the same solution to rounding), the hard case, saddle-free, k = 1,
pad values against extreme spectra.

### Krylov solvers as bounded loops

Each solver becomes `start -> loop(done, step) -> result`, with a frozen
dataclass carry holding the step count, a status code, the vectors and
blocks, the padded coordinate arrays, and four padded trace arrays. Stop
reasons become integer codes inside the core and strings in
`SubproblemStats`. Besides the stop reasons of today, the loop stops with
status FULL when the subproblem was solved at step k, the stopping test
failed, k < min(max_krylov, dimension), and the next step has no slot for
its new basis vector. Nothing is computed and thrown away.

CG, as an example of the style (CG has no basis, so it never fills):

```python
def solve_cg(bk, space, g, hessvec, metric_solve, radius, resid_tol, max_steps, ...):
    def done(c):
        return c.status != RUNNING

    def step(c):
        Hp = hessvec(c.p)
        curv = space.pair(Hp, c.p)
        alpha = safe_div(c.rz, curv)
        leaves = (curv <= 0) | (c.s_norm2 + 2 * alpha * c.sp + alpha**2 * c.p_norm2 >= radius**2)
        # to_boundary: tau, s + tau p, status BOUNDARY or NEGATIVE (no operator applied)
        # advance: the CG update, including the M^{-1} solve of the new residual
        return when_running(c, bk.cond(leaves, to_boundary, advance, (c, Hp, curv, alpha)))

    return finish(bk.loop(done, step, cg_start(bk, space, g, metric_solve, ...), max_steps))
```

Per solver:

- **CG**: as above. `theta_min` becomes an `xp.minimum` carried along; the
  trace is written with `index_update` at k.
- **Lanczos**: the first step is peeled (no previous vector); the pending
  vector joins the basis with `set_vector` at k; the tridiagonal is two
  padded arrays; `prev_model_value` starts as nan for the stall test; the
  explicit residual's `metric_apply` (only when lam > 0) goes under `cond`;
  breakdown sets the status and ends the loop.
- **Flexible**: each step first proposes P^{-1} R and orthogonalizes it;
  `cond` on "not finite or nothing new" computes the fallback M^{-1} R (with
  its `metric_apply`); breakdown skips the Hessian product with `cond`;
  otherwise the three blocks are set at k and T grows with `index_update`.

Operator call counts are computed arithmetically in the carry, so they mean
"calls made by the algorithm" on every backend (under `vmap` both branches
of a `cond` run, but the counts do not double).

### One outer iteration, in pieces

```python
@dataclass(frozen=True)
class OuterState:
    x: Point;  f: Array;  g: Covector;  data: Any        # point_data(x)
    gnorm: Array;  metric_gnorm: Array;  start: Vector    # norms and the Krylov start vector at x
    radius: Array;  status: Array
```

The pieces, each pure and compiled by the backend:

```
init(x0, g0)              -> OuterState    f(x0), gradient (unless g0), point_data, norms, start vector
begin_c(state)            -> KrylovState   operators from (x, data); the Krylov start at capacity c
advance_c(kstate)         -> KrylovState   the bounded loop: stops when done or FULL
grow(kstate, c2)          -> KrylovState   the same state at capacity c2: array copies only
finish_c(state, kstate)   -> (OuterState, RawRecord)
                                           the step from the basis, retract, f(trial), rho, radius
                                           update (where form), cond(accepted: gradient,
                                           point_data, norms at the new point)
```

- **Continuation.** `grow` copies the blocks (k vectors each) and the small
  coordinate arrays into larger ones. Nothing is recomputed: no Hessian
  products, no preconditioner or metric applications, no inner products.
  The continued solve gives exactly what an unlimited basis would (numpy:
  bit for bit; static backends: to rounding, since the projected problem
  is solved afresh from T at every step). The cost of one growth is the copy,
  O(k n), small next to the O(k^2 n) reorthogonalization already done, plus
  one compilation per new capacity, cached for the rest of the run.
- **CG** has no basis, so for the CG path the pieces fuse into one compiled
  `iterate(state)`.
- **Three calls per iteration** in the common case (begin, advance, finish)
  for Lanczos and flexible. The Krylov state passes between them on the
  device, with no host transfer. Measured in phase 6; if the dispatch
  overhead matters, begin and advance fuse for the common path.
- **Per-point data (D1).** Today `operators_at(x)` returns closures and is
  called once per new iterate, so expensive setup is shared by all the
  solves and rejected steps at one point. Closures cannot cross the boundary
  of a compiled piece, so the setup becomes data: `point_data(x)` returns
  arrays in tuples or dicts (for numpy, anything: a pyamg hierarchy works,
  since nothing is compiled), carried in the outer state, and
  `operators_at(x, data)` rebuilds the operators cheaply in whichever piece
  needs them. Without `point_data`, `data` is `None`. The keyword operators
  (`hessvec(x, v)`, `metric_solve(x, w)`, ...) need no change. The examples
  that use `operators_at` today (p_laplacian, p_laplacian_eigen, the two
  experiment studies) move their setup into `point_data`.

The stopping tests stay in the driver, on the floats of the returned state,
so `gradient_converged`, `check_resume_matches` and the max_iter and radius
tests are untouched. `forcing_term`, `reduction_ratio` and `update_radius`
get where-forms used in the core (the Python versions stay as the
reference, tested against them).

### The driver

```python
def trust_region_newton(fun, grad, hessvec, x0, *, backend=None, point_data=None,
                        krylov_capacity=None, ...):
    bk = numpy_backend if backend is None else backend
    core = make_core(bk, fun, grad, ..., options)           # pure pieces; options are static
    pieces = CompiledPieces(bk, core)                       # compiles lazily, one set per capacity
    c = initial_capacity(bk, krylov_capacity, max_krylov, dimension)
    state = pieces.init(x0, g0)
    raise_on_status(state.status)
    while True:
        gnorm = float(state.gnorm)                          # one host transfer per iteration
        ... the stopping tests of today ...
        k = pieces.advance(c, pieces.begin(c, state))
        while is_full(k):                                   # static-shape backends only
            c = min(2 * c, capacity_bound)
            k = pieces.advance(c, pieces.grow(k, c))
        state, raw = pieces.finish(c, state, k)
        raise_on_status(state.status)
        record = make_record(raw, iteration=it, elapsed=time.perf_counter() - start)
        ... display, callback(state.x, info), ResumeState: as today ...
```

`make_record` converts the raw record to the existing `IterationRecord` and
`SubproblemStats` (Python floats, traces trimmed to k), so the records,
`format_*`, `history_array`, `merge_summaries` and resuming are unchanged.
A counter of capacity growths goes into the summary. The capacity is a
performance state, not part of `ResumeState`: a resumed run starts again at
`krylov_capacity` and may grow once more, with the same iterates. Timing
includes the device wait, since converting the record's scalars to floats
waits for the result.

Compilation happens per `trust_region_newton` call (the compiled pieces
close over the user's functions and the static options) and per capacity
used. A rank continuation therefore compiles once per stage, which it would
anyway since the shapes change. A reusable compiled solver (a factory) is
left for later.

Transitional rule: until phase 6, static-shape backends support only the CG
path, and the driver says so. The numpy backend runs unconverted solvers
unchanged, since its `jit` is the identity.

## Guarding against drift

The pressure toward JAX arrives as small conveniences (pytrees, `.at[]`,
`jnp`-only functions, `vmap`, `static_argnums`, JAX random keys,
workarounds for JAX's quirks). Each guardrail below is enforced by a test,
so drifting takes a deliberate change rather than a slip:

1. **Import guard**: a test parses every module in `src/` and fails on
   imports other than the standard library and numpy. The core modules
   (`dense`, `krylov`, the subproblem solvers, the outer-iteration core,
   `criteria`) do not import numpy at all; they use `bk.xp`. The test suite
   runs without JAX installed.
2. **Contract check**: a test parses the core modules and fails on any
   `xp.<name>` outside the contract list.
3. **Strict numpy backend** (in `tests/`): numpy underneath, but
   - its scalars and arrays are an ndarray subclass whose `__bool__`,
     `__float__`, `__int__` and `__index__` raise, so a Python `if` on data in
     the core fails an ordinary test;
   - its `xp` exposes only the contract names (anything else is an
     `AttributeError`);
   - `loop` checks that the carry keeps its structure, shapes and dtypes from
     one step to the next, and the bounded-loop contract (it applies the body
     once more to a finished carry and checks nothing changed); `cond`
     checks that both branches return the same structure. These are JAX's
     rules plus ours, checked without JAX;
   - `cond` evaluates both branches and selects, as `vmap` does, so a branch
     that raises or leaks nan on the data of the other case fails;
   - `static_shapes=True`, so the padded path runs, with capacity 1 or 2 so
     that growth and continuation happen constantly.

   Used with `array_space(strict_backend)`, it exercises the code path a
   compiled backend takes. (Tests that count operator calls by
   instrumenting the operators use the plain numpy backend, since this
   `cond` runs both branches.)
4. **Containers**: the backend only ever sees tuples, arrays and the user's
   own vector types; our dataclasses are flattened at the boundary, and no
   type of ours needs registering with a framework (hence also the removal of
   `LeafBlock`).
5. **Framework-specific code lives with the framework**: the JAX backend
   record and anything tuned for XLA (e.g. `lax.dynamic_slice` instead of
   `take` for chunks) stay in the JAX example. The core is written for the
   contract, not for the fastest JAX program.
6. **numpy stays the reference**: every test runs on numpy; the JAX tests
   are an optional CI job that compares against numpy, never the other way
   around.

## Testing

- **Pre-refactor comparison (D7).** Phase 0 vendors a copy of the current
  package as `tests/legacy/` (renamed, used by tests only, deleted when the
  refactor is done). A test runs the old and new packages side by side in
  the same process on a matrix of configurations (Rosenbrock with the
  identity, smoothing and x-dependent metrics; the double well; the
  dense-reference quadratics at several radii; CG, Lanczos, flexible;
  saddle-free; P = M and separate P; tuple trees; the distributed numpy
  space with a trivial allreduce) and compares the records by `repr`. Same
  machine, so BLAS differences cannot interfere. Where a sound
  restructuring changes rounding, that comparison switches to tolerances
  plus identical counts and stop reasons, with a note in the commit.
- **Backend matrix**: the core tests run on `numpy_backend + numpy_space` and
  on `strict_backend + array_space(strict_backend)`; the two agree to
  rounding over the first 40 iterations (as the dense-vs-tree comparisons do
  today), with identical stop reasons and Krylov counts on the short runs.
- **Loop helpers**: `loop_from_scan` and `loop_from_while` built from plain
  Python implementations of `scan` and `while_loop` give the same results as
  the default loop.
- **Capacity**: runs with capacities 1, 2, 3 and the full bound give the
  same iterates and Krylov counts (to rounding on static backends); `grow`
  preserves the state exactly.
- **Contract tests**: the import guard and the `xp` name check; tests of the
  strict backend itself (that it catches an `if` on data, a shape-changing
  carry, a body that changes a finished carry, a branch that raises); the
  dataclass flattening round trip.
- **Kernel tests**: padding invariance of the dense kernels; chunked block
  operations against full ones for every k and chunk size.
- **Optional JAX job** in CI (`pip install jax[cpu]`, x64 on): the JAX example
  against numpy on the same problems, with both loop helpers; one trace per
  compiled piece per capacity (counted with a Python counter inside the
  traced function); no host callbacks in the compiled pieces.
- **Performance**: example run times on numpy before and after (phase 0
  baseline), with any slowdown over 5% explained or fixed.

## Phases

Each phase ends with the full test suite and the CI examples passing, and
is committed separately.

| | Phase | Delivers | Gate | Size |
|---|---|---|---|---|
| 0 | Baseline | vendored legacy package and the comparison harness; benchmark script and baseline timings of the examples | harness passes trivially | S |
| 1 | Backend and guardrails | `backend.py` (`Backend`, `numpy_backend`, `loop_from_scan`, `loop_from_while`, the contract list); dataclass flattening helpers; strict backend in `tests/`; import and contract tests | new tests | S |
| 2 | Vertical slice: core and driver, CG | `OuterState`, `init`, the fused CG `iterate`, the driver; where-forms of the criteria and radius update; `point_data` and two-argument `operators_at`, examples updated; CG as a bounded loop with status codes; `backend=` keyword | legacy comparison (D7); CG on the strict backend (numpy vectors with strict pairing results, since CG needs no blocks) | M |
| 3 | JAX proof of concept (CG path) | `experiments/jax_backend/`: the backend record with both loop helpers, Rosenbrock, a JAX p-Laplacian-style problem with `jax.jvp` Hessian products, timings, compile counts, a README with results | agreement with numpy; **check in (D8)** | M |
| 4 | Blocks with capacity | new block entries in `VectorSpace` (including `grow_*`); `NumpyBlock`, `TupleBlock`; `array_space(backend)` (unchunked); `LeafBlock` removed; tests, examples and experiments updated | legacy comparison | M |
| 5 | Padded dense kernels | `dense.py` in `xp` with padding and trimming; the secular-equation bound | legacy comparison; padding invariance | M |
| 6 | Lanczos and flexible in pieces | both solvers as bounded loops with the FULL status; `begin`, `advance`, `grow`, `finish` and the driver's growth loop; `krylov_capacity`; static-shape backends get every solver; the JAX experiment extended | legacy comparison; strict backend with tiny capacities; JAX agreement | L |
| 7 | Storage, chunking, buffers | numpy blocks preallocated on the leading axis (lazy zero pages); chunked `array_space`; measure dispatch overhead and Krylov-state copies between pieces (fuse begin and advance, or add a donation hint, only if needed) | rounding-level agreement (the layout change alters BLAS calls); timings; memory check | M |
| 8 | Graduation | JAX example in `examples/` with tests, optional CI job; README section on backends and the traceability rules for user functions; legacy copy deleted; TODO updated | CI | S |

## API changes

Breaking (pre-1.0, all callers are in this repository):

- `VectorSpace` block entries: `vector_block(v, capacity)`,
  `covector_block(w, capacity)`, `set_vector` / `set_covector` instead of
  `append_*`, `grow_*`, and the hint k in `pair_block` and `combine_*`. The
  constructors (`numpy_space`, `tuple_tree_space`, `tuple_block_space`,
  `numpy_space_with_reductions`) are updated; hand-built `VectorSpace(...)`
  records need the new entries.
- `LeafBlock` removed (the matrix-completion type aliases change).
- `operators_at(x, data)` with `point_data` (D1).
- Krylov process functions take a backend and a capacity; `LanczosState`
  and `CGState` stay frozen dataclasses, with array fields in place of
  growing tuples (D3).
- `pair` returns a 0-d array (a numpy scalar for numpy) rather than a Python
  float; existing code that returns floats keeps working on numpy.

New: `backend=`, `point_data=` and `krylov_capacity=` keywords; `Backend`,
`numpy_backend`, `loop_from_scan`, `loop_from_while`, `array_space`.

## Risks and open problems

- **Readability.** where-style code is harder to read than plain ifs. The
  helpers and the peeling rule help; the Python versions of the criteria
  stay as documentation and test oracles.
- **numpy overhead.** Python loop closures, dataclass rebuilding and
  flattening cost microseconds per Krylov step: negligible against a Hessian
  product except on toy problems. Measured in phase 0 and after phase 6.
- **Leaks in the strict backend.** numpy subclasses can lose their class
  (numpy scalars from indexing, 0-d results of ufuncs); the subclass must
  override `__getitem__` and `__array_wrap__`. If it proves leaky, the JAX
  job's tracing is the real check, and the strict backend is best effort.
- **Padded vs exact dense algebra** differ at rounding level between
  backends; long runs drift apart (as dense vs tree runs already do), so
  cross-backend tests compare the first iterations.
- **Compile time** under JAX: seconds per piece and capacity, so a growth in
  the middle of a run shows up as a pause. Measured in phases 3 and 6; a
  reusable compiled solver is the remedy if it matters.
- **Skipped steps with a scan-built loop**, paid at every Krylov step by the
  inner secular-equation loop; cheap on CPU (measured), unknown on GPU.
  Users who care can build the loop from `while_loop`.
- **Copies between pieces.** Without buffer reuse, passing the Krylov state
  from one compiled piece to the next may copy it, briefly doubling the
  basis memory. Measured in phase 7.
- **`vmap` over problems** works in principle (everything is `where`, `cond`
  and bounded loops), but the slowest problem sets the pace, both branches
  of every `cond` run, and a scan-built loop does full work in every step.
  Not a goal now.
- **Distributed runs**: unchanged for numpy (`numpy_space_with_reductions`,
  every process runs the same driver and loops). JAX with `mpi4jax` would
  need an allreduce usable inside compiled code; not planned.

## Out of scope for now

A fused run (the outer loop compiled too), batched solves with `vmap`, a
reusable compiled solver across calls, float32, `mpi4jax`, other frameworks
beyond checking that the contract does not exclude them.

## Appendix: data-dependent control flow in the current code

| Where | Today | Becomes |
|---|---|---|
| `trust_region_newton` outer loop | Python `while` with breaks | the driver (Python), around the compiled pieces |
| `float(fun(x))`, `bool(rho > eta)` | host conversions | arrays in the core; the driver converts records |
| `if accepted:` gradient, operators, norms | Python `if` | `cond` (applies grad, point_data, M^{-1}) |
| `update_radius`, `reduction_ratio`, `forcing_term` | ifs and early returns | where-forms |
| `gradient_norms` branches | on configuration | unchanged (fixed at trace time) |
| `lanczos_initial_pair`, `cg_start` | raise `ValueError` | status code, raised by the driver |
| Lanczos `for` loop with early return | Python loop | bounded loop with a status (including FULL) |
| traces | Python lists | padded arrays, `index_update` at k |
| `G = np.eye(k + 1)`, `extend_metric_gram` | growing arrays | padded c x c, `index_update` |
| `krylov_residual_norm`: `if lam > 0` | `if`, then `metric_apply` | `cond` |
| `boundary_stalled` | `Optional` previous value | nan start, `where` |
| `lanczos_step`: `if v_prev is not None`, `gamma == 0` | ifs | first step peeled; guarded division |
| `LanczosState.deltas`, `gammas` | growing tuples | padded arrays |
| `solve_projected_trs`, `pencil_ritz` | k x k numpy | padded c x c in `xp`; trimmed on numpy |
| `solve_sphere_trs_eig` | three branches, Newton loop with breaks | three candidates and `where`; bounded loop |
| CG boundary or negative curvature | `if` / `break` | `cond(leaves, to_boundary, advance)` |
| CG `theta_min = min(...)` | Python `min` | `xp.minimum` |
| flexible candidate loop | `for` / `break`, `np.isfinite` | first candidate, `cond` to the fallback |
| flexible `append_*`, `grow_gram` | concatenation | `set_*` at k, `index_update` |
| `SubproblemStats(...)` | built in the solver | built by the driver from the raw record |
| `space.dimension` cap | Python int | unchanged (static) |
| `pair` returning `float(...)` | Python float | 0-d array |
