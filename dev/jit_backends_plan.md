# Plan: compiled iterations with a pluggable backend

October 2026. **Status: proposal for review.** Nothing here is implemented
yet.

## Summary

Let a user compile each trust-region iteration with a framework of their
choice (JAX being the first target) by passing a `Backend` record, while
keeping numpy as the default and reference, and keeping the library free of
any dependency on, or drift toward, a particular framework.

What changes:

- The method is split into a **Python driver** and a **pure core**. The core
  computes one outer iteration (subproblem solve, trial point, acceptance,
  gradient at the new point) without Python control flow on data, so a
  backend can compile it. The driver keeps everything that talks to the
  outside world: stopping tests, records, display, the callback, resuming,
  timing, and raising errors.
- Loops on data become `backend.while_loop`, branches on data become
  `xp.where` (on scalars) or `backend.cond` (when a branch applies an
  operator).
- Krylov bases get a **fixed capacity** C = min(max_krylov, dimension), so
  shapes are static; the small projected problems are padded to C x C.
  Backends with dynamic shapes (numpy) keep computing at the exact size k.
- Per-point setup becomes data (`point_data(x)`), because closures cannot
  cross the boundary of a compiled function (decision D1).

What does not change: the arguments of `trust_region_newton` (apart from a
new `backend=` keyword and decision D1), the records, the display, the
callback, checkpointing and resuming, distributed numpy runs, and the
results of numpy runs (gated bit for bit during the restructuring phases).

## Decisions already made

1. A `Backend` record, not separate `jit=` / `scan=` keywords.
2. Fixed-capacity blocks, with block operations restricted to the active
   members behind the `VectorSpace` interface (option 3 of the discussion:
   chunked operations for static-shape backends, exact sizes for numpy).
3. Compile one iteration; the outer loop stays in Python. A fully fused run
   (the whole loop compiled) is out of scope for now.
4. No numba. No framework imports in `src/`, and guardrails against drifting
   toward JAX (section "Guarding against drift").
5. The cond question is settled: a `cond` inside a loop costs at most the
   sum of its branches per step, never 2^k (a chain of diamonds, not a tree;
   checked with JAX: the vmapped program has `select_n` and no `cond`, and
   its time grows linearly in the number of steps). Exponential growth only
   comes from static nesting, which the design avoids.

## Decisions for review

Recommendations first; each is discussed where it comes up.

| | Question | Recommendation | Alternative |
|---|---|---|---|
| D1 | Per-point setup under compilation | `point_data(x) -> data` (arrays, tuples, dicts) and `operators_at(x, data) -> PointOperators`, always two arguments | keep `operators_at(x)`; compiled backends recompute it at the start of every iteration (one setup and one M^{-1} g wasted per rejected step) |
| D2 | How block operations know the active size | an explicit hint k in `pair_block(B, w, k)` and `combine_*(B, c, k)` | spaces track counts inside blocks (blocks become (array, count) pairs, which reintroduces the tree ambiguity that `LeafBlock` papered over) |
| D3 | The Krylov process functions (`lanczos_start`, `cg_start`, ...), public today | keep them public, with the new arguments (backend, capacity) | make them internal, a smaller surface to keep to the contract |
| D4 | Where JAX code lives | `experiments/jax_backend/` first, graduating to `examples/`; never `src/`. The framework-agnostic `array_space(backend)` lives in `src/` | a separate optional package |
| D5 | Array namespace contract | names common to numpy >= 1.24, `jax.numpy` and the array API standard; keep `numpy>=1.24` | require numpy >= 2 and use its array API namespace |
| D6 | Python version | require 3.11 (3.10 reaches end of life this month), for generic `NamedTuple` states | stay on 3.10 with untyped (`Any`) state fields |
| D7 | Gate for the restructuring | bit-for-bit agreement with the pre-refactor code on numpy (phases 2, 4 to 6), via a temporary vendored copy of the old package | tolerance-based comparison from the start (less work, less assurance) |
| D8 | When to check in | after the vertical slice (phase 3: CG path end to end, plus the JAX proof of concept), before converting the Lanczos and flexible solvers | after everything |

## Architecture

### Layers

```
trust_region_newton (driver, plain Python)
  argument checks, stopping tests, records, display, callback, resume,
  timing, raising errors
      |
      |  init = backend.jit(core.init), iterate = backend.jit(core.iterate)
      v
core (pure, backend-generic): one outer iteration
  operators at x, subproblem solve (CG / Lanczos / flexible as while_loops),
  retraction, f(trial), rho, radius update, cond(accepted: grad, point data,
  gradient norms)
      |
      |  space.*, backend.while_loop / cond / index_update, xp.*
      v
Backend record (numpy default)          VectorSpace record
```

The driver only sees arrays coming out of `init` and `iterate`; it converts
them to Python floats for records and tests (one small host transfer per
iteration). User functions (fun, grad, hessvec, metric and preconditioner
operators, retract, project_covector, point_data) run inside the core, so
with a compiled backend they must be traceable by that backend.

### The Backend record

```python
@dataclass(frozen=True, kw_only=True)
class Backend:
    jit: Callable[[F], F]               # compile a pure function (identity for numpy)
    while_loop: Callable                # while_loop(cond_fun, body_fun, init) -> final
    cond: Callable                      # cond(pred, true_fun, false_fun, operand) -> result
    index_update: Callable              # index_update(a, index, value): a copy of a with a[index] = value
    xp: Any                             # array namespace, used only through the contract below
    static_shapes: bool                 # True: shapes fixed at trace time (pad to capacity)
```

The numpy default is a few lines in `src/`:

```python
def _while_loop(cond_fun, body_fun, state):
    while cond_fun(state):
        state = body_fun(state)
    return state

numpy_backend = Backend(jit=lambda f: f, while_loop=_while_loop,
                        cond=lambda pred, t, f, x: t(x) if pred else f(x),
                        index_update=_copy_and_set, xp=np, static_shapes=False)
```

and the JAX one is just as thin (in the example, not in `src/`):

```python
jax_backend = Backend(jit=jax.jit, while_loop=jax.lax.while_loop, cond=jax.lax.cond,
                      index_update=lambda a, i, v: a.at[i].set(v), xp=jnp, static_shapes=True)
```

No `scan`: with the outer loop in Python, the only loops inside compiled
code are the Krylov loop and the secular-equation iteration, both
`while_loop`s (a fixed-length `scan` with `cond` to skip finished steps
would save nothing under `vmap`). No host-callback primitive: display stays
in the driver.

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

### Rules for core code

1. **No Python control flow on data**: no `if`, `while`, `and`/`or`,
   `bool()`, `float()` or `int()` on values computed from the inputs.
   Control flow on configuration (krylov method, saddle_free, whether a
   preconditioner was given) is fine: it is fixed when the iteration is
   traced.
2. **Choose the scalar, then do one vector operation.** Most branches in a
   Krylov step differ only in a coefficient (boundary step tau or interior
   step alpha, a done flag): pick it with `xp.where` and do a single axpy.
3. **Operator applications that are conditional go under `backend.cond`**
   (H, M, M^{-1}, P^{-1}, fun, grad, retract, point_data). Under plain
   compilation only the taken branch runs; under `vmap` both do, which costs
   the sum of the branches (decision 5).
4. **Select with `where`, never multiply by a 0/1 mask**, and guard
   denominators (`a / where(b > 0, b, 1)`): the branch not taken still runs
   under `vmap`, on whatever data it gets, and 0 * inf = nan.
5. **Kernels never raise on bad data.** numpy raises (`LinAlgError`) where
   JAX returns nan, so factorizations get finite, well-posed inputs (padding
   by the identity, non-finite entries replaced before factorizing).
6. **Errors become status codes.** Today `lanczos_start` and `cg_start`
   raise when <g, P^{-1} g> <= 0. In the core this becomes a status the
   driver turns into the same `ValueError` after the compiled call returns.
7. **States are `NamedTuple`s** of arrays and vectors (JAX treats named
   tuples as containers without registration; plain Python does too).
   User-facing records stay frozen dataclasses, built by the driver.
8. **Peel the first step** of a loop when step 0 is special (Lanczos with
   no previous vector), instead of branching on the step number.
9. **No hard-coded dtype.** Coordinate arrays take the dtype of the first
   pairing, and `finfo` of that dtype replaces `np.finfo(float)`. (float32
   is not a goal; the tolerances assume float64, and the JAX example turns
   on `jax_enable_x64`.)

Small helpers keep the where-style code readable: `safe_div`,
`count_where(mask)`, `when(pred, fn, operand)` (a `cond` whose other branch
returns the operand).

### Blocks with fixed capacity

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
```

(replacing `append_*` and the one-argument `vector_block`). Members at and
beyond k are zero, so k is only a hint that lets an implementation skip
work (D2). Capacity C = min(max_krylov, dimension), a Python int known
before tracing. CG uses no blocks at all (five vectors), so the default
path for P = M costs no extra memory under compilation; Lanczos stores two
blocks (Q, V = P Q) and the flexible solver three (Q, M Q, H Q).

Implementations:

- **numpy (dynamic shapes).** k is a concrete integer, so operations use
  exactly the first k members. During the restructuring, storage stays as it
  is today (k columns, concatenated on `set_vector` at j = k, with a `NumpyBlock`
  wrapper that knows the capacity and pads coordinate results to it), so
  the arithmetic is unchanged. Phase 7 switches to preallocated storage
  with the members along the leading axis: `np.zeros` memory is committed
  lazily by the operating system, page by page as members are written, so
  memory stays proportional to k with no copies at all (to be verified).
- **`array_space(backend, chunk=None)`** (new, in `src/`, framework-agnostic):
  one array per block of shape (capacity,) + vector shape. Without `chunk`,
  products run over all C members (zeros contribute nothing). With
  `chunk=c`, a `backend.while_loop` over fixed-width chunks takes members
  `j*c + arange(c)` with `xp.take` and stops after the chunk containing
  k - 1, so the work is that of k members rounded up to c (option 3). It is
  written against the contract, so it also runs on numpy with
  `static_shapes=True` semantics, which is how the static path gets tested
  without JAX.
- **Tuple trees**: blocks are plain tuple trees of leaf blocks. `LeafBlock`
  goes away: leaf blocks are arrays or `NumpyBlock`s, never tuples, so they
  cannot be mistaken for tree nodes.
- **`tuple_block_space`**: blocks become a small `TupleBlock` dataclass. It
  cannot index a Python tuple with a traced j, so it is documented as
  numpy-backend only (it is meant for small custom representations anyway).

Under a static-shape backend memory is C vectors per block regardless of k:
`max_krylov` becomes the memory knob, documented as such.

### Small dense algebra, padded

The projected problems (T, G, the M-Ritz pairs, the trust-region problem on
a ball) are C x C arrays with an active leading k x k block:

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
  `while_loop` whose condition is false from the start when another case
  applies.
- **Dynamic-shape backends trim** to the leading k x k block at the entry of
  the kernel and pad the results back to C at the exit. The masks inside the
  kernel are then all true, so numpy computes exactly what it computes today,
  at the same cost.

Tests: padding invariance (the same problem at capacities k, k + 1, 2k
gives the same solution to rounding), the hard case, saddle-free, k = 1,
pad values against extreme spectra.

### Krylov solvers as while loops

Each solver becomes `init -> while_loop(running, step) -> finish`, with a
`NamedTuple` carry holding the step count, a status code, the vectors and
blocks, the padded coordinate arrays, and four padded trace arrays of length
C. Stop reasons become integer codes inside the core and strings in
`SubproblemStats`.

CG, as an example of the style:

```python
def solve_cg(bk, space, g, hessvec, metric_solve, radius, resid_tol, max_steps, ...):
    xp = bk.xp

    def running(c):
        return (c.status == RUNNING) & (c.k < max_steps)

    def step(c):
        Hp = hessvec(c.p)
        curv = space.pair(Hp, c.p)
        alpha = safe_div(c.rz, curv)
        leaves = (curv <= 0) | (c.s_norm2 + 2 * alpha * c.sp + alpha**2 * c.p_norm2 >= radius**2)
        # to_boundary: tau, s + tau p, status BOUNDARY or NEGATIVE (no operator applied)
        # advance: the CG update, including the M^{-1} solve of the new residual
        return bk.cond(leaves, to_boundary, advance, (c, Hp, curv, alpha))

    return finish(bk.while_loop(running, step, cg_start(bk, space, g, metric_solve, ...)))
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

### One outer iteration

```python
class OuterState(NamedTuple):
    x: Point;  f: Array;  g: Covector;  data: Any        # point_data(x)
    gnorm: Array;  metric_gnorm: Array;  start: Vector    # norms and the Krylov start vector at x
    radius: Array;  status: Array
```

- `init(x0, g0)`: f(x0), the gradient (unless g0 is given),
  `point_data(x0)`, the operators, the gradient norms and the start vector
  (M^{-1} g or P^{-1} g, as today).
- `iterate(state, gnorm0, forcing_max, ...)`: the operators from
  `operators_at(x, data)` (cheap: closures over data); the subproblem solve;
  `retract`; f at the trial point; rho; the radius update (in where form);
  then `cond(accepted, ...)` computes the gradient, `point_data` and the
  gradient norms at the new point, or returns the old ones. Returns the new
  state and a raw record (a `NamedTuple` of scalars and padded traces).

**Per-point data (D1).** Today `operators_at(x)` returns closures and is
called once per new iterate, so expensive setup is shared by all the solves
and rejected steps at one point. Closures cannot be returned from a compiled
function, so with D1 the setup becomes data: `point_data(x)` returns arrays
in tuples or dicts (for numpy, anything: a pyamg hierarchy works, since
nothing is compiled), carried in the state, and `operators_at(x, data)`
builds the operators from it inside the iteration. Without `point_data`,
`data` is `None`. The keyword operators (`hessvec(x, v)`,
`metric_solve(x, w)`, ...) need no change. The examples that use
`operators_at` today (p_laplacian, p_laplacian_eigen, the two experiment
studies) move their setup into `point_data`.

The stopping tests stay in the driver, on the floats of the returned state,
so `gradient_converged`, `check_resume_matches` and the max_iter and radius
tests are untouched. `forcing_term`, `reduction_ratio` and `update_radius`
get where-forms used in the core (the Python versions stay as the
reference, tested against them).

### The driver

```python
def trust_region_newton(fun, grad, hessvec, x0, *, backend=None, point_data=None, ...):
    bk = numpy_backend if backend is None else backend
    core = make_core(bk, fun, grad, ..., options)           # pure functions; options are static
    init, iterate = bk.jit(core.init), bk.jit(core.iterate)
    state = init(x0, g0)
    raise_on_status(state.status)
    while True:
        gnorm = float(state.gnorm)                          # one host transfer per iteration
        ... the stopping tests of today ...
        state, raw = iterate(state, gnorm0, ...)
        raise_on_status(state.status)
        record = make_record(raw, iteration=it, elapsed=time.perf_counter() - start)
        ... display, callback(state.x, info), ResumeState: as today ...
```

`make_record` converts the raw record to the existing `IterationRecord` and
`SubproblemStats` (Python floats, traces trimmed to k), so the records,
`format_*`, `history_array`, `merge_summaries` and resuming are unchanged.
Timing includes the device wait, since converting the record's scalars to
floats waits for the result.

Compilation happens once per `trust_region_newton` call (the jitted
functions close over the user's functions and the static options). A
rank continuation therefore compiles once per stage, which it would anyway
since the shapes change. A reusable compiled solver (a factory) is left for
later.

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
   - `while_loop` checks that the carry keeps its structure, shapes and
     dtypes from one step to the next, and `cond` that both branches return
     the same structure, which are JAX's rules, checked without JAX;
   - `cond` evaluates both branches and selects, as `vmap` does, so a branch
     that raises or leaks nan on the data of the other case fails;
   - `static_shapes=True`, so the padded path runs.

   Used with `array_space(strict_backend)`, it exercises the code path a
   compiled backend takes. (Tests that count operator calls by
   instrumenting the operators use the plain numpy backend, since this
   `cond` runs both branches.)
4. **Containers**: only tuples, named tuples, dicts and arrays cross the
   core's boundary or live in loop carries; no dataclasses that a framework
   would need to register (hence the removal of `LeafBlock`).
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
  space with a trivial allreduce) and compares the records by `repr`,
  bit for bit. Same machine, so BLAS differences cannot interfere.
- **Backend matrix**: the core tests run on `numpy_backend + numpy_space` and
  on `strict_backend + array_space(strict_backend)`; the two agree to
  rounding over the first 40 iterations (as the dense-vs-tree comparisons
  do today), with identical stop reasons and Krylov counts on the short
  runs.
- **Contract tests**: the import guard and the `xp` name check; tests of the
  strict backend itself (that it catches an `if` on data, a shape-changing
  carry, a branch that raises).
- **Kernel tests**: padding invariance of the dense kernels; chunked block
  operations against full ones for every k and chunk size.
- **Optional JAX job** in CI (`pip install jax[cpu]`, x64 on): the JAX example
  against numpy on the same problems; one trace per compiled function per
  run (counted with a Python counter inside the traced function); the
  jaxpr of an iteration contains no host callbacks.
- **Performance**: example run times on numpy before and after (phase 0
  baseline), with any slowdown over 5% explained or fixed.

## Phases

Each phase ends with the full test suite and the CI examples passing, and
is committed separately.

| | Phase | Delivers | Gate | Size |
|---|---|---|---|---|
| 0 | Baseline | vendored legacy package and the comparison harness; benchmark script and baseline timings of the examples | harness passes trivially | S |
| 1 | Backend and guardrails | `backend.py` (`Backend`, `numpy_backend`, the contract list); strict backend in `tests/`; import and contract tests; Python 3.11 floor (D6) | new tests | S |
| 2 | Vertical slice: core and driver, CG | `OuterState`, `init` / `iterate`, the driver; where-forms of the criteria and radius update; `point_data` and two-argument `operators_at` (D1), examples updated; CG as a `while_loop` with status codes; `backend=` keyword | bit for bit against legacy on numpy; CG on the strict backend (numpy vectors with strict pairing results, since CG needs no blocks) | M |
| 3 | JAX proof of concept (CG path) | `experiments/jax_backend/`: the backend record, Rosenbrock, a JAX p-Laplacian-style problem with `jax.jvp` Hessian products, timings, compile counts, a README with results | agreement with numpy; **check in with you (D8)** | M |
| 4 | Fixed-capacity blocks | new block entries in `VectorSpace`; `NumpyBlock`, `TupleBlock`; `array_space(backend)` (unchunked); `LeafBlock` removed; tests, examples and experiments updated | bit for bit against legacy | M |
| 5 | Padded dense kernels | `dense.py` in `xp` with padding and trimming | bit for bit against legacy; padding invariance | M |
| 6 | Lanczos and flexible as while loops | both solvers converted; static-shape backends get every solver; the JAX experiment extended to Lanczos and flexible | bit for bit against legacy; strict backend; JAX agreement | L |
| 7 | Storage and chunking | numpy blocks preallocated on the leading axis (lazy zero pages); chunked `array_space` | rounding-level agreement (the layout change alters BLAS calls); timings; memory check | M |
| 8 | Graduation | JAX example in `examples/` with tests, optional CI job; README section on backends and the traceability rules for user functions; legacy copy deleted; TODO updated | CI | S |

## API changes

Breaking (pre-1.0, all callers are in this repository):

- `VectorSpace` block entries: `vector_block(v, capacity)`,
  `covector_block(w, capacity)`, `set_vector` / `set_covector` instead of
  `append_*`, and the hint k in `pair_block` and `combine_*`. The
  constructors (`numpy_space`, `tuple_tree_space`, `tuple_block_space`,
  `numpy_space_with_reductions`) are updated; hand-built `VectorSpace(...)`
  records need the new entries.
- `LeafBlock` removed (the matrix-completion type aliases change).
- `operators_at(x, data)` with `point_data` (D1).
- Krylov process functions take a backend and a capacity; `LanczosState`
  and `CGState` become named tuples with array fields (D3).
- `pair` returns a 0-d array (a numpy scalar for numpy) rather than a Python
  float; existing code that returns floats keeps working on numpy.
- Python >= 3.11 (D6).

New: `backend=` keyword, `Backend`, `numpy_backend`, `array_space`,
`point_data`.

## Risks and open problems

- **Readability.** where-style code is harder to read than plain ifs. The
  helpers and the peeling rule help; the Python versions of the criteria
  stay as documentation and test oracles.
- **numpy overhead.** Python `while_loop` closures and named-tuple
  rebuilding cost microseconds per Krylov step: negligible against a
  Hessian product except on toy problems. Measured in phase 0 and after
  phase 6.
- **Leaks in the strict backend.** numpy subclasses can lose their class
  (numpy scalars from indexing, 0-d results of ufuncs); the subclass must
  override `__getitem__` and `__array_wrap__`. If it proves leaky, the JAX
  job's tracing is the real check, and the strict backend is best effort.
- **Padded vs exact dense algebra** differ at rounding level between backends;
  long runs drift apart (as dense vs tree runs already do), so cross-backend
  tests compare the first iterations.
- **Compile time** under JAX (the whole iteration, including user functions
  and nested while loops): seconds per `trust_region_newton` call. Measured
  in phase 3; a reusable compiled solver is the remedy if it matters.
- **Memory under static shapes**: C vectors per block for Lanczos (2 blocks)
  and flexible (3); none for CG.
- **`vmap` over problems** works in principle (everything is `where`, `cond`
  and `while_loop`), but the slowest problem sets the pace and both branches
  of every `cond` run. Not a goal now.
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
| `trust_region_newton` outer loop | Python `while` with breaks | the driver (Python), around `iterate` |
| `float(fun(x))`, `bool(rho > eta)` | host conversions | arrays in the core; the driver converts records |
| `if accepted:` gradient, operators, norms | Python `if` | `cond` (applies grad, point_data, M^{-1}) |
| `update_radius`, `reduction_ratio`, `forcing_term` | ifs and early returns | where-forms |
| `gradient_norms` branches | on configuration | unchanged (fixed at trace time) |
| `lanczos_initial_pair`, `cg_start` | raise `ValueError` | status code, raised by the driver |
| Lanczos `for` loop with early return | Python loop | `while_loop` with a status |
| traces | Python lists | padded arrays, `index_update` at k |
| `G = np.eye(k + 1)`, `extend_metric_gram` | growing arrays | padded C x C, `index_update` |
| `krylov_residual_norm`: `if lam > 0` | `if`, then `metric_apply` | `cond` |
| `boundary_stalled` | `Optional` previous value | nan start, `where` |
| `lanczos_step`: `if v_prev is not None`, `gamma == 0` | ifs | first step peeled; guarded division |
| `LanczosState.deltas`, `gammas` | growing tuples | padded arrays |
| `solve_projected_trs`, `pencil_ritz` | k x k numpy | padded C x C in `xp`; trimmed on numpy |
| `solve_sphere_trs_eig` | three branches, Newton loop with breaks | three candidates and `where`; `while_loop` |
| CG boundary or negative curvature | `if` / `break` | `cond(leaves, to_boundary, advance)` |
| CG `theta_min = min(...)` | Python `min` | `xp.minimum` |
| flexible candidate loop | `for` / `break`, `np.isfinite` | first candidate, `cond` to the fallback |
| flexible `append_*`, `grow_gram` | concatenation | `set_*` at k, `index_update` |
| `SubproblemStats(...)` | built in the solver | built by the driver from the raw record |
| `space.dimension` cap | Python int | unchanged (static) |
| `pair` returning `float(...)` | Python float | 0-d array |
