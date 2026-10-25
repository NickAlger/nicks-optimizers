# nicks-optimizers

Matrix-free optimization methods in Python. The first is `trust_region_newton`:
an inexact Newton trust-region method built around a **metric** `M`, the
geometry of the problem, with an optional **preconditioner** `P` as a separate
accelerant. It takes three structures separately: the vector space (how
vectors and covectors are added, scaled and paired), the Hilbert structure
(the metric) and the point structure (the retraction).

- **The metric comes first.** `M(x)` (symmetric positive definite, possibly
  depending on the iterate) shapes the trust region
  `{s : ||s||_M <= radius}`, measures the gradient in the stopping test
  `||g||_{M^{-1}} <= rtol ||g(x0)||_{M^{-1}}`, and defines the curvature used
  by the saddle-free model. You give it by two separate operators:
  `metric_solve` (`M^{-1}`), plus `metric_apply` (`M`) only when a separate
  preconditioner is used. For plain numpy arrays the default is `M = I`.
- **Preconditioning through the metric.** By default `P = M`: the
  conventional (Steihaug-Toint) trust region, where only `M^{-1}` is ever
  applied. Any symmetric positive definite operator passed as `metric_solve`,
  a multigrid cycle for instance, defines `M` as its inverse. The
  subproblems are solved by truncated conjugate gradients: five vectors, two
  reductions per step.
- **A separate preconditioner when the geometry should differ.**
  `precond_solve` (`P^{-1}`) shapes the Krylov space while `M` keeps the
  geometry, e.g. a mesh-independent L^2 trust region with a multigrid
  preconditioner for the Hessian.
- **Lanczos subproblem solver.** An explicitly reorthogonalized,
  P-preconditioned Lanczos basis; the small trust-region problem is solved
  exactly in M-orthonormal coordinates (including the hard case). It handles
  a separate `P`, the saddle-free model, and is available for `P = M` on
  request. No Hessian-vector products are spent beyond one per Krylov step.
- **Saddle-free option.** `saddle_free=True` replaces the Hessian by its absolute
  value *with respect to M* (generalized eigenvalues of `H u = theta M u`), so
  saddle points repel iterates instead of attracting them.
- **Representation-generic.** Vectors and covectors can be numpy arrays of
  any shape (the default), tuple trees of arrays, distributed objects, or
  anything else, through a `VectorSpace` record of functions.
- **Distributed.** Every reduction goes through the space, so vectors can
  live on many MPI ranks and the method runs in lockstep.
  [`examples/p_laplacian.py`](examples/p_laplacian.py) minimizes a
  p-Laplacian energy with mpi4py and a multigrid preconditioner.
- **Manifolds.** Points move only through a retraction, and tangent vectors may
  be stored redundantly (e.g. gauged factor variations), with an optional
  covector projection that keeps the Krylov process inside the tangent space.
  [`examples/matrix_completion.py`](examples/matrix_completion.py) completes a
  matrix on the manifold of rank-r matrices.
- **Diagnostics, callback and exact resumption.** Per-iteration records, a
  readable iteration log, `callback(x, info)`, and a `ResumeState` that lets an
  interrupted run continue exactly as if it had not stopped.
- **Typed.** Primal/dual roles are spelled out as types (`Point`, `Vector`,
  `Covector`), erased at run time, and the package passes `mypy --strict`.

## Repository layout

- `src/`: the library, tested and type-checked (`mypy --strict`).
- `examples/`: demonstrations, tested and type-checked, run in CI.
- `experiments/`: exploratory studies of open questions, not part of the
  tested surface. An experiment that pans out graduates to `src/` and
  `examples/`. See [`experiments/README.md`](experiments/README.md).
- `dev/TODO.md`: findings and open questions.

## Installation

```bash
pip install git+https://github.com/NickAlger/nicks-optimizers
```

Requires Python 3.10+ and numpy. The distributed p-Laplacian examples also
need `pip install -e ".[examples]"` and an MPI library (e.g. `pip install mpich`).
For development:

```bash
git clone https://github.com/NickAlger/nicks-optimizers
cd nicks-optimizers
pip install -e ".[test,examples]" mpich   # without the extras, MPI tests are skipped
pytest          # the test suite
mypy            # strict type checking of src, tests and examples
```

## Quick example

```python
import numpy as np
from nicks_optimizers import trust_region_newton

A = np.diag(np.arange(1.0, 11.0))
c = np.ones(10)
d = np.arange(1.0, 11.0)                   # a diagonal preconditioner, here exact

x, summary = trust_region_newton(
    lambda x: 0.5 * x @ A @ x - c @ x,     # objective
    lambda x: A @ x - c,                   # gradient (a covector)
    lambda x, v: A @ v,                    # Hessian-vector product
    np.zeros(10),                          # starting point
    metric_solve=lambda x, w: w / d,       # M^{-1} w: the metric, and (P = M) the preconditioner
    display=1,                             # print one line per iteration
)
print(summary.reason, summary.f)
```

Without `metric_solve` the metric is the identity (for numpy arrays). See
[`examples/quickstart.py`](examples/quickstart.py) for a typed, matrix-free
version on the chained Rosenbrock function, with a separate preconditioner
as well.

## The problem pieces

```python
trust_region_newton(fun, grad, hessvec, x0, *, space=None, retract=None,
                    metric_solve=None, metric_apply=None, precond_solve=None,
                    operators_at=None, point_data=None, project_covector=None,
                    backend=None, ...)
```

| argument | signature | meaning |
|---|---|---|
| `fun` | `(x) -> float` | objective |
| `grad` | `(x) -> Covector` | differential `df(x)` |
| `hessvec` | `(x, v) -> Covector` | `H(x) v` |
| `x0` | `Point` | starting point |
| `space` | `VectorSpace` | the vector space structure; default numpy arrays (see [Other representations](#other-representations)) |
| `retract` | `(x, s) -> Point` | the step update; default `x + s` for numpy arrays, required with a `space` |
| `metric_solve` | `(x, w) -> Vector` | `M(x)^{-1} w`; default `M = I` for numpy arrays, required with a `space` |
| `metric_apply` | `(x, v) -> Covector` | `M(x) v`; needed only with `precond_solve`, and only with `metric_solve` |
| `precond_solve` | `(x, w) -> Vector` | `P(x)^{-1} w`, if `P` should differ from `M` |

`M` and `P` are symmetric positive definite. `metric_solve` and
`metric_apply` are separate: with `P = M` only `M^{-1}` is ever applied, and
`M` is needed only for a separate preconditioner, where the Lanczos solver
builds the M-Gram matrix of its basis with `metric_apply` and measures
residuals with `metric_solve`. When both are given they must be inverse to
each other.

`operators_at(x, data) -> PointOperators` gives all the operators at a
point instead, `PointOperators(hessvec=..., metric_solve=...,
precond_solve=..., metric_apply=...)` (pass `hessvec=None` and none of the
operator keywords). Setup the operators share (assembling a Hessian,
building a multigrid hierarchy) goes into `point_data(x) -> data`, which is
called once per new iterate and reused after rejected steps; `operators_at`
may be called several times per point and should only wrap `data` in
functions. Without `point_data`, `data` is `None`. With the default numpy
backend `data` can be any object, including the operators themselves
(`point_data=setup, operators_at=lambda x, ops: ops`, as the p-Laplacian
example does). `project_covector` is for redundant representations (see
[Manifolds](#manifolds)).

`backend` (see [Backends](#backends)) says how the iterations are computed;
the default runs them as plain Python on numpy.

The initial `radius` (default 1.0) is an absolute length in the M-norm, so
scale it to the problem when you know the size of a sensible first step. The
radius adapts after that, but a start that is far too large costs one
rejected step per factor of 4. An automatic choice from the model at `x0`
proved unreliable, and the evidence is in [`dev/TODO.md`](dev/TODO.md).

### Stopping test

Gradient norms are dual norms, `||g||_{M^{-1}} = sqrt(<g, M^{-1} g>)`, so the
test is invariant under changes of variables. The run stops when
`||g|| <= max(gtol, rtol * gnorm0)`, with `rtol = 1e-8`, `gtol = 0` and
`gnorm0 = ||g(x0)||` by default. With `convergence_norm="preconditioner"`
the norm is `||g||_{P^{-1}}` instead (the same unless `precond_solve` is
given; then it costs one more `P` solve per new iterate, which the Krylov
solve reuses). The Krylov tolerance is a forcing term times `||g||_{M^{-1}}`.

### Subproblem solvers

`krylov` chooses the subproblem solver:

- `"auto"` (default): truncated CG when `P = M` and `saddle_free=False`, and
  Lanczos otherwise.
- `"cg"`: Steihaug-Toint truncated CG (`solve_tr_subproblem_cg`). It runs
  CG preconditioned by `M` from `s = 0` and stops on the residual, at the
  first crossing of the boundary, or on negative curvature (then it follows
  that direction to the boundary). It keeps five vectors, does two reductions
  per step, and starts from the `M^{-1} g` already computed for the gradient
  norm. It requires `P = M` and `saddle_free=False`.
- `"lanczos"`: the reorthogonalized Lanczos solver, which is more robust in
  floating point. To tight tolerances on ill-conditioned problems, CG loses
  conjugacy and needs more steps; the tests include such a case. Lanczos
  also solves the trust-region problem exactly on the subspace when the
  boundary is active, where CG stops at the first crossing. That means
  fewer outer iterations, at more cost per step. With `P = M` its basis is
  M-orthonormal, so it applies only `M^{-1}` and gets the residual norm from
  the Lanczos recurrence for free.
- `"flexible"`: for a separate preconditioner that is not a fixed SPD
  operator: nonsymmetric (block triangular, Gauss-Seidel), nonlinear
  (inexact inner solves) or stochastic (randomized sweeps, minibatch SGD)
  (`solve_tr_subproblem_flexible`). P only proposes directions: the basis
  starts with `M^{-1} g`, grows by `P^{-1}` applied to the KKT residual, is
  made M-orthonormal, and the projected Hessian is formed explicitly, so every
  step is the exact model minimizer on its subspace and beats the Cauchy
  point whatever P does. It requires `precond_solve` and `metric_apply`.
  With a good fixed SPD P, Lanczos was as fast or faster in the outer loop;
  with nonsymmetric or stochastic P, Lanczos failed where the flexible solver
  converged. See
  [`experiments/flexible_preconditioners`](experiments/flexible_preconditioners/).

On the distributed p-Laplacian with M = P (2 ranks, 80 rings), CG took 116
Krylov steps at p = 3 against 140 for Lanczos, and 1.7 s against 2.7 s. At
p = 5 it needed 98 outer iterations against 88, with 251 Krylov steps against
344.

### Matched and mismatched metrics

When the trust region is active the exact step solves `(H + lam M) s = -g`. If
`M` is (a multiple of) `P`, that solution lies in the Krylov space of `P^{-1} H`
for every `lam` and a few Lanczos steps suffice. If `M` differs from `P` it does
not, and the subspace solutions improve slowly. Global convergence is
unaffected (every step beats the Cauchy point along `-P^{-1} g`), and boundary
solves stop when the model stops improving (`stall_tol`) rather than on the
residual. A separate `P` also costs, per Krylov step, one product with `M`
and one solve with `M^{-1}` for the explicit residual.
[`examples/matched_vs_mismatched.py`](examples/matched_vs_mismatched.py)
shows the effect.

## Saddle-free Newton

With `saddle_free=True` the model Hessian is `|H|_M = M U |Lambda| U^T M`, from
the generalized eigenproblem `H u = lambda M u`, approximated through the
M-Ritz pairs of the Krylov space. In the positive definite regime it equals the
Newton model, so fast local convergence is unchanged. The Krylov stopping test
uses the residual of the flipped system, `H s_flip + lam M s + g`, computed
from the Lanczos relation (for free when `P = M`).

On the problems studied so far it has not paid off inside a trust region:

- **Escaping saddles:** plain trust-region Newton already uses negative
  curvature through boundary steps, and escaped every saddle on its own.
- **Cost:** the |H| model needs every negative eigenvalue resolved, so it
  costs more Krylov steps when there are many.
- **Matrix completion:** it took a worse path.

See [`dev/TODO.md`](dev/TODO.md).

## Other representations

`trust_region_newton(..., space=...)` takes a `VectorSpace`: a frozen record of
functions for the vector space structure only (`scale_vector`,
`scale_covector`, `axpy_vector`, `axpy_covector`, `pair`, block operations,
and optionally `dimension`). Ready-made spaces:

- `numpy_space`: the default, one numpy array per vector, of any shape (a
  matrix-valued unknown can stay a matrix).
- `numpy_space_with_reductions(allreduce)`: one local numpy array per process
  (e.g. MPI ranks); only the reductions change. All decisions come from reduced
  numbers, so processes stay in lockstep.
- `tuple_tree_space(leaf_space)`: nested tuples whose leaves belong to
  `leaf_space`, numpy arrays by default (`tuple_tree_space()`), e.g.
  `((a, b), c)` with arrays of different shapes. Every operation works leaf
  by leaf with the leaf space's functions, so trees can hold distributed
  arrays (`numpy_space_with_reductions` leaves) or any other basic vector
  type. Trees are tuples only: a list raises, and `as_tuple_tree` converts
  one (e.g. `x0 = as_tuple_tree(x0)`). Leaves must not be tuples or lists.
- `tuple_block_space(...)`: any vector type from five functions; blocks are
  tuples of vectors. Override `pair_block` and `combine_*` with
  `dataclasses.replace` for batched versions.

With a space, pass the metric and the retraction too. Helpers for the simple
cases:

- `euclidean_metric_solve`, `euclidean_metric_apply`: `M = I`, the identity
  on components. For a tuple tree of arrays that is the direct sum of the
  leaves' Euclidean inner products; it also serves distributed arrays.
- `direct_sum(tree_of_leaf_operators)`: a different metric on each leaf of a
  tuple tree, for `M^{-1}` and `M` separately:

  ```python
  space = tuple_tree_space()
  M_inv = direct_sum(((lambda w: w / m1, lambda w: w / m2), lambda w: w / m3))
  trust_region_newton(..., space=space, retract=flat_retraction(space),
                      metric_solve=lambda x, w: M_inv(w))
  ```
- `flat_retraction(space)`: `x + s` for points stored like vectors, with the
  space's `axpy_vector`.
- `tree_map`, `tree_map2`, `tree_leaves`: maps over tuple trees, for writing
  your own functions.

`check_space(space, v1, v2, w1, w2)` verifies a space's internal consistency on
sample vectors. The method never modifies objects in place.

## Distributed vectors: the p-Laplacian

[`examples/p_laplacian.py`](examples/p_laplacian.py) solves
`-div(|grad u|^{p-2} grad u) = f` on the unit disk by minimizing the
regularized energy with linear finite elements, with the unknowns split
across MPI ranks:

```bash
pip install -e ".[examples]" mpich        # mpi4py, pyamg, scipy, and an MPI
mpiexec -n 4 python examples/p_laplacian.py --p 1.5 3 5
```

- **Mesh:** coarse at the center and finer toward the circle, so the lumped
  mass matrix varies 16-fold across the mesh.
- **Metric:** the L^2 norm of the lumped mass matrix, given by `metric_solve`
  and `metric_apply` (both diagonal).
- **Preconditioner:** separate from the metric. It is block Jacobi over the
  ranks, each block being the rank's part of the current Hessian, applied by
  one pyamg V-cycle. It is built once per iterate, as the point data.
  `--metric precond` uses it as the metric instead (`M = P`, CG).
- **Distribution:** vectors are `numpy_space_with_reductions` arrays of each
  rank's nodal values. Ghost values come in by halo exchange, and reductions
  use MPI Allreduce.
- **Right-hand side:** chosen so that the exact discrete solution is known.
  Every run reaches it to 3e-9 or better.
- **Rank count:** with `--precond jacobi` (partition-independent) the
  iterates are the same on any number of ranks, and the tests check this.

p is a dial: p = 1.5 takes 9 Newton iterations on every mesh, while p = 5 takes
40 to 120 and grows with refinement. The example's docstring explains why (the
L^2 trust region admits rough steps) and compares an H^1 metric.

[`examples/p_laplacian_eigen.py`](examples/p_laplacian_eigen.py) finds the first
p-Laplacian eigenvalue on the same distributed setup, by unconstrained
minimization of `f(u) = A(u)/p - log(B(u))/p` with `A = int |grad u|^p` and
`B = int |u|^p`:

- **Why this f:** its minimizers are exactly the ground states, scaled to
  `A = 1`, so `lambda_1 = A / B` there. Unlike the Rayleigh quotient, f has
  no flat directions.
- **Saddles:** f is nonconvex, and the higher eigenfunctions are its saddle
  points.
- **Accuracy:** the discrete eigenvalues converge at second order to the
  value from a radial shooting computation (`j_{0,1}^2` for p = 2).
- **Escaping saddles:** from random sign-mixed starts, plain trust-region
  Newton reaches the ground state every time. The saddle-free model gets
  there too, at up to 6 times the Krylov cost (see
  [`dev/TODO.md`](dev/TODO.md)).

## Manifolds

Nothing in the method assumes a flat space. On a manifold:

- **Retraction.** Pass `retract(x, s)`, the point structure.
  `hessvec` is the Hessian of the pullback `s -> f(retract(x, s))` at `s = 0`,
  which for a second-order retraction is the Riemannian Hessian.
- **Tangent and cotangent vectors** are whatever the representation says,
  tied to the current point. The method never moves a vector from one point
  to another, so no vector transport is needed.
- **Redundant representations.** Tangent spaces are often stored with more
  numbers than their dimension, e.g. low-rank tangent vectors as gauged
  factor variations `(dU, dV)` with `U^T dU = 0`. A covector component that
  pairs to zero with every tangent vector is invisible to every pairing, so
  reorthogonalization cannot remove its rounding errors, and the Lanczos
  recurrence amplifies them exponentially. Within tens of steps they swamp
  the covectors, the basis leaves the tangent space, and spurious Ritz values
  appear. Pass `project_covector(x, w)`, the projection onto canonical
  representatives; it is applied to gradients and to every Krylov covector.

[`examples/matrix_completion.py`](examples/matrix_completion.py) recovers a
1000 x 800 matrix with decaying spectrum from 7% of its entries by
trust-region Newton on the manifold of rank-r matrices:

- points as thin SVD factors;
- tangent vectors as gauged variations `(dU, dV)`, a `tuple_tree_space` of
  matrices, so the Frobenius metric is the identity in the components: the
  metric is `euclidean_metric_solve`;
- the truncated SVD of a rank-2r factorization as the retraction;
- the exact Riemannian Hessian, curvature term included.

It starts from `X = 0`, which the gauged representation handles, and fits
ranks 1, ..., r in turn, adding a component of size zero each time. Every
stage comes within 15% of the best rank-k error, and the last converges
quadratically.

## Diagnostics and the callback

`trust_region_newton` returns `(x, summary)`. The `RunSummary` holds the
stopping reason, one `IterationRecord` per iteration (objective, gradient norm,
radius, predicted and actual reduction, the Krylov solve's statistics and
per-step trace) and the total work. `history_array(summary)` gives a numeric
array. `display=1` prints a line per iteration and a summary, `display=2` adds
each Krylov trace, and `print_fn` redirects the output.

`callback(x, info)` is called after every iteration with the current iterate
and an `IterationInfo` holding the record, `f` and `g` at `x`, the step, and the
`ResumeState`. Returning `True` stops the run.

## Checkpointing and resuming

Besides `x`, the state carried between iterations is the trust-region radius,
the reference gradient norm and the iteration count; it is collected in
`ResumeState`, a record of plain numbers. Save `x`, `info.g` and `info.resume`
from the callback, and continue with

```python
x, summary = trust_region_newton(..., x_saved, resume=state, g0=g_saved)
```

With the same options and deterministic functions, the resumed run reproduces
the uninterrupted one exactly. A state saved at a different point than `x` is
detected and rejected. See [`examples/checkpointing.py`](examples/checkpointing.py).

## Backends

The iterations are pure pieces (`trust_region_core`) under a Python driver
(`trust_region_newton`), which keeps the stopping tests, records, display,
callback and resuming. A `Backend` record says how the pieces run: how they
are compiled (`jit`), how they loop (a bounded loop, built from a
framework's `scan` or `while_loop` by `loop_from_scan` or
`loop_from_while`), how they branch (`cond`), how arrays are updated
(`index_update`) and which array namespace they use (`xp`, through a fixed
list of names common to numpy, `jax.numpy` and the array API standard).
`numpy_backend`, the default, runs everything as plain Python. The library
imports no framework; a backend for one is a few lines in the user's code.

With a compiling backend, the functions given to `trust_region_newton` must
be traceable by it: no Python control flow on computed values, and arrays
out (`fun` returns a 0-d array rather than a float). Compiled backends run
the CG path for now; the Lanczos and flexible solvers follow
([`dev/jit_backends_plan.md`](dev/jit_backends_plan.md)).

## Package layout

Modules in dependency order. The first seven do not depend on the
trust-region code and are meant to be reused by other optimizers.

| module | contents |
|---|---|
| `backend` | how loops, branches and arrays are computed: `Backend`, `numpy_backend`, `loop_from_scan`, `loop_from_while` |
| `trees` | tuple trees: `Tree`, `as_tuple_tree`, `tree_map`, `tree_map2`, `tree_leaves` |
| `spaces` | the vector space structure: `VectorSpace`, `numpy_space`, `numpy_space_with_reductions`, `tuple_tree_space`, `tuple_block_space`, `flat_retraction`, `check_space` |
| `metrics` | the Hilbert structure, the metric given by operators: `euclidean_metric_solve`, `euclidean_metric_apply`, `direct_sum`, `vector_norm`, `covector_norm`, M-Gram matrix growth |
| `dense` | small dense linear algebra in coordinates: Rayleigh–Ritz for a projected pencil, the trust-region problem on a ball (with the hard case) |
| `krylov` | Krylov processes, immutable: preconditioned Lanczos (`lanczos_start`, `lanczos_advance`, `LanczosState`, `lanczos_tridiagonal`) and conjugate gradients (`cg_start`, `cg_curvature`, `cg_advance`, `CGState`, with recurrences for P-norms); optional covector projection for redundant representations |
| `criteria` | tests shared by inexact Newton methods: `gradient_converged`, `forcing_term`, `reduction_ratio` |
| `trust_region_subproblem` | the trust-region subproblem over a Krylov space: `solve_tr_subproblem` (Lanczos, any M, saddle-free), `solve_tr_subproblem_cg` (Steihaug-Toint, M = P) and `solve_tr_subproblem_flexible` (any preconditioner) |
| `trust_region_diagnostics` | records, `ResumeState`, and text output |
| `trust_region_core` | one iteration as pure pieces a backend can compile; `PointOperators` |
| `trust_region` | `trust_region_newton`, the driver |

The Lanczos process knows nothing about trust regions or the metric, so the
same `lanczos_advance` loop serves other consumers, e.g. Ritz values of the
pencil `(H, P)` for spectral estimates or low-rank approximations
([`tests/test_krylov.py`](tests/test_krylov.py) uses it as an eigensolver).

## License

MIT, see [LICENSE](LICENSE).
