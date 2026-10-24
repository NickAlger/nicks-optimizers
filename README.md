# nicks-optimizers

Matrix-free optimization methods in Python. The first is `trust_region_newton`:
an inexact Newton trust-region method in which the **Hessian preconditioner** and
the **shape of the trust region** are independent choices.

- **Ellipsoidal trust region given by inner products.** The trust region is
  `{s : (s, s)_M <= radius^2}` for a metric `M(x)` that may depend on the iterate.
  You supply the inner product and its dual, not the operators `M` and `M^{-1}`;
  for an ellipsoid with covariance `B`, `M = B^{-1}`.
- **Preconditioner as an accelerant only.** `P(x)` shapes the Krylov space;
  all geometry (step lengths, curvature, stopping tests) is measured with `M`.
- **Lanczos subproblem solver.** An explicitly reorthogonalized, P-preconditioned
  Lanczos basis; the small trust-region problem is solved exactly in
  M-orthonormal coordinates (including the hard case). No Hessian-vector
  products are spent beyond one per Lanczos step.
- **Saddle-free option.** `saddle_free=True` replaces the Hessian by its absolute
  value *with respect to M* (generalized eigenvalues of `H u = theta M u`), so
  saddle points repel iterates instead of attracting them.
- **Representation-generic.** Points, vectors and covectors can be numpy arrays
  (the default), distributed objects, collections of arrays, or anything else,
  through a `VectorSpace` record of functions.
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

## Installation

```bash
pip install git+https://github.com/NickAlger/nicks-optimizers
```

Requires Python 3.10+ and numpy. For development:

```bash
git clone https://github.com/NickAlger/nicks-optimizers
cd nicks-optimizers
pip install -e ".[test]"
pytest          # the test suite
mypy            # strict type checking of src, tests and examples
```

## Quick example

```python
import numpy as np
from nicks_optimizers import trust_region_newton

A = np.diag(np.arange(1.0, 11.0))
c = np.ones(10)
b = np.linspace(0.5, 2.0, 10)              # covariance of the trust-region ellipsoid

x, summary = trust_region_newton(
    lambda x: 0.5 * x @ A @ x - c @ x,     # objective
    lambda x: A @ x - c,                   # gradient (a covector)
    lambda x, v: A @ v,                    # Hessian-vector product
    lambda x, w: w,                        # preconditioner solve P^{-1} w (here P = I)
    lambda x, U, v: U.T @ (v / b),         # ball inner products (u_i, v)_M, M = B^{-1}
    lambda x, Psi, phi: Psi.T @ (b * phi), # their exact dual, (psi_i, phi)_{M^{-1}}
    np.zeros(10),                          # starting point
    display=1,                             # print one line per iteration
)
print(summary.reason, summary.f)
```

See [`examples/quickstart.py`](examples/quickstart.py) for a typed, matrix-free
version on the chained Rosenbrock function.

## The problem pieces

| argument | signature | meaning |
|---|---|---|
| `fun` | `(x) -> float` | objective |
| `grad` | `(x) -> Covector` | differential `df(x)` |
| `hessvec` | `(x, v) -> Covector` | `H(x) v` |
| `precond_solve` | `(x, w) -> Vector` | `P(x)^{-1} w`, `P(x)` symmetric positive definite |
| `ball_inner` | `(x, U, v) -> array` | `(u_i, v)_M` for each vector `u_i` in the block `U` |
| `ball_dual_inner` | `(x, Psi, phi) -> array` | `(psi_i, phi)_{M^{-1}}` for each covector in `Psi` |
| `x0` | `Point` | starting point, in the representation of `space` |

Optional: `space` (see [Other representations](#other-representations)),
`retract(x, s) -> Point` and `project_covector(x, w) -> Covector` (see
[Manifolds](#manifolds)).

`H` and `P^{-1}` are the only operators the method applies. The two inner
products must be **exact duals** of each other: the residual norm on the
trust-region boundary relies on `(M u, M v)_{M^{-1}} = (u, v)_M`. For `M = I`
pass `euclidean_inner` and `euclidean_dual_inner`. Inner products take a block
first so an implementation can apply `M` once per call, e.g.
`lambda x, U, v: U.T @ (M @ v)`.

Gradient norms are dual norms, `||g||_{M^{-1}}`, so the stopping tests are
invariant under changes of variables. The run stops when
`||g|| <= max(gtol, rtol * gnorm0)`, where `gnorm0` defaults to `||g(x0)||`.

### Matched and mismatched metrics

When the trust region is active the exact step solves `(H + lam M) s = -g`. If
`M` is (a multiple of) `P`, that solution lies in the Krylov space of `P^{-1} H`
for every `lam` and a few Lanczos steps suffice. If `M` differs from `P` it does
not, and the subspace solutions improve slowly. Global convergence is
unaffected (every step beats the Cauchy point along `-P^{-1} g`), and boundary
solves stop when the model stops improving (`stall_tol`) rather than on the
residual. [`examples/matched_vs_mismatched.py`](examples/matched_vs_mismatched.py)
shows the effect.

## Saddle-free Newton

With `saddle_free=True` the model Hessian is `|H|_M = M U |Lambda| U^T M`, from
the generalized eigenproblem `H u = lambda M u`, approximated through the
M-Ritz pairs of the Krylov space. In the positive definite regime it equals the
Newton model, so fast local convergence is unchanged. The Krylov stopping test
uses the residual of the flipped system, `H s_flip + lam M s + g`, computed
from the Lanczos relation at no extra cost.

## Other representations

`trust_region_newton(..., space=...)` takes a `VectorSpace`: a frozen record of
functions (`scale_vector`, `scale_covector`, `axpy_covector`, `pair`, block
operations, and optionally `translate` and `dimension`). Ready-made spaces:

- `numpy_space`: the default, one numpy array per vector.
- `numpy_space_with_reductions(allreduce)`: one local numpy array per process
  (e.g. MPI ranks); only the reductions change. All decisions come from reduced
  numbers, so processes stay in lockstep.
- `tuple_block_space(...)`: any vector type from five functions; blocks are
  tuples of vectors. Override `pair_block` and `combine_*` with
  `dataclasses.replace` for batched versions.

`check_space(space, v1, v2, w1, w2)` verifies a space's internal consistency on
sample vectors. The method never modifies objects in place.

## Manifolds

Nothing in the method assumes a flat space. On a manifold:

- **Retraction.** Pass `retract(x, s)`; the space needs no `translate`.
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
- tangent vectors as gauged variations, so the Frobenius inner product is the
  dot product of the components;
- the truncated SVD of a rank-2r factorization as the retraction;
- the exact Riemannian Hessian, curvature term included.

It starts from `X = 0`, which the gauged representation handles, and fits
ranks 1, ..., r in turn, adding a component of size zero each time. Every
stage comes within 15% of the best rank-k error, and the last converges
quadratically.

## Diagnostics and the callback

`trust_region_newton` returns `(x, summary)`. The `RunSummary` holds the
stopping reason, one `IterationRecord` per iteration (objective, gradient norm,
radius, predicted and actual reduction, the Lanczos solve's statistics and
per-step trace) and the total work. `history_array(summary)` gives a numeric
array. `display=1` prints a line per iteration and a summary, `display=2` adds
each Lanczos trace, and `print_fn` redirects the output.

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

## Package layout

Modules in dependency order. The first five do not depend on the trust-region
code and are meant to be reused by other optimizers.

| module | contents |
|---|---|
| `spaces` | how points, vectors and covectors are represented: `VectorSpace`, `numpy_space`, `numpy_space_with_reductions`, `tuple_block_space`, `check_space` |
| `metrics` | inner products and norms: `InnerProduct`, `DualInnerProduct`, `euclidean_inner`, `vector_norm`, `covector_norm`, Gram-matrix growth |
| `dense` | small dense linear algebra in coordinates: Rayleigh–Ritz for a projected pencil, the trust-region problem on a ball (with the hard case) |
| `krylov` | preconditioned Lanczos as an immutable process: `lanczos_start`, `lanczos_advance`, `LanczosState`, `lanczos_tridiagonal`; optional covector projection for redundant representations |
| `criteria` | tests shared by inexact Newton methods: `gradient_converged`, `forcing_term`, `reduction_ratio` |
| `trust_region_subproblem` | the trust-region subproblem over a Krylov space: `solve_tr_subproblem` |
| `trust_region_diagnostics` | records, `ResumeState`, and text output |
| `trust_region` | `trust_region_newton` |

The Lanczos process knows nothing about trust regions or inner products, so the
same `lanczos_advance` loop serves other consumers, e.g. Ritz values of the
pencil `(H, P)` for spectral estimates or low-rank approximations
([`tests/test_krylov.py`](tests/test_krylov.py) uses it as an eigensolver).

## License

MIT, see [LICENSE](LICENSE).
