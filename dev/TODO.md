# Open questions and things to investigate

Findings from stress tests that are not resolved yet. Each entry says what
was seen, how to reproduce it, and what to try.

## Saddle-free Newton underperforms on low-rank matrix completion

**Seen** (2026-10-24, `examples/matrix_completion.py`, direct rank-r fits
from `zero_point`, no rank continuation):

| problem | plain Newton | saddle-free |
|---|---|---|
| 600 x 500, r = 5, singular values 2^-k, K = 4 dof | converged: 41 it, 464 Lanczos steps, error 1.16x best | max_iter: 200 it, 9006 Lanczos steps, error 16x best |
| 400 x 300, r = 5, singular values 4^-k, K = 4 dof | (stalls too; rank continuation: 58 it, 1.16x best) | max_iter: 200 it, 9246 Lanczos steps, error 390x best |

Per-iteration behavior of the saddle-free runs (400 x 300, 2^-k):

- Whenever the projected Hessian has negative M-Ritz values, the Krylov
  solve tends to run to `max_lanczos` (stop reason `max`) or to stall; the
  surrogate residual of the flipped system does not reach its tolerance.
- Boundary steps in those iterations are often rejected outright
  (rho = -1.7, -2.8, -14): the objective increases.
- `true_model_value` differs from `model_value` by 30-60% there, as
  expected when the flipped directions carry weight.
- Once no negative Ritz values remain, solves are short and rho is about 1,
  as for plain Newton.

**Context.** The negative curvature comes from the manifold's curvature
term, 2 <R, dU S^-1 dV_perp^T>, which grows like ||R|| / s_r. It is part of the
Hessian of the pullback (verified by finite differences), so plain
trust-region Newton exploits it through boundary steps; |H| treats the same
directions as strongly positively curved.

**Hypotheses.**

1. The |H| model is too conservative in exactly the directions that need
   large moves (the components that have not converged yet), so progress
   is slow.
2. The surrogate residual (from flipped Ritz vectors) converges slowly
   while the negative Ritz pairs are inaccurate, so every solve costs
   `max_lanczos` steps.
3. The rejections come from steps along directions whose true curvature
   is large and negative, where higher-order terms dominate over the radius
   used. If so, plain Newton would be rejected there too; check whether it
   is.

**To try.**

- Log, per Krylov step, the surrogate residual next to the true residual
  of |H|_M s + lam M s + g (dense |H|_M on a small instance).
- Compare the model's predicted change along each M-Ritz direction of the
  step with finite differences of f(retract(x, t s)).
- Run saddle-free on standard nonconvex test problems with known saddles
  (e.g. the double well in the tests, the p-Laplacian eigenproblem) to see
  whether the behavior is specific to the curvature term of fixed-rank
  manifolds.
- Variants: flip only Ritz values below -tau * |theta|_max; switch to |H|
  only when the plain model's step is rejected; stop on the plain residual.

**Reproduce.**

```python
import numpy as np, matrix_completion as mc          # examples/ on sys.path
rng = np.random.default_rng(0)
N, M, r = 600, 500, 5
a = mc.random_low_rank((N, M), 0.5 ** np.arange(30, dtype=float), rng)
S = mc.sample_uniformly(a, 4 * (N + M - r) * r, rng)
x0 = mc.zero_point((N, M), r, rng)
for sf in (False, True):
    x, summary = mc.complete(S, x0, saddle_free=sf, display=2)
```

## The default initial radius is scale-dependent

`radius=1.0` is an absolute length in the M-norm. In the matrix-completion
rank continuation, each new stage started 100-1000 times too large, and the
first iterations were rejections that shrank it by 4x each. The example now
passes the smallest singular value found so far. Library options: a radius
derived from the problem when none is given (e.g. the length of the Cauchy
step, or ||g||_{M^-1} / ||H||), or documenting the choice prominently.
