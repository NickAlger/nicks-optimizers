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

**Update (2026-10-24, later).** Two more experiments; hypotheses 1 and 2
both hold, in different problems, and neither looks like a bug.

*Problem with known saddles* (`examples/p_laplacian_eigen.py`, 1898
unknowns, 10 random sign-mixed starts per p; the higher eigenfunctions are
saddles):

| p | plain: ground state, median it / Lanczos | saddle-free: ground state, median it / Lanczos |
|---|---|---|
| 2 | 10/10, 8 / 35 | 10/10, 13 / 250 |
| 3 | 10/10, 14 / 51 | 10/10, 14 / 71 |
| 4 | 10/10, 19 / 70 | 10/10, 19 / 71 |

Plain trust-region Newton escapes every saddle by itself. At a random start
the Hessian has dozens of negative eigenvalues (the -diag(m |u|^{p-2}) / B
term). Plain TR needs only the leftmost one: its solves go to the boundary
(lam = 360) and stall after 3-5 Lanczos steps. The saddle-free model needs
all of them resolved. In one solve the count of negative Ritz values grew
1, 2, 4, 7, 13, 25, 44, 62 over 100 steps while the surrogate residual stayed
near 20 times its tolerance (hypothesis 2). Its interior steps also
overshot the non-quadratic -log B term (rho = -15).

*Matrix completion, traced* (600 x 500, r = 5, 2^-k, from X = 0):
saddle-free takes a different path rather than a slower one. By iteration
2 it has grown all components together: singular values
[.65 .49 .44 .43 .41], against [.26 .17 .16 .15 .15] for plain TR. Flipping the
negative curvature along junk components turns them into ordinary descent
directions. From iteration 80 it sits in a nearly flat region,
f = 4.40e-5 for 120 iterations against the optimum 3.85e-5. Every Krylov
solve hits the cap with rho = 1.01, and the full-matrix error drifts from
0.28 to 0.50. This is hypothesis 1: inside a trust region the |H| model
discards information that plain TR exploits through boundary steps.

*Tried:* letting the model-stall test also stop interior saddle-free solves
(lam = 0). No real effect: eigen p = 2 median Lanczos 250 -> 239, and
completion still fails (error 16x -> 12x best).

*Current view.* No evidence so far that saddle-free helps inside a trust
region; it costs more whenever H has many negative eigenvalues. Possible
roles to test: a fallback only when plain TR stalls with a tiny gradient
and negative curvature; or line-search Newton, where nothing else handles
negative curvature. A test problem where plain TR genuinely stalls at a
saddle is still missing. Symmetric starts are no good: they trap both
models, since the Krylov space never sees the symmetric negative direction.

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

## Trust region in the preconditioner's norm (M = P)

The classic Steihaug-Toint choice, and the natural one when P is spectrally
equivalent to the Hessian (multigrid): mesh-independent and free. It cannot
be expressed today, since `ball_inner` needs (u, v)_P and multigrid only gives
P^{-1}. In the matched case nothing has to be computed: the Lanczos basis is
P-orthonormal, so G = I and W = I (including the pending vector), and
||g||_{P^{-1}} = gamma0; the residual norm reduces to
gamma_next |y_flip[-1]|. Proposal: omitting both inner products means M = P.
The outer loop still needs ||g||_{P^{-1}} = sqrt(<g, P^{-1} g>) before the
subproblem (one extra solve per iteration, or share it with the first
Lanczos vector).

Evidence (2026-10-24, `examples/p_laplacian.py`, serial, p = 5, eps = 1e-3,
20/40/80 rings): the L^2 ball needed 58 / 99 / 156 iterations and 298 / 542 / 1127
Lanczos steps; an H^1 ball (stiffness matrix, exact dual by sparse LU)
needed 22 / 27 / 40 and 78 / 125 / 163. At p = 3: 11 / 13 / 16 against 8 / 9 / 10. The
L^2 ball admits rough steps that a gradient-dependent energy punishes, and
more of them as the mesh is refined.

## Communication per Lanczos step

In a distributed run each Lanczos step does six global reductions:
<Hq, q>, two reorthogonalization passes (one `pair_block` each),
<t, P^{-1} t>, and one call to each inner product (Gram matrices). Fusing
them (delta is the newest coefficient of the first reorthogonalization pass;
the two Gram rows could go in one reduction) would cut latency for small
local problems. Not urgent: with multigrid the local work dominates.
