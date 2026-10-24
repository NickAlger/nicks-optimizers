# Open questions and things to investigate

Findings from stress tests that are not resolved yet. Each entry says what
was seen, how to reproduce it, and what to try.

## Saddle-free Newton inside a trust region (first seen on matrix completion)

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
saddles; deterministic multigrid):

| p | plain: ground state, median it / Lanczos | saddle-free: ground state, median it / Lanczos |
|---|---|---|
| 2 | 10/10, 8 / 36 | 10/10, 13 / 208 |
| 3 | 10/10, 14 / 54 | 10/10, 15 / 75 |
| 4 | 10/10, 18 / 67 | 10/10, 18 / 67 |

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
(lam = 0). No real effect: eigen p = 2 median Lanczos 250 -> 239 (measured
before the multigrid was made deterministic), and
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

**Experiments (2026-10-24), no library change made.** Iterations /
rejections on every example problem:

| problem | radius 1 | Cauchy-step radius | shrink x1/16 when rho < 0 |
|---|---|---|---|
| Rosenbrock, identity ball | 119 / 19 | 91 / 4 | 109 / 3 |
| Rosenbrock, smoothing ball | 171 / 47 | - | 229 / 42 |
| Rosenbrock, x-dependent ball | 118 / 9 | - | 229 / 42 |
| double well | 16 / 1 | 17 / 2 (fallback) | 18 / 1 |
| matrix completion from X = 0 | 41 / 7 | 42 / 8 | 38 / 5 |
| p-Laplacian p = 3 from 0 | 12 / 1 | 18 / 7 (radius 1.6e3) | 12 / 1 |
| p-Laplacian p = 5 from 0 | 62 / 16 | 64 / 27 (radius 7e9) | 61 / 10 |
| eigenproblem p = 2, random start | 8 / 1 | 8 / 1 | 11 / 1 |

Cauchy-step radius: the M-norm length of the unconstrained Cauchy step
along -P^{-1} g at x0, falling back to ||P^{-1} g||_M when the curvature
is not positive. Neither change is a clear win. The Cauchy length is
absurd exactly where the Hessian at x0 is degenerate (the p-Laplacian at
u = 0, curvature ~ eps^{p-2}), which is where a good guess would matter.
Faster shrinking helps some problems and hurts others. Recommendation: keep
the defaults, say plainly in the docs that the radius is an absolute
M-norm length, and choose it from the problem's scale when one is known (as
the matrix-completion continuation does).

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
20/40/80 rings): the L^2 ball needed 62 / 91 / 176 iterations and 311 / 512 / 1274
Lanczos steps; an H^1 ball (stiffness matrix, exact dual by sparse LU)
needed 23 / 26 / 42 and 78 / 108 / 212. At p = 3: 12 / 14 / 17 against 8 / 9 / 10. The
L^2 ball admits rough steps that a gradient-dependent energy punishes, and
more of them as the mesh is refined.

**Implemented** (PR #1, merged 2026-10-24): passing `None` for both inner
products means M = P, in `solve_tr_subproblem` and `trust_region_newton`. It agrees with explicit (u, v)_P inner products step
for step (Jacobi on Rosenbrock; dense tests at all radii, plain and
saddle-free) and makes no inner-product calls. `p_laplacian.py` gains
`--ball precond` and a fixed `--precond laplace`. Results on 2 ranks,
iterations / Lanczos steps on 20/40/80 rings:

| p | L^2 ball, Hessian AMG | M = P, Hessian AMG | M = P, Laplacian AMG |
|---|---|---|---|
| 1.5 | 9/9/9; 85/120/167 | 12/13/13; 63/101/126 | 9/10/13; 81/111/215 |
| 3 | 16/17/23; 89/102/130 | 16/19/21; 59/90/140 | 12/18/21; 83/177/290 |
| 5 | 43/89/120; 310/507/790 | 44/56/88; 170/259/344 | too slow: P far from H |

Fewer Lanczos steps (the matched solves stop on the residual, not the stall
test), and milder mesh dependence at p = 5. Not the serial H^1 ball's
23/26/42: block Jacobi breaks the H^1 norm at rank interfaces.

Still open: with an x-dependent P, gradient norms change meaning between
iterates. At u = 0, p = 5, P(0) is about eps^3 K, so ||g0||_{P^{-1}} is huge,
and rtol * gnorm0 stops at L^2 errors of 3e-6 to 3e-5 (versus 1e-9 for the
L^2 ball). Users would pass gtol or gnorm0, or the test could use a fixed
norm.

**Conventional path** (2026-10-24): with M = P and no saddle-free model,
`krylov="auto"` now uses Steihaug-Toint truncated CG
(`solve_tr_subproblem_cg`): five vectors, two reductions per step, and it
reuses the `P^{-1} g` of the gradient norm. `krylov="lanczos"` keeps the
reorthogonalized solver. On the p-Laplacian with M = P, CG took the same or
fewer Krylov steps and less time; at p = 5 it took about 10-25% more outer
iterations, since it stops at the first boundary crossing.

## Communication per Lanczos step

The CG path for M = P needs only two reductions per step (<H p, p> and
<r, P^{-1} r>). The rest of this entry is about the Lanczos path.

In a distributed run each Lanczos step does six global reductions:
<Hq, q>, two reorthogonalization passes (one `pair_block` each),
<t, P^{-1} t>, and one call to each inner product (Gram matrices). Fusing
them (delta is the newest coefficient of the first reorthogonalization pass;
the two Gram rows could go in one reduction) would cut latency for small
local problems. Not urgent: with multigrid the local work dominates.
