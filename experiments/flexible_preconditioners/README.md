# Flexible preconditioners

**Question.** Can the trust-region subproblem be made robust to
preconditioners that are not fixed symmetric positive definite operators:
nonsymmetric ones (block triangular, Gauss-Seidel), nonlinear ones (inexact
inner Krylov solves), and stochastic ones (randomized row-action methods,
minibatch SGD)?

**Status.** In progress. The solver below is in the library (tested, opt-in
with `krylov="flexible"`); whether it should become the default for a
separate preconditioner is open (see the results).

## Approach

The Lanczos and CG solvers get the projected Hessian and the residual from
short recurrences that assume P is fixed SPD. With any other P those
quantities are wrong, so the step can increase the model or the solve can
fail. `solve_tr_subproblem_flexible` (in `src/nicks_optimizers/trust_region_subproblem.py`)
uses P only to propose directions:

- the basis starts with M^{-1} g, so every step beats the Cauchy point;
- each new direction is P^{-1} applied to the KKT residual R = H s + lam M s + g
  of the current step, made M-orthonormal (two Gram-Schmidt passes), with
  M^{-1} R as the fallback when P's output is not finite or adds nothing;
- the projected Hessian Q^T H Q is formed from stored products, and the small
  problem is solved exactly, so the step's model value is the true one;
- the stopping test uses the explicit residual norm ||R||_{M^{-1}}.

Applying P to the residual, rather than to the newest Hessian product as
Lanczos does, matters: it targets the system (H + lam M) s = -g that the step
solves. Per step it costs one product with H, one application of P, one
product with M and one solve with M^{-1}, like the Lanczos path with a
separate P; it stores Q, M Q and H Q. With P = M it spans the Lanczos space.

## Scripts

    python experiments/flexible_preconditioners/subproblem_study.py
    mpiexec -n 2 python experiments/flexible_preconditioners/p_laplacian_study.py

## Results

### The subproblem (dense, n = 200, M diagonal SPD, P built from H + 3 I)

`subproblem_study.py`. Relative gap (m(s) - m*)/|m*| of the true model after
40 steps, boundary active (radius 0.3); above 1 means the step is worse than
not moving. "flex-Hq" is the flexible basis expanded Lanczos-style with
P^{-1} H q_k.

| preconditioner | lanczos | flex-Hq | flexible |
|---|---|---|---|
| fixed SPD (P != M) | 3e-2 | 2e-2 | 3e-16 |
| block triangular | 0.35, own estimate wrong | 9e-2 | 7e-10 |
| inner CG, tol 0.3 | crash (2.4 at k = 20) | 2e-2 | 7e-16 |
| random coordinate descent | 0.10 | 0.11 | 2e-15 |
| minibatch SGD | 0.20 | 0.20 | 2e-15 |
| 30% noise | 2.0, own estimate wrong | 0.17 | 1e-8 |

The flexible solver converges in every case, also for interior steps and the
hard case (g orthogonal to the bottom eigenvector). Lanczos with the
nonlinear or stochastic preconditioners returns steps that increase the
model, by up to 5e11 times |m*| for interior steps. Applying P to H q_k is
robust but converges slowly: the residual is the right thing to precondition.

### Trust-region Newton, distributed p-Laplacian (2 ranks, 3011 unknowns)

`p_laplacian_study.py`: L^2 metric, P on each rank's owned Hessian block,
rtol 1e-10, at most 200 iterations and 100 Krylov steps per solve. Iterations
/ Krylov steps / final error ||u - u*|| / ||u*||:

| p | preconditioner | lanczos | flexible |
|---|---|---|---|
| 1.5 | AMG V-cycle (fixed SPD) | 10 / 118 / 1e-16 | 9 / 100 / 2e-13 |
| 1.5 | AMG-PCG to 30% (nonlinear) | 10 / 116 / 5e-13 | 10 / 125 / 1e-16 |
| 1.5 | Gauss-Seidel (nonsymmetric) | 200 / 1642 / 1e-5, max_iter | 10 / 425 / 4e-15 |
| 1.5 | random-order GS (stochastic) | 200 / 2849 / 2e-3, max_iter | 10 / 429 / 3e-16 |
| 1.5 | AMG with 30% noise | crash | 9 / 208 / 1e-13 |
| 3 | AMG V-cycle | 17 / 94 / 2e-16 | 23 / 108 / 8e-13 |
| 3 | AMG-PCG to 30% | 16 / 84 / 8e-12 | 21 / 102 / 3e-12 |
| 3 | Gauss-Seidel | 200 / 2918 / 1e-8, max_iter | 26 / 293 / 4e-16 |
| 3 | random-order GS | 200 / 2910 / 4e-9, max_iter | 25 / 296 / 5e-15 |
| 3 | AMG with 30% noise | crash | 23 / 194 / 3e-15 |
| 5 | AMG V-cycle | 68 / 450 / 1e-10 | 74 / 706 / 6e-10 |
| 5 | AMG-PCG to 30% | 52 / 277 / 2e-10 | 72 / 631 / 3e-12 |
| 5 | Gauss-Seidel | 200 / 1476 / 3e-4, max_iter | 113 / 2466 / 5e-13 |
| 5 | random-order GS | 200 / 2357 / 3e-3, max_iter | 114 / 2165 / 1e-11 |
| 5 | AMG with 30% noise | crash | 83 / 1198 / 2e-12 |

No fallbacks were needed in any run.

## Findings

1. **Robustness holds in the outer loop.** The flexible solver reached the
   exact discrete solution in all 15 runs. Lanczos failed in 9: with the
   Gauss-Seidel preconditioners it ran out of iterations at errors from
   1e-8 to 3e-3, and with the noisy one it crashed (an indefinite Gram
   matrix).
2. **With a good fixed SPD preconditioner, Lanczos is as good or better
   here**: 17 against 23 iterations at p = 3, and 450 against 706 Krylov
   steps at p = 5. The dense study's large win for a fixed separate P does
   not carry over. Inexact AMG-PCG to 30% did not break Lanczos in the outer
   loop, unlike in the dense study (possibly because the stall test ends its
   boundary solves after few steps).
3. Nonsymmetric Gauss-Seidel is a usable preconditioner with the flexible
   solver: 10, 26 and 113 iterations at p = 1.5, 3 and 5, against 9, 23 and
   74 for AMG with the same solver.

## Open questions

- Why does Lanczos win with a fixed SPD P in the outer loop? Hypothesis: its
  stall test ends boundary solves early with adequate steps, while the
  flexible solver spends Krylov steps reducing the boundary residual to the
  forcing tolerance far from the solution. Try a stall test or a looser
  boundary tolerance for the flexible solver.
- Memory: the basis grows with k. Restarting or truncating (keeping the
  current step and a few directions) is untested.
- The saddle-free model with the flexible solver is tested on dense
  problems but not studied in the outer loop.
- A problem where a stochastic preconditioner is natural: e.g. matrix
  completion with randomized row-action sweeps over the sampled entries.
  That needs preconditioner outputs that respect the gauge (a vector
  projection, which the library does not have yet).
- Stochastic preconditioners make runs nondeterministic; exact resumption
  then needs seeded randomness.
