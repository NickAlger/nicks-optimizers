# Line search against trust region

**Question.** On the same problems, operators and options, where does
`line_search_newton` (dev/line_search_plan.md) spend less work than
`trust_region_newton`, and where more?

**Status.** First round (the phase 5 check-in of the plan), CPU, numpy
backend, default options. The line search wins clearly on the convex
p-Laplacian with a separate or a weak preconditioner; the trust region wins
on Rosenbrock's curved valley; elsewhere they are close.

    python experiments/line_search_vs_trust_region/compare.py [--json results.json]

## Problems

- **rosenbrock**: the tests' chained Rosenbrock function, n = 20, from
  (-1.2, 1, ...): nonconvex, a curved valley, and a second local minimizer
  near (-1, 1, ..., 1); with M = I (CG), the Jacobi metric M = P (CG), a
  smoothing metric with a separate Jacobi preconditioner (Lanczos), and the
  saddle-free model.
- **double well**: the tests' double well, n = 30 (nonconvex, saddles), with
  CG and with the saddle-free model.
- **p-laplacian**: a finite-difference p-Laplacian energy (p = 4, eps = 1e-2,
  convex) on the unit square, from u = 0, with M = P = the discrete H^1
  inner product (a sparse LU), M = h^2 I with P = H^1 (Lanczos with a
  separate preconditioner), and M = P = diag(H^1) (a weak preconditioner).
- **matrix completion**: the tests' rank-3 fit of a rank-8 matrix on the
  fixed-rank manifold.

## Results

Counts over the run: iterations, evaluations of f and g, Hessian-vector
products, solves with M^{-1} or P^{-1}, products with M; `results.json` has
them all.

| problem | method | its | f evals | Hv | solves | M products | time |
|---|---|---|---|---|---|---|---|
| rosenbrock, M = I | TR | 54 | 55 | 409 | 436 | 0 | 0.04 s |
| | LS | 114 | 203 | 1049 | 1152 | 0 | 0.13 s |
| rosenbrock, Jacobi M = P | TR | 57 | 58 | 343 | 364 | 0 | 0.05 s |
| | LS | 91 | 152 | 834 | 913 | 0 | 0.12 s |
| rosenbrock, smoothing M, Jacobi P | TR | 73 | 74 | 665 | 1459 | 911 | 0.51 s |
| | LS | 118 | 190 | 965 | 2142 | 118 | 0.50 s |
| rosenbrock, saddle-free | TR | 54 | 55 | 411 | 462 | 0 | 0.22 s |
| | LS | 47 | 61 | 405 | 453 | 0 | 0.17 s |
| double well | TR | 13 | 14 | 24 | 31 | 0 | 0.01 s |
| | LS | 13 | 16 | 27 | 35 | 0 | 0.01 s |
| double well, saddle-free | TR | 20 | 21 | 49 | 135 | 86 | 0.04 s |
| | LS | 18 | 33 | 58 | 153 | 58 | 0.04 s |
| p-laplacian n = 3,969, H^1 | TR | 10 | 11 | 56 | 66 | 0 | 0.09 s |
| | LS | 10 | 20 | 55 | 66 | 0 | 0.13 s |
| p-laplacian n = 3,969, L^2 / H^1 | TR | 18 | 19 | 117 | 266 | 154 | 0.28 s |
| | LS | 10 | 20 | 52 | 125 | 10 | 0.11 s |
| p-laplacian n = 3,969, Jacobi | TR | 211 | 212 | 2117 | 2208 | 0 | 1.15 s |
| | LS | 47 | 72 | 2123 | 2171 | 0 | 0.99 s |
| p-laplacian n = 16,129, H^1 | TR | 10 | 11 | 60 | 70 | 0 | 0.46 s |
| | LS | 10 | 20 | 59 | 70 | 0 | 0.41 s |
| p-laplacian n = 16,129, L^2 / H^1 | TR | 18 | 19 | 132 | 296 | 169 | 1.20 s |
| | LS | 10 | 20 | 71 | 163 | 10 | 0.57 s |
| p-laplacian n = 16,129, Jacobi | TR | 522 | 523 | 6748 | 7026 | 0 | 7.98 s |
| | LS | 184 | 274 | 6672 | 6857 | 0 | 7.61 s |
| matrix completion | TR | 23 | 24 | 80 | 82 | 0 | 0.10 s |
| | LS | 18 | 26 | 75 | 84 | 0 | 0.09 s |

Every run converged. On Rosenbrock the methods sometimes end at different
local minimizers (f = 0 or f = 3.99), as either may from this start.

## Findings

1. **Curved valleys favor the trust region.** On Rosenbrock the Newton
   direction overshoots the valley, the first trial fails badly, and the
   interpolation asks for less than a tenth of the step, which the
   safeguard clips to 0.1: many iterations take alpha = 0.1, about twice
   the Hessian products of the trust region, whose radius remembers the
   right length. SciPy's line-search Newton-CG needs 156 iterations on the
   same problem (ours: 114). The saddle-free model removes the gap (47
   against 54 iterations).
2. **A separate preconditioner favors the line search.** With M = L^2 and
   P = H^1 the trust-region steps often lie on the boundary, where the
   Krylov space of P^{-1} H approximates the solution of (H + lam M) s = -g
   slowly (the caveat in trust_region_subproblem); a line search has no
   boundary. 10 against 18 iterations, half the Hessian products and
   solves, and 10 products with M instead of 154: truncating Lanczos keeps
   no Gram matrix, so only ||d||_M costs one product per iteration.
3. **A weak preconditioner favors the line search in iterations** (4.5
   times fewer at n = 3,969, 2.8 at n = 16,129), but the Krylov work is the
   same, so time is too.
4. **A degenerate start costs function evaluations.** At u = 0 the p = 4
   energy has a nearly singular Hessian, so the first Newton direction is
   about 4,000 long; the line search cuts it to alpha = 2e-4 in 10 trials,
   and then takes unit steps to the end. step_max would shorten the first
   iteration; the trust region pays instead with its initial radius.
5. **Elsewhere the two are close**: the double well, matrix completion.

## Open questions

- An extrapolating line search (lengthening steps when the first trial is
  accepted with room to spare) or a remembered step length might close the
  gap on curved valleys.
- The comparison under compilation (JAX) is phase 6 of the plan.
