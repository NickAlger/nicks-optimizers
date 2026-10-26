# lbfgs at the phase-2 check-in

**Questions** (dev/lbfgs_plan.md, phase 2):

- How does lbfgs compare with the Newton methods on the same problems?
- Does the metric give mesh independence on the p-Laplacian?
- How often are pairs skipped, line searches restarted, or directions
  replaced by the angle test, and what should angle_tol be?
- A first look at the scalings and the memory (phase 8 studies the options
  fully).

**Run:** `python experiments/lbfgs_checkin/compare.py [--json results.json] [--parts ABCD]`.
`results.json` holds the run below: 73 runs, all deterministic except the
times.

**Problems:**
- chained Rosenbrock (n = 20) and the double well (n = 30) from the tests;
- the p-Laplacian energy (p = 4, eps = 1e-2; convex, degenerate at the
  start u = 0) with N x N interior nodes.

**Metrics:**
- I (Euclidean), L2 (h^2 I), Jacobi (diag H^1) and H1 (sparse LU solve,
  sparse product);
- on Rosenbrock, the tests' point-dependent Jacobi metric
  diag(max(|H_ii(x)|, 1)).

lbfgs runs with its defaults (memory 10, scaling "y", gamma0 = 1, "relative"
curvature test, angle_tol = 1e-8). The Newton methods use the same metric as
M = P.

## A. Against the Newton methods

grad + Hv counts derivative evaluations (a Hessian-vector product costs
about a gradient); solves are applications of M^{-1}.

| problem | method | its | grad | Hv | grad + Hv | solves | M products |
|---|---|---|---|---|---|---|---|
| Rosenbrock, I | TR | 54 | 51 | 409 | 460 | 436 | 0 |
| | LS | 114 | 115 | 1049 | 1164 | 1152 | 0 |
| | lbfgs | 128 | 129 | 0 | **129** | 384 | 128 |
| Rosenbrock, Jacobi(x) | TR | 57 | 53 | 343 | 396 | 364 | 0 |
| | LS | 91 | 92 | 834 | 926 | 913 | 0 |
| | lbfgs | 116 | 117 | 0 | **117** | 348 | 116 |
| double well, I | TR | 13 | 13 | 24 | 37 | 31 | 0 |
| | lbfgs | 22 | 23 | 0 | **23** | 64 | 22 |
| p-Laplacian 63^2, Jacobi | LS | 47 | 48 | 2123 | 2171 | 2171 | 0 |
| | lbfgs | 907 | 908 | 0 | **908** | 2721 | 907 |
| p-Laplacian 127^2, Jacobi | LS | 184 | 185 | 6672 | 6857 | 6857 | 0 |
| | lbfgs | 1826 | 1827 | 0 | **1827** | 5478 | 1826 |
| p-Laplacian 63^2, H1 | TR | 10 | 11 | 56 | 67 | **66** | 0 |
| | lbfgs | 41 | 42 | 0 | **42** | 123 | 41 |
| p-Laplacian 127^2, H1 | TR | 10 | 11 | 60 | 71 | **70** | 0 |
| | lbfgs | 49 | 50 | 0 | **50** | 147 | 49 |

- **Derivative evaluations.** lbfgs spends fewer everywhere: 2 to 4 times
  fewer than the best Newton method on Rosenbrock and with the Jacobi
  metric, 30-40% fewer with the H1 metric.
- **Solves.** It spends more: three M^{-1} solves per iteration (the
  gradient norm, the base, the "y" fit). With the H1 metric that is 123
  against 66 solves, and the sparse LU solves dominate the time (in the
  recorded run 0.20 s against 0.09 s at N = 63). With scaling="none", which works as well here
  (39 iterations, below), it is two per iteration. The fixed path (phase 6)
  makes it one for a constant metric: 41 solves against 66.
- **Same minimizers.** All three methods reach the same minimizers: on
  Rosenbrock with the Jacobi metric, all reach the second local minimizer.

## B. The metric across meshes (iterations)

| N | I | L2 | Jacobi | H1 |
|---|---|---|---|---|
| 15 | 150 | 150 | 157 | 25 |
| 31 | 376 | 344 | 329 | 33 |
| 63 | 787 | 871 | 907 | 41 |
| 127 | 1748 | 1681 | 1826 | 49 |

- **H1:** about 8 more iterations per refinement (logarithmic growth), so
  nearly mesh-independent.
- **I, L2 and Jacobi:** the iterations double with each refinement, as the
  square root of the condition number h^-2 predicts.
- **The gap:** 36 times fewer iterations at N = 127. L2 and Jacobi are
  nearly multiples of I here, so they behave alike.

## C. The angle test

| problem | angle_tol 0, 1e-8, 1e-4, 1e-2 | 0.1 |
|---|---|---|
| Rosenbrock, I | 128 its, 0 fallbacks | 778 its, 73 fallbacks |
| p-Laplacian 63^2, H1 | 41, 0 | 41, 0 |
| p-Laplacian 63^2, I | 787, 0 | 1753, 45 |

Up to 1e-2 the test never fires. At 0.1 it rejects good quasi-Newton
directions, whose cosines are often below 0.1, and costs 2 to 6 times the
iterations. The default 1e-8 is a pure safeguard: keep it.

## Skips, restarts, fallbacks

Over all 73 runs with the default angle test:

- no restart, and no fallback;
- skipped pairs only on Rosenbrock with the point-dependent Jacobi metric
  (3 of 116) and on the double well (2 of 22, near its local maximum).

Armijo backtracking without Wolfe has not caused trouble on these
problems.

## D. Scalings, first step, memory (iterations / f evaluations / solves)

| | Rosenbrock, I | p-Laplacian 63^2, H1 | p-Laplacian 63^2, I |
|---|---|---|---|
| "y" (default) | **128** / 150 / 384 | 41 / 45 / 123 | 787 / 816 / 2361 |
| "s" | 171 / 390 / 342 | 41 / 50 / 82 | 1029 / 2221 / 2058 |
| "geometric" | 142 / 254 / 426 | 39 / 46 / 117 | 826 / 1314 / 2478 |
| "none" | 313 / 1507 / 626 | **39 / 41 / 78** | max_iter (20000) |
| initial_step = 1 | 134 / 156 / 402 | 41 / 45 / 123 | 768 / 781 / 2304 |
| memory 3 | 162 / 186 / 486 | 41 / 44 / 123 | 1015 / 1067 / 3045 |
| memory 30 | 118 / 139 / 354 | 40 / 44 / 120 | 689 / 714 / 2067 |

- **"y" is the right default.** It is best or close everywhere.
- **"s" and "geometric"** cost more line-search trials: they overestimate
  the step.
- **"none"** (gamma = 1, B0 = M) fails with the Euclidean metric, which has
  no scale. With the H1 metric it is the best: H1 is scaled like this
  Hessian, and it saves a solve per iteration.
- **Memory:** the H1 runs do not care; the Euclidean runs gain 12% from
  30 pairs.

## Status

First round done.

- **Mesh independence.** The metric delivers it, as intended.
- **Against the Newton methods.** lbfgs is the cheapest method in
  derivative evaluations, and the most expensive in solves until the fixed
  path (phase 6).
- **Safeguards.** They stay quiet at their defaults.
- **Next.** Phase 3 (P as an accelerant), phase 4 (point-dependent metrics
  and manifolds), and the full option study (phase 8).
