# lbfgs: what do the memory and the transport add over the metric?

**Question.** With the Kacanov metric, memory 1 was best on the p-Laplacian.
Is lbfgs then doing anything beyond -M(x)^{-1} g? Do any of our test
problems get their good results from the low-rank memory and the
transport, rather than from M?

**Baseline.** `memory=0` keeps no pairs. Every direction is -gamma M^{-1} g,
with gamma fitted to the newest pair (<y, s> / <y, M^{-1} y>). This is
steepest descent in M with Barzilai-Borwein steps, run with the same line
search, tests and counts as lbfgs.

**Control for the transport.** On the two manifolds, the identity
"transport": stored vectors reused as they are, and y = g_new - g by
subtraction, as in textbook L-BFGS.

**Run:**
- `python experiments/lbfgs_memory/compare.py [--json results.json] [--only sphere ...]`
- `python experiments/lbfgs_memory/p_laplacian.py [--json p_laplacian.json]`

The JSON files hold the runs below, which are deterministic. compare.py's
results were produced in four parts with `--only` and merged.

## Iterations to converge

Each run allows 20000 iterations. `*` marks a run that did not converge;
`+` marks a completion run that ended at a worse point than the best fit,
f = 5.8707e-05.

| problem | m = 0 | 1 | 3 | 10 | 30 |
|---|---|---|---|---|---|
| convex quartic, n = 40, M = I | 94 | 86 | 79 | 75 | 60 |
| chained Rosenbrock, n = 20, M = I | 20000* | 240 | 162 | 128 | 118 |
| chained Rosenbrock, Jacobi M(x) | 1936 | 187 | 143 | 116 | 110 |
| p-Laplacian N = 63, M = I | 2127 | 1359 | 1015 | 787 | 689 |
| p-Laplacian N = 63, H1 | 48 | 43 | 41 | 41 | 40 |
| p-Laplacian N = 63, Kacanov M(u) | **14** | 17 | 22 | 22 | 24 |
| sphere, projection | 3463 | 2241 | 2265 | 1487 | 1551 |
| sphere, no transport | 4458 | 2672 | 2247 | 1630 | 1571 |
| completion from X = 0, projection | 1559 | 1039 | 20000*+ | 10274+ | 20000*+ |
| completion from X = 0, no transport | 20000*+ | 20000*+ | 1147 | 20000*+ | 20000*+ |
| completion, spectral start, projection | 18057 | 830 | 1269 | 2588 | 989 |
| completion, spectral start, no transport | 20000*+ | 20000*+ | 2898 | 20000*+ | 20000*+ |

**The problems.**
- Sphere: min x^T A x / 2 on the unit sphere, n = 300, with A's eigenvalues
  spaced geometrically from 1 to 1000. The eigengap is 0.023, so the
  Riemannian Hessian at the minimizer has condition number about 4 x 10^4.
- Completion: rank 5, a 400 x 300 matrix with singular values 2^-k, 4
  samples per degree of freedom, and the Frobenius metric in the gauged
  components. The spectral start is the rank-5 truncated SVD of
  (NM/K) P_Omega(A).

## The p-Laplacian across meshes and exponents

| N | H1: m = 0 | 1 | 3 | 10 | 30 | Kacanov: m = 0 | 1 | 3 | 10 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|
| 31 | 37 | 37 | 34 | 33 | 32 | 14 | 18 | 13 | 15 | 15 |
| 63 | 48 | 43 | 41 | 41 | 40 | 14 | 17 | 22 | 22 | 24 |
| 127 | 56 | 51 | 51 | 49 | 50 | 15 | 17 | 32 | 33 | 45 |
| 255 | 62 | 65 | 60 | 59 | 58 | 15 | 17 | 44 | 74 | 81 |

| p (N = 63) | H1: m = 0 | 1 | 3 | 10 | 30 | Kacanov: m = 0 | 1 | 3 | 10 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 28 | 28 | 25 | 26 | 26 | 11 | 12 | 11 | 11 | 11 |
| 4 | 48 | 43 | 41 | 41 | 40 | 14 | 17 | 22 | 22 | 24 |
| 6 | 77 | 73 | 71 | 66 | 66 | 16 | 16 | 19 | 24 | 34 |

With the Euclidean metric (N = 31, 63, 127), memory 0 takes 1153, 2127 and
7050 iterations, and memory 30 takes 319, 689 and 1580.

## Findings

**1. With the Kacanov metric, memory adds nothing; the metric does the
work.**
- Memory 0 is the best setting, or within one iteration of it, at every
  mesh (14 or 15 iterations from N = 31 to 255) and at every p.
- Memory 1 is close behind (12 to 18).
- From N = 63 on, more memory slows the run, increasingly on finer meshes
  (memory 30 takes 24 to 81 iterations).
- On this problem, memory 0 with this metric is the classical Kacanov
  iteration, with a Barzilai-Borwein step length and a line search. The
  gradient is M(u) u - load, so the unit step -M(u)^{-1} g lands exactly on
  the Kacanov iterate M(u)^{-1} load.
- The earlier mesh-independence result (17 iterations with memory 1) is the
  metric's doing.

**2. With the constant H1 metric, the memory helps a little.**
- Memory 10 to 30 saves 5 to 17% over memory 0 at every mesh and p (for
  example 48 to 40 at N = 63, and 77 to 66 at p = 6).
- Memory 0 already grows only from 37 to 62 iterations as N goes from 31 to
  255, so the near mesh-independence also comes from M.

**3. The memory matters where M is a poor model of the Hessian, including
a point-dependent M.**
- With M = I:
  - Rosenbrock fails at memory 0 and converges in 118 to 240 iterations
    with memory.
  - On the p-Laplacian, memory 30 cuts the iterations 3.1 to 4.5 times.
  - The convex quartic, already well conditioned, gains 1.6 times.
- With the diagonal Jacobi metric on Rosenbrock, which changes at every
  point but misses the valley's off-diagonal coupling, memory cuts 1936
  iterations to 110 to 187.
- So a point-dependent metric does not by itself make the memory useless.
  What matters is how much of the Hessian M leaves out.

**4. On the sphere, the memory helps and the transport helps slightly.**
- Memory 10 to 30 needs 2.2 to 2.3 times fewer iterations than memory 0.
- Reusing the vectors as they are takes 29%, 19% and 10% more iterations
  than the projection transport at memory 0, 1 and 10, and about the same
  at memory 3 and 30.
- At memory 0 the transport enters only through the gamma fit (y =
  g_new - T g).
- The projection's defects are second order in the step, and near the
  minimizer the steps are short, so the two transports converge together.

**5. In completion, the transport is essential.**
- From the spectral start with the projection transport, all five memories
  find the minimizer.
- Without the transport, four of five stop after 20000 iterations, at worse
  points. Only memory 3 gets through, and more slowly (2898 iterations
  against 1269).
- From X = 0 the picture is the same: without the transport, only memory 3
  converges.

**6. Completion also shows the memory hurting.**
- From X = 0 with the projection transport:
  - Memory 0 and 1 find the minimizer.
  - Memory 3, 10 and 30 end at f = 7.585e-05 instead of 5.871e-05, and
    converge slowly there or not at all.
  - In the memory-10 run, one of the five singular values grows to about
    2300 while the fit barely changes. A component that large must be
    nearly invisible to the samples, so the fit is effectively rank 4.
  - Presumably the quasi-Newton model sees almost no curvature along that
    component and extrapolates along it.
  - The completion example avoids this start by rank continuation.
- From the spectral start, memory 1 is fastest (830 iterations against 989
  to 2588 with more memory).

**7. The memory-0 baseline is weak on nonconvex problems.**
- When <y, s> <= 0 the pair fails the curvature test and gamma is kept.
- From the spectral start, the memory-0 run spends 17753 of its 18057
  iterations with <y, s> < 0. It creeps through a flat, slightly concave
  region (f about 7.2e-05, |g| about 5e-6 against 0.6 at the start) with
  gamma frozen at 0.5.
- Barzilai-Borwein codes that take a long step on negative curvature would
  escape sooner. So the factor of 18057 / 830 overstates what the memory
  gives there; memory 1 against more memory is the fairer comparison.

## Answer, so far

On these problems:
- **The memory earns its keep where M leaves much of the Hessian out:** the
  Euclidean runs, Rosenbrock with a diagonal point-dependent metric, and
  the ill-conditioned sphere.
- **Where M is a good model, the results are the metric's:**
  - H1 gains 5 to 17% from the memory.
  - Kacanov gains nothing: memory 0 is best.
  - The suspicion that the Kacanov runs lean on M is right.
- **The transport matters on a strongly curved manifold:** in completion,
  without it most runs fail. On the sphere it is a modest gain.
- **No problem here shows a good metric, memory and a nontrivial transport
  each pulling their weight together.** That would be the case to look for
  next.

## What this does not show

- **A few problems, one start each.**
  - Completion is nonconvex. There, which critical point a run reaches
    depends on the path, and a different seed or rank could reorder the
    memories.
  - The p-Laplacian is the only problem with a good metric.
- **Why more memory hurts with the Kacanov metric.** One candidate is
  stale curvature. From the degenerate start u = 0, the Hessian changes by
  orders of magnitude, and M(u) tracks it while older pairs record
  curvature from where the iterates were. Ways to test this:
  - periodic restarts;
  - a memory policy that drops pairs when M changes a lot;
  - a structured secant update (Dennis-Martinez-Tapia). That would model
    H = M(u) + R with the memory spent on R alone, using y - M(u_new) s in
    place of y (for the p-Laplacian, R is the rank-structured, positive
    semidefinite (p - 2) part).
- **Cost per iteration.** Memory 0 needs no two-loop recursion, but it
  still needs the M^{-1} solve, which dominates here. With the Kacanov
  metric, the fewest iterations are also the fewest factorizations.

**Status:** first round.
