# lbfgs: a constant metric against a point-dependent one

**Question** (after the phase-2 check-in): on one problem, how do these
compare?

- the fixed path for a constant metric (`fixed_metric=True`) against the
  general path;
- a constant metric against a point-dependent one.

This is a note on one example. Both kinds of metric are supported, and
other problems may well favor the other kind; see "What this does not
show".

**Run:** `python experiments/lbfgs_metrics/compare.py [--json results.json] [--N 31 63 127 255]`.
`results.json` holds the run below; everything is deterministic except the
times.

**Problem:** the finite-difference p-Laplacian energy (p = 4, eps = 1e-2;
convex, degenerate at the start u = 0) on the unit square, N x N interior
nodes, scipy sparse LU for every solve.

**Metrics:**
- **H1**, the discrete H^1 inner product: constant, one factorization per
  run.
- **Kacanov**, M(u) = Dx^T S(u) Dx + Dy^T S(u) Dy with
  S = h^2 (eps^2 + |grad u|^2)^((p-2)/2): the Hessian without its
  rank-structured (p - 2) part, so M(u) <= H(u) <= (p - 1) M(u) whatever
  the mesh and the degeneracy. It is a new matrix at every point, passed
  through operators_at with point_data, so each point is factorized once.

lbfgs runs with its defaults (memory 10, scaling "y") unless noted. TR and
LS are trust_region_newton and line_search_newton with M = P (CG). The
times include every factorization.

## The fixed path for a constant metric

| N | path | its | M^{-1} solves | M products | time |
|---|---|---|---|---|---|
| 127 | general | 49 | 147 | 49 | 1.00 s |
| 127 | fixed | 49 | **50** | **0** | **0.61 s** |
| 255 | general | 59 | 177 | 59 | 5.99 s |
| 255 | fixed | 59 | **60** | **0** | **3.61 s** |

- **The same runs.** The decisions and iterates are the same (the tests
  check this to rounding).
- **The cost.** A third of the solves, no products with M, about 60% of the
  time.
- **The price.** Two more blocks of memory (M^{-1} Y and M S).

## Iterations across meshes

| N | lbfgs H1 | lbfgs H1, m=1 | lbfgs Kacanov | Kacanov, m=3 | Kacanov, m=1 | TR H1 | TR Kacanov |
|---|---|---|---|---|---|---|---|
| 31 | 33 | 37 | 15 | 13 | 18 | 9 | 12 |
| 63 | 41 | 43 | 22 | 22 | 17 | 10 | 13 |
| 127 | 49 | 51 | 33 | 32 | 17 | 10 | 13 |
| 255 | 59 | 65 | 74 | 44 | 17 | 11 | 13 |

**With the constant H1 metric**, lbfgs grows slowly (about 8 to 10 more
iterations per refinement). Memory 1 costs a few iterations more than
memory 10.

**With the point-dependent Kacanov metric**, the result depends strongly on
the memory:

- **Memory 10 (the default).** It starts better than H1 (15 against 33
  iterations at N = 31) but grows faster, and is worse at N = 255 (74
  against 59). Unit steps become rare: 77% of steps at N = 63, 48% at
  N = 127, 9% at N = 255, with about two line-search trials per iteration.
  gamma stays at 0.3 to 0.5, consistent with H between M and 3 M, so the
  scale is right and it is the direction that degrades.
- **Memory 1.** It takes 17 iterations at every N from 63 to 255, with 88%
  unit steps. The newest pair with the new base works; older pairs, formed
  while the metric was different, hurt once they are combined with it.
  This is the risk the plan listed for strongly varying metrics. On this
  problem the weights S(u) change by orders of magnitude across the domain
  and from iterate to iterate.
- **The trust region with the Kacanov metric** takes 13 iterations at every
  N. Its inner CG uses the metric at one point only, so a changing metric
  costs it nothing in direction quality.

## Cost at N = 255

| method | its | grad | Hv | solves | factorizations | time |
|---|---|---|---|---|---|---|
| lbfgs H1 fixed | 59 | 60 | 0 | 60 | 1 | 3.61 s |
| lbfgs H1 fixed, m=1 | 65 | 66 | 0 | 66 | 1 | **2.99 s** |
| lbfgs H1 fixed, P = Kacanov, m=1 | 21 | 22 | 0 | 64 | 23 | 16.77 s |
| lbfgs Kacanov, m=1 | 17 | 18 | 0 | 51 | 18 | 14.08 s |
| lbfgs Kacanov | 74 | 75 | 0 | 222 | 75 | 50.55 s |
| TR H1 | 11 | 12 | 95 | 106 | 1 | 4.76 s |
| TR Kacanov | 13 | 10 | 23 | 24 | 10 | 8.73 s |

- **The fastest runs here.** With sparse LU, a factorization costs far more
  than a gradient or a solve. So the constant metric on the fixed path is
  the fastest lbfgs configuration, and it beats the trust region at the
  finest mesh: memory 1 is quicker per iteration (smaller blocks) despite
  more iterations.
- **The point-dependent metric** needs the fewest gradients by far (18 with
  memory 1, against 60 to 66), and loses only on factorizations.

## Sharing the setup at a point (operators_at, decision D9)

At N = 63, the Kacanov metric passed as keyword operators refactorizes at
every call:

- 66 factorizations for 22 iterations, 1.20 s;
- with operators_at and point_data, 23 factorizations, 0.63 s, the same
  iterates.

## A constant metric with a point-dependent preconditioner (phase 3)

M = H1 on the fixed path (the geometry: norms, angles, the M direction),
and P = M(u), the Kacanov operator, as the safeguarded accelerant (the same
point_data, so one factorization per point). Iterations, with the
iterations whose direction came from P in brackets:

| N | P Kacanov, m=10 | P Kacanov, m=1 | P Kacanov, scaling_P "none" | for reference: H1 alone | Kacanov as M, m=1 |
|---|---|---|---|---|---|
| 31 | 17 (17) | 20 (20) | 33 (33) | 33 | 18 |
| 63 | 19 (19) | 20 (20) | 51 (51) | 41 | 17 |
| 127 | 27 (27) | 21 (21) | 63 (63) | 49 | 17 |
| 255 | 36 (36) | 21 (21) | 75 (75) | 59 | 17 |

- **P's proposal is always taken.** It passes the angle test in H1 every
  time.
- **Memory.** With memory 1 the iterations stay at about 21; with memory 10
  they grow as before. P's base changes from point to point, and the older
  pairs conflict with it, just as with the Kacanov metric.
- **The scale fit matters.** Without it (gamma_P = 1) the runs are worse
  than H1 alone. The Kacanov operator lies below the Hessian (M(u) <= H <=
  3 M(u)), and the fit in the H1 norm finds the factor.
- **Cost.** Per iteration this costs one H1 solve (the gradient norm), two
  P solves (the base and the fit) and two products with H1. One
  factorization per point again dominates the time.

## What this does not show

The comparisons above come from one convex problem with smooth data, one
nonlinearity (p = 4) and one kind of solver (sparse LU).

- **The cost of a point-dependent metric depends on how it is solved.**
  With multigrid instead of LU, a new point costs a hierarchy setup or,
  with fixed smoothers on a new operator, nearly nothing. The balance
  between gradients (where Kacanov wins) and setups (where it loses) then
  shifts.
- **Problems where no constant metric is close to the Hessian** may need a
  point-dependent one: coefficients that change strongly with the
  solution, other nonlinearities, larger p, or a degeneracy that moves.
  This p-Laplacian happens to be served well by H1.
- **One data point on memory.** That memory 1 beats memory 10 for the
  Kacanov metric is one observation on one problem. It suggests a memory
  policy for changing metrics, not a default.

## To try

- **The preconditioner as accelerant:** done (section above). With
  memory 1 it gets the Kacanov iteration counts with the H1 geometry.
- **Memory policies for point-dependent metrics or preconditioners**
  (phase 8): clear or age
  out pairs when the metric changes by more than a threshold, measured for
  example on the newest pair through <y, M_new^{-1} y> / <y, M_old^{-1} y>,
  or keep only the pairs formed since the last large change.
- **The damped Kacanov iteration** (the memory cleared at every step), as
  a baseline for the memory-1 runs.
- **The same comparison with multigrid solves**, and on a problem where H1
  is a poor metric.

## Status

First round done (2026-10-26).
