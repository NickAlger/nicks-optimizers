# Plan: a line-search Newton-Krylov optimizer

October 2026. **Status: agreed (2026-10-25); phases 0 to 5 done, see
[Progress](#progress); phases 6 and 7 after the check-in.** Decisions D1 to
D10 were settled on review: all as recommended, with step_max an option the
user may set, off by default (D5).

## Goal

`line_search_newton`: the inexact Newton method of `trust_region_newton`,
globalized by a line search instead of a trust region. The same problem
pieces (fun, grad, hessvec or operators_at, the metric M, the optional
preconditioner P, spaces, retractions, point data, covector projection),
the same Krylov solvers (CG, Lanczos, flexible) and options (saddle-free,
convergence norm, forcing term), the same backends (numpy by default,
compiled by JAX or another framework, the line search included), the same
callback and exact resumption, and a display that shows line-search
diagnostics instead of trust-region ones.

Not in scope: Wolfe line searches, nonmonotone searches, curvilinear
searches along negative curvature (More-Sorensen), hybrids of trust region
and line search. Each could follow on top of this design.

## The method

At the iterate x_k, with f, g = grad f, the gradient norms and the point
data known (as in `trust_region_newton`):

1. **Forcing term** eta_k = min(forcing_max, sqrt(||g_k|| / ||g_0||)), the
   same function.
2. **Direction.** Solve H d = -g approximately over a Krylov space, until
   ||H d + g||_{M^{-1}} <= eta_k ||g||_{M^{-1}}, with one of the existing
   solvers in *line-search mode* (below). The direction is a descent
   direction: <g, d> < 0.
3. **Line search** along the retraction curve, phi(alpha) = f(retract(x, alpha
   d)), phi'(0) = <g, d>: backtracking from alpha_0 = min(1, step_max /
   ||d||_M) until the Armijo condition phi(alpha) <= phi(0) + c1 alpha
   phi'(0) holds (c1 = 1e-4), with safeguarded interpolation (D2).
4. **Accept** x_{k+1} = retract(x_k, alpha d): its gradient, point data and
   gradient norms (the same `state_at`). If the search fails within
   max_backtracks trials, the run stops with reason "line_search" (D4).

Stopping tests (gtol, rtol, max_iter, callback) are those of
`trust_region_newton`.

### Directions: the Krylov solvers in line-search mode

A trust-region solve minimizes the model in a ball; a line-search solve
minimizes it without a bound, so the solvers differ only where the radius
or indefiniteness enters. Negative curvature is handled in one of two ways
(D1):

- **Truncation** (default, Newton-CG semantics, Nocedal and Wright Alg.
  7.1): stop as soon as the Krylov space shows nonpositive curvature and
  return the last direction found with positive curvature; if that happens
  at the first step, return the preconditioned steepest descent direction
  d = -P^{-1} g. Every CG iterate and every Galerkin solution on a positive
  definite projected Hessian is a descent direction.
- **Saddle-free** (`saddle_free=True`, Lanczos and flexible): the model
  Hessian is |H|_M on the Krylov space, as in the trust-region solver, with
  the M-Ritz values floored at curvature_floor times the largest |theta|
  (D6), since nothing else bounds the step. Always a descent direction; the
  solve runs to the (surrogate) residual tolerance.

Per solver:

| solver | trust-region mode (today) | line-search mode (new) |
|---|---|---|
| CG (P = M) | stops on the boundary or follows negative curvature to it | no boundary test; stops at nonpositive curvature: s_k, or p_0 = -P^{-1} g at the first step |
| Lanczos | projected trust-region problem at each step (`solve_projected_trs`) | projected Newton system (`solve_projected_newton`): y = -gamma0 T^{-1} e_1 while T is positive definite, else the previous y (truncation); or |theta| floored (saddle-free) |
| flexible | the same, with T from the stored H Q | the same as Lanczos |

The residual is the trust-region one with lam = 0: for Lanczos it is
gamma_next y_k v_next, so with P = M its norm is free, and the stall test
(which needs lam > 0) never fires.

**No user operation is spent beyond what the direction needs** (D10). The
Galerkin solution T y = -gamma0 e_1 does not depend on M (Q^T g = gamma0 e_1),
and the sign of the M-Ritz values is the sign of T's eigenvalues (Sylvester),
so with P != M and truncation the M-Gram matrix, and with it one product
with M per Lanczos step, is not needed: Lanczos keeps it only for the
trust-region mode and for saddle-free. ||d||_M (for step_max and the
display) then costs one product with M per iteration; it comes free from
the coordinates otherwise (y^T G y; the CG recurrences; y^T y for
flexible). The fallback direction -P^{-1} g is the Krylov start vector,
already computed. Per Krylov step, line-search mode applies:

| solver | H | P^{-1} | M^{-1} | M |
|---|---|---|---|---|
| CG (P = M) | 1 | 1 (= M^{-1}) | - | - |
| Lanczos, P = M | 1 | 1 (= M^{-1}) | - | - |
| Lanczos, P != M, truncation | 1 | 1 | 1 (residual norm) | - (trust region: 1) |
| Lanczos, P != M, saddle-free | 1 | 1 | 1 | 1 (Gram matrix) |
| flexible | 1 | 1 | 1 | 1 (orthonormalization) |

Growing a basis repeats nothing, as now (capacity per solve, doubling).

### The line search

A pure function of the backend, reusable by other optimizers:

```python
backtracking(backend, phi, f0, slope, alpha0, *, c1=1e-4, max_trials=30,
             shrink=(0.1, 0.5)) -> LineSearchResult
```

phi(alpha) returns (f(retract(x, alpha d)), the point). Each trial costs one
evaluation of fun (and a retraction). The next trial is the minimizer of
the quadratic interpolating phi(0), phi'(0) and phi(alpha_j), then of the
cubic through the last two trials (Dennis and Schnabel, A6.3.1), clipped to
[0.1 alpha_j, 0.5 alpha_j]; a non-finite phi(alpha_j) gives 0.1 alpha_j.
Near convergence the decrease falls to the rounding level of f; as with the
trust region's rho, the test allows delta = 10 eps max(1, |f0|) (D3), so a
unit Newton step that changes f only by rounding is accepted.

It is a bounded loop (`backend.loop`) whose carry holds alpha, phi(alpha),
the trial point, the two last trials for the interpolation, the status and
per-trial traces (alpha and phi, for the display), with `where` instead of
branches: so it compiles, and with a loop built from scan its skipped trials
evaluate nothing. Results: alpha, the accepted point and f there, the
number of trials, the status (ARMIJO, FAILED, NOT_DESCENT) and the traces.

If <g, d> is not negative (rounding, or a non-finite direction), the
direction is replaced by -P^{-1} g under `cond`, which costs a solve only
when it happens and only if the start vector was not computed.

### One iteration, in pieces

The trust-region pieces, with step_finish replaced:

```
init(x0, g0)                       -> OuterState          (shared, unchanged)
step_start_c(state, gnorm0)        -> carry               the direction solve, begin + run
step_grow(state, carry, c2)        -> carry               (shared)
step_continue(state, carry)        -> carry               (shared)
line_search_finish(state, carry)   -> (OuterState, RawLineSearchIteration)
                                      the direction, slope and ||d||_M, the
                                      line search, cond(success: gradient,
                                      point data, norms at the new point)
```

The driver keeps the stopping tests, records, display, callback and
resuming, as for the trust region, and shares the code that compiles the
pieces and grows the basis.

## Diagnostics and display

`LineSearchRecord`, one per iteration:

| field | meaning |
|---|---|
| iteration, f, gnorm, gnorm_ratio, forcing | as in IterationRecord |
| direction | "newton" (the solve met its tolerance or the step limit), "truncated" (negative curvature after k > 0 steps), "steepest" (negative curvature at the first step, or not a descent direction) |
| step_norm | \|\|d\|\|_M |
| slope | <g, d> |
| alpha, trials | the step length taken and the evaluations of fun it cost |
| predicted, actual | decrease of the quadratic model along alpha d, and f(x_k) - f(x_{k+1}) |
| accepted | whether the line search succeeded |
| elapsed, subproblem | as in IterationRecord (SubproblemStats, lam = 0) |
| trace_alpha, trace_f | the trials of the line search |

The display (level 1):

```
  it              f       |g|  |g|/|g0|      eta    k   stop  neg  direction      |d|     slope    alpha  ls  act/pred
   0   2.420000e+01   2.3e+02   1.0e+00  5.0e-01    1    tol    0     newton  1.2e+00  -4.5e+01  2.3e-01   3      0.98
   1   1.913200e+01   9.1e+01   4.0e-01  5.0e-01    2    tol    0     newton  8.7e-01  -2.1e+01  1.0e+00   1      1.02
```

Level 2 adds the Krylov trace (as now, without the lam column) and the line
search trials:

```
        trial      alpha             f  (f - f0)/(alpha slope)
            1   1.00e+00   2.51234e+01   -0.02
            2   2.31e-01   1.91320e+01    0.49  accepted
```

`LineSearchResume` (gnorm0, next_iteration, f: no radius), `LineSearchSummary`
(reason in "converged", "max_iter", "callback", "line_search"; the work,
with line-search trials in fun_calls), `LineSearchInfo` for the callback,
and `format_*`, `display_*`, `history_array`, `merge_summaries` versions for
them (D7).

## Reuse

| component | reuse |
|---|---|
| backend, spaces, metrics, trees, krylov processes, dense (pencil_ritz) | unchanged |
| criteria (gradient_converged, forcing_term) | unchanged |
| SubproblemStats, RawSubproblem, status codes, KrylovSolver, solve_with, capacity growth | moved to a shared module, unchanged |
| CG, Lanczos, flexible solvers | a static `trust_region` flag (D8): radius and boundary logic, the projected trust-region problem and the Gram matrix in trust-region mode; truncation, the projected Newton system and the fallback direction in line-search mode |
| OuterState, init, state_at, operators, gradient_norms, canonical, subproblem_at, step_start/grow/continue | moved to a shared core, unchanged |
| the driver's piece compilation, capacity growth loop, stopping tests, callback and resume handling | shared helpers |
| records, display, summary | new types for the line search; SubproblemStats and the Krylov trace shared |
| new | `solve_projected_newton` (dense), `backtracking` (the line search), `line_search_finish`, `line_search_newton`, the line-search records and display |

Proposed modules (D8): `newton_core` (the shared outer pieces),
`krylov_solvers` (the shared solver machinery and the three solvers, with
`trust_region_subproblem` keeping `solve_tr_subproblem*` and re-exporting),
`line_search` (`backtracking`, reusable), `newton_line_search` (driver,
pieces, records and display, or split as the trust-region modules are).
Public: `line_search_newton`, `backtracking`, `solve_newton_direction`,
`solve_newton_direction_cg`, `solve_newton_direction_flexible` (standalone
direction solves, like `solve_tr_subproblem*`), the record types and
formatting functions.

## API

```python
line_search_newton(fun, grad, hessvec, x0, *, space=None, retract=None,
                   metric_solve=None, metric_apply=None, precond_solve=None,
                   operators_at=None, point_data=None, project_covector=None,
                   saddle_free=False, krylov="auto", convergence_norm="metric",
                   gtol=0.0, rtol=1e-8, gnorm0=None, forcing_max=0.5,
                   max_iter=500, max_krylov=50, c1=1e-4, max_backtracks=30,
                   step_max=None, curvature_floor=1e-6, display=0, print_fn=print,
                   callback=None, resume=None, g0=None, backend=None,
                   krylov_capacity=None) -> (x, LineSearchSummary)
```

The arguments of `trust_region_newton` without radius, radius_max,
radius_min, eta and stall_tol; new: c1, max_backtracks, step_max,
curvature_floor.

## Decisions

Settled on 2026-10-25: all as recommended. D5: step_max is an option the
user may set; no cap by default. D8: module names as proposed. D10: the
M^{-1} norm, one M^{-1} solve per Lanczos step, as for the trust region.

- **D1. Negative curvature.** (a) Truncation by default, saddle-free on
  request (Lanczos and flexible). (b) Saddle-free by default whenever
  Lanczos runs. *Recommendation: (a)*: it is the standard Newton-CG
  behavior, the same for all three solvers, and needs no curvature floor;
  saddle-free remains one flag away, as for the trust region. **Agreed.**
- **D2. Line search.** (a) Armijo backtracking with safeguarded
  quadratic/cubic interpolation, alpha_0 = 1. (b) Strong Wolfe: a gradient
  at every trial, which Newton steps rarely need. (c) Nonmonotone (Grippo
  et al., Zhang-Hager). *Recommendation: (a)*; (b) and (c) can be added as
  options of the same loop later. **Agreed.**
- **D3. Rounding allowance in the Armijo test** (10 eps max(1, |f|), like
  rho). *Recommendation: yes.* Without it, runs near convergence can fail
  the line search on noise. **Agreed.**
- **D4. When the search fails.** (a) Stop with reason "line_search", x
  unchanged. (b) Retry once along -P^{-1} g, then stop. *Recommendation:
  (a)*: failures come from inaccurate gradients or rounding, where a retry
  rarely helps, and an honest stop is easier to diagnose; (b) is a small
  addition if wanted. **Agreed.**
- **D5. step_max**, a cap on ||alpha d||_M. *Recommendation: default None*
  (no cap), documented as the remedy for nearly singular Hessians and
  steepest-descent fallbacks on badly scaled problems. **Agreed**: an option
  the user may set, off by default.
- **D6. Saddle-free floor**: |theta| floored at curvature_floor * max |theta|.
  *Recommendation: relative, default 1e-6.* An absolute floor would depend
  on the scale of f. **Agreed.**
- **D7. Records.** (a) New types for the line search (LineSearchRecord,
  LineSearchResume, LineSearchSummary), sharing SubproblemStats. (b) A
  generic RunSummary[RecordT, ResumeT] for both methods. *Recommendation:
  (a)*: the trust-region types stay as they are, and each method's records
  say exactly what it does; the shared helpers stay internal. **Agreed.**
- **D8. Code organization.** (a) One set of Krylov solvers with a static
  trust_region flag, shared modules extracted from the trust-region ones.
  (b) Separate line-search solvers. *Recommendation: (a)*: the solvers
  differ in a few lines (the boundary test, the projected solve, the Gram
  matrix), and one copy keeps fixes and capacity growth in one place.
  Module names as proposed above, or yours. **Agreed, with the proposed
  names.**
- **D9. Safety net for the refactoring.** The shared-module extraction must
  not change `trust_region_newton`. *Recommendation:* vendor the current
  package again (as phase 0 of the backend plan did) and compare 40-odd
  trust-region configurations bit for bit until the line search is done;
  then delete the copy. **Agreed.**
- **D10. The inner residual norm with P != M.** (a) ||R||_{M^{-1}}, like
  the trust region: one M^{-1} solve per Lanczos step. (b) ||R||_{P^{-1}},
  free from the Lanczos relation, but then the forcing test is not in the
  convergence norm. *Recommendation: (a)* for consistency, (b) as a later
  option if the solve matters. In both, truncating Lanczos keeps no Gram
  matrix (no product with M per step). **Agreed: (a).**

## Testing

- **Directions**, against dense references: the Galerkin Newton solution on
  positive definite problems (equal to CG's iterates in exact arithmetic
  when P = M), truncation at the first nonpositive curvature with the right
  fallback, saddle-free equal to the |H|_M solution on the Krylov space with
  the floor, descent in every case; capacities 1, 2, 3 and the bound give
  the same directions; counted operator calls match the table above.
- **The line search** on one-dimensional functions: unit steps accepted when
  Armijo holds, interpolation steps within their safeguards, non-finite
  values, failure after max_trials, the rounding allowance, the traces; on
  the strict backend.
- **The optimizer**: a convex quadratic converges with one unit Newton step
  per iteration (exact solves); Rosenbrock and the double well, with
  truncation and saddle-free; P = M and separate P; tuple trees, the
  distributed numpy space, the matrix-completion manifold; resumption
  exact; display text; callback; operator calls instrumented (none
  repeated, as for the trust region).
- **Backends**: the strict backend runs every configuration with tiny
  capacities; the JAX tests (job "jax") compare compiled runs with numpy,
  trace counts, no host callbacks.
- **Trust region unchanged** during the refactoring (D9).

## Phases

Each phase ends with the test suite, mypy and the CI examples passing, and
is committed separately.

| | Phase | Delivers | Gate |
|---|---|---|---|
| 0 | Safety net | the current package vendored, the trust-region comparison harness (restored from the backend plan) | harness passes |
| 1 | Shared modules | `newton_core` and `krylov_solvers` extracted; shared driver helpers; no behavior change | bit for bit |
| 2 | Projected Newton | `solve_projected_newton` in `dense` (truncation, saddle-free floor, padding for static shapes) | dense references; padding invariance |
| 3 | Directions | line-search mode of CG, Lanczos and flexible; `solve_newton_direction*` | references; capacities; strict backend; call counts; trust region bit for bit |
| 4 | Line search | `backtracking`, compiled-ready | unit tests; strict backend |
| 5 | Optimizer | `line_search_finish`, `line_search_newton`, records, display, resume, callback | convergence tests; resume; display; call counts; **check in** with a comparison of the two methods on the tests' problems and the p-Laplacian |
| 6 | Compiled | JAX tests and the example extended to the line search | job "jax" |
| 7 | Wrap-up | README section, an example comparing the methods, TODO; vendored copy deleted | CI |

## Risks

- **Nonconvex problems.** A backtracking line search never lengthens a step,
  so along negative curvature it is cautious, and truncated Newton can stall
  near saddle points where the trust region escapes. Saddle-free helps; the
  README will say that the trust region remains the robust choice for
  nonconvex problems and the line search suits convex or nearly convex ones
  (and problems where function evaluations are much cheaper than Krylov
  solves, since a rejected trust-region step costs a solve).
- **Badly scaled steepest-descent fallbacks** may cost many trials; the
  interpolation cuts alpha by up to 10 per trial, and step_max caps the
  first one.
- **Records and display** grow a second family of types; keeping their
  shared parts (SubproblemStats, the Krylov trace) common limits it.

## Progress

- **Phase 0** (24ce709): the package vendored as
  `tests/nicks_optimizers_baseline/`, and `tests/test_baseline.py` (the
  backend refactor's harness, restored) comparing 42 trust-region
  configurations and the subproblem solvers bit for bit. It held through
  every phase below.
- **Phase 1** (4dc0513): `krylov_solvers` (the solver machinery and the three
  solvers) out of `trust_region_subproblem`, which keeps the
  `solve_tr_subproblem*` functions and re-exports; `newton_core` (the shared
  outer pieces) out of `trust_region_core`, which keeps rho, the radius update
  and `step_finish` (eta and radius_max in `TrustRegionOptions`). Addition: a
  module `newton_driver` for the shared driver helpers (argument checks, the
  CoreProblem, the compiled step-solve pieces with basis growth, the resume
  check).
- **Phase 2** (823bb1b): `dense.solve_projected_newton`, exported.
- **Phase 3** (a709ece): the line-search mode (`trust_region=False`) of the
  three solvers, the Gram matrix kept only when needed, and
  `solve_newton_direction*`. Deviation: the flexible solver falls back to
  -M^{-1} g, its first basis vector (P may be nonsymmetric, so -P^{-1} g need
  not descend). With a separate P and truncation, theta are the eigenvalues
  of T (same signs as the M-Ritz values). The truncating step measures no
  residual (its M^{-1} solve is skipped under cond).
- **Phase 4** (29b017d): `line_search.backtracking` (quadratic then cubic
  interpolation, the rounding allowance, traces), exported.
- **Phase 5** (185d930): `line_search_finish`, `line_search_newton`, the
  records and display. Deviation: the line-search modules are
  `newton_line_search` (the driver and the direction functions),
  `newton_line_search_core` (the pure piece) and
  `newton_line_search_diagnostics` (records and text, functions named
  `format_ls_*`); a fourth direction kind, "fallback", marks the
  replacement of a non-descent direction. Check-in comparison in
  `experiments/line_search_vs_trust_region/` (6846513): the line search wins
  on the convex p-Laplacian with a separate preconditioner (10 against 18
  iterations, half the Hessian products, 10 products with M instead of 154)
  or a weak one (up to 4.5 times fewer iterations), the trust region on
  Rosenbrock's curved valley (half the Hessian products), close elsewhere.
