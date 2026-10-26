# Plan: metric-aware L-BFGS

October 2026. **Status: proposed**, decisions D1 to D11 open.

## Goal

`lbfgs`: a limited-memory BFGS method that works in the geometry of the
metric M (and the preconditioner P), on flat spaces and manifolds, with the
problem pieces of the Newton methods (fun, grad, space, retract, metric_solve,
metric_apply, precond_solve, operators_at with point data, project_covector),
their backends (numpy by default, compiled by JAX or another framework, the
whole iteration in one compiled piece), callback, resumption and a display of
its own. No Hessian is needed. New ingredient: a **vector transport** supplied
by the user, which carries the memory from one tangent space to the next.

Not in scope: Wolfe and extrapolating line searches (deferred, as agreed;
the design leaves room for them), bounds (L-BFGS-B), trust-region L-BFGS
(it needs products with B, the compact representation), limited-memory SR1,
and L-BFGS as a preconditioner for the Newton-Krylov solvers. Each could
follow on top of this design.

## Where the metric enters

Typed, the objects of L-BFGS are:

| object | type | where |
|---|---|---|
| x_k | Point | |
| g_k = Df(x_k) | Covector | T*_{x_k} |
| d_k, s_k = alpha_k d_k | Vector | T_{x_k} |
| y_k = g_{k+1} - (g_k transported) | Covector | T*_{x_{k+1}} |
| B_k, the Hessian approximation | Vector -> Covector | |
| H_k = B_k^{-1} | Covector -> Vector | |
| H0_k = gamma_k P_k^{-1} | Covector -> Vector | |
| S, the stored steps | VectorBlock | T_{x_k} |
| Y, the stored gradient changes | CovectorBlock | T*_{x_k} |
| rho_i, cross pairings, gamma | Coords | numbers |
| T_{x -> x'} | Vector at x -> Vector at x' | |
| Tb_{x -> x'} | Covector at x -> Covector at x' | |

Written this way, textbook L-BFGS has its Euclidean assumptions in a few
specific places, and each one has a typed replacement:

| step | textbook (R^n) | metric-aware | operators |
|---|---|---|---|
| direction | d = -H g, H0 = gamma I | d = -H g, H0 = gamma P^{-1} (P = M by default) | P^{-1}: the only covector-to-vector map |
| scaling | gamma = s^T y / y^T y | gamma = <y, s> / <y, P^{-1} y> | P^{-1} |
| update | s^T q, y^T r, q - a y, r + c s | <q, s_i>, <y_i, r>, covector and vector combinations | none: pairings only |
| new pair | s = x_{k+1} - x_k, y = g_{k+1} - g_k | s~ = T(alpha d), y = g_{k+1} - Tb(g_k) | transports |
| old pairs | reused as they are | S <- T(S), Y <- Tb(Y) | transports |
| curvature test | s^T y > 0 | <y, s~> > 0, relative (D6) | none |
| stopping | \|\|g\|\|_2 | \|\|g\|\|_{M^{-1}} | M^{-1} |
| line search | slope g^T d | slope <g, d>; \|\|alpha d\|\|_M only for step_max | M, only for lengths |

The update itself needs no metric. It is
H+ = V* H V + rho s (x) s with V = I - rho y (x) s, and in typed form every
term is a pairing. Its variational characterization (Nocedal and Wright,
section 6.1) is also intrinsic: H+ is the symmetric operator closest to H
with H+ y = s in the weighted norm ||E||_W^2 = tr(W E W E), for any W with
W s = y. E maps covectors to vectors and W maps vectors to covectors, so the
trace is defined without coordinates. BFGS is therefore affine covariant:
under any linear change of variables it maps to itself.

Consequences:

- **Only a few places change.** The base H0, the scaling, the transport
  and the norms. Converting between vectors and covectors happens exactly
  once per iteration, P^{-1} applied to one covector in the two-loop
  recursion, and never the other way (M is applied only to measure
  ||d||_M, D9).
- **A free reference test.** With a constant metric M = L L^T and
  H0 = M^{-1}, the iterates are exactly those of textbook L-BFGS on
  f(L^{-T} z), mapped back by x = L^{-T} z. Checked numerically for the
  operator, to 1e-15. The method generalizes textbook L-BFGS by a change of
  variables while M is constant, and goes beyond it when M varies or the
  space is curved.

## Transports, and what they are for

**Flat spaces: the identity, even when M varies.** The stored pairs are
data about the Hessian: for a quadratic, y = A s exactly, whatever M is. A
transport that tried to follow the metric would break this. For example, a
transport isometric from M_x to M_x' = c^2 M_x scales s by 1/c and, to keep
pairings, y by c, so the pair no longer satisfies B s = y.

How the operator adapts to a new metric: each iteration rebuilds H from the
newest base gamma_k P_k^{-1} and the stored pairs. The newest secant
equation H y_k = s_k holds exactly whatever the base, and the older pairs
pin H approximately along their directions. So when P changes from point to
point, the stored pairs keep H in place along the directions they have
seen; the new P shapes H elsewhere, and gamma sets the scale there. This
happens by construction. The transport's job is only geometric: moving
vectors between tangent spaces.

**The Riesz convention mixes the metric in.** Riemannian L-BFGS (Huang,
Gallivan and Absil 2015; Ring and Wirth 2012) represents y as a vector
through the Riesz map:

    y = grad f(x') - T grad f(x) = M'^{-1} g' - T M^{-1} g.

As a covector this is the transport Tb = M' T M^{-1}. With a point-dependent
metric on a flat space and T = I, that mixes the change of metric into y
(for a quadratic, y is no longer A s). Keeping y a covector and transporting
it as one avoids this, and saves the 2(m + 1) metric operations per
iteration that Riesz-converting the memory would cost (D1).

**Curved spaces.** The user supplies T and Tb: for an embedded manifold,
typically the projection onto the new tangent space, with the same formula
on canonical covector representatives. Pairing-preserving pairs,
<Tb w, T v> = <w, v>, are the ideal; an M-isometric T with
Tb = M' T M^{-1} is one example. Projections preserve pairings only
approximately, and the memory below is built to tolerate that.

The transport is a record of functions, given separately like the retraction
(the point structure):

```python
@dataclass(frozen=True, kw_only=True)
class VectorTransport(Generic[PointT, VectorT, CovectorT, VectorBlockT, CovectorBlockT]):
    vector: Callable[[PointT, PointT, VectorT], VectorT]          # T_{x -> x'} v
    covector: Callable[[PointT, PointT, CovectorT], CovectorT]    # Tb_{x -> x'} w
    vector_block: Optional[Callable[[PointT, PointT, VectorBlockT], VectorBlockT]] = None
    covector_block: Optional[Callable[[PointT, PointT, CovectorBlockT], CovectorBlockT]] = None

identity_transport     # flat spaces: nothing is called, nothing is copied
```

- **Single vectors.** The memory is transported member by member, through
  new space entries for reading block members (D4).
- **Block forms** are optional: one call per block, in the space's block
  representation (e.g. one batched projection of an array_space block),
  for when batching pays.
- **Requirements.** Both maps must be linear, and the covector map must
  return canonical covectors when project_covector is in use.

## The memory, and the two-loop recursion in coordinates

**The memory** holds up to m pairs in circular slots of two blocks with
capacity m:

- S, a VectorBlock, and Y, a CovectorBlock;
- rho_i = 1 / <y_i, s_i> and the cross pairings U_ij = <y_j, s_i> (y_j newer
  than s_i), both computed when a pair is stored, at the point where it is
  formed;
- gamma, the count, and the newest slot.

The active slots are always a prefix until the memory is full; clearing it
sets the count to zero.

**The direction.** In logical order (oldest first), with
R = diag(1/rho) + U upper triangular:

```
c  = S^T g               one block pairing             (<g, s_i>)_i
a  = R^{-1} c            m x m triangular solve
q  = g - Y a             one block combination of covectors
r0 = gamma P^{-1} q      the one covector-to-vector map
e  = Y^T r0              one block pairing of covectors with a vector (new space entry, D4)
b  = R^{-T} (e + U^T a)  m x m triangular solve
d  = -(r0 + S (a - b))   one block combination of vectors
```

This is the textbook two-loop recursion (Nocedal and Wright, Algorithm 7.4)
with its 2m sequential pairings replaced by two batched ones, using
pairings between the y_j and s_i stored when they met. It is the
Byrd-Nocedal-Schnabel compact form without the m applications of H0 to Y.
What it computes, for any stored rho > 0 and any U:

    H = W* H0 W + S R^{-T} D R^{-1} S^T,   W = I - Y R^{-1} S^T,  D = diag(1/rho).

- **Exact when the stored pairings are current.** Always with the identity
  transport, and with any pairing-preserving one.
- **Safe when they are not.** H is still symmetric and positive definite,
  and the newest secant equation H y_newest = s_newest holds exactly,
  because the newest pair is formed at the current point.

So no transport, however inexact, can make the direction ascend or the
operator indefinite. All three properties were checked numerically: 2000
random trials with transports that do not preserve pairings and arbitrary
stale U and positive rho (D3).

**Storing a pair** (s~, y) at x_{k+1}: write s~ and y into the next slot
(the oldest when full); one block pairing of S with y gives the row
(<y, s_i>)_i, which holds the new cross pairings and 1/rho; zero the stale
column of the overwritten slot.

## One iteration

At x_k, with f, the canonical gradient g, its norm for the stopping test and
the memory at x_k:

1. **Direction** d = -H_k g by the two-loop recursion. With an empty memory,
   that is -gamma P^{-1} g, which reuses the gradient norm's M^{-1} g when
   P = M.
2. **Slope** <g, d> < 0. Otherwise (rounding or non-finite values), use the
   fallback -gamma P^{-1} g and clear the memory (D7). If M can be
   applied, also compute ||d||_M (D9).
3. **Line search.** Backtracking along retract(x, alpha d) from alpha = 1,
   capped by step_max (line_search.backtracking, unchanged).
4. **If accepted:** x_{k+1}, the canonical g_{k+1} and its norm.
   - Transport: s~ = T(alpha d), Tb(g_k), and the memory (S, Y).
   - Form y = g_{k+1} - Tb(g_k), canonical.
   - If <y, s~> > curvature_tol alpha |<g_k, d>| (D6): store the pair, and
     set gamma = <y, s~> / <y, P^{-1} y> (D5).
   - Otherwise skip the pair; the memory is still transported.
5. **If the search fails:** with a nonempty memory, clear it and continue
   from the next iteration (a restart); with an empty memory, stop with
   reason "line_search" (D7).

Steps 1 to 4 form one pure piece of fixed shapes: the capacity is m, and
nothing grows. The driver keeps the stopping tests, restarts, records,
display, callback and resumption.

```
init(x0, g0)                -> LbfgsState      x, f, g, point data, norms, empty memory
lbfgs_step(state, gnorm0)   -> (LbfgsState, RawLbfgsIteration)
clear(state)                -> LbfgsState      the count set to zero (Python, no compiling)
```

### Work per iteration

| | fun | grad | M^{-1} | P^{-1} | M | transports | block operations |
|---|---|---|---|---|---|---|---|
| P = M | trials | 1 | 3: norm, two-loop, scaling | - | 1 if given (D9) | m+1 vectors, m+1 covectors (none for the identity) | 3 pairings, 2 combinations |
| separate P | trials | 1 | 1: norm | 2: two-loop, scaling | 1 if given | the same | the same |
| P = M, fixed (D10) | trials | 1 | 1: norm | - | 1 if given | none | 3 pairings, 3 combinations |

- **Reductions** (pairings that are global in a distributed run): about six
  per iteration. scaling=False saves one solve.
- **Against the Newton methods:** no Hessian products and no Krylov
  solves. Each Newton iteration spends 1 + k operator applications for k
  Krylov steps.

With P = M, the honest price of a point-dependent metric is three M^{-1}
solves per iteration. When P and M are fixed and the transport is the
identity, the fixed path (D10) stores Z = P^{-1} Y and takes P^{-1} y_k as
P^{-1} g_{k+1} - P^{-1} g_k from the gradient-norm solves, so that
r0 = gamma (P^{-1} g - Z a). That is one solve per iteration, with the same
iterates.

## Typing

The memory, the two-loop recursion, the transport and the iteration are
written generically, in type variables for points, vectors, covectors and
the two block types, with no Any. Under mypy --strict every operation is
then checked against its kind: storing y in the vector block, applying T to
a covector, or combining Y with vector coefficients is a type error. The
tests use the NewTypes Point, Vector, Covector, VectorBlock and
CovectorBlock. A small set of deliberately mistyped snippets, kept outside
mypy's file list, must each fail mypy with the expected error; a test runs
them through mypy.api, and is skipped where mypy is not installed. That
pins the typing down: a refactor that turned the core back to Any would
fail it.

## Diagnostics and display

`LbfgsRecord`, one per iteration:

| field | meaning |
|---|---|
| iteration, f, gnorm, gnorm_ratio | as in the other records |
| direction | "quasi-newton" (from the memory), "steepest" (empty memory: -gamma P^{-1} g), "fallback" (the memory's direction was not a descent direction) |
| pairs | pairs in the memory used for the direction |
| gamma | the scaling of H0 |
| step_norm | \|\|d\|\|_M, or nan when M is not available |
| slope, alpha, trials, accepted | as in LineSearchRecord |
| curvature | <y, s~> / (alpha \|<g_k, d>\|): in a flat space 1 - phi'(alpha)/phi'(0), so 1 when the step reached the one-dimensional minimizer, below 1 when short, above when past it |
| update | "stored", "skipped" or "-" (search failed) |
| restart | whether the memory was cleared after this iteration |
| elapsed, trace_alpha, trace_f | as in LineSearchRecord |

```
  it              f       |g|  |g|/|g0|  mem     direction       |d|      slope     alpha  ls     gamma   curv  update
   0   2.420000e+01   2.3e+02   1.0e+00    0      steepest   2.3e+02  -5.3e+04   1.2e-02   3   1.0e+00   0.97  stored
   1   1.913200e+01   9.1e+01   4.0e-01    1  quasi-newton   8.7e-01  -2.1e+01   1.0e+00   1   4.1e-03   1.02  stored
```

Level 2 adds the line-search trials (format_ls_trials, shared).
`LbfgsSummary` has the reason (converged, max_iter, callback, line_search),
the work (calls of every operator, transports, trials), the restarts and
the skipped pairs. `LbfgsInfo` is what the callback receives. `LbfgsResume`
holds gnorm0, next_iteration, f, gamma and the memory, so that resuming is
exact (D8).

## Reuse

| component | reuse |
|---|---|
| backend, metrics, trees | unchanged |
| spaces | three new entries (D4), in every constructor and in check_space |
| line_search (backtracking, statuses), format_ls_trials | unchanged |
| newton_core | canonical; the operators at a point and the gradient norms, with PointOperators.hessvec optional (D9) |
| newton_driver | number, check_resumed_f; the shared argument checks split from the Newton-specific ones |
| criteria (gradient_converged) | unchanged |
| new | transport (VectorTransport, identity_transport), lbfgs_memory (memory, store, clear, transport, two-loop: generic, pure), lbfgs_core (state, init, lbfgs_step), lbfgs_diagnostics (records, display), quasi_newton (the driver, lbfgs) |

## API

```python
lbfgs(fun, grad, x0, *, space=None, retract=None, transport=None,
      metric_solve=None, metric_apply=None, precond_solve=None,
      operators_at=None, point_data=None, project_covector=None,
      memory=10, scaling=True, convergence_norm="metric",
      gtol=0.0, rtol=1e-8, gnorm0=None, max_iter=1000,
      c1=1e-4, max_backtracks=30, step_max=None, curvature_tol=1e-8,
      display=0, print_fn=print, callback=None, resume=None, g0=None,
      backend=None) -> (x, LbfgsSummary)
```

- **Defaults.** As for the Newton methods: without a space, numpy arrays,
  M = I, x + s and the identity transport; with a space, metric_solve,
  retract and transport are required.
- **The rest:** memory is m; scaling turns gamma on (D5); curvature_tol
  sets the test for storing a pair (D6).

## Decisions

- **D1. Typed memory.** (a) Steps s_i as vectors and gradient changes y_i
  as covectors, with H0 = gamma P^{-1} as the only covector-to-vector map.
  (b) The Riemannian convention: y_i as vectors through the Riesz map.
  *Recommendation: (a)*. It is exact for quadratics under any metric, it
  needs no metric operations for the memory, and it is what the typing
  says.
- **D2. Transports.** (a) A user-supplied `VectorTransport` with a vector
  map and a covector map (block forms optional), and `identity_transport`
  for flat spaces. The identity is the default without a space and the
  transport is required with one, like retract. (b) Only a vector
  transport, with Tb = M' T M^{-1} derived through the metric: 2(m + 1)
  metric operations per iteration, and pairing-preserving only if T is
  isometric. (c) The identity as the default everywhere. *Recommendation:
  (a)*. A forgotten transport on a manifold would silently put the memory
  in the wrong tangent space.
- **D3. The two-loop recursion.** (a) In coordinates, with rho and the
  cross pairings stored when a pair is formed: three batched pairings per
  iteration; the operator is always symmetric positive definite and
  satisfies the newest secant equation exactly. (b) Sequential over
  members: always current pairings, but 2m sequential reductions. (c)
  Coordinates with all pairings recomputed after each transport: the
  textbook operator under any transport, but m more block pairings per
  iteration, and a recomputed rho can turn negative. *Recommendation: (a)*,
  with (c) as a possible later option.
- **D4. New VectorSpace entries.** `pair_covector_block(C, v, k)` (the
  pairings <c_i, v>, for e = Y^T r0) and `vector_member(B, j)`,
  `covector_member(C, j)` (for member-by-member transports; for numpy a
  view). (a) Required, implemented by every constructor and tested by
  check_space; nothing outside spaces.py builds a VectorSpace directly.
  (b) Optional, checked when lbfgs runs. *Recommendation: (a)*.
- **D5. Initial Hessian and scaling.** B0 = P (M when P = M) with
  gamma_0 = 1: the first step is -P^{-1} g. That is a full step when P is
  scaled like the Hessian; otherwise the line search finds the scale in a
  few trials, the first stored pair fixes gamma, and step_max can cap the
  step. Then gamma = <y, s~> / <y, P^{-1} y> after each stored pair
  (scaling=True by default), at one P^{-1} solve per iteration;
  scaling=False keeps gamma = 1 and saves that solve when P already tracks
  the Hessian's magnitude. Alternatives: no scaling by default, or the
  textbook first step of unit length. *Recommendation: as stated.*
- **D6. Line search and the curvature test.** Reuse backtracking unchanged
  (Armijo from alpha = 1, step_max off by default), with Wolfe deferred.
  Store a pair only if <y, s~> > curvature_tol alpha |<g_k, d>|, with
  curvature_tol = 1e-8. The test uses pairings only, so it is invariant
  under scaling f and under changes of variables; the ratio is the
  "curvature" column, which a Wolfe search would keep above 1 - c2.
  Alternatives: plain <y, s> > 0; Dennis-Schnabel's
  sqrt(eps) ||s|| ||y||, which needs an M product and an M^{-1} solve; the
  cautious update of Li and Fukushima. *Recommendation: as stated*, with
  skip rates measured at the check-in.
- **D7. Failures.** (a) If the search fails with a nonempty memory, clear
  the memory and continue from -gamma P^{-1} g (a restart, recorded); stop
  with "line_search" only when it fails with an empty memory. A non-descent
  direction from the memory gives the fallback -gamma P^{-1} g and clears
  the memory. (b) Stop at the first failure, as line_search_newton does.
  *Recommendation: (a)*. A quasi-Newton direction can be poor because the
  memory is stale (skipped pairs, a metric that changed a lot), which a
  restart fixes; the Newton direction carries no such state.
- **D8. Resuming.** (a) Exact: LbfgsResume carries the memory (the pairs as
  the user's vectors and covectors, rho, the cross pairings, gamma), which
  the user saves like x and g. (b) Plain numbers only, so the memory is
  dropped on resume (one restart). *Recommendation: (a)*.
- **D9. Operators.** Keyword operators, or operators_at with point_data, as
  in the Newton methods; PointOperators.hessvec becomes optional (the
  Newton methods still require it). metric_apply is optional: when given
  (or with the numpy default M = I), ||d||_M is computed for step_max and
  the display at one M product per iteration. Without it, |d| shows "-"
  and step_max is refused. A separate precond_solve does not require
  metric_apply here. *Recommendation: as stated.*
- **D10. The fixed path.** `fixed_preconditioner=True` declares that P
  (and M) do not depend on x; it requires the identity transport. Z =
  P^{-1} Y is then stored, and a P = M iteration costs one M^{-1} solve
  instead of three, with the same iterates. (a) In this plan, as phase 5.
  (b) Later. *Recommendation: (a)*. A constant metric with an expensive
  solve (the H^1 metric by multigrid) is a main use, and there the solves
  dominate.
- **D11. Names.** `lbfgs` in module `quasi_newton`; modules `transport`,
  `lbfgs_memory`, `lbfgs_core`, `lbfgs_diagnostics`; records LbfgsRecord,
  LbfgsSummary, LbfgsInfo, LbfgsResume; memory = 10 by default.
  *Recommendation: these, or yours.*

## Testing

- **Memory and two-loop recursion**, against dense references:
  - with current pairings, coordinates = sequential = the dense BFGS
    recursion;
  - with stale pairings, symmetric, positive definite and the newest
    secant equation exact;
  - circular slots (the oldest dropped), clearing, capacities 1, 2 and n;
  - the strict backend.
- **Equivalences:**
  - with M = I, the same iterates as a textbook L-BFGS written plainly in
    the test (the same backtracking and scaling);
  - with a constant M = L L^T, the iterates are L^{-T} times textbook
    iterates on f(L^{-T} z);
  - with a constant separate P, the same with P as the change of variables
    (M then only measures);
  - all to rounding.
- **Point-dependent metric, flat space:** a quadratic with a strongly
  varying M(x) converges, and the pairs are exact.
- **Manifolds:**
  - the Rayleigh quotient on the sphere, with projection transports;
  - fixed-rank matrix completion (tuple trees, project_covector);
  - the distributed numpy space.
- **The optimizer:**
  - Rosenbrock and the double well, with skips and restarts;
  - the p-Laplacian;
  - exact resumption, memory included;
  - callback and display;
  - operator calls counted against the work table: each operator and each
    transport as many times as it says, none repeated.
- **Typing:** mypy --strict on the generic core; the mistyped snippets
  rejected.
- **Backends:**
  - the strict backend on every configuration;
  - the JAX example and tests extended to lbfgs: compiled against numpy,
    trace counts, no host callbacks.

## Phases

Each phase ends with the test suite, mypy and the CI examples passing, and
is committed separately.

| | Phase | Delivers | Gate |
|---|---|---|---|
| 0 | Space entries | pair_covector_block, vector_member, covector_member in every constructor; check_space | tests; mypy |
| 1 | Memory | lbfgs_memory (store, clear, transport member by member and by block, the two-loop recursion), VectorTransport, identity_transport; generic types; the mistyped snippets | dense references; stale-pairing properties; strict backend |
| 2 | Optimizer | lbfgs_core, the driver, records, display, resumption, callback; PointOperators.hessvec optional | equivalence tests; convergence; resumption; call counts; **check in** with a comparison against the Newton methods on the tests' problems, and Euclidean against H^1 metric on the p-Laplacian over N = 15, 31, 63 |
| 3 | Varying metrics and manifolds | a point-dependent metric (the p-Laplacian with its linearized, Kacanov-type metric M(u)), the sphere and fixed-rank completion with transports, distributed and tree spaces | tests; examples |
| 4 | Compiled | the JAX example and tests extended to lbfgs | job "jax" |
| 5 | Fixed path | fixed_preconditioner (D10) | the same iterates as the general path; call counts |
| 6 | Wrap-up | README section, a comparison example, TODO | CI |

## Risks

- **Nonconvex problems with Armijo only.** Without a Wolfe search, steps
  can give <y, s> <= 0, and pairs are skipped. Skip rates will be measured
  at the check-in; Wolfe, with the deferred extrapolating search, is the
  remedy if they are high.
- **Transports that do not preserve pairings.** The operator stays positive
  definite and the newest secant equation exact, but the older pairs'
  information degrades; restarts and option D3 (c) are the remedies.
- **Strongly varying metrics.** The base changes every iteration. Outside
  the directions the pairs pin down, the stored pairs and the new base can
  disagree; H stays positive definite, and gamma follows the scale. The
  p-Laplacian with M(u) in phase 3 tests this.
- **Solves.** With P = M, three M^{-1} solves per iteration; D10 and
  scaling=False reduce them.
- **Copies.** numpy blocks copy on write, so once the memory is full,
  overwriting a slot copies the block: one copy per iteration, the same
  order as the block operations.
- **Theory on manifolds.** Superlinear convergence results for Riemannian
  BFGS assume isometric transports, and some also a locking condition with
  the retraction (Huang, Gallivan and Absil). Projections are the standard
  practice (Manopt) without these guarantees.

## References

- Nocedal and Wright, *Numerical Optimization* (2nd ed.), chapters 6 and 7:
  the variational characterization of BFGS, the two-loop recursion, the
  scaling of H0.
- Byrd, Nocedal and Schnabel, "Representations of quasi-Newton matrices and
  their use in limited memory methods", Math. Programming 63 (1994).
- Huang, Gallivan and Absil, "A Broyden class of quasi-Newton methods for
  Riemannian optimization", SIAM J. Optim. 25 (2015).
- Ring and Wirth, "Optimization methods on Riemannian manifolds and their
  application to shape space", SIAM J. Optim. 22 (2012).
- Li and Fukushima, "On the global convergence of the BFGS method for
  nonconvex unconstrained optimization problems", SIAM J. Optim. 11 (2001).

## Progress

Not started.
