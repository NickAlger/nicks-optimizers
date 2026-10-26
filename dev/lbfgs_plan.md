# Plan: metric-aware L-BFGS

October 2026. **Status: all decisions settled (2026-10-26)**, in progress,
see [Progress](#progress).

- **First review (2026-10-25)** settled D2, D4, D6 (every curvature test,
  as options), D7, D8, D10 (fixed M and fixed P, independently) and D11. It
  reworked the role of M (now the principle of the design), D1, D3, D5 and
  D9, and added D12 to D14.
- **Second review (2026-10-26)** agreed to the revised recommendations for
  D1, D3, D5, D9 and D12 to D14. Several options are uncharted, so once
  the code is built, experiments compare them (phase 8).

## Goal

`lbfgs`: a limited-memory BFGS method defined in the geometry of the metric
M, on flat spaces and manifolds. It takes the problem pieces of the Newton
methods (fun, grad, space, retract, metric_solve, metric_apply,
precond_solve, project_covector, point data), their backends (numpy by
default, compiled by JAX or another framework, the whole iteration in one
compiled piece), callback, resumption and a display of its own. No Hessian
is needed. New ingredient: a **vector transport** supplied by the user,
which carries the memory from one tangent space to the next.

Not in scope: Wolfe and extrapolating line searches (deferred; the design
leaves room for them), bounds (L-BFGS-B), trust-region L-BFGS, limited-memory
SR1, L-BFGS as a preconditioner for the Newton-Krylov solvers.

## The principle: M first

M is the Hilbert structure of the tangent spaces, and it is given by both of
its operators: M^{-1} (`metric_solve`, the Riesz map from covectors to
vectors) and M (`metric_apply`). Both are required here (D12); for plain
numpy arrays both default to the identity. They are not assumed cheap, but
they are used wherever they make the method cleaner or better motivated.
Options that spend fewer applications are offered next to the clean
defaults, with their costs in the work table, so the user can judge the
trade.

- **M defines everything geometric:** the initial guess B0 = M / gamma,
  every scaling fit, every norm (stopping, step lengths, angles), and the
  corrections of inexact transports (D3).
- **P, if given, is only an accelerant.** It proposes directions; it never
  defines a norm or a fit. It need not be symmetric or linear (D13): as in
  the flexible Newton solver, P chooses and M keeps the method safe.

## Where the metric enters

Typed, the objects of L-BFGS are:

| object | type | where |
|---|---|---|
| x_k | Point | |
| g_k = Df(x_k) | Covector | T*_{x_k} |
| d_k, s_k = alpha_k d_k | Vector | T_{x_k} |
| y_k = g_{k+1} - (g_k transported) | Covector | T*_{x_{k+1}} |
| B_k, the Hessian approximation | symmetric bilinear form on T_{x_k} (Vector -> Covector) | |
| H_k = B_k^{-1} | Covector -> Vector | |
| H0_k = gamma_k M_k^{-1} (or a P proposal, D13) | Covector -> Vector | |
| S, the stored steps | VectorBlock | T_{x_k} |
| Y, the stored gradient changes | CovectorBlock | T*_{x_k} |
| rho_i, cross pairings, gamma | Coords | numbers |
| T_{x -> x'} | Vector at x -> Vector at x' | |
| Tb_{x -> x'} | Covector at x -> Covector at x' | |

Textbook L-BFGS has its Euclidean assumptions in a few specific places, and
each one has a typed replacement:

| step | textbook (R^n) | metric-aware |
|---|---|---|
| initial guess | B0 = I / gamma | B0 = M / gamma: a multiple of the inner product itself |
| scaling | gamma = s^T y / y^T y | gamma = <y, s> / <y, M^{-1} y> (or another fit in M, D5) |
| update | s^T q, y^T r, q - a y, r + c s | <q, s_i>, <y_i, r>, combinations of covectors and of vectors: pairings only |
| new pair | s = x_{k+1} - x_k, y = g_{k+1} - g_k | s~ = T(alpha d), y = g_{k+1} - Tb(g_k) |
| old pairs | reused as they are | S <- T(S), Y <- Tb(Y), the stored pairings per D3 |
| curvature test | s^T y > 0 | pairings, or norms in M (D6) |
| stopping, lengths, angles | Euclidean | \|\|g\|\|_{M^{-1}}, \|\|d\|\|_M, the angle between -g and d in M |

The update itself needs no metric. It is
H+ = V* H V + rho s (x) s with V = I - rho y (x) s, every term a pairing.
Its variational characterization (Nocedal and Wright, section 6.1) is
intrinsic too: H+ is the symmetric operator closest to H with H+ y = s in
the weighted norm ||E||_W^2 = tr(W E W E), for any W with W s = y, and that
trace needs no coordinates. BFGS is affine covariant: under a linear change
of variables it maps to itself. So the metric enters through the initial
guess, its scaling, the transports and the measurements, and nowhere else.

**A reference test.** With a constant metric M = L L^T, the iterates are
exactly those of textbook L-BFGS on f(L^{-T} z), mapped back by
x = L^{-T} z. Checked numerically for the operator, to 1e-15.

## Initial guess and scaling (D1, D5)

**The base.** H0 = gamma M^{-1}, so B0 = M / gamma.

**The scaling** fits the scaled base to the newest pair, in the M geometry.
The options:

| scaling | gamma | fits | cost at the new point |
|---|---|---|---|
| "y" (default; Shanno-Phua, BB2) | <y, s~> / <y, M^{-1} y> | gamma M^{-1} y = s~, in the M-norm | one M^{-1} solve |
| "s" (BB1) | <M s~, s~> / <y, s~> | M s~ / gamma = y, in the M^{-1}-norm | one M product |
| "geometric" | \|\|s~\|\|_M / \|\|y\|\|_{M^{-1}} | equal lengths: \|\|gamma M^{-1} y\|\|_M = \|\|s~\|\|_M | both |
| "none" | gamma0 throughout | | none |

The three fits are ordered "y" <= "geometric" <= "s", by Cauchy-Schwarz.
"y" is the conservative textbook choice.

**The first step.** Before any pair is stored:

- gamma0 = 1 (default): B0 = M, the first step is -M^{-1} g. It is a full
  step when M is scaled like the Hessian.
- initial_step = l: gamma0 = l / ||g0||_{M^{-1}}, a first step of M-length
  l. This is the textbook rule typed: Liu and Nocedal take a Euclidean unit
  step.

After a restart (D7), the last fitted gamma is kept, since it carries
curvature information.

**Why ||g||_{M^{-1}} is not used as the scaling after the first step.** It
sets the length of the base step, not the curvature. As ||g|| -> 0 near a
minimizer, gamma = l / ||g|| would keep base steps of length l while Newton
steps shrink, which loses superlinear convergence. The fits above measure
curvature, which stays bounded.

**The denominator follows the base.** In a fit, the denominator is the
thing being scaled, applied to y. With the base M^{-1} it is
<y, M^{-1} y>. If the base were P^{-1} and the denominator still
<y, M^{-1} y>, gamma would be the curvature scale relative to M, applied to
P^{-1}.

*Example (checked):* the L^2 metric M = h^2 I and the H^1 operator P of the
comparison example (N = 63), with curvature equal to P, so the right gamma
is 1. That gamma comes out between 7e-3 (smooth modes) and 3e-5 (rough
modes). The M-norm fit of gamma P^{-1} y = s gives 1.000 for every mode
(D13).

## The preconditioner as an accelerant (D13)

The memory is data from f only: any step s and its gradient change y form
a valid pair, whatever produced the step. So P can never corrupt the
memory; it can only influence which steps are taken. The two-loop
recursion splits into a base-independent part and the base:

```
c  = S^T g               one block pairing                      (<g, s_i>)_i
a  = R^{-1} c            m x m triangular solve
q  = g - Y a             one block combination of covectors
r0 = base(q)             gamma M^{-1} q   or   gamma_P P^{-1}(q)
e  = Y^T r0              one block pairing of covectors with a vector
b  = R^{-T} (e + U^T a)  m x m triangular solve
d  = -(r0 + S (a - b))   one block combination of vectors
```

Here R = diag(1/rho) + U is upper triangular, in logical order (oldest
first). The P direction d_P is the same recursion with P^{-1} as the base,
applied once as a black box. The modes:

- **"safeguarded"** (recommended default when P is given): use d_P if it
  passes the angle test in M (D14),
  -<g, d_P> >= angle_tol ||g||_{M^{-1}} ||d_P||_M; otherwise use the M
  direction (the first half of the recursion is shared). Every direction
  taken is then gradient-related in the M geometry, which is what global
  convergence of Armijo backtracking needs, whatever P is: nonsymmetric,
  nonlinear or stochastic. P only changes the speed.
- **"trusted"**: d_P with the descent test only. This is classic
  preconditioned L-BFGS, for a symmetric positive definite P.
- **"subspace"** (later phase): minimize the BFGS model, built on the M
  base, over span{d_M, d_P}. The result is never worse than d_M in the
  model, the flexible solver's guarantee. It needs B applied to d_P (the
  direct compact form), hence the M-Gram matrix of the memory: one more M
  product per pair when M is fixed or the transport is isometric, m
  otherwise.

**The scale of the P base** (option scaling_P):

- "fit" (default): gamma_P = <M P^{-1} y, s~> / <M P^{-1} y, P^{-1} y>, the
  fit of gamma_P P^{-1}(y) = s~ in the M-norm. It costs one P^{-1} solve
  and one M product per stored pair. It equals the "y" scaling when P = M,
  and is defined for any P, since it is a scalar fit of one vector.
- "none": gamma_P = 1, trusting P's scale (e.g. multigrid on the Hessian).

## Transports

**Flat spaces: the identity, even when M varies.** The stored pairs are
data about the Hessian: for a quadratic, y = A s whatever M is. Each
iteration rebuilds H from the base at the new point and the stored pairs.
The newest secant equation H y_k = s_k holds exactly whatever the base, and
the older pairs pin H approximately along their directions. So a change of
M is absorbed along the directions the pairs have seen, and gamma sets the
scale elsewhere.

**The Riesz convention mixes the metric in.** Riemannian L-BFGS (Huang,
Gallivan and Absil 2015; Ring and Wirth 2012) keeps y as a vector,
M'^{-1} g' - T M^{-1} g. That amounts to the covector transport
Tb = M' T M^{-1}: with a varying metric on a flat space and T = I it
changes y (for a quadratic, y is no longer A s). Keeping y a covector
avoids this (D1).

**Curved spaces.** The user supplies T and Tb (settled, D2):

```python
@dataclass(frozen=True, kw_only=True)
class VectorTransport(Generic[PointT, VectorT, CovectorT, VectorBlockT, CovectorBlockT]):
    vector: Callable[[PointT, PointT, VectorT], VectorT]          # T_{x -> x'} v
    covector: Callable[[PointT, PointT, CovectorT], CovectorT]    # Tb_{x -> x'} w
    vector_block: Optional[Callable[[PointT, PointT, VectorBlockT], VectorBlockT]] = None
    covector_block: Optional[Callable[[PointT, PointT, CovectorBlockT], CovectorBlockT]] = None
```

- **Helpers:** `identity_transport` (flat spaces: nothing is called,
  nothing is copied) and `riesz_covector_transport(vector_transport)`
  (Tb = M' T M^{-1}: pairing-preserving when T is M-isometric, at two
  operator applications per covector).
- **Without block forms**, the memory is transported member by member,
  through the new space entries (D4).
- **Requirements:** both maps linear; Tb returns canonical covectors when
  project_covector is in use.

### What it means to transport B (D3)

B is a symmetric bilinear form. Transporting it means
T(B)(T u, T v) = B(u, v). As operators, T(B) = Tb B T^{-1} with
Tb = T^{-*}, the inverse adjoint.

- **The pairs.** The secant data B s_i = y_i must become
  T(B)(T s_i) = T^{-*} y_i: the identity holds for the pairs exactly when
  the covector transport is the inverse adjoint of the vector transport,
  i.e. preserves pairings, <Tb w, T v> = <w, v>.
- **The base.** B0 = (., .)_M / gamma transports to the new base exactly
  when T is isometric from M_x to M_x'. In a flat space with a varying M,
  T = I is not isometric, and the base is deliberately the new point's
  metric.
- **The stored scalars** are values of bilinear forms on memory vectors.
  In a flat space, y_j = A_j s_j with A_j the Hessian averaged over step j,
  so 1/rho_i = A_i(s_i, s_i) and U_ij = <y_j, s_i> = A_j(s_j, s_i). The
  identity says these values do not change under transport.

That decides between the options:

- **"stored"** (a): rho and U kept from when the pairs were formed. These
  are exactly the values the identity prescribes, whatever the transports
  are. The operator is always symmetric positive definite and satisfies the
  newest secant equation exactly (checked: 2000 random trials with stale
  pairings).
- **"recomputed"** (c): rho and U recomputed from the transported pairs
  after each transport, at m more block pairings per iteration. This
  equals "stored" when the transports satisfy the identity. Otherwise it
  inherits their defect, and a recomputed curvature can turn nonpositive;
  such pairs are dropped. The operator is still positive definite whenever
  all rho are positive.
- **"corrected"** (d): enforce the identity on the memory. Replace each
  transported covector Tb y_j by the nearest covector, in the
  M'^{-1}-norm, that pairs with the transported steps as y_j paired with
  the steps before:

      y_j' = Tb y_j + sum_l C_jl M' s_l',    C = (Pi - (Tb Y)^T S') G^{-1},

  with G = S'^T M' S' the M'-Gram matrix of the transported steps and Pi
  the matrix of stored pairings Pi_ji = <y_j, s_i>, all of them (one more
  block pairing per stored pair). After the correction "stored" and "recomputed" agree. It
  costs m M products and about 2m block pairings per iteration; G may need
  regularizing when steps are nearly dependent.
- **The transport defect.** A free diagnostic, per iteration:
  delta_k = <Tb g_k, T s_k> / <g_k, s_k> - 1. It is the defect of the
  identity on data at hand: <Tb g_k, s~> is part of <y, s~> anyway, and
  <g_k, s_k> = alpha * slope. It is zero for the identity transport, and
  reported in the records and the display. With an M product, the isometry
  defect ||T s_k||_{M'} / ||s_k||_M - 1 is reported too.

With the identity transport, all three options are the same, and nothing is
recomputed.

## The memory

The memory holds up to m pairs in circular slots of S (a VectorBlock) and Y
(a CovectorBlock), with rho, the cross pairings, both gammas, the count and
the newest slot. The active slots are a prefix until the memory is full;
clearing sets the count to zero.

**Storing a pair** (s~, y) at x_{k+1}: write it into the next slot (the
oldest when full). One block pairing of S with y gives the new row of
pairings, 1/rho included; "corrected" also needs the new column, from a
block pairing of Y with s~.

**The recursion** is the one under "The preconditioner as an accelerant":
the textbook two-loop (Nocedal and Wright, Algorithm 7.4) with its 2m
sequential pairings replaced by two batched ones. What it computes, for any
stored rho > 0 and U:

    H = W* H0 W + S R^{-T} D R^{-1} S^T,   W = I - Y R^{-1} S^T,  D = diag(1/rho).

For a symmetric positive definite base, H is symmetric positive definite,
and H y_newest = s_newest holds exactly.

## The curvature test (D6: every option)

A pair (s~, y) is stored only if:

| test | condition | needs |
|---|---|---|
| "relative" (default) | <y, s~> > curvature_tol alpha \|<g_k, d>\| | pairings only; invariant under scaling f and changes of variables; a Wolfe search would keep the ratio above 1 - c2 |
| "positive" | <y, s~> > 0 | nothing |
| "dennis-schnabel" | <y, s~> > sqrt(eps) \|\|s~\|\|_M \|\|y\|\|_{M^{-1}} | M s~ and M^{-1} y (shared with the "s" and "y" scalings) |
| "cautious" (Li and Fukushima) | <y, s~> >= eps_c \|\|g_k\|\|_{M^{-1}}^beta \|\|s~\|\|_M^2 | M s~ |

## The angle test (D14)

-<g, d> >= angle_tol ||g||_{M^{-1}} ||d||_M.

- **For d_P** (safeguarded mode): if it fails, use d_M.
- **For d_M:** if it fails, the fallback is -gamma M^{-1} g and the memory
  is cleared.
- **Cost:** ||d||_M is computed every iteration anyway (one M product), and
  ||g||_{M^{-1}} is known.
- **Default threshold:** small (1e-8 proposed, tuned at the check-in). For
  d = -H g, cos theta >= 2 sqrt(kappa) / (1 + kappa) with kappa the
  condition number of H in M (Kantorovich), so a healthy quasi-Newton
  direction trips the test only past kappa ~ 4e16.

## One iteration

At x_k, with f, the canonical gradient g, ||g||_{M^{-1}} and the memory at
x_k:

1. **Direction:**
   - the two-loop recursion with the M base, or with the P base in the
     mode of D13;
   - with an empty memory, the base applied to g, which reuses the
     gradient norm's M^{-1} g;
   - ||d||_M; the descent and angle tests (D14).
2. **Line search:** backtracking along retract(x, alpha d) from alpha = 1,
   capped by step_max (line_search.backtracking, unchanged).
3. **If accepted:** x_{k+1}, the canonical g_{k+1}, ||g_{k+1}||_{M^{-1}}.
   - Transport s~ = T(alpha d), Tb(g_k) and the memory; then the D3 mode,
     and the transport defect.
   - Form y = g_{k+1} - Tb(g_k).
   - Curvature test (D6): store the pair and refit gamma (and gamma_P), or
     skip it.
4. **If the search fails:** with a nonempty memory, clear it and continue
   (a restart); with an empty memory, stop with "line_search" (D7).

Steps 1 to 3 form one pure piece of fixed shapes (the capacity is m). The
driver keeps the stopping tests, restarts, records, display, callback and
resumption.

## Work per iteration

**Default, no P** ("y" scaling, "stored", "relative" test):

| | fun | grad | M^{-1} | M | P^{-1} | transports | block operations |
|---|---|---|---|---|---|---|---|
| per iteration | trials | 1 | 3: norm, base, "y" fit | 1: \|\|d\|\|_M | - | m+1 vectors, m+1 covectors (none for the identity) | 3 pairings, 2 combinations |

**With P, safeguarded, "fit":** M^{-1} 1 (the norm), P^{-1} 2 (base, fit),
M 2 (||d_P||_M, fit). A fallback to d_M adds an M^{-1} solve, and another
for gamma if it is stale.

**The options add:**

| option | adds per iteration |
|---|---|
| scaling "s" | 1 M product |
| scaling "geometric" | 1 M product and 1 M^{-1} solve |
| curvature tests "dennis-schnabel", "cautious" | shared with the scalings, else 1 M product and 1 M^{-1} solve |
| "recomputed" (D3) | m block pairings |
| "corrected" (D3) | m M products, about 2m block pairings |
| isometry defect | 1 M product |
| scaling "none", scaling_P "none" | saves the corresponding fit |

**Fixed paths** (D10, settled; each needs the identity transport):

- **fixed_metric=True** (M independent of x): store M^{-1} Y and M S as two
  more blocks.
  - M^{-1} y_k = M^{-1} g_{k+1} - M^{-1} g_k comes from the gradient-norm
    solves.
  - The base is r0 = gamma (M^{-1} g - (M^{-1} Y) a).
  - M d = -(gamma q + (M S)(a - b)), so ||d||_M is free; M s~ = alpha M d.
  - The M-Gram matrix of the memory is available (for "subspace").
  - An iteration then costs one M^{-1} solve and no M product, for twice
    the memory.
- **fixed_preconditioner=True** (P linear and independent of x): store
  P^{-1} Y. P^{-1} g replaces P^{-1}(q) as the one P^{-1} solve, and the
  fit's P^{-1} y comes free.

Against the Newton methods: no Hessian products and no Krylov solves.

## Typing

The memory, the recursion, the transports and the iteration are written
generically, in type variables for points, vectors, covectors and the two
block types, with no Any. Under mypy --strict every operation is checked
against its kind. The tests use the NewTypes. A set of deliberately
mistyped snippets, kept out of mypy's file list, must each fail mypy with
the expected error; a test runs them through mypy.api, and is skipped where
mypy is not installed.

## Diagnostics and display

`LbfgsRecord`, one per iteration:

| field | meaning |
|---|---|
| iteration, f, gnorm, gnorm_ratio | as in the other records |
| direction | "quasi-newton", "steepest" (empty memory) or "fallback" (failed the tests; memory cleared) |
| base | "M" or "P": which base the direction used |
| pairs, gamma, gamma_P | memory in use, and the scalings |
| step_norm, angle | \|\|d\|\|_M and the cosine in the angle test |
| slope, alpha, trials, accepted | as in LineSearchRecord |
| curvature | <y, s~> / (alpha \|<g_k, d>\|): in a flat space 1 - phi'(alpha)/phi'(0), so 1 when the step reached the one-dimensional minimizer |
| update | "stored", "skipped", "dropped" (pairs removed by "recomputed") or "-" |
| transport_defect, isometry_defect | D3 diagnostics (0 for the identity; nan when not computed) |
| restart | whether the memory was cleared after this iteration |
| elapsed, trace_alpha, trace_f | as in LineSearchRecord |

```
  it              f       |g|  |g|/|g0|  mem      direction  base       |d|    angle      slope     alpha  ls     gamma   curv    tdef  update
   0   2.420000e+01   2.3e+02   1.0e+00    0       steepest     M   2.3e+02  1.0e+00  -5.3e+04   1.2e-02   3   1.0e+00   0.97   0e+00  stored
   1   1.913200e+01   9.1e+01   4.0e-01    1   quasi-newton     M   8.7e-01  6.2e-01  -2.1e+01   1.0e+00   1   4.1e-03   1.02   0e+00  stored
```

- **Level 2** adds the line-search trials (format_ls_trials, shared).
- **LbfgsSummary:** the reason (converged, max_iter, callback,
  line_search), the work (each operator, transports, trials), the
  restarts, the skipped pairs, and how often P's proposal was taken.
- **LbfgsInfo** is what the callback receives; **LbfgsResume** holds the
  memory, so resuming is exact (D8).

## Passing the operators (D9)

**Keyword operators only:** metric_solve(x, w), metric_apply(x, v),
precond_solve(x, w).

Pros:

- the smallest API, with one signature style shared with the transports;
- nothing to change in PointOperators;
- fewer ways to do the same thing in docs and tests;
- with numpy, a user can memoize a per-point setup.

Cons:

- setup that depends on the point is repeated on every call. Assembling
  M(x), factorizing it, building a multigrid hierarchy: L-BFGS applies the
  operators 3 to 5 times per point, and point-dependent metrics are the
  case this method is for.
- Under compilation, Python-side caching is impossible. Within one compiled
  piece XLA may merge identical setups, but each point spans two calls of
  the piece (the norm and fits at the end of one, the base and ||d||_M at
  the start of the next), so at least two setups per point remain.

**Also operators_at(x, data) with point_data(x)**, as in the Newton
methods.

Pros:

- the setup is computed once per point and carried in the state between
  compiled pieces;
- users can reuse the operators_at they wrote for the Newton methods;
- the code exists and is tested.

Cons:

- a second way to pass operators;
- PointOperators.hessvec must become optional (L-BFGS ignores it, the
  Newton methods still require it);
- the transports would not see the point data (a later extension could
  pass data at both points).

*Recommendation:* both, with keyword operators as the documented default.
operators_at is what makes a point-dependent metric with an expensive setup
pay one setup per point.

## API

```python
lbfgs(fun, grad, x0, *, space=None, retract=None, transport=None,
      metric_solve=None, metric_apply=None, precond_solve=None,
      operators_at=None, point_data=None, project_covector=None,
      memory=10, scaling="y", gamma0=1.0, initial_step=None,
      precond_mode="safeguarded", scaling_P="fit",
      pairings="stored", curvature_test="relative", curvature_tol=1e-8,
      angle_tol=1e-8, fixed_metric=False, fixed_preconditioner=False,
      convergence_norm="metric", gtol=0.0, rtol=1e-8, gnorm0=None,
      max_iter=1000, c1=1e-4, max_backtracks=30, step_max=None,
      display=0, print_fn=print, callback=None, resume=None, g0=None,
      backend=None) -> (x, LbfgsSummary)
```

- **Defaults:** without a space, numpy arrays, M = I (both operators), x + s
  and the identity transport. With a space, metric_solve, metric_apply,
  retract and transport are required.
- **Option names** are provisional, to be fixed in phase 2.

## Decisions

All settled: D2, D4, D6, D7, D8, D10 and D11 on 2026-10-25, the rest as
recommended on 2026-10-26.

- **D2.** User-supplied transports (vector and covector maps, block forms
  optional), `identity_transport`; the identity is the default without a
  space, and the transport is required with one.
- **D4.** pair_covector_block, vector_member and covector_member as
  required VectorSpace entries, in every constructor and check_space.
- **D6.** Every curvature test, as options; "relative" by default.
- **D7.** Restart (clear the memory) when the line search fails with a
  nonempty memory; stop when it fails with an empty one; fallback with a
  cleared memory for a direction that fails the tests.
- **D8.** Exact resumption, with the memory in LbfgsResume.
- **D10.** Take advantage of fixed operators, with M and P fixed
  independently (fixed_metric, fixed_preconditioner).
- **D11.** The names `lbfgs` (in `quasi_newton`), `transport`,
  `lbfgs_memory`, `lbfgs_core`, `lbfgs_diagnostics`; memory = 10.

Settled on 2026-10-26, as recommended:

- **D1 (revised). The geometry of the method.** Steps as vectors, gradient
  changes as covectors (not the Riesz convention), base M^{-1}, every fit
  and norm in M; P never defines a norm. *Recommendation: yes.*
- **D3 (revised). The stored pairings under transport.** "stored" by
  default: it keeps exactly the values the identity
  T(B)(T u, T v) = B(u, v) prescribes, and always gives a positive definite
  operator. "recomputed" (dropping pairs with nonpositive curvature) and
  "corrected" (the identity enforced on the memory, using M) as options.
  The transport defect is always reported. *Recommendation: as stated;
  "corrected" in phase 4.*
- **D5 (revised). Scaling.** "y" by default; "s", "geometric" and "none"
  as options. gamma0 = 1 (B0 = M) by default, initial_step for a first step
  of given M-length; the gradient norm only for the first step.
  *Recommendation: as stated.*
- **D9. Operators.** Keyword operators and operators_at, as laid out above.
  *Recommendation: both, keyword operators the documented default.*
- **D12 (new). M required.** metric_apply joins metric_solve as a required
  operator for lbfgs (the identity for numpy arrays without a space). A user
  whose metric exists only through M^{-1} (a multigrid cycle) passes that
  cycle as P with a simple M, the library's pattern for separating geometry
  from acceleration. *Recommendation: yes.*
- **D13 (new). How P enters.** "safeguarded" by default (P's direction when
  it passes the angle test in M, else the M direction), "trusted" as an
  option, "subspace" in a later phase; scaling_P "fit" (in the M-norm) by
  default, "none" as an option. *Recommendation: as stated.*
- **D14 (new). The angle test** for every direction (d_P falls back to
  d_M, d_M to steepest descent with a cleared memory), with angle_tol =
  1e-8 to be tuned at the check-in. *Recommendation: yes.*

## Testing

- **Memory and recursion**, against dense references:
  - with current pairings, coordinates = sequential = the dense BFGS
    recursion;
  - with stale pairings, symmetric, positive definite and the newest
    secant equation exact;
  - the three pairing modes: equal under pairing-preserving transports;
    after "corrected", <y_j', s_i'> = <y_j, s_i>, with the correction of
    least M'^{-1}-norm;
  - circular slots, clearing, capacities 1, 2 and n;
  - the strict backend.
- **Equivalences:**
  - with M = I, a textbook L-BFGS written plainly in the test (Euclidean
    unit first step, Shanno-Phua, the same backtracking);
  - with a constant M = L L^T, the iterates are L^{-T} times textbook
    iterates on f(L^{-T} z);
  - all to rounding.
- **Scalings:** the fits solve their least-squares problems, and the order
  "y" <= "geometric" <= "s" holds.
- **P:**
  - "trusted" with a symmetric positive definite P;
  - "safeguarded" with the preconditioners of the flexible experiments
    (nonsymmetric, nonlinear, stochastic): convergence, the angle test
    taking over when P misleads, and the M-norm fit giving gamma_P = 1 when
    P is the curvature.
- **Transports:**
  - a quadratic with a strongly varying M(x) on a flat space (pairs
    exact);
  - the Rayleigh quotient on the sphere with projection transports
    (defects of second order in the step) and with an isometric transport
    plus riesz_covector_transport (zero pairing defect);
  - fixed-rank matrix completion (tuple trees, project_covector);
  - the distributed numpy space.
- **The optimizer:**
  - Rosenbrock and the double well, with skips and restarts;
  - the p-Laplacian;
  - exact resumption, memory included;
  - callback and display;
  - operator calls counted against the work tables for every option,
    none repeated.
- **Fixed paths:** the same iterates as the general path, and the call
  counts of the table.
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
| 1 | Memory | lbfgs_memory (store, clear, transport by member and by block, the split recursion, "stored" and "recomputed"), VectorTransport, identity_transport, riesz_covector_transport; generic types; the mistyped snippets | dense references; stale-pairing properties; strict backend |
| 2 | Optimizer, M only | lbfgs_core, the driver, scalings, curvature tests, angle test, restarts, records, display, resumption, callback; operators_at with PointOperators.hessvec optional | equivalence tests; convergence; resumption; call counts; **check in**: comparison with the Newton methods, Euclidean against H^1 metric on the p-Laplacian over N = 15, 31, 63, skip and restart rates, angle_tol |
| 3 | The preconditioner | "safeguarded" and "trusted", scaling_P | weird-P tests; call counts |
| 4 | Varying metrics and manifolds | the p-Laplacian with its linearized, Kacanov-type metric M(u); sphere and fixed-rank completion with transports; "corrected" and the defect diagnostics | tests; examples |
| 5 | Compiled | the JAX example and tests extended to lbfgs | job "jax" |
| 6 | Fixed paths | fixed_metric, fixed_preconditioner | same iterates; call counts |
| 7 | Wrap-up | README section, a comparison example, TODO | CI |
| 8 | Experiments | the options against each other: the scalings, gamma0 and initial_step, the pairing modes, the curvature tests, the P modes, angle_tol, the fixed paths; on the test problems, varying metrics, manifolds and unconventional preconditioners; written up in `experiments/lbfgs_options/` with recommended defaults | write-up |

"subspace" (D13) is left for after phase 7, or earlier if the check-in
asks for it.

## Risks

- **Nonconvex problems with Armijo only.** Pairs are skipped when
  <y, s> <= 0; skip rates will be measured at the check-in. Wolfe, with the
  deferred extrapolating search, is the remedy if they are high.
- **Inexact transports.** "stored" stays positive definite and keeps the
  identity's values, but its covectors carry the transport's defect;
  "corrected" removes it on the memory, at m M products.
- **Strongly varying metrics.** The base changes every iteration. Outside
  the directions the pairs pin down, the pairs and the new base can
  disagree; H stays positive definite, and gamma follows the scale. Phase 4
  tests this on the p-Laplacian with M(u).
- **Solves.** The clean default spends three M^{-1} solves and one M
  product per iteration (without P). The fixed paths, scaling="none", and
  the work table let the user trade.
- **Many options.** Each needs tests and call counts. They are kept
  orthogonal (base, scaling, pairing mode, curvature test, fixed paths) so
  that each is a small, separate branch.
- **Theory on manifolds.** Superlinear convergence results for Riemannian
  BFGS assume isometric transports and, for some, a locking condition with
  the retraction (Huang, Gallivan and Absil). Projections are the standard
  practice (Manopt) without these guarantees.

## References

- Nocedal and Wright, *Numerical Optimization* (2nd ed.), chapters 6 and 7.
- Byrd, Nocedal and Schnabel, "Representations of quasi-Newton matrices and
  their use in limited memory methods", Math. Programming 63 (1994).
- Huang, Gallivan and Absil, "A Broyden class of quasi-Newton methods for
  Riemannian optimization", SIAM J. Optim. 25 (2015).
- Ring and Wirth, "Optimization methods on Riemannian manifolds and their
  application to shape space", SIAM J. Optim. 22 (2012).
- Li and Fukushima, "On the global convergence of the BFGS method for
  nonconvex unconstrained optimization problems", SIAM J. Optim. 11 (2001).
- Liu and Nocedal, "On the limited memory BFGS method for large scale
  optimization", Math. Programming 45 (1989).

## Progress

- **Phase 0 (2026-10-26).** pair_covector_block, vector_member and
  covector_member in VectorSpace and every constructor:
  - numpy: a read-only view of a member in use, zeros for a row only
    another block has set;
  - array_space: a dynamic index;
  - tuple trees: leafwise;
  - tuple_block_space: from pair.

  check_space tests the three entries.
- **Phase 1 (2026-10-26).**
  - `transport` holds VectorTransport, identity_transport and
    riesz_covector_transport.
  - `lbfgs_memory` holds the memory and the split two-loop recursion,
    generic and numpy-free:
    - slots ordered by stamps;
    - free slots hold zero, so linear transports and masked coefficients
      are safe;
    - a pair is stored in the lowest free slot, else the oldest;
    - transports go by block or member by member (fresh blocks, built by
      appending);
    - memory_recompute drops pairs whose curvature falls to the floor they
      passed when stored.
  - Tests: the coordinate recursion against the textbook sequential
    recursion and the dense BFGS recursion (capacities 1, 2, 5, n,
    wrapped); 200 trials of stale pairings (symmetric positive definite,
    newest secant exact); pairing-preserving transports keep the stored
    pairings current; recomputed pairings give the BFGS operator of the
    transported pairs, and a flipped pair is dropped; block and member
    transports agree; riesz_covector_transport preserves pairings for
    isometries; the strict backend agrees with numpy.
  - Typing: `tests/typing_errors/` holds mix-ups that mypy must reject
    (arg-type, or misc where swapped generic arguments defeat inference),
    checked by `tests/test_typing.py`. With memory_store untyped, the test
    fails.
- **Phase 2 (2026-10-26).** `lbfgs_core` (LbfgsProblem, LbfgsState, init,
  lbfgs_step, restart: one compiled piece per iteration),
  `lbfgs_diagnostics` (records, resume, summary, display) and
  `quasi_newton.lbfgs`, with the scalings, gamma0 and initial_step, the
  "stored" and "recomputed" pairings, the four curvature tests, the angle
  test with fallback, restarts and step_max. PointOperators.hessvec became
  optional. Resumption keeps each pair's slot, so it is bitwise exact. The
  trials table (format_trials) is shared with line_search_newton.
  - Tests:
    - with M = I, textbook L-BFGS over a whole convex run (1e-8) and the
      first 40 Rosenbrock iterates (1e-10); beyond that, rounding
      differences grow along the valley, as between any two
      implementations;
    - with a constant M, textbook L-BFGS on f(L^{-T} z);
    - every option converges to a local minimizer;
    - a quadratic with a strongly varying M(x) keeps exact pairs;
    - on the sphere with projection transports, the transport defect equals
      -a^2 / (1 + a^2), and the transports are called as counted;
    - restarts, fallbacks and skipped pairs;
    - bitwise resumption;
    - display;
    - operator calls against the work table for four option sets;
    - argument checks;
    - the strict backend, also with member-by-member transports.
  - A quick check: compiled with JAX on the p-Laplacian (H^1 metric), the
    same 32 iterations as numpy, f to 1e-15. The full JAX tests come in
    phase 5.
- **Check-in (2026-10-26)**, `experiments/lbfgs_checkin/`:
  - **The metric.** With the H^1 metric lbfgs is nearly mesh-independent on
    the p-Laplacian: 25, 33, 41 and 49 iterations for N = 15 to 127,
    against 150 to 1750 with the Euclidean, L^2 or Jacobi metrics.
  - **Against the Newton methods.** lbfgs spends the fewest derivative
    evaluations: 2 to 4 times fewer on Rosenbrock and with the Jacobi
    metric, 30-40% fewer with H^1. It spends the most solves, three per
    iteration.
  - **Safeguards.** No restarts or fallbacks at the defaults, and skipped
    pairs only on two nonconvex runs. angle_tol has no effect up to 1e-2
    and hurts at 0.1, so the default 1e-8 stays.
  - **Scalings.** "y" is the best default; "none" fails with the Euclidean
    metric but is the best with H^1.
- **Fixed metric (2026-10-26)**, moved ahead of phase 3 on review.
  - `fixed_metric=True` keeps Z = M^{-1} Y and W = M S slot for slot
    (Memory.Z, Memory.W).
  - The base is gamma (M^{-1} g - Z a), with M d = gamma q + W (a - b);
    M^{-1} y = M^{-1} g_new - M^{-1} g and M s = alpha M d. So an iteration
    applies M^{-1} once (the gradient norm) and M never.
  - It requires the identity transport and a fixed linear M^{-1}.
  - LbfgsResume.metric_pairs makes resumption bitwise. A state saved
    without them gets them recomputed (one solve and one product per pair).
  - Tests: the general path's decisions and iterates, for four option sets
    and with fallbacks; the call counts; resumption both ways; the strict
    backend; refusing a non-identity transport.
  - fixed_preconditioner comes with phase 3.
- **Note: constant against point-dependent metrics (2026-10-26)**,
  `experiments/lbfgs_metrics/`, one problem (the p-Laplacian, sparse LU).
  - **Fixed against general path.** The same iterates with a third of the
    solves, about 60% of the time.
  - **The Kacanov metric M(u)** needs the fewest gradients. With memory 1
    it takes 17 iterations at every N; with memory 10 it degrades on fine
    meshes (74 iterations at N = 255, 9% unit steps): older pairs conflict
    with a changing base.
  - **Cost.** One factorization per point makes it slower than the
    constant H^1 metric on the fixed path here.
  - **Both kinds stay supported.** To try: P = Kacanov with M = H^1
    (phase 3), and memory policies for changing metrics (phase 8).
- **Phase 3 (2026-10-26).** precond_solve (keyword or from operators_at)
  as an accelerant.
  - The memory's recursion with the base gamma_P P^{-1} proposes each
    direction.
  - "safeguarded" (default) takes it if it passes the angle test in M,
    else the M direction. "trusted" applies the descent test only.
  - The M base's gamma is refitted lazily, to the newest pair, only when
    the M direction is used.
  - scaling_P: "fit" (the fit of gamma_P P^{-1} y = s in the M-norm) or
    "none".
  - convergence_norm="preconditioner" stops on ||g||_{P^{-1}}.
  - fixed_preconditioner keeps P^{-1} Y (Memory.ZP) and P^{-1} g per
    point, so phase 6 is done.
  - Records show the base of each direction; the summary counts P solves
    and the directions taken from P.
  - Tests:
    - P = M gives the run without P;
    - a P that proposes ascent falls back to the M direction, bitwise the
      run without P;
    - trusted and safeguarded runs with an SPD P;
    - nonsymmetric, nonlinear, random and rotated P all converge (the
      rotated one is often refused);
    - gamma_P = 1 when P is the curvature;
    - the P convergence norm;
    - call counts;
    - fixed_preconditioner, with fixed_metric too, and resumption;
    - operators_at giving P; argument checks;
    - the strict backend with unconventional P.
  - In `experiments/lbfgs_metrics/`, M = H^1 (fixed) with P = Kacanov
    takes P's proposal every iteration. With memory 1 that is about 21
    iterations at every N; scaling_P "none" is much worse than "fit".
- **Phase 4 (2026-10-26).**
  - **pairings="corrected"** (memory_correct): the transported covectors
    get the least correction in the M'^{-1}-norm that makes them pair with
    the transported steps as stored. One M product per pair; the full
    stored pairings are kept for it.
  - **The isometry defect** ||T s||_{M'} / ||s||_M - 1 in the records, for
    a non-identity transport.
  - **Fixed-rank matrix completion** (`examples/matrix_completion.py`):
    the projection transport for tangent vectors and canonical covectors,
    complete_lbfgs, and lbfgs rank continuation, printed next to the trust
    region's (the same fit at every rank).
  - **`examples/lbfgs_p_laplacian.py`** (run in CI, checked by
    `tests/test_lbfgs_example.py`): the Euclidean metric, the constant H^1
    metric on the fixed path, the Kacanov metric through operators_at and
    point_data, and H^1 with the Kacanov operator as P.
  - **Tests:**
    - corrected covectors pair as stored and change least;
    - the strict backend;
    - the sphere in all three pairing modes, with the exact isometry defect
      1 / sqrt(1 + a^2) - 1;
    - the fixed-rank transport is the Frobenius projection and preserves
      pairings to second order;
    - lbfgs reaches the trust region's fit, also in rank continuation;
    - three threads in lockstep with a serial run (with M, M and P, fixed M
      and P);
    - tuple trees;
    - tuple_block_space.
  - **A bug fixed on the way:** tuple_block_space's block pairings assumed
    that k members were set. check_space now tests k beyond the members
    set.
