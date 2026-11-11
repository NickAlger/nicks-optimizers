# Plan: switching from Gauss-Newton to the full Hessian

November 2026. **Status: approved (2026-11-11), in implementation**; see
[Progress](#progress). Nick settled the mode name ("switch"), line-search
support, the detector's parameters (tau = 0.25 and k = 2, exposed as
keywords), a one-way switch, the forcing powers (0.5 before the switch, 1.0
after it, with the keyword logic of [Forcing power](#forcing-power)), and
D1 to D9 under [Decisions](#decisions).

Background: the study "Gauss-Newton to full-Newton switching" (Nick Alger
and Claude, November 2026; artifact V9rpRsqpod3aSuBVNbxhLo), on a
regularized extended Rosenbrock problem (n = 200), a diffusion inversion
(1024 parameters) and matrix completion (4000 unknowns). Over 29 cases,
switching on the detector below beat pure Gauss-Newton every time, using
5% to 92% fewer Hessian products. It also avoided the cost that full Newton
from the start pays on ill-posed problems: 23% to 55% more PDE solves on
the diffusion inversion with small regularization.

## Goal

Add `hessian="switch"` to `trust_region_newton` and `line_search_newton`.
For least-squares-like problems the user gives two Hessian actions:

- `gn_hessvec`: a cheap model Hessian that is positive semidefinite, usually
  Gauss-Newton `J^T J` (plus any regularization).
- `hessvec`: the full Hessian, `J^T J + S` with `S = sum_k r_k Hess(r_k)`.

The run starts with `gn_hessvec`. After every step it also evaluates the
full model along that step, at the cost of one extra product with the full
Hessian. When the full model has been clearly better than the Gauss-Newton
model for k steps in a row, the run switches to `hessvec` for good.

The default `hessian="full"` is today's behavior: `hessvec` is the model
Hessian, and every existing run stays bit for bit the same (see D9).

Not in scope:

- switching back after the switch (Nick: one-way);
- secant approximations of S (NL2SOL-style);
- the Fletcher-Xu switching rule, which needs no extra product. This can be
  added later as an alternative trigger (study's follow-up 1);
- delaying the start of tracking to save the tracking products (see Risks).

## The method

The switch run has two phases: before the switch and after it.

**Before the switch.** Each iteration is an ordinary iteration (trust
region or line search) with these settings:

- the model Hessian is `gn_hessvec`;
- the forcing term is `min(forcing_max, (||g|| / ||g0||)^p_pre)`, with
  `p_pre = 0.5` by default;
- rho and the radius update use the Gauss-Newton model.

Then one more operation: the full model along the step taken, `s` (the
trust-region step, or `alpha d` for the line search):

```
slope    = <g, s>
m_GN(s)  = model value of the Gauss-Newton model at s      (known already)
m_N(s)   = slope + 1/2 <H s, s>                            (ONE product with hessvec)
change   = f(retract(x, s)) - f(x)                         (known already)
e_GN     = |change - m_GN(s)|
e_N      = |change - m_N(s)|
```

`m_GN(s)` is already known:

- Trust region: the solver's `true_model_value`.
- Line search: `-predicted`, which `line_search_finish` already computes as
  `-(alpha slope + 1/2 alpha^2 <H_GN d, d>)`.

A rejected trust-region trial resets the hit count, whatever its errors
(D5). A failed line search ends the run, so the rule does not arise there.

**The detector** runs in the driver, in Python floats, like the stopping
tests:

```
valid = isfinite(e_GN) and isfinite(e_N) and e_GN > c_round * delta
        where delta = 10 eps max(1, |f(x)|)        (the rounding allowance of rho, D4)
hit   = valid and e_N < tau * e_GN                   (tau = switch_tau = 0.25)
hits  = hits + 1 if hit and accepted else 0          (D3: an invalid sample resets;
                                                      D5: so does a rejected trial)
if hits >= k:  switch                                (k = switch_k = 2)
```

The switch takes effect at the next iteration. Why the detector works: for
a step `s`, `e_GN` is about `1/2 |<S s, s>|`, which is of order `||s||^2`,
while `e_N` is of order `||s||^3` along a second-order retraction. Near a
minimizer with a nonzero residual, `e_N / e_GN` therefore falls toward
zero. In the noisy early iterations it is about 1 or larger, so a threshold
well below 1, required on k consecutive steps, rarely fires too early.

The rounding gate replaces the study's `||g|| > 1e-12 ||g0||`. It is
scale-free and measures the right quantity. If Gauss-Newton's own error is
already at the level of rounding in f, the Gauss-Newton model is as good as
any model can be measured to be, and there is nothing to gain by switching.
`e_N` itself may be at the rounding level: that is a hit, and a correct
one.

**After the switch.** These are ordinary iterations with these settings:

- the model Hessian is `hessvec`;
- the forcing power is `p_post = 1.0` by default;
- there is no tracking product.

The trust-region radius carries over unchanged, and rho uses the full
model. The line search needs nothing carried over. Negative curvature is
handled as in any full-Hessian run: Steihaug or Lanczos boundary steps in
the trust region, truncation in the line search (or `saddle_free`).

Nothing about the step solve changes. Before the switch the run is exactly
a Gauss-Newton run with forcing power `p_pre`. After the switch it is
exactly a full-Hessian run with forcing power `p_post`, resumed from the
switch point. The tests use both facts (see [Testing](#testing)).

## Forcing power

The forcing term is now hard-coded as
`min(forcing_max, sqrt(||g|| / ||g0||))`. It gets a power:
`min(forcing_max, (||g|| / ||g0||)^p)`.

- `forcing_power` (new, all modes): the power. With `hessian="full"`,
  `None` means 0.5 (today's term). With `hessian="switch"`, it is the power
  after the switch.
- `pre_switch_forcing_power` (new, switch only): the power before the
  switch. Given with `hessian="full"`, it raises.

In switch mode the two resolve as (D2):

| `pre_switch_forcing_power` | `forcing_power` | before the switch | after the switch |
|---|---|---|---|
| None | None | 0.5 | 1.0 |
| p | None | p | 1.0 |
| None | q | q | q |
| p | q | p | q |

The power lives in `CoreProblem`, so it is static and closed over by the
compiled pieces, like `forcing_max`. A power of exactly 0.5 keeps the
`xp.sqrt` path, decided in Python because the power is static.
`xp.power(r, 0.5)` and `xp.sqrt(r)` need not agree to the last bit, on
every backend, so this keeps the default runs bit for bit the same.
`criteria.forcing_term` (the public float version) gets the same optional
`power=0.5` argument.

## How it fits the code

The switch run uses two `CoreProblem`s that differ in two fields:

| | before the switch | after the switch |
|---|---|---|
| model Hessian | `gn_hessvec` | `hessvec` |
| forcing power | `pre_switch_forcing_power` | `forcing_power` |

The two share `fun`, `grad`, the metric, the preconditioner, `point_data`,
the retraction and the space. So the `OuterState` computed by either is
valid for the other, and the switch costs nothing at the point where it
happens: same `x`, `f`, `g`, point data and gradient norms.

Each phase has its own pieces, built by the existing `step_solve` and
`compile_function`. The pieces after the switch are built only when the
switch fires, so a compiling backend compiles them at most once per run.

### newton_core

- `PointOperators` gets a field `gn_hessvec: Optional[Callable[[VectorT],
  CovectorT]] = None`. lbfgs ignores it, as it ignores `hessvec`.
- `CoreProblem` gets:
  - `gn_hessvec`, the keyword-argument version, like `hessvec`;
  - `model: Literal["full", "gauss_newton"] = "full"`: which operator
    `operators()` returns as `hessvec`;
  - `forcing_power: float = 0.5`.
- `operators(problem, x, data)` gives `hessvec = gn_hessvec` when
  `problem.model == "gauss_newton"`. It raises if that operator is missing,
  as it does today for a missing `hessvec`. Everything downstream
  (`subproblem_at`, the solvers, `SubproblemOperators`) is unchanged: they
  see one `hessvec`.
- `forcing_term(xp, gnorm, gnorm0, forcing_max, power=0.5)`: as above.
- `full_hessvec_at(problem, state)` (new, small): the full Hessian's action
  at the state (`hessvec`, from the keyword arguments or `operators_at`),
  for the tracking product.

### trust_region_core and newton_line_search_core

`step_finish` and `line_search_finish` get an optional static argument,
`track: Optional[CoreProblem] = None`. This is the problem for after the
switch, used only to apply its Hessian. When it is given (that is, before
the switch), the piece:

1. applies `Hs = hessvec(s)` once, on the step tried: `s` for the trust
   region, `alpha d` for the line search;
2. returns `full_model = <g, s> + 1/2 <Hs, s>` in the raw iteration
   (`RawIteration.full_model`, `RawLineSearchIteration.full_model`), and
   `gn_model` (the Gauss-Newton model value at the same step);
3. otherwise computes exactly what it computes today.

When `track` is `None` (`hessian="full"`, or after the switch), the fields
are `nan` and no product is applied. The presence of the product is decided
in Python, by whether `track` is `None`, so the pieces stay traceable and
the default pieces are unchanged.

There is no `project_covector` on `Hs`, because components that are
invisible to the pairing contribute nothing to `<Hs, s>`. In the line
search, a fallback direction has no Gauss-Newton curvature, so its
`gn_model` is `nan`, and the sample is invalid (D3).

### The drivers (trust_region, newton_line_search, newton_driver)

- `check_arguments` gets the new keywords (see [API](#api)) and raises for
  inconsistent combinations, in the library's style:
  - `hessian="switch"` without `gn_hessvec`, or with `operators_at` and no
    `gn_hessvec` in the returned `PointOperators` (checked at the first
    point);
  - `gn_hessvec`, `switch_tau`, `switch_k` or `pre_switch_forcing_power` given
    with `hessian="full"`;
  - `switch_tau` not in (0, 1);
  - `switch_k < 1`;
  - a forcing power `<= 0`.
- `make_problem` builds the two problems; `step_solve` is called per phase.
- The detector is a pure function in `criteria`:
  `switch_sample(change, gn_model, full_model, f, tau) -> (valid, hit,
  e_gn, e_full)`. The driver keeps `hits`, `switched` and
  `switch_iteration`.
- The work count is split by Hessian. Today `hessvec_calls` sums the
  subproblems' products, which all go to `hessvec`. In a switch run:
  - `hessvec_calls` counts the full Hessian (solves after the switch, plus
    the tracking products);
  - a new `gn_hessvec_calls` counts the Gauss-Newton solves before the
    switch.
  
  With `hessian="full"`, `gn_hessvec_calls` is 0 and `hessvec_calls` is
  unchanged.

## API

```python
trust_region_newton(fun, grad, hessvec, x0, *, ...,          # as today, plus
                    hessian="full",              # "full" | "switch"
                    gn_hessvec=None,             # (x, v) -> Covector; required with "switch"
                    switch_tau=0.25,             # hit when e_N < tau * e_GN
                    switch_k=2,                  # consecutive hits that trigger the switch
                    forcing_power=None,          # None: 0.5 ("full"), 1.0 (after a switch)
                    pre_switch_forcing_power=None,  # see Forcing power
                    ...)
line_search_newton(...)                          # the same five keywords
```

With `operators_at`, the returned `PointOperators` carry `gn_hessvec`
beside `hessvec`. The keyword defaults shown are the effective ones. In
code, `switch_tau`, `switch_k` and `pre_switch_forcing_power` default to `None`,
so that giving them with `hessian="full"` can raise.

On a manifold, `hessvec` is the Hessian of the pullback (the Riemannian
Hessian for a second-order retraction). `gn_hessvec` is its Gauss-Newton
part. `examples/matrix_completion.py` already has both:
`hessian_action(..., curvature=True/False)`.

## Diagnostics and display

**Records.** `IterationRecord` and `LineSearchRecord` get fields with
default `None`, so that the records of a `hessian="full"` run compare equal
to today's:

| field | meaning |
|---|---|
| `model` | `"gauss_newton"` or `"full"`: the model Hessian of this iteration |
| `error_gn`, `error_full` | `e_GN` and `e_N` (before the switch only) |
| `switch_hit` | whether this iteration counted as a hit (`None` after the switch) |

**Summaries.** `RunSummary` and `LineSearchSummary` get:

- `switch_iteration: Optional[int] = None`: the first iteration that used
  the full Hessian;
- `gn_hessvec_calls: int = 0`.

`merge_summaries` and `merge_ls_summaries` carry both. `history_array` keeps
its eight columns (D6).

**Resume.** `ResumeState` and `LineSearchResume` get `switched: bool =
False` and `switch_hits: int = 0`. They are plain numbers, still
round-tripped through JSON. A resumed switch run continues exactly: in the
same phase, with the same hit count. Resuming a switched state with
`hessian="full"` raises.

**Display.** In switch mode only, so that the default display is unchanged
character for character, level 1 gets two more columns:

- `H`: `GN` or `N`;
- `eN/eGN`: blank after the switch; `-` for an invalid sample.

The row where the switch fires is marked `*` after the ratio. The summary
adds a line, `switched to the full Hessian at iteration 23` or `did not
switch`. The work line splits the products into Gauss-Newton and full.

```
  it              f       |g|  ...  rho  acc    k       stop  neg   H   eN/eGN
  21   3.104552e+01   2.1e-03  ...  +0.999  yes   14        tol    0  GN    0.31
  22   3.104552e+01   6.0e-04  ...  +1.000  yes   15        tol    0  GN    0.12
  23   3.104552e+01   1.8e-04  ...  +1.000  yes   16        tol    0  GN    0.07 *
  24   3.104552e+01   4.4e-06  ...  +1.000  yes   21        tol    0   N
```

## Decisions

Settled by Nick (2026-11-11): the mode name `"switch"`; allowed with the
line search as well as the trust region; `tau = 0.25` and `k = 2`, exposed
as keywords; one-way; a keyword for the forcing power before the switch,
with defaults of 0.5 before and 1.0 after.

- **D1. The names of the keywords.** **Decided: `gn_hessvec`.** `hessian="full" | "switch"`,
  `gn_hessvec`, `switch_tau`, `switch_k`, `pre_switch_forcing_power`,
  `forcing_power`. Alternative for the operator: `model_hessvec` or
  `approx_hessvec`, since nothing in the method needs it to be
  Gauss-Newton. Any positive semidefinite model whose error is of second
  order in the step works, e.g. one that drops a different curvature term.
  *Recommendation: `gn_hessvec`.* That is what it will be in nearly every
  use, and the docstring can say it may be any positive semidefinite model
  Hessian.
- **D2. The forcing keywords.** (a) A general `forcing_power` (all modes;
  `None` means 0.5 normally and 1.0 after a switch), plus
  `switch_forcing_power=0.5` for the phase before the switch. (b) Two
  switch-only keywords, `pre_switch_forcing_power=0.5` and
  `post_switch_forcing_power=1.0`, with no general one. *Recommendation:
  (a).* A forcing power is useful without the switch: the study's point is
  that 1.0 gives the full Hessian its q-quadratic tail. (a) adds that knob
  to every run, and the switch keyword is then "the other phase's power".
  The cost is a default that depends on the mode, which the docstring must
  state. **Decided:** `forcing_power` and `pre_switch_forcing_power`, with
  the table in [Forcing power](#forcing-power).
- **D3. An invalid sample** (a rounding-level `e_GN`, a non-finite value,
  or a line-search fallback direction): (a) resets the hit count; (b)
  leaves the count unchanged. *Recommendation: (a)*, the conservative
  reading of "k in a row". **Decided: (a).**
- **D4. The rounding gate.** A sample is valid when `e_GN > c_round *
  delta`, with `delta = 10 eps max(1, |f|)`, the same allowance as rho.
  *Recommendation: `c_round = 100`*, a module constant, not a keyword. With
  that value, `e_GN` is at least about `1e-12 |f|` in float64. **Decided:
  as recommended.**
- **D5. Rejected trust-region trials.** *Recommendation was: count them as
  samples.* **Decided: a rejection resets the hit count** (conservative,
  like D3).
- **D6. history_array** keeps its eight columns. The switch fields are on
  the records only. *Recommendation: yes*, so that existing code that
  slices the array keeps working. A `switch_history_array` can be added if
  it is wanted. **Agreed.**
- **D7. Display columns** only in switch mode. *Recommendation: yes*, so
  that the display of other runs does not change. **Agreed.**
- **D8. Tracking starts at iteration 0**, with no delay or cheap
  pre-trigger. *Recommendation: yes, for now.* It is the method the study
  measured. A delayed start is a later option, if the experiments show the
  tracking product matters (see Risks). **Agreed, for now**; a delayed
  start may follow.
- **D9. Safety net.** As in the line-search plan: vendor the current
  package as `tests/nicks_optimizers_baseline/`, and compare the trust-region
  configurations (42 last time) and line-search ones bit for bit. This
  covers `hessian="full"` and `forcing_power=None`. Delete the copy at the
  end. *Recommendation: yes.* **Agreed.**

## Testing

- **Trajectory identity**, the central tests:
  - Before the switch, a switch run's iterates, records (except the switch
    fields) and work are bit for bit those of a `hessian="full"` run with
    `hessvec=gn_hessvec` and `forcing_power=<the pre-switch power>`, because
    tracking never changes the trajectory.
  - After the switch, they are those of a `hessian="full"` run with the
    full Hessian, resumed at the switch point with the switch run's radius
    and `gnorm0`.
  
  Both for the trust region and the line search; CG, Lanczos and flexible;
  `P = M` and separate `P`.
- **The detector** (`criteria.switch_sample` and the driver's counter) on
  scripted sequences:
  - fires after exactly k hits;
  - an invalid or missing sample resets the count;
  - never fires on rounding-level errors;
  - `tau` and `k` are respected.
- **Problems** (`tests/problems.py`):
  - **(i) Regularized extended Rosenbrock**, small n (e.g. 20), from the
    study, with `r`, `J^T J v` and `S v` explicit. The residual is nonzero
    at the solution, so the switch fires and the run beats pure
    Gauss-Newton in Hessian products. Ported from the study's
    `gn_vs_newton_tr.py`.
  - **(ii) The same problem with a zero residual** at the solution (c = 1).
    Gauss-Newton is already fast; the run converges, whether or not it
    switches.
  - **(iii) A linear least-squares problem** (`S = 0`). Then `e_N = e_GN`,
    and the switch never fires.
  - **(iv) The matrix-completion manifold**, with `hessian_action(...,
    curvature=False)` as `gn_hessvec`.
- **Work**: exactly one product with `hessvec` per iteration before the
  switch, rejected trust-region trials included, and none in a
  `hessian="full"` run. Every product is counted in the right counter.
- **Resume**: interrupted before the switch, at the switch iteration, and
  after it, the resumed run is bit for bit the uninterrupted one. A
  switched state with `hessian="full"` raises.
- **Arguments**: every inconsistent combination raises (see the drivers
  section).
- **Display**: the switch-mode text; the default display unchanged
  character for character.
- **Backends**:
  - the strict backend runs switch mode with tiny capacities;
  - the JAX tests compare a compiled switch run with numpy;
  - the pieces after the switch are compiled once, and only after the
    switch fires.
- **Unchanged**: the vendored baseline, bit for bit (D9).

## Phases

Each phase ends with the test suite, mypy --strict and the CI examples
passing, and is committed separately.

| | Phase | Delivers | Gate |
|---|---|---|---|
| 0 | Safety net | baseline vendored, comparison harness | harness passes |
| 1 | Forcing power | `forcing_power` in `CoreProblem`, `forcing_term(..., power)`, the keyword in both drivers (no switch yet) | bit for bit with `None`; power 1.0 tested on a quadratic |
| 2 | Second operator | `gn_hessvec` in `PointOperators`, `CoreProblem.model`, `operators()`, the tracking product in both finish pieces (`track=`), the raw fields | bit for bit without `track`; `full_model` against dense references |
| 3 | Switch, trust region | detector in `criteria`, the two-phase driver, records, summary, resume, display, argument checks | trajectory identity; detector; resume; problems (i) to (iii) |
| 4 | Switch, line search | the same for `line_search_newton` | the same |
| 5 | Backends | strict backend and JAX tests in switch mode | job "jax" |
| 6 | Experiments | `experiments/gn_newton_switch/`: the study's three problems on the library, trust region and line search, work counted by Hessian, wall time | **check in** with Nick on the results |
| 7 | Wrap-up | README section, switch run in `examples/matrix_completion.py`, `dev/TODO.md`; baseline deleted | CI |

## Risks

- **The tracking product costs more than the study charged.** The study
  treated it as free. When the Gauss-Newton solves are short (one to five
  Krylov steps in the early iterations), one full product per iteration is
  30% to 40% more work, and more if a full product costs more than a
  Gauss-Newton one (1.7 times in the study's matrix completion, about 3 to
  4 times in T3Toolbox). Phase 6 measures it. The remedy, if needed, is D8's
  later option: start tracking only once Gauss-Newton is visibly linear
  (its `|1 - rho|` or the ratio of successive gradient norms levels off).
- **The retraction must be of second order** for `e_N` to be `O(||s||^3)`.
  With a first-order retraction, `e_N` stays of order `||s||^2`, and the
  switch may never fire. That is a safe failure (the run stays
  Gauss-Newton), but the docstring must say so.
- **The full Hessian may be indefinite at the switch.** The detector fires
  only when the full model predicts the steps better, which in the study
  happened near the solution, where H is positive definite. The study did
  not check definiteness at the switch point except in 2D. The trust region
  handles indefiniteness. The line search truncates, which is cautious but
  safe.
- **The best tau depends on the problem**: larger values fired earlier,
  which helped matrix completion and hurt the diffusion inversion. `tau` is
  a keyword for that reason. The default is the value that never fired too
  early in the study.

## Progress

- **Plan approved** (2026-11-11), with the decisions above.
