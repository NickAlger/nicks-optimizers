# Switching from Gauss-Newton to the full Hessian

**Question.** On the switching study's problems, run through the library
(`hessian="switch"`, dev/gauss_newton_switch_plan.md), how much work does
the switch save against pure Gauss-Newton and against the full Hessian from
the start, with the trust region and with the line search? And what do the
defaults (tau = 0.25, k = 2, forcing power 0.5 before the switch and 1.0
after it) cost?

**Status.** First round (the plan's phase 6 check-in), CPU, numpy backend.
With the defaults the switch beats pure Gauss-Newton in 41 of the 44 runs on
the diffusion inversion and matrix completion, by up to 94%, and every
switch run converged, where pure Gauss-Newton did not within 500 iterations
in 12 of the 58 runs. On Rosenbrock the default switch costs more than
Gauss-Newton in 10 of 14 runs. A forcing power of 0.5 after the switch was
cheaper than the default 1.0 in 46 of 58 runs (1.0 cheaper in 5, by 0% to
15%). Against the full Hessian from the start the switch is mixed: it
usually wins with the line search and usually loses or ties with the trust
region.

    python compare.py [--only rosenbrock,diffusion,mc] [--jobs 8]   # appends to results.jsonl, resumes
    python summarize.py > tables.md

## Problems and setup

The study's three problems (artifact "GN vs Newton switching: figures";
Nick Alger and Claude, November 2026):

- **rosenbrock**: the regularized extended Rosenbrock problem
  (`tests/problems.py:regularized_rosenbrock`) in the study's seven
  configurations (n, alpha, c-scale, seed).
- **diffusion**: inversion for the log-conductivity in
  `-div(exp(m) grad u) = 1` on a 32 x 32 grid (1024 parameters), 150 noisy
  point observations, contrast 3, regularization beta = 1e-4 to 1e-9
  (`study_problems.py`, ported from the study). Cost: all PDE solves (state,
  adjoint, products: a Gauss-Newton product costs 2, a full product 3).
- **mc**: rank-10 matrix completion, 200 x 200, 25% observed, 1% noise,
  lam = 1e-3, and the study's 16 variations (`study_problems.py`). Cost:
  Hessian products, unweighted; a full product took 1.6 times a
  Gauss-Newton one, and the last column of the table weighs it so.

Every run uses the library defaults (initial radius 1, eta = 1e-4) except
rtol = 1e-9 (relative to `||g(x0)||`, the study's stop), max_iter = 500
and max_krylov = 200. Five variants per case and method:

| variant | model Hessian | forcing power |
|---|---|---|
| gn | Gauss-Newton | 0.5 |
| switch | `hessian="switch"`, defaults | 0.5, then 1.0 |
| switch p=0.5 | `hessian="switch"`, `forcing_power=0.5` | 0.5, then 0.5 |
| full | full | 0.5 |
| full p=1 | full | 1.0 |

The cost of the switch runs includes the tracking products (one full
product per iteration before the switch).

## Results

[`tables.md`](tables.md) has every run: the cost and iterations of each
variant, the iteration of the switch, and the savings. In summary (counts
of runs where the first is cheaper):

| | rosenbrock (14) | diffusion (12) | mc (32) | all (58) |
|---|---|---|---|---|
| switch < gn | 4 | 11 | 30 | 45 |
| switch p=0.5 < gn | 9 | 11 | 31 | 51 |
| switch p=0.5 < switch | 14 | 9 | 23 | 46 |
| switch < the better full run | 1 | 3 | 7 | 11 |
| switch p=0.5 < full | 6 | 9 | 12 | 27 |
| gn did not converge (500 its) | 0 | 1 | 11 | 12 |

By method: switch p=0.5 is cheaper than full (power 0.5) in 21 of 29
line-search runs and 6 of 29 trust-region runs.

Observations:

- **Against Gauss-Newton, mc and diffusion.** On matrix completion the
  default switch saves 52% to 94% in 28 of 32 runs (lower bounds where
  Gauss-Newton did not converge), and with the full products weighted 1.6
  still 25% to 92% in 28 of 32. The exceptions: lam = 1e-5 (-9% trust
  region, +11% line search) and rank 5 with the line search (-73%; +19%
  with power 0.5). On the diffusion inversion it saves 30% to 76% with the
  trust region and 20% to 91% with the line search, more the smaller beta,
  except beta = 1e-9 with the line search (-40%; -5% with power 0.5): the
  detector fired at iteration 10 and the full-Hessian line search then took
  103 iterations.
- **Rosenbrock, trust region.** The detector fires one or two iterations
  before convergence in 6 of 7 runs (iteration 31 to 35 of 33 to 37; with
  alpha = 1, 41 of 54). Gauss-Newton with the library's relative forcing
  term converges fast enough that little is left to gain, and the tracking
  products (one per iteration until the switch, a median 16% of the run's
  products) and a tight final solve make the switch cost 17% to 52% more in
  5 of 7 runs (10% and 20% less in the other two). The study had found 16%
  to 38% savings here, with an absolute forcing term (sqrt(||g||)), a
  radius cap of 1e3 and cancellation-free differences of f.
- **Rosenbrock, line search.** The detector fires early (iteration 3 to 13),
  and with the default power 1.0 the full-Hessian phase spends more
  products than it saves: worse than Gauss-Newton in 5 of 7 runs. With
  power 0.5 the same switch is cheaper than Gauss-Newton in all 7, by 19% to
  31%.
- **The power after the switch.** 1.0 tightens every solve after the
  switch; on these problems that rarely bought enough fewer iterations, and
  0.5 was cheaper in 46 of 58 runs. 1.0 was cheaper in 5 (by 0% to 15%). On
  mc with the trust region the two are often identical: every solve after
  the switch stops on the boundary or at max_krylov, so the forcing term
  never binds.
- **Against the full Hessian from the start.** The study found full Newton
  23% to 55% worse than switching on the ill-posed diffusion problems. Here
  the library's full-Hessian trust region is close to the switch there (the
  better switch run is 14% worse to 11% better over beta = 1e-4 to 1e-9),
  while with the line search the switch wins at every beta (by 3% to 23%).
  On mc with the trust region the better full run is cheaper in 9 of 16
  cases (by 14% to 125%) and within 1% in 6, those where the detector fired
  at iteration 2 to 4. With the line search the switch (power 0.5) is cheaper
  than the full run in 21 of 29 runs over all problems, by up to 64%; its
  worst case is mc rank 5 (4.4 times the full run's products).
- **Where it fires.** At iteration 2 to 4 on 6 of the 16 mc trust-region
  runs (the run is then the full run plus a few tracking products), one or
  two iterations before the end on Rosenbrock with the trust region, and in
  between elsewhere. It never fired on the tests' linear least-squares
  problem, as it should not.

Caveats: one seed per configuration (as in the study); unweighted product
counts except where noted; f(x + s) - f(x) by subtraction, as the library
computes it (the study used cancellation-free differences).

## Questions for the check-in

1. The default power after the switch: 1.0 (decided) or 0.5, which was
   cheaper in 46 of 58 runs here?
2. Rosenbrock with the trust region: the switch costs 17% to 52% more than
   Gauss-Newton in 5 of 7 runs, mostly the tracking products of a switch
   that comes too late to help. A delayed start of the tracking (D8's later option) would
   remove most of it.
3. The switch against the full Hessian from the start is less favorable
   than in the study, especially with the trust region. Is that a reason to
   revisit tau, or just a property of these problems with this solver?
