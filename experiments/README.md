# Experiments

Exploratory studies. The code here is not part of the tested surface: it is
not run in CI or type-checked, it may change or disappear, and its
conclusions are provisional.

The repository has three tiers:

| folder | what it holds | standard |
|---|---|---|
| `src/` | the library | solid: tested, `mypy --strict` |
| `examples/` | demonstrations of the library | solid: tested, `mypy --strict`, run in CI |
| `experiments/` | open questions | exploratory |

An experiment that pans out and is validated graduates: its algorithmic
pieces move into `src/` with tests, and a demonstration moves into
`examples/`.

Each experiment has its own folder with a README stating the question, how to
run it, the results so far, and its status.

| experiment | question | status |
|---|---|---|
| [`flexible_preconditioners/`](flexible_preconditioners/) | Can the trust-region subproblem be made robust to preconditioners that are nonsymmetric, nonlinear (inexact inner solves) or stochastic? | in progress |
| [`sgd_preconditioner/`](sgd_preconditioner/) | Can minibatch SGD with Cauchy steps precondition matrix completion, with nonuniform sampling or a cold start at full rank? | first round done: SGD costs more than it saves; block Jacobi helps with nonuniform sampling; cold starts not rescued |
| [`jax_backend/`](jax_backend/) | Does the backend design (dev/jit_backends_plan.md) work with JAX, and what does it buy? | graduated to `examples/jax_p_laplacian.py`. CG: same iterates, one compilation per run, up to 2.3 times faster per iteration on CPU. Lanczos and flexible: same iterates; with each solve starting small and doubling its basis, 0.5 to 1.1 times numpy's time per iteration; chunked block products do not pay on CPU |
| [`line_search_vs_trust_region/`](line_search_vs_trust_region/) | Where does line_search_newton spend less work than trust_region_newton, and where more? | first round: the line search wins on the convex p-Laplacian with a separate or weak preconditioner (up to 4.5 times fewer iterations); the trust region on Rosenbrock's curved valley (half the Hessian products); close elsewhere |
| [`lbfgs_checkin/`](lbfgs_checkin/) | lbfgs at its phase-2 check-in: against the Newton methods, the metric across meshes, skips and restarts, the angle test | first round: the H^1 metric makes lbfgs nearly mesh-independent (25 to 49 iterations for N = 15 to 127, against 150 to 1750 with the Euclidean metric); fewest derivative evaluations of the three methods, but three solves per iteration; no restarts or fallbacks at the defaults |
| [`lbfgs_metrics/`](lbfgs_metrics/) | lbfgs with a constant metric (general and fixed paths) against a point-dependent one (Kacanov), on the p-Laplacian | first round, one problem: the fixed path gives the same run with a third of the solves (60% of the time); the Kacanov metric needs the fewest gradients and with memory 1 is mesh-independent (17 iterations), but with the default memory degrades on fine meshes, and its factorizations make it slower than the constant metric here. Both kinds stay supported |
| [`lbfgs_memory/`](lbfgs_memory/) | What do lbfgs's memory and transport add over the metric alone (against memory 0, steepest descent in M with Barzilai-Borwein steps)? | first round: with the Kacanov metric the memory adds nothing (memory 0 is best, 14 or 15 iterations at every mesh); with H1 it saves 5 to 17%; where M leaves much of the Hessian out it matters, also for a point-dependent M (Rosenbrock with the Jacobi metric: 1936 iterations against 110 to 187); the transport is essential in fixed-rank completion; on the sphere no difference beyond rounding (corrected: single long runs spread by some 15%) |
| [`lbfgs_step_length/`](lbfgs_step_length/) | Is memory 0's weakness its step length? Directions and step lengths of memories 0 and 1 crossed, 8 starts each | first round: memory 0's frozen gamma is why it fails on Rosenbrock, but memory 1 stays 4 to 9 times faster than a memory 0 that refits gamma; on the convex problems memory 1's gain is its direction, not its length; memory 10 is 1.4 to 1.9 times faster than memory 1 where the metric is poor, and no faster with H1 or Kacanov |
