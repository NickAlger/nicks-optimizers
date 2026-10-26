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
