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
| [`jax_backend/`](jax_backend/) | Does the backend design (dev/jit_backends_plan.md) work with JAX, and what does it buy? | CG: same iterates, one compilation per run, up to 2.3 times faster per iteration on CPU. Lanczos and flexible: correct with growing bases, not faster than numpy yet (fixed capacity costs work; chunking next) |
