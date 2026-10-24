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
