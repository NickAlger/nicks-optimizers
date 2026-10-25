# Minibatch SGD as a preconditioner for matrix completion

**Question.** Can a few epochs of minibatch SGD with Cauchy steps serve as
the preconditioner of trust-region Newton for matrix completion
(`examples/matrix_completion.py`)? Does it help more with nonuniform
sampling, or when the rank continuation is skipped (a cold start at full
rank)? Gauss-Newton or full Hessian inside the SGD? Cyclic or reshuffled
batches?

**Status.** RESULTS PENDING

## The preconditioner

When the trust-region solver asks for P^{-1} b, approximately minimize
0.5 <H p, p> - <b, p> over tangent vectors p by minibatch SGD from p = 0: each
step goes along -g, g = H_B p - b with H_B the Hessian restricted to a batch
B of the sampled entries (scaled by K / |B|), to the minimizer of the
minibatch quadratic along that ray (the Cauchy step, one minibatch Hessian
product). Choices (`preconditioners.py`):

- **Gauss-Newton or full Hessian for H_B.** Gauss-Newton: each sampled entry
  contributes a nonnegative term, so every minibatch quadratic is convex
  and the Cauchy step is defined. The full Hessian's curvature term has
  either sign per entry and is huge at stage starts (below).
- **Cyclic or reshuffled batches.** Cyclic: a fixed partition in a fixed
  order, a deterministic map. Reshuffled: a new random partition for every
  epoch of every application (from a seeded generator).
- **Against the noise floor:** averaging the last epoch's iterates, or SVRG
  variance reduction (one full product per epoch).
- Step cap alpha <= 10 alpha_0, alpha_0 the first Gauss-Newton Cauchy step:
  scale-invariant, so the map stays homogeneous.
- The SGD gradients are canonical covectors, so the iterates are valid
  tangent vectors (U^T dU = 0); no vector projection is needed.

The flexible subproblem solver (`krylov="flexible"`, see
`experiments/flexible_preconditioners`) makes any such P safe to use.

Baseline: **block Jacobi** on the rows of the factors, the block diagonal of
the Gauss-Newton Hessian (the normal equations of alternating least
squares): linear, cheap, aimed at the imbalance of row and column counts.

## Scripts

    python experiments/sgd_preconditioner/nonlinearity.py
    python experiments/sgd_preconditioner/accuracy.py
    python experiments/sgd_preconditioner/study.py --problem uniform --strategy continuation
    (--problem uniform|powerlaw, --strategy continuation|cold, --seeds, --budget, --json)

`/tmp/claude-0/-home-claude/f173fb43-8092-5fb9-ac2a-935ec4724e87/scratchpad/mc_conditioning.py`
measured the Hessian spectra below (a scratch script, not kept).
