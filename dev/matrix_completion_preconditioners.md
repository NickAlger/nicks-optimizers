# Preconditioning matrix completion: minibatch SGD, block Jacobi, cold starts

October 2026. Code, scripts and the complete tables:
[`experiments/sgd_preconditioner/`](../experiments/sgd_preconditioner/).

## Summary

- Minibatch SGD with Cauchy steps, used as the preconditioner of the
  trust-region method (through the flexible subproblem solver), is safe but
  not worth its cost on matrix completion: it cuts Krylov steps when the
  sampling is nonuniform, yet each application costs 1 to 4 passes over the
  data, so a run needs 1.4 to 4.2 times the passes of plain CG.
- Block Jacobi on the rows of the factors (the alternating-least-squares
  normal equations) is the preconditioner that helps: with power-law
  sampling it saves 32% of the passes and 45% of the Krylov steps; with
  uniform sampling it neither helps nor hurts.
- No preconditioner makes a cold start at full rank work. Rank continuation
  stays essential: it reached 1.15 times the best rank-8 error, while the
  best cold start (block Jacobi) ended at 13 times after 150 iterations.
- The reasons are visible in the Hessian spectra: with uniform sampling the
  Gauss-Newton Hessian is already well conditioned on the tangent space, and
  the hard part at full rank is the curvature term, which Gauss-Newton
  preconditioners do not see.

## Setup

Matrix completion as in `examples/matrix_completion.py`: a 1000 x 800 matrix
with singular values 2^-k, fitted at rank 8 from 4 sampled entries per degree
of freedom (57,344 entries), on the fixed-rank manifold with the Frobenius
metric. Two samplings: uniform, and power law (rows and columns drawn with
probability proportional to (index + 1)^-1, after 8 uniform entries per row
and column to keep the fit identifiable; 18 to 781 entries per row). Two
strategies: rank continuation (ranks 1, ..., 8 in turn, as the example does)
and a cold start at rank 8 from X = 0.

Cost is counted in passes over the sampled entries: one Hessian product, one
gradient or one objective evaluation is one pass, and a preconditioner's work
is counted in the same unit. Wall times are reported too.

**The SGD preconditioner.** For P^{-1} b, minimize 0.5 <H p, p> - <b, p> over
tangent vectors by minibatch SGD from p = 0. Each step moves along the
negative minibatch gradient to the minimizer of the minibatch quadratic on
that ray (the Cauchy step, one minibatch Hessian product). Variants:
Gauss-Newton or full Hessian for the minibatch quadratic; cyclic batches (a
fixed map) or batches reshuffled for every epoch (a stochastic map); plain,
with the last epoch averaged, or variance-reduced (SVRG). The SGD gradients
are canonical covectors, so the iterates respect the gauge without a vector
projection.

## Results

**Spectra** (dense, small instance: 120 x 100, rank 4). Condition number of
the Gauss-Newton Hessian on the tangent space:

| sampling | at the fit | at a stage start (new singular value 1e-6 s_1) |
|---|---|---|
| uniform | 26 | 23 |
| power law | 220 | 290 |

At the fit the full Hessian has the same spectrum. At a stage start its
eigenvalues spread over [-3e4, 3e4] (uniform) and [-4e4, 4e4] (power law),
with 96 negative ones: the curvature term ||R|| / s_r.

**The SGD map.** At the fitted point:

- It is homogeneous of degree one, P^{-1}(c b) = c P^{-1}(b) for all real c,
  to rounding: the Cauchy step lengths are ratios of quadratics in the
  gradient. <b, P^{-1} b> was positive for every sample, so its unit set is
  star-shaped.
- With cyclic batches it is nearly linear: 1 to 3% additivity defect, 0.5%
  symmetry defect. Reshuffled applications differ from each other by about
  60%.
- As an approximate inverse of the Gauss-Newton Hessian it is weak. The
  residual ||H p - b|| / ||b|| stalls at 0.5 to 0.77, whatever the number of
  epochs. Averaging the last epoch reaches 0.35 at about 6 passes per
  application; SVRG reaches 0.27 at 10 passes and 0.19 at 38. Block Jacobi
  reaches 0.37 at almost no cost per application.

**Trust-region runs with rank continuation.** Passes over the data / Krylov
steps; every run reached the same fit (1.15 and 1.25 times the best rank-8
error with uniform and power-law sampling):

| preconditioner | uniform | power law |
|---|---|---|
| plain CG (the example) | 474 / 262 | 1209 / 962 |
| block Jacobi | 532 / 261 | **821 / 525** |
| SGD, Gauss-Newton, 1 epoch of 10 batches (best seed) | 1068 / 362 | 1660 / 566 |
| SGD, Gauss-Newton, 3 epochs, last averaged (best seed) | 1399 / 250 | 2220 / 374 |
| SGD, full Hessian, 1 epoch of 10 batches (best seed) | 3552 / 1182 | 3701 / 1231 |

**Cold starts.** Final error relative to the best rank-8 error after 150
iterations (or 200 s): plain CG 68 (uniform) and 50 (power law); the flexible
solver without a preconditioner 23 and 45; block Jacobi 13 and 38; the SGD
variants 26 to 101 and 46 to 52.

## Interpretation

1. **Uniform sampling leaves little to precondition.** Restricted to the
   tangent space of an incoherent low-rank matrix, uniform sampling acts
   almost like a multiple of the identity, which is the property matrix
   completion theory rests on. Plain CG needs only about 2.5 Krylov steps
   per outer iteration, and a preconditioner can save little of that.
2. **Minibatches are too small to be informative here.** At 4 entries per
   degree of freedom, a tenth of the data has fewer entries than the tangent
   space has dimensions, so each minibatch Hessian is singular and Cauchy
   steps based on it overshoot in the directions the batch does not see.
   Constant-step SGD then stalls at a noise floor. Averaging and variance
   reduction remove the floor, but only by spending passes.
3. **Nonuniform sampling is a row and column imbalance**, which is close to
   block diagonal in the factor rows. Block Jacobi corrects it exactly where
   it lives, and costs almost nothing per application; SGD corrects it only
   statistically.
4. **The cold start fails for a reason no Gauss-Newton preconditioner
   addresses.** At full rank, components that have not been found yet have
   tiny singular values, and the curvature term (indefinite, of size
   ||R|| / s_r) dominates the Hessian. The trust region then has to stay
   small. Rank continuation avoids this: each new component starts with the
   residual already fitted by the others, and the radius starts at the scale
   of the smallest singular value found so far.
5. **Gauss-Newton is the right quadratic for an SGD preconditioner.** Each
   sampled entry contributes a nonnegative term, so minibatch Cauchy steps
   are defined; the full Hessian's per-entry curvature terms have either
   sign. The full-Hessian SGD needed 2 to 3.6 times the Krylov steps.
6. **Randomness did not help.** Cyclic and reshuffled batches gave the same
   results within the spread over seeds.

What the experiment does show about the method: the flexible subproblem
solver made every one of these preconditioners safe, including the
stochastic and nonlinear ones, and it beat plain CG on the cold start (23
against 68 times the best error) because it solves the boundary subproblems
more accurately.

## Open questions

- **Heavy oversampling.** With many more entries than unknowns, minibatch
  Hessians are nonsingular and SGD becomes a real approximate inverse; it may
  also pay off when a full pass is expensive (out-of-core or streaming data).
  The oversampling ratio is the dial to turn.
- **The curvature term.** A metric or preconditioner scaled by the singular
  values (scaled gradients on Grassmann manifolds, Ngo and Saad 2012;
  Riemannian preconditioning, Mishra and Sepulchre 2016) could make cold
  starts at full rank viable.
- **Block Jacobi in the example.** It could become an option of
  `examples/matrix_completion.py` for nonuniform sampling, after tests on more
  problems.
