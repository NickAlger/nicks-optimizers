# Minibatch SGD as a preconditioner for matrix completion

**Question.** Can a few epochs of minibatch SGD with Cauchy steps serve as
the preconditioner of trust-region Newton for matrix completion
(`examples/matrix_completion.py`)? Does it help more with nonuniform
sampling, or when the rank continuation is skipped (a cold start at full
rank)? Gauss-Newton or full Hessian inside the SGD? Cyclic or reshuffled
batches?

**Status.** First round done. Negative for SGD: it works (the flexible
solver makes it safe) but costs more than it saves. Block Jacobi helps with
nonuniform sampling. No preconditioner rescues a cold start at full rank.

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

    python experiments/sgd_preconditioner/spectra.py
    python experiments/sgd_preconditioner/nonlinearity.py
    python experiments/sgd_preconditioner/accuracy.py
    python experiments/sgd_preconditioner/study.py --problem uniform --strategy continuation
    (--problem uniform|powerlaw, --strategy continuation|cold, --seeds, --budget, --json)

`results.jsonl` holds the raw results of the study runs below.

## Results

### How much room is there? (`spectra.py`)

Dense spectra on the tangent space (Frobenius metric), 120 x 100, rank 4,
4 samples per degree of freedom, singular values 2^-k:

| sampling | point | Gauss-Newton: condition number | full Hessian |
|---|---|---|---|
| uniform | fitted | 26 | same as Gauss-Newton |
| uniform | stage start, s_r = 1e-6 s_1 | 23 | eigenvalues in [-3e4, 3e4], 96 negative |
| power law (exponent 1) | fitted | 220 | same as Gauss-Newton |
| power law | stage start | 290 | eigenvalues in [-4e4, 4e4] |

With uniform sampling the Gauss-Newton Hessian is already well conditioned
on the tangent space (sampling restricted to the tangent space is close to
a scaled isometry); power-law sampling leaves about ten times more room. The
extreme spectrum at stage starts comes from the curvature term ||R|| / s_r,
which a Gauss-Newton preconditioner does not see.

### The preconditioner as a map (`nonlinearity.py`, fitted point of the example)

| preconditioner | homogeneity defect | additivity defect | symmetry defect | min <b, P b> / norms | residual of H_GN p = b |
|---|---|---|---|---|---|
| block Jacobi | 2e-16 | 2e-16 | 6e-18 | 0.91 | 0.37 |
| SGD GN cyclic, 1 epoch x 10 batches | 4e-16 | 1.4e-2 | 5e-3 | 0.80 | 0.73 |
| SGD GN cyclic, 3 x 10 | 4e-16 | 1.4e-2 | 4e-3 | 0.77 | 0.76 |
| SGD GN cyclic, 1 x 50 | 4e-16 | 2.6e-2 | 5e-3 | 0.76 | 0.86 |
| SGD GN reshuffled, 1 x 10 | 4e-16 | (0.59: randomness) | 6e-3 | 0.80 | 0.73 |

- The Cauchy-step map is homogeneous of degree one, P^{-1}(c b) = c P^{-1}(b)
  for every real c (to rounding), as argued: the step lengths are ratios of
  quadratics in the gradient. <b, P^{-1} b> was positive for every sample,
  so the set { b : <b, P^{-1} b> <= 1 } was star-shaped.
- With cyclic batches the map is nearly linear (1 to 3% additivity defect,
  0.5% symmetry defect). With reshuffled batches, two applications differ by
  about 60%: the additivity column then measures randomness.
- As an approximate inverse of the Gauss-Newton Hessian it is weak, and more
  epochs do not help (`accuracy.py`): constant-step SGD stalls at a noise
  floor (residual 0.5 to 0.77). Each batch here holds fewer sampled entries
  than the tangent space has dimensions (4 samples per degree of freedom,
  10 batches), so every minibatch Hessian is singular. Averaging the last
  epoch's iterates brings the residual to 0.35 (about 6 passes per
  application); SVRG to 0.27 (10 passes) or 0.19 (38 passes). Block Jacobi
  reaches 0.37 for almost nothing per application.

### Trust-region Newton (`study.py`; 1000 x 800, rank 8, 4 samples per degree of freedom)

Rank continuation, passes over the samples / Krylov steps / time. Every run
reached the same fit (1.15 and 1.25 times the best rank-8 error); ranges are
over 3 seeds for reshuffled batches.

| preconditioner | uniform | power law |
|---|---|---|
| CG, P = M = I (the example) | 474 / 262 / 4.3 s | 1209 / 962 / 13.6 s |
| flexible, P = I | 498 / 303 / 5.1 s | 1238 / 1014 / 17.0 s |
| flexible, block Jacobi | 532 / 261 / 8.9 s | **821 / 525 / 13.4 s** |
| SGD GN cyclic 1 x 10 | 1306 / 444 / 11 s | 1668 / 569 / 15 s |
| SGD GN reshuffled 1 x 10 | 1068-1144 / 362-388 / 10-11 s | 1660-1782 / 566-608 / 16-17 s |
| SGD GN cyclic 3 x 10, averaged | 1444 / 256 / 11 s | 2350 / 392 / 17 s |
| SGD GN reshuffled 3 x 10, averaged | 1399-1615 / 250-282 / 11-13 s | 2220-2428 / 374-405 / 17-18 s |
| SGD GN cyclic 3 x 4, SVRG | 1998 / 247 / 11 s | 3599 / 404 / 21 s |
| SGD full Hessian reshuffled 1 x 10 | 3552-3890 / 1182-1295 / 42-48 s | 3701-3800 / 1231-1264 / 46-47 s |

Cold start at rank 8 from X = 0, stopped after 150 iterations or 200 s;
final error / best rank-8 error (iterations done, passes):

| preconditioner | uniform | power law |
|---|---|---|
| CG | 68 (150, 3818) | 50 (150, 3760) |
| flexible, P = I | 23 (150, 6143) | 45 (150, 6714) |
| flexible, block Jacobi | 13 (150, 6266) | 38 (150, 7194) |
| SGD GN cyclic 1 x 10 | 33 (120 in 200 s) | 52 (135) |
| SGD GN reshuffled 1 x 10 | 41 (114) | 46 (123) |
| SGD GN cyclic 3 x 10, averaged | 26 (98) | 46 (114) |
| SGD GN reshuffled 3 x 10, averaged | 33 (85) | 49 (104) |
| SGD GN cyclic 3 x 4, SVRG | 101 (52) | 50 (87) |
| SGD full reshuffled 1 x 10 | 28 (91) | 48 (99) |

## Findings

1. **SGD costs more than it saves.** It cuts Krylov steps with power-law
   sampling (962 to 374-608) but each application costs 1 to 4 passes, so
   the total is 1.4 to 3 times the passes of plain CG (2.3 to 4.2 times with
   uniform sampling). With uniform sampling it does not cut Krylov steps at
   all (plain SGD even needs more, 362-444 against 262), as the spectra
   predicted.
2. **Block Jacobi is the useful preconditioner here**: with power-law
   sampling, 32% fewer passes and 45% fewer Krylov steps than CG at equal
   wall time; with uniform sampling no gain.
3. **Gauss-Newton, not the full Hessian**, inside the SGD: the full-Hessian
   version needed 2 to 3.6 times the Krylov steps and 2 to 3.6 times the
   passes of the same configuration with Gauss-Newton.
4. **Cyclic or reshuffled made no consistent difference**, within the spread
   over seeds; randomness did not reveal useful directions here.
5. **No preconditioner rescues the cold start.** The best, block Jacobi,
   ended at 13 times the best error after 150 iterations (uniform), against
   1.15 times in about 90 iterations with continuation; with power-law
   sampling every run ended at 38 to 52 times. This fits the spectra: at
   full rank the difficulty is the curvature term (indefinite, of size
   ||R|| / s_r), which Gauss-Newton preconditioning leaves alone. The rank
   continuation remains essential. (The flexible solver alone did better
   than CG on the cold start, 23 against 68 times the best error, at 1.6
   times the passes: it solves the boundary subproblems more accurately.)

## Open questions

- SGD may pay off with heavy oversampling (K much larger than the dimension),
  where minibatch Hessians are nonsingular, or when a full pass over the
  data is expensive (out of core, streaming). The oversampling ratio is the
  dial to turn.
- The cold start needs something aimed at the curvature term, e.g. a metric
  scaled by the singular values (scaled gradients on Grassmann manifolds,
  Ngo and Saad 2012; Riemannian preconditioning, Mishra and Sepulchre 2016).
- Block Jacobi could graduate to the matrix-completion example as an option
  for nonuniform sampling, after tests on more problems.
