# lbfgs: is memory 0's weakness its step length?

**Question.** In `experiments/lbfgs_memory/`, going from memory 0 to memory 1
helped far more than going from memory 1 to more. Memory 0 is steepest
descent in M, with gamma fitted to each pair that passes the curvature test
and kept otherwise. Is memory 1's advantage just a better step length? And
is the small gain beyond memory 1 real?

**Run:**
- `python experiments/lbfgs_step_length/step_length.py --validate` checks the
  driver against lbfgs.
- `python experiments/lbfgs_step_length/step_length.py [--json results.json] [--starts 8] [--only sphere ...]`
  runs the arms.

`results.json` holds every run.

## Design

The arms cross memory 0's and memory 1's directions with their step lengths.
Here the length means the trial step at alpha = 1, before the line search.
G = ||g||^2_{M^{-1}}, and H1 is memory 1's operator.

| arm | direction | step length |
|---|---|---|
| A | memory 0: -M^{-1} g | gamma (memory 0 as in lbfgs) |
| B1 | memory 0 | t minimizing memory 1's model along -M^{-1} g: t = G / <M^{-1} g, B1 M^{-1} g>, with B1 = H1^{-1} |
| B2 | memory 0 | t giving memory 1's predicted decrease: t = <g, H1 g> / G |
| C1 | memory 1: d1 = -H1 g | c minimizing memory 0's model (B0 = M / gamma) along d1: c = gamma <g, H1 g> / \|\|d1\|\|_M^2 |
| C2 | memory 1 | c giving memory 0's predicted decrease: c = gamma G / <g, H1 g> |
| D | memory 1 | 1 (memory 1 as in lbfgs) |
| E1 | memory 0 | gamma; when a pair fails the curvature test, gamma is refitted anyway as \|\|s\|\|_M / \|\|y\|\|_{M^{-1}}, which is positive whatever the sign of <y, s> |
| E2 | memory 0 | gamma; doubled when a pair fails the curvature test after a unit step |
| m1, m3, m10 | lbfgs with memory 1, 3 and 10, for reference | |

- **Pairs.** B, C and D keep memory 1's pair and test it as memory 1 does.
  Every arm refits gamma to each pair that passes.
- **Empty memory.** Every arm then steps along -gamma M^{-1} g.
- **Everything else is lbfgs's:** the backtracking line search from alpha =
  1, the curvature test, the angle test and fallback, restarts, the stopping
  test (||g||_{M^{-1}} <= 1e-8 ||g0||), and the transport on the sphere.
- **Problems:** those of `experiments/lbfgs_memory/` without completion. All
  runs allow 20000 iterations.

**Validation.**
- Arm A reproduces lbfgs with memory 0 exactly on every problem, so column
  A is also lbfgs's memory 0.
- Arm D matches memory 1's first 40 iterates to rounding (at most 1.5e-9).
- Over whole runs from the same 8 starts, D's medians and lbfgs's own
  memory-1 medians (column m1) agree within the spread described next:
  233 and 236 on Rosenbrock, 1382 and 1277 on the p-Laplacian, 2035 and
  2179 on the sphere, and equal on the short runs.

**Rounding sets the precision of a single run.**
- Over thousands of iterations, rounding differences grow until single
  iteration counts are unreliable. Moving x0 by 1e-12 changes lbfgs's
  memory-1 counts across these ranges (column m1 below):
  - Rosenbrock: 203 to 255;
  - p-Laplacian, M = I: 1096 to 1506;
  - sphere: 2014 to 2441.
- So each arm runs from 8 starts: x0, and x0 moved by 1e-12 times a random
  vector, renormalized on the sphere.
- The table gives the median and, in parentheses, the range.
- Short runs (the convex quartic, H1, Kacanov) barely move.

## Results

Iterations: median over 8 starts (range).

| problem | A (m0) | B1 | B2 | C1 | C2 | D (m1) | E1 | E2 | m1 | m3 | m10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| convex quartic, M = I | 94 | 108 (96–122) | 120 (114–134) | 237 (230–250) | 109 (101–112) | 86 (86–87) | 94 | 94 | 86 (86–87) | 79 | 75 |
| Rosenbrock, M = I | fails | 2158 (1462–2382) | 859 (729–999) | fails | 7016 (6952–7683) | 233 (212–295) | 2006 (1828–2255) | 13058 (12745–13481) | 236 (203–255) | 163 (149–171) | 127 (124–130) |
| Rosenbrock, Jacobi M(x) | 2018 (1840–2235) | 1661 (1311–1822) | 1775 (679–2131) | fails | fails | 181 (167–189) | 802 (661–912) | 955 (820–997) | 185 (167–196) | 130 (123–150) | 116 (114–122) |
| p-Laplacian, M = I | 2624 (1875–2855) | 4118 (3447–4900) | 2834 (2416–2957) | 12548 (11908–13852) | 1675 (1435–1776) | 1382 (1165–1544) | 2624 (1875–2855) | 2624 (1875–2855) | 1277 (1096–1506) | 1084 (909–1152) | 795 (730–862) |
| p-Laplacian, H1 | 48 | 46 | 45 | 64 (59–67) | 50 | 43 | 48 | 48 | 43 | 41 | 41 |
| p-Laplacian, Kačanov M(u) | 14 | 15 | 16 | 17 | 16 | 17 | 14 | 14 | 17 | 22 (21–22) | 22 |
| sphere | 3435 (2634–3967) | 5597 (4167–6773) | 2466 (2106–2983) | fails | 2349 (2083–2564) | 2035 (1907–2336) | 3435 (2634–3967) | 3435 (2634–3967) | 2179 (2014–2441) | 2045 (1881–2308) | 1568 (1481–1853) |

"fails": no start converged within 20000 iterations.

**Diagnostics** (medians over the starts; per run in `results.json`):
- **Pairs that fail the curvature test, memory 0 (A):** 93% of iterations
  on Rosenbrock with M = I and 70% with the Jacobi metric; none on the other
  problems. On Rosenbrock with M = I, gamma freezes at 4.9e-5 after 11
  iterations, and the steps are so short that <y, s> is about -6e-7
  relative. E1 brings the skip rate down to 2% and 5%.
- **Length of B's step relative to gamma:** 1.3 to 2.3 with the poor
  metrics (except B2 with the Jacobi metric, 1.05); 1.07 to 1.33 with H1
  and Kacanov. Memory 1's model asks for longer steps along -M^{-1} g than
  gamma gives.
- **Length of C1's step relative to memory 0's:** 0.08 to 0.55 with the
  poor metrics. Memory 0's model cuts memory 1's direction short.

## Findings

**1. The frozen gamma is why memory 0 fails on Rosenbrock, but it explains
only part of the jump to memory 1.**
- Refitting gamma on rejected pairs (E1) takes Rosenbrock with M = I from
  failure to 2006 iterations, and with the Jacobi metric from 2018 to 802.
- Memory 1 still needs 8.6 and 4.4 times fewer iterations (233 and 181).
- Doubling gamma (E2) also helps, but less.
- On the convex problems no pair is rejected, so gamma never freezes and A,
  E1 and E2 coincide.

**2. On the convex problems, memory 1's gain comes from its direction, not
its step length.**
- Memory 1's direction at memory 0's predicted decrease (C2) keeps most of
  the gain:
  - p-Laplacian with M = I: 1675, against 2624 for A and 1382 for D;
  - sphere: 2349, against 3435 and 2035.
- Memory 0's direction at memory 1's length (B) mostly does not:
  - p-Laplacian: 2834 and 4118;
  - sphere: 2466 and 5597.
  - The exception is B2 on the sphere, which gets about two thirds of the
    way.
- B1, the exact minimizer of a model along the steepest direction, is the
  worst of the memory-0 lengths. This fits what is known of steepest
  descent: steps that minimize along the gradient zigzag, and
  Barzilai-Borwein's lagged steps do better.

**3. On Rosenbrock, neither half works alone.**
- With M = I, B2 needs 859 iterations and C2 7016, against 233 for memory
  1.
- With the Jacobi metric, B needs 1661 to 1775, and C fails.
- Memory 1's direction at memory 0's lengths (C1, C2) inherits the frozen
  gamma.

**4. Memory beyond 1 does help, once rounding is accounted for.** All
the numbers here are lbfgs's own (columns A, m1, m3, m10).
- From memory 1 to memory 10:
  - Rosenbrock: 1.9 times fewer iterations with M = I, 1.6 with the Jacobi
    metric;
  - p-Laplacian with M = I: 1.6;
  - sphere: 1.4 (memory 3 is within memory 1's spread there);
  - convex quartic: 1.15.
- With the good metrics it does not: H1 goes from 43 to 41 iterations, and
  Kacanov from 17 to 22.
- Where M is poor and memory 0 does not break down, the first pair is worth
  somewhat more than the next nine together, but not overwhelmingly:

  | problem | memory 0 to 1 | memory 1 to 10 |
  |---|---|---|
  | p-Laplacian, M = I | 2.1 | 1.6 |
  | sphere | 1.6 | 1.4 |
  | convex quartic | 1.09 | 1.15 |

- The jump from memory 0 looked outsized for two reasons:
  - memory 0 breaking down where gamma freezes (Rosenbrock, completion);
  - single-run comparisons inside the rounding spread (see the correction
    in `experiments/lbfgs_memory/`).

## Answer, so far

- **The frozen gamma is real.** On nonconvex problems memory 0 stalls
  because its gamma freezes, so there the jump to memory 1 overstates what
  the memory adds. Against a memory 0 that refits gamma on rejected pairs,
  memory 1 is still 4 to 9 times faster on Rosenbrock.
- **The step length is not the main story.** Where gamma never freezes,
  memory 1's advantage is its direction. Giving memory 0 memory 1's step
  length does not reproduce it.
- **More memory helps where the metric is poor.** Memory 10 is 1.4 to 1.9
  times faster than memory 1 there. With H1 and Kacanov, the metric already
  does that work.

## What this does not show

- **The line search.** It is lbfgs's monotone Armijo backtracking.
  Barzilai-Borwein methods are usually run with a nonmonotone line search
  (Grippo-Lampariello-Lucidi; Raydan 1997), which lets their characteristic
  nonmonotone steps through. Memory 0 here may be handicapped more than
  L-BFGS is. A nonmonotone arm would be the next test of the baseline.
- **Other choices.** E1 and E2 are two simple safeguards, and B and C use two
  definitions of "the other memory's length" each. Others exist.
- **The starts sample rounding, not the problem.** All starts are within
  1e-12 of the same x0, and there are seven problems.
- **Iterations, not cost.**
  - E1 adds one M product and one M^{-1} solve per rejected pair.
  - B1 adds one M product per iteration.
  - C's length uses the M product that the angle test already makes.

**Status:** first round.
