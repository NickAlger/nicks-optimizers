"""Spectra of the Gauss-Newton and full Riemannian Hessians of matrix
completion on the tangent space (Frobenius metric), for uniform and
nonuniform (power-law) sampling, at the fitted point and at the start of the
last rank-continuation stage (a new component of size 1e-6 s_1). Dense, on a
small instance (120 x 100, rank 4).

Run:  python experiments/sgd_preconditioner/spectra.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples"))
import matrix_completion as mc  # noqa: E402

def tangent_basis(x):
    U, s, V = x
    N, r = U.shape; M = V.shape[0]
    Uperp = np.linalg.svd(np.eye(N) - U @ U.T)[0][:, : N - r]
    basis = []
    for k in range(r):
        for i in range(N - r):
            dU = np.zeros((N, r)); dU[:, k] = Uperp[:, i]; basis.append((dU, np.zeros((M, r))))
    for k in range(r):
        for j in range(M):
            dV = np.zeros((M, r)); dV[j, k] = 1.0; basis.append((np.zeros((N, r)), dV))
    return basis

def spectrum(samples, x, curvature):
    B = tangent_basis(x)
    cols = [mc.hessian_action(samples, x, b, curvature) for b in B]
    Hm = np.array([[np.vdot(c[0], b[0]) + np.vdot(c[1], b[1]) for c in cols] for b in B])
    return np.linalg.eigvalsh(0.5 * (Hm + Hm.T))

def sample(a, K, rng, power):
    N, M = a[0].shape[0], a[2].shape[0]
    if power == 0:
        return mc.sample_uniformly(a, K, rng)
    pr = np.arange(1, N + 1, dtype=float) ** -power; pr /= pr.sum()
    pc = np.arange(1, M + 1, dtype=float) ** -power; pc /= pc.sum()
    flat = set()
    while len(flat) < K:
        flat.update(zip(rng.choice(N, K, p=pr), rng.choice(M, K, p=pc)))
    idx = np.array(list(flat))[:K]
    rows, cols = idx[:, 0].astype(np.intp), idx[:, 1].astype(np.intp)
    return mc.Samples(shape=(N, M), rows=rows, cols=cols, values=mc.sampled_entries(a, rows, cols))

rng = np.random.default_rng(1)
N, M, r = 120, 100, 4
d = 0.5 ** np.arange(20, dtype=float)
a = mc.random_low_rank((N, M), d, rng)
K = 4 * (N + M - r) * r
print(f"{N} x {M}, rank {r} fit, K = {K} samples (4 per degree of freedom), singular values 2^-k\n")
for power in (0.0, 1.0):
    smp = sample(a, K, rng, power)
    stages = mc.complete_by_rank_continuation(smp, r, rng)
    x_fit = stages[-1][0]
    x_start = mc.add_zero_component(stages[-2][0], rng)          # rank-r start of the last stage
    x_start = mc.LowRank((x_start[0], np.append(x_start[1][:-1], 1e-6 * x_start[1][0]), x_start[2]))
    label = "uniform" if power == 0 else f"power law (exponent {power})"
    for where, x in (("fitted point", x_fit), ("start of last stage (s_r = 1e-6 s_1)", x_start)):
        for curvature in (False, True):
            ev = spectrum(smp, x, curvature)
            neg = int(np.sum(ev < 0))
            pos = ev[ev > 0]
            print(f"{label:28s} {where:38s} {'full' if curvature else 'GN  '}: "
                  f"lambda in [{ev[0]:9.2e}, {ev[-1]:9.2e}], {neg:3d} negative, "
                  f"cond(positive part) = {pos[-1] / pos[0]:9.2e}")
    print()
