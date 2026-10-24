"""How well the Krylov space of the preconditioned Hessian captures the
trust-region solution when the trust-region metric M matches the
preconditioner P, and when it does not.

When the trust region is active, the exact step solves (H + lam M) s = -g.
With M = P that solution lies in the Krylov space K(P^{-1} H, P^{-1} g) for
every lam (shift invariance), so a few Lanczos steps suffice. With M != P it
does not, and the subspace solutions improve slowly; the method still
converges globally (every step beats the Cauchy point), and boundary solves
stop when the model stops improving.

Run:  python examples/matched_vs_mismatched.py
"""

import numpy as np

from nicks_optimizers import Covector, Vector, numpy_space, solve_tr_subproblem


def random_spd(n: int, cond: float, rng: np.random.Generator) -> np.ndarray:
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return (U * np.logspace(0.0, np.log10(cond), n)) @ U.T


def model_value(H: np.ndarray, P: np.ndarray, M: np.ndarray, g: np.ndarray, radius: float,
                k: int, resid_tol: float = 0.0, stall_tol: float = 0.0) -> tuple[float, int]:
    """Lanczos model value and step count. With M = P only M^{-1} is passed
    (the matched path); otherwise P^{-1} as the preconditioner and M too."""
    separate = {} if M is P else {
        "precond_solve": lambda w: Vector(np.linalg.solve(P, w)),
        "metric_apply": lambda v: Covector(M @ v)}
    _, stats = solve_tr_subproblem(
        numpy_space, Covector(g), lambda v: Covector(H @ v), lambda w: Vector(np.linalg.solve(M, w)),
        radius, resid_tol, k, stall_tol=stall_tol, **separate)  # type: ignore[arg-type]
    return stats.model_value, stats.krylov_steps


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, radius = 200, 0.5
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    H = (U * np.concatenate([np.logspace(0, 3, n - 3), [-5.0, -1.0, -0.1]])) @ U.T
    P, M_other = random_spd(n, 1e2, rng), random_spd(n, 1e2, rng)
    g = rng.standard_normal(n)
    ks = (5, 10, 20, 40, 100)
    print("relative model-value gap to the exact solution after k Lanczos steps")
    print(" " * 24 + "".join(f"k={k:<8d}" for k in ks))
    for name, M in (("M = P (matched)", P), ("M != P (mismatched)", M_other)):
        best = model_value(H, P, M, g, radius, n)[0]
        gaps = [(model_value(H, P, M, g, radius, k)[0] - best) / abs(best) for k in ks]
        print(f"{name:24s}" + "".join(f"{gap:<10.1e}" for gap in gaps))
        gnorm = float(np.sqrt(g @ np.linalg.solve(M, g)))
        value, k_used = model_value(H, P, M, g, radius, 50, resid_tol=0.5 * gnorm, stall_tol=1e-3)
        print(f"{'':24s}default stopping rules stop at k = {k_used}, gap {(value - best) / abs(best):.1e}")
