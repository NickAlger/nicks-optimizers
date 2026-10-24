"""The trust-region subproblem with unconventional preconditioners, on a
dense problem where the exact solution is known.

Solvers compared, all with the same metric M (SPD, diagonal) and the same
separate preconditioner P:

  lanczos       the library's solve_tr_subproblem: P-preconditioned Lanczos
                over K_k(P^{-1} H, P^{-1} g), relying on P being fixed SPD
  flex-Hq       a local prototype: flexible basis (M-orthonormalized,
                explicit T = Q^T H Q, start M^{-1} g) expanded with
                P^{-1}(H q_k), the Lanczos-style direction
  flexible      the library's solve_tr_subproblem_flexible: the same, but
                expanded with P^{-1} R, R the KKT residual of the current step

Prints the relative gap (m(s) - m*) / |m*| of the TRUE model at the
returned step (above 1: worse than not moving), with '!' when the step
leaves the ball and '?' when the solver's own model value is off by more
than 10%.

Run:  python experiments/flexible_preconditioners/subproblem_study.py
"""
import warnings
from typing import Callable

import numpy as np

from nicks_optimizers import (
    Covector, Vector, numpy_space, solve_sphere_trs, solve_tr_subproblem,
    solve_tr_subproblem_flexible,
)
from nicks_optimizers.trust_region_subproblem import solve_projected_trs

warnings.simplefilter("ignore")            # the Lanczos runs overflow with the worst preconditioners
Precond = Callable[[np.ndarray], np.ndarray]


def preconditioners(Hs: np.ndarray, rng: np.random.Generator) -> dict[str, Precond]:
    """Maps covector -> vector built from Hs, an SPD shift of H."""
    n = Hs.shape[0]
    tri = Hs.copy()
    tri[: n // 2, n // 2:] = 0.0
    d = np.diag(Hs)

    def inner_cg(r: np.ndarray, tol: float = 0.3) -> np.ndarray:
        z, res = np.zeros(n), r.copy()
        p, rr = res.copy(), res @ res
        r0 = np.sqrt(rr)
        while np.sqrt(rr) > tol * r0:
            Ap = Hs @ p
            a = rr / (p @ Ap)
            z, res = z + a * p, res - a * Ap
            rr_new = res @ res
            p, rr = res + (rr_new / rr) * p, rr_new
        return z

    def random_coordinate_descent(r: np.ndarray, sweeps: int = 3) -> np.ndarray:
        z, res = np.zeros(n), r.copy()
        for _ in range(sweeps):
            for i in rng.permutation(n):
                delta = res[i] / d[i]
                z[i] += delta
                res -= delta * Hs[:, i]
        return z

    def minibatch_sgd(r: np.ndarray, steps: int = 40, batch: int = 20) -> np.ndarray:
        z, lr = np.zeros(n), 1.0 / np.linalg.norm(Hs, 2)
        for _ in range(steps):
            rows = rng.choice(n, batch, replace=False)
            grad = np.zeros(n)
            grad[rows] = (Hs[rows] @ z - r[rows]) * (n / batch)
            z = z - lr * grad
        return z

    def noisy(r: np.ndarray) -> np.ndarray:
        z = np.linalg.solve(Hs, r)
        return z + 0.3 * np.linalg.norm(z) * rng.standard_normal(n) / np.sqrt(n)

    return {"fixed SPD": lambda r: np.linalg.solve(Hs, r),
            "block triangular": lambda r: np.linalg.solve(tri, r),
            "inner CG (tol 0.3)": inner_cg, "random coord. descent": random_coordinate_descent,
            "minibatch SGD": minibatch_sgd, "noisy (30%)": noisy}


def flex_hq(H: np.ndarray, M: np.ndarray, g: np.ndarray, pinv: Precond, radius: float,
            k_max: int) -> np.ndarray:
    """Prototype: flexible basis expanded with P^{-1}(H q_k)."""
    gamma0 = float(np.sqrt(g @ np.linalg.solve(M, g)))
    q = np.linalg.solve(M, g) / gamma0
    Q, MQ, HQ = q[:, None], (g / gamma0)[:, None], (H @ q)[:, None]
    for k in range(1, k_max + 1):
        T = Q.T @ HQ
        y = solve_projected_trs(0.5 * (T + T.T), np.eye(k), gamma0, radius)[0]
        if k == k_max:
            return Q @ y
        z = pinv(HQ[:, -1])
        Mz = M @ z
        for _ in range(2):
            c = Q.T @ Mz
            z, Mz = z - Q @ c, Mz - MQ @ c
        nz = float(np.sqrt(max(z @ Mz, 0.0)))
        Q, MQ = np.column_stack([Q, z / nz]), np.column_stack([MQ, Mz / nz])
        HQ = np.column_stack([HQ, H @ (z / nz)])
    raise AssertionError


def study(title: str, H: np.ndarray, M: np.ndarray, g: np.ndarray, radius: float,
          names: tuple[str, ...], seed: int = 0, ks: tuple[int, ...] = (5, 10, 20, 40)) -> None:
    rng = np.random.default_rng(seed)
    pre = preconditioners(H + 3.0 * np.eye(H.shape[0]), rng)
    Li = np.linalg.inv(np.linalg.cholesky(M))
    z, lam = solve_sphere_trs(Li @ H @ Li.T, Li @ g, radius)
    s_star = Li.T @ z
    model = lambda s: float(g @ s + 0.5 * s @ H @ s)
    mnorm = lambda s: float(np.sqrt(s @ M @ s))
    m_star = model(s_star)
    print(f"\n=== {title}: radius {radius:g}, ||s*||_M = {mnorm(s_star):.3g}, lam* = {lam:.3g} ===")
    print(f"{'preconditioner':24s} {'solver':9s}" + "".join(f"{'k=' + str(k):>11s}" for k in ks))
    Minv = lambda w: Vector(np.linalg.solve(M, w))
    for name in names:
        pinv = pre[name]
        for solver in ("lanczos", "flex-Hq", "flexible"):
            row = []
            for k in ks:
                reported = None
                try:
                    if solver == "lanczos":
                        s, st = solve_tr_subproblem(
                            numpy_space, Covector(g), lambda v: Covector(H @ v), Minv, radius, 0.0, k,
                            precond_solve=lambda w: Vector(pinv(w)),
                            metric_apply=lambda v: Covector(M @ v), stall_tol=0.0)
                        reported = st.model_value
                    elif solver == "flex-Hq":
                        s = flex_hq(H, M, g, pinv, radius, k)
                    else:
                        s, st = solve_tr_subproblem_flexible(
                            numpy_space, Covector(g), lambda v: Covector(H @ v), Minv,
                            lambda v: Covector(M @ v), lambda w: Vector(pinv(w)), radius, 0.0, k)
                        reported = st.model_value
                except (np.linalg.LinAlgError, ValueError):
                    row.append(f"{'crash':>9s}  ")
                    continue
                if not np.all(np.isfinite(s)):
                    row.append(f"{'nan':>9s}  ")
                    continue
                m = model(s)
                flags = ("!" if mnorm(s) > 1.01 * radius else " ") + \
                    ("?" if reported is not None and abs(reported - m) > 0.1 * abs(m) + 1e-12 else " ")
                row.append(f"{(m - m_star) / abs(m_star):9.1e}{flags}")
            print(f"{name:24s} {solver:9s}" + "".join(row))


def main() -> None:
    rng = np.random.default_rng(0)
    n = 200
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    M = np.diag(np.linspace(0.5, 2.0, n))
    g = rng.standard_normal(n)
    H = (U * np.concatenate([[-2.0, -0.5, -0.1], np.logspace(-1, 3, n - 3)])) @ U.T
    everything = tuple(preconditioners(H, rng))
    study("indefinite H, boundary active", H, M, g, 0.3, everything)
    study("indefinite H, large radius (still on the boundary)", H, M, g, 30.0, everything)
    H_spd = (U * np.logspace(-1, 3, n)) @ U.T
    study("SPD H, interior", H_spd, M, g, 1e6,
          ("fixed SPD", "inner CG (tol 0.3)", "random coord. descent", "noisy (30%)"))
    # Hard case: g M-orthogonal to the bottom eigenvector of the pencil (H, M).
    Li = np.linalg.inv(np.linalg.cholesky(M))
    _, V = np.linalg.eigh(Li @ H @ Li.T)
    u = Li.T @ V[:, 0]
    study("hard case", H, M, g - (M @ u) * (u @ g), 50.0,
          ("fixed SPD", "inner CG (tol 0.3)", "noisy (30%)"))


if __name__ == "__main__":
    main()
