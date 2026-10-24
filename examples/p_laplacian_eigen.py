"""The first p-Laplacian eigenvalue by unconstrained energy minimization,
distributed over MPI.

The first eigenvalue of -div(|grad u|^{p-2} grad u) = lambda |u|^{p-2} u on
the unit disk (u = 0 on the circle) is the minimum of the Rayleigh quotient

    lambda_1 = min_u A(u) / B(u),   A(u) = int |grad u|^p,  B(u) = int |u|^p.

R = A / B is invariant under scaling u, so its minimizers form rays and its
Hessian is singular along them. This example minimizes instead

    f(u) = A(u) / p - log(B(u)) / p,

which fixes the scale: along a ray u = t v, f is smallest where A(u) = 1, and
there f = 1/p + log(A(v) / B(v)) / p. So the minimizers of f are exactly the
ground states scaled to A = 1, the minimum value gives lambda_1, and f has no
flat directions. f is not convex: the higher eigenfunctions are its saddle
points (the first eigenfunction is the only one of one sign), which makes
this a test problem for escaping saddles, with or without the saddle-free
model.

Discretization and distribution are those of p_laplacian.py: linear finite
elements on the graded disk mesh, slabs of nodes per rank, the L^2 ball of
the lumped mass matrix, and block Jacobi multigrid on the gradient part of
the Hessian (A is regularized with a small eps, which also keeps that part
positive definite). B uses the lumped mass, B(u) = sum_i m_i |u_i|^p, and
p >= 2 keeps it twice differentiable. The Hessian of f is

    H_A / p - (p - 1) diag(m |u|^{p-2}) / B + p z z^T / B^2,   z = m |u|^{p-2} u,

whose last term needs one global reduction per product.

Check. The ground state on the disk is radial, so lambda_1 also follows from
the radial equation (r |u'|^{p-2} u')' = -lambda r |u|^{p-2} u, solved here by
shooting (j_{0,1}^2 = 5.78319 for p = 2). The discrete eigenvalues converge
to it under refinement.

Run:   mpiexec -n 4 python examples/p_laplacian_eigen.py --p 2 3 4
       python examples/p_laplacian_eigen.py --start random --seeds 5 --saddle-free
"""

import argparse
import json
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from mpi4py import MPI
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

import p_laplacian as pl
from nicks_optimizers import Array, Covector, Point, RunSummary, Vector, trust_region_newton


# --- the reference value ------------------------------------------------------------


def radial_shot(lam: float, p: float, r0: float = 1e-6) -> float:
    """u(1) for the radial solution with u(0) = 1 (series start at r0)."""
    def rhs(r: float, y: Array) -> list[float]:
        u, phi = float(y[0]), float(y[1])                    # phi = |u'|^{p-2} u'
        du = float(np.sign(phi) * abs(phi) ** (1.0 / (p - 1.0)))
        return [du, -phi / r - lam * abs(u) ** (p - 2.0) * u]
    u0 = 1.0 - (p - 1.0) / p * (lam / 2.0) ** (1.0 / (p - 1.0)) * r0 ** (p / (p - 1.0))
    sol = solve_ivp(rhs, (r0, 1.0), [u0, -lam * r0 / 2.0], rtol=1e-12, atol=1e-14, method="DOP853")
    return float(sol.y[0, -1])


def radial_eigenvalue(p: float) -> float:
    """lambda_1 of the unit disk: the smallest lambda with u(1) = 0."""
    lo, hi = 0.5, 1.0
    while radial_shot(hi, p) > 0.0:
        lo, hi = hi, 2.0 * hi
    return float(brentq(lambda lam: radial_shot(lam, p), lo, hi, xtol=1e-13, rtol=1e-13))


# --- the energy ---------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Eigen:
    p: float
    eps: float


def gradient_energy(e: Eigen) -> pl.Energy:
    """A / p (regularized) in the form p_laplacian uses, without a load."""
    return pl.Energy(p=e.p, eps=e.eps, b=np.zeros(0))


def mass_power(loc: pl.Local, e: Eigen, u: Array) -> tuple[float, Array]:
    """B(u) = sum_i m_i |u_i|^p and z = m |u|^{p-2} u (so grad B = p z)."""
    z = loc.mass * np.abs(u) ** (e.p - 2.0) * u
    return float(pl.allreduce(loc, np.array([z @ u]))[0]), z


def objective(loc: pl.Local, e: Eigen, u: Point) -> float:
    E = pl.objective(loc, pl.Energy(p=e.p, eps=e.eps, b=np.zeros(loc.n_owned)), u)
    B, _ = mass_power(loc, e, u)
    return float(E - np.log(B) / e.p) if B > 0.0 else float("inf")


def gradient(loc: pl.Local, e: Eigen, u: Point) -> Covector:
    B, z = mass_power(loc, e, u)
    return Covector(pl.gradient_rows(loc, e.p, e.eps, u) - z / B)


def operators_at(loc: pl.Local, e: Eigen, precond: str, u: Point
                 ) -> tuple[Callable[[Vector], Covector], Callable[[Covector], Vector]]:
    """H(u) (see the module docstring) and P(u)^{-1}: block Jacobi multigrid
    (or the diagonal) of the gradient part H_A / p, which is positive
    definite."""
    hess_A, g, a, c = pl.hessian_at(loc, gradient_energy(e), u)
    B, z = mass_power(loc, e, u)
    diag = (e.p - 1.0) * loc.mass * np.abs(u) ** (e.p - 2.0) / B

    def hessvec(v: Vector) -> Covector:
        zv = float(pl.allreduce(loc, np.array([z @ v]))[0])
        return Covector(hess_A(v) - diag * v + (e.p * zv / B ** 2) * z)

    block = pl.owned_block(loc, g, a, c)
    if precond == "jacobi":
        d = block.diagonal()
        return hessvec, lambda w: Vector(w / d)
    return hessvec, pl.multigrid_cycle(block)


def eigenvalue(loc: pl.Local, e: Eigen, u: Array) -> float:
    """A(u) / B(u) without the regularization."""
    grads = pl.element_gradients(loc, pl.extend(loc, u))
    local_A = float(np.sum((loc.area * np.sum(grads ** 2, axis=1) ** (e.p / 2.0))[loc.count_energy]))
    A = float(pl.allreduce(loc, np.array([local_A]))[0])
    B, _ = mass_power(loc, e, u)
    return A / B


def sign_definite(loc: pl.Local, u: Array) -> bool:
    """True for a function of one sign (up to rounding): the ground state."""
    scale = float(loc.comm.allreduce(float(np.max(np.abs(u))), op=MPI.MAX))
    lo = float(loc.comm.allreduce(float(np.min(u)), op=MPI.MIN))
    hi = float(loc.comm.allreduce(float(np.max(u)), op=MPI.MAX))
    return min(-lo, hi) <= 1e-8 * scale


# --- starting points and the driver ---------------------------------------------------------


def starting_point(loc: pl.Local, kind: str, seed: int) -> Point:
    """'bump': 1 - |x|^2. 'random': independent normal nodal values, drawn
    for all unknowns from one seed, so every partition starts at the same
    point."""
    if kind == "bump":
        return Point(1.0 - np.sum(loc.xy ** 2, axis=1))
    values = np.random.default_rng(seed).standard_normal(loc.n_unknowns)
    return Point(values[loc.owned])


def solve(loc: pl.Local, e: Eigen, u0: Point, precond: str = "amg", saddle_free: bool = False,
          display: int = 0, max_iter: int = 300, max_lanczos: int = 100) -> tuple[Point, RunSummary]:
    root = loc.comm.Get_rank() == 0
    return trust_region_newton(
        lambda u: objective(loc, e, u),
        lambda u: gradient(loc, e, u),
        None, None,
        lambda u, U, v: pl.l2_inner(loc, U, v),
        lambda u, Psi, phi: pl.l2_dual_inner(loc, Psi, phi),
        u0, space=pl.distributed_space(loc),
        operators_at=lambda u: operators_at(loc, e, precond, u),
        saddle_free=saddle_free, rtol=1e-10, max_iter=max_iter, max_lanczos=max_lanczos,
        display=display, print_fn=print if root else (lambda line: None))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--p", type=float, nargs="+", default=[2.0, 3.0, 4.0])
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--rings", type=int, default=40)
    parser.add_argument("--start", choices=["bump", "random"], default="bump")
    parser.add_argument("--seeds", type=int, default=1, help="random starts per p")
    parser.add_argument("--saddle-free", action="store_true")
    parser.add_argument("--precond", choices=["amg", "jacobi"], default="amg")
    parser.add_argument("--display", type=int, default=0)
    parser.add_argument("--json", help="write the results here (rank 0)")
    args = parser.parse_args()
    if min(args.p) < 2.0:
        parser.error("p >= 2 is needed for B = sum m |u|^p to be twice differentiable")

    comm = MPI.COMM_WORLD
    root = comm.Get_rank() == 0
    mesh = pl.graded_disk_mesh(args.rings)
    loc = pl.partition(mesh, comm)
    if root:
        print(f"mesh: {loc.n_unknowns} unknowns on {comm.Get_size()} ranks; start: {args.start}; "
              f"{'saddle-free' if args.saddle_free else 'plain'} model; preconditioner: {args.precond}\n")
        print("    p  seed  iterations  Lanczos  lambda_h        lambda (radial)  rel. diff  "
              "ground state  time")
    results = []
    for p in args.p:
        e = Eigen(p=p, eps=args.eps)
        reference = radial_eigenvalue(p) if root else 0.0
        for seed in range(args.seeds):
            start = time.perf_counter()
            u, summary = solve(loc, e, starting_point(loc, args.start, seed), args.precond,
                               args.saddle_free, args.display)
            elapsed = comm.allreduce(time.perf_counter() - start, op=MPI.MAX)
            lam, ground = eigenvalue(loc, e, u), sign_definite(loc, u)
            lanczos = sum(r.subproblem.lanczos_steps for r in summary.iterations)
            if root:
                print(f"{p:5.2f}  {seed:4d}  {len(summary.iterations):10d}  {lanczos:7d}  {lam:.10f}  "
                      f"{reference:.10f}  {abs(lam - reference) / reference:9.1e}  "
                      f"{'yes' if ground else 'no':>12s}  {elapsed:4.1f}s  ({summary.reason})")
            results.append({"p": p, "seed": seed, "lambda": lam, "reference": reference,
                            "ground": ground, "reason": summary.reason,
                            "iterations": len(summary.iterations), "lanczos": lanczos,
                            "records": [{"f": r.f, "accepted": r.accepted,
                                         "lanczos": r.subproblem.lanczos_steps,
                                         "negative": r.subproblem.n_negative}
                                        for r in summary.iterations]})
    if root and args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh)


if __name__ == "__main__":
    main()
