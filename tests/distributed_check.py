"""MPI program used by test_distributed.py: run it under mpiexec.

Checks that the distributed p-Laplacian operators of examples/p_laplacian.py
(energy, gradient, Hessian action, Jacobi preconditioner, load) agree with
the same operators on one rank, and that the L^2 inner products are exact
duals. Exits with status 1 on failure.
"""

import sys

import numpy as np
from mpi4py import MPI

import p_laplacian as pl
from nicks_optimizers import Covector, CovectorBlock, Point, Vector, VectorBlock


def main() -> int:
    comm = MPI.COMM_WORLD
    mesh = pl.graded_disk_mesh(12)
    loc = pl.partition(mesh, comm)
    serial = pl.partition(mesh, MPI.COMM_SELF)
    rng = np.random.default_rng(1)
    u_all, v_all = rng.standard_normal(loc.n_unknowns), rng.standard_normal(loc.n_unknowns)
    worst = 0.0
    for p in (1.5, 3.0):
        e, _ = pl.manufactured_energy(loc, p, 1e-2)
        e_serial, _ = pl.manufactured_energy(serial, p, 1e-2)
        u, v = Point(u_all[loc.owned]), Vector(v_all[loc.owned])
        hessvec, precond = pl.operators_at(loc, e, "jacobi", u)
        parts = (pl.gradient(loc, e, u), hessvec(v), precond(Covector(np.asarray(v))), e.b)
        hessvec_s, precond_s = pl.operators_at(serial, e_serial, "jacobi", Point(u_all))
        references = (pl.gradient(serial, e_serial, Point(u_all)), hessvec_s(Vector(v_all)),
                      precond_s(Covector(v_all)), e_serial.b)
        for part, reference in zip(parts, references):
            gathered = np.zeros(loc.n_unknowns)
            gathered[loc.owned] = part
            total = pl.allreduce(loc, gathered)
            worst = max(worst, float(np.max(np.abs(total - reference)) / np.max(np.abs(reference))))
        f, f_serial = pl.objective(loc, e, u), pl.objective(serial, e_serial, Point(u_all))
        worst = max(worst, abs(f - f_serial) / abs(f_serial))

    # Exact duality of the L^2 pair: (M u, M v)_{M^{-1}} = (u, v)_M.
    s, t = Vector(u_all[loc.owned]), Vector(v_all[loc.owned])
    Ms, Mt = Covector(loc.mass * s), Covector(loc.mass * t)
    primal = float(pl.l2_inner(loc, VectorBlock(s[:, None]), t)[0])
    dual = float(pl.l2_dual_inner(loc, CovectorBlock(Ms[:, None]), Mt)[0])
    worst = max(worst, abs(primal - dual) / abs(primal))

    ok = worst < 1e-12
    if comm.Get_rank() == 0:
        print(f"{comm.Get_size()} ranks: largest relative difference from one rank {worst:.1e}: "
              f"{'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
