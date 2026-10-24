"""MPI program used by test_distributed.py: run it under mpiexec.

Checks that the distributed p-Laplacian operators of examples/p_laplacian.py
(energy, gradient, Hessian action, Jacobi preconditioner, load) agree with
the same operators on one rank, and that the L^2 metric operators are
inverse to each other and symmetric across ranks. Exits with status 1 on
failure.
"""

import sys

import numpy as np
from mpi4py import MPI

import p_laplacian as pl
from nicks_optimizers import Covector, Point, Vector


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
        ops = pl.operators_at(loc, e, "jacobi", u)
        ops_s = pl.operators_at(serial, e_serial, "jacobi", Point(u_all))
        assert ops.precond_solve is not None and ops_s.precond_solve is not None
        parts = (pl.gradient(loc, e, u), ops.hessvec(v), ops.precond_solve(Covector(np.asarray(v))), e.b)
        references = (pl.gradient(serial, e_serial, Point(u_all)), ops_s.hessvec(Vector(v_all)),
                      ops_s.precond_solve(Covector(v_all)), e_serial.b)
        for part, reference in zip(parts, references):
            gathered = np.zeros(loc.n_unknowns)
            gathered[loc.owned] = part
            total = pl.allreduce(loc, gathered)
            worst = max(worst, float(np.max(np.abs(total - reference)) / np.max(np.abs(reference))))
        f, f_serial = pl.objective(loc, e, u), pl.objective(serial, e_serial, Point(u_all))
        worst = max(worst, abs(f - f_serial) / abs(f_serial))

    # The L^2 metric: M^{-1} M = I, and <M s, t> = <M t, s> = <M s, M^{-1} M t> across ranks.
    space = pl.distributed_space(loc)
    s, t = Vector(u_all[loc.owned]), Vector(v_all[loc.owned])
    Ms, Mt = pl.l2_metric_apply(loc, s), pl.l2_metric_apply(loc, t)
    worst = max(worst, float(np.max(np.abs(pl.l2_metric_solve(loc, Ms) - s))) / float(np.max(np.abs(s))))
    primal = space.pair(Ms, t)
    worst = max(worst, abs(primal - space.pair(Mt, s)) / abs(primal))
    worst = max(worst, abs(primal - space.pair(Ms, pl.l2_metric_solve(loc, Mt))) / abs(primal))

    ok = worst < 1e-12
    if comm.Get_rank() == 0:
        print(f"{comm.Get_size()} ranks: largest relative difference from one rank {worst:.1e}: "
              f"{'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
