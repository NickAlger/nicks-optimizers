"""The p-Laplacian eigenvalue example (examples/p_laplacian_eigen.py).

Skipped unless mpi4py, pyamg and scipy are available; the multi-rank test
also needs mpiexec.
"""

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from nicks_optimizers import Covector, Point, Vector

MISSING = [m for m in ("mpi4py", "pyamg", "scipy") if importlib.util.find_spec(m) is None]
pytestmark = pytest.mark.skipif(bool(MISSING), reason="needs mpi4py, pyamg and scipy")


def test_radial_reference_matches_bessel_zero() -> None:
    """For p = 2 the radial shooting gives j_{0,1}^2."""
    import p_laplacian_eigen as pe
    j01 = 2.404825557695773
    assert abs(pe.radial_eigenvalue(2.0) - j01 ** 2) < 1e-10 * j01 ** 2


def test_derivatives_of_the_eigen_functional() -> None:
    """f, its gradient and Hessian agree with central differences (errors
    scale like t^2); H is symmetric and the preconditioner SPD."""
    from mpi4py import MPI
    import p_laplacian as pl
    import p_laplacian_eigen as pe
    loc = pl.partition(pl.graded_disk_mesh(10), MPI.COMM_SELF)
    rng = np.random.default_rng(0)
    for p in (2.0, 3.0, 4.5):
        e = pe.Eigen(p=p, eps=1e-2)
        u = Point(rng.standard_normal(loc.n_owned))
        v, w = (Vector(rng.standard_normal(loc.n_owned)) for _ in range(2))
        g = pe.gradient(loc, e, u)
        ops = pe.operators_at(loc, e, "amg", u)
        hessvec, precond = ops.hessvec, ops.precond_solve
        assert hessvec is not None and precond is not None
        Hv = np.asarray(hessvec(v))
        errors: list[tuple[float, float]] = []
        for t in (1e-3, 1e-4):
            slope = (pe.objective(loc, e, Point(u + t * v))
                     - pe.objective(loc, e, Point(u - t * v))) / (2 * t)
            column = (np.asarray(pe.gradient(loc, e, Point(u + t * v)))
                      - np.asarray(pe.gradient(loc, e, Point(u - t * v)))) / (2 * t)
            errors.append((abs(slope - float(g @ v)) / abs(float(g @ v)),
                           float(np.max(np.abs(column - Hv))) / float(np.max(np.abs(Hv)))))
        assert errors[1][0] < 1e-6 and errors[1][1] < 1e-6
        assert errors[0][0] / max(errors[1][0], 1e-300) > 30 or errors[1][0] < 1e-10
        assert errors[0][1] / max(errors[1][1], 1e-300) > 30 or errors[1][1] < 1e-10
        assert abs(hessvec(v) @ w - hessvec(w) @ v) < 1e-12 * abs(hessvec(v) @ w)
        Pv, Pw = precond(Covector(np.asarray(v))), precond(Covector(np.asarray(w)))
        assert abs(Pv @ w - Pw @ v) < 1e-12 * abs(Pv @ w) and Pv @ v > 0


def test_ground_state_and_eigenvalue() -> None:
    """From a random sign-mixed start the method reaches the (one-signed)
    ground state; lambda_h is within the discretization error of the radial
    value and converges at second order."""
    from mpi4py import MPI
    import p_laplacian as pl
    import p_laplacian_eigen as pe
    for p in (2.0, 3.0):
        e = pe.Eigen(p=p, eps=1e-6)
        reference = pe.radial_eigenvalue(p)
        errors = []
        for rings in (10, 20):
            loc = pl.partition(pl.graded_disk_mesh(rings), MPI.COMM_SELF)
            u, summary = pe.solve(loc, e, pe.starting_point(loc, "random", 0))
            assert summary.reason == "converged" and pe.sign_definite(loc, u)
            errors.append(abs(pe.eigenvalue(loc, e, u) - reference) / reference)
        assert errors[1] < 5e-3
        assert 3.0 < errors[0] / errors[1] < 5.5            # about 4: second order


@pytest.mark.skipif(shutil.which("mpiexec") is None, reason="needs mpiexec")
def test_same_eigenvalue_on_three_ranks(tmp_path: Path) -> None:
    from test_distributed import ROOT, mpirun
    out = {}
    for ranks in (1, 3):
        path = tmp_path / f"eigen_{ranks}.json"
        mpirun(ranks, ROOT / "examples" / "p_laplacian_eigen.py", "--rings", "12", "--p", "2", "3",
               "--json", str(path))
        out[ranks] = json.loads(path.read_text())
    for one, three in zip(out[1], out[3]):
        assert one["ground"] and three["ground"]
        assert abs(one["lambda"] - three["lambda"]) < 1e-10 * one["lambda"]
