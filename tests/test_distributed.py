"""Distributed vectors: examples/p_laplacian.py run under mpiexec.

Skipped unless mpi4py, pyamg, scipy and an mpiexec are available
(pip install -e ".[examples]" plus an MPI, e.g. pip install mpich).
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
MISSING = [m for m in ("mpi4py", "pyamg", "scipy") if importlib.util.find_spec(m) is None]
pytestmark = pytest.mark.skipif(bool(MISSING) or shutil.which("mpiexec") is None,
                                reason="needs mpi4py, pyamg, scipy and mpiexec")


def mpirun(ranks: int, script: Path, *args: str) -> str:
    """Run a script under mpiexec on this machine. UCX (MPICH's transport) is
    limited to shared memory and TCP unless UCX_TLS is set: on some cloud VMs
    it otherwise picks a network device it cannot use and aborts."""
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'tests'}:{ROOT / 'examples'}:{ROOT / 'src'}"}
    env.setdefault("UCX_TLS", "self,sm,tcp")
    result = subprocess.run(
        ["mpiexec", "-n", str(ranks), sys.executable, str(script), *args],
        capture_output=True, text=True, timeout=600, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.mark.parametrize("ranks", [1, 2, 3])
def test_distributed_operators_match_one_rank(ranks: int) -> None:
    assert "OK" in mpirun(ranks, ROOT / "tests" / "distributed_check.py")


def run_example(ranks: int, tmp_path: Path, *args: str) -> Any:
    out = tmp_path / f"history_{ranks}.json"
    mpirun(ranks, ROOT / "examples" / "p_laplacian.py", "--rings", "12", "--json", str(out), *args)
    return json.loads(out.read_text())


def test_runs_follow_the_same_iterates_on_any_number_of_ranks(tmp_path: Path) -> None:
    """With the (partition-independent) Jacobi preconditioner, 1 and 3 ranks
    take the same steps: same acceptances and Lanczos counts, objective
    values equal up to the order of floating-point sums."""
    one = run_example(1, tmp_path, "--p", "1.5", "3", "--precond", "jacobi")
    three = run_example(3, tmp_path, "--p", "1.5", "3", "--precond", "jacobi")
    for p in one:
        a, b = one[p]["records"], three[p]["records"]
        assert len(a) == len(b)
        assert [(r["accepted"], r["krylov_steps"]) for r in a] == [(r["accepted"], r["krylov_steps"]) for r in b]
        assert max(abs(r["f"] - s["f"]) / max(1.0, abs(r["f"])) for r, s in zip(a, b)) < 1e-12
        assert one[p]["error"] < 1e-9 and three[p]["error"] < 1e-9


def test_block_jacobi_multigrid_converges(tmp_path: Path) -> None:
    """The default preconditioner (block Jacobi over ranks, one AMG V-cycle per
    block) finds the exact discrete solution on 2 ranks."""
    history = run_example(2, tmp_path, "--p", "1.5", "3", "5")
    assert all(entry["error"] < 1e-8 for entry in history.values())


def test_amg_runs_are_reproducible(tmp_path: Path) -> None:
    """Two identical runs give identical histories. (pyamg's default
    prolongation smoothing estimates a spectral radius from an unseeded random
    vector; the example uses 'local' weighting to avoid it. The p = 5 path is
    sensitive enough to show any difference.)"""
    first = run_example(2, tmp_path, "--p", "5")
    (tmp_path / "history_2.json").rename(tmp_path / "first.json")
    second = run_example(2, tmp_path, "--p", "5")
    assert first == second
