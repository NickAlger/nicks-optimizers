"""Vector spaces other than plain numpy arrays: consistency, a composite
representation, and an SPMD run with threads standing in for MPI ranks."""

import dataclasses
import threading

import numpy as np

from nicks_optimizers import (
    Covector,
    NumpySpace,
    Point,
    RunSummary,
    Vector,
    check_space,
    history_array,
    numpy_space,
    numpy_space_with_reductions,
    trust_region_newton,
)
from helpers import (
    Rng,
    report,
)
from problems import (
    PAIR_SPACE,
    CoupledProblem,
    coupled_problem,
    jacobi_precond_solve,
    join_pair,
    make_thread_allreduce,
    pa_jacobi_precond_solve,
    pa_rosen_fun,
    pa_rosen_grad,
    pa_rosen_hessvec,
    pa_smoothing_metric_apply,
    pa_smoothing_metric_solve,
    rosen_fun,
    rosen_grad,
    rosen_hessvec,
    smoothing_metric_apply,
    smoothing_metric_solve,
    split_pair,
)


def check_space_consistency(rng: Rng) -> bool:
    print("VectorSpace records: internal consistency (check_space)")
    n = 30
    v1, v2 = Vector(rng.standard_normal(n)), Vector(rng.standard_normal(n))
    w1, w2 = Covector(rng.standard_normal(n)), Covector(rng.standard_normal(n))
    ok = report("numpy_space", check_space(numpy_space, v1, v2, w1, w2), 1e-14)
    single = numpy_space_with_reductions(lambda a: a)
    ok &= report("numpy_space_with_reductions, one process", check_space(single, v1, v2, w1, w2), 1e-14)
    ok &= report("two-array space from tuple_block_space",
                 check_space(PAIR_SPACE, split_pair(v1), split_pair(v2), split_pair(w1), split_pair(w2)), 1e-14)
    broken = dataclasses.replace(numpy_space, pair_block=lambda B, w: 2.0 * (B.T @ w))
    err = check_space(broken, v1, v2, w1, w2)
    detected = err > 0.1
    print(f"  [{'PASS' if detected else 'FAIL'}] a space whose pair_block disagrees with pair is "
          f"flagged: {err:.2e}")
    return ok and detected


def check_composite_representation() -> bool:
    print("Composite representation (each vector a pair of arrays) vs numpy, Rosenbrock n=50, smoothing metric")
    x0 = np.tile([-1.2, 1.0], 25)
    ok = True
    for label, separate in (("separate Jacobi P, Lanczos", True), ("P = M, CG", False)):
        np_ops: dict[str, object] = {"metric_solve": smoothing_metric_solve}
        pa_ops: dict[str, object] = {"metric_solve": pa_smoothing_metric_solve}
        if separate:
            np_ops.update(metric_apply=smoothing_metric_apply, precond_solve=jacobi_precond_solve)
            pa_ops.update(metric_apply=pa_smoothing_metric_apply, precond_solve=pa_jacobi_precond_solve)
        x_np, s_np = trust_region_newton(
            rosen_fun, rosen_grad, rosen_hessvec, Point(x0), gtol=1e-8, rtol=0.0, max_iter=1000,
            max_krylov=100, **np_ops)  # type: ignore[arg-type]
        x_pa, s_pa = trust_region_newton(
            pa_rosen_fun, pa_rosen_grad, pa_rosen_hessvec, split_pair(x0), space=PAIR_SPACE,
            gtol=1e-8, rtol=0.0, max_iter=1000, max_krylov=100, **pa_ops)  # type: ignore[arg-type]
        h_np, h_pa = history_array(s_np), history_array(s_pa)
        print(f"  {label}: numpy: {len(h_np)} its, {int(h_np[:, 5].sum())} Hv;  pair of arrays: "
              f"{len(h_pa)} its, {int(h_pa[:, 5].sum())} Hv")
        # Pairings of pairs add two partial sums, so rounding differs from numpy. Lanczos (with
        # reorthogonalization) keeps the paths together; along CG's path the differences grow,
        # so only its first 60 iterations are compared.
        m = len(h_np) if separate else 60
        same_path = (len(h_np) == len(h_pa) or not separate) and \
            np.allclose(h_np[:m, 1:4], h_pa[:m, 1:4], rtol=1e-8, atol=0.0)
        print(f"  [{'PASS' if same_path else 'FAIL'}] {label}: same iteration path (f, |g|, radius "
              f"agree to 1e-8 for {m} iterations)")
        ok &= same_path and s_np.reason == "converged"
        ok &= report(f"{label}: same final point", float(np.max(np.abs(join_pair(x_pa) - x_np))), 1e-10)
    return ok


def check_distributed_lockstep(rng: Rng) -> bool:
    print("numpy_space_with_reductions: 3 threads standing in for MPI ranks vs a serial run (n=40)")
    n, n_ranks = 40, 3
    b, d = rng.standard_normal(n), 1.0 + np.linspace(0.0, 4.0, n)
    x0 = 0.1 * rng.standard_normal(n)
    pieces = np.array_split(np.arange(n), n_ranks)          # uneven: 14, 13, 13

    def solve(problem: CoupledProblem, x_start: Point, space: NumpySpace) -> tuple[Point, RunSummary]:
        fun, grad, hessvec, precond_solve, metric_solve, metric_apply = problem
        return trust_region_newton(fun, grad, hessvec, x_start, metric_solve=metric_solve,
                                   metric_apply=metric_apply, precond_solve=precond_solve,
                                   space=space, gtol=1e-10, rtol=0.0, max_iter=200)

    serial = coupled_problem(b, d, lambda a: a)
    x_serial, s_serial = solve(serial, Point(x0.copy()), space=numpy_space)
    h_serial = history_array(s_serial)

    for_rank = make_thread_allreduce(n_ranks)
    results: list[tuple[Point, RunSummary] | None] = [None] * n_ranks
    errors: list[BaseException] = []

    def worker(rank: int) -> None:
        try:
            allreduce = for_rank(rank)
            idx = pieces[rank]
            local = coupled_problem(b[idx], d[idx], allreduce)
            results[rank] = solve(local, Point(x0[idx].copy()),
                                  space=numpy_space_with_reductions(allreduce))
        except BaseException as exc:      # surfaced below, so a failure cannot hang the test
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(r,)) for r in range(n_ranks)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors or any(r is None for r in results):
        print(f"  [FAIL] a rank failed: {errors[:1]}")
        return False
    done = [r for r in results if r is not None]
    hists = [history_array(summary) for _, summary in done]
    x_dist = np.concatenate([x for x, _ in done])
    lockstep = all(np.array_equal(hists[0], h) for h in hists[1:])
    print(f"  serial: {len(h_serial)} its;  distributed: {len(hists[0])} its on every rank")
    print(f"  [{'PASS' if lockstep else 'FAIL'}] all ranks record bitwise-identical histories (lockstep)")
    ok = report("distributed result matches the serial run", float(np.max(np.abs(x_dist - x_serial))), 1e-10)
    return ok and lockstep


def test_space_consistency() -> None:
    assert check_space_consistency(np.random.default_rng(7))


def test_composite_representation() -> None:
    assert check_composite_representation()


def test_distributed_lockstep() -> None:
    assert check_distributed_lockstep(np.random.default_rng(8))
