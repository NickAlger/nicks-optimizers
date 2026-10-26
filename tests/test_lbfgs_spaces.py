"""lbfgs in other vector spaces (dev/lbfgs_plan.md, phase 4): distributed
arrays (threads standing in for MPI ranks, in lockstep with a serial run),
tuple trees, and a composite representation from tuple_block_space."""

import threading
from typing import Any, Optional

import numpy as np
import pytest

from nicks_optimizers import (
    LbfgsSummary,
    Point,
    euclidean_metric_apply,
    euclidean_metric_solve,
    flat_retraction,
    identity_transport,
    lbfgs,
    lbfgs_history_array,
    numpy_space,
    numpy_space_with_reductions,
    tuple_tree_space,
)
from problems import (
    PAIR_SPACE,
    coupled_problem,
    join_pair,
    make_thread_allreduce,
    pa_retract,
    pa_rosen_fun,
    pa_rosen_grad,
    pa_smoothing_metric_apply,
    pa_smoothing_metric_solve,
    rosen_fun,
    rosen_grad,
    smoothing_metric_apply,
    smoothing_metric_solve,
    split_pair,
)
from test_lbfgs import convex, rel


@pytest.mark.parametrize("options", [dict(), dict(precond=True), dict(precond=True, fixed_metric=True)],
                         ids=["M", "M and P", "fixed M and P"])
def test_distributed_runs_stay_in_lockstep(options: dict[str, Any]) -> None:
    """Three threads with their shares of x, every reduction through an
    allreduce: all ranks record bitwise-identical histories, and the result
    is the serial run's."""
    rng = np.random.default_rng(8)
    n, n_ranks = 40, 3
    b, d = rng.standard_normal(n), 1.0 + np.linspace(0.0, 4.0, n)
    x0 = 0.1 * rng.standard_normal(n)
    pieces = np.array_split(np.arange(n), n_ranks)
    use_P, fixed = options.get("precond", False), options.get("fixed_metric", False)

    def solve(problem: Any, x_start: Point, space: Any) -> tuple[Any, LbfgsSummary[Any, Any]]:
        fun, grad, _, precond_solve, metric_solve, metric_apply = problem
        return lbfgs(fun, grad, x_start, space=space, retract=flat_retraction(space),
                     transport=identity_transport, metric_solve=metric_solve, metric_apply=metric_apply,
                     precond_solve=precond_solve if use_P else None, fixed_metric=fixed,
                     gtol=1e-10, rtol=0.0, max_iter=300)

    x_serial, s_serial = solve(coupled_problem(b, d, lambda a: a), Point(x0.copy()), numpy_space)
    for_rank = make_thread_allreduce(n_ranks)
    results: list[Optional[tuple[Any, LbfgsSummary[Any, Any]]]] = [None] * n_ranks
    errors: list[BaseException] = []

    def worker(rank: int) -> None:
        try:
            allreduce = for_rank(rank)
            idx = pieces[rank]
            results[rank] = solve(coupled_problem(b[idx], d[idx], allreduce), Point(x0[idx].copy()),
                                  numpy_space_with_reductions(allreduce))
        except BaseException as exc:          # surfaced below, so a failure cannot hang the test
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(r,)) for r in range(n_ranks)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and all(r is not None for r in results), errors[:1]
    done = [r for r in results if r is not None]
    histories = [lbfgs_history_array(summary) for _, summary in done]
    assert all(np.array_equal(histories[0], h) for h in histories[1:])
    assert s_serial.reason == "converged" and all(summary.reason == "converged" for _, summary in done)
    assert np.max(np.abs(np.concatenate([x for x, _ in done]) - x_serial)) < 1e-10
    if use_P:
        assert done[0][1].precond_directions > 0


def test_tuple_trees_give_the_flat_run() -> None:
    """The convex problem on a tuple tree of two arrays (pairings add the
    leaves' sums, so rounding differs): the flat run's iterates."""
    fun, grad, x0 = convex()
    space = tuple_tree_space()

    def split(x: Any) -> Any:
        return (x[:15], (x[15:],))

    def join(t: Any) -> Any:
        return np.concatenate([t[0], t[1][0]])
    flat_xs: list[Any] = [x0]
    tree_xs: list[Any] = [x0]
    lbfgs(fun, grad, x0, callback=lambda x, info: flat_xs.append(np.array(x)) if info.record.accepted else None)
    _, s = lbfgs(lambda t: fun(join(t)), lambda t: split(grad(join(t))), split(x0), space=space,
                 retract=flat_retraction(space), transport=identity_transport,
                 metric_solve=euclidean_metric_solve, metric_apply=euclidean_metric_apply,
                 callback=lambda t, info: tree_xs.append(join(t)) if info.record.accepted else None)
    assert s.reason == "converged" and len(tree_xs) == len(flat_xs)
    assert max(rel(a, b) for a, b in zip(tree_xs, flat_xs)) < 1e-8


def test_a_composite_representation_gives_the_numpy_run() -> None:
    """Rosenbrock with x stored as a pair of arrays (tuple_block_space, blocks
    as tuples of members) and a point-dependent smoothing metric: the numpy
    run, to rounding, while the paths agree."""
    x0 = np.tile([-1.2, 1.0], 10)
    a_xs: list[Any] = [x0]
    b_xs: list[Any] = [x0]
    lbfgs(rosen_fun, rosen_grad, Point(x0), metric_solve=smoothing_metric_solve,
          metric_apply=smoothing_metric_apply,
          callback=lambda x, info: a_xs.append(np.array(x)) if info.record.accepted else None)
    _, s = lbfgs(pa_rosen_fun, pa_rosen_grad, split_pair(x0), space=PAIR_SPACE, retract=pa_retract,
                 transport=identity_transport, metric_solve=pa_smoothing_metric_solve,
                 metric_apply=pa_smoothing_metric_apply,
                 callback=lambda x, info: b_xs.append(join_pair(x)) if info.record.accepted else None)
    assert s.reason == "converged"
    assert max(rel(a, b) for a, b in zip(a_xs[:40], b_xs[:40])) < 1e-9
