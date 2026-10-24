"""Vector spaces other than plain 1-D numpy arrays: consistency, arrays of
any shape, tuple trees (of numpy arrays and of other leaf types, given as
tuples or lists), a composite representation, and an SPMD run with threads
standing in for MPI ranks."""

import dataclasses
import threading
from typing import Any, Callable, Literal

import numpy as np
import pytest

from nicks_optimizers import (
    Array,
    Covector,
    NumpySpace,
    Point,
    RunSummary,
    Tree,
    Vector,
    VectorSpace,
    as_tuple_tree,
    check_space,
    history_array,
    numpy_space,
    numpy_space_with_reductions,
    trust_region_newton,
    tuple_block_space,
    tuple_tree_space,
)
from helpers import (
    Rng,
    report,
    without_times,
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


@dataclasses.dataclass(frozen=True)
class Box:
    """A leaf type that is not a numpy array (nor a tuple): an array in a box."""
    data: Array


BOX_SPACE = tuple_block_space(
    scale_vector=lambda a, v: Box(a * v.data), scale_covector=lambda a, w: Box(a * w.data),
    axpy_vector=lambda a, v, z: Box(a * v.data + z.data),
    axpy_covector=lambda a, w, z: Box(a * w.data + z.data),
    pair=lambda w, v: float(w.data.ravel() @ v.data.ravel()),
    translate=lambda x, s: Box(x.data + s.data), dimension=lambda v: int(v.data.size))


def random_tree(rng: Rng) -> Tree[Array]:
    """((1-D, 2-D), 0-d, ((), 1-D)): mixed shapes, a scalar leaf and an empty subtree."""
    return ((rng.standard_normal(3), rng.standard_normal((2, 4))), rng.standard_normal(()),
            ((), rng.standard_normal(5)))


def test_tree_spaces_are_consistent() -> None:
    """check_space for numpy arrays of any shape and for tuple trees over
    numpy, distributed numpy and boxed leaves; the dimension adds up the
    leaves."""
    rng = np.random.default_rng(9)
    samples: list[Any] = [rng.standard_normal((3, 4, 2)) for _ in range(4)]
    assert check_space(numpy_space, *samples) < 1e-14
    trees: list[Any] = [random_tree(rng) for _ in range(4)]
    tree_space = tuple_tree_space()
    assert check_space(tree_space, *trees) < 1e-14
    assert tree_space.dimension is not None and tree_space.dimension(trees[0]) == 3 + 8 + 1 + 5
    distributed = tuple_tree_space(numpy_space_with_reductions(lambda a: a))
    assert check_space(distributed, *trees) < 1e-14
    boxed: list[Any] = [as_tuple_tree([Box(rng.standard_normal(4)), [Box(rng.standard_normal((2, 2)))]])
                        for _ in range(4)]
    assert check_space(tuple_tree_space(BOX_SPACE), *boxed) < 1e-14


def test_tree_operations_refuse_lists_and_mismatched_structures() -> None:
    """Inside the method trees are tuples: a list is an error (user objects are
    converted on entry), and so is a pairing of trees with different shapes."""
    sp = tuple_tree_space()
    a, b = np.ones(2), np.ones(3)
    with pytest.raises(TypeError):
        sp.pair((a, [b]), (a, (b,)))                       # type: ignore[arg-type]
    with pytest.raises(ValueError):
        sp.pair((a, b), (a, (b,)))                         # type: ignore[arg-type]
    with pytest.raises(ValueError):
        sp.axpy_vector(1.0, (a, b), (a, b, b))             # type: ignore[arg-type]
    assert as_tuple_tree([a, [b, (a, [b])]]) == (a, (b, (a, (b,))))   # leaves are not copied


def test_numpy_space_takes_arrays_of_any_shape() -> None:
    """Rosenbrock with x stored as a 25 x 2 array takes exactly the steps of
    the 1-D run: products flatten the vector axes."""
    x0 = np.tile([-1.2, 1.0], 25)
    _, flat = trust_region_newton(rosen_fun, rosen_grad, rosen_hessvec, Point(x0),
                                  metric_solve=jacobi_precond_solve)
    krylov: Literal["auto", "lanczos"]
    for krylov in ("auto", "lanczos"):
        x, shaped = trust_region_newton(
            lambda X: rosen_fun(Point(X.ravel())),
            lambda X: Covector(rosen_grad(Point(X.ravel())).reshape(X.shape)),
            lambda X, V: Covector(rosen_hessvec(Point(X.ravel()), Vector(V.ravel())).reshape(X.shape)),
            Point(x0.reshape(25, 2)),
            metric_solve=lambda X, W: Vector(jacobi_precond_solve(Point(X.ravel()), Covector(W.ravel()))
                                             .reshape(X.shape)),
            krylov=krylov)
        assert x.shape == (25, 2)
        if krylov == "auto":
            assert without_times(shaped.iterations) == without_times(flat.iterations)
        assert shaped.reason == "converged"


def to_tree(x: Array) -> Tree[Array]:
    """Rosenbrock's 50 unknowns as the tuple tree ((x[:10], 4 x 5 matrix), x[30:])."""
    return ((x[:10], x[10:30].reshape(4, 5)), x[30:])


def to_list_tree(x: Array) -> Any:
    """The same tree written with lists."""
    return [[x[:10], x[10:30].reshape(4, 5)], x[30:]]


def from_tree(t: Any) -> Array:
    """Flatten a tree (tuples or lists) of to_tree's structure."""
    return np.concatenate([t[0][0], t[0][1].ravel(), t[1]])


def _map_leaves(f: Callable[[Any], Any], t: Any) -> Any:
    """f applied to the leaves of a tree, keeping tuples and lists as they are."""
    if isinstance(t, (tuple, list)):
        return type(t)(_map_leaves(f, c) for c in t)
    return f(t)


def tree_run(build: Callable[[Array], Any], boxed: bool = False,
             saddle_free: bool = False) -> tuple[Any, RunSummary]:
    """Rosenbrock on trees made by build (to_tree or to_list_tree), with the
    Jacobi preconditioner as the metric; boxed puts every leaf in a Box."""
    def wrap(a: Array) -> Any:
        return _map_leaves(Box, build(a)) if boxed else build(a)

    def flat(t: Any) -> Array:
        return from_tree(_map_leaves(lambda b: b.data, t) if boxed else t)

    def fun(x: Any) -> float:
        return rosen_fun(Point(flat(x)))

    def grad(x: Any) -> Any:
        return wrap(rosen_grad(Point(flat(x))))

    def hessvec(x: Any, v: Any) -> Any:
        return wrap(rosen_hessvec(Point(flat(x)), Vector(flat(v))))

    def metric_solve(x: Any, w: Any) -> Any:
        return wrap(jacobi_precond_solve(Point(flat(x)), Covector(flat(w))))

    space: VectorSpace[Any, Any, Any, Any, Any] = \
        tuple_tree_space(BOX_SPACE) if boxed else tuple_tree_space()
    return trust_region_newton(fun, grad, hessvec, wrap(np.tile([-1.2, 1.0], 25)),
                               metric_solve=metric_solve, space=space, saddle_free=saddle_free)


def _is_tuple_tree(t: Any) -> bool:
    if isinstance(t, list):
        return False
    return all(_is_tuple_tree(c) for c in t) if isinstance(t, tuple) else True


def test_tuple_tree_runs() -> None:
    """Rosenbrock on a nested tuple tree (a 1-D, a 2-D and a 1-D leaf):
    the same path as the flat numpy run, up to rounding (pairings add up the
    leaves). Given as lists (x0 and every function result), the run is
    identical and returns tuple trees. Leaves of another type (boxed
    arrays) give the same path up to rounding. Plain (CG) and saddle-free
    (Lanczos)."""
    x0 = Point(np.tile([-1.2, 1.0], 25))
    for saddle_free in (False, True):
        _, flat = trust_region_newton(rosen_fun, rosen_grad, rosen_hessvec, x0,
                                      metric_solve=jacobi_precond_solve, saddle_free=saddle_free)
        x_t, tuples = tree_run(to_tree, saddle_free=saddle_free)
        x_l, lists = tree_run(to_list_tree, saddle_free=saddle_free)
        x_b, boxes = tree_run(to_tree, boxed=True, saddle_free=saddle_free)
        h_flat, h_t = history_array(flat), history_array(tuples)
        assert tuples.reason == "converged" and len(h_t) == len(h_flat)
        assert np.allclose(h_t[:, 1:4], h_flat[:, 1:4], rtol=1e-8, atol=0.0)
        assert without_times(lists.iterations) == without_times(tuples.iterations)
        assert _is_tuple_tree(x_l) and np.array_equal(from_tree(x_l), from_tree(x_t))
        # Boxed leaves come with tuple blocks (tuple_block_space): block products
        # one vector at a time, so Lanczos rounds differently.
        h_b = history_array(boxes)
        assert len(h_b) == len(h_t) and np.allclose(h_b[:, 1:4], h_t[:, 1:4], rtol=1e-8, atol=0.0)
        x_unboxed = from_tree(_map_leaves(lambda b: b.data, x_b))
        assert np.allclose(x_unboxed, from_tree(x_t), rtol=0.0, atol=1e-10)


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
