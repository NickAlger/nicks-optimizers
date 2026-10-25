"""Side-by-side comparison with the package as it was before the backend
refactor (dev/jit_backends_plan.md), vendored as nicks_optimizers_legacy.

Each case builds the same problem with each package's own constructors and
compares the runs. The default is bit for bit: the records, the summary and
the final point compared through repr. A case marked exact=False (after a
deliberate, numerically sound change that alters rounding) is compared with
tolerances instead, with identical counts, stop reasons and acceptances.
This file and the vendored copy go away when the refactor is done.
"""

import dataclasses
import math
from typing import Any, Callable, NamedTuple

import numpy as np
import pytest

import matrix_completion as mc
import nicks_optimizers as new
import nicks_optimizers_legacy as old
from problems import (
    METRICS,
    PAIR_SPACE,
    jacobi_precond_solve,
    join_pair,
    pa_jacobi_precond_solve,
    pa_retract,
    pa_rosen_fun,
    pa_rosen_grad,
    pa_rosen_hessvec,
    rosen_fun,
    rosen_grad,
    rosen_hessvec,
    split_pair,
    well_fun,
    well_grad,
    well_hessvec,
    well_precond_solve,
)

Run = Any                      # (x, summary), from either package


def is_new(pkg: Any) -> bool:
    return pkg is new


def fingerprint(run: Run) -> list[str]:
    """Everything a run returns, as text (repr gives floats exactly and nan
    compares equal), with wall-clock fields cleared."""
    x, summary = run
    records = [repr(dataclasses.replace(r, elapsed=0.0)) for r in summary.iterations]
    rest = repr(dataclasses.replace(summary, iterations=(), elapsed=0.0))
    return records + [rest, repr(as_lists(x))]


def as_lists(x: Any) -> Any:
    if isinstance(x, tuple):
        return tuple(as_lists(leaf) for leaf in x)
    return np.asarray(x).tolist()


def close(a: Any, b: Any, rtol: float, path: str = "") -> list[str]:
    """Differences between two records (or values), recursively: exact for
    ints, bools and strings, relative tolerance for floats."""
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        out: list[str] = []
        for f in dataclasses.fields(a):
            if f.name != "elapsed":
                out += close(getattr(a, f.name), getattr(b, f.name), rtol, f"{path}.{f.name}")
        return out
    if isinstance(a, (tuple, list)):
        if len(a) != len(b):
            return [f"{path}: lengths {len(a)} and {len(b)}"]
        return [d for i, (u, v) in enumerate(zip(a, b)) for d in close(u, v, rtol, f"{path}[{i}]")]
    if isinstance(a, float) or isinstance(b, float):
        fa, fb = float(a), float(b)
        if math.isnan(fa) and math.isnan(fb) or fa == fb:
            return []
        ok = abs(fa - fb) <= rtol * max(abs(fa), abs(fb), 1e-300)
        return [] if ok else [f"{path}: {fa!r} vs {fb!r}"]
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


class Case(NamedTuple):
    name: str
    run: Callable[[Any], Run]
    exact: bool = True
    rtol: float = 1e-9


def rosenbrock_x0(n: int) -> Any:
    return np.tile([-1.2, 1.0], n // 2)


def well_x0(n: int) -> Any:
    return np.random.default_rng(3).uniform(-0.3, 0.3, n)


def newton(fun: Any, grad: Any, hessvec: Any, x0: Any, **options: Any) -> Callable[[Any], Run]:
    """A case: trust_region_newton with these arguments. An option given as
    "pkg.<name>" is looked up in the package under test."""
    def run(pkg: Any) -> Run:
        opts = {k: getattr(pkg, v[4:]) if isinstance(v, str) and v.startswith("pkg.") else v
                for k, v in options.items()}
        return pkg.trust_region_newton(fun, grad, hessvec, x0, **opts)
    return run


def numpy_cases() -> list[Case]:
    cases: list[Case] = []
    rosen = (rosen_fun, rosen_grad, rosen_hessvec, rosenbrock_x0(20))
    for label, solve, apply in METRICS:
        tag = label.split(",")[0]
        for krylov in ("auto", "lanczos"):
            cases.append(Case(f"rosenbrock {tag} {krylov}",
                              newton(*rosen, metric_solve=solve, krylov=krylov)))
        for krylov in ("lanczos", "flexible"):
            cases.append(Case(f"rosenbrock {tag} jacobi {krylov}",
                              newton(*rosen, metric_solve=solve, metric_apply=apply,
                                     precond_solve=jacobi_precond_solve, krylov=krylov)))
    cases.append(Case("rosenbrock saddle-free", newton(*rosen, saddle_free=True)))
    cases.append(Case("rosenbrock preconditioner norm",
                      newton(*rosen, precond_solve=jacobi_precond_solve,
                             convergence_norm="preconditioner")))
    cases.append(Case("rosenbrock small radius, few Krylov steps",
                      newton(*rosen, radius=1e-3, max_krylov=3, max_iter=60)))
    well = (well_fun, well_grad, well_hessvec, well_x0(30))
    for sf in (False, True):
        for krylov in ("auto", "lanczos", "flexible"):
            if krylov == "flexible" or sf:
                runner = newton(*well, precond_solve=well_precond_solve,
                                metric_solve="pkg.euclidean_metric_solve",
                                metric_apply="pkg.euclidean_metric_apply", saddle_free=sf,
                                krylov=krylov)
            else:
                runner = newton(*well, krylov=krylov)
            cases.append(Case(f"double well saddle_free={sf} {krylov}", runner))
    return cases


def operators_at_case(pkg: Any) -> Run:
    """Per-point operators with a separate preconditioner."""
    x0 = rosenbrock_x0(20)

    def operators_at(x: Any) -> Any:
        return pkg.PointOperators(hessvec=lambda v: rosen_hessvec(x, v),
                                  precond_solve=lambda w: jacobi_precond_solve(x, w))
    if is_new(pkg):                    # the setup is point data, operators_at(x, data) wraps it
        return pkg.trust_region_newton(rosen_fun, rosen_grad, None, x0, point_data=operators_at,
                                       operators_at=lambda x, ops: ops)
    return pkg.trust_region_newton(rosen_fun, rosen_grad, None, x0, operators_at=operators_at)


def resumed_case(pkg: Any) -> Run:
    """A run stopped by max_iter and resumed from its summary."""
    x0 = rosenbrock_x0(20)
    x, first = pkg.trust_region_newton(rosen_fun, rosen_grad, rosen_hessvec, x0, max_iter=15)
    x, second = pkg.trust_region_newton(rosen_fun, rosen_grad, rosen_hessvec, x, resume=first.resume,
                                        max_iter=500)
    return x, pkg.merge_summaries(first, second)


def tree_case(pkg: Any, krylov: str) -> Run:
    x0 = split_pair(rosenbrock_x0(20))
    space = pkg.tuple_tree_space()
    precond = None if krylov == "auto" else pa_jacobi_precond_solve
    apply = None if krylov == "auto" else pkg.euclidean_metric_apply
    return pkg.trust_region_newton(pa_rosen_fun, pa_rosen_grad, pa_rosen_hessvec, x0, space=space,
                                   retract=pkg.flat_retraction(space),
                                   metric_solve=pkg.euclidean_metric_solve,
                                   metric_apply=apply, precond_solve=precond, krylov=krylov)


def pair_space_case(pkg: Any) -> Run:
    """A space from tuple_block_space (each package builds its own)."""
    sp = PAIR_SPACE
    space = pkg.tuple_block_space(
        scale_vector=sp.scale_vector, scale_covector=sp.scale_covector,
        axpy_vector=sp.axpy_vector, axpy_covector=sp.axpy_covector, pair=sp.pair,
        dimension=sp.dimension)
    return pkg.trust_region_newton(pa_rosen_fun, pa_rosen_grad, pa_rosen_hessvec,
                                   split_pair(rosenbrock_x0(20)), space=space, retract=pa_retract,
                                   metric_solve=pkg.euclidean_metric_solve, krylov="lanczos")


def distributed_case(pkg: Any) -> Run:
    """numpy_space_with_reductions with a one-process allreduce."""
    space = pkg.numpy_space_with_reductions(lambda a: a)
    return pkg.trust_region_newton(rosen_fun, rosen_grad, rosen_hessvec, rosenbrock_x0(20),
                                   space=space, retract=pkg.flat_retraction(space),
                                   metric_solve=pkg.euclidean_metric_solve)


def matrix_completion_case(pkg: Any, krylov: str) -> Run:
    """A rank-3 fit on the fixed-rank manifold (tuple trees, gauge
    projection), from the example's problem functions."""
    rng = np.random.default_rng(0)
    shape, r = (60, 50), 3
    a = mc.random_low_rank(shape, 0.5 ** np.arange(8, dtype=float), rng)
    samples = mc.sample_uniformly(a, 4 * (shape[0] + shape[1] - r) * r, rng)
    x0 = mc.random_point(shape, r, rng)
    space = dataclasses.replace(pkg.tuple_tree_space(), dimension=mc.tangent_dimension)
    return pkg.trust_region_newton(
        lambda x: mc.objective(samples, x), lambda x: mc.gradient(samples, x),
        lambda x, v: mc.hessian_action(samples, x, v), x0, space=space, retract=mc.retract,
        metric_solve=pkg.euclidean_metric_solve, project_covector=mc.gauge_projection,
        krylov=krylov, max_iter=60)


CASES: list[Case] = numpy_cases() + [
    Case("operators_at", operators_at_case),
    Case("resumed", resumed_case),
    Case("tuple tree CG", lambda pkg: tree_case(pkg, "auto")),
    Case("tuple tree flexible", lambda pkg: tree_case(pkg, "flexible")),
    Case("tuple_block_space", pair_space_case),
    Case("distributed numpy space", distributed_case),
    Case("matrix completion CG", lambda pkg: matrix_completion_case(pkg, "auto")),
    Case("matrix completion lanczos", lambda pkg: matrix_completion_case(pkg, "lanczos")),
]


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_matches_legacy(case: Case) -> None:
    a, b = case.run(old), case.run(new)
    assert len(a[1].iterations) > 1
    if case.exact:
        assert fingerprint(a) == fingerprint(b)
        return
    differences = close(a[1], b[1], case.rtol)
    assert not differences, differences[:5]


def dense_subproblems() -> list[tuple[str, Callable[[Any], Any]]]:
    """Subproblem solves on dense random problems at several radii: plain,
    saddle-free, separate P, CG and flexible."""
    rng = np.random.default_rng(11)
    n = 30
    Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
    H = (Q * np.linspace(-2.0, 10.0, n)) @ Q.T
    S = rng.standard_normal((n, n))
    M = S @ S.T / n + np.eye(n)
    P = np.diag(np.abs(np.diag(H)) + 1.0)
    g = rng.standard_normal(n)
    out: list[tuple[str, Callable[[Any], Any]]] = []
    for radius in (1e-2, 1.0, 1e2):
        def lanczos(pkg: Any, r: float = radius, sf: bool = False) -> Any:
            return pkg.solve_tr_subproblem(
                pkg.numpy_space, g, lambda v: H @ v, lambda w: np.linalg.solve(M, w), r, 1e-10, 40,
                precond_solve=lambda w: np.linalg.solve(P, w), metric_apply=lambda v: M @ v,
                saddle_free=sf)

        def cg(pkg: Any, r: float = radius) -> Any:
            return pkg.solve_tr_subproblem_cg(pkg.numpy_space, g, lambda v: H @ v,
                                              lambda w: np.linalg.solve(M, w), r, 1e-10, 40)

        def flexible(pkg: Any, r: float = radius) -> Any:
            return pkg.solve_tr_subproblem_flexible(
                pkg.numpy_space, g, lambda v: H @ v, lambda w: np.linalg.solve(M, w),
                lambda v: M @ v, lambda w: np.linalg.solve(P, w), r, 1e-10, 40)

        def saddle_free(pkg: Any, r: float = radius) -> Any:
            return lanczos(pkg, r, True)
        out += [(f"lanczos r={radius}", lanczos), (f"lanczos saddle-free r={radius}", saddle_free),
                (f"cg r={radius}", cg), (f"flexible r={radius}", flexible)]
    return out


SUBPROBLEMS = dense_subproblems()


@pytest.mark.parametrize("name,solve", SUBPROBLEMS, ids=[s[0] for s in SUBPROBLEMS])
def test_subproblem_matches_legacy(name: str, solve: Callable[[Any], Any]) -> None:
    (s_old, st_old), (s_new, st_new) = solve(old), solve(new)
    assert repr(st_old) == repr(st_new)
    assert np.asarray(s_old).tobytes() == np.asarray(s_new).tobytes()


def test_cases_exercise_every_solver() -> None:
    """The matrix covers CG, Lanczos and flexible solves, and rejected steps."""
    seen: set[str] = set()
    rejected = 0
    for case in CASES:
        _, summary = case.run(old)
        seen |= {r.subproblem.krylov for r in summary.iterations}
        rejected += sum(not r.accepted for r in summary.iterations)
    assert seen == {"cg", "lanczos", "flexible"} and rejected > 0
    assert join_pair(split_pair(rosenbrock_x0(4))).tolist() == rosenbrock_x0(4).tolist()
