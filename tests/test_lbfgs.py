"""lbfgs (dev/lbfgs_plan.md, phase 2): equivalence with textbook L-BFGS
(M = I, and a constant metric through a change of variables), convergence
with every option, a point-dependent metric on a flat space, a manifold with
projection transports, restarts, fallbacks and skipped pairs, exact
resumption, display, operator calls, and the strict backend."""

from typing import Any, Callable, Optional

import numpy as np
import pytest

from nicks_optimizers import (
    Array,
    Covector,
    LbfgsInfo,
    LbfgsResume,
    LbfgsSummary,
    Point,
    Vector,
    VectorTransport,
    array_space,
    backtracking,
    euclidean_metric_apply,
    euclidean_metric_solve,
    format_lbfgs_header,
    format_lbfgs_row,
    format_lbfgs_summary,
    identity_transport,
    lbfgs,
    lbfgs_history_array,
    merge_lbfgs_summaries,
    numpy_backend,
    numpy_space,
)
from nicks_optimizers.lbfgs_core import CurvatureTest, Pairings, Scaling
from nicks_optimizers.line_search import ARMIJO
from problems import rosen_fun, rosen_grad, rosen_hessvec, well_fun, well_grad
from strict_backend import strict_backend

X0 = Point(np.tile([-1.2, 1.0], 10))


def iterates(fun: Callable[..., Any], grad: Callable[..., Any], x0: Any, **options: Any
             ) -> tuple[list[Array], Any]:
    """lbfgs's iterates (x0 and every accepted point) and its summary."""
    xs = [np.array(x0, dtype=float)]

    def collect(x: Any, info: LbfgsInfo[Any, Any]) -> None:
        if info.record.accepted:
            xs.append(np.array(x, dtype=float))
    _, summary = lbfgs(fun, grad, x0, callback=collect, **options)
    return xs, summary


def textbook(fun: Callable[[Any], float], grad: Callable[[Any], Any], x0: Array, *, m: int = 10,
             max_iter: int = 1000, rtol: float = 1e-8, curvature_tol: float = 1e-8,
             angle_tol: float = 1e-8) -> list[Array]:
    """Textbook L-BFGS with H0 = gamma I (Nocedal and Wright, Algorithms 7.4
    and 7.5): the sequential two-loop recursion, gamma = s^T y / y^T y, the
    same backtracking line search, curvature test, angle test and restarts
    as lbfgs. Returns the iterates."""
    x, f, g = x0.copy(), fun(x0), grad(x0)
    gnorm0, gamma = float(np.linalg.norm(g)), 1.0
    S: list[Array] = []
    Y: list[Array] = []
    xs = [x.copy()]
    for _ in range(max_iter):
        if np.linalg.norm(g) <= rtol * gnorm0:
            break
        q, alphas = g.copy(), []
        for s, y in zip(reversed(S), reversed(Y)):
            a = float(s @ q) / float(y @ s)
            alphas.append(a)
            q = q - a * y
        r = gamma * q
        for (s, y), a in zip(zip(S, Y), reversed(alphas)):
            r = r + (a - float(y @ r) / float(y @ s)) * s
        d = -r
        slope = float(g @ d)
        cosine = -slope / (np.linalg.norm(g) * np.linalg.norm(d))
        fallback = bool(S) and not (np.isfinite(slope) and slope < 0.0 and cosine >= angle_tol)
        if fallback:
            S, Y = [], []
        if not S:
            d = -gamma * g
            slope = float(g @ d)
        found = backtracking(numpy_backend, lambda alpha: (fun(x + alpha * d), x + alpha * d), x, f,
                             slope, 1.0)
        if found.status != ARMIJO:
            if S:
                S, Y = [], []
                continue
            break
        x1 = np.asarray(found.point)
        g1 = grad(x1)
        s, y = float(found.alpha) * d, g1 - g
        if float(y @ s) > curvature_tol * abs(float(found.alpha) * slope):
            S, Y = (S + [s])[-m:], (Y + [y])[-m:]
            gamma = float(y @ s) / float(y @ y)
        x, f, g = x1, float(found.f), g1
        xs.append(x.copy())
    return xs


def rel(a: Array, b: Array) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def convex(n: int = 40, seed: int = 4) -> tuple[Callable[[Array], float], Callable[[Array], Array], Array]:
    """A strictly convex quartic, 0.5 x^T A x - b^T x + 0.25 sum(x^4), on
    which rounding differences between runs grow only mildly."""
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
    A = Q @ np.diag(np.geomspace(1.0, 100.0, n)) @ Q.T
    b = rng.standard_normal(n)
    return (lambda x: float(0.5 * x @ A @ x - b @ x + 0.25 * np.sum(x ** 4)),
            lambda x: A @ x - b + x ** 3, rng.standard_normal(n))


def test_euclidean_metric_is_textbook_lbfgs() -> None:
    """With M = I (the default for numpy arrays), the iterates are those of
    textbook L-BFGS, to rounding: over a whole convex run, and over the first
    40 iterations on Rosenbrock (along its valley, rounding differences grow
    by orders of magnitude, as between any two implementations)."""
    fun, grad, x0 = convex()
    xs, summary = iterates(fun, grad, x0)
    reference = textbook(fun, grad, x0)
    assert summary.reason == "converged" and len(xs) == len(reference)
    assert max(rel(a, b) for a, b in zip(xs, reference)) < 1e-8
    xs, summary = iterates(rosen_fun, rosen_grad, X0)
    reference = textbook(rosen_fun, rosen_grad, X0)
    assert summary.reason == "converged"
    assert max(rel(a, b) for a, b in zip(xs[:40], reference[:40])) < 1e-10


def test_constant_metric_is_textbook_lbfgs_after_a_change_of_variables() -> None:
    """With a constant metric M = L L^T, the iterates are L^{-T} times those of
    textbook L-BFGS on f(L^{-T} z): BFGS is affine covariant, and the base,
    the scaling, the tests and the norms are all taken in M."""
    rng = np.random.default_rng(0)
    n = X0.shape[0]
    Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
    M = Q @ np.diag(np.geomspace(0.3, 3.0, n)) @ Q.T
    L = np.linalg.cholesky(M)
    xs, summary = iterates(rosen_fun, rosen_grad, X0, metric_solve=lambda x, w: np.linalg.solve(M, w),
                           metric_apply=lambda x, v: M @ v)

    def back(z: Array) -> Array:
        out: Array = np.linalg.solve(L.T, z)
        return out
    reference = textbook(lambda z: rosen_fun(Point(back(z))),
                         lambda z: np.linalg.solve(L, rosen_grad(Point(back(z)))), L.T @ X0)
    assert summary.reason == "converged"
    assert max(rel(xs[k], back(reference[k])) for k in range(40)) < 1e-9
    fun, grad, x0 = convex()
    Q2 = np.linalg.qr(np.random.default_rng(5).standard_normal((x0.shape[0],) * 2))[0]
    M2 = Q2 @ np.diag(np.geomspace(0.1, 10.0, x0.shape[0])) @ Q2.T
    L2 = np.linalg.cholesky(M2)
    xs, summary = iterates(fun, grad, x0, metric_solve=lambda x, w: np.linalg.solve(M2, w),
                           metric_apply=lambda x, v: M2 @ v)
    reference = textbook(lambda z: fun(np.linalg.solve(L2.T, z)),
                         lambda z: np.linalg.solve(L2, grad(np.linalg.solve(L2.T, z))), L2.T @ x0)
    assert summary.reason == "converged"            # (a long run: rounding grows over it)
    assert max(rel(a, np.linalg.solve(L2.T, z)) for a, z in zip(xs[:60], reference[:60])) < 1e-9


@pytest.mark.parametrize("options", [
    dict(), dict(scaling="s"), dict(scaling="geometric"), dict(scaling="none", initial_step=0.1),
    dict(memory=1), dict(memory=3), dict(curvature_test="positive"),
    dict(curvature_test="dennis-schnabel"), dict(curvature_test="cautious"),
    dict(pairings="recomputed"), dict(initial_step=1.0), dict(step_max=0.5),
], ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items()) or "defaults")
def test_options_converge_on_rosenbrock(options: dict[str, Any]) -> None:
    """Every option converges to a local minimizer (chained Rosenbrock has
    two: x = 1 and one with x_1 near -1)."""
    x, s = lbfgs(rosen_fun, rosen_grad, X0, max_iter=3000, **options)
    assert s.reason == "converged"
    H = np.column_stack([rosen_hessvec(x, Vector(e)) for e in np.eye(x.shape[0])])
    assert np.linalg.eigvalsh(0.5 * (H + H.T)).min() > 0.0
    if "step_max" in options:
        assert all(a * r.step_norm <= options["step_max"] * (1 + 1e-12)
                   for r in s.iterations for a in r.trace_alpha)
    if options.get("initial_step") is not None:
        assert abs(s.iterations[0].step_norm - options["initial_step"]) < 1e-12


def test_recomputed_pairings_change_nothing_on_a_flat_space() -> None:
    """With the identity transport the stored pairings are current, so both
    modes give the same run."""
    a, sa = iterates(rosen_fun, rosen_grad, X0)
    b, sb = iterates(rosen_fun, rosen_grad, X0, pairings="recomputed")
    assert len(a) == len(b) and all(np.array_equal(u, v) for u, v in zip(a, b))


def test_quadratic_with_a_point_dependent_metric() -> None:
    """A quadratic on a flat space with a metric that changes strongly from
    point to point: the identity transport keeps the pairs exact (y = A s),
    and the run converges."""
    rng = np.random.default_rng(1)
    n = 30
    A = rng.standard_normal((n, n))
    A = A @ A.T / n + np.diag(np.geomspace(1e-2, 1e2, n))
    b = rng.standard_normal(n)

    def weights(x: Array) -> Array:
        out: Array = 1.0 + 10.0 * x ** 2
        return out
    newest: list[tuple[Array, Array]] = []

    def watch(x: Any, info: LbfgsInfo[Any, Any]) -> None:
        if info.resume.pairs:
            newest.append(tuple(np.asarray(v) for v in info.resume.pairs[-1]))  # type: ignore[arg-type]
    x, s = lbfgs(lambda x: 0.5 * x @ A @ x - b @ x, lambda x: A @ x - b, np.zeros(n),
                 metric_solve=lambda x, w: w / weights(x), metric_apply=lambda x, v: weights(x) * v,
                 callback=watch, max_iter=2000)
    assert s.reason == "converged"
    assert rel(x, np.linalg.solve(A, b)) < 1e-6
    # exact up to the rounding of a difference of gradients near convergence
    assert max(rel(y, A @ s_) for s_, y in newest) < 1e-6


# --- the sphere: Rayleigh quotient with projection transports ---------------------------


def sphere_problem(n: int = 30, seed: int = 2) -> dict[str, Any]:
    """min 0.5 x^T A x on the unit sphere: tangent vectors v with x.v = 0,
    canonical covectors likewise (the Euclidean pairing), the normalizing
    retraction, and projection onto the new tangent space as the transport of
    both."""
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
    A = Q @ np.diag(np.arange(1.0, n + 1.0)) @ Q.T

    def project(x: Any, w: Any) -> Any:
        return w - (x @ w) * x

    move = lambda x, x1, v: project(x1, v)
    x0 = rng.standard_normal(n)
    return dict(fun=lambda x: 0.5 * x @ A @ x, grad=lambda x: project(x, A @ x),
                x0=x0 / np.linalg.norm(x0), smallest=Q[:, 0],
                options=dict(space=numpy_space, retract=lambda x, s: (x + s) / np.linalg.norm(x + s),
                             transport=VectorTransport(vector=move, covector=move),
                             metric_solve=euclidean_metric_solve, metric_apply=euclidean_metric_apply,
                             project_covector=project))


@pytest.mark.parametrize("pairings", ["stored", "recomputed"])
def test_sphere_with_projection_transports(pairings: Pairings) -> None:
    """The smallest eigenvector is found; the transports are called as
    counted, and their pairing defect is second order in the step: for a
    step s of length a, <P' g, P' s> = <g, s> - (x'.g)(x'.s) with x' the
    normalized x + s, so the defect is -a^2 / (1 + a^2) exactly."""
    p = sphere_problem()
    x, s = lbfgs(p["fun"], p["grad"], p["x0"], pairings=pairings, max_iter=500, **p["options"])
    assert s.reason == "converged"
    assert abs(abs(x @ p["smallest"]) - 1.0) < 1e-10
    for r in s.iterations:
        if r.accepted:
            a2 = (r.alpha * r.step_norm) ** 2
            assert abs(r.transport_defect + a2 / (1.0 + a2)) <= 1e-9 * a2 + 1e-15
    accepted = sum(r.accepted for r in s.iterations)
    used = sum(r.pairs for r in s.iterations if r.accepted and r.direction != "fallback")
    assert s.vector_transport_calls == s.covector_transport_calls == accepted + used
    assert s.block_transport_calls == 0


# --- restarts, fallbacks, skipped pairs -----------------------------------------------------


def test_restarts_after_failed_line_searches() -> None:
    """With one trial per search, failures happen; each clears the memory,
    and the next iteration starts from an empty memory."""
    s: LbfgsSummary[Any, Any]
    x, s = lbfgs(rosen_fun, rosen_grad, X0, max_backtracks=1, max_iter=3000, initial_step=0.1)
    assert s.restarts > 0
    records = s.iterations
    for before, after in zip(records, records[1:]):
        if before.restart:
            assert not before.accepted and after.pairs == 0 and after.direction == "steepest"
    assert s.reason in ("converged", "line_search")
    if s.reason == "line_search":                 # from an empty memory, or a fallback
        last = records[-1]
        assert not last.accepted and not last.restart
        assert last.pairs == 0 or last.direction == "fallback"


def test_fallbacks_clear_the_memory() -> None:
    """A strict angle test rejects quasi-Newton directions: the steepest
    direction is used instead, and the memory restarts from the new pair."""
    s: LbfgsSummary[Any, Any]
    x, s = lbfgs(rosen_fun, rosen_grad, X0, angle_tol=0.95, max_iter=5000)
    fallbacks = [k for k, r in enumerate(s.iterations) if r.direction == "fallback"]
    assert fallbacks
    for k in fallbacks:
        r = s.iterations[k]
        assert r.cosine == 1.0 and r.pairs > 0
        if r.accepted and k + 1 < len(s.iterations):
            assert s.iterations[k + 1].pairs <= 1


def test_negative_curvature_pairs_are_skipped() -> None:
    """Near the local maximum of the double well, steps see negative
    curvature: those pairs are skipped, and the run still converges."""
    x0 = Point(np.random.default_rng(3).uniform(-0.1, 0.1, 30))
    s: LbfgsSummary[Any, Any]
    x, s = lbfgs(well_fun, well_grad, x0, curvature_test="positive", max_iter=2000)
    assert s.reason == "converged" and s.skipped > 0
    skipped = [r for r in s.iterations if r.update == "skipped"]
    assert all(r.curvature <= 0.0 for r in skipped)


# --- resuming, display, calls ------------------------------------------------------------


def test_resuming_is_exact() -> None:
    """Saving x, info.resume and info.g in the callback and resuming gives
    the same iterates as the uninterrupted run, memory included."""
    full, s_full = iterates(rosen_fun, rosen_grad, X0)
    saved: dict[str, Any] = {}

    def stop_at_12(x: Any, info: LbfgsInfo[Any, Any]) -> bool:
        if info.record.iteration == 11:
            saved.update(x=np.array(x), resume=info.resume, g=np.array(info.g))
            return True
        return False
    _, first = lbfgs(rosen_fun, rosen_grad, X0, callback=stop_at_12)
    resume: LbfgsResume[Any, Any] = saved["resume"]
    assert len(resume.pairs) == 10 and resume.next_iteration == 12
    rest, second = iterates(rosen_fun, rosen_grad, saved["x"], resume=resume, g0=saved["g"])
    k = len(full) - len(rest)
    assert all(np.array_equal(a, b) for a, b in zip(full[k:], rest))
    merged = merge_lbfgs_summaries(first, second)
    assert len(merged.iterations) == len(s_full.iterations)
    assert merged.fun_calls == s_full.fun_calls + 1 and merged.grad_calls == s_full.grad_calls


def test_display_and_history() -> None:
    lines: list[str] = []
    s: LbfgsSummary[Any, Any]
    x, s = lbfgs(rosen_fun, rosen_grad, X0, display=2, print_fn=lines.append, max_iter=25)
    assert lines[0] == format_lbfgs_header()
    assert format_lbfgs_row(s.iterations[3]) in lines
    assert lines[-1] == format_lbfgs_summary(s)
    assert any("trial" in line for line in lines)
    history = lbfgs_history_array(s)
    assert history.shape == (25, 9) and np.all(history[:, 0] == np.arange(25))


def counted(f: Callable[..., Any], counter: dict[str, int], key: str) -> Callable[..., Any]:
    def wrapper(*args: Any) -> Any:
        counter[key] += 1
        return f(*args)
    return wrapper


@pytest.mark.parametrize("scaling,test", [("y", "relative"), ("s", "relative"), ("geometric", "cautious"),
                                          ("none", "dennis-schnabel")])
def test_operator_calls_match_the_work_table(scaling: Scaling, test: CurvatureTest) -> None:
    """Every operator is called as the plan's work table says, and as the
    summary reports: M^{-1} once per point for the norm, once per direction
    from a nonempty memory, once more for the "y" fit; M once per direction
    and once more for the "s" fit."""
    calls = dict(fun=0, grad=0, solve=0, apply=0)
    x, s = lbfgs(counted(rosen_fun, calls, "fun"), counted(rosen_grad, calls, "grad"), X0,
                 metric_solve=counted(euclidean_metric_solve, calls, "solve"),
                 metric_apply=counted(euclidean_metric_apply, calls, "apply"),
                 scaling=scaling, curvature_test=test, max_iter=60, initial_step=0.1)
    assert (calls["fun"], calls["grad"]) == (s.fun_calls, s.grad_calls)
    assert (calls["solve"], calls["apply"]) == (s.metric_solve_calls, s.metric_apply_calls)
    accepted = sum(r.accepted for r in s.iterations)
    from_memory = sum(r.pairs > 0 for r in s.iterations)
    need_yy = scaling in ("y", "geometric") or test == "dennis-schnabel"
    need_ss = scaling in ("s", "geometric") or test in ("dennis-schnabel", "cautious")
    assert s.metric_solve_calls == 1 + from_memory + accepted * (1 + need_yy)
    assert s.metric_apply_calls == len(s.iterations) + accepted * need_ss
    assert s.fun_calls == 1 + s.trials and s.grad_calls == 1 + accepted


def test_arguments_are_checked() -> None:
    f, g = rosen_fun, rosen_grad
    bad: list[dict[str, Any]] = [
        dict(metric_solve=euclidean_metric_solve),                       # M without M^{-1}'s partner
        dict(space=numpy_space, retract=lambda x, s: x + s),             # no transport
        dict(space=numpy_space, transport=identity_transport),           # no retract
        dict(space=numpy_space, retract=lambda x, s: x + s, transport=identity_transport),  # no metric
        dict(point_data=lambda x: None), dict(memory=0), dict(scaling="bb"), dict(pairings="corrected"),
        dict(curvature_test="wolfe"), dict(curvature_tol=-1.0), dict(angle_tol=1.0), dict(gamma0=0.0),
        dict(initial_step=0.0), dict(c1=1.0), dict(max_backtracks=0), dict(step_max=0.0),
        dict(operators_at=lambda x, d: None, metric_solve=euclidean_metric_solve,
             metric_apply=euclidean_metric_apply),
    ]
    for options in bad:
        with pytest.raises(ValueError):
            lbfgs(f, g, X0, max_iter=1, **options)


# --- backends -------------------------------------------------------------------------------


def test_strict_backend_agrees_with_numpy() -> None:
    """The code path of a compiling backend: array_space, fixed shapes, every
    iteration one compiled piece, no Python control flow on values."""
    options: dict[str, Any] = dict(retract=lambda x, s: x + s, transport=identity_transport,
                                   metric_solve=euclidean_metric_solve,
                                   metric_apply=euclidean_metric_apply, max_iter=40, memory=5)
    _, ref = lbfgs(rosen_fun, rosen_grad, X0, space=numpy_space, **options)
    _, run = lbfgs(lambda x: 100.0 * np.sum((x[1:] - x[:-1] ** 2) ** 2) + np.sum((1.0 - x[:-1]) ** 2),
                   rosen_grad, X0, space=array_space(strict_backend), backend=strict_backend, **options)
    assert [(r.direction, r.trials, r.update) for r in run.iterations] == \
        [(r.direction, r.trials, r.update) for r in ref.iterations]
    assert max(abs(a.f - b.f) / max(abs(b.f), 1.0) for a, b in zip(run.iterations, ref.iterations)) < 1e-10


def test_strict_backend_transports_member_by_member() -> None:
    p = sphere_problem(n=12)
    options = dict(p["options"], max_iter=40, memory=4)
    _, ref = lbfgs(p["fun"], p["grad"], p["x0"], **options)
    options["space"] = array_space(strict_backend)
    _, run = lbfgs(p["fun"], p["grad"], p["x0"], backend=strict_backend, **options)
    assert [(r.direction, r.trials) for r in run.iterations] == [(r.direction, r.trials) for r in ref.iterations]
    assert abs(run.f - ref.f) < 1e-12 and run.vector_transport_calls == ref.vector_transport_calls
