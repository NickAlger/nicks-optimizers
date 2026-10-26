"""lbfgs with a preconditioner (dev/lbfgs_plan.md, phase 3): P as an
accelerant whose proposals are tested in the geometry of M. P = M gives the
run without P; a P whose proposals always fail gives it too, through the M
direction; unconventional preconditioners (nonsymmetric, nonlinear, random)
converge; the scale fit in the M-norm; the preconditioner's convergence
norm; fixed_preconditioner; operator calls; the strict backend."""

from typing import Any, Callable, cast

import numpy as np
import pytest

from nicks_optimizers import (
    Array,
    LbfgsInfo,
    LbfgsSummary,
    PointOperators,
    array_space,
    format_lbfgs_row,
    format_lbfgs_summary,
    identity_transport,
    lbfgs,
    numpy_space,
)
from nicks_optimizers.lbfgs_core import PrecondMode
from test_lbfgs import convex, counted, iterates, rel, sphere_problem

N = 40


def matrices(seed: int = 7) -> tuple[Array, Array, Array]:
    """The Hessian-like part A of the convex test problem, a metric M and a
    rough approximation P0 of A (its diagonal times 2)."""
    rng = np.random.default_rng(4)                    # as in convex(): the same A
    Q = np.linalg.qr(rng.standard_normal((N, N)))[0]
    A = Q @ np.diag(np.geomspace(1.0, 100.0, N)) @ Q.T
    Qm = np.linalg.qr(np.random.default_rng(seed).standard_normal((N, N)))[0]
    M = Qm @ np.diag(np.geomspace(0.5, 2.0, N)) @ Qm.T
    return A, M, 2.0 * np.diag(np.diag(A))


def metric(M: Array) -> dict[str, Any]:
    return dict(metric_solve=lambda x, w: np.linalg.solve(M, w), metric_apply=lambda x, v: M @ v)


def test_P_equal_to_M_gives_the_run_without_P() -> None:
    """With P = M, the preconditioner's base and its fit are those of M, so
    the run is the run without P, every direction from P's base."""
    fun, grad, x0 = convex()
    _, M, _ = matrices()
    plain, s_plain = iterates(fun, grad, x0, **metric(M))
    with_P, s_P = iterates(fun, grad, x0, precond_solve=metric(M)["metric_solve"], **metric(M))
    assert len(plain) == len(with_P)
    assert max(rel(a, b) for a, b in zip(plain, with_P)) < 1e-9
    assert all(r.base == "P" for r in s_P.iterations) and s_P.precond_directions == len(s_P.iterations)


def test_a_misleading_P_falls_back_to_the_M_direction() -> None:
    """P^{-1} = -M^{-1} proposes ascent directions: every one fails, the M
    direction (its scaling refitted when used) is taken, and the run is the
    run without P."""
    fun, grad, x0 = convex()
    _, M, _ = matrices()
    plain, s_plain = iterates(fun, grad, x0, **metric(M))
    ascent, s_a = iterates(fun, grad, x0, precond_solve=lambda x, w: -np.linalg.solve(M, w), **metric(M))
    assert all(np.array_equal(a, b) for a, b in zip(plain, ascent)) and len(plain) == len(ascent)
    assert all(r.base == "M" for r in s_a.iterations) and s_a.precond_directions == 0


@pytest.mark.parametrize("mode", ["safeguarded", "trusted"])
def test_a_symmetric_positive_definite_P(mode: PrecondMode) -> None:
    """A rough approximation of the Hessian as P, with the Euclidean metric:
    both modes converge, and P's proposals are taken."""
    fun, grad, x0 = convex()
    A, _, P0 = matrices()
    x, s = lbfgs(fun, grad, x0, precond_solve=lambda x, w: w / np.diag(P0), precond_mode=mode, max_iter=500)
    s_plain: LbfgsSummary[Any, Any]
    _, s_plain = lbfgs(fun, grad, x0, max_iter=500)
    assert s.reason == "converged" and s.precond_directions >= len(s.iterations) - 2
    assert len(s.iterations) < len(s_plain.iterations)


def test_the_scale_fit_is_one_when_P_is_the_curvature() -> None:
    """On a quadratic with P = the Hessian, u = P^{-1} y = s, so gamma_P =
    <M u, s> / <M u, u> = 1 in any metric M; the first step -P^{-1} g is the
    Newton step."""
    rng = np.random.default_rng(8)
    A, M, _ = matrices()
    b = rng.standard_normal(N)
    x, s = lbfgs(lambda x: 0.5 * x @ A @ x - b @ x, lambda x: A @ x - b, np.zeros(N),
                 precond_solve=lambda x, w: np.linalg.solve(A, w), **metric(M))
    assert s.reason == "converged" and len(s.iterations) <= 2
    assert rel(x, np.linalg.solve(A, b)) < 1e-10
    if len(s.iterations) == 2:
        assert abs(s.iterations[1].gamma - 1.0) < 1e-10


def unconventional(kind: str, A: Array) -> Callable[[Any, Array], Array]:
    """P^{-1} that is not a symmetric positive definite operator."""
    rng = np.random.default_rng(9)
    D = np.diag(A)
    K = rng.standard_normal((N, N))
    K = 0.3 * (K - K.T) / np.linalg.norm(K - K.T, 2)
    noise = np.random.default_rng(10)
    if kind == "nonsymmetric":
        return lambda x, w: (np.eye(N) + K) @ (w / D)
    if kind == "nonlinear":
        return lambda x, w: w / D + 0.05 * np.linalg.norm(w / D) * np.tanh(w / np.max(np.abs(w)))
    if kind == "random":
        return lambda x, w: (w / D) * noise.uniform(0.5, 1.5, N)
    if kind == "rotated":                     # often no descent direction at all
        Q = np.linalg.qr(rng.standard_normal((N, N)))[0]
        return lambda x, w: Q @ (w / D)
    raise ValueError(kind)


@pytest.mark.parametrize("kind", ["nonsymmetric", "nonlinear", "random", "rotated"])
def test_unconventional_preconditioners_are_safe(kind: str) -> None:
    """Safeguarded by the angle test in M, any P gives a convergent run: P
    only chooses among directions, and the pairs are data from f."""
    fun, grad, x0 = convex()
    A, _, _ = matrices()
    x, s = lbfgs(fun, grad, x0, precond_solve=unconventional(kind, A), max_iter=2000)
    assert s.reason == "converged"
    taken = s.precond_directions / len(s.iterations)
    if kind == "rotated":
        assert taken < 1.0 and any(r.base == "M" for r in s.iterations)
    else:
        assert taken > 0.5


def test_the_preconditioner_convergence_norm() -> None:
    fun, grad, x0 = convex()
    A, M, P0 = matrices()
    def Pinv(x: Any, w: Array) -> Array:
        out: Array = w / np.diag(P0)
        return out
    x, s = lbfgs(fun, grad, x0, precond_solve=Pinv, convergence_norm="preconditioner", **metric(M))
    g = grad(x)
    assert s.reason == "converged"
    assert abs(s.gnorm - np.sqrt(g @ Pinv(x, g))) < 1e-12 * s.gnorm0
    assert s.gnorm <= 1e-8 * s.gnorm0


def test_operator_calls_with_a_preconditioner() -> None:
    """With a P that is always taken: P^{-1} once per iteration for the base
    and once per accepted step for the fit, M once per iteration (||d||_M)
    and once per accepted step (the fit), M^{-1} once per point (the norm)."""
    fun, grad, x0 = convex()
    _, M, P0 = matrices()
    calls = dict(solve=0, apply=0, precond=0)
    x, s = lbfgs(fun, grad, x0, metric_solve=counted(lambda x, w: np.linalg.solve(M, w), calls, "solve"),
                 metric_apply=counted(lambda x, v: M @ v, calls, "apply"),
                 precond_solve=counted(lambda x, w: w / np.diag(P0), calls, "precond"), max_iter=200)
    accepted = sum(r.accepted for r in s.iterations)
    assert s.precond_directions == len(s.iterations)
    assert (calls["solve"], calls["apply"], calls["precond"]) == \
        (s.metric_solve_calls, s.metric_apply_calls, s.precond_solve_calls)
    assert s.precond_solve_calls == len(s.iterations) + accepted
    assert s.metric_apply_calls == len(s.iterations) + accepted
    assert s.metric_solve_calls == 1 + accepted
    lines = format_lbfgs_summary(s)
    assert "preconditioner solves" in lines and " P " in format_lbfgs_row(s.iterations[3])


def test_fixed_preconditioner() -> None:
    """A constant linear P: the same run as the general path, with P^{-1}
    applied once per point (P^{-1} g) instead of twice per iteration; exact
    resumption keeps P^{-1} Y, and a state without it gets it recomputed."""
    fun, grad, x0 = convex()
    _, M, P0 = matrices()
    Pinv = lambda x, w: w / np.diag(P0)
    options: dict[str, Any] = dict(precond_solve=Pinv, **metric(M))
    general, sg = iterates(fun, grad, x0, **options)
    fixed, sf = iterates(fun, grad, x0, fixed_preconditioner=True, **options)
    assert [(r.base, r.trials, r.update) for r in sf.iterations[:60]] == \
        [(r.base, r.trials, r.update) for r in sg.iterations[:60]]
    assert max(rel(a, b) for a, b in zip(fixed[:60], general[:60])) < 1e-9
    assert sf.precond_solve_calls == 1 + sum(r.accepted for r in sf.iterations)
    both, sb = iterates(fun, grad, x0, fixed_preconditioner=True, fixed_metric=True, **options)
    assert max(rel(a, b) for a, b in zip(both[:60], general[:60])) < 1e-9
    assert sb.metric_solve_calls == 1 + sum(r.accepted for r in sb.iterations)
    firsts: list[dict[str, Any]] = [{"fixed_preconditioner": True}, {}]
    for first in firsts:
        saved: dict[str, Any] = {}

        def stop(x: Any, info: LbfgsInfo[Any, Any]) -> bool:
            if info.record.iteration == 14:
                saved.update(x=np.array(x), resume=info.resume, g=np.array(info.g))
                return True
            return False
        lbfgs(fun, grad, x0, callback=stop, **options, **first)
        rest, _ = iterates(fun, grad, saved["x"], resume=saved["resume"], g0=saved["g"],
                           fixed_preconditioner=True, **options)
        if first:
            assert all(np.array_equal(u, v) for u, v in zip(fixed[15:], rest))
        else:
            assert max(rel(u, v) for u, v in zip(fixed[15:45], rest[:30])) < 1e-9


def test_operators_at_can_give_the_preconditioner() -> None:
    fun, grad, x0 = convex()
    _, M, P0 = matrices()
    ops = PointOperators(metric_solve=lambda w: np.linalg.solve(M, w), metric_apply=lambda v: M @ v,
                         precond_solve=lambda w: w / np.diag(P0))
    a, sa = iterates(fun, grad, x0, operators_at=lambda x, d: ops)
    b, sb = iterates(fun, grad, x0, precond_solve=lambda x, w: w / np.diag(P0), **metric(M))
    assert all(np.array_equal(u, v) for u, v in zip(a, b)) and sa.precond_directions == sb.precond_directions


def test_preconditioner_arguments_are_checked() -> None:
    fun, grad, x0 = convex()
    p = sphere_problem()
    bad: list[tuple[Any, ...]] = [
        (fun, grad, x0, dict(convergence_norm="preconditioner")),
        (fun, grad, x0, dict(fixed_preconditioner=True)),
        (fun, grad, x0, dict(precond_solve=lambda x, w: w, precond_mode="always")),
        (fun, grad, x0, dict(precond_solve=lambda x, w: w, scaling_P="bb")),
        (fun, grad, x0, dict(precond_solve=lambda x, w: w, convergence_norm="P")),
        (fun, grad, x0, dict(operators_at=lambda x, d: None, precond_solve=lambda x, w: w)),
        (p["fun"], p["grad"], p["x0"], dict(p["options"], precond_solve=lambda x, w: w,
                                            fixed_preconditioner=True)),
    ]
    for f, g, x, options in bad:
        with pytest.raises(ValueError):
            lbfgs(f, g, x, max_iter=1, **options)


def test_strict_backend_with_unconventional_preconditioners() -> None:
    from strict_backend import strict_backend
    fun, grad, x0 = convex()
    A, M, _ = matrices()
    D = np.diag(A)
    K = np.random.default_rng(9).standard_normal((N, N))
    K = 0.3 * (K - K.T) / np.linalg.norm(K - K.T, 2)
    rotated = np.linalg.qr(np.random.default_rng(11).standard_normal((N, N)))[0]
    for Pinv in (lambda x, w: (np.eye(N) + K) @ (w / D), lambda x, w: rotated @ (w / D)):
        options: dict[str, Any] = dict(retract=lambda x, s: x + s, transport=identity_transport,
                                       precond_solve=Pinv, max_iter=60, memory=5, **metric(M))
        _, ref = lbfgs(fun, cast(Any, grad), x0, space=numpy_space, **options)
        _, run = lbfgs(strict_convex(A), cast(Any, grad), x0, space=array_space(strict_backend),
                       backend=strict_backend, **options)
        assert [(r.base, r.direction, r.trials) for r in run.iterations] == \
            [(r.base, r.direction, r.trials) for r in ref.iterations]
        # (the traceable objective rounds differently from the numpy one)
        assert max(abs(a.f - b.f) / max(abs(b.f), 1.0) for a, b in zip(run.iterations, ref.iterations)) < 1e-8


def strict_convex(A: Array) -> Callable[[Any], Any]:
    """convex()'s objective without float(): traceable."""
    rng = np.random.default_rng(4)
    rng.standard_normal((N, N))
    b = rng.standard_normal(N)
    return lambda x: 0.5 * x @ (A @ x) - b @ x + 0.25 * np.sum(x ** 4)
