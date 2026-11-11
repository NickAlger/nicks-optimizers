"""hessian="switch": Gauss-Newton first, the full Hessian once the full model
predicts the steps clearly better (dev/gauss_newton_switch_plan.md).

The pieces: with track (the problem with the full Hessian) the finish
pieces report the Gauss-Newton and the full model values at the step tried,
by one product with the full Hessian; without it they report nan and apply
nothing more.

The drivers: the detector (criteria.switch_sample, newton_switch.SwitchTracker);
before the switch a switch run is the Gauss-Newton run, after it the
full-Hessian run resumed at the switch point, bit for bit; resuming; the work
by Hessian; the test problems; operators_at; the keyword checks; the display;
the forcing powers of the two phases.
"""

import dataclasses
from typing import Any, Callable, Optional

import numpy as np
import pytest

from nicks_optimizers import (
    PointOperators,
    ResumeState,
    array_space,
    euclidean_metric_apply,
    euclidean_metric_solve,
    format_ls_summary,
    format_summary,
    line_search_newton,
    merge_ls_summaries,
    merge_summaries,
    switch_sample,
    trust_region_newton,
)
from nicks_optimizers.backend import numpy_backend
from nicks_optimizers.newton_core import CoreProblem, OuterState
from nicks_optimizers.newton_driver import make_problem, step_solve
from nicks_optimizers.newton_line_search_core import LineSearchOptions, line_search_finish
from nicks_optimizers.newton_switch import SwitchSettings, SwitchTracker
from nicks_optimizers.trust_region_core import TrustRegionOptions, step_finish
from problems import LeastSquares, linear_least_squares, regularized_rosenbrock
from strict_backend import strict_backend


class Counter:
    """Counts the calls of a Hessian-vector product."""

    def __init__(self, hv: Callable[[Any, Any], Any]) -> None:
        self.hv, self.calls = hv, 0

    def __call__(self, x: Any, v: Any) -> Any:
        self.calls += 1
        return self.hv(x, v)


def problems(p: LeastSquares, trust_region: bool, full: Counter, gn: Counter
             ) -> tuple[CoreProblem, CoreProblem]:
    """The two phases' problems: Gauss-Newton (before) and full (after)."""
    base = make_problem(fun=p.fun, grad=p.grad, hessvec=full, space=None, retract=None,
                        metric_solve=None, metric_apply=None, precond_solve=None,
                        operators_at=None, point_data=None, project_covector=None,
                        saddle_free=False, krylov="auto", convergence_norm="metric",
                        forcing_max=0.5, stall_tol=1e-3, max_krylov=50,
                        trust_region=trust_region)
    base = dataclasses.replace(base, gn_hessvec=gn)
    return dataclasses.replace(base, model="gauss_newton"), base


def dense(hv: Callable[[Any, Any], Any], x: Any) -> Any:
    n = x.shape[0]
    return np.column_stack([hv(x, np.eye(n)[i]) for i in range(n)])


def one_iteration(trust_region: bool, track: bool, x0: Optional[Any] = None
                  ) -> tuple[Any, Any, Counter, Counter, OuterState]:
    p = regularized_rosenbrock()
    full, gn = Counter(p.hessvec), Counter(p.gn_hessvec)
    before, after = problems(p, trust_region, full, gn)
    bk = numpy_backend
    pieces = step_solve(before, bk, None)
    state = pieces.start(p.x0 if x0 is None else x0, None)
    gnorm0 = bk.xp.asarray(float(state.gnorm))
    radius = bk.xp.asarray(1.0 if trust_region else np.inf)
    carry, _ = pieces.solve(state, radius, gnorm0)
    gn_calls_in_solve = gn.calls
    kw = {"track": after} if track else {}
    raw: Any
    if trust_region:
        _, raw = step_finish(before, TrustRegionOptions(eta=1e-4, radius_max=1e4), bk, state,
                             carry, radius, gnorm0, **kw)
    else:
        _, raw = line_search_finish(before, LineSearchOptions(c1=1e-4, max_backtracks=30,
                                                              step_max=None), bk, state,
                                    carry, gnorm0, **kw)
    assert gn.calls == gn_calls_in_solve          # the finish piece applies no GN product
    return p, raw, full, gn, state


@pytest.mark.parametrize("trust_region", [True, False], ids=["trust region", "line search"])
@pytest.mark.parametrize("start", ["far", "near"])
def test_tracked_model_values(trust_region: bool, start: str) -> None:
    x0 = None if start == "far" else np.ones(20) + 0.05
    p, raw, full, gn, state = one_iteration(trust_region, True, x0)
    assert gn.calls >= 1                          # the solve used the Gauss-Newton Hessian
    assert full.calls == 1                        # exactly one product with the full Hessian
    x, g, s = state.x, state.g, raw.step
    H, G = dense(p.hessvec, x), dense(p.gn_hessvec, x)
    assert float(raw.full_model) == pytest.approx(g @ s + 0.5 * s @ H @ s, rel=1e-12)
    assert float(raw.gn_model) == pytest.approx(g @ s + 0.5 * s @ G @ s, rel=1e-12)
    assert float(raw.gn_model) == pytest.approx(-float(raw.predicted), rel=1e-12)
    change = p.fun(x + s) - p.fun(x)
    assert float(raw.actual) == pytest.approx(-change, rel=1e-12)


@pytest.mark.parametrize("trust_region", [True, False], ids=["trust region", "line search"])
def test_no_tracking_without_track(trust_region: bool) -> None:
    _, raw, full, gn, _ = one_iteration(trust_region, False)
    assert full.calls == 0 and gn.calls >= 1
    assert np.isnan(float(raw.gn_model)) and np.isnan(float(raw.full_model))


def test_gauss_newton_model_needs_gn_hessvec() -> None:
    p = regularized_rosenbrock()
    before, _ = problems(p, True, Counter(p.hessvec), Counter(p.gn_hessvec))
    without = dataclasses.replace(before, gn_hessvec=None)
    with pytest.raises(ValueError, match="gn_hessvec"):
        step_solve(without, numpy_backend, None).start(p.x0, None)


# --- the detector ---------------------------------------------------------------------------


def test_switch_sample() -> None:
    # change -1: the Gauss-Newton model is off by 0.5, the full one by 0.01: a hit
    valid, hit, e_gn, e_full = switch_sample(1.0, -1.0, -0.5, -0.99, 0.25)
    assert valid and hit and e_gn == 0.5 and e_full == pytest.approx(0.01)
    valid, hit, _, _ = switch_sample(1.0, -1.0, -0.5, -0.8, 0.25)       # 0.2 >= 0.25 * 0.5
    assert valid and not hit
    valid, hit, _, _ = switch_sample(1.0, -1.0, -0.5, -0.8, 0.5)        # a larger tau
    assert valid and hit
    eps = float(np.finfo(float).eps)
    valid, hit, _, _ = switch_sample(1.0, -1e-3, -1e-3 + 500 * eps, -1e-3, 0.25)  # rounding
    assert not valid and not hit
    valid, hit, _, _ = switch_sample(1.0, -1e-3, -1e-3 + 5000 * eps, -1e-3, 0.25)
    assert valid and hit
    valid, _, _, _ = switch_sample(1e6, -1.0, -1.0 + 1e-7, -1.0, 0.25)  # rounding scales with |f|
    assert not valid
    assert switch_sample(1.0, -1.0, float("nan"), -1.0, 0.25)[:2] == (False, False)
    assert switch_sample(1.0, -1.0, -0.5, float("nan"), 0.25)[:2] == (False, False)


HIT = (1.0, 1.0, -0.5, -0.99)          # (f, actual, gn_model, full_model): a hit
MISS = (1.0, 1.0, -0.5, -0.5)
INVALID = (1.0, 1.0, -1.0, -1.0)        # e_gn = 0


def settings(k: int = 2, tau: float = 0.25) -> SwitchSettings:
    return SwitchSettings(switch=True, tau=tau, k=k, pre_power=0.5, post_power=1.0)


def script(tracker: SwitchTracker, steps: list[tuple[tuple[float, float, float, float], bool]]
           ) -> list[dict[str, Any]]:
    return [tracker.observe(i, *sample, accepted) for i, (sample, accepted) in enumerate(steps)]


def test_tracker_fires_after_k_consecutive_hits() -> None:
    t = SwitchTracker(settings())
    out = script(t, [(HIT, True), (MISS, True), (HIT, True), (HIT, True), (MISS, True)])
    assert [o["model"] for o in out] == ["gauss_newton"] * 4 + ["full"]
    assert [o.get("switch_hits") for o in out] == [1, 0, 1, 2, None]
    assert [o.get("switch_fired") for o in out] == [False, False, False, True, None]
    assert t.switched and t.switch_iteration == 4 and t.hits == 0 and not t.tracking


@pytest.mark.parametrize("breaker", ["miss", "invalid", "rejected"])
def test_tracker_resets(breaker: str) -> None:
    """A miss, an invalid sample (D3) or a rejected step (D5) resets the count."""
    step = {"miss": (MISS, True), "invalid": (INVALID, True), "rejected": (HIT, False)}[breaker]
    t = SwitchTracker(settings(k=3))
    out = script(t, [(HIT, True), (HIT, True), step, (HIT, True), (HIT, True)])
    assert [o["switch_hits"] for o in out] == [1, 2, 0, 1, 2] and not t.switched


def test_tracker_without_switch_and_resumed() -> None:
    off = SwitchSettings(switch=False, tau=0.25, k=2, pre_power=0.5, post_power=0.5)
    assert SwitchTracker(off).observe(0, *HIT, True) == {}
    t = SwitchTracker(settings(), switched=False, hits=1)               # resumed with one hit
    assert t.observe(7, *HIT, True)["switch_fired"] and t.switch_iteration == 8
    t = SwitchTracker(settings(), switched=True)
    assert t.observe(7, *HIT, True) == {"model": "full"} and t.switch_iteration is None


# --- the drivers ----------------------------------------------------------------------------

METHODS = ("trust region", "line search")
SWITCH_FIELDS = {"model": None, "error_gn": None, "error_full": None, "switch_hits": None,
                 "switch_fired": None}


def optimizer(method: str) -> Any:
    return trust_region_newton if method == "trust region" else line_search_newton


def merge(method: str) -> Any:
    return merge_summaries if method == "trust region" else merge_ls_summaries


def plain(records: Any) -> list[str]:
    """Records without the switch fields and times, as text (repr gives
    floats exactly, and nan equals nan)."""
    return [repr(dataclasses.replace(r, elapsed=0.0, **SWITCH_FIELDS)) for r in records]


def timeless(records: Any) -> list[str]:
    return [repr(dataclasses.replace(r, elapsed=0.0)) for r in records]


def switch_run(method: str, p: LeastSquares, x0: Optional[Any] = None, **options: Any
               ) -> tuple[Any, Any]:
    run: tuple[Any, Any] = optimizer(method)(
        p.fun, p.grad, p.hessvec, p.x0 if x0 is None else x0, hessian="switch",
        gn_hessvec=p.gn_hessvec, rtol=1e-12, **options)
    return run


def solver_options(krylov: str) -> dict[str, Any]:
    """CG, Lanczos with P = M, and Lanczos and flexible with a separate
    (constant) preconditioner."""
    if krylov in ("cg", "lanczos"):
        return {"krylov": krylov}
    return {"krylov": krylov.split()[0], "metric_solve": euclidean_metric_solve,
            "metric_apply": euclidean_metric_apply,
            "precond_solve": lambda x, w: w / (1.0 + 0.5 * np.arange(w.shape[0]) / w.shape[0])}


KRYLOV = ["cg", "lanczos", "lanczos P", "flexible P"]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("krylov", KRYLOV)
def test_before_the_switch_it_is_the_gauss_newton_run(method: str, krylov: str) -> None:
    p = regularized_rosenbrock()
    opts = solver_options(krylov)
    x_at: dict[int, Any] = {}

    def keep(x: Any, info: Any) -> None:
        x_at[info.record.iteration] = x

    _, s = switch_run(method, p, callback=keep, **opts)
    j = s.switch_iteration
    assert j is not None and j >= 2
    x, gn = optimizer(method)(p.fun, p.grad, p.gn_hessvec, p.x0, rtol=1e-12, max_iter=j, **opts)
    assert plain(s.iterations[:j]) == plain(gn.iterations)
    assert all(r.model == "gauss_newton" for r in s.iterations[:j])
    assert x.tobytes() == x_at[j - 1].tobytes()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("krylov", KRYLOV)
def test_after_the_switch_it_is_the_full_run_resumed(method: str, krylov: str) -> None:
    p = regularized_rosenbrock()
    opts = solver_options(krylov)
    saved: dict[str, Any] = {}

    def keep(x: Any, info: Any) -> None:
        if info.record.switch_fired:
            saved.update(x=x, g=info.g, resume=info.resume)

    x_switch, s = switch_run(method, p, callback=keep, **opts)
    j = s.switch_iteration
    assert j is not None and saved["resume"].switched and saved["resume"].next_iteration == j
    resume = dataclasses.replace(saved["resume"], switched=False, switch_hits=0)
    x_full, full = optimizer(method)(p.fun, p.grad, p.hessvec, saved["x"], rtol=1e-12,
                                     resume=resume, g0=saved["g"], **opts)
    assert plain(s.iterations[j:]) == plain(full.iterations)
    assert all(r.model == "full" for r in s.iterations[j:])
    assert x_switch.tobytes() == x_full.tobytes()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("where", ["one hit", "at the switch", "after"])
def test_resumed_switch_run_is_the_same_run(method: str, where: str) -> None:
    p = regularized_rosenbrock()
    x_ref, ref = switch_run(method, p)
    j = ref.switch_iteration
    assert j is not None
    if where == "one hit":
        stop = next(r.iteration for r in ref.iterations if r.switch_hits == 1 and not r.switch_fired)
    else:
        stop = j - 1 if where == "at the switch" else j + 1
    saved: dict[str, Any] = {}

    def interrupt(x: Any, info: Any) -> bool:
        saved.update(g=info.g, resume=info.resume)
        return bool(info.record.iteration == stop)

    x1, first = switch_run(method, p, callback=interrupt)
    assert first.reason == "callback"
    assert saved["resume"].switch_hits == (1 if where == "one hit" else 0)
    assert saved["resume"].switched == (where != "one hit")
    x2, second = switch_run(method, p, x1, resume=saved["resume"], g0=saved["g"])
    both = merge(method)(first, second)
    assert timeless(both.iterations) == timeless(ref.iterations)
    assert both.switch_iteration == j and both.reason == ref.reason
    assert x2.tobytes() == x_ref.tobytes()


@pytest.mark.parametrize("method", METHODS)
def test_switched_state_resumed_without_switch_raises(method: str) -> None:
    p = regularized_rosenbrock()
    _, s = switch_run(method, p)
    assert s.resume.switched
    with pytest.raises(ValueError, match="hessian='switch'"):
        optimizer(method)(p.fun, p.grad, p.hessvec, p.x0, resume=s.resume)


@pytest.mark.parametrize("method", METHODS)
def test_work_by_hessian(method: str) -> None:
    """One product with hessvec per iteration before the switch; the solves
    use gn_hessvec before it and hessvec after; the summary counts both."""
    p = regularized_rosenbrock()
    full, gn = Counter(p.hessvec), Counter(p.gn_hessvec)
    _, s = optimizer(method)(p.fun, p.grad, full, p.x0, hessian="switch", gn_hessvec=gn,
                             rtol=1e-12)
    before = [r for r in s.iterations if r.model == "gauss_newton"]
    after = [r for r in s.iterations if r.model == "full"]
    assert before and after
    assert gn.calls == s.gn_hessvec_calls == sum(r.subproblem.hessvec_calls for r in before)
    assert full.calls == s.hessvec_calls == len(before) + sum(r.subproblem.hessvec_calls
                                                              for r in after)
    full2, gn2 = Counter(p.hessvec), Counter(p.gn_hessvec)
    _, plain_run = optimizer(method)(p.fun, p.grad, full2, p.x0, rtol=1e-12)
    assert gn2.calls == plain_run.gn_hessvec_calls == 0
    assert full2.calls == plain_run.hessvec_calls
    assert plain_run.hessian == "full" and plain_run.switch_iteration is None


@pytest.mark.parametrize("method", METHODS)
def test_nonzero_residual(method: str) -> None:
    """The study's problem: the switch fires, the run reaches the same
    minimizer as Gauss-Newton, in no more iterations, and converges fast
    once switched."""
    p = regularized_rosenbrock()
    _, s = switch_run(method, p)
    _, gn = optimizer(method)(p.fun, p.grad, p.gn_hessvec, p.x0, rtol=1e-12)
    assert s.reason == gn.reason == "converged" and s.switch_iteration is not None
    assert s.f == pytest.approx(gn.f, rel=1e-12) and s.f > 1e-3
    assert len(s.iterations) <= len(gn.iterations)
    errors = [r for r in s.iterations if r.model == "gauss_newton"][-2:]
    assert all(r.error_full < 0.25 * r.error_gn for r in errors)


@pytest.mark.parametrize("method", METHODS)
def test_zero_residual(method: str) -> None:
    p = regularized_rosenbrock(cscale=0.0)
    _, s = switch_run(method, p)
    assert s.reason == "converged" and s.f < 1e-20


@pytest.mark.parametrize("method", METHODS)
def test_linear_least_squares_never_switches(method: str) -> None:
    """S = 0: both models are exact, the Gauss-Newton error is rounding, and
    no sample is valid."""
    p = linear_least_squares()
    _, s = switch_run(method, p)
    assert s.reason == "converged" and s.switch_iteration is None
    assert all(r.model == "gauss_newton" and not r.switch_hits for r in s.iterations)


@pytest.mark.parametrize("method", METHODS)
def test_operators_at(method: str) -> None:
    """gn_hessvec in the PointOperators of operators_at: the run of the keywords."""
    p = regularized_rosenbrock()
    _, by_keywords = switch_run(method, p)

    def operators_at(x: Any, data: Any) -> Any:
        return PointOperators(hessvec=lambda v: p.hessvec(x, v),
                              gn_hessvec=lambda v: p.gn_hessvec(x, v))

    _, by_operators = optimizer(method)(p.fun, p.grad, None, p.x0, operators_at=operators_at,
                                        hessian="switch", rtol=1e-12)
    assert timeless(by_operators.iterations) == timeless(by_keywords.iterations)

    def without_gn(x: Any, data: Any) -> Any:
        return PointOperators(hessvec=lambda v: p.hessvec(x, v))

    with pytest.raises(ValueError, match="gn_hessvec"):
        optimizer(method)(p.fun, p.grad, None, p.x0, operators_at=without_gn, hessian="switch")


@pytest.mark.parametrize("method", METHODS)
def test_keyword_checks(method: str) -> None:
    p = regularized_rosenbrock()
    opt = optimizer(method)

    def run(**kw: Any) -> Any:
        return opt(p.fun, p.grad, p.hessvec, p.x0, **kw)

    cases: list[tuple[dict[str, Any], str]] = [
        ({"hessian": "newton"}, "hessian must be"),
        ({"hessian": "switch"}, "needs gn_hessvec"),
        ({"gn_hessvec": p.gn_hessvec}, "gn_hessvec only with hessian='switch'"),
        ({"switch_tau": 0.5}, "switch_tau only with"),
        ({"switch_k": 3}, "switch_k only with"),
        ({"pre_switch_forcing_power": 0.5}, "pre_switch_forcing_power only with"),
        ({"hessian": "switch", "gn_hessvec": p.gn_hessvec, "switch_tau": 1.0}, "switch_tau must"),
        ({"hessian": "switch", "gn_hessvec": p.gn_hessvec, "switch_tau": 0.0}, "switch_tau must"),
        ({"hessian": "switch", "gn_hessvec": p.gn_hessvec, "switch_k": 0}, "switch_k must"),
        ({"hessian": "switch", "gn_hessvec": p.gn_hessvec, "switch_k": 1.5}, "switch_k must"),
        ({"hessian": "switch", "gn_hessvec": p.gn_hessvec, "pre_switch_forcing_power": -1.0},
         "pre_switch_forcing_power must"),
    ]
    for kw, message in cases:
        with pytest.raises(ValueError, match=message):
            run(**kw)
    with pytest.raises(ValueError, match="not as a keyword"):
        opt(p.fun, p.grad, None, p.x0, operators_at=lambda x, d: None, hessian="switch",
            gn_hessvec=p.gn_hessvec)


@pytest.mark.parametrize("method", METHODS)
def test_display(method: str) -> None:
    p = regularized_rosenbrock()
    lines: list[str] = []
    _, s = switch_run(method, p, display=1, print_fn=lines.append)
    j = s.switch_iteration
    assert j is not None
    rows = [line for line in lines if line.strip()[:1].isdigit()]
    assert "eN/eGN" in lines[0]
    assert all(row.rstrip().endswith(" *") == (i == j - 1) for i, row in enumerate(rows[:j]))
    assert all("  GN  " in row for row in rows[:j]) and all("   N" in row for row in rows[j:])
    summary = format_summary(s) if method == "trust region" else format_ls_summary(s)
    assert f"switched to the full Hessian at iteration {j}" in summary
    assert f"{s.gn_hessvec_calls} Gauss-Newton and {s.hessvec_calls} full" in summary
    plain_lines: list[str] = []
    _, s0 = optimizer(method)(p.fun, p.grad, p.hessvec, p.x0, display=1, print_fn=plain_lines.append)
    assert "eN/eGN" not in plain_lines[0] and not any("switch" in line for line in plain_lines)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("pre,power,expected", [
    (None, None, (0.5, 0.5)), (0.75, None, (0.75, 0.5)), (None, 0.75, (0.75, 0.75)),
    (0.5, 1.0, (0.5, 1.0)),
    (0.25, 2.0, (0.25, 2.0))])
def test_forcing_powers_of_the_phases(method: str, pre: Optional[float], power: Optional[float],
                                      expected: tuple[float, float]) -> None:
    """Each phase's forcing follows its power. (A tighter Gauss-Newton phase
    may converge before the detector fires; the default case switches.)"""
    p = regularized_rosenbrock()
    _, s = switch_run(method, p, pre_switch_forcing_power=pre, forcing_power=power)
    assert s.reason == "converged"
    if pre is None and power is None:
        assert s.switch_iteration is not None
    for r in s.iterations:
        q = expected[0] if r.model == "gauss_newton" else expected[1]
        assert r.forcing == pytest.approx(min(0.5, r.gnorm_ratio ** q), rel=1e-13)


# --- the strict backend ---------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("krylov", KRYLOV)
def test_strict_backend(method: str, krylov: str) -> None:
    """The code path of a compiling backend, without a framework: the switch
    run on the strict backend (tiny capacities) is the numpy run."""
    p = regularized_rosenbrock()
    opts = solver_options(krylov)

    def spaced(backend: Any) -> dict[str, Any]:
        """Lanczos and flexible keep Krylov blocks of fixed capacity, as
        compiling backends need: array_space (on both sides, so that the
        comparison sees the backend only)."""
        if krylov == "cg":
            return opts
        return {"metric_solve": euclidean_metric_solve, **opts, "space": array_space(backend),
                "retract": lambda x, s: x + s}

    _, reference = switch_run(method, p, krylov_capacity=2, **spaced(numpy_backend))
    _, strict = switch_run(method, p, backend=strict_backend, krylov_capacity=2,
                           **spaced(strict_backend))
    assert reference.switch_iteration is not None
    assert strict.switch_iteration == reference.switch_iteration
    assert [(r.model, r.accepted, r.subproblem.krylov_steps, r.switch_hits)
            for r in strict.iterations] == \
        [(r.model, r.accepted, r.subproblem.krylov_steps, r.switch_hits)
         for r in reference.iterations]
    for a, b in zip(strict.iterations, reference.iterations):     # as tests/test_newton_line_search
        assert abs(a.f - b.f) <= 1e-10 * max(1.0, abs(b.f))
    assert (strict.hessvec_calls, strict.gn_hessvec_calls) == \
        (reference.hessvec_calls, reference.gn_hessvec_calls)
