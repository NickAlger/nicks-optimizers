"""hessian="switch": Gauss-Newton first, the full Hessian once the full model
predicts the steps clearly better (dev/gauss_newton_switch_plan.md).

Phase 2: the pieces. With track (the problem with the full Hessian) the
finish pieces report the Gauss-Newton and the full model values at the step
tried, by one product with the full Hessian; without it they report nan and
apply nothing more.
"""

import dataclasses
from typing import Any, Callable, Optional

import numpy as np
import pytest

from nicks_optimizers.backend import numpy_backend
from nicks_optimizers.newton_core import CoreProblem, OuterState
from nicks_optimizers.newton_driver import make_problem, step_solve
from nicks_optimizers.newton_line_search_core import LineSearchOptions, line_search_finish
from nicks_optimizers.trust_region_core import TrustRegionOptions, step_finish
from problems import LeastSquares, regularized_rosenbrock


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
