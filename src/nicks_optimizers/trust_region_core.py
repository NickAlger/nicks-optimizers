"""The trust-region part of an iteration of trust_region_newton, as a pure
piece a backend can compile (see backend and dev/jit_backends_plan.md).

The step is computed by the shared pieces of newton_core (step_start,
step_grow, step_continue); then

  step_finish(problem, options, backend, state, carry, radius, gnorm0)
                 -> (OuterState, RawIteration)
                 the step, the trial point, the reduction ratio, the new
                 radius, and, when the step is accepted, the gradient, the
                 point data and the gradient norms at the new point (under
                 backend.cond)

  TrustRegionOptions  eta and radius_max, the options of the update
  RawIteration   what one iteration reports, as arrays; the driver turns it
                 into an IterationRecord

Nothing here branches in Python on computed values, and nothing imports
numpy. The stopping tests, records, display, callback and resuming stay in
the driver (trust_region).
"""

from dataclasses import dataclass
from typing import Any

from .backend import Backend, backend_state, run_cond
from .newton_core import (
    CoreProblem,
    OuterState,
    PointOperators,
    canonical,
    forcing_term,
    init,
    state_at,
    step_continue,
    step_grow,
    step_start,
    subproblem_at,
)

__all__ = ["CoreProblem", "OuterState", "PointOperators", "RawIteration", "TrustRegionOptions",
           "forcing_term", "init", "reduction_ratio", "step_continue", "step_finish", "step_grow",
           "step_start", "update_radius"]


@dataclass(frozen=True, kw_only=True)
class TrustRegionOptions:
    """The options of the trust-region update: the acceptance threshold eta
    for rho, and the largest radius."""
    eta: float
    radius_max: float


@backend_state
@dataclass(frozen=True, kw_only=True)
class RawIteration:
    """What one iteration reports (arrays): the forcing term, the predicted
    and actual reductions, rho, acceptance, the new radius, the subproblem's
    statistics (RawSubproblem; SubproblemStats from the solvers not yet
    converted) and the step tried."""
    forcing: Any
    predicted: Any
    actual: Any
    rho: Any
    accepted: Any
    new_radius: Any
    subproblem: Any
    step: Any


# --- the tests of criteria.py and the radius update, in where form -------------------------


def reduction_ratio(xp: Any, f_old: Any, f_trial: Any, predicted: Any) -> Any:
    """criteria.reduction_ratio: actual over predicted reduction, both
    shifted by 10 eps max(1, |f_old|); -inf when the predicted reduction is
    not positive or f_trial is not finite."""
    valid = (predicted > 0.0) & xp.isfinite(f_trial)
    eps = xp.finfo(xp.asarray(f_old).dtype).eps
    delta = 10.0 * eps * xp.maximum(1.0, xp.abs(f_old))
    rho = ((f_old - f_trial) + delta) / xp.where(valid, predicted + delta, 1.0)
    return xp.where(valid, rho, -xp.inf)


def update_radius(xp: Any, radius: Any, rho: Any, step_norm: Any, radius_max: Any) -> Any:
    """The standard radius update, measuring the step in the M-norm: shrink
    to a quarter of the step when rho < 1/4, double (up to radius_max) when
    rho > 3/4 and the step reached the boundary."""
    expand = (rho > 0.75) & (step_norm >= 0.99 * radius)
    return xp.where(rho < 0.25, 0.25 * step_norm,
                    xp.where(expand, xp.minimum(2.0 * radius, radius_max), radius))


# --- the piece ---------------------------------------------------------------------------


def step_finish(problem: CoreProblem, options: TrustRegionOptions, backend: Backend,
                state: OuterState, carry: Any, radius: Any, gnorm0: Any
                ) -> tuple[OuterState, RawIteration]:
    """The rest of the iteration once the solve has stopped: the step, the
    trial point, rho, the new radius, and the state at the next point (the
    trial point if accepted, else the same)."""
    xp = backend.xp
    solver, ops = subproblem_at(problem, state)
    s, sub = solver.result(backend, problem.space, ops, carry)
    x_trial = problem.retract(state.x, s)
    f_trial = problem.fun(x_trial)
    predicted = -sub.model_value
    rho = reduction_ratio(xp, state.f, f_trial, predicted)
    accepted = rho > options.eta
    new_radius = update_radius(xp, radius, rho, sub.step_norm, options.radius_max)

    def move(_: Any) -> OuterState:
        g = canonical(problem, x_trial, problem.grad(x_trial))
        return state_at(problem, backend, x_trial, f_trial, g)

    def stay(_: Any) -> OuterState:
        return state

    new_state = run_cond(backend, accepted, move, stay, None)
    forcing = forcing_term(xp, state.gnorm, gnorm0, problem.forcing_max)
    return new_state, RawIteration(forcing=forcing, predicted=predicted, actual=state.f - f_trial,
                                   rho=rho, accepted=accepted, new_radius=new_radius,
                                   subproblem=sub, step=s)
