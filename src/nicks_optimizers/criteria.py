"""Tests shared by inexact Newton methods: gradient convergence, the forcing
term for the inner solve, a rounding-safe ratio of actual to predicted
reduction, and one sample of the Gauss-Newton to full-Hessian switch
detector.
"""

import numpy as np


def gradient_converged(gnorm: float, gnorm0: float, gtol: float, rtol: float) -> bool:
    """Newton convergence test:  ||g|| <= max(gtol, rtol * ||g0||)."""
    return gnorm <= max(gtol, rtol * gnorm0)


def forcing_term(gnorm: float, gnorm0: float, forcing_max: float, power: float = 0.5) -> float:
    """Relative Krylov tolerance min(forcing_max, (||g|| / ||g0||)^power).

    Using the ratio makes it invariant to scaling f; it tends to zero as the
    gradient does, which gives superlinear local convergence (of order
    1 + power for an exact Hessian, quadratic with power = 1). The default
    power 0.5 is the square root.
    """
    if gnorm0 <= 0.0:
        return forcing_max
    ratio = gnorm / gnorm0
    return min(forcing_max, float(np.sqrt(ratio) if power == 0.5 else ratio ** power))


def reduction_ratio(f_old: float, f_trial: float, pred: float) -> float:
    """rho = actual reduction / predicted reduction, with a rounding safeguard.

    Near convergence both reductions fall to the level of rounding error in f,
    and the raw ratio becomes noise that can reject good steps and collapse
    the radius. The standard fix (see Conn, Gould & Toint, Trust-Region
    Methods) adds delta = 10 eps max(1, |f_old|) to both, so rho tends to 1
    in that regime and is unaffected otherwise.
    """
    if not (pred > 0.0 and np.isfinite(f_trial)):
        return -np.inf
    delta = 10.0 * float(np.finfo(float).eps) * max(1.0, abs(f_old))
    return ((f_old - f_trial) + delta) / (pred + delta)


SWITCH_ROUNDING = 100.0
"""A switch sample is valid when e_GN > SWITCH_ROUNDING * 10 eps max(1, |f|):
Gauss-Newton's own error must stand clear of the rounding in f
(dev/archive/gauss_newton_switch_plan.md, D4)."""


def switch_sample(f: float, change: float, gn_model: float, full_model: float,
                  tau: float) -> tuple[bool, bool, float, float]:
    """One sample of the Gauss-Newton to full-Hessian switch detector.

    f is f(x), change = f(retract(x, s)) - f(x) for the step s tried, and
    gn_model, full_model the two quadratic models' values at s (the change
    each predicts). Returns (valid, hit, e_gn, e_full) with

        e_gn = |change - gn_model|,  e_full = |change - full_model|,

    valid when both are finite and e_gn > SWITCH_ROUNDING * 10 eps max(1, |f|)
    (a Gauss-Newton error at the level of rounding in f leaves nothing to
    gain), and hit when valid and e_full < tau * e_gn. Near a minimizer with
    a nonzero residual e_gn is of order ||s||^2 and e_full of order ||s||^3
    (along a second-order retraction), so hits become persistent there.
    """
    e_gn, e_full = abs(change - gn_model), abs(change - full_model)
    delta = 10.0 * float(np.finfo(float).eps) * max(1.0, abs(f))
    valid = bool(np.isfinite(e_gn) and np.isfinite(e_full) and e_gn > SWITCH_ROUNDING * delta)
    return valid, valid and e_full < tau * e_gn, e_gn, e_full
