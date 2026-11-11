"""Tests shared by inexact Newton methods: gradient convergence, the forcing
term for the inner solve, and a rounding-safe ratio of actual to predicted
reduction.
"""

import numpy as np


def gradient_converged(gnorm: float, gnorm0: float, gtol: float, rtol: float) -> bool:
    """Newton convergence test:  ||g|| <= max(gtol, rtol * ||g0||)."""
    return gnorm <= max(gtol, rtol * gnorm0)


def forcing_term(gnorm: float, gnorm0: float, forcing_max: float) -> float:
    """Relative Krylov tolerance min(forcing_max, sqrt(||g|| / ||g0||)).

    Using the ratio makes it invariant to scaling f; it tends to zero as the
    gradient does, which gives superlinear local convergence.
    """
    if gnorm0 <= 0.0:
        return forcing_max
    return min(forcing_max, float(np.sqrt(gnorm / gnorm0)))


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
