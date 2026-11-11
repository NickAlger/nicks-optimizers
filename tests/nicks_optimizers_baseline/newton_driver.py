"""What the drivers of the inexact Newton optimizers share (Python, around
the compiled pieces of newton_core): the checks of the arguments, the
CoreProblem with its defaults, the compiled pieces of the step solve with
the growth of its Krylov basis, and the consistency check of a resumed run.
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Literal, Optional, cast

import numpy as np

from .backend import Backend, compile_function
from .krylov_solvers import DEFAULT_CAPACITY, FULL, as_integer
from .newton_core import CoreProblem, OuterState, PointOperators, init, step_continue, step_grow, step_start
from .spaces import VectorSpace, numpy_space

Krylov = Literal["auto", "cg", "lanczos", "flexible"]
ConvergenceNorm = Literal["metric", "preconditioner"]


def check_arguments(*, hessvec: Optional[Callable[..., Any]], space: Optional[VectorSpace[Any, Any, Any, Any]],
                    retract: Optional[Callable[..., Any]], metric_solve: Optional[Callable[..., Any]],
                    metric_apply: Optional[Callable[..., Any]], precond_solve: Optional[Callable[..., Any]],
                    operators_at: Optional[Callable[..., Any]], point_data: Optional[Callable[..., Any]],
                    saddle_free: bool, krylov: str, convergence_norm: str,
                    krylov_capacity: Optional[int]) -> None:
    """Raise ValueError for an inconsistent combination of arguments."""
    if operators_at is not None:
        if any(f is not None for f in (hessvec, metric_solve, metric_apply, precond_solve)):
            raise ValueError("operators_at gives all the operators at a point: pass hessvec=None "
                             "and no metric_solve, metric_apply or precond_solve")
    elif hessvec is None:
        raise ValueError("pass hessvec, or operators_at")
    elif point_data is not None:
        raise ValueError("point_data is passed to operators_at: give operators_at too")
    elif metric_apply is not None and metric_solve is None:
        raise ValueError("metric_apply needs metric_solve (its inverse)")
    numpy_default = space is None
    if not numpy_default and retract is None:
        raise ValueError("with a space, pass retract (flat_retraction(space) gives x + s)")
    if not numpy_default and operators_at is None and metric_solve is None:
        raise ValueError("with a space, pass metric_solve (euclidean_metric_solve gives M = I)")
    if krylov not in ("auto", "cg", "lanczos", "flexible"):
        raise ValueError(f"krylov must be 'auto', 'cg', 'lanczos' or 'flexible', not {krylov!r}")
    if krylov == "flexible" and operators_at is None and (precond_solve is None or metric_apply is None):
        raise ValueError("krylov='flexible' needs a separate precond_solve, and metric_apply")
    if krylov == "cg" and (saddle_free or precond_solve is not None):
        raise ValueError("krylov='cg' needs P = M (no precond_solve) and saddle_free=False")
    if convergence_norm not in ("metric", "preconditioner"):
        raise ValueError(f"convergence_norm must be 'metric' or 'preconditioner', "
                         f"not {convergence_norm!r}")
    if krylov_capacity is not None and krylov_capacity < 1:
        raise ValueError(f"krylov_capacity must be positive, not {krylov_capacity}")


def make_problem(*, fun: Callable[[Any], Any], grad: Callable[[Any], Any],
                 hessvec: Optional[Callable[[Any, Any], Any]],
                 space: Optional[VectorSpace[Any, Any, Any, Any]], retract: Optional[Callable[[Any, Any], Any]],
                 metric_solve: Optional[Callable[[Any, Any], Any]],
                 metric_apply: Optional[Callable[[Any, Any], Any]],
                 precond_solve: Optional[Callable[[Any, Any], Any]],
                 operators_at: Optional[Callable[[Any, Any], PointOperators[Any, Any]]],
                 point_data: Optional[Callable[[Any], Any]],
                 project_covector: Optional[Callable[[Any, Any], Any]], saddle_free: bool,
                 krylov: Krylov, convergence_norm: ConvergenceNorm, forcing_max: float,
                 stall_tol: float, max_krylov: int, trust_region: bool = True,
                 curvature_floor: float = 1e-6) -> CoreProblem:
    """The CoreProblem, with the numpy defaults (numpy_space, x + s) filled in."""
    return CoreProblem(
        fun=fun, grad=grad, hessvec=hessvec,
        space=space if space is not None else cast(VectorSpace[Any, Any, Any, Any], numpy_space),
        numpy_default=space is None,
        retract=retract if retract is not None else (lambda x, s: x + s),
        metric_solve=metric_solve, metric_apply=metric_apply, precond_solve=precond_solve,
        operators_at=operators_at, point_data=point_data, project_covector=project_covector,
        saddle_free=saddle_free, krylov=krylov, convergence_norm=convergence_norm,
        forcing_max=forcing_max, stall_tol=stall_tol, max_krylov=max_krylov,
        trust_region=trust_region, curvature_floor=curvature_floor)


@dataclass(frozen=True)
class StepSolve:
    """The compiled pieces of an iteration's start and step solve.

      start(x0, g0) -> OuterState
      solve(state, radius, gnorm0) -> (carry, growths)
                     the Krylov solve, started with room for the initial
                     capacity and grown (doubled, up to max_krylov) and
                     continued while its basis fills up

    The step_start and step_grow pieces are compiled once per capacity and
    kept for the run."""
    start: Callable[..., OuterState]
    solve: Callable[[OuterState, Any, Any], tuple[Any, int]]


def step_solve(problem: CoreProblem, backend: Backend, krylov_capacity: Optional[int]) -> StepSolve:
    """The StepSolve of a run: each solve starts with room for krylov_capacity
    vectors (DEFAULT_CAPACITY if None), at most max_krylov."""
    initial = min(DEFAULT_CAPACITY if krylov_capacity is None else krylov_capacity,
                  problem.max_krylov)
    continue_piece = compile_function(backend, partial(step_continue, problem, backend))
    begin_pieces: dict[int, Callable[..., Any]] = {}
    grow_pieces: dict[int, Callable[..., Any]] = {}

    def begin_piece(c: int) -> Callable[..., Any]:
        """step_start compiled for basis capacity c (one version per capacity)."""
        if c not in begin_pieces:
            begin_pieces[c] = compile_function(backend, partial(step_start, problem, backend, capacity=c))
        return begin_pieces[c]

    def grow_piece(c: int) -> Callable[..., Any]:
        if c not in grow_pieces:
            grow_pieces[c] = compile_function(backend, partial(step_grow, problem, backend, capacity=c))
        return grow_pieces[c]

    def solve(state: OuterState, radius: Any, gnorm0: Any) -> tuple[Any, int]:
        capacity, growths = initial, 0
        carry = begin_piece(capacity)(state, radius, gnorm0)
        while as_integer(carry.status) == FULL:
            capacity = min(2 * capacity, problem.max_krylov)
            carry = continue_piece(state, grow_piece(capacity)(state, carry))
            growths += 1
        return carry, growths

    return StepSolve(start=compile_function(backend, partial(init, problem, backend)), solve=solve)


def number(v: Any) -> float:
    """A 0-d array of any backend as a Python float (records hold floats)."""
    return float(np.asarray(v))


def check_resumed_f(f_saved: float, f_x0: float, rtol: float = 1e-10) -> None:
    """A resume state carries f at its checkpoint; f(x0) must match it, or the
    state belongs to another point."""
    if abs(f_x0 - f_saved) > rtol * max(1.0, abs(f_saved)):
        raise ValueError(
            f"resume state does not belong to x0: f(x0) = {f_x0!r}, but the state was saved at "
            f"f = {f_saved!r}. Save x, the state (and g) from the same callback call.")
