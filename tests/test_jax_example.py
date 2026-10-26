"""The JAX example (examples/jax_p_laplacian.py): compiled runs of both
optimizers against the numpy backend, how often the user's functions are
traced, and no host callbacks in the compiled pieces. Skipped without JAX
(the CI job "jax" installs it)."""

import dataclasses
from typing import Any, Callable

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

import jax_p_laplacian as ex  # noqa: E402
from nicks_optimizers import array_space  # noqa: E402

N = 15


def agree(reference: Any, run: Any) -> None:
    """Same iterations, Krylov steps, stop reasons (and for the line search
    directions and trials); values to rounding."""
    assert run.reason == reference.reason
    assert [r.subproblem.krylov_steps for r in run.iterations] == \
        [r.subproblem.krylov_steps for r in reference.iterations]
    assert [r.accepted for r in run.iterations] == [r.accepted for r in reference.iterations]
    if hasattr(reference.iterations[0], "trials"):
        assert [(r.direction, r.trials) for r in run.iterations] == \
            [(r.direction, r.trials) for r in reference.iterations]
    for a, b in zip(run.iterations, reference.iterations):
        assert abs(a.f - b.f) <= 1e-12 * abs(b.f)
        assert abs(a.gnorm - b.gnorm) <= 1e-8 * reference.gnorm0


@pytest.mark.parametrize("loop", ["scan", "while"])
@pytest.mark.parametrize("method", ex.METHODS)
@pytest.mark.parametrize("solver", ex.SOLVERS)
def test_compiled_runs_agree_with_numpy(solver: str, method: str, loop: str) -> None:
    problem = ex.p_laplacian(N)
    _, reference = ex.solve_numpy(problem, solver, method)
    _, run = ex.solve_compiled(problem, solver, loop, method)
    assert reference.reason == "converged"
    agree(reference, run)


@pytest.mark.parametrize("method", ex.METHODS)
@pytest.mark.parametrize("solver", ["lanczos", "flexible"])
def test_compiled_bases_grow_and_continue(solver: str, method: str) -> None:
    """A basis with room for two vectors doubles whenever a solve fills it,
    and the run is the numpy run."""
    problem = ex.p_laplacian(N)
    _, reference = ex.solve_numpy(problem, solver, method)
    _, run = ex.solve_compiled(problem, solver, method=method, krylov_capacity=2)
    agree(reference, run)
    assert run.capacity_growths > 0


@pytest.mark.parametrize("method", ex.METHODS)
def test_functions_are_traced_once_per_piece(method: str) -> None:
    """With a fixed capacity, fun and hessvec are traced a fixed number of
    times (once per compiled piece that uses them), however many iterations
    and Krylov steps run."""
    problem = ex.p_laplacian(N)
    traces: dict[int, dict[str, int]] = {}
    for max_iter in (2, 100):
        count = {"fun": 0, "hessvec": 0}

        def counted(key: str, f: Callable[..., Any]) -> Callable[..., Any]:
            def wrapper(*args: Any) -> Any:
                count[key] += 1
                return f(*args)
            return wrapper
        traced = problem._replace(fun=counted("fun", problem.fun),
                                  hessvec=counted("hessvec", problem.hessvec))
        _, run = ex.solve_compiled(traced, "lanczos", method=method, krylov_capacity=50,
                                   max_krylov=50, max_iter=max_iter)
        traces[max_iter] = count
    assert len(run.iterations) > 2 and run.hessvec_calls > 20
    assert traces[2] == traces[100] and 0 < traces[100]["hessvec"] <= 4
    assert 0 < traces[100]["fun"] <= 3          # the line search's trials included


def test_compiled_pieces_make_no_host_callbacks() -> None:
    """Every compiled piece, lowered: no host callback (the driver alone
    talks to Python, between pieces)."""
    texts: list[str] = []
    base = ex.jax_backend("scan")

    def jit(f: Callable[..., Any]) -> Callable[..., Any]:
        compiled = jax.jit(f)

        def call(*args: Any) -> Any:
            texts.append(compiled.lower(*args).as_text())
            return compiled(*args)
        return call
    backend = dataclasses.replace(base, jit=jit)
    problem = ex.p_laplacian(7)
    for method in ex.METHODS:
        for solver in ex.SOLVERS:
            ex.optimize(method, problem.fun, problem.grad, problem.hessvec, jnp.zeros(49),
                        space=array_space(backend), retract=lambda x, s: x + s, backend=backend,
                        krylov_capacity=2, max_iter=4, **ex.operators(problem, solver))
    assert len(texts) > 10 and not any("callback" in text for text in texts)
    probe = jax.jit(lambda x: (jax.debug.callback(lambda v: None, x), x)[1])
    assert "callback" in probe.lower(jnp.ones(2)).as_text()      # the test would see one


def test_unknown_options_are_refused() -> None:
    with pytest.raises(ValueError):
        ex.jax_backend("fori")
    with pytest.raises(ValueError):
        ex.operators(ex.p_laplacian(3), "gmres")
    with pytest.raises(ValueError):
        ex.optimize("bisection")
    assert np.isfinite(float(ex.p_laplacian(3).fun(jnp.zeros(9))))
