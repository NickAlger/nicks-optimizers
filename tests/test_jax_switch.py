"""hessian="switch" compiled with JAX (dev/archive/gauss_newton_switch_plan.md, phase
5): compiled runs against numpy, how often each Hessian is traced, and the
full-Hessian pieces compiled only once the switch fires. Skipped without JAX
(the CI job "jax" installs it)."""

from typing import Any, Callable, NamedTuple

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

import jax_p_laplacian as ex  # noqa: E402  (enables float64; the JAX backend)
from nicks_optimizers import (  # noqa: E402
    array_space,
    euclidean_metric_apply,
    euclidean_metric_solve,
    line_search_newton,
    trust_region_newton,
)

METHODS: dict[str, Any] = {"trust region": trust_region_newton, "line search": line_search_newton}
SOLVERS = ("cg", "lanczos", "flexible")


class JaxLeastSquares(NamedTuple):
    fun: Callable[[Any], Any]
    grad: Callable[[Any], Any]
    gn_hessvec: Callable[[Any, Any], Any]
    hessvec: Callable[[Any, Any], Any]
    x0: Any


def rosenbrock(n: int = 20, alpha: float = 0.1, cscale: float = 0.5, seed: int = 1,
               a: float = 10.0) -> JaxLeastSquares:
    """problems.regularized_rosenbrock in jax.numpy: the residual
    r(x), and J, J^T from jax.jvp and jax.vjp; S v = J'(x)[v]^T r from
    the derivative of the gradient minus the Gauss-Newton part."""
    c = jnp.asarray(1.0 + cscale * np.random.default_rng(seed).standard_normal(n))
    root = float(np.sqrt(alpha))

    def residual(x: Any) -> Any:
        u, w = x[0::2], x[1::2]
        return jnp.concatenate([a * (w - u**2), 1.0 - u, root * (x - c)])

    def fun(x: Any) -> Any:
        return 0.5 * jnp.sum(residual(x) ** 2)

    def grad(x: Any) -> Any:
        r, vjp = jax.vjp(residual, x)
        return vjp(r)[0]

    def gn_hessvec(x: Any, v: Any) -> Any:
        _, jv = jax.jvp(residual, (x,), (v,))
        _, vjp = jax.vjp(residual, x)
        return vjp(jv)[0]

    def hessvec(x: Any, v: Any) -> Any:
        return jax.jvp(grad, (x,), (v,))[1]

    return JaxLeastSquares(fun, grad, gn_hessvec, hessvec, jnp.tile(jnp.asarray([-1.2, 1.0]), n // 2))


def solver_options(solver: str) -> dict[str, Any]:
    if solver in ("cg", "lanczos"):
        return {"krylov": solver, "metric_solve": euclidean_metric_solve}
    return {"krylov": solver, "metric_solve": euclidean_metric_solve,
            "metric_apply": euclidean_metric_apply, "precond_solve": lambda x, w: w / 1.5}


def compiled(p: JaxLeastSquares, method: str, solver: str, loop: str = "scan",
             **options: Any) -> Any:
    backend = ex.jax_backend(loop)
    return METHODS[method](p.fun, p.grad, p.hessvec, p.x0, hessian="switch",
                           gn_hessvec=p.gn_hessvec, space=array_space(backend),
                           retract=lambda x, s: x + s, backend=backend, rtol=1e-12,
                           **solver_options(solver), **options)[1]


def numpy_run(p: JaxLeastSquares, method: str, solver: str) -> Any:
    """The numpy backend: each JAX function compiled on its own."""
    def np_fn(f: Callable[..., Any]) -> Callable[..., Any]:
        g = jax.jit(f)
        return lambda *args: np.asarray(g(*args))
    fun = jax.jit(p.fun)
    ops = {k: np_fn(v) if k in ("precond_solve",) else v for k, v in solver_options(solver).items()}
    return METHODS[method](lambda x: float(fun(x)), np_fn(p.grad), np_fn(p.hessvec),
                           np.asarray(p.x0), hessian="switch", gn_hessvec=np_fn(p.gn_hessvec),
                           rtol=1e-12, **ops)[1]


@pytest.mark.parametrize("loop", ["scan", "while"])
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("solver", SOLVERS)
def test_compiled_switch_agrees_with_numpy(solver: str, method: str, loop: str) -> None:
    p = rosenbrock()
    reference = numpy_run(p, method, solver)
    run = compiled(p, method, solver, loop)
    assert reference.reason == run.reason == "converged"
    assert reference.switch_iteration is not None
    assert run.switch_iteration == reference.switch_iteration
    assert [(r.model, r.accepted, r.subproblem.krylov_steps) for r in run.iterations] == \
        [(r.model, r.accepted, r.subproblem.krylov_steps) for r in reference.iterations]
    for a, b in zip(run.iterations, reference.iterations):
        assert abs(a.f - b.f) <= 1e-10 * max(1.0, abs(b.f))
    assert (run.hessvec_calls, run.gn_hessvec_calls) == \
        (reference.hessvec_calls, reference.gn_hessvec_calls)


@pytest.mark.parametrize("method", METHODS)
def test_traces_and_lazy_full_phase(method: str) -> None:
    """With a fixed capacity, each Hessian is traced a fixed number of times
    (once per compiled piece that uses it), however long the run. Before the
    switch hessvec is traced only by the tracking finish piece; the
    full-Hessian pieces are compiled only once the switch fires."""
    p = rosenbrock()
    traces: dict[int, dict[str, int]] = {}
    summaries: dict[int, Any] = {}
    for max_iter in (2, 200):
        count = {"hessvec": 0, "gn_hessvec": 0}

        def counted(key: str, f: Callable[..., Any]) -> Callable[..., Any]:
            def wrapper(*args: Any) -> Any:
                count[key] += 1
                return f(*args)
            return wrapper
        traced = p._replace(hessvec=counted("hessvec", p.hessvec),
                            gn_hessvec=counted("gn_hessvec", p.gn_hessvec))
        summaries[max_iter] = compiled(traced, method, "lanczos", krylov_capacity=50,
                                       max_krylov=50, max_iter=max_iter)
        traces[max_iter] = count
    short, long = summaries[2], summaries[200]
    assert short.switch_iteration is None and long.switch_iteration is not None
    assert long.hessvec_calls > 20 and long.gn_hessvec_calls > 20
    assert traces[2]["hessvec"] == 1                      # the tracking product only
    assert 1 < traces[200]["hessvec"] <= 4                # + the full phase's pieces
    assert traces[2]["gn_hessvec"] == traces[200]["gn_hessvec"] <= 3
