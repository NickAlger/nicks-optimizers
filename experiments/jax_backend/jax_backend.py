"""Backend records for JAX (dev/jit_backends_plan.md, phase 3): everything
JAX-specific the library needs, in a few lines outside the library.

    import jax_backend
    trust_region_newton(..., backend=jax_backend.backend("scan"))

Importing this module turns on 64-bit floats in JAX (the method's
tolerances assume float64).
"""

import jax
import jax.numpy as jnp

from nicks_optimizers import Backend, loop_from_scan, loop_from_while

jax.config.update("jax_enable_x64", True)


def backend(loop: str = "scan") -> Backend:
    """A JAX backend whose bounded loop is built from jax.lax.scan (a fixed
    trip count, finished steps skipped) or jax.lax.while_loop (early exit)."""
    if loop == "scan":
        bounded = loop_from_scan(jax.lax.scan, jax.lax.cond)
    elif loop == "while":
        bounded = loop_from_while(jax.lax.while_loop)
    else:
        raise ValueError(f"loop must be 'scan' or 'while', not {loop!r}")
    return Backend(jit=jax.jit, loop=bounded, cond=jax.lax.cond,
                   index_update=lambda a, i, v: a.at[i].set(v), xp=jnp, static_shapes=True)
