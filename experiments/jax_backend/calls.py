"""How often the user's operators run inside compiled code (phase 7 of
dev/jit_backends_plan.md): each operator is wrapped with a host callback
that counts its executions at run time (a trace happens once, so counting
calls in Python would not do). Checked: the counts equal the summary's, and
they do not depend on the room the basis starts with (2 or 8 vectors, so it
grows many times, or 100), for loops built from scan (whose skipped steps
must not run anything) and from while_loop. Growing the basis copies it and
repeats no Hessian product, preconditioner solve or metric operation.

Run:  python experiments/jax_backend/calls.py
"""
import collections
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, jax, jax.numpy as jnp
from jax.experimental import io_callback
import jax_backend
from solvers import configurations
from study import plaplace
from nicks_optimizers import array_space, trust_region_newton

N = 31; P = plaplace(N); n = N * N
for solver in ("lanczos", "lanczos P!=M", "flexible"):
    for loop in ("scan", "while"):
        line = f"{solver:13s} {loop:5s}"
        seen = {}
        for cap in (2, 8, 100):
            count = collections.Counter()
            def counted(key, f):
                def bump(_):
                    count[key] += 1
                    return np.int32(0)
                def wrapper(*args):
                    token = io_callback(bump, jax.ShapeDtypeStruct((), jnp.int32), args[-1][0], ordered=True)
                    return f(*args) + 0.0 * token
                return wrapper
            opts = {k: counted(k, f) if callable(f) and k != "krylov" else f
                    for k, f in configurations(N, P)[solver].items()}
            bk = jax_backend.backend(loop)
            _, s = trust_region_newton(P["fun"], P["grad"], counted("hessvec", P["hessvec"]), jnp.zeros(n),
                                       space=array_space(bk), retract=lambda x, s: x + s, backend=bk,
                                       krylov_capacity=cap, max_iter=300, max_krylov=100, **opts)
            summary = dict(hessvec=s.hessvec_calls, metric_solve=s.metric_solve_calls,
                           metric_apply=s.metric_apply_calls, precond_solve=s.precond_solve_calls)
            ran = {k: count.get(k, 0) for k in summary}
            assert ran == summary, (solver, loop, cap, ran, summary)
            seen[cap] = ran
            line += f" | room {cap:3d}: {s.capacity_growths:2d} growths, executed {ran}"
        assert seen[2] == seen[8] == seen[100]
        print(line, flush=True)
print("all executions equal the summary counts, and do not depend on the room")
