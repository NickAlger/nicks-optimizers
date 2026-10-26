"""Deliberate mix-ups of vectors, covectors and their blocks, each marked
with the error code mypy --strict must report on that line (a trailing
comment "E:" and the code).
tests/test_typing.py runs mypy on this file and compares; the file is left
out of the project's own mypy run (pyproject.toml, exclude). Unmarked lines
are correct and must pass."""

from typing import Any, Callable

import numpy as np

from nicks_optimizers import Covector, Point, Vector, VectorTransport, numpy_space
from nicks_optimizers.backend import Backend
from nicks_optimizers.lbfgs_memory import Memory, apply_inverse, memory_store
from nicks_optimizers.spaces import CovectorBlockT, CovectorT, PointT, VectorBlockT, VectorSpace, VectorT


def generic(space: VectorSpace[VectorT, CovectorT, VectorBlockT, CovectorBlockT], backend: Backend,
            memory: Memory[VectorBlockT, CovectorBlockT], s: VectorT, y: CovectorT,
            transport: VectorTransport[PointT, VectorT, CovectorT, VectorBlockT, CovectorBlockT],
            x: PointT, base: Callable[[CovectorT], VectorT]) -> None:
    space.set_vector(memory.S, 0, s)
    space.set_covector(memory.Y, 0, y)
    space.set_vector(memory.S, 0, y)                                 # E: arg-type
    space.set_covector(memory.S, 0, y)                               # E: arg-type
    space.combine_vectors(memory.Y, np.zeros(3), 3)                  # E: arg-type
    space.pair_block(memory.S, s, 3)                                 # E: arg-type
    transport.vector(x, x, s)
    transport.vector(x, x, y)                                        # E: arg-type
    transport.covector(x, x, s)                                      # E: arg-type
    memory_store(space, backend, memory, s, y, 0, 0.0)
    memory_store(space, backend, memory, y, s, 0, 0.0)               # E: misc
    d: VectorT = apply_inverse(space, backend, memory, y, base)
    apply_inverse(space, backend, memory, s, base)                   # E: misc
    w: CovectorT = apply_inverse(space, backend, memory, y, base)    # E: assignment
    del d, w


def concrete(g: Covector, d: Vector, x: Point, retract: Callable[[Point, Vector], Point]) -> Any:
    numpy_space.pair(g, d)
    numpy_space.pair(d, g)                                           # E: arg-type
    numpy_space.axpy_vector(1.0, g, d)                               # E: arg-type
    retract(x, d)
    return retract(d, x)                                             # E: arg-type
