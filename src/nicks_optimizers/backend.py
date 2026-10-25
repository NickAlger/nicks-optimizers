"""The backend: how the core's loops, branches and array operations run.

The trust-region method is a Python driver around pure pieces (see
trust_region_core): functions without Python control flow on data, so that
a framework can compile them. A Backend record says how:

  jit(f) -> f                          compile a pure function (identity for
                                       numpy)
  loop(done, body, carry, max_steps)   the bounded loop: apply body at most
                                       max_steps times (a Python int), stopping
                                       at the first carry with done(carry)
                                       true; equivalent to
                                           for _ in range(max_steps):
                                               if done(carry): break
                                               carry = body(carry)
  cond(pred, true_fun, false_fun, operand)
                                       true_fun(operand) if pred else
                                       false_fun(operand)
  index_update(a, index, value)        a copy of the array a with a[index] =
                                       value (index may be computed)
  xp                                   the array namespace, used only through
                                       the names in CONTRACT
  static_shapes                        True if shapes are fixed when a piece
                                       is compiled (arrays are then padded to
                                       a capacity); False for numpy

numpy_backend is the default. loop_from_scan and loop_from_while build the
bounded loop from either primitive of a framework, e.g. for JAX

    Backend(jit=jax.jit, loop=loop_from_scan(jax.lax.scan, jax.lax.cond),
            cond=jax.lax.cond, index_update=lambda a, i, v: a.at[i].set(v),
            xp=jax.numpy, static_shapes=True)

Built from scan, the loop has a known trip count (finished steps are
skipped); built from while_loop, it exits early. Either way the result is
that of the Python loop above. This module imports no framework: backends
for frameworks live with the code that uses them.

States crossing the backend
---------------------------
The core's states are frozen dataclasses, marked with @backend_state. A
framework can only carry containers it knows, so the backend never sees
them: run_loop, run_cond and compile_function flatten states to plain
tuples (recursively, through tuples and nested states; everything else,
such as arrays and the user's vectors, is left as it is) and rebuild them
on the other side. The plain Python primitives of numpy_backend carry any
object, so for them nothing is flattened.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar

import numpy as np

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")
Done = Callable[[Any], Any]
Body = Callable[[Any], Any]
Loop = Callable[[Done, Body, Any, int], Any]
Cond = Callable[[Any, Callable[[Any], Any], Callable[[Any], Any], Any], Any]


# The names the core may use from xp: a subset common to numpy >= 1.24,
# jax.numpy and the array API standard. Adding a name is a design change
# (dev/jit_backends_plan.md); tests/test_backend.py checks the core against it.
CONTRACT: frozenset[str] = frozenset({
    # creation
    "asarray", "zeros", "ones", "full", "eye", "arange", "stack",
    # elementwise
    "sqrt", "abs", "maximum", "minimum", "where", "isfinite",
    "logical_and", "logical_or", "logical_not",
    # reductions and sorting
    "sum", "max", "min", "argsort", "take",
    # linear algebra (see CONTRACT_LINALG) and dtypes
    "matmul", "linalg", "finfo",
    # constants
    "inf", "nan",
})
CONTRACT_LINALG: frozenset[str] = frozenset({"eigh", "cholesky", "solve"})


@dataclass(frozen=True, kw_only=True)
class Backend:
    """How the core's pieces are compiled and how they loop, branch and
    compute (see the module docstring)."""
    jit: Callable[[Any], Any]
    loop: Loop
    cond: Cond
    index_update: Callable[[Any, Any, Any], Any]
    xp: Any
    static_shapes: bool


def python_loop(done: Done, body: Body, carry: Any, max_steps: int) -> Any:
    """The bounded loop in plain Python (the definition)."""
    for _ in range(max_steps):
        if done(carry):
            break
        carry = body(carry)
    return carry


def python_cond(pred: Any, true_fun: Callable[[Any], Any], false_fun: Callable[[Any], Any],
                operand: Any) -> Any:
    return true_fun(operand) if pred else false_fun(operand)


def numpy_index_update(a: Any, index: Any, value: Any) -> Any:
    out = np.array(a, copy=True)
    out[index] = value
    return out


def _identity(f: F) -> F:
    return f


numpy_backend = Backend(jit=_identity, loop=python_loop, cond=python_cond,
                        index_update=numpy_index_update, xp=np, static_shapes=False)


def loop_from_scan(scan: Callable[..., Any], cond: Cond) -> Loop:
    """The bounded loop from scan(f, init, xs, length=...) -> (carry, ys) and
    cond: max_steps steps, where the steps after done(carry) are skipped by
    cond. done must return a boolean array."""
    def loop(done: Done, body: Body, carry: Any, max_steps: int) -> Any:
        def step(c: Any, _: Any) -> tuple[Any, None]:
            return cond(done(c), lambda c: c, body, c), None
        return scan(step, carry, None, length=max_steps)[0]
    return loop


def loop_from_while(while_loop: Callable[..., Any]) -> Loop:
    """The bounded loop from while_loop(cond_fun, body_fun, init) -> carry:
    it stops at the first carry with done(carry), or after max_steps steps.
    done must return a boolean array."""
    def loop(done: Done, body: Body, carry: Any, max_steps: int) -> Any:
        def going(ic: tuple[Any, Any]) -> Any:
            return (ic[0] < max_steps) & ~done(ic[1])

        def step(ic: tuple[Any, Any]) -> tuple[Any, Any]:
            return ic[0] + 1, body(ic[1])
        return while_loop(going, step, (0, carry))[1]
    return loop


# --- states crossing the backend ----------------------------------------------------------


_STATE_TYPES: set[type] = set()


def backend_state(cls: type[T]) -> type[T]:
    """Class decorator for the core's frozen dataclasses: their instances are
    flattened to tuples wherever the backend sees them."""
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls.__name__} is not a dataclass")
    _STATE_TYPES.add(cls)
    return cls


# A structure is None (a leaf), ("tuple", children) or (state class, children).
Structure = Optional[tuple[Any, tuple[Any, ...]]]


def structure_of(value: Any) -> Structure:
    """The shape of a value's nesting of states and tuples (hashable)."""
    if type(value) in _STATE_TYPES:
        return (type(value), tuple(structure_of(getattr(value, f.name))
                                   for f in dataclasses.fields(value)))
    if type(value) is tuple:
        return ("tuple", tuple(structure_of(v) for v in value))
    return None


def flatten(value: Any) -> Any:
    """value with every state replaced by the tuple of its fields."""
    if type(value) in _STATE_TYPES:
        return tuple(flatten(getattr(value, f.name)) for f in dataclasses.fields(value))
    if type(value) is tuple:
        return tuple(flatten(v) for v in value)
    return value


def unflatten(flat: Any, structure: Structure) -> Any:
    """The inverse of flatten, given the structure of the original value."""
    if structure is None:
        return flat
    kind, children = structure
    parts = [unflatten(v, s) for v, s in zip(flat, children)]
    if kind == "tuple":
        return tuple(parts)
    return kind(**{f.name: p for f, p in zip(dataclasses.fields(kind), parts)})


def _check_structure(value: Any, expected: Structure, what: str) -> Any:
    found = structure_of(value)
    if found != expected:
        raise TypeError(f"{what} changed the structure of its value: {found} instead of {expected}")
    return flatten(value)


def run_loop(backend: Backend, done: Callable[[T], Any], body: Callable[[T], T], carry: T,
             max_steps: int) -> T:
    """backend.loop on a carry that may contain states."""
    if backend.loop is python_loop:
        out: T = python_loop(done, body, carry, max_steps)
        return out
    s = structure_of(carry)
    result = backend.loop(lambda c: done(unflatten(c, s)),
                          lambda c: _check_structure(body(unflatten(c, s)), s, "a loop body"),
                          flatten(carry), max_steps)
    flat_out: T = unflatten(result, s)
    return flat_out


def run_cond(backend: Backend, pred: Any, true_fun: Callable[[Any], T],
             false_fun: Callable[[Any], T], operand: Any) -> T:
    """backend.cond on operands and results that may contain states."""
    if backend.cond is python_cond:
        return true_fun(operand) if pred else false_fun(operand)
    s_in = structure_of(operand)
    found: list[Structure] = []

    def wrap(f: Callable[[Any], T]) -> Callable[[Any], Any]:
        def branch(flat: Any) -> Any:
            out = f(unflatten(flat, s_in))
            found.append(structure_of(out))
            if found[0] != found[-1]:
                raise TypeError(f"the branches of a cond return different structures: "
                                f"{found[0]} and {found[-1]}")
            return flatten(out)
        return branch
    result = backend.cond(pred, wrap(true_fun), wrap(false_fun), flatten(operand))
    out: T = unflatten(result, found[-1])
    return out


def compile_function(backend: Backend, f: Callable[..., T]) -> Callable[..., T]:
    """backend.jit(f) for arguments and results that may contain states. One
    compiled version per structure of the arguments."""
    if backend.jit is _identity:
        return f
    compiled: dict[tuple[Structure, ...], tuple[Callable[..., Any], list[Structure]]] = {}

    def call(*args: Any) -> T:
        s_in = tuple(structure_of(a) for a in args)
        entry = compiled.get(s_in)
        if entry is None:
            s_out: list[Structure] = [None]

            def flat_f(*flat_args: Any) -> Any:
                result = f(*(unflatten(a, s) for a, s in zip(flat_args, s_in)))
                s_out[0] = structure_of(result)
                return flatten(result)
            entry = (backend.jit(flat_f), s_out)
            compiled[s_in] = entry
        fn, s_out_cell = entry
        result = fn(*(flatten(a) for a in args))
        out: T = unflatten(result, s_out_cell[0])
        return out
    return call
