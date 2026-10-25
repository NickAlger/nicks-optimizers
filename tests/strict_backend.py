"""A strict numpy backend: checks, without any framework, that the core
could be compiled (dev/jit_backends_plan.md, "Guarding against drift").

  * Values are Strict arrays, an ndarray subclass whose bool(), float(),
    int() and index() raise StrictError, so Python control flow on data
    (an `if`, `and`, `min()` on values) fails.
  * xp exposes only the contract names (backend.CONTRACT); anything else is
    an AttributeError.
  * loop is built from a scan with loop_from_scan, as a JAX user would.
    The scan checks that the carry keeps its structure, shapes and dtypes,
    and that it holds arrays, not Python numbers.
  * cond evaluates both branches and selects, as vmap does: a branch must
    not raise on the other case's data, and both must return the same
    structure, shapes and dtypes.
  * static_shapes is True, so the padded code paths run.
  * jit makes the arguments of a compiled piece Strict, as tracing would.

The user's functions receive Strict arrays too, so test problems used here
must be traceable (no float() on values, no ifs on data), as with JAX.
"""

import types
from typing import Any, Callable

import numpy as np

from nicks_optimizers.backend import CONTRACT, CONTRACT_LINALG, Backend, loop_from_scan


class StrictError(TypeError):
    """Python control flow on a value computed by the core."""


def _refuse(what: str) -> Callable[..., Any]:
    def method(self: Any, *args: Any) -> Any:
        raise StrictError(f"{what} of a computed value: Python control flow on data")
    return method


class Strict(np.ndarray):
    __bool__ = _refuse("bool()")
    __float__ = _refuse("float()")
    __int__ = _refuse("int()")
    __index__ = _refuse("operator.index()")
    __complex__ = _refuse("complex()")

    def __array_wrap__(self, obj: Any, context: Any = None, return_scalar: bool = False) -> Any:
        return np.asarray(obj).view(Strict)

    def __getitem__(self, index: Any) -> Any:
        return strict(np.ndarray.__getitem__(self.view(np.ndarray), plain(index)))

    def __setitem__(self, index: Any, value: Any) -> None:
        np.ndarray.__setitem__(self.view(np.ndarray), plain(index), plain(value))


def strict(value: Any) -> Any:
    """value with every array or number in it (through tuples) made Strict."""
    if isinstance(value, tuple):
        return tuple(strict(v) for v in value)
    if isinstance(value, (np.ndarray, np.generic, float, int, bool)):
        return np.asarray(value).view(Strict)
    return value


def plain(value: Any) -> Any:
    """value with Strict arrays (through tuples) turned back into ndarrays."""
    if isinstance(value, tuple):
        return tuple(plain(v) for v in value)
    if isinstance(value, Strict):
        return value.view(np.ndarray)
    return value


def truth(value: Any) -> bool:
    """bool() for the backend itself, which may look at values."""
    return bool(np.asarray(plain(value)))


def _lift(fn: Callable[..., Any]) -> Callable[..., Any]:
    def lifted(*args: Any, **kwargs: Any) -> Any:
        return strict(fn(*args, **kwargs))
    return lifted


def _linalg(name: str) -> Callable[..., Any]:
    f = getattr(np.linalg, name)
    if name == "eigh":                       # a named tuple in numpy 2: make it a plain pair
        return _lift(lambda a: tuple(f(a)))
    return _lift(f)


def _namespace() -> types.SimpleNamespace:
    ns = types.SimpleNamespace()
    for name in CONTRACT:
        if name == "linalg":
            ns.linalg = types.SimpleNamespace(**{n: _linalg(n) for n in CONTRACT_LINALG})
        elif name in ("inf", "nan"):
            setattr(ns, name, getattr(np, name))
        elif name == "finfo":
            ns.finfo = np.finfo
        else:
            setattr(ns, name, _lift(getattr(np, name)))
    return ns


strict_xp = _namespace()


def leaves(value: Any) -> list[Any]:
    """The leaves of a container (tuples, lists, dicts), JAX-style."""
    if isinstance(value, (tuple, list)):
        return [leaf for v in value for leaf in leaves(v)]
    if isinstance(value, dict):
        return [leaf for k in sorted(value) for leaf in leaves(value[k])]
    return [value]


def layout(value: Any) -> Any:
    """Structure, shapes and dtypes of a container's leaves."""
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(layout(v) for v in value))
    if isinstance(value, dict):
        return ("dict", tuple((k, layout(value[k])) for k in sorted(value)))
    if value is None:
        return None
    if isinstance(value, np.generic):                # a numpy scalar: a 0-d array
        return ((), value.dtype.str)
    if isinstance(value, (bool, int, float)):
        raise StrictError(f"a Python number ({value!r}) in a carry or a branch result: "
                          f"make it an array with xp.asarray")
    if isinstance(value, np.ndarray):
        return (value.shape, value.dtype.str)
    return ("object", id(value))


def strict_scan(f: Callable[[Any, Any], tuple[Any, Any]], init: Any, xs: Any, length: int
                ) -> tuple[Any, None]:
    carry = init
    expected = layout(carry)
    for _ in range(length):
        carry, _ys = f(carry, None)
        found = layout(carry)
        if found != expected:
            raise StrictError(f"the loop carry changed: {found} instead of {expected}")
    return carry, None


def strict_cond(pred: Any, true_fun: Callable[[Any], Any], false_fun: Callable[[Any], Any],
                operand: Any) -> Any:
    a, b = true_fun(operand), false_fun(operand)
    if layout(a) != layout(b):
        raise StrictError(f"cond branches differ: {layout(a)} and {layout(b)}")
    take = truth(pred)
    return select(take, a, b)


def select(take: bool, a: Any, b: Any) -> Any:
    if isinstance(a, (tuple, list)):
        return type(a)(select(take, u, v) for u, v in zip(a, b))
    if isinstance(a, dict):
        return {k: select(take, a[k], b[k]) for k in a}
    if isinstance(a, np.ndarray):
        return strict(np.where(take, plain(a), plain(b)))
    return a if take else b


def strict_index_update(a: Any, index: Any, value: Any) -> Any:
    out = np.array(plain(a), copy=True)
    out[plain(index)] = plain(value)
    return out.view(Strict)


def strict_jit(f: Callable[..., Any]) -> Callable[..., Any]:
    """Arguments become Strict, as they become tracers under compilation."""
    return lambda *args: f(*strict(args))


strict_backend = Backend(jit=strict_jit, loop=loop_from_scan(strict_scan, strict_cond),
                         cond=strict_cond, index_update=strict_index_update, xp=strict_xp,
                         static_shapes=True)
