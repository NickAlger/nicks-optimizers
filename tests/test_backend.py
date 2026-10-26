"""The backend record, the bounded loop, state flattening, and the guardrails
against drifting toward one framework (dev/jit_backends_plan.md):

  * src/ imports nothing but the standard library and numpy, and the core
    modules (NUMPY_FREE) not even numpy;
  * the core uses only the contract names of the array namespace xp;
  * the strict backend catches what a compiling framework would refuse.
"""

import ast
import dataclasses
import pathlib
import sys
from typing import Any, Callable

import numpy as np
import pytest

from nicks_optimizers.backend import (
    CONTRACT,
    CONTRACT_LINALG,
    backend_state,
    compile_function,
    flatten,
    loop_from_scan,
    loop_from_while,
    numpy_backend,
    python_cond,
    python_loop,
    run_cond,
    run_loop,
    structure_of,
    unflatten,
)
from nicks_optimizers import Covector, Vector, array_space, check_space, numpy_space, tuple_tree_space
from strict_backend import Strict, StrictError, strict, strict_backend, strict_xp

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "nicks_optimizers"

# Core modules: everything in them runs inside compiled pieces, so they use
# the backend's xp and never numpy. Grows as the refactor converts modules.
NUMPY_FREE: list[str] = ["newton_core", "trust_region_core", "dense", "krylov", "krylov_solvers",
                         "trust_region_subproblem"]


def modules() -> dict[str, ast.Module]:
    return {p.stem: ast.parse(p.read_text()) for p in sorted(SRC.glob("*.py"))}


def imported_roots(tree: ast.Module) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_src_imports_only_the_standard_library_and_numpy() -> None:
    allowed = set(sys.stdlib_module_names) | {"numpy", "__future__"}
    for name, tree in modules().items():
        extra = imported_roots(tree) - allowed
        assert not extra, f"{name}.py imports {sorted(extra)}"


def test_core_modules_do_not_import_numpy() -> None:
    found = modules()
    for name in NUMPY_FREE:
        assert name in found, name
        assert "numpy" not in imported_roots(found[name]), f"{name}.py imports numpy"


def is_xp(node: ast.expr) -> bool:
    return ((isinstance(node, ast.Name) and node.id == "xp")
            or (isinstance(node, ast.Attribute) and node.attr == "xp"))


def test_core_uses_only_contract_names() -> None:
    for name, tree in modules().items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            if is_xp(node.value):
                assert node.attr in CONTRACT, f"{name}.py uses xp.{node.attr}"
            elif (isinstance(node.value, ast.Attribute) and node.value.attr == "linalg"
                  and is_xp(node.value.value)):
                assert node.attr in CONTRACT_LINALG, f"{name}.py uses xp.linalg.{node.attr}"


def test_contract_names_exist_in_numpy() -> None:
    for name in CONTRACT:
        assert hasattr(np, name), name
    for name in CONTRACT_LINALG:
        assert hasattr(np.linalg, name), name


# --- the bounded loop -----------------------------------------------------------------------


def python_scan(f: Callable[[Any, Any], Any], init: Any, xs: Any, length: int) -> tuple[Any, None]:
    carry = init
    for _ in range(length):
        carry, _ = f(carry, None)
    return carry, None


def python_while(cond_fun: Callable[[Any], Any], body_fun: Callable[[Any], Any], init: Any) -> Any:
    carry = init
    while cond_fun(carry):
        carry = body_fun(carry)
    return carry


LOOPS = {"python": python_loop, "from scan": loop_from_scan(python_scan, python_cond),
         "from while": loop_from_while(python_while)}

# Python primitives that are not numpy_backend's own: states get flattened.
FLATTENING = dataclasses.replace(numpy_backend, jit=lambda f: f, loop=LOOPS["from while"],
                                 cond=lambda p, t, f, x: t(x) if p else f(x))


@pytest.mark.parametrize("which", list(LOOPS))
def test_bounded_loops_agree(which: str) -> None:
    """Counting up to a target: the loop stops at the first done carry, or
    after max_steps; the body never sees a done carry."""
    loop = LOOPS[which]
    for target, max_steps in ((3, 10), (10, 3), (0, 5), (4, 4)):
        calls: list[int] = []

        def body(c: Any) -> Any:
            assert not c >= target
            calls.append(int(c))
            return c + 1
        out = loop(lambda c: np.asarray(c >= target), body, np.asarray(0), max_steps)
        assert int(out) == min(target, max_steps), (which, target, max_steps)
        if which != "from scan":
            assert len(calls) == min(target, max_steps)


# --- states ---------------------------------------------------------------------------------


@backend_state
@dataclasses.dataclass(frozen=True, kw_only=True)
class Inner:
    a: Any
    b: Any


@backend_state
@dataclasses.dataclass(frozen=True, kw_only=True)
class Outer:
    inner: Inner
    vec: Any                     # a user vector: arrays in tuples
    label: Any = None


def sample_state() -> Outer:
    return Outer(inner=Inner(a=np.asarray(1.0), b=(np.ones(2), np.zeros(3))),
                 vec=(np.arange(2.0), (np.ones(1),)))


def test_flatten_round_trip() -> None:
    s = sample_state()
    flat = flatten(s)
    assert type(flat) is tuple and not any(dataclasses.is_dataclass(v) for v in flat)
    back = unflatten(flat, structure_of(s))
    assert type(back) is Outer and type(back.inner) is Inner
    assert repr(back) == repr(s)
    assert structure_of(np.ones(2)) is None


@pytest.mark.parametrize("backend", [numpy_backend, FLATTENING, strict_backend],
                         ids=["numpy", "flattening", "strict"])
def test_states_through_loops_branches_and_compilation(backend: Any) -> None:
    xp = backend.xp
    s = Outer(inner=Inner(a=xp.asarray(0.0), b=xp.zeros(3)), vec=(xp.ones(2),))

    def body(c: Outer) -> Outer:
        b = backend.index_update(c.inner.b, 0, c.inner.b[0] + 1.0)
        return dataclasses.replace(c, inner=Inner(a=c.inner.a + 1.0, b=b))
    out = run_loop(backend, lambda c: c.inner.a >= 2.5, body, s, 10)
    assert float(np.asarray(out.inner.a)) == 3.0 and float(np.asarray(out.inner.b)[0]) == 3.0

    picked = run_cond(backend, out.inner.a > 1.0, lambda c: c.vec, lambda c: (c.vec[0] * 0.0,), out)
    assert np.asarray(picked[0]).tolist() == [1.0, 1.0]

    double = compile_function(backend, lambda c, k: dataclasses.replace(c, vec=(c.vec[0] * k,)))
    assert np.asarray(double(out, 2.0).vec[0]).tolist() == [2.0, 2.0]


def test_loop_bodies_must_keep_the_structure() -> None:
    s = Outer(inner=Inner(a=np.asarray(0.0), b=None), vec=None)
    with pytest.raises(TypeError):
        run_loop(FLATTENING, lambda c: c.inner.a > 1.0,
                 lambda c: dataclasses.replace(c, vec=(c.inner.a,)), s, 3)


# --- the strict backend ---------------------------------------------------------------------


def test_strict_values_refuse_python_control_flow() -> None:
    x = strict_xp.asarray(1.5)
    assert isinstance(x * 2.0, Strict) and isinstance(strict_xp.sum(strict_xp.ones(3)), Strict)
    assert isinstance(strict_xp.ones(3)[1], Strict)
    for refuse in (bool, float, int, lambda v: v if v > 0 else -v, lambda v: max(v, 0.0)):
        with pytest.raises(StrictError):
            refuse(x)


def test_strict_namespace_has_only_the_contract() -> None:
    with pytest.raises(AttributeError):
        strict_xp.cos  # noqa: B018
    w, V = strict_xp.linalg.eigh(strict_xp.eye(2))
    assert isinstance(w, Strict) and isinstance(V, Strict)


def test_strict_factorizations_fail_with_nan_as_jax_does() -> None:
    # A body run on a discarded carry may factor a singular matrix: NaN, not
    # an exception (numpy raises, JAX does not).
    singular = strict_xp.asarray([[1.0, 2.0], [2.0, 1.0]])
    L = strict_xp.linalg.cholesky(singular)
    x = strict_xp.linalg.solve(strict_xp.zeros((2, 2)), strict_xp.ones(2))
    assert isinstance(L, Strict) and np.isnan(np.asarray(L).view(np.ndarray)).all()
    assert x.shape == (2,) and np.isnan(x.view(np.ndarray)).all()


def test_strict_backend_catches_what_a_compiler_would_refuse() -> None:
    bk = strict_backend
    zero = strict(np.asarray(0.0))
    # an if on data inside a loop body
    with pytest.raises(StrictError):
        bk.loop(lambda c: c > 2.0, lambda c: c + 1.0 if c < 1.0 else c + 2.0, zero, 5)
    # a carry that changes shape
    with pytest.raises(StrictError):
        bk.loop(lambda c: c.sum() > 2.0, lambda c: strict_xp.zeros(2) + 1.0, strict_xp.zeros(1), 5)
    # a Python number in the carry
    with pytest.raises(StrictError):
        bk.loop(lambda c: strict(c >= 2), lambda c: c + 1, 0, 5)
    # branches of different shapes
    with pytest.raises(StrictError):
        bk.cond(zero > 1.0, lambda c: strict_xp.zeros(2), lambda c: strict_xp.zeros(3), zero)
    # both branches run: one that fails on the other case's data fails the test
    with pytest.raises(ZeroDivisionError):
        bk.cond(zero > 1.0, lambda c: 1 / 0, lambda c: c, zero)


# --- spaces with blocks of fixed capacity ----------------------------------------------------


@pytest.mark.parametrize("backend", [numpy_backend, strict_backend], ids=["numpy", "strict"])
@pytest.mark.parametrize("shape", [(7,), (3, 4)], ids=["1-D", "2-D"])
def test_array_space_satisfies_the_space_identities(backend: Any, shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(0)
    space = array_space(backend)
    v1, v2, w1, w2 = (strict(rng.standard_normal(shape)) if backend is strict_backend
                      else rng.standard_normal(shape) for _ in range(4))
    assert check_space(space, v1, v2, w1, w2) < 1e-14
    tree = tuple_tree_space(space)
    assert check_space(tree, (v1, (w2,)), (v2, (v1,)), (w1, (v2,)), (w2, (w1,))) < 1e-14
    assert space.dimension is not None and space.dimension(v1) == int(np.prod(shape))


def test_array_space_blocks_have_fixed_shapes() -> None:
    """A block is one array of capacity members; growing keeps the members,
    and members at and beyond k do not count."""
    space = array_space(numpy_backend)
    B = space.set_vector(space.vector_block(np.ones(3), 4), 1, 2.0 * np.ones(3))
    assert np.asarray(B).shape == (4, 3)
    grown = space.grow_vector_block(B, 6)
    assert np.asarray(grown).shape == (6, 3) and np.array_equal(np.asarray(grown)[:4], np.asarray(B))
    w = np.array([1.0, 0.0, 0.0])
    assert np.asarray(space.pair_block(B, w, 1)).tolist() == [1.0, 0.0, 0.0, 0.0]
    assert np.asarray(space.combine_vectors(B, np.array([1.0, 1.0, 9.0, 9.0]), 2)).tolist() == [3.0] * 3


def test_numpy_blocks_store_the_members_along_the_leading_axis() -> None:
    """numpy_space blocks preallocate their capacity and track the members
    set; they replace or fill members out of order, and refuse members
    beyond the capacity."""
    B = numpy_space.vector_block(Vector(np.ones(2)), 3)
    assert B.members.tolist() == [[1.0, 1.0]] and B.capacity == 3 and B.count == 1
    B = numpy_space.set_vector(B, 2, Vector(np.full(2, 3.0)))         # member 1 stays zero
    assert B.members.tolist() == [[1.0, 1.0], [0.0, 0.0], [3.0, 3.0]]
    B = numpy_space.set_vector(B, 1, Vector(np.full(2, 2.0)))
    assert B.members[1].tolist() == [2.0, 2.0]
    assert numpy_space.pair_block(B, Covector(np.ones(2)), 2).tolist() == [2.0, 4.0, 0.0]
    with pytest.raises(IndexError):
        numpy_space.set_vector(B, 3, Vector(np.ones(2)))
    grown = numpy_space.grow_vector_block(B, 5)
    assert grown.capacity == 5 and np.array_equal(grown.members, B.members)


def test_numpy_blocks_are_values() -> None:
    """Appending writes in place only while no other block owns the row: a
    block never changes, however many blocks are built from it."""
    base = numpy_space.vector_block(Vector(np.zeros(3)), 4)
    a = numpy_space.set_vector(base, 1, Vector(np.ones(3)))          # in place
    b = numpy_space.set_vector(base, 1, Vector(np.full(3, 2.0)))     # row 1 is a's: a copy
    c = numpy_space.set_vector(a, 2, Vector(np.full(3, 3.0)))        # a's next row: in place
    assert a.storage is base.storage and b.storage is not base.storage and c.storage is a.storage
    assert base.count == 1 and base.members.tolist() == [[0.0] * 3]
    assert a.members.tolist() == [[0.0] * 3, [1.0] * 3]
    assert b.members.tolist() == [[0.0] * 3, [2.0] * 3]
    assert c.members.tolist() == [[0.0] * 3, [1.0] * 3, [3.0] * 3]
    w = Covector(np.ones(3))                                          # k beyond the members: zeros
    assert numpy_space.pair_block(a, w, 4).tolist() == [0.0, 3.0, 0.0, 0.0]
    assert numpy_space.combine_vectors(a, np.ones(4), 4).tolist() == [1.0] * 3
    shaped = numpy_space.vector_block(Vector(np.ones((2, 3))), 2)     # vectors of any shape
    shaped = numpy_space.set_vector(shaped, 1, Vector(np.full((2, 3), 2.0)))
    assert numpy_space.pair_block(shaped, Covector(np.ones((2, 3))), 2).tolist() == [6.0, 12.0]
    assert numpy_space.combine_vectors(shaped, np.array([1.0, 1.0]), 2).shape == (2, 3)
