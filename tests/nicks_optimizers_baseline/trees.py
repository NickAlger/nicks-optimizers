"""Tuple trees: nested tuples whose leaves are objects of any other type.

A tuple tree is a leaf, or a tuple of tuple trees. Anything that is not a
tuple is a leaf, so leaves must not themselves be tuples (or lists). Only
tuples make trees: the functions here raise on a list, and as_tuple_tree
converts lists, so convert user input once (x0 = as_tuple_tree(x0)) and
have your functions return tuples. Named tuples are tuples, and come back
from these functions as plain tuples.

  as_tuple_tree(t)         lists to tuples, recursively (leaves not copied)
  tree_map(f, t)           f applied to every leaf
  tree_map2(f, a, b)       f applied to corresponding leaves of two trees
  tree_leaves(t)           the leaves, in order
  tree_leaf_pairs(a, b)    the corresponding leaves of two trees, in order

Two trees must have the same structure wherever they are combined; a
mismatch raises. These functions carry the vector space of tuple trees
(spaces.tuple_tree_space) and direct sums of leaf operators
(metrics.direct_sum).
"""

from typing import Any, Callable, Iterator, TypeVar, Union


LeafT = TypeVar("LeafT")
Tree = Union[LeafT, tuple["Tree[LeafT]", ...]]
"""A tuple tree with leaves of type LeafT."""


def as_tuple_tree(tree: Any) -> Any:
    """The same tree with every list turned into a tuple (recursively).
    Leaves are returned as they are, not copied."""
    if isinstance(tree, (tuple, list)):
        return tuple(as_tuple_tree(child) for child in tree)
    return tree


def _refuse_list(tree: Any) -> None:
    if isinstance(tree, list):
        raise TypeError("a list in a tuple tree: only tuples make trees "
                        "(convert with as_tuple_tree)")


def _mismatch() -> ValueError:
    return ValueError("tuple trees with different structures")


def tree_map(f: Callable[[Any], Any], tree: Any) -> Any:
    """f applied to every leaf; the result has the same structure."""
    if isinstance(tree, tuple):
        return tuple(tree_map(f, child) for child in tree)
    _refuse_list(tree)
    return f(tree)


def tree_map2(f: Callable[[Any, Any], Any], a: Any, b: Any) -> Any:
    """f applied to corresponding leaves of two trees of the same structure."""
    if isinstance(a, tuple):
        if not (isinstance(b, tuple) and len(a) == len(b)):
            _refuse_list(b)
            raise _mismatch()
        return tuple(tree_map2(f, x, y) for x, y in zip(a, b))
    _refuse_list(a)
    _refuse_list(b)
    if isinstance(b, tuple):
        raise _mismatch()
    return f(a, b)


def tree_leaves(tree: Any) -> Iterator[Any]:
    """The leaves of a tree, in order."""
    if isinstance(tree, tuple):
        for child in tree:
            yield from tree_leaves(child)
        return
    _refuse_list(tree)
    yield tree


def tree_leaf_pairs(a: Any, b: Any) -> Iterator[tuple[Any, Any]]:
    """The corresponding leaves of two trees of the same structure, in order."""
    if isinstance(a, tuple):
        if not (isinstance(b, tuple) and len(a) == len(b)):
            _refuse_list(b)
            raise _mismatch()
        for x, y in zip(a, b):
            yield from tree_leaf_pairs(x, y)
        return
    _refuse_list(a)
    _refuse_list(b)
    if isinstance(b, tuple):
        raise _mismatch()
    yield a, b
