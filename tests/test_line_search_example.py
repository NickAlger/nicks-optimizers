"""The comparison example (examples/line_search_vs_trust_region.py): both
optimizers converge on every problem, and the differences it describes hold
on a smaller grid."""

import line_search_vs_trust_region as ex


def test_the_example_shows_where_each_method_wins() -> None:
    rows = {(r.problem, r.method): r for r in ex.compare(N=31)}
    assert all(r.reason == "converged" for r in rows.values())

    def both(problem: str) -> tuple[ex.Row, ex.Row]:
        return rows[(problem, "trust region")], rows[(problem, "line search")]
    tr, ls = both("rosenbrock")                       # a curved valley: the trust region
    assert tr.hessvec_calls < ls.hessvec_calls and tr.iterations < ls.iterations
    tr, ls = both("p-laplacian, L^2 / H^1")           # a separate preconditioner: the line search
    assert ls.iterations < tr.iterations and ls.hessvec_calls < tr.hessvec_calls
    assert ls.metric_applies == ls.iterations < tr.metric_applies   # no Gram matrix: one per iteration
    tr, ls = both("p-laplacian, Jacobi")              # a weak preconditioner: fewer iterations
    assert ls.iterations < tr.iterations
    for problem in ("p-laplacian, H^1", "p-laplacian, L^2 / H^1", "p-laplacian, Jacobi"):
        tr, ls = both(problem)
        assert abs(tr.f - ls.f) <= 1e-10 * abs(tr.f)
