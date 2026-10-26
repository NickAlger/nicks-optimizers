"""The metrics example (examples/lbfgs_p_laplacian.py): every configuration
converges to the same energy, and the work it describes holds on two small
grids (what each configuration costs, not which is best in general)."""

import lbfgs_p_laplacian as ex


def test_the_example_runs_as_described() -> None:
    rows = {(N, c): ex.run(N, c)[0] for N in (15, 31) for c in ("Euclidean", "H1", "Kacanov", "H1, P Kacanov")}
    assert all(r.reason == "converged" for r in rows.values())
    for N in (15, 31):
        f = rows[(N, "H1")].f
        assert all(abs(rows[(N, c)].f - f) <= 1e-10 * abs(f) for c in ("Euclidean", "Kacanov", "H1, P Kacanov"))
        h1 = rows[(N, "H1")]                          # fixed metric: one solve per point, no products
        assert h1.metric_solves == h1.gradients and h1.metric_products == 0 and h1.factorizations == 1
        kac = rows[(N, "Kacanov")]                    # point_data: one factorization per point
        assert kac.factorizations == kac.gradients
        both = rows[(N, "H1, P Kacanov")]             # P's directions taken; H1 solved once per point
        assert both.precond_directions == both.iterations and both.metric_solves == both.gradients
        assert h1.iterations < rows[(N, "Euclidean")].iterations / 3
    # the H1 metric grows slowly with the grid; the Euclidean one roughly doubles
    assert rows[(31, "Euclidean")].iterations > 2 * rows[(15, "Euclidean")].iterations
    assert rows[(31, "H1")].iterations < 1.5 * rows[(15, "H1")].iterations
