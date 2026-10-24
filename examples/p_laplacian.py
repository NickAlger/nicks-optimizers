"""The p-Laplacian by energy minimization, with vectors distributed over MPI.

Solve  -div(|grad u|^{p-2} grad u) = f  on the unit disk, u = 0 on the
circle, by minimizing the (regularized) energy

    J(u) = int (1/p) (eps^2 + |grad u|^2)^{p/2} dx - <b, u>

with linear finite elements. J is convex for every p > 1, and p is a dial:
p = 2 is the Poisson problem (one Newton step), and the problem gets harder
as p moves away from 2 (the Hessian degenerates where grad u = 0 for p > 2
and blows up there for p < 2; eps keeps it finite).

Mesh. Rings of nodes, coarse at the center and about 4 times finer near the
circle, triangulated by Delaunay: element areas vary by a factor of 16, so the
lumped mass matrix is far from a multiple of the identity.

Exact discrete solution. The right-hand side is b = grad E(u*), where E is the
gradient part of J and u* a smooth function vanishing on the circle. Then the
nodal values of u* minimize the discrete J exactly, and the error of the
optimizer can be measured down to rounding, free of discretization error.

Metric and preconditioner. The metric M, which shapes the trust region and
measures the gradient in the stopping test, is the lumped mass matrix,
||u||_M^2 = sum_i m_i u_i^2: the L^2 norm, mesh-independent. It is given by
metric_solve (w / m) and metric_apply (m v). The preconditioner P is
different: block Jacobi over the MPI ranks, each block being the rank's
diagonal block of the Hessian at the current point, applied by one algebraic
multigrid V-cycle (pyamg). The Hessian and the multigrid hierarchy are built
once per iterate through operators_at. With --precond jacobi, P is the
diagonal of the Hessian instead, which does not depend on the partition, so
runs on any number of ranks follow the same iterates (up to the order of
floating-point sums). With --metric precond, the preconditioner is the
metric (M = P): the conventional trust region, solved by truncated CG, where
only P^{-1} is ever applied. --precond laplace takes a fixed P, multigrid for
the stiffness matrix.

Distribution. The unknowns are the interior nodes, split into vertical slabs
by x coordinate, one slab per rank; a vector is the array of a rank's own
nodal values. Each rank evaluates every element touching its nodes, so it
needs the values of a layer of ghost nodes owned by its neighbors, received
by a halo exchange, and it computes complete gradient and Hessian rows for
its own nodes without sending anything back. Reductions (pairings and the
energy) are completed with MPI Allreduce, so all ranks see the
same numbers and the optimizer runs in lockstep. For simplicity every rank
builds the whole mesh before keeping its part.

What to expect (from u = 0, eps = 1e-3, 2 ranks, 20/40/80 rings, i.e. about
2k/8k/32k unknowns, stopping at ||g||_{M^{-1}} <= 1e-10 ||g(0)||_{M^{-1}}).
Every run reaches the exact discrete solution to 3e-9 or better.

  p = 1.5   9 / 9 / 9 Newton iterations
  p = 3     16 / 18 / 23
  p = 5     43 / 89 / 120

Near u = 0 the Hessian of the p = 5 energy is eps^3 times a Laplacian, so
Newton steps are huge and the method spends most iterations on the
trust-region boundary. Those iterations grow with refinement because the L^2
ball admits rough steps, cheap in L^2 but expensive for an energy that grows
like |grad s|^5. In a serial experiment, an H^1 ball (stiffness matrix, exact
dual by sparse LU) took 23 / 26 / 42 iterations instead of 62 / 91 / 176.
Block Jacobi has no coarse space, so Krylov steps per solve grow with the
number of ranks.

Run:   mpiexec -n 4 python examples/p_laplacian.py --p 1.5 3 5
       python examples/p_laplacian.py --help
"""

import argparse
import json
import time
from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np
import numpy.typing as npt
import pyamg
import scipy.sparse
from mpi4py import MPI
from scipy.spatial import Delaunay

from nicks_optimizers import (
    Array,
    Covector,
    NumpySpace,
    Point,
    PointOperators,
    RunSummary,
    Vector,
    numpy_space_with_reductions,
    trust_region_newton,
)


IntArray = npt.NDArray[np.intp]
BoolArray = npt.NDArray[np.bool_]
Operators = PointOperators[Vector, Covector]


# --- mesh --------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Mesh:
    """A triangle mesh: node coordinates, triangles (node indices) and which
    nodes lie on the boundary."""
    points: Array
    triangles: IntArray
    boundary: BoolArray


def graded_disk_mesh(n_rings: int, grading: float = 0.6, seed: int = 0) -> Mesh:
    """Rings of nodes on the unit disk at radii r(t) = (1 - grading) t +
    grading (1 - (1 - t)^2), t = k / n_rings, so the ring spacing shrinks from
    (1 + grading) / n_rings at the center to (1 - grading) / n_rings at the
    circle. Each ring gets nodes at about the local ring spacing, with a
    random angular offset, and the nodes are triangulated by Delaunay.
    Deterministic for a given seed, so every rank builds the same mesh."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_rings + 1)
    r = (1.0 - grading) * t + grading * (1.0 - (1.0 - t) ** 2)
    spacing = np.diff(r)
    points = [np.zeros((1, 2))]
    for k in range(1, n_rings + 1):
        h = spacing[k - 1] if k == n_rings else 0.5 * (spacing[k - 1] + spacing[k])
        count = max(6, int(round(2.0 * np.pi * r[k] / h)))
        theta = 2.0 * np.pi * (np.arange(count) + rng.random()) / count
        points.append(r[k] * np.column_stack([np.cos(theta), np.sin(theta)]))
    xy = np.vstack(points)
    boundary = np.zeros(xy.shape[0], dtype=bool)
    boundary[-points[-1].shape[0]:] = True
    triangles: IntArray = Delaunay(xy).simplices.astype(np.intp)
    return Mesh(points=xy, triangles=triangles, boundary=boundary)


def hat_gradients(points: Array, triangles: IntArray) -> tuple[Array, Array]:
    """Gradients of the three hat functions on each triangle (n x 3 x 2) and
    the triangle areas."""
    p0, p1, p2 = (points[triangles[:, j]] for j in range(3))
    J = np.stack([p1 - p0, p2 - p0], axis=2)                 # x = p0 + J xi
    reference = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    grads: Array = reference @ np.linalg.inv(J)
    area: Array = 0.5 * np.abs(np.linalg.det(J))
    return grads, area


# --- this rank's part of the problem -----------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Local:
    """What one rank keeps.

    Unknowns are the interior nodes. The rank owns n_owned of them; its
    extended array holds the owned values, then the ghosts, then one zero
    slot standing for every boundary node. Elements are all triangles
    touching an owned node, with vertices as indices into the extended
    array; count_energy marks the ones whose energy this rank adds (each
    element is counted by exactly one rank). send[q] lists the owned entries
    rank q needs, recv[q] the ghost entries received from q.
    """
    comm: MPI.Comm
    n_owned: int
    n_ghost: int
    owned: IntArray                    # global unknown numbers
    xy: Array                          # coordinates of the owned nodes
    vertices: IntArray                 # elements x 3, into the extended array
    grads: Array                       # elements x 3 x 2
    area: Array
    count_energy: BoolArray
    mass: Array                        # lumped mass of the owned nodes
    send: dict[int, IntArray]
    recv: dict[int, IntArray]
    n_unknowns: int                    # global


def partition(mesh: Mesh, comm: MPI.Comm) -> Local:
    """Split the interior nodes into vertical slabs (equal counts, by x) and
    set up this rank's elements, ghosts and communication lists."""
    rank, size = comm.Get_rank(), comm.Get_size()
    interior = np.flatnonzero(~mesh.boundary)
    n = interior.shape[0]
    unknown = np.full(mesh.points.shape[0], -1, dtype=np.intp)
    unknown[interior] = np.arange(n)
    owner = np.empty(n, dtype=np.intp)
    for q, chunk in enumerate(np.array_split(np.argsort(mesh.points[interior, 0], kind="stable"), size)):
        owner[chunk] = q
    owned = np.flatnonzero(owner == rank)

    tri = unknown[mesh.triangles]                            # -1 marks boundary nodes
    tri_owner = np.where(tri >= 0, owner[np.maximum(tri, 0)], -1)
    mine = np.any(tri_owner == rank, axis=1)
    tri, tri_owner = tri[mine], tri_owner[mine]
    ghosts = np.setdiff1d(np.unique(tri[tri >= 0]), owned)
    extended = np.concatenate([owned, ghosts])
    position = np.full(n, -1, dtype=np.intp)
    position[extended] = np.arange(extended.shape[0])
    zero_slot = extended.shape[0]
    vertices = np.where(tri >= 0, position[np.maximum(tri, 0)], zero_slot)

    # Each element's energy is counted by the owner of its lowest-numbered unknown.
    lowest = np.where(tri >= 0, tri, n).min(axis=1)
    count_energy = (lowest < n) & (owner[np.minimum(lowest, n - 1)] == rank)

    grads, area = hat_gradients(mesh.points, mesh.triangles[mine])
    mass = np.bincount(vertices.ravel(), weights=np.repeat(area / 3.0, 3),
                       minlength=zero_slot + 1)[:owned.shape[0]]

    # Ask each owner for the ghosts it owns; learn what the others need from us.
    wanted = [ghosts[owner[ghosts] == q] for q in range(size)]
    needed_by = comm.alltoall(wanted)
    send = {q: position[np.asarray(needed_by[q], dtype=np.intp)]
            for q in range(size) if q != rank and len(needed_by[q]) > 0}
    recv = {q: position[wanted[q]] - owned.shape[0] for q in range(size)
            if q != rank and wanted[q].shape[0] > 0}
    return Local(comm=comm, n_owned=owned.shape[0], n_ghost=ghosts.shape[0], owned=owned,
                 xy=mesh.points[interior[owned]], vertices=vertices, grads=grads, area=area,
                 count_energy=count_energy, mass=mass, send=send, recv=recv, n_unknowns=n)


def extend(loc: Local, u: Array) -> Array:
    """Owned values, then the ghost values from their owners, then 0 for the
    boundary (a halo exchange)."""
    ghost = np.empty(loc.n_ghost)
    requests = []
    buffers = {q: np.empty(idx.shape[0]) for q, idx in loc.recv.items()}
    for q in sorted(buffers):
        requests.append(loc.comm.Irecv(buffers[q], source=q, tag=7))
    for q in sorted(loc.send):
        requests.append(loc.comm.Isend(np.ascontiguousarray(u[loc.send[q]]), dest=q, tag=7))
    MPI.Request.Waitall(requests)
    for q, idx in loc.recv.items():
        ghost[idx] = buffers[q]
    return np.concatenate([u, ghost, [0.0]])


def allreduce(loc: Local, a: Array) -> Array:
    out = np.empty_like(a)
    loc.comm.Allreduce(np.ascontiguousarray(a), out, op=MPI.SUM)
    return out


def element_gradients(loc: Local, u_ext: Array) -> Array:
    """grad u on every local element (elements x 2)."""
    g: Array = np.einsum("tj,tjd->td", u_ext[loc.vertices], loc.grads)
    return g


def assemble_rows(loc: Local, contributions: Array) -> Array:
    """Sum element contributions (elements x 3) into the owned nodes; parts
    for ghosts are dropped, since their owners compute them in full."""
    summed: Array = np.bincount(loc.vertices.ravel(), weights=contributions.ravel(),
                                minlength=loc.n_owned + loc.n_ghost + 1)[:loc.n_owned]
    return summed


# --- the energy ----------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Energy:
    """W(g) = (1/p) (eps^2 + |g|^2)^{p/2} and the load b (a covector)."""
    p: float
    eps: float
    b: Array


def energy_density(e: Energy, g: Array) -> tuple[Array, Array, Array]:
    """W, and the coefficients of dW = a g and d2W = a I + c g g^T."""
    s = e.eps ** 2 + np.sum(g * g, axis=1)
    return s ** (e.p / 2.0) / e.p, s ** (e.p / 2.0 - 1.0), (e.p - 2.0) * s ** (e.p / 2.0 - 2.0)


def objective(loc: Local, e: Energy, u: Point) -> float:
    W, _, _ = energy_density(e, element_gradients(loc, extend(loc, u)))
    local = float(np.sum((loc.area * W)[loc.count_energy]) - e.b @ u)
    return float(allreduce(loc, np.array([local]))[0])


def gradient_rows(loc: Local, e_p: float, e_eps: float, u: Array) -> Array:
    """grad E(u) on the owned nodes, E the gradient part of J."""
    g = element_gradients(loc, extend(loc, u))
    _, a, _ = energy_density(Energy(p=e_p, eps=e_eps, b=np.zeros(0)), g)
    return assemble_rows(loc, np.einsum("tjd,td->tj", loc.grads, (loc.area * a)[:, None] * g))


def gradient(loc: Local, e: Energy, u: Point) -> Covector:
    return Covector(gradient_rows(loc, e.p, e.eps, u) - e.b)


def hessian_at(loc: Local, e: Energy, u: Point) -> tuple[Callable[[Vector], Covector], Array, Array, Array]:
    """H(u) as an operator, plus the element data (g, a, c) it uses."""
    g = element_gradients(loc, extend(loc, u))
    _, a, c = energy_density(e, g)

    def hessvec(v: Vector) -> Covector:
        G = element_gradients(loc, extend(loc, v))
        q = a[:, None] * G + (c * np.sum(g * G, axis=1))[:, None] * g
        return Covector(assemble_rows(loc, np.einsum("tjd,td->tj", loc.grads, loc.area[:, None] * q)))

    return hessvec, g, a, c


def owned_block(loc: Local, g: Array, a: Array, c: Array) -> scipy.sparse.csr_matrix:
    """The owned x owned block of the Hessian (exact: every element touching
    an owned node is local)."""
    Bg = np.einsum("tjd,td->tj", loc.grads, g)                       # grad phi_j . g
    K = (loc.area * a)[:, None, None] * np.einsum("tjd,tkd->tjk", loc.grads, loc.grads) \
        + (loc.area * c)[:, None, None] * Bg[:, :, None] * Bg[:, None, :]
    rows = np.repeat(loc.vertices, 3, axis=1).ravel()
    cols = np.tile(loc.vertices, (1, 3)).ravel()
    keep = (rows < loc.n_owned) & (cols < loc.n_owned)
    block: scipy.sparse.csr_matrix = scipy.sparse.coo_matrix(
        (K.ravel()[keep], (rows[keep], cols[keep])), shape=(loc.n_owned, loc.n_owned)).tocsr()
    return block


def multigrid_cycle(A: scipy.sparse.csr_matrix) -> Callable[[Covector], Vector]:
    """One smoothed-aggregation V-cycle for A (symmetric positive definite).

    'local' weighting of the prolongation smoother uses a Gershgorin bound
    instead of pyamg's default spectral radius estimate, which starts from an
    unseeded random vector and makes runs irreproducible: with p = 5 the
    iteration count varied from 49 to 63 between identical runs."""
    cycle = pyamg.smoothed_aggregation_solver(
        A, symmetry="symmetric", smooth=("jacobi", {"weighting": "local"})).aspreconditioner(cycle="V")
    return lambda w: Vector(np.asarray(cycle @ w, dtype=float))


def laplacian_cycle(loc: Local) -> Callable[[Covector], Vector]:
    """Block Jacobi multigrid for the stiffness matrix (the Hessian at p = 2),
    which does not depend on the point."""
    n = loc.vertices.shape[0]
    return multigrid_cycle(owned_block(loc, np.zeros((n, 2)), np.ones(n), np.zeros(n)))


def preconditioner_at(loc: Local, precond: str, g: Array, a: Array, c: Array,
                      fixed: Optional[Callable[[Covector], Vector]] = None
                      ) -> Callable[[Covector], Vector]:
    """P^{-1}: block Jacobi multigrid of the Hessian with element data
    (g, a, c), its diagonal ("jacobi"), or the fixed operator ("laplace")."""
    if precond == "laplace":
        assert fixed is not None
        return fixed
    A = owned_block(loc, g, a, c)
    if precond == "jacobi":
        diagonal = A.diagonal()
        return lambda w: Vector(w / diagonal)
    return multigrid_cycle(A)


def operators_at(loc: Local, e: Energy, precond: str, u: Point, metric: str = "l2",
                 fixed: Optional[Callable[[Covector], Vector]] = None) -> Operators:
    """All the operators at u, set up once per point: H(u) from the Hessian's
    element data, and P(u)^{-1} (for block Jacobi, the owned block and its
    multigrid hierarchy). With metric "l2", M is the lumped mass matrix and P
    is a separate preconditioner; with metric "precond", P is the metric
    itself (M = P)."""
    hessvec, g, a, c = hessian_at(loc, e, u)
    P_solve = preconditioner_at(loc, precond, g, a, c, fixed)
    if metric == "precond":
        return PointOperators(hessvec=hessvec, metric_solve=P_solve)
    return PointOperators(hessvec=hessvec, metric_solve=lambda w: l2_metric_solve(loc, w),
                          metric_apply=lambda v: l2_metric_apply(loc, v), precond_solve=P_solve)


# --- the space and the L^2 metric --------------------------------------------------------


def distributed_space(loc: Local) -> NumpySpace:
    return numpy_space_with_reductions(lambda a: allreduce(loc, a))


def l2_metric_solve(loc: Local, w: Covector) -> Vector:
    """M^{-1} w with M the lumped mass matrix: local, the pairing reduces."""
    return Vector(w / loc.mass)


def l2_metric_apply(loc: Local, v: Vector) -> Covector:
    """M v with M the lumped mass matrix."""
    return Covector(loc.mass * v)


# --- the exact solution ----------------------------------------------------------------


def exact_solution(xy: Array) -> Array:
    """A smooth, non-radial function vanishing on the unit circle."""
    x, y = xy[:, 0], xy[:, 1]
    u: Array = (1.0 - x ** 2 - y ** 2) * (1.0 + 0.5 * x + 0.5 * x * y)
    return u


def manufactured_energy(loc: Local, p: float, eps: float) -> tuple[Energy, Array]:
    """The energy whose exact discrete minimizer is u*: b = grad E(u*)."""
    u_star = exact_solution(loc.xy)
    return Energy(p=p, eps=eps, b=gradient_rows(loc, p, eps, u_star)), u_star


def l2_error(loc: Local, u: Array, u_star: Array) -> float:
    d = allreduce(loc, np.array([loc.mass @ (u - u_star) ** 2, loc.mass @ u_star ** 2]))
    return float(np.sqrt(d[0] / d[1]))


# --- driver ------------------------------------------------------------------------


def solve(loc: Local, e: Energy, precond: str, metric: str = "l2", display: int = 0,
          max_iter: int = 200, max_krylov: int = 100,
          krylov: Literal["auto", "cg", "lanczos", "flexible"] = "auto") -> tuple[Point, RunSummary]:
    """Minimize J from u = 0. metric = "l2" (lumped mass, with P separate:
    Lanczos) or "precond" (M = P, solved by truncated CG unless
    krylov = "lanczos")."""
    root = loc.comm.Get_rank() == 0
    fixed = laplacian_cycle(loc) if precond == "laplace" else None
    return trust_region_newton(
        lambda u: objective(loc, e, u),
        lambda u: gradient(loc, e, u),
        None,
        Point(np.zeros(loc.n_owned)),
        space=distributed_space(loc), retract=lambda u, s: Point(u + s),
        operators_at=lambda u: operators_at(loc, e, precond, u, metric, fixed),
        rtol=1e-10, max_iter=max_iter, max_krylov=max_krylov, krylov=krylov,
        display=display, print_fn=print if root else (lambda line: None))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--p", type=float, nargs="+", default=[1.5, 3.0, 5.0])
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--rings", type=int, default=40, help="mesh resolution")
    parser.add_argument("--precond", choices=["amg", "jacobi", "laplace"], default="amg")
    parser.add_argument("--metric", choices=["l2", "precond"], default="l2",
                        help="L^2 (lumped mass), or the preconditioner itself (M = P)")
    parser.add_argument("--krylov", choices=["auto", "cg", "lanczos", "flexible"], default="auto",
                        help="subproblem solver (auto: CG for --metric precond, else Lanczos; "
                             "flexible needs --metric l2)")
    parser.add_argument("--display", type=int, default=0)
    parser.add_argument("--json", help="write the iteration histories here (rank 0)")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    root = comm.Get_rank() == 0
    mesh = graded_disk_mesh(args.rings)
    loc = partition(mesh, comm)
    counts = comm.gather(loc.n_owned)
    ghosts = comm.gather(loc.n_ghost)
    if root:
        _, area = hat_gradients(mesh.points, mesh.triangles)
        print(f"mesh: {mesh.points.shape[0]} nodes, {mesh.triangles.shape[0]} triangles, "
              f"element areas {area.min():.1e} to {area.max():.1e}; {loc.n_unknowns} unknowns")
        print(f"{comm.Get_size()} ranks: owned nodes {counts}, ghost nodes {ghosts}")
        metric = "L^2 (lumped mass)" if args.metric == "l2" else "the preconditioner (M = P)"
        print(f"metric: {metric}; preconditioner: {args.precond}; eps = {args.eps:g}\n")
        print("    p  iterations  accepted  Krylov steps  ||u - u*|| / ||u*||    time")
    histories = {}
    for p in args.p:
        e, u_star = manufactured_energy(loc, p, args.eps)
        start = time.perf_counter()
        u, summary = solve(loc, e, args.precond, args.metric, display=args.display, krylov=args.krylov)
        elapsed = comm.allreduce(time.perf_counter() - start, op=MPI.MAX)
        error = l2_error(loc, u, u_star)
        krylov_steps = sum(r.subproblem.krylov_steps for r in summary.iterations)
        if root:
            print(f"{p:5.2f}  {len(summary.iterations):10d}  {summary.accepted_steps:8d}  "
                  f"{krylov_steps:12d}  {error:20.2e}  {elapsed:6.1f}s  ({summary.reason})")
        histories[str(p)] = {"error": error, "records": [
            {"f": r.f, "gnorm": r.gnorm, "radius": r.radius, "rho": r.rho, "accepted": r.accepted,
             "krylov_steps": r.subproblem.krylov_steps} for r in summary.iterations]}
    if root and args.json:
        with open(args.json, "w") as fh:
            json.dump(histories, fh)


if __name__ == "__main__":
    main()
