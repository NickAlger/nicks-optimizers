"""The switching study's problems P2 and P3 (Nick Alger and Claude, November
2026; artifact V9rpRsqpod3aSuBVNbxhLo, code/diffusion_inversion.py and
code/matrix_completion.py), ported to the library's interface: fun, grad,
gn_hessvec and hessvec as functions, with the same data, seeds and
operators. P1 (regularized extended Rosenbrock) is tests/problems.py's
regularized_rosenbrock.

Differences from the study's code: the library computes f(x + s) - f(x) by
subtraction (the study had cancellation-free differences), and the full
Hessian-vector product of the diffusion problem shares the state increment
between its Gauss-Newton and curvature parts (3 PDE solves, as the study's
accounting assumed; its code spent 4).
"""

from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class _CountingLU:
    """splu wrapper that counts triangular solves (the dominant cost)."""
    def __init__(self, lu: Any, owner: Any) -> None:
        self.lu, self.owner = lu, owner

    def solve(self, b: Any) -> Any:
        self.owner.nsolve += 1
        return self.lu.solve(b)


class DiffusionInversion:
    """P2: -div(exp(m) grad u) = 1 on (0,1)^2, u = 0 on the boundary, P1
    triangles on an N x N grid of cells, m piecewise constant per cell;
    J(m) = 1/2 ||B u(m) - d||^2 + beta/2 m^T R m. nsolve counts PDE solves
    (the state, its adjoint, and those of the Hessian products); a new
    point costs a factorization and two solves, a GN product two solves, a
    full product three."""

    def __init__(self, N: int = 32, beta: float = 1e-5, n_obs: int = 150, noise: float = 0.01,
                 seed: int = 0, contrast: float = 3.0) -> None:
        self.N, self.beta = N, beta
        h = 1.0 / N
        nn = (N + 1) ** 2

        def node(i: Any, j: Any) -> Any:
            return i * (N + 1) + j
        ci, cj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        ci, cj = ci.ravel(), cj.ravel()
        c = np.arange(N * N)
        n00, n10, n01, n11 = node(ci, cj), node(ci + 1, cj), node(ci, cj + 1), node(ci + 1, cj + 1)

        def rows(r: Any, cols: Any, vals: Any) -> Any:
            return sp.coo_matrix((np.concatenate(vals), (np.concatenate(r), np.concatenate(cols))),
                                 shape=(2 * N * N, nn))
        ra, rb = c, N * N + c
        one = np.ones(N * N) / h
        Gx = rows([ra, ra, rb, rb], [n00, n10, n01, n11], [-one, one, -one, one]).tocsr()
        Gy = rows([ra, ra, rb, rb], [n00, n01, n10, n11], [-one, one, -one, one]).tocsr()
        gi, gj = np.meshgrid(np.arange(N + 1), np.arange(N + 1), indexing="ij")
        interior = ((gi > 0) & (gi < N) & (gj > 0) & (gj < N)).ravel()
        self.Gx, self.Gy = Gx[:, interior], Gy[:, interior]
        self.nu = int(interior.sum())
        self.P = sp.vstack([sp.identity(N * N), sp.identity(N * N)]).tocsr()
        self.a = np.full(2 * N * N, 0.5 * h * h)
        self.n = N * N
        e = np.ones(N)
        D1 = sp.diags([-e[:-1], e[:-1]], [0, 1], shape=(N - 1, N)) / h
        eye = sp.identity(N)
        self.R = (sp.kron(D1.T @ D1, eye) + sp.kron(eye, D1.T @ D1)
                  + 1e-6 * sp.identity(N * N)).tocsr() * h * h
        self.b = np.full(self.nu, h * h)
        rng = np.random.default_rng(seed)
        self.obs = rng.choice(self.nu, size=n_obs, replace=False)
        self.B = sp.csr_matrix((np.ones(n_obs), (np.arange(n_obs), self.obs)), shape=(n_obs, self.nu))
        xc, yc = (ci + 0.5) * h, (cj + 0.5) * h
        m_true = contrast * (np.exp(-((xc - .35) ** 2 + (yc - .65) ** 2) / .03)
                             - 0.8 * np.exp(-((xc - .7) ** 2 + (yc - .3) ** 2) / .04))
        self.m_true = m_true
        clean = self.B @ spla.splu(self._D(np.exp(m_true)).tocsc()).solve(self.b)
        self.sigma = noise * np.abs(clean).max()
        self.d = clean + self.sigma * rng.standard_normal(n_obs)
        self._x: Any = None
        self._st: Any = None
        self.nsolve = 0

    def _D(self, q: Any) -> Any:
        w = self.a * (self.P @ q)
        return self.Gx.T @ sp.diags(w) @ self.Gx + self.Gy.T @ sp.diags(w) @ self.Gy

    def _E(self, lam: Any, u: Any) -> Any:
        t = self.a * ((self.Gx @ lam) * (self.Gx @ u) + (self.Gy @ lam) * (self.Gy @ u))
        return self.P.T @ t

    def _state(self, x: Any) -> Any:
        if self._x is not None and np.array_equal(x, self._x):
            return self._st
        with np.errstate(over="ignore"):
            kappa = np.exp(x)
        try:
            if not np.all(np.isfinite(kappa)):
                raise RuntimeError("conductivity overflow")
            lu = _CountingLU(spla.splu(self._D(kappa).tocsc()), self)
        except RuntimeError:              # a wild line-search trial: f = inf rejects it
            self._x, self._st = x.copy(), None
            return None
        u = lu.solve(self.b)
        rho = self.B @ u - self.d
        lam = lu.solve(self.B.T @ rho)
        self._x, self._st = x.copy(), dict(kappa=kappa, lu=lu, u=u, rho=rho, lam=lam)
        return self._st

    def fun(self, x: Any) -> float:
        st = self._state(x)
        if st is None:
            return float("inf")
        return float(0.5 * st["rho"] @ st["rho"] + 0.5 * self.beta * x @ (self.R @ x))

    def grad(self, x: Any) -> Any:
        st = self._state(x)
        return -st["kappa"] * self._E(st["lam"], st["u"]) + self.beta * (self.R @ x)

    def gn_hessvec(self, x: Any, v: Any) -> Any:
        st = self._state(x)
        du = -st["lu"].solve(self._D(st["kappa"] * v) @ st["u"])
        mu = st["lu"].solve(self.B.T @ (self.B @ du))
        return -st["kappa"] * self._E(mu, st["u"]) + self.beta * (self.R @ v)

    def hessvec(self, x: Any, v: Any) -> Any:
        st = self._state(x)
        k, u, lam, lu = st["kappa"], st["u"], st["lam"], st["lu"]
        Dkv = self._D(k * v)
        du = -lu.solve(Dkv @ u)
        mu = lu.solve(self.B.T @ (self.B @ du))
        w = lu.solve(Dkv @ lam)
        gn = -k * self._E(mu, u) + self.beta * (self.R @ v)
        return gn - (k * v) * self._E(lam, u) - k * self._E(lam, du) + k * self._E(w, u)

    def x0(self) -> Any:
        return np.zeros(self.n)


class MatrixCompletion:
    """P3: rank-r factorization X Y^T fitted to |Omega| noisy entries of a
    matrix with singular values sqrt(mn)/k^2, with lam/2 (||X||^2 + ||Y||^2);
    unknowns z = [vec X; vec Y]."""

    def __init__(self, m: int = 200, n: int = 200, r: int = 10, obs_frac: float = 0.25,
                 noise: float = 0.01, lam: float = 1e-3, seed: int = 0, decay: float = 2.0) -> None:
        self.m, self.n_cols, self.r, self.lam = m, n, r, lam
        self.n = (m + n) * r
        rng = np.random.default_rng(seed)
        p = min(m, n)
        U, _ = np.linalg.qr(rng.standard_normal((m, p)))
        V, _ = np.linalg.qr(rng.standard_normal((n, p)))
        self.sing = np.sqrt(m * n) / np.arange(1, p + 1) ** decay
        self.A = (U * self.sing) @ V.T
        nobs = int(obs_frac * m * n)
        idx = rng.choice(m * n, size=nobs, replace=False)
        self.I, self.J = np.divmod(idx, n)
        clean = self.A[self.I, self.J]
        self.sigma = noise * np.sqrt(np.mean(clean ** 2))
        self.d = clean + self.sigma * rng.standard_normal(nobs)
        self.seed = seed

    def split(self, z: Any) -> tuple[Any, Any]:
        k = self.m * self.r
        return z[:k].reshape(self.m, self.r), z[k:].reshape(self.n_cols, self.r)

    def join(self, X: Any, Y: Any) -> Any:
        return np.concatenate([X.ravel(), Y.ravel()])

    def _sp(self, w: Any) -> Any:
        return sp.csr_matrix((w, (self.I, self.J)), shape=(self.m, self.n_cols))

    def resid(self, z: Any) -> Any:
        X, Y = self.split(z)
        return np.einsum("ij,ij->i", X[self.I], Y[self.J]) - self.d

    def fun(self, z: Any) -> float:
        rd = self.resid(z)
        return float(0.5 * rd @ rd + 0.5 * self.lam * z @ z)

    def grad(self, z: Any) -> Any:
        X, Y = self.split(z)
        W = self._sp(self.resid(z))
        return self.join(W @ Y, W.T @ X) + self.lam * z

    def gn_hessvec(self, z: Any, v: Any) -> Any:
        X, Y = self.split(z)
        Vx, Vy = self.split(v)
        jv = np.einsum("ij,ij->i", Vx[self.I], Y[self.J]) + np.einsum("ij,ij->i", X[self.I], Vy[self.J])
        W = self._sp(jv)
        return self.join(W @ Y, W.T @ X) + self.lam * v

    def hessvec(self, z: Any, v: Any) -> Any:
        Vx, Vy = self.split(v)
        R = self._sp(self.resid(z))
        return self.gn_hessvec(z, v) + self.join(R @ Vy, R.T @ Vx)

    def x0(self, scale: float = 0.3) -> Any:
        return scale * np.random.default_rng(self.seed + 1000).standard_normal(self.n)


def check(p: Any, x: Any, rng: Any) -> tuple[float, float, float]:
    """Relative finite-difference errors of grad and hessvec, and the GN
    symmetry defect."""
    v, w, h = rng.standard_normal(x.shape[0]), rng.standard_normal(x.shape[0]), 1e-5
    g_fd = (p.fun(x + h * v) - p.fun(x - h * v)) / (2 * h)
    H_fd = (p.grad(x + h * v) - p.grad(x - h * v)) / (2 * h)
    sym = abs(w @ p.gn_hessvec(x, v) - v @ p.gn_hessvec(x, w)) / abs(w @ p.gn_hessvec(x, v))
    return (abs(g_fd - p.grad(x) @ v) / abs(g_fd),
            float(np.linalg.norm(H_fd - p.hessvec(x, v)) / np.linalg.norm(H_fd)), sym)
