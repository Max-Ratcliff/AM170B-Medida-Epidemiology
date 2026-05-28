import numpy as np
from abc import ABC, abstractmethod
from .integrators import integrate

class DynamicalSystem(ABC):
    @abstractmethod
    def rhs(self, u):
        pass

    def __call__(self, u):
        return self.rhs(u)

    def step(self, u, dt, method="rk4", substeps=1):
        """Advance one observation interval ``dt``."""
        return integrate(self.rhs, u, dt, 1, method=method, substeps=substeps)[-1]

    def trajectory(self, u0, dt, n_steps, method="rk4", substeps=1):
        """Integrate the system and return the full trajectory."""
        u = np.asarray(u0, dtype=float)
        out = np.empty((n_steps + 1,) + u.shape, dtype=float)
        out[0] = u
        for i in range(n_steps):
            u = self.step(u, dt, method=method, substeps=substeps)
            out[i + 1] = u
        return out

    def coefficients(self, library, n_probe=4000, probe_scale=12.0, seed=0):
        """Express ``rhs`` in ``library`` as a coefficient matrix."""
        rng = np.random.default_rng(seed)
        U = rng.standard_normal((n_probe, self.dim)) * probe_scale
        Phi = library.transform(U)
        target = np.asarray(self.rhs(U), dtype=float)
        if hasattr(library, "kind") and library.kind == "scalar_pde":
            target = target.reshape(-1)
        else:
            target = target.reshape(n_probe, self.dim)
        coeffs, *_ = np.linalg.lstsq(Phi, target, rcond=None)
        return coeffs

class PolynomialODE(DynamicalSystem):
    def __init__(self, coefficients, library, state_names=None):
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.library = library
        self.state_names = state_names
        self.dim = self.coefficients.shape[1]

    def rhs(self, u):
        u = np.atleast_2d(u)
        phi = self.library.transform(u)
        res = phi @ self.coefficients
        return res.squeeze()

class Lorenz63(DynamicalSystem):
    dim = 3
    state_names = ("x", "y", "z")
    def __init__(self, sigma=10.0, rho=28.0, beta=8/3):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def rhs(self, u):
        u = np.asarray(u, dtype=float)
        x, y, z = u[..., 0], u[..., 1], u[..., 2]
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return np.stack([dx, dy, dz], axis=-1)

class KSSystem(DynamicalSystem):
    """Kuramoto-Sivashinsky on [0, L) periodic, integrated with ETDRK4.
        u_t = -beta * u * u_x  -  nu_2 * u_xx  -  nu_4 * u_xxxx
    """
    state_names = ("u",)
    def __init__(self, L=22.0, N=64, beta=1.0, nu_2=1.0, nu_4=1.0):
        self.L = float(L)
        self.N = int(N)
        self.dim = self.N
        self.beta = float(beta)
        self.nu_2 = float(nu_2)
        self.nu_4 = float(nu_4)
        self.k = 2.0 * np.pi * np.fft.fftfreq(self.N, d=self.L / self.N)
        self.x = np.arange(self.N) * self.L / self.N
        self._etdrk4_cache = {}

    def _spec_deriv(self, u, order):
        fhat = np.fft.fft(u, axis=-1) * (1j * self.k) ** order
        return np.real(np.fft.ifft(fhat, axis=-1))

    def rhs(self, u):
        u = np.asarray(u, dtype=float)
        ux    = self._spec_deriv(u, 1)
        uxx   = self._spec_deriv(u, 2)
        uxxxx = self._spec_deriv(u, 4)
        return -self.beta * u * ux - self.nu_2 * uxx - self.nu_4 * uxxxx

    def _etdrk4_coeffs(self, dt):
        if dt in self._etdrk4_cache:
            return self._etdrk4_cache[dt]
        Llin = self.nu_2 * self.k ** 2 - self.nu_4 * self.k ** 4
        E = np.exp(dt * Llin)
        E2 = np.exp(dt * Llin / 2.0)
        M = 32
        r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
        LR = dt * Llin[:, None] + r[None, :]
        Q  = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
        f1 = dt * np.real(np.mean((-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR ** 2)) / LR ** 3, axis=1))
        f2 = dt * np.real(np.mean(( 2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR ** 3, axis=1))
        f3 = dt * np.real(np.mean((-4.0 - 3.0 * LR - LR ** 2 + np.exp(LR) * (4.0 - LR)) / LR ** 3, axis=1))
        g = -0.5j * self.k * self.beta
        coeffs = (E, E2, Q, f1, f2, f3, g)
        self._etdrk4_cache[dt] = coeffs
        return coeffs

    def step(self, u, dt, **_):
        E, E2, Q, f1, f2, f3, g = self._etdrk4_coeffs(dt)
        v = np.fft.fft(u, axis=-1)
        def Nfun(v):
            ur = np.real(np.fft.ifft(v, axis=-1))
            return g * np.fft.fft(ur * ur, axis=-1)
        Nv = Nfun(v)
        a = E2 * v + Q * Nv;  Na = Nfun(a)
        b = E2 * v + Q * Na;  Nb = Nfun(b)
        c = E2 * a + Q * (2.0 * Nb - Nv);  Nc = Nfun(c)
        v = E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3
        return np.real(np.fft.ifft(v, axis=-1))

class SIRSystem(DynamicalSystem):
    dim = 3
    state_names = ("S", "I", "R")
    def __init__(self, beta=0.6, gamma=0.18):
        self.beta = float(beta)
        self.gamma = float(gamma)

    def rhs(self, u):
        u = np.asarray(u, dtype=float)
        S, I, R = u[..., 0], u[..., 1], u[..., 2]
        dS = -self.beta * S * I
        dI =  self.beta * S * I - self.gamma * I
        dR =  self.gamma * I
        return np.stack([dS, dI, dR], axis=-1)

class SIRSSystem(DynamicalSystem):
    dim = 3
    state_names = ("S", "I", "R")
    def __init__(self, beta=0.6, gamma=0.18, xi=0.08):
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.xi = float(xi)

    def rhs(self, u):
        u = np.asarray(u, dtype=float)
        S, I, R = u[..., 0], u[..., 1], u[..., 2]
        dS = -self.beta * S * I + self.xi * R
        dI =  self.beta * S * I - self.gamma * I
        dR =  self.gamma * I - self.xi * R
        return np.stack([dS, dI, dR], axis=-1)

class SIRDSystem(DynamicalSystem):
    dim = 4
    state_names = ("S", "I", "R", "D")
    def __init__(self, beta=0.6, gamma=0.14, mu=0.04):
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.mu = float(mu)

    def rhs(self, u):
        u = np.asarray(u, dtype=float)
        S, I, R, D = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
        dS = -self.beta * S * I
        dI =  self.beta * S * I - (self.gamma + self.mu) * I
        dR =  self.gamma * I
        dD =  self.mu * I
        return np.stack([dS, dI, dR, dD], axis=-1)

class SEIRSystem(DynamicalSystem):
    dim = 4
    state_names = ("S", "E", "I", "R")
    def __init__(self, beta=0.6, sigma=0.2, gamma=0.18):
        self.beta = float(beta)
        self.sigma = float(sigma)
        self.gamma = float(gamma)

    def rhs(self, u):
        u = np.asarray(u, dtype=float)
        S, E, I, R = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
        dS = -self.beta * S * I
        dE =  self.beta * S * I - self.sigma * E
        dI =  self.sigma * E - self.gamma * I
        dR =  self.gamma * I
        return np.stack([dS, dE, dI, dR], axis=-1)

class SIRNonlinearSystem(DynamicalSystem):
    dim = 3
    state_names = ("S", "I", "R")
    def __init__(self, beta=0.6, gamma=0.18, a=8.0):
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.a = float(a)

    def rhs(self, u):
        u = np.asarray(u, dtype=float)
        S, I, R = u[..., 0], u[..., 1], u[..., 2]
        incidence = self.beta * S * I / (1.0 + self.a * I)
        dS = -incidence
        dI =  incidence - self.gamma * I
        dR =  self.gamma * I
        return np.stack([dS, dI, dR], axis=-1)

class ProjectedSIRFromSEIRSystem(DynamicalSystem):
    """3D SIR system that is really a projection of a 4D SEIR system."""
    dim = 3
    state_names = ("S", "I", "R")
    def __init__(self, beta=0.6, sigma=0.2, gamma=0.18, e_ratio=0.5):
        self.beta = float(beta)
        self.sigma = float(sigma)
        self.gamma = float(gamma)
        self.e_ratio = float(e_ratio)

    def rhs(self, u):
        u = np.asarray(u, dtype=float)
        S, I, R = u[..., 0], u[..., 1], u[..., 2]
        # In this projection, we assume E is proportional to I for the RHS evaluation
        E = self.e_ratio * I
        dS = -self.beta * S * I
        dI =  self.sigma * E - self.gamma * I
        dR =  self.gamma * I
        return np.stack([dS, dI, dR], axis=-1)

def make_ks_model(beta=1.0, nu_2=1.0, nu_4=1.0, L=22.0, N=64):
    """Helper to build a Kuramoto-Sivashinsky system."""
    return KSSystem(L=L, N=N, beta=beta, nu_2=nu_2, nu_4=nu_4)

def make_sir_model(beta=0.6, gamma=0.18):
    """Helper to build a standard SIR system."""
    return SIRSystem(beta=beta, gamma=gamma)
