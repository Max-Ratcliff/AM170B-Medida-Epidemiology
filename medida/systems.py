import numpy as np
from abc import ABC, abstractmethod
from .integrators import integrate


class DynamicalSystem(ABC):
    """Abstract base class for dynamical systems.

    Provides interfaces for evaluating the right-hand side (RHS), stepping
    the system forward in time, and integrating trajectories.
    """

    @abstractmethod
    def rhs(self, u):
        """Evaluate the system tendencies at state u."""
        pass

    def __call__(self, u):
        return self.rhs(u)

    def step(self, u, dt, method="rk4", substeps=1):
        """Advance the system state by a time interval dt.

        Args:
            u (array-like): Current state of the system.
            dt (float): Time interval to advance.
            method (str): Numerical integration method ('rk4' or 'euler').
            substeps (int): Number of internal integration steps per dt.

        Returns:
            np.ndarray: The advanced state.
        """
        return integrate(self.rhs, u, dt, 1, method=method, substeps=substeps)[
            -1
        ]

    def trajectory(self, u0, dt, n_steps, method="rk4", substeps=1):
        """Integrate the system from u0 and return the full trajectory.

        Args:
            u0 (array-like): Initial state of the system.
            dt (float): Time interval between returned trajectory points.
            n_steps (int): Number of time steps to integrate.
            method (str): Numerical integration method ('rk4' or 'euler').
            substeps (int): Number of internal integration steps per dt.

        Returns:
            np.ndarray: Trajectory of shape (n_steps + 1, ...).
        """
        u = np.asarray(u0, dtype=float)
        out = np.empty((n_steps + 1,) + u.shape, dtype=float)
        out[0] = u
        for i in range(n_steps):
            u = self.step(u, dt, method=method, substeps=substeps)
            out[i + 1] = u
        return out

    def coefficients(self, library, n_probe=4000, probe_scale=12.0, seed=0):
        """Express the RHS as a coefficient matrix in the provided library.

        Uses a Monte Carlo approach to probe the system dynamics and solve
        a least-squares problem to map the RHS onto library features.

        Args:
            library (FeatureLibrary): The library to project the RHS onto.
            n_probe (int): Number of random states used to probe the system.
            probe_scale (float): Standard deviation of the probing states.
            seed (int): Random seed for generating probing states.

        Returns:
            np.ndarray: The coefficient matrix of shape (n_features, dim).
        """
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
    """Dynamical system defined by a polynomial coefficient matrix.

    Typically used for discovered corrections or baseline ODE models.
    """

    def __init__(self, coefficients, library, state_names=None):
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.library = library
        self.state_names = state_names
        self._etdrk4_cache = {}
        if self.coefficients.ndim == 1:
            self.dim = 1
        else:
            self.dim = self.coefficients.shape[1]

    def rhs(self, u):
        """Evaluate the polynomial RHS using the feature library."""
        u_in = np.asarray(u, dtype=float)
        # PDELibrary.transform handles both (n_grid,) and (n_samples, n_grid)
        phi = self.library.transform(u_in)
        res = phi @ self.coefficients

        # If it's a scalar PDE, reshape the flattened grid points back to the input shape
        if getattr(self.library, "kind", None) == "scalar_pde":
            return res.reshape(u_in.shape)

        # For ODEs
        if self.dim == 1:
            return res.reshape(u_in.shape)

        # Multivariable ODEs (dim > 1)
        if u_in.ndim == 1:
            return res.squeeze()  # (1, dim) -> (dim,)
        return res  # (n_samples, dim)

    def _get_etdrk4_coeffs(self, dt):
        """Compute and cache ETDRK4 operators for the linear spectral part.

        Identifies purely linear terms in the library to build the diagonal
        linear operator Llin in Fourier space.
        """
        if dt in self._etdrk4_cache:
            return self._etdrk4_cache[dt]
        lib = self.library
        k = lib.k
        c = self.coefficients
        Llin = np.zeros(lib.n_grid, dtype=complex)
        for idx, term in enumerate(lib.terms):
            if term == ("bias", "bias"):
                continue
            r_t, s_t = term
            if r_t == 0:
                # Linear term: contributes c * (ik)^s to the spectral operator
                Llin += float(c[idx]) * (1j * k) ** s_t

        # Cox-Matthews contour-integral quadrature for stiff coefficients
        E = np.exp(dt * Llin)
        E2 = np.exp(dt * Llin / 2.0)
        M = 32
        r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
        LR = dt * Llin[:, None] + r[None, :]
        Q = dt * np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1)
        f1 = dt * np.mean(
            (-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / LR**3, axis=1
        )
        f2 = dt * np.mean(
            (2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR**3, axis=1
        )
        f3 = dt * np.mean(
            (-4.0 - 3.0 * LR - LR**2 + np.exp(LR) * (4.0 - LR)) / LR**3, axis=1
        )
        coeffs = (E, E2, Q, f1, f2, f3, Llin)
        self._etdrk4_cache[dt] = coeffs
        return coeffs

    def _etdrk4_step(self, u, dt):
        """Perform one ETDRK4 step for stiff 1-D scalar PDE systems."""
        lib = self.library
        E, E2, Q, f1, f2, f3, Llin = self._get_etdrk4_coeffs(dt)

        def Nfun(vhat):
            # Nonlinear residual: full system tendencies minus the linear part
            u_r = np.real(np.fft.ifft(vhat, axis=-1))
            rhs_r = self.rhs(u_r)
            return np.fft.fft(rhs_r, axis=-1) - Llin * vhat

        vhat = np.fft.fft(u, axis=-1)
        Nv = Nfun(vhat)
        a = E2 * vhat + Q * Nv
        Na = Nfun(a)
        b = E2 * vhat + Q * Na
        Nb = Nfun(b)
        cv = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = Nfun(cv)
        vhat = E * vhat + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3
        return np.real(np.fft.ifft(vhat, axis=-1))

    def step(self, u, dt, method="rk4", substeps=1):
        """Advance the state. Routes to ETDRK4 if the system is a stiff PDE."""
        u_arr = np.asarray(u, dtype=float)
        if getattr(self.library, "kind", None) == "scalar_pde":
            return self._etdrk4_step(u_arr, dt)
        return super().step(u, dt, method=method, substeps=substeps)


class Lorenz63(DynamicalSystem):
    """The classical Lorenz-63 chaotic system."""

    dim = 3
    state_names = ("x", "y", "z")

    def __init__(self, sigma=10.0, rho=28.0, beta=8 / 3):
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
    """Kuramoto-Sivashinsky system on a periodic domain.

    Integrated using the Exponential Time-Differencing Runge-Kutta 4 (ETDRK4)
    scheme to handle high-order stiff derivatives.
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
        """Compute spatial derivatives in Fourier space."""
        fhat = np.fft.fft(u, axis=-1) * (1j * self.k) ** order
        return np.real(np.fft.ifft(fhat, axis=-1))

    def rhs(self, u):
        u = np.asarray(u, dtype=float)
        ux = self._spec_deriv(u, 1)
        uxx = self._spec_deriv(u, 2)
        uxxxx = self._spec_deriv(u, 4)
        return -self.beta * u * ux - self.nu_2 * uxx - self.nu_4 * uxxxx

    def _etdrk4_coeffs(self, dt):
        """Compute and cache ETDRK4 coefficients for the KS linear operator."""
        if dt in self._etdrk4_cache:
            return self._etdrk4_cache[dt]
        Llin = self.nu_2 * self.k**2 - self.nu_4 * self.k**4
        E = np.exp(dt * Llin)
        E2 = np.exp(dt * Llin / 2.0)
        M = 32
        r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
        LR = dt * Llin[:, None] + r[None, :]
        Q = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
        f1 = dt * np.real(
            np.mean(
                (-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / LR**3,
                axis=1,
            )
        )
        f2 = dt * np.real(
            np.mean((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR**3, axis=1)
        )
        f3 = dt * np.real(
            np.mean(
                (-4.0 - 3.0 * LR - LR**2 + np.exp(LR) * (4.0 - LR)) / LR**3,
                axis=1,
            )
        )
        g = -0.5j * self.k * self.beta
        coeffs = (E, E2, Q, f1, f2, f3, g)
        self._etdrk4_cache[dt] = coeffs
        return coeffs

    def step(self, u, dt, **_):
        """Step forward using ETDRK4 for stability."""
        E, E2, Q, f1, f2, f3, g = self._etdrk4_coeffs(dt)
        v = np.fft.fft(u, axis=-1)

        def Nfun(v):
            ur = np.real(np.fft.ifft(v, axis=-1))
            return g * np.fft.fft(ur * ur, axis=-1)

        Nv = Nfun(v)
        a = E2 * v + Q * Nv
        Na = Nfun(a)
        b = E2 * v + Q * Na
        Nb = Nfun(b)
        c = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = Nfun(c)
        v = E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3
        return np.real(np.fft.ifft(v, axis=-1))


class SIRSystem(DynamicalSystem):
    """Standard Susceptible-Infectious-Recovered (SIR) epidemic model."""

    dim = 3
    state_names = ("S", "I", "R")

    def __init__(self, beta=0.6, gamma=0.18):
        self.beta = float(beta)
        self.gamma = float(gamma)

    def rhs(self, u):
        u = np.asarray(u, dtype=float)
        S, I, R = u[..., 0], u[..., 1], u[..., 2]
        dS = -self.beta * S * I
        dI = self.beta * S * I - self.gamma * I
        dR = self.gamma * I
        return np.stack([dS, dI, dR], axis=-1)


class SIRSSystem(DynamicalSystem):
    """SIR model with waning immunity (R -> S)."""

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
        dI = self.beta * S * I - self.gamma * I
        dR = self.gamma * I - self.xi * R
        return np.stack([dS, dI, dR], axis=-1)


class SIRDSystem(DynamicalSystem):
    """SIR model with an explicit Death compartment (D)."""

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
        dI = self.beta * S * I - (self.gamma + self.mu) * I
        dR = self.gamma * I
        dD = self.mu * I
        return np.stack([dS, dI, dR, dD], axis=-1)


class SEIRSystem(DynamicalSystem):
    """SIR model with an explicit latent period (Exposed compartment E)."""

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
        dE = self.beta * S * I - self.sigma * E
        dI = self.sigma * E - self.gamma * I
        dR = self.gamma * I
        return np.stack([dS, dE, dI, dR], axis=-1)


class SIRNonlinearSystem(DynamicalSystem):
    """SIR model with saturated incidence (nonlinear coupling)."""

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
        dI = incidence - self.gamma * I
        dR = self.gamma * I
        return np.stack([dS, dI, dR], axis=-1)


class ProjectedSIRFromSEIRSystem(DynamicalSystem):
    """3D SIR system modeling SEIR dynamics via a latent ratio assumption."""

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
        dI = self.sigma * E - self.gamma * I
        dR = self.gamma * I
        return np.stack([dS, dI, dR], axis=-1)


def make_ks_model(beta=1.0, nu_2=1.0, nu_4=1.0, L=22.0, N=64):
    """Build a Kuramoto-Sivashinsky system instance."""
    return KSSystem(L=L, N=N, beta=beta, nu_2=nu_2, nu_4=nu_4)


def make_sir_model(beta=0.6, gamma=0.18):
    """Build a standard SIR system instance."""
    return SIRSystem(beta=beta, gamma=gamma)
