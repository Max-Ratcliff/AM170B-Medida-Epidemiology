import numpy as np
from abc import ABC, abstractmethod
from itertools import combinations_with_replacement


class FeatureLibrary(ABC):
    """Abstract base class for feature libraries used in sparse regression.

    Subclasses must implement the transform method to map raw states into
    a feature space.
    """

    @abstractmethod
    def transform(self, states):
        """Map raw state observations into the feature space."""
        pass

    def index(self, name):
        """Find the index of a feature by its human-readable name."""
        return self.feature_names.index(name)


class PolynomialLibrary(FeatureLibrary):
    """Library consisting of multivariate polynomial features.

    Typically used for finite-dimensional Ordinary Differential Equations (ODEs).
    """

    kind = "vector_ode"

    def __init__(self, n_vars, degree=2, include_bias=True, var_names=None):
        self.n_vars = int(n_vars)
        self.degree = int(degree)
        self.include_bias = bool(include_bias)
        if var_names is None:
            var_names = [f"u{i}" for i in range(self.n_vars)]
        self.var_names = list(var_names)
        self.exponents = self._build_exponents()
        self.feature_names = [self._name(e) for e in self.exponents]
        self.n_features = len(self.exponents)

    def _build_exponents(self):
        """Generate all unique polynomial exponent combinations up to the given degree."""
        exps = []
        lowest = 0 if self.include_bias else 1
        for total in range(lowest, self.degree + 1):
            for combo in combinations_with_replacement(
                range(self.n_vars), total
            ):
                e = [0] * self.n_vars
                for idx in combo:
                    e[idx] += 1
                exps.append(tuple(e))
        return exps

    def _name(self, exponent):
        """Construct a human-readable string representation of a polynomial term."""
        if all(p == 0 for p in exponent):
            return "1"
        parts = []
        for var, p in zip(self.var_names, exponent):
            if p == 1:
                parts.append(var)
            elif p > 1:
                parts.append(f"{var}^{p}")
        return " ".join(parts)

    def transform(self, states):
        """Transform input states into the polynomial feature matrix."""
        U = np.atleast_2d(np.asarray(states, dtype=float))
        n = U.shape[0]
        Phi = np.empty((n, self.n_features), dtype=float)
        for j, exponent in enumerate(self.exponents):
            col = np.ones(n, dtype=float)
            for var, p in enumerate(exponent):
                if p:
                    col = col * U[:, var] ** p
            Phi[:, j] = col
        return Phi


class PDELibrary(FeatureLibrary):
    """Library consisting of spatial-derivative and nonlinear terms for 1-D PDEs.

    Uses spectral methods (FFT) to compute high-order spatial derivatives.
    """

    kind = "scalar_pde"

    def __init__(
        self, n_grid, length, poly_order=4, deriv_order=4, include_bias=True
    ):
        self.n_grid = int(n_grid)
        self.length = float(length)
        self.poly_order = int(poly_order)
        self.deriv_order = int(deriv_order)
        self.include_bias = bool(include_bias)
        self.k = (
            2.0
            * np.pi
            * np.fft.fftfreq(self.n_grid, d=self.length / self.n_grid)
        )
        self.terms = []
        self.feature_names = []
        if self.include_bias:
            self.terms.append(("bias", "bias"))
            self.feature_names.append("1")
        for r in range(0, self.poly_order + 1):
            for s in range(0, self.deriv_order + 1):
                self.terms.append((r, s))
                self.feature_names.append(self._name(r, s))
        self.n_features = len(self.terms)

    @staticmethod
    def _name(r, s):
        """Construct a string name for terms like u^r * u_{x...x}."""
        deriv = "u" if s == 0 else "u_" + "x" * s
        if r == 0:
            return deriv
        poly = "u" if r == 1 else f"u^{r}"
        return f"{poly} {deriv}"

    def derivative(self, field, order):
        """Compute the spatial derivative of the field using the spectral method."""
        if order == 0:
            return np.asarray(field, dtype=float)
        fhat = np.fft.fft(field, axis=-1)
        fhat = fhat * (1j * self.k) ** order
        return np.real(np.fft.ifft(fhat, axis=-1))

    def transform(self, states):
        """Build the PDE feature library matrix."""
        U = np.atleast_2d(np.asarray(states, dtype=float))
        cols = []
        for term in self.terms:
            if term == ("bias", "bias"):
                cols.append(np.ones(U.size, dtype=float))
                continue
            r, s = term
            value = self.derivative(U, s)
            if r > 0:
                value = U**r * value
            cols.append(value.reshape(-1))
        return np.stack(cols, axis=1)


class SaturatedSIRLibrary(FeatureLibrary):
    """Custom library for nonlinear incidence SIR models.

    Includes terms for linear incidence and saturated incidence (Holling Type II).
    """

    def __init__(self, a=8.0):
        self.a = float(a)
        self.feature_names = ["I", "S I", f"S I / (1 + {self.a:g} I)"]
        self.n_features = len(self.feature_names)

    def transform(self, states):
        """Map S, I, R states into the saturated incidence feature space."""
        states = np.asarray(states, dtype=float)
        if states.ndim == 1:
            states = states[None, :]
        S = states[:, 0]
        I = states[:, 1]
        saturated = S * I / (1.0 + self.a * I)
        return np.column_stack([I, S * I, saturated])
