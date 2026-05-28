import numpy as np
from abc import ABC, abstractmethod

class FeatureLibrary(ABC):
    @abstractmethod
    def transform(self, states):
        pass
    def index(self, name):
        return self.feature_names.index(name)

class PolynomialLibrary(FeatureLibrary):
    def __init__(self, n_vars, degree=2, include_constant=True, var_names=None):
        self.n_vars = n_vars
        self.degree = degree
        self.include_constant = include_constant
        self.var_names = var_names if var_names else [f"u{i}" for i in range(n_vars)]
        
        self.feature_names = []
        if include_constant:
            self.feature_names.append("1")
            
        # First degree
        for i in range(n_vars):
            self.feature_names.append(self.var_names[i])
            
        # Second degree
        if degree >= 2:
            for i in range(n_vars):
                for j in range(i, n_vars):
                    self.feature_names.append(f"{self.var_names[i]} {self.var_names[j]}")
                    
        self.n_features = len(self.feature_names)

    def transform(self, states):
        states = np.asarray(states)
        if states.ndim == 1:
            states = states[None, :]
        n_samples = states.shape[0]
        
        features = []
        if self.include_constant:
            features.append(np.ones(n_samples))
            
        for i in range(self.n_vars):
            features.append(states[:, i])
            
        if self.degree >= 2:
            for i in range(self.n_vars):
                for j in range(i, self.n_vars):
                    features.append(states[:, i] * states[:, j])
                    
        return np.column_stack(features)

class PDELibrary(FeatureLibrary):
    """Library for 1D PDEs (spectral features)."""
    kind = "scalar_pde"
    def __init__(self, n_grid=64, length=22.0, poly_order=2, deriv_order=4):
        self.n_grid = int(n_grid)
        self.L = float(length)
        self.dx = self.L / self.n_grid
        self.k = 2.0 * np.pi * np.fft.fftfreq(self.n_grid, d=self.dx)
        
        self.feature_names = []
        # Matches notebook: u, u u_x, u_xx, u_xxxx etc.
        # Simple hardcoded list for KS as per notebook
        self.feature_names = ["u", "u u_x", "u_x", "u_xx", "u_xxx", "u_xxxx"]
        self.n_features = len(self.feature_names)

    def transform(self, u):
        u = np.asarray(u, dtype=float)
        if u.ndim == 1:
            u = u[None, :]
        n_samples, N = u.shape
        
        def spec_deriv(u, order):
            fhat = np.fft.fft(u, axis=-1) * (1j * self.k) ** order
            return np.real(np.fft.ifft(fhat, axis=-1))

        ux = spec_deriv(u, 1)
        uxx = spec_deriv(u, 2)
        uxxx = spec_deriv(u, 3)
        uxxxx = spec_deriv(u, 4)
        
        Phi = np.column_stack([
            u.flatten(),
            (u * ux).flatten(),
            ux.flatten(),
            uxx.flatten(),
            uxxx.flatten(),
            uxxxx.flatten()
        ])
        return Phi

class SaturatedSIRLibrary(FeatureLibrary):
    """
    Library for nonlinear-incidence SIR.
    Features: I, S I, S I / (1 + a I)
    """
    def __init__(self, a=8.0):
        self.a = float(a)
        self.feature_names = ["I", "S I", f"S I / (1 + {self.a:g} I)"]
        self.n_features = len(self.feature_names)

    def transform(self, states):
        states = np.asarray(states, dtype=float)
        if states.ndim == 1:
            states = states[None, :]
        S = states[:, 0]
        I = states[:, 1]
        saturated = S * I / (1.0 + self.a * I)
        return np.column_stack([I, S * I, saturated])
