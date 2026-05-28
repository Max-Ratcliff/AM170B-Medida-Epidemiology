import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Any
from .integrators import integrate

@dataclass
class MedidaResult:
    error_coefficients: np.ndarray
    supports: List[np.ndarray]
    delta_u: np.ndarray
    state: np.ndarray
    forecast: np.ndarray
    library: Any
    feature_names: List[str]
    state_names: Optional[List[str]] = None

    def corrected_coefficients(self, imperfect_coeffs):
        return imperfect_coeffs + self.error_coefficients

class MEDIDA:
    """The MEDIDA model-error-discovery framework."""

    def __init__(self, model, library, rvm=None, dt=1e-3,
                 enkf=None, sigma_obs=0.0, significance=1e-3):
        self.model = model
        self.library = library
        from .regression import RelevanceVectorMachine
        self.rvm = rvm if rvm is not None else RelevanceVectorMachine()
        self.dt = float(dt)
        self.enkf = enkf
        self.sigma_obs = float(sigma_obs)
        self.significance = float(significance)
        self.result_ = None

    def model_step(self, u):
        """Advance the model one observation interval ``dt``."""
        return self.model.step(u, self.dt)

    def estimate_model_error(self, obs_prev, obs_curr):
        """Step 1 (and 3): return ``(delta_u, library_state, forecast)``."""
        obs_prev = np.atleast_2d(np.asarray(obs_prev, dtype=float))
        obs_curr = np.atleast_2d(np.asarray(obs_curr, dtype=float))

        if self.enkf is None:
            forecast = self.model_step(obs_prev)
            delta_u = (obs_curr - forecast) / self.dt
            state = obs_curr
        else:
            # EnKF replaces raw observed state with analysis state
            analysis, forecast = self.enkf.analysis(
                self.model, obs_prev, obs_curr, self.sigma_obs, self.dt)
            delta_u = (analysis - forecast) / self.dt
            state = analysis
            
        return delta_u, state, forecast

    def fit(self, obs_prev, obs_curr):
        """Discover the model error from observation pairs."""
        delta_u, state, forecast = self.estimate_model_error(obs_prev, obs_curr)
        Phi = self.library.transform(state)
        kind = getattr(self.library, "kind", "vector_ode")
        n_features = self.library.n_features
        
        if kind == "scalar_pde":
            target = delta_u.reshape(-1)
            tendency = np.asarray(self.model.rhs(state), dtype=float).reshape(-1)
            if np.linalg.norm(target) <= self.significance * np.linalg.norm(tendency):
                error_coeffs = np.zeros(n_features)
                supports = [np.array([], dtype=int)]
            else:
                self.rvm.fit(Phi, target)
                error_coeffs = self.rvm.coef_.copy()
                supports = [self.rvm.support_.copy()]
        else:
            dim = delta_u.shape[1]
            error_coeffs = np.zeros((n_features, dim))
            supports = []
            tendency = np.asarray(self.model.rhs(state), dtype=float).reshape(-1, dim)
            for j in range(dim):
                target = delta_u[:, j]
                tend_norm = np.linalg.norm(tendency[:, j])
                if np.linalg.norm(target) <= self.significance * tend_norm:
                    error_coeffs[:, j] = 0.0
                    supports.append(np.array([], dtype=int))
                else:
                    self.rvm.fit(Phi, target)
                    error_coeffs[:, j] = self.rvm.coef_
                    supports.append(self.rvm.support_.copy())

        self.result_ = MedidaResult(
            error_coefficients=error_coeffs,
            supports=supports,
            delta_u=delta_u,
            state=state,
            forecast=forecast,
            library=self.library,
            feature_names=self.library.feature_names,
            state_names=getattr(self.model, 'state_names', None)
        )
        return self.result_

    @property
    def error_coefficients_(self):
        if self.result_ is None:
            raise RuntimeError("call fit() first")
        return self.result_.error_coefficients

    def corrected_coefficients(self, model_coefficients):
        if self.result_ is None:
            raise RuntimeError("call fit() first")
        return self.result_.corrected_coefficients(model_coefficients)

def sample_observations(system, n_samples, dt, spinup=25.0, spin_dt=0.01,
                        ic_scale=8.0, method="rk4", substeps=4, seed=0,
                        sigma_obs=0.0, noise_seed=1):
    """Generate sporadic observation pairs (vector ODE)."""
    rng = np.random.default_rng(seed)
    ic = rng.standard_normal((n_samples, system.dim)) * ic_scale

    spin_steps = max(int(round(spinup / spin_dt)), 1)
    truth_prev = integrate(system.rhs, ic, spin_dt, spin_steps, method=method)[-1]
    truth_curr = integrate(system.rhs, truth_prev, dt, 1,
                           method=method, substeps=substeps)[-1]

    if sigma_obs > 0.0:
        noise_rng = np.random.default_rng(noise_seed)
        obs_prev = truth_prev + noise_rng.normal(0.0, sigma_obs, truth_prev.shape)
        obs_curr = truth_curr + noise_rng.normal(0.0, sigma_obs, truth_curr.shape)
    else:
        obs_prev = truth_prev.copy()
        obs_curr = truth_curr.copy()

    return obs_prev, obs_curr, truth_prev, truth_curr

def sample_simplex_observations(system, n_samples, dt, seed=0,
                                sigma_obs=0.0, noise_seed=1,
                                alpha=None):
    """Generate random states on the probability simplex, then advance one step."""
    rng = np.random.default_rng(seed)
    if alpha is None:
        dim = getattr(system, 'dim', 3)
        alpha = np.ones(dim)

    truth_prev = rng.dirichlet(alpha=np.asarray(alpha, dtype=float), size=n_samples)
    truth_curr = integrate(system.rhs, truth_prev, dt, 1, substeps=8)[-1]

    if sigma_obs > 0.0:
        noise_rng = np.random.default_rng(noise_seed)
        obs_prev = truth_prev + noise_rng.normal(0.0, sigma_obs, truth_prev.shape)
        obs_curr = truth_curr + noise_rng.normal(0.0, sigma_obs, truth_curr.shape)
        obs_prev = np.clip(obs_prev, 0.0, 1.2)
        obs_curr = np.clip(obs_curr, 0.0, 1.2)
    else:
        obs_prev = truth_prev.copy()
        obs_curr = truth_curr.copy()

    return obs_prev, obs_curr, truth_prev, truth_curr

def sample_ks_observations(system, n_samples, dt, spinup_time=80.0,
                           spin_dt=0.05, seed=0, sigma_obs=0.0, noise_seed=1):
    """Generate KS observation pairs using ETDRK4."""
    rng = np.random.default_rng(seed)
    # Random initial condition in spectral space
    v0 = rng.standard_normal(system.N) + 1j * rng.standard_normal(system.N)
    u0 = np.real(np.fft.ifft(v0))
    
    # Spin-up
    n_spin = int(spinup_time / spin_dt)
    u = u0
    for _ in range(n_spin):
        u = system.step(u, spin_dt)
    
    # Collect samples separated by dt
    truth_prev = np.empty((n_samples, system.N))
    truth_curr = np.empty((n_samples, system.N))
    
    for i in range(n_samples):
        truth_prev[i] = u
        u = system.step(u, dt)
        truth_curr[i] = u
        # Add some extra evolution to decorrelate samples
        for _ in range(5):
            u = system.step(u, dt)
            
    if sigma_obs > 0.0:
        noise_rng = np.random.default_rng(noise_seed)
        obs_prev = truth_prev + noise_rng.normal(0.0, sigma_obs, truth_prev.shape)
        obs_curr = truth_curr + noise_rng.normal(0.0, sigma_obs, truth_curr.shape)
    else:
        obs_prev = truth_prev.copy()
        obs_curr = truth_curr.copy()
        
    return obs_prev, obs_curr, truth_prev, truth_curr

def sample_hidden_E_seir_observations(system, n_samples, dt, seed=0,
                                      sigma_obs=0.0, noise_seed=1):
    """Generate SEIR data, then hide E."""
    rng = np.random.default_rng(seed)
    truth_prev_4d = rng.dirichlet(alpha=np.array([3.0, 0.8, 0.8, 2.0]), size=n_samples)
    truth_prev_4d[:, 1] = np.maximum(truth_prev_4d[:, 1], 1e-4) # E
    truth_prev_4d[:, 2] = np.maximum(truth_prev_4d[:, 2], 1e-4) # I
    truth_prev_4d /= truth_prev_4d.sum(axis=1, keepdims=True)

    truth_curr_4d = integrate(system.rhs, truth_prev_4d, dt, 1, method="rk4", substeps=8)[-1]
    
    # Hide E: keep only S, I, R (indices 0, 2, 3)
    truth_prev_3d = truth_prev_4d[:, [0, 2, 3]]
    truth_curr_3d = truth_curr_4d[:, [0, 2, 3]]

    if sigma_obs > 0.0:
        noise_rng = np.random.default_rng(noise_seed)
        obs_prev_3d = truth_prev_3d + noise_rng.normal(0.0, sigma_obs, truth_prev_3d.shape)
        obs_curr_3d = truth_curr_3d + noise_rng.normal(0.0, sigma_obs, truth_curr_3d.shape)
        obs_prev_3d = np.clip(obs_prev_3d, 0.0, 1.2)
        obs_curr_3d = np.clip(obs_curr_3d, 0.0, 1.2)
    else:
        obs_prev_3d = truth_prev_3d.copy()
        obs_curr_3d = truth_curr_3d.copy()

    return obs_prev_3d, obs_curr_3d, truth_prev_4d, truth_curr_4d
