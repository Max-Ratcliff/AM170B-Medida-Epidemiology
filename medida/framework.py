import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from .integrators import integrate

@dataclass
class MedidaResult:
    error_coefficients: np.ndarray
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
            # EnKF integration not fully implemented in previous turn,
            # but matching the logic for noise-free first.
            forecast = self.model_step(obs_prev)
            # In notebook Step 3, analysis replaces raw obs
            analysis, forecast = self.enkf.analysis(
                self.model, obs_prev, obs_curr, self.sigma_obs, self.dt)
            delta_u = (analysis - forecast) / self.dt
            state = analysis
            
        return delta_u, state, forecast

    def fit(self, obs_prev, obs_curr):
        """Discover the model error from observation pairs."""
        delta_u, state, forecast = self.estimate_model_error(obs_prev, obs_curr)
        Phi = self.library.transform(state)
        
        # Determine if scalar PDE or vector ODE
        if hasattr(self.library, "kind") and self.library.kind == "scalar_pde":
            target = delta_u.reshape(-1)
            tendency = np.asarray(self.model.rhs(state), dtype=float).reshape(-1)
            if np.linalg.norm(target) <= self.significance * np.linalg.norm(tendency):
                error_coeffs = np.zeros(self.library.n_features)
            else:
                self.rvm.fit(Phi, target)
                error_coeffs = self.rvm.coef_.copy()
        else:
            dim = delta_u.shape[1]
            error_coeffs = np.zeros((self.library.n_features, dim))
            tendency = np.asarray(self.model.rhs(state), dtype=float).reshape(-1, dim)
            for j in range(dim):
                target = delta_u[:, j]
                tend_norm = np.linalg.norm(tendency[:, j])
                if np.linalg.norm(target) <= self.significance * tend_norm:
                    error_coeffs[:, j] = 0.0
                else:
                    self.rvm.fit(Phi, target)
                    error_coeffs[:, j] = self.rvm.coef_

        return MedidaResult(
            error_coefficients=error_coeffs,
            feature_names=self.library.feature_names,
            state_names=getattr(self.model, 'state_names', None)
        )

def sample_simplex_observations(system, n_samples, dt, seed=0,
                                sigma_obs=0.0, noise_seed=1,
                                alpha=None):
    """
    Generate random states on the probability simplex, then advance one step.
    Matches the notebook's Dirichlet-based sampling.
    """
    rng = np.random.default_rng(seed)

    if alpha is None:
        dim = getattr(system, 'dim', 3)
        alpha = np.ones(dim)

    truth_prev = rng.dirichlet(alpha=np.asarray(alpha, dtype=float),
                               size=n_samples)

    # In notebook, integrate returns (n_steps + 1, *u.shape)
    # substeps=8 is used for accuracy
    truth_curr = system.trajectory(
        truth_prev,
        dt=dt,
        n_steps=1,
        method="rk4",
        substeps=8
    )[-1]

    if sigma_obs > 0.0:
        noise_rng = np.random.default_rng(noise_seed)
        obs_prev = truth_prev + noise_rng.normal(0.0, sigma_obs, truth_prev.shape)
        obs_curr = truth_curr + noise_rng.normal(0.0, sigma_obs, truth_curr.shape)
    else:
        obs_prev, obs_curr = truth_prev, truth_curr

    return obs_prev, obs_curr, truth_prev, truth_curr

def sample_observations(system, u0, t_end, dt_obs, sigma_obs=0.0, seed=None):
    """Integrate system and return noisy observations."""
    if seed is not None:
        np.random.seed(seed)
    
    n_steps = int(t_end / dt_obs)
    u_true = system.trajectory(u0, dt=dt_obs, n_steps=n_steps)
    
    u_obs = u_true + sigma_obs * np.random.randn(*u_true.shape)
    return u_obs, u_true, np.linspace(0, t_end, n_steps + 1)

def sample_ks_observations(system, n_samples, dt, spinup_time=80.0,
                           spinup_dt=0.25, ic_amp=0.6, seed=0,
                           sigma_obs=0.0, noise_seed=1):
    """Sporadic KS observation pairs.
    Matches notebook logic for KS IC generation and spin-up.
    """
    rng = np.random.default_rng(seed)
    x = system.x; L = system.L
    ic = np.zeros((n_samples, system.N))
    for i in range(n_samples):
        amps = rng.standard_normal(8) * ic_amp
        phs  = rng.uniform(0, 2 * np.pi, 8)
        for m, (a, p) in enumerate(zip(amps, phs), start=1):
            ic[i] += a * np.cos(2 * np.pi * m * x / L + p)
    # spinup at coarse dt
    n_spin = max(int(spinup_time / spinup_dt), 1)
    truth_prev = ic.copy()
    for _ in range(n_spin):
        truth_prev = system.step(truth_prev, spinup_dt)
    # one accurate dt-step gives the later member
    truth_curr = system.step(truth_prev, dt)
    if sigma_obs > 0.0:
        nrng = np.random.default_rng(noise_seed)
        obs_prev = truth_prev + nrng.normal(0.0, sigma_obs, truth_prev.shape)
        obs_curr = truth_curr + nrng.normal(0.0, sigma_obs, truth_curr.shape)
    else:
        obs_prev = truth_prev.copy()
        obs_curr = truth_curr.copy()
    return obs_prev, obs_curr, truth_prev, truth_curr
