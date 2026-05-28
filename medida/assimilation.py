import numpy as np


class EnsembleKalmanFilter:
    """Stochastic Ensemble Kalman Filter (EnKF) for denoising observations.

    Used by MEDIDA to provide a cleaner state estimate ('analysis') from
    noisy observation pairs, improving the reliability of model-error
    identification.
    """

    def __init__(self, n_ensemble=200, inflation=1.05, seed=0):
        self.n_ensemble = int(n_ensemble)
        self.inflation = float(inflation)
        self.seed = seed

    def analysis(self, model, obs_prev, obs_curr, sigma_obs, dt):
        """Perform EnKF analysis to estimate the true state at the current time.

        This implementation is vectorized across n_samples observation pairs.
        """
        obs_prev = np.atleast_2d(np.asarray(obs_prev, dtype=float))
        obs_curr = np.atleast_2d(np.asarray(obs_curr, dtype=float))
        n_samples, dim = obs_prev.shape
        N_ens = self.n_ensemble
        rng = np.random.default_rng(self.seed)

        # Initialize the ensemble by perturbing the previous observation
        ic = obs_prev[:, None, :] + rng.normal(0.0, sigma_obs, size=(n_samples, N_ens, dim))

        # Perform a one-step forecast with the imperfect model
        # The ensemble is flattened to (n*N, dim) to be compatible with library transforms
        ic_flat = ic.reshape(n_samples * N_ens, dim)
        fc = np.asarray(model.step(ic_flat, dt), dtype=float).reshape(n_samples, N_ens, dim)
        forecast_mean = fc.mean(axis=1)

        # Estimate the background covariance (P) from the forecast ensemble
        dev = fc - forecast_mean[:, None, :]
        P = np.einsum("nki,nkj->nij", dev, dev) / max(N_ens - 1, 1)

        # Apply multiplicative inflation to the background covariance
        P = self.inflation * P

        # Compute the Kalman Gain matrix (K)
        # Solve (P + R) K.T = P.T -> K = P @ (P + R)^-1
        R = sigma_obs**2 * np.eye(dim)
        try:
            K = P @ np.linalg.inv(P + R)
        except np.linalg.LinAlgError:
            K = P @ np.linalg.pinv(P + R)

        # Generate perturbed observations for the current time step
        obs_ens = obs_curr[:, None, :] + rng.normal(0.0, sigma_obs, size=(n_samples, N_ens, dim))

        # Update the ensemble to obtain the analysis state
        innovation = obs_ens - fc
        analysis = fc + np.einsum("nij,nkj->nki", K, innovation)

        return analysis.mean(axis=1), forecast_mean
