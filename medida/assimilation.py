import numpy as np

class EnsembleKalmanFilter:
    """Stochastic EnKF used to denoise observations for MEDIDA."""

    def __init__(self, n_ensemble=200, inflation=1.05, seed=0):
        self.n_ensemble = int(n_ensemble)
        self.inflation = float(inflation)
        self.seed = seed

    def analysis(self, model, obs_prev, obs_curr, sigma_obs, dt):
        """Run the EnKF for every observation pair (vectorised).

        Parameters
        ----------
        model : DynamicalSystem
            The imperfect model used for forecasting.
        obs_prev, obs_curr : array_like, shape ``(n_samples, dim)``
            Noisy observations at ``t_i - dt`` and ``t_i``.
        sigma_obs : float
            Standard deviation of the observation noise.
        dt : float
            Time step between observations.

        Returns
        -------
        analysis_mean : numpy.ndarray, shape ``(n_samples, dim)``
        forecast_mean : numpy.ndarray, shape ``(n_samples, dim)``
        """
        obs_prev = np.atleast_2d(np.asarray(obs_prev, dtype=float))
        obs_curr = np.atleast_2d(np.asarray(obs_curr, dtype=float))
        n, dim = obs_prev.shape
        N = self.n_ensemble
        rng = np.random.default_rng(self.seed)

        # (Eq. 10) IC ensemble perturbed by the TRUE observation noise sigma
        ic = obs_prev[:, None, :] + rng.normal(0.0, sigma_obs, size=(n, N, dim))

        # one-step forecast of every ensemble member
        # model.step handles vectorised input (n, N, dim)
        fc = np.asarray(model.step(ic, dt), dtype=float).reshape(n, N, dim)
        forecast_mean = fc.mean(axis=1)

        # (Eq. 11) per-sample background covariance
        dev = fc - forecast_mean[:, None, :]
        P = np.einsum("nki,nkj->nij", dev, dev) / max(N - 1, 1)

        # multiplicative inflation of the background covariance
        P = self.inflation * P

        # (Eq. 12) Kalman gain
        R = sigma_obs ** 2 * np.eye(dim)
        # Solve (P + R) K.T = P.T -> K = P (P + R)^-1
        # For batch: use np.linalg.solve if possible or inv
        # P is (n, dim, dim), R is (dim, dim)
        try:
            K = P @ np.linalg.inv(P + R)
        except np.linalg.LinAlgError:
            K = P @ np.linalg.pinv(P + R)

        # (Eq. 13) perturbed observations at t_i
        obs_ens = obs_curr[:, None, :] + rng.normal(0.0, sigma_obs,
                                                    size=(n, N, dim))

        # (Eq. 14) analysis ensemble
        innovation = obs_ens - fc
        analysis = fc + np.einsum("nij,nkj->nki", K, innovation)

        return analysis.mean(axis=1), forecast_mean
