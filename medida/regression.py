import numpy as np

class RelevanceVectorMachine:
    """Sparse Bayesian linear regression (RVM)."""

    def __init__(self, max_iter=1000, tol=1e-4, alpha_threshold=1e12,
                 normalize=True, t_min=0.0, threshold=0.0, debias=True):
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.alpha_threshold = float(alpha_threshold)
        self.normalize = bool(normalize)
        self.t_min = float(t_min)
        self.threshold = float(threshold)
        self.debias = bool(debias)

    def fit(self, Phi, y):
        """Fit ``y ~ Phi c`` and store the sparse solution in ``coef_``."""
        Phi = np.asarray(Phi, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n_samples, n_features = Phi.shape

        if self.normalize:
            scales = np.linalg.norm(Phi, axis=0)
            scales[scales == 0.0] = 1.0
        else:
            scales = np.ones(n_features)
        Phi_n = Phi / scales

        y_var = float(np.var(y))
        if y_var <= 1e-30 or np.linalg.norm(y) <= 1e-14:
            self._store_empty(n_features)
            return self

        # --- evidence-maximisation (sparse Bayesian learning) ---------
        alpha = np.ones(n_features)
        beta = 1.0 / max(0.1 * y_var, 1e-12)
        active = np.arange(n_features)
        Phi_t_y = Phi_n.T @ y

        for _ in range(self.max_iter):
            Pa = Phi_n[:, active]
            gram = beta * (Pa.T @ Pa) + np.diag(alpha[active])
            sigma = self._safe_inverse(gram)
            mu = beta * (sigma @ Phi_t_y[active])

            gamma = np.clip(1.0 - alpha[active] * np.clip(np.diag(sigma), 0.0, None),
                            1e-12, 1.0)
            alpha_new = gamma / (mu ** 2 + 1e-300)
            residual = y - Pa @ mu
            beta = max((n_samples - gamma.sum()) / (residual @ residual + 1e-300),
                       1e-12)

            delta = np.max(np.abs(np.log(alpha_new + 1e-300) - np.log(alpha[active] + 1e-300)))
            alpha[active] = alpha_new
            keep = alpha[active] < self.alpha_threshold
            pruned = not np.all(keep)
            active = active[keep]
            if active.size == 0:
                break
            if delta < self.tol and not pruned:
                break

        support = active.copy()

        # --- significance pruning: t-statistic backward elimination ---
        if self.t_min > 0.0 and support.size > 0:
            support = self._t_eliminate(Phi_n, y, support, self.t_min)

        # --- final (debiased) coefficients on the support -------------
        coef_n = np.zeros(n_features)
        tstat = np.zeros(n_features)
        if support.size > 0:
            coef_s, t_s = self._ols_with_tstat(Phi_n[:, support], y)
            coef_n[support] = coef_s
            tstat[support] = t_s

        # --- optional relative-magnitude clean-up ---------------------
        if self.threshold > 0.0 and support.size > 0:
            biggest = np.max(np.abs(coef_n[support]))
            if biggest > 0.0:
                keep = np.abs(coef_n) >= self.threshold * biggest
                support = np.intersect1d(support, np.where(keep)[0])
                coef_n = np.zeros(n_features)
                tstat = np.zeros(n_features)
                if support.size > 0:
                    coef_s, t_s = self._ols_with_tstat(Phi_n[:, support], y)
                    coef_n[support] = coef_s
                    tstat[support] = t_s

        self.coef_ = coef_n / scales
        self.support_ = np.sort(support)
        self.tstat_ = tstat
        self.beta_ = beta
        self.n_features_ = n_features
        
        # compatibility with notebook properties
        self.coefficients = self.coef_
        self.active_idx = self.support_
        
        return self

    def predict(self, Phi):
        return np.asarray(Phi, dtype=float) @ self.coef_

    def _t_eliminate(self, Phi_n, y, support, t_min):
        support = list(np.sort(support))
        while support:
            coef, tstat = self._ols_with_tstat(Phi_n[:, support], y)
            worst = int(np.argmin(tstat))
            if tstat[worst] < t_min:
                support.pop(worst)
            else:
                break
        return np.array(support, dtype=int)

    @staticmethod
    def _ols_with_tstat(design, y):
        design = np.asarray(design, dtype=float)
        n, k = design.shape
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ coef
        dof = max(n - k, 1)
        s2 = float(residual @ residual) / dof
        gram = design.T @ design
        try:
            gram_inv = np.linalg.inv(gram)
        except np.linalg.LinAlgError:
            gram_inv = np.linalg.pinv(gram)
        var = s2 * np.clip(np.diag(gram_inv), 0.0, None)
        std = np.sqrt(var)
        if s2 <= 1e-300:
            tstat = np.full(k, np.inf)
        else:
            tstat = np.abs(coef) / (std + 1e-300)
        return coef, tstat

    def _store_empty(self, n_features):
        self.coef_ = np.zeros(n_features)
        self.support_ = np.array([], dtype=int)
        self.tstat_ = np.zeros(n_features)
        self.beta_ = np.inf
        self.n_features_ = n_features
        self.coefficients = self.coef_
        self.active_idx = self.support_

    @staticmethod
    def _safe_inverse(matrix):
        try:
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(matrix)

class RidgeRVM(RelevanceVectorMachine):
    """RVM whose support-fitting OLS step carries a small Tikhonov ridge."""

    def __init__(self, *args, ridge=1e-2, **kwargs):
        super().__init__(*args, **kwargs)
        self.ridge = float(ridge)

    def _ols_with_tstat(self, design, y):
        design = np.asarray(design, dtype=float)
        n, k = design.shape
        gram = design.T @ design
        lam = self.ridge * (np.trace(gram) / max(k, 1))
        gram_r = gram + lam * np.eye(k)
        try:
            coef = np.linalg.solve(gram_r, design.T @ y)
        except np.linalg.LinAlgError:
            coef, *_ = np.linalg.lstsq(gram_r, design.T @ y, rcond=None)
            
        residual = y - design @ coef
        dof = max(n - k, 1)
        s2 = float(residual @ residual) / dof
        try:
            gram_ri = np.linalg.inv(gram_r)
        except np.linalg.LinAlgError:
            gram_ri = np.linalg.pinv(gram_r)
            
        var = s2 * np.clip(np.diag(gram_ri @ gram @ gram_ri), 0.0, None)
        std = np.sqrt(var)
        if s2 <= 1e-300:
            tstat = np.full(k, np.inf)
        else:
            tstat = np.abs(coef) / (std + 1e-300)
        return coef, tstat
