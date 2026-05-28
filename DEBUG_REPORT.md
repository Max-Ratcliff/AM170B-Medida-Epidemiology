# DEBUG REPORT: KS PDE Shape Broadcast Error

## 1. The Error
During the execution of `scripts/verification.py`, the `example_ks()` function crashes with the following error:
```
ValueError: operands could not be broadcast together with shapes (800,64) (51200,) 
```
This occurs inside the `rk4_step` integrator in `medida/integrators.py` at the line:
```python
k2 = rhs(u + 0.5 * dt * k1)
```

## 2. Root Cause Analysis
The error arises from a shape mismatch between the input state `u` (shape: `800` samples $\times$ `64` spatial grid points) and the output of the imperfect model's RHS function `k1` (shape: `51200,`, which is flattened).

1.  **Input State `u`**: `obs_prev` has shape `(800, 64)`.
2.  **Feature Transformation**: `PDELibrary.transform(states)` receives this `(800, 64)` array and intentionally flattens each computed spatial derivative into a 1D column vector of size `51200` to perform linear regression over all samples and spatial points simultaneously.
3.  **PolynomialODE.rhs Evaluation**: The `PolynomialODE.rhs(u)` method calls `phi = self.library.transform(u)`. `phi` is thus `(51200, n_features)`. It performs the matrix multiplication `res = phi @ self.coefficients`.
4.  **The Flaw**: `PolynomialODE.rhs` returns `res.ravel()` or `res.squeeze()`, yielding a 1D array of shape `(51200,)`. It **does not reshape the result back to the original input shape of `u`**.
5.  **Integration Failure**: The `rk4_step` attempts to do `u + 0.5 * dt * k1`. `u` is `(800, 64)` and `k1` is `(51200,)`, leading to the NumPy broadcast failure.

## 3. Recommended Fixes for the Fresh Context

You have two paths to resolve this quickly in the new context:

### Path A: Fix `PolynomialODE.rhs` (Recommended)
Modify `PolynomialODE.rhs` in `medida/systems.py` to restore the original shape of `u` if the library flattened it (i.e., when dealing with PDEs).

```python
    def rhs(self, u):
        """Evaluate the polynomial RHS using the feature library."""
        u_arr = np.asarray(u)
        u_2d = np.atleast_2d(u_arr)
        phi = self.library.transform(u_2d)
        res = phi @ self.coefficients
        
        # If the library flattened spatial dimensions (like PDELibrary), 
        # res will be larger than u_2d.shape[0]. We must reshape it back.
        if hasattr(self.library, "kind") and self.library.kind == "scalar_pde":
            res = res.reshape(u_arr.shape)
            return res

        if self.dim == 1:
            return res.ravel()
        return res.squeeze()
```

### Path B: Git Rollback
If you pushed a successful checkpoint to Git *before* the refactoring of `systems.py` and `libraries.py` that caused this dimensionality mismatch, you can simply run:
```bash
git checkout <working_commit_hash> -- medida/systems.py medida/libraries.py
```

## 4. Next Steps
Once this fix is applied, the synthetic suite (`verification.py`) will run completely. The COVID analysis is already fully working and generated all its artifacts successfully.
