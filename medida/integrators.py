import numpy as np

def euler_step(rhs, u, dt):
    """One step of the explicit (forward) Euler scheme."""
    return u + dt * rhs(u)

def rk4_step(rhs, u, dt):
    """One step of the classical 4th-order Runge-Kutta scheme."""
    k1 = rhs(u)
    k2 = rhs(u + 0.5 * dt * k1)
    k3 = rhs(u + 0.5 * dt * k2)
    k4 = rhs(u + dt * k3)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

INTEGRATORS = {
    "euler": euler_step,
    "rk4": rk4_step,
}

def integrate(rhs, u0, dt, n_steps, method="rk4", substeps=1):
    """Integrate ``du/dt = rhs(u)`` and return the full trajectory.
    
    Supports batch initial conditions of shape (n_samples, dim).
    """
    if method not in INTEGRATORS:
        raise KeyError(f"unknown integrator {method!r}; choose from {list(INTEGRATORS)}")
    step = INTEGRATORS[method]
    h = dt / substeps
    u = np.array(u0, dtype=float)
    traj = np.empty((n_steps + 1,) + u.shape, dtype=float)
    traj[0] = u
    for i in range(n_steps):
        for _ in range(substeps):
            u = step(rhs, u, h)
        traj[i + 1] = u
    return traj
