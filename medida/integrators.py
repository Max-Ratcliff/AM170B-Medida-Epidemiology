import numpy as np


def euler_step(rhs, u, dt):
    """Advance the state by dt using the explicit forward Euler scheme.

    Args:
        rhs (callable): Function evaluating the system tendencies du/dt.
        u (array-like): Current state of the system.
        dt (float): Time step size.

    Returns:
        np.ndarray: The advanced state.
    """
    return u + dt * rhs(u)


def rk4_step(rhs, u, dt):
    """Advance the state by dt using the 4th-order Runge-Kutta scheme.

    Args:
        rhs (callable): Function evaluating the system tendencies du/dt.
        u (array-like): Current state of the system.
        dt (float): Time step size.

    Returns:
        np.ndarray: The advanced state.
    """
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
    """Integrate the system du/dt = rhs(u) over multiple steps.

    Supports batch initial conditions of shape (n_samples, dim).

    Args:
        rhs: Function evaluating the system tendencies.
        u0: Initial state or batch of initial states.
        dt: Observation interval (or total integration time per step).
        n_steps: Number of dt intervals to integrate.
        method: Integration scheme ('rk4' or 'euler').
        substeps: Number of internal integrator steps per dt interval.

    Returns:
        Full trajectory array of shape (n_steps + 1, ...).
    """
    if method not in INTEGRATORS:
        raise KeyError(
            f"Unsupported integrator '{method}'; "
            f"choose from {list(INTEGRATORS)}"
        )

    step_func = INTEGRATORS[method]
    h = dt / substeps
    u = np.array(u0, dtype=float)
    traj = np.empty((n_steps + 1,) + u.shape, dtype=float)
    traj[0] = u

    for i in range(n_steps):
        for _ in range(substeps):
            u = step_func(rhs, u, h)
        traj[i + 1] = u

    return traj
