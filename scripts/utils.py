import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from medida import (
    sample_simplex_observations, 
    MEDIDA, 
    PolynomialODE, 
    coefficient_error,
    relative_error,
    format_system,
    RidgeRVM
)

def run_medida_experiment(
    name, true_system, library, true_coeffs, imperfect_coeffs,
    u0, alpha, dt_fit=0.01, n_samples=1000, sigma_obs=0.0,
    seed=0, noise_seed=1, rvm=None, significance=1e-6,
    ridge=0.0, plot_index=1
):
    print("\n" + "="*70)
    print(name)
    print("="*70)
    
    if rvm is None:
        if ridge > 0:
            rvm = RidgeRVM(ridge=ridge, t_min=2.0, threshold=0.01)
        else:
            from medida import RelevanceVectorMachine
            rvm = RelevanceVectorMachine(t_min=2.0, threshold=0.01)

    # 1. Sample observations using Dirichlet simplex sampling (matches notebook)
    print(f"[*] Generating {n_samples} Dirichlet-simplex observation pairs (dt={dt_fit})...")
    obs_prev, obs_curr, truth_prev, truth_curr = sample_simplex_observations(
        true_system, n_samples, dt_fit, seed=seed, noise_seed=noise_seed,
        sigma_obs=sigma_obs, alpha=alpha
    )
    
    # 2. Setup MEDIDA
    model_m = PolynomialODE(imperfect_coeffs, library, state_names=true_system.state_names)
    medida = MEDIDA(model_m, library, dt=dt_fit, rvm=rvm, significance=significance)
    
    # 3. Fit model error
    print("[*] Running MEDIDA error discovery (Step 1-2)...")
    result = medida.fit(obs_prev, obs_curr)
    
    # 4. Correct model
    print("[*] Applying structural corrections to model...")
    corrected_coeffs = result.corrected_coefficients(imperfect_coeffs)
    corrected_model = PolynomialODE(corrected_coeffs, library, state_names=true_system.state_names)
    
    # 5. Metrics
    print("[*] Calculating performance metrics...")
    eps_m = coefficient_error(true_coeffs, imperfect_coeffs)
    eps_star = coefficient_error(true_coeffs, corrected_coeffs)
    
    print(f"\nResults:")
    print(f"  Observation noise sigma: {sigma_obs}")
    print(f"  Original coefficient error eps_m: {eps_m:.6e}")
    print(f"  Corrected coefficient error eps_star: {eps_star:.6e}")
    if eps_star > 0:
        print(f"  Improvement factor: {eps_m/eps_star:.2f}x")
    
    # Trajectory verification
    print(f"[*] Validating corrected trajectory (t_end=50, substeps=8)...")
    t_end = 50.0
    dt = 0.05
    n_steps = int(t_end / dt)
    
    true_traj = true_system.trajectory(u0, dt=dt, n_steps=n_steps, method="rk4", substeps=8)
    bad_traj = model_m.trajectory(u0, dt=dt, n_steps=n_steps, method="rk4", substeps=8)
    corrected_traj = corrected_model.trajectory(u0, dt=dt, n_steps=n_steps, method="rk4", substeps=8)
    
    rel_m = relative_error(true_traj, bad_traj)
    rel_star = relative_error(true_traj, corrected_traj)
    
    print(f"  Original trajectory relative error: {rel_m:.6e}")
    print(f"  Corrected trajectory relative error: {rel_star:.6e}")
    if rel_star > 0:
        print(f"  Trajectory improvement factor: {rel_m/rel_star:.2f}x")

    print("\n[Models]")
    print("True model:")
    print(format_system(true_coeffs, library.feature_names, true_system.state_names))
    print("\nImperfect model:")
    print(format_system(imperfect_coeffs, library.feature_names, true_system.state_names))
    print("\nDiscovered MEDIDA correction:")
    print(format_system(result.error_coefficients, library.feature_names, true_system.state_names))
    print("\nCorrected model:")
    print(format_system(corrected_coeffs, library.feature_names, true_system.state_names))

    # Plotting
    print(f"[*] Generating diagnostic plots: outputs/figures/{name.lower().replace(' ', '_').replace('-', '_')}.png")
    t = np.linspace(0, t_end, n_steps + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot component comparison
    ax = axes[0]
    ax.plot(t, true_traj[:, plot_index], 'k-', label='True')
    ax.plot(t, bad_traj[:, plot_index], 'r--', label='Imperfect')
    ax.plot(t, corrected_traj[:, plot_index], 'b:', label='Corrected')
    ax.set_title(f"{name}: {true_system.state_names[plot_index]} comparison")
    ax.set_xlabel("time")
    ax.set_ylabel(true_system.state_names[plot_index])
    ax.legend()
    
    # Plot residuals
    ax = axes[1]
    ax.plot(t, bad_traj[:, plot_index] - true_traj[:, plot_index], 'r--', label='Imperfect residual')
    ax.plot(t, corrected_traj[:, plot_index] - true_traj[:, plot_index], 'b:', label='Corrected residual')
    ax.set_title("Residuals")
    ax.set_xlabel("time")
    ax.legend()
    
    plt.tight_layout()
    filename = name.lower().replace(" ", "_").replace("-", "_") + ".png"
    plt.savefig(f"outputs/figures/{filename}")
    
    return result, corrected_coeffs, corrected_model
