import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Add the project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import run_medida_experiment
from medida import (
    SIRSystem, SIRSSystem, SIRDSystem, SIRNonlinearSystem,
    Lorenz63, KSSystem, SEIRSystem, ProjectedSIRFromSEIRSystem,
    PolynomialLibrary, SaturatedSIRLibrary, PDELibrary,
    PolynomialODE, RelevanceVectorMachine, MEDIDA, RidgeRVM,
    format_system, sample_ks_observations, coefficient_error,
    sample_hidden_E_seir_observations, sample_observations
)

def example_sir_vs_sirs():
    true_system = SIRSSystem(beta=0.6, gamma=0.18, xi=0.08)
    library = PolynomialLibrary(n_vars=3, degree=2, var_names=["S", "I", "R"])
    true_coeffs = true_system.coefficients(library)
    imperfect_system = SIRSystem(beta=0.6, gamma=0.18)
    imperfect_coeffs = imperfect_system.coefficients(library)

    run_medida_experiment(
        "Noise-free SIRS truth vs plain SIR",
        true_system, library, true_coeffs, imperfect_coeffs,
        u0=np.array([0.98, 0.02, 0.0]),
        alpha=np.array([3.0, 0.8, 2.0]),
        dt_fit=0.01, n_samples=800, sigma_obs=0.0
    )

def example_hidden_e():
    print("\n" + "="*70)
    print("Hidden-E experiment: SEIR truth vs SIR model")
    print("="*70)
    
    true_system = SEIRSystem(beta=0.6, sigma=0.2, gamma=0.18)
    library = PolynomialLibrary(n_vars=3, degree=2, var_names=["S", "I", "R"])
    imperfect_system = SIRSystem(beta=0.6, gamma=0.18)
    imperfect_coeffs = imperfect_system.coefficients(library)
    imperfect_model = PolynomialODE(imperfect_coeffs, library, state_names=["S", "I", "R"])
    
    dt_fit = 0.01
    n_samples = 1200
    obs_prev, obs_curr, _, _ = sample_hidden_E_seir_observations(
        true_system, n_samples=n_samples, dt=dt_fit, seed=404
    )
    
    medida = MEDIDA(imperfect_model, library, dt=dt_fit, significance=1e-6)
    result = medida.fit(obs_prev, obs_curr)
    
    corrected_coeffs = result.corrected_coefficients(imperfect_coeffs)
    corrected_model = PolynomialODE(corrected_coeffs, library, state_names=["S", "I", "R"])
    
    # Visualization
    sns.set_theme(style="ticks")
    u0_seir = np.array([0.98, 0.01, 0.01, 0.0])
    u0_sir = u0_seir[[0, 2, 3]]
    t_end, dt = 50.0, 0.1
    n_steps = int(t_end / dt)
    
    true_traj = true_system.trajectory(u0_seir, dt=dt, n_steps=n_steps, substeps=8)[:, [0, 2, 3]]
    bad_traj = imperfect_model.trajectory(u0_sir, dt=dt, n_steps=n_steps, substeps=8)
    cor_traj = corrected_model.trajectory(u0_sir, dt=dt, n_steps=n_steps, substeps=8)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    t = np.linspace(0, t_end, n_steps + 1)
    ax.plot(t, true_traj[:, 1], 'k-', label='True I (from SEIR)')
    ax.plot(t, bad_traj[:, 1], 'r--', label='Naive SIR I')
    ax.plot(t, cor_traj[:, 1], 'g:', lw=2, label='MEDIDA-Corrected SIR I')
    ax.set_title("Hidden-E: Trajectory Recovery", fontweight="bold")
    ax.legend(frameon=False)
    
    ax = axes[1]
    ax.plot(true_traj[:, 0], true_traj[:, 1], 'k-', alpha=0.3)
    ax.plot(cor_traj[:, 0], cor_traj[:, 1], 'g:', lw=2, label='Corrected Phase')
    ax.set_title("Hidden-E: Phase Space (S vs I)", fontweight="bold")
    
    plt.tight_layout()
    plt.savefig("outputs/figures/hidden_e_recovery.png", dpi=150)
    plt.close()

def example_lorenz():
    print("\n" + "="*70)
    print("Lorenz-63 Structural Discovery & 3D Visualization")
    print("="*70)
    true_system = Lorenz63()
    library = PolynomialLibrary(n_vars=3, degree=2, var_names=["x", "y", "z"])
    true_coeffs = true_system.coefficients(library)
    
    imperfect_coeffs = true_coeffs.copy()
    imperfect_coeffs[library.index("x z"), 1] = 0.0  # Remove xz from dy/dt
    imperfect_model = PolynomialODE(imperfect_coeffs, library, state_names=["x", "y", "z"])
    
    dt_fit, n_samples = 0.01, 2000
    obs_prev, obs_curr, _, _ = sample_observations(true_system, n_samples, dt_fit, spinup=20.0, seed=42)
    
    medida = MEDIDA(imperfect_model, library, dt=dt_fit, significance=1e-5)
    result = medida.fit(obs_prev, obs_curr)
    cor_model = PolynomialODE(result.corrected_coefficients(imperfect_coeffs), library)
    
    # 3D Plotting
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    u0 = np.array([1.0, 1.0, 1.0])
    traj_true = true_system.trajectory(u0, 0.01, 3000)
    traj_cor = cor_model.trajectory(u0, 0.01, 3000)
    
    ax.plot(traj_true[:, 0], traj_true[:, 1], traj_true[:, 2], 'k-', alpha=0.2, label='True Attractor')
    ax.plot(traj_cor[:, 0], traj_cor[:, 1], traj_cor[:, 2], 'g-', alpha=0.7, label='MEDIDA Corrected')
    ax.set_title("Lorenz-63: Recovering the Chaotic Attractor", fontweight="bold")
    ax.legend(frameon=False)
    plt.savefig("outputs/figures/lorenz_3d_attractor.png", dpi=150)
    plt.close()

def example_ks():
    print("\n" + "="*70)
    print("Kuramoto-Sivashinsky (KS) Spatio-Temporal Discovery")
    print("="*70)
    true_system = KSSystem(beta=1.0, nu_2=1.0, nu_4=1.0)
    library = PDELibrary(n_grid=64, length=22.0, poly_order=2, deriv_order=4)
    true_coeffs = true_system.coefficients(library)
    
    imp_system = KSSystem(beta=0.0, nu_2=1.0, nu_4=1.0) # Missing advection
    dt_fit, n_samples = 0.01, 400
    obs_prev, obs_curr, _, _ = sample_ks_observations(true_system, n_samples, dt_fit)
    
    medida = MEDIDA(imp_system, library, dt=dt_fit, significance=1e-4)
    result = medida.fit(obs_prev, obs_curr)
    cor_coeffs = result.corrected_coefficients(imp_system.coefficients(library))
    # Hovmoller Plots
    t_end, dt_plot = 20.0, 0.1
    n_plot = int(t_end / dt_plot)
    u0 = np.cos(np.linspace(0, 2*np.pi, 64)) * (1 + np.sin(np.linspace(0, 2*np.pi, 64)))
    
    traj_true = true_system.trajectory(u0, dt_plot, n_plot)
    traj_imp = imp_system.trajectory(u0, dt_plot, n_plot)
    
    # Corrected model manual integration (avoiding PolynomialODE dim check for PDE)
    u = u0.copy()
    traj_cor = [u]
    for _ in range(n_plot):
        for _ in range(4): # 4 substeps
            phi = library.transform(u)
            du = (phi @ cor_coeffs).reshape(64)
            u = u + (dt_plot/4) * du
        traj_cor.append(u)
    traj_cor = np.array(traj_cor)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = ["True (KS Chaos)", "Imperfect (No Advection)", "MEDIDA Corrected"]
    trajs = [traj_true, traj_imp, traj_cor]
    
    for ax, traj, title in zip(axes, trajs, titles):
        im = ax.imshow(traj, aspect='auto', extent=[0, 22, t_end, 0], cmap='magma')
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("x (Space)")
        if ax == axes[0]: ax.set_ylabel("t (Time)")
        plt.colorbar(im, ax=ax)
        
    plt.tight_layout()
    plt.savefig("outputs/figures/ks_pde_hovmoller.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    example_sir_vs_sirs()
    example_hidden_e()
    example_lorenz()
    example_ks()
