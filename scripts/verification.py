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

from utils import run_medida_experiment, save_discovery_card, save_latex_correction, apply_publication_theme
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
        "Noise-free SIRS",
        true_system, library, true_coeffs, imperfect_coeffs,
        u0=np.array([0.98, 0.02, 0.0]), alpha=np.array([3.0, 0.8, 2.0]),
        dt_fit=0.01, n_samples=800, sigma_obs=0.0,
        output_dir="outputs/synthetic/sirs/noise_free"
    )

def example_hidden_e():
    output_dir = "outputs/synthetic/hidden_e"
    os.makedirs(output_dir, exist_ok=True)
    apply_publication_theme()
    
    true_system = SEIRSystem(beta=0.6, sigma=0.2, gamma=0.18)
    library = PolynomialLibrary(n_vars=3, degree=2, var_names=["S", "I", "R"])
    imperfect_system = SIRSystem(beta=0.6, gamma=0.18)
    imperfect_coeffs = imperfect_system.coefficients(library)
    imperfect_model = PolynomialODE(imperfect_coeffs, library, state_names=["S", "I", "R"])
    
    dt_fit, n_samples = 0.01, 1200
    obs_prev, obs_curr, _, _ = sample_hidden_E_seir_observations(
        true_system, n_samples=n_samples, dt=dt_fit, seed=404
    )
    
    medida = MEDIDA(imperfect_model, library, dt=dt_fit, significance=1e-6)
    result = medida.fit(obs_prev, obs_curr)
    corrected_coeffs = result.corrected_coefficients(imperfect_coeffs)
    corrected_model = PolynomialODE(corrected_coeffs, library, state_names=["S", "I", "R"])
    
    # Premium Card
    save_discovery_card(result.error_coefficients, library.feature_names, ["S", "I", "R"], 
                       {"Discovery": "Success", "Type": "Hidden Variable"}, 
                       os.path.join(output_dir, "discovery_card.png"), title="Hidden E Discovery")
    
    u0_seir = np.array([0.98, 0.01, 0.01, 0.0])
    u0_sir = u0_seir[[0, 2, 3]]
    t_end, dt = 50.0, 0.1
    n_steps = int(t_end / dt)
    
    true_traj = true_system.trajectory(u0_seir, dt=dt, n_steps=n_steps, substeps=8)[:, [0, 2, 3]]
    cor_traj = corrected_model.trajectory(u0_sir, dt=dt, n_steps=n_steps, substeps=8)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    t = np.linspace(0, t_end, n_steps + 1)
    axes[0].plot(t, true_traj[:, 1], 'k-', lw=4, label='TRUTH (SEIR)')
    axes[0].plot(t, cor_traj[:, 1], '--', color='#33a02c', lw=4, label='MEDIDA (SIR)')
    axes[0].set_title("TRAJECTORY RECOVERY", fontweight="black")
    axes[0].legend()
    
    axes[1].plot(true_traj[:, 0], true_traj[:, 1], 'k-', alpha=0.2, lw=2)
    axes[1].plot(cor_traj[:, 0], cor_traj[:, 1], '--', color='#33a02c', lw=4, label='CORRECTED')
    axes[1].set_title("PHASE SPACE RECOVERY", fontweight="black")
    axes[1].set_xlabel("S"); axes[1].set_ylabel("I")
    
    sns.despine()
    plt.savefig(os.path.join(output_dir, "diagnostic_plot.png"))
    plt.close()

def example_lorenz():
    output_dir = "outputs/synthetic/lorenz"
    os.makedirs(output_dir, exist_ok=True)
    apply_publication_theme()
    
    true_system = Lorenz63()
    library = PolynomialLibrary(n_vars=3, degree=2, var_names=["x", "y", "z"])
    true_coeffs = true_system.coefficients(library)
    imperfect_coeffs = true_coeffs.copy()
    imperfect_coeffs[library.index("x z"), 1] = 0.0
    imperfect_model = PolynomialODE(imperfect_coeffs, library, state_names=["x", "y", "z"])
    
    dt_fit, n_samples = 0.01, 2000
    obs_prev, obs_curr, _, _ = sample_observations(true_system, n_samples, dt_fit, spinup=20.0, seed=42)
    
    medida = MEDIDA(imperfect_model, library, dt=dt_fit, significance=1e-5)
    result = medida.fit(obs_prev, obs_curr)
    
    save_discovery_card(result.error_coefficients, library.feature_names, ["x", "y", "z"], 
                       {"Regime": "Chaotic", "System": "Lorenz-63"}, 
                       os.path.join(output_dir, "discovery_card.png"), title="Chaos Discovery")

    cor_model = PolynomialODE(result.corrected_coefficients(imperfect_coeffs), library)
    u0 = np.array([1.0, 1.0, 1.0])
    traj_true = true_system.trajectory(u0, 0.01, 4000)
    traj_cor = cor_model.trajectory(u0, 0.01, 4000)
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj_true[:, 0], traj_true[:, 1], traj_true[:, 2], 'k-', alpha=0.15, lw=1.5)
    ax.plot(traj_cor[:, 0], traj_cor[:, 1], traj_cor[:, 2], color='#33a02c', alpha=0.8, lw=2.5, label='RECOVERED')
    ax.set_axis_off()
    ax.set_title("RECOVERING THE BUTTERFLY ATTRACTOR", fontsize=20, fontweight="black")
    plt.savefig(os.path.join(output_dir, "attractor_3d.png"), dpi=300)
    plt.close()

def example_ks():
    output_dir = "outputs/synthetic/ks_pde"
    os.makedirs(output_dir, exist_ok=True)
    apply_publication_theme()
    
    true_system = KSSystem(beta=1.0)
    library = PDELibrary(n_grid=64, length=22.0, poly_order=2, deriv_order=4)
    true_coeffs = true_system.coefficients(library)
    imp_system = KSSystem(beta=0.0)
    
    dt_fit, n_samples = 0.01, 400
    obs_prev, obs_curr, _, _ = sample_ks_observations(true_system, n_samples, dt_fit)
    medida = MEDIDA(imp_system, library, dt=dt_fit, significance=1e-4)
    result = medida.fit(obs_prev, obs_curr)
    
    # LaTeX Only for KS
    save_latex_correction(result.error_coefficients, library.feature_names, ["u"], 
                          {"Experiment": "KS PDE"}, os.path.join(output_dir, "discovery.tex"))

    t_end, dt_plot = 50.0, 0.1
    n_plot = int(t_end / dt_plot)
    u0 = np.cos(np.linspace(0, 2*np.pi, 64, endpoint=False)) * (1 + np.sin(np.linspace(0, 2*np.pi, 64, endpoint=False)))
    
    traj_true = true_system.trajectory(u0, dt_plot, n_plot)
    traj_imp = imp_system.trajectory(u0, dt_plot, n_plot)
    cor_model = PolynomialODE(result.corrected_coefficients(imp_system.coefficients(library)), library)
    traj_cor = cor_model.trajectory(u0, dt_plot, n_plot, substeps=10)

    # Calculate global vmin/vmax for unified scale
    v_min = min(traj_true.min(), traj_cor.min())
    v_max = max(traj_true.max(), traj_cor.max())

    fig, axes = plt.subplots(1, 3, figsize=(22, 8), sharey=True)
    trajs = [traj_true, traj_imp, traj_cor]
    titles = ["GROUND TRUTH", "IMPERFECT (NO ADVECTION)", "MEDIDA RECONSTRUCTION"]
    
    for ax, traj, title in zip(axes, trajs, titles):
        im = ax.imshow(traj, aspect='auto', extent=[0, 22, t_end, 0], 
                       cmap='inferno', origin='upper', vmin=v_min, vmax=v_max)
        ax.set_title(title, fontweight="black", fontsize=16)
        ax.set_xlabel("SPACE (x)")
        if ax == axes[0]: ax.set_ylabel("TIME (t)")
        
    # Shared Colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Field Amplitude $u(x,t)$")
        
    plt.savefig(os.path.join(output_dir, "hovmoller_comparison.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    example_sir_vs_sirs()
    example_hidden_e()
    example_lorenz()
    example_ks()
