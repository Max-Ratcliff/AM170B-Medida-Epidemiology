import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from medida import (
    sample_simplex_observations, 
    MEDIDA, 
    PolynomialODE, 
    coefficient_error,
    relative_error,
    format_system,
)
from medida.metrics import format_latex_system

def apply_publication_theme():
    """Apply a premium 'Presentation-Grade' theme (16pt baseline)."""
    sns.set_theme(style="ticks", context="talk", font_scale=1.2)
    plt.rcParams.update({
        "axes.spines.right": False,
        "axes.spines.top": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "legend.frameon": False,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "lines.linewidth": 3.5,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18
    })

def save_latex_correction(coeffs, feature_names, state_names, metrics, filename, title="Discovered Correction"):
    """Export raw LaTeX for manual compilation."""
    latex_code = format_latex_system(coeffs, feature_names, state_names)
    with open(filename, "w") as f:
        f.write(f"% {title}\n")
        f.write("% Metrics: " + ", ".join([f"{k}: {v}" for k, v in metrics.items()]) + "\n\n")
        f.write(latex_code)
        f.write("\n")

def save_discovery_card(coeffs, feature_names, state_names, metrics, filename, title=""):
    """Render a high-end visual card using Matplotlib's internal TeX rendering."""
    apply_publication_theme()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis('off')
    
    # We use format_latex_system but wrap in $ $ for matplotlib's mathtext
    # Simple substitution for some environments that matplotlib doesn't support
    latex_code = format_latex_system(coeffs, feature_names, state_names)
    clean_latex = latex_code.replace("\\begin{aligned}", "").replace("\\end{aligned}", "").replace("&", "")
    
    # Render with mathtext
    plt.text(0.5, 0.5, f"${clean_latex}$", ha='center', va='center', fontsize=20, 
             transform=ax.transAxes, fontweight='bold',
             bbox=dict(boxstyle="round,pad=1.5", fc="#ffffff", ec="#333333", lw=2))
    
    if title:
        plt.text(0.5, 0.94, title.upper(), ha='center', transform=ax.transAxes, 
                 fontsize=16, fontweight='black', color="#111111")
    
    # Metric Footer (Slide friendly)
    m_parts = [f"\\mathbf{{{k.upper()}}}: {v}" for k, v in metrics.items()]
    m_text = " \\quad \\bullet \\quad ".join(m_parts)
    plt.text(0.5, 0.05, f"${m_text}$", ha='center', transform=ax.transAxes, 
             fontsize=13, color="#555555")
    
    plt.savefig(filename, dpi=300, facecolor='white')
    plt.close()

def run_medida_experiment(name, true_system, library, true_coeffs, imperfect_coeffs, 
                          u0, alpha=None, t_end=50.0, dt_fit=0.01, 
                          n_samples=1000, sigma_obs=0.0, seed=0, 
                          noise_seed=1, significance=1e-3, ridge=0.0,
                          output_dir="outputs/synthetic/misc",
                          generate_card=True):
    """Experiment runner with premium visuals and high-contrast lines."""
    os.makedirs(output_dir, exist_ok=True)
    apply_publication_theme()
    
    obs_prev, obs_curr, _, _ = sample_simplex_observations(
        true_system, n_samples, dt_fit, seed=seed, sigma_obs=sigma_obs, 
        noise_seed=noise_seed, alpha=alpha)

    imperfect_model = PolynomialODE(imperfect_coeffs, library, state_names=true_system.state_names)
    from medida import RelevanceVectorMachine, RidgeRVM
    rvm = RidgeRVM(ridge=ridge) if ridge > 0 else RelevanceVectorMachine()
    medida = MEDIDA(imperfect_model, library, rvm=rvm, dt=dt_fit, significance=significance)
    result = medida.fit(obs_prev, obs_curr)
    
    corrected_coeffs = result.corrected_coefficients(imperfect_coeffs)
    corrected_model = PolynomialODE(corrected_coeffs, library, state_names=true_system.state_names)

    # Performance
    eps_m = coefficient_error(true_coeffs, imperfect_coeffs)
    eps_star = coefficient_error(true_coeffs, corrected_coeffs)
    n_steps = int(t_end / 0.1)
    true_traj = true_system.trajectory(u0, 0.1, n_steps, substeps=8)
    cor_traj = corrected_model.trajectory(u0, 0.1, n_steps, substeps=8)
    err_cor = relative_error(true_traj, cor_traj)

    metrics = {"Improvement": f"{eps_m/eps_star:.1f}x", "L2 Error": f"{err_cor:.1e}"}
    
    # Save Artifacts
    save_latex_correction(result.error_coefficients, library.feature_names, true_system.state_names, 
                          metrics, os.path.join(output_dir, "discovery.tex"), title=name)
    
    if generate_card:
        save_discovery_card(result.error_coefficients, library.feature_names, true_system.state_names, 
                            metrics, os.path.join(output_dir, "discovery_card.png"), title=f"Correction: {name}")

    # Diagnostic Plot (High Contrast / Wide Aspect)
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    colors = ["#111111", "#e31a1c", "#33a02c"] # High-contrast Black, Red, Green
    t = np.linspace(0, t_end, n_steps + 1)
    
    # 1. Trajectory
    ax = axes[0]
    if true_system.dim >= 2:
        ax.plot(t, true_traj[:, 1], '-', color=colors[0], lw=4, label='TRUTH')
        ax.plot(t, cor_traj[:, 1], '--', color=colors[2], lw=4, label='MEDIDA')
    ax.set_title("TRAJECTORY RECOVERY")
    ax.set_xlabel("TIME (DAYS)"); ax.set_ylabel("INFECTIOUS RATE")
    ax.legend(loc='upper right')

    # 2. Phase Space
    ax = axes[1]
    if true_system.dim >= 2:
        ax.plot(true_traj[:, 0], true_traj[:, 1], color=colors[0], alpha=0.15, lw=2.5)
        ax.plot(cor_traj[:, 0], cor_traj[:, 1], '--', color=colors[2], lw=4, label='RECOVERED')
    ax.set_title("PHASE MANIFOLD")
    ax.set_xlabel("SUSCEPTIBLE"); ax.set_ylabel("INFECTIOUS")

    # 3. Log-Error
    ax = axes[2]
    res_cor = np.linalg.norm(cor_traj - true_traj, axis=-1)
    ax.semilogy(t, res_cor, '-', color=colors[2], lw=3)
    ax.set_title("DISCOVERY ACCURACY")
    ax.set_xlabel("TIME"); ax.set_ylabel("L2 NORM ERROR")
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "diagnostic_plot.png"), dpi=200)
    plt.close()
    return result
