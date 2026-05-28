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
    """Apply a standardized plotting theme for research publications.

    Sets font families, sizes, and axis styles for consistent visual output.
    """
    sns.set_theme(style="ticks", context="talk", font_scale=1.2)
    plt.rcParams.update(
        {
            "axes.spines.right": False,
            "axes.spines.top": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "figure.dpi": 200,
            "savefig.bbox": "tight",
            "legend.frameon": False,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "lines.linewidth": 3.5,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
        }
    )


def save_latex_correction(
    coeffs, feature_names, state_names, metrics, filename, title="Discovered Correction"
):
    """Export the discovered model correction as a LaTeX-formatted file."""
    latex_code = format_latex_system(coeffs, feature_names, state_names)
    with open(filename, "w") as f:
        f.write(f"% {title}\n")
        f.write("% Metrics: " + ", ".join([f"{k}: {v}" for k, v in metrics.items()]) + "\n\n")
        f.write(latex_code)
        f.write("\n")


def save_discovery_card(coeffs, feature_names, state_names, metrics, filename, title=""):
    """Generate a summary PNG card displaying the discovered system equations.

    Renders plain text equations in a monospace font for high reliability.
    """
    apply_publication_theme()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")

    eq_text = format_system(coeffs, feature_names, state_names, precision=4)

    # Render discovery box with monospace font for equation alignment
    plt.text(
        0.5,
        0.55,
        eq_text,
        ha="center",
        va="center",
        fontsize=18,
        family="monospace",
        transform=ax.transAxes,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=1.5", fc="#ffffff", ec="#111111", lw=2),
    )

    if title:
        plt.text(
            0.5,
            0.95,
            title.upper(),
            ha="center",
            transform=ax.transAxes,
            fontsize=16,
            fontweight="black",
            color="#111111",
        )

    # Metadata footer for metrics
    m_text = "  •  ".join([f"{k.upper()}: {v}" for k, v in metrics.items()])
    plt.text(
        0.5,
        0.05,
        m_text,
        ha="center",
        transform=ax.transAxes,
        fontsize=14,
        color="#444444",
        fontweight="bold",
    )

    plt.savefig(filename, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close()


def run_medida_experiment(
    name,
    true_system,
    library,
    true_coeffs,
    imperfect_coeffs,
    u0,
    alpha=None,
    t_end=50.0,
    dt_fit=0.01,
    n_samples=1000,
    sigma_obs=0.0,
    seed=0,
    noise_seed=1,
    significance=1e-3,
    ridge=0.0,
    output_dir="outputs/synthetic/misc",
    generate_card=False,
):
    """Execute a single MEDIDA discovery experiment and save diagnostic results.

    Includes data generation, fitting, metric calculation, and visualization.
    """
    os.makedirs(output_dir, exist_ok=True)
    apply_publication_theme()

    obs_prev, obs_curr, _, _ = sample_simplex_observations(
        true_system,
        n_samples,
        dt_fit,
        seed=seed,
        sigma_obs=sigma_obs,
        noise_seed=noise_seed,
        alpha=alpha,
    )

    imperfect_model = PolynomialODE(imperfect_coeffs, library, state_names=true_system.state_names)
    from medida import RelevanceVectorMachine, RidgeRVM

    rvm = RidgeRVM(ridge=ridge) if ridge > 0 else RelevanceVectorMachine()
    medida = MEDIDA(imperfect_model, library, rvm=rvm, dt=dt_fit, significance=significance)
    result = medida.fit(obs_prev, obs_curr)

    corrected_coeffs = result.corrected_coefficients(imperfect_coeffs)
    corrected_model = PolynomialODE(corrected_coeffs, library, state_names=true_system.state_names)

    # Calculate discovery metrics
    eps_m = coefficient_error(true_coeffs, imperfect_coeffs)
    eps_star = coefficient_error(true_coeffs, corrected_coeffs)
    n_steps = int(t_end / 0.1)
    # trajectory integration uses increased substeps for stability verification
    true_traj = true_system.trajectory(u0, 0.1, n_steps, substeps=16)
    cor_traj = corrected_model.trajectory(u0, 0.1, n_steps, substeps=16)

    err_cor = float(relative_error(true_traj, cor_traj))
    improvement = eps_m / eps_star if eps_star > 1e-15 else float("inf")
    metrics = {"Improvement": f"{improvement:.1f}x", "L2 Error": f"{err_cor:.1e}"}

    save_latex_correction(
        result.error_coefficients,
        library.feature_names,
        true_system.state_names,
        metrics,
        os.path.join(output_dir, "discovery.tex"),
        title=name,
    )

    if generate_card:
        save_discovery_card(
            result.error_coefficients,
            library.feature_names,
            true_system.state_names,
            metrics,
            os.path.join(output_dir, "discovery_card.png"),
            title=f"Correction: {name}",
        )

    # Generate multi-panel diagnostic figure
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    colors = ["#111111", "#e31a1c", "#33a02c"]
    t = np.linspace(0, t_end, n_steps + 1)

    # Panel 1: Time-series comparison of infectious rates
    ax = axes[0]
    if true_system.dim >= 2:
        ax.plot(t, true_traj[:, 1], "-", color=colors[0], lw=4, label="TRUTH")
        ax.plot(t, cor_traj[:, 1], "--", color=colors[2], lw=4, label="MEDIDA")
    ax.set_title("TRAJECTORY RECOVERY")
    ax.set_xlabel("TIME (DAYS)")
    ax.set_ylabel("INFECTIOUS RATE")
    ax.legend(loc="upper right")

    # Panel 2: Phase manifold reconstruction
    ax = axes[1]
    if true_system.dim >= 2:
        ax.plot(true_traj[:, 0], true_traj[:, 1], color=colors[0], alpha=0.15, lw=2.5)
        ax.plot(cor_traj[:, 0], cor_traj[:, 1], "--", color=colors[2], lw=4, label="RECOVERED")
    ax.set_title("PHASE MANIFOLD")
    ax.set_xlabel("SUSCEPTIBLE")
    ax.set_ylabel("INFECTIOUS")

    # Panel 3: Temporal L2-norm error evolution
    ax = axes[2]
    res_cor = np.linalg.norm(cor_traj - true_traj, axis=-1)
    ax.semilogy(t, res_cor, "-", color=colors[2], lw=3)
    ax.set_title("DISCOVERY ACCURACY")
    ax.set_xlabel("TIME")
    ax.set_ylabel("L2 NORM ERROR")

    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "diagnostic_plot.png"), dpi=200)
    plt.close()
    return result


def _ps_calibrate_sigma(true_sys, alpha, dt, noise_frac, n_probe=300):
    """Calibrate observation noise level relative to state variability."""
    obs_prev, obs_curr, _, _ = sample_simplex_observations(
        true_sys, n_samples=n_probe, dt=dt, seed=99999, sigma_obs=0.0, alpha=alpha
    )
    sigma_u = float(np.mean(np.std(np.vstack([obs_prev, obs_curr]), axis=0)))
    return noise_frac * sigma_u


def run_parameter_sweep(model_type, grid_size=13, n_seeds=4, noise="free"):
    """Execute a global parameter sweep to test algorithm robustness.

    Iterates over beta and another model parameter (gamma, xi, etc.) to
    map discovery performance across a parameter space.
    """
    from medida import (
        SIRSystem,
        SIRSSystem,
        SIRDSystem,
        SIRNonlinearSystem,
        PolynomialLibrary,
        SaturatedSIRLibrary,
        RelevanceVectorMachine,
        RidgeRVM,
        MEDIDA,
        PolynomialODE,
        coefficient_error,
    )

    bg = np.linspace(0.20, 1.00, grid_size)
    if model_type == "sir":
        pg = np.linspace(0.05, 0.45, grid_size)
        var_names = ["S", "I", "R"]
        n_vars = 3
    elif model_type == "sirs":
        pg = np.linspace(0.01, 0.30, grid_size)
        var_names = ["S", "I", "R"]
        n_vars = 3
    elif model_type == "sird":
        pg = np.linspace(0.00, 0.20, grid_size)
        var_names = ["S", "I", "R", "D"]
        n_vars = 4
    else:
        pg = np.linspace(0.5, 20.0, grid_size)
        n_vars = 3

    em = np.full((grid_size, grid_size, n_seeds), np.nan)
    es = np.full_like(em, np.nan)
    dt = 0.01 if noise == "free" else 0.02
    nsamp = 400 if noise == "free" else 500
    ridge = 1e-2 if noise == "noisy" else 0.0
    alpha = np.array([3.0, 0.8, 2.0] if n_vars == 3 else [3.0, 0.8, 2.0, 0.2])

    for i, beta in enumerate(bg):
        for j, p_val in enumerate(pg):
            if model_type == "sir":
                sys_t = SIRSystem(beta=beta, gamma=p_val)
                lib = PolynomialLibrary(n_vars=3, degree=2, var_names=var_names)
                c_t = sys_t.coefficients(lib)
                c_i = c_t.copy()
                c_i[lib.index("I"), 2] = 0.0  # missing recovery
            elif model_type == "sirs":
                gamma = 0.18
                sys_t = SIRSSystem(beta=beta, gamma=gamma, xi=p_val)
                lib = PolynomialLibrary(n_vars=3, degree=2, var_names=var_names)
                c_t = sys_t.coefficients(lib)
                c_i = np.zeros_like(c_t)
                c_i[lib.index("S I"), 0], c_i[lib.index("S I"), 1] = -beta, beta
                c_i[lib.index("I"), 1], c_i[lib.index("I"), 2] = -gamma, gamma
            elif model_type == "sird":
                sys_t = SIRDSystem(beta=0.6, gamma=beta, mu=p_val)
                lib = PolynomialLibrary(n_vars=4, degree=2, var_names=var_names)
                c_t = sys_t.coefficients(lib)
                c_i = c_t.copy()
                c_i[lib.index("I"), 3] = 0.0
                c_i[lib.index("I"), 2] = beta + p_val
            else:
                sys_t = SIRNonlinearSystem(beta=beta, gamma=0.18, a=p_val)
                lib = SaturatedSIRLibrary(a=p_val)
                c_t = sys_t.coefficients(lib)
                c_i = np.zeros((lib.n_features, 3))
                c_i[lib.index("S I"), 0], c_i[lib.index("S I"), 1] = -beta, beta
                c_i[lib.index("I"), 1], c_i[lib.index("I"), 2] = -0.18, 0.18

            sigma = _ps_calibrate_sigma(sys_t, alpha, dt, 0.003) if noise == "noisy" else 0.0
            for s in range(n_seeds):
                obs_prev, obs_curr, _, _ = sample_simplex_observations(
                    sys_t, nsamp, dt, seed=2000 + s, sigma_obs=sigma, alpha=alpha
                )
                rvm = (
                    RidgeRVM(ridge=ridge, t_min=3.0, threshold=0.02)
                    if noise == "noisy"
                    else RelevanceVectorMachine(t_min=2.0, threshold=0.01)
                )
                med = MEDIDA(
                    PolynomialODE(c_i, lib),
                    lib,
                    rvm=rvm,
                    dt=dt,
                    significance=1e-4 if noise == "noisy" else 1e-6,
                )
                res = med.fit(obs_prev, obs_curr)
                em[i, j, s] = coefficient_error(c_t, c_i)
                es[i, j, s] = coefficient_error(c_t, res.corrected_coefficients(c_i))

    return bg, pg, em, es


def plot_meta_heatmaps(sweep_data, output_path):
    """Generate heatmaps of discovery performance across parameter sweeps."""
    apply_publication_theme()
    fig, axes = plt.subplots(4, 2, figsize=(18, 26))

    for row, (title, g0, g1, em_f, es_f, em_n, es_n, xl, yl) in enumerate(sweep_data):
        for col, (label, em, es) in enumerate([("free", em_f, es_f), ("noisy", em_n, es_n)]):
            ax = axes[row, col]
            impr_mean = np.nanmean(em, axis=2) / np.clip(np.nanmean(es, axis=2), 1e-12, None)
            log_impr = np.log10(np.clip(impr_mean, 1e-3, None))

            im = ax.imshow(
                log_impr.T,
                origin="lower",
                aspect="auto",
                extent=[g0[0], g0[-1], g1[0], g1[-1]],
                cmap="RdYlGn",
                vmin=-1,
                vmax=4,
            )
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.set_title(f"{title}\n[{label.upper()}]", fontsize=14)
            plt.colorbar(im, ax=ax)

    fig.suptitle("MEDIDA PERFORMANCE STRESS-TEST", fontweight="black", fontsize=26, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_robustness_heatmaps(sweep_data, output_path):
    """Generate heatmaps showing the coefficient of variation (CV) across random seeds."""
    apply_publication_theme()
    fig, axes = plt.subplots(2, 4, figsize=(24, 11))

    for col, (title, g0, g1, em_f, es_f, em_n, es_n, xl, yl) in enumerate(sweep_data):
        short_title = title.split("truth")[0].strip()
        for row, (label, es) in enumerate([("free", es_f), ("noisy", es_n)]):
            ax = axes[row, col]
            mean_e = np.nanmean(es, axis=2)
            std_e = np.nanstd(es, axis=2)
            cv = std_e / np.clip(mean_e, 1e-12, None)

            im = ax.imshow(
                cv.T,
                origin="lower",
                aspect="auto",
                extent=[g0[0], g0[-1], g1[0], g1[-1]],
                cmap="viridis",
                vmin=0,
                vmax=2,
            )
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.set_title(f"{short_title} [{label.upper()}]")
            plt.colorbar(im, ax=ax)

    fig.suptitle("ROBUSTNESS (LOWER IS BETTER)", fontweight="black", fontsize=26, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300)
    plt.close()
