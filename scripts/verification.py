import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Ensure project root is in path for module imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.utils import (
    run_medida_experiment,
    save_discovery_card,
    save_latex_correction,
    apply_publication_theme,
    run_parameter_sweep,
    plot_meta_heatmaps,
    plot_robustness_heatmaps,
)
from medida import (
    SIRSystem,
    SIRSSystem,
    SIRDSystem,
    SIRNonlinearSystem,
    Lorenz63,
    KSSystem,
    SEIRSystem,
    ProjectedSIRFromSEIRSystem,
    PolynomialLibrary,
    SaturatedSIRLibrary,
    PDELibrary,
    PolynomialODE,
    RelevanceVectorMachine,
    MEDIDA,
    RidgeRVM,
    EnsembleKalmanFilter,
    format_system,
    sample_ks_observations,
    coefficient_error,
    relative_error,
    sample_hidden_E_seir_observations,
    sample_observations,
    sample_simplex_observations,
)


def example_sir_vs_sirs():
    """Verify MEDIDA's ability to recover waning immunity (xi) from a standard SIR model."""
    true_system = SIRSSystem(beta=0.6, gamma=0.18, xi=0.08)
    library = PolynomialLibrary(n_vars=3, degree=2, var_names=["S", "I", "R"])
    true_coeffs = true_system.coefficients(library)

    # Imperfect model assumes standard SIR (no waning immunity)
    imperfect_system = SIRSystem(beta=0.6, gamma=0.18)
    imperfect_coeffs = imperfect_system.coefficients(library)

    run_medida_experiment(
        "Noise-free SIRS",
        true_system,
        library,
        true_coeffs,
        imperfect_coeffs,
        u0=np.array([0.98, 0.02, 0.0]),
        alpha=np.array([3.0, 0.8, 2.0]),
        dt_fit=0.01,
        n_samples=800,
        sigma_obs=0.0,
        output_dir="outputs/synthetic/sirs/noise_free",
    )

    run_medida_experiment(
        "Noisy SIRS",
        true_system,
        library,
        true_coeffs,
        imperfect_coeffs,
        u0=np.array([0.98, 0.02, 0.0]),
        alpha=np.array([3.0, 0.8, 2.0]),
        dt_fit=0.05,
        n_samples=2000,
        sigma_obs=0.002,
        noise_seed=2,
        significance=1e-4,
        ridge=1e-2,
        output_dir="outputs/synthetic/sirs/noisy",
    )


def example_sird():
    """Verify recovery of mortality (mu) when missing from a 3-compartment SIR model."""
    true_system = SIRDSystem(beta=0.6, gamma=0.14, mu=0.04)
    library = PolynomialLibrary(n_vars=4, degree=2, var_names=["S", "I", "R", "D"])
    true_coeffs = true_system.coefficients(library)

    # Imperfect model: removal rate (gamma + mu) is lumped into the recovery (R) compartment
    imperfect_coeffs = np.zeros_like(true_coeffs)
    imperfect_coeffs[library.index("S I"), 0] = -true_system.beta
    imperfect_coeffs[library.index("S I"), 1] = true_system.beta
    imperfect_coeffs[library.index("I"), 1] = -(true_system.gamma + true_system.mu)
    imperfect_coeffs[library.index("I"), 2] = true_system.gamma + true_system.mu

    run_medida_experiment(
        "Noise-free SIRD",
        true_system,
        library,
        true_coeffs,
        imperfect_coeffs,
        u0=np.array([0.98, 0.02, 0.0, 0.0]),
        alpha=np.array([3.0, 0.8, 2.0, 0.2]),
        dt_fit=0.01,
        n_samples=800,
        sigma_obs=0.0,
        seed=211,
        significance=1e-6,
        output_dir="outputs/synthetic/sird/noise_free",
    )

    run_medida_experiment(
        "Noisy SIRD",
        true_system,
        library,
        true_coeffs,
        imperfect_coeffs,
        u0=np.array([0.98, 0.02, 0.0, 0.0]),
        alpha=np.array([3.0, 0.8, 2.0, 0.2]),
        dt_fit=0.05,
        n_samples=2000,
        sigma_obs=0.002,
        seed=212,
        noise_seed=213,
        significance=1e-4,
        ridge=1e-2,
        output_dir="outputs/synthetic/sird/noisy",
    )


def example_nonlinear_sir():
    """Verify recovery of saturated incidence (Holling Type II) using a custom library."""
    true_system = SIRNonlinearSystem(beta=0.6, gamma=0.18, a=8.0)
    library = SaturatedSIRLibrary(a=true_system.a)
    true_coeffs = true_system.coefficients(library)

    # Imperfect model assumes linear mass-action kinetics
    imperfect_coeffs = np.zeros_like(true_coeffs)
    imperfect_coeffs[library.index("S I"), 0] = -true_system.beta
    imperfect_coeffs[library.index("S I"), 1] = true_system.beta
    imperfect_coeffs[library.index("I"), 1] = -true_system.gamma
    imperfect_coeffs[library.index("I"), 2] = true_system.gamma

    run_medida_experiment(
        "Noise-free Nonlinear SIR",
        true_system,
        library,
        true_coeffs,
        imperfect_coeffs,
        u0=np.array([0.98, 0.02, 0.0]),
        alpha=np.array([3.0, 0.8, 2.0]),
        dt_fit=0.01,
        n_samples=800,
        sigma_obs=0.0,
        output_dir="outputs/synthetic/nonlinear_sir/noise_free",
    )


def example_seir_from_sir():
    """Demonstrate recovery of unobserved 'Exposed' dynamics from SIR-only data."""
    output_dir = "outputs/synthetic/seir_from_sir"
    os.makedirs(output_dir, exist_ok=True)
    apply_publication_theme()

    true_system = SEIRSystem(beta=0.6, sigma=0.2, gamma=0.18)
    library = PolynomialLibrary(n_vars=3, degree=2, var_names=["S", "I", "R"])

    # Baseline SIR model uses linear S-I incidence
    sir_coeffs = np.zeros((library.n_features, 3))
    sir_coeffs[library.index("S I"), 0] = -0.6
    sir_coeffs[library.index("S I"), 1] = 0.6
    sir_coeffs[library.index("I"), 1] = -0.18
    sir_coeffs[library.index("I"), 2] = 0.18
    imperfect_model = PolynomialODE(sir_coeffs, library, state_names=["S", "I", "R"])

    # Generate 4D SEIR data but only expose S, I, R for fitting
    obs_prev_3d, obs_curr_3d, truth_prev_4d, truth_curr_4d = sample_hidden_E_seir_observations(
        true_system, n_samples=1500, dt=0.05, sigma_obs=0.00
    )

    medida = MEDIDA(imperfect_model, library, dt=0.05, significance=1e-5)
    result = medida.fit(obs_prev_3d, obs_curr_3d)

    # Plot true vs corrected infectious curve
    u0 = np.array([0.99, 0.005, 0.005, 0.0])  # S, E, I, R
    n_steps = 400
    true_traj_4d = true_system.trajectory(u0, 0.1, n_steps, substeps=16)

    cor_coeffs = result.corrected_coefficients(sir_coeffs)
    cor_model = PolynomialODE(cor_coeffs, library, state_names=["S", "I", "R"])
    cor_traj_3d = cor_model.trajectory(u0[[0, 2, 3]], 0.1, n_steps, substeps=16)

    imp_traj_3d = imperfect_model.trajectory(u0[[0, 2, 3]], 0.1, n_steps, substeps=16)

    plt.figure(figsize=(10, 6))
    plt.plot(true_traj_4d[:, 2], "k-", lw=5, label="True SEIR (I)")
    plt.plot(imp_traj_3d[:, 1], "r--", lw=3, label="Imperfect SIR")
    plt.plot(cor_traj_3d[:, 1], "g-", lw=4, label="MEDIDA (Corrected)")
    plt.title("SEIR RECOVERY: I Trajectory")
    plt.legend()
    plt.grid(alpha=0.2)
    sns.despine()
    plt.savefig(os.path.join(output_dir, "seir_correction.png"))
    plt.close()

    # Reconstruction of the 'Exposed' tendency discrepancy
    phi_all = library.transform(true_traj_4d[:, [0, 2, 3]])
    discovered_correction = phi_all @ result.error_coefficients
    true_exposed_outflow = true_system.sigma * true_traj_4d[:, 1]

    plt.figure(figsize=(10, 6))
    plt.plot(true_exposed_outflow, "k-", lw=5, label="True E->I Outflow")
    plt.plot(discovered_correction[:, 1], "g--", lw=4, label="Discovered dI correction")
    plt.title("SEIR RECOVERY: Exposed Compartment Outflow")
    plt.legend()
    plt.grid(alpha=0.2)
    sns.despine()
    plt.savefig(os.path.join(output_dir, "exposed_recovery.png"))
    plt.close()


def example_hidden_E():
    """Analyze impact of high observation noise on hidden compartment recovery."""
    output_dir = "outputs/synthetic/hidden_e"
    os.makedirs(output_dir, exist_ok=True)
    apply_publication_theme()

    true_system = SEIRSystem(beta=0.6, sigma=0.2, gamma=0.18)
    library = PolynomialLibrary(n_vars=3, degree=2, var_names=["S", "I", "R"])
    
    # Baseline SIR model
    sir_coeffs = np.zeros((library.n_features, 3))
    sir_coeffs[library.index("S I"), 0] = -0.6
    sir_coeffs[library.index("S I"), 1] = 0.6
    sir_coeffs[library.index("I"), 1] = -0.18
    sir_coeffs[library.index("I"), 2] = 0.18
    imperfect_model = PolynomialODE(sir_coeffs, library, state_names=["S", "I", "R"])

    # Generate 4D data, hide E, and add SIGNIFICANT noise
    obs_prev, obs_curr, truth_prev_4d, truth_curr_4d = sample_hidden_E_seir_observations(
        true_system, n_samples=2500, dt=0.05, sigma_obs=0.005, noise_seed=42
    )

    # Use RidgeRVM to handle noise during hidden variable discovery
    rvm = RidgeRVM(ridge=1e-2)
    medida = MEDIDA(imperfect_model, library, rvm=rvm, dt=0.05, significance=1e-3)
    result = medida.fit(obs_prev, obs_curr)
    
    # Generate diagnostic plots manually for the hidden variable case
    u0 = np.array([0.99, 0.005, 0.005, 0.0])
    n_steps = 400
    true_traj_4d = true_system.trajectory(u0, 0.1, n_steps, substeps=16)
    
    cor_coeffs = result.corrected_coefficients(sir_coeffs)
    cor_model = PolynomialODE(cor_coeffs, library, state_names=["S", "I", "R"])
    cor_traj_3d = cor_model.trajectory(u0[[0, 2, 3]], 0.1, n_steps, substeps=16)

    plt.figure(figsize=(10, 6))
    plt.plot(true_traj_4d[:, 2], 'k-', lw=5, label='True SEIR (I)')
    plt.plot(cor_traj_3d[:, 1], 'g--', lw=4, label='MEDIDA (Noisy Recovery)')
    plt.title("HIDDEN-E RECOVERY UNDER 0.5% NOISE")
    plt.legend(); plt.grid(alpha=0.2); sns.despine()
    plt.savefig(os.path.join(output_dir, "diagnostic_plot.png"))
    plt.close()



def example_lorenz():
    """Verify recovery of chaotic dynamics from sparse observations."""
    output_dir = "outputs/synthetic/lorenz"
    os.makedirs(output_dir, exist_ok=True)
    apply_publication_theme()

    true_system = Lorenz63()
    library = PolynomialLibrary(n_vars=3, degree=2, var_names=["x", "y", "z"])
    c_true = true_system.coefficients(library)

    # Imperfect model is missing the 'x*z' cross-term in the second dimension
    c_imperfect = c_true.copy()
    c_imperfect[library.index("x z"), 1] = 0.0
    imperfect_model = PolynomialODE(c_imperfect, library, state_names=["x", "y", "z"])

    # High-fidelity noise-free recovery
    obs_prev, obs_curr, _, _ = sample_observations(true_system, 1200, dt=1e-4)
    medida = MEDIDA(imperfect_model, library, dt=1e-4, significance=1e-4)
    result = medida.fit(obs_prev, obs_curr)

    c_star = result.corrected_coefficients(c_imperfect)
    cor_model = PolynomialODE(c_star, library)

    u0 = np.array([-8.0, 7.0, 27.0])
    # Full attractor for background reference
    t_traj_full = true_system.trajectory(u0, 0.01, 3000, substeps=8)
    # Short trajectory (within Lyapunov time ~5) so true and recovered overlap visibly
    n_short = 600
    t_traj = true_system.trajectory(u0, 0.01, n_short, substeps=8)
    c_traj = cor_model.trajectory(u0, 0.01, n_short, substeps=8)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        t_traj_full[:, 0],
        t_traj_full[:, 1],
        t_traj_full[:, 2],
        color="lightgray",
        alpha=0.3,
        lw=0.8,
        label="True attractor",
    )
    ax.plot(
        t_traj[:, 0],
        t_traj[:, 1],
        t_traj[:, 2],
        color="steelblue",
        alpha=0.9,
        lw=2,
        label="Ground truth (t=6)",
    )
    ax.plot(c_traj[:, 0], c_traj[:, 1], c_traj[:, 2], "g--", lw=2.5, label="MEDIDA recovered (t=6)")
    ax.set_xlabel("x", fontsize=12, labelpad=8)
    ax.set_ylabel("y", fontsize=12, labelpad=8)
    ax.set_zlabel("z", fontsize=12, labelpad=8)
    ax.set_title("LORENZ-63 ATTRACTOR RECOVERY", fontweight="black", pad=15)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "attractor_3d.png"), dpi=300, bbox_inches="tight")
    plt.close()


def example_lorenz_verification_summary():
    """Benchmark discovery precision across 8 structural perturbation cases."""
    output_dir = "outputs/synthetic/lorenz"
    os.makedirs(output_dir, exist_ok=True)
    apply_publication_theme()

    true_system = Lorenz63()
    library = PolynomialLibrary(n_vars=3, degree=2, var_names=["x", "y", "z"])
    c_true = true_system.coefficients(library)

    # Dictionary of test cases: (label, list of (feature_name, dimension, new_value))
    CASES = [
        ("missing x z", [("x z", 1, 0.0)]),
        ("missing x y", [("x y", 2, 0.0)]),
        ("missing sigma·y", [("y", 0, 0.0)]),
        ("wrong rho (20→28)", [("x", 1, 20.0)]),
        ("wrong sigma (4→10)", [("x", 0, -4.0), ("y", 0, 4.0)]),
        ("wrong beta (1→8/3)", [("z", 2, -1.0)]),
        ("extra z + miss xz", [("z", 0, 0.5), ("x z", 1, 0.0)]),
        ("wrong rho + miss xy", [("x", 1, 20.0), ("x y", 2, 0.0)]),
    ]

    def make_imperfect(mods):
        c = c_true.copy()
        for feat, comp, val in mods:
            c[library.index(feat), comp] = val
        return c

    # Step 1-2: Noise-free recovery benchmarking
    DT_FREE, N_FREE = 1e-4, 400
    free_results = []
    print("[lorenz] noise-free pass...")
    for label, mods in CASES:
        c_model = make_imperfect(mods)
        model = PolynomialODE(c_model, library)
        obs_prev, obs_curr, _, _ = sample_observations(true_system, N_FREE, dt=DT_FREE)
        medida = MEDIDA(
            model,
            library,
            dt=DT_FREE,
            significance=1e-3,
            rvm=RelevanceVectorMachine(t_min=2.0, threshold=0.02),
        )
        result = medida.fit(obs_prev, obs_curr)
        c_star = result.corrected_coefficients(c_model)
        eps_m = coefficient_error(c_true, c_model)
        eps_star = coefficient_error(c_true, c_star)
        free_results.append(dict(label=label, eps_m=eps_m, eps_star=eps_star))
        print(f"  {label:28s}  eps_m={eps_m*100:.2f}%  eps*={eps_star*100:.4f}%")

    # Step 3: Denoising benchmark with EnKF
    DT_NOISY, N_NOISY = 2e-3, 800
    NOISE_LEVEL, N_ENS, INFLATION = 0.01, 150, 1.05
    probe_prev, probe_curr, _, _ = sample_observations(true_system, 1000, dt=1e-3)
    sigma_u = float(np.mean(np.std(np.vstack([probe_prev, probe_curr]), axis=0)))
    sigma_obs = NOISE_LEVEL * sigma_u

    noisy_results = []
    print("[lorenz] noisy pass...")
    for label, mods in CASES:
        c_model = make_imperfect(mods)
        model = PolynomialODE(c_model, library)
        obs_prev, obs_curr, _, _ = sample_observations(
            true_system, N_NOISY, dt=DT_NOISY, sigma_obs=sigma_obs, noise_seed=7
        )

        # Baseline: MEDIDA without Data Assimilation
        med_nd = MEDIDA(model, library, dt=DT_NOISY, rvm=RelevanceVectorMachine(t_min=5.0))
        eps_nd = coefficient_error(
            c_true, med_nd.fit(obs_prev, obs_curr).corrected_coefficients(c_model)
        )

        # Proposed: MEDIDA with EnKF preprocessing
        enkf = EnsembleKalmanFilter(n_ensemble=N_ENS, inflation=INFLATION, seed=3)
        med_da = MEDIDA(
            model,
            library,
            dt=DT_NOISY,
            rvm=RelevanceVectorMachine(t_min=5.0),
            enkf=enkf,
            sigma_obs=sigma_obs,
        )
        res_da = med_da.fit(obs_prev, obs_curr)
        eps_da = coefficient_error(c_true, res_da.corrected_coefficients(c_model))
        eps_m = coefficient_error(c_true, c_model)
        noisy_results.append(dict(label=label, eps_m=eps_m, eps_nd=eps_nd, eps_da=eps_da))
        print(f"  {label:28s}  eps*_nd={eps_nd*100:.3f}%  eps*_da={eps_da*100:.3f}%")

    # Multi-panel summary visualization
    short = [r["label"][:22] for r in free_results]
    idx = np.arange(len(CASES))
    fig, ax = plt.subplots(2, 2, figsize=(18, 14))

    a = ax[0, 0]
    a.bar(
        idx - 0.2,
        [r["eps_m"] * 100 for r in free_results],
        0.4,
        label="ε_m (Imperfect)",
        color="#bcbddc",
    )
    a.bar(
        idx + 0.2,
        [max(r["eps_star"] * 100, 1e-4) for r in free_results],
        0.4,
        label="ε* (Corrected)",
        color="#2c7fb8",
    )
    a.set_yscale("log")
    a.set_ylabel("Coefficient Error [%]", fontweight="bold")
    a.set_xticks(idx)
    a.set_xticklabels(short, rotation=35, ha="right", fontsize=9)
    a.set_title("(a) NOISE-FREE — STEPS 1-2")
    a.legend(fontsize=9)
    a.axhline(2.0, ls="--", lw=1, color="grey")

    b = ax[0, 1]
    b.bar(idx - 0.3, [r["eps_m"] * 100 for r in noisy_results], 0.3, label="ε_m", color="#bcbddc")
    b.bar(
        idx,
        [r["eps_nd"] * 100 for r in noisy_results],
        0.3,
        label="ε*_noDA",
        color="#f03b20",
        alpha=0.5,
    )
    b.bar(
        idx + 0.3, [r["eps_da"] * 100 for r in noisy_results], 0.3, label="ε*_DA", color="#feb24c"
    )
    b.set_yscale("log")
    b.set_ylabel("Coefficient Error [%]", fontweight="bold")
    b.set_xticks(idx)
    b.set_xticklabels(short, rotation=35, ha="right", fontsize=9)
    b.set_title(f"(b) NOISY ({NOISE_LEVEL*100:.0f}%) — STEP 3 (EnKF)")
    b.legend(fontsize=9)
    b.axhline(2.0, ls="--", lw=1, color="grey")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "verification_summary.png"), dpi=300)
    plt.close()


def example_sir_verification_summary():
    """Benchmark discovery of structural SIR errors (missing compartment, wrong rate)."""
    output_dir = "outputs/synthetic/sir"
    os.makedirs(output_dir, exist_ok=True)
    apply_publication_theme()

    true_system = SIRSystem(beta=0.6, gamma=0.18)
    library = PolynomialLibrary(n_vars=3, degree=2, var_names=["S", "I", "R"])
    c_true = true_system.coefficients(library)

    CASES = [
        ("missing infection dS", [("S I", 0, 0.0)]),
        ("missing infection dI", [("S I", 1, 0.0)]),
        ("missing recovery dR", [("I", 2, 0.0)]),
        ("wrong beta (0.4→0.6)", [("S I", 0, -0.4), ("S I", 1, 0.4)]),
        ("wrong gamma (0.10→0.18)", [("I", 1, -0.10), ("I", 2, 0.10)]),
        ("extra S decay", [("S", 0, -0.05)]),
        ("wrong beta + miss dR", [("S I", 0, -0.4), ("S I", 1, 0.4), ("I", 2, 0.0)]),
    ]

    def make_imperfect(mods):
        c = c_true.copy()
        for feat, comp, val in mods:
            c[library.index(feat), comp] = val
        return c

    print("[sir-verify] noise-free pass...")
    free_results = []
    for label, mods in CASES:
        c_model = make_imperfect(mods)
        model = PolynomialODE(c_model, library)
        obs_prev, obs_curr, _, _ = sample_simplex_observations(true_system, 500, dt=0.01)
        res = MEDIDA(model, library, dt=0.01).fit(obs_prev, obs_curr)
        eps_m = coefficient_error(c_true, c_model)
        eps_star = coefficient_error(c_true, res.corrected_coefficients(c_model))
        free_results.append(dict(label=label, eps_m=eps_m, eps_star=eps_star))
        print(f"  {label:28s}  {eps_m*100:5.2f}% → {eps_star*100:.4f}%")

    print("[sir-verify] noisy pass...")
    noisy_results = []
    for label, mods in CASES:
        c_model = make_imperfect(mods)
        model = PolynomialODE(c_model, library)
        obs_prev, obs_curr, _, _ = sample_simplex_observations(
            true_system, 1000, dt=0.05, sigma_obs=0.005
        )

        # no-DA
        med_nd = MEDIDA(model, library, dt=0.05, rvm=RelevanceVectorMachine(t_min=4.0))
        eps_nd = coefficient_error(
            c_true, med_nd.fit(obs_prev, obs_curr).corrected_coefficients(c_model)
        )

        # DA
        enkf = EnsembleKalmanFilter(n_ensemble=200, inflation=1.05)
        med_da = MEDIDA(
            model,
            library,
            dt=0.05,
            enkf=enkf,
            sigma_obs=0.005,
            rvm=RelevanceVectorMachine(t_min=4.0),
        )
        eps_da = coefficient_error(
            c_true, med_da.fit(obs_prev, obs_curr).corrected_coefficients(c_model)
        )
        noisy_results.append(dict(label=label, eps_nd=eps_nd, eps_da=eps_da))
        print(f"  {label:28s}  eps*_nd={eps_nd*100:.3f}%  eps*_da={eps_da*100:.3f}%")


def example_ks():
    """Verify recovery of the Kuramoto-Sivashinsky PDE using spectral integration."""
    output_dir = "outputs/synthetic/ks_pde"
    os.makedirs(output_dir, exist_ok=True)
    apply_publication_theme()

    true_system = KSSystem(L=22.0, N=64, beta=1.0, nu_2=1.0, nu_4=1.0)
    library = PDELibrary(n_grid=64, length=22.0, poly_order=1, deriv_order=4)
    c_true = true_system.coefficients(library)

    # Imperfect model is missing the advection term (u * u_x)
    c_imperfect = c_true.copy()
    c_imperfect[library.index("u u_x")] = 0.0
    imperfect_model = PolynomialODE(c_imperfect, library)

    # Training phase
    obs_prev, obs_curr, _, _ = sample_ks_observations(true_system, 800, dt=0.1)
    medida = MEDIDA(imperfect_model, library, dt=0.1, significance=1e-4)
    result = medida.fit(obs_prev, obs_curr)

    c_star = result.corrected_coefficients(c_imperfect)
    cor_model = PolynomialODE(c_star, library)

    # Forecast comparison
    u0 = obs_prev[0]
    n_steps = 200
    t_traj = true_system.trajectory(u0, 0.1, n_steps)
    i_traj = imperfect_model.trajectory(u0, 0.1, n_steps)
    c_traj = cor_model.trajectory(u0, 0.1, n_steps)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    extent = [0, 22, 20, 0]
    im0 = axes[0].imshow(t_traj, extent=extent, cmap="inferno", aspect="auto")
    axes[0].set_title("GROUND TRUTH")
    axes[0].set_ylabel("TIME")

    im1 = axes[1].imshow(i_traj, extent=extent, cmap="inferno", aspect="auto")
    axes[1].set_title("IMPERFECT (NO ADVECTION)")

    im2 = axes[2].imshow(c_traj, extent=extent, cmap="inferno", aspect="auto")
    axes[2].set_title("MEDIDA RECONSTRUCTION")

    for ax in axes:
        ax.set_xlabel("SPACE (x)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hovmoller_comparison.png"), dpi=300)
    plt.close()


def run_meta_analysis():
    """Execute global robustness sweeps across all synthetic systems."""
    output_dir = "outputs/summary/sweeps"
    os.makedirs(output_dir, exist_ok=True)

    def get_full_sweep(model_type):
        bg, pg, em_f, es_f = run_parameter_sweep(model_type, grid_size=7, n_seeds=2, noise="free")
        _, _, em_n, es_n = run_parameter_sweep(model_type, grid_size=7, n_seeds=2, noise="noisy")
        return bg, pg, em_f, es_f, em_n, es_n

    # Run sweeps for different model structural error types
    sweep_data = [
        ("SIR (missing recovery)", *get_full_sweep("sir"), "BETA", "GAMMA"),
        ("SIRS (waning immunity)", *get_full_sweep("sirs"), "BETA", "XI"),
        ("SIRD (mortality)", *get_full_sweep("sird"), "GAMMA", "MU"),
        ("Nonlinear (saturation)", *get_full_sweep("sat"), "BETA", "ALPHA"),
    ]

    plot_meta_heatmaps(sweep_data, os.path.join(output_dir, "performance_heatmaps.png"))
    plot_robustness_heatmaps(sweep_data, os.path.join(output_dir, "robustness_heatmaps.png"))



if __name__ == "__main__":
    example_sir_vs_sirs()
    example_sird()
    example_nonlinear_sir()
    example_seir_from_sir()
    example_hidden_E()
    example_lorenz()
    example_lorenz_verification_summary()
    example_sir_verification_summary()
    example_ks()
    run_meta_analysis()
