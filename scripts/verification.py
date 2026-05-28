import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to sys.path to allow importing the 'medida' package
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Also add the current examples directory for local utility imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import run_medida_experiment
from medida import (
    SIRSystem, SIRSSystem, SIRDSystem, SIRNonlinearSystem,
    Lorenz63, KSSystem,
    PolynomialLibrary, SaturatedSIRLibrary, PDELibrary,
    PolynomialODE, RelevanceVectorMachine, MEDIDA, RidgeRVM,
    format_system, sample_ks_observations, coefficient_error
)

def example_sir_vs_sirs():
    # Matches Notebook 2.1: Noise-free SIRS truth vs plain SIR
    true_system = SIRSSystem(beta=0.6, gamma=0.18, xi=0.08)
    library = PolynomialLibrary(n_vars=3, degree=2, var_names=["S", "I", "R"])
    
    true_coeffs = np.zeros((library.n_features, 3))
    true_coeffs[library.index("S I"), 0] = -0.6
    true_coeffs[library.index("R"), 0] = 0.08
    true_coeffs[library.index("S I"), 1] = 0.6
    true_coeffs[library.index("I"), 1] = -0.18
    true_coeffs[library.index("I"), 2] = 0.18
    true_coeffs[library.index("R"), 2] = -0.08

    imperfect_coeffs = np.zeros_like(true_coeffs)
    imperfect_coeffs[library.index("S I"), 0] = -0.6
    imperfect_coeffs[library.index("S I"), 1] = 0.6
    imperfect_coeffs[library.index("I"), 1] = -0.18
    imperfect_coeffs[library.index("I"), 2] = 0.18

    run_medida_experiment(
        "Noise-free SIRS truth vs plain SIR",
        true_system, library, true_coeffs, imperfect_coeffs,
        u0=np.array([0.98, 0.02, 0.0]),
        alpha=np.array([3.0, 0.8, 2.0]),
        dt_fit=0.01,
        n_samples=800,
        sigma_obs=0.0,
        seed=0,
        significance=1e-6
    )

    # Matches Notebook 2.7 Noisy Case
    run_medida_experiment(
        "Noisy SIRS truth vs plain SIR",
        true_system, library, true_coeffs, imperfect_coeffs,
        u0=np.array([0.98, 0.02, 0.0]),
        alpha=np.array([3.0, 0.8, 2.0]),
        dt_fit=0.05,
        n_samples=3000,
        sigma_obs=0.002,
        seed=0,
        noise_seed=1,
        ridge=0.01,
        significance=1e-6
    )

def example_sird():
    # Matches Notebook 2.2: Noise-free SIRD truth vs plain SIR
    true_system = SIRDSystem(beta=0.6, gamma=0.14, mu=0.04)
    library = PolynomialLibrary(n_vars=4, degree=2, var_names=["S", "I", "R", "D"])
    
    true_coeffs = np.zeros((library.n_features, 4))
    true_coeffs[library.index("S I"), 0] = -0.6
    true_coeffs[library.index("S I"), 1] = 0.6
    true_coeffs[library.index("I"), 1] = -(0.14 + 0.04)
    true_coeffs[library.index("I"), 2] = 0.14
    true_coeffs[library.index("I"), 3] = 0.04

    imperfect_coeffs = np.zeros_like(true_coeffs)
    imperfect_coeffs[library.index("S I"), 0] = -0.6
    imperfect_coeffs[library.index("S I"), 1] = 0.6
    imperfect_coeffs[library.index("I"), 1] = -0.18
    imperfect_coeffs[library.index("I"), 2] = 0.18

    run_medida_experiment(
        "Noise-free SIRD truth vs plain SIR",
        true_system, library, true_coeffs, imperfect_coeffs,
        u0=np.array([0.98, 0.02, 0.0, 0.0]),
        alpha=np.array([3.0, 0.8, 2.0, 0.2]),
        dt_fit=0.01,
        n_samples=1000,
        sigma_obs=0.0,
        seed=0,
        significance=1e-6
    )

def example_nonlinear():
    # Matches Notebook 2.3: Nonlinear-incidence SIR truth vs SIR
    true_system = SIRNonlinearSystem(beta=0.6, gamma=0.18, a=8.0)
    library = SaturatedSIRLibrary(a=8.0)
    
    true_coeffs = np.zeros((library.n_features, 3))
    sat_name = f"S I / (1 + 8 I)"
    true_coeffs[library.index(sat_name), 0] = -0.6
    true_coeffs[library.index(sat_name), 1] = 0.6
    true_coeffs[library.index("I"), 1] = -0.18
    true_coeffs[library.index("I"), 2] = 0.18

    imperfect_coeffs = np.zeros_like(true_coeffs)
    # plain SIR features in SaturatedSIRLibrary? No, we need to map S*I
    # SaturatedSIRLibrary has: I, S*I, S*I/(1+a*I)
    imperfect_coeffs[library.index("S I"), 0] = -0.6
    imperfect_coeffs[library.index("S I"), 1] = 0.6
    imperfect_coeffs[library.index("I"), 1] = -0.18
    imperfect_coeffs[library.index("I"), 2] = 0.18

    run_medida_experiment(
        "Nonlinear SIR truth vs plain SIR",
        true_system, library, true_coeffs, imperfect_coeffs,
        u0=np.array([0.98, 0.02, 0.0]),
        alpha=np.array([3.0, 0.8, 2.0]),
        dt_fit=0.01,
        n_samples=1000,
        sigma_obs=0.0,
        seed=0,
        significance=1e-6
    )

def example_lorenz():
    true_system = Lorenz63(sigma=10.0, rho=28.0, beta=8/3)
    library = PolynomialLibrary(n_vars=3, degree=2, var_names=["x", "y", "z"])
    
    true_coeffs = np.zeros((library.n_features, 3))
    true_coeffs[library.index("x"), 0] = -10.0
    true_coeffs[library.index("y"), 0] = 10.0
    true_coeffs[library.index("x"), 1] = 28.0
    true_coeffs[library.index("y"), 1] = -1.0
    true_coeffs[library.index("x z"), 1] = -1.0
    true_coeffs[library.index("x y"), 2] = 1.0
    true_coeffs[library.index("z"), 2] = -8/3
    
    imperfect_coeffs = true_coeffs.copy()
    imperfect_coeffs[library.index("x z"), 1] = 0.0
    
    from medida.framework import sample_observations
    print("\n" + "="*70)
    print("Lorenz-63 structural discovery")
    print("="*70)
    
    u0 = np.array([-8.0, 7.0, 27.0])
    dt_fit = 0.01
    t_end = 20.0
    
    print(f"[*] Simulating Lorenz-63 for {t_end}s...")
    u_obs, u_true, t = sample_observations(true_system, u0, t_end, dt_fit, seed=0)
    
    model_m = PolynomialODE(imperfect_coeffs, library, state_names=true_system.state_names)
    rvm = RelevanceVectorMachine(t_min=2.0, threshold=0.01)
    medida = MEDIDA(model_m, library, dt=dt_fit, rvm=rvm)
    
    print("[*] Running MEDIDA...")
    result = medida.fit(u_obs[:-1], u_obs[1:])
    
    print("\nDiscovered MEDIDA correction:")
    print(format_system(result.error_coefficients, library.feature_names, true_system.state_names))
    
def example_ks():
    print("\n" + "="*70)
    print("Kuramoto-Sivashinsky (KS) PDE discovery")
    print("="*70)
    
    L, N = 22.0, 64
    true_system = KSSystem(L=L, N=N, beta=1.0, nu_2=1.0, nu_4=1.0)
    library = PDELibrary(n_grid=N, length=L)
    
    true_coeffs = np.zeros(library.n_features)
    true_coeffs[library.index('u u_x')]   = -1.0
    true_coeffs[library.index('u_xx')]    = -1.0
    true_coeffs[library.index('u_xxxx')]  = -1.0
    
    model_m = KSSystem(L=L, N=N, beta=0.0, nu_2=1.0, nu_4=1.0)
    imperfect_coeffs = np.zeros(library.n_features)
    imperfect_coeffs[library.index('u_xx')]    = -1.0
    imperfect_coeffs[library.index('u_xxxx')]  = -1.0
    
    dt_fit = 0.01
    n_samples = 200
    
    print(f"[*] Generating {n_samples} KS observation pairs (dt={dt_fit})...")
    obs_prev, obs_curr, _, _ = sample_ks_observations(
        true_system, n_samples, dt_fit, seed=7)
    
    print("[*] Running MEDIDA for KS...")
    rvm = RelevanceVectorMachine(t_min=3.0, threshold=0.03, max_iter=400)
    medida = MEDIDA(model_m, library, dt=dt_fit, rvm=rvm, significance=1e-4)
    result = medida.fit(obs_prev, obs_curr)
    
    print("\nDiscovered Structural Correction for KS:")
    for i, c in enumerate(result.error_coefficients):
        if abs(c) > 1e-3:
            print(f"  {library.feature_names[i]}: {c:+.4f}")
    
    corrected_coeffs = imperfect_coeffs + result.error_coefficients
    eps_m = coefficient_error(true_coeffs, imperfect_coeffs)
    eps_star = coefficient_error(true_coeffs, corrected_coeffs)
    
    print(f"\nResults:")
    print(f"  Original coefficient error eps_m: {eps_m*100:.3f}%")
    print(f"  Corrected coefficient error eps_star: {eps_star*100:.4f}%")
    if eps_star > 0:
        print(f"  Improvement factor: {eps_m/eps_star:.0f}x")

if __name__ == "__main__":
    example_sir_vs_sirs()
    example_sird()
    example_nonlinear()
    example_lorenz()
    example_ks()
