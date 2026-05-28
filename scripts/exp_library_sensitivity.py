import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.utils import apply_publication_theme  # noqa: E402
from medida import (  # noqa: E402
    SIRNonlinearSystem,
    PolynomialLibrary,
    SaturatedSIRLibrary,
    MEDIDA,
    PolynomialODE,
    coefficient_error,
    sample_simplex_observations,
)


def run_library_sensitivity():
    """Analyze how library selection affects nonlinear discovery."""
    output_dir = "outputs/experiments/library_sensitivity"
    os.makedirs(output_dir, exist_ok=True)
    apply_publication_theme()

    # Ground truth: SIR with saturated incidence (Holling Type II)
    a_true = 8.0
    true_system = SIRNonlinearSystem(beta=0.6, gamma=0.18, a=a_true)

    # Common test data
    dt = 0.05
    n_samples = 1500
    obs_prev, obs_curr, _, _ = sample_simplex_observations(
        true_system, n_samples, dt, seed=42
    )

    # Libraries to test
    LIBRARIES = {
        "Linear": PolynomialLibrary(
            n_vars=3, degree=1, var_names=["S", "I", "R"]
        ),
        "Quad-Poly": PolynomialLibrary(
            n_vars=3, degree=2, var_names=["S", "I", "R"]
        ),
        "Saturated": SaturatedSIRLibrary(a=a_true),
    }

    # Naive SIR baseline coefficients (assuming linear mass-action)
    def get_baseline(lib):
        c = np.zeros((lib.n_features, 3))
        try:
            idx_si = lib.index("S I")
            c[idx_si, 0], c[idx_si, 1] = -0.6, 0.6
        except ValueError:
            pass  # Linear library doesn't have S I

        try:
            idx_i = lib.index("I")
            c[idx_i, 1], c[idx_i, 2] = -0.18, 0.18
        except ValueError:
            pass
        return c

    results = {}
    for name, lib in LIBRARIES.items():
        baseline_c = get_baseline(lib)
        model = PolynomialODE(baseline_c, lib)
        medida = MEDIDA(model, lib, dt=dt, significance=1e-6)
        res = medida.fit(obs_prev, obs_curr)

        # Calculate improvement: ε_m / ε*
        # Use a surrogate true coefficient map for non-saturated libraries.
        c_true_map = true_system.coefficients(lib)
        eps_m = coefficient_error(c_true_map, baseline_c)
        eps_s = coefficient_error(
            c_true_map, res.corrected_coefficients(baseline_c)
        )
        results[name] = eps_m / eps_s if eps_s > 1e-15 else 1000.0

    # Visualization
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    values = list(results.values())
    plt.bar(
        names,
        values,
        color=["#fb9a99", "#a6cee3", "#b2df8a"],
        edgecolor="black",
    )
    plt.yscale("log")
    plt.ylabel("Improvement Factor (Log Scale)")
    plt.title("LIBRARY SENSITIVITY: DISCOVERING SATURATED INCIDENCE")
    plt.grid(axis="y", alpha=0.3)
    sns.despine()
    plt.savefig(os.path.join(output_dir, "library_comparison.png"))
    plt.close()

    print(f"[*] Library sensitivity results saved to {output_dir}")


if __name__ == "__main__":
    run_library_sensitivity()
