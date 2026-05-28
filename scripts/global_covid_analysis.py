import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "medida_mplconfig")
)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Ensure project root is in path for module imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from medida import (  # noqa: E402
    MEDIDA,
    PolynomialLibrary,
    PolynomialODE,
)
from scripts.covid_analysis import (  # noqa: E402
    COVID_DT,
    COVID_WINDOW,
    LOCKDOWN_COUNTRIES,
    load_and_process_country,
    load_owid_data,
)
from scripts.utils import apply_publication_theme  # noqa: E402


DEFAULT_TARGETS = ["Italy", "Sweden", "India"]


def estimate_country_beta(states, gamma):
    """Estimate a simple SIR beta from the early infectious growth rate."""
    infected = np.asarray(states[:, 1], dtype=float)
    positive = infected > 1e-12
    if np.count_nonzero(positive) < 5:
        return None

    horizon = min(9, len(infected))
    growth = infected[:horizon]
    if np.count_nonzero(growth > 0) < 5:
        return None

    slope, _ = np.polyfit(
        np.arange(len(growth)), np.log(growth + 1e-15), 1
    )
    return float(slope + gamma)


def build_global_training_set(df_all, targets):
    """Concatenate country trajectories into one pooled training set."""
    obs_prev_list = []
    obs_curr_list = []
    used_countries = []
    skipped = []
    gamma = 1.0 / COVID_WINDOW

    for country in sorted(df_all["location"].dropna().unique()):
        if country in targets:
            continue

        states, _ = load_and_process_country(df_all, country)
        if states is None or len(states) < 30:
            skipped.append(country)
            continue

        beta = estimate_country_beta(states, gamma)
        if beta is None or not np.isfinite(beta):
            skipped.append(country)
            continue

        obs_prev_list.append(states[:-1])
        obs_curr_list.append(states[1:])
        used_countries.append((country, beta))

    if not obs_prev_list:
        raise RuntimeError("No valid countries were available for training.")

    obs_prev = np.concatenate(obs_prev_list, axis=0)
    obs_curr = np.concatenate(obs_curr_list, axis=0)
    beta_global = float(np.median([b for _, b in used_countries]))

    return obs_prev, obs_curr, beta_global, used_countries, skipped


def build_baseline_coeffs(lib, beta, gamma):
    coeffs = np.zeros((lib.n_features, 3))
    coeffs[lib.index("S I"), 0], coeffs[lib.index("S I"), 1] = -beta, beta
    coeffs[lib.index("I"), 1], coeffs[lib.index("I"), 2] = -gamma, gamma
    return coeffs


def evaluate_target_country(df_all, target, imp_coeffs, corrected_coeffs, lib):
    states, N_pop = load_and_process_country(df_all, target)
    if states is None:
        return None

    Phi = lib.transform(states[:-1])
    pred_naive = states[:-1] + COVID_DT * (Phi @ imp_coeffs)
    pred_medida = states[:-1] + COVID_DT * (Phi @ corrected_coeffs)

    rmse_naive = float(
        np.sqrt(np.mean((states[1:, 1] - pred_naive[:, 1]) ** 2))
    )
    rmse_medida = float(
        np.sqrt(np.mean((states[1:, 1] - pred_medida[:, 1]) ** 2))
    )

    return {
        "country": target,
        "population": float(N_pop),
        "states": states,
        "pred_naive": pred_naive,
        "pred_medida": pred_medida,
        "naive_rmse": rmse_naive,
        "medida_rmse": rmse_medida,
        "improvement": rmse_naive / rmse_medida if rmse_medida > 0 else np.inf,
    }


def evaluate_pooled_training_fit(
    obs_prev, obs_curr, imp_coeffs, corrected_coeffs, lib
):
    """Evaluate the pooled global training set as one aggregated fit."""
    pred_naive = obs_prev + COVID_DT * (lib.transform(obs_prev) @ imp_coeffs)
    pred_medida = (
        obs_prev + COVID_DT * (lib.transform(obs_prev) @ corrected_coeffs)
    )

    rmse_naive = float(
        np.sqrt(np.mean((obs_curr[:, 1] - pred_naive[:, 1]) ** 2))
    )
    rmse_medida = float(
        np.sqrt(np.mean((obs_curr[:, 1] - pred_medida[:, 1]) ** 2))
    )

    return {
        "country": "Global",
        "population": 1.0,
        "states": obs_curr,
        "pred_naive": pred_naive,
        "pred_medida": pred_medida,
        "naive_rmse": rmse_naive,
        "medida_rmse": rmse_medida,
        "improvement": rmse_naive / rmse_medida if rmse_medida > 0 else np.inf,
    }


def _plot_fit_error(result, output_path):
    apply_publication_theme()
    states = result["states"]
    pred_naive = result["pred_naive"]
    pred_medida = result["pred_medida"]
    n_pop = float(result["population"])

    t = np.arange(len(states))
    is_global = result["country"] == "Global"
    t_pred = t if is_global else np.arange(1, len(states))
    obs_I = states[:, 1] * n_pop
    naive_I = pred_naive[:, 1] * n_pop
    medida_I = pred_medida[:, 1] * n_pop
    obs_next_I = obs_I if is_global else states[1:, 1] * n_pop

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    ax = axes[0]
    ax.fill_between(t, obs_I, alpha=0.15, color="black", label="Observed")
    ax.plot(t, obs_I, color="black", lw=2.4)
    ax.plot(t_pred, naive_I, "r--", lw=2.0, label="Naive SIR")
    ax.plot(t_pred, medida_I, color="#33a02c", lw=2.5, label="Global MEDIDA")
    ax.set_title(
        f"{result['country']} epidemic curve  ({result['improvement']:.2f}x)",
        fontweight="black",
    )
    ax.set_xlabel("Days since start")
    ax.set_ylabel("Infectious fraction" if n_pop == 1.0 else "Infectious population")
    if result["country"] in LOCKDOWN_COUNTRIES:
        ax.axvline(
            8,
            color="#6a3d9a",
            lw=1.8,
            ls=":",
            alpha=0.8,
            label="Lockdown onset",
        )
    ax.legend(loc="upper left")

    ax = axes[1]
    ax.axhline(0, color="black", lw=1, alpha=0.35)
    ax.plot(
        t_pred,
        naive_I - obs_next_I,
        "r--",
        lw=2.0,
        label="Naive residuals",
    )
    ax.plot(
        t_pred,
        medida_I - obs_next_I,
        color="#33a02c",
        lw=2.5,
        label="MEDIDA residuals",
    )
    ax.set_title(f"{result['country']} residuals", fontweight="black")
    ax.set_xlabel("Days since start")
    ax.set_ylabel(
        "1-step residual (fraction)" if n_pop == 1.0 else "1-step residual (people)"
    )
    ax.legend(loc="upper left")

    fig.suptitle(
        "GLOBAL TRAINED MEDIDA TRANSFER",
        fontsize=20,
        fontweight="black",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_effective_beta(result, beta_global, corrected_coeffs, lib, output_path):
    apply_publication_theme()
    states = result["states"]
    op = states[:-1]
    Phi = lib.transform(op)
    dS_corr = (Phi @ corrected_coeffs)[:, 0]
    S = op[:, 0]
    I = op[:, 1]
    SI = np.clip(S * I, 1e-12, None)
    beta_eff = np.clip(-dS_corr / SI, 0, None)
    y_max = max(beta_global * 1.8, 0.05)
    beta_eff = np.clip(beta_eff, 0, y_max)

    t = np.arange(len(beta_eff))
    fig, ax = plt.subplots(figsize=(16, 4.8))
    ax.plot(t, beta_eff, color="#33a02c", lw=2.6, label="Global MEDIDA β(t)")
    ax.axhline(
        beta_global,
        color="#e31a1c",
        lw=1.8,
        ls="--",
        alpha=0.8,
        label="Global baseline β",
    )
    if result["country"] in LOCKDOWN_COUNTRIES:
        ax.axvline(
            8,
            color="#6a3d9a",
            lw=1.8,
            ls=":",
            alpha=0.8,
            label="Lockdown onset",
        )
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Days since start")
    ax.set_ylabel("Effective β(t)")
    ax.set_title(
        f"{result['country']} effective transmission  ({result['improvement']:.2f}x)",
        fontweight="black",
    )
    ax.legend(loc="upper right")

    fig.suptitle(
        "GLOBAL TRAINED MEDIDA EFFECTIVE TRANSMISSION",
        fontsize=20,
        fontweight="black",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.96])
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_global_training_curve(global_result, output_path):
    """Plot the pooled global training fit as a single aggregated series."""
    apply_publication_theme()
    states = global_result["states"]
    pred_naive = global_result["pred_naive"]
    pred_medida = global_result["pred_medida"]

    t = np.arange(len(states))
    t_pred = np.arange(1, len(states))
    obs_I = states[:, 1]
    naive_I = pred_naive[:, 1]
    medida_I = pred_medida[:, 1]
    obs_next_I = states[1:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    ax = axes[0]
    ax.fill_between(t, obs_I, alpha=0.15, color="black", label="Observed")
    ax.plot(t, obs_I, color="black", lw=2.4)
    ax.plot(t_pred, naive_I, "r--", lw=2.0, label="Naive SIR")
    ax.plot(t_pred, medida_I, color="#33a02c", lw=2.5, label="Global MEDIDA")
    ax.set_title(
        f"Global pooled fit  ({global_result['improvement']:.2f}x)",
        fontweight="black",
    )
    ax.set_xlabel("Pooled sample index")
    ax.set_ylabel("Infectious fraction")
    ax.legend(loc="upper left")

    ax = axes[1]
    ax.axhline(0, color="black", lw=1, alpha=0.35)
    ax.plot(
        t_pred,
        naive_I - obs_next_I,
        "r--",
        lw=2.0,
        label="Naive residuals",
    )
    ax.plot(
        t_pred,
        medida_I - obs_next_I,
        color="#33a02c",
        lw=2.5,
        label="MEDIDA residuals",
    )
    ax.set_title("Global pooled residuals", fontweight="black")
    ax.set_xlabel("Pooled sample index")
    ax.set_ylabel("1-step residual (fraction)")
    ax.legend(loc="upper left")

    fig.suptitle(
        "GLOBAL TRAINED MEDIDA: POOLED FIT AND ERROR",
        fontsize=20,
        fontweight="black",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.96])
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train a global MEDIDA correction and test it on a few countries."
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=DEFAULT_TARGETS,
        help="Countries to evaluate the global-trained correction on.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/global_transfer_slide",
        help="Directory for the summary CSV and slide-ready PNGs.",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Force re-download of the OWID CSV before training.",
    )
    args = parser.parse_args()

    df_all = load_owid_data(force_download=args.refresh_data)
    output_dir = (
        args.output_dir
        if os.path.isabs(args.output_dir)
        else os.path.join(project_root, args.output_dir)
    )
    os.makedirs(output_dir, exist_ok=True)
    apply_publication_theme()

    gamma = 1.0 / COVID_WINDOW
    lib = PolynomialLibrary(n_vars=3, degree=2, var_names=["S", "I", "R"])

    obs_prev, obs_curr, beta_global, used_countries, skipped = (
        build_global_training_set(df_all, set(args.targets))
    )
    imp_coeffs = build_baseline_coeffs(lib, beta_global, gamma)

    medida = MEDIDA(
        PolynomialODE(imp_coeffs, lib),
        lib,
        dt=COVID_DT,
        significance=1e-8,
    )
    result = medida.fit(obs_prev, obs_curr)
    corrected_coeffs = result.corrected_coefficients(imp_coeffs)

    global_result = evaluate_pooled_training_fit(
        obs_prev, obs_curr, imp_coeffs, corrected_coeffs, lib
    )

    target_rows = []
    for target in args.targets:
        row = evaluate_target_country(
            df_all, target, imp_coeffs, corrected_coeffs, lib
        )
        if row is not None:
            target_rows.append(row)

    results_df = pd.DataFrame(target_rows)
    csv_path = os.path.join(output_dir, "global_transfer_summary.csv")
    results_df.to_csv(csv_path, index=False)

    fit_specs = [global_result] + [row for _, row in results_df.iterrows()]
    for spec in fit_specs:
        country = spec["country"].lower().replace(" ", "_")
        plot_path = os.path.join(output_dir, f"{country}_fit_error.png")
        beta_path = os.path.join(output_dir, f"{country}_effective_beta.png")
        _plot_fit_error(spec, plot_path)
        plot_effective_beta(spec, beta_global, corrected_coeffs, lib, beta_path)

    print(f"Global beta estimate: {beta_global:.4f}")
    print(f"Training countries used: {len(used_countries)}")
    print(f"Skipped countries: {len(skipped)}")
    print("\nTarget-country transfer:")
    print(
        results_df[
            ["country", "naive_rmse", "medida_rmse", "improvement"]
        ].to_string(index=False)
    )
    print(f"\nWrote {csv_path}")
    print(f"Wrote {os.path.join(output_dir, 'global_fit_error.png')}")
    for target in args.targets:
        country = target.lower().replace(" ", "_")
        print(f"Wrote {os.path.join(output_dir, f'{country}_fit_error.png')}")
        print(
            f"Wrote {os.path.join(output_dir, f'{country}_effective_beta.png')}"
        )


if __name__ == "__main__":
    main()
