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
    COVID_END,
    COVID_START,
    COVID_WINDOW,
    LOCKDOWN_COUNTRIES,
    load_and_process_country,
    load_owid_data,
)
from scripts.utils import apply_publication_theme  # noqa: E402


DEFAULT_TARGETS = ["Italy", "Sweden", "India"]

# Extend targets to 2-year window to capture second-wave / Delta dynamics
COVID_END_GLOBAL = "2021-12-31"

COUNTRY_COLORS = {
    "Italy": "#1f78b4",
    "Sweden": "#e31a1c",
    "India": "#ff7f00",
}


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


def evaluate_target_country(
    df_all, target, imp_coeffs, corrected_coeffs, lib
):
    states, N_pop = load_and_process_country(
        df_all, target, end=COVID_END_GLOBAL
    )
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
        "improvement": (
            rmse_naive / rmse_medida if rmse_medida > 0 else np.inf
        ),
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


# ---------------------------------------------------------------------------
# Per-country: epidemic curve + residuals
# ---------------------------------------------------------------------------

def plot_country_fit(result, output_path):
    """Side-by-side epidemic curve and one-step residuals for one country."""
    apply_publication_theme()
    states = result["states"]
    pred_naive = result["pred_naive"]
    pred_medida = result["pred_medida"]
    n_pop = float(result["population"])
    country = result["country"]
    color = COUNTRY_COLORS.get(country, "#33a02c")

    t_obs = np.arange(len(states))
    t_pred = np.arange(1, len(states))
    obs_I = states[:, 1] * n_pop
    naive_I = pred_naive[:, 1] * n_pop
    medida_I = pred_medida[:, 1] * n_pop
    obs_next_I = states[1:, 1] * n_pop

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    ax.fill_between(t_obs, obs_I, alpha=0.15, color="black", label="Observed")
    ax.plot(t_obs, obs_I, color="black", lw=2.4)
    ax.plot(t_pred, naive_I, "r--", lw=2.0, label="Naive SIR")
    ax.plot(
        t_pred, medida_I, color=color, lw=2.5, label="Global MEDIDA"
    )
    if country in LOCKDOWN_COUNTRIES:
        ax.axvline(
            8, color="#6a3d9a", lw=1.8, ls=":", alpha=0.8,
            label="Lockdown onset",
        )
    ax.set_xlabel("Days since start")
    ax.set_xlim(0, 700)
    ax.set_ylabel("Infectious population")
    ax.set_title(
        f"{country} epidemic curve  ({result['improvement']:.2f}x)",
        fontweight="black",
    )
    ax.legend(loc="best", fontsize="small", frameon=True, framealpha=0.8)

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
        color=color,
        lw=2.5,
        label="MEDIDA residuals",
    )
    if country in LOCKDOWN_COUNTRIES:
        ax.axvline(
            8,
            color="#6a3d9a",
            lw=1.8,
            ls=":",
            alpha=0.8,
            label="Lockdown onset",
        )
    ax.set_xlabel("Days since start")
    ax.set_xlim(0, 700)
    ax.set_ylabel("1-step residual (people)")
    ax.set_title(f"{country} residuals", fontweight="black")
    ax.legend(loc="best", fontsize="small", frameon=True, framealpha=0.8)

    fig.suptitle(
        "GLOBAL TRAINED MEDIDA TRANSFER",
        fontsize=22,
        fontweight="black",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Global epidemic curve (calendar-date aligned sum across training countries)
# ---------------------------------------------------------------------------

def build_global_aggregate(df_all, targets, imp_coeffs, corrected_coeffs, lib):
    """Sum I(t)*N_pop across all training countries aligned to calendar dates."""
    dates = pd.date_range(start=COVID_START, end=COVID_END_GLOBAL, freq="D")
    n_days = len(dates)
    global_obs = np.zeros(n_days)
    global_naive = np.zeros(n_days)
    global_medida = np.zeros(n_days)
    gamma = 1.0 / COVID_WINDOW

    for country in sorted(df_all["location"].dropna().unique()):
        if country in targets:
            continue
        states, N_pop = load_and_process_country(
            df_all, country, end=COVID_END_GLOBAL
        )
        if states is None or len(states) < 30:
            continue
        beta = estimate_country_beta(states, gamma)
        if beta is None or not np.isfinite(beta):
            continue

        n = len(states)
        limit = min(n, n_days)
        Phi = lib.transform(states[: limit - 1])
        pred_naive = states[: limit - 1] + COVID_DT * (Phi @ imp_coeffs)
        pred_medida = states[: limit - 1] + COVID_DT * (Phi @ corrected_coeffs)

        global_obs[:limit] += states[:limit, 1] * N_pop
        global_naive[1:limit] += pred_naive[: limit - 1, 1] * N_pop
        global_medida[1:limit] += pred_medida[: limit - 1, 1] * N_pop

    return dates, global_obs, global_naive, global_medida


def plot_global_fit_error(
    df_all, targets, imp_coeffs, corrected_coeffs, lib, output_path
):
    """Global aggregate epidemic curve + residuals, styled to match per-country fit_error."""
    apply_publication_theme()
    dates, obs, naive, medida = build_global_aggregate(
        df_all, targets, imp_coeffs, corrected_coeffs, lib
    )

    obs_M = obs / 1e6
    naive_M = naive / 1e6
    medida_M = medida / 1e6
    naive_res = naive_M[1:] - obs_M[1:]
    medida_res = medida_M[1:] - obs_M[1:]
    n = len(obs_M)
    t = np.arange(n)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left: epidemic curve
    ax = axes[0]
    ax.fill_between(t, obs_M, alpha=0.15, color="black", label="Observed")
    ax.plot(t, obs_M, color="black", lw=2.4)
    ax.plot(t[1:], naive_M[1:], "r--", lw=2.0, label="Naive SIR")
    ax.plot(t[1:], medida_M[1:], color="#33a02c", lw=2.5,
            label="Global MEDIDA")
    ax.set_xlim(0, 700)
    ax.set_xlabel("Days since start (2020-03-01)")
    ax.set_ylabel("Infectious population (millions)")
    ax.set_title(
        "Global aggregate epidemic curve  (60 training countries)",
        fontweight="black",
    )
    ax.legend(loc="upper left", fontsize="small", frameon=True, framealpha=0.8)

    # Right: residuals
    ax = axes[1]
    ax.axhline(0, color="black", lw=1, alpha=0.35)
    ax.plot(t[1:], naive_res, "r--", lw=2.0, label="Naive residuals")
    ax.plot(t[1:], medida_res, color="#33a02c", lw=2.5,
            label="MEDIDA residuals")
    ax.set_xlim(0, 700)
    ax.set_xlabel("Days since start (2020-03-01)")
    ax.set_ylabel("1-step residual (millions)")
    ax.set_title("Global aggregate residuals", fontweight="black")
    ax.legend(loc="upper left", fontsize="small", frameon=True, framealpha=0.8)

    fig.suptitle(
        "GLOBAL TRAINED MEDIDA TRANSFER",
        fontsize=22, fontweight="black", y=0.98,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Global summary: 3-panel β(t) comparison (replaces concatenated beta)
# ---------------------------------------------------------------------------
def plot_global_beta_comparison(
    target_rows, beta_global, corrected_coeffs, lib, output_path
):
    """Three-panel effective β(t) — one panel per target country."""
    apply_publication_theme()

    n = len(target_rows)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), sharey=True) # Taller figure
    if n == 1:
        axes = [axes]

    y_max = 1.3 # Standardized limit

    for ax, result in zip(axes, target_rows):
        country = result["country"]
        color = COUNTRY_COLORS.get(country, "#33a02c")
        states = result["states"]
        op = states[:-1]
        Phi = lib.transform(op)
        dS_corr = (Phi @ corrected_coeffs)[:, 0]
        S, I = op[:, 0], op[:, 1]
        SI = np.clip(S * I, 1e-12, None)
        beta_eff = np.clip(-dS_corr / SI, 0, y_max)

        t = np.arange(len(beta_eff))
        ax.plot(t, beta_eff, color=color, lw=2.4, label="β(t)")
        ax.axhline(
            beta_global, color="#e31a1c", lw=1.6, ls="--", alpha=0.75,
            label=f"Baseline β = {beta_global:.3f}",
        )
        if country in LOCKDOWN_COUNTRIES:
            ax.axvline(
                8, color="#6a3d9a", lw=1.6, ls=":", alpha=0.8,
                label="Lockdown onset",
            )
        ax.set_ylim(0, y_max)
        ax.set_xlabel("Days since start")
        ax.set_xlim(0, 700)
        ax.set_title(
            f"{country}  ({result['improvement']:.2f}×)",
            fontweight="black",
        )
        ax.legend(fontsize=10)

    axes[0].set_ylabel("Effective β(t)")
    fig.suptitle(
        "GLOBAL TRAINED MEDIDA: EFFECTIVE TRANSMISSION BY COUNTRY",
        fontsize=18, fontweight="black", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train a global MEDIDA correction on all non-target countries "
            "and evaluate transfer to held-out targets."
        )
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=DEFAULT_TARGETS,
        help="Countries to evaluate the global-trained correction on.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run evaluation across all ~200 countries for global statistics.",
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

    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    gamma = 1.0 / COVID_WINDOW
    lib = PolynomialLibrary(n_vars=3, degree=2, var_names=["S", "I", "R"])

    print("Building global pooled training set...")
    obs_prev, obs_curr, beta_global, used_countries, skipped = (
        build_global_training_set(df_all, set(args.targets))
    )
    print(
        f"  Training on {len(used_countries)} countries "
        f"({len(skipped)} skipped), global β̄ = {beta_global:.4f}"
    )

    imp_coeffs = build_baseline_coeffs(lib, beta_global, gamma)

    print("Running MEDIDA fit on pooled data...")
    medida = MEDIDA(
        PolynomialODE(imp_coeffs, lib),
        lib,
        dt=COVID_DT,
        significance=1e-8,
    )
    fit_result = medida.fit(obs_prev, obs_curr)
    corrected_coeffs = fit_result.corrected_coefficients(imp_coeffs)

    print(f"Evaluating accuracy gains (window: 2020-03-01 → {COVID_END_GLOBAL})...")
    target_rows = []
    eval_list = sorted(df_all["location"].unique()) if args.sweep else args.targets

    for target in eval_list:
        row = evaluate_target_country(
            df_all, target, imp_coeffs, corrected_coeffs, lib
        )
        if row is not None:
            target_rows.append(row)
            if target in args.targets:
                print(
                    f"  {target}: {row['improvement']:.2f}× improvement  "
                    f"(naive RMSE {row['naive_rmse']:.5f}, "
                    f"MEDIDA RMSE {row['medida_rmse']:.5f})"
                )

    csv_path = os.path.join(output_dir, "global_transfer_summary.csv")
    pd.DataFrame(target_rows).drop(
        columns=["states", "pred_naive", "pred_medida"], errors="ignore"
    ).to_csv(csv_path, index=False)

    # Separate target rows from full sweep rows for plotting
    target_plot_rows = [r for r in target_rows if r["country"] in args.targets]

    # --- Global summary figures ---
    print("Plotting global summary figures...")
    plot_global_fit_error(
        df_all, set(args.targets), imp_coeffs, corrected_coeffs, lib,
        os.path.join(output_dir, "global_fit_error.png"),
    )
    plot_global_beta_comparison(
        target_plot_rows, beta_global, corrected_coeffs, lib,
        os.path.join(output_dir, "global_beta_comparison.png"),
    )

    # --- Per-country fit figures (targets only) ---
    print("Plotting per-country fit figures...")
    for row in target_plot_rows:
        slug = row["country"].lower().replace(" ", "_")
        plot_country_fit(row, os.path.join(output_dir, f"{slug}_fit_error.png"))

    print(f"\nOutputs written to {output_dir}/")
    for fname in sorted(os.listdir(output_dir)):
        print(f"  {fname}")


if __name__ == "__main__":
    main()
