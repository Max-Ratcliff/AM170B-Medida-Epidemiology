import sys
import os
import tempfile
import pandas as pd
import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "medida_mplconfig")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from matplotlib.patches import Patch

# Ensure project root is in path for module imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.utils import (
    save_latex_correction,
    apply_publication_theme,
    save_discovery_card,
)
from medida import (
    MEDIDA,
    PolynomialODE,
    PolynomialLibrary,
    RelevanceVectorMachine,
    format_system,
    relative_error,
)

# Configuration for COVID-19 data processing
COVID_URL = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
COVID_START = "2020-03-01"
COVID_END = "2021-01-15"
COVID_MULTIPLIER = 4  # Factor to estimate total infections from reported cases
COVID_WINDOW = 14  # Infectious period duration (days)
COVID_DT = 1.0  # Time step for real-world daily reports
COVID_LOCKDOWN_DAY = (
    8  # Estimated onset of social distancing in training window
)

# List of countries with documented national lockdowns in early 2020
LOCKDOWN_COUNTRIES = {
    "Austria",
    "Belgium",
    "Bolivia",
    "Bosnia and Herzegovina",
    "Bulgaria",
    "Colombia",
    "Croatia",
    "Czechia",
    "Ecuador",
    "El Salvador",
    "France",
    "Germany",
    "Greece",
    "Hungary",
    "India",
    "Ireland",
    "Israel",
    "Italy",
    "Jordan",
    "Lebanon",
    "Lithuania",
    "Malaysia",
    "Malta",
    "Morocco",
    "New Zealand",
    "North Macedonia",
    "Pakistan",
    "Peru",
    "Philippines",
    "Poland",
    "Portugal",
    "Romania",
    "Serbia",
    "Slovakia",
    "Slovenia",
    "South Africa",
    "Spain",
    "Tunisia",
    "United Kingdom",
}


def load_and_process_country(
    df_all, country, start=COVID_START, end=COVID_END
):
    """Filter OWID data and compute S-I-R compartment trajectories for a country.

    Args:
        df_all (pd.DataFrame): The full OWID dataset.
        country (str): Name of the country to process.
        start (str): Start date for filtering.
        end (str): End date for filtering.

    Returns:
        tuple: (states, N_pop) where states is an (n_days, 3) numpy array
            containing (S, I, R) fractions, and N_pop is the population size.
    """
    df = (
        df_all.query("location == @country")
        .assign(date=lambda d: pd.to_datetime(d["date"]))
        .query("@start <= date <= @end")
        .sort_values("date")
        .reset_index(drop=True)
    )

    if df.empty or len(df) < 30:
        return None, None

    df["new_cases_smoothed"] = (
        df["new_cases_smoothed"].fillna(0.0).clip(lower=0.0)
    )
    N_pop = float(df["population"].iloc[0])

    # Map reported cases to estimated infectious population I(t)
    daily_infections = COVID_MULTIPLIER * df["new_cases_smoothed"].to_numpy()
    I_counts = (
        pd.Series(daily_infections)
        .rolling(COVID_WINDOW, min_periods=1)
        .sum()
        .to_numpy()
    )
    cum_infections = np.cumsum(daily_infections)

    # Approximate recovery (R) and susceptible (S) compartments
    R_counts = np.maximum(cum_infections - I_counts, 0.0)
    S_counts = np.maximum(N_pop - I_counts - R_counts, 0.0)

    states = np.column_stack(
        [S_counts / N_pop, I_counts / N_pop, R_counts / N_pop]
    )
    return states, N_pop


def plot_coefficients_bar(coeffs, feature_names, output_path):
    """Generate horizontal bar charts of discovered model-error correction terms."""
    apply_publication_theme()
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    for i, label in enumerate(["S", "I", "R"]):
        ax = axes[i]
        c = coeffs[:, i]
        mask = np.abs(c) > 1e-4

        if not np.any(mask):
            ax.text(
                0.5,
                0.5,
                "NO CORRECTION",
                ha="center",
                fontweight="bold",
                alpha=0.5,
                transform=ax.transAxes,
            )
        else:
            sorted_idx = np.argsort(np.abs(c[mask]))
            vals = c[mask][sorted_idx]
            ylabs = [feature_names[j] for j, m in enumerate(mask) if m]
            ylabs = [ylabs[k] for k in sorted_idx]

            # Use diverging palette to distinguish positive and negative contributions
            palette = sns.color_palette("vlag", n_colors=len(vals))
            ax.barh(ylabs, vals, color=palette, edgecolor="white", height=0.6)
            ax.axvline(0, color="black", lw=1, alpha=0.4)

        ax.set_title(f"CORRECTION FOR d{label}/dt", fontweight="black")
        ax.set_xlabel("COEFFICIENT VALUE")

    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_country_comparison(country_results, output_path):
    """3×3 grid comparing discovered correction terms across Italy, Sweden, Germany."""
    apply_publication_theme()
    countries = [cr["country"] for cr in country_results]
    fig, axes = plt.subplots(
        len(countries), 3, figsize=(20, 5 * len(countries))
    )

    labels = ["S", "I", "R"]

    for row, cr in enumerate(country_results):
        coeffs = cr["coeffs"]
        feature_names = cr["feature_names"]
        beta = cr["beta"]
        n_terms = int(np.sum(np.abs(coeffs) > 1e-4))
        # Scale x-axis to each country's own range so small corrections are readable
        row_max = max(np.max(np.abs(coeffs)), 1e-3)

        for col, label in enumerate(labels):
            ax = axes[row, col]
            c = coeffs[:, col]
            mask = np.abs(c) > 1e-4

            if not np.any(mask):
                ax.text(
                    0.5,
                    0.5,
                    "NO CORRECTION",
                    ha="center",
                    fontweight="bold",
                    alpha=0.5,
                    transform=ax.transAxes,
                )
            else:
                sorted_idx = np.argsort(np.abs(c[mask]))
                vals = c[mask][sorted_idx]
                ylabs = [feature_names[j] for j, m in enumerate(mask) if m]
                ylabs = [ylabs[k] for k in sorted_idx]
                palette = sns.color_palette("vlag", n_colors=len(vals))
                ax.barh(
                    ylabs, vals, color=palette, edgecolor="white", height=0.6
                )
                ax.axvline(0, color="black", lw=1, alpha=0.4)

            ax.set_xlim(-row_max * 1.2, row_max * 1.2)
            if col == 0:
                ax.set_ylabel(
                    f"{cr['country'].upper()}\nβ={beta:.3f} | {n_terms} terms",
                    fontweight="black",
                    fontsize=13,
                )
            if row == 0:
                ax.set_title(f"CORRECTION FOR d{label}/dt", fontweight="black")
            ax.set_xlabel("COEFFICIENT VALUE")

    fig.suptitle(
        "HOW LOCKDOWN POLICY SHAPES THE DISCOVERED CORRECTION",
        fontsize=18,
        fontweight="black",
        y=1.01,
    )
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_landscape_ranking(df, title, filename, train_country):
    """Visualize accuracy improvements across a set of countries as a ranked bar chart."""
    apply_publication_theme()
    df = df.copy().reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharex=True)

    success_color, fail_color = "#2c7bb6", "#d7191c"
    mid = len(df) // 2

    for i, ax in enumerate(axes):
        start_idx, end_idx = i * mid, (i + 1) * mid if i == 0 else len(df)
        sub_df = df.iloc[start_idx:end_idx].copy().sort_values("improvement")
        y_pos = np.arange(len(sub_df))

        # Color-code based on whether the country implemented a national lockdown
        colors = [
            success_color if lock else fail_color
            for lock in sub_df["lockdown"]
        ]
        ax.barh(
            y_pos,
            sub_df["improvement"],
            color=colors,
            height=0.7,
            edgecolor="white",
            lw=1,
            alpha=0.9,
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub_df["country"], fontsize=13, fontweight="bold")

        for j, x in enumerate(sub_df["improvement"]):
            ax.text(
                x + 0.05,
                j,
                f"{x:.1f}x",
                va="center",
                fontsize=12,
                fontweight="black",
            )

        ax.axvline(1, color="black", lw=2, ls="-", alpha=0.5)
        ax.set_xlabel("ACCURACY IMPROVEMENT FACTOR", fontweight="bold")

    fig.suptitle(title.upper(), fontsize=22, fontweight="black", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_global_choropleth(results_df, train_country, output_path):
    """Generate a global choropleth map showing model-error reduction by geography."""
    import plotly.express as px

    # Map country names to Plotly-compatible ISO conventions
    NAME_FIXES = {
        "United States": "United States of America",
        "Democratic Republic of Congo": "Democratic Republic of the Congo",
        "Congo": "Republic of the Congo",
        "Czechia": "Czech Republic",
    }

    plot_df = results_df.copy()
    plot_df["country_plot"] = plot_df["country"].replace(NAME_FIXES)

    # Cap improvement factor for map visualization to prevent outliers from saturating the scale
    cap = plot_df["improvement"].quantile(0.95)
    plot_df["improvement_capped"] = plot_df["improvement"].clip(upper=cap)

    fig = px.choropleth(
        plot_df,
        locations="country_plot",
        locationmode="country names",
        color="improvement_capped",
        hover_name="country",
        color_continuous_scale="RdYlGn",
        range_color=(0, cap),
        title=f"GLOBAL ACCURACY GAIN (TRAINED ON {train_country.upper()})",
    )

    fig.update_layout(
        title_font_size=18,
        title_x=0.5,
        margin=dict(l=0, r=0, t=40, b=0),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type="equirectangular",
        ),
    )
    fig.write_image(output_path, scale=2)


def plot_effective_beta(
    states,
    imp_coeffs,
    train_coeffs,
    lib,
    beta_est,
    output_path,
    has_lockdown=True,
):
    """Visualize the time-varying transmission rate discovered by MEDIDA."""
    apply_publication_theme()
    op = states[:-1]
    Phi = lib.transform(op)
    S, I = op[:, 0], op[:, 1]

    # Derive effective beta: beta(t) = - (dS/dt) / (S * I)
    dS_corr = (Phi @ train_coeffs)[:, 0]
    SI = np.clip(S * I, 1e-12, None)
    beta_eff_corr = -dS_corr / SI

    # Cap y-axis at a fraction of naive β so the variation in β(t) fills the plot.
    y_cap = beta_est * 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    t = np.arange(len(op))
    ax.plot(
        t,
        np.clip(beta_eff_corr, 0, y_cap),
        color="#33a02c",
        lw=3,
        label="MEDIDA β(t)",
    )
    if has_lockdown:
        ax.axvline(
            COVID_LOCKDOWN_DAY,
            color="#6a3d9a",
            lw=2,
            ls=":",
            label="Lockdown onset (Day 8)",
        )

    # Naive β is above the visible range — draw it as a clipped line + annotation
    ax.axhline(y_cap, color="#e31a1c", lw=1.5, ls="--", alpha=0.4)
    ax.annotate(
        f"Naive SIR: β = {beta_est:.3f} (constant, above scale)",
        xy=(len(op) * 0.05, y_cap),
        xytext=(len(op) * 0.05, y_cap * 0.88),
        color="#e31a1c",
        fontsize=12,
        fontweight="bold",
    )

    ax.set_ylim(0, y_cap)
    ax.set_xlabel("DAYS SINCE START")
    ax.set_ylabel("EFFECTIVE β(t)")
    ax.set_title(
        "CORRECTED MODEL DISCOVERS TIME-VARYING TRANSMISSION",
        fontweight="black",
    )
    ax.legend(loc="upper right")
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_epidemic_residuals(
    states, N_pop, imp_coeffs, train_coeffs, lib, output_path
):
    """Compare population-level forecasts and residuals between baseline and MEDIDA."""
    apply_publication_theme()
    op = states[:-1]
    oc = states[1:]
    Phi = lib.transform(op)
    p_m = op + COVID_DT * (Phi @ imp_coeffs)
    p_s = op + COVID_DT * (Phi @ train_coeffs)
    t = np.arange(1, len(states))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left Panel: Absolute Infectious curves
    ax = axes[0]
    ax.fill_between(
        np.arange(len(states)),
        states[:, 1] * N_pop,
        alpha=0.15,
        color="k",
        label="Observed",
    )
    ax.plot(np.arange(len(states)), states[:, 1] * N_pop, "k-", lw=3)
    ax.plot(t, p_m[:, 1] * N_pop, "r--", lw=2.5, label="Naive SIR")
    ax.plot(t, p_s[:, 1] * N_pop, color="#33a02c", lw=3, label="MEDIDA")
    ax.set_xlabel("DAYS SINCE START")
    ax.set_ylabel("INFECTIOUS POPULATION")
    ax.set_title("EPIDEMIC CURVE", fontweight="black")
    ax.legend()

    # Right Panel: One-step prediction residuals in people counts
    ax = axes[1]
    ax.axhline(0, color="k", lw=1, alpha=0.3)
    ax.plot(
        t,
        (p_m[:, 1] - oc[:, 1]) * N_pop,
        "r--",
        lw=2.5,
        label="Naive residuals",
    )
    ax.plot(
        t,
        (p_s[:, 1] - oc[:, 1]) * N_pop,
        color="#33a02c",
        lw=3,
        label="MEDIDA residuals",
    )
    ax.set_xlabel("DAYS SINCE START")
    ax.set_ylabel("ONE-STEP RESIDUAL (PEOPLE)")
    ax.set_title("PREDICTION RESIDUALS", fontweight="black")
    ax.legend()

    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_transfer_country(
    df_all,
    train_coeffs,
    imp_coeffs,
    lib,
    train_country,
    transfer_country,
    output_path,
):
    """One-step accuracy transfer test: apply correction learned from one country to another."""
    apply_publication_theme()
    states, N_pop = load_and_process_country(df_all, transfer_country)
    if states is None:
        return

    op = states[:-1]
    oc = states[1:]
    Phi = lib.transform(op)
    p_m = op + COVID_DT * (Phi @ imp_coeffs)
    p_s = op + COVID_DT * (Phi @ train_coeffs)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    ax.fill_between(
        np.arange(len(states)),
        states[:, 1] * N_pop,
        alpha=0.15,
        color="k",
        label="Observed",
    )
    ax.plot(np.arange(len(states)), states[:, 1] * N_pop, "k-", lw=3)
    ax.plot(
        np.arange(1, len(states)),
        p_m[:, 1] * N_pop,
        "r--",
        lw=2.5,
        label="Naive SIR",
    )
    ax.plot(
        np.arange(1, len(states)),
        p_s[:, 1] * N_pop,
        color="#33a02c",
        lw=3,
        label=f"{train_country} Correction",
    )
    ax.set_xlabel("DAYS SINCE START")
    ax.set_ylabel("INFECTIOUS POPULATION")
    ax.set_title(
        f"TRANSFER TO {transfer_country.upper()}: EPIDEMIC CURVE",
        fontweight="black",
    )
    ax.legend()

    ax = axes[1]
    ax.axhline(0, color="k", lw=1, alpha=0.3)
    ax.plot(
        np.arange(1, len(states)),
        (p_m[:, 1] - oc[:, 1]) * N_pop,
        "r--",
        lw=2.5,
        label="Naive residuals",
    )
    ax.plot(
        np.arange(1, len(states)),
        (p_s[:, 1] - oc[:, 1]) * N_pop,
        color="#33a02c",
        lw=3,
        label=f"{train_country} correction residuals",
    )
    ax.set_xlabel("DAYS SINCE START")
    ax.set_ylabel("ONE-STEP RESIDUAL (PEOPLE)")
    ax.set_title("PREDICTION RESIDUALS", fontweight="black")
    ax.legend()

    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_failure_grid(
    df_all, results_df, train_coeffs, imp_coeffs, lib, output_path
):
    """Visualize trajectories for countries where the correction failed to improve accuracy."""
    apply_publication_theme()
    failures = results_df[results_df["improvement"] < 0.95].nsmallest(
        6, "improvement"
    )
    if failures.empty:
        return

    fig, axes = plt.subplots(6, 3, figsize=(20, 28))
    for row, (_, fail_row) in enumerate(failures.iterrows()):
        country = fail_row["country"]
        states, N_pop = load_and_process_country(df_all, country)
        if states is None:
            continue

        op = states[:-1]
        oc = states[1:]
        Phi = lib.transform(op)
        p_m = op + COVID_DT * (Phi @ imp_coeffs)
        p_s = op + COVID_DT * (Phi @ train_coeffs)

        for col, label in enumerate(["S", "I", "R"]):
            ax = axes[row, col]
            ax.plot(
                states[:, col] * N_pop,
                "k-",
                lw=3,
                label="Observed" if row == 0 else None,
            )
            ax.plot(
                np.arange(1, len(states)),
                p_m[:, col] * N_pop,
                "r--",
                alpha=0.6,
                label="Naive" if row == 0 else None,
            )
            ax.plot(
                np.arange(1, len(states)),
                p_s[:, col] * N_pop,
                ":",
                color="#2ca02c",
                lw=3,
                label="MEDIDA" if row == 0 else None,
            )
            if col == 0:
                ax.set_ylabel(
                    f"{country.upper()}", fontweight="black", fontsize=14
                )
            if row == 0:
                ax.set_title(f"COMPARTMENT {label}", fontweight="black")

    axes[0, 0].legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def load_owid_data(force_download=False):
    """Load the cached OWID COVID data, downloading it when needed."""
    data_dir = os.path.join(project_root, "data")
    local_path = os.path.join(data_dir, "owid-covid-data.csv")
    os.makedirs(data_dir, exist_ok=True)

    if force_download or not os.path.exists(local_path):
        print(f"Downloading OWID COVID data to {local_path}")
        df_all = pd.read_csv(COVID_URL)
        df_all.to_csv(local_path, index=False)
        return df_all

    return pd.read_csv(local_path)


def main():
    parser = argparse.ArgumentParser(
        description="MEDIDA COVID-19 Model Discovery and Validation"
    )
    parser.add_argument(
        "--train-country",
        type=str,
        default="Italy",
        help="Country used for error-term discovery",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Execute global accuracy gain sweep across all countries",
    )
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Download/cache the OWID COVID CSV and exit without analysis",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Force re-download of the OWID COVID CSV before continuing",
    )
    args = parser.parse_args()

    df_all = load_owid_data(force_download=args.refresh_data)
    if args.download_data:
        print("OWID COVID data is ready in data/owid-covid-data.csv")
        return

    output_dir = (
        f"outputs/covid/{args.train_country.lower().replace(' ', '_')}"
    )
    os.makedirs(output_dir, exist_ok=True)
    apply_publication_theme()

    states_train, N_pop_train = load_and_process_country(
        df_all, args.train_country
    )
    if states_train is None:
        print(f"Error: Could not load data for {args.train_country}")
        return

    # Estimate baseline SIR parameters using pre-lockdown growth rate
    gamma_est = 1.0 / COVID_WINDOW
    growth_I = states_train[: COVID_LOCKDOWN_DAY + 1, 1]
    slope, _ = np.polyfit(
        np.arange(len(growth_I)), np.log(growth_I + 1e-15), 1
    )
    beta_est = float(slope) + gamma_est

    lib = PolynomialLibrary(n_vars=3, degree=2, var_names=["S", "I", "R"])
    imp_coeffs = np.zeros((lib.n_features, 3))
    imp_coeffs[lib.index("S I"), 0], imp_coeffs[lib.index("S I"), 1] = (
        -beta_est,
        beta_est,
    )
    imp_coeffs[lib.index("I"), 1], imp_coeffs[lib.index("I"), 2] = (
        -gamma_est,
        gamma_est,
    )

    # Identify model-error correction terms
    medida = MEDIDA(
        PolynomialODE(imp_coeffs, lib), lib, dt=COVID_DT, significance=1e-8
    )
    result = medida.fit(states_train[:-1], states_train[1:])
    train_coeffs = result.corrected_coefficients(imp_coeffs)

    # Core Visual Artifacts
    save_discovery_card(
        result.error_coefficients,
        lib.feature_names,
        ["S", "I", "R"],
        {"Dataset": "OWID", "Training": args.train_country},
        os.path.join(output_dir, "discovery_card.png"),
        title=f"Real Data: {args.train_country}",
    )

    plot_coefficients_bar(
        result.error_coefficients,
        lib.feature_names,
        os.path.join(output_dir, "discovered_terms_bar.png"),
    )

    plot_effective_beta(
        states_train,
        imp_coeffs,
        train_coeffs,
        lib,
        beta_est,
        os.path.join(output_dir, "effective_beta.png"),
        has_lockdown=args.train_country in LOCKDOWN_COUNTRIES,
    )

    plot_epidemic_residuals(
        states_train,
        N_pop_train,
        imp_coeffs,
        train_coeffs,
        lib,
        os.path.join(output_dir, "epidemic_curve_residuals.png"),
    )

    # Temporal holdout: train on days 0-199, then FREE ROLLOUT from day 200 (no observations)
    if len(states_train) > 250:
        res_h = medida.fit(states_train[:199], states_train[1:200])
        c_h = res_h.corrected_coefficients(imp_coeffs)

        n_forecast = len(states_train) - 200
        # Multi-step rollout: integrate corrected and naive models forward from day 200
        rollout_medida = np.zeros((n_forecast + 1, 3))
        rollout_naive = np.zeros((n_forecast + 1, 3))
        rollout_medida[0] = rollout_naive[0] = states_train[200]
        for step in range(n_forecast):
            Phi_s = lib.transform(rollout_medida[step : step + 1])
            rollout_medida[step + 1] = np.clip(
                rollout_medida[step] + COVID_DT * (Phi_s @ c_h), 0, 1
            )
            Phi_n = lib.transform(rollout_naive[step : step + 1])
            rollout_naive[step + 1] = np.clip(
                rollout_naive[step] + COVID_DT * (Phi_n @ imp_coeffs), 0, 1
            )

        fig, ax = plt.subplots(figsize=(14, 7))
        t_full = np.arange(len(states_train))
        t_fore = np.arange(200, 200 + n_forecast + 1)
        obs_peak = float(np.max(states_train[:, 1]) * N_pop_train)
        y_cap = obs_peak * 2.5

        naive_I = rollout_naive[:, 1] * N_pop_train
        naive_peak = float(np.max(naive_I))

        ax.plot(
            t_full,
            states_train[:, 1] * N_pop_train,
            "k-",
            lw=4,
            label=f"Observed ({args.train_country} I)",
        )
        ax.plot(
            t_fore,
            np.clip(naive_I, 0, y_cap),
            "r--",
            lw=2.5,
            label="Naive SIR (free rollout)",
            alpha=0.7,
        )
        ax.plot(
            t_fore,
            rollout_medida[:, 1] * N_pop_train,
            color="#33a02c",
            lw=3,
            ls=":",
            label="MEDIDA (free rollout from Day 200)",
        )
        ax.axvspan(0, 200, color="gray", alpha=0.1, label="Training Period")
        ax.set_ylim(0, y_cap)

        if naive_peak > y_cap:
            ax.annotate(
                f"Naive SIR peaks at {naive_peak/1e6:.0f}M (off-scale)",
                xy=(t_fore[np.argmax(naive_I)], y_cap),
                xytext=(t_fore[5], y_cap * 0.88),
                color="#e31a1c",
                fontsize=11,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#e31a1c"),
            )

        ax.set_title(
            f"{args.train_country.upper()} TEMPORAL HOLDOUT: FREE ROLLOUT FROM DAY 200",
            fontweight="black",
        )
        ax.set_xlabel("DAYS SINCE START")
        ax.set_ylabel("INFECTIOUS POPULATION")
        ax.legend()
        sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "temporal_holdout.png"), dpi=200)
        plt.close()

    # Transfer test: apply correction to France as a benchmark
    if args.train_country != "France":
        plot_transfer_country(
            df_all,
            train_coeffs,
            imp_coeffs,
            lib,
            args.train_country,
            "France",
            os.path.join(output_dir, f"transfer_france.png"),
        )

    if args.train_country == "Italy":
        # 3-country comparison: how lockdown policy shapes the discovered correction
        country_comp = [
            {
                "country": "Italy",
                "coeffs": result.error_coefficients,
                "feature_names": lib.feature_names,
                "beta": beta_est,
            }
        ]
        for comp_country in ["Sweden", "Germany"]:
            s_c, _ = load_and_process_country(df_all, comp_country)
            g_c = s_c[: COVID_LOCKDOWN_DAY + 1, 1]
            sl, _ = np.polyfit(np.arange(len(g_c)), np.log(g_c + 1e-15), 1)
            b_c = float(sl) + gamma_est
            ic_c = np.zeros((lib.n_features, 3))
            ic_c[lib.index("S I"), 0], ic_c[lib.index("S I"), 1] = -b_c, b_c
            ic_c[lib.index("I"), 1], ic_c[lib.index("I"), 2] = (
                -gamma_est,
                gamma_est,
            )
            m_c = MEDIDA(
                PolynomialODE(ic_c, lib), lib, dt=COVID_DT, significance=1e-8
            )
            r_c = m_c.fit(s_c[:-1], s_c[1:])
            country_comp.append(
                {
                    "country": comp_country,
                    "coeffs": r_c.error_coefficients,
                    "feature_names": lib.feature_names,
                    "beta": b_c,
                }
            )
        plot_country_comparison(
            country_comp, os.path.join(output_dir, "country_comparison.png")
        )

    if args.sweep:
        results = []
        for country in df_all["location"].unique():
            try:
                states_c, N_c = load_and_process_country(df_all, country)
                if states_c is None or len(states_c) < 200:
                    continue
                Phi = lib.transform(states_c[:-1])
                p_m, p_s = states_c[:-1] + COVID_DT * (
                    Phi @ imp_coeffs
                ), states_c[:-1] + COVID_DT * (Phi @ train_coeffs)
                r_m, r_s = float(
                    np.sqrt(np.mean((states_c[1:, 1] - p_m[:, 1]) ** 2))
                ), float(np.sqrt(np.mean((states_c[1:, 1] - p_s[:, 1]) ** 2)))
                results.append(
                    {
                        "country": country,
                        "improvement": r_m / r_s,
                        "lockdown": country in LOCKDOWN_COUNTRIES,
                    }
                )
            except:
                continue

        results_df = pd.DataFrame(results).sort_values(
            "improvement", ascending=False
        )
        results_df.to_csv(
            os.path.join(output_dir, "sweep_results.csv"), index=False
        )
        plot_global_choropleth(
            results_df,
            args.train_country,
            os.path.join(output_dir, "global_map.png"),
        )

        plot_landscape_ranking(
            results_df.nlargest(50, "improvement"),
            f"Top 50 Accuracy Gains: {args.train_country}",
            os.path.join(output_dir, "top50_landscape.png"),
            args.train_country,
        )

        plot_landscape_ranking(
            results_df.nsmallest(50, "improvement"),
            f"Bottom 50 Accuracy Gains: {args.train_country}",
            os.path.join(output_dir, "bottom50_landscape.png"),
            args.train_country,
        )

        plot_failure_grid(
            df_all,
            results_df,
            train_coeffs,
            imp_coeffs,
            lib,
            os.path.join(output_dir, "failure_analysis_grid.png"),
        )


if __name__ == "__main__":
    main()
