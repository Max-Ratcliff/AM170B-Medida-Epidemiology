import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from matplotlib.patches import Patch

# Add the project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils import save_latex_correction, apply_publication_theme
from medida import (
    MEDIDA, PolynomialODE, PolynomialLibrary, 
    RelevanceVectorMachine, format_system, relative_error
)

# Global configuration
COVID_URL = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
COVID_START = "2020-03-01"
COVID_END = "2021-01-15"
COVID_MULTIPLIER = 4
COVID_WINDOW = 14
COVID_DT = 1.0
COVID_LOCKDOWN_DAY = 8

LOCKDOWN_COUNTRIES = {
    "Austria", "Belgium", "Bolivia", "Bosnia and Herzegovina", "Bulgaria",
    "Colombia", "Croatia", "Czechia", "Ecuador", "El Salvador", "France",
    "Germany", "Greece", "Hungary", "India", "Ireland", "Israel", "Italy",
    "Jordan", "Lebanon", "Lithuania", "Malaysia", "Malta", "Morocco",
    "New Zealand", "North Macedonia", "Pakistan", "Peru", "Philippines",
    "Poland", "Portugal", "Romania", "Serbia", "Slovakia", "Slovenia",
    "South Africa", "Spain", "Tunisia", "United Kingdom",
}

def load_and_process_country(df_all, country, start=COVID_START, end=COVID_END):
    df = (df_all
          .query("location == @country")
          .assign(date=lambda d: pd.to_datetime(d["date"]))
          .query("@start <= date <= @end")
          .sort_values("date")
          .reset_index(drop=True))
    
    if df.empty or len(df) < 30:
        return None, None
        
    df["new_cases_smoothed"] = df["new_cases_smoothed"].fillna(0.0).clip(lower=0.0)
    N_pop = float(df["population"].iloc[0])
    daily_infections = COVID_MULTIPLIER * df["new_cases_smoothed"].to_numpy()

    I_counts = pd.Series(daily_infections).rolling(COVID_WINDOW, min_periods=1).sum().to_numpy()
    cum_infections = np.cumsum(daily_infections)
    R_counts = np.maximum(cum_infections - I_counts, 0.0)
    S_counts = np.maximum(N_pop - I_counts - R_counts, 0.0)

    states = np.column_stack([S_counts / N_pop, I_counts / N_pop, R_counts / N_pop])
    return states, N_pop

def plot_landscape_ranking(df, title, filename, train_country):
    """Create a presentation-ready 2-column landscape ranking plot."""
    apply_publication_theme()
    df = df.copy().reset_index(drop=True)
    n = len(df)
    mid = n // 2
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 10), sharex=True)
    for i, ax in enumerate(axes):
        start_idx = i * mid
        end_idx = (i + 1) * mid if i == 0 else n
        sub_df = df.iloc[start_idx:end_idx].copy().sort_values("improvement")
        
        y_pos = np.arange(len(sub_df))
        colors = ["#1f77b4" if lock else "#d62728" for lock in sub_df["lockdown"]]
        ax.hlines(y_pos, 1, sub_df["improvement"], color='gray', alpha=0.2, lw=0.8)
        ax.scatter(sub_df["improvement"], y_pos, color=colors, s=70, edgecolors='white', zorder=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub_df["country"], fontsize=11)
        ax.axvline(1, color="black", lw=1.5, ls="--", alpha=0.3)
        for y in range(len(sub_df)):
            if y % 2 == 0:
                ax.axhspan(y-0.5, y+0.5, color='gray', alpha=0.03, zorder=0)
        ax.set_xlabel("RMSE Improvement Ratio", fontsize=12)

    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
    legend_elements = [
        Patch(color="#1f77b4", label="Lockdown (2020)"),
        Patch(color="#d62728", label="No Lockdown"),
        plt.Line2D([0], [0], color="black", ls="--", label="Break-even")
    ]
    axes[1].legend(handles=legend_elements, loc='lower right', fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename)
    plt.close()

def run_global_sweep(df_all, train_country, corrected_coeffs, imp_coeffs, lib, output_dir):
    print(f"[*] Running Global Sweep...")
    results = []
    MIN_DAYS, MIN_CASES, MIN_POP = 200, 100, 1e6
    candidate_locations = df_all.groupby("location").filter(
        lambda g: g["new_cases_smoothed"].max() >= MIN_CASES and g["population"].iloc[0] >= MIN_POP
    )["location"].unique()

    for country in candidate_locations:
        try:
            states_c, N_c = load_and_process_country(df_all, country)
            if states_c is None or len(states_c) < MIN_DAYS: continue
            op, oc = states_c[:-1], states_c[1:]
            Phi_c = lib.transform(op)
            p_bl = op + COVID_DT * (Phi_c @ imp_coeffs)
            p_mc = op + COVID_DT * (Phi_c @ corrected_coeffs)
            r_bl = float(np.sqrt(np.mean((oc[:, 1] - p_bl[:, 1]) ** 2)))
            r_mc = float(np.sqrt(np.mean((oc[:, 1] - p_mc[:, 1]) ** 2)))
            if r_mc > 0:
                results.append({"country": country, "improvement": r_bl / r_mc, "lockdown": country in LOCKDOWN_COUNTRIES})
        except: continue

    results_df = pd.DataFrame(results).sort_values("improvement", ascending=False)
    results_df.to_csv(os.path.join(output_dir, "sweep_data.csv"), index=False)
    
    top50 = results_df.nlargest(50, "improvement")
    plot_landscape_ranking(top50, f"Top 50 Improvements: Discovered in {train_country}",
                           os.path.join(output_dir, "top50_landscape.png"), train_country)
    
    print(f"[*] Sweep complete. Median Improvement: {results_df['improvement'].median():.2f}x")

def main():
    parser = argparse.ArgumentParser(description="MEDIDA COVID-19 Research Tool")
    parser.add_argument("--train-country", type=str, default="Italy", help="Country to train on")
    parser.add_argument("--sweep", action="store_true", help="Run global sweep across all countries")
    args = parser.parse_args()

    output_dir = f"outputs/covid/{args.train_country.lower().replace(' ', '_')}"
    os.makedirs(output_dir, exist_ok=True)
    apply_publication_theme()

    print("\n" + "="*70)
    print(f"COVID-19 MEDIDA: {args.train_country}")
    print("="*70)

    # Data Loading
    data_dir = os.path.join(project_root, "data")
    local_path = os.path.join(data_dir, "owid-covid-data.csv")
    if os.path.exists(local_path):
        df_all = pd.read_csv(local_path, usecols=["location", "date", "new_cases_smoothed", "population"])
    else:
        os.makedirs(data_dir, exist_ok=True)
        df_all = pd.read_csv(COVID_URL, usecols=["location", "date", "new_cases_smoothed", "population"])
        df_all.to_csv(local_path, index=False)
    
    # Training
    states_train, N_pop_train = load_and_process_country(df_all, args.train_country)
    gamma_est = 1.0 / COVID_WINDOW
    growth_I = states_train[:COVID_LOCKDOWN_DAY+1, 1]
    slope, _ = np.polyfit(np.arange(len(growth_I)), np.log(growth_I + 1e-15), 1)
    beta_est = float(slope) + gamma_est
    
    lib = PolynomialLibrary(n_vars=3, degree=2, var_names=["S", "I", "R"])
    imp_coeffs = np.zeros((lib.n_features, 3))
    imp_coeffs[lib.index("S I"), 0], imp_coeffs[lib.index("S I"), 1] = -beta_est, beta_est
    imp_coeffs[lib.index("I"), 1], imp_coeffs[lib.index("I"), 2] = -gamma_est, gamma_est
    
    model_m = PolynomialODE(imp_coeffs, lib, state_names=["S", "I", "R"])
    medida = MEDIDA(model_m, lib, dt=COVID_DT, significance=1e-8)
    result = medida.fit(states_train[:-1], states_train[1:])
    corrected_coeffs = result.corrected_coefficients(imp_coeffs)

    # LaTeX Save
    save_latex_correction(result.error_coefficients, lib.feature_names, ["S", "I", "R"], 
                       {"Train Country": args.train_country}, 
                       os.path.join(output_dir, "discovery.tex"),
                       title=f"Correction: {args.train_country}")

    # Visual (Effective Beta)
    Phi = lib.transform(states_train[:-1])
    h_S = Phi @ result.error_coefficients[:, 0]
    delta_beta = -h_S / np.clip(states_train[:-1, 0] * states_train[:-1, 1], 1e-12, None)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(beta_est + delta_beta, "-", color="#2ca02c", lw=3, label="MEDIDA $\\beta(t)$")
    ax.axhline(beta_est, color="#d62728", ls="--", label="Naive constant $\\beta$")
    ax.set_title(f"{args.train_country}: Discovered Transmission", fontweight="bold")
    ax.set_ylabel("Transmission Rate $\\beta$")
    ax.legend()
    plt.savefig(os.path.join(output_dir, "effective_beta.png"))
    plt.close()

    if args.sweep:
        run_global_sweep(df_all, args.train_country, corrected_coeffs, imp_coeffs, lib, output_dir)

if __name__ == "__main__":
    main()
