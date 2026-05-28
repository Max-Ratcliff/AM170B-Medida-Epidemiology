import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to sys.path to allow importing the 'medida' package
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from medida import (
    MEDIDA, PolynomialODE, PolynomialLibrary, 
    RelevanceVectorMachine, format_system
)

def run_covid_example():
    print("\n" + "="*70)
    print("COVID-19 Real-World Analysis: Italy")
    print("="*70)

    # 1. Data Download
    COVID_URL = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
    print(f"[*] Downloading OWID COVID-19 dataset...")
    try:
        df_all = pd.read_csv(COVID_URL, usecols=["location", "date", "new_cases_smoothed", "population"])
    except Exception as e:
        print(f"  Error: {e}")
        return

    # 2. Parameters (Matching Notebook 6.1)
    COVID_COUNTRY = "Italy"
    COVID_START = "2020-03-01"
    COVID_END = "2021-01-15"
    COVID_LOCKDOWN_DAY = 8
    COVID_MULTIPLIER = 4
    COVID_WINDOW = 14
    COVID_DT = 1.0

    print(f"[*] Processing data for {COVID_COUNTRY} ({COVID_START} to {COVID_END})...")
    df = (df_all
          .query("location == @COVID_COUNTRY")
          .assign(date=lambda d: pd.to_datetime(d["date"]))
          .query("@COVID_START <= date <= @COVID_END")
          .copy())
    
    df["new_cases_smoothed"] = df["new_cases_smoothed"].fillna(0.0).clip(lower=0.0)
    N_pop = float(df["population"].iloc[0])
    daily_infections = COVID_MULTIPLIER * df["new_cases_smoothed"].to_numpy()

    # 3. State Reconstruction (Matching Notebook 6.1)
    I_counts = pd.Series(daily_infections).rolling(COVID_WINDOW, min_periods=1).sum().to_numpy()
    cum_infections = np.cumsum(daily_infections)
    R_counts = np.maximum(cum_infections - I_counts, 0.0)
    S_counts = np.maximum(N_pop - I_counts - R_counts, 0.0)

    covid_states = np.column_stack([S_counts / N_pop, I_counts / N_pop, R_counts / N_pop])
    
    # 4. Imperfect Model: Naive SIR with early beta (Matching Notebook 6.2)
    print("[*] Estimating naive SIR parameters from pre-lockdown growth...")
    lockdown_date = pd.Timestamp(COVID_START) + pd.Timedelta(days=COVID_LOCKDOWN_DAY)
    dates = df["date"].to_numpy()
    growth_mask = ((dates >= pd.Timestamp(COVID_START)) & (dates <= lockdown_date) & (covid_states[:, 1] > 0))
    
    growth_t = np.arange(np.sum(growth_mask), dtype=float)
    growth_I = covid_states[growth_mask, 1]
    
    gamma_est = 1.0 / COVID_WINDOW
    if len(growth_I) >= 3:
        slope, _ = np.polyfit(growth_t, np.log(growth_I + 1e-15), 1)
        growth_rate = float(slope)
    else:
        growth_rate = 0.25
    beta_est = growth_rate + gamma_est

    print(f"  Estimated Gamma: {gamma_est:.4f}")
    print(f"  Estimated Early Beta: {beta_est:.4f}")

    lib = PolynomialLibrary(n_vars=3, degree=2, var_names=["S", "I", "R"])
    imp_coeffs = np.zeros((lib.n_features, 3))
    imp_coeffs[lib.index("S I"), 0] = -beta_est
    imp_coeffs[lib.index("S I"), 1] = beta_est
    imp_coeffs[lib.index("I"), 1] = -gamma_est
    imp_coeffs[lib.index("I"), 2] = gamma_est
    
    model_m = PolynomialODE(imp_coeffs, lib, state_names=["S", "I", "R"])

    # 5. MEDIDA Fit (Matching Notebook 6.3)
    print("[*] Running MEDIDA structural discovery...")
    rvm = RelevanceVectorMachine(max_iter=2000, tol=1e-5, t_min=1.5, threshold=0.005, normalize=True)
    medida = MEDIDA(model_m, lib, dt=COVID_DT, rvm=rvm, significance=1e-8)
    
    u_prev = covid_states[:-1]
    u_curr = covid_states[1:]
    
    result = medida.fit(u_prev, u_curr)

    print("\nResults:")
    print("Discovered Structural Correction:")
    print(format_system(result.error_coefficients, lib.feature_names, ["S", "I", "R"]))

    # 6. Prediction Comparison
    print(f"[*] Generating diagnostic plots: figures/covid_{COVID_COUNTRY.lower()}.png")
    t = np.arange(len(covid_states))
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # 1-step prediction comparison
    Phi = lib.transform(u_prev)
    pred_m = u_prev + COVID_DT * (Phi @ imp_coeffs)
    pred_star = u_prev + COVID_DT * (Phi @ (imp_coeffs + result.error_coefficients))
    
    ax.plot(t, covid_states[:, 1] * N_pop, 'r-', lw=2, label="Observed I")
    ax.plot(t[1:], pred_m[:, 1] * N_pop, 'b--', alpha=0.6, label="Naive SIR 1-step")
    ax.plot(t[1:], pred_star[:, 1] * N_pop, 'g:', lw=2, label="MEDIDA-corrected 1-step")
    
    ax.set_title(f"COVID-19 Model Correction: {COVID_COUNTRY}", fontweight="bold")
    ax.set_xlabel("Days since start")
    ax.set_ylabel("Active infectious people")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"figures/covid_{COVID_COUNTRY.lower()}.png")
    print(f"\nAnalysis complete for {COVID_COUNTRY}.")

if __name__ == "__main__":
    run_covid_example()
