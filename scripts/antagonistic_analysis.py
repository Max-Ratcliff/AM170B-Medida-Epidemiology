import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def analyze_bias(csv_path):
    """Analyze if improvement factors correlate with population size or case counts (Antagonistic Audit)."""
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    # Load raw data to get population and total cases
    raw_data_path = os.path.join(project_root, "data/owid-covid-data.csv")
    if not os.path.exists(raw_data_path):
        print(f"Error: {raw_data_path} not found.")
        return

    raw_df = pd.read_csv(raw_data_path)
    # Aggregate total cases and population per country
    stats = (
        raw_df.groupby("location")
        .agg({"population": "first", "new_cases_smoothed": "sum"})
        .reset_index()
    )

    # Merge with sweep results
    merged = df.merge(stats, left_on="country", right_on="location")

    # Filter out zero improvement (failed runs or missing data)
    merged = merged[merged["improvement"] > 0]

    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.scatter(
        merged["population"],
        merged["improvement"],
        alpha=0.6,
        color="#2c7fb8",
        edgecolor="white",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Population (Log)", fontweight="bold")
    plt.ylabel("Improvement Factor (Log)", fontweight="bold")
    plt.title("IMPROVEMENT VS POPULATION", fontweight="black")
    plt.grid(True, which="both", ls="-", alpha=0.1)

    plt.subplot(1, 2, 2)
    plt.scatter(
        merged["new_cases_smoothed"],
        merged["improvement"],
        alpha=0.6,
        color="#e31a1c",
        edgecolor="white",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Total New Cases (Log)", fontweight="bold")
    plt.ylabel("Improvement Factor (Log)", fontweight="bold")
    plt.title("IMPROVEMENT VS TOTAL CASES", fontweight="black")
    plt.grid(True, which="both", ls="-", alpha=0.1)

    plt.tight_layout()
    output_png = os.path.join(
        project_root, "outputs/antagonistic_bias_check.png"
    )
    plt.savefig(output_png, dpi=300)
    print(f"[*] Bias check plot saved to {output_png}")

    # Correlation analysis
    corr_pop = merged["improvement"].corr(
        merged["population"], method="spearman"
    )
    corr_cases = merged["improvement"].corr(
        merged["new_cases_smoothed"], method="spearman"
    )

    print("-" * 40)
    print("ANTAGONISTIC AUDIT: SPEARMAN CORRELATION")
    print(f"Improvement vs Population:  {corr_pop:.3f}")
    print(f"Improvement vs Total Cases: {corr_cases:.3f}")
    print("-" * 40)

    # Check if high improvement is driven by low-data countries
    high_imp = merged.nlargest(10, "improvement")
    print("\nTOP 10 IMPROVED COUNTRIES (STATS):")
    print(
        high_imp[
            ["country", "improvement", "population", "new_cases_smoothed"]
        ]
    )


if __name__ == "__main__":
    # Target the Italy sweep as the primary benchmark
    analyze_bias("outputs/covid/italy/sweep_results.csv")
