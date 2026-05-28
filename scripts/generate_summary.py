import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.utils import apply_publication_theme

SWEEP_CSV = os.path.join(project_root, "outputs", "covid", "italy", "sweep_results.csv")


def load_sweep_results():
    if not os.path.exists(SWEEP_CSV):
        raise FileNotFoundError(
            f"Sweep results not found at {SWEEP_CSV}.\n"
            "Run: python scripts/covid_analysis.py --train-country Italy --sweep"
        )
    return pd.read_csv(SWEEP_CSV)


def plot_ablation_summary(results_df, output_path):
    """Bar chart showing top-10 and bottom-10 countries by improvement ratio."""
    apply_publication_theme()

    top10 = results_df.nlargest(10, "improvement")[["country", "improvement", "lockdown"]]
    bottom10 = results_df.nsmallest(10, "improvement")[["country", "improvement", "lockdown"]]
    display = pd.concat([top10, bottom10]).drop_duplicates("country").sort_values("improvement")

    success_color, fail_color = "#2c7bb6", "#d7191c"

    def bar_color(row):
        if row["improvement"] >= 1.5:
            return success_color
        if row["improvement"] >= 1.0:
            return "#abd9e9"
        return fail_color

    colors = [bar_color(r) for _, r in display.iterrows()]

    fig, ax = plt.subplots(figsize=(16, 12))
    y_pos = np.arange(len(display))
    ax.barh(
        y_pos, display["improvement"], color=colors, height=0.7, edgecolor="white", lw=1.5, zorder=3
    )

    for i, (_, row) in enumerate(display.iterrows()):
        ax.text(
            row["improvement"] + 0.05,
            i,
            f"{row['improvement']:.2f}×",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="#333333",
        )

    ax.axvline(1.0, color="black", ls="-", lw=2.5, alpha=0.8, zorder=4)
    ax.text(
        1.02,
        len(display) - 0.5,
        "Baseline\n(Naive SIR)",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="black",
    )

    n_countries = len(results_df)
    n_improve = int((results_df["improvement"] >= 1.0).sum())
    median_imp = results_df["improvement"].median()
    ax.set_title(
        f"MEDIDA GLOBAL TRANSFER: {n_improve}/{n_countries} COUNTRIES IMPROVE\n"
        f"Median improvement: {median_imp:.2f}×  |  Trained on Italy",
        fontsize=18,
        fontweight="black",
        pad=15,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(display["country"], fontsize=14, fontweight="bold")
    ax.set_xlabel(
        "ACCURACY IMPROVEMENT RATIO  (Naive SIR RMSE ÷ MEDIDA RMSE)",
        fontsize=14,
        labelpad=12,
        fontweight="bold",
    )
    ax.set_xlim(0, display["improvement"].max() * 1.2)
    sns.despine(left=True)
    ax.grid(axis="x", ls=":", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[summary] Saved {output_path}")


def main():
    results_df = load_sweep_results()
    output_dir = os.path.join(project_root, "outputs", "summary")
    os.makedirs(output_dir, exist_ok=True)
    plot_ablation_summary(results_df, os.path.join(output_dir, "master_ablation_summary.png"))


if __name__ == "__main__":
    main()
