import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.utils import apply_publication_theme  # noqa: E402

SWEEP_CSV = os.path.join(
    project_root, "outputs", "covid", "italy", "sweep_results.csv"
)


def load_sweep_results():
    if not os.path.exists(SWEEP_CSV):
        raise FileNotFoundError(
            f"Sweep results not found at {SWEEP_CSV}.\n"
            "Run: python scripts/covid_analysis.py "
            "--train-country Italy --sweep"
        )
    df = pd.read_csv(SWEEP_CSV)
    # Drop territories with too little data for a defined improvement ratio.
    return df[df["improvement"] > 0.05].copy()


def plot_ablation_summary(results_df, output_path):
    """Top-10 / bottom-10 bar chart colored by lockdown status."""
    apply_publication_theme()

    top10 = results_df.nlargest(10, "improvement")
    bottom10 = results_df.nsmallest(10, "improvement")
    display = (
        pd.concat([top10, bottom10])
        .drop_duplicates("country")
        .sort_values("improvement")
    )

    lockdown_color = "#2c7bb6"
    no_lockdown_color = "#d7191c"
    colors = [
        lockdown_color if r["lockdown"] else no_lockdown_color
        for _, r in display.iterrows()
    ]

    n_countries = len(results_df)
    n_improve = int((results_df["improvement"] >= 1.0).sum())
    med_lock = results_df[results_df["lockdown"]]["improvement"].median()
    med_nolock = results_df[~results_df["lockdown"]]["improvement"].median()

    fig, ax = plt.subplots(figsize=(16, 12))
    y_pos = np.arange(len(display))
    ax.barh(
        y_pos,
        display["improvement"],
        color=colors,
        height=0.7,
        edgecolor="white",
        lw=1.5,
        zorder=3,
    )

    for i, (_, row) in enumerate(display.iterrows()):
        ax.text(
            row["improvement"] + 0.1,
            i,
            f"{row['improvement']:.1f}×",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="#333333",
        )

    ax.axvline(1.0, color="black", ls="-", lw=2.5, alpha=0.8, zorder=4)
    ax.text(
        1.04,
        len(display) - 0.5,
        "Naive SIR\nbaseline",
        va="top",
        fontsize=10,
        color="black",
    )

    # Legend
    leg = [
        mpatches.Patch(
            color=lockdown_color,
            label=f"National lockdown (median {med_lock:.1f}×)",
        ),
        mpatches.Patch(
            color=no_lockdown_color,
            label=f"No lockdown (median {med_nolock:.1f}×)",
        ),
    ]
    ax.legend(handles=leg, loc="lower right", fontsize=13)

    ax.set_title(
        "MEDIDA GLOBAL TRANSFER: "
        f"{n_improve}/{n_countries} COUNTRIES IMPROVE  "
        "(Trained on Italy)\n"
        f"Lockdown countries: {med_lock:.1f}x median  |  "
        f"Non-lockdown: {med_nolock:.1f}x median",
        fontsize=16,
        fontweight="black",
        pad=12,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display["country"], fontsize=13, fontweight="bold")
    ax.set_xlabel(
        "ACCURACY IMPROVEMENT RATIO  (Naive SIR RMSE ÷ MEDIDA RMSE)",
        fontsize=13,
        labelpad=10,
        fontweight="bold",
    )
    ax.set_xlim(0, display["improvement"].max() * 1.15)
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
    plot_ablation_summary(
        results_df, os.path.join(output_dir, "master_ablation_summary.png")
    )


if __name__ == "__main__":
    main()
