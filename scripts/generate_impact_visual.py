import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.utils import apply_publication_theme

def generate_efficacy_distribution(df, output_path):
    """Generate a histogram + CDF of global improvement factors."""
    apply_publication_theme()
    fig, ax = plt.subplots(figsize=(10, 7))

    # Calculate metrics
    improvements = df['improvement'].clip(0, 50)  # Clip for better histogram viz
    median_imp = improvements.median()
    
    # Histogram
    sns.histplot(improvements, bins=30, kde=False, color='#33a02c', alpha=0.6, ax=ax, label='Country Count')
    
    # Cumulative Distribution (CDF) on twin axis
    ax2 = ax.twinx()
    sns.ecdfplot(improvements, color='#e31a1c', lw=3, ax=ax2, label='Cumulative %')
    ax2.set_ylabel('Cumulative Percentage of Countries', color='#e31a1c', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#e31a1c')
    ax2.grid(False)

    ax.axvline(median_imp, color='black', ls='--', lw=2, label=f'Median Gain ({median_imp:.1f}x)')
    
    ax.set_xlabel('Accuracy Improvement Factor (RMSE Ratio)', fontweight='bold')
    ax.set_ylabel('Number of Countries', fontweight='bold')
    ax.set_title('GLOBAL ACCURACY GAIN: DISTRIBUTION', fontsize=18, fontweight='black')
    
    # Combined legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='center right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved efficacy distribution to {output_path}")
    plt.close()

def generate_case_study_slope(df, output_path):
    """Generate a high-impact 'Slope Plot' for key target countries."""
    targets = ['Italy', 'Sweden', 'India']
    subset = df[df['country'].isin(targets)].copy()
    if subset.empty:
        return

    apply_publication_theme()
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = {'Italy': '#1f78b4', 'Sweden': '#e31a1c', 'India': '#ff7f00'}

    for _, row in subset.iterrows():
        country = row['country']
        # Normalize to Naive=100% for easier relative comparison
        y = [1.0, row['medida_rmse'] / row['naive_rmse']]
        x = [0, 1]
        
        ax.plot(x, y, marker='o', markersize=12, lw=4, color=colors.get(country, 'gray'), label=country)
        ax.text(-0.05, y[0], f"Naive SIR", ha='right', va='center', fontweight='bold')
        ax.text(1.05, y[1], f"{row['improvement']:.1f}x Better", ha='left', va='center', fontweight='bold', color=colors.get(country, 'gray'))

    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['BASELINE\n(Naive SIR)', 'MEDIDA\n(Universal Correction)'], fontweight='bold', fontsize=14)
    ax.set_ylabel('Relative Error (Normalized)', fontweight='bold')
    ax.set_title('CASE STUDY: ERROR COLLAPSE', fontsize=20, fontweight='black', pad=20)
    
    ax.set_ylim(0, 1.2)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved case-study slope plot to {output_path}")
    plt.close()

if __name__ == "__main__":
    csv_file = "outputs/global_transfer_slide/global_transfer_summary.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Run scripts/global_covid_analysis.py --sweep first.")
    else:
        df_all = pd.read_csv(csv_file)
        generate_efficacy_distribution(df_all, "outputs/global_transfer_slide/global_efficacy_dist.png")
        generate_case_study_slope(df_all, "outputs/global_transfer_slide/case_study_slope.png")
