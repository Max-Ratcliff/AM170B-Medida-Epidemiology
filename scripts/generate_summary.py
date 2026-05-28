import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Add the project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.utils import apply_publication_theme

# Final compiled data
data = [
    {"Country": "Brazil", "Improvement": 4.32},
    {"Country": "Norway", "Improvement": 4.10},
    {"Country": "Sweden", "Improvement": 4.06},
    {"Country": "United States", "Improvement": 3.98},
    {"Country": "Germany", "Improvement": 3.52},
    {"Country": "Italy", "Improvement": 1.87},
    {"Country": "South Korea", "Improvement": 0.86},
    {"Country": "Australia", "Improvement": 0.60},
    {"Country": "India", "Improvement": 0.54},
    {"Country": "Poland", "Improvement": 0.51},
    {"Country": "South Africa", "Improvement": 0.50},
    {"Country": "Japan", "Improvement": 0.43},
]

df = pd.DataFrame(data).sort_values("Improvement", ascending=True)

# Visual Setup
apply_publication_theme()
# Tall figure to prevent crowding
fig, ax = plt.subplots(figsize=(16, 12))

# Professional Palette
success_color = "#2c7bb6" # Strong Blue (Generalization)
fail_color = "#d7191c"    # Strong Red (Specificity)

def get_color(val):
    if val >= 2.0: return success_color
    if val >= 1.0: return "#abd9e9" # Light Blue
    return fail_color

colors = [get_color(x) for x in df["Improvement"]]

# Plot: Professional Full Bars
y_pos = np.arange(len(df))
bars = ax.barh(y_pos, df["Improvement"], color=colors, height=0.7, edgecolor='white', lw=1.5, zorder=3)

# Accuracy labels inside or next to bars
for i, x in enumerate(df["Improvement"]):
    ax.text(x + 0.1, i, f"{x:.2f}x Accuracy", va='center', fontsize=16, fontweight='bold', color="#333333")

# The "Break-even" Baseline (Moved to top/bottom cleanly)
ax.axvline(1.0, color='black', ls='-', lw=3, alpha=0.8, zorder=4)
# Top baseline label
ax.text(1.0, 11.6, "STANDARD SIR MODEL\n(BASELINE ACCURACY)", ha='center', va='bottom', 
         fontsize=12, fontweight='black', color='black', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

# LARGE CATEGORY HEADERS (Top section)
# Universal Side
plt.text(3.5, 13.0, "UNIVERSAL GENERALIZATION\n(FUNDAMENTAL HUMAN DYNAMICS)", 
         ha='center', va='bottom', fontsize=18, fontweight='black', color=success_color,
         bbox=dict(boxstyle="round,pad=0.6", fc="#f7f7f7", ec=success_color, lw=2.5))

# Local Side
plt.text(0.5, 13.0, "LOCAL SPECIFICITY\n(POLICY & DATA ARTIFACTS)", 
         ha='center', va='bottom', fontsize=18, fontweight='black', color=fail_color,
         bbox=dict(boxstyle="round,pad=0.6", fc="#f7f7f7", ec=fail_color, lw=2.5))

# Formatting
ax.set_yticks(y_pos)
ax.set_yticklabels(df["Country"], fontsize=18, fontweight='bold')
ax.set_xlim(0, 6.0) # More padding for labels
ax.set_xlabel("Global Median Improvement Ratio ($SIR \\div MEDIDA$)", fontsize=20, labelpad=30, fontweight='bold')
ax.set_title("DISCOVERING UNIVERSAL EPIDEMIOLOGICAL LAWS", 
             fontsize=24, fontweight="black", pad=120)

# Clean up axes
sns.despine(left=True, bottom=False, trim=True)
ax.grid(axis='x', ls=':', alpha=0.3)

plt.tight_layout()
output_dir = "outputs/summary"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "master_ablation_summary.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"[*] Master Research Summary finalized with full bars: {output_dir}/master_ablation_summary.png")
