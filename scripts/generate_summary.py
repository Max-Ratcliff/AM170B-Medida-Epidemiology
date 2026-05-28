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
# Increase height to 12 to prevent vertical crowding
fig, ax = plt.subplots(figsize=(15, 12))

# Professional Palette
success_color = "#2c7bb6" # Strong Blue (Generalization)
fail_color = "#d7191c"    # Strong Red (Specificity)
neutral_color = "#999999" # Gray

def get_color(val):
    if val >= 2.0: return success_color
    if val >= 1.0: return "#abd9e9" # Light Blue
    return fail_color

colors = [get_color(x) for x in df["Improvement"]]

# Plot: Modern Clean Lollipop
y_pos = np.arange(len(df))
# Faint horizontal tracks
ax.hlines(y_pos, 0, 5.5, color='gray', alpha=0.05, lw=1, zorder=0)
# Data lines
ax.hlines(y_pos, 1, df["Improvement"], color=colors, alpha=0.6, lw=3, zorder=1)
# Large Markers
ax.scatter(df["Improvement"], y_pos, color=colors, s=400, edgecolors='white', lw=2.5, zorder=3)

# Value labels (Clearly to the right)
for i, x in enumerate(df["Improvement"]):
    ax.text(x + 0.15, i, f"{x:.2f}x", va='center', fontsize=16, fontweight='bold', color="#333333")

# The "Break-even" Wall
ax.axvline(1.0, color='#333333', ls='-', lw=3, alpha=0.9, zorder=2)
plt.text(1.0, -1.2, "NAIVE SIR MODEL\n(BREAK-EVEN)", ha='center', va='top', 
         fontsize=13, fontweight='black', color='#333333')

# LARGE CATEGORY HEADERS (Moved far from data)
# Universal Side
plt.text(3.5, 12.2, "UNIVERSAL GENERALIZATION\n(FUNDAMENTAL DYNAMICS)", 
         ha='center', va='bottom', fontsize=16, fontweight='black', color=success_color,
         bbox=dict(boxstyle="round,pad=0.5", fc="#f7f7f7", ec=success_color, lw=2))

# Local Side
plt.text(0.5, 12.2, "LOCAL SPECIFICITY\n(POLICY ARTIFACTS)", 
         ha='center', va='bottom', fontsize=16, fontweight='black', color=fail_color,
         bbox=dict(boxstyle="round,pad=0.5", fc="#f7f7f7", ec=fail_color, lw=2))

# Formatting
ax.set_yticks(y_pos)
ax.set_yticklabels(df["Country"], fontsize=18, fontweight='bold')
ax.set_xlim(0, 5.5)
ax.set_xlabel("Median Global Improvement Ratio ($SIR \\div MEDIDA$)", fontsize=20, labelpad=25, fontweight='bold')
ax.set_title("GLOBAL GENERALIZABILITY OF DISCOVERED MODEL CORRECTIONS", 
             fontsize=22, fontweight="black", pad=100)

sns.despine(left=True, bottom=False, trim=True)

output_dir = "outputs/summary"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "master_ablation_summary.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"[*] Master Research Summary finalized: {output_dir}/master_ablation_summary.png")
