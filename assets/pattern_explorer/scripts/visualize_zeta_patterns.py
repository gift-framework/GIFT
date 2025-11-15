#!/usr/bin/env python3
"""
Visualization of zeta ratio patterns in GIFT observables.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Get data directory relative to this script
# scripts/ -> pattern_explorer/ -> data/
data_dir = Path(__file__).resolve().parent.parent / 'data'
viz_dir = Path(__file__).resolve().parent.parent / 'visualizations'

# Create visualizations directory if it doesn't exist
viz_dir.mkdir(parents=True, exist_ok=True)

# Read results
csv_path = data_dir / 'zeta_ratio_matches.csv'
if not csv_path.exists():
    print(f"Error: {csv_path} not found. Please run zeta_ratio_discovery.py first.")
    exit(1)

df = pd.read_csv(csv_path)

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 12))

# 1. Deviation by Observable (sorted)
ax1 = plt.subplot(3, 2, 1)
df_plot = df.sort_values('deviation_pct')
colors = ['darkgreen' if x < 0.1 else 'green' if x < 0.5 else 'orange' if x < 1.0 else 'red'
          for x in df_plot['deviation_pct']]
ax1.barh(range(len(df_plot)), df_plot['deviation_pct'], color=colors)
ax1.set_yticks(range(len(df_plot)))
ax1.set_yticklabels(df_plot['observable'], fontsize=8)
ax1.set_xlabel('Deviation (%)', fontsize=10)
ax1.set_title('Zeta Ratio Match Quality by Observable', fontsize=12, fontweight='bold')
ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
ax1.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
ax1.grid(axis='x', alpha=0.3)

# 2. Category Statistics
ax2 = plt.subplot(3, 2, 2)
category_stats = df.groupby('category').agg({
    'deviation_pct': 'mean',
    'observable': 'count'
}).sort_values('deviation_pct')
colors_cat = ['darkgreen' if x < 0.3 else 'green' if x < 0.6 else 'orange'
              for x in category_stats['deviation_pct']]
bars = ax2.bar(range(len(category_stats)), category_stats['deviation_pct'], color=colors_cat)
ax2.set_xticks(range(len(category_stats)))
ax2.set_xticklabels(category_stats.index, rotation=45, ha='right', fontsize=9)
ax2.set_ylabel('Mean Deviation (%)', fontsize=10)
ax2.set_title('Mean Deviation by Category', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Add count labels on bars
for i, (idx, row) in enumerate(category_stats.iterrows()):
    ax2.text(i, row['deviation_pct'] + 0.05, f"n={int(row['observable'])}",
             ha='center', va='bottom', fontsize=8)

# 3. Zeta Ratio Usage Frequency
ax3 = plt.subplot(3, 2, 3)
ratio_counts = df['best_ratio'].value_counts()
ax3.barh(range(len(ratio_counts)), ratio_counts.values, color='steelblue')
ax3.set_yticks(range(len(ratio_counts)))
ax3.set_yticklabels(ratio_counts.index, fontsize=9)
ax3.set_xlabel('Number of Observables', fontsize=10)
ax3.set_title('Most Common Zeta Ratios', fontsize=12, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# 4. Scaling Factor Distribution
ax4 = plt.subplot(3, 2, 4)
scaling_counts = df['scaling_factor'].value_counts().head(10)
ax4.bar(range(len(scaling_counts)), scaling_counts.values, color='coral')
ax4.set_xticks(range(len(scaling_counts)))
ax4.set_xticklabels(scaling_counts.index, rotation=45, ha='right', fontsize=8)
ax4.set_ylabel('Number of Observables', fontsize=10)
ax4.set_title('Most Common Scaling Factors', fontsize=12, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# 5. Predicted vs Observed (log scale for large range)
ax5 = plt.subplot(3, 2, 5)
# Separate into different ranges for better visualization
df_small = df[df['experimental_value'] < 10]
df_large = df[df['experimental_value'] >= 10]

if len(df_small) > 0:
    ax5.scatter(df_small['experimental_value'], df_small['predicted_value'],
               s=100, alpha=0.6, c='blue', label='< 10')
if len(df_large) > 0:
    ax5.scatter(df_large['experimental_value'], df_large['predicted_value'],
               s=100, alpha=0.6, c='red', label='≥ 10')

# Perfect correlation line
all_vals = pd.concat([df['experimental_value'], df['predicted_value']])
min_val, max_val = all_vals.min() * 0.9, all_vals.max() * 1.1
ax5.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1, label='Perfect match')

ax5.set_xlabel('Experimental Value', fontsize=10)
ax5.set_ylabel('Predicted Value (Zeta Ratio)', fontsize=10)
ax5.set_title('Predicted vs Observed Values', fontsize=12, fontweight='bold')
ax5.legend(fontsize=8)
ax5.grid(alpha=0.3)
ax5.set_xscale('log')
ax5.set_yscale('log')

# 6. Confidence Score Distribution
ax6 = plt.subplot(3, 2, 6)
confidence_bins = [0, 60, 75, 85, 95, 100]
confidence_labels = ['Moderate\n(60)', 'Strong\n(75)', 'High\n(85)', 'Very High\n(95)', 'Exceptional\n(100)']
conf_hist, _ = np.histogram(df['confidence_score'], bins=confidence_bins)
colors_conf = ['orange', 'gold', 'lightgreen', 'green', 'darkgreen']
ax6.bar(range(len(conf_hist)), conf_hist, color=colors_conf)
ax6.set_xticks(range(len(conf_hist)))
ax6.set_xticklabels(confidence_labels, fontsize=8)
ax6.set_ylabel('Number of Observables', fontsize=10)
ax6.set_title('Confidence Score Distribution', fontsize=12, fontweight='bold')
ax6.grid(axis='y', alpha=0.3)

plt.tight_layout()
output_path1 = viz_dir / 'zeta_patterns_visualization.png'
plt.savefig(output_path1, dpi=300, bbox_inches='tight')
print(f"Visualization saved to: {output_path1}")

# Create a second figure showing zeta ratio network
fig2, ax = plt.subplots(figsize=(14, 10))

# Extract numerator and denominator from ratio names
import re
zeta_indices = set()
edges = []
for _, row in df.iterrows():
    match = re.search(r'ζ\((\d+)\)/ζ\((\d+)\)', row['best_ratio'])
    if match:
        num, den = int(match.group(1)), int(match.group(2))
        zeta_indices.add(num)
        zeta_indices.add(den)
        edges.append((den, num, row['observable'], row['deviation_pct']))

# Position zeta values on a circle
zeta_indices = sorted(zeta_indices)
n = len(zeta_indices)
angles = np.linspace(0, 2*np.pi, n, endpoint=False)
positions = {idx: (np.cos(angle), np.sin(angle)) for idx, angle in zip(zeta_indices, angles)}

# Draw nodes
for idx in zeta_indices:
    x, y = positions[idx]
    circle = plt.Circle((x, y), 0.15, color='lightblue', ec='darkblue', linewidth=2, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, f'ζ({idx})', ha='center', va='center', fontsize=11, fontweight='bold', zorder=4)

# Draw edges
for den, num, obs, dev in edges:
    x1, y1 = positions[den]
    x2, y2 = positions[num]
    # Color by deviation
    if dev < 0.1:
        color = 'darkgreen'
        width = 2.5
        alpha = 0.8
    elif dev < 0.5:
        color = 'green'
        width = 2.0
        alpha = 0.6
    elif dev < 1.0:
        color = 'orange'
        width = 1.5
        alpha = 0.5
    else:
        color = 'red'
        width = 1.0
        alpha = 0.4

    ax.arrow(x1*1.05, y1*1.05, (x2-x1)*0.85, (y2-y1)*0.85,
             head_width=0.05, head_length=0.05, fc=color, ec=color,
             alpha=alpha, linewidth=width, zorder=2)

    # Add observable label at midpoint
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    ax.text(mid_x*1.3, mid_y*1.3, obs, fontsize=7, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7), zorder=5)

ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.8, 1.8)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Zeta Ratio Network: Observables as Connections', fontsize=14, fontweight='bold', pad=20)

# Legend
legend_elements = [
    mpatches.Patch(color='darkgreen', label='< 0.1% deviation'),
    mpatches.Patch(color='green', label='0.1-0.5% deviation'),
    mpatches.Patch(color='orange', label='0.5-1.0% deviation'),
    mpatches.Patch(color='red', label='> 1.0% deviation')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.tight_layout()
output_path2 = viz_dir / 'zeta_network_graph.png'
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"Network graph saved to: {output_path2}")

print("\nVisualization complete!")
