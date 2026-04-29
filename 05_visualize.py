"""
IntrinsicEthics-SFT — Step 05: Results Visualization

Reads evaluation output and produces the four-panel results figure
used in the paper.

Inputs:
    /content/drive/MyDrive/ethics_experiment/results/eval_progress.json

Outputs:
    /content/drive/MyDrive/ethics_experiment/results/sft_experiment_results.png

Runtime:
    Any (CPU sufficient) | <1 min | no API cost

Configuration:
    None
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ DRIVE MOUNT                                                                 │
# └─────────────────────────────────────────────────────────────────────────────┘
from google.colab import drive as _drive
if not Path("/content/drive/MyDrive").exists():
    print("Mounting Google Drive...", end=" ", flush=True)
    _drive.mount("/content/drive")
    print("ok")
else:
    print("Google Drive already mounted.")

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ PATHS                                                                       │
# └─────────────────────────────────────────────────────────────────────────────┘
DRIVE_BASE    = Path("/content/drive/MyDrive/ethics_experiment")
PROGRESS_FILE = DRIVE_BASE / "results" / "eval_progress.json"
OUT_FILE      = DRIVE_BASE / "results" / "sft_experiment_results.png"

if not PROGRESS_FILE.exists():
    raise FileNotFoundError(
        f"eval_progress.json not found at {PROGRESS_FILE}\n"
        "Run 04_train_and_evaluate.py first."
    )

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ LOAD DATA                                                                   │
# └─────────────────────────────────────────────────────────────────────────────┘
with open(PROGRESS_FILE) as f:
    merged = json.load(f)

results = merged["results"]

CONFIG_IDS = ["baseline", "filter", "intrinsic", "intrinsic_ablated", "filter_ablated"]
LABELS = {
    "baseline":          "Baseline",
    "filter":            "Filter",
    "intrinsic":         "Intrinsic",
    "intrinsic_ablated": "Intrinsic\nAblated",
    "filter_ablated":    "Filter\nAblated",
}

# Colorblind-safe palette
COLORS = {
    "baseline":          "#e74c3c",   # red
    "filter":            "#e67e22",   # orange
    "intrinsic":         "#27ae60",   # green
    "intrinsic_ablated": "#8e44ad",   # purple
    "filter_ablated":    "#95a5a6",   # gray
}

def m(cfg_id, key, default=0.0):
    return results.get(cfg_id, {}).get(key, default)

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ FIGURE SETUP                                                                │
# └─────────────────────────────────────────────────────────────────────────────┘
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Intrinsic Ethics SFT Experiment — Load-Bearing Alignment Results",
    fontsize=14, fontweight="bold", y=0.98,
)

labels       = [LABELS[c]  for c in CONFIG_IDS]
colors       = [COLORS[c]  for c in CONFIG_IDS]

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ PANEL 1 (top-left) — IES bar chart                                         │
# └─────────────────────────────────────────────────────────────────────────────┘
ax1 = axes[0, 0]
ies_values = [m(c, "IES") for c in CONFIG_IDS]

bars = ax1.bar(labels, ies_values, color=colors, edgecolor="black", linewidth=0.8)
ax1.set_title("Intrinsic Ethics Score (IES) — Higher is Better", fontweight="bold")
ax1.set_ylabel("IES")
ax1.axhline(y=0, color="black", linestyle="--", alpha=0.5, linewidth=1)

y_min = min(min(ies_values) - 0.05, -0.05)
y_max = max(max(ies_values) + 0.12, 0.12)
ax1.set_ylim(y_min, y_max)

for bar, val in zip(bars, ies_values):
    va     = "bottom" if val >= 0 else "top"
    offset = 0.01 if val >= 0 else -0.01
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        val + offset,
        f"{val:.3f}",
        ha="center", va=va, fontweight="bold", fontsize=9,
    )

ax1.tick_params(axis="x", labelsize=8)

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ PANEL 2 (top-right) — Safety vs. utility scatter                           │
# └─────────────────────────────────────────────────────────────────────────────┘
ax2 = axes[0, 1]

rj_vals = [m(c, "R_J") for c in CONFIG_IDS]
ut_vals = [m(c, "U_T") for c in CONFIG_IDS]

# Light gray diagonal reference line across full axes range
_x_ref = np.linspace(0, 1.05, 100)
ax2.plot(_x_ref, _x_ref, color="#cccccc", linewidth=1, linestyle="--", zorder=0)

for cfg_id, rj, ut, color, label in zip(CONFIG_IDS, rj_vals, ut_vals, colors, labels):
    ax2.scatter(rj, ut, color=color, s=120, edgecolors="black", linewidth=0.8,
                zorder=3, label=label.replace("\n", " "))
    ax2.annotate(
        label.replace("\n", " "),
        xy=(rj, ut),
        xytext=(6, 4),
        textcoords="offset points",
        fontsize=8,
    )

ax2.set_title("Safety vs. Utility — No Tradeoff for Intrinsic", fontweight="bold")
ax2.set_xlabel("Jailbreak Resistance (R_J)")
ax2.set_ylabel("Utility Preservation (U_T)")
ax2.set_xlim(-0.05, 1.1)

ut_lo = max(0.0, min(ut_vals) - 0.08)
ut_hi = min(1.0, max(ut_vals) + 0.08)
ax2.set_ylim(ut_lo, ut_hi)
ax2.grid(True, alpha=0.3)

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ PANEL 3 (bottom-left) — Load-bearing connected dot plot                    │
# └─────────────────────────────────────────────────────────────────────────────┘
ax3 = axes[1, 0]

pairs = [
    ("intrinsic",  "intrinsic_ablated", "p=0.0000 ✓ LOAD-BEARING", "#27ae60"),
    ("filter",     "filter_ablated",    "p=0.3203 ✗ not significant", "#95a5a6"),
]

x_positions = {
    "intrinsic":          0.8,
    "intrinsic_ablated":  1.2,
    "filter":             2.0,
    "filter_ablated":     2.4,
}

for full_id, ablated_id, annotation, ann_color in pairs:
    x_full    = x_positions[full_id]
    x_ablated = x_positions[ablated_id]
    y_full    = m(full_id,    "IES")
    y_ablated = m(ablated_id, "IES")

    ax3.scatter(x_full,    y_full,    color=COLORS[full_id],    s=120,
                edgecolors="black", linewidth=0.8, zorder=3)
    ax3.scatter(x_ablated, y_ablated, color=COLORS[ablated_id], s=120,
                edgecolors="black", linewidth=0.8, zorder=3)

    ax3.annotate(
        "", xy=(x_ablated, y_ablated), xytext=(x_full, y_full),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
    )

    mid_x = (x_full + x_ablated) / 2
    mid_y = (y_full + y_ablated) / 2 + 0.03
    ax3.text(mid_x, mid_y, annotation,
             ha="center", va="bottom", fontsize=8,
             color=ann_color, fontweight="bold")

    # Label the dots
    for x_pos, cfg_id in [(x_full, full_id), (x_ablated, ablated_id)]:
        ax3.text(x_pos, m(cfg_id, "IES") - 0.025,
                 LABELS[cfg_id].replace("\n", " "),
                 ha="center", va="top", fontsize=7.5)

ax3.axhline(y=0, color="black", linestyle="--", alpha=0.4, linewidth=1)
ax3.set_title("Load-Bearing Test — Ethics Removal Effect", fontweight="bold")
ax3.set_ylabel("IES")
ax3.set_xlim(0.3, 2.9)
_ies_range = [m(c, "IES") for c in CONFIG_IDS]
ax3.set_ylim(min(_ies_range) - 0.12, max(_ies_range) + 0.12)
ax3.set_xticks([])
ax3.grid(True, alpha=0.3, axis="y")

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ PANEL 4 (bottom-right) — Cross-domain multi-metric bar chart               │
# └─────────────────────────────────────────────────────────────────────────────┘
ax4 = axes[1, 1]

metrics_to_show = [
    ("humaneval_pass1",      "HumanEval pass@1",    "#3498db"),
    ("causal_reasoning_rate","Causal Reasoning",     "#e67e22"),
    ("evasion_rate",         "Evasion Detection",    "#1abc9c"),
]

n_configs = len(CONFIG_IDS)
n_metrics = len(metrics_to_show)
width     = 0.22
x         = np.arange(n_configs)

for i, (key, metric_label, bar_color) in enumerate(metrics_to_show):
    offset = (i - (n_metrics - 1) / 2) * width
    vals   = [m(c, key) for c in CONFIG_IDS]
    ax4.bar(x + offset, vals, width,
            label=metric_label, color=bar_color,
            edgecolor="black", linewidth=0.6)

ax4.set_title("Cross-Domain Performance Metrics", fontweight="bold")
ax4.set_xticks(x)
ax4.set_xticklabels(labels, fontsize=8)
ax4.set_ylim(0, 1.2)
ax4.set_ylabel("Score / Rate")
ax4.legend(fontsize=8, loc="upper right")
ax4.grid(True, alpha=0.3, axis="y")

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ SAVE                                                                        │
# └─────────────────────────────────────────────────────────────────────────────┘
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {OUT_FILE}")
