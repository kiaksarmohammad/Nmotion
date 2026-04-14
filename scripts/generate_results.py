"""Generate publication-quality results from feature discovery output."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# ── paths ──────────────────────────────────────────────────────────────
DISCOVERY_DIR = Path("output/feature_discovery_full")
RESULTS_DIR = Path("output/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ranking = pd.read_csv(DISCOVERY_DIR / "ranking.csv")
features = pd.read_csv(DISCOVERY_DIR / "all_features.csv")

GROUP_COLORS = {
    "normal": "#2ecc71",
    "seizure": "#e74c3c",
    "hypotonic": "#3498db",
}
GROUP_ORDER = ["normal", "seizure", "hypotonic"]

# ── Figure 1: Feature Ranking Waterfall ────────────────────────────────
# Top-30 features, horizontal bars colored by -log10(p)
top_n = 30
top = ranking.head(top_n).copy()
top["neg_log_p"] = -np.log10(top["kw_pvalue"].clip(lower=1e-300))
# best pairwise effect size (max across 3 comparisons)
top["best_effect"] = top[
    ["effect_hypotonic_vs_normal", "effect_hypotonic_vs_seizure", "effect_normal_vs_seizure"]
].max(axis=1)

fig, ax = plt.subplots(figsize=(12, 10))
y_pos = np.arange(top_n)[::-1]

# color by -log10(p)
norm = plt.Normalize(vmin=top["neg_log_p"].min(), vmax=top["neg_log_p"].max())
cmap = LinearSegmentedColormap.from_list("pval", ["#fee08b", "#d73027"])
colors = cmap(norm(top["neg_log_p"].values))

bars = ax.barh(y_pos, top["best_effect"], color=colors, edgecolor="white", linewidth=0.5, height=0.75)

# p-value annotation on bar
for i, (_, row) in enumerate(top.iterrows()):
    p = row["kw_pvalue"]
    if p < 1e-10:
        label = f"p = {p:.1e}"
    else:
        label = f"p = {p:.2e}"
    ax.text(row["best_effect"] + 0.008, y_pos[i], label, va="center", fontsize=7.5, color="#555")

ax.set_yticks(y_pos)
ax.set_yticklabels(top["feature"], fontsize=9)
ax.set_xlabel("Best Pairwise Effect Size (rank-biserial)", fontsize=11)
ax.set_title("Top 30 Features Ranked by Kruskal-Wallis Significance", fontsize=14, fontweight="bold")
ax.set_xlim(0, 1.15)
ax.axvline(0.5, color="#999", linestyle="--", linewidth=0.8, alpha=0.6)
ax.text(0.51, top_n - 0.5, "large effect\nthreshold", fontsize=8, color="#777", va="top")

# colorbar for p-value
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.6)
cbar.set_label("-log₁₀(p-value)", fontsize=10)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "feature_ranking_waterfall.png", dpi=300, bbox_inches="tight")
fig.savefig(RESULTS_DIR / "feature_ranking_waterfall.pdf", bbox_inches="tight")
plt.close(fig)
print("✓ Figure 1: feature_ranking_waterfall.png")


# ── Figure 2: Group Separation Violin Plots (top 8) ───────────────────
top_features = ranking["feature"].head(8).tolist()

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.ravel()

for i, feat in enumerate(top_features):
    ax = axes[i]
    data_by_group = []
    positions = []
    colors_list = []

    for j, grp in enumerate(GROUP_ORDER):
        vals = features.loc[features["group"] == grp, feat].dropna()
        if len(vals) > 0:
            data_by_group.append(vals.values)
            positions.append(j)
            colors_list.append(GROUP_COLORS[grp])

    parts = ax.violinplot(data_by_group, positions=positions, showmeans=False, showmedians=False, showextrema=False)
    for pc, c in zip(parts["bodies"], colors_list):
        pc.set_facecolor(c)
        pc.set_alpha(0.7)
        pc.set_edgecolor("white")
        pc.set_linewidth(0.8)

    # overlay box plots
    bp = ax.boxplot(data_by_group, positions=positions, widths=0.15, patch_artist=True,
                    showfliers=False, zorder=3)
    for patch, c in zip(bp["boxes"], colors_list):
        patch.set_facecolor(c)
        patch.set_alpha(0.9)
    for element in ["whiskers", "caps", "medians"]:
        plt.setp(bp[element], color="#333", linewidth=1.2)

    # strip plot (jittered dots)
    for j, (grp, vals) in enumerate(zip(GROUP_ORDER, data_by_group)):
        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full_like(vals, j) + jitter, vals,
                   c=GROUP_COLORS[grp], s=12, alpha=0.5, edgecolors="white", linewidths=0.3, zorder=4)

    # p-value annotation
    row = ranking[ranking["feature"] == feat].iloc[0]
    p = row["kw_pvalue"]
    ax.set_title(feat.replace("_", " "), fontsize=10, fontweight="bold")
    ax.text(0.98, 0.97, f"p = {p:.1e}", transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="#c0392b", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ddd", alpha=0.9))

    ax.set_xticks(range(len(GROUP_ORDER)))
    ax.set_xticklabels([g.capitalize() for g in GROUP_ORDER], fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("Group Separation — Top 8 Discriminating Features", fontsize=15, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "group_separation_violins.png", dpi=300, bbox_inches="tight")
fig.savefig(RESULTS_DIR / "group_separation_violins.pdf", bbox_inches="tight")
plt.close(fig)
print("✓ Figure 2: group_separation_violins.png")


# ── Figure 3: Pairwise AUROC Heatmap ──────────────────────────────────
# Show AUROC for each feature × pairwise comparison (top 25)
top25 = ranking.head(25).copy()
auroc_cols = ["auroc_hypotonic_vs_normal", "auroc_hypotonic_vs_seizure", "auroc_normal_vs_seizure"]
auroc_labels = ["Hypo vs Normal", "Hypo vs Seizure", "Normal vs Seizure"]

auroc_matrix = top25[auroc_cols].values

fig, ax = plt.subplots(figsize=(8, 12))
im = ax.imshow(auroc_matrix, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0)

ax.set_xticks(range(3))
ax.set_xticklabels(auroc_labels, fontsize=10, rotation=20, ha="right")
ax.set_yticks(range(25))
ax.set_yticklabels(top25["feature"], fontsize=9)

# annotate cells
for i in range(25):
    for j in range(3):
        val = auroc_matrix[i, j]
        text_color = "white" if val > 0.85 or val < 0.55 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=text_color, fontweight="bold")

ax.set_title("Pairwise AUROC — Top 25 Features", fontsize=14, fontweight="bold")
cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
cbar.set_label("AUROC", fontsize=10)

fig.tight_layout()
fig.savefig(RESULTS_DIR / "pairwise_auroc_heatmap.png", dpi=300, bbox_inches="tight")
fig.savefig(RESULTS_DIR / "pairwise_auroc_heatmap.pdf", bbox_inches="tight")
plt.close(fig)
print("✓ Figure 3: pairwise_auroc_heatmap.png")


# ── Figure 4: Feature Correlation Clusters (top 20) ───────────────────
top20_feats = ranking["feature"].head(20).tolist()
corr = features[top20_feats].corr(method="spearman").abs()

# simple hierarchical clustering for ordering
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

dist = 1 - corr.values
np.fill_diagonal(dist, 0)
dist = np.clip(dist, 0, None)  # numerical safety
Z = linkage(squareform(dist), method="average")
order = leaves_list(Z)

corr_ordered = corr.iloc[order, order]

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_ordered, dtype=bool), k=1)
data = corr_ordered.values.copy()
data[mask] = np.nan

im = ax.imshow(data, cmap="coolwarm", vmin=0, vmax=1, aspect="equal")
ax.set_xticks(range(20))
ax.set_xticklabels(corr_ordered.columns, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(20))
ax.set_yticklabels(corr_ordered.index, fontsize=9)

# annotate high correlations
for i in range(20):
    for j in range(i + 1):
        val = corr_ordered.values[i, j]
        if val > 0.8:
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, fontweight="bold", color="white")

ax.set_title("Feature Redundancy — Spearman |ρ| (Top 20, Clustered)", fontsize=14, fontweight="bold")
cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
cbar.set_label("Spearman |ρ|", fontsize=10)

fig.tight_layout()
fig.savefig(RESULTS_DIR / "feature_correlation_clusters.png", dpi=300, bbox_inches="tight")
fig.savefig(RESULTS_DIR / "feature_correlation_clusters.pdf", bbox_inches="tight")
plt.close(fig)
print("✓ Figure 4: feature_correlation_clusters.png")


# ── Summary CSV ────────────────────────────────────────────────────────
summary = ranking.copy()
summary["significance"] = summary["kw_pvalue"].apply(
    lambda p: "★★★" if p < 0.001 else ("★★" if p < 0.01 else ("★" if p < 0.05 else "n.s."))
)
summary["best_pairwise_effect"] = summary[
    ["effect_hypotonic_vs_normal", "effect_hypotonic_vs_seizure", "effect_normal_vs_seizure"]
].max(axis=1)
summary["best_auroc"] = summary[
    ["auroc_hypotonic_vs_normal", "auroc_hypotonic_vs_seizure", "auroc_normal_vs_seizure"]
].max(axis=1)
summary["best_separation_pair"] = summary.apply(
    lambda r: ["hypo-norm", "hypo-seiz", "norm-seiz"][
        np.argmax([r["effect_hypotonic_vs_normal"], r["effect_hypotonic_vs_seizure"], r["effect_normal_vs_seizure"]])
    ], axis=1
)

out_cols = [
    "feature", "significance", "kw_pvalue", "kw_stat",
    "best_pairwise_effect", "best_auroc", "best_separation_pair",
    "effect_normal_vs_seizure", "auroc_normal_vs_seizure",
    "effect_hypotonic_vs_normal", "auroc_hypotonic_vs_normal",
    "effect_hypotonic_vs_seizure", "auroc_hypotonic_vs_seizure",
]
out = summary[out_cols].copy()
out.columns = [
    "Feature", "Sig", "KW p-value", "KW H-stat",
    "Best Effect", "Best AUROC", "Best Pair",
    "Effect (N-S)", "AUROC (N-S)",
    "Effect (H-N)", "AUROC (H-N)",
    "Effect (H-S)", "AUROC (H-S)",
]

out.to_csv(RESULTS_DIR / "summary_statistics.csv", index=False, float_format="%.6g")
print(f"✓ Summary CSV: summary_statistics.csv ({len(out)} features)")


# ── Print quick stats ──────────────────────────────────────────────────
n_sig_001 = (ranking["kw_pvalue"] < 0.001).sum()
n_sig_01 = (ranking["kw_pvalue"] < 0.01).sum()
n_sig_05 = (ranking["kw_pvalue"] < 0.05).sum()
n_total = len(ranking)
n_videos = len(features)
n_groups = features["group"].value_counts().to_dict()

print(f"\n{'='*60}")
print(f"Dataset: {n_videos} videos — {n_groups}")
print(f"Features tested: {n_total}")
print(f"Significant at p < 0.05:  {n_sig_05}/{n_total} ({100*n_sig_05/n_total:.0f}%)")
print(f"Significant at p < 0.01:  {n_sig_01}/{n_total} ({100*n_sig_01/n_total:.0f}%)")
print(f"Significant at p < 0.001: {n_sig_001}/{n_total} ({100*n_sig_001/n_total:.0f}%)")
print(f"Top feature: {ranking.iloc[0]['feature']} (p = {ranking.iloc[0]['kw_pvalue']:.2e})")
print(f"{'='*60}")
