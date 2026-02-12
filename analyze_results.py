"""Analyse statistique et génération de figures à partir d'un CSV de résultats.

Usage:
    python analyze_results.py results/resultats_all_20260209.csv
"""

import argparse
import csv
import statistics
from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.size"] = 11


def load_csv(filepath):
    with open(filepath, newline="") as f:
        return list(csv.DictReader(f))


def analyze(rows, output_dir):
    deg_cols = [c for c in rows[0].keys() if c.startswith("mono_")]
    deg_labels = [c.replace("mono_", "").replace("_", "+").title() for c in deg_cols]
    backbones = sorted(set(r["modele_backbone"] for r in rows))
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0", "#00BCD4"]

    # ─── Best configs ───
    best_overall = {}
    best_per_deg = {bb: {} for bb in backbones}

    for bb in backbones:
        bb_rows = [r for r in rows if r["modele_backbone"] == bb]
        best_overall[bb] = max(bb_rows, key=lambda r: float(r["monotonie_moyenne"]))
        for deg in deg_cols:
            best_per_deg[bb][deg] = max(bb_rows, key=lambda r: float(r[deg]))

    # ─── Print stats ───
    print(f"\n{'=' * 90}")
    print(f"DATASET: {len(rows)} lignes, {len(backbones)} backbones")
    print(f"{'=' * 90}")

    print(f"\nMEILLEURE CONFIG PAR BACKBONE:")
    for bb in backbones:
        b = best_overall[bb]
        print(f"  {bb:15s} | mono={b['monotonie_moyenne']:>9s} | "
              f"couche={b['modules_extraits']} | metric={b['metrique_distance']} | dist={b['distribution']}")

    print(f"\nMONOTONIE MOYENNE PAR BACKBONE:")
    by_backbone = defaultdict(list)
    for r in rows:
        by_backbone[r["modele_backbone"]].append(float(r["monotonie_moyenne"]))
    for bb in sorted(by_backbone, key=lambda k: statistics.mean(by_backbone[k]), reverse=True):
        vals = by_backbone[bb]
        print(f"  {bb:15s} | mean={statistics.mean(vals):+.4f} | std={statistics.stdev(vals):.4f} | "
              f"min={min(vals):+.4f} | max={max(vals):+.4f} | n={len(vals)}")

    print(f"\nMEILLEURE COUCHE PAR BACKBONE (single layers):")
    by_bb_layer = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["config_couches"].startswith("single_layer"):
            by_bb_layer[r["modele_backbone"]][r["modules_extraits"]].append(float(r["monotonie_moyenne"]))
    for bb in sorted(by_bb_layer):
        print(f"  --- {bb} ---")
        layers = by_bb_layer[bb]
        for layer in sorted(layers, key=lambda k: statistics.mean(layers[k]), reverse=True):
            print(f"    {layer:40s} | mean={statistics.mean(layers[layer]):+.4f} | n={len(layers[layer])}")

    # ─── FIGURE 1: Barres groupées best mono par dégradation x backbone ───
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(deg_labels))
    width = 0.8 / len(backbones)
    for i, bb in enumerate(backbones):
        vals = [float(best_per_deg[bb][d][d]) for d in deg_cols]
        bars = ax.bar(x + i * width, vals, width, label=bb, color=colors[i % len(colors)], edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("Type de dégradation")
    ax.set_ylabel("Meilleure monotonie (Spearman)")
    ax.set_title("Meilleure monotonie par type de dégradation et par backbone")
    ax.set_xticks(x + width * (len(backbones) - 1) / 2)
    ax.set_xticklabels(deg_labels)
    ax.legend(title="Backbone")
    ax.set_ylim(0, 1.15)
    ax.axhline(y=1.0, color="grey", linestyle="--", alpha=0.3)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig1_best_mono_per_degradation.png", dpi=150)
    print(f"\nSaved fig1_best_mono_per_degradation.png")

    # ─── FIGURE 2: Barres horizontales best mono moyenne par backbone ───
    fig, ax = plt.subplots(figsize=(10, max(4, len(backbones) * 1.2)))
    best_means = [float(best_overall[bb]["monotonie_moyenne"]) for bb in backbones]
    bars = ax.barh(backbones, best_means, color=[colors[i % len(colors)] for i in range(len(backbones))],
                   edgecolor="white", height=0.5)
    for bar, v, bb in zip(bars, best_means, backbones):
        b = best_overall[bb]
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{v:.4f}  ({b["modules_extraits"]}, {b["metrique_distance"]})',
                ha="left", va="center", fontsize=9)
    ax.set_xlabel("Monotonie moyenne")
    ax.set_title("Meilleure monotonie moyenne par backbone (config optimale)")
    ax.set_xlim(0, 1.25)
    ax.axvline(x=1.0, color="grey", linestyle="--", alpha=0.3)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig2_best_mean_mono_per_backbone.png", dpi=150)
    print("Saved fig2_best_mean_mono_per_backbone.png")

    # ─── FIGURE 3: Heatmap best mono (backbone x dégradation) ───
    fig, ax = plt.subplots(figsize=(9, max(4, len(backbones) * 1.2)))
    data = np.array([[float(best_per_deg[bb][d][d]) for d in deg_cols] for bb in backbones])
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-0.2, vmax=1.0)
    ax.set_xticks(range(len(deg_labels)))
    ax.set_xticklabels(deg_labels)
    ax.set_yticks(range(len(backbones)))
    ax.set_yticklabels(backbones)
    for i in range(len(backbones)):
        for j in range(len(deg_labels)):
            val = data[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Monotonie (Spearman)")
    ax.set_title("Heatmap : meilleure monotonie par backbone x dégradation")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig3_heatmap_best_mono.png", dpi=150)
    print("Saved fig3_heatmap_best_mono.png")

    # ─── FIGURE 4: Radar chart ───
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(deg_labels), endpoint=False).tolist()
    angles += angles[:1]
    for i, bb in enumerate(backbones):
        vals = [float(best_per_deg[bb][d][d]) for d in deg_cols]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=bb, color=colors[i % len(colors)])
        ax.fill(angles, vals, alpha=0.1, color=colors[i % len(colors)])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(deg_labels, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title("Radar : meilleure monotonie par dégradation", y=1.08)
    ax.legend(loc="lower right", bbox_to_anchor=(1.25, 0))
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig4_radar_best_mono.png", dpi=150)
    print("Saved fig4_radar_best_mono.png")

    # ─── FIGURE 5: Profils par couche pour chaque backbone (2x2 grid ou plus) ───
    n_bb = len(backbones)
    ncols = min(2, n_bb)
    nrows = (n_bb + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5 * nrows))
    if n_bb == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    metrics_in_data = sorted(set(r["metrique_distance"] for r in rows))
    metric_colors = {"mahalanobis": "#2196F3", "neg_loglik": "#E91E63", "fid": "#4CAF50", "cmmd": "#FF9800"}
    metric_markers = {"mahalanobis": "o", "neg_loglik": "s", "fid": "^", "cmmd": "D"}

    for idx, bb in enumerate(backbones):
        ax = axes[idx]
        bb_single = [r for r in rows if r["modele_backbone"] == bb
                     and r["config_couches"].startswith("single_layer")
                     and r["distribution"] == "gmm_diag"]
        for metric in metrics_in_data:
            metric_rows = sorted(
                [r for r in bb_single if r["metrique_distance"] == metric],
                key=lambda r: int(r["indices_couches"].split("|")[0].strip()),
            )
            if not metric_rows:
                continue
            names = [r["modules_extraits"] for r in metric_rows]
            vals = [float(r["monotonie_moyenne"]) for r in metric_rows]
            ax.plot(range(len(names)), vals,
                    marker=metric_markers.get(metric, "o"),
                    color=metric_colors.get(metric, "#666"),
                    label=metric.upper(), linewidth=2, markersize=6)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_title(bb, fontsize=13, fontweight="bold")
        ax.set_ylabel("Monotonie moyenne")
        ax.legend(fontsize=9)
        ax.axhline(y=0, color="grey", linestyle="--", alpha=0.3)
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.7, 1.05)

    # Hide unused subplots
    for idx in range(n_bb, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Profil de monotonie par couche pour chaque backbone", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig5_all_backbones_layer_profiles.png", dpi=150)
    print("Saved fig5_all_backbones_layer_profiles.png")

    print(f"\n=== {5} figures générées dans {output_dir}/ ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse des résultats de monotonie")
    parser.add_argument("csv_file", help="CSV de résultats à analyser")
    parser.add_argument("-o", "--output-dir", default="results", help="Dossier de sortie pour les figures")
    args = parser.parse_args()

    rows = load_csv(args.csv_file)
    analyze(rows, args.output_dir)
