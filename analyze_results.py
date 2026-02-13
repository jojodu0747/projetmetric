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

    # ─── FIGURE 5: 6 figures séparées (une par backbone) avec noms de couches ───
    metrics_in_data = sorted(set(r["metrique_distance"] for r in rows))

    # Palette de couleurs vibrante et distinctive
    metric_colors = {
        "mmd_rbf_auto": "#FF6B6B",     # Rouge corail
        "mahalanobis": "#4ECDC4",      # Turquoise
        "neg_loglik": "#FFD93D",       # Jaune doré
        "fid": "#95E1D3",              # Vert menthe
        "cmmd": "#A8E6CF",             # Vert pastel
        "energy": "#FF8C94",           # Rose saumon
        "sinkhorn": "#C7CEEA"          # Bleu lavande
    }
    metric_markers = {
        "mmd_rbf_auto": "o",
        "mahalanobis": "s",
        "neg_loglik": "^",
        "fid": "D",
        "cmmd": "v",
        "energy": "p",
        "sinkhorn": "*"
    }

    # Créer une figure séparée pour chaque backbone
    for bb in backbones:
        fig, ax = plt.subplots(figsize=(16, 8), facecolor='white')

        # Fond légèrement coloré
        ax.set_facecolor('#F8F9FA')

        bb_single = [r for r in rows if r["modele_backbone"] == bb
                     and r["config_couches"].startswith("single_layer")]

        if not bb_single:
            plt.close(fig)
            continue

        # Extraire tous les layer names disponibles
        layer_data = {}  # {layer_idx: {"name": ..., "metrics": {metric: value}}}
        for r in bb_single:
            layer_idx = int(r["config_couches"].split("_")[-1])
            if layer_idx not in layer_data:
                layer_data[layer_idx] = {
                    "name": r["modules_extraits"] if r["modules_extraits"] else f"Layer {layer_idx}",
                    "metrics": {}
                }
            metric = r["metrique_distance"]
            layer_data[layer_idx]["metrics"][metric] = float(r["monotonie_moyenne"])

        # Trier par indice
        sorted_layers = sorted(layer_data.items())
        layer_indices = [idx for idx, _ in sorted_layers]
        layer_names = [data["name"] for _, data in sorted_layers]

        # Tracer chaque métrique
        for metric in metrics_in_data:
            vals = []
            indices_with_data = []
            for idx in layer_indices:
                if metric in layer_data[idx]["metrics"]:
                    vals.append(layer_data[idx]["metrics"][metric])
                    indices_with_data.append(idx)

            if not vals:
                continue

            color = metric_colors.get(metric, "#666666")
            marker = metric_markers.get(metric, "o")

            # Tracer avec style amélioré
            ax.plot(indices_with_data, vals,
                    marker=marker,
                    color=color,
                    label=metric.upper().replace("_", " "),
                    linewidth=3,
                    markersize=10,
                    markeredgewidth=2,
                    markeredgecolor='white',
                    alpha=0.9)

            # Marquer le meilleur point pour cette métrique
            if vals:
                best_idx = np.argmax(vals)
                ax.scatter(indices_with_data[best_idx], vals[best_idx],
                          s=300, color=color, marker='*',
                          edgecolors='gold', linewidths=3, zorder=10)

        # Styling amélioré
        ax.set_title(f"Profil de monotonie - {bb.upper().replace('_', ' ')}\n★ = Meilleur score par métrique",
                    fontsize=18, fontweight="bold",
                    color='#2C3E50', pad=20)
        ax.set_xlabel("Couche (indice | nom du module)", fontsize=13, fontweight='bold')
        ax.set_ylabel("Monotonie moyenne (Spearman ρ)", fontsize=13, fontweight='bold')

        # X-axis avec indices ET noms
        ax.set_xticks(layer_indices)
        # Créer des labels combinés: "idx\nname"
        combined_labels = [f"{idx}\n{name}" for idx, name in zip(layer_indices, layer_names)]
        ax.set_xticklabels(combined_labels, rotation=45, ha='right', fontsize=9)

        # Légende améliorée
        ax.legend(fontsize=11, loc='best', framealpha=0.95,
                 edgecolor='#CCCCCC', fancybox=True, shadow=True)

        # Lignes de référence
        ax.axhline(y=0, color='#E74C3C', linestyle='--', alpha=0.5, linewidth=2, label='_nolegend_')
        ax.axhline(y=0.8, color='#27AE60', linestyle='--', alpha=0.4, linewidth=2, label='_nolegend_')

        # Grille améliorée
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, color='#95A5A6')
        ax.set_axisbelow(True)

        # Limites et ticks
        ax.set_ylim(-0.15, 1.05)
        if layer_indices:
            ax.set_xlim(min(layer_indices) - 0.5, max(layer_indices) + 0.5)

        # Bordure du plot
        for spine in ax.spines.values():
            spine.set_edgecolor('#BDC3C7')
            spine.set_linewidth(2)

        # Sauvegarder la figure individuelle
        plt.tight_layout()
        output_path = f"{output_dir}/fig5_{bb}_layer_profile.png"
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved {output_path}")
        plt.close(fig)

    print(f"\n=== {5} figures générées dans {output_dir}/ ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse des résultats de monotonie")
    parser.add_argument("csv_file", help="CSV de résultats à analyser")
    parser.add_argument("-o", "--output-dir", default="results", help="Dossier de sortie pour les figures")
    args = parser.parse_args()

    rows = load_csv(args.csv_file)
    analyze(rows, args.output_dir)
