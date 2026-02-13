"""Analyse Top 20 des meilleures configurations avec statistiques par backbone."""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Style matplotlib
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150

# Détails des couches par backbone (extraits de config.py)
LAYER_DETAILS = {
    "sd_vae": {
        0: "conv_in (128ch, 256×256)",
        1: "down_blocks.0.resnets.0 (128ch, 256×256)",
        2: "down_blocks.0.resnets.1 (128ch, 256×256)",
        3: "down_blocks.0.downsamplers.0 (128ch, 128×128)",
        4: "down_blocks.1.resnets.0 (256ch, 128×128)",
        5: "down_blocks.1.resnets.1 (256ch, 128×128)",
        6: "down_blocks.1.downsamplers.0 (256ch, 64×64)",
        7: "down_blocks.2.resnets.0 (512ch, 64×64)",
        8: "down_blocks.2.resnets.1 (512ch, 64×64)",
        9: "down_blocks.2.downsamplers.0 (512ch, 32×32)",
        10: "down_blocks.3.resnets.0 (512ch, 32×32)",
        11: "down_blocks.3.resnets.1 (512ch, 32×32)",
        12: "mid_block.resnets.0 (512ch, 32×32)",
        13: "mid_block.attentions.0 (512ch, 32×32)",
        14: "mid_block.resnets.1 (512ch, 32×32)",
        15: "conv_norm_out (512ch, 32×32)",
        16: "conv_out (8ch, 32×32)",
    },
    "dinov2_vitb14": {
        0: "patch_embed (768d, 1369 tokens)",
        1: "blocks.0 (768d, early)",
        2: "blocks.1 (768d, early-mid)",
        3: "blocks.2 (768d)",
        4: "blocks.3 (768d)",
        5: "blocks.4 (768d, mid)",
        6: "blocks.5 (768d)",
        7: "blocks.6 (768d)",
        8: "blocks.7 (768d, mid-late)",
        9: "blocks.8 (768d)",
        10: "blocks.9 (768d)",
        11: "blocks.10 (768d, late)",
        12: "blocks.11 (768d)",
        13: "norm (768d, final)",
    },
    "vgg19": {
        0: "features.0 - conv1_1 (64ch, 224×224)",
        1: "features.2 - conv1_2 (64ch, 224×224)",
        2: "features.5 - conv2_1 (128ch, 112×112)",
        3: "features.7 - conv2_2 (128ch, 112×112)",
        4: "features.10 - conv3_1 (256ch, 56×56)",
        5: "features.12 - conv3_2 (256ch, 56×56)",
        6: "features.14 - conv3_3 (256ch, 56×56)",
        7: "features.16 - conv3_4 (256ch, 56×56)",
        8: "features.19 - conv4_1 (512ch, 28×28)",
        9: "features.21 - conv4_2 (512ch, 28×28)",
        10: "features.23 - conv4_3 (512ch, 28×28)",
        11: "features.25 - conv4_4 (512ch, 28×28)",
        12: "features.28 - conv5_1 (512ch, 14×14)",
        13: "features.30 - conv5_2 (512ch, 14×14)",
        14: "features.32 - conv5_3 (512ch, 14×14)",
        15: "features.34 - conv5_4 (512ch, 14×14)",
        16: "classifier.0 - FC1 (4096d)",
        17: "classifier.3 - FC2 (4096d)",
    },
    "lpips_vgg": {
        0: "slice1.0 - conv1_1 (64ch, 224×224)",
        1: "slice1.2 - conv1_2 (64ch, 224×224)",
        2: "slice2.5 - conv2_1 (128ch, 112×112)",
        3: "slice2.7 - conv2_2 (128ch, 112×112)",
        4: "slice3.10 - conv3_1 (256ch, 56×56)",
        5: "slice3.12 - conv3_2 (256ch, 56×56)",
        6: "slice3.14 - conv3_3 (256ch, 56×56)",
        7: "slice4.17 - conv4_1 (512ch, 28×28)",
        8: "slice4.19 - conv4_2 (512ch, 28×28)",
        9: "slice4.21 - conv4_3 (512ch, 28×28)",
        10: "slice5.24 - conv5_1 (512ch, 14×14)",
        11: "slice5.26 - conv5_2 (512ch, 14×14)",
        12: "slice5.28 - conv5_3 (512ch, 14×14)",
    },
    "clip_vit_base": {
        0: "embeddings (768d, 50 tokens)",
        1: "encoder.layers.0 (768d, early)",
        2: "encoder.layers.1 (768d)",
        3: "encoder.layers.2 (768d, early-mid)",
        4: "encoder.layers.3 (768d)",
        5: "encoder.layers.4 (768d)",
        6: "encoder.layers.5 (768d, mid)",
        7: "encoder.layers.6 (768d)",
        8: "encoder.layers.7 (768d)",
        9: "encoder.layers.8 (768d, mid-late)",
        10: "encoder.layers.9 (768d)",
        11: "encoder.layers.10 (768d)",
        12: "encoder.layers.11 (768d, final)",
    },
    "resnet50": {
        7: "layer3 (1024ch, 14×14)",
        8: "layer4.0 (2048ch, 7×7)",
        12: "layer4.1 (2048ch, 7×7)",
        13: "layer4.2 (2048ch, 7×7)",
        14: "layer3.5 (1024ch, 14×14)",
        15: "layer3.4 (1024ch, 14×14)",
    }
}


def get_layer_details(backbone, layer_idx):
    """Retourne les détails d'une couche."""
    if backbone in LAYER_DETAILS and layer_idx in LAYER_DETAILS[backbone]:
        return LAYER_DETAILS[backbone][layer_idx]
    return f"Layer {layer_idx}"


def create_top20_overall(df, output_dir):
    """Figure: Top 20 meilleures configurations (monotonie moyenne)."""
    # Trier par monotonie moyenne
    df_sorted = df.sort_values('monotonie_moyenne', ascending=False).head(20)

    # Créer des labels lisibles
    labels = []
    for _, row in df_sorted.iterrows():
        bb = row['modele_backbone']
        layer_idx = int(row['config_couches'].split('_')[-1]) if 'single_layer' in row['config_couches'] else -1
        layer_detail = get_layer_details(bb, layer_idx) if layer_idx >= 0 else "multi-layer"
        metric = row['metrique_distance']
        labels.append(f"{bb} | {layer_detail} | {metric}")

    # Calculer statistiques par backbone
    backbone_counts = Counter(df_sorted['modele_backbone'])
    total = len(df_sorted)
    stats_text = "Proportion dans Top 20:\n"
    for bb, count in backbone_counts.most_common():
        pct = 100 * count / total
        stats_text += f"  {bb}: {count}/20 ({pct:.1f}%)\n"

    # Créer figure
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3])

    # Panneau supérieur pour le titre du pie chart
    ax_title = fig.add_subplot(gs[0, 1])
    ax_title.axis('off')
    ax_title.text(0.5, 0.5, 'Répartition par Backbone',
                 ha='center', va='center',
                 fontsize=12, fontweight='bold')

    # Barres horizontales
    ax1 = fig.add_subplot(gs[:, 0])

    colors_bb = {'sd_vae': '#FF6B6B', 'dinov2_vitb14': '#4ECDC4', 'vgg19': '#FFD93D',
                 'lpips_vgg': '#95E1D3', 'clip_vit_base': '#A8E6CF', 'resnet50': '#FF8C94'}
    colors = [colors_bb.get(row['modele_backbone'], '#999') for _, row in df_sorted.iterrows()]

    y_pos = np.arange(len(labels))
    bars = ax1.barh(y_pos, df_sorted['monotonie_moyenne'].values, color=colors, edgecolor='white', linewidth=1.5)

    # Ajouter valeurs
    for i, (bar, val) in enumerate(zip(bars, df_sorted['monotonie_moyenne'].values)):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', ha='left', fontsize=9, fontweight='bold')

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel('Monotonie Moyenne (Spearman ρ)', fontsize=12, fontweight='bold')
    ax1.set_title('Top 20 Meilleures Configurations (Monotonie Moyenne)', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 1.1)
    ax1.axvline(x=1.0, color='grey', linestyle='--', alpha=0.3)
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()

    # Pie chart dans le panneau de droite
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.axis('off')

    # Pie chart des proportions
    pie_labels = [f"{bb}\n({count})" for bb, count in backbone_counts.most_common()]
    pie_colors = [colors_bb.get(bb, '#999') for bb, _ in backbone_counts.most_common()]
    ax2.pie([count for _, count in backbone_counts.most_common()],
           labels=pie_labels, colors=pie_colors, autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 9})

    plt.tight_layout()
    output_path = f"{output_dir}/fig_top20_overall.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved {output_path}")
    plt.close()


def create_top20_per_degradation(df, output_dir):
    """Figures: Top 20 par type de dégradation."""
    deg_cols = [c for c in df.columns if c.startswith('mono_')]
    deg_names = {'mono_blur': 'Blur', 'mono_noise': 'Noise',
                 'mono_aliasing': 'Aliasing', 'mono_blur_contrast': 'Blur+Contrast'}

    colors_bb = {'sd_vae': '#FF6B6B', 'dinov2_vitb14': '#4ECDC4', 'vgg19': '#FFD93D',
                 'lpips_vgg': '#95E1D3', 'clip_vit_base': '#A8E6CF', 'resnet50': '#FF8C94'}

    for deg_col in deg_cols:
        if deg_col not in df.columns:
            continue

        deg_name = deg_names.get(deg_col, deg_col.replace('mono_', ''))

        # Trier par ce type de dégradation
        df_sorted = df.sort_values(deg_col, ascending=False).head(20)

        # Créer labels
        labels = []
        for _, row in df_sorted.iterrows():
            bb = row['modele_backbone']
            layer_idx = int(row['config_couches'].split('_')[-1]) if 'single_layer' in row['config_couches'] else -1
            layer_detail = get_layer_details(bb, layer_idx) if layer_idx >= 0 else "multi-layer"
            metric = row['metrique_distance']
            labels.append(f"{bb} | {layer_detail} | {metric}")

        # Stats
        backbone_counts = Counter(df_sorted['modele_backbone'])
        total = len(df_sorted)
        stats_text = f"Top 20 - {deg_name}:\n"
        for bb, count in backbone_counts.most_common():
            pct = 100 * count / total
            stats_text += f"  {bb}: {count}/20 ({pct:.1f}%)\n"

        # Figure
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3])

        # Titre pie chart
        ax_title = fig.add_subplot(gs[0, 1])
        ax_title.axis('off')
        ax_title.text(0.5, 0.5, f'Répartition - {deg_name}',
                     ha='center', va='center',
                     fontsize=12, fontweight='bold')

        # Barres
        ax1 = fig.add_subplot(gs[:, 0])

        colors = [colors_bb.get(row['modele_backbone'], '#999') for _, row in df_sorted.iterrows()]
        y_pos = np.arange(len(labels))
        bars = ax1.barh(y_pos, df_sorted[deg_col].values, color=colors, edgecolor='white', linewidth=1.5)

        for i, (bar, val) in enumerate(zip(bars, df_sorted[deg_col].values)):
            ax1.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', ha='left', fontsize=9, fontweight='bold')

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(labels, fontsize=8)
        ax1.set_xlabel(f'Monotonie {deg_name} (Spearman ρ)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Top 20 Meilleures Configurations - {deg_name}', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 1.1)
        ax1.axvline(x=1.0, color='grey', linestyle='--', alpha=0.3)
        ax1.grid(axis='x', alpha=0.3)
        ax1.invert_yaxis()

        # Pie chart panneau droit
        ax2 = fig.add_subplot(gs[1, 1])
        ax2.axis('off')

        # Pie chart
        if backbone_counts:
            pie_labels = [f"{bb}\n({count})" for bb, count in backbone_counts.most_common()]
            pie_colors = [colors_bb.get(bb, '#999') for bb, _ in backbone_counts.most_common()]
            ax2.pie([count for _, count in backbone_counts.most_common()],
                   labels=pie_labels, colors=pie_colors, autopct='%1.1f%%',
                   startangle=90, textprops={'fontsize': 9})

        plt.tight_layout()
        output_path = f"{output_dir}/fig_top20_{deg_col.replace('mono_', '')}.png"
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"✓ Saved {output_path}")
        plt.close()


def create_layer_profiles_simple(df, output_dir):
    """Profils de couches simples avec détails en abscisse."""
    backbones = df['modele_backbone'].unique()

    for bb in backbones:
        bb_data = df[df['modele_backbone'] == bb]
        bb_single = bb_data[bb_data['config_couches'].str.startswith('single_layer')]

        if len(bb_single) == 0:
            continue

        # Extraire données
        layer_indices = []
        layer_labels = []
        monotonies = []

        for _, row in bb_single.iterrows():
            layer_idx = int(row['config_couches'].split('_')[-1])
            layer_indices.append(layer_idx)
            layer_labels.append(f"{layer_idx}\n{get_layer_details(bb, layer_idx)}")
            monotonies.append(row['monotonie_moyenne'])

        # Trier par indice
        sorted_data = sorted(zip(layer_indices, layer_labels, monotonies))
        layer_indices, layer_labels, monotonies = zip(*sorted_data)

        # Figure plus carrée pour mieux voir les variations
        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot ligne + points
        ax.plot(range(len(layer_indices)), monotonies, 'o-', linewidth=2, markersize=8,
               color='#2196F3', markerfacecolor='#FF6B6B', markeredgewidth=2, markeredgecolor='white')

        # X-axis
        ax.set_xticks(range(len(layer_indices)))
        ax.set_xticklabels(layer_labels, rotation=45, ha='right', fontsize=8)

        # Style
        ax.set_ylabel('Monotonie Moyenne', fontsize=12, fontweight='bold')
        ax.set_title(f'{bb.upper()} - Profil de monotonie par couche', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.3)
        ax.axhline(y=0.9, color='darkgreen', linestyle='--', alpha=0.4)
        ax.set_ylim(-0.1, 1.05)

        plt.tight_layout()
        output_path = f"{output_dir}/fig_profile_{bb}.png"
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"✓ Saved {output_path}")
        plt.close()


def create_layer_bars_ranked(df, output_dir):
    """Barres horizontales avec couches triées par monotonie décroissante."""
    backbones = df['modele_backbone'].unique()

    for bb in backbones:
        bb_data = df[df['modele_backbone'] == bb]
        bb_single = bb_data[bb_data['config_couches'].str.startswith('single_layer')]

        if len(bb_single) == 0:
            continue

        # Extraire données
        layer_data = []
        for _, row in bb_single.iterrows():
            layer_idx = int(row['config_couches'].split('_')[-1])
            layer_name = get_layer_details(bb, layer_idx)
            mono = row['monotonie_moyenne']
            # Format: "Layer X: description"
            label = f"Layer {layer_idx}: {layer_name}"
            layer_data.append((layer_idx, label, mono))

        # Trier par monotonie DÉCROISSANTE (meilleure en haut)
        layer_data.sort(key=lambda x: x[2], reverse=True)

        # Extraire pour le plot
        indices, labels, monotonies = zip(*layer_data)

        # Figure carrée pour avoir de la hauteur
        n_layers = len(labels)
        fig_height = max(10, n_layers * 0.5)  # Au moins 10, ou 0.5 par couche
        fig, ax = plt.subplots(figsize=(12, fig_height))

        # Positions Y (inversées car barh plot de haut en bas)
        y_pos = range(len(labels))

        # Couleurs basées sur la monotonie
        colors = []
        for mono in monotonies:
            if mono > 0.9:
                colors.append('#4CAF50')  # Vert pour excellent
            elif mono >= 0.8:
                colors.append('#8BC34A')  # Vert clair pour très bon
            elif mono >= 0.6:
                colors.append('#2196F3')  # Bleu pour bon
            elif mono >= 0.4:
                colors.append('#FF9800')  # Orange pour moyen
            else:
                colors.append('#F44336')  # Rouge pour faible

        # Barres horizontales
        bars = ax.barh(y_pos, monotonies, color=colors, edgecolor='white', linewidth=1.5)

        # Ajouter les valeurs à droite des barres
        for i, (bar, mono) in enumerate(zip(bars, monotonies)):
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{mono:.3f}',
                   ha='left', va='center', fontsize=9, fontweight='bold')

        # Configuration des axes
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Monotonie Moyenne', fontsize=12, fontweight='bold')
        ax.set_title(f'{bb.upper()} - Classement des couches par monotonie',
                    fontsize=14, fontweight='bold', pad=20)

        # Lignes de référence
        ax.axvline(x=0.9, color='green', linestyle='--', alpha=0.3, linewidth=2, label='Excellent (>0.9)')
        ax.axvline(x=0.8, color='lightgreen', linestyle='--', alpha=0.3, linewidth=2, label='Très bon (≥0.8)')
        ax.axvline(x=0.6, color='blue', linestyle='--', alpha=0.3, linewidth=2, label='Bon (≥0.6)')
        ax.axvline(x=0.4, color='orange', linestyle='--', alpha=0.3, linewidth=2, label='Moyen (≥0.4)')

        # Grille et limites
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        ax.set_xlim(0, 1.05)
        ax.legend(loc='lower right', fontsize=9)

        # Inverser l'axe Y pour avoir la meilleure en haut
        ax.invert_yaxis()

        plt.tight_layout()
        output_path = f"{output_dir}/fig_bars_ranked_{bb}.png"
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"✓ Saved {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyse Top 20 avec stats")
    parser.add_argument("csv_file", help="CSV de résultats")
    parser.add_argument("-o", "--output-dir", default="results", help="Dossier de sortie")
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("ANALYSE TOP 20 + PROFILS SIMPLIFIÉS")
    print(f"{'='*80}\n")

    df = pd.read_csv(args.csv_file)
    print(f"✓ Loaded {len(df)} rows from {args.csv_file}\n")

    print("Génération des figures...\n")

    # Top 20 overall
    create_top20_overall(df, args.output_dir)

    # Top 20 per degradation
    create_top20_per_degradation(df, args.output_dir)

    # Layer profiles simples
    create_layer_profiles_simple(df, args.output_dir)

    # Barres horizontales classées par monotonie
    create_layer_bars_ranked(df, args.output_dir)

    print(f"\n{'='*80}")
    print(f"✓ Toutes les figures générées dans {args.output_dir}/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
