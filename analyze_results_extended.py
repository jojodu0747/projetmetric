"""Analyse statistique étendue avec visualisations supplémentaires.

Génère 4 figures additionnelles pour une analyse plus approfondie :
- Fig 6: Boxplots de distribution des monotonies par backbone
- Fig 7: Comparaison de performance des métriques de distance
- Fig 8: Analyse par position de couche (early/mid/late)
- Fig 9: Heatmap de corrélation entre types de dégradation

Usage:
    python analyze_results_extended.py results/resultats_medium_20260213_all.csv -o results
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Style matplotlib
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

def load_data(csv_path):
    """Charge les données avec pandas."""
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} rows from {csv_path}")
    return df

def fig6_boxplots_distribution(df, output_dir):
    """Figure 6: Boxplots de la distribution des monotonies par backbone."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Sort backbones by median monotonicity
    backbone_order = df.groupby('modele_backbone')['monotonie_moyenne'].median().sort_values(ascending=False).index

    # Create boxplot
    bp = df.boxplot(
        column='monotonie_moyenne',
        by='modele_backbone',
        ax=ax,
        positions=range(len(backbone_order)),
        patch_artist=True,
        return_type='dict'
    )

    # Color boxes
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0', '#00BCD4']
    for patch, color in zip(bp['monotonie_moyenne']['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Styling
    ax.set_xlabel('Backbone Model', fontsize=12)
    ax.set_ylabel('Monotonie Moyenne (Spearman)', fontsize=12)
    ax.set_title('Distribution de la monotonie par backbone (variabilité et robustesse)',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(-0.2, 1.05)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.3, linewidth=1, label='Seuil 0.8')
    plt.suptitle('')  # Remove automatic suptitle

    # Add statistics annotations
    for i, bb in enumerate(backbone_order):
        bb_data = df[df['modele_backbone'] == bb]['monotonie_moyenne']
        median = bb_data.median()
        q1 = bb_data.quantile(0.25)
        q3 = bb_data.quantile(0.75)
        ax.text(i, -0.15, f'n={len(bb_data)}', ha='center', va='top', fontsize=9)

    plt.tight_layout()
    output_path = f"{output_dir}/fig6_boxplots_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved {output_path}")
    plt.close()

def fig7_metric_comparison(df, output_dir):
    """Figure 7: Comparaison de performance des métriques de distance."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Group by backbone and metric
    metric_means = df.groupby(['modele_backbone', 'metrique_distance'])['monotonie_moyenne'].mean().unstack(fill_value=0)

    # Sort backbones by overall performance
    backbone_order = df.groupby('modele_backbone')['monotonie_moyenne'].mean().sort_values(ascending=False).index
    metric_means = metric_means.loc[backbone_order]

    # Create grouped bar chart
    x = np.arange(len(backbone_order))
    width = 0.15
    metrics = metric_means.columns
    colors_metric = {'mmd_rbf_auto': '#2196F3', 'mahalanobis': '#FF9800',
                     'neg_loglik': '#E91E63', 'fid': '#4CAF50', 'cmmd': '#9C27B0'}

    for i, metric in enumerate(metrics):
        values = metric_means[metric].values
        bars = ax.bar(x + i * width, values, width, label=metric,
                     color=colors_metric.get(metric, '#999'), edgecolor='white', alpha=0.9)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0.1:  # Only label significant values
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Backbone Model', fontsize=12)
    ax.set_ylabel('Monotonie Moyenne', fontsize=12)
    ax.set_title('Performance moyenne par métrique de distance × backbone',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(backbone_order, rotation=30, ha='right')
    ax.legend(title='Métrique Distance', loc='upper right', fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.3, linewidth=1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = f"{output_dir}/fig7_metric_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved {output_path}")
    plt.close()

def fig8_layer_position_analysis(df, output_dir):
    """Figure 8: Analyse par position de couche (early/mid/late)."""
    # Parse position_couche to extract layer stage
    def categorize_layer_stage(position_str, total_layers):
        """Categorize layer as early/mid/late based on position."""
        try:
            if pd.isna(position_str) or position_str == '':
                return 'unknown'

            # Extract first index from "X / Y" or "X | Y | Z / Y" format
            position_part = position_str.split('/')[0].strip()
            if '|' in position_part:
                position_part = position_part.split('|')[0].strip()

            layer_idx = int(position_part)

            # Categorize based on position
            if layer_idx < total_layers / 3:
                return 'Early'
            elif layer_idx < 2 * total_layers / 3:
                return 'Mid'
            else:
                return 'Late'
        except:
            return 'unknown'

    # Add stage column
    df_copy = df.copy()
    df_copy['layer_stage'] = df_copy.apply(
        lambda row: categorize_layer_stage(row['position_couche'], row['nb_couches_modele']),
        axis=1
    )

    # Filter out unknown stages
    df_stage = df_copy[df_copy['layer_stage'] != 'unknown']

    # Group by backbone and stage
    stage_means = df_stage.groupby(['modele_backbone', 'layer_stage'])['monotonie_moyenne'].mean().unstack(fill_value=np.nan)

    # Reorder stages
    stage_order = ['Early', 'Mid', 'Late']
    stage_means = stage_means[[s for s in stage_order if s in stage_means.columns]]

    # Sort backbones (only those with stage data)
    backbone_order_all = df.groupby('modele_backbone')['monotonie_moyenne'].mean().sort_values(ascending=False).index
    backbone_order = [bb for bb in backbone_order_all if bb in stage_means.index]
    stage_means = stage_means.loc[backbone_order]

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(backbone_order))
    width = 0.25
    colors_stage = {'Early': '#4CAF50', 'Mid': '#FF9800', 'Late': '#E91E63'}

    for i, stage in enumerate(stage_order):
        if stage in stage_means.columns:
            values = stage_means[stage].values
            bars = ax.bar(x + i * width, values, width, label=stage,
                         color=colors_stage[stage], edgecolor='white', alpha=0.9)

            # Add value labels
            for bar, val in zip(bars, values):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                           f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Backbone Model', fontsize=12)
    ax.set_ylabel('Monotonie Moyenne', fontsize=12)
    ax.set_title('Performance par position de couche (Early/Mid/Late) × backbone',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(backbone_order, rotation=30, ha='right')
    ax.legend(title='Layer Stage', loc='upper right', fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.3, linewidth=1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = f"{output_dir}/fig8_layer_position_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved {output_path}")
    plt.close()

def fig9_degradation_correlation(df, output_dir):
    """Figure 9: Heatmap de corrélation entre types de dégradation."""
    # Extract monotonicity columns
    mono_cols = [c for c in df.columns if c.startswith('mono_')]
    mono_data = df[mono_cols].copy()

    # Rename columns for better display
    mono_data.columns = [c.replace('mono_', '').replace('_', '+').title() for c in mono_cols]

    # Compute correlation matrix
    corr_matrix = mono_data.corr()

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    # Mask upper triangle for cleaner display
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1, ax=ax)

    ax.set_title('Corrélation entre types de dégradation (Pearson)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    output_path = f"{output_dir}/fig9_degradation_correlation.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyse étendue des résultats")
    parser.add_argument("csv_file", help="CSV de résultats à analyser")
    parser.add_argument("-o", "--output-dir", default="results", help="Dossier de sortie")
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("ANALYSE STATISTIQUE ÉTENDUE")
    print(f"{'='*80}\n")

    # Load data
    df = load_data(args.csv_file)

    # Generate figures
    print("\nGénération des figures...")
    fig6_boxplots_distribution(df, args.output_dir)
    fig7_metric_comparison(df, args.output_dir)
    fig8_layer_position_analysis(df, args.output_dir)
    fig9_degradation_correlation(df, args.output_dir)

    print(f"\n{'='*80}")
    print(f"✓ 4 figures supplémentaires générées dans {args.output_dir}/")
    print(f"{'='*80}\n")

    # Print additional insights
    print("\n📊 INSIGHTS SUPPLÉMENTAIRES:")

    # Variability ranking
    print("\n1. Variabilité (std) par backbone (du plus stable au plus variable):")
    variability = df.groupby('modele_backbone')['monotonie_moyenne'].std().sort_values()
    for bb, std in variability.items():
        print(f"   {bb:15s} | std = {std:.4f}")

    # Best metric per backbone
    print("\n2. Meilleure métrique par backbone:")
    for bb in df['modele_backbone'].unique():
        bb_data = df[df['modele_backbone'] == bb]
        best_metric = bb_data.groupby('metrique_distance')['monotonie_moyenne'].mean().idxmax()
        best_value = bb_data.groupby('metrique_distance')['monotonie_moyenne'].mean().max()
        print(f"   {bb:15s} | {best_metric:20s} = {best_value:.4f}")

    # Degradation difficulty ranking
    print("\n3. Difficultés des dégradations (monotonie moyenne décroissante):")
    mono_cols = [c for c in df.columns if c.startswith('mono_')]
    for col in mono_cols:
        mean_val = df[col].astype(float).mean()
        deg_name = col.replace('mono_', '').replace('_', '+')
        print(f"   {deg_name:20s} | mean = {mean_val:+.4f}")

if __name__ == "__main__":
    main()
