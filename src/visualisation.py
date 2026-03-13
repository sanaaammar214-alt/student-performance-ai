import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os

# Palette cohérente avec 3 clusters
CLUSTER_COLORS = ['#534AB7', '#0F6E56', '#993C1D']
CLUSTER_NAMES  = ['Profil Équilibré', 'Excellence Académique', "Besoin d'Accompagnement"]


def plot_pca(pca_results, clusters=None, pca_model=None,
             output_path='outputs/figures/pca_plot.png'):
    """
    Visualise les résultats PCA avec clusters colorés et axes labellisés
    par leur % de variance expliquée.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))

    if clusters is not None:
        for cid, (color, name) in enumerate(zip(CLUSTER_COLORS, CLUSTER_NAMES)):
            mask = clusters == cid
            ax.scatter(pca_results[mask, 0], pca_results[mask, 1],
                       c=color, label=f'Cluster {cid} — {name}',
                       alpha=0.7, s=60, edgecolors='white', linewidths=0.4)
        ax.legend(fontsize=11, framealpha=0.9)
    else:
        ax.scatter(pca_results[:, 0], pca_results[:, 1],
                   alpha=0.6, s=60, color='#185FA5')

    # Axes avec variance expliquée
    if pca_model is not None:
        var = pca_model.explained_variance_ratio_
        ax.set_xlabel(f'PC1 — Performance globale ({var[0]*100:.1f}% variance)', fontsize=12)
        ax.set_ylabel(f'PC2 — Nuance du profil ({var[1]*100:.1f}% variance)', fontsize=12)
        ax.set_title(
            f'Projection PCA des étudiants (PC1+PC2 = {sum(var)*100:.1f}% de variance conservée)',
            fontsize=13, fontweight='bold'
        )
    else:
        ax.set_xlabel('PC1', fontsize=12)
        ax.set_ylabel('PC2', fontsize=12)
        ax.set_title('Projection PCA des étudiants', fontsize=13)

    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Visualisation] PCA plot sauvegardé → {output_path}")


def plot_elbow(inertias, silhouette_scores, k_range=range(2, 9),
               output_path='outputs/figures/elbow_plot.png'):
    """
    Graphique Elbow Method + Score Silhouette pour justifier le choix de k.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ks = list(k_range)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Elbow
    ax1.plot(ks, inertias, 'o-', color='#185FA5', linewidth=2, markersize=8)
    ax1.axvline(x=3, color='#993C1D', linestyle='--', linewidth=1.5, label='k=3 choisi')
    ax1.set_xlabel('Nombre de clusters k', fontsize=12)
    ax1.set_ylabel('Inertie (WCSS)', fontsize=12)
    ax1.set_title("Méthode du coude (Elbow Method)", fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Silhouette
    colors_bar = ['#639922' if s == max(silhouette_scores) else '#B4B2A9' for s in silhouette_scores]
    bars = ax2.bar(ks, silhouette_scores, color=colors_bar, edgecolor='white', linewidth=0.5)
    ax2.axvline(x=3, color='#993C1D', linestyle='--', linewidth=1.5, label='k=3 choisi')
    ax2.set_xlabel('Nombre de clusters k', fontsize=12)
    ax2.set_ylabel('Score Silhouette', fontsize=12)
    ax2.set_title("Score Silhouette par k", fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    # Annoter le max
    best_idx = silhouette_scores.index(max(silhouette_scores))
    ax2.annotate(f'{max(silhouette_scores):.3f}',
                 xy=(ks[best_idx], max(silhouette_scores)),
                 xytext=(0, 8), textcoords='offset points',
                 ha='center', fontsize=11, fontweight='bold', color='#3B6D11')

    plt.suptitle("Justification du choix de k pour K-Means", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Visualisation] Elbow + Silhouette sauvegardé → {output_path}")


def plot_variance_explained(pca_model, output_path='outputs/figures/pca_variance.png'):
    """
    Graphique de la variance expliquée par chaque composante PCA.
    Répond directement à la question d'évaluation du sujet.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    var = pca_model.explained_variance_ratio_
    n = len(var)
    ks = [f'PC{i+1}' for i in range(n)]
    cumulative = np.cumsum(var)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(ks, var * 100, color=['#185FA5', '#0F6E56'][:n],
                  edgecolor='white', linewidth=0.5, width=0.5)
    ax.plot(ks, cumulative * 100, 'o-', color='#993C1D', linewidth=2,
            markersize=9, label='Variance cumulée', zorder=5)

    # Annotations sur les barres
    for i, (bar, v) in enumerate(zip(bars, var)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{v*100:.2f}%', ha='center', va='bottom',
                fontsize=12, fontweight='bold', color='#0C447C')

    # Annotation variance cumulée
    ax.annotate(f'Total : {cumulative[-1]*100:.2f}%',
                xy=(ks[-1], cumulative[-1]*100),
                xytext=(20, -15), textcoords='offset points',
                fontsize=11, color='#993C1D', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#993C1D'))

    ax.set_ylim(0, 115)
    ax.set_xlabel('Composantes Principales', fontsize=12)
    ax.set_ylabel('Variance Expliquée (%)', fontsize=12)
    ax.set_title('Variance expliquée par les composantes PCA\n(Réponse à la question d\'évaluation)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Visualisation] Variance PCA sauvegardé → {output_path}")


def plot_cluster_profiles(df_numeric, clusters, output_path='outputs/figures/cluster_profiles.png'):
    """
    Profil moyen de chaque cluster (radar chart simplifié en barres groupées).
    """
    import pandas as pd
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_plot = df_numeric.copy()
    df_plot['Cluster'] = clusters

    means = df_plot.groupby('Cluster').mean()
    features = means.columns.tolist()
    x = np.arange(len(features))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (cid, color, name) in enumerate(zip(range(3), CLUSTER_COLORS, CLUSTER_NAMES)):
        if cid in means.index:
            ax.bar(x + i*width, means.loc[cid], width,
                   label=f'Cluster {cid} — {name}',
                   color=color, alpha=0.85, edgecolor='white')

    ax.set_xticks(x + width)
    ax.set_xticklabels(['Math', 'Lecture', 'Écriture'], fontsize=12)
    ax.set_ylabel('Score moyen (/100)', fontsize=12)
    ax.set_title('Profil académique moyen par cluster', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Visualisation] Profils cluster sauvegardé → {output_path}")