import matplotlib.pyplot as plt
import seaborn as sns

def plot_pca(pca_results, clusters=None, output_path='outputs/figures/pca_plot.png'):
    """
    Visualise les résultats de la PCA.
    """
    plt.figure(figsize=(10, 6))
    if clusters is not None:
        sns.scatterplot(x=pca_results[:, 0], y=pca_results[:, 1], hue=clusters, palette='viridis')
    else:
        sns.scatterplot(x=pca_results[:, 0], y=pca_results[:, 1])
    plt.title('PCA Results')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig(output_path)
    plt.close()
