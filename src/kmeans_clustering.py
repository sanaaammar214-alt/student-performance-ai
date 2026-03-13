from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

def find_optimal_k(data, k_range=range(2, 9)):
    """
    Méthode du coude (Elbow Method) + Score Silhouette pour déterminer le k optimal.
    
    Retourne :
    - inertias : liste des inerties pour chaque k
    - silhouette_scores : liste des scores silhouette pour chaque k
    - optimal_k : k recommandé selon le score silhouette
    """
    inertias = []
    silhouette_scores = []

    print("  [K-Means] Recherche du k optimal :")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(data)
        inertias.append(km.inertia_)
        sil = silhouette_score(data, labels)
        silhouette_scores.append(sil)
        print(f"    k={k} | Inertie={km.inertia_:.1f} | Silhouette={sil:.4f}")

    # k optimal = celui avec le meilleur score silhouette
    optimal_k = list(k_range)[np.argmax(silhouette_scores)]
    print(f"  [K-Means] k optimal recommandé : {optimal_k} (silhouette={max(silhouette_scores):.4f})")

    return inertias, silhouette_scores, optimal_k


def perform_kmeans(data, n_clusters=3):
    """
    Exécute l'algorithme K-Means avec k=3 (justifié par l'Elbow Method et le score silhouette).
    
    Paramètres :
    - data : array PCA (résultats de apply_pca)
    - n_clusters : nombre de clusters (défaut=3, validé par find_optimal_k)
    
    Retourne :
    - kmeans : modèle KMeans entraîné
    - clusters : labels de cluster pour chaque observation
    - sil_score : score silhouette du clustering final
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(data)

    sil_score = silhouette_score(data, clusters)
    print(f"  [K-Means] k={n_clusters} | Inertie finale : {kmeans.inertia_:.2f}")
    print(f"  [K-Means] Score Silhouette final : {sil_score:.4f} (plus proche de 1 = meilleur)")

    # Distribution des clusters
    unique, counts = np.unique(clusters, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"    Cluster {cluster_id} : {count} étudiants ({count/len(clusters)*100:.1f}%)")

    return kmeans, clusters, sil_score