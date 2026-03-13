from sklearn.cluster import KMeans
import pandas as pd

def perform_kmeans(data, n_clusters=3):
    """
    Exécute l'algorithme K-Means.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    return kmeans, clusters
