from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

def apply_pca(df, n_components=2):
    """
    Applique une Analyse en Composantes Principales.
    """
    # Standardisation
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    # PCA
    pca = PCA(n_components=n_components)
    pca_results = pca.fit_transform(scaled_data)
    
    return pca, scaler, pca_results
