from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

def apply_pca(df, n_components=2):
    """
    Applique une Analyse en Composantes Principales (PCA) sur les données numériques.
    
    Paramètres :
    - df : DataFrame contenant UNIQUEMENT les colonnes numériques (math, reading, writing scores)
    - n_components : nombre de composantes à conserver (défaut=2)
    
    Retourne :
    - pca : modèle PCA entraîné
    - scaler : modèle StandardScaler entraîné
    - pca_results : array des données projetées dans l'espace PCA
    """
    # Standardisation (moyenne=0, écart-type=1) — obligatoire avant PCA
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Application PCA
    pca = PCA(n_components=n_components)
    pca_results = pca.fit_transform(scaled_data)

    # Rapport détaillé de la variance expliquée
    var_ratios = pca.explained_variance_ratio_
    cumulative = np.cumsum(var_ratios)

    print(f"  [PCA] Variance expliquée par composante :")
    for i, (r, c) in enumerate(zip(var_ratios, cumulative)):
        print(f"    PC{i+1} : {r*100:.2f}%  (cumulé : {c*100:.2f}%)")
    print(f"  [PCA] Variance totale conservée (PC1+PC2) : {cumulative[-1]*100:.2f}%")

    return pca, scaler, pca_results