import os
import joblib
import warnings

# Ignorer les warnings de noms de features scikit-learn
warnings.filterwarnings("ignore", category=UserWarning)

from src.load_data import load_raw_data
from src.preprocessing import clean_data, encode_data, get_numeric_features
from src.pca_analysis import apply_pca
from src.kmeans_clustering import find_optimal_k, perform_kmeans
from src.visualisation import plot_pca, plot_elbow, plot_variance_explained, plot_cluster_profiles


def main():
    print("=" * 55)
    print("  STUDENT PERFORMANCE AI — Pipeline complet")
    print("=" * 55)

    # 1. Chargement
    print("\n[1/6] Chargement des données...")
    df = load_raw_data('data/raw/StudentsPerformance.csv')
    if df is None:
        print("ERREUR : fichier introuvable.")
        return

    print(f"  Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"  Colonnes : {list(df.columns)}")

    # 2. Preprocessing
    print("\n[2/6] Prétraitement des données...")
    df_clean = clean_data(df)

    # Sauvegarde données nettoyées
    os.makedirs('data/processed', exist_ok=True)
    df_clean.to_csv('data/processed/students_clean.csv', index=False)
    print(f"  Données nettoyées sauvegardées → data/processed/students_clean.csv")

    # Extraction des features numériques uniquement (math, reading, writing)
    # C'est ces 3 colonnes qui seront utilisées dans l'app Streamlit
    df_numeric = get_numeric_features(df_clean)
    print(f"  Features numériques retenues : {list(df_numeric.columns)}")

    # 3. PCA — sur les scores numériques uniquement
    print("\n[3/6] Analyse PCA (sur features numériques)...")
    pca_model, scaler_model, pca_results = apply_pca(df_numeric, n_components=2)

    # Réponse à la question d'évaluation du sujet
    var = pca_model.explained_variance_ratio_
    print(f"\n  *** RÉPONSE QUESTION D'ÉVALUATION ***")
    print(f"  PC1 conserve {var[0]*100:.2f}% de la variance")
    print(f"  PC2 conserve {var[1]*100:.2f}% de la variance")
    print(f"  PC1 + PC2 conservent {sum(var)*100:.2f}% de la variance totale")
    print(f"  *** FIN RÉPONSE ***\n")

    # 4. Choix de k optimal par Elbow + Silhouette
    print("\n[4/6] Recherche du k optimal (Elbow + Silhouette)...")
    inertias, silhouette_scores, optimal_k = find_optimal_k(pca_results, k_range=range(2, 9))

    # Clustering K-Means avec k=3 (validé par l'analyse)
    print(f"\n  Clustering avec k=3 (justifié par Elbow + Silhouette)...")
    kmeans_model, clusters, silhouette = perform_kmeans(pca_results, n_clusters=3)

    # Ajout des clusters au dataframe propre
    df_clean['cluster'] = clusters
    df_clean.to_csv('data/processed/students_clustered.csv', index=False)
    print(f"  Données avec clusters sauvegardées → data/processed/students_clustered.csv")

    # 5. Sauvegarde des modèles
    print("\n[5/6] Sauvegarde des modèles...")
    os.makedirs('outputs/models', exist_ok=True)
    joblib.dump(pca_model,    'outputs/models/pca_model.pkl')
    joblib.dump(scaler_model, 'outputs/models/scaler_model.pkl')
    joblib.dump(kmeans_model, 'outputs/models/kmeans_model.pkl')
    # Sauvegarde du score silhouette réel pour l'app
    joblib.dump({'silhouette': silhouette, 'variance_pc1': var[0], 'variance_pc2': var[1]},
                'outputs/models/metrics.pkl')
    print("  Modèles sauvegardés : pca_model.pkl, scaler_model.pkl, kmeans_model.pkl, metrics.pkl")

    # 6. Visualisations
    print("\n[6/6] Génération des graphiques...")
    os.makedirs('outputs/figures', exist_ok=True)
    plot_variance_explained(pca_model)
    plot_elbow(inertias, silhouette_scores)
    plot_pca(pca_results, clusters, pca_model)
    plot_cluster_profiles(df_numeric, clusters)

    print("\n" + "=" * 55)
    print("  Pipeline terminé avec succès.")
    print(f"  Score Silhouette final : {silhouette:.4f}")
    print(f"  Variance conservée (PC1+PC2) : {sum(var)*100:.2f}%")
    print("=" * 55)


if __name__ == "__main__":
    main()