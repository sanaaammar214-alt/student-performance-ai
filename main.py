import os
import joblib
from src.load_data import load_raw_data
from src.preprocessing import clean_data, encode_data
from src.pca_analysis import apply_pca
from src.kmeans_clustering import perform_kmeans
from src.visualisation import plot_pca

def main():
    # 1. Chargement
    print("Chargement des données...")
    df = load_raw_data('data/raw/StudentsPerformance.csv')
    
    if df is not None:
        # 2. Preprocessing
        print("Prétraitement...")
        df_clean = clean_data(df)
        
        # Sauvegarde cleaned data
        os.makedirs('data/processed', exist_ok=True)
        df_clean.to_csv('data/processed/students_clean.csv', index=False)
        
        # Encodage pour ML
        df_encoded = encode_data(df_clean) 
        
        # 3. PCA
        print("Analyse PCA...")
        pca_model, scaler_model, pca_results = apply_pca(df_encoded)
        
        # 4. Clustering
        print("Clustering K-Means...")
        kmeans_model, clusters = perform_kmeans(pca_results)
        
        # Sauvegarde des modèles
        print("Sauvegarde des modèles...")
        os.makedirs('outputs/models', exist_ok=True)
        joblib.dump(pca_model, 'outputs/models/pca_model.pkl')
        joblib.dump(scaler_model, 'outputs/models/scaler_model.pkl')
        joblib.dump(kmeans_model, 'outputs/models/kmeans_model.pkl')
        
        # 5. Visualisation
        print("Génération des graphiques...")
        os.makedirs('outputs/figures', exist_ok=True)
        plot_pca(pca_results, clusters)
        
        print("Pipeline terminé avec succès.")

if __name__ == "__main__":
    main()
