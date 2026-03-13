import unittest
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.load_data import load_raw_data
from src.preprocessing import clean_data, encode_data, get_numeric_features
from src.pca_analysis import apply_pca
from src.kmeans_clustering import perform_kmeans


class TestLoadData(unittest.TestCase):
    """Tests du module de chargement des données."""

    def setUp(self):
        self.raw_path = 'data/raw/StudentsPerformance.csv'

    def test_load_data_returns_dataframe(self):
        df = load_raw_data(self.raw_path)
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)

    def test_load_data_not_empty(self):
        df = load_raw_data(self.raw_path)
        self.assertFalse(df.empty)

    def test_load_data_expected_columns(self):
        df = load_raw_data(self.raw_path)
        expected = ['math score', 'reading score', 'writing score']
        for col in expected:
            self.assertIn(col, df.columns)

    def test_load_data_expected_rows(self):
        df = load_raw_data(self.raw_path)
        self.assertEqual(len(df), 1000)

    def test_load_data_invalid_path(self):
        df = load_raw_data('chemin/inexistant.csv')
        self.assertIsNone(df)


class TestPreprocessing(unittest.TestCase):
    """Tests du module de prétraitement."""

    def setUp(self):
        self.df_sample = pd.DataFrame({
            'math score':    [80, 80, 60, 90, 45],
            'reading score': [75, 75, 55, 88, 40],
            'writing score': [78, 78, 58, 85, 42],
            'gender':        ['male', 'male', 'female', 'female', 'male']
        })

    def test_clean_data_removes_duplicates(self):
        df_clean = clean_data(self.df_sample)
        self.assertEqual(len(df_clean), 4)  # 1 doublon supprimé

    def test_clean_data_handles_missing_values(self):
        df_with_nan = self.df_sample.copy()
        df_with_nan.loc[0, 'math score'] = np.nan
        df_clean = clean_data(df_with_nan)
        self.assertEqual(df_clean['math score'].isnull().sum(), 0)

    def test_get_numeric_features(self):
        df_num = get_numeric_features(self.df_sample)
        self.assertListEqual(list(df_num.columns), ['math score', 'reading score', 'writing score'])

    def test_encode_data_creates_dummies(self):
        # Avec drop_first=True et une colonne binaire (male/female), on obtient 1 colonne pour gender
        df_enc = encode_data(self.df_sample)
        # Le nb de colonnes doit être >= nb de colonnes numériques (les catégorielles sont encodées)
        self.assertIn('math score', df_enc.columns)
        self.assertTrue(all(df_enc.dtypes != 'object'))


class TestPCA(unittest.TestCase):
    """Tests du module PCA."""

    def setUp(self):
        df = load_raw_data('data/raw/StudentsPerformance.csv')
        df_clean = clean_data(df)
        self.df_numeric = get_numeric_features(df_clean)

    def test_pca_returns_correct_shapes(self):
        pca, scaler, results = apply_pca(self.df_numeric, n_components=2)
        self.assertEqual(results.shape[1], 2)
        self.assertEqual(results.shape[0], len(self.df_numeric))

    def test_pca_variance_high(self):
        """PC1+PC2 doivent conserver >= 95% de la variance sur les 3 scores."""
        pca, scaler, results = apply_pca(self.df_numeric, n_components=2)
        total_variance = sum(pca.explained_variance_ratio_)
        self.assertGreater(total_variance, 0.95,
                           f"Variance {total_variance:.4f} trop faible — attendu > 95%")

    def test_pca_variance_answer(self):
        """Réponse à la question d'évaluation : ~98.5%."""
        pca, scaler, results = apply_pca(self.df_numeric, n_components=2)
        total = sum(pca.explained_variance_ratio_) * 100
        self.assertAlmostEqual(total, 98.5, delta=0.5,
                               msg=f"Variance totale = {total:.2f}%, attendu ≈ 98.5%")

    def test_pca_scaler_fitted(self):
        pca, scaler, results = apply_pca(self.df_numeric, n_components=2)
        # Le scaler doit accepter les 3 colonnes numériques
        test_input = [[70, 75, 72]]
        transformed = scaler.transform(test_input)
        self.assertEqual(transformed.shape, (1, 3))


class TestKMeans(unittest.TestCase):
    """Tests du module K-Means."""

    def setUp(self):
        df = load_raw_data('data/raw/StudentsPerformance.csv')
        df_clean = clean_data(df)
        df_numeric = get_numeric_features(df_clean)
        _, _, self.pca_results = apply_pca(df_numeric, n_components=2)

    def test_kmeans_returns_correct_nb_clusters(self):
        kmeans, clusters, sil = perform_kmeans(self.pca_results, n_clusters=3)
        unique_clusters = set(clusters)
        self.assertEqual(len(unique_clusters), 3)

    def test_kmeans_clusters_correct_length(self):
        kmeans, clusters, sil = perform_kmeans(self.pca_results, n_clusters=3)
        self.assertEqual(len(clusters), len(self.pca_results))

    def test_kmeans_silhouette_positive(self):
        """Le score Silhouette doit être positif (clustering cohérent)."""
        kmeans, clusters, sil = perform_kmeans(self.pca_results, n_clusters=3)
        self.assertGreater(sil, 0, f"Silhouette négatif : {sil:.4f}")

    def test_kmeans_predict_new_point(self):
        """Le modèle doit prédire un cluster pour un nouveau point."""
        kmeans, clusters, sil = perform_kmeans(self.pca_results, n_clusters=3)
        new_point = np.array([[0.5, -0.2]])
        prediction = kmeans.predict(new_point)
        self.assertIn(prediction[0], [0, 1, 2])


if __name__ == '__main__':
    unittest.main(verbosity=2)