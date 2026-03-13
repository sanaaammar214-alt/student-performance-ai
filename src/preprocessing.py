import pandas as pd

def clean_data(df):
    """
    Nettoyage des données :
    - Suppression des doublons
    - Gestion des valeurs manquantes (remplacement par la médiane pour les numériques,
      le mode pour les catégorielles)
    - Vérification et correction des types
    """
    df_clean = df.copy()

    # Rapport avant nettoyage
    n_before = len(df_clean)
    n_dup = df_clean.duplicated().sum()
    n_missing = df_clean.isnull().sum().sum()

    print(f"  [Preprocessing] Lignes avant : {n_before}")
    print(f"  [Preprocessing] Doublons détectés : {n_dup}")
    print(f"  [Preprocessing] Valeurs manquantes : {n_missing}")

    # Suppression des doublons
    df_clean = df_clean.drop_duplicates()

    # Gestion des valeurs manquantes
    for col in df_clean.columns:
        if df_clean[col].isnull().any():
            if df_clean[col].dtype in ['float64', 'int64']:
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                print(f"  [Preprocessing] '{col}' : {df_clean[col].isnull().sum()} NaN remplacés par médiane ({median_val})")
            else:
                mode_val = df_clean[col].mode()[0]
                df_clean[col].fillna(mode_val, inplace=True)
                print(f"  [Preprocessing] '{col}' : NaN remplacés par mode ('{mode_val}')")

    # Vérification des scores (doivent être entre 0 et 100)
    score_cols = ['math score', 'reading score', 'writing score']
    for col in score_cols:
        if col in df_clean.columns:
            invalid = df_clean[(df_clean[col] < 0) | (df_clean[col] > 100)]
            if len(invalid) > 0:
                print(f"  [Preprocessing] '{col}' : {len(invalid)} valeurs hors plage [0-100] supprimées")
                df_clean = df_clean[(df_clean[col] >= 0) & (df_clean[col] <= 100)]

    print(f"  [Preprocessing] Lignes après nettoyage : {len(df_clean)}")
    return df_clean


def encode_data(df):
    """
    Encodage des variables catégorielles avec pd.get_dummies (one-hot encoding).
    drop_first=True pour éviter la multicolinéarité.
    """
    df_encoded = pd.get_dummies(df, drop_first=True)
    print(f"  [Preprocessing] Colonnes après encodage : {df_encoded.shape[1]}")
    return df_encoded


def get_numeric_features(df):
    """
    Retourne uniquement les colonnes numériques (scores) pour la PCA.
    Correspond aux features : math score, reading score, writing score.
    """
    return df.select_dtypes(include=['number'])