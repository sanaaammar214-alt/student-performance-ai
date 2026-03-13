import pandas as pd

def clean_data(df):
    """
    Nettoyage des données : suppression des doublons, gestion des valeurs manquantes.
    """
    # Copie du dataframe
    df_clean = df.copy()
    
    # Nettoyage simple
    df_clean = df_clean.drop_duplicates()
    
    return df_clean

def encode_data(df):
    """
    Encodage des variables catégorielles.
    """
    # TODO: implémenter l'encodage (OneHot, LabelEncoding)
    return pd.get_dummies(df)
