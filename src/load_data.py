import pandas as pd

def load_raw_data(path='data/raw/StudentsPerformance.csv'):
    """
    Charge le dataset brut StudentsPerformance.csv.
    """
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print(f"Fichier non trouvé à {path}")
        return None
