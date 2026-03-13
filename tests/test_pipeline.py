import unittest
import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.load_data import load_raw_data
from src.preprocessing import clean_data

class TestPipeline(unittest.TestCase):
    def setUp(self):
        # Path to dummy data created in main
        self.raw_path = 'projet_notes_etudiants/data/raw/StudentsPerformance.csv'
        # Adjust path if running from root or tests/
        if not os.path.exists(self.raw_path):
             self.raw_path = 'data/raw/StudentsPerformance.csv'

    def test_load_data(self):
        df = load_raw_data(self.raw_path)
        self.assertIsNotNone(df)
        self.assertFalse(df.empty)
        self.assertIn('math score', df.columns)

    def test_clean_data(self):
        df = pd.DataFrame({'a': [1, 1, 2], 'b': [3, 3, 4]})
        df_clean = clean_data(df)
        self.assertEqual(len(df_clean), 2)

if __name__ == '__main__':
    unittest.main()
