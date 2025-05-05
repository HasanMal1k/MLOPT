import unittest
import pandas as pd
import numpy as np
from transformations import engineer_features

class TestTransformations(unittest.TestCase):
    def setUp(self):
        # Create a simple test DataFrame
        self.test_df = pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5],
            'category_col': ['A', 'B', 'A', 'C', 'B'],
            'date_col': pd.date_range(start='2023-01-01', periods=5),
            'skewed_col': [1, 1, 1, 1, 100]
        })
    
    def test_engineer_features(self):
        # Test the automatic feature engineering
        transformed_df, results = engineer_features(self.test_df)
        
        # Check that the original columns are preserved
        for col in self.test_df.columns:
            self.assertIn(col, transformed_df.columns)
        
        # Check that datetime features were created
        self.assertIn('date_col_year', transformed_df.columns)
        self.assertIn('date_col_month', transformed_df.columns)
        
        # Check that categorical encodings were created
        self.assertTrue(any(col.startswith('category_col_') for col in transformed_df.columns))
        
        # Check that numeric transformations were created for skewed column
        self.assertIn('skewed_col_log', transformed_df.columns)
        
        # Verify the results structure
        self.assertIn('datetime_features', results)
        self.assertIn('categorical_encodings', results)
        self.assertIn('numeric_transformations', results)
        
        # Print the columns to verify
        print("Original columns:", self.test_df.columns.tolist())
        print("Transformed columns:", transformed_df.columns.tolist())
        print("New columns count:", len(transformed_df.columns) - len(self.test_df.columns))

if __name__ == '__main__':
    unittest.main()