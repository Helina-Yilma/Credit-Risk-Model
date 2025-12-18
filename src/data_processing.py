import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xverse.transformer import WOE

# --- 1. Custom Transformer for Aggregates ---
class FeatureAggregator(BaseEstimator, TransformerMixin):
    """Calculates Total, Average, Count, and Std Dev per Customer."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Group by CustomerId for aggregations
        agg = X.groupby('CustomerId')['Amount'].agg(['sum', 'mean', 'count', 'std']).reset_index()
        agg.columns = ['CustomerId', 'Total_Amount', 'Average_Amount', 'Transaction_Count', 'Std_Dev_Amount']
        
        # Merge back to original data
        X = X.merge(agg, on='CustomerId', how='left')
        X['Std_Dev_Amount'] = X['Std_Dev_Amount'].fillna(0) # Handle single-transaction customers
        return X

# --- 2. Custom Transformer for Time Extraction ---
class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts Hour, Day, Month, and Year from TransactionStartTime."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        X['TransactionHour'] = X['TransactionStartTime'].dt.hour
        X['TransactionDay'] = X['TransactionStartTime'].dt.day
        X['TransactionMonth'] = X['TransactionStartTime'].dt.month
        X['TransactionYear'] = X['TransactionStartTime'].dt.year
        return X

# --- 3. The Main Processing Function ---
def get_preprocessing_pipeline(num_cols, cat_cols):
    """
    Builds a robust sklearn Pipeline.
    Note: num_cols should include the new aggregate feature names.
    """
    
    # Numerical: Impute missing then Standardize
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical: Impute missing then One-Hot Encode
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )
    
    return preprocessor