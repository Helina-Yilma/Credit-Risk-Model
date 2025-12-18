import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# --- 1. Custom Transformer for Aggregates ---
class FeatureAggregator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Check if we already have the features to avoid re-calculating and causing KeyErrors
        target_cols = ['Total_Amount', 'Average_Amount', 'Transaction_Count', 'Std_Dev_Amount']
        if all(col in X.columns for col in target_cols):
            return X 

        # If not present, calculate them
        agg = X.groupby('CustomerId')['Amount'].agg(
            Total_Amount='sum',
            Average_Amount='mean',
            Transaction_Count='count',
            Std_Dev_Amount='std'
        ).reset_index()
        
        X = X.merge(agg, on='CustomerId', how='left')
        X['Std_Dev_Amount'] = X['Std_Dev_Amount'].fillna(0)
        return X

# --- 2. Custom Transformer for Time Extraction ---
class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
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

# --- 3. Custom Transformer to Drop Constant/ID Columns ---
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Drop columns if they exist in the dataframe
        return X.drop(columns=[col for col in self.columns_to_drop if col in X.columns])

# --- 4. The Main Processing Function ---
def get_preprocessing_pipeline(num_cols, cat_cols):
    """
    Builds a robust sklearn Pipeline that handles strings and scaling.
    """
    
    # Numerical: Impute then Standardize
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical: Impute then One-Hot Encode (This fixes your String to Float error)
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