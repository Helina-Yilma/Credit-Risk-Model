import pytest
import pandas as pd
from src.data_processing import TimeFeatureExtractor

def test_time_feature_columns():
    df = pd.DataFrame({'TransactionStartTime': ['2018-11-15T02:18:49Z']})
    extractor = TimeFeatureExtractor()
    transformed = extractor.transform(df)
    assert 'TransactionHour' in transformed.columns
    assert transformed['TransactionHour'].iloc[0] == 2