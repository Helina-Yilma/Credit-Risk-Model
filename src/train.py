import pandas as pd
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from src.data_processing import (
    FeatureAggregator, 
    TimeFeatureExtractor, 
    ColumnDropper, 
    get_preprocessing_pipeline
)

def run_training():
    # 1. Load Data
    df = pd.read_csv('data/processed/labeled_data.csv')

    # 2. Apply Custom Transforms
    df = FeatureAggregator().transform(df)
    df = TimeFeatureExtractor().transform(df)

    # 3. Define Column Groups
    num_cols = ['Amount', 'Value', 'Total_Amount', 'Average_Amount', 'Transaction_Count']
    cat_cols = ['ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']
    drop_cols = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 'TransactionStartTime', 'CurrencyCode', 'CountryCode']

    # 4. Clean and Preprocess
    df_cleaned = ColumnDropper(drop_cols).transform(df)
    pipeline = get_preprocessing_pipeline(num_cols, cat_cols)

    # This is the step that turns strings into floats/ints
    X = pipeline.fit_transform(df_cleaned)
    y = df['is_high_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("Credit_Score_Analysis")

    with mlflow.start_run(run_name="XGBoost_Tuning_Search"):
        # 2. Hyperparameter Tuning (Random Search)
        param_dist = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        
        search = RandomizedSearchCV(XGBClassifier(), param_dist, n_iter=5, scoring='roc_auc', cv=3)
        search.fit(X_train, y_train)
        
        # 3. Log Best Model & Params
        best_model = search.best_estimator_
        mlflow.log_params(search.best_params_)
        
        auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
        mlflow.log_metric("roc_auc", auc)
        
        # Save the model
        mlflow.xgboost.log_model(best_model, "model")
        print(f"Model trained and logged with AUC: {auc}")

if __name__ == "__main__":
    run_training()