"""
Production-grade machine learning training pipeline for California Housing Dataset using XGBoost and MLflow.
"""

import os
import logging
import sys
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.xgboost
import joblib

# Constants
TRAIN_DATA_PATH = "data/featured/train_featured.csv"
TEST_DATA_PATH = "data/featured/test_featured.csv"
MODEL_DIR = "models"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "model_training.log")

TARGET_COL = "median_house_value"
DROP_COLS = ["house_id", "event_timestamp"]

def setup_logging() -> logging.Logger:
    """Sets up logging configuration."""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def load_and_prepare_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Loads featured data and separates features from target."""
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Drop Feast-specific columns and target
        X_train = train_df.drop(columns=[TARGET_COL] + DROP_COLS)
        y_train = train_df[TARGET_COL]
        
        X_test = test_df.drop(columns=[TARGET_COL] + DROP_COLS)
        y_test = test_df[TARGET_COL]
        
        logger.info(f"Data loaded. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return X_train, y_train, X_test, y_test
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def eval_metrics(actual: np.ndarray, pred: np.ndarray) -> Tuple[float, float, float]:
    """Calculates RMSE, MAE, and R2 metrics."""
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train_and_log_model():
    """Trains the XGBoost model and logs metrics/artifacts to MLflow."""
    try:
        logger.info("Starting model training pipeline...")
        
        # Load data
        X_train, y_train, X_test, y_test = load_and_prepare_data(TRAIN_DATA_PATH, TEST_DATA_PATH)
        
        # Hyperparameters
        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "objective": "reg:squarederror",
            "random_state": 42
        }
        
        # MLflow Tracking
        mlflow.set_experiment("California_Housing_Regression")
        
        with mlflow.start_run() as run:
            logger.info(f"MLflow Run ID: {run.info.run_id}")
            
            # Initialize and train model
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            
            # Predictions
            predictions = model.predict(X_test)
            
            # Evaluate
            rmse, mae, r2 = eval_metrics(y_test, predictions)
            
            logger.info(f"Metrics: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.2f}")
            
            # Log params and metrics
            mlflow.log_params(params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            
            # Log model to MLflow
            mlflow.xgboost.log_model(
                model, 
                artifact_path="model",
                registered_model_name="CaliforniaHousingXGB"
            )
            
            # Save model locally
            if not os.path.exists(MODEL_DIR):
                os.makedirs(MODEL_DIR)
            
            model_path = os.path.join(MODEL_DIR, "xgboost_model.joblib")
            joblib.dump(model, model_path)
            logger.info(f"Model saved locally to {model_path}")
            
            print("\nModel Training and Tracking Successful!")
            print(f"RMSE: {rmse:.2f}")
            print(f"MAE: {mae:.2f}")
            print(f"R2: {r2:.2f}")
            
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    train_and_log_model()
