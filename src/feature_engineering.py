"""
Production-grade feature engineering pipeline for California Housing Dataset.
Includes custom feature creation, imputation, scaling, and artifact persistence.
"""

import os
import logging
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from typing import Tuple, Dict, Any
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Constants
TRAIN_DATA_PATH = "data/processed/train.csv"
TEST_DATA_PATH = "data/processed/test.csv"
FEATURED_DATA_DIR = "data/featured"
PREPROCESSOR_DIR = "artifacts/preprocessor"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "feature_engineering.log")

TARGET_COL = "median_house_value"

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

def load_datasets(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads train and test datasets."""
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        logger.info(f"Loaded train set ({len(train_df)} rows) and test set ({len(test_df)} rows).")
        return train_df, test_df
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        raise

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates new features based on domain knowledge."""
    logger.info("Creating engineered features...")
    df = df.copy()
    
    # avoid division by zero errors by adding a small epsilon if households/total_rooms is 0
    # though in this dataset they should be > 0
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]
    
    return df

def get_preprocessing_pipeline(numerical_cols: list) -> Pipeline:
    """Defines the preprocessing pipeline (imputation + scaling)."""
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols)
        ],
        remainder='drop' # Drop categorical 'ocean_proximity' unless encoding requested
    )
    
    return preprocessor

def run_feature_engineering():
    """Main function to run the feature engineering pipeline."""
    try:
        logger.info("Starting Feature Engineering Pipeline...")
        
        # 1. Load Data
        train_df, test_df = load_datasets(TRAIN_DATA_PATH, TEST_DATA_PATH)
        
        # 2. Add Engineered Features
        train_df = add_engineered_features(train_df)
        test_df = add_engineered_features(test_df)
        
        # 3. Separate features and target
        X_train = train_df.drop(columns=[TARGET_COL])
        y_train = train_df[TARGET_COL]
        X_test = test_df.drop(columns=[TARGET_COL])
        y_test = test_df[TARGET_COL]
        
        # Identify numerical columns for scaling
        # (Exclude ocean_proximity if it's there)
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        logger.info(f"Numerical columns for transformation: {numerical_cols}")
        
        # 4. Create and fit pipeline
        preprocessor = get_preprocessing_pipeline(numerical_cols)
        
        logger.info("Fitting and transforming data...")
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        # Convert back to DataFrame for saving
        # Get column names after transformation (ColumnTransformer might change order or drop columns)
        # In this case, we dropped 'ocean_proximity', so we use numerical_cols names
        feature_names = numerical_cols
        
        train_featured = pd.DataFrame(X_train_transformed, columns=feature_names)
        train_featured[TARGET_COL] = y_train.values
        
        test_featured = pd.DataFrame(X_test_transformed, columns=feature_names)
        test_featured[TARGET_COL] = y_test.values

        # Add house_id and event_timestamp for Feast
        train_featured["house_id"] = range(len(train_featured))
        test_featured["house_id"] = range(len(train_featured), len(train_featured) + len(test_featured))
        
        current_time = datetime.now()
        train_featured["event_timestamp"] = current_time
        test_featured["event_timestamp"] = current_time
        
        # 5. Save Artifacts
        if not os.path.exists(FEATURED_DATA_DIR):
            os.makedirs(FEATURED_DATA_DIR)
        
        train_featured.to_csv(os.path.join(FEATURED_DATA_DIR, "train_featured.csv"), index=False)
        test_featured.to_csv(os.path.join(FEATURED_DATA_DIR, "test_featured.csv"), index=False)
        
        # Also save as Parquet for Feast (preferred)
        train_featured.to_parquet(os.path.join(FEATURED_DATA_DIR, "train_featured.parquet"), index=False)
        
        logger.info(f"Transformed datasets saved to {FEATURED_DATA_DIR}")
        
        if not os.path.exists(PREPROCESSOR_DIR):
            os.makedirs(PREPROCESSOR_DIR)
            
        joblib.dump(preprocessor, os.path.join(PREPROCESSOR_DIR, "preprocessor.joblib"))
        logger.info(f"Preprocessing pipeline saved to {PREPROCESSOR_DIR}")
        
        print("\nFeature Engineering Pipeline executed successfully!")
        print(f"Transformed train shape: {train_featured.shape}")
        print(f"Transformed test shape: {test_featured.shape}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_feature_engineering()
