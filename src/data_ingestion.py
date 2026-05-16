"""
Data Ingestion and Splitting for California Housing Dataset.
Splits raw data into train and test sets.
"""

import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

# Constants
RAW_DATA_PATH = "data/raw/housing.csv"
PROCESSED_DIR = "data/processed"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "data_ingestion.log")

def setup_logging():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8'), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def ingest_data():
    try:
        if not os.path.exists(PROCESSED_DIR):
            os.makedirs(PROCESSED_DIR)
            
        logger.info(f"Loading raw data from {RAW_DATA_PATH}")
        df = pd.read_csv(RAW_DATA_PATH)
        
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        train_path = os.path.join(PROCESSED_DIR, "train.csv")
        test_path = os.path.join(PROCESSED_DIR, "test.csv")
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"Data split successful. Saved to {PROCESSED_DIR}")
        print("Data Ingestion and Splitting successful!")
        
    except Exception as e:
        logger.error(f"Error in data ingestion: {e}")
        raise

if __name__ == "__main__":
    ingest_data()
