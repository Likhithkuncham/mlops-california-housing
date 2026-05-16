"""
Production-grade data validation pipeline for California Housing Dataset using Pandas.
This script performs rigorous data quality checks and generates a validation report.
"""

import os
import logging
import sys
import json
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd

# Constants
DATA_PATH = "data/raw/housing.csv"
REPORT_DIR = "artifacts/validation_reports"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "data_validation.log")

def setup_logging() -> logging.Logger:
    """Sets up logging configuration with UTF-8 encoding."""
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

class DataValidator:
    """Class to handle dataset validation using Pandas logic."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "success": True
        }

    def validate_not_empty(self) -> bool:
        """Checks if the dataset is not empty."""
        is_not_empty = len(self.df) > 0
        self.validation_results["checks"]["not_empty"] = {
            "success": is_not_empty,
            "details": f"Row count: {len(self.df)}"
        }
        if not is_not_empty:
            self.validation_results["success"] = False
        return is_not_empty

    def validate_columns_exist(self, required_columns: List[str]) -> bool:
        """Checks if all required columns are present."""
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        success = len(missing_columns) == 0
        self.validation_results["checks"]["columns_existence"] = {
            "success": success,
            "details": "All columns present" if success else f"Missing: {missing_columns}"
        }
        if not success:
            self.validation_results["success"] = False
        return success

    def validate_no_duplicates(self) -> bool:
        """Checks for duplicate rows."""
        duplicate_count = self.df.duplicated().sum()
        success = duplicate_count == 0
        self.validation_results["checks"]["no_duplicates"] = {
            "success": success,
            "details": f"Found {duplicate_count} duplicate rows"
        }
        if not success:
            self.validation_results["success"] = False
        return success

    def validate_no_nulls(self, column: str) -> bool:
        """Checks for null values in a specific column."""
        null_count = self.df[column].isnull().sum()
        success = null_count == 0
        self.validation_results["checks"][f"no_nulls_{column}"] = {
            "success": success,
            "details": f"Found {null_count} null values in {column}"
        }
        if not success:
            self.validation_results["success"] = False
        return success

    def validate_numerical_ranges(self, range_config: Dict[str, Dict[str, float]]) -> bool:
        """Checks if numerical columns are within specified ranges."""
        range_failures = {}
        overall_success = True

        for col, bounds in range_config.items():
            if col in self.df.columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                
                success = (min_val >= bounds["min"]) and (max_val <= bounds["max"])
                if not success:
                    overall_success = False
                    range_failures[col] = f"Actual range: [{min_val}, {max_val}], Expected: [{bounds['min']}, {bounds['max']}]"

        self.validation_results["checks"]["numerical_ranges"] = {
            "success": overall_success,
            "details": "All ranges valid" if overall_success else range_failures
        }
        if not overall_success:
            self.validation_results["success"] = False
        return overall_success

    def get_results(self) -> Dict[str, Any]:
        """Returns the final validation results."""
        return self.validation_results

def load_data(file_path: str) -> pd.DataFrame:
    """Loads dataset from CSV file."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found at {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded dataset: {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def save_report(results: Dict[str, Any]):
    """Saves validation summary report as JSON."""
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)
    
    # Ensure all values are JSON serializable (convert numpy types)
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        if hasattr(obj, 'item'): # numpy types
            return obj.item()
        return obj

    report_path = os.path.join(REPORT_DIR, "validation_summary.json")
    with open(report_path, "w", encoding='utf-8') as f:
        json.dump(make_serializable(results), f, indent=4)
    
    logger.info(f"Validation summary report saved to {report_path}")

def main():
    """Main execution function."""
    try:
        logger.info("Starting Pandas Data Validation Pipeline...")
        
        # Load data
        df = load_data(DATA_PATH)
        
        # Initialize Validator
        validator = DataValidator(df)
        
        # Define Configuration
        required_columns = [
            "longitude", "latitude", "housing_median_age", "total_rooms",
            "total_bedrooms", "population", "households", "median_income",
            "median_house_value", "ocean_proximity"
        ]
        
        range_config = {
            "longitude": {"min": -125, "max": -114},
            "latitude": {"min": 32, "max": 42},
            "housing_median_age": {"min": 1, "max": 52},
            "median_house_value": {"min": 1, "max": 500001}
        }
        
        # Run Validations
        validator.validate_not_empty()
        validator.validate_columns_exist(required_columns)
        validator.validate_no_duplicates()
        validator.validate_no_nulls("median_house_value")
        validator.validate_numerical_ranges(range_config)
        
        # Get and save results
        results = validator.get_results()
        save_report(results)
        
        if results["success"]:
            logger.info("DATA VALIDATION SUCCESSFUL!")
            print("\nData Validation passed successfully!")
        else:
            logger.warning("DATA VALIDATION FAILED!")
            print("\nData Validation failed! Check the report for details.")
            
        logger.info("Pipeline completed.")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
