"""
Modern production-grade drift detection pipeline using Evidently AI for California Housing Dataset.
Compares reference (train) and current (test) datasets to identify data and feature drift.
"""

import os
import logging
import sys
import json
from typing import Tuple
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# Constants
REF_DATA_PATH = "data/featured/train_featured.csv"
CURR_DATA_PATH = "data/featured/test_featured.csv"
REPORT_DIR = "monitoring/reports"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "drift_detection.log")

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

def load_data(ref_path: str, curr_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads reference and current datasets."""
    try:
        ref_df = pd.read_csv(ref_path)
        curr_df = pd.read_csv(curr_path)
        
        # Clean data for drift analysis (remove non-feature columns)
        ref_df = ref_df.drop(columns=DROP_COLS)
        curr_df = curr_df.drop(columns=DROP_COLS)
        
        logger.info(f"Loaded reference ({len(ref_df)} rows) and current ({len(curr_df)} rows) datasets.")
        return ref_df, curr_df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def run_drift_analysis(ref_df: pd.DataFrame, curr_df: pd.DataFrame):
    """Generates modern Evidently drift reports (HTML and JSON)."""
    try:
        logger.info("Initializing modern Evidently Drift Report...")
        
        # Define the report with DataDriftPreset (handles both data and feature drift)
        report = Report(metrics=[
            DataDriftPreset()
        ])
        
        logger.info("Running drift analysis...")
        snapshot = report.run(reference_data=ref_df, current_data=curr_df)
        
        # Ensure report directory exists
        if not os.path.exists(REPORT_DIR):
            os.makedirs(REPORT_DIR)
            
        # 1. Save HTML report
        html_path = os.path.join(REPORT_DIR, "drift_report.html")
        snapshot.save_html(html_path)
        logger.info(f"HTML drift report saved to {html_path}")
        
        # 2. Save JSON report
        json_path = os.path.join(REPORT_DIR, "drift_report.json")
        snapshot.save_json(json_path)
        logger.info(f"JSON drift report saved to {json_path}")
        
        # Extract summary from the metrics
        result_dict = snapshot.dict()
        
        # In Evidently 0.7.x, DataDriftPreset expands into several metrics.
        # We look for the summary metric 'DriftedColumnsCount'
        try:
            drift_metric = next(m for m in result_dict['metrics'] if "DriftedColumnsCount" in m['metric_name'])
            n_drifted = drift_metric['value']['count']
            drift_share = drift_metric['value']['share']
            threshold = drift_metric['config'].get('drift_share', 0.5)
            dataset_drift = drift_share > threshold
            
            # Count total features analyzed
            n_total = sum(1 for m in result_dict['metrics'] if "ValueDrift" in m['metric_name'])
        except (StopIteration, KeyError) as e:
            logger.warning(f"Could not extract detailed drift metrics: {e}")
            n_drifted, drift_share, dataset_drift, n_total = 0, 0, False, 0

        print("\n" + "="*40)
        print("MODERN DRIFT ANALYSIS SUMMARY")
        print("="*40)
        print(f"Dataset Drift Status: {'DRIFT DETECTED' if dataset_drift else 'NO DRIFT'}")
        print(f"Drifted Columns: {int(n_drifted)} / {n_total}")
        print(f"Share of Drifted Columns: {drift_share:.2%}")
        print("="*40)
        
        if dataset_drift:
            logger.warning("Significant data drift detected between reference and current datasets.")
        else:
            logger.info("No significant data drift detected.")
            
    except Exception as e:
        logger.error(f"Error during drift analysis: {e}")
        raise

def main():
    """Main execution function."""
    try:
        logger.info("Starting Modern Drift Detection Pipeline...")
        
        # Load Data
        ref_df, curr_df = load_data(REF_DATA_PATH, CURR_DATA_PATH)
        
        # Run Analysis
        run_drift_analysis(ref_df, curr_df)
        
        logger.info("Drift Detection Pipeline completed successfully.")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
