"""
ClickHouse HTTP client utility for production-grade data ingestion.
Uses zero-dependency HTTP interface to insert Pandas DataFrames seamlessly.
"""

import os
import logging
import requests
import pandas as pd

logger = logging.getLogger(__name__)

# Fetch environment variables with production fallbacks
CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST", "localhost")
CLICKHOUSE_PORT = os.getenv("CLICKHOUSE_PORT", "8123")
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "default")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "clickhouse_password")

BASE_URL = f"http://{CLICKHOUSE_HOST}:{CLICKHOUSE_PORT}/"

def execute_query(query: str, data: str = None) -> bool:
    """Executes a SQL query or inserts data into ClickHouse via HTTP interface."""
    try:
        params = {"query": query}
        # Setup authentication
        auth = (CLICKHOUSE_USER, CLICKHOUSE_PASSWORD) if CLICKHOUSE_PASSWORD else None
        
        response = requests.post(BASE_URL, params=params, data=data, auth=auth, timeout=30)
        
        if response.status_code == 200:
            return True
        else:
            logger.error(f"ClickHouse HTTP Query failed with code {response.status_code}: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error connecting to ClickHouse: {e}")
        return False

def save_dataframe_to_clickhouse(df: pd.DataFrame, table_name: str, create_table_query: str = None) -> bool:
    """Creates a table and inserts a Pandas DataFrame into ClickHouse."""
    if df.empty:
        logger.warning(f"DataFrame for table {table_name} is empty. Skipping save.")
        return False
        
    logger.info(f"Saving {len(df)} rows to ClickHouse table: {table_name}...")
    
    # 1. Create table if query provided
    if create_table_query:
        if not execute_query(create_table_query):
            logger.error(f"Failed to create table {table_name}.")
            return False

    # 2. Format DataFrame to JSONEachRow (newline-delimited JSON)
    # ClickHouse parses this format natively and extremely fast!
    try:
        # Convert timestamp columns to ISO strings to avoid encoding issues
        df_copy = df.copy()
        for col in df_copy.columns:
            if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                
        json_data = df_copy.to_json(orient='records', lines=True)
        
        # 3. Perform HTTP Bulk Insert
        insert_query = f"INSERT INTO {table_name} FORMAT JSONEachRow"
        success = execute_query(insert_query, data=json_data)
        
        if success:
            logger.info(f"Successfully saved all records to {table_name} in ClickHouse.")
        return success
    except Exception as e:
        logger.error(f"Failed to format or insert DataFrame to {table_name}: {e}")
        return False
