"""
Production-grade FastAPI application for California Housing price prediction.
Includes Pydantic validation, Prometheus metrics, and custom logging middleware.
"""

import os
import time
import logging
from typing import Dict, Any
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

# Constants
MODEL_PATH = "models/xgboost_model.joblib"
PREPROCESSOR_PATH = "artifacts/preprocessor/preprocessor.joblib"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "api.log")

# Setup Logging
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load Model and Preprocessor
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"Preprocessor file not found at {PREPROCESSOR_PATH}")
        
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    logger.info("Model and Preprocessor loaded successfully.")
except Exception as e:
    logger.error(f"Error loading assets: {e}")
    # In a real production app, we might want to exit here
    model = None
    preprocessor = None

# FastAPI App
app = FastAPI(
    title="California Housing Price Prediction API",
    description="MLOps Pipeline API for predicting house prices.",
    version="1.0.0"
)

# Prometheus Metrics
Instrumentator().instrument(app).expose(app)

# Request Schema
class HouseFeatures(BaseModel):
    """Input features for house price prediction."""
    longitude: float = Field(..., json_schema_extra={"example": -122.23})
    latitude: float = Field(..., example=37.88)
    housing_median_age: float = Field(..., example=41.0)
    total_rooms: float = Field(..., example=880.0)
    total_bedrooms: float = Field(..., example=129.0)
    population: float = Field(..., example=322.0)
    households: float = Field(..., example=126.0)
    median_income: float = Field(..., example=8.3252)
    ocean_proximity: str = Field(..., example="NEAR BAY")

# Middleware for Logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(f"Method: {request.method} Path: {request.url.path} Status: {response.status_code} Duration: {duration:.4f}s")
    return response

# Utility for Feature Engineering
def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Replicates feature engineering from training pipeline."""
    df = data.copy()
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]
    return df

@app.get("/health")
def health_check():
    """Returns the health status of the API."""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model or Preprocessor not loaded.")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
async def predict(features: HouseFeatures):
    """Predicts house price based on input features."""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    try:
        # 1. Convert input to DataFrame
        input_dict = features.dict()
        input_df = pd.DataFrame([input_dict])
        
        # 2. Apply Feature Engineering
        input_df = engineer_features(input_df)
        
        # 3. Preprocess (Scaling + Imputation)
        # We need to make sure we pass only the columns expected by the preprocessor
        # The preprocessor was fit on: ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 
        # 'total_bedrooms', 'population', 'households', 'median_income', 
        # 'rooms_per_household', 'bedrooms_per_room', 'population_per_household']
        
        X_transformed = preprocessor.transform(input_df)
        
        # 4. Predict
        prediction = model.predict(X_transformed)
        
        result = {
            "prediction": float(prediction[0]),
            "unit": "USD",
            "model_version": "1.0.0"
        }
        
        logger.info(f"Prediction successful: {result['prediction']}")
        return result
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Use 0.0.0.0 for containerization compatibility
    uvicorn.run(app, host="0.0.0.0", port=8000)
