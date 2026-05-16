import pandas as pd
import pytest
from src.feature_engineering import add_engineered_features

def test_feature_creation():
    """Test that custom features are created correctly."""
    data = {
        "total_rooms": [100, 200],
        "total_bedrooms": [20, 50],
        "population": [500, 1000],
        "households": [50, 100]
    }
    df = pd.DataFrame(data)
    
    # Apply engineering
    processed_df = add_engineered_features(df)
    
    # Assert columns exist
    assert "rooms_per_household" in processed_df.columns
    assert "bedrooms_per_room" in processed_df.columns
    assert "population_per_household" in processed_df.columns
    
    # Assert calculations are correct
    assert processed_df["rooms_per_household"].iloc[0] == 2.0
    assert processed_df["bedrooms_per_room"].iloc[0] == 0.2
    assert processed_df["population_per_household"].iloc[0] == 10.0
