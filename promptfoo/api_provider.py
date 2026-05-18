import requests
import json
import os

def call_api(prompt, options, context):
    """
    Custom provider for Promptfoo to evaluate prompts through the active
    FastAPI real-estate description generation service.
    """
    # 1. Access Promptfoo variables
    vars = context.get('vars', {})
    
    # 2. Extract context parameters or fall back to default housing inputs
    predicted_price = float(vars.get('predicted_price', 350000.0))
    ocean_proximity = str(vars.get('ocean_proximity', "NEAR BAY"))
    total_rooms = float(vars.get('total_rooms', 6.0))
    housing_median_age = float(vars.get('housing_median_age', 15.0))
    
    # 3. Formulate the POST payload to target our local FastAPI service
    payload = {
        "predicted_price": predicted_price,
        "ocean_proximity": ocean_proximity,
        "total_rooms": total_rooms,
        "housing_median_age": housing_median_age,
        "prompt_template": prompt
    }
    
    # 4. Target the API server URL (defaulting to container compose routing if set)
    api_url = os.getenv("API_URL", "http://localhost:8000")
    endpoint = f"{api_url}/generate_description"
    
    try:
        response = requests.post(endpoint, json=payload, timeout=15)
        if response.status_code == 200:
            description = response.json().get("description", "")
            return {"output": description}
            
        return {"error": f"API returned status {response.status_code}: {response.text}"}
        
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to connect to FastAPI server at {endpoint}: {str(e)}"}
