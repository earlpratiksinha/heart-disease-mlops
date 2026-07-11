import os
import joblib
import pandas as pd
from fastapi.testclient import TestClient
from src.api import app

# Create a test client using your FastAPI app
client = TestClient(app)

def test_model_file_exists():
    """Test if the trained model file was generated and saved."""
    assert os.path.exists("models/best_model.pkl"), "Model file does not exist!"

def test_model_loads_and_predicts():
    """Test if the model can be loaded and handles data properly."""
    model = joblib.load("models/best_model.pkl")
    
    # Dummy data with correct features
    dummy_data = pd.DataFrame([{
        "age": 50, "sex": 1, "cp": 2, "trestbps": 120, "chol": 200,
        "fbs": 0, "restecg": 1, "thalach": 140, "exang": 0,
        "oldpeak": 1.5, "slope": 1, "ca": 0, "thal": 2
    }])
    
    prediction = model.predict(dummy_data)
    assert prediction[0] in [0, 1], "Prediction should be 0 or 1"

def test_api_home():
    """Test the root endpoint of the API."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_api_predict():
    """Test the /predict endpoint with a standard JSON payload."""
    sample_payload = {
      "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
      "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
      "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
    }
    
    response = client.post("/predict", json=sample_payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["confidence"] <= 1.0