from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Heart Disease Prediction API")

# Load the saved model pipeline (which includes the StandardScaler)
try:
    model = joblib.load('models/best_model.pkl')
except Exception as e:
    model = None

# Define the expected JSON input structure using Pydantic
class PatientData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running. Use /predict to get predictions."}

@app.post("/predict")
def predict(data: PatientData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")
    
    # Convert input data to a Pandas DataFrame
    input_data = pd.DataFrame([data.model_dump()])
    
    # Make prediction
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        return {
            "prediction": int(prediction),
            "confidence": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)