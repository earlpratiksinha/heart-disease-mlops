import joblib
import pandas as pd
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

# 1. Setup Basic Logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Heart Disease Prediction API")

# 2. Add Prometheus Metrics
Instrumentator().instrument(app).expose(app)

class PatientData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

try:
    model = joblib.load("models/best_model.pkl")
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

@app.get("/")
def home():
    logger.info("Home endpoint accessed.")
    return {"message": "Heart Disease Prediction API is running"}

@app.post("/predict")
def predict(data: PatientData):
    if model is None:
        logger.error("Prediction failed: Model is not loaded.")
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        logger.info(f"Incoming prediction request for age: {data.age}, sex: {data.sex}")
        input_data = pd.DataFrame([data.model_dump()])
        
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0].max()
        
        result = int(prediction[0])
        logger.info(f"Prediction successful: {result} (Confidence: {probability:.2f})")
        
        return {
            "prediction": result,
            "confidence": float(probability)
        }
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e))