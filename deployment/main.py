import joblib
import pandas as pd
import s3fs
from fastapi import FastAPI, HTTPException

# Automatically detects AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY from environment
fs = s3fs.S3FileSystem()

# Stream model from S3 into memory on startup
model_path = 's3://forex-data2026/Fx_data/predictive_models/hour_open_trigger_model.joblib'

with fs.open(model_path, "rb") as f:
    model = joblib.load(f)
    print("Successfully streamed model from AWS S3")

app = FastAPI(title="Inference API", version="1.0")

@app.post("/predict")
def predict(payload: dict):
    try:
        df = pd.DataFrame(payload["data"], columns=payload["columns"])

        predictions = model.predict(df)
        probabilities = model.predict_proba(df)

        return {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}
