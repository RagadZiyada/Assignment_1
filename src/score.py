import json
import os
import joblib
import numpy as np

model = None

def init():
    global model
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    model_path = os.path.join(model_dir, "model.pkl")
    if not os.path.exists(model_path):
        # fallback in case Azure nests model under a versioned folder
        for root, dirs, files in os.walk(model_dir):
            if "model.pkl" in files:
                model_path = os.path.join(root, "model.pkl")
                break
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)

        if "data" not in data:
            return {"error": "Request JSON must contain a 'data' field."}

        X = np.array(data["data"])
        preds = model.predict(X)

        return {"predictions": preds.tolist()}
    except Exception as e:
        return {"error": str(e)}
