import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../models/energy_estimator_model.pkl')

# Load once at import
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"⚠️ Failed to load model: {e}")

def preprocess(prompt: str):
    new_data = pd.DataFrame([{
        'prompt': prompt,
        'model_name': "llama3:70b",
    }])
    new_data_encoded = pd.get_dummies(new_data, drop_first=True)
    for col in model.feature_names_in_:
        if col not in new_data_encoded.columns:
            new_data_encoded[col] = 0
    new_data_encoded = new_data_encoded[model.feature_names_in_]
    return new_data_encoded.values[0]

def predict_energy(prompt: str):
    if model is None:
        raise ValueError("Model not loaded.")
    
    features = preprocess(prompt)
    prediction = model.predict([features])[0]
    return prediction