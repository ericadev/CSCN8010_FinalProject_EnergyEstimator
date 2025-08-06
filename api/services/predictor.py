# predictor.py

from api.services.util.BartModel import BartModel
import joblib
import os

# Load energy prediction model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../models/energy_model_rf.pkl')

try:
    
    bart = BartModel()
    rf_model = joblib.load(MODEL_PATH)
    # SBertModel is temporarily disabled
except Exception as e:
    raise RuntimeError(f"Failed to initialize models: {str(e)}")

def predict_energy(prompt: str):
    optimized_prompt = bart.optimize_prompt(prompt)
    
    if optimized_prompt:
        # Simulate prediction structure since SBert is disabled
        # You can modify this later once SBert is re-integrated
        prediction = {
            "original_prompt": prompt,
            "optimized_prompt": optimized_prompt,
            "token_count": len(optimized_prompt.split()),
            "energy_saved": round(0.0025, 4),  # placeholder
            "similarity_score": 1.0            # placeholder
        }
        return prediction
    else:
        raise ValueError("Optimized prompt is empty or too similar to the original.")
