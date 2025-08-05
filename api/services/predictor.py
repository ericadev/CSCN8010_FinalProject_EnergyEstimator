#from services.util.BartModel import BartModel
#from services.util.SBertModel import SBertModel
from api.services.util.SBertModel import SBertModel
from api.services.util.BartModel import BartModel

import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../models/energy_model_rf.pkl')

try:
    rf_model = joblib.load(MODEL_PATH)
    bart = BartModel()
    bert = SBertModel()
except Exception as e:
    raise RuntimeError(f"Failed to initialize models: {str(e)}")

def predict_energy(prompt: str):
    optimized_prompt = bart.optimize_prompt(prompt)
    if optimized_prompt:
        prediction = bert.compare_prompts(prompt, optimized_prompt, rf_model)
        return prediction
    else:
        raise ValueError("Optimized prompt is empty or too similar to the original.")