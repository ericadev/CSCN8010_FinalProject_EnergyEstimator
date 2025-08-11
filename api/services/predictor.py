# predictor.py

from api.services.util.BartModel import BartModel
from api.services.util.SBertModel import SBertModel
import joblib
import os
import numpy as np

# === Load Models ===
ENERGY_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../models/energy_model_rf.pkl')
SBERT_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../models/sbert_model.pkl')

try:
    bart = BartModel()
    rf_model = joblib.load(ENERGY_MODEL_PATH)
    sbert = SBertModel(SBERT_MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to initialize models: {str(e)}")

# === Utility ===
def cosine_similarity(vec1, vec2):
    a, b = np.array(vec1), np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# === Main Function ===
def predict_energy(prompt: str):
    optimized_prompt = bart.optimize_prompt(prompt)

    if not optimized_prompt:
        raise ValueError("Optimized prompt is empty or too similar to the original.")

    # --- Energy Calculation ---
    def extract_features(text):
        token_len = len(text.split())
        char_len = len(text)
        avg_word_len = char_len / token_len if token_len > 0 else 0
        return [[token_len, char_len, avg_word_len]]

    original_features = extract_features(prompt)
    optimized_features = extract_features(optimized_prompt)

    original_energy = rf_model.predict(original_features)[0]
    optimized_energy = rf_model.predict(optimized_features)[0]

    energy_saved = round(original_energy - optimized_energy, 6)

    # --- Similarity Calculation ---
    emb1 = sbert.model.encode(prompt)
    emb2 = sbert.model.encode(optimized_prompt)
    similarity = round(cosine_similarity(emb1, emb2), 4)

    # --- Final Output ---
    return {
        "original_prompt": prompt,
        "optimized_prompt": optimized_prompt,
        "token_count": len(optimized_prompt.split()),
        "original_energy": round(original_energy, 6),
        "optimized_energy": round(optimized_energy, 6),
        "energy_saved": energy_saved,
        "similarity_score": similarity
    }
