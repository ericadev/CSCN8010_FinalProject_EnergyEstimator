from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import joblib

class SBertModel:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = joblib.load("../models/sbert_model.pkl")

    def encode(self, texts, convert_to_tensor=False):
        return self.model.encode(texts, convert_to_tensor=convert_to_tensor)
    
    def get_semantic_similarity(self, original, optimized):
        emb1 = self.model.encode(original, convert_to_tensor=True)
        emb2 = self.model.encode(optimized, convert_to_tensor=True)
        return float(util.cos_sim(emb1, emb2))

    def predict_energy(self, model, prompt):
        token_length = len(prompt.split())
        char_length = len(prompt)
        avg_word_length = char_length / token_length if token_length > 0 else 0
        features = [[token_length, char_length, avg_word_length]]
        log_prediction = model.predict(features)[0]
        return max(np.expm1(log_prediction), 1e-6)  # Reverse log1p and clamp minimum

    def compare_prompts(self, original, optimized, model):
        original_energy = self.predict_energy(model, original)
        optimized_energy = self.predict_energy(model, optimized)
        similarity = self.get_semantic_similarity(original, optimized)

        # Reject if similarity too low
        if similarity < 0.80:
            return {
                "original_prompt": original,
                "optimized_prompt": optimized,
                "original_energy": round(original_energy, 6),
                "optimized_energy": round(original_energy, 6),  # force same
                "energy_saved (%)": 0.0,
                "shortening_coeff": round(len(optimized) / len(original), 2),
                "semantic_similarity": round(similarity, 4),
                "output_confidence": "Rejected (Low Similarity)"
            }

        # Handle cosmetic changes
        if similarity > 0.97 and abs(len(optimized) - len(original)) < 5:
            optimized_energy = original_energy
            energy_saving = 0.0
        else:
            energy_saving = (original_energy - optimized_energy) / original_energy
            if abs(energy_saving) < 0.01:
                energy_saving = 0.0
                optimized_energy = original_energy

        return {
            "original_prompt": original,
            "optimized_prompt": optimized,
            "original_energy": round(original_energy, 6),
            "optimized_energy": round(optimized_energy, 6),
            "energy_saved (%)": round(energy_saving * 100, 2),
            "shortening_coeff": round(len(optimized) / len(original), 2),
            "semantic_similarity": round(similarity, 4),
            "output_confidence": "High" if similarity >= 0.85 else "Low"
        }
    
    def return_results(self, original, optimized, model): 
        return self.compare_prompts(original, optimized, model)

