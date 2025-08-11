from sentence_transformers import SentenceTransformer, util
import numpy as np
import joblib
import os
import logging

log = logging.getLogger(__name__)

DEFAULT_HF_MODEL = "all-MiniLM-L6-v2"

class SBertModel:
    def __init__(self, model_name: str = DEFAULT_HF_MODEL):
        """
        Try local pickle if present; otherwise load a HF model.
        If model_name looks like a local path (e.g., .pkl or has a path separator),
        ignore it for HF loading and use DEFAULT_HF_MODEL.
        """
        # legacy pickle path used by repo
        pickle_path = os.path.join(
            os.path.dirname(__file__), "../../../models/sbert_model.pkl"
        )

        self.model = None

        # 1) Try the pickle (backward-compatible)
        try:
            if os.path.exists(pickle_path) and os.path.getsize(pickle_path) > 0:
                self.model = joblib.load(pickle_path)
                log.info(f"Loaded SBERT from pickle: {pickle_path}")
        except Exception as e:
            log.warning(f"Failed to load SBERT pickle ({pickle_path}): {e}")

        # 2) Fallback to HF model
        if self.model is None:
            # If caller passed a path or a .pkl by mistake, force default HF id
            looks_like_path = (
                os.path.sep in model_name
                or model_name.lower().endswith((".pkl", ".pt", ".bin", ".zip"))
            )
            hf_name = DEFAULT_HF_MODEL if looks_like_path else model_name
            log.info(f"Loading SentenceTransformer('{hf_name}')")
            self.model = SentenceTransformer(hf_name)

    def encode(self, texts, convert_to_tensor: bool = False):
        return self.model.encode(texts, convert_to_tensor=convert_to_tensor)

    def get_semantic_similarity(self, original: str, optimized: str) -> float:
        emb1 = self.model.encode(original, convert_to_tensor=True)
        emb2 = self.model.encode(optimized, convert_to_tensor=True)
        return float(util.cos_sim(emb1, emb2))

    def predict_energy(self, model, prompt: str) -> float:
        token_length = len(prompt.split())
        char_length = len(prompt)
        avg_word_length = char_length / token_length if token_length > 0 else 0
        features = [[token_length, char_length, avg_word_length]]
        log_prediction = model.predict(features)[0]
        return max(np.expm1(log_prediction), 1e-6)

    def compare_prompts(self, original: str, optimized: str, model):
        original_energy = self.predict_energy(model, original)
        optimized_energy = self.predict_energy(model, optimized)
        similarity = self.get_semantic_similarity(original, optimized)

        if similarity < 0.80:
            return {
                "original_prompt": original,
                "optimized_prompt": optimized,
                "original_energy": round(original_energy, 6),
                "optimized_energy": round(original_energy, 6),
                "energy_saved (%)": 0.0,
                "shortening_coeff": round(len(optimized) / len(original), 2),
                "semantic_similarity": round(similarity, 4),
                "output_confidence": "Rejected (Low Similarity)",
            }

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
            "output_confidence": "High" if similarity >= 0.85 else "Low",
        }

    def return_results(self, original: str, optimized: str, model):
        return self.compare_prompts(original, optimized, model)
