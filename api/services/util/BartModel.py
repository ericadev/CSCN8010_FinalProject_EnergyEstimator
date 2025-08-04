from transformers import BartTokenizer, BartForConditionalGeneration
import difflib
import torch

class BartModel:
    def __init__(self, model_name='facebook/bart-large-cnn'):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode

    def is_too_similar(a, b, threshold=0.95):
        return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold

    def generate_summary(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=60,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

    def optimize_prompt(self, prompt):
        prompt = prompt.strip()
        if len(prompt.split()) <= 7:
            return prompt  # Skip very short prompts

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)

        # Generate summary
        summary_ids = self.model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=60,
            length_penalty=2.0,
            early_stopping=True
        )

        # Decode output
        optimized = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

        # ✅ Repetition check
        if not optimized or self.is_too_similar(prompt, optimized):
            return prompt  # Use original if too similar or empty

        return optimized


# ✅ Test example
# original = "Explain the core differences between supervised and unsupervised learning models in machine learning."
# optimized = optimize_prompt(original)
# print("Original:", original)
# print("Optimized:", optimized)
