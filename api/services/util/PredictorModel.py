import spacy 

class PredictorModel:
    def __init__(self, model):
        self.model = model
        self.nlp = spacy.load("en_core_web_sm")

    def count_verbs(self, text):
        doc = self.nlp(text)
        verbs = [token.text for token in doc if token.pos_ == "VERB"]
        return len(verbs)

    def predict_energy(self, prompt):
        verb_count = self.count_verbs(prompt)
        return self.model.predict(verb_count)
