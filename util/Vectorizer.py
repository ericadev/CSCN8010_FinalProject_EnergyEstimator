# A class that handles vectorization of text data using Word2Vec
import tiktoken
from gensim.models import Word2Vec
import numpy as np
import re

class Vectorizer:
    def __init__(self, sentences, vector_size=100, window=5, min_count=1, workers=4):
        """
        Initializes the Vectorizer with the given parameters and trains a Word2Vec model.

        :param sentences: List of tokenized sentences (list of lists of words)
        :param vector_size: Dimensionality of the word vectors
        :param window: Maximum distance between the current and predicted word within a sentence
        :param min_count: Ignores all words with total frequency lower than this
        :param workers: Number of worker threads to train the model
        """
        cleaned_prompts = [self.clean_text(s) for s in sentences]

        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        tokenized_prompts = [self.tokenizer.encode(p) for p in cleaned_prompts]
        self.model = Word2Vec(tokenized_prompts, vector_size=vector_size, window=window,
                              min_count=min_count, workers=workers)
        self.model.train(sentences, total_examples=len(sentences), epochs=10)
    
    def clean_text(self, text):
        text = re.sub(r"\s+", " ", text)  # Remove extra whitespace
        text = text.strip().lower()
        return text

    def sentence_vector(self, sentence):
        """Compute average vector for a tokenized sentence"""
        valid_tokens = self.tokenizer.encode(self.clean_text(sentence))
        if not valid_tokens or len(valid_tokens) == 0:
            raise ValueError("No valid tokens found in the sentence.")
        # Return the average vector for the valid tokens
        return np.mean([self.model.wv[token] for token in valid_tokens], axis=0)

    def save_model(self, path):
        """
        Saves the trained model to the specified path.

        :param path: Path to save the model
        """
        self.model.save(path)

    def load_model(self, path):
        """
        Loads a model from the specified path.

        :param path: Path to load the model from
        """
        self.model = Word2Vec.load(path)
    
