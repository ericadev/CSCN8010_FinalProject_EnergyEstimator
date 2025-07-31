# A class that handles vectorization of text data using Word2Vec
from gensim.models import Word2Vec
import numpy as np

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
        self.model = Word2Vec(sentences, vector_size=vector_size, window=window,
                              min_count=min_count, workers=workers)
    
    def print_sentence_vector(self, sentence_tokens, model):
        """Prints the average vector for a tokenized sentence"""
        vector = self.sentence_vector(sentence_tokens, model)
        print(f"Vector for sentence '{' '.join(sentence_tokens)}': {vector}")

    def sentence_vector(self, sentence_tokens, model):
        """Compute average vector for a tokenized sentence"""
        valid_tokens = [token for token in sentence_tokens if token in model.wv]
        if not valid_tokens:
            return np.zeros(model.vector_size)
        return np.mean([model.wv[token] for token in valid_tokens], axis=0)
    
    def get_vector(self, word):
        """
        Returns the vector for a given word.

        :param word: The word to get the vector for
        :return: The vector for the word or None if the word is not in vocabulary
        """
        return self.model.wv[word] if word in self.model.wv else None

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

    def update_dataframe_with_prompt_vector(self, df, column_name):
        """
        Computes the average vector for each sentence in a specified column of a DataFrame.
        """
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
        if df[column_name].dtype != 'object':
            raise ValueError(f"Column '{column_name}' must contain text data (dtype 'object').")
        df['prompt_vector'] = df[column_name].apply(lambda x: self.sentence_vector(x, self.model) if isinstance(x, list) else np.zeros(self.model.vector_size))
        return df
    
