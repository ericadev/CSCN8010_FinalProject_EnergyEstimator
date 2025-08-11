import numpy as np

class ExpOffsetModel:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
    def predict(self, X):
        X = np.array(X, dtype=float)
        return self.c + self.a * np.exp(-self.b * X)