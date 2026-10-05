from joblib import load
import numpy as np

labels = {0: "setosa", 1: "versicolor", 2: "virginica"}


def load_model():
    return load("model.joblib")


def predict(model, data):
    data = np.array([val for val in data.values()]).reshape(1, -1)
    return labels[model.predict(data)[0]]
