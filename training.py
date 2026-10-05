from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from joblib import dump


def load_data():
    iris = load_iris()
    return iris


def train_model(data):
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(data.data, data.target)
    return model


def save_model(model):
    dump(model, "model.joblib")


if __name__ == "__main__":
    data = load_data()
    model = train_model(data)
    save_model(model)
