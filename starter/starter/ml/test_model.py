import pytest
import numpy as np
import pandas as pd
from .model import (
    train_model, compute_model_metrics, inference
)
from .data import process_data
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier


@pytest.fixture()
def data():
    """Fixture: generate a random 2-class classification problem data
    """
    X, y = make_classification(n_samples=100)
    return X, y


@pytest.fixture()
def model(data):
    """Fixture: return dummy model
    """
    dummy_model = DummyClassifier()
    X, y = data
    dummy_model.fit(X, y)
    return dummy_model


@pytest.fixture()
def real_data():
    data = pd.read_csv(r"starter/data/census_clean.csv")

    train, _ = train_test_split(data, test_size=0.20, random_state=42)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X_train, y_train, _, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    return X_train, y_train


def test_train_model(data):
    """Test train_model
    """
    X, y = data
    model = train_model(X, y)
    assert type(model) == RandomForestClassifier


def test_inference(model, data):
    X, y = data
    y_pred = inference(model, X)
    assert type(y_pred) == np.ndarray
    assert len(y) == len(y_pred)


def test_compute_model_metrics(real_data):
    X, y = real_data
    model = train_model(X, y)
    y_pred = inference(model, X)
    precision, recall, fbeta = compute_model_metrics(y, y_pred)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1
