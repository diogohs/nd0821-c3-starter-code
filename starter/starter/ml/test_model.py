# import pytest
from model import train_model  # , compute_model_metrics, inference
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


def test_train_model():
    """Test train_model
    """
    X, y = make_classification(n_samples=100, n_features=3)
    model = train_model(X, y)
    assert model == RandomForestClassifier
