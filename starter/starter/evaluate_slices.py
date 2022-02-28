import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import compute_model_metrics, inference


def compute_metrics_on_test_set():
    """Compute metrics on test set"""
    # Load test data
    data = pd.read_csv(r"../data/census_clean.csv", index_col=0)

    # Train test split
    _, test = train_test_split(data, test_size=0.20, random_state=42)

    # Load model, encoder and label binarizer
    with open("../model/model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("../model/encoder.pkl", "rb") as file:
        encoder = pickle.load(file)

    with open("../model/label_binarizer.pkl", "rb") as file:
        lb = pickle.load(file)

    # Categorical features
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

    with open("slice_output.txt", "w") as file:
        for col_value in test["education"].unique():
            file.write(f"Slice - column value: {col_value}\n")

            # Process the test data with the process_data function.
            X_test, y_test, encoder, lb = process_data(
                test.loc[test["education"] == col_value],
                categorical_features=cat_features,
                label="salary",
                training=False,
                encoder=encoder,
                lb=lb,
            )

            # Get predictions
            preds = inference(model, X_test)

            # Calculate metrics
            precision, recall, fbeta = compute_model_metrics(y_test, preds)

            # Write slice metrics to file
            file.write(f"Precision: {precision:.4f}\n")
            file.write(f"Recall:    {recall:.4f}\n")
            file.write(f"FBeta:     {fbeta:.4f}\n")
            file.write(80 * "*")
            file.write("\n")


if __name__ == "__main__":
    compute_metrics_on_test_set()
