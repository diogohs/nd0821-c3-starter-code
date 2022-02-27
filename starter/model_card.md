# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

We use a Random Forest Classifier with default parameters for this classification task. This project is part of the Machine Learning DevOps Nanodegree from Udacity.

## Intended Use

The main goal is to predict whether a person income is greater or lower than $ 50k per year based on census data. Besides, we intend to apply recent acquired skills to develop an API using FastAPI to serve a ML model.

## Training Data

The dataser description can be found at: https://archive.ics.uci.edu/ml/datasets/census+income

For training, 80% of the original dataset was randomly selected.

## Evaluation Data

The remaining 20% of the data is part of the test set.

## Metrics

Precision, recall and F-beta metrics were chosen to evaluate the model's performance:

- Precision: 0.7488954344624448
- Recall: 0.6344354335620711
- F-beta: 0.6869300911854104 

## Ethical Considerations

No considerations.

## Caveats and Recommendations

The dataset was donated to the UCI repository in 1996.
