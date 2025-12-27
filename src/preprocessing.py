import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_data(df):

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Columns with invalid zero values
    invalid_zero_cols = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]


    for col in invalid_zero_cols:
        X_train[col] = X_train[col].replace(0, np.nan)
        X_test[col] = X_test[col].replace(0, np.nan)


    #Replacing zero values with median
    median = {}

    for col in invalid_zero_cols:
        median[col] = X_train[col].median()
        X_train[col] = X_train[col].fillna(median[col])
        X_test[col] = X_test[col].fillna(median[col])



    #Scaling
    scaler = {}

    for col in X_train.columns:
        mean = X_train[col].mean()
        std = X_train[col].std()

        if std == 0:
            std = 1.0

        scaler[col] = {
            "mean": mean,
            "std" : std
        }

    for col in X_train.columns:

        X_train[col] = (X_train[col] - scaler[col]["mean"]) / scaler[col]["std"]
        X_test[col] = (X_test[col] - scaler[col]["mean"]) / scaler[col]["std"]



    # Convert to NumPy arrays

    X_train = X_train.values
    X_test = X_test.values

    return X_train, X_test, y_train, y_test
