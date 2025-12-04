# SimpleModel.py
"""
Goal:
- take the engineered value-investing dataset that my partner built
- pick a few simple features (P/B, P/E, NCAV)
- train a basic logistic regression model to predict "Optimality"
- save the trained model + scaler for later use in the app
"""

import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# go up one level from ModelTraining → MLLifecycle → then into ModelDevelopment
DATA_PATH = os.path.join(
    BASE_DIR, "..", "ModelDevelopment", "preparedDataset.csv"
)
DATA_PATH = os.path.abspath(DATA_PATH)  # make it a nice absolute path

# where to save my model + scaler
MODEL_DIR = os.path.join(BASE_DIR, "models_simple")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "simple_value_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "simple_scaler.joblib")

def load_data():
    """
    load the engineered dataset from CSV and do some basic sanity checks
    """
    print(f"Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    print("Columns in dataset:")
    print(df.columns)

    # I’m assuming these columns exist based on partner’s pipeline.
    # If the names are slightly different, I’ll just tweak them.
    needed_cols = ["P/B", "P/E", "NCAV", "Optimality"]

    for col in needed_cols:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in preparedDataset.csv")

    # drop rows that don’t have full info
    df = df.dropna(subset=needed_cols).copy()

    print("Sample of cleaned data:")
    print(df[needed_cols].head())

    return df


def build_features_and_labels(df: pd.DataFrame):
    """
    create X (features) and y (labels) from the dataframe

    Features:
      - P/B
      - P/E
      - NCAV

    Label:
      - Optimality (assumed to be something like 0/1 or a small integer)
    """
    feature_cols = ["P/B", "P/E", "NCAV"]
    target_col = "Optimality"

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(int)  # treat it as a class label

    print(f"Feature matrix shape: {X.shape}")
    print(f"Label vector shape:   {y.shape}")

    return X, y, feature_cols

def split_and_scale(X, y):
    """
    basic train/test split + standardization
    """

    # if each class doesn't have at least 2 samples,
    # we can't use stratify (sklearn will complain)
    unique, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    if min_count < 2:
        print(
            "NOTE: some classes have < 2 samples, "
            "so I'm skipping stratify in train_test_split."
        )
        stratify_arg = None
    else:
        stratify_arg = y

    # 80/20 split, random_state just so it's reproducible
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_arg,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Train shape:", X_train_scaled.shape, "Test shape:", X_test_scaled.shape)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train, y_train):
    """
    train a basic logistic regression model as a simple baseline

    this is intentionally simple and beginner-friendly:
    - no neural nets here
    - just a straightforward classifier on top of partner’s features
    """
    model = LogisticRegression(
        max_iter=1000,
        multi_class="auto",
    )

    print("Training Logistic Regression model...")
    model.fit(X_train, y_train)
    print("Done training.")

    return model

def evaluate_model(model, X_test, y_test):
    """
    print some basic evaluation metrics so I can see how the model did
    """
    y_pred = model.predict(X_test)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def save_artifacts(model, scaler):
    """
    save the trained model + scaler so they can be loaded later
    by the web app or another script
    """
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"\nSaved simple model to:   {MODEL_PATH}")
    print(f"Saved simple scaler to:  {SCALER_PATH}")

def main():
    print("=== STEP 1: Load engineered dataset ===")
    df = load_data()

    print("\n=== STEP 2: Build features + labels ===")
    X, y, feature_cols = build_features_and_labels(df)

    print("\n=== STEP 3: Train/test split + scaling ===")
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    print("\n=== STEP 4: Train simple model ===")
    model = train_model(X_train, y_train)

    print("\n=== STEP 5: Evaluate model ===")
    evaluate_model(model, X_test, y_test)

    print("\n=== STEP 6: Save model + scaler ===")
    save_artifacts(model, scaler)

    print("\nAll done. This is my simple baseline training pipeline built on top of the engineered dataset.")

if __name__ == "__main__":
    main()