import joblib

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    top_k_accuracy_score,
    classification_report,
)

from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# SETTINGS
# -----------------------------

DATASET_FILE = "kmer_dataset.csv"

TEST_SIZE = 0.2

# -----------------------------
# LOAD DATASET
# -----------------------------

print("Loading dataset...")

df = pd.read_csv(DATASET_FILE)

X = df.drop(columns="species").astype("float32")

y = df["species"]

print(f"Dataset shape: {X.shape}, classes: {y.nunique()}")

# -----------------------------
# ENCODE LABELS
# -----------------------------

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y).astype(np.int64)

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------

print("Splitting dataset (stratified)...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=TEST_SIZE,
    random_state=42,
    stratify=y_encoded,
)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# -----------------------------
# TRAIN MODEL
# -----------------------------

print("Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=30,
    max_features="sqrt",
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

# -----------------------------
# PREDICTIONS
# -----------------------------

print("Making predictions...")

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)

# -----------------------------
# EVALUATION
# -----------------------------

accuracy = accuracy_score(y_test, predictions)

top3 = top_k_accuracy_score(y_test, probabilities, k=3)

top5 = top_k_accuracy_score(y_test, probabilities, k=5)

print("\n===== RESULTS =====\n")

print(f"Top-1 accuracy: {round(accuracy * 100, 2)}%")

print(f"Top-3 accuracy: {round(top3 * 100, 2)}%")

print(f"Top-5 accuracy: {round(top5 * 100, 2)}%")

report = classification_report(
    y_test,
    predictions,
    target_names=label_encoder.classes_,
    zero_division=0,
)

with open("rf_report.txt", "w") as f:

    f.write(report)

print("\nFull classification report saved to rf_report.txt")

# -----------------------------
# SAVE MODEL
# -----------------------------

joblib.dump(model, "aquatic_species_model.pkl")

joblib.dump(label_encoder, "rf_label_encoder.pkl")

print("\nModel saved as aquatic_species_model.pkl")
