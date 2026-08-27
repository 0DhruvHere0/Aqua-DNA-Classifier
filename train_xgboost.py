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
from xgboost import XGBClassifier
DATASET_FILE = "kmer_dataset.csv"
TEST_SIZE = 0.2
print("Loading dataset...")
df = pd.read_csv(DATASET_FILE)
X = df.drop(columns="species").astype("float32")
y = df["species"]
print(f"Dataset shape: {X.shape}, classes: {y.nunique()}")
print("Encoding species labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y).astype(np.int64)
print("Splitting dataset (stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=TEST_SIZE,
    random_state=42,
    stratify=y_encoded,
)
model = XGBClassifier(
    n_estimators=20,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    min_child_weight=2,
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)
print("Training XGBoost model...")
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print("Making predictions...")
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
accuracy = accuracy_score(y_test, predictions)
print("\n===== RESULTS =====\n")
print("Accuracy:")
print(round(accuracy * 100, 2), "%")
print("\nClassification Report:\n")
print(
    classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_,
        zero_division=0,
    )
)
joblib.dump(model, "xgboost_species_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("\nModel saved!")