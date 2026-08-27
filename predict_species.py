import json
import joblib
import pandas as pd
from Bio import SeqIO
from kmer_utils import (
    FEATURE_COLUMNS,
    MIN_LENGTH,
    CONFIDENCE_THRESHOLD,
    clean_sequence,
    kmer_frequencies,
)
MODEL_FILE = "aquatic_species_model.pkl"
ENCODER_FILE = "rf_label_encoder.pkl"
META_FILE = "feature_metadata.json"
INPUT_FILE = "unknown.fasta"
TOP_N = 5
print("Loading model...")
model = joblib.load(MODEL_FILE)
label_encoder = joblib.load(ENCODER_FILE)
with open(META_FILE) as f:
    meta = json.load(f)
if meta["feature_columns"] != FEATURE_COLUMNS:
    raise RuntimeError(
        "Feature mismatch between model and kmer_utils.py. "
        "Rebuild the dataset and retrain the model."
    )
found_any = False
for record in SeqIO.parse(INPUT_FILE, "fasta"):
    sequence = clean_sequence(str(record.seq))
    if len(sequence) < MIN_LENGTH:
        print(
            f"Skipping {record.description}: "
            f"sequence too short after cleaning ({len(sequence)} bp)"
        )
        continue
    found_any = True
    features = kmer_frequencies(sequence)
    X = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    probabilities = model.predict_proba(X)[0]
    classes = label_encoder.inverse_transform(model.classes_)
    top_indices = probabilities.argsort()[-TOP_N:][::-1]
    print(f"\n===== TOP {TOP_N} PREDICTIONS for {record.description} =====\n")
    for rank, idx in enumerate(top_indices, start=1):
        species = classes[idx]
        confidence = probabilities[idx] * 100
        print(f"{rank}. {species} --> {confidence:.2f}%")
    best_confidence = probabilities[top_indices[0]] * 100
    if best_confidence < CONFIDENCE_THRESHOLD:
        print(
            "\nWARNING: low confidence. The true species may be missing "
            "from the training data."
        )
if not found_any:
    print("No usable sequences found in", INPUT_FILE)