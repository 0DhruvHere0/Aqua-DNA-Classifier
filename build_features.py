from Bio import SeqIO

from multiprocessing import Pool

import pandas as pd

import json

from kmer_utils import (
    FEATURE_COLUMNS,
    MIN_LENGTH,
    MAX_LENGTH,
    kmer_frequencies,
    extract_species_name,
)

# -----------------------------
# SETTINGS
# -----------------------------

INPUT_FILE = "filtered_sequences.fasta"

OUTPUT_FILE = "kmer_dataset.csv"

META_FILE = "feature_metadata.json"

MIN_SAMPLES_PER_CLASS = 20

# -----------------------------
# PER-SEQUENCE WORKER
# -----------------------------


def process_record(record):

    species = extract_species_name(record.description)

    if species is None:

        return None

    sequence = "".join(
        base for base in str(record.seq).upper() if base in "ACGT"
    )

    if not (MIN_LENGTH <= len(sequence) <= MAX_LENGTH):

        return None

    return kmer_frequencies(sequence), species

# -----------------------------
# BUILD DATASET
# -----------------------------


if __name__ == "__main__":

    print("Reading sequences...")

    records = list(SeqIO.parse(INPUT_FILE, "fasta"))

    print(f"Loaded {len(records)} sequences")

    X = []

    y = []

    skipped = 0

    with Pool() as pool:

        for processed, result in enumerate(
            pool.imap(process_record, records, chunksize=256),
            start=1
        ):

            if result is None:

                skipped += 1

            else:

                features, species = result

                X.append(features)

                y.append(species)

            if processed % 10000 == 0:

                print(f"Processed {processed}/{len(records)}")

    print(f"Kept {len(y)} sequences, skipped {skipped}")

    # -----------------------------
    # CREATE DATAFRAME
    # -----------------------------

    df = pd.DataFrame(X, columns=FEATURE_COLUMNS, dtype="float32")

    df["species"] = y

    # -----------------------------
    # FILTER RARE CLASSES
    # -----------------------------

    species_counts = df["species"].value_counts()

    keep_species = species_counts[species_counts >= MIN_SAMPLES_PER_CLASS].index

    df = df[df["species"].isin(keep_species)].reset_index(drop=True)

    print(f"Kept {len(keep_species)} species with >= {MIN_SAMPLES_PER_CLASS} samples")

    # -----------------------------
    # SAVE DATASET + METADATA
    # -----------------------------

    df.to_csv(OUTPUT_FILE, index=False, float_format="%.6f")

    meta = {
        "k": 5,
        "canonical": True,
        "normalized": True,
        "feature_columns": FEATURE_COLUMNS,
        "min_samples_per_class": MIN_SAMPLES_PER_CLASS,
    }

    with open(META_FILE, "w") as f:

        json.dump(meta, f)

    print("\nDONE")

    print("Dataset shape:", df.shape)

    print("Unique species:", df["species"].nunique())

    print("\nSaved as:")

    print(OUTPUT_FILE)
