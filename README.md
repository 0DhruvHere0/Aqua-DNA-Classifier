# Aqua Species Finder

DNA barcode-based aquatic species classification using k-mer frequencies and machine learning.

## Quick Start

```bash
# 1. Activate environment (requires Python 3.10+)
conda create -n aqua-species python=3.12
conda activate aqua-species
pip install biopython pandas scikit-learn xgboost joblib

# 2. Build dataset from COI sequences (takes ~2 min)
python build_features.py

# 3. Train model (RandomForest, ~40 sec)
python train_model.py

# 4. Predict unknown sequences
python predict_species.py
```

## Pipeline

```
sequence.fasta
     │
     ▼ (filter_dataset.py)
filtered_sequences.fasta  ──► build_features.py ──► kmer_dataset.csv + feature_metadata.json
                                                    │
                                                    ▼
                                            train_model.py
                                                    │
                                                    ▼
                                  aquatic_species_model.pkl + rf_label_encoder.pkl
                                                    │
                                                    ▼
                                          predict_species.py
```

## Key Features

- **Normalized k-mer frequencies** (not raw counts) — fixes length-dependent prediction errors
- **Canonical 5-mers** (512 features) — orientation-invariant for COI barcodes
- **Stratified train/test splits** — reliable accuracy estimates
- **Low-confidence warnings** — flags when true species may be missing from training data
- **Fast RandomForest** — trains in <1 min on 600+ species

## Files

| File | Description |
|------|-------------|
| `kmer_utils.py` | Shared k-mer vocabulary, feature extraction |
| `filter_dataset.py` | Filters raw FASTA to species with ≥20 sequences |
| `build_features.py` | Extracts normalized k-mer frequencies |
| `train_model.py` | Trains RandomForest classifier |
| `train_xgboost.py` | Trains XGBoost (slower, optional) |
| `predict_species.py` | Predicts species from unknown FASTA |
| `predict_xgboost.py` | XGBoost prediction script |

## Input Data

Place your COI barcode FASTA as `sequence.fasta` with headers like:
```
>ACCESSION Genus species voucher ...
ATGC...
```

## Output

`predict_species.py` prints top-5 predictions with confidence:
```
===== TOP 5 PREDICTIONS for unknown_sample =====

1. Gadus morhua --> 94.20%
2. Gadus macrocephalus --> 3.10%
3. Merluccius merluccius --> 1.20%
...

WARNING: low confidence. The true species may be missing from the training data.
```

## Tuning

Edit `build_features.py`:
- `MIN_SAMPLES_PER_CLASS = 50` — keep species with ≥N samples (reduce for more classes, increase for faster training)

## Requirements

- Python 3.10+
- biopython, pandas, scikit-learn, xgboost, joblib

## License

MIT