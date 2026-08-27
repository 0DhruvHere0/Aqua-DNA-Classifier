# Why the training run failed

Diagnosis of `Aqua_Species_Finder`, 2026-08-26. Every number below was measured on your
actual `kmer_dataset.csv`, not estimated.

## Short version

Your data is fine. A plain 1-nearest-neighbour baseline on your own feature matrix gets
**93% accuracy on a strictly honest split**, so the signal is real and the task is learnable.

The run failed because of one line in `train_xgboost.py`:

```python
min_child_weight=2,
```

With 2,115 classes this single parameter makes it mathematically impossible for the trees to
isolate any species. It looks harmless — the XGBoost default is 1 — but that default is
calibrated for binary problems, and it silently destroys many-class softmax training.

## The mechanism

`min_child_weight` is not a minimum row count. It is a minimum **sum of Hessians** in a child
node. For multiclass softmax the per-sample Hessian is `p(1-p)`, and at the first boosting
round `p = 1/n_classes`:

```
p        = 1/2115          = 4.728e-04
Hessian  = p(1-p)          = 4.726e-04   per sample
```

So the number of rows a leaf must contain before XGBoost will allow the split is:

| min_child_weight | rows required per leaf |
|---|---|
| 2 (your setting) | 4,232 |
| 1 (XGBoost default) | 2,116 |
| 1e-3 (correct here) | 2 |

Your median species has **42 training rows**. You are asking the tree to isolate 42 rows while
forbidding any leaf smaller than 4,232 — a constraint that is **101x too strict**.

## What this does to the trees, measured

I grew a real depth-6 tree on your feature matrix using exact XGBoost split gains and softmax
gradients, targeting `Ostorhinchus fasciatus` (42 train rows, a typical class):

| | your `min_child_weight=2` | corrected `1e-3` |
|---|---|---|
| leaves grown (max_depth=6 allows 64) | **3** | 10 |
| smallest leaf | 4,483 rows | 3 rows |
| purest leaf | 6,905 rows, **0.58% pure** | 6 rows, **100% pure** |
| species sharing that leaf | **~296** | 1 |

The tree collapses to three leaves. The best leaf it can build lumps your target species
together with roughly 296 other species and assigns all of them the same positive margin.
Every one of the 2,115 per-class trees is this blunt, so the model is very nearly a constant
function. Predicted probabilities stay near uniform (`1/2115 ≈ 0.05%`), which also means
`CONFIDENCE_THRESHOLD = 50.0` in `kmer_utils.py` would flag literally every prediction as
"low confidence".

## Why the run never even saved

`xgboost_species_model.pkl` is dated May 29, while you edited `train_xgboost.py` on Aug 26 at
22:21. The `joblib.dump` at the bottom never executed, so the process died or was interrupted
before finishing. That is consistent with the cost of the configuration:

```
trees to build = n_estimators x n_classes = 20 x 2115 = 42,300
```

and with the memory profile:

```
train margin cache   88,443 x 2115 x 4B  = 0.75 GB
gradients            (same shape)        = 0.75 GB
Hessians             (same shape)        = 0.75 GB
eval_set cache       22,111 x 2115 x 4B  = 0.19 GB
read_csv as float64  110,554 x 512 x 8B  = 0.45 GB
.astype("float32") copy                  = 0.23 GB
train_test_split copies                  = 0.23 GB
                                   PEAK ~= 3.34 GB
```

`pd.read_csv` parses to float64 first, so `.astype("float32")` doubles the frame instead of
saving memory. Passing an explicit dtype map to `read_csv` avoids that entirely.

## Second, independent bug: the prediction path is broken

`predict_xgboost.py` cannot work even with a good model, and the guard you wrote will not
catch it:

```python
if meta["feature_columns"] != FEATURE_COLUMNS:   # compares metadata to kmer_utils
    raise RuntimeError(...)                      # never compares either to the MODEL
```

Reading the pickle directly, the saved model records `num_feature = 1024` and
`num_class = 2115`. But you later switched `kmer_utils.py` to `CANONICAL = True`, which
collapses the vocabulary from 1024 k-mers to 512, and you regenerated
`feature_metadata.json` at the same time as the CSV. So metadata and `kmer_utils` agree with
each other, the guard passes, and then 512 columns are handed to a model expecting 1024.

The fix is to validate against the model, not the metadata:

```python
assert model.n_features_in_ == len(FEATURE_COLUMNS)
```

## Third: 50.7% of your dataset is duplicate rows

```
unique feature vectors : 54,475 / 110,554
exact duplicate rows   : 56,079  (50.7%)
largest identical group : 314 rows
```

Because `train_test_split` splits rows at random, **60.0% of your test rows have a
byte-identical copy sitting in the training set**. That is textbook leakage — and it is the
reason the 96.22% in `output.txt` should not be believed at face value.

To be fair to the data, I measured how much this actually inflates the score using 1-NN:

| evaluation | accuracy |
|---|---|
| random split (what your scripts do) | 95.72% |
| duplicate-aware split (honest) | 92.97% |

So the inflation is about 2.8 points, not 40. This is genuinely good news: the k-mer
representation works, and 93% is a real, defensible number. But you are also paying to train
on 56,079 redundant rows, and any future headline accuracy should come from the grouped split.

Only 4.1% of rows sit in duplicate groups with *conflicting* species labels, putting the hard
ceiling from identical vectors at **99.01%** — comfortably above where you are.

## Fourth: ~12% of labels are not species

`extract_species_name` just takes words 2 and 3 of the FASTA header, which is right for most
NCBI records but produces garbage for the rest:

```
 97,559 rows  1,931 classes   valid binomial
  9,015 rows    146 classes   genus + "sp."   (unidentified)
  2,328 rows      4 classes   "environmental" (not a species at all)
  1,196 rows     23 classes   "cf." / "aff."  (uncertain ID)
    432 rows     10 classes   header prefix junk (UNVERIFIED:, PREDICTED:, ...)
```

Your single largest class, at 2,254 rows, is `Actinopterygii environmental` — 2x the next
biggest and not a species. `Betta cf.` and 145 `... sp.` classes are unresolvable by
construction: no model can separate `Astyanax sp.` from a named *Astyanax*.

## Fix order

1. Set `min_child_weight=1e-3`. Nothing else matters until this is done.
2. Raise `n_estimators` (20 rounds cannot separate 2,115 classes; `ln(2115) = 7.66` of logit
   gap is needed before you even beat the uniform prior) and add early stopping on the eval set.
3. Load with an explicit float32 dtype map and `del` the frame before `fit`.
4. Drop exact-duplicate feature vectors, and split by duplicate group rather than by row.
5. Filter labels matching `sp.`, `cf.`, `aff.`, `environmental`, and the `UNVERIFIED:`-style prefixes.
6. Validate `model.n_features_in_` at predict time, and retrain before using the stale pickle.
7. Consider whether XGBoost is the right tool at all. 2,115 classes means 42,300 trees per
   fit. Your May Random Forest ran fine, and 1-NN already delivers 93% — for barcode data,
   distance-based methods are both faster and standard practice.

Also worth noting: `MAX_LENGTH = 1500` in `kmer_utils.py` silently discards full mitochondrial
genomes (~16 kb), which is why the rebuilt dataset is 110,554 rows against roughly 143,000
behind the old `output.txt`.
