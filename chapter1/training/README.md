# Training and evaluation

`train.py` retains the architecture and training procedure in the five uploaded `training-pytorch_ver8.0_GPU_ablation.py` files. Input paths, alphabet selection, training settings, and ablations are command-line parameters.

## Inputs

Provide one negative-pair HDF5 and one positive-pair HDF5 generated with the same alphabet. The negative file must contain label 0, and the positive file label 1. Both files require the 25 datasets documented in [the feature README](../feature/README.md). Each class must contain enough rows to produce nonempty train, validation, and test subsets.

```bash
python training/train.py --negative data/negatome_rhp4.hdf5 --positive data/biogrid_rhp4.hdf5 --alphabet rhp4 --output-root runs/rhp4
```

## Feature views and model

For each protein, the position view concatenates 1-mer, 2-mer, 3-mer, H1-matrix, and H2-matrix features along the channel axis. Its shape is `(K + K² + K³ + 60, 20)`, where K includes the placeholder.

The summary view concatenates H1 features, H2 features, APAAC1, APAAC2, composition, and scaled length in that order. It contains `120 + K + 1` values.

Protein A and protein B have separate position and summary encoders. Each position encoder maps the flattened input to 512 and then 128 units. Each summary encoder maps its input to 64 and then 16 units. Their outputs form a 144-dimensional representation of each protein.

Four tokens are passed to the Transformer: the A representation, the B representation, their elementwise absolute difference, and their elementwise product. The Transformer output is flattened and passed through a 128-unit layer, a 32-unit layer, and a single logit. Sigmoid converts the logit to the reported probability.

The original LayerNorm, activations, dropout placement, separate A/B encoder weights, and Transformer settings are retained. A and B use distinct learned branches, so pair order remains part of the implementation.

## Original defaults

| Setting | Default |
| --- | --- |
| Epochs | 24 |
| Batch size | 16 |
| Learning rate | HP2: 1e-5; other alphabets: 5e-6 |
| Optimizer | AdamW |
| Weight decay | 1e-5 |
| Loss | BCEWithLogitsLoss |
| Gradient clipping | Maximum norm 5.0 |
| Dropout | 0.25 |
| Transformer layers | 1 |
| Attention heads | 4 |
| Feed-forward dimension | 384 |
| Test fraction within each selected class | 0.20 |
| Validation fraction of the remaining rows | 0.40 |
| Classification threshold | 0.5 |
| Checkpoint selection | Highest validation AUC |
| Early-stopping patience | 100 epochs |
| Ablations | Empty list; all features are used |
| Seeds | 0, 11111, 22222, 33333, 44444, 55555, 66666, 77777, 88888, 99999 |

The original files contained comments and run labels suggesting a summary-only ablation, but their actual default ablation lists were empty and the model consumed both views. The released defaults follow that executable behavior.

Use `--epochs`, `--batch-size`, `--learning-rate`, `--weight-decay`, `--seeds`, `--device`, `--test-ratio`, `--validation-ratio`, `--threshold`, `--dropout`, `--transformer-layers`, `--transformer-heads`, `--transformer-ff-dim`, `--save-best-by`, and `--patience` to override defaults. `--cpu-threads` controls CPU thread count. CUDA requires a compatible PyTorch installation; `--device cpu` always selects the CPU.

## Balanced splitting

For each seed, the program samples `min(number_of_negative_rows, number_of_positive_rows)` rows from each class without replacement. Consequently, when positives outnumber negatives, only a balanced subset of positives is used in that seed.

Each class is shuffled independently. The first rounded 20% is assigned to test. The remaining rows are reshuffled; a rounded 40% of those rows is assigned to validation, with the rest used for training. This corresponds approximately to 48% training, 32% validation, and 20% test. Each concatenated subset is then shuffled. Its row indices remain fixed across epochs within the seed, and are saved in `fixed_split_indices.json`.

Each selected row belongs to one subset. This is an **interaction-row split**; protein IDs may recur across subsets. Pair deduplication and sequence filtering belong to the upstream dataset preparation. The program does not replace the supplied split with a protein-disjoint or sequence-cluster split.

The files use the original `untouched_test` naming. This set is excluded from gradient updates and checkpoint selection. Its metrics are nevertheless evaluated and logged each epoch, as in the originals; it is not a set first opened only after training. The best checkpoint is chosen using the validation criterion and evaluated again on test.

## Feature ablation

Pass one or more feature-block names with `--ablate`.

```bash
python training/train.py --negative work/rhp4_negative.hdf5 --positive work/rhp4_positive.hdf5 --alphabet rhp4 --output-root runs/rhp4_without_positions --ablate one_mer_counts two_mer_counts three_mer_counts H1_matrix H2_matrix
```

This masks every position block. The model architecture and input dimensions remain unchanged. To mask summary blocks, use `H1_features`, `H2_features`, `apaac1`, `apaac2`, `proportion`, or `length`. Masks apply to both proteins and both source datasets. The replacement constant defaults to 0 and can be changed with `--ablation-constant`.

## Outputs

Each seed creates a timestamped run directory containing:

| File | Contents |
| --- | --- |
| `config.json` | Paths, alphabet mapping, H1/H2 vectors, software versions, hyperparameters, dimensions, and parameter count |
| `fixed_split_indices.json` | Selected source-row indices and post-concatenation shuffle orders |
| `best_model.pt` | Best model and optimizer states, configuration, selected epoch, score, and history |
| `epoch_history.csv` | Train, validation, and test metrics at each epoch |
| `best_model_untouched_test_predictions.csv` | Protein IDs, true label, predicted probability, and thresholded label |
| `best_model_untouched_test_summary.json` | Final test metrics for the selected checkpoint |
| `best_model_untouched_test_roc_curve.csv` | False-positive rate, true-positive rate, and ROC thresholds |
| `best_model_untouched_test_threshold_sweep.csv` | Test metrics at thresholds 0–1 in steps of 0.001 |
| `best_model_untouched_test_TPR_at_fixed_untouched_test_TNR.csv` | TPR at target test TNR values 0.90, 0.95, 0.97, 0.98, and 0.99 |
| `loss_curve.png`, `accuracy_curve.png`, `auc_curve.png` | Epoch curves |
| `untouched_test_tpr_tnr_curve.png` | Epoch test TPR and TNR |
| `best_model_untouched_test_roc_curve.png` | Final test ROC |

TPR is recall on positive pairs; TNR is the fraction of negative pairs classified as negative. Threshold-sweep and fixed-TNR tables are descriptive evaluations on the test data. Their thresholds are selected from those test predictions and should be interpreted accordingly.

Only the best checkpoint is saved. The program does not aggregate the ten seeds into a single mean/SD report; per-seed outputs are retained for downstream analysis. Random seeds are set for NumPy, Python, and PyTorch, including CUDA when available. Exact numerical reproducibility also depends on the software and hardware environment recorded for the run.
