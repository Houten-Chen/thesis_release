# Dataset preparation and feature extraction

`build_dataset.py` encodes amino-acid sequences. `extract_features.py` calculates the original numerical features for both proteins in each pair. Both programs require an explicit `--alphabet`.

## Prepared database input

The prepared BioGRID-positive and Negatome-negative CSVs use the same schema.

| Required column | Type | Example |
| --- | --- | --- |
| `Entry A` | Nonempty protein identifier | `SYN_POS_A01` |
| `Entry B` | Nonempty protein identifier | `SYN_POS_B01` |
| `Letters_A` | Nonempty amino-acid sequence | `ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQ` |
| `Letters_B` | Nonempty amino-acid sequence | `MKWVTFISLLFLFSSAYSACDEFGHIKLMNPQRS` |

Column names are case-sensitive. Additional columns are ignored. Sequences are stripped of surrounding whitespace and converted to uppercase. The groups cover the 20 standard amino acids. Characters outside those groups, including X, B, and Z, map to the alphabet's placeholder. Empty IDs or sequences cause an error.

The supplied database builders assigned all BioGRID rows label 1. The shared builder makes that source label explicit and also accepts 0 for prepared negative pairs. It writes UTF-8 CSV with a BOM and a `<output>.csv.json` sidecar containing the input checksum, label, alphabet configuration, and row count.

## Reduced pair-table schema

The feature extractor requires the following columns:

| Required column | Type | Meaning |
| --- | --- | --- |
| `proteinA` | String | Protein A identifier |
| `proteinB` | String | Protein B identifier |
| `Sequence A` | String of reduced-alphabet digits | Protein A encoded sequence |
| `Sequence B` | String of reduced-alphabet digits | Protein B encoded sequence |
| `interaction` | 0 or 1 | Negative or positive pair label |

For `rhp4`, `ACDEFGHIKLMNPQRSTVWY` becomes `11443123433222421331`. The same input has a different encoding under each other alphabet. Only digits belonging to the selected alphabet, including its placeholder, are accepted. Sequences are read as strings to preserve the complete digit sequence.

`examples/encoded_rhp4_example.csv` illustrates this schema. Its mixed labels demonstrate the column format; the training command takes separate positive and negative HDF5 files.

## Numerical feature definition

The feature equations follow the uploaded extraction functions. Their scaling and treatment of unknown residues are retained.

- **Composition and length.** Composition includes every reduced symbol, including the placeholder, divided by sequence length. Length is divided by 1000.
- **Position-resolved n-mers.** All 1-, 2-, and 3-mers over the reduced symbols are listed in Cartesian-product order. Occurrences are found with overlapping searches and assigned to 20 relative-position bins using the original start/end rule. For each 1-mer, its bin counts are normalized to sum to 10 when that symbol occurs. The 2- and 3-mer features retain counts. For a multi-bin occurrence, bins from the start bin up to, but excluding, the end bin are incremented, as in the original code.
- **H1/H2 lag features.** For lags 1–30, the code computes products of the corresponding H values. The original placeholder-filling rule is retained, including its treatment of leading placeholder residues. The global vectors and 20-bin lag matrices are scaled by 10. Lags beyond a sequence's valid range are padded with zeros.
- **APAAC arrays.** With `w = 0.1`, the original denominator is `1 + w * sum(H1_lag_values + H2_lag_values)`. Each array contains `100 * w * lag_value / denominator`. These are the uploaded implementation's features; no external APAAC package is substituted.

H1/H2 values are copied from each corresponding source file. In particular, the RHP four-class vectors have their own numerical values and are not replaced with the group averages used by the other variants. All values and residue mappings are in `alphabets.json` and the tables below.

## HDF5 interface

Let `N` be the number of pairs and `K` the number of residue classes plus the placeholder. The extractor writes the same 25 dataset keys as the originals.

| Dataset key(s) | Shape | Contents |
| --- | --- | --- |
| `proteinA`, `proteinB` | `(N,)` | UTF-8-encoded identifiers |
| `interaction` | `(N,)` | Integer labels |
| `proportion_A`, `proportion_B` | `(N, K)` | Symbol fractions |
| `length_A`, `length_B` | `(N,)` | Sequence length / 1000 |
| `one_mer_countsA`, `one_mer_countsB` | `(N, K, 20)` | Scaled 1-mer position profiles |
| `two_mer_countsA`, `two_mer_countsB` | `(N, K², 20)` | 2-mer position counts |
| `three_mer_countsA`, `three_mer_countsB` | `(N, K³, 20)` | 3-mer position counts |
| `H1_featuresA`, `H1_featuresB`, `H2_featuresA`, `H2_featuresB` | `(N, 30)` | Global scaled lag features |
| `apaac1A`, `apaac1B`, `apaac2A`, `apaac2B` | `(N, 30)` | APAAC arrays |
| `H1_matrixA`, `H1_matrixB`, `H2_matrixA`, `H2_matrixB` | `(N, 30, 20)` | Position-resolved lag matrices |

The file attributes record `alphabet`, `alphabet_config`, `input_csv_sha256`, `position_bins`, `maxlag`, and `feature_version`. Features are written one pair at a time, preserving numerical values while avoiding storage of all feature arrays in RAM during extraction. A `.partial` file is replaced atomically on successful completion.

The training loader checks the dataset dimensions and finite values. For new HDF5 files, it also checks the stored mapping and H1/H2 configuration. Legacy HDF5 files with the original 25 keys and correct dimensions can be used; their mapping and numerical vectors must correspond to the selected alphabet.

## Commands

```bash
python feature/build_dataset.py --input examples/biogrid_prepared_example.csv --output work/positive.csv --alphabet hp2 --label 1
python feature/extract_features.py --input work/positive.csv --output work/positive.hdf5 --alphabet hp2
```

Run these commands from `chapter2`. Use `--help` for all options. The files under `original_scripts/` retain the uploaded implementations; they contain the original paths and may require manual edits and an interactive exit response.

## Exact alphabet configurations

The following tables are generated from the released `alphabets.json`.

### hp2 — HP2-style

| Code | Amino acids |
| --- | --- |
| `1` | C, F, I, L, M, V, W, Y |
| `2` | A, G, P, S, T, D, E, H, K, N, Q, R |
| `3` | Other characters (placeholder) |

H1 in group order: `[2.1875, -2.28]`.

H2 in group order: `[-1.56, 0.24]`.

### li3 — Li3-style

| Code | Amino acids |
| --- | --- |
| `1` | C, F, Y, W, M, L, I, V |
| `2` | G, P, A, T, S |
| `3` | N, H, Q, E, D, R, K |
| `4` | Other characters (placeholder) |

H1 in group order: `[2.1875, -0.34, -3.66]`.

H2 in group order: `[-1.56, -0.25, 0.59]`.

### rhp4 — RHP four-class

| Code | Amino acids |
| --- | --- |
| `1` | C, Y, A, T, G |
| `2` | S, Q, H, N, P |
| `3` | L, I, F, W, V, M |
| `4` | E, D, R, K |
| `5` | Other characters (placeholder) |

H1 in group order: `[-0.512233954, -0.38383707, 1.710733781, -0.814662758]`.

H2 in group order: `[-0.959054378, 1.359590767, -0.959054378, 0.558517989]`.

### wang5 — Wang5-style

| Code | Amino acids |
| --- | --- |
| `1` | C, M, F, I, L, V, W, Y |
| `2` | A, H, T |
| `3` | G, P |
| `4` | D, E |
| `5` | N, Q, R, K, S |
| `6` | Other characters (placeholder) |

H1 in group order: `[2.1875, -0.7, -1.0, -3.5, -3.24]`.

H2 in group order: `[-1.56, -0.23, -0.36, 0.71, 0.57]`.

### aa7 — Seven-group

| Code | Amino acids |
| --- | --- |
| `1` | A, G, V |
| `2` | I, L, F, P |
| `3` | Y, M, T, S |
| `4` | H, N, Q, W |
| `5` | R, K |
| `6` | D, E |
| `7` | C |
| `8` | Other characters (placeholder) |

H1 in group order: `[1.8667, 2.375, -0.225, -2.775, -4.2, -3.5, 2.5]`.

H2 in group order: `[-0.51, -1.5, -0.6, -0.39, 1.0, 0.71, -1.54]`.
