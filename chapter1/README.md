# Chapter 1 — PPI prediction with reduced amino-acid alphabets

This release organizes the 15 supplied scripts into a shared dataset builder, feature extractor, and PyTorch training program. The five alphabet mappings and their numerical H1/H2 vectors are stored separately in `feature/alphabets.json`. The original files are retained byte-for-byte in the two `original_scripts/` directories and listed in `source_manifest.json`.

## Contents

| Location | Purpose |
| --- | --- |
| `feature/build_dataset.py` | Convert prepared amino-acid pair tables to reduced-alphabet pair tables |
| `feature/extract_features.py` | Write the original sequence features to HDF5 |
| `feature/alphabets.json` | Exact residue groups, placeholder codes, H1/H2 values, and alphabet-specific learning rates |
| `feature/README.md` | Input schemas, alphabet definitions, feature equations, and HDF5 layout |
| `training/train.py` | Pair-token Transformer, balanced row splits, ablations, and evaluation outputs |
| `training/README.md` | Model, defaults, split procedure, and output definitions |
| `examples/` | Synthetic database-format examples; see `examples/README.md` |
| `validation/validate_release.py` | Regression checks against the archived originals and a small CPU workflow test |
| `source_manifest.json` | Original filenames, repository locations, and SHA-256 checksums |

## Supported alphabets

The numerical codes follow the uploaded scripts. The additional placeholder is included in composition and n-mer features, so the feature alphabet contains one more symbol than the nominal biological class count.

| `--alphabet` | Original filename prefix | Residue classes | Placeholder | Position view per protein | Summary view per protein | Default learning rate |
| --- | --- | --- | --- | --- | --- | --- |
| `hp2` | `2_` | 2 | `3` | 99 × 20 | 124 | 1e-5 |
| `li3` | `3_` | 3 | `4` | 144 × 20 | 125 | 5e-6 |
| `rhp4` | `4_` | 4 | `5` | 215 × 20 | 126 | 5e-6 |
| `wang5` | `5_` | 5 | `6` | 318 × 20 | 127 | 5e-6 |
| `aa7` | `7_` | 7 | `8` | 644 × 20 | 129 | 5e-6 |

Group membership and all H1/H2 numbers are documented in [feature/README.md](feature/README.md). The scheme names identify the uploaded implementations; the mapping table is authoritative for this release.

## Installation

Use Python 3.10 or later. From the `chapter1` directory:

```bash
python -m pip install -r requirements.txt
```

The training program uses CUDA when available and otherwise uses the CPU. For a specific CUDA installation, use the PyTorch package appropriate to that environment.

## Run the format examples

The bundled sequences, IDs, and labels are synthetic. They demonstrate the file interfaces and exercise the code; they are not BioGRID or Negatome records and do not reproduce dissertation performance.

First encode the prepared positive and negative example tables using the same alphabet.

```bash
python feature/build_dataset.py --input examples/biogrid_prepared_example.csv --output work/rhp4_positive.csv --alphabet rhp4 --label 1
python feature/build_dataset.py --input examples/negatome_prepared_example.csv --output work/rhp4_negative.csv --alphabet rhp4 --label 0
```

Then extract both feature files.

```bash
python feature/extract_features.py --input work/rhp4_positive.csv --output work/rhp4_positive.hdf5 --alphabet rhp4
python feature/extract_features.py --input work/rhp4_negative.csv --output work/rhp4_negative.hdf5 --alphabet rhp4
```

Run a short CPU check with one seed and two epochs.

```bash
python training/train.py --positive work/rhp4_positive.hdf5 --negative work/rhp4_negative.hdf5 --alphabet rhp4 --output-root runs/example_rhp4 --seeds 0 --epochs 2 --device cpu --cpu-threads 2
```

Paths containing spaces should be quoted. The preparation and extraction programs preserve existing outputs unless `--overwrite` is supplied. Training creates a separate timestamped directory for each seed.

To use another alphabet, change `--alphabet` in all three stages and use separate output paths. Feature files from different alphabets cannot be mixed.

## Inputs for the research dataset

The dataset builder expects a **prepared sequence-pair CSV**, not an unmodified database download. Both positive and negative tables require these four columns:

| Column | Contents |
| --- | --- |
| `Entry A` | Protein A identifier |
| `Entry B` | Protein B identifier |
| `Letters_A` | Protein A amino-acid sequence |
| `Letters_B` | Protein B amino-acid sequence |

The original BioGRID scripts read `BIOGRID_schemeA_reduced_with_letters_dedup_len30.csv`. Sequence retrieval, identifier matching, biological interaction filtering, duplicate removal, and the upstream length filter used to prepare that file are outside the uploaded scripts. Apply the same preparation to the research inputs used for reproduction. The shared builder preserves input row order and performs alphabet encoding; it does not repeat those upstream filters.

Use `--label 1` for prepared positive pairs and `--label 0` for prepared negative pairs. The added negative-table entry point uses the same mapping and feature equations as the positive-table entry point. If negative pairs have already been encoded, use the five-column reduced schema described in [feature/README.md](feature/README.md) and proceed directly to feature extraction.

## Training with the original defaults

With the actual research feature files:

```bash
python training/train.py --positive data/biogrid_rhp4.hdf5 --negative data/negatome_rhp4.hdf5 --alphabet rhp4 --output-root runs/rhp4
```

This runs 24 epochs for each of the ten original seeds. It retains the supplied model, loss, optimizer, feature concatenation, and balanced interaction-row splitting procedure. The default ablation list is empty, so both position and summary features are used. See [training/README.md](training/README.md) for the full defaults and evaluation details.

The full databases, precomputed research features, trained research weights, and dissertation result tables are not included in the uploaded material or this release.

## Validation and provenance

Run the bundled checks from `chapter1`:

```bash
python validation/validate_release.py
```

The checks compare residue encoding, numerical features, feature views, ablation blocks, split indices, model architecture, and model outputs with the archived scripts for all five alphabets. They also exercise CSV preparation, HDF5 extraction, checkpoint saving/loading, and evaluation on the synthetic examples.

The refactor replaces private Windows paths with parameters, removes interactive exit prompts, writes features incrementally to HDF5, adds input/configuration checks, records alphabet metadata and software versions, and includes protein IDs in prediction tables. Original equations and valid-input defaults are retained. The archived scripts preserve the original implementation and its comments, paths, and prompts.
