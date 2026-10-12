# Chapter 3 — Protein-derived sequence space and RHP design

This release turns the eight uploaded analysis scripts into explicit command-line steps. It covers sequence encoding, upstream-model inference, fixed PCA projection, KDE visualization, HLB lag statistics, Q3 structure-tendency maps, RHP overlap scores, and pseudo-block controls. The uploaded files are preserved byte-for-byte in `original_scripts/`; their SHA-256 checksums are recorded in `source_manifest.json`.

## Upstream projects and attribution

| Resource | Role in this workflow | Original source |
| --- | --- | --- |
| Shuni Li's AE-RHP / DeepRHP-related implementation | `AE_MP` sequence encoder and one-hot convention used by the uploaded scripts | [Shunili/AE-RHP](https://github.com/Shunili/AE-RHP) |
| ESM2 | Layer-33, 1280-dimensional residue representations for Q3 prediction | [facebookresearch/esm](https://github.com/facebookresearch/esm) |
| Porter6 | The `predictor3` classifier converting ESM2 representations to H/E/C labels | [WafaAlanazi/Porter6](https://github.com/WafaAlanazi/Porter6) |
| Ivan Jayapurna and collaborators' RHPapp | Polymerization simulation generating the standard RHP sequence ensembles | [ivanjayapurna/RHPapp](https://github.com/ivanjayapurna/RHPapp) |
| UniProtKB and MobiDB | Protein sequences and disorder-related reference populations | [UniProtKB](https://www.uniprot.org/) and [MobiDB](https://mobidb.org/) |

The upstream implementations, model weights, and standard RHP simulator remain with their authors. This directory contains the chapter-specific preparation, inference adapters, analyses, and pseudo-block control. The `pseudo-block` command implements the uploaded control; standard polymerization simulation is performed with RHPapp.

The public AE-RHP repository identifies itself as supplementary software for Ruan et al., *Population-based heteropolymer design to mimic protein mixtures* (Nature, 2023), and contains the `AE_MP` architecture actually imported by the uploaded scripts. Li et al., *DeepRHP: A Hybrid Variational Autoencoder for Designing Random Heteropolymers as Protein Mimics* ([arXiv:2606.11651](https://arxiv.org/abs/2606.11651)), describes the related embedding framework. This adapter specifically loads `models/AE_MP.py`; it does not assume that checkpoints from other DeepRHP architectures are interchangeable. For standard sequence simulation, cite Jayapurna et al., *Sequence Design of Random Heteropolymers as Protein Mimics* ([Biomacromolecules, 2023](https://doi.org/10.1021/acs.biomac.2c01036)). Follow the ESM2 and Porter6 repositories' citation guidance when using their predictions.

Upstream source inspected while preparing this release was AE-RHP commit `ca739439a716153741fc59314b2bcd73eb538581`, Porter6 commit `90c3dd9489ea1acbbfd94061e4d171af21b86fbf`, and RHPapp commit `2131549ed8eeedef4c9cb5e8acd6bc20278e947d`. These are inspection references, not a claim that the dissertation used these exact checkouts.

## Layout and original-script mapping

| Uploaded script | Reusable command |
| --- | --- |
| `protein_convert-notail.py` | `windows` |
| `generate_z_row_for_protein.py` | `embed --kind protein` |
| `generate_z_row_for_RHP.py` | `embed --kind rhp` |
| `PCA_2D_KDE_visualize_by_file_protein_groups_with_RHP_2of5.py` | `fit-pca` and `plot-kde` |
| `Batch_prediction.py` | `predict-q3` |
| `self-correlation heatmap.py` | `lag-map` |
| `RHP structure prediction(1).py` | `structure-map` and `score-rhp` |
| `pseudo-block_generator_-ver2.py` | `pseudo-block` |

`chapter3.py` is the entry point. Shared functions are in `ch3lib/`. `config/ae_model.json` records the uploaded architecture settings. `config/study_groups.template.json` gives the chapter's protein groups and five RHP colors, with explicit reference-selection flags. `examples/` contains synthetic format examples. `validation/` contains regression checks and the release report. `DATA_AND_CODE_AVAILABILITY.md` supplies the proposed thesis Section 3.8.

## Install and run a small example

Use Python 3.10 or later. Run the commands below from this `chapter3` directory. Core analysis and plotting do not require neural-network weights.

```bash
python -m pip install -r requirements.txt
python chapter3.py --help
python chapter3.py fit-pca --groups examples/groups.json --output work/example_pca.json
python chapter3.py plot-kde --groups examples/groups.json --pca work/example_pca.json --output-dir work/example_kde
python chapter3.py lag-map --input examples/protein_cytoplasm_z.csv --pca work/example_pca.json --min-points 1 --output-dir work/example_lag
python chapter3.py structure-map --input examples/protein_cytoplasm_z.csv --pca work/example_pca.json --min-points 1 --output-dir work/example_structure
python chapter3.py score-rhp --input examples/rhp_z.csv --pca work/example_pca.json --map work/example_structure/structure_map.csv --output-dir work/example_scores
python chapter3.py pseudo-block --output work/example_pseudo.csv --count 20
```

Each analysis writes CSV results and JSON metadata; plots are saved as PNG and PDF. Outputs are protected from replacement by default. Add `--overwrite` for an intentional rerun. Paths containing spaces must be quoted. Use `python chapter3.py COMMAND --help` for every option. Plot fonts default to Arial when available and otherwise use DejaVu Sans; `--font` selects another installed font. Plot limits follow the data unless supplied with `--xlim` and `--ylim`.

For heatmap colors, `--value-limits MIN MAX`, `--entropy-limits MIN MAX`, and `--cmap NAME` expose the originally hand-edited settings. Original color limits are retained by default: 0–0.15 for lag variance, 3.388–3.395 for entropy, and −1–1 for structure tendency. Change the limits when inspecting data outside those ranges.

The example sequences, Q3 labels, and latent vectors are synthetic. They demonstrate the interfaces without downloading models or reproducing biological findings. See [examples/README.md](examples/README.md).

## Research inputs and model setup

For the chapter's reference space, prepare the MobiDB sequences/regions of length at least 50, the reviewed UniProtKB set with annotation score at least 4, and the Cytoplasm and Membrane subsets described in the thesis. Dataset retrieval, annotation filtering, redundancy filtering, and the standard RHP simulation recipes are upstream inputs to this release. Preserve the database versions, sequence identifiers, and actual filtering/simulation settings alongside a research run.

The supplied files do not contain the study's trained `checkpoint_041.pkl`, original PCA mean/loading matrix, complete protein tables, or simulated research ensembles. Exact figure reproduction requires those assets. The embedding scripts reference an `AE_MP` checkpoint under `N=4_MP_de2concat`; obtaining or training the same architecture does not establish that it is the same trained encoder. Refit PCA only when intentionally defining a new reference space.

Install optional inference dependencies and obtain the upstream implementations separately:

```bash
python -m pip install -r requirements-inference.txt
git clone https://github.com/Shunili/AE-RHP.git ../AE-RHP
git clone https://github.com/ivanjayapurna/RHPapp.git ../RHPapp
```

Obtain Porter6 from its repository and extract `porter6.zip`. The Q3 adapter needs the `predictor3/model/cbrcnn_stage1.py` file and `predictor3/output/trained_model/esm2/cbrcnn_stage1_f3.pth` checkpoint. Use the extracted source-file path directly; this avoids assumptions about how many nested `porter6` directories were created. Obtain `esm2_t33_650M_UR50D.pt` and the companion contact-regression weights required by the upstream ESM loader from the official ESM repository. Neural-network inference can use a CPU or GPU; `--device auto` selects CUDA when available.

## Prepare standard RHP sequences

Generate standard RHP ensembles with Ivan's RHPapp using the actual feed composition, reactivity ratios, chain count, conversion, and other simulation parameters. The inspected simulator's `exportPolymerArray` writes headerless CSV rows containing one monomer ID per cell. Convert that export with `import-rhpapp`, explicitly declaring which source IDs correspond to the four chemistries.

For example, if a simulation uses `1=MMA, 2=EHMA, 3=OEGMA, 4=SPMA`, the adapter command is:

```bash
python chapter3.py import-rhpapp --input data/rhpapp_export.csv --source-codes 1 3 2 4 --output data/RHP_chains.csv
```

`--source-codes` is always given in **MMA, OEGMA, EHMA, SPMA** order. Set it to the IDs used in the actual simulation. The resulting `Sequence` strings use the target encoding below. A prepared CSV already containing a `Sequence` column in this encoding can be used directly.

| Target code | Monomer | Protein residues | HLB used for lag calculations |
| --- | --- | --- | --- |
| `1` | MMA | A, C, Y, T, G | 8.45 |
| `2` | OEGMA | S, Q, H, N, P | 11.42 |
| `3` | EHMA | L, I, F, W, V, M | 5.125 |
| `4` | SPMA | E, D, R, K | 18.5 |

The target ordering also applies to `--composition` in the pseudo-block command. It differs from the common shorthand MMA/EHMA used in RHP composition labels. Thus RHP4, labeled RHP(50/20), uses `--composition 50 25 20 5`.

## Q3 prediction and sequence embedding

Protein input tables require a `Sequence` column containing amino-acid strings. `Entry` is useful for tracking records; other columns are retained. CSV and Excel inputs are supported. Q3 output is a CSV with a per-residue `Q3` string and an error column.

```bash
python chapter3.py predict-q3 --input data/proteins.csv --esm-checkpoint checkpoints/esm2_t33_650M_UR50D.pt --porter-source ../porter6/predictor3/model/cbrcnn_stage1.py --porter-checkpoint checkpoints/cbrcnn_stage1_f3.pth --chunk-size 1022 --output-dir work/q3
python chapter3.py embed --kind protein --input work/q3/proteins_q3.csv --ae-repo ../AE-RHP --checkpoint checkpoints/checkpoint_041.pkl --output-dir work/protein_z
python chapter3.py embed --kind rhp --input data/RHP_chains.csv --ae-repo ../AE-RHP --checkpoint checkpoints/checkpoint_041.pkl --output-dir work/rhp_z
```

The uploaded Q3 script uses ESM2 layer 33, 1280-dimensional representations, and one Porter6 `predictor3` checkpoint with label order `0=H, 1=E, 2=C`. This release retains that model path rather than replacing it with the full upstream Porter6 ensemble. The original chunk size is 9,500 residues. The command above demonstrates a smaller chunk size for memory-limited inference; chunk boundaries change sequence context and can change predictions. Record the size actually used. Prediction errors stop the run by default; `--on-error record` writes `ERROR` and its reason. Such rows must be resolved before constructing a structure map.

Both embedding modes use complete 50-mers with a default step of 15, a 5 × 51 one-hot input including the termination channel, and a 16-dimensional latent vector. Batches are bounded by `--batch-size`, default 1,000. Protein rows with no complete window remain in the output with empty latent cells. Short RHP rows are omitted, matching the uploaded scripts. Whitespace is removed and protein letters are uppercased. Noncanonical amino acids are rejected by default; `--unknown-policy mma` explicitly restores the original assignment of unrecognized residues to code `1`.

The uploaded standalone converter defaults to 100-mers despite printing “50-mer.” `windows` retains its actual default and reports the requested length correctly. It is an optional sequence-window export; the embedding commands read full sequences directly.

```bash
python chapter3.py windows --input data/proteins.csv --window 100 --step 15 --output work/protein_100mers.csv
```

## Keep one PCA reference

Every projection or map command requires `--pca`. Missing parameters cause an error; analysis never silently fits a replacement. Legacy PCA JSON files with `mean` and `components` are accepted. The mean must have length 16 and every component must have length 16. Projection is `(z - mean) @ components.T`, retaining the first two components.

For the study's coordinate system, use its original PCA parameters and original encoder checkpoint. When intentionally making a new protein-derived reference, fit it explicitly to the selected protein datasets:

```bash
python chapter3.py fit-pca --input data/z_membrane.csv data/z_cytoplasma.csv data/z_reviewed_annotation_4_5.csv data/z_IDR_sequences_over50.csv --output work/pca_reference.json
```

Alternatively, edit `config/study_groups.template.json` to point to the actual inputs and run `fit-pca --groups`. Only entries with `fit_reference=true` are fitted; RHP groups are projected into that reference. This makes the reference pool explicit. The original plotting script pooled every loaded group, including RHPs and Fusion peptide, if its saved PCA file was absent. That fallback is removed here so the reference does not depend on which overlay files happen to be present.

Use the same group configuration to plot the populations:

```bash
python chapter3.py plot-kde --groups config/study_groups.template.json --pca work/pca_reference.json --output-dir work/sequence_space --xlim -57 65 --ylim -42 42
```

Group paths are relative to the configuration file. `kind`, `fit_reference`, `plot`, `color`, `display_label`, `fill`, and `bw_adjust` configure each group without editing Python. All valid window vectors are used; no random shuffle or subsampling is performed. The default KDE threshold is 0.04. Degenerate groups with insufficient independent points are displayed as points.

## Lag maps, structure maps, and RHP overlap

```bash
python chapter3.py lag-map --input work/protein_z/z_proteins_q3.csv --pca work/pca_reference.json --output-dir work/lag
python chapter3.py structure-map --input work/protein_z/z_proteins_q3.csv --pca work/pca_reference.json --output-dir work/structure
python chapter3.py score-rhp --input work/rhp_z/z_RHP_chains.csv --pca work/pca_reference.json --map work/structure/structure_map.csv --output-dir work/overlap
```

Wide embedding CSVs use `z_0`, `z_1`, ... columns, each containing a serialized list of 16 numbers. The suffix is the actual window index. For example, `z_2` begins at residue 30 when the step is 15, even if `z_1` is absent or columns have been reordered. Blank cells are skipped. Nonfinite or malformed vectors are rejected. The structure map additionally requires a valid `Q3` string with the same length as `Sequence`.

The lag profile is the mean absolute HLB difference at each separation n=1...30. The default grid spacing is 2 PCA units and the lag map requires at least 10 windows per bin. Output statistics distinguish the aggregation order:

| CSV field | Definition |
| --- | --- |
| `variance_of_mean_lag_profile` / legacy `var_mean_after` | First average the lag profiles in a bin, then take population variance over the 30 mean lag values; this is the uploaded plotted “Lag variance” |
| `mean_lag_entropy` / legacy `var_point_first` | Mean Shannon entropy of individually normalized lag profiles, using natural logarithms and the uploaded 1e-12 epsilon |
| `mean_segment_lag_variance` | Mean of the individual windows' lag variances; an additional explicit statistic, separate from the original plotted value |
| `lag_1` ... `lag_30` | Bin-mean absolute HLB differences |

Zero-difference profiles have undefined entropy and are excluded from the entropy mean. The uploaded names of the two legacy fields are retained for compatibility. In particular, `var_point_first` contains entropy, not variance. The thesis method wording should distinguish the variance of a bin-mean profile from the bin mean of per-window variances.

The structure map requires at least 20 protein windows per bin. H, E, and C fractions are averaged within a bin, then the uploaded score is calculated as `10 * ln((mean_H + mean_E) / (mean_C + 1e-12))`. Pure-coil bins have a score of negative infinity; unsupported bins have missing values. Neither is a finite score for RHP overlap. A positive score expresses combined H/E enrichment relative to C; this scalar alone does not separate helix from sheet. The three fraction fields remain available for that distinction.

`score-rhp` reports the empirical window-weighted mean of finite protein-derived grid scores and the fraction of eligible RHP windows that fall in finite supported bins. It also saves every window's coordinates and assigned score. These are protein-derived statistical scores applied to RHP sequences, not predictions that a synthetic polymer forms protein secondary structure. The default includes all RHP windows, as in the uploaded script. `--pc2-limit VALUE` explicitly restricts eligibility to `|PC2| <= VALUE`; the report then includes `N_points`, `N_region`, and `N_scored` so the denominator remains visible.

Grid sidecars store lower edges, spacing, and dimensions. Legacy CSV-only maps are reconstructed from their center coordinates, sorted before reshaping, and matched to the same cell edges. This fixes the original cached-map half-bin shift. Points exactly on a previous maximum grid boundary are retained in a final cell, avoiding dropped edge points. The variable-name errors that prevented uncached lag-map recomputation are also corrected. The residue mapping, HLB values, lag equations, Q3 formula, and pseudo-block stochastic rules are retained.

## Pseudo-block controls

```bash
python chapter3.py pseudo-block --output work/RHP4_pseudo.csv --composition 50 25 20 5 --count 10000 --length 150 --seed 777
```

The default algorithm matches the uploaded version. Per-chain monomer counts fluctuate independently by ±30% about their expected counts, with no renormalization of chain length. The requested length is therefore nominal. Hydrophobic block bodies use codes 1/3 with a forced 3 after two consecutive 1s; hydrophilic block bodies use codes 2/4. Original sealing and leftover-token rules are retained. Chain i uses seed `seed + i`, starting at i=0. The main CSV contains `Sequence`; a companion `.chains.csv` records lengths, block counts, and monomer counts. Generation streams to disk and prints one summary rather than a line for every chain.

## Validation

```bash
python validation/validate_release.py
```

Checks cover all eight original checksums, canonical residue mapping, complete-window positions, lag profiles and entropy, Q3 fractions, aggregation order, numeric window-column alignment, grid boundaries, cached-map geometry, score coverage, exact pseudo-block sequences for 60 chains across the five compositions, output guards, and the full synthetic CPU workflow. Core validation skips optional upstream-model checks unless their source paths are supplied.

```bash
python validation/validate_release.py --ae-repo ../AE-RHP --porter-source ../porter6/predictor3/model/cbrcnn_stage1.py --report work/validation_report.json
```

The release was also checked against the inspected upstream AE_MP implementation with newly initialized fixture weights, including one-hot equality and batching agreement. The Porter6 check uses its actual `predictor3` Net with fixture weights and mocked ESM representations to verify tensor orientation, chunk handling, and one-residue outputs. These checks validate adapters, not the dissertation's trained-model predictions. The exact test environment and limitations are recorded in `validation/validation_report.json`.

An additional test loads a small real ESM2 fixture through fair-esm, including Namespace checkpoint metadata and the contact-regression companion file. The adapter scopes the Namespace allowance needed by recent PyTorch versions to this load operation.
