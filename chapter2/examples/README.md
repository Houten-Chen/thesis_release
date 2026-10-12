# Database-format examples

All rows in this directory are **synthetic format examples**. IDs, sequences, and interaction labels were created for exercising the file interfaces. They are not downloaded BioGRID or Negatome records, and their model metrics have no biological interpretation.

| File | Format | Rows |
| --- | --- | --- |
| `biogrid_prepared_example.csv` | Four-column prepared amino-acid pair table; encode with `--label 1` | 12 |
| `negatome_prepared_example.csv` | Same prepared format; encode with `--label 0` | 12 |
| `encoded_rhp4_example.csv` | Five-column reduced pair table, with a few rows from each example class | 6 |

The prepared headers are exactly `Entry A,Entry B,Letters_A,Letters_B`. The reduced headers are exactly `proteinA,proteinB,Sequence A,Sequence B,interaction`. The examples include X placeholders in some sequences to exercise unknown-residue handling.

The Chapter 2 README gives commands that turn the two prepared tables into separate positive/negative CSV and HDF5 files, then run a short CPU training check. Larger research databases should provide the same required columns after identifier matching and sequence preparation.
