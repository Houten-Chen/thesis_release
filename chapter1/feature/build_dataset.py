#!/usr/bin/env python3
"""Encode a prepared protein-pair CSV using a selected reduced alphabet."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from alphabet_config import ALPHABETS, get_alphabet, translation_map

REQUIRED_COLUMNS = ("Entry A", "Entry B", "Letters_A", "Letters_B")


def encode_sequence(sequence, config):
    sequence = str(sequence).strip().upper()
    mapping = translation_map(config)
    return "".join(mapping.get(residue, config["unknown"]) for residue in sequence)


def build_dataset(input_path, output_path, alphabet, label, overwrite=False):
    if label not in (0, 1):
        raise ValueError("label must be 0 or 1")
    if output_path.exists() and not overwrite:
        raise ValueError(
            f"Output exists: {output_path}; choose another path or use --overwrite"
        )
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must be different files")
    config = get_alphabet(alphabet)
    data = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    missing = [column for column in REQUIRED_COLUMNS if column not in data]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if data.empty:
        raise ValueError("Input has no protein pairs")
    for column in REQUIRED_COLUMNS:
        data[column] = data[column].str.strip()
        empty = data.index[data[column] == ""].tolist()
        if empty:
            raise ValueError(
                f"{column}: empty values at CSV data rows {[i + 1 for i in empty[:10]]}"
            )
    for column in ("Letters_A", "Letters_B"):
        data[column] = data[column].str.upper()
        if data[column].str.contains(r"\d", regex=True).any():
            raise ValueError(
                f"{column} contains digits; supply amino-acid letters, not already encoded sequences"
            )
    result = pd.DataFrame(
        {
            "proteinA": data["Entry A"],
            "proteinB": data["Entry B"],
            "Sequence A": data["Letters_A"].map(
                lambda seq: encode_sequence(seq, config)
            ),
            "Sequence B": data["Letters_B"].map(
                lambda seq: encode_sequence(seq, config)
            ),
            "interaction": label,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    metadata = {
        "input_file": input_path.name,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "rows": len(result),
        "label": label,
        "alphabet_config": config,
        "output_columns": list(result.columns),
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(result)} encoded pairs to {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--alphabet", required=True, choices=ALPHABETS)
    parser.add_argument(
        "--label",
        required=True,
        type=int,
        choices=(0, 1),
        help="1 for positive pairs; 0 for negative pairs",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        build_dataset(
            args.input, args.output, args.alphabet, args.label, args.overwrite
        )
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
