#!/usr/bin/env python3
"""Extract the original position and sequence-summary features into HDF5."""

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from alphabet_config import ALPHABETS, get_alphabet

BINS = 20
MAXLAG = 30


def all_n_mers(n, alphabet_size):
    return [
        "".join(p)
        for p in product([str(i) for i in range(1, alphabet_size + 1)], repeat=n)
    ]


def nmer_positions(sequence, n, alphabet_size):
    """Preserve overlapping searches and the original 20-bin counting rule."""
    counts_list = []
    for mer in all_n_mers(n, alphabet_size):
        counts = [0] * BINS
        start = 0
        while start < len(sequence):
            start_index = sequence.find(mer, start)
            if start_index == -1:
                break
            end_index = start_index + len(mer) - 1
            start_position = start_index / len(sequence)
            end_position = end_index / len(sequence)
            if start_position >= 1:
                start_position = 0.99999
            if end_position >= 1:
                end_position = 0.99999
            start_bin = int(BINS * start_position)
            end_bin = int(BINS * end_position)
            if start_bin == end_bin:
                counts[start_bin] += 1
            for i in range(start_bin, end_bin):
                counts[i] += 1
            start = start_index + 1
        if n == 1:
            total = sum(counts)
            if total != 0:
                counts = [10 * (i / total) for i in counts]
        counts_list.append(counts)
    return np.asarray(counts_list, dtype=np.float64 if n == 1 else np.int64)


def apaac(sequence, config, maxlag=MAXLAG):
    """Preserve lag products, placeholder filling, scaling, and normalization."""

    def calculate_h_values(sequence, H):
        sequence = list(map(int, sequence))
        h_values = []
        hdiff_matrix = []
        valid_maxlag = min(maxlag, len(sequence) - 1)
        unknown = int(config["unknown"])
        for d in range(1, valid_maxlag + 1):
            hdiff_in_lengths = [0] * BINS
            sum_of_h = 0
            blur = 0
            item = 0
            skip_initial_unknown = True
            for i in range(len(sequence) - d):
                if skip_initial_unknown and sequence[i] == unknown:
                    continue
                elif skip_initial_unknown:
                    skip_initial_unknown = False
                    blur = i
                if sequence[i] == unknown or sequence[i + d] == unknown:
                    blur += 1
                else:
                    h_val = H[sequence[i] - 1] * H[sequence[i + d] - 1]
                    item += 1
                    sum_of_h += h_val
            if item == 0:
                ave_h = 0
            else:
                sum_of_h = blur * (sum_of_h / item) + sum_of_h
                ave_h = sum_of_h / (blur + item)
            for k in range(len(sequence) - d):
                length_percentage = k / (len(sequence) - d)
                if length_percentage >= 1:
                    length_percentage = 0.99999
                if sequence[k] == unknown or sequence[k + d] == unknown:
                    h_val = ave_h
                else:
                    h_val = H[sequence[k] - 1] * H[sequence[k + d] - 1]
                hdiff_in_lengths[int(BINS * length_percentage)] += h_val
            denominator = max(len(sequence) - d, 1)
            h_values.append(sum_of_h / denominator)
            hdiff_matrix.append(
                [10 * (block / denominator) for block in hdiff_in_lengths]
            )
        while len(h_values) < maxlag:
            h_values.append(0)
            hdiff_matrix.append([0] * BINS)
        return h_values, hdiff_matrix

    H1_values, H1_matrix = calculate_h_values(sequence, np.asarray(config["H1"]))
    H2_values, H2_matrix = calculate_h_values(sequence, np.asarray(config["H2"]))
    w = 0.1
    denominator = sum(H1_values + H2_values) * w + 1
    return {
        "H1_features": np.asarray([10 * v for v in H1_values]),
        "H2_features": np.asarray([10 * v for v in H2_values]),
        "H1_matrix": np.asarray(H1_matrix),
        "H2_matrix": np.asarray(H2_matrix),
        "apaac1": np.asarray([100 * w * v / denominator for v in H1_values]),
        "apaac2": np.asarray([100 * w * v / denominator for v in H2_values]),
    }


def sequence_features(sequence, config):
    alphabet_size = len(config["groups"]) + 1
    features = {
        "proportion": np.asarray(
            [
                sequence.count(str(i)) / len(sequence)
                for i in range(1, alphabet_size + 1)
            ]
        ),
        "length": np.asarray(len(sequence) / 1000),
        "one_mer_counts": nmer_positions(sequence, 1, alphabet_size),
        "two_mer_counts": nmer_positions(sequence, 2, alphabet_size),
        "three_mer_counts": nmer_positions(sequence, 3, alphabet_size),
    }
    features.update(apaac(sequence, config))
    if not all(np.isfinite(value).all() for value in features.values()):
        raise ValueError(
            "Non-finite feature values; inspect the original APAAC denominator and input sequence"
        )
    return features


def dataset_key(name, side):
    return f"{name}_{side}" if name in ("proportion", "length") else f"{name}{side}"


def extract(input_path, output_path, alphabet, overwrite=False):
    if output_path.exists() and not overwrite:
        raise ValueError(f"Output exists: {output_path}; use --overwrite to replace it")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must be different files")
    config = get_alphabet(alphabet)
    data = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    required = ["proteinA", "proteinB", "Sequence A", "Sequence B", "interaction"]
    missing = [column for column in required if column not in data]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if data.empty:
        raise ValueError("Input has no pairs")
    for column in required:
        data[column] = data[column].str.strip()
        if (data[column] == "").any():
            raise ValueError(f"{column} contains empty values")
    if not data["interaction"].isin(("0", "1")).all():
        raise ValueError("interaction must contain only 0 or 1")
    allowed = set(str(i) for i in range(1, len(config["groups"]) + 2))
    for column in ("Sequence A", "Sequence B"):
        for i, sequence in enumerate(data[column]):
            if set(sequence) - allowed:
                raise ValueError(
                    f"{column}, data row {i + 1}: invalid codes for {alphabet}; encode with build_dataset.py first"
                )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write one pair at a time so the full feature arrays need not reside in RAM.
    temporary = output_path.with_name(output_path.name + ".partial")
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["alphabet"] = alphabet
            handle.attrs["alphabet_config"] = json.dumps(config)
            handle.attrs["input_csv_sha256"] = hashlib.sha256(
                input_path.read_bytes()
            ).hexdigest()
            handle.attrs["position_bins"] = BINS
            handle.attrs["maxlag"] = MAXLAG
            handle.attrs["feature_version"] = "chapter1-original-equations-v1"
            for column in ("proteinA", "proteinB"):
                ids = np.asarray(
                    [value.encode("utf-8") for value in data[column]], dtype="S"
                )
                handle.create_dataset(column, data=ids)
            handle.create_dataset(
                "interaction", data=data["interaction"].astype(np.int64).to_numpy()
            )
            n_rows = len(data)
            for i, (seqA, seqB) in enumerate(
                tqdm(
                    zip(data["Sequence A"], data["Sequence B"]),
                    total=n_rows,
                    desc=alphabet,
                )
            ):
                for side, sequence in (("A", seqA), ("B", seqB)):
                    for name, values in sequence_features(sequence, config).items():
                        key = dataset_key(name, side)
                        if i == 0:
                            dtype = (
                                np.int64
                                if name in ("two_mer_counts", "three_mer_counts")
                                else np.float64
                            )
                            handle.create_dataset(
                                key,
                                shape=(n_rows,) + values.shape,
                                dtype=dtype,
                                chunks=True,
                            )
                        handle[key][i] = values
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Saved {len(data)} pairs to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alphabet", choices=ALPHABETS, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        extract(args.input, args.output, args.alphabet, args.overwrite)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
