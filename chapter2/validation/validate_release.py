#!/usr/bin/env python3
"""Check the shared release against archived originals and run a CPU example."""

import ast
import contextlib
import hashlib
import io
import json
from itertools import product
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "feature"))
sys.path.insert(0, str(ROOT / "training"))
from alphabet_config import ALPHABETS, get_alphabet
from build_dataset import build_dataset, encode_sequence
from extract_features import extract, sequence_features
import train

torch.set_num_threads(2)


def load_original_functions(path, include_h=False):
    """Load only definitions and H arrays; skip original top-level file I/O."""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            nodes.append(node)
        elif (
            include_h
            and isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id in ("H1", "H2") for t in node.targets
            )
        ):
            nodes.append(node)
    scope = {
        "np": np,
        "product": product,
        "nn": nn,
        "torch": torch,
        "ablation_constant": 0.0,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


class ReleaseChecks(unittest.TestCase):
    def test_original_file_checksums(self):
        manifest = json.loads((ROOT / "source_manifest.json").read_text())
        self.assertEqual(len(manifest), 15)
        for record in manifest:
            self.assertEqual(
                hashlib.sha256(
                    (ROOT / record["repository_path"]).read_bytes()
                ).hexdigest(),
                record["sha256"],
            )

    def test_encoding_and_numerical_features(self):
        for name in ALPHABETS:
            config = get_alphabet(name)
            prefix = config["source_prefix"]
            builder_path = (
                ROOT
                / f"feature/original_scripts/{prefix}_training_dataset_build_biogrid.py"
            )
            tree = ast.parse(builder_path.read_text(encoding="utf-8-sig"))
            mapping = next(
                ast.literal_eval(node.value)
                for node in tree.body
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "translation_dict"
                    for t in node.targets
                )
            )
            residues = "acdefghiklmnpqrstvwyXBZJUO-*"
            expected = "".join(
                mapping.get(letter, config["unknown"]) for letter in residues.upper()
            )
            self.assertEqual(encode_sequence(residues, config), expected)
            original = load_original_functions(
                ROOT
                / f"feature/original_scripts/{prefix}_feature_extraction_biogrid.py",
                include_h=True,
            )
            np.testing.assert_array_equal(original["H1"], config["H1"])
            np.testing.assert_array_equal(original["H2"], config["H2"])
            examples = [
                encode_sequence("ACDEFGHIKLMNPQRSTVWY" * 2, config),
                encode_sequence("XXACDXEFGXHIKLMNPQRSTVWYX", config),
                config["unknown"] * 12,
                "1",
                "123" * 20,
                "".join(str(i) for i in range(1, len(config["groups"]) + 2)) * 9,
            ]
            for sequence in examples:
                values = sequence_features(sequence, config)
                proportions, length = original["calculate_amino_acid_proportion"](
                    sequence
                )
                np.testing.assert_array_equal(values["proportion"], proportions)
                np.testing.assert_array_equal(values["length"], length)
                for n, field in (
                    (1, "one_mer_counts"),
                    (2, "two_mer_counts"),
                    (3, "three_mer_counts"),
                ):
                    np.testing.assert_array_equal(
                        values[field],
                        original["calculate_nmer_position_percentage"](sequence, n),
                    )
                lag_values = original["APAAC"](sequence, 30)
                for field, expected_values in zip(
                    (
                        "H1_features",
                        "H1_matrix",
                        "H2_features",
                        "H2_matrix",
                        "apaac1",
                        "apaac2",
                    ),
                    lag_values,
                ):
                    np.testing.assert_array_equal(values[field], expected_values)

    def test_views_ablations_splits_and_model(self):
        rng = np.random.default_rng(12)
        for name in ALPHABETS:
            config = get_alphabet(name)
            prefix = config["source_prefix"]
            original = load_original_functions(
                ROOT
                / f"training/original_scripts/{prefix}_training-pytorch_ver8.0_GPU_ablation.py"
            )
            k = len(config["groups"]) + 1
            raw = {
                "proteinA": np.asarray([f"A{i}" for i in range(12)], dtype="S"),
                "proteinB": np.asarray([f"B{i}" for i in range(12)], dtype="S"),
                "interaction": np.zeros(12, dtype=np.int64),
            }
            shapes = {
                "proportion": (k,),
                "length": (),
                "one_mer_counts": (k, 20),
                "two_mer_counts": (k**2, 20),
                "three_mer_counts": (k**3, 20),
                "H1_features": (30,),
                "H2_features": (30,),
                "apaac1": (30,),
                "apaac2": (30,),
                "H1_matrix": (30, 20),
                "H2_matrix": (30, 20),
            }
            for side in ("A", "B"):
                for field, shape in shapes.items():
                    key = (
                        f"{field}_{side}"
                        if field in ("proportion", "length")
                        else f"{field}{side}"
                    )
                    raw[key] = rng.normal(size=(12,) + shape)
            for ablations in (
                [],
                ["two_mer_counts", "H2_matrix", "apaac1", "proportion", "length"],
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    expected = original["build_feature_views"](raw, ablations)
                    actual = train.build_feature_views(raw, ablations)
                for key in actual:
                    np.testing.assert_array_equal(actual[key], expected[key])
            neg = train.build_feature_views(raw)
            pos = {key: value.copy() for key, value in neg.items()}
            pos["interaction"][:] = 1
            old_split = original["build_balanced_train_val_untouched_test_split"](
                neg, pos, np.random.default_rng(11111)
            )
            new_split = train.build_balanced_train_val_untouched_test_split(
                neg, pos, np.random.default_rng(11111)
            )
            self.assertEqual(old_split[3], new_split[3])
            for old_view, new_view in zip(old_split[:3], new_split[:3]):
                for key in new_view:
                    np.testing.assert_array_equal(old_view[key], new_view[key])
            for source in ("neg", "pos"):
                groups = [
                    set(new_split[3][f"{source}_{part}_indices"])
                    for part in ("train", "val", "untouched_test")
                ]
                self.assertFalse(
                    groups[0] & groups[1]
                    or groups[0] & groups[2]
                    or groups[1] & groups[2]
                )
            kwargs = {
                "pos_shape": neg["posA"].shape[1:],
                "sum_shape": neg["sumA"].shape[1:],
            }
            torch.manual_seed(123)
            old_model = original["PRIPredictorModelBTransformer"](**kwargs).eval()
            torch.manual_seed(123)
            new_model = train.PRIPredictorModelBTransformer(**kwargs).eval()
            for key, tensor in old_model.state_dict().items():
                torch.testing.assert_close(
                    tensor, new_model.state_dict()[key], rtol=0, atol=0
                )
            inputs = [
                torch.from_numpy(neg[key][:2])
                for key in ("posA", "posB", "sumA", "sumB")
            ]
            with torch.no_grad():
                torch.testing.assert_close(
                    old_model(*inputs), new_model(*inputs), rtol=0, atol=0
                )

    def test_csv_hdf5_and_cpu_training(self):
        with tempfile.TemporaryDirectory(prefix="chapter2_validation_") as directory:
            work = Path(directory)
            for alphabet in ALPHABETS:
                for label, source in ((1, "biogrid"), (0, "negatome")):
                    csv_path = work / f"{alphabet}_{source}.csv"
                    h5_path = work / f"{alphabet}_{source}.hdf5"
                    with contextlib.redirect_stdout(
                        io.StringIO()
                    ), contextlib.redirect_stderr(io.StringIO()):
                        result = build_dataset(
                            ROOT / f"examples/{source}_prepared_example.csv",
                            csv_path,
                            alphabet,
                            label,
                        )
                        extract(csv_path, h5_path, alphabet)
                    with h5py.File(h5_path, "r") as handle:
                        self.assertEqual(len(handle.keys()), 25)
                        self.assertEqual(handle.attrs["alphabet"], alphabet)
                        np.testing.assert_array_equal(
                            handle["interaction"][:], result["interaction"]
                        )
                        for side in ("A", "B"):
                            expected = sequence_features(
                                result[f"Sequence {side}"].iloc[0],
                                get_alphabet(alphabet),
                            )
                            for key, values in expected.items():
                                dataset = (
                                    f"{key}_{side}"
                                    if key in ("proportion", "length")
                                    else f"{key}{side}"
                                )
                                np.testing.assert_array_equal(
                                    handle[dataset][0], values
                                )
            command = [
                sys.executable,
                str(ROOT / "training/train.py"),
                "--negative",
                str(work / "rhp4_negatome.hdf5"),
                "--positive",
                str(work / "rhp4_biogrid.hdf5"),
                "--alphabet",
                "rhp4",
                "--output-root",
                str(work / "runs"),
                "--seeds",
                "0",
                "--epochs",
                "2",
                "--device",
                "cpu",
                "--cpu-threads",
                "2",
            ]
            run = subprocess.run(
                command, cwd=ROOT, text=True, capture_output=True, timeout=180
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            folders = list((work / "runs").iterdir())
            self.assertEqual(len(folders), 1)
            folder = folders[0]
            for name in (
                "config.json",
                "fixed_split_indices.json",
                "best_model.pt",
                "epoch_history.csv",
                "best_model_untouched_test_summary.json",
                "best_model_untouched_test_predictions.csv",
                "best_model_untouched_test_threshold_sweep.csv",
                "best_model_untouched_test_roc_curve.png",
            ):
                self.assertTrue((folder / name).is_file(), name)
            predictions = pd.read_csv(
                folder / "best_model_untouched_test_predictions.csv"
            )
            self.assertTrue(
                {"proteinA", "proteinB", "y_true", "pred_prob", "pred_label"}
                <= set(predictions.columns)
            )
            self.assertEqual(len(predictions), 4)
            history = pd.read_csv(folder / "epoch_history.csv")
            self.assertEqual(len(history), 2)
            sweep = pd.read_csv(
                folder / "best_model_untouched_test_threshold_sweep.csv"
            )
            self.assertEqual(len(sweep), 1001)

    def test_invalid_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            missing = directory / "missing.csv"
            missing.write_text("Entry A,Entry B\nA,B\n")
            with self.assertRaises(ValueError):
                build_dataset(missing, directory / "out.csv", "hp2", 1)
            malformed = directory / "malformed.csv"
            malformed.write_text(
                "proteinA,proteinB,Sequence A,Sequence B,interaction\nA,B,999,111,1\n"
            )
            with self.assertRaises(ValueError):
                extract(malformed, directory / "out.hdf5", "hp2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
