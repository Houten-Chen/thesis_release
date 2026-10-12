#!/usr/bin/env python3
"""Regression checks, boundary checks, CLI workflow, and optional upstream adapter checks."""
import argparse
import ast
from collections import Counter
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import typing
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ch3lib import analysis, common
from ch3lib.blocks import generate_chain

AE_REPO = PORTER_SOURCE = None


def original(filename):
    tree = ast.parse((ROOT / "original_scripts" / filename).read_text(encoding="utf-8-sig"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    nodes.extend(n for n in tree.body if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in {"aa_to_monomer", "RNG_SEED", "MONOMERS"} for t in n.targets))
    scope = {"np": np, "pd": pd, "ast": ast, "random": random, "Counter": Counter,
             "Dict": typing.Dict, "List": typing.List, "Tuple": typing.Tuple, "Optional": typing.Optional}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), scope)
    return scope


def identity_pca(path):
    common.save_json(path, {"mean": [0.] * 16, "components": np.eye(16)[:5].tolist()})


def run_cli(*args):
    command = [sys.executable, str(ROOT / "chapter3.py"), *map(str, args)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    return result


class ReleaseChecks(unittest.TestCase):
    def test_original_checksums(self):
        manifest = json.loads((ROOT / "source_manifest.json").read_text())
        self.assertEqual(len(manifest), 8)
        for record in manifest:
            self.assertEqual(hashlib.sha256((ROOT / record["repository_path"]).read_bytes()).hexdigest(), record["sha256"])

    def test_original_residue_mapping_and_windows(self):
        reference = original("protein_convert-notail.py")
        sequence = "ACDEFGHIKLMNPQRSTVWY" * 8
        self.assertEqual(common.encode_sequence(sequence), reference["encode_sequence_to_rhp"](sequence))
        for window in [50, 100]:
            encoded = common.encode_sequence(sequence)
            actual = [encoded[i:i + window] for i in common.sliding_positions(len(encoded), window, 15)]
            self.assertEqual(actual, reference["sliding_window_sequence"](encoded, window, 15))
        self.assertEqual(common.sliding_positions(49), [])
        self.assertEqual(common.sliding_positions(50), [0])
        self.assertEqual(common.encode_sequence("a c d"), "114")
        with self.assertRaises(ValueError): common.encode_sequence("AX")
        self.assertEqual(common.encode_sequence("AX", unknown_policy="mma"), "11")

    def test_lag_and_entropy_match_originals(self):
        reference = original("self-correlation heatmap.py")
        rng = np.random.default_rng(101)
        for values in [np.repeat(8.45, 50), np.tile([5.125, 18.5], 25), rng.choice(list(common.HLB.values()), 50)]:
            expected = reference["calc_lag_diffs"](values, 30)
            actual = analysis.lag_profile(values, 30)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_allclose(analysis.lag_entropy(actual), reference["calc_entropy"](expected), equal_nan=True)
        with self.assertRaises(ValueError): analysis.lag_profile([1, 2], 2)

    def test_q3_fraction_matches_original(self):
        reference = original("RHP structure prediction(1).py")
        for q3 in ["H" * 50, "C" * 50, "H" * 20 + "E" * 10 + "C" * 20]:
            np.testing.assert_array_equal(analysis.q3_fraction(q3), reference["q3_fraction"](q3))
        with self.assertRaises(ValueError): analysis.q3_fraction("ERROR")
        with np.errstate(divide="ignore"):
            expected = 10 * np.log((.4 + .2) / (.4 + 1e-12))
        self.assertEqual(float(analysis.structure_tendency([.4, .2, .4])), expected)

    def test_numeric_z_suffix_preserves_window_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = directory / "input.csv"
            sequence = "A" * 110
            q3 = "H" * 50 + "E" * 50 + "C" * 10
            vector = json.dumps([.25] * 16)
            pd.DataFrame([{"Sequence": sequence, "Q3": q3, "z_2": vector, "z_0": vector}]).to_csv(path, index=False)
            pca = {"mean": [0.] * 16, "components": np.eye(16)[:5].tolist()}
            points, fractions = analysis.feature_points(path, pca, feature="structure")
            self.assertEqual(points.window_index.tolist(), [0, 2])
            self.assertEqual(points.start.tolist(), [0, 30])
            np.testing.assert_array_equal(fractions, [[1, 0, 0], [.4, .6, 0]])

    def test_invalid_vectors_and_q3_are_rejected(self):
        for value in ["[1, 2]", "not-a-vector", json.dumps([True] * 16), str([float("inf")] * 16)]:
            with self.assertRaises(ValueError): common.parse_z(value)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            pd.DataFrame([{"Sequence": "A" * 50, "Q3": "H" * 49, "z_0": json.dumps([0.] * 16)}]).to_csv(path, index=False)
            with self.assertRaises(ValueError): analysis.feature_points(path, {"mean": [0.] * 16, "components": np.eye(16)[:2].tolist()}, feature="structure")

    def test_grid_includes_exact_upper_boundary(self):
        coords = np.array([[0, 0], [2, 2]], float)
        grid = common.Grid.from_points(coords, 2)
        self.assertEqual((grid.nx, grid.ny), (2, 2))
        self.assertEqual(sum(map(len, analysis.bin_members(coords, grid).values())), 2)

    def test_legacy_centers_and_shuffled_grid_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.csv"
            pd.DataFrame({"grid_x": [3, 1], "grid_y": [1, 1], "structure_tendency": [20, 10]}).to_csv(path, index=False)
            frame, grid, _ = common.load_grid_table(path)
            self.assertEqual(grid.extent, [0, 4, 0, 2])
            scores, _, _ = analysis.score_points(np.array([[1.9, 1], [2.1, 1]]), frame.structure_tendency.to_numpy().reshape(1, 2), grid)
            np.testing.assert_array_equal(scores, [10, 20])

    def test_scoring_and_optional_region_denominators(self):
        grid = common.Grid(0, -2, 1, 2, 2)
        coords = np.array([[1, -1], [1, 1], [1, 3], [3, 0]])
        score_grid = np.array([[2.], [4.]])
        _, _, summary = analysis.score_points(coords, score_grid, grid)
        self.assertEqual(summary["mean_structure_score"], 3)
        self.assertEqual(summary["coverage_ratio"], .5)
        _, _, limited = analysis.score_points(coords, score_grid, grid, pc2_limit=1)
        self.assertEqual(limited["N_region"], 3)
        self.assertAlmostEqual(limited["coverage_ratio"], 2 / 3)

    def test_pseudo_blocks_exact_sequences_and_counts(self):
        legacy = original("pseudo-block_generator_-ver2.py")
        for ratios in [(.15,.25,.55,.05), (.30,.25,.40,.05), (.40,.25,.30,.05), (.50,.25,.20,.05), (.65,.25,.05,.05)]:
            for seed in range(777, 789):
                random.seed(seed)
                sequence, _, decided, nh, npolar = legacy["generate_one_chain_mixed"](150, dict(enumerate(ratios, 1)), .30, 9, .1, 9, .1)
                actual, info = generate_chain(150, ratios, .30, 9, .1, 9, .1, seed)
                self.assertEqual(actual, "".join(map(str, sequence)))
                self.assertEqual(info["counts"], decided)
                self.assertEqual((info["hydrophobic_blocks"], info["hydrophilic_blocks"]), (nh, npolar))

    def test_lag_map_recomputation_and_statistical_definitions(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path, pca_path = directory / "input.csv", directory / "pca.json"
            identity_pca(pca_path)
            sequences = ["A" * 50, "L" * 25 + "S" * 25, "LS" * 25]
            vector = json.dumps([.1] * 16)
            pd.DataFrame([{"Sequence": seq, "z_0": vector} for seq in sequences]).to_csv(path, index=False)
            frame, _, _ = analysis.make_map(path, pca_path, directory / "maps", min_points=1)
            profiles = np.stack([analysis.lag_profile([common.HLB[m] for m in common.encode_sequence(seq)]) for seq in sequences])
            self.assertEqual(frame.loc[0, "var_mean_after"], np.var(profiles.mean(axis=0)))
            self.assertEqual(frame.loc[0, "var_point_first"], np.nanmean([analysis.lag_entropy(x) for x in profiles]))
            self.assertEqual(frame.loc[0, "mean_segment_lag_variance"], np.mean(np.var(profiles, axis=1)))
            self.assertNotAlmostEqual(frame.loc[0, "var_mean_after"], frame.loc[0, "mean_segment_lag_variance"])

    def test_overwrite_and_metadata_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            path.write_text("Sequence\nAAA\n")
            with self.assertRaises(FileExistsError): common.prepare_outputs([path])
            with self.assertRaises(ValueError): common.prepare_outputs([path], True, [path])
            common.save_json(common.meta_path(path), {"window": 50, "step": 15})
            with self.assertRaises(ValueError): common.check_embedding_metadata([path], window=100)

    def test_all_cli_help_and_cpu_example_workflow(self):
        for command in ["windows", "import-rhpapp", "embed", "predict-q3", "fit-pca", "plot-kde", "lag-map", "structure-map", "score-rhp", "pseudo-block"]:
            self.assertIn("usage:", run_cli(command, "--help").stdout)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            pca = directory / "pca.json"
            run_cli("windows", "--input", ROOT / "examples/protein_sequences.csv", "--window", 50, "--output", directory / "windows.csv")
            run_cli("fit-pca", "--groups", ROOT / "examples/groups.json", "--output", pca)
            run_cli("plot-kde", "--groups", ROOT / "examples/groups.json", "--pca", pca, "--output-dir", directory / "kde")
            run_cli("lag-map", "--input", ROOT / "examples/protein_cytoplasm_z.csv", "--pca", pca, "--min-points", 1, "--output-dir", directory / "lag")
            run_cli("structure-map", "--input", ROOT / "examples/protein_cytoplasm_z.csv", "--pca", pca, "--min-points", 1, "--output-dir", directory / "structure")
            run_cli("score-rhp", "--input", ROOT / "examples/rhp_z.csv", "--pca", pca, "--map", directory / "structure/structure_map.csv", "--output-dir", directory / "scores")
            run_cli("pseudo-block", "--output", directory / "pseudo.csv", "--count", 20)
            self.assertEqual(len(pd.read_csv(directory / "windows.csv")), 40)
            self.assertEqual(len(pd.read_csv(directory / "pseudo.csv")), 20)
            for path in [directory / "kde/sequence_space.png", directory / "lag/lag_variance.png", directory / "structure/structure_tendency.pdf"]:
                self.assertTrue(path.stat().st_size > 1000)

    def test_rhpapp_export_recode_declares_source_order(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source, output = directory / "arrays.csv", directory / "sequences.csv"
            source.write_text("0,1,2,3\n3,1,0\n")
            run_cli("import-rhpapp", "--input", source, "--output", output, "--source-codes", 0, 2, 1, 3)
            self.assertEqual(pd.read_csv(output, dtype=str).Sequence.tolist(), ["1324", "431"])
            previous = output.read_bytes()
            source.write_text("0,1,9\n")
            failed = subprocess.run([sys.executable, str(ROOT / "chapter3.py"), "import-rhpapp", "--input", str(source),
                                     "--output", str(output), "--source-codes", "0", "2", "1", "3", "--overwrite"], capture_output=True)
            self.assertEqual(failed.returncode, 2)
            self.assertEqual(output.read_bytes(), previous)

    def test_upstream_encoder_adapter_and_batching(self):
        if AE_REPO is None: self.skipTest("Pass --ae-repo to test against separately obtained upstream code")
        import torch
        from ch3lib import inference
        torch.set_num_threads(2)
        torch.manual_seed(119)
        module = inference.upstream_module(AE_REPO / "models/AE_MP.py", "_ae_reference")
        util = inference.upstream_module(AE_REPO / "utils/util_functions.py", "_ae_util_reference")
        model = module.build_model(**inference.AE_CONFIG).eval()
        encoded = "1234" * 45
        starts = common.sliding_positions(len(encoded))
        tensors = torch.stack([util.seq_one_hot_AE(encoded[i:i + 50], "1234", 50).flatten() for i in starts])
        np.testing.assert_array_equal(common.one_hot_numpy(encoded[:50]), util.seq_one_hot_AE(encoded[:50], "1234", 50).numpy())
        with torch.no_grad(): expected = model.encoder(tensors).numpy()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            checkpoint = directory / "weights.pt"
            torch.save({**model.state_dict(), "optimizer": {}}, checkpoint)
            loaded, torch, device, _ = inference.load_encoder(AE_REPO, checkpoint, "cpu")
            for batch_size in [1, 2, 1000]:
                actual = inference.embed_windows(encoded, loaded, torch, device, batch_size=batch_size)
                np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
            raw = directory / "raw.csv"
            pd.DataFrame({"Entry": ["LONG", "SHORT"], "Sequence": ["L" * 65, "A" * 49]}).to_csv(raw, index=False)
            inference.embed_files([raw], directory / "embedded", AE_REPO, checkpoint, device="cpu")
            frame = pd.read_csv(directory / "embedded/z_raw.csv")
            self.assertEqual(len(frame), 2)
            self.assertEqual(len(common.z_columns(frame)), 2)
            metadata = json.loads(common.meta_path(directory / "embedded/z_raw.csv").read_text())
            self.assertEqual(metadata["short_rows"], 1)

    def test_porter_adapter_q3_length_and_chunk_interface(self):
        if PORTER_SOURCE is None: self.skipTest("Pass --porter-source to test the Q3 interface with upstream Net")
        import torch
        import esm
        from ch3lib import inference
        torch.set_num_threads(2)
        module = inference.upstream_module(PORTER_SOURCE, "_porter_reference")
        module.device = torch.device("cpu")
        model = module.Net(input_size=1280).eval()

        class FakeESM(torch.nn.Module):
            def forward(self, tokens, repr_layers, return_contacts):
                assert repr_layers == [33] and return_contacts is False
                return {"representations": {33: torch.zeros(tokens.shape[0], tokens.shape[1], 1280)}}

        class FakeAlphabet:
            def get_batch_converter(self):
                return lambda data: (["id"], [data[0][1]], torch.zeros(1, len(data[0][1]) + 2, dtype=torch.long))

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            checkpoint = directory / "porter.pth"
            esm_path = directory / "esm.pt"
            torch.save({"model_state_dict": model.state_dict()}, checkpoint)
            esm_path.write_bytes(b"mock ESM loader fixture")
            path = directory / "protein.csv"
            pd.DataFrame({"Sequence": ["A", "ACDEFGH"]}).to_csv(path, index=False)
            with patch.object(esm.pretrained, "load_model_and_alphabet_local", return_value=(FakeESM(), FakeAlphabet())):
                inference.predict_q3_files([path], directory / "out", esm_path, PORTER_SOURCE, checkpoint, device="cpu", chunk_size=3)
            result = pd.read_csv(directory / "out/protein_q3.csv")
            self.assertEqual(result.Q3.str.len().tolist(), [1, 7])
            self.assertTrue(all(set(q3) <= set("HEC") for q3 in result.Q3))

    def test_real_esm_loader_namespace_metadata_compatibility(self):
        if PORTER_SOURCE is None: self.skipTest("Optional PyTorch/fair-esm inference check")
        import torch, esm
        from ch3lib.inference import load_esm_local
        alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        model = esm.model.esm2.ESM2(num_layers=1, embed_dim=16, attention_heads=2, alphabet=alphabet)
        config = argparse.Namespace(encoder_layers=1, encoder_embed_dim=16, encoder_attention_heads=2, token_dropout=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "esm2_t1_fixture.pt"
            torch.save({"cfg": {"model": config}, "model": model.state_dict()}, path)
            torch.save({"model": {k: v for k, v in model.state_dict().items() if k.startswith("contact_head.regression.")}},
                       path.with_name("esm2_t1_fixture-contact-regression.pt"))
            loaded, loaded_alphabet = load_esm_local(esm, torch, path)
            loaded.eval()
            _, _, tokens = loaded_alphabet.get_batch_converter()([("fixture", "ACDE")])
            with torch.inference_mode(): result = loaded(tokens, repr_layers=[1])
            self.assertEqual(tuple(result["representations"][1].shape), (1, 6, 16))


def main():
    global AE_REPO, PORTER_SOURCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ae-repo", type=Path)
    parser.add_argument("--porter-source", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    AE_REPO, PORTER_SOURCE = args.ae_repo, args.porter_source
    start = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReleaseChecks))
    if args.report:
        import matplotlib, sklearn
        record = {"validation_date_utc": datetime.now(timezone.utc).date().isoformat(),
                  "status": "passed" if result.wasSuccessful() else "failed", "tests_run": result.testsRun,
                  "skipped": len(result.skipped), "original_scripts_verified": 8, "pseudo_block_regression_chains": 60,
                  "duration_seconds": round(time.monotonic() - start, 3), "research_checkpoints_tested": False,
                  "ae_adapter": "upstream AE_MP with newly initialized fixture weights" if AE_REPO else "not run",
                  "q3_adapter": "upstream Porter6 Net with fixture weights and mocked ESM representations" if PORTER_SOURCE else "not run",
                  "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                               "matplotlib": matplotlib.__version__, "scikit_learn": sklearn.__version__}}
        if AE_REPO or PORTER_SOURCE:
            import torch
            record["versions"]["torch"] = torch.__version__
        if PORTER_SOURCE:
            import esm
            record["versions"]["fair_esm"] = esm.__version__
        common.prepare_outputs([args.report], overwrite=True)
        common.save_json(args.report, record)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
