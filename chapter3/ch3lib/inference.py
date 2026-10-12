"""Adapters to separately obtained upstream code and model checkpoints."""
from pathlib import Path
import argparse
import contextlib
import importlib.util
import json
import subprocess

import numpy as np
import pandas as pd

from .common import (encode_sequence, input_records, meta_path, one_hot_numpy, prepare_outputs,
                     read_table, save_json, sha256, sliding_positions)

AE_CONFIG = dict(input_dim=[5, 51], latent_dim=16, win_length=5,
                 enc_hidden_sizes=[512, 256, 64], dec_hidden_sizes=[64, 256, 512],
                 compute_HLB=True, MLP_HLB_size=[32, 32])


def require_torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise ImportError("Inference needs PyTorch; install requirements-inference.txt") from exc


def select_device(torch, requested):
    name = "cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested)
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return torch.device(name)


def upstream_module(path, name):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Upstream source file not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_revision(path):
    try:
        result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def state_dict(torch, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a state-dictionary or a dictionary containing one")
    for key in ("model_state_dict", "state_dict"):
        if key in checkpoint:
            checkpoint = checkpoint[key]
            break
    return {key: value for key, value in checkpoint.items() if key != "optimizer"}


def load_esm_local(esm, torch, checkpoint):
    # Public ESM checkpoints use Namespace configuration metadata. PyTorch 2.6+
    # needs this specific class allowed while keeping weights-only loading.
    safe_globals = getattr(torch.serialization, "safe_globals", None)
    context = safe_globals([argparse.Namespace]) if safe_globals else contextlib.nullcontext()
    with context:
        return esm.pretrained.load_model_and_alphabet_local(str(checkpoint))


def load_encoder(ae_repo, checkpoint, device="auto", config_path=None):
    torch = require_torch()
    target = select_device(torch, device)
    config = dict(AE_CONFIG)
    if config_path:
        config.update(json.loads(Path(config_path).read_text()))
    if config["input_dim"] != [5, 51] or config["latent_dim"] != 16:
        raise ValueError("This adapter requires the uploaded 50-mer AE_MP layout and 16 latent dimensions")
    # Loading just the public model file avoids unrelated torchvision imports and generic models-module collisions.
    module = upstream_module(Path(ae_repo) / "models" / "AE_MP.py", "_chapter3_ae_mp")
    model = module.build_model(**config)
    model.load_state_dict(state_dict(torch, checkpoint, target), strict=True)
    model.to(target).eval()
    return model, torch, target, config


def embed_windows(encoded, model, torch, device, window=50, step=15, batch_size=1000):
    if window != 50 or batch_size < 1:
        raise ValueError("AE_MP inference requires window=50 and a positive batch size")
    starts = sliding_positions(len(encoded), window, step)
    vectors = []
    for offset in range(0, len(starts), batch_size):
        part = starts[offset:offset + batch_size]
        batch = np.stack([one_hot_numpy(encoded[start:start + window]).reshape(-1) for start in part])
        tensor = torch.from_numpy(batch).to(device)
        with torch.inference_mode():
            z = model.encoder(tensor).detach().cpu().numpy()
        if z.shape != (len(part), 16) or not np.isfinite(z).all():
            raise ValueError("Encoder did not return finite 16-dimensional vectors")
        vectors.extend(z.tolist())
    return vectors


def embed_files(paths, output_dir, ae_repo, checkpoint, kind="protein", sequence_column="Sequence",
                step=15, batch_size=1000, device="auto", unknown_policy="error", overwrite=False, config_path=None):
    names = ["z_" + Path(p).stem + ".csv" for p in paths]
    if len(set(names)) != len(names):
        raise ValueError("Input basenames collide; run these inputs in separate output directories")
    outputs = [Path(output_dir) / name for name in names]
    prepare_outputs([p for output in outputs for p in (output, meta_path(output))], overwrite, paths)
    model, torch, target, config = load_encoder(ae_repo, checkpoint, device, config_path)
    checkpoint_hash = sha256(checkpoint)
    for path, output in zip(paths, outputs):
        frame = read_table(path)
        if sequence_column not in frame:
            raise ValueError(f"{path}: missing {sequence_column}")
        if any(str(c).startswith("z_") for c in frame.columns):
            raise ValueError(f"{path} already has z_ columns; supply an unembedded input table")
        records, short = [], 0
        for index, row in frame.iterrows():
            try:
                encoded = encode_sequence(row[sequence_column], kind, unknown_policy)
                vectors = embed_windows(encoded, model, torch, target, 50, step, batch_size)
            except ValueError as exc:
                raise ValueError(f"{path}, data row {index + 1}: {exc}") from exc
            if not vectors:
                short += 1
                if kind == "rhp":
                    continue  # Retain the uploaded RHP/protein row-handling distinction.
            record = row.to_dict()
            record[sequence_column] = "".join(row[sequence_column].split()).upper()
            record.update({f"z_{i}": json.dumps(vector) for i, vector in enumerate(vectors)})
            records.append(record)
        result = pd.DataFrame(records, columns=list(frame.columns) if not records else None)
        result.to_csv(output, index=False, encoding="utf-8-sig")
        save_json(meta_path(output), {"inputs": input_records([path]), "kind": kind, "window": 50, "step": step,
                                     "batch_size": batch_size, "model": "AE_MP", "model_config": config,
                                     "checkpoint_sha256": checkpoint_hash, "ae_repo_revision": git_revision(ae_repo),
                                     "ae_source_sha256": sha256(Path(ae_repo) / "models/AE_MP.py"),
                                     "device": str(target), "torch_version": torch.__version__,
                                     "unknown_policy": unknown_policy, "short_rows": short,
                                     "output_csv_sha256": sha256(output)})
        print(f"Saved {len(result)} rows to {output}; {short} rows had no complete 50-mer")


def predict_q3_files(paths, output_dir, esm_checkpoint, porter_source, porter_checkpoint,
                     device="auto", chunk_size=9500, overwrite=False, on_error="raise", sequence_column="Sequence"):
    torch = require_torch()
    try:
        import esm
    except ImportError as exc:
        raise ImportError("Q3 prediction needs fair-esm; install requirements-inference.txt") from exc
    if chunk_size < 1:
        raise ValueError("Chunk size must be positive")
    target = select_device(torch, device)
    esm_model, alphabet = load_esm_local(esm, torch, esm_checkpoint)
    esm_model.to(target).eval()
    converter = alphabet.get_batch_converter()
    module = upstream_module(porter_source, "_chapter3_porter_q3")
    module.device = target  # Upstream Net creates its recurrent initial state using this module variable.
    porter = module.Net(input_size=1280).to(target)
    porter.load_state_dict(state_dict(torch, porter_checkpoint, target), strict=True)
    porter.eval()
    names = [Path(p).stem + "_q3.csv" for p in paths]
    if len(set(names)) != len(names):
        raise ValueError("Input basenames collide")
    outputs = [Path(output_dir) / name for name in names]
    prepare_outputs([p for output in outputs for p in (output, meta_path(output))], overwrite, paths)
    for path, output in zip(paths, outputs):
        frame = read_table(path)
        if sequence_column not in frame:
            raise ValueError(f"{path}: missing {sequence_column}")
        sequences, labels, errors = [], [], []
        for index, row in frame.iterrows():
            sequence = "".join(row[sequence_column].split()).upper()
            sequences.append(sequence)
            try:
                if not sequence or set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
                    raise ValueError("Expected a nonempty canonical amino-acid sequence")
                parts = []
                for start in range(0, len(sequence), chunk_size):
                    part = sequence[start:start + chunk_size]
                    _, _, tokens = converter([("sequence", part)])
                    with torch.inference_mode():
                        embedding = esm_model(tokens.to(target), repr_layers=[33], return_contacts=False)["representations"][33]
                        residue_embedding = embedding[0, 1:len(part) + 1]
                        prediction = porter(residue_embedding.unsqueeze(0).transpose(1, 2))
                        indices = prediction.argmax(dim=2).reshape(-1).cpu().tolist()
                    if len(indices) != len(part):
                        raise ValueError("Porter output length differs from the input sequence")
                    parts.append("".join("HEC"[i] for i in indices))
                labels.append("".join(parts)); errors.append("")
            except Exception as exc:
                if on_error == "raise":
                    raise RuntimeError(f"{path}, data row {index + 1}: {exc}") from exc
                labels.append("ERROR"); errors.append(str(exc))
        frame[sequence_column], frame["Q3"] = sequences, labels
        frame["Q3_error"] = errors
        frame.to_csv(output, index=False)
        save_json(meta_path(output), {"inputs": input_records([path]), "esm_checkpoint_sha256": sha256(esm_checkpoint),
                                     "porter_checkpoint_sha256": sha256(porter_checkpoint),
                                     "porter_source_sha256": sha256(porter_source), "esm_layer": 33, "embedding_dimension": 1280,
                                     "label_order": "HEC", "chunk_size": chunk_size, "device": str(target),
                                     "error_rows": sum(bool(x) for x in errors)})
        print(f"Saved Q3 labels to {output}; {sum(bool(x) for x in errors)} errors")
