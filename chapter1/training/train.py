#!/usr/bin/env python3
"""Train the original pair-token Transformer with configurable alphabet and paths."""

import argparse
import os
import json
import random
import sys
import platform
from pathlib import Path
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import sklearn
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "feature"))
from alphabet_config import ALPHABETS, get_alphabet

# Historical defaults; main() applies command-line values.
num_epochs = 24
batch_size = 16
learning_rate = 5e-6
weight_decay = 1e-5
untouched_test_ratio = 0.20
validation_ratio_within_remaining = 0.40
prediction_threshold = 0.5
force_labels_by_source = True
dropout = 0.25
transformer_layers = 1
transformer_heads = 4
transformer_ff_dim = 384
save_best_by = "val_auc"
early_stopping_patience = 100
ablation_features = []
ablation_constant = 0.0
device = torch.device("cpu")


def get_run_dir(seed):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    return os.path.join(
        output_root,
        f"run_ModelB_transformer_{alphabet_name}_seed{seed:05d}_{timestamp}",
    )


def decode_if_bytes(arr):
    return np.asarray(
        [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in arr]
    )


def apply_pos_ablation(pos_features, ablation_list):
    if not ablation_list:
        return pos_features
    candidates = [
        k for k in (3, 4, 5, 6, 8) if k + k**2 + k**3 + 60 == pos_features.shape[1]
    ]
    if len(candidates) != 1:
        raise ValueError("Unexpected position-feature shape")
    k = candidates[0]
    widths = {
        "one_mer_counts": k,
        "two_mer_counts": k**2,
        "three_mer_counts": k**3,
        "H1_matrix": 30,
        "H2_matrix": 30,
    }
    masked = pos_features.copy()
    start = 0
    for name, width in widths.items():
        if name in ablation_list:
            masked[:, start : start + width, :] = ablation_constant
        start += width
    return masked


def apply_sum_ablation(sum_features, ablation_list):
    if not ablation_list:
        return sum_features
    k = sum_features.shape[1] - 121
    widths = {
        "H1_features": 30,
        "H2_features": 30,
        "apaac1": 30,
        "apaac2": 30,
        "proportion": k,
        "length": 1,
    }
    masked = sum_features.copy()
    start = 0
    for name, width in widths.items():
        if name in ablation_list:
            masked[:, start : start + width] = ablation_constant
        start += width
    return masked


def validate_feature_tables(neg, pos, alphabet):
    k = len(get_alphabet(alphabet)["groups"]) + 1
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
    for source, data, label in (("negative", neg, 0), ("positive", pos, 1)):
        n = len(data["interaction"])
        if n == 0 or data["interaction"].shape != (n,):
            raise ValueError(f"{source}: empty or invalid interaction dataset")
        if not np.all(data["interaction"] == label):
            raise ValueError(
                f"{source}: interaction labels must all equal {label}; prepare the CSV with --label {label}"
            )
        for key in ("proteinA", "proteinB"):
            if data[key].shape != (n,):
                raise ValueError(f"{source}: invalid {key} shape")
        for side in ("A", "B"):
            for name, shape in shapes.items():
                key = (
                    f"{name}_{side}"
                    if name in ("proportion", "length")
                    else f"{name}{side}"
                )
                if data[key].shape != (n,) + shape:
                    raise ValueError(
                        f"{source}: {key} has shape {data[key].shape}; expected {(n,) + shape} for {alphabet}"
                    )
                if not np.isfinite(data[key]).all():
                    raise ValueError(f"{source}: non-finite values in {key}")
    n_pair = min(len(neg["interaction"]), len(pos["interaction"]))
    n_test = round(n_pair * untouched_test_ratio)
    n_val = round((n_pair - n_test) * validation_ratio_within_remaining)
    if min(n_test, n_val, n_pair - n_test - n_val) < 1:
        raise ValueError(
            "Too few rows for nonempty train, validation, and test subsets in each class"
        )


def read_h5_to_dict(h5_path):
    data = {}
    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            data[key] = f[key][:]
    return data


def ensure_required_keys(data_dict, source_name):
    required = [
        "interaction",
        "proteinA",
        "proteinB",
        "one_mer_countsA",
        "two_mer_countsA",
        "three_mer_countsA",
        "one_mer_countsB",
        "two_mer_countsB",
        "three_mer_countsB",
        "proportion_A",
        "length_A",
        "H1_featuresA",
        "H2_featuresA",
        "H1_matrixA",
        "H2_matrixA",
        "apaac1A",
        "apaac2A",
        "proportion_B",
        "length_B",
        "H1_featuresB",
        "H2_featuresB",
        "H1_matrixB",
        "H2_matrixB",
        "apaac1B",
        "apaac2B",
    ]
    missing = [k for k in required if k not in data_dict]
    if missing:
        raise KeyError(f"{source_name} is missing required keys: {missing}")


def build_feature_views(data_dict, ablation_features=None):
    one_mer_countsA = data_dict["one_mer_countsA"]
    two_mer_countsA = data_dict["two_mer_countsA"]
    three_mer_countsA = data_dict["three_mer_countsA"]
    proportion_A = data_dict["proportion_A"]
    length_A = data_dict["length_A"]
    H1_featuresA = data_dict["H1_featuresA"]
    H2_featuresA = data_dict["H2_featuresA"]
    H1_matrixA = data_dict["H1_matrixA"]
    H2_matrixA = data_dict["H2_matrixA"]
    apaac1A = data_dict["apaac1A"]
    apaac2A = data_dict["apaac2A"]
    one_mer_countsB = data_dict["one_mer_countsB"]
    two_mer_countsB = data_dict["two_mer_countsB"]
    three_mer_countsB = data_dict["three_mer_countsB"]
    proportion_B = data_dict["proportion_B"]
    length_B = data_dict["length_B"]
    H1_featuresB = data_dict["H1_featuresB"]
    H2_featuresB = data_dict["H2_featuresB"]
    H1_matrixB = data_dict["H1_matrixB"]
    H2_matrixB = data_dict["H2_matrixB"]
    apaac1B = data_dict["apaac1B"]
    apaac2B = data_dict["apaac2B"]
    posA = np.concatenate(
        (one_mer_countsA, two_mer_countsA, three_mer_countsA, H1_matrixA, H2_matrixA),
        axis=1,
    )
    posB = np.concatenate(
        (one_mer_countsB, two_mer_countsB, three_mer_countsB, H1_matrixB, H2_matrixB),
        axis=1,
    )
    if ablation_features:
        posA = apply_pos_ablation(posA, ablation_features)
        posB = apply_pos_ablation(posB, ablation_features)
    sumA = np.concatenate((H1_featuresA, H2_featuresA, apaac1A, apaac2A), axis=1)
    sumB = np.concatenate((H1_featuresB, H2_featuresB, apaac1B, apaac2B), axis=1)
    length_A_2d = length_A[:, np.newaxis]
    length_B_2d = length_B[:, np.newaxis]
    sumA = np.concatenate((sumA, proportion_A, length_A_2d), axis=1)
    sumB = np.concatenate((sumB, proportion_B, length_B_2d), axis=1)
    if ablation_features:
        sumA = apply_sum_ablation(sumA, ablation_features)
        sumB = apply_sum_ablation(sumB, ablation_features)
    return {
        "posA": posA.astype(np.float32),
        "posB": posB.astype(np.float32),
        "sumA": sumA.astype(np.float32),
        "sumB": sumB.astype(np.float32),
        "interaction": data_dict["interaction"].astype(np.float32),
        "proteinA": decode_if_bytes(data_dict["proteinA"]),
        "proteinB": decode_if_bytes(data_dict["proteinB"]),
    }


def subset_features(view_dict, indices):
    return {
        "posA": view_dict["posA"][indices],
        "posB": view_dict["posB"][indices],
        "sumA": view_dict["sumA"][indices],
        "sumB": view_dict["sumB"][indices],
        "interaction": view_dict["interaction"][indices],
        "proteinA": view_dict["proteinA"][indices],
        "proteinB": view_dict["proteinB"][indices],
    }


def concat_feature_dicts(a, b):
    return {
        "posA": np.concatenate([a["posA"], b["posA"]], axis=0),
        "posB": np.concatenate([a["posB"], b["posB"]], axis=0),
        "sumA": np.concatenate([a["sumA"], b["sumA"]], axis=0),
        "sumB": np.concatenate([a["sumB"], b["sumB"]], axis=0),
        "interaction": np.concatenate([a["interaction"], b["interaction"]], axis=0),
        "proteinA": np.concatenate([a["proteinA"], b["proteinA"]], axis=0),
        "proteinB": np.concatenate([a["proteinB"], b["proteinB"]], axis=0),
    }


def shuffle_feature_dict(d, rng):
    n = len(d["interaction"])
    indices = np.arange(n)
    rng.shuffle(indices)
    return (subset_features(d, indices), indices)


def make_dataloader(feature_dict, batch_size=32, shuffle=True):
    ds = TensorDataset(
        torch.from_numpy(feature_dict["posA"]),
        torch.from_numpy(feature_dict["posB"]),
        torch.from_numpy(feature_dict["sumA"]),
        torch.from_numpy(feature_dict["sumB"]),
        torch.from_numpy(feature_dict["interaction"]).unsqueeze(1),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def choose_indices(n_total, n_pick, rng, name):
    if n_total < n_pick:
        raise ValueError(f"{name} has only {n_total} samples, cannot sample {n_pick}.")
    return rng.choice(n_total, size=n_pick, replace=False)


def split_class_indices(
    indices_all, untouched_test_ratio, validation_ratio_within_remaining, rng
):
    indices_all = np.array(indices_all)
    rng.shuffle(indices_all)
    n_total = len(indices_all)
    n_test = int(round(n_total * untouched_test_ratio))
    test_idx = indices_all[:n_test]
    remaining_idx = indices_all[n_test:]
    rng.shuffle(remaining_idx)
    n_val = int(round(len(remaining_idx) * validation_ratio_within_remaining))
    val_idx = remaining_idx[:n_val]
    train_idx = remaining_idx[n_val:]
    return (train_idx, val_idx, test_idx)


def build_balanced_train_val_untouched_test_split(
    neg, pos, rng, untouched_test_ratio=0.2, validation_ratio_within_remaining=0.4
):
    n_neg_total = len(neg["interaction"])
    n_pos_total = len(pos["interaction"])
    n_pair = min(n_neg_total, n_pos_total)
    neg_idx_all = choose_indices(n_neg_total, n_pair, rng, "Negatome")
    pos_idx_all = choose_indices(n_pos_total, n_pair, rng, "BioGRID")
    neg_train_idx, neg_val_idx, neg_test_idx = split_class_indices(
        neg_idx_all,
        untouched_test_ratio=untouched_test_ratio,
        validation_ratio_within_remaining=validation_ratio_within_remaining,
        rng=rng,
    )
    pos_train_idx, pos_val_idx, pos_test_idx = split_class_indices(
        pos_idx_all,
        untouched_test_ratio=untouched_test_ratio,
        validation_ratio_within_remaining=validation_ratio_within_remaining,
        rng=rng,
    )
    neg_train = subset_features(neg, neg_train_idx)
    pos_train = subset_features(pos, pos_train_idx)
    neg_val = subset_features(neg, neg_val_idx)
    pos_val = subset_features(pos, pos_val_idx)
    neg_test = subset_features(neg, neg_test_idx)
    pos_test = subset_features(pos, pos_test_idx)
    train_set = concat_feature_dicts(neg_train, pos_train)
    val_set = concat_feature_dicts(neg_val, pos_val)
    untouched_test_set = concat_feature_dicts(neg_test, pos_test)
    train_set, train_shuffle_order = shuffle_feature_dict(train_set, rng)
    val_set, val_shuffle_order = shuffle_feature_dict(val_set, rng)
    untouched_test_set, untouched_test_shuffle_order = shuffle_feature_dict(
        untouched_test_set, rng
    )
    split_records = {
        "n_pair_each_class": int(n_pair),
        "untouched_test_ratio": float(untouched_test_ratio),
        "validation_ratio_within_remaining": float(validation_ratio_within_remaining),
        "n_neg_train": int(len(neg_train_idx)),
        "n_neg_val": int(len(neg_val_idx)),
        "n_neg_untouched_test": int(len(neg_test_idx)),
        "n_pos_train": int(len(pos_train_idx)),
        "n_pos_val": int(len(pos_val_idx)),
        "n_pos_untouched_test": int(len(pos_test_idx)),
        "neg_train_indices": neg_train_idx.tolist(),
        "neg_val_indices": neg_val_idx.tolist(),
        "neg_untouched_test_indices": neg_test_idx.tolist(),
        "pos_train_indices": pos_train_idx.tolist(),
        "pos_val_indices": pos_val_idx.tolist(),
        "pos_untouched_test_indices": pos_test_idx.tolist(),
        "train_shuffle_order_after_concat": train_shuffle_order.tolist(),
        "val_shuffle_order_after_concat": val_shuffle_order.tolist(),
        "untouched_test_shuffle_order_after_concat": untouched_test_shuffle_order.tolist(),
    }
    return (train_set, val_set, untouched_test_set, split_records)


def safe_auc(y_true, pred_prob):
    try:
        fpr, tpr, _ = roc_curve(y_true, pred_prob)
        return auc(fpr, tpr)
    except Exception:
        return np.nan


def compute_basic_metrics(y_true, pred_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    pred_prob = np.asarray(pred_prob).astype(float)
    pred_label = (pred_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred_label)),
        "precision": float(precision_score(y_true, pred_label, zero_division=0)),
        "recall": float(recall_score(y_true, pred_label, zero_division=0)),
        "f1": float(f1_score(y_true, pred_label, zero_division=0)),
        "auc": float(safe_auc(y_true, pred_prob)),
        "n_pos": int(np.sum(y_true == 1)),
        "n_neg": int(np.sum(y_true == 0)),
        "n_pred_pos": int(np.sum(pred_label == 1)),
        "n_pred_neg": int(np.sum(pred_label == 0)),
    }


def compute_untouched_test_tpr_tnr(y_true, pred_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    pred_prob = np.asarray(pred_prob).astype(float)
    pred_label = (pred_prob >= threshold).astype(int)
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    untouched_test_tpr = (
        float(np.mean(pred_label[pos_mask] == 1)) if np.sum(pos_mask) > 0 else np.nan
    )
    untouched_test_tnr = (
        float(np.mean(pred_label[neg_mask] == 0)) if np.sum(neg_mask) > 0 else np.nan
    )
    return (untouched_test_tpr, untouched_test_tnr)


def get_selection_score(record, save_best_by):
    if save_best_by == "val_auc":
        return (record["val_auc"], True)
    if save_best_by == "val_accuracy":
        return (record["val_accuracy"], True)
    if save_best_by == "val_loss":
        return (record["val_loss"], False)
    raise ValueError(f"Unknown save_best_by: {save_best_by}")


class PRIPredictorModelBTransformer(nn.Module):

    def __init__(
        self,
        pos_shape,
        sum_shape,
        dropout=0.25,
        transformer_layers=1,
        transformer_heads=4,
        transformer_ff_dim=384,
    ):
        super().__init__()
        pos_flat_dim = int(np.prod(pos_shape))
        sum_dim = int(sum_shape[0])
        self.posA_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(pos_flat_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.posB_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(pos_flat_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.sumA_net = nn.Sequential(
            nn.Linear(sum_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
        )
        self.sumB_net = nn.Sequential(
            nn.Linear(sum_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
        )
        self.embed_dim = 144
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.embed_dim * 4, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, posA, posB, sumA, sumB):
        featA_pos = self.posA_net(posA)
        featB_pos = self.posB_net(posB)
        featA_sum = self.sumA_net(sumA)
        featB_sum = self.sumB_net(sumB)
        featA = torch.cat([featA_pos, featA_sum], dim=-1)
        featB = torch.cat([featB_pos, featB_sum], dim=-1)
        abs_diff = torch.abs(featA - featB)
        prod = featA * featB
        tokens = torch.stack([featA, featB, abs_diff, prod], dim=1)
        tokens = self.transformer(tokens)
        logits = self.head(tokens)
        return logits


def run_one_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []
    for batch in loader:
        posA, posB, sumA, sumB, y = [x.to(device) for x in batch]
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            logits = model(posA, posB, sumA, sumB)
            loss = criterion(logits, y)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        probs = torch.sigmoid(logits)
        total_loss += loss.item() * y.size(0)
        all_probs.append(probs.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0).ravel()
    all_labels = np.concatenate(all_labels, axis=0).ravel()
    avg_loss = total_loss / len(all_labels)
    return (avg_loss, all_probs, all_labels)


def run_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    run_dir = get_run_dir(seed)
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n=== Ablation Run - Seed {seed:05d} ===\nRun folder: {run_dir}")
    print("Loading HDF5 files.")
    neg_raw = read_h5_to_dict(neg_h5)
    pos_raw = read_h5_to_dict(pos_h5)
    ensure_required_keys(neg_raw, "Negatome HDF5")
    ensure_required_keys(pos_raw, "BioGRID HDF5")
    validate_feature_tables(neg_raw, pos_raw, alphabet_name)
    neg = build_feature_views(neg_raw, ablation_features=ablation_features)
    pos = build_feature_views(pos_raw, ablation_features=ablation_features)
    if force_labels_by_source:
        neg["interaction"] = np.zeros_like(neg["interaction"], dtype=np.float32)
        pos["interaction"] = np.ones_like(pos["interaction"], dtype=np.float32)
    print(f"Negatome total: {len(neg['interaction'])}")
    print(f"BioGRID total: {len(pos['interaction'])}")
    train_set, val_set, untouched_test_set, split_records = (
        build_balanced_train_val_untouched_test_split(
            neg=neg,
            pos=pos,
            rng=rng,
            untouched_test_ratio=untouched_test_ratio,
            validation_ratio_within_remaining=validation_ratio_within_remaining,
        )
    )
    train_loader = make_dataloader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = make_dataloader(val_set, batch_size=batch_size, shuffle=False)
    untouched_test_loader = make_dataloader(
        untouched_test_set, batch_size=batch_size, shuffle=False
    )
    print("\nFixed balanced split:")
    print(
        f"  train total: {len(train_set['interaction'])} (pos: {int(np.sum(train_set['interaction'] == 1))})"
    )
    print(f"  validation total: {len(val_set['interaction'])}")
    print(f"  untouched test total: {len(untouched_test_set['interaction'])}")
    with open(
        os.path.join(run_dir, "fixed_split_indices.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(split_records, f, indent=2)
    input_shape_posA = neg["posA"].shape[1:]
    input_shape_sumA = neg["sumA"].shape[1:]
    model = PRIPredictorModelBTransformer(
        pos_shape=input_shape_posA,
        sum_shape=input_shape_sumA,
        dropout=dropout,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        transformer_ff_dim=transformer_ff_dim,
    ).to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()
    n_params = sum((p.numel() for p in model.parameters()))
    print(f"\nModel parameters: total={n_params:,}")
    config = {
        "neg_h5": neg_h5,
        "pos_h5": pos_h5,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "untouched_test_ratio": untouched_test_ratio,
        "validation_ratio_within_remaining": validation_ratio_within_remaining,
        "prediction_threshold": prediction_threshold,
        "random_seed": seed,
        "force_labels_by_source": force_labels_by_source,
        "dropout": dropout,
        "transformer_layers": transformer_layers,
        "transformer_heads": transformer_heads,
        "transformer_ff_dim": transformer_ff_dim,
        "save_best_by": save_best_by,
        "early_stopping_patience": early_stopping_patience,
        "device": str(device),
        "ablation_features": ablation_features,
        "ablation_constant": ablation_constant,
        "model_type": "ModelB_pair_tokens_transformer",
        "n_params": int(n_params),
        "input_shape_posA": list(input_shape_posA),
        "input_shape_sumA": list(input_shape_sumA),
        "alphabet": alphabet_name,
        "alphabet_config": get_alphabet(alphabet_name),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "h5py": h5py.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    history_records = []
    best_score = None
    best_epoch = None
    epochs_without_improvement = 0
    best_model_path = os.path.join(run_dir, "best_model.pt")
    for epoch in range(1, num_epochs + 1):
        print(
            f"\n================ Epoch {epoch}/{num_epochs} (seed={seed}) ================"
        )
        train_loss, train_pred_prob, y_train = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_pred_prob, y_val = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
        )
        untouched_test_loss, untouched_test_pred_prob, y_untouched_test = run_one_epoch(
            model=model,
            loader=untouched_test_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
        )
        train_metrics = compute_basic_metrics(
            y_train, train_pred_prob, threshold=prediction_threshold
        )
        val_metrics = compute_basic_metrics(
            y_val, val_pred_prob, threshold=prediction_threshold
        )
        untouched_test_metrics = compute_basic_metrics(
            y_untouched_test, untouched_test_pred_prob, threshold=prediction_threshold
        )
        untouched_test_TPR, untouched_test_TNR = compute_untouched_test_tpr_tnr(
            y_untouched_test, untouched_test_pred_prob, threshold=prediction_threshold
        )
        record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_metrics["accuracy"]),
            "train_precision": float(train_metrics["precision"]),
            "train_recall": float(train_metrics["recall"]),
            "train_f1": float(train_metrics["f1"]),
            "train_auc": (
                float(train_metrics["auc"])
                if not np.isnan(train_metrics["auc"])
                else np.nan
            ),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_precision": float(val_metrics["precision"]),
            "val_recall": float(val_metrics["recall"]),
            "val_f1": float(val_metrics["f1"]),
            "val_auc": (
                float(val_metrics["auc"])
                if not np.isnan(val_metrics["auc"])
                else np.nan
            ),
            "untouched_test_loss": float(untouched_test_loss),
            "untouched_test_accuracy": float(untouched_test_metrics["accuracy"]),
            "untouched_test_precision": float(untouched_test_metrics["precision"]),
            "untouched_test_recall": float(untouched_test_metrics["recall"]),
            "untouched_test_f1": float(untouched_test_metrics["f1"]),
            "untouched_test_auc": (
                float(untouched_test_metrics["auc"])
                if not np.isnan(untouched_test_metrics["auc"])
                else np.nan
            ),
            "untouched_test_TPR": (
                float(untouched_test_TPR)
                if not np.isnan(untouched_test_TPR)
                else np.nan
            ),
            "untouched_test_TNR": (
                float(untouched_test_TNR)
                if not np.isnan(untouched_test_TNR)
                else np.nan
            ),
            "untouched_test_n_pos": int(untouched_test_metrics["n_pos"]),
            "untouched_test_n_neg": int(untouched_test_metrics["n_neg"]),
            "untouched_test_n_pred_pos": int(untouched_test_metrics["n_pred_pos"]),
            "untouched_test_n_pred_neg": int(untouched_test_metrics["n_pred_neg"]),
        }
        history_records.append(record)
        current_score, larger_is_better = get_selection_score(record, save_best_by)
        if (
            best_score is None
            or (larger_is_better and current_score > best_score)
            or (not larger_is_better and current_score < best_score)
        ):
            best_score = current_score
            best_epoch = epoch
            epochs_without_improvement = 0
            best_checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history_records": history_records,
                "config": config,
                "best_score": best_score,
                "save_best_by": save_best_by,
            }
            torch.save(best_checkpoint, best_model_path)
        else:
            epochs_without_improvement += 1
        print(
            f"  train_loss={train_loss:.4f}  val_auc={val_metrics['auc']:.4f}  untouched_test_auc={untouched_test_metrics['auc']:.4f}"
        )
        print(
            f"  best_epoch={best_epoch}  best_{save_best_by}={best_score:.4f}  no_improve={epochs_without_improvement}"
        )
        history_df = pd.DataFrame(history_records)
        history_df.to_csv(
            os.path.join(run_dir, "epoch_history.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        if (
            early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break
    print("\nLoading best model for final untouched test evaluation.")
    best_checkpoint = torch.load(
        best_model_path, map_location=device, weights_only=False
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.to(device)
    untouched_test_loss, untouched_test_pred_prob, y_untouched_test = run_one_epoch(
        model=model,
        loader=untouched_test_loader,
        criterion=criterion,
        optimizer=None,
        device=device,
    )
    untouched_test_metrics = compute_basic_metrics(
        y_untouched_test, untouched_test_pred_prob, threshold=prediction_threshold
    )
    untouched_test_TPR, untouched_test_TNR = compute_untouched_test_tpr_tnr(
        y_untouched_test, untouched_test_pred_prob, threshold=prediction_threshold
    )
    untouched_test_pred_label = (
        untouched_test_pred_prob >= prediction_threshold
    ).astype(int)
    pred_df = pd.DataFrame(
        {
            "proteinA": untouched_test_set["proteinA"],
            "proteinB": untouched_test_set["proteinB"],
            "y_true": y_untouched_test.astype(int),
            "pred_prob": untouched_test_pred_prob,
            "pred_label": untouched_test_pred_label,
        }
    )
    pred_df.to_csv(
        os.path.join(run_dir, "best_model_untouched_test_predictions.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    try:
        fpr, roc_tpr, thresholds = roc_curve(
            y_untouched_test.astype(int), untouched_test_pred_prob
        )
        roc_auc = auc(fpr, roc_tpr)
        roc_df = pd.DataFrame({"fpr": fpr, "roc_tpr": roc_tpr, "threshold": thresholds})
        roc_df.to_csv(
            os.path.join(run_dir, "best_model_untouched_test_roc_curve.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        sweep_records = []
        for th in np.linspace(0, 1, 1001):
            m = compute_basic_metrics(
                y_untouched_test, untouched_test_pred_prob, threshold=th
            )
            th_tpr, th_tnr = compute_untouched_test_tpr_tnr(
                y_untouched_test, untouched_test_pred_prob, threshold=th
            )
            sweep_records.append(
                {
                    "threshold": float(th),
                    "untouched_test_accuracy": m["accuracy"],
                    "untouched_test_precision": m["precision"],
                    "untouched_test_recall": m["recall"],
                    "untouched_test_f1": m["f1"],
                    "untouched_test_TPR": th_tpr,
                    "untouched_test_TNR": th_tnr,
                    "untouched_test_n_pred_pos": m["n_pred_pos"],
                    "untouched_test_n_pred_neg": m["n_pred_neg"],
                }
            )
        sweep_df = pd.DataFrame(sweep_records)
        sweep_df.to_csv(
            os.path.join(run_dir, "best_model_untouched_test_threshold_sweep.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        fixed_tnr_records = []
        for target_tnr in [0.9, 0.95, 0.97, 0.98, 0.99]:
            candidates = sweep_df[sweep_df["untouched_test_TNR"] >= target_tnr].copy()
            if len(candidates) == 0:
                fixed_tnr_records.append(
                    {
                        "target_untouched_test_TNR": target_tnr,
                        "found": False,
                        "threshold": np.nan,
                        "untouched_test_TPR": np.nan,
                        "untouched_test_TNR": np.nan,
                    }
                )
            else:
                best_row = candidates.sort_values(
                    ["untouched_test_TPR", "untouched_test_accuracy"], ascending=False
                ).iloc[0]
                fixed_tnr_records.append(
                    {
                        "target_untouched_test_TNR": target_tnr,
                        "found": True,
                        "threshold": float(best_row["threshold"]),
                        "untouched_test_TPR": float(best_row["untouched_test_TPR"]),
                        "untouched_test_TNR": float(best_row["untouched_test_TNR"]),
                    }
                )
        pd.DataFrame(fixed_tnr_records).to_csv(
            os.path.join(
                run_dir, "best_model_untouched_test_TPR_at_fixed_untouched_test_TNR.csv"
            ),
            index=False,
            encoding="utf-8-sig",
        )
    except Exception as e:
        print("ROC / sweep generation skipped:", e)
    history_df = pd.DataFrame(history_records)
    for metric, filename, ylabel in [
        (["train_loss", "val_loss", "untouched_test_loss"], "loss_curve.png", "Loss"),
        (
            ["train_accuracy", "val_accuracy", "untouched_test_accuracy"],
            "accuracy_curve.png",
            "Accuracy",
        ),
        (["train_auc", "val_auc", "untouched_test_auc"], "auc_curve.png", "AUC"),
        (
            ["untouched_test_TPR", "untouched_test_TNR"],
            "untouched_test_tpr_tnr_curve.png",
            "Rate",
        ),
    ]:
        plt.figure(figsize=(8, 6))
        for m in metric:
            plt.plot(history_df["epoch"], history_df[m], label=m)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} by Epoch (seed={seed})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, filename), dpi=300)
        plt.close()
    try:
        plt.figure(figsize=(7, 7))
        plt.plot(fpr, roc_tpr, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title(f"Best Model Untouched Test ROC (seed={seed})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(run_dir, "best_model_untouched_test_roc_curve.png"), dpi=300
        )
        plt.close()
    except:
        pass
    final_untouched_test_summary = {
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_score": float(best_checkpoint["best_score"]),
        "save_best_by": save_best_by,
        "untouched_test_loss": float(untouched_test_loss),
        "untouched_test_accuracy": float(untouched_test_metrics["accuracy"]),
        "untouched_test_precision": float(untouched_test_metrics["precision"]),
        "untouched_test_recall": float(untouched_test_metrics["recall"]),
        "untouched_test_f1": float(untouched_test_metrics["f1"]),
        "untouched_test_auc": (
            float(untouched_test_metrics["auc"])
            if not np.isnan(untouched_test_metrics["auc"])
            else np.nan
        ),
        "untouched_test_TPR": (
            float(untouched_test_TPR) if not np.isnan(untouched_test_TPR) else np.nan
        ),
        "untouched_test_TNR": (
            float(untouched_test_TNR) if not np.isnan(untouched_test_TNR) else np.nan
        ),
        "untouched_test_n_pos": int(untouched_test_metrics["n_pos"]),
        "untouched_test_n_neg": int(untouched_test_metrics["n_neg"]),
        "untouched_test_n_pred_pos": int(untouched_test_metrics["n_pred_pos"]),
        "untouched_test_n_pred_neg": int(untouched_test_metrics["n_pred_neg"]),
    }
    with open(
        os.path.join(run_dir, "best_model_untouched_test_summary.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(final_untouched_test_summary, f, indent=2)
    print(f"\nSeed {seed:05d} finished. Best model saved: {best_model_path}")
    print(f"Run folder: {run_dir}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--negative", type=Path, required=True, help="negative-pair HDF5"
    )
    parser.add_argument(
        "--positive", type=Path, required=True, help="positive-pair HDF5"
    )
    parser.add_argument("--alphabet", choices=ALPHABETS, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="default: HP2 1e-5; all others 5e-6",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[int(str(i) * 5) for i in range(10)]
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.40,
        help="fraction of rows remaining after test selection",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--transformer-layers", type=int, default=1)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-ff-dim", type=int, default=384)
    parser.add_argument(
        "--save-best-by",
        choices=("val_auc", "val_accuracy", "val_loss"),
        default="val_auc",
    )
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument(
        "--ablate",
        nargs="*",
        default=[],
        choices=(
            "one_mer_counts",
            "two_mer_counts",
            "three_mer_counts",
            "H1_matrix",
            "H2_matrix",
            "H1_features",
            "H2_features",
            "apaac1",
            "apaac2",
            "proportion",
            "length",
        ),
    )
    parser.add_argument("--ablation-constant", type=float, default=0.0)
    args = parser.parse_args()
    if (
        min(
            args.epochs,
            args.batch_size,
            args.patience,
            args.transformer_layers,
            args.transformer_heads,
            args.transformer_ff_dim,
        )
        < 1
    ):
        parser.error(
            "epoch, batch, patience, and Transformer dimensions must be positive"
        )
    if 144 % args.transformer_heads:
        parser.error("Transformer heads must divide the 144-dimensional embedding")
    if not 0 < args.test_ratio < 1 or not 0 < args.validation_ratio < 1:
        parser.error("split ratios must lie strictly between 0 and 1")
    if not 0 <= args.threshold <= 1 or not 0 <= args.dropout < 1:
        parser.error("invalid threshold or dropout")
    if args.cpu_threads is not None:
        if args.cpu_threads < 1:
            parser.error("--cpu-threads must be positive")
        torch.set_num_threads(args.cpu_threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error(
            "CUDA is unavailable; use --device cpu or a CUDA-enabled PyTorch installation"
        )
    config = get_alphabet(args.alphabet)
    selected_device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    for path in (args.negative, args.positive):
        if not path.is_file():
            parser.error(f"Input file does not exist: {path}")
        with h5py.File(path, "r") as handle:
            if "alphabet" in handle.attrs and handle.attrs["alphabet"] != args.alphabet:
                parser.error(f"Alphabet metadata mismatch in {path}")
            if (
                "alphabet_config" in handle.attrs
                and json.loads(handle.attrs["alphabet_config"]) != config
            ):
                parser.error(f"Alphabet/H1/H2 configuration mismatch in {path}")
    chosen_learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else config["learning_rate"]
    )
    if chosen_learning_rate <= 0 or args.weight_decay < 0:
        parser.error("learning rate must be positive and weight decay nonnegative")
    globals().update(
        {
            "neg_h5": str(args.negative.resolve()),
            "pos_h5": str(args.positive.resolve()),
            "output_root": str(args.output_root.resolve()),
            "alphabet_name": args.alphabet,
            "num_epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": chosen_learning_rate,
            "weight_decay": args.weight_decay,
            "untouched_test_ratio": args.test_ratio,
            "validation_ratio_within_remaining": args.validation_ratio,
            "prediction_threshold": args.threshold,
            "dropout": args.dropout,
            "transformer_layers": args.transformer_layers,
            "transformer_heads": args.transformer_heads,
            "transformer_ff_dim": args.transformer_ff_dim,
            "save_best_by": args.save_best_by,
            "early_stopping_patience": args.patience,
            "ablation_features": args.ablate,
            "ablation_constant": args.ablation_constant,
            "device": torch.device(selected_device),
        }
    )
    print(
        f"Alphabet: {args.alphabet}; device: {selected_device}; ablations: {args.ablate}; seeds: {args.seeds}"
    )
    for seed in args.seeds:
        run_seed(seed)
    print(f"Completed {len(args.seeds)} seed(s).")


if __name__ == "__main__":
    main()
