import os
import json
import h5py
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    roc_curve, auc,
    precision_score, recall_score, f1_score, accuracy_score
)

# =========================================================
# Device
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# =========================================================
# Paths
# =========================================================
neg_h5 = r"Z:\PRI deeplearning\training_dataset\extracted_features_negatome.hdf5"
pos_h5 = r"Z:\PRI deeplearning\training_dataset\extracted_features_biogrid.hdf5"

output_root = r"Z:\PRI deeplearning\training_runs_pytorch"

# =========================================================
# Ablation Configuration（本版本：pos全部不使用，只保留sum）
# =========================================================
# ablation_features list（对proteinA和proteinB同时生效，不需要分别写A/B）
# 支持的key（注释中列出所有可能，后续ablation版本直接修改这个list即可）：
# "one_mer_counts", "two_mer_counts", "three_mer_counts",
# "H1_matrix", "H2_matrix",
# "H1_features", "H2_features", "apaac1", "apaac2", "proportion", "length"
ablation_features = [
#    "one_mer_counts",
#    "two_mer_counts",
#    "three_mer_counts",
#    "H1_matrix",
#    "H2_matrix",
#    "proportion",
#    "length",
#    "apaac1",
#    "apaac2",
]  # ← 当前版本：完全抹除pos特征（只保留sum特征）

ablation_constant = 0.0  # 抹除时全部替换成这个常数（保持输入shape不变）

# =========================================================
# Config
# =========================================================
num_epochs = 24
batch_size = 16
learning_rate = 5e-6
weight_decay = 1e-5

untouched_test_ratio = 0.20
validation_ratio_within_remaining = 0.40

prediction_threshold = 0.5
force_labels_by_source = True

dropout = 0.25

# Transformer config
transformer_layers = 1
transformer_heads = 4
transformer_ff_dim = 384

# Saving and selection
save_every_epoch = False          # Ablation版本：只保存best model
save_best_by = "val_auc"          # options: val_auc, val_accuracy, val_loss
early_stopping_patience = 24

# =========================================================
# Seeds to run（10个seed循环）
# =========================================================
seeds = [int(str(i) * 5) for i in range(10)]   # 00000, 11111, ..., 99999

# =========================================================
# Reproducibility
# =========================================================
rng_global = None  # 会在每个seed中重新设置

# =========================================================
# Output folder helper
# =========================================================
def get_run_dir(seed):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ablation_tag = "ablation_pos_onlysum"   # 本版本标识
    return os.path.join(
        output_root,
        f"run_ModelB_transformer_{ablation_tag}_seed{seed:05d}_{timestamp}"
    )

# =========================================================
# Utilities（原函数全部保留，仅新增ablation相关）
# =========================================================
def read_h5_to_dict(h5_path):
    data = {}
    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            data[key] = f[key][:]
    return data


def ensure_required_keys(data_dict, source_name):
    required = [
        "interaction", "proteinA", "proteinB",
        "one_mer_countsA", "two_mer_countsA", "three_mer_countsA",
        "one_mer_countsB", "two_mer_countsB", "three_mer_countsB",
        "proportion_A", "length_A",
        "H1_featuresA", "H2_featuresA", "H1_matrixA", "H2_matrixA", "apaac1A", "apaac2A",
        "proportion_B", "length_B",
        "H1_featuresB", "H2_featuresB", "H1_matrixB", "H2_matrixB", "apaac1B", "apaac2B",
    ]
    missing = [k for k in required if k not in data_dict]
    if missing:
        raise KeyError(f"{source_name} is missing required keys: {missing}")


def decode_if_bytes(arr):
    if hasattr(arr, "dtype") and arr.dtype.kind == "S":
        return arr.astype(str)
    return arr


def apply_pos_ablation(pos_features, ablation_list):
    """对posA/posB进行抹除（只修改指定的块，保持整体shape [N,215,20]不变）"""
    if not ablation_list:
        return pos_features

    # pos特征拼接顺序（必须与build_feature_views中一致）
    blocks = {
        "one_mer_counts": (0, 5),
        "two_mer_counts": (5, 30),      # 5 + 25
        "three_mer_counts": (30, 155),  # 30 + 125
        "H1_matrix": (155, 185),
        "H2_matrix": (185, 215),
    }

    masked = pos_features.copy()
    for feat_name in ablation_list:
        if feat_name in blocks:
            start, end = blocks[feat_name]
            masked[:, start:end, :] = ablation_constant
            print(f"  [Ablation] {feat_name} (pos) 已抹除 → 常数 {ablation_constant}")
    return masked

def apply_sum_ablation(sum_features, ablation_list):
    """对sumA/sumB进行抹除（保持 [N, 126] shape 不变）"""
    if not ablation_list:
        return sum_features

    # sum特征拼接顺序（必须与 build_feature_views 中一致）
    blocks = {
        "H1_features": (0, 30),
        "H2_features": (30, 60),
        "apaac1": (60, 90),
        "apaac2": (90, 120),
        "proportion": (120, 125),
        "length": (125, 126),
    }

    masked = sum_features.copy()
    for feat_name in ablation_list:
        if feat_name in blocks:
            start, end = blocks[feat_name]
            masked[:, start:end] = ablation_constant
            print(f"  [Ablation] {feat_name} (sum) 已抹除 → 常数 {ablation_constant}")
    return masked

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

# 构建posA / posB（保持原样）
    posA = np.concatenate(
        (one_mer_countsA, two_mer_countsA, three_mer_countsA, H1_matrixA, H2_matrixA),
        axis=1
    )
    posB = np.concatenate(
        (one_mer_countsB, two_mer_countsB, three_mer_countsB, H1_matrixB, H2_matrixB),
        axis=1
    )

    # === pos特征抹除 ===
    if ablation_features:
        posA = apply_pos_ablation(posA, ablation_features)
        posB = apply_pos_ablation(posB, ablation_features)

    # 构建sumA / sumB
    sumA = np.concatenate((H1_featuresA, H2_featuresA, apaac1A, apaac2A), axis=1)
    sumB = np.concatenate((H1_featuresB, H2_featuresB, apaac1B, apaac2B), axis=1)

    length_A_2d = length_A[:, np.newaxis]
    length_B_2d = length_B[:, np.newaxis]

    sumA = np.concatenate((sumA, proportion_A, length_A_2d), axis=1)
    sumB = np.concatenate((sumB, proportion_B, length_B_2d), axis=1)

    # === 新增：sum特征抹除 ===
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
    return subset_features(d, indices), indices


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


def split_class_indices(indices_all, untouched_test_ratio, validation_ratio_within_remaining, rng):
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

    return train_idx, val_idx, test_idx


def build_balanced_train_val_untouched_test_split(
    neg,
    pos,
    rng,
    untouched_test_ratio=0.20,
    validation_ratio_within_remaining=0.40,
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
    untouched_test_set, untouched_test_shuffle_order = shuffle_feature_dict(untouched_test_set, rng)

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

    return train_set, val_set, untouched_test_set, split_records


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

    untouched_test_tpr = float(np.mean(pred_label[pos_mask] == 1)) if np.sum(pos_mask) > 0 else np.nan
    untouched_test_tnr = float(np.mean(pred_label[neg_mask] == 0)) if np.sum(neg_mask) > 0 else np.nan

    return untouched_test_tpr, untouched_test_tnr


def get_selection_score(record, save_best_by):
    if save_best_by == "val_auc":
        return record["val_auc"], True
    if save_best_by == "val_accuracy":
        return record["val_accuracy"], True
    if save_best_by == "val_loss":
        return record["val_loss"], False
    raise ValueError(f"Unknown save_best_by: {save_best_by}")


# =========================================================
# Model（保持完全不变）
# =========================================================
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

        pos_flat_dim = int(np.prod(pos_shape))  # 4300
        sum_dim = int(sum_shape[0])             # 126

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
            encoder_layer,
            num_layers=transformer_layers,
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


# =========================================================
# Train / eval helpers
# =========================================================
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

    return avg_loss, all_probs, all_labels


# =========================================================
# 单个seed的完整训练流程
# =========================================================
def run_seed(seed: int):
    # Reproducibility
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)

    run_dir = get_run_dir(seed)
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n=== Ablation Run - Seed {seed:05d} ===\nRun folder: {run_dir}")

    # Load data（每次seed都重建feature views，保证ablation生效）
    print("Loading HDF5 files.")
    neg_raw = read_h5_to_dict(neg_h5)
    pos_raw = read_h5_to_dict(pos_h5)

    ensure_required_keys(neg_raw, "Negatome HDF5")
    ensure_required_keys(pos_raw, "BioGRID HDF5")

    neg = build_feature_views(neg_raw, ablation_features=ablation_features)
    pos = build_feature_views(pos_raw, ablation_features=ablation_features)

    if force_labels_by_source:
        neg["interaction"] = np.zeros_like(neg["interaction"], dtype=np.float32)
        pos["interaction"] = np.ones_like(pos["interaction"], dtype=np.float32)

    print(f"Negatome total: {len(neg['interaction'])}")
    print(f"BioGRID total: {len(pos['interaction'])}")

    # Fixed split
    train_set, val_set, untouched_test_set, split_records = build_balanced_train_val_untouched_test_split(
        neg=neg,
        pos=pos,
        rng=rng,
        untouched_test_ratio=untouched_test_ratio,
        validation_ratio_within_remaining=validation_ratio_within_remaining,
    )

    train_loader = make_dataloader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = make_dataloader(val_set, batch_size=batch_size, shuffle=False)
    untouched_test_loader = make_dataloader(untouched_test_set, batch_size=batch_size, shuffle=False)

    print("\nFixed balanced split:")
    print(f"  train total: {len(train_set['interaction'])} (pos: {int(np.sum(train_set['interaction']==1))})")
    print(f"  validation total: {len(val_set['interaction'])}")
    print(f"  untouched test total: {len(untouched_test_set['interaction'])}")

    with open(os.path.join(run_dir, "fixed_split_indices.json"), "w", encoding="utf-8") as f:
        json.dump(split_records, f, indent=2)

    # Model
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
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: total={n_params:,}")

    # Save config (包含ablation信息)
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
        "model_type": "ModelB_pair_tokens_transformer_untouched20_ablation_pos_onlysum",
        "n_params": int(n_params),
        "input_shape_posA": list(input_shape_posA),
        "input_shape_sumA": list(input_shape_sumA),
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Training loop
    history_records = []
    best_score = None
    best_epoch = None
    epochs_without_improvement = 0
    best_model_path = os.path.join(run_dir, "best_model.pt")

    for epoch in range(1, num_epochs + 1):
        print(f"\n================ Epoch {epoch}/{num_epochs} (seed={seed}) ================")

        train_loss, train_pred_prob, y_train = run_one_epoch(
            model=model, loader=train_loader, criterion=criterion, optimizer=optimizer, device=device
        )
        val_loss, val_pred_prob, y_val = run_one_epoch(
            model=model, loader=val_loader, criterion=criterion, optimizer=None, device=device
        )
        untouched_test_loss, untouched_test_pred_prob, y_untouched_test = run_one_epoch(
            model=model, loader=untouched_test_loader, criterion=criterion, optimizer=None, device=device
        )

        train_metrics = compute_basic_metrics(y_train, train_pred_prob, threshold=prediction_threshold)
        val_metrics = compute_basic_metrics(y_val, val_pred_prob, threshold=prediction_threshold)
        untouched_test_metrics = compute_basic_metrics(y_untouched_test, untouched_test_pred_prob, threshold=prediction_threshold)

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
            "train_auc": float(train_metrics["auc"]) if not np.isnan(train_metrics["auc"]) else np.nan,
            "val_loss": float(val_loss),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_precision": float(val_metrics["precision"]),
            "val_recall": float(val_metrics["recall"]),
            "val_f1": float(val_metrics["f1"]),
            "val_auc": float(val_metrics["auc"]) if not np.isnan(val_metrics["auc"]) else np.nan,
            "untouched_test_loss": float(untouched_test_loss),
            "untouched_test_accuracy": float(untouched_test_metrics["accuracy"]),
            "untouched_test_precision": float(untouched_test_metrics["precision"]),
            "untouched_test_recall": float(untouched_test_metrics["recall"]),
            "untouched_test_f1": float(untouched_test_metrics["f1"]),
            "untouched_test_auc": float(untouched_test_metrics["auc"]) if not np.isnan(untouched_test_metrics["auc"]) else np.nan,
            "untouched_test_TPR": float(untouched_test_TPR) if not np.isnan(untouched_test_TPR) else np.nan,
            "untouched_test_TNR": float(untouched_test_TNR) if not np.isnan(untouched_test_TNR) else np.nan,
            "untouched_test_n_pos": int(untouched_test_metrics["n_pos"]),
            "untouched_test_n_neg": int(untouched_test_metrics["n_neg"]),
            "untouched_test_n_pred_pos": int(untouched_test_metrics["n_pred_pos"]),
            "untouched_test_n_pred_neg": int(untouched_test_metrics["n_pred_neg"]),
        }
        history_records.append(record)

        current_score, larger_is_better = get_selection_score(record, save_best_by)
        if best_score is None or (larger_is_better and current_score > best_score) or (not larger_is_better and current_score < best_score):
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

        print(f"  train_loss={train_loss:.4f}  val_auc={val_metrics['auc']:.4f}  untouched_test_auc={untouched_test_metrics['auc']:.4f}")
        print(f"  best_epoch={best_epoch}  best_{save_best_by}={best_score:.4f}  no_improve={epochs_without_improvement}")

        # 保存历史CSV（每次epoch都更新）
        history_df = pd.DataFrame(history_records)
        history_df.to_csv(os.path.join(run_dir, "epoch_history.csv"), index=False, encoding="utf-8-sig")

        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    # Final untouched test evaluation using best model
    print("\nLoading best model for final untouched test evaluation.")
    best_checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.to(device)

    untouched_test_loss, untouched_test_pred_prob, y_untouched_test = run_one_epoch(
        model=model, loader=untouched_test_loader, criterion=criterion, optimizer=None, device=device
    )

    untouched_test_metrics = compute_basic_metrics(y_untouched_test, untouched_test_pred_prob, threshold=prediction_threshold)
    untouched_test_TPR, untouched_test_TNR = compute_untouched_test_tpr_tnr(
        y_untouched_test, untouched_test_pred_prob, threshold=prediction_threshold
    )

    # 保存预测结果
    untouched_test_pred_label = (untouched_test_pred_prob >= prediction_threshold).astype(int)
    pred_df = pd.DataFrame({
        "y_true": y_untouched_test.astype(int),
        "pred_prob": untouched_test_pred_prob,
        "pred_label": untouched_test_pred_label,
    })
    pred_df.to_csv(os.path.join(run_dir, "best_model_untouched_test_predictions.csv"), index=False, encoding="utf-8-sig")

    # ROC + threshold sweep + fixed TNR tables（全部保留）
    try:
        fpr, roc_tpr, thresholds = roc_curve(y_untouched_test.astype(int), untouched_test_pred_prob)
        roc_auc = auc(fpr, roc_tpr)
        roc_df = pd.DataFrame({"fpr": fpr, "roc_tpr": roc_tpr, "threshold": thresholds})
        roc_df.to_csv(os.path.join(run_dir, "best_model_untouched_test_roc_curve.csv"), index=False, encoding="utf-8-sig")

        # threshold sweep
        sweep_records = []
        for th in np.linspace(0, 1, 1001):
            m = compute_basic_metrics(y_untouched_test, untouched_test_pred_prob, threshold=th)
            th_tpr, th_tnr = compute_untouched_test_tpr_tnr(y_untouched_test, untouched_test_pred_prob, threshold=th)
            sweep_records.append({
                "threshold": float(th),
                "untouched_test_accuracy": m["accuracy"],
                "untouched_test_precision": m["precision"],
                "untouched_test_recall": m["recall"],
                "untouched_test_f1": m["f1"],
                "untouched_test_TPR": th_tpr,
                "untouched_test_TNR": th_tnr,
                "untouched_test_n_pred_pos": m["n_pred_pos"],
                "untouched_test_n_pred_neg": m["n_pred_neg"],
            })
        sweep_df = pd.DataFrame(sweep_records)
        sweep_df.to_csv(os.path.join(run_dir, "best_model_untouched_test_threshold_sweep.csv"), index=False, encoding="utf-8-sig")

        # fixed TNR
        fixed_tnr_records = []
        for target_tnr in [0.90, 0.95, 0.97, 0.98, 0.99]:
            candidates = sweep_df[sweep_df["untouched_test_TNR"] >= target_tnr].copy()
            if len(candidates) == 0:
                fixed_tnr_records.append({"target_untouched_test_TNR": target_tnr, "found": False, "threshold": np.nan, "untouched_test_TPR": np.nan, "untouched_test_TNR": np.nan})
            else:
                best_row = candidates.sort_values(["untouched_test_TPR", "untouched_test_accuracy"], ascending=False).iloc[0]
                fixed_tnr_records.append({
                    "target_untouched_test_TNR": target_tnr,
                    "found": True,
                    "threshold": float(best_row["threshold"]),
                    "untouched_test_TPR": float(best_row["untouched_test_TPR"]),
                    "untouched_test_TNR": float(best_row["untouched_test_TNR"]),
                })
        pd.DataFrame(fixed_tnr_records).to_csv(
            os.path.join(run_dir, "best_model_untouched_test_TPR_at_fixed_untouched_test_TNR.csv"),
            index=False, encoding="utf-8-sig"
        )
    except Exception as e:
        print("ROC / sweep generation skipped:", e)

    # Plots（全部保留）
    history_df = pd.DataFrame(history_records)
    for metric, filename, ylabel in [
        (["train_loss", "val_loss", "untouched_test_loss"], "loss_curve.png", "Loss"),
        (["train_accuracy", "val_accuracy", "untouched_test_accuracy"], "accuracy_curve.png", "Accuracy"),
        (["train_auc", "val_auc", "untouched_test_auc"], "auc_curve.png", "AUC"),
        (["untouched_test_TPR", "untouched_test_TNR"], "untouched_test_tpr_tnr_curve.png", "Rate"),
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

    # ROC plot
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
        plt.savefig(os.path.join(run_dir, "best_model_untouched_test_roc_curve.png"), dpi=300)
        plt.close()
    except:
        pass

    # Final summary JSON
    final_untouched_test_summary = {
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_score": float(best_checkpoint["best_score"]),
        "save_best_by": save_best_by,
        "untouched_test_loss": float(untouched_test_loss),
        "untouched_test_accuracy": float(untouched_test_metrics["accuracy"]),
        "untouched_test_precision": float(untouched_test_metrics["precision"]),
        "untouched_test_recall": float(untouched_test_metrics["recall"]),
        "untouched_test_f1": float(untouched_test_metrics["f1"]),
        "untouched_test_auc": float(untouched_test_metrics["auc"]) if not np.isnan(untouched_test_metrics["auc"]) else np.nan,
        "untouched_test_TPR": float(untouched_test_TPR) if not np.isnan(untouched_test_TPR) else np.nan,
        "untouched_test_TNR": float(untouched_test_TNR) if not np.isnan(untouched_test_TNR) else np.nan,
        "untouched_test_n_pos": int(untouched_test_metrics["n_pos"]),
        "untouched_test_n_neg": int(untouched_test_metrics["n_neg"]),
        "untouched_test_n_pred_pos": int(untouched_test_metrics["n_pred_pos"]),
        "untouched_test_n_pred_neg": int(untouched_test_metrics["n_pred_neg"]),
    }
    with open(os.path.join(run_dir, "best_model_untouched_test_summary.json"), "w", encoding="utf-8") as f:
        json.dump(final_untouched_test_summary, f, indent=2)

    print(f"\nSeed {seed:05d} finished. Best model saved: {best_model_path}")
    print(f"Run folder: {run_dir}\n")


# =========================================================
# Main: 循环跑10个seed
# =========================================================
if __name__ == "__main__":
    print("=== Starting Ablation Test (pos全部不使用，只用sum) ===")
    print(f"Ablation features: {ablation_features}")
    print(f"Seeds: {seeds}\n")

    for seed in seeds:
        run_seed(seed)

    print("\nAll 10 seeds completed successfully!")
    print("每个seed只保存了 best_model.pt，其余图表、CSV、ROC、threshold sweep 全部保留。")
