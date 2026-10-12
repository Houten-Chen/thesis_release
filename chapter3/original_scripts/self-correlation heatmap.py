import os
import ast
import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm
from sklearn.decomposition import PCA

# =====================================
# 可调参数区
# =====================================
protein_csv = r"Z:\Structure_prediction\Original protein data\z_output_Q3\z_reviewed_annotation_4_5.csv"
pca_json_path = "pca_weights_2d_2of5_groups.json"

# 输出文件
output_csv = "HLB_lag_combined_map.csv"  # 包含平均→方差
output_mean_png = "HLB_lag_mean_heatmap.png"
output_var_png = "HLB_lag_var_heatmap.png"

# 参数设置
use_existing_csv = True           # 是否直接读取已有csv
grid_size = 2                      # PCA bin大小
max_lag = 30                       # 最大自相关距离
window = 50                        # 滑窗长度
step = 15                          # 滑窗步长
min_points_for_bin = 10            # 每个bin最少点数
cmap = "RdBu_r"                  # 热图配色

# 🎨 热图颜色上下限（可调）
vmin_mean, vmax_mean = 0.0, 0.15    # 对应 HLB_lag_mean_heatmap
vmin_var, vmax_var = 3.388, 3.395      # 对应 entropy 热图范围
# =====================================

plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False

# =====================================
# ====== 氨基酸 → RHP monomer 编码 ======
# =====================================
def map_sequence_to_rhp(seq):
    return ''.join([
        '4' if aa in 'EDRK' else
        '3' if aa in 'LIFWVM' else
        '2' if aa in 'SQHNP' else
        '1'
        for aa in seq
    ])

# HLB 映射表
hlb_dict = {
    '1': 8.45,   # MMA hydrophobic
    '2': 11.42,  # OEGMA500 hydrophilic
    '3': 5.125,  # EHMA very hydrophobic
    '4': 18.5    # SPMA
}

# =====================================
# ====== 基础函数 ======
# =====================================
def parse_z_vector(cell):
    if isinstance(cell, str) and cell.strip():
        try:
            v = ast.literal_eval(cell)
            if isinstance(v, list) and len(v) == 16:
                return np.array(v, float)
        except:
            return None
    return None

def sliding_positions(seq_len, window=50, step=15):
    if seq_len < window:
        return []
    return list(range(0, seq_len - window + 1, step))

def calc_lag_diffs(hlb_arr, max_lag=30):
    diffs = []
    for lag in range(1, max_lag + 1):
        vals = [abs(hlb_arr[i] - hlb_arr[i + lag]) for i in range(len(hlb_arr) - lag)]
        diffs.append(np.mean(vals) if vals else np.nan)
    return np.array(diffs)

# ===== 熵计算函数 =====
def calc_entropy(lag_vals, eps=1e-12):
    """计算单条序列 lag 分布的 Shannon 熵"""
    p = np.abs(lag_vals)
    total = np.nansum(p)
    if total == 0 or np.isnan(total):
        return np.nan
    p = p / total
    return -np.nansum(p * np.log(p + eps))

# =====================================
# ====== 主体计算部分 ======
# =====================================
if use_existing_csv and os.path.exists(output_csv):
    print(f"♻️ 使用已有 lag 结果文件：{output_csv}")
    df_combined = pd.read_csv(output_csv)
    nx = len(df_combined["grid_x"].unique())
    ny = len(df_combined["grid_y"].unique())
    lag_mean_after = df_combined["var_mean_after"].values.reshape(ny, nx)
    lag_var_first = df_combined["var_point_first"].values.reshape(ny, nx)
    # ===== 从 CSV 中恢复坐标范围 =====
    x0, x1 = df_combined["grid_x"].min(), df_combined["grid_x"].max()
    y0, y1 = df_combined["grid_y"].min(), df_combined["grid_y"].max()

else:
    print("📘 读取蛋白文件...")
    df = pd.read_csv(protein_csv, low_memory=False)
    z_cols = [c for c in df.columns if c.startswith("z_")]

    all_z, all_lag_vals, all_lag_var = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        seq = str(row["Sequence"]) if not pd.isna(row["Sequence"]) else ""
        seq_rhp = map_sequence_to_rhp(seq)
        seq_len = len(seq_rhp)
        pos_list = sliding_positions(seq_len, window=window, step=step)

        for k, zcol in enumerate(z_cols):
            vec = parse_z_vector(row[zcol])
            if vec is None or k >= len(pos_list):
                continue
            i0 = pos_list[k]
            sub_seq = seq_rhp[i0:i0 + window]
            hlb_vals = np.array([hlb_dict[m] for m in sub_seq])
            lag_vals = calc_lag_diffs(hlb_vals, max_lag)
            entropy_val = calc_entropy(lag_vals)
            all_z.append(vec)
            all_lag_vals.append(lag_vals)
            all_lag_var.append(entropy_val)

    all_z = np.array(all_z, float)
    all_lag_vals = np.array(all_lag_vals, float)
    all_lag_var = np.array(all_lag_var, float)
    print(f"✅ Parsed {len(all_z)} z-points with lag profiles.")

    # ====== PCA 投影（修正） ======
    if os.path.exists(pca_json_path):
        with open(pca_json_path, "r") as f:
            pca_info = json.load(f)
        mean = np.array(pca_info["mean"])
        comp = np.array(pca_info["components"])
        z_2d = (all_z - mean) @ comp.T[:, :2]  # ✅ 与 RHP structure prediction 一致
        print("♻️ 使用已有 PCA 权重（已修正轴方向）。")
    else:
        pca = PCA(n_components=5, random_state=42)
        z_2d = pca.fit_transform(all_z)[:, :2]
        print("⚠️ 未找到 PCA 权重，使用新拟合结果（会导致轴偏移）。")

    PC1, PC2 = z_2d[:, 0], z_2d[:, 1]
    x0, y0 = math.floor(PC1.min()/grid_size)*grid_size, math.floor(PC2.min()/grid_size)*grid_size
    x1, y1 = math.ceil(PC1.max()/grid_size)*grid_size, math.ceil(PC2.max()/grid_size)*grid_size
    ix, iy = np.floor((PC1 - x0)/grid_size).astype(int), np.floor((PC2 - y0)/grid_size).astype(int)
    nx, ny = int((x1 - x0)/grid_size), int((y1 - y0)/grid_size)

    bins = {}
    for i, (bx, by) in enumerate(zip(ix, iy)):
        if 0 <= bx < nx and 0 <= by < ny:
            bins.setdefault((bx, by), []).append(i)

    var_mean_after_map = np.full((ny, nx), np.nan)
    var_point_first_map = np.full((ny, nx), np.nan)
    
    # ⭐⭐ 新增：保存每个 bin 的 lag 平均值
    lag_profile_map = {}  # key = (bx,by), val = 1×max_lag vector
    
    for (bx, by), idxs in tqdm(bins.items(), desc="Calculating bin stats"):
        if len(idxs) < min_points_for_bin:
            continue
        
        subset = all_lag_vals[idxs, :]  # shape = (#points, max_lag)

        # ====== 计算每个 lag 的平均值（你要保存的 n 个 lag 值） ======
        mean_lag_profile = np.nanmean(subset, axis=0)  # shape = (max_lag,)
        lag_profile_map[(bx, by)] = mean_lag_profile
    
        # ====== 原本的 2 个指标 ======
        lag_mean_after_map[by, bx] = np.nanvar(mean_lag_profile)
        var_point_first_map[by, bx] = np.nanmean(all_lag_var[idxs])

    gx_centers = x0 + (np.arange(nx) + 0.5) * grid_size
    gy_centers = y0 + (np.arange(ny) + 0.5) * grid_size
    GX, GY = np.meshgrid(gx_centers, gy_centers)

    rows = []
    for by in range(ny):
        for bx in range(nx):

            # 基本字段
            row = {
                "grid_x": gx_centers[bx],
                "grid_y": gy_centers[by],
                "var_mean_after": lag_mean_after_map[by, bx],
                "var_point_first": lag_var_first_map[by, bx],
            }

            # ⭐⭐ 加入 lag_1 ~ lag_30
            if (bx, by) in lag_profile_map:
                lag_vec = lag_profile_map[(bx, by)]
                for i in range(max_lag):
                    row[f"lag_{i+1}"] = lag_vec[i]
            else:
                for i in range(max_lag):
                    row[f"lag_{i+1}"] = np.nan

            rows.append(row)

    df_combined = pd.DataFrame(rows)
    df_combined.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"✅ 结果文件已保存：{output_csv}")

    lag_mean_after = var_mean_after_map
    lag_var_first = var_point_first_map

# ====== 绘制 mean→var 热图 ======
plt.figure(figsize=(3.6, 2.2))
plt.imshow(lag_mean_after, origin="lower", cmap=cmap, aspect="auto",
           vmin=vmin_mean, vmax=vmax_mean, extent=[x0, x1, y0, y1])
cb = plt.colorbar(label="Lag variance")
cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # 保留三位小数

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.ylim(-50, 50)
plt.xlim(-60, 80)
#plt.xticks([])   # 去掉 x 轴数字
#plt.yticks([])   # 去掉 y 轴数字
plt.tight_layout()
plt.savefig(output_mean_png, dpi=300)
plt.savefig(output_mean_png+".pdf")
plt.close()

# ===== 绘制 entropy→mean 热图 =====
plt.figure(figsize=(3.5, 2))
plt.imshow(lag_var_first, origin="lower", cmap=cmap, aspect="auto",
           vmin=vmin_var, vmax=vmax_var, extent=[x0, x1, y0, y1])
cb = plt.colorbar(label="Entropy (point→mean)")
cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.xticks([])   # 去掉 x 轴数字
plt.yticks([])   # 去掉 y 轴数字
plt.tight_layout()
plt.savefig(output_var_png, dpi=300)
plt.close()

print(f"🎯 热图已保存：{output_mean_png}, {output_var_png}")
