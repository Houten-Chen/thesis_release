import os
import ast
import json
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
from tqdm import tqdm

# =========================
# 路径与参数配置
# =========================
protein_csv = r"Z:\Structure_prediction\Original protein data\z_output_Q3\z_reviewed_annotation_4_5.csv"
rhp_dir = r"Z:\PCA\Original RHP data\z_output"
pca_json_path = "pca_weights_2d_2of5_groups.json"
grid_size = 2
min_points_for_bin = 20  # bin内蛋白z点数量下限
use_existing_map = True  # 是否使用已有结构map，若False则重新计算

# 输出文件路径
out_map_csv = "PCA_structure_map.csv"
out_map_png = "PCA_structure_map.png"
out_rhp_csv = "RHP_structure_tendency_score.csv"
out_curve_folder = "RHP_structure_rank_curves"
os.makedirs(out_curve_folder, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=UserWarning)

# =========================
# 通用函数
# =========================
def parse_z_vector(cell):
    if isinstance(cell, str) and cell.strip():
        try:
            vec = ast.literal_eval(cell)
            if isinstance(vec, list) and len(vec) == 16 and all(isinstance(x, (int, float)) for x in vec):
                return vec
        except Exception:
            return None
    return None

def q3_fraction(q3_window: str):
    L = len(q3_window)
    if L == 0:
        return 0.0, 0.0, 0.0
    c = Counter(q3_window)
    return c.get("H", 0)/L, c.get("E", 0)/L, c.get("C", 0)/L

def sliding_positions(seq_len: int, window: int = 50, step: int = 15):
    if seq_len < window:
        return []
    return list(range(0, seq_len - window + 1, step))

# =========================
# 1️⃣ 生成或读取结构倾向 map
# =========================
if use_existing_map and os.path.exists(out_map_csv):
    print("♻️ 使用已有结构map。")
    df_map = pd.read_csv(out_map_csv)
    gx = df_map["grid_x"].to_numpy()
    gy = df_map["grid_y"].to_numpy()
    H_vals = df_map["mean_H"].to_numpy()
    E_vals = df_map["mean_E"].to_numpy()
    C_vals = df_map["mean_C"].to_numpy()
    S_vals = df_map["structure_tendency"].to_numpy()
    x_unique = np.unique(gx)
    y_unique = np.unique(gy)
    nx, ny = len(x_unique), len(y_unique)
    x0, x1 = x_unique.min(), x_unique.max()
    y0, y1 = y_unique.min(), y_unique.max()
    H_grid = E_grid = C_grid = S_grid = None  # 占位，稍后从csv读出不再重算
else:
    print("📘 读取蛋白文件并计算三结构map...")
    df = pd.read_csv(protein_csv, low_memory=False)
    z_cols = [c for c in df.columns if c.startswith("z_")]

    all_z, all_H, all_E, all_C = [], [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        q3 = str(row["Q3"]) if not pd.isna(row["Q3"]) else ""
        L = len(q3)
        pos = sliding_positions(L, window=50, step=15)
        for k, zcol in enumerate(z_cols):
            vec = parse_z_vector(row[zcol])
            if vec is None or k >= len(pos):
                continue
            i0 = pos[k]
            q3_sub = q3[i0:i0+50]
            Hf, Ef, Cf = q3_fraction(q3_sub)
            all_z.append(vec)
            all_H.append(Hf)
            all_E.append(Ef)
            all_C.append(Cf)
    all_z, all_H, all_E, all_C = np.array(all_z, float), np.array(all_H, float), np.array(all_E, float), np.array(all_C, float)
    print(f"✅ Parsed {len(all_z)} protein z-points with H/E/C fractions.")

    # ===== PCA 投影 =====
    if os.path.exists(pca_json_path):
        with open(pca_json_path, "r") as f:
            pca_info = json.load(f)
        mean, comp = np.array(pca_info["mean"]), np.array(pca_info["components"])
        z_2d = (all_z - mean) @ comp.T[:, :2]
        print("♻️ 使用已有 PCA 权重。")
    else:
        pca = PCA(n_components=5, random_state=42)
        z_2d = pca.fit_transform(all_z)[:, :2]
        pca_info = {"mean": pca.mean_.tolist(), "components": pca.components_.tolist()}
        with open(pca_json_path, "w") as f:
            json.dump(pca_info, f, indent=2)
        mean, comp = pca.mean_, pca.components_
        print("📀 已保存新 PCA 权重。")

    PC1, PC2 = z_2d[:, 0], z_2d[:, 1]
    x0, y0 = math.floor(PC1.min()/grid_size)*grid_size, math.floor(PC2.min()/grid_size)*grid_size
    x1, y1 = math.ceil(PC1.max()/grid_size)*grid_size, math.ceil(PC2.max()/grid_size)*grid_size
    ix, iy = np.floor((PC1-x0)/grid_size).astype(int), np.floor((PC2-y0)/grid_size).astype(int)
    nx, ny = int((x1-x0)/grid_size), int((y1-y0)/grid_size)

    df_grid = pd.DataFrame({"ix": ix, "iy": iy, "H": all_H, "E": all_E, "C": all_C})
    grid_stats = df_grid.groupby(["ix", "iy"]).agg(
        mean_H=("H", "mean"),
        mean_E=("E", "mean"),
        mean_C=("C", "mean"),
        count=("C", "size")
    ).reset_index()

    H_grid = np.full((ny, nx), np.nan)
    E_grid = np.full((ny, nx), np.nan)
    C_grid = np.full((ny, nx), np.nan)

    for _, r in grid_stats.iterrows():
        gx, gy = int(r["ix"]), int(r["iy"])
        if 0 <= gx < nx and 0 <= gy < ny and r["count"] >= min_points_for_bin:
            H_grid[gy, gx] = r["mean_H"]
            E_grid[gy, gx] = r["mean_E"]
            C_grid[gy, gx] = r["mean_C"]

    # ===== 结构倾向 map =====
    S_grid = np.full_like(C_grid, np.nan)
    mask = np.isfinite(H_grid) & np.isfinite(E_grid) & np.isfinite(C_grid)
    S_grid[mask] = (H_grid[mask] + E_grid[mask]) / (C_grid[mask] + 1e-12)
    S_grid[mask] = 10 * np.log((H_grid[mask] + E_grid[mask]) / (C_grid[mask] + 1e-12))

    gx_centers = x0 + (np.arange(nx) + 0.5) * grid_size
    gy_centers = y0 + (np.arange(ny) + 0.5) * grid_size
    GX, GY = np.meshgrid(gx_centers, gy_centers)

    df_map = pd.DataFrame({
        "grid_x": GX.ravel(),
        "grid_y": GY.ravel(),
        "mean_H": H_grid.ravel(),
        "mean_E": E_grid.ravel(),
        "mean_C": C_grid.ravel(),
        "structure_tendency": S_grid.ravel()
    })
    df_map.to_csv(out_map_csv, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(3, 2))
    plt.imshow(S_grid, origin="lower", extent=[x0,x1,y0,y1],
               cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title("Structure tendency map (10ln((H+E)/C))")
    plt.colorbar(label="Structure tendency score")
    plt.tight_layout(); plt.savefig(out_map_png, dpi=300); plt.close()
    print(f"✅ Structure map saved: {out_map_csv}, {out_map_png}")

# =========================
# ✨ 无论是否使用已有 map，都重新绘制一份 map PNG
# =========================

print("🖼️ 正在重新绘制 Structure tendency map PNG...")

S_mat = df_map["structure_tendency"].to_numpy().reshape(ny, nx)

plt.figure(figsize=(3.6, 2.2))
plt.imshow(
    S_mat, origin="lower",
    extent=[x0, x1, y0, y1],
    cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto"
)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.ylim(-50, 50)
plt.xlim(-60, 80)
#plt.xticks([])   # 去掉 x 轴数字
#plt.yticks([])   # 去掉 y 轴数字
plt.colorbar(label="Structure tendency score")
plt.tight_layout()
plt.savefig(out_map_png, dpi=300)
plt.savefig(out_map_png+".pdf")
plt.close()

print(f"🎨 新的结构 map PNG 已重新绘制：{out_map_png}")

# =========================
# 2️⃣ 读取 RHP z 向量
# =========================
def read_rhp_z_vectors(rhp_dir):
    rhp_data = {}
    for file in os.listdir(rhp_dir):
        if file.startswith("z_RHP(") and file.endswith(".csv"):
            name = os.path.splitext(file)[0]
            path = os.path.join(rhp_dir, file)
            df_header = pd.read_csv(path, nrows=0)
            z_cols = [c for c in df_header.columns if c.startswith("z_")]
            if not z_cols:
                continue
            df_z = pd.read_csv(path, usecols=z_cols, dtype='string', low_memory=False)
            vecs = []
            for _, row in df_z.iterrows():
                for col in z_cols:
                    cell = row[col]
                    if isinstance(cell, str) and cell.strip():
                        try:
                            v = ast.literal_eval(cell)
                            if isinstance(v, list) and len(v)==16 and all(isinstance(x,(int,float)) for x in v):
                                vecs.append(v)
                        except:
                            continue
            if vecs:
                rhp_data[name] = np.array(vecs, float)
    return rhp_data

rhp_dict = read_rhp_z_vectors(rhp_dir)
print(f"✅ 共载入 {len(rhp_dict)} 个 RHP 文件。")

# =========================
# 3️⃣ 计算每个 RHP 的结构倾向分数
# =========================
with open(pca_json_path, "r") as f:
    pca_info = json.load(f)
mean, comp = np.array(pca_info["mean"]), np.array(pca_info["components"])

results = []
for rhp_name, z_arr in tqdm(rhp_dict.items()):
    z_2d = (z_arr - mean) @ comp.T[:, :2]
    PC1, PC2 = z_2d[:,0], z_2d[:,1]
    ix = np.floor((PC1-x0)/grid_size).astype(int)
    iy = np.floor((PC2-y0)/grid_size).astype(int)

    s_vals, nan_vals = [], []
    for x, y in zip(ix, iy):
        if 0 <= x < nx and 0 <= y < ny and np.isfinite(df_map["structure_tendency"].to_numpy().reshape(ny, nx)[y, x]):
            val = df_map["structure_tendency"].to_numpy().reshape(ny, nx)[y, x]
            s_vals.append(val)
        else:
            nan_vals.append(np.nan)
    combined = sorted(s_vals, reverse=True) + nan_vals

    avg_s = np.nanmean(s_vals) if s_vals else np.nan
    coverage = len(s_vals) / (len(s_vals) + len(nan_vals)) if (s_vals or nan_vals) else np.nan
    results.append({
        "RHP": rhp_name,
        "mean_structure_score": avg_s,
        "coverage_ratio": coverage,
        "N_points": len(s_vals) + len(nan_vals)
    })

    # 绘制排序曲线
    x_vals = np.arange(len(combined))
    plt.figure(figsize=(3,3))
    plt.plot(x_vals, combined, color="#007ACC", lw=1.8)
    plt.xlabel(f"{rhp_name} structure tendency rank")
    plt.ylabel("Structure tendency 10ln((H+E)/C)")
    plt.ylim(0, 10)
    plt.xlim(0, 52000)
    plt.tight_layout()
    out_curve_path = os.path.join(out_curve_folder, f"{rhp_name}_structure_curve.png")
    plt.savefig(out_curve_path, dpi=300)
    plt.close()
    print(f"  ✅ 曲线保存: {out_curve_path}")

# =========================
# 4️⃣ 汇总输出
# =========================
df_res = pd.DataFrame(results)
df_res.to_csv(out_rhp_csv, index=False, encoding="utf-8-sig")
print(f"✅ RHP 平均结构倾向值表已保存：{out_rhp_csv}")
print("🎯 All done.")
