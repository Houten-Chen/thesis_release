import os
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import random
import warnings
from tqdm import tqdm
from matplotlib.patches import Patch
import json

#warnings.filterwarnings("ignore", category=UserWarning)

# ======= 配置路径 =======
protein_dir = r"Z:\PCA\Original protein data\z_output"
rhp_dir = r"Z:\PCA\Original RHP data\z_output"

use_existing_pca_weights = True
pca_json_path = "pca_weights_2d_2of5_groups.json"

# ======= 统一绘图字体（Nature style）=======
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False

# ======= 收集所有 z 向量 =======
all_z = []
labels = []

def read_z_vectors_from_csv(path, label_name):
    df_header = pd.read_csv(path, nrows=0)
    z_cols = [col for col in df_header.columns if col.startswith("z_")]
    if not z_cols:
        print(f"⚠ {label_name} 文件中未找到 z_ 列")
        return []

    df = pd.read_csv(path, usecols=z_cols, dtype='string', keep_default_na=False, low_memory=False)
    result = []
    for _, row in df.iterrows():
        for col in z_cols:
            cell = row[col]
            if isinstance(cell, str) and cell.strip():
                try:
                    vec = ast.literal_eval(cell)
                    if isinstance(vec, list) and len(vec) == 16 and all(isinstance(x, (int, float)) for x in vec):
                        result.append(vec)
                        labels.append(label_name)
                except:
                    continue
    return result

# --- 蛋白来源（全部为 CSV）---
protein_files = {
    "Membrane": os.path.join(protein_dir, "z_membrane.csv"),
    "Cytoplasm": os.path.join(protein_dir, "z_cytoplasma.csv"),
    "UniProt protein set": os.path.join(protein_dir, "z_reviewed_annotation_4_5.csv"),
    "Disordered region": os.path.join(protein_dir, "z_IDR_sequences_over50.csv"),
    "Fusion peptide": os.path.join(protein_dir, "z_fusion_peptides_padded.csv"),
}

for label, path in protein_files.items():
    print(f"📄 读取蛋白文件：{label}")
    z_vecs = read_z_vectors_from_csv(path, label)
    print(f"✅ 收到 {len(z_vecs)} 个合法 z 向量")
    all_z.extend(z_vecs)

# --- RHP 来源（全部为 CSV）---
for file in os.listdir(rhp_dir):
    if file.startswith("z_RHP(") and file.endswith(".csv"):
        rhp_label = os.path.splitext(file)[0]
        path = os.path.join(rhp_dir, file)
        print(f"📄 读取 RHP 文件：{file}")
        z_vecs = read_z_vectors_from_csv(path, rhp_label)
        print(f"✅ 收到 {len(z_vecs)} 个合法 z 向量")
        all_z.extend(z_vecs)

# ======= 不随机采样 =======
sample_indices = random.sample(range(len(all_z)), k=int(len(all_z) * 1))
sampled_z = [all_z[i] for i in sample_indices]
sampled_labels = [labels[i] for i in sample_indices]

# ======= PCA 降维 =======
print("\U0001f9e0 开始进行 PCA 分析...")
if use_existing_pca_weights and os.path.exists(pca_json_path):
    with open(pca_json_path, "r") as f:
        pca_info = json.load(f)
    mean = np.array(pca_info["mean"])
    components = np.array(pca_info["components"])
    sampled_z_centered = np.array(sampled_z) - mean
    z_all_pcs = sampled_z_centered @ components.T
    z_2d = z_all_pcs[:, :2]
    print(f"♻️ 使用已有 PCA 权重：{pca_json_path}")
else:
    pca = PCA(n_components=5)
    z_all_pcs = pca.fit_transform(sampled_z)
    z_2d = z_all_pcs[:, :2]
    pca_info = {
        "mean": pca.mean_.tolist(),
        "components": pca.components_.tolist()
    }
    with open(pca_json_path, "w") as f:
        json.dump(pca_info, f, indent=2)
    print(f"📀 PCA 权重保存至：{pca_json_path}")

# ======= 转为 DataFrame =======
df_pca = pd.DataFrame(z_2d, columns=["PC1", "PC2"])
df_pca["group"] = sampled_labels

# ======= 自定义配色 =======
protein_colors = {
    "Cytoplasm": "#1F78B4",
    "Membrane": "#E31A1C",
    "UniProt protein set": "#000000",
    "Disordered region": "#0313FF",
    "Fusion peptide": "#FC9F42",
}
rhp_colors = {
    "RHP(65/5)": "#A6CEE3",
    "RHP(50/20)": "#B2DF8A",
    "RHP(40/30)": "#33A02C",
    "RHP(30/40)": "#FC633D",
    "RHP(15/55)": "#9C27B0",
}
group_colors = {**protein_colors, **rhp_colors}

rhp_display_map = {
    "RHP(15/55)": "RHP1",
    "RHP(30/40)": "RHP2",
    "RHP(40/30)": "RHP3",
    "RHP(50/20)": "RHP4",
    "RHP(65/5)":  "RHP5",
}

def display_label(group_name: str) -> str:
    return rhp_display_map.get(group_name, group_name)

# --- 修正 label 格式 ---
def format_rhp_label(raw_label):
    if raw_label.startswith("z_RHP(") and raw_label.endswith(")"):
        content = raw_label[6:-1]
        parts = content.split("_")
        if len(parts) == 2:
            a, b = map(int, parts)
            return f"RHP({a}/{b})"
    return raw_label

df_pca["group"] = df_pca["group"].apply(format_rhp_label)
unique_groups = sorted(df_pca["group"].unique())

# --- 拆分蛋白和 RHP 数据 ---
df_protein = df_pca[~df_pca["group"].str.contains("RHP")]
df_rhp = df_pca[df_pca["group"].str.contains("RHP")]

# --- 拆分 total_protein 出来，做黑色线框 ---
df_total = df_protein[df_protein["group"] == "UniProt protein set"]
df_protein = df_protein[df_protein["group"] != "UniProt protein set"]

# --- 拆分 Fusion peptide 和 Disordered region ---
df_fusion = df_protein[df_protein["group"] == "Fusion peptide"]
df_idr = df_protein[df_protein["group"] == "Disordered region"]
df_protein = df_protein[~df_protein["group"].isin(["Fusion peptide", "Disordered region"])]

# ======= 绘图 =======
print(f"📊 参与绘图的 group：{sorted(df_pca['group'].unique())}")
fig, ax = plt.subplots(figsize=(3, 3))
handles = []

# --- 绘制 UniProt protein set 黑色轮廓 ---
if not df_total.empty:
    sns.kdeplot(
        data=df_total, x="PC1", y="PC2", fill=False,
        thresh=0.04, levels=1, color="black", alpha=0.5,
        linewidths=2, label="UniProt protein set", bw_adjust=1, ax=ax
    )
    handles.append(Patch(facecolor="none", edgecolor="black", label="UniProt protein set", linewidth=1.8))

# --- 绘制 protein 组（不含 Fusion 和 IDR） ---
for group in sorted(df_protein["group"].unique()):
    subset = df_protein[df_protein["group"] == group]
    color = group_colors.get(group, "#999999")
    sns.kdeplot(
        data=subset, x="PC1", y="PC2", fill=True,
        thresh=0.04, levels=4,
        cmap=sns.light_palette(color, as_cmap=True),
        alpha=0.7, bw_adjust=1, ax=ax
    )
    sns.kdeplot(
        data=subset, x="PC1", y="PC2", fill=False,
        thresh=0.04, levels=1, color=color,
        linewidths=2, label=group, alpha=0.7,
        bw_adjust=1, ax=ax
    )
    handles.append(Patch(facecolor=color, edgecolor=color, label=group))

# --- 单独绘制 Fusion peptide ---
#if not df_fusion.empty:
#    color = group_colors.get("Fusion peptide", "#999999")
#    sns.kdeplot(
#        data=df_fusion, x="PC1", y="PC2", fill=True,
#        thresh=0.01, levels=4,
#        cmap=sns.light_palette(color, as_cmap=True),
#        alpha=0.7, bw_adjust=0.9, ax=ax
#    )
#    sns.kdeplot(
#        data=df_fusion, x="PC1", y="PC2", fill=False,
#        thresh=0.01, levels=1, color=color,
#        linewidths=2.5, label="Fusion peptide", alpha=0.7,
#        bw_adjust=0.9, ax=ax
#    )
#    handles.append(Patch(facecolor=color, edgecolor=color, label="Fusion peptide"))

# --- 单独绘制 Disordered region ---
if not df_idr.empty:
    color = group_colors.get("Disordered region", "#999999")
    sns.kdeplot(
        data=df_idr, x="PC1", y="PC2", fill=True,
        thresh=0.04, levels=4,
        cmap=sns.light_palette(color, as_cmap=True),
        alpha=0.4, bw_adjust=1.35, ax=ax
    )
    sns.kdeplot(
        data=df_idr, x="PC1", y="PC2", fill=False,
        thresh=0.04, levels=1, color=color,
        linewidths=2, label="Disordered region", alpha=0.7,
        bw_adjust=1.35, ax=ax
    )
    handles.append(Patch(facecolor=color, edgecolor=color, label="Disordered region"))

# --- 绘制 RHP（实线轮廓，无填充） ---
for group in ["RHP(15/55)", "RHP(30/40)", "RHP(40/30)", "RHP(50/20)", "RHP(65/5)"]:
    subset = df_rhp[df_rhp["group"] == group]
    if subset.empty:
        continue
    color = group_colors.get(group, "#999999")

    sns.kdeplot(
        data=subset, x="PC1", y="PC2", fill=False,
        thresh=0.04, levels=4, color=color,
        alpha=0.7, linewidths=0.8, label=display_label(group),
        bw_adjust=1, ax=ax
    )

    handles.append(
        Patch(facecolor="none", edgecolor=color,
              label=display_label(group), linewidth=1.8)
    )

# --- 图像样式设置 ---
ax.set_xlim(-57, 65)
ax.set_ylim(-42, 42)
ax.set_xlabel("PC1", fontsize=10)
ax.set_ylabel("PC2", fontsize=10)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.grid(False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig("2D_PCA_3protein_5RHP_2of5_groups.png", dpi=300, bbox_inches="tight")
fig.savefig("2D_PCA_3protein_5RHP_2of5_groups.pdf", bbox_inches="tight")
print("✅ 图像已保存：2D_PCA_3protein_5RHP_2of5_groups.png")
plt.close(fig)

# ======= Legend 单独保存 =======
fig_legend, ax_legend = plt.subplots(figsize=(10, 10))
ax_legend.axis("off")
ax_legend.legend(
    handles=handles,
    title="Group",
    loc="center",
    frameon=True,
    framealpha=0.8,
    borderpad=1,
    fontsize=12,
    title_fontsize=13
)
fig_legend.savefig("2D_PCA_3protein_5RHP_legend_2of5_groups.png", dpi=300, bbox_inches="tight")
fig_legend.savefig("2D_PCA_3protein_5RHP_legend_2of5_groups.pdf", bbox_inches="tight")
print("✅ Legend 已保存：2D_PCA_3protein_5RHP_legend_2of5_groups.png")
