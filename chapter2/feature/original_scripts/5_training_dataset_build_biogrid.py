import pandas as pd

# =========================
# Paths
# =========================
input_file = r"BIOGRID_schemeA_reduced_with_letters_dedup_len30.csv"
output_file = r"BIOGRID_Wang5_style.csv"   # ← 输出改为 Wang5 风格


translation_dict = {
    # Group I (疏水组, 8个) → 1
    "C": "1", "M": "1", "F": "1", "I": "1", "L": "1", "V": "1", "W": "1", "Y": "1",

    # Group II (3个) → 2
    "A": "2", "H": "2", "T": "2",

    # Group III (2个) → 3
    "G": "3", "P": "3",

    # Group IV (2个) → 4
    "D": "4", "E": "4",

    # Group V (极性组, 5个) → 5
    "N": "5", "Q": "5", "R": "5", "K": "5", "S": "5",

    # 未知/容错占位符 → 6
}

def translate_sequence(seq):
    """
    将氨基酸序列转换为 Wang5-style 5+1 字母（数字串）。
    来源：Wang & Wang 1999
    未知字符或不在1-5组的残基全部映射为容错占位符 "6"。
    """
    seq = str(seq).strip().upper()
    return "".join(translation_dict.get(aa, "6") for aa in seq)

# =========================
# Read input
# =========================
df = pd.read_csv(input_file, low_memory=False)

required_cols = ["Entry A", "Entry B", "Letters_A", "Letters_B"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# =========================
# Build output table
# =========================
out_df = pd.DataFrame()

out_df["proteinA"] = df["Entry A"].astype(str).str.strip()
out_df["proteinB"] = df["Entry B"].astype(str).str.strip()

# 原始氨基酸序列 -> Wang5-style 数字串
out_df["Sequence A"] = df["Letters_A"].astype(str).str.strip().str.upper().apply(translate_sequence)
out_df["Sequence B"] = df["Letters_B"].astype(str).str.strip().str.upper().apply(translate_sequence)

# BioGRID 正样本一律设为 1
out_df["interaction"] = 1

# =========================
# Optional cleanup
# 去掉空序列
# =========================
before_rows = len(out_df)
out_df = out_df[
    (out_df["Sequence A"] != "") &
    (out_df["Sequence B"] != "")
].copy()
after_rows = len(out_df)

# =========================
# Save
# =========================
out_df.to_csv(output_file, index=False, encoding="utf-8-sig")

# =========================
# Report
# =========================
print(f"Original rows: {before_rows}")
print(f"Rows kept: {after_rows}")
print(f"Saved to: {output_file}")

print("\nPreview:")
print(out_df.head())

print("\ninteraction counts:")
print(out_df["interaction"].value_counts(dropna=False))
