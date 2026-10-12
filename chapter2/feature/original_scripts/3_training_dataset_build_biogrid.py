import pandas as pd

# =========================
# Paths
# =========================
input_file = r"BIOGRID_schemeA_reduced_with_letters_dedup_len30.csv"
output_file = r"BIOGRID_Li3_style.csv"   # ← 输出改为 Li3 风格

# =========================
# AA -> Li3-style 1~4 (3+1 letter)
# 来源：Li et al. 2003 Reduced amino acid alphabet
# 1 = Hydrophobic          (C F Y W M L I V)
# 2 = Small/Polar          (G P A T S)
# 3 = Charged/Polar        (N H Q E D R K)
# 4 = 容错占位符（未知字符、X、B、Z 等全部归到 4）
# =========================
translation_dict = {
    # Group 1: Hydrophobic (Li3 I组)
    "C": "1", "F": "1", "Y": "1", "W": "1",
    "M": "1", "L": "1", "I": "1", "V": "1",

    # Group 2: Small / Polar (Li3 S组)
    "G": "2", "P": "2", "A": "2", "T": "2", "S": "2",

    # Group 3: Charged / Polar (Li3 E组)
    "N": "3", "H": "3", "Q": "3", "E": "3", "D": "3", "R": "3", "K": "3",
}

def translate_sequence(seq):
    """
    将氨基酸序列转换为 Li3-style 3+1 字母（数字串）。
    未知字符或不在123组的残基全部映射为容错占位符 "4"。
    """
    seq = str(seq).strip().upper()
    return "".join(translation_dict.get(aa, "4") for aa in seq)

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

# 原始氨基酸序列 -> Li3-style 数字串
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
