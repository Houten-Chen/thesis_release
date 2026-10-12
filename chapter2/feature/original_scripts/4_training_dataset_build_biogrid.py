import pandas as pd

# =========================
# Paths
# =========================
input_file = r"BIOGRID_schemeA_reduced_with_letters_dedup_len30.csv"
output_file = r"BIOGRID_RHP_style.csv"

# =========================
# AA -> RHP-style 1~5
# 与你旧脚本一致
# =========================
translation_dict = {
    "C": "1", "Y": "1", "A": "1", "T": "1", "G": "1",
    "S": "2", "Q": "2", "H": "2", "N": "2", "P": "2",
    "L": "3", "I": "3", "F": "3", "W": "3", "V": "3", "M": "3",
    "E": "4", "D": "4", "R": "4", "K": "4",
    "X": "5",  # unknown
}

def translate_sequence(seq):
    seq = str(seq).strip().upper()
    return "".join(translation_dict.get(aa, "5") for aa in seq)

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

# 原始氨基酸序列 -> RHP-style 数字串
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
