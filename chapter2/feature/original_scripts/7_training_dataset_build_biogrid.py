import pandas as pd

# =========================
# Paths
# =========================
input_file = r"BIOGRID_schemeA_reduced_with_letters_dedup_len30.csv"
output_file = r"BIOGRID_7group_style.csv"

# =========================
# 7-group amino acid classification (与 Negatome 完全一致)
# =========================
translation_dict = {
    "A": "1", "G": "1", "V": "1",
    "I": "2", "L": "2", "F": "2", "P": "2",
    "Y": "3", "M": "3", "T": "3", "S": "3",
    "H": "4", "N": "4", "Q": "4", "W": "4",
    "R": "5", "K": "5",
    "D": "6", "E": "6",
    "C": "7",
}

def translate_sequence(seq):
    seq = str(seq).strip().upper()
    return "".join(translation_dict.get(aa, "8") for aa in seq)

# =========================
# Read & Process
# =========================
df = pd.read_csv(input_file, low_memory=False)

out_df = pd.DataFrame()
out_df["proteinA"] = df["Entry A"].astype(str).str.strip()
out_df["proteinB"] = df["Entry B"].astype(str).str.strip()
out_df["Sequence A"] = df["Letters_A"].astype(str).str.strip().str.upper().apply(translate_sequence)
out_df["Sequence B"] = df["Letters_B"].astype(str).str.strip().str.upper().apply(translate_sequence)
out_df["interaction"] = 1   # BioGRID 正样本

out_df = out_df[(out_df["Sequence A"] != "") & (out_df["Sequence B"] != "")].copy()

out_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"Saved to: {output_file}")
print(f"Rows kept: {len(out_df)}")
