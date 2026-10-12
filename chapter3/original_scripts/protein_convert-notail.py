import pandas as pd

# === 参数设置 ===
input_files = ["average_protein_dedup.csv"]
output_file = "Uniprot_100.csv"

# 单体编码规则：氨基酸 → 数字（1-4）
aa_to_monomer = {
    'E': '4', 'D': '4', 'R': '4', 'K': '4',        # SPMA
    'L': '3', 'I': '3', 'F': '3', 'W': '3', 'V': '3', 'M': '3',  # EHMA
    'S': '2', 'Q': '2', 'H': '2', 'N': '2', 'P': '2',           # OEGMA
    # 其他氨基酸默认为 MMA（编号 '1'）
}

# === 函数定义 ===

# 将蛋白质序列编码为 RHP 数字序列
def encode_sequence_to_rhp(seq):
    return ''.join([aa_to_monomer.get(aa.upper(), '1') for aa in seq])

# 滑窗函数：从左到右滑动，提取所有完整的 100-mer，步长为15
def sliding_window_sequence(seq, window_size=100, step=15):
    return [seq[i:i + window_size] for i in range(0, len(seq) - window_size + 1, step)]

# === 主程序 ===

def main():
    print("📥 读取输入文件...")
    dfs = [pd.read_csv(file) for file in input_files]
    merged_df = pd.concat(dfs, ignore_index=True)

    if "Sequence" not in merged_df.columns:
        raise ValueError("❌ 输入文件缺少 'Sequence' 列，请检查 Excel 文件格式。")

    sequences = merged_df["Sequence"].dropna()

    print(f"🧬 总共读取 {len(sequences)} 条蛋白质序列")

    final_sequences = []

    for seq in sequences:
        encoded = encode_sequence_to_rhp(seq)
        fragments = sliding_window_sequence(encoded)
        final_sequences.extend(fragments)

    print(f"✅ 最终提取出 {len(final_sequences)} 条 50-mer 片段")

    df_out = pd.DataFrame({"sequence": final_sequences})
    df_out.to_csv(output_file, index=False)
    print(f"💾 已保存到文件: {output_file}")

if __name__ == "__main__":
    main()
