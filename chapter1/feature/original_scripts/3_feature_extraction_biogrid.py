from itertools import product
import numpy as np
from tqdm import tqdm
import pandas as pd
import h5py
import os

# =========================
# Read csv data
# =========================
# 注意：请使用 Li3 版本的 CSV 文件（Negatome2_Li3_style.csv）
data = pd.read_csv(r'Z:\PRI deeplearning_3_letters\training_dataset\seperate\BIOGRID_Li3_style.csv')
output_file = 'extracted_features_biogrid_Li3.hdf5'

# 如果旧文件已存在，先删掉，避免重复 create_dataset 报错
if os.path.exists(output_file):
    os.remove(output_file)

# =========================
# Feature extraction functions
# =========================
def calculate_amino_acid_proportion(sequence):
    sequence = str(sequence)
    total_length = len(sequence)
    # Li3 版本只有 1~4，因此 proportion 只计算 4 组
    proportion = [sequence.count(str(i)) / total_length for i in range(1, 5)]
    length = len(sequence) / 1000
    return proportion, length


def get_all_n_mers(n):
    # Li3 版本：只有 4 个字母 (1,2,3 + 占位符4)
    amino_acids = ['1', '2', '3', '4']
    all_n_mers = [''.join(p) for p in product(amino_acids, repeat=n)]
    return all_n_mers


def calculate_frequencies(sequence, all_n_mers):
    sequence = str(sequence)
    n_mer_counts = [sequence.count(mer) for mer in all_n_mers]
    total_n_mer_counts = sum(n_mer_counts)
    n_mer_frequencies = [count / total_n_mer_counts for count in n_mer_counts]
    return n_mer_frequencies


def calculate_nmer_position_percentage(sequence, n):
    sequence = str(sequence)
    n_mer_list = get_all_n_mers(n)
    counts_list = []

    def count_n_mers(sequence, mer, n):
        counts = [0] * 20
        start = 0
        while start < len(sequence):
            start_index = sequence.find(mer, start)
            if start_index == -1:
                break

            end_index = start_index + len(mer) - 1
            start_position = start_index / len(sequence)
            end_position = end_index / len(sequence)

            if start_position >= 1:
                start_position = 0.99999
            if end_position >= 1:
                end_position = 0.99999

            start_percentage = int(20 * start_position)
            end_percentage = int(20 * end_position)

            if start_percentage == end_percentage:
                counts[start_percentage] += 1

            for i in range(start_percentage, end_percentage):
                counts[i] += 1

            start = start_index + 1

        if n == 1:
            total = sum(counts)
            if total != 0:
                counts = [10 * (i / total) for i in counts]

        return counts

    for n_mer in n_mer_list:
        con = count_n_mers(sequence, n_mer, n)
        counts_list.append(con)

    return counts_list


def APAAC(sequence, maxlag):
    sequence = str(sequence)

    def calculate_h_values(sequence, H):
        sequence = list(map(int, str(sequence)))
        h_values = []
        hdiff_matrix = []

        # 只计算有效 lag；不足的后面补零
        valid_maxlag = min(maxlag, len(sequence) - 1)

        for d in range(1, valid_maxlag + 1):
            hdiff_in_lengths = [0] * 20
            sum_of_h = 0
            blur = 0
            item = 0

            # Li3 版本：占位符改为 4（原来是 5）
            skip_initial_fours = True

            # 计算 lag=d 时的 H 统计
            for i in range(len(sequence) - d):
                if skip_initial_fours and sequence[i] == 4:
                    continue
                elif skip_initial_fours:
                    skip_initial_fours = False
                    blur = i

                # 占位符 4 视为模糊区域
                if sequence[i] == 4 or sequence[i + d] == 4:
                    blur += 1
                else:
                    # H 是长度为 3 的数组，对应字母 1,2,3
                    h_val = H[sequence[i] - 1] * H[sequence[i + d] - 1]
                    item += 1
                    sum_of_h += h_val

            # 避免 division by zero
            if item == 0:
                ave_h = 0
            else:
                sum_of_h = blur * (sum_of_h / item) + sum_of_h
                ave_h = sum_of_h / (blur + item)

            # 按序列长度位置统计 H 分布
            for k in range(len(sequence) - d):
                length_percentage = k / (len(sequence) - d)
                if length_percentage >= 1:
                    length_percentage = 0.99999

                if sequence[k] == 4 or sequence[k + d] == 4:
                    h_val = ave_h
                else:
                    h_val = H[sequence[k] - 1] * H[sequence[k + d] - 1]

                hdiff_in_lengths[int(20 * length_percentage)] += h_val

            denom = max(len(sequence) - d, 1)
            h_values.append(sum_of_h / denom)
            hdiff_in_lengths = [10 * (block / denom) for block in hdiff_in_lengths]
            hdiff_matrix.append(hdiff_in_lengths)

        # 不足 maxlag 的 lag 用 0 补齐，保证输出维度固定
        while len(h_values) < maxlag:
            h_values.append(0)
            hdiff_matrix.append([0] * 20)

        return h_values, hdiff_matrix

    # Calculate H1 / H2
    H1_value, H1_matrix = calculate_h_values(sequence, H1)
    H2_value, H2_matrix = calculate_h_values(sequence, H2)
    H1_features = [10 * i for i in H1_value]
    H2_features = [10 * i for i in H2_value]

    # Calculate APAAC descriptor
    w = 0.1
    denominator = sum(H1_value + H2_value) * w + 1

    apaac1 = [100 * w * h_value / denominator for h_value in H1_value]
    apaac2 = [100 * w * h_value / denominator for h_value in H2_value]

    return H1_features, H1_matrix, H2_features, H2_matrix, apaac1, apaac2


# =========================
# Hydrophobicity lists (Li3 版本)
# =========================
# H1 / H2 仅对应真实字母 1,2,3（长度为 3）
# 占位符 4 的处理逻辑与原来 5 完全一致（在 calculate_h_values 中特殊处理）
# 请根据实际计算结果填写下面两个数组（推荐使用 Kyte-Doolittle + Fauchère-Pliska / XlogP3 side-chain）
H1 = np.array([2.1875, -0.34, -3.66])      # Group1, Group2, Group3 的平均值
H2 = np.array([-1.56, -0.25, 0.59])        # 侧链贡献平均值（可替换为你的精确计算结果）

# =========================
# Extract file data
# =========================
proteinA = data['proteinA'].astype(str).values
proteinB = data['proteinB'].astype(str).values
sequenceA = data['Sequence A']
sequenceB = data['Sequence B']
interaction = data['interaction'].astype(int).values

# =========================
# Prepare feature lists
# =========================
# sequenceA
proportionA_list = []
lengthA_list = []
one_mer_counts_listA_list = []
two_mer_counts_listA_list = []
three_mer_counts_listA_list = []
H1_featuresA_list = []
H2_featuresA_list = []
H1_matrixA_list = []
H2_matrixA_list = []
apaac1A_list = []
apaac2A_list = []

# sequenceB
proportionB_list = []
lengthB_list = []
one_mer_counts_listB_list = []
two_mer_counts_listB_list = []
three_mer_counts_listB_list = []
H1_featuresB_list = []
H2_featuresB_list = []
H1_matrixB_list = []
H2_matrixB_list = []
apaac1B_list = []
apaac2B_list = []

# =========================
# Extract basic features for A
# =========================
print('Extracting proportion/length/n-mer-position features, from protein A sequences...')
for lineA in tqdm(sequenceA):
    proportionA, lengthA = calculate_amino_acid_proportion(lineA)
    one_mer_countsA = calculate_nmer_position_percentage(lineA, 1)
    two_mer_countsA = calculate_nmer_position_percentage(lineA, 2)
    three_mer_countsA = calculate_nmer_position_percentage(lineA, 3)

    proportionA_list.append(proportionA)
    lengthA_list.append(lengthA)
    one_mer_counts_listA_list.append(one_mer_countsA)
    two_mer_counts_listA_list.append(two_mer_countsA)
    three_mer_counts_listA_list.append(three_mer_countsA)

# =========================
# Extract APAAC features for A
# =========================
print('Extracting APAAC features, from protein A sequences...')
for lineA in tqdm(sequenceA):
    H1_featuresA, H1_matrixA, H2_featuresA, H2_matrixA, apaac1A, apaac2A = APAAC(lineA, 30)

    H1_featuresA_list.append(H1_featuresA)
    H2_featuresA_list.append(H2_featuresA)
    H1_matrixA_list.append(H1_matrixA)
    H2_matrixA_list.append(H2_matrixA)
    apaac1A_list.append(apaac1A)
    apaac2A_list.append(apaac2A)

# =========================
# Extract basic features for B
# =========================
print('Extracting proportion/length/n-mer-position features, from protein B sequences...')
for lineB in tqdm(sequenceB):
    proportionB, lengthB = calculate_amino_acid_proportion(lineB)
    one_mer_countsB = calculate_nmer_position_percentage(lineB, 1)
    two_mer_countsB = calculate_nmer_position_percentage(lineB, 2)
    three_mer_countsB = calculate_nmer_position_percentage(lineB, 3)

    proportionB_list.append(proportionB)
    lengthB_list.append(lengthB)
    one_mer_counts_listB_list.append(one_mer_countsB)
    two_mer_counts_listB_list.append(two_mer_countsB)
    three_mer_counts_listB_list.append(three_mer_countsB)

# =========================
# Extract APAAC features for B
# =========================
print('Extracting APAAC features, from protein B sequences...')
for lineB in tqdm(sequenceB):
    H1_featuresB, H1_matrixB, H2_featuresB, H2_matrixB, apaac1B, apaac2B = APAAC(lineB, 30)

    H1_featuresB_list.append(H1_featuresB)
    H2_featuresB_list.append(H2_featuresB)
    H1_matrixB_list.append(H1_matrixB)
    H2_matrixB_list.append(H2_matrixB)
    apaac1B_list.append(apaac1B)
    apaac2B_list.append(apaac2B)

# =========================
# Convert to arrays
# =========================
proportion_A = np.array(proportionA_list)
length_A = np.array(lengthA_list)
one_mer_countsA = np.array(one_mer_counts_listA_list)
two_mer_countsA = np.array(two_mer_counts_listA_list)
three_mer_countsA = np.array(three_mer_counts_listA_list)
H1_featuresA = np.array(H1_featuresA_list)
H2_featuresA = np.array(H2_featuresA_list)
H1_matrixA = np.array(H1_matrixA_list)
H2_matrixA = np.array(H2_matrixA_list)
apaac1A = np.array(apaac1A_list)
apaac2A = np.array(apaac2A_list)

proportion_B = np.array(proportionB_list)
length_B = np.array(lengthB_list)
one_mer_countsB = np.array(one_mer_counts_listB_list)
two_mer_countsB = np.array(two_mer_counts_listB_list)
three_mer_countsB = np.array(three_mer_counts_listB_list)
H1_featuresB = np.array(H1_featuresB_list)
H2_featuresB = np.array(H2_featuresB_list)
H1_matrixB = np.array(H1_matrixB_list)
H2_matrixB = np.array(H2_matrixB_list)
apaac1B = np.array(apaac1B_list)
apaac2B = np.array(apaac2B_list)

# =========================
# Print shapes
# =========================
print('\n=== Shapes for A (Li3 3+1 version) ===')
print('proportion_A:', np.shape(proportion_A))
print('length_A:', np.shape(length_A))
print('one_mer_countsA:', np.shape(one_mer_countsA))
print('two_mer_countsA:', np.shape(two_mer_countsA))
print('three_mer_countsA:', np.shape(three_mer_countsA))
print('H1_featuresA:', np.shape(H1_featuresA))
print('H2_featuresA:', np.shape(H2_featuresA))
print('apaac1A:', np.shape(apaac1A))
print('apaac2A:', np.shape(apaac2A))
print('H1_matrixA:', np.shape(H1_matrixA))
print('H2_matrixA:', np.shape(H2_matrixA))

print('\n=== Shapes for B (Li3 3+1 version) ===')
print('proportion_B:', np.shape(proportion_B))
print('length_B:', np.shape(length_B))
print('one_mer_countsB:', np.shape(one_mer_countsB))
print('two_mer_countsB:', np.shape(two_mer_countsB))
print('three_mer_countsB:', np.shape(three_mer_countsB))
print('H1_featuresB:', np.shape(H1_featuresB))
print('H2_featuresB:', np.shape(H2_featuresB))
print('apaac1B:', np.shape(apaac1B))
print('apaac2B:', np.shape(apaac2B))
print('H1_matrixB:', np.shape(H1_matrixB))
print('H2_matrixB:', np.shape(H2_matrixB))

# =========================
# Save all datasets into HDF5
# =========================
with h5py.File(output_file, 'w') as f:
    # original identifiers / labels
    f.create_dataset('proteinA', data=proteinA.astype('S'))
    f.create_dataset('proteinB', data=proteinB.astype('S'))
    f.create_dataset('interaction', data=interaction)

    # A side
    f.create_dataset('proportion_A', data=proportion_A)
    f.create_dataset('length_A', data=length_A)
    f.create_dataset('one_mer_countsA', data=one_mer_countsA)
    f.create_dataset('two_mer_countsA', data=two_mer_countsA)
    f.create_dataset('three_mer_countsA', data=three_mer_countsA)
    f.create_dataset('H1_featuresA', data=H1_featuresA)
    f.create_dataset('H2_featuresA', data=H2_featuresA)
    f.create_dataset('apaac1A', data=apaac1A)
    f.create_dataset('apaac2A', data=apaac2A)
    f.create_dataset('H1_matrixA', data=H1_matrixA)
    f.create_dataset('H2_matrixA', data=H2_matrixA)

    # B side
    f.create_dataset('proportion_B', data=proportion_B)
    f.create_dataset('length_B', data=length_B)
    f.create_dataset('one_mer_countsB', data=one_mer_countsB)
    f.create_dataset('two_mer_countsB', data=two_mer_countsB)
    f.create_dataset('three_mer_countsB', data=three_mer_countsB)
    f.create_dataset('H1_featuresB', data=H1_featuresB)
    f.create_dataset('H2_featuresB', data=H2_featuresB)
    f.create_dataset('apaac1B', data=apaac1B)
    f.create_dataset('apaac2B', data=apaac2B)
    f.create_dataset('H1_matrixB', data=H1_matrixB)
    f.create_dataset('H2_matrixB', data=H2_matrixB)

print(f'\nExtraction complete. Saved to: {output_file}')
print("Li3 (3+1) 版本特征提取完成！H1/H2 长度为 3，n-mer 已适配 4 字母，占位符 4 处理逻辑与原来 5 完全一致。")
input("Press any key to exit...")
