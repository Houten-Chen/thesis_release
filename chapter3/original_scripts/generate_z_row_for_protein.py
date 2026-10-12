import sys
import os
import torch
import pandas as pd
import warnings
from tqdm import tqdm

# 消除 openpyxl 样式警告
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# 添加 AE-RHP 代码路径
sys.path.append("Z:/PCA/AE-RHP-main")

from models import get_model
from utils.util_functions import seq_one_hot_AE

# ====== 路径设置 ======
model_path = r'Z:\PCA\AE-RHP-main\trained_models\N=4_MP_de2concat\checkpoints\checkpoint_041.pkl'
data_dir = r'Z:\PCA\Original protein data'
output_dir = os.path.join(r'Z:\PCA\Original protein data', 'z_output')
os.makedirs(output_dir, exist_ok=True)

# ====== 设备选择 ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ 当前推理设备：{device}")

# ====== 加载模型 ======
checkpoint = torch.load(model_path, map_location=device)
if "optimizer" in checkpoint:
    del checkpoint["optimizer"]

model = get_model(
    name="AE_MP",
    input_dim=[5, 51],
    latent_dim=16,
    win_length=5,
    enc_hidden_sizes=[512, 256, 64],
    dec_hidden_sizes=[64, 256, 512],
    compute_HLB=True,
    MLP_HLB_size=[32, 32]
)
model.load_state_dict(checkpoint)
model.eval()
model.to(device)  # 将模型转移到GPU

# ====== 氨基酸 → RHP monomer 编码 ======
def map_sequence_to_rhp(seq):
    return ''.join([
        '4' if aa in 'EDRK' else
        '3' if aa in 'LIFWVM' else
        '2' if aa in 'SQHNP' else
        '1'
        for aa in seq
    ])

def process_file(filename):
    filepath = os.path.join(data_dir, filename)
    df = pd.read_csv(filepath)
    sequences = df["Sequence"].fillna("")

    z_data = []

    for seq in tqdm(sequences, desc=f"Processing {filename}"):
        mapped = map_sequence_to_rhp(seq)
        window_size = 50
        step = 15
        positions = list(range(0, len(mapped) - window_size + 1, step))

        # 生成所有窗口子串
        substrings = [mapped[i:i + window_size] for i in positions]
        if not substrings:
            z_data.append({})  # 保持行数一致
            continue

        # 批量 one-hot 编码
        onehots = [seq_one_hot_AE(sub, monomer_codes='1234', max_length=50).flatten() for sub in substrings]
        onehots = torch.stack(onehots).to(device)  # [batch, 5*51]

        # 批量推理
        with torch.no_grad():
            z_batch = model.encoder(onehots).detach().cpu().numpy()

        # 构造 z_0, z_1, ...
        z_row = {f"z_{idx}": z.tolist() for idx, z in enumerate(z_batch)}
        z_data.append(z_row)

    # 输出保存
    z_df = pd.DataFrame(z_data)
    df_out = pd.concat([df.reset_index(drop=True), z_df], axis=1)

    base_name = os.path.splitext(filename)[0]
    save_path = os.path.join(output_dir, f'z_{base_name}.csv')
    df_out.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"✅ 已保存：{save_path}")

# ====== 批量执行 ======
file_list = [
    "reviewed_annotation_4_5.csv",
]

for file in file_list:
    process_file(file)

