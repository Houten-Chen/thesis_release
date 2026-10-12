import os
import sys
import torch
import pandas as pd
import warnings
from tqdm import tqdm

# 忽略 openpyxl 样式警告
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# 添加 AE-RHP 代码路径
sys.path.append("Z:/PCA/AE-RHP-main")
from models import get_model
from utils.util_functions import seq_one_hot_AE

# ======= 自动检测是否有GPU =======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ 当前推理设备：{device}")

# ======= 路径配置 =======
#model_path = r'Z:\PCA\AE-RHP-main\trained_models\N=4_MP_largescale_small_step3\checkpoints\checkpoint_036.pkl'
model_path = r'Z:\PCA\AE-RHP-main\trained_models\N=4_MP_de2concat\checkpoints\checkpoint_041.pkl'
input_dir = r'F:\PCA\Generated artificial RHP data\wave_generation'
output_dir = os.path.join(input_dir, 'z_output')
os.makedirs(output_dir, exist_ok=True)

# ======= 加载模型 =======
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
model.to(device)

# ======= 主函数 =======
def process_rhp_file(filename, input_dir, output_dir):
    input_path = os.path.join(input_dir, filename)
    base_name = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, f'z_{base_name}.csv')

    print(f"🔄 正在处理: {filename}")

    df = pd.read_csv(input_path)
    sequences = df["Sequence"].fillna("").astype(str)

    all_records = []
    window_size = 50
    step = 15
    max_batch_windows = 1000

    for idx, seq in enumerate(tqdm(sequences, desc=f"Processing {filename}")):
        positions = list(range(0, len(seq) - window_size + 1, step))
        if not positions:
            continue  # 序列太短，跳过

        onehots = []
        for i in positions:
            sub = seq[i:i + window_size]
            onehot = seq_one_hot_AE(sub, monomer_codes='1234', max_length=50).flatten()
            onehots.append(onehot)

        z_vectors = []
        for start in range(0, len(onehots), max_batch_windows):
            end = start + max_batch_windows
            batch_tensor = torch.stack(onehots[start:end]).to(device)
            with torch.no_grad():
                z_batch = model.encoder(batch_tensor).detach().cpu().numpy()
            z_vectors.extend(z_batch)

        z_row = {f"z_{i}": z.tolist() for i, z in enumerate(z_vectors)}
        record = df.iloc[idx].to_dict()
        record.update(z_row)
        all_records.append(record)

    df_out = pd.DataFrame(all_records)
    df_out.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ 已保存：{output_path}")

# ======= 批量执行所有 RHP(xx_xx).csv 文件 =======
for file in os.listdir(input_dir):
    if file.startswith("RHP(") and file.endswith(".csv"):
        process_rhp_file(file, input_dir, output_dir)
