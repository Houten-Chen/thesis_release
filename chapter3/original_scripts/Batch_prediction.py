import os
import sys
import torch
import esm
import pandas as pd
from tqdm import tqdm

# ==== 设定要处理的目标文件 ====
target_files = [
    "secreted.xlsx"
]

input_dir = r"Original protein data"
output_dir = os.path.join(input_dir, "output")
os.makedirs(output_dir, exist_ok=True)

# ==== 模型导入路径设置 ====
sys.path.append(r"Porter6-main\porter6\porter6\predictor3\model")
from cbrcnn_stage1 import Net

# ==== 模型加载（全局一次） ====
input_size = 1280
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"🖥️ 当前推理设备：CUDA ({torch.cuda.get_device_name(device)})")
else:
    device = torch.device("cpu")
    print("🖥️ 当前推理设备：CPU")

# ---- 加载 ESM 模型 ----
esm_model_path = r"esm-main\trained_model\esm2_t33_650M_UR50D.pt"
esm_model, alphabet = esm.pretrained.load_model_and_alphabet_local(esm_model_path)
esm_model = esm_model.to(device)
batch_converter = alphabet.get_batch_converter()
esm_model.eval()

# ---- 加载 Porter6 模型 ----
porter_model_path = r"Porter6-main\porter6\porter6\predictor3\output\trained_model\esm2\cbrcnn_stage1_f3.pth"
porter_model = Net(input_size=input_size).to(device)
checkpoint = torch.load(porter_model_path, map_location=device)
if "model_state_dict" in checkpoint:
    porter_model.load_state_dict(checkpoint["model_state_dict"])
else:
    porter_model.load_state_dict(checkpoint)
porter_model.eval()

# ==== 定义预测函数 ====
def predict_secondary_structure(sequence: str) -> str:
    MAX_LEN = 9500
    if len(sequence) > MAX_LEN:
        print(f"\n⚠️ 序列长度 {len(sequence)} 超出 {MAX_LEN}，将分段预测")
        parts = [sequence[i:i+MAX_LEN] for i in range(0, len(sequence), MAX_LEN)]
        q3_parts = [predict_secondary_structure(part) for part in parts]
        return "".join(q3_parts)

    # ===== 正常预测流程 =====
    data = [("input_seq", sequence)]
    _, _, batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = esm_model(batch_tokens.to(device), repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]
    embedding = token_representations[0, 1:-1].to("cpu", non_blocking=True).numpy()

    embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = porter_model(embedding_tensor.transpose(1, 2))
        pred = torch.argmax(output, dim=2).squeeze().cpu().tolist()
    torch.cuda.empty_cache()
    idx2ss = {0: "H", 1: "E", 2: "C"}
    return "".join([idx2ss[i] for i in pred])

# ==== 主处理循环 ====
for filename in target_files:
    file_path = os.path.join(input_dir, filename)
    if not os.path.isfile(file_path):
        print(f"❌ 文件未找到：{filename}")
        continue

    print(f"📄 正在处理：{filename}")
    df = pd.read_excel(file_path)

    if "Sequence" not in df.columns:
        print(f"⚠️  文件 {filename} 缺少 Sequence 列，跳过")
        continue

    q3_list = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"🔬 {filename}", ncols=80):
        entry = row.get("Entry", f"row{idx}")  # 如果没有 Entry 列则用行号代替
        seq = row["Sequence"]
        try:
            if not isinstance(seq, str):
                raise ValueError("序列不是字符串")
            q3 = predict_secondary_structure(seq)
        except Exception as e:
            print(f"❌ 出错序列跳过 Entry={entry}（序列前10: {str(seq)[:10]}...）原因：{e}")
            q3 = "ERROR"
        q3_list.append(q3)


    df["Q3"] = q3_list
    out_path = os.path.join(output_dir, filename)
    df.to_excel(out_path, index=False)
    print(f"✅ 已保存至：{out_path}")

print("🎉 所有指定文件已处理完毕。")
