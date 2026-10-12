import random
import csv
from collections import Counter
from typing import Dict, List, Tuple, Optional

# =========================
# 全局配置（可调整）
# =========================
RNG_SEED = 777
N_SEQUENCES = 100000
TOTAL_LENGTH = 150
FLUCT_GLOBAL = 0.30

# block 目标长度 & 浮动
HYDRO_BLOCK_EXPECTED = 9
HYDRO_BLOCK_FLUCT = 0.1
HYDROPHILIC_BLOCK_EXPECTED = 9
HYDROPHILIC_BLOCK_FLUCT = 0.1

CSV_OUTPUT = "RHP_sequences_mixed.csv"

# 单体字典与默认比例
MONOMERS = {
    1: "MMA1",   # 中性
    2: "OEGMA2", # 亲水
    3: "EHMA3",  # 疏水
    4: "SPMA4",  # 亲水
    # 5: 疏水段（封尾后）的占位符
    # 6: 亲水段（封尾后）的占位符
}
DEFAULT_RATIOS = {1: 0.50, 2: 0.25, 3: 0.20, 4: 0.05}

# =========================
# 工具函数
# =========================
def choose_from_counts(counts: Dict[int, int], keys: List[int]) -> Optional[int]:
    total = sum(counts.get(k, 0) for k in keys)
    if total <= 0:
        return None
    r = random.randint(1, total)
    cum = 0
    for k in keys:
        c = counts.get(k, 0)
        if c <= 0:
            continue
        cum += c
        if r <= cum:
            return k
    return None

def count_monomers(seq: List[int]) -> Dict[int, int]:
    return dict(Counter(seq))

def build_global_pool(total_length: int,
                      ratios: Dict[int, float],
                      fluct: float) -> Dict[int, int]:
    """决定单条链的 1/2/3/4 数量（四舍五入，不二次矫正）。"""
    counts = {}
    for mono, ratio in ratios.items():
        expected = total_length * ratio
        factor = 1.0 + random.uniform(-fluct, fluct)
        counts[mono] = int(round(expected * factor))
    return counts

# =========================
# 疏水 block：构建 + 立刻封尾（成功则消费；失败则不改 counts）
# =========================
def make_one_hydrophobic_block(consumable: Dict[int,int],
                               expected_len: int,
                               fluct: float) -> Tuple[bool, Optional[List[int]], Dict[int,int], str]:
    """
    构建规则：
      - 仅 {1,3}；按比例抽取；若出现连续两个1，下一位必须为3（无3则失败）
    封尾规则（同你之前设计）：
      - 用 {2,4} 按比例封尾；若 2/4 都无，则尝试 1,1,1；若 1<3 则失败
    返回：
      (ok, sealed_block_or_none, new_counts_if_ok_else_original, reason)
    """
    # 先复制一份用于试拼（失败可回退）
    local = consumable.copy()

    # 目标长度
    target_len = int(round(expected_len * (1.0 + random.uniform(-fluct, fluct))))
    target_len = max(2, target_len)

    used: List[int] = []
    consecutive_ones = 0

    # —— 构建主体 {1,3}
    while len(used) < target_len:
        if consecutive_ones >= 2:
            # 强制 3
            if local.get(3, 0) <= 0:
                return False, None, consumable, "疏水block失败：需要强制3但已无3"
            pick = 3
        else:
            pick = choose_from_counts(local, [1,3])
            if pick is None:
                return False, None, consumable, "疏水block失败：1/3均不足"
        # 扣减
        local[pick] -= 1
        used.append(pick)
        consecutive_ones = consecutive_ones + 1 if pick == 1 else 0

    # —— 封尾：优先 {2,4}；兜底 1,1,1
    # 先按 2/4 比例抽
    tail_pick = choose_from_counts(local, [2,4])
    if tail_pick is not None:
        local[tail_pick] -= 1
        sealed = used + [tail_pick]
        return True, sealed, local, ""
    # 否则尝试 1*3
    if local.get(1,0) >= 3:
        local[1] -= 3
        sealed = used + [1,1,1]
        return True, sealed, local, ""
    return False, None, consumable, "疏水block封尾失败：2/4不足且1不足3个"

# =========================
# 亲水 block：构建 + 立刻封尾（成功则消费；失败则不改 counts）
# =========================
def make_one_hydrophilic_block(consumable: Dict[int,int],
                               expected_len: int,
                               fluct: float) -> Tuple[bool, Optional[List[int]], Dict[int,int], str]:
    """
    改进版（2025-11-03）：
      - 主体部分完全不允许出现 1（MMA）
      - 仅使用 {2,4} 生成亲水段
      - 封尾逻辑保持原样（优先 3，若无则用 1,1,1）
    """
    local = consumable.copy()

    # 目标长度
    target_len = int(round(expected_len * (1.0 + random.uniform(-fluct, fluct))))
    target_len = max(2, target_len)

    used: List[int] = []

    # —— 主体构建：仅从 {2,4} 中抽取
    while len(used) < target_len:
        pick = choose_from_counts(local, [2,4])
        if pick is None:
            return False, None, consumable, "亲水block失败：2/4均不足（不允许1）"
        local[pick] -= 1
        used.append(pick)

    # —— 封尾：优先 3；兜底 1*3
    if local.get(3,0) > 0:
        local[3] -= 1
        sealed = used + [3]
        return True, sealed, local, ""
    if local.get(1,0) >= 3:
        local[1] -= 3
        sealed = used + [1,1,1]
        return True, sealed, local, ""
    return False, None, consumable, "亲水block封尾失败：3不足且1不足3个"

# =========================
# token 展开
# =========================
def expand_tokens(token_seq: List[int],
                  hydro_blocks: List[List[int]],
                  hydrophilic_blocks: List[List[int]]) -> List[int]:
    seq = []
    h = hydro_blocks.copy()
    p = hydrophilic_blocks.copy()
    random.shuffle(h)
    random.shuffle(p)
    ih = 0
    ip = 0
    for t in token_seq:
        if t == 5:
            if ih >= len(h): raise RuntimeError("5 的数量超过疏水block数量")
            seq.extend(h[ih]); ih += 1
        elif t == 6:
            if ip >= len(p): raise RuntimeError("6 的数量超过亲水block数量")
            seq.extend(p[ip]); ip += 1
        else:
            seq.append(t)
    return seq

# =========================
# 生成单条链（交替 block）
# =========================
def generate_one_chain_mixed(total_length: int,
                             ratios: Dict[int,float],
                             fluct_global: float,
                             hydro_expected: int,
                             hydro_fluct: float,
                             hydrophilic_expected: int,
                             hydrophilic_fluct: float) -> Tuple[Optional[List[int]], Dict[str,str], Dict[int,int], int, int]:
    """
    返回：
      - final_seq 或 None
      - debug（含失败原因）
      - decided_counts（最初决定的 1/2/3/4）
      - n_hydro_blocks（本链疏水段数）
      - n_hydrophilic_blocks（本链亲水段数）
    """
    # 1) 决定目标计数
    decided = build_global_pool(total_length, ratios, fluct_global)
    available = decided.copy()

    hydro_blocks: List[List[int]] = []
    hydrophilic_blocks: List[List[int]] = []
    n_hydro = 0
    n_hydrophil = 0

    # 2) 交替生成（疏水→亲水），同一轮两个失败则退出
    while True:
        round_fail = 0

        # 疏水一次
        ok, block, new_counts, reason = make_one_hydrophobic_block(available, hydro_expected, hydro_fluct)
        if ok and block is not None:
            available = new_counts
            hydro_blocks.append(block)
            n_hydro += 1
        else:
            round_fail += 1

        # 亲水一次
        ok, block, new_counts, reason2 = make_one_hydrophilic_block(available, hydrophilic_expected, hydrophilic_fluct)
        if ok and block is not None:
            available = new_counts
            hydrophilic_blocks.append(block)
            n_hydrophil += 1
        else:
            round_fail += 1

        if round_fail == 2:
            break  # 同一轮两个都失败 → 停止 block 阶段

    # 3) 组装 token 池：1/2/3/4 + 5（疏水） + 6（亲水）
    token_counts = {
        1: max(0, available.get(1,0)),
        2: max(0, available.get(2,0)),
        3: max(0, available.get(3,0)),
        4: max(0, available.get(4,0)),
        5: len(hydro_blocks),
        6: len(hydrophilic_blocks),
    }

    # 4) 无放回按比例抽 token
    total_tokens = sum(token_counts.values())
    token_seq: List[int] = []
    for _ in range(total_tokens):
        pick = choose_from_counts(token_counts, [1,2,3,4,5,6])
        if pick is None:
            return None, {"fail":"token 池耗尽异常"}, decided, n_hydro, n_hydrophil
        token_counts[pick] -= 1
        token_seq.append(pick)

    # 5) 展开 5/6
    final_seq = expand_tokens(token_seq, hydro_blocks, hydrophilic_blocks)

    # 6) 强一致性校验
    if len(final_seq) != sum(decided.values()):
        return None, {"fail":f"长度不匹配：final={len(final_seq)} vs decided={sum(decided.values())}"}, decided, n_hydro, n_hydrophil
    final_counts = count_monomers(final_seq)
    for k in [1,2,3,4]:
        if final_counts.get(k,0) != decided.get(k,0):
            return None, {"fail":f"计数不匹配：final={final_counts} vs decided={decided}"}, decided, n_hydro, n_hydrophil

    return final_seq, {}, decided, n_hydro, n_hydrophil

# =========================
# 批量生成 + 导出 + 统计
# =========================
def generate_sequences_mixed(n_sequences: int,
                             total_length: int,
                             ratios: Dict[int,float],
                             fluct_global: float,
                             hydro_expected: int,
                             hydro_fluct: float,
                             hydrophilic_expected: int,
                             hydrophilic_fluct: float,
                             csv_output: str) -> None:
    random.seed(RNG_SEED)

    successes: List[List[int]] = []
    failures: List[Dict[str,str]] = []
    hydro_counts: List[int] = []
    hydrophil_counts: List[int] = []

    for i in range(n_sequences):
        random.seed(RNG_SEED + i)
        seq, dbg, decided, n_h, n_p = generate_one_chain_mixed(
            total_length, ratios, fluct_global,
            hydro_expected, hydro_fluct,
            hydrophilic_expected, hydrophilic_fluct
        )
        # 打印本链的 block 数量
        print(f"[Chain {i+1}] Hydrophobic blocks={n_h}, Hydrophilic blocks={n_p}")
        hydro_counts.append(n_h)
        hydrophil_counts.append(n_p)

        if seq is None:
            failures.append({"idx": str(i+1), **dbg})
        else:
            successes.append(seq)

    # CSV 输出（仅 Sequence 列）
    with open(csv_output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Sequence"])
        for seq in successes:
            w.writerow(["".join(map(str, seq))])

    # 汇总统计
    print("======================================")
    print("双 block 交替版本 —— 生成完成")
    print(f"目标链条数：{n_sequences}")
    print(f"成功：{len(successes)} 条")
    print(f"失败：{len(failures)} 条")
    if failures:
        reason_counter = Counter()
        for fr in failures:
            reason_counter[fr.get("fail","未知失败")] += 1
        print("—— 失败原因汇总 ——")
        for r, c in reason_counter.items():
            print(f"{r}  -> {c} 次")

    # 平均 block 数
    if hydro_counts:
        avg_h = sum(hydro_counts)/len(hydro_counts)
        avg_p = sum(hydrophil_counts)/len(hydrophil_counts)
        print(f"平均疏水 block 数：{avg_h:.2f}")
        print(f"平均亲水 block 数：{avg_p:.2f}")

    # 成功序列整体单体比例（再次确认）
    total_counts_success: Dict[int,int] = Counter()
    for seq in successes:
        total_counts_success.update(seq)
    total_len = sum(total_counts_success.values())
    if total_len > 0:
        print("—— 成功序列的总体单体比例（汇总） ——")
        for k in [1,2,3,4]:
            frac = total_counts_success.get(k,0)/total_len
            print(f"{k}({MONOMERS[k]}): {total_counts_success.get(k,0)} | 比例: {frac:.4f}")
    else:
        print("无成功序列，无法计算比例。")

    print(f"\nCSV 已输出到：{csv_output}")
    print("======================================")

if __name__ == "__main__":
    generate_sequences_mixed(
        n_sequences=N_SEQUENCES,
        total_length=TOTAL_LENGTH,
        ratios=DEFAULT_RATIOS,
        fluct_global=FLUCT_GLOBAL,
        hydro_expected=HYDRO_BLOCK_EXPECTED,
        hydro_fluct=HYDRO_BLOCK_FLUCT,
        hydrophilic_expected=HYDROPHILIC_BLOCK_EXPECTED,
        hydrophilic_fluct=HYDROPHILIC_BLOCK_FLUCT,
        csv_output=CSV_OUTPUT
    )
    input("\n按 Enter 键结束程序...")
