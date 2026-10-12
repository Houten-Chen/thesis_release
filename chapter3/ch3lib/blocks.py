"""Pseudo-block control; the uploaded stochastic rules and per-chain seeds are retained."""
from collections import Counter
import random


def choose(counts, keys, rng):
    total = sum(counts.get(k, 0) for k in keys)
    if total <= 0:
        return None
    target = rng.randint(1, total)
    cumulative = 0
    for key in keys:
        count = counts.get(key, 0)
        if count > 0:
            cumulative += count
            if target <= cumulative:
                return key
    return None


def make_block(available, expected, fluctuation, hydrophobic, rng):
    local = available.copy()
    length = max(2, int(round(expected * (1 + rng.uniform(-fluctuation, fluctuation)))))
    block = []
    ones = 0
    while len(block) < length:
        if hydrophobic and ones >= 2:
            if local.get(3, 0) <= 0:
                return None, available
            pick = 3
        else:
            pick = choose(local, [1, 3] if hydrophobic else [2, 4], rng)
            if pick is None:
                return None, available
        local[pick] -= 1
        block.append(pick)
        ones = ones + 1 if pick == 1 else 0
    tail = choose(local, [2, 4], rng) if hydrophobic else (3 if local.get(3, 0) > 0 else None)
    if tail is not None:
        local[tail] -= 1
        return block + [tail], local
    if local.get(1, 0) >= 3:
        local[1] -= 3
        return block + [1, 1, 1], local
    return None, available


def generate_chain(nominal_length=150, ratios=(.50, .25, .20, .05), global_fluct=.30,
                   hydro_length=9, hydro_fluct=.1, polar_length=9, polar_fluct=.1, seed=777):
    rng = random.Random(seed)
    decided = {code: int(round(nominal_length * ratio * (1 + rng.uniform(-global_fluct, global_fluct))))
               for code, ratio in enumerate(ratios, 1)}
    available = decided.copy()
    hydro, polar = [], []
    while True:
        h, after_h = make_block(available, hydro_length, hydro_fluct, True, rng)
        if h is not None:
            hydro.append(h)
            available = after_h
        p, after_p = make_block(available, polar_length, polar_fluct, False, rng)
        if p is not None:
            polar.append(p)
            available = after_p
        if h is None and p is None:
            break
    tokens = {1: available[1], 2: available[2], 3: available[3], 4: available[4], 5: len(hydro), 6: len(polar)}
    order = []
    for _ in range(sum(tokens.values())):
        pick = choose(tokens, [1, 2, 3, 4, 5, 6], rng)
        if pick is None:
            raise RuntimeError("Unexpected exhausted token pool")
        tokens[pick] -= 1
        order.append(pick)
    rng.shuffle(hydro)
    rng.shuffle(polar)
    chain = []
    hi = pi = 0
    for token in order:
        if token == 5:
            chain.extend(hydro[hi]); hi += 1
        elif token == 6:
            chain.extend(polar[pi]); pi += 1
        else:
            chain.append(token)
    counts = Counter(chain)
    if len(chain) != sum(decided.values()) or any(counts.get(k, 0) != decided[k] for k in decided):
        raise RuntimeError("Pseudo-block assembly did not conserve monomer counts")
    return "".join(map(str, chain)), {"length": len(chain), "hydrophobic_blocks": len(hydro),
                                      "hydrophilic_blocks": len(polar), "counts": decided}
