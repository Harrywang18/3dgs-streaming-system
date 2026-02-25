import os
import re
from collections import defaultdict, Counter

import numpy as np
from plyfile import PlyData


# =========================
# Config
# =========================
ROOT_DIR = "scenes/longdress/layered_pure_streaming_cuboids"

# 如果 {id}-Lk.ply 是累积层（L1含L0，L2含L0+L1...），用 True（你之前说“叠加出来的”，通常是 True）
# 如果 {id}-Lk.ply 已经是非叠加层（每层仅自己的点），设 False
ASSUME_CUMULATIVE = False

# K=32 的 token 分桶（你要求的版本）
K = 32

T0_FIXED = 100  # L0 输入固定 token 数

# 分桶阈值与 T（按 delta 点数 |Δk|）
# 你可以根据显存/预算改 T 的上限，例如把 512 改 400
BUCKETS = {
    1: [  # Δ1
        (1000, 64),
        (4000, 128),
        (float("inf"), 256),
    ],
    2: [  # Δ2
        (2000, 128),
        (8000, 256),
        (float("inf"), 384),
    ],
    3: [  # Δ3
        (3000, 256),
        (16000, 384),
        (float("inf"), 512),
    ],
}

# 如果你想强制每层 token 上限（例如 streaming 预算），填这里；不需要就设 None
# 例：CAP = {1: 200, 2: 300, 3: 400}
CAP = None


# =========================
# Helpers
# =========================
def count_vertices(ply_path: str) -> int:
    ply = PlyData.read(ply_path)
    if "vertex" not in ply:
        raise ValueError(f"No vertex element in {ply_path}")
    return int(ply["vertex"].count)


def choose_T(layer: int, n_delta: int) -> int:
    if layer == 0:
        return T0_FIXED
    for thr, T in BUCKETS[layer]:
        if n_delta <= thr:
            if CAP is not None and layer in CAP:
                return int(min(T, CAP[layer]))
            return int(T)
    raise RuntimeError("Bucket config error")


def safe_int_id(x: str):
    return int(x) if x.isdigit() else x


# =========================
# Main
# =========================
def main():
    pattern = re.compile(r"(.+)-L(\d+)\.ply$")
    counts = defaultdict(dict)  # counts[id][layer] = Ck (vertex count read from ply)

    for fname in os.listdir(ROOT_DIR):
        if not fname.endswith(".ply"):
            continue
        m = pattern.match(fname)
        if not m:
            continue
        cid, layer = m.group(1), int(m.group(2))
        path = os.path.join(ROOT_DIR, fname)
        try:
            counts[cid][layer] = count_vertices(path)
        except Exception as e:
            print(f"[Warning] Failed to read {fname}: {e}")

    if not counts:
        raise RuntimeError(f"No matched ply files found in: {ROOT_DIR}")

    # Prepare outputs
    per_cuboid = []
    bucket_counter = {1: Counter(), 2: Counter(), 3: Counter()}

    for cid in sorted(counts.keys(), key=safe_int_id):
        C = [counts[cid].get(i, None) for i in range(4)]

        # If missing layers, skip or treat as 0
        if any(v is None for v in C):
            print(f"[Warning] Missing some layers for cuboid {cid}: {C}  -> skip")
            continue

        C0, C1, C2, C3 = C

        if ASSUME_CUMULATIVE:
            d0 = C0
            d1 = max(0, C1 - C0)
            d2 = max(0, C2 - C1)
            d3 = max(0, C3 - C2)
        else:
            # already non-cumulative (each ply is its own layer)
            d0, d1, d2, d3 = C0, C1, C2, C3

        # token allocation
        T0 = choose_T(0, d0)
        T1 = choose_T(1, d1)
        T2 = choose_T(2, d2)
        T3 = choose_T(3, d3)

        # record bucket name for summary
        def bucket_name(layer, n):
            for thr, T in BUCKETS[layer]:
                if n <= thr:
                    return f"≤{int(thr) if np.isfinite(thr) else 'inf'}:T{T}"
            return "unknown"

        b1 = bucket_name(1, d1); bucket_counter[1][b1] += 1
        b2 = bucket_name(2, d2); bucket_counter[2][b2] += 1
        b3 = bucket_name(3, d3); bucket_counter[3][b3] += 1

        per_cuboid.append({
            "id": cid,
            "C0": C0, "C1": C1, "C2": C2, "C3": C3,
            "d0": d0, "d1": d1, "d2": d2, "d3": d3,
            "T0": T0, "T1": T1, "T2": T2, "T3": T3,
        })

    # Print per-cuboid result (compact)
    print("\nPer-cuboid (non-cumulative deltas + token allocation)")
    print("=" * 90)
    print(f"ASSUME_CUMULATIVE = {ASSUME_CUMULATIVE}, K = {K}, T0 = {T0_FIXED}, CAP = {CAP}")
    print("-" * 90)
    print("id | d0 d1 d2 d3 | T0 T1 T2 T3")
    print("-" * 90)
    for r in per_cuboid:
        print(f"{r['id']:>4} | "
              f"{r['d0']:>5} {r['d1']:>6} {r['d2']:>6} {r['d3']:>6} | "
              f"{r['T0']:>3} {r['T1']:>3} {r['T2']:>3} {r['T3']:>3}")

    # Stats on deltas
    deltas = {k: [] for k in range(4)}
    for r in per_cuboid:
        deltas[0].append(r["d0"])
        deltas[1].append(r["d1"])
        deltas[2].append(r["d2"])
        deltas[3].append(r["d3"])

    print("\nDelta stats (over cuboids)")
    print("=" * 90)
    for k in range(4):
        arr = np.array(deltas[k], dtype=np.float64)
        print(f"Δ{k}: min={int(arr.min()):>6}, max={int(arr.max()):>6}, mean={arr.mean():>10.2f}, n={len(arr)}")

    # Bucket summary
    print("\nBucket summary (counts)")
    print("=" * 90)
    for layer in [1, 2, 3]:
        print(f"Δ{layer}:")
        for name, c in bucket_counter[layer].most_common():
            print(f"  {name:<18} -> {c}")
        print()

    # Optional: save CSV
    out_csv = os.path.join(ROOT_DIR, "token_allocation_delta.csv")
    with open(out_csv, "w") as f:
        header = ["id", "C0","C1","C2","C3","d0","d1","d2","d3","T0","T1","T2","T3"]
        f.write(",".join(header) + "\n")
        for r in per_cuboid:
            f.write(",".join(str(r[h]) for h in header) + "\n")
    print(f"[Done] Saved: {out_csv}")


if __name__ == "__main__":
    main()
