import numpy as np
import argparse
from plyfile import PlyData, PlyElement


def load_ply_as_array(path):
    ply = PlyData.read(path)
    v = ply["vertex"].data
    names = v.dtype.names
    arr = np.vstack([v[n] for n in names]).T.astype(np.float64)
    return arr, names


def save_ply(arr, names, out_path):
    dtype = [(n, "f4") for n in names]
    structured = np.array([tuple(row) for row in arr], dtype=dtype)
    el = PlyElement.describe(structured, "vertex")
    PlyData([el]).write(out_path)


def quantize(arr, tol):
    return np.round(arr / tol).astype(np.int64)


def array_to_hashset(arr_q):
    return {tuple(row) for row in arr_q}


def check_append_only(low_arr, high_arr, tol, low_name, high_name):

    low_q = quantize(low_arr, tol)
    high_q = quantize(high_arr, tol)

    low_set = array_to_hashset(low_q)
    high_set = array_to_hashset(high_q)

    missing = low_set - high_set
    added = high_set - low_set

    stats = {
        "low_count": len(low_set),
        "high_count": len(high_set),
        "added": len(added),
        "missing": len(missing),
    }

    ok = (len(missing) == 0)

    print(f"\n=== Check {low_name} → {high_name} (append-only) ===")
    print(f"Low  count   : {stats['low_count']}")
    print(f"High count   : {stats['high_count']}")
    print(f"Added points : {stats['added']} (OK)")
    if ok:
        print(f"Missing      : {stats['missing']} ✓")
        print("✓ Append-only property holds (no deletions / no modifications).")
    else:
        print(f"Missing      : {stats['missing']} ❌")
        print("✗ Found missing points: lower-layer Gaussians were deleted or modified.")
    return ok, stats


def extract_delta(high_arr, low_arr, tol):
    high_q = quantize(high_arr, tol)
    low_q = quantize(low_arr, tol)

    low_set = array_to_hashset(low_q)
    mask = np.array([tuple(row) not in low_set for row in high_q])
    return high_arr[mask]


def main(args):
    # Load
    L0, names0 = load_ply_as_array(args.L0)
    L1, names1 = load_ply_as_array(args.L1)
    L2, names2 = load_ply_as_array(args.L2)
    L3, names3 = load_ply_as_array(args.L3)

    # Basic schema check
    if not (names0 == names1 == names2 == names3):
        raise ValueError(
            "PLY vertex properties are not identical across layers. "
            "This script assumes all layers share the same vertex fields."
        )

    print("Loaded shapes:", L0.shape, L1.shape, L2.shape, L3.shape)
    tol = args.tol

    # Check append-only
    ok01, _ = check_append_only(L0, L1, tol, "L0", "L1")
    ok12, _ = check_append_only(L1, L2, tol, "L1", "L2")
    ok23, _ = check_append_only(L2, L3, tol, "L2", "L3")

    all_ok = ok01 and ok12 and ok23
    if (not all_ok) and (not args.no_fail):
        raise RuntimeError(
            "Append-only check failed (missing points found). "
            "Use --no_fail to still export deltas, or fix your layered PLY generation."
        )

    # Extract deltas
    delta_L1 = extract_delta(L1, L0, tol)
    delta_L2 = extract_delta(L2, L1, tol)
    delta_L3 = extract_delta(L3, L2, tol)

    print("\n=== Delta sizes ===")
    print("ΔL1:", delta_L1.shape[0])
    print("ΔL2:", delta_L2.shape[0])
    print("ΔL3:", delta_L3.shape[0])

    # Save
    save_ply(delta_L1, names0, args.out1)
    save_ply(delta_L2, names0, args.out2)
    save_ply(delta_L3, names0, args.out3)

    print("\nDone.")
    print("Outputs:", args.out1, args.out2, args.out3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract delta Gaussians between layered 3DGS PLY files, with append-only consistency checks."
    )
    parser.add_argument("--L0", required=True, help="Path to L0.ply")
    parser.add_argument("--L1", required=True, help="Path to L1.ply")
    parser.add_argument("--L2", required=True, help="Path to L2.ply")
    parser.add_argument("--L3", required=True, help="Path to L3.ply")

    parser.add_argument("--out1", default="delta_L1.ply", help="Output ΔL1")
    parser.add_argument("--out2", default="delta_L2.ply", help="Output ΔL2")
    parser.add_argument("--out3", default="delta_L3.ply", help="Output ΔL3")

    parser.add_argument("--tol", type=float, default=1e-6, help="Float tolerance for matching (default: 1e-6)")
    parser.add_argument(
        "--no_fail",
        action="store_true",
        help="Do not fail even if append-only check fails; still export delta files."
    )

    args = parser.parse_args()
    main(args)


'''

python extract_3dgs_layer.py \
    --L0 examples/webxr/room_res16.ply \
    --L1 examples/webxr/room_res8.ply \
    --L2 examples/webxr/room_res4.ply \
    --L3 examples/webxr/room_res2.ply \
    --out1 examples/webxr/room_delta_res8.ply \
    --out2 examples/webxr/room_delta_res4.ply \
    --out3 examples/webxr/room_delta_res2.ply



'''