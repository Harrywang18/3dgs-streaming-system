#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
from typing import Dict, List, Tuple

import numpy as np
from plyfile import PlyData, PlyElement


def load_ply_vertices(path: str) -> np.ndarray:
    ply = PlyData.read(path)
    if "vertex" not in ply:
        raise RuntimeError(f"No 'vertex' in {path}")
    return ply["vertex"].data


def write_ply_vertices(path: str, verts: np.ndarray) -> None:
    el = PlyElement.describe(verts, "vertex")
    PlyData([el], text=False).write(path)


def parse_layered_files(layered_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Expect files like:
      {id}-L0.ply, {id}-L1.ply, {id}-L2.ply, {id}-L3.ply
    Returns:
      mapping[id][layer] = filepath
    """
    pat = re.compile(r"^(?P<id>.+)-(?P<layer>L[0-3])\.ply$", re.IGNORECASE)
    mp: Dict[str, Dict[str, str]] = {}
    for fn in os.listdir(layered_dir):
        m = pat.match(fn)
        if not m:
            continue
        cid = m.group("id")
        layer = m.group("layer").upper()
        mp.setdefault(cid, {})[layer] = os.path.join(layered_dir, fn)
    return mp


def sort_cuboid_ids(ids: List[str]) -> List[str]:
    """
    Sort ids numerically if possible, else lexicographically.
    """
    def key(x: str):
        try:
            return (0, int(x), x)
        except ValueError:
            return (1, 0, x)
    return sorted(ids, key=key)


def main():
    ap = argparse.ArgumentParser(description="Assemble per-cuboid layered PLYs into full-scene L0~L3.")
    ap.add_argument("--layered_dir", required=True, help="Directory containing {id}-L0.ply..{id}-L3.ply")
    ap.add_argument("--out_dir", required=True, help="Output directory for assembled full-scene layers")
    ap.add_argument(
        "--layers",
        default="L0,L1,L2,L3",
        help="Comma-separated layers to assemble (default: L0,L1,L2,L3)"
    )
    args = ap.parse_args()

    layers = [x.strip().upper() for x in args.layers.split(",") if x.strip()]
    for L in layers:
        if L not in {"L0", "L1", "L2", "L3"}:
            raise ValueError(f"Invalid layer: {L}")

    mp = parse_layered_files(args.layered_dir)
    if not mp:
        raise RuntimeError(f"No matching files found in {args.layered_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    cuboid_ids = sort_cuboid_ids(list(mp.keys()))
    print(f"Found {len(cuboid_ids)} cuboids. Example ids: {cuboid_ids[:10]}")

    # Load one file to get dtype sanity
    first_id = cuboid_ids[0]
    first_layer = layers[0]
    if first_layer not in mp[first_id]:
        raise RuntimeError(f"Missing {first_layer} for cuboid {first_id}")
    dtype_ref = load_ply_vertices(mp[first_id][first_layer]).dtype

    for L in layers:
        parts: List[np.ndarray] = []
        total = 0
        missing = 0

        for cid in cuboid_ids:
            if L not in mp[cid]:
                missing += 1
                continue
            v = load_ply_vertices(mp[cid][L])
            if v.dtype != dtype_ref:
                raise RuntimeError(f"Dtype mismatch at cuboid {cid} layer {L}: {v.dtype} vs {dtype_ref}")
            parts.append(v)
            total += len(v)

        if not parts:
            print(f"[{L}] No parts to assemble (missing={missing}). Skipped.")
            continue

        full = np.concatenate(parts, axis=0)
        out_path = os.path.join(args.out_dir, f"assembled_{L}.ply")
        write_ply_vertices(out_path, full)
        print(f"[{L}] assembled verts={len(full)} from cuboids={len(parts)} missing_cuboids={missing} -> {out_path}")


if __name__ == "__main__":
    main()

'''

python assemble_full_layers.py \
  --layered_dir scenes/longdress/layered_streaming_cuboids \
  --out_dir scenes/longdress/assembled_full_scene



'''