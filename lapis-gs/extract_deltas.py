#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
from plyfile import PlyData, PlyElement


def load_vertex(path):
    ply = PlyData.read(path)
    return ply, ply.elements[0].data


def save_vertex_like(src_ply, vertex, out_path):
    elem_name = src_ply.elements[0].name
    elem = PlyElement.describe(vertex, elem_name)
    PlyData([elem], text=src_ply.text).write(out_path)


def quant_keys(v, quant):
    """
    只用 xyz 做 key（最稳妥，适合你这种“直接 append 点”的分层）
    """
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    q = np.round(xyz / quant).astype(np.int64)
    return np.ascontiguousarray(q).view(
        np.dtype((np.void, q.dtype.itemsize * q.shape[1]))
    ).ravel()


def extract_delta(curr, prev, quant):
    """
    curr - prev（多重集差分）
    """
    curr_key = quant_keys(curr, quant)
    prev_key = quant_keys(prev, quant)

    uniq, cnt = np.unique(prev_key, return_counts=True)
    counter = dict(zip(uniq.tolist(), cnt.tolist()))

    keep = np.ones(len(curr), dtype=bool)
    for i, k in enumerate(curr_key.tolist()):
        if counter.get(k, 0) > 0:
            keep[i] = False
            counter[k] -= 1

    return curr[keep]


def main():
    ap = argparse.ArgumentParser(
        description="Extract L1/L2/L3 delta from cumulative 3DGS point_cloud.ply"
    )
    ap.add_argument("--l0", required=True, help="L0 directory")
    ap.add_argument("--l1", required=True, help="L1 directory")
    ap.add_argument("--l2", required=True, help="L2 directory")
    ap.add_argument("--l3", required=True, help="L3 directory")
    ap.add_argument("--quant", type=float, default=1e-6, help="quantization step")
    args = ap.parse_args()

    p0 = os.path.join(args.l0, "point_cloud.ply")
    p1 = os.path.join(args.l1, "point_cloud.ply")
    p2 = os.path.join(args.l2, "point_cloud.ply")
    p3 = os.path.join(args.l3, "point_cloud.ply")

    ply0, v0 = load_vertex(p0)
    ply1, v1 = load_vertex(p1)
    ply2, v2 = load_vertex(p2)
    ply3, v3 = load_vertex(p3)

    d1 = extract_delta(v1, v0, args.quant)
    d2 = extract_delta(v2, v1, args.quant)
    d3 = extract_delta(v3, v2, args.quant)

    save_vertex_like(ply1, d1, os.path.join(args.l1, "delta.ply"))
    save_vertex_like(ply2, d2, os.path.join(args.l2, "delta.ply"))
    save_vertex_like(ply3, d3, os.path.join(args.l3, "delta.ply"))

    print("[Done]")
    print(f"L1 delta: {len(d1)}")
    print(f"L2 delta: {len(d2)}")
    print(f"L3 delta: {len(d3)}")


if __name__ == "__main__":
    main()

'''
python extract_deltas.py \
  --l0 model_freeze_opacity/360/room/freeze/room_res16/point_cloud/iteration_30000 \
  --l1 model_freeze_opacity/360/room/freeze/room_res8/point_cloud/iteration_30000 \
  --l2 model_freeze_opacity/360/room/freeze/room_res4/point_cloud/iteration_30000 \
  --l3 model_freeze_opacity/360/room/freeze/room_res2/point_cloud/iteration_30000


'''