#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
from plyfile import PlyData, PlyElement


@dataclass
class Cuboid:
    cid: str
    lb: np.ndarray  # (3,)  in *8 coord
    rt: np.ndarray  # (3,)  in *8 coord


def load_ply_vertices(path: str) -> np.ndarray:
    ply = PlyData.read(path)
    if "vertex" not in ply:
        raise RuntimeError(f"No 'vertex' element in {path}")
    return ply["vertex"].data  # structured ndarray


def write_ply_vertices(path: str, verts: np.ndarray) -> None:
    el = PlyElement.describe(verts, "vertex")
    PlyData([el], text=False).write(path)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def parse_ids_from_streaming_dir(streaming_dir: str) -> Set[str]:
    """
    Extract the LAST integer group from each .ply filename stem.
    e.g. cuboid_3.ply -> "3", 000123.ply -> "000123", foo_12_bar.ply -> "12"
    """
    ids: Set[str] = set()
    for fn in os.listdir(streaming_dir):
        if not fn.lower().endswith(".ply"):
            continue
        stem = os.path.splitext(fn)[0]
        nums = re.findall(r"\d+", stem)
        if nums:
            ids.add(nums[-1])
    return ids


def load_cuboids_dict_json(json_path: str, filter_ids: Optional[Set[str]] = None) -> List[Cuboid]:
    """
    Your aabb_json format (dict):
    {
      "0": {"leftBottom":[...], "rightTop":[...], "num":..., "density":...},
      "3": {...}
    }
    Values are already in *8 coord in your setting.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cuboids: List[Cuboid] = []
    for cid, info in data.items():
        cid_str = str(cid)
        if filter_ids is not None and cid_str not in filter_ids:
            continue
        lb = np.array(info["leftBottom"], dtype=np.float64)
        rt = np.array(info["rightTop"], dtype=np.float64)
        cuboids.append(Cuboid(cid=cid_str, lb=lb, rt=rt))
    return cuboids


def build_hash_grid(cuboids: List[Cuboid]) -> Tuple[Dict[Tuple[int, int, int], List[int]], np.ndarray]:
    """
    Spatial hash for fast AABB lookup.
    """
    sizes = np.array([c.rt - c.lb for c in cuboids], dtype=np.float64)
    cell_size = np.median(sizes, axis=0)
    cell_size = np.maximum(cell_size, 1e-9)

    grid: Dict[Tuple[int, int, int], List[int]] = {}
    for idx, c in enumerate(cuboids):
        mn = np.floor(c.lb / cell_size).astype(int)
        mx = np.floor(c.rt / cell_size).astype(int)
        for ix in range(mn[0], mx[0] + 1):
            for iy in range(mn[1], mx[1] + 1):
                for iz in range(mn[2], mx[2] + 1):
                    grid.setdefault((ix, iy, iz), []).append(idx)
    return grid, cell_size


def point_in_aabb(p: np.ndarray, lb: np.ndarray, rt: np.ndarray, eps: float) -> bool:
    return (
        (p[0] >= lb[0] - eps) and (p[0] <= rt[0] + eps) and
        (p[1] >= lb[1] - eps) and (p[1] <= rt[1] + eps) and
        (p[2] >= lb[2] - eps) and (p[2] <= rt[2] + eps)
    )


def find_cuboid_for_point(
    p: np.ndarray,
    cuboids: List[Cuboid],
    grid: Dict[Tuple[int, int, int], List[int]],
    cell_size: np.ndarray,
    eps: float,
) -> Optional[int]:
    """
    Return index in cuboids list.
    If multiple matches (boundary), pick smallest numeric cid if possible.
    """
    cell = tuple(np.floor(p / cell_size).astype(int).tolist())
    cand = grid.get(cell, [])
    if not cand:
        cx, cy, cz = cell
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    cand.extend(grid.get((cx + dx, cy + dy, cz + dz), []))

    best_idx = None
    best_key = None
    for idx in cand:
        c = cuboids[idx]
        if point_in_aabb(p, c.lb, c.rt, eps):
            try:
                k = (int(c.cid), c.cid)
            except ValueError:
                k = (10**18, c.cid)
            if best_key is None or k < best_key:
                best_key = k
                best_idx = idx
    return best_idx


def scale_vertices_inplace(verts, pos_fields, scale):
    for pf in pos_fields:
        verts[pf] = verts[pf] * scale

    log_delta = math.log(scale)
    # only adjust if these fields exist
    for k in ["scale_0", "scale_1", "scale_2"]:
        if k in verts.dtype.names:
            verts[k] += log_delta


def ensure_same_schema(v_ref: np.ndarray, v_cur: np.ndarray, name_ref: str, name_cur: str):
    if v_ref.dtype.names != v_cur.dtype.names:
        raise RuntimeError(
            f"Vertex schema mismatch between {name_ref} and {name_cur}.\n"
            f"{name_ref}: {v_ref.dtype.names}\n"
            f"{name_cur}: {v_cur.dtype.names}\n"
            f"Need identical field names/order for --preserve_old_attrs."
        )
    for n in v_ref.dtype.names:
        if v_ref.dtype[n] != v_cur.dtype[n]:
            raise RuntimeError(
                f"Vertex dtype mismatch for field '{n}' between {name_ref} and {name_cur}: "
                f"{v_ref.dtype[n]} vs {v_cur.dtype[n]}"
            )


def main():
    ap = argparse.ArgumentParser(
        description="Given aabb_json (*8) and selected streaming_dir (*8), split full-scene L0~L3 (original scale) into per-cuboid layered PLYs: {id}-Lk.ply."
    )
    ap.add_argument("--aabb_json", required=True)
    ap.add_argument("--streaming_dir", required=True, help="Directory containing selected cuboid ply files (*8). Used to choose ids.")
    ap.add_argument("--l0", required=True, help="Full-scene L0 ply (original scale)")
    ap.add_argument("--l1", required=True, help="Full-scene L1 ply (original scale)")
    ap.add_argument("--l2", required=True, help="Full-scene L2 ply (original scale)")
    ap.add_argument("--l3", required=True, help="Full-scene L3 ply (original scale)")
    ap.add_argument("--scale", type=float, default=1.0, help="Scale factor to align points to AABB (*8). Default 8.")
    ap.add_argument("--pos_fields", type=str, default="x,y,z")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--flush", type=int, default=200000, help="Flush buffer size per (cuboid,seg)")
    ap.add_argument(
        "--output_scale",
        choices=["original", "scaled"],
        default="original",
        help="Coordinate scale of output PLYs. 'original' = keep original coords; 'scaled' = multiply position fields by --scale before writing."
    )

    # �? new flag
    ap.add_argument(
        "--preserve_old_attrs",
        action="store_true",
        help="If set: preserve old layers' attributes by writing seg0 from L0, seg1 from L1, seg2 from L2, seg3 from L3. "
             "Cuboid assignment still uses L3 xyz. Default (unset): use L3 rows for everything."
    )

    args = ap.parse_args()

    # Output dir: parent(streaming_dir) / layered_streaming_cuboids
    parent = os.path.dirname(os.path.abspath(args.streaming_dir.rstrip("/")))
    out_dir = os.path.join(parent, "layered_streaming_cuboids_scaled" if args.output_scale == "scaled" else "layered_streaming_cuboids")
    ensure_dir(out_dir)
    tmp_dir = os.path.join(out_dir, "_tmp_segments")
    ensure_dir(tmp_dir)

    # Choose ids based on files in streaming_dir
    ids_in_dir = parse_ids_from_streaming_dir(args.streaming_dir)
    if not ids_in_dir:
        raise RuntimeError(f"No .ply files / no numeric ids found in streaming_dir: {args.streaming_dir}")
    print(f"Found {len(ids_in_dir)} cuboid ids in streaming_dir (from filenames). Example: {sorted(list(ids_in_dir))[:10]}")

    # Load only those ids from json
    cuboids = load_cuboids_dict_json(args.aabb_json, filter_ids=ids_in_dir)
    found_ids = {c.cid for c in cuboids}
    missing_in_json = sorted(list(ids_in_dir - found_ids))
    if missing_in_json:
        print(f"WARNING: {len(missing_in_json)} ids exist in streaming_dir but NOT in aabb_json. Example: {missing_in_json[:10]}")
    print(f"Loaded cuboids from json after filtering: {len(cuboids)}")
    if not cuboids:
        raise RuntimeError("No cuboids remain after filtering. Check json keys vs filenames.")

    # Read full-scene layers
    v0 = load_ply_vertices(args.l0)
    v1 = load_ply_vertices(args.l1)
    v2 = load_ply_vertices(args.l2)
    v3 = load_ply_vertices(args.l3)

    # If we want to preserve old attrs, schema must match
    if args.preserve_old_attrs:
        ensure_same_schema(v3, v0, "L3", "L0")
        ensure_same_schema(v3, v1, "L3", "L1")
        ensure_same_schema(v3, v2, "L3", "L2")

    N0, N1, N2, N3 = len(v0), len(v1), len(v2), len(v3)
    print(f"Counts: N0={N0}, N1={N1}, N2={N2}, N3={N3}")
    if not (N0 <= N1 <= N2 <= N3):
        raise RuntimeError("Counts are not non-decreasing; append-at-end assumption fails.")

    pos_fields = [s.strip() for s in args.pos_fields.split(",")]
    for pf in pos_fields:
        if pf not in v3.dtype.names:
            raise RuntimeError(f"Missing position field '{pf}' in L3 vertex. Available: {v3.dtype.names}")

    # Hash grid on cuboids (AABB are already *8)
    grid, cell_size = build_hash_grid(cuboids)
    print(f"Hash grid built. cell_size={cell_size.tolist()}")

    dtype = v3.dtype

    # buffers[(cuboid_index, seg)] : seg 0=L0, 1=Δ1, 2=Δ2, 3=Δ3
    buffers: Dict[Tuple[int, int], List[np.ndarray]] = {}
    part_idx: Dict[Tuple[int, int], int] = {}

    def flush(ci: int, seg: int):
        key = (ci, seg)
        if key not in buffers or not buffers[key]:
            return
        arr = np.concatenate(buffers[key], axis=0)
        buffers[key] = []
        part = part_idx.get(key, 0)
        part_idx[key] = part + 1

        cid = cuboids[ci].cid
        out_path = os.path.join(tmp_dir, f"cuboid_{cid}_seg{seg}_part{part:04d}.npy")
        np.save(out_path, arr)

    assigned = 0
    missed = 0

    print(f"[Mode] preserve_old_attrs={args.preserve_old_attrs} | output_scale={args.output_scale}")

    # One pass over full L3 for cuboid assignment; write rows depending on mode
    for j in range(N3):
        # seg by prefix counts
        if j < N0:
            seg = 0
        elif j < N1:
            seg = 1
        elif j < N2:
            seg = 2
        else:
            seg = 3

        # ALWAYS use L3 xyz to decide cuboid (stable)
        row_pos = v3[j]
        p = np.array([row_pos[pos_fields[0]], row_pos[pos_fields[1]], row_pos[pos_fields[2]]], dtype=np.float64)
        p = p * args.scale  # points are original, AABB is *8

        ci = find_cuboid_for_point(p, cuboids, grid, cell_size, eps=args.eps)
        if ci is None:
            missed += 1
            continue

        # choose which layer's attributes to write
        if args.preserve_old_attrs:
            if seg == 0:
                row = v0[j]
            elif seg == 1:
                row = v1[j]
            elif seg == 2:
                row = v2[j]
            else:
                row = v3[j]
        else:
            row = v3[j]

        key = (ci, seg)
        buffers.setdefault(key, [])
        buffers[key].append(np.array([row], dtype=dtype))
        if len(buffers[key]) >= args.flush:
            flush(ci, seg)

        assigned += 1
        if (j + 1) % 1_000_000 == 0:
            print(f"Processed {j+1}/{N3} | assigned={assigned} missed(outside selected cuboids)={missed}")

    # flush all
    for (ci, seg) in list(buffers.keys()):
        flush(ci, seg)

    print(f"Done streaming full L3. assigned={assigned}, missed(outside selected cuboids)={missed}")

    def load_seg(cid: str, seg: int) -> np.ndarray:
        prefix = f"cuboid_{cid}_seg{seg}_part"
        parts = []
        for fn in sorted(os.listdir(tmp_dir)):
            if fn.startswith(prefix) and fn.endswith(".npy"):
                parts.append(np.load(os.path.join(tmp_dir, fn), allow_pickle=False))
        if not parts:
            return np.empty((0,), dtype=dtype)
        return np.concatenate(parts, axis=0)

    # Write {id}-{layer}.ply directly into out_dir
    for c in cuboids:
        cid = c.cid
        base = load_seg(cid, 0)
        d1 = load_seg(cid, 1)
        d2 = load_seg(cid, 2)
        d3 = load_seg(cid, 3)

        L0c = base
        L1c = np.concatenate([base, d1], axis=0)
        L2c = np.concatenate([base, d1, d2], axis=0)
        L3c = np.concatenate([base, d1, d2, d3], axis=0)

        if args.output_scale == "scaled":
            scale_vertices_inplace(L0c, pos_fields, args.scale)
            scale_vertices_inplace(L1c, pos_fields, args.scale)
            scale_vertices_inplace(L2c, pos_fields, args.scale)
            scale_vertices_inplace(L3c, pos_fields, args.scale)

        write_ply_vertices(os.path.join(out_dir, f"{cid}-L0.ply"), L0c)
        write_ply_vertices(os.path.join(out_dir, f"{cid}-L1.ply"), L1c)
        write_ply_vertices(os.path.join(out_dir, f"{cid}-L2.ply"), L2c)
        write_ply_vertices(os.path.join(out_dir, f"{cid}-L3.ply"), L3c)

        print(f"[cuboid {cid}] L0={len(L0c)} L1={len(L1c)} L2={len(L2c)} L3={len(L3c)}")

    print(f"\n�? Output written to: {out_dir}")
    print(f"Temp segment files in: {tmp_dir} (delete after verification)")


if __name__ == "__main__":
    main()


'''
python split_streaming_cuboids.py \
  --aabb_json scenes/longdress/optimal_voxels/voxel_ilp.json \
  --streaming_dir scenes/longdress/optimal_voxels \
  --l0 /home/why/dynamic-lapis-gs/model/8i/longdress/dynamic-lapis/longdress_res8/1051/point_cloud/iteration_30000/point_cloud.ply \
  --l1 /home/why/dynamic-lapis-gs/model/8i/longdress/dynamic-lapis/longdress_res4/1051/point_cloud/iteration_30000/point_cloud.ply \
  --l2 /home/why/dynamic-lapis-gs/model/8i/longdress/dynamic-lapis/longdress_res2/1051/point_cloud/iteration_30000/point_cloud.ply \
  --l3 /home/why/dynamic-lapis-gs/model/8i/longdress/dynamic-lapis/longdress_res1/1051/point_cloud/iteration_30000/point_cloud.ply \
  --scale 8 \
  --pos_fields x,y,z \
  --output_scale original \
  --preserve_old_attrs



python split_streaming_cuboids.py \
  --aabb_json scenes/room/optimal_voxels/voxel_ilp.json \
  --streaming_dir scenes/room/optimal_voxels \
  --l0 model_freeze_opacity/360/room/freeze/room_res16/point_cloud/iteration_30000/point_cloud.ply \
  --l1 model_freeze_opacity/360/room/freeze/room_res8/point_cloud/iteration_30000/point_cloud.ply \
  --l2 model_freeze_opacity/360/room/freeze/room_res4/point_cloud/iteration_30000/point_cloud.ply \
  --l3 model_freeze_opacity/360/room/freeze/room_res2/point_cloud/iteration_30000/point_cloud.ply \
  --pos_fields x,y,z \
  --output_scale original \
  --preserve_old_attrs


python split_streaming_cuboids.py \
  --aabb_json scenes/room/optimal_voxels/voxel_ilp.json \
  --streaming_dir scenes/room/optimal_voxels \
  --l0 /home/why/lapis-gs/model_freeze_all_res_fewer_splats/360/room/freeze/room_res16/point_cloud/iteration_30000/point_cloud_nonan.ply \
  --l1 /home/why/lapis-gs/model_freeze_all_res_fewer_splats/360/room/freeze/room_res8/point_cloud/iteration_30000/point_cloud_nonan.ply \
  --l2 /home/why/lapis-gs/model_freeze_all_res_fewer_splats/360/room/freeze/room_res4/point_cloud/iteration_30000/point_cloud_nonan.ply \
  --l3 /home/why/lapis-gs/model_freeze_all_res_fewer_splats/360/room/freeze/room_res2/point_cloud/iteration_30000/point_cloud_nonan.ply \
  --pos_fields x,y,z \
  --output_scale original \
  --preserve_old_attrs



'''