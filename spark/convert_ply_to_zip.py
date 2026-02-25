import argparse
import re
from pathlib import Path

import argparse
import json
import math
import os
import zipfile
from dataclasses import dataclass

import numpy as np
from PIL import Image
from plyfile import PlyData


SH_C0 = 0.28209479177387814  # used in decoding, here for reference only


@dataclass
class GaussianPly:
    means: np.ndarray   # (N,3) float32
    scales: np.ndarray  # (N,3) float32
    quats: np.ndarray   # (N,4) float32  (x,y,z,w)
    sh0: np.ndarray     # (N,3) float32  DC coeffs per channel
    opacity: np.ndarray # (N,) float32 in [0,1]


def _sym_log(x: np.ndarray) -> np.ndarray:
    # symmetric log transform used by SOG for positions (see spec)
    # n = sign(x) * log(1 + |x|)
    return np.sign(x) * np.log1p(np.abs(x))


def _ensure_quat_xyzw(q: np.ndarray) -> np.ndarray:
    """
    Input is (N,4). Many PLYs store rot_0..3 as (w,x,y,z) or (x,y,z,w).
    We'll try to detect by heuristics:
      - if abs(q[...,0]) tends to be largest, it might be w-first.
    You can override with --quat-order.
    """
    return q


def _pack_quat_smallest_three(q_xyzw: np.ndarray) -> np.ndarray:
    """
    SOG v2 quaternion packing:
      - choose the component with largest abs to omit (mode 0..3 for x,y,z,w)
      - flip sign so omitted component is non-negative
      - store remaining 3 comps quantized uniformly over [-sqrt(2)/2, +sqrt(2)/2]
      - A stores mode in 252..255 (A=252+mode)
    Spec: "26-bit smallest-three". :contentReference[oaicite:1]{index=1}
    """
    q = q_xyzw.astype(np.float32).copy()
    # normalize to be safe
    nrm = np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    q /= nrm

    absq = np.abs(q)
    mode = np.argmax(absq, axis=1).astype(np.int32)  # 0..3 for x,y,z,w

    # flip sign so omitted component is >=0
    omitted = q[np.arange(q.shape[0]), mode]
    flip = (omitted < 0).astype(np.float32) * -2.0 + 1.0  # +1 or -1
    q *= flip[:, None]

    # gather remaining three components in order (a,b,c) matching spec's decode slots
    # decode uses (r,g,b) -> (a,b,c) then inserts d at omitted slot.
    comps = np.zeros((q.shape[0], 3), dtype=np.float32)
    for i in range(q.shape[0]):
        m = mode[i]
        # omit m
        kept = [0, 1, 2, 3]
        kept.pop(m)
        comps[i, :] = q[i, kept]

    # quantize to uint8 over [-1/sqrt2, +1/sqrt2]
    lo, hi = -1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)
    t = (comps - lo) / (hi - lo)
    t = np.clip(t, 0.0, 1.0)
    rgb = np.round(t * 255.0).astype(np.uint8)

    a = (252 + mode).astype(np.uint8)[:, None]
    rgba = np.concatenate([rgb, a], axis=1)  # (N,4) uint8
    return rgba


def _choose_texture_dims(count: int) -> tuple[int, int]:
    # Simple near-square packing
    w = int(math.ceil(math.sqrt(count)))
    h = int(math.ceil(count / w))
    return w, h


def _pack_to_image(data_u8: np.ndarray, width: int, height: int, channels: int) -> np.ndarray:
    """
    data_u8: (N,channels) uint8
    return: (H,W,channels) uint8
    """
    img = np.zeros((height * width, channels), dtype=np.uint8)
    img[: data_u8.shape[0], :] = data_u8
    img = img.reshape(height, width, channels)
    return img


def _save_webp(path: str, arr: np.ndarray):
    mode = "RGB" if arr.shape[2] == 3 else "RGBA"
    im = Image.fromarray(arr, mode=mode)
    im.save(path, format="WEBP", lossless=True, quality=100, method=6)


def _make_codebook_256(values: np.ndarray) -> np.ndarray:
    """
    Uniform codebook with 256 entries spanning [min,max].
    SOG spec uses codebook indexing for scales and sh0. :contentReference[oaicite:2]{index=2}
    """
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax - vmin < 1e-12:
        return np.full((256,), vmin, dtype=np.float32)
    return np.linspace(vmin, vmax, 256, dtype=np.float32)


def _quantize_to_codebook(values: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """
    values: (...,) float32
    returns: (...,) uint8 indices 0..255 (nearest)
    """
    cb = codebook.astype(np.float32)
    # map to [0,255] by linear inverse (since cb is uniform linspace)
    vmin = cb[0]
    vmax = cb[-1]
    if vmax - vmin < 1e-12:
        return np.zeros(values.shape, dtype=np.uint8)
    t = (values - vmin) / (vmax - vmin)
    idx = np.round(np.clip(t, 0.0, 1.0) * 255.0).astype(np.uint8)
    return idx


def read_gaussian_ply(ply_path: str, quat_order: str = "xyzw") -> GaussianPly:
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data
    names = v.dtype.names

    def need(fields):
        for f in fields:
            if f not in names:
                raise KeyError(f"Missing field '{f}' in PLY. Available: {names}")

    # Common gaussian splat PLY fields:
    # x,y,z
    need(["x", "y", "z"])
    means = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    # scales: scale_0..2 or scales_0..2
    if all(f"scale_{i}" in names for i in range(3)):
        scales = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32)
    elif all(f"scales_{i}" in names for i in range(3)):
        scales = np.stack([v["scales_0"], v["scales_1"], v["scales_2"]], axis=1).astype(np.float32)
    else:
        raise KeyError("Cannot find scales fields (scale_0..2 or scales_0..2).")

    # quats: rot_0..3 or quat_0..3
    if all(f"rot_{i}" in names for i in range(4)):
        q = np.stack([v[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float32)
    elif all(f"quat_{i}" in names for i in range(4)):
        q = np.stack([v[f"quat_{i}"] for i in range(4)], axis=1).astype(np.float32)
    else:
        raise KeyError("Cannot find quaternion fields (rot_0..3 or quat_0..3).")

    # order convert
    if quat_order == "wxyz":
        # stored as (w,x,y,z) -> (x,y,z,w)
        quats = np.stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]], axis=1)
    elif quat_order == "xyzw":
        quats = q
    else:
        raise ValueError("--quat-order must be 'xyzw' or 'wxyz'")

    # sh0 DC coeffs:
    # Many 3DGS PLYs use f_dc_0..2 (or features_dc_0..2)
    if all(f"f_dc_{i}" in names for i in range(3)):
        sh0 = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
    elif all(f"features_dc_{i}" in names for i in range(3)):
        sh0 = np.stack([v["features_dc_0"], v["features_dc_1"], v["features_dc_2"]], axis=1).astype(np.float32)
    else:
        raise KeyError("Cannot find DC SH fields (f_dc_0..2 or features_dc_0..2).")

    # opacity
    if "opacity" in names:
        opacity = np.array(v["opacity"], dtype=np.float32)
    elif "alpha" in names:
        opacity = np.array(v["alpha"], dtype=np.float32)
    else:
        raise KeyError("Cannot find opacity field ('opacity' or 'alpha').")

    # clamp opacity to [0,1]
    opacity = np.clip(opacity, 0.0, 1.0)

    return GaussianPly(means=means, scales=scales, quats=quats, sh0=sh0, opacity=opacity)


def convert_to_sog(ply_path: str, out_dir: str, antialias: bool = False, quat_order: str = "xyzw") -> str:
    os.makedirs(out_dir, exist_ok=True)
    g = read_gaussian_ply(ply_path, quat_order=quat_order)
    N = g.means.shape[0]
    W, H = _choose_texture_dims(N)

    # ---- means: sym-log -> per-axis uint16 -> split into low/high bytes across 2 RGB images
    npos = _sym_log(g.means)
    mins = npos.min(axis=0).astype(np.float32)
    maxs = npos.max(axis=0).astype(np.float32)
    span = np.maximum(maxs - mins, 1e-12)

    q16 = np.round((npos - mins[None, :]) / span[None, :] * 65535.0).astype(np.uint16)
    means_l = (q16 & 0xFF).astype(np.uint8)         # (N,3)
    means_u = ((q16 >> 8) & 0xFF).astype(np.uint8)  # (N,3)

    means_l_img = _pack_to_image(means_l, W, H, 3)
    means_u_img = _pack_to_image(means_u, W, H, 3)
    _save_webp(os.path.join(out_dir, "means_l.webp"), means_l_img)
    _save_webp(os.path.join(out_dir, "means_u.webp"), means_u_img)

    # ---- scales: codebook index per axis (RGB)
    scales_cb = _make_codebook_256(g.scales)
    sx = _quantize_to_codebook(g.scales[:, 0], scales_cb)
    sy = _quantize_to_codebook(g.scales[:, 1], scales_cb)
    sz = _quantize_to_codebook(g.scales[:, 2], scales_cb)
    scales_rgb = np.stack([sx, sy, sz], axis=1).astype(np.uint8)
    scales_img = _pack_to_image(scales_rgb, W, H, 3)
    _save_webp(os.path.join(out_dir, "scales.webp"), scales_img)

    # ---- quats: smallest-three pack (RGBA)
    quats_rgba = _pack_quat_smallest_three(g.quats)
    quats_img = _pack_to_image(quats_rgba, W, H, 4)
    _save_webp(os.path.join(out_dir, "quats.webp"), quats_img)

    # ---- sh0: R,G,B are indices into sh0.codebook; A is opacity in [0,1] as uint8
    sh0_cb = _make_codebook_256(g.sh0)
    r = _quantize_to_codebook(g.sh0[:, 0], sh0_cb)
    gg = _quantize_to_codebook(g.sh0[:, 1], sh0_cb)
    b = _quantize_to_codebook(g.sh0[:, 2], sh0_cb)
    a = np.round(g.opacity * 255.0).astype(np.uint8)
    sh0_rgba = np.stack([r, gg, b, a], axis=1).astype(np.uint8)
    sh0_img = _pack_to_image(sh0_rgba, W, H, 4)
    _save_webp(os.path.join(out_dir, "sh0.webp"), sh0_img)

    # ---- meta.json (SOG v2)
    meta = {
        "version": 2,
        "count": int(N),
        "antialias": bool(antialias),
        "means": {
            "mins": [float(mins[0]), float(mins[1]), float(mins[2])],
            "maxs": [float(maxs[0]), float(maxs[1]), float(maxs[2])],
            "files": ["means_l.webp", "means_u.webp"],
        },
        "scales": {
            "codebook": [float(x) for x in scales_cb.tolist()],
            "files": ["scales.webp"],
        },
        "quats": {
            "files": ["quats.webp"],
        },
        "sh0": {
            "codebook": [float(x) for x in sh0_cb.tolist()],
            "files": ["sh0.webp"],
        },
        # shN optional (not generated in this minimal script)
    }

    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # ---- zip it
    base = os.path.splitext(os.path.basename(ply_path))[0]
    zip_path = os.path.join(os.path.dirname(out_dir), f"{base}.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fn in ["meta.json", "means_l.webp", "means_u.webp", "scales.webp", "quats.webp", "sh0.webp"]:
            z.write(os.path.join(out_dir, fn), arcname=fn)

    return zip_path




# 你需要把我之前给的单文件函数放在同一个文件里：
# convert_to_sog(ply_path: str, out_dir: str, antialias=False, quat_order="xyzw") -> str

PAT = re.compile(r"^(?P<cuboid>\d+)-L(?P<layer>\d+)\.ply$", re.IGNORECASE)

def collect_plys(in_dir: Path, recursive: bool):
    if recursive:
        files = list(in_dir.rglob("*.ply"))
    else:
        files = list(in_dir.glob("*.ply"))

    out = []
    for p in files:
        m = PAT.match(p.name)
        if m:
            out.append(p)
    return sorted(out)

def out_paths_for(ply_path: Path, in_dir: Path, out_root: Path, keep_structure: bool):
    """
    输出：
      out_root / (relative_dir) / {stem}/   (存meta+webp)
      out_root / (relative_dir) / {stem}.zip
    其中 stem = 263-L0
    """
    rel_dir = ply_path.parent.relative_to(in_dir) if keep_structure else Path(".")
    out_dir = (out_root / rel_dir / ply_path.stem)
    zip_path = (out_root / rel_dir / f"{ply_path.stem}.zip")
    return out_dir, zip_path

def batch_convert(in_dir: str, out_root: str, recursive: bool, keep_structure: bool,
                  overwrite: bool, antialias: bool, quat_order: str):
    in_dir = Path(in_dir).resolve()
    out_root = Path(out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    plys = collect_plys(in_dir, recursive)
    if not plys:
        print(f"[Warn] No files matching '*-L*.ply' under: {in_dir}")
        return

    print(f"[Info] Found {len(plys)} layer PLYs under: {in_dir}")

    ok = skipped = failed = 0
    for i, ply_path in enumerate(plys, 1):
        try:
            out_dir, zip_path = out_paths_for(ply_path, in_dir, out_root, keep_structure)

            if zip_path.exists() and not overwrite:
                skipped += 1
                print(f"[{i}/{len(plys)}][Skip] zip exists: {zip_path}")
                continue

            out_dir.mkdir(parents=True, exist_ok=True)

            zip_created = Path(convert_to_sog(
                ply_path=str(ply_path),
                out_dir=str(out_dir),
                antialias=antialias,
                quat_order=quat_order,
            ))

            # 统一 zip 名字为 {stem}.zip（比如 263-L0.zip）
            if zip_created.resolve() != zip_path.resolve():
                if zip_path.exists():
                    zip_path.unlink()
                zip_created.replace(zip_path)

            ok += 1
            print(f"[{i}/{len(plys)}][OK] {ply_path.name} -> {zip_path.name}")

        except Exception as e:
            failed += 1
            print(f"[{i}/{len(plys)}][Fail] {ply_path}: {e}")

    print(f"\n[Done] OK={ok}, Skipped={skipped}, Failed={failed}")
    print(f"[Out] {out_root}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="Input directory containing 263-L0.ply style files")
    ap.add_argument("--out-root", required=True, help="Output root directory for zips")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    ap.add_argument("--keep-structure", action="store_true",
                    help="Keep subfolder structure under out-root (recommended if recursive)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing zips")
    ap.add_argument("--antialias", action="store_true", help="Set meta.antialias=true")
    ap.add_argument("--quat-order", choices=["xyzw", "wxyz"], default="xyzw",
                    help="Quaternion order in PLY (default xyzw)")

    args = ap.parse_args()

    batch_convert(
        in_dir=args.in_dir,
        out_root=args.out_root,
        recursive=args.recursive,
        keep_structure=args.keep_structure,
        overwrite=args.overwrite,
        antialias=args.antialias,
        quat_order=args.quat_order,
    )

if __name__ == "__main__":
    main()

'''

python convert_ply_to_zip.py \
  --in-dir examples/webxr/layered_pure_streaming_cuboids_room \
  --out-root examples/webxr/layered_pure_streaming_cuboids_room_zip \
  --overwrite

'''