import os
import pickle
from pathlib import Path

import numpy as np
from PIL import Image


def save_rgb_png(img_float01: np.ndarray, out_path: Path):
    """
    img_float01: (H,W,3) float32, typically in [0,1]
    """
    img = np.clip(img_float01, 0.0, 1.0)
    img_u8 = (img * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(img_u8).save(out_path)


def export_scene(scene_name: str, scene_data: dict, out_root: Path,
                 export_images=True, export_depths=True, export_cam=True,
                 depth_format="npy"):
    """
    depth_format: "npy" (推荐) or "png16" (需要深度已归一化/或你自己定义scale)
    """
    out_scene = out_root / scene_name
    out_scene.mkdir(parents=True, exist_ok=True)

    images = scene_data.get("images", None)        # (N,256,256,3) float32
    depths = scene_data.get("depths", None)        # (N,256,256,1) float32
    extr = scene_data.get("extrinsics", None)      # (N,4,4) float32
    intr = scene_data.get("intrinsics", None)      # (N,4,4) float32

    # -------- images --------
    if export_images and images is not None:
        img_dir = out_scene / "images"
        img_dir.mkdir(exist_ok=True)
        N = images.shape[0]
        for i in range(N):
            save_rgb_png(images[i], img_dir / f"{i:05d}.png")
        print(f"[{scene_name}] images -> {img_dir} ({N} files)")

    # -------- depths --------
    if export_depths and depths is not None:
        dep_dir = out_scene / "depths"
        dep_dir.mkdir(exist_ok=True)
        N = depths.shape[0]

        if depth_format == "npy":
            for i in range(N):
                # 保存 (H,W) float32 深度
                np.save(dep_dir / f"{i:05d}.npy", depths[i, :, :, 0].astype(np.float32))
            print(f"[{scene_name}] depths(.npy) -> {dep_dir} ({N} files)")

        elif depth_format == "png16":
            # 只有当你明确深度范围/单位时才建议用 png16
            # 这里给一个通用归一化写法：按每张图 min/max 归一化（会丢失绝对尺度）
            for i in range(N):
                d = depths[i, :, :, 0].astype(np.float32)
                dmin, dmax = float(np.min(d)), float(np.max(d))
                if dmax <= dmin:
                    d_u16 = np.zeros_like(d, dtype=np.uint16)
                else:
                    dn = (d - dmin) / (dmax - dmin)
                    d_u16 = (dn * 65535.0 + 0.5).astype(np.uint16)
                Image.fromarray(d_u16).save(dep_dir / f"{i:05d}.png")
            print(f"[{scene_name}] depths(.png16, per-image normalized) -> {dep_dir} ({N} files)")
        else:
            raise ValueError(f"Unknown depth_format: {depth_format}")

    # -------- cameras --------
    if export_cam and (extr is not None or intr is not None):
        # 保存成 npy（也可以保存成 npz）
        if extr is not None:
            np.save(out_scene / "extrinsics.npy", extr.astype(np.float32))
        if intr is not None:
            np.save(out_scene / "intrinsics.npy", intr.astype(np.float32))

        meta = {
            "scene": scene_name,
            "num_frames": int(images.shape[0]) if images is not None else (
                int(depths.shape[0]) if depths is not None else None
            ),
            "images_shape": tuple(images.shape) if images is not None else None,
            "depths_shape": tuple(depths.shape) if depths is not None else None,
            "extrinsics_shape": tuple(extr.shape) if extr is not None else None,
            "intrinsics_shape": tuple(intr.shape) if intr is not None else None,
        }
        # 简单文本信息
        with open(out_scene / "meta.txt", "w") as f:
            for k, v in meta.items():
                f.write(f"{k}: {v}\n")

        print(f"[{scene_name}] cameras -> {out_scene} (extrinsics.npy / intrinsics.npy / meta.txt)")


def export_all(pkl_path: str, out_root: str = "mipnerf360_extracted",
               export_images=True, export_depths=True, export_cam=True,
               depth_format="npy"):
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    print(f"✅ Loaded: {pkl_path}")
    print(f"Scenes: {list(data.keys())}\n")

    for scene_name, scene_data in data.items():
        export_scene(
            scene_name, scene_data, out_root,
            export_images=export_images,
            export_depths=export_depths,
            export_cam=export_cam,
            depth_format=depth_format
        )

    print("\n[Done]")


if __name__ == "__main__":
    PKL_PATH = "/data/why/mipnerf360_datasets_preprocessed.pkl"

    export_all(
        pkl_path=PKL_PATH,
        out_root="/data/why/mipnerf360",
        export_images=True,
        export_depths=True,
        export_cam=True,
        depth_format="npy"   # 推荐：保留 float32 深度，不丢尺度
    )
