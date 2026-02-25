import numpy as np
import math
from plyfile import PlyData, PlyElement


def scale_gs_ply(in_ply, out_ply, s=2.0):
    ply = PlyData.read(in_ply)
    v = ply["vertex"]
    data = v.data

    # name -> column index
    name2i = {p.name: i for i, p in enumerate(v.properties)}

    # 必备字段检查
    for key in ["x", "y", "z"]:
        if key not in name2i:
            raise KeyError(f"PLY vertex missing field: {key}")

    # 位置放大
    data["x"] *= s
    data["y"] *= s
    data["z"] *= s

    # scale 字段：3DGS 通常是 log-scale: scale_0/1/2
    log_delta = math.log(s)
    scale_keys = [k for k in ["scale_0", "scale_1", "scale_2"] if k in data.dtype.names]

    if len(scale_keys) == 3:
        # 默认按 3DGS 官方：log-scale
        for k in scale_keys:
            data[k] += log_delta
    else:
        # 你的 ply 可能叫别的名字，打印出来方便你确认
        print("Warning: scale_0/1/2 not found. Available fields:")
        print(data.dtype.names)

    # 写回（保留原 dtype / 属性）
    out_vertex = PlyElement.describe(data, "vertex")
    PlyData([out_vertex], text=ply.text).write(out_ply)


if __name__ == "__main__":
    scale_gs_ply(
        in_ply="/home/why/dynamic-lapis-gs/model/8i/longdress/dynamic-lapis-freeze/longdress_res1/1051/point_cloud/iteration_30000/point_cloud.ply",
        out_ply="/home/why/dynamic-lapis-gs/model/8i/longdress/dynamic-lapis-freeze/longdress_res1/1051/point_cloud/iteration_30000/point_cloud_x8.ply",
        s=8.0
    )
