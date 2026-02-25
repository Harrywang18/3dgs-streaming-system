import numpy as np
from plyfile import PlyData, PlyElement
import argparse
import os


def clean_ply_remove_nan(ply_path, out_path):
    """
    Remove any Gaussian (row) that contains NaN or Inf in ANY float attribute.
    """
    plydata = PlyData.read(ply_path)

    assert len(plydata.elements) == 1, "Only support single-element PLY (vertex)"
    elem = plydata.elements[0]
    data = elem.data

    names = data.dtype.names
    N = len(data)

    # start with all valid
    valid = np.ones(N, dtype=bool)

    for k in names:
        v = data[k]
        if np.issubdtype(v.dtype, np.floating):
            valid &= np.isfinite(v)

    removed = N - valid.sum()

    print(f"[Clean PLY]")
    print(f"  input  : {ply_path}")
    print(f"  output : {out_path}")
    print(f"  total  : {N}")
    print(f"  kept   : {valid.sum()}")
    print(f"  removed: {removed}")

    clean_data = data[valid]

    # rebuild ply
    clean_elem = PlyElement.describe(clean_data, elem.name)
    PlyData([clean_elem], text=plydata.text).write(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input ply path")
    parser.add_argument("--output", type=str, required=True, help="output clean ply path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    clean_ply_remove_nan(args.input, args.output)


if __name__ == "__main__":
    main()

'''

# python clean_ply_remove_nan.py \
#     --input  /home/why/lapis-gs/model_freeze_opacity/360/room/freeze/room_res2/point_cloud/iteration_30000/point_cloud.ply \
#     --output /home/why/lapis-gs/model_freeze_opacity/360/room/freeze/room_res2/point_cloud/iteration_30000/point_cloud.ply

# python clean_ply_remove_nan.py \
#     --input  /home/why/lapis-gs/model_freeze_opacity/360/room/freeze/room_res4/point_cloud/iteration_30000/point_cloud.ply \
#     --output /home/why/lapis-gs/model_freeze_opacity/360/room/freeze/room_res4/point_cloud/iteration_30000/point_cloud.ply

# python clean_ply_remove_nan.py \
#     --input  /home/why/lapis-gs/model_freeze_opacity/360/room/freeze/room_res8/point_cloud/iteration_30000/point_cloud.ply \
#     --output /home/why/lapis-gs/model_freeze_opacity/360/room/freeze/room_res8/point_cloud/iteration_30000/point_cloud.ply

# python clean_ply_remove_nan.py \
#     --input  /home/why/lapis-gs/model_freeze_opacity/360/room/freeze/room_res16/point_cloud/iteration_30000/point_cloud.ply \
#     --output /home/why/lapis-gs/model_freeze_opacity/360/room/freeze/room_res16/point_cloud/iteration_30000/point_cloud.ply


python clean_ply_remove_nan.py \
    --input  /home/why/lapis-gs/model_freeze_all_res_fewer_splats/360/room/freeze/room_res2/point_cloud/iteration_30000/point_cloud.ply  \
    --output /home/why/lapis-gs/model_freeze_all_res_fewer_splats/360/room/freeze/room_res2/point_cloud/iteration_30000/point_cloud_nonan.ply

python clean_ply_remove_nan.py \
    --input  /home/why/lapis-gs/model_freeze_all_res_fewer_splats/360/room/freeze/room_res4/point_cloud/iteration_30000/point_cloud.ply  \
    --output /home/why/lapis-gs/model_freeze_all_res_fewer_splats/360/room/freeze/room_res4/point_cloud/iteration_30000/point_cloud_nonan.ply

python clean_ply_remove_nan.py \
    --input  /home/why/lapis-gs/model_freeze_all_res_fewer_splats/360/room/freeze/room_res8/point_cloud/iteration_30000/point_cloud.ply  \
    --output /home/why/lapis-gs/model_freeze_all_res_fewer_splats/360/room/freeze/room_res8/point_cloud/iteration_30000/point_cloud_nonan.ply

python clean_ply_remove_nan.py \
    --input  /home/why/lapis-gs/model_freeze_all_res_fewer_splats/360/room/freeze/room_res16/point_cloud/iteration_30000/point_cloud.ply  \
    --output /home/why/lapis-gs/model_freeze_all_res_fewer_splats/360/room/freeze/room_res16/point_cloud/iteration_30000/point_cloud_nonan.ply

python clean_ply_remove_nan.py \
    --input  /home/why/lapis-gs/model_freeze_all_res_fewer_splats/360/room/freeze/room_res1/point_cloud/iteration_30000/point_cloud.ply  \
    --output /home/why/lapis-gs/model_freeze_all_res_fewer_splats/360/room/freeze/room_res1/point_cloud/iteration_30000/point_cloud_nonan.ply


python clean_ply_remove_nan.py \
    --input  /home/why/lapis-gs/model_freeze_all_res_fewer_splats/360/garden/freeze/garden_res8/point_cloud/iteration_30000/point_cloud.ply  \
    --output /home/why/lapis-gs/model_freeze_all_res_fewer_splats/360/garden/freeze/garden_res8/point_cloud/iteration_30000/point_cloud_nonan.ply

python clean_ply_remove_nan.py \
    --input  /home/why/lapis-gs/model_freeze_all_res_fewer_splats/360/garden/freeze/garden_res16/point_cloud/iteration_30000/point_cloud.ply  \
    --output /home/why/lapis-gs/model_freeze_all_res_fewer_splats/360/garden/freeze/garden_res16/point_cloud/iteration_30000/point_cloud_nonan.ply

python clean_ply_remove_nan.py \
    --input  /home/why/lapis-gs/model_freeze_all_res_fewer_splats/360/garden/freeze/garden_res4/point_cloud/iteration_30000/point_cloud.ply  \
    --output /home/why/lapis-gs/model_freeze_all_res_fewer_splats/360/garden/freeze/garden_res4/point_cloud/iteration_30000/point_cloud_nonan.ply

python clean_ply_remove_nan.py \
    --input  /home/why/lapis-gs/model_freeze_all_res_fewer_splats/360/garden/freeze/garden_res2/point_cloud/iteration_30000/point_cloud.ply  \
    --output /home/why/lapis-gs/model_freeze_all_res_fewer_splats/360/garden/freeze/garden_res2/point_cloud/iteration_30000/point_cloud_nonan.ply


'''