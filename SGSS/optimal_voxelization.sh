scene="room"

python pre_processing/optimal_voxelization.py \
    --matrix_a_path scenes/${scene}/matrix_A.npy \
    --c_cost_path scenes/${scene}/C_cost.npy \
    --ply_file_path /home/why/lapis-gs/model_freeze_opacity/360/room/freeze/${scene}_res2/point_cloud/iteration_30000/point_cloud.ply \
    --x_solution_path scenes/${scene}/x_solution.npy \
    --output_folder scenes/${scene}/optimal_voxels \
    --scene_name ${scene}

