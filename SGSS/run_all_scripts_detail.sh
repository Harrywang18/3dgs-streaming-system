scene="treehill"
# ply_file_path="/home/why/lapis-gs/model_freeze_all_res_fewer_splats/360/${scene}/freeze/${scene}_res2/point_cloud/iteration_30000/point_cloud_nonan.ply"
# camera_path="/home/why/lapis-gs/model_freeze_all_res_fewer_splats/360/${scene}/freeze/${scene}_res2/cameras.json"


camera_path="/home/why/lapis-gs/model_freeze_full_res_fewer_splats/360/${scene}/freeze/${scene}_res2/cameras.json"

for res in "2"; do
    python clean_ply_remove_nan.py \
        --input  /home/why/lapis-gs/model_freeze_full_res_fewer_splats/360/${scene}/freeze/${scene}_res${res}/point_cloud/iteration_30000/point_cloud.ply  \
        --output /home/why/lapis-gs/model_freeze_full_res_fewer_splats/360/${scene}/freeze/${scene}_res${res}/point_cloud/iteration_30000/point_cloud_nonan.ply
done

ply_file_path="/home/why/lapis-gs/model_freeze_full_res_fewer_splats/360/${scene}/freeze/${scene}_res2/point_cloud/iteration_30000/point_cloud_nonan.ply"

python pre_processing/voxel_gaussian.py \
    --ply_file ${ply_file_path} \
    --output_folder scenes/${scene} \
    --scene_name ${scene}

python pre_processing/build_matrix_A.py \
    --ply_file_path ${ply_file_path} \
    --output_folder scenes/${scene} \
    --scene_name ${scene}

python pre_processing/projection_model.py \
    --ply_file_path ${ply_file_path} \
    --cameras_path ${camera_path} \
    --matrix_a_path scenes/${scene}/matrix_A.npy \
    --c_store_path scenes/${scene}/C_cost.npy \
    --voxel_path scenes/${scene}/voxel_new.json \
    --output_folder scenes/${scene} \
    --scene_name ${scene}

tiles=$(ls scenes/${scene}/voxels_new | wc -l)

python pre_processing/run_gurobi_flow.py --scene ${scene} --tiles ${tiles}

python pre_processing/optimal_voxelization.py \
    --matrix_a_path scenes/${scene}/matrix_A.npy \
    --c_cost_path scenes/${scene}/C_cost.npy \
    --ply_file_path ${ply_file_path} \
    --x_solution_path scenes/${scene}/x_solution.npy \
    --output_folder scenes/${scene}/optimal_voxels \
    --scene_name ${scene}

python pre_processing/streaming_cuboids.py \
    --ply_file_path ${ply_file_path} \
    --method sgss \
    --output_folder scenes/${scene}/streaming_cuboids \
    --scene_name ${scene} \
    --mode voxel \
    --input_folder scenes/${scene}/optimal_voxels \
    --json_file scenes/${scene}/optimal_voxels/voxel_ilp.json


python experiment/cam_trace.py --scene_name ${scene} \
    --input_camera_path ${camera_path} \
    --output_folder scenes/${scene} \
    --mode full


python experiment/cam_trace.py --scene_name ${scene} \
    --input_camera_path ${camera_path} \
    --output_folder scenes/${scene} \
    --mode limit
