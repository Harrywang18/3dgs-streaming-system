scene="room"

python pre_processing/projection_model.py \
    --ply_file_path /home/why/lapis-gs/model_freeze_opacity/360/room/freeze/${scene}_res2/point_cloud/iteration_30000/point_cloud.ply \
    --cameras_path /home/why/lapis-gs/model_freeze_opacity/360/room/freeze/${scene}_res2/cameras.json \
    --matrix_a_path scenes/${scene}/matrix_A.npy \
    --c_store_path scenes/${scene}/C_cost.npy \
    --voxel_path scenes/${scene}/voxel_new.json \
    --output_folder scenes/${scene} \
    --scene_name ${scene}