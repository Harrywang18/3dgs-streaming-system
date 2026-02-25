scene="room"

python pre_processing/streaming_cuboids.py \
    --ply_file_path /home/why/lapis-gs/model_freeze_opacity/360/room/freeze/${scene}_res2/point_cloud/iteration_30000/point_cloud.ply \
    --method sgss \
    --output_folder scenes/${scene}/streaming_cuboids \
    --scene_name ${scene} \
    --mode voxel \
    --input_folder scenes/${scene}/optimal_voxels \
    --json_file scenes/${scene}/optimal_voxels/voxel_ilp.json