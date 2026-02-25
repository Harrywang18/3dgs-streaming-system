scene="room"

python pre_processing/build_matrix_A.py \
    --ply_file_path /home/why/lapis-gs/model_freeze_opacity/360/room/freeze/${scene}_res2/point_cloud/iteration_30000/point_cloud.ply \
    --output_folder scenes/${scene} \
    --scene_name ${scene}