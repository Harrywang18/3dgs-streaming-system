scene="room_layered_res16_8_4_2"


camera_path_org="/home/why/lapis-gs/model_freeze_opacity/360/room/freeze/room_res2/cameras.json"
camera_path_sgss="/data/why/SGSS_scenes/room/cameras.json"

python experiment/cam_trace.py --scene_name ${scene} \
    --input_camera_path ${camera_path_org} \
     --output_folder scenes/${scene} \
      --mode full


python experiment/cam_trace.py --scene_name ${scene} \
    --input_camera_path ${camera_path_sgss} \
     --output_folder scenes/${scene} \
      --mode limit