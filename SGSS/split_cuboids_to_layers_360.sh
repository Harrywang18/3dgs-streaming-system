scene="flowers"



python split_streaming_cuboids.py \
  --aabb_json scenes/${scene}/optimal_voxels/voxel_ilp.json \
  --streaming_dir scenes/${scene}/optimal_voxels \
  --l0 /home/why/lapis-gs/model_freeze_all_res_fewer_splats/360/${scene}/freeze/${scene}_res16/point_cloud/iteration_30000/point_cloud_nonan.ply \
  --l1 /home/why/lapis-gs/model_freeze_all_res_fewer_splats/360/${scene}/freeze/${scene}_res8/point_cloud/iteration_30000/point_cloud_nonan.ply \
  --l2 /home/why/lapis-gs/model_freeze_all_res_fewer_splats/360/${scene}/freeze/${scene}_res4/point_cloud/iteration_30000/point_cloud_nonan.ply \
  --l3 /home/why/lapis-gs/model_freeze_all_res_fewer_splats/360/${scene}/freeze/${scene}_res2/point_cloud/iteration_30000/point_cloud_nonan.ply \
  --pos_fields x,y,z \
  --output_scale original \
  --preserve_old_attrs


python assemble_full_layers.py \
  --layered_dir scenes/${scene}/layered_streaming_cuboids \
  --out_dir scenes/${scene}/assembled_full_scene



python split_cumulative_layers.py \
  --in_dir scenes/${scene}/layered_streaming_cuboids \
  --out_dir scenes/${scene}/layered_pure_streaming_cuboids


python pre_processing/streaming_cuboids.py \
    --ply_file_path /home/why/lapis-gs/model/360/${scene}/lapis/${scene}_res2/point_cloud/iteration_30000/point_cloud.ply \
    --method sgss \
    --output_folder scenes/${scene}/streaming_layered_cuboids \
    --scene_name ${scene} \
    --mode voxel \
    --input_folder scenes/${scene}/layered_pure_streaming_cuboids \
    --json_file scenes/${scene}/optimal_voxels/voxel_ilp.json


