# for scene in "counter" "garden" "kitchen" "bonsai" "flowers" "treehill";
for scene in "garden" "bonsai" "flowers" "treehill";
do
    python train_full_pipeline.py \
        --model_base ./model_freeze_full_res_fewer_splats \
        --dataset_base ./source_360 \
        --dataset_name 360 \
        --scene $scene \
        --method freeze
done
