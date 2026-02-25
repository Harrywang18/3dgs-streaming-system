for scene in "flowers" "treehill";
# for scene in "room";
do
    python train_full_pipeline.py \
        --model_base ./model_freeze_all_res \
        --dataset_base ./source_360 \
        --dataset_name 360 \
        --scene $scene \
        --method freeze
done
