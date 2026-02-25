for scene in "lego" "chair" "drums" "ficus" "hotdog" "materials" "mic" "ship";
do
    for res in 1 2 4 8;
    do
        python render.py -m ./model/nerf_synthetic/$scene/lapis/${scene}_res$res --eval
    done
done
