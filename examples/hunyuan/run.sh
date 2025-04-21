WORLD_SIZE=1

python3 sample_video.py \
    --video-size 720 1280 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "a bear catching a salmon in the river" \
    --seed 786966 \
    --embedded-cfg-scale 6.0 \
    --flow-shift 7.0 \
    --flow-reverse \
    --ulysses-degree "$WORLD_SIZE" \
    --ring-degree 1