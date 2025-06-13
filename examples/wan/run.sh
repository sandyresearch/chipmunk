CONFIG_DIR=configs/cuda-test
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python generate.py \
--task t2v-14B \
--size 832*480 \
--ckpt_dir ./Wan2.1-T2V-14B \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
--base_seed 42 \
--offload_model True \
--chipmunk-config ${CONFIG_DIR}/chipmunk-config.yml \
--output-dir ${CONFIG_DIR}/media
