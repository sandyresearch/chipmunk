# avoid OOMs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# python generate.py  --task t2v-1.3B --size "832*480" --ckpt_dir ./Wan2.1-T2V-1.3B --offload_model True --t5_cpu --sample_shift 8 --sample_guide_scale 6 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."

python generate.py  --task t2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-T2V-14B --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." --offload_model True --chipmunk-config chipmunk-config.yml