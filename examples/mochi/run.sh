export HF_HOME=/data/shared/huggingface


MODEL_DIR=/data/austin/mochi
OUT_DIR=/data/austin/results/mochi

CHIPMUNK_ATTENTION=1 \
RAY_DEDUP_LOGS=0 \
COMPILE_DIT=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2 \
python3 demos/cli.py --model_dir ${MODEL_DIR} --out_dir ${OUT_DIR} --cpu_offload