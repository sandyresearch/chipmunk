export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
printf '%s\n' "A person is eating a big fat orange cat for lunch sitting on a table." 831 |
python3 sample_video.py --video-size 544 960 --video-length 129 --infer-steps 50 --flow-reverse --chipmunk-config ./chipmunk-config.yml
# python3 sample_video.py --flow-reverse --chipmunk-config ./chipmunk-config.yml 