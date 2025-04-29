prompts=(
    "anthropomorphic crow werecreature, photograph captured in a forest"
    "a concept art of a vehicle, cyberpunk"
    "astronaut drifting afloat in space, in the darkness away from anyone else, alone, black background dotted with stars, realistic"
    "photo of a interior taken with a cheap digital camera at night flash lighting"
    "A realistic photo of a man with big ears"
    "delicious plate of food"
    "tumultuous plunging waves, anime, artwork, studio ghibli, stylized, in an anime format"
    "an alien planet viewed from space, extremely, beautiful, dynamic, creative, cinematic"
)


# Largest image size on fp8: 2560x1920

rm -rf output
python -m flux.cli --name flux-dev \
    --prompt "${prompts[0]}|${prompts[1]}|${prompts[2]}|${prompts[3]}|${prompts[4]}|${prompts[5]}|${prompts[6]}|${prompts[7]}" \
    --output_dir output \
    --width 1360 \
    --height 768 \
    --num_steps 50 \
    --seed 42