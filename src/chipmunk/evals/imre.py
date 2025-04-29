import argparse
import json
import os
import torch
import ImageReward as RM
from tqdm import tqdm

# Set HF_HOME if needed
if os.path.exists('/workspace/.hf-home') and os.environ.get('HF_HOME') is None:
    os.environ['HF_HOME'] = '/workspace/.hf-home'

prompts = json.load(open('evals/prompts/imre.json', 'r'))

def eval(folder: str, imre_json_result_path: str):
    model = RM.load("ImageReward-v1.0")
    image_count = len([x for x in os.listdir(folder + '/imre') if x.endswith('.png')])
    score = 0
    ascore = 0
    count = 0
    data = {}
    for i in range(0, len(prompts), 4):
        for generation in prompts[i:i+4]:
            max_score = 0
            avg_score = 0
            out_path = os.path.join(folder, generation['output_path'])
            if not os.path.exists(out_path):
                print(f'missing: {out_path}')
                continue
            with torch.no_grad():
                seed_score = model.score(generation['prompt'], out_path)

                avg_score += seed_score
                if seed_score > max_score:
                    max_score = seed_score
            score += max_score
            ascore += (avg_score / image_count)
            count += 1
    if count > 0:
        data['avg_score'] = ascore
        print('score', ascore)
        with open(imre_json_result_path, 'w') as f:
            json.dump(data, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=str, required=True)
    parser.add_argument("--out-path", type=str, required=True)
    args = parser.parse_args()
    eval(args.experiment_dir, args.out_path)

if __name__ == "__main__":
    main()
