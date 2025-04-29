import os
import pandas as pd
import json
import yaml
model_dirs = os.listdir('evals/models')

results_per_exp = {}

for model_dir in model_dirs:
    if not os.path.isdir(os.path.join('evals/models', model_dir)):
        continue
    
    results_this_exp = []
    for exp_name in os.listdir(os.path.join('evals/models', model_dir)):
        exp_eval_log = os.path.join('evals/models', model_dir, exp_name, 'logs', 'gpu0.out')
        if not os.path.exists(exp_eval_log):
            continue
        
        config = os.path.join('evals/models', model_dir, exp_name, 'chipmunk-config.yml')
        if not os.path.exists(config):
            continue
        with open(config, 'r') as f:
            config = yaml.safe_load(f)

        imre_file = os.path.join('evals/models', model_dir, exp_name, 'imre.json')
        if not os.path.exists(imre_file):
            continue
        with open(imre_file, 'r') as f:
            imre_score = json.load(f)['avg_score']
        all_times = []
        for i in range(8):
            exp_gen_log = os.path.join('evals/models', model_dir, exp_name, 'logs', f'gpu{i}.out')
            if not os.path.exists(exp_gen_log):
                continue
                
            times = []
            with open(exp_gen_log, 'r') as f:
                for line in f:
                    try:
                        if ' in ' in line and 'finished' in line:
                            time_str = line.split(' in ')[1].split('s')[0].strip()
                            times.append(float(time_str))
                    except Exception as e:
                        print(f'Error reading generation times from {exp_gen_log}: {e}')
                        continue

            times.sort()
            times = times[:-3]
            all_times.extend(times)

        results_this_exp.append({ 
            'imre': imre_score*100,
            'dtype': 'fp8' if config.get('fp8', {}).get('is_enabled', False) else 'bf16',
            # we measured the amount of time it takes to write the file and all overhead, so we subtract that
            'time': round(sum(all_times) / len(all_times) - 0.44, 3),
            'mlp_tk': config['mlp']['top_keys'] if config['mlp']['is_enabled'] else 1,
            'attn_tk': config['attn']['top_keys'] if config['attn']['is_enabled'] else 1,
            'mlp_rk': config['mlp']['random_keys'] if config['mlp']['is_enabled'] else 0,
            'mlp_mbm': config['mlp']['mbm'] if config['mlp']['is_enabled'] else 1,
            'attn_full_step_every': config['attn']['full_step_every'] if config['attn']['is_enabled'] else 1,
            'attn_full_step_schedule': config['attn']['full_step_schedule'] if config['attn']['is_enabled'] else None,
            'attn_local_voxels': config['attn']['local_voxels'] if config['attn']['is_enabled'] else 0,
            'attn_local_1d_window': config['attn']['local_1d_window'] if config['attn']['is_enabled'] else 0,
            'attn_recompute_mask': config['attn']['recompute_mask'] if config['attn']['is_enabled'] else False,
            'mlp_full_step_every': config['mlp']['full_step_every'] if config['mlp']['is_enabled'] else 1,
            'mlp_block_mask_cache': config['mlp']['block_mask_cache'] if config['mlp']['is_enabled'] else 1,
            'step_caching': config['step_caching']['is_enabled'],
        })
    
    results_per_exp[model_dir] = results_this_exp

for model_dir, results in results_per_exp.items():
    df = pd.DataFrame(results)
    df.to_csv(f'evals/results/{model_dir}_imre_table.csv', index=False)
    print(f'found {len(results)} experiments for {model_dir}')