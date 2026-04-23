#!/usr/bin/env python3
import os, sys, json, shutil, pickle

TWGNN_DIR = '../TWGNN'
LAH_DIR = '.'
PROVER_DATA = os.path.join(TWGNN_DIR, 'prover_data')

datasets = {
    'D1': 'D1_hard_negative_mining.json',
    'D2': 'D2_contrastive_finetune.json',
    'D3a': 'D3a_expert_iteration_holist.json',
    'D3b': 'D3b_expert_iteration_gptf.json',
    'D4': 'D4_frequency_weighting.json',
    'D5': 'D5_hardneg_freqweight_combined.json',
}

STATEMENTS_PKL = os.path.join(TWGNN_DIR, 'data/mptp2078_custom/statements.pkl')
NODE_DICT_SRC = os.path.join(TWGNN_DIR, 'data/mptp2078_custom/node_dict.pkl')
TEST_PROBS_SRC = os.path.join(TWGNN_DIR, 'data/mptp2078_custom/test_problems_provable.json')

def pkl_statements_to_text(pkl_path, output_path):
    with open(pkl_path, 'rb') as f:
        stmts = pickle.load(f)
    with open(output_path, 'w') as f:
        for name, formula in sorted(stmts.items()):
            f.write(formula + '\n')
    return len(stmts)

def convert_dict_to_linejson(dict_json_path, output_path):
    with open(dict_json_path) as f:
        data = json.load(f)
    count = 0
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps([item['conjecture'], item['premise'], item['label']]) + '\n')
            count += 1
    return count

def main():
    print("=== Converting TWGNN prover_data for LAH_TWGNN ===\n")

    statements_text = os.path.join(LAH_DIR, 'statements')
    if not os.path.exists(statements_text):
        n = pkl_statements_to_text(STATEMENTS_PKL, statements_text)
        print(f"Generated statements: {n} formulas")
    else:
        print("Statements already exists")

    for src, name in [(NODE_DICT_SRC, 'node_dict.pkl'), (TEST_PROBS_SRC, 'test_problems_provable.json')]:
        dst = os.path.join(LAH_DIR, name)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"Copied {name}")

    for ds_name, src_file in datasets.items():
        print(f"\n--- {ds_name} ---")
        ds_dir = os.path.join(LAH_DIR, f'dataset_{ds_name}')
        train_raw = os.path.join(ds_dir, 'train', 'raw')
        os.makedirs(train_raw, exist_ok=True)

        src_path = os.path.join(PROVER_DATA, src_file)
        dst_path = os.path.join(train_raw, 'train.json')
        if os.path.exists(src_path):
            count = convert_dict_to_linejson(src_path, dst_path)
            print(f"  Converted {count} samples")
        else:
            print(f"  [SKIP] {src_path} not found")
            continue

        for f in ['statements', 'node_dict.pkl', 'test_problems_provable.json']:
            src = os.path.join(LAH_DIR, f)
            dst = os.path.join(ds_dir, f)
            if os.path.exists(src) and not os.path.exists(dst):
                os.symlink(src, dst)

    print("\n=== Done ===")

if __name__ == '__main__':
    main()
