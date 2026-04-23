#!/usr/bin/env python3
import os, sys, json, pickle, re, argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PremiseSelectionModel
from graph import Graph, Node
from formula_parser import fof_formula_transformer
from dataset import PairData
from utils import Statements
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

def find_problem_file(problem_name, problem_dir):
    match = re.match(r'[a-z]+\d*_(.+)', problem_name)
    if match:
        theory = match.group(1)
        candidate = os.path.join(problem_dir, f"{theory}__{problem_name}.p")
        if os.path.exists(candidate):
            return candidate
    for fn in os.listdir(problem_dir):
        if fn.endswith('.p') and problem_name in fn:
            return os.path.join(problem_dir, fn)
    return None

def parse_axioms_from_file(filepath):
    axioms = {}
    buffer = []
    with open(filepath) as f:
        for line in f:
            if '%' in line:
                line = line.split('%')[0]
            line = line.strip()
            if not line:
                continue
            buffer.append(line)
            if line.endswith('.'):
                full = ' '.join(buffer)
                buffer = []
                clean = full.replace(' ', '')
                if full.startswith('fof') and ',axiom,' in clean:
                    idx = full.find('(') + 1
                    name = full[idx:full.find(',', idx)].strip()
                    axioms[name] = full
    return axioms

def main():
    parser = argparse.ArgumentParser(description='LAH_TWGNN Premise Selection')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--statements', type=str, default='statements')
    parser.add_argument('--node_dict', type=str, default='node_dict.pkl')
    parser.add_argument('--chainy_dir', type=str, required=True)
    parser.add_argument('--test_problems', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--top_k', type=int, default=1024)
    parser.add_argument('--dim', type=int, default=512, help='node_out_channels')
    parser.add_argument('--layers', type=int, default=1, help='TW layers')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for inference')
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    with open(args.node_dict, 'rb') as f:
        node_dict = pickle.load(f)

    checkpoint = torch.load(args.model, map_location=args.device)
    model = PremiseSelectionModel(
        node_in_channels=len(node_dict),
        node_out_channels=args.dim,
        layers=args.layers
    ).to(args.device)

    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()
    print(f"Model loaded: {len(node_dict)} nodes, dim={args.dim}, layers={args.layers}")

    stmts = Statements(args.statements)
    print(f"Statements: {len(stmts)}")

    with open(args.test_problems) as f:
        test_probs = json.load(f)
    prob_names = sorted(test_probs.keys()) if isinstance(test_probs, dict) else sorted(test_probs)
    print(f"Test problems: {len(prob_names)}")

    print("Building graph cache...")
    graph_cache = {}
    for name in tqdm(stmts.statements.keys(), desc="Caching"):
        try:
            formula = stmts.statements[name]
            Node.rest_id()
            g = Graph(fof_formula_transformer(formula), rename=True)
            nodes = []
            tw_indices = []
            for node in g:
                nodes.append(node.name)
                if node.parents and node.children:
                    for parent in node.parents:
                        for child in node.children:
                            tw_indices.append([parent.id, node.id, child.id])
            tw = np.array(tw_indices, dtype=np.int64).reshape(-1, 3).T if tw_indices else np.zeros((3, 0), dtype=np.int64)
            indices = [node_dict[n] for n in nodes]
            x = F.one_hot(torch.LongTensor(indices), len(node_dict)).float()
            graph_cache[name] = (x, torch.from_numpy(tw))
        except Exception:
            pass
    print(f"Cached {len(graph_cache)} graphs")

    rankings = {}

    for prob_name in tqdm(prob_names, desc="Ranking"):
        pf = find_problem_file(prob_name, args.chainy_dir)
        if not pf or prob_name not in graph_cache:
            rankings[prob_name] = []
            continue

        conj_x, conj_tw = graph_cache[prob_name]
        local_axioms = parse_axioms_from_file(pf)

        valid_axioms = [ax for ax in local_axioms if ax in graph_cache]
        if not valid_axioms:
            rankings[prob_name] = []
            continue

        all_scores = []
        with torch.no_grad():
            for i in range(0, len(valid_axioms), args.batch_size):
                batch_axioms = valid_axioms[i:i + args.batch_size]
                pair_list = []
                for ax_name in batch_axioms:
                    prem_x, prem_tw = graph_cache[ax_name]
                    pair = PairData(
                        x_s=conj_x, term_walk_index_s=conj_tw,
                        x_t=prem_x, term_walk_index_t=prem_tw,
                        y=torch.LongTensor([0])
                    )
                    pair_list.append(pair)

                loader = DataLoader(pair_list, batch_size=len(pair_list),
                                    follow_batch=['x_s', 'x_t'])
                for batch in loader:
                    batch = batch.to(args.device)
                    h_s = model.initial(batch.x_s)
                    h_t = model.initial(batch.x_t)
                    h_s = model.dag_emb(h_s, batch.term_walk_index_s)
                    h_t = model.dag_emb(h_t, batch.term_walk_index_t)

                    h_g_s = global_mean_pool(h_s, batch.x_s_batch)
                    h_g_t = global_mean_pool(h_t, batch.x_t_batch)

                    x_concat = torch.cat([h_g_s, h_g_t], dim=1)
                    pred_y = model.classifier.classifier(x_concat)
                    probs = F.softmax(pred_y, dim=1)[:, 1]
                    all_scores.extend(list(zip(batch_axioms, probs.cpu().tolist())))

        all_scores.sort(key=lambda x: x[1], reverse=True)
        rankings[prob_name] = [name for name, _ in all_scores[:args.top_k]]

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(rankings, f, indent=2)

    print(f"\nSaved: {args.output}")
    print(f"Problems: {len(rankings)}")
    if rankings:
        print(f"Avg premises/problem: {sum(len(v) for v in rankings.values()) / len(rankings):.0f}")

if __name__ == '__main__':
    main()
