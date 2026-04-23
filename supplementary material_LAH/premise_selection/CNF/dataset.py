import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset, dataset
from torch.nn.functional import one_hot
import json
import re
import os
import networkx as nx

from utils import load_pickle_file, Statements, read_file
from formula_parser import fof_formula_transformer
from graph import Graph

VARIABLE_PATTERN = re.compile(r"[A-Z][A-Z0-9_]*")
FUNCTOR_PATTERN = re.compile(r"[a-z0-9][a-z0-9_]*")
CONNECTIVE_PATTERN = {"!", "?", "|", "&", "=>", "<=>", "~"}
BOOL = "$true"

class PairData(Data):
    def __init__(self, x_s=None, term_walk_index_s=None,
                 x_t=None, term_walk_index_t=None, y=None,
                 prem_graph=None, conj_graph=None):
        super().__init__()
        self.x_s = x_s if x_s is not None else torch.empty(0)
        self.x_t = x_t if x_t is not None else torch.empty(0)
        self.term_walk_index_s = term_walk_index_s if term_walk_index_s is not None else torch.empty(0)
        self.term_walk_index_t = term_walk_index_t if term_walk_index_t is not None else torch.empty(0)
        self.y = y if y is not None else torch.empty(0)

        self._prem_graph = prem_graph
        self._conj_graph = conj_graph

    @property
    def prem_graph(self):
        return self._prem_graph

    @property
    def conj_graph(self):
        return self._conj_graph

    def __inc__(self, key, value, *args, **kwargs):
        if key == "term_walk_index_s":
            return self.x_s.size(0) if self.x_s.numel() > 0 else 0
        if key == "term_walk_index_t":
            return self.x_t.size(0) if self.x_t.numel() > 0 else 0
        return super().__inc__(key, value, *args, **kwargs)

class FormulaGraphDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 data_class,
                 statements_file,
                 node_dict_file,
                 rename=True):
        self.root = root
        self.data_class = data_class
        self.statements = Statements(statements_file)
        self.rename = rename
        self.node_dict = load_pickle_file(node_dict_file)
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["{}.json".format(self.data_class)]

    @property
    def processed_file_names(self):
        return ["{}.pt".format(self.data_class)]

    def graph_process(self, G):
        nodes = []
        term_walk_indices = []

        for node in G:
            nodes.append(node.name)
            if node.parents and node.children:
                for parent in node.parents:
                    for child in node.children:
                        term_walk_indices.append([parent.id,
                                                  node.id,
                                                  child.id])

        term_walk_indices = np.array(
            term_walk_indices, dtype=np.int64).reshape(-1, 3).T

        return nodes, term_walk_indices

    def vectorization(self, objects, object_dict):
        indices = [object_dict[obj] for obj in objects]
        onehot = one_hot(torch.LongTensor(indices), len(object_dict)).float()
        return onehot

    def get_raw_examples(self):

        with open(os.path.join(self.raw_dir, self.raw_file_names[0]), 'r') as f:
            return [json.loads(line) for line in f]

    def process(self):
        raw_examples = [json.loads(line) for line in read_file(self.raw_paths[0])]
        data_list = []
        for example in raw_examples:
            conj, prem, label = example
            if not all([conj, prem, label is not None]):
                continue

            conj_graph = Graph(fof_formula_transformer(self.statements[conj]), rename=self.rename)
            prem_graph = Graph(fof_formula_transformer(self.statements[prem]), rename=self.rename)
            if len(conj_graph.graph) == 0 or len(prem_graph.graph) == 0:
                continue

            c_nodes, c_term_walk_indices = self.graph_process(conj_graph)
            p_nodes, p_term_walk_indices = self.graph_process(prem_graph)
            if not c_nodes or not p_nodes:
                continue

            data = PairData(
                x_s=self.vectorization(c_nodes, self.node_dict),
                term_walk_index_s=torch.from_numpy(c_term_walk_indices) if c_term_walk_indices.size > 0 else None,
                x_t=self.vectorization(p_nodes, self.node_dict),
                term_walk_index_t=torch.from_numpy(p_term_walk_indices) if p_term_walk_indices.size > 0 else None,
                y=torch.LongTensor([label])
            )

            data._prem_graph = prem_graph.to_dict()
            data._conj_graph = conj_graph.to_dict()
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def formula_graph_collate(self, data_list):

    data, slices, _ = super().collate(data_list)

    prem_graphs = [getattr(d, "_prem_graph", None) for d in data_list]
    conj_graphs = [getattr(d, "_conj_graph", None) for d in data_list]

    setattr(data, "prem_graphs", prem_graphs)
    setattr(data, "conj_graphs", conj_graphs)
    return data, slices
