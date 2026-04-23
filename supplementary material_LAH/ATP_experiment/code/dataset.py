import torch
import os
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from torch.nn.functional import one_hot
import json

from formula_parser import fof_formula_transformer
from graph import Graph
from utils import read_file, load_pickle_file, Statements
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
        self['prem_graph'] = prem_graph
        self['conj_graph'] = conj_graph
    @property
    def prem_graph(self):
        return self['prem_graph']

    @property
    def conj_graph(self):
        return self['conj_graph']

    def __inc__(self, key, value, *args, **kwargs):
        if key == "term_walk_index_s":
            return self.x_s.size(0) if self.x_s.numel() > 0 else 0
        if key == "term_walk_index_t":
            return self.x_t.size(0) if self.x_t.numel() > 0 else 0
        else:
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
      dataList = []
      for example in raw_examples:
        conj, prem, label = example

        if not all([conj, prem, label is not None]):
            continue

        conj_graph = Graph(fof_formula_transformer(self.statements[conj]),
                           rename=self.rename)
        prem_graph = Graph(fof_formula_transformer(self.statements[prem]),
                           rename=self.rename)

        assert prem_graph is not None, f"Premise graph is none: {prem}"
        assert conj_graph is not None, f"Conjecture graph is none: {conj}"
        assert len(prem_graph.graph) > 0, f"Premise graph is none: {prem}"
        assert len(conj_graph.graph) > 0, f"Conjecture graph is none: {conj}"

        c_nodes, c_term_walk_indices = self.graph_process(conj_graph)
        p_nodes, p_term_walk_indices = self.graph_process(prem_graph) 

        if not c_nodes or not p_nodes:
            continue

        data = PairData(
            x_s=self.vectorization(c_nodes, self.node_dict),
            term_walk_index_s=torch.from_numpy(c_term_walk_indices) if c_term_walk_indices.size > 0 else None,
            x_t=self.vectorization(p_nodes, self.node_dict),
            term_walk_index_t=torch.from_numpy(p_term_walk_indices) if p_term_walk_indices.size > 0 else None,
            prem_graph=prem_graph,
            conj_graph=conj_graph,
            y=torch.LongTensor([label])
        )

        assert data.prem_graph is not None, f"Premise graph is None for {prem}"
        assert data.conj_graph is not None, f"Conjecture graph is None for {conj}"
        dataList.append(data)

      for data in dataList:
        assert 'prem_graph' in data._store, "Premise graph not stored correctly"
        assert 'conj_graph' in data._store, "Conjecture graph not stored correctly"
        assert data['prem_graph'] is not None, "Premise graph is None"
        assert data['conj_graph'] is not None, "Conjecture graph is None"

      data, slices = self.collate(data_list=dataList)
      torch.save((data, slices), self.processed_paths[0])

def formula_graph_collate(batch):

    prem_graphs = [data['prem_graph'] for data in batch]
    conj_graphs = [data['conj_graph'] for data in batch]

    data_list = [Data(
        x_s=data.x_s,
        x_t=data.x_t,
        term_walk_index_s=data.term_walk_index_s,
        term_walk_index_t=data.term_walk_index_t,
        y=data.y,
    ) for data in batch]

    pyg_batch = Batch.from_data_list(data_list, follow_batch=['x_s', 'x_t'])

    setattr(pyg_batch, 'prem_graphs', prem_graphs)
    setattr(pyg_batch, 'conj_graphs', conj_graphs)

    return pyg_batch
