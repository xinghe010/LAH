import re
import matplotlib.pyplot as plt
import networkx as nx

from formula_parser import fof_formula_transformer

NEGATIVE_CONNECTIVE = {"~"}
BINARY_CONNECTIVE = {"=>", "<=>"}
BINARY_CONNECTIVE1 = {"=>"}
ASSOC_CONNECTIVE = {"|", "&"}
QUANTIFIER = {"!", "?"}
EQUAL = {"=", "!="}
BOOL = {"$true"}
VARIABLE_PATTERN = re.compile(r"[A-Z][A-Z0-9_]*")
FUNCTOR_PATTERN = re.compile(r"[a-z0-9][a-z0-9_]*")

class Node:
    def __init__(self, name):
        self.id = self.__class__.id
        self.name = name
        self.children = []
        self.parents = []
        self.quantifer = None
        self.quantified_variable = []
        self.scoped = []
        self.__class__.id += 1
        self.time = 0

    @classmethod
    def rest_id(cls):
        cls.id = 0

    def __reduce__(self):
        return (self.__class__, (self.name,), {
            'id': self.id,
            'children': [child.id for child in self.children],
            'parents': [parent.id for parent in self.parents],
            'quantifer_id': self.quantifer.id if self.quantifer else None,
            'quantified_variable_ids': [v.id for v in self.quantified_variable],
            'scoped_ids': [s.id for s in self.scoped],
            'time': self.time
        })

    def __setstate__(self, state):
        self.id = state['id']
        self.name = state['name']
        self.children = state['children']
        self.parents = state['parents']
        self.quantifer = state['quantifer_id']
        self.quantified_variable = state['quantified_variable_ids']
        self.scoped = state['scoped_ids']
        self.time = state['time']

    def __str__(self):
        extra = ''
        if re.match(VARIABLE_PATTERN, self.name):
            extra = 'quantifer {}'.format(
                self.quantifer.id if self.quantifer is not None else '')
        elif self.name in QUANTIFIER:
            extra = 'quantified variable {}'.format(
                ' '.join(str(x.id) for x in self.quantified_variable))

        parents_info = ' '.join([str(x.id) for x in self.parents])
        children_info = ' '.join([str(x.id) for x in self.children])
        scoped_info = ' '.join(([str(x.id) for x in self.scoped]))
        return '<{}>: token {} | parents {} | children {} | scoped {} | {}'.\
            format(
                self.id,
                self.name,
                parents_info,
                children_info,
                scoped_info,
                extra)

    def __repr__(self):
        return self.__str__()

class Graph:
    def __init__(self, formula, rename=True):
        self.nodes = {} 
        self.graph = []
        self.id2subterm = dict()
        self.convert(formula, rename)
        self.rename = rename

    def __iter__(self):
        return self.graph.__iter__()

    def __getitem__(self, index):
        return self.graph[index]

    def __len__(self):
        return len(self.graph)

    def rebuild_node_references(self):
        for node in self.graph:
            node.children = [self.get_node_by_id(cid) for cid in node.children]
            node.parents = [self.get_node_by_id(pid) for pid in node.parents]
            if node.quantifer_id:
                node.quantifer = self.get_node_by_id(node.quantifer_id)
            node.quantified_variable = [self.get_node_by_id(vid) for vid in node.quantified_variable_ids]
            node.scoped = [self.get_node_by_id(sid) for sid in node.scoped_ids]

    def to_networkx(self):
        G = nx.DiGraph()
        for node in self.graph:
            for child in node.children:
                G.add_edge(node.id, child.id)
        return G

    def get_node_by_id(self, node_id):
        return next((n for n in self.graph if n.id == node_id), None)

    def to_dict(self):
        return {
            'nodes': [{
                'id': node.id,
                'name': node.name,
                'children': [c.id for c in node.children],
                'parents': [p.id for p in node.parents],
                'quantifer': node.quantifer.id if node.quantifer else None,
                'quantified_variable': [v.id for v in node.quantified_variable],
                'scoped': [s.id for s in node.scoped]
            } for node in self.graph],
            'id2subterm': {k: (v[0], v[1].id) for k, v in self.id2subterm.items()}
        }

    @classmethod
    def from_dict(cls, data):
        graph = cls.__new__(cls)
        graph.graph = []
        graph.id2subterm = {}

        id_to_node = {}
        for node_data in data['nodes']:
            node = Node(node_data['name'])
            node.id = node_data['id']
            id_to_node[node.id] = node
            graph.graph.append(node)

        for node_data in data['nodes']:
            node = id_to_node[node_data['id']]
            node.children = [id_to_node[cid] for cid in node_data['children']]
            node.parents = [id_to_node[pid] for pid in node_data['parents']]
            if node_data['quantifer']:
                node.quantifer = id_to_node[node_data['quantifer']]
            node.quantified_variable = [id_to_node[vid] for vid in node_data['quantified_variable']]
            node.scoped = [id_to_node[sid] for sid in node_data['scoped']]

        for subterm_id, (formula, node_id) in data['id2subterm'].items():
            graph.id2subterm[subterm_id] = (formula, id_to_node[node_id])

        return graph

    def create_quantifer_node(self, name, parent):
        quantifer_node = Node(name)
        if parent:
            quantifer_node.parents.append(parent)
            quantifer_node.scoped.extend(parent.scoped)
            if parent.name in QUANTIFIER:
                quantifer_node.scoped.append(parent)

        self.graph.append(quantifer_node)
        return quantifer_node

    def create_functor_node(self, name, parent):
        functor_node = Node(name)
        self.graph.append(functor_node)
        if parent:
            functor_node.parents.append(parent)
            for node in parent.scoped:
                if node not in functor_node.scoped:
                    functor_node.scoped.append(node)
            if parent.name in QUANTIFIER:
                functor_node.scoped.append(parent)

        return functor_node

    def create_variable_node(self, name, parent):
        assert parent is not None, \
            "a variable should have at least one parent"

        variable_node = Node(name)
        variable_node.quantifer = parent
        variable_node.scoped.extend(parent.scoped)
        variable_node.scoped.append(parent)
        parent.quantified_variable.append(variable_node)
        if variable_node not in self.graph:
            self.graph.append(variable_node)
        else:
            for node in parent.scoped:
                for child in node.children:
                    if child.name == name:
                        variable_node = child
                        break
        variable_node.parents.append(parent)
        return variable_node

    def create_constant_node(self, name, parent):
        exsiting_same_constant_nodes = [
            node for node in self.graph if node.name == name]
        if len(exsiting_same_constant_nodes) == 1 and \
                exsiting_same_constant_nodes[0].children == []:
            constant_node = exsiting_same_constant_nodes[0]
        else:
            constant_node = Node(name)
            self.graph.append(constant_node)
        if parent:
            constant_node.parents.append(parent)
            for node in parent.scoped:
                if node not in constant_node.scoped:
                    constant_node.scoped.append(node)
        return constant_node

    def create_connective_node(self, name, parent):
        connective_node = Node(name)
        if parent:
            connective_node.parents.append(parent)
            connective_node.scoped.extend(parent.scoped)
            if parent.name in QUANTIFIER:
                connective_node.scoped.append(parent)
        self.graph.append(connective_node)
        return connective_node

    def create_negative_node(self, name, parent):
        negative_node = Node(name)
        if parent:
            negative_node.parents.append(parent)
            negative_node.scoped.extend(parent.scoped)
            if parent.name in QUANTIFIER:
                negative_node.scoped.append(parent)
        self.graph.append(negative_node)
        return negative_node

    def merge_sub(self, formula, parent):
        for id in self.id2subterm:
            if self.id2subterm[id][0] == formula:
                pre_node = self.id2subterm[id][1]
                pre_node.parents.append(parent)
                return pre_node

    def check_merge(self, formula, parent):
        if formula in [self.id2subterm[id][0] for id in self.id2subterm]:
            for id in self.id2subterm:
                if self.id2subterm[id][0] == formula:
                    pre_node = self.id2subterm[id][1]
                    var_flag = self.check_variable(pre_node)
                    if not var_flag:
                        return True
                    else:
                        if set(pre_node.scoped).issubset(set(parent.scoped)):
                            return True
                        else:
                            return False

    def check_variable(self, node):
        if re.match(VARIABLE_PATTERN, node.name):
            return True
        else:
            if not node.children:
                return False
            else:
                for child in node.children:
                    flag = self.check_variable(child)
                    if flag:
                        return True
                else:
                    return False

    def formula_to_dense_graph(self, formula, parent=None):
        number = 0
        if isinstance(formula, str):
            if re.match(VARIABLE_PATTERN, formula):
                variable_node = self.create_variable_node(formula, parent)
                return variable_node
            if re.match(FUNCTOR_PATTERN, formula) or (formula in BOOL):
                constant_node = self.create_constant_node(formula, parent)
                return constant_node

        if isinstance(formula[0], str) and \
                formula[0] in QUANTIFIER and len(formula) == 3:
            if self.check_merge(formula, parent):
                return self.merge_sub(formula, parent)
            else:
                if number == 0:
                    quantifer_node = self.create_quantifer_node(formula[0], parent)
                    self.id2subterm[quantifer_node.id] = (formula, quantifer_node)
                    for variable in formula[1]:
                        variable_node = self.create_variable_node(
                            variable, quantifer_node)
                        quantifer_node.children.append(variable_node)
                if formula[2][0] is not None:
                    if formula[2][0] == formula[0]:
                        number = 1
                        for variable in formula[2][1]:
                            variable_node = self.create_variable_node(
                                variable, quantifer_node)
                            quantifer_node.children.append(variable_node)
                        if formula[2][2][0] != formula[2][0]:
                            node = self.formula_to_dense_graph(formula[2][2], quantifer_node)
                        if formula[2][2][0] == formula[2][0]:
                            for variable in formula[2][2][1]:
                                variable_node = self.create_variable_node(
                                    variable, quantifer_node)
                                quantifer_node.children.append(variable_node)
                            if formula[2][2][2][0] != formula[2][2][0]:
                                node = self.formula_to_dense_graph(formula[2][2][2],
                                                                   quantifer_node)
                            if formula[2][2][2][0] == formula[2][2][0]:
                                number = 1
                                for variable in formula[2][2][2][1]:
                                    variable_node = self.create_variable_node(
                                        variable, quantifer_node)
                                    quantifer_node.children.append(variable_node)
                                if formula[2][2][2][2][0] != formula[2][2][2][0]:
                                    node = self.formula_to_dense_graph(formula[2][2][2][2],
                                                                       quantifer_node)
                                if formula[2][2][2][2][0] == formula[2][2][2][0]:
                                    number = 1
                                    for variable in formula[2][2][2][2][1]:
                                        variable_node = self.create_variable_node(
                                            variable, quantifer_node)
                                        quantifer_node.children.append(variable_node)
                                    if formula[2][2][2][2][2][0] != formula[2][2][2][2][0]:
                                        node = self.formula_to_dense_graph(formula[2][2][2][2][2],
                                                                           quantifer_node)
                                    if formula[2][2][2][2][2][0] == formula[2][2][2][2][0]:
                                        number = 1
                                        for variable in formula[2][2][2][2][2][1]:
                                            variable_node = self.create_variable_node(
                                                variable, quantifer_node)
                                            quantifer_node.children.append(variable_node)
                                        if formula[2][2][2][2][2][2][0] != formula[2][2][2][2][2][0]:
                                            node = self.formula_to_dense_graph(formula[2][2][2][2][2][2],
                                                                               quantifer_node)
                                        if formula[2][2][2][2][2][2][0] == formula[2][2][2][2][2][0]:
                                            number = 1
                                            for variable in formula[2][2][2][2][2][2][1]:
                                                variable_node = self.create_variable_node(
                                                    variable, quantifer_node)
                                                quantifer_node.children.append(variable_node)
                                            if formula[2][2][2][2][2][2][2][0] != formula[2][2][2][2][2][2][0]:
                                                node = self.formula_to_dense_graph(formula[2][2][2][2][2][2][2],
                                                                                   quantifer_node)
                                            if formula[2][2][2][2][2][2][2][0] == formula[2][2][2][2][2][2][0]:
                                                number = 1
                                                for variable in formula[2][2][2][2][2][2][2][1]:
                                                    variable_node = self.create_variable_node(
                                                        variable, quantifer_node)
                                                    quantifer_node.children.append(variable_node)
                                                if formula[2][2][2][2][2][2][2][2][0] != formula[2][2][2][2][2][2][2][
                                                    0]:
                                                    node = self.formula_to_dense_graph(formula[2][2][2][2][2][2][2][2],
                                                                                       quantifer_node)
                                                if formula[2][2][2][2][2][2][2][2][0] == formula[2][2][2][2][2][2][2][
                                                    0]:
                                                    number = 1
                                                    for variable in formula[2][2][2][2][2][2][2][2][1]:
                                                        variable_node = self.create_variable_node(
                                                            variable, quantifer_node)
                                                        quantifer_node.children.append(variable_node)
                                                    if formula[2][2][2][2][2][2][2][2][2][0] != \
                                                            formula[2][2][2][2][2][2][2][2][0]:
                                                        node = self.formula_to_dense_graph(
                                                            formula[2][2][2][2][2][2][2][2][2],
                                                            quantifer_node)
                                                    if formula[2][2][2][2][2][2][2][2][2][0] == \
                                                            formula[2][2][2][2][2][2][2][2][0]:
                                                        number = 1
                                                        for variable in formula[2][2][2][2][2][2][2][2][2][1]:
                                                            variable_node = self.create_variable_node(
                                                                variable, quantifer_node)
                                                            quantifer_node.children.append(
                                                                variable_node)
                                                        if formula[2][2][2][2][2][2][2][2][2][2][0] != \
                                                                formula[2][2][2][2][2][2][2][2][2][0]:
                                                            node = self.formula_to_dense_graph(
                                                                formula[2][2][2][2][2][2][2][2][2][2],
                                                                quantifer_node)
                                                        if formula[2][2][2][2][2][2][2][2][2][2][0] == \
                                                                formula[2][2][2][2][2][2][2][2][2][0]:
                                                            number = 1
                                                            for variable in formula[2][2][2][2][2][2][2][2][2][2][1]:
                                                                variable_node = self.create_variable_node(variable,
                                                                                                          quantifer_node)
                                                                quantifer_node.children.append(
                                                                    variable_node)
                                                            node = self.formula_to_dense_graph(
                                                                formula[2][2][2][2][2][2][2][2][2][2][2],
                                                                quantifer_node)
                    else:
                        number = 0
                        node = self.formula_to_dense_graph(formula[2], quantifer_node)
                    quantifer_node.children.append(node)
                return quantifer_node

        if isinstance(formula[0], str) and \
                formula[0] in NEGATIVE_CONNECTIVE and len(formula) == 2:
            if self.check_merge(formula, parent):
                return self.merge_sub(formula, parent)
            else:
                negative_node = self.create_negative_node(formula[0], parent)
                self.id2subterm[negative_node.id] = (formula, negative_node)

                node = self.formula_to_dense_graph(formula[1], negative_node)
                negative_node.children.append(node)
                return negative_node

        if isinstance(formula[1], str) and \
                formula[1] in (BINARY_CONNECTIVE | ASSOC_CONNECTIVE | EQUAL) \
                and len(formula) == 3:
            if self.check_merge(formula, parent):
                return self.merge_sub(formula, parent)
            else:
                connective_node = self.create_connective_node(
                    formula[1], parent)
                self.id2subterm[connective_node.id] = (
                    formula, connective_node)
                left_node = self.formula_to_dense_graph(
                    formula[0], connective_node)
                connective_node.children.append(left_node)
                right_node = self.formula_to_dense_graph(
                    formula[2], connective_node)
                connective_node.children.append(right_node)
                return connective_node

        if isinstance(formula[0], str) and \
                re.match(FUNCTOR_PATTERN, formula[0]) and len(formula) == 2:
            if self.check_merge(formula, parent):
                return self.merge_sub(formula, parent)
            else:
                functor_node = self.create_functor_node(formula[0], parent)
                self.id2subterm[functor_node.id] = (formula, functor_node)
                for argument in formula[1]:
                    argument_node = self.formula_to_dense_graph(
                        argument, functor_node)
                    functor_node.children.append(argument_node)
                return functor_node

    def convert(self, formula, rename):
        Node.rest_id()
        self.formula_to_dense_graph(formula)

        function_nodes = [node for node in self.graph if re.match(FUNCTOR_PATTERN, node.name)]

        quantifer_nodes = [node for node in self.graph if node.name in QUANTIFIER]
        variable_nodes = [node for node in self.graph if re.match(VARIABLE_PATTERN, node.name)]

        if rename:
            variable_nodes = [node for node in self.graph if re.match(
               VARIABLE_PATTERN, node.name)]
            for node in variable_nodes:
                node.name = 'VAR'
        return self.graph

def Draw(graph):
    MG = nx.MultiDiGraph()
    for node in graph.graph:
        for child in node.children:
            MG.add_edge(node.id, child.id)
    nx.draw_networkx(MG, pos=nx.nx_agraph.graphviz_layout(
        MG, root=0), with_labels=True)
    plt.show()
    return MG
