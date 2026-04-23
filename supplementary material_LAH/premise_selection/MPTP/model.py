import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from torch_scatter import scatter_mean
from torch_geometric.nn import global_mean_pool
from LibMTL.weighting import MGDA 
from collections import defaultdict

class MLPBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 activation=F.relu,
                 bias=True,
                 batch=False,
                 drop=False):
        super(MLPBlock, self).__init__()
        self.activation = activation
        self.bias = bias
        self.batch = batch
        self.drop = drop
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)
        if batch:
            self.BN = nn.BatchNorm1d(out_channels)
        if self.drop:
            self.drop = nn.Dropout(0.7)
        self.reset_parameters()

    def reset_parameters(self):
        if self.activation == F.relu:
            nn.init.kaiming_normal_(self.lin.weight, nonlinearity="relu")
        elif self.activation == F.leaky_relu:
            nn.init.kaiming_normal_(self.lin.weight)
        else:
            nn.init.xavier_normal_(self.lin.weight)
        if self.bias:
            nn.init.zeros_(self.lin.bias)

    def forward(self, x):
        x = self.lin(x)
        if self.batch and x.size()[0] > 1:
            x = self.BN(x)
        if self.drop:
            x = self.drop(x)
        x = self.activation(x)
        return x

class Initialization(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Initialization, self).__init__()
        self.embedding = nn.Embedding(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.embedding.weight)

    def forward(self, x):
        indices = torch.argmax(x, dim=1)
        output = self.embedding(indices)
        return output

class DAGEmbedding(nn.Module):
    def __init__(self, node_out_channels, layers):
        super().__init__()
        self.K = layers
        self.F_T = nn.ModuleList([
            MLPBlock(3 * node_out_channels,
                     node_out_channels,
                     batch=True) for i in range(layers)
        ])
        self.F_M = nn.ModuleList([
            MLPBlock(3 * node_out_channels,
                     node_out_channels,
                     batch=True) for i in range(layers)
        ])
        self.F_B = nn.ModuleList([
            MLPBlock(3 * node_out_channels,
                     node_out_channels,
                     batch=True) for i in range(layers)
        ])
        self.F_trans_TW = nn.ModuleList([
            MLPBlock(node_out_channels,
                     node_out_channels,
                     batch=True) for i in range(layers)
        ])

        self.gate = nn.Sequential(
            nn.Linear(2 * node_out_channels, 1),  
            nn.Sigmoid()
        )
        self.reset_gate_parameters()

    def reset_gate_parameters(self):

        nn.init.xavier_normal_(self.gate[0].weight)
        nn.init.constant_(self.gate[0].bias, 0.0)

    def forward(self, x, term_walk_index):
        term_walk_index = term_walk_index.to(x.device)
        N = x.size()[0]
        term_walk_index = term_walk_index.long()
        for i in range(self.K):
            term_walk_feat = torch.cat([x[term_walk_index[0]],
                                        x[term_walk_index[1]],
                                        x[term_walk_index[2]]], dim=1)

            trans_T = self.F_T[i](term_walk_feat)
            m_T = scatter_mean(trans_T,
                               index=term_walk_index[0],
                               dim=0, dim_size=N)

            trans_M = self.F_M[i](term_walk_feat)
            m_M = scatter_mean(trans_M,
                               index=term_walk_index[1],
                               dim=0, dim_size=N)

            trans_B = self.F_B[i](term_walk_feat)
            m_B = scatter_mean(trans_B,
                               index=term_walk_index[2],
                               dim=0, dim_size=N)

            m_TW = m_T + m_M + m_B
            m_TW = self.F_trans_TW[i](m_TW)

            gate_input = torch.cat([x, m_TW], dim=1)
            alpha = self.gate(gate_input)  
            x = alpha * m_TW + (1 - alpha) * x
        return x

class AttentionMechanism(nn.Module):
    def __init__(self, node_out_channels):
        super().__init__()

        self.node_attn = nn.Sequential(
            nn.Linear(node_out_channels, node_out_channels),
            nn.LeakyReLU(0.2),
            nn.Linear(node_out_channels, 1)
        )

        self.subterm_attn = nn.Sequential(
            nn.Linear(node_out_channels, node_out_channels),
            nn.ReLU(),
            nn.Linear(node_out_channels, 1),
            nn.Sigmoid()
        )

        nn.init.xavier_uniform_(self.node_attn[0].weight)
        nn.init.xavier_uniform_(self.subterm_attn[0].weight)

    def forward(self, node_features):

        alpha = torch.softmax(self.node_attn(node_features).squeeze(-1), dim=0)

        subterm_feat = node_features.mean(dim=0, keepdim=True)
        beta = self.subterm_attn(subterm_feat).squeeze()

        return alpha,beta

class HierarchicalPooling(nn.Module):
    def __init__(self, node_out_channels, pooling_mode="full"):
        super().__init__()
        self.node_out_channels = node_out_channels
        self.pooling_mode = pooling_mode
        self.attention = AttentionMechanism(node_out_channels)
        self.mean_proj = nn.Linear(node_out_channels, node_out_channels)

    def forward(self, x, batch_index, graph=None):
        if graph is None or self.pooling_mode == "mean_pool":
            return global_mean_pool(x, batch_index), {}

        super_nodes = {}
        batch_size = batch_index.max().item() + 1
        graph_embeddings = torch.zeros(
            batch_index.max().item() + 1,
            self.node_out_channels,
            device=x.device
        )

        sorted_subterms = sorted(graph.id2subterm.items(),
                               key=lambda item: len(str(item[1][0])))

        for subterm_id, (subterm, node) in sorted_subterms:
            node_ids = self._get_subterm_node_ids(node, graph)
            node_features = x[node_ids]

            if self.pooling_mode == "alpha_only":
                alpha, _ = self.attention(node_features)
                subterm_feature = torch.sum(alpha.unsqueeze(-1) * node_features, dim=0)
            elif self.pooling_mode == "beta_only":
                _, beta = self.attention(node_features)
                subterm_feature = beta * node_features.mean(dim=0)
            else:
                alpha, beta = self.attention(node_features)
                subterm_feature_base = torch.sum(alpha.unsqueeze(-1) * node_features, dim=0)
                subterm_feature = beta * subterm_feature_base

            super_nodes[subterm_id] = subterm_feature
            x[node_ids] = subterm_feature.unsqueeze(0).expand(len(node_ids), -1)

        for i in range(batch_size):
            graph_mask = (batch_index == i)
            if graph_mask.any():
                outermost_id = self._find_outermost_subterm(graph, graph_mask)
                if outermost_id is not None:
                    graph_embeddings[i] = super_nodes[outermost_id]
                else:

                    graph_embeddings[i] = self.mean_proj(x[graph_mask].mean(dim=0))

        return graph_embeddings, super_nodes

class Classifier(nn.Module):
    def __init__(self, node_out_channels, use_r1=True, use_r2=True, use_r3=True,
                 use_hard_r_true=True, use_hard_r_disj=True):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            MLPBlock(2 * node_out_channels, node_out_channels // 2, batch=True),
            nn.Linear(node_out_channels // 2, 2)
        )
        self.alpha_r1 = nn.Parameter(torch.tensor(1.0))
        self.alpha_r2 = nn.Parameter(torch.tensor(1.0))
        self.alpha_r3 = nn.Parameter(torch.tensor(1.0))
        self.predicate_pattern = re.compile(r"[a-z][a-z0-9_]*")

        self.use_r1 = use_r1
        self.use_r2 = use_r2
        self.use_r3 = use_r3
        self.use_hard_r_true = use_hard_r_true
        self.use_hard_r_disj = use_hard_r_disj

    def apply_hard_rules(self, conj_graph, prem_graph, pred_y):
        if prem_graph is None and conj_graph is None:
            return pred_y

        if self.use_hard_r_true:
            if (len(prem_graph.graph) == 1 or
                prem_graph.graph[0].name == "$true" or
                not prem_graph.graph[0].children):
                pred_y[:, 0] = 1.0
                pred_y[:, 1] = 0.0
                return pred_y

        if self.use_hard_r_disj:
            pred_s = self.extract_predicates(conj_graph)
            pred_t = self.extract_predicates(prem_graph)
            if not pred_s & pred_t:
                pred_y[:, 0] = 1.0
                pred_y[:, 1] = 0.0

        return pred_y

    def extract_quantifier_paths(formula):
        paths = []
        current_path = []
        stack = []

        tokens = re.findall(r'(!|\?|\(|\)|\||&)', formula)
        for token in tokens:
            if token in ('!', '?'):  
                current_path.append(token)
            elif token == '(':  
                stack.append((list(current_path), len(paths)))
                current_path = []
            elif token == ')':  
                if current_path:
                    paths.append(current_path)
                if stack:
                    current_path, path_idx = stack.pop()
            elif token in ('|', '&'): 
                if current_path:
                    paths.append(current_path)
                current_path = stack[-1][0].copy() if stack else []

        if current_path:
            paths.append(current_path)
        return paths

    def extract_predicates(self, graph):
        if not hasattr(graph, 'graph'): 
            return set()
        predicates = set()
        for node in graph.graph:
            if self.predicate_pattern.fullmatch(node.name):
                predicates.add(node.name)
        return predicates

    def path_matching_ratio(self, conj_graph, prem_graph):
        if conj_graph is None or prem_graph is None:
            return 0.0
        if not hasattr(conj_graph, 'formula') or not hasattr(prem_graph, 'formula'):
            return 0.0

        conj_paths = extract_quantifier_paths(conj_graph.formula)
        prem_paths = extract_quantifier_paths(prem_graph.formula)

        max_match_len = 0
        for prem_path in prem_paths:
            for conj_path in conj_paths:
                for i in range(len(conj_path) - len(prem_path) + 1):
                    if prem_path == conj_path[i:i+len(prem_path)]:
                        max_match_len = max(max_match_len, len(prem_path))
                        break

        max_conj_len = max(len(p) for p in conj_paths) if conj_paths else 1
        return max_match_len / max_conj_len

    def apply_soft_rules(self, conj_graph, prem_graph, super_nodes_s, super_nodes_t, pred_y):

        if conj_graph is None or prem_graph is None:
            return pred_y
        mask = (pred_y[:, 1] != 0.0) 

        xi_r1 = torch.zeros(pred_y.size(0), device=pred_y.device)
        xi_r2 = torch.zeros(pred_y.size(0), device=pred_y.device)
        xi_r3 = torch.zeros(pred_y.size(0), device=pred_y.device)

        if self.use_r1 and super_nodes_s and super_nodes_t and prem_graph is not None:       
            feat_s = torch.stack([v for v in super_nodes_s.values()]).to(device)
            feat_t = torch.stack([v for v in super_nodes_t.values()]).to(device)
            sim = F.cosine_similarity(feat_s, feat_t, dim=1)
            xi_r1[mask] = sim.max() if len(sim) > 0 else 0.0

        if self.use_r2:               
            pred_s = extract_predicates(conj_graph.formula)
            pred_t = extract_predicates(conj_graph.formula)
            match_count = len(pred_s & pred_t)
            total_count = max(len(pred_t), 1) 
            xi_r2[mask] += match_count / total_count

        if self.use_r3:        
            xi_r3[mask] = self.path_matching_ratio(conj_graph, prem_graph)

        rule_weights = (torch.sigmoid(self.alpha_r1) * xi_r1 if self.use_r1 else 0) + \
                      (torch.sigmoid(self.alpha_r2) * xi_r2 if self.use_r2 else 0) + \
                      (torch.sigmoid(self.alpha_r3) * xi_r3 if self.use_r3 else 0)

        pred_y[mask, 1] = pred_y[mask, 1] * torch.exp(rule_weights[mask])
        pred_y = pred_y / pred_y.sum(dim=1, keepdim=True)        
        return pred_y

    def forward(self, conj_batch, prem_batch, conj_graph=None, prem_graph=None, super_nodes_s=None, super_nodes_t=None):      
        x_concat = torch.cat([conj_batch, prem_batch], dim=1)
        pred_y = self.classifier(x_concat)

        pred_y = self.apply_hard_rules(conj_graph, prem_graph, pred_y)        
        pred_y = self.apply_soft_rules(conj_graph, prem_graph, super_nodes_s, super_nodes_t, pred_y)        
        return pred_y

class MGDA(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, losses, shared_params):
        if not torch.is_grad_enabled():  
            return torch.sum(losses) 
        if len(losses) == 0:
            return torch.tensor(0.0, device=losses.device)

        losses = [loss if loss.requires_grad else loss.requires_grad_(True) for loss in losses]
        losses = torch.stack(losses)

        grads = []
        for loss in losses:
            params = [p for p in shared_params if p.requires_grad]
            if not params:  
                  grads.append(torch.zeros(1, device=loss.device))  
                  continue

            grad = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            filtered_grad = [g.view(-1) for g in grad if g is not None]
            if not filtered_grad:  
                flat_grad = torch.zeros(1, device=loss.device)  
            else:
                flat_grad = torch.cat(filtered_grad)
            grads.append(flat_grad)

        if not grads:  
            return torch.sum(losses)  

        grads = torch.stack(grads)
        alpha = self.solve_mgda(grads)

        return torch.sum(alpha * losses)

    def solve_mgda(self, grads, max_iter=20):
        num_tasks = grads.size(0)
        alpha = torch.ones(num_tasks, device=grads.device) / num_tasks

        for k in range(max_iter):
            combined_grad = torch.matmul(alpha, grads)

            dot_products = torch.matmul(grads, combined_grad)
            s = torch.zeros_like(alpha)
            s[dot_products.argmin()] = 1.0

            gamma = 2.0 / (k + 2.0)
            alpha = (1 - gamma) * alpha + gamma * s

        return alpha.detach()

class LogicLoss(nn.Module):
    def __init__(self, node_out_channels, tau=0.2, use_r1=True, use_r2=True, use_r3=True,
                 weighting_mode="mgda"):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.tau = tau
        self.node_out_channels = node_out_channels
        self.pooling = HierarchicalPooling(node_out_channels)
        self.graph = []
        self.db = Classifier(node_out_channels)

        self.use_r1 = use_r1
        self.use_r2 = use_r2
        self.use_r3 = use_r3

        self.weighting_mode = weighting_mode
        self.mgda = MGDA().to(self.device)
        if weighting_mode == "uncertainty":
            self.log_vars = nn.Parameter(torch.zeros(3, device=self.device))

    def compute_similarity_loss(self, super_nodes_s, super_nodes_t, y):
        if not super_nodes_s or not super_nodes_t:
            return torch.tensor(0.0, device=y.device)
        try:
            feat_s = torch.stack([v for _,v in super_nodes_s.items()])
            feat_t = torch.stack([v for _,v in super_nodes_t.items()])

            sim = F.cosine_similarity(feat_s, feat_t, dim=1)
            max_sim = sim.max()
            pos_loss = F.relu(1 - max_sim) * y.float().mean()
            neg_loss = F.relu(max_sim - self.tau) * (1 - y.float().mean())
            losses.append(pos_loss + neg_loss)
            return losses

        except RuntimeError as e:
            print(f"Error in similarity calculation: {e}")
            return torch.tensor(0.0, device=y.device)

    def compute_predicate_loss(self, conj_graph, prem_graph, y):
        pred_s = self.db.extract_predicates(conj_graph)
        pred_t = self.db.extract_predicates(prem_graph)

        match_count = len(pred_s & pred_t)
        unique_count = len(pred_t - pred_s)      
        return (-match_count * (2 * y.float().mean() - 1) + unique_count * (1 - y.float().mean()))

    def compute_quant_path_loss(self, conj_graph, prem_graph, y):

        if not hasattr(conj_graph, 'formula') or not hasattr(prem_graph, 'formula'):
            return torch.tensor(0.0, device=self.device)

        conj_paths = self.db.extract_quantifier_paths(conj_graph.formula)
        prem_paths = self.db.extract_quantifier_paths(prem_graph.formula)

        max_match_len = 0
        for prem_path in prem_paths:
            for conj_path in conj_paths:
                for i in range(len(conj_path) - len(prem_path) + 1):
                    if prem_path == conj_path[i:i+len(prem_path)]:
                        max_match_len = max(max_match_len, len(prem_path))
                        break

        max_conj_len = max(len(p) for p in conj_paths) if conj_paths else 1
        ratio = max_match_len / max_conj_len        
        return torch.where(y == 1, 1 - ratio, ratio).mean()

    def forward(self, h_g_s, h_g_t, conj_graph, prem_graph, y, shared_params):

         h_g_s = h_g_s.requires_grad_(True) if not h_g_s.requires_grad else h_g_s
         h_g_t = h_g_t.requires_grad_(True) if not h_g_t.requires_grad else h_g_t

         _, super_nodes_s = self.pooling(h_g_s, None, conj_graph)
         _, super_nodes_t = self.pooling(h_g_t, None, prem_graph)

         losses = []

         if self.use_r1 and super_nodes_s and super_nodes_t:
             L_sim = self.compute_similarity_loss(super_nodes_s, super_nodes_t, y)
             losses.append(L_sim)      
         if self.use_r2 and conj_graph is not None and prem_graph is not None:
             L_match = self.compute_predicate_loss(conj_graph, prem_graph, y)
             losses.append(L_match)
         if self.use_r3 and conj_graph is not None and prem_graph is not None:
             L_quant = self.compute_quant_path_loss(conj_graph, prem_graph, y)
             losses.append(L_quant)   
         if not losses:  
             return torch.tensor(0.0, device=y.device)

         losses = torch.stack(losses)

         if self.weighting_mode == "equal":
             total_loss = losses.mean()
         elif self.weighting_mode == "sum":
             total_loss = losses.sum()
         elif self.weighting_mode == "uncertainty":

             stacked = []
             for i in range(losses.size(0)):
                 lv = self.log_vars[i]
                 stacked.append(torch.exp(-lv) * losses[i] + lv)
             total_loss = torch.stack(stacked).sum()
         else:
             total_loss = self.mgda(losses, shared_params)
         return total_loss

class PremiseSelectionModel(nn.Module):
    def __init__(self, node_in_channels, node_out_channels, layers, logic_rules="all",
                 pooling_mode="full", use_hard_r_true=True, use_hard_r_disj=True,
                 logic_loss_weight=0.2, tau=0.2, weighting_mode="mgda"):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.node_out_channels = node_out_channels
        self.initial = Initialization(node_in_channels, node_out_channels)
        self.dag_emb = DAGEmbedding(node_out_channels, layers)
        self.pooling = HierarchicalPooling(node_out_channels, pooling_mode=pooling_mode)
        self.criterion = nn.CrossEntropyLoss()
        self.corrects = None
        self.logic_loss_weight = logic_loss_weight

        rule_cfg = {
            "all":    (True, True, True,   True, True, True),
            "r1":     (True, False, False,  True, False, False),
            "r2":     (False, True, False,  False, True, False),
            "r3":     (False, False, True,  False, False, True),
            "r1_r2":  (True, True, False,   True, True, False),
            "r1_r3":  (True, False, True,   True, False, True),
            "r2_r3":  (False, True, True,   False, True, True),
            "none":   (False, False, False, False, False, False),
            "all_c":  (True, True, True,   False, False, False),
            "all_l":  (False, False, False, True, True, True),
        }
        cr1, cr2, cr3, lr1, lr2, lr3 = rule_cfg.get(logic_rules, (True, True, True, True, True, True))
        self.classifier = Classifier(node_out_channels, use_r1=cr1, use_r2=cr2, use_r3=cr3,
                                     use_hard_r_true=use_hard_r_true, use_hard_r_disj=use_hard_r_disj)
        self.logic_loss = LogicLoss(node_out_channels, tau=tau, use_r1=lr1, use_r2=lr2, use_r3=lr3,
                                    weighting_mode=weighting_mode)
        self.to(self.device)

    def compute_logic_loss(self, h_g_s, h_g_t, y, conj_graph, prem_graph):     
        shared_params = list(self.dag_emb.parameters()) + list(self.pooling.parameters())
        return self.logic_loss(h_g_s, h_g_t, conj_graph, prem_graph, y, shared_params)

    def forward(self, batch):
        batch = batch.to(self.device)

        h_s = self.initial(batch.x_s)
        h_t = self.initial(batch.x_t)

        if hasattr(batch, 'term_walk_index_t') and batch.term_walk_index_t.numel() > 0:
            h_t = self.dag_emb(h_t, batch.term_walk_index_t)

        if hasattr(batch, 'term_walk_index_s') and batch.term_walk_index_s.numel() > 0:
            h_s = self.dag_emb(h_s, batch.term_walk_index_s)

        prem_graphs = batch._store.get('prem_graphs', [None]*len(batch.y))
        conj_graphs = batch._store.get('conj_graphs', [None]*len(batch.y))

        prem_graph = batch.prem_graphs[0] if hasattr(batch, 'prem_graphs') and len(batch.prem_graphs) > 0 else None
        conj_graph = batch.conj_graphs[0] if hasattr(batch, 'conj_graphs') and len(batch.conj_graphs) > 0 else None

        if not hasattr(batch, 'x_s'):
            batch.x_s = torch.empty(0, device=self.device)
        if not hasattr(batch, 'x_t'):
            batch.x_t = torch.empty(0, device=self.device)
        x_s_batch = getattr(batch, 'x_s_batch', torch.zeros(batch.x_s.size(0), dtype=torch.long, device=self.device))
        x_t_batch = getattr(batch, 'x_t_batch', torch.zeros(batch.x_t.size(0), dtype=torch.long, device=self.device))

        h_g_s, super_nodes_s = self.pooling(h_s, x_s_batch, conj_graph)
        h_g_t, super_nodes_t = self.pooling(h_t, x_t_batch, prem_graph)

        pred_y = self.classifier(h_g_s, h_g_t, conj_graph, prem_graph, super_nodes_s, super_nodes_t)

        cls_loss = self.criterion(pred_y, batch.y)
        logic_loss = self.compute_logic_loss(h_s, h_t, batch.y, conj_graph, prem_graph)
        total_loss = (1-self.logic_loss_weight)*cls_loss + self.logic_loss_weight * logic_loss

        pred_label = torch.max(pred_y, dim=1)[1]
        self.corrects = (pred_label == batch.y).sum().cpu().item()

        return total_loss, batch.y, pred_label
