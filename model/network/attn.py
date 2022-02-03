import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}

    return func


def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}

    return func


def message_func(edges):
    # message UDF for equation (3) & (4)
    return {"V_h": edges.src["V_h"], "score": edges.data["score"]}


def reduce_func(nodes):
    # reduce UDF for equation (3) & (4)
    # equation (3)
    alpha = F.softmax(nodes.mailbox["score"], dim=1)
    # equation (4)
    h = torch.sum(alpha * nodes.mailbox["V_h"], dim=1)
    return {"V_h": h}


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, use_bias=False):
        super(MultiHeadAttention,self).__init__()
        assert d_model % n_head == 0, "d_model must can be divisible by n_head"
        self.dim_head = d_model // n_head
        self.n_head = n_head

        self.W_q = nn.Linear(d_model, d_model, bias=use_bias)
        self.W_k = nn.Linear(d_model, d_model, bias=use_bias)
        self.W_v = nn.Linear(d_model, d_model, bias=use_bias)

    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst("K_h", "Q_h", "score"))
        g.apply_edges(scaled_exp("score", np.sqrt(self.dim_head)))

        g.update_all(message_func, reduce_func)

    def forward(self, g, h):
        Q_h = self.W_q(h).view(-1, self.n_head, self.dim_head)
        K_h = self.W_k(h).view(-1, self.n_head, self.dim_head)
        V_h = self.W_v(h).view(-1, self.n_head, self.dim_head)

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        g.ndata["Q_h"] = Q_h
        g.ndata["K_h"] = K_h
        g.ndata["V_h"] = V_h

        self.propagate_attention(g)

        head_out = g.ndata["V_h"]

        return head_out #【node num, n_head, d_head】

