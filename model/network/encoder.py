import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from model.network.attn import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout, layer_norm, batch_norm):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.W_o = nn.Linear(d_model, d_model)

        if batch_norm:
            self.norm_part1 = nn.BatchNorm1d(d_model)
        elif layer_norm:
            self.norm_part1 = nn.LayerNorm(d_model)

        # FFN
        self.FFN_layer1 = nn.Linear(d_model, d_model * 2)
        self.FFN_layer2 = nn.Linear(d_model * 2, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if batch_norm:
            self.norm_part2 = nn.BatchNorm1d(d_model)
        elif layer_norm:
            self.norm_part2 = nn.LayerNorm(d_model)

    def forward(self, g, h):
        h_in1 = h  # for first residual connection

        # multi-head attention out
        attn_out = self.attention(g, h)
        h = attn_out.view(-1, self.d_model)

        h = self.dropout1(h)
        h = self.W_o(h)
        h = self.norm_part1(h_in1 + h)  # residual connection & normalization

        h_in2 = h  # for second residual connection

        # FFN
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = self.dropout2(h)
        h = self.FFN_layer2(h)

        h = self.norm_part2(h + h_in2)

        return h


class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(Encoder, self).__init__()
        self.encoder = _get_clones(encoder_layer, num_layers)

    def forward(self, g: dgl.DGLGraph, h):
        for layer in self.encoder:
            h = layer(g, h)
        g.ndata["h"] = h

        vectors = dgl.readout_nodes(g, "h", op="mean")

        # # 只选取 真实节点 进行mean
        # all_feat = g.ndata["h"]
        # all_flag = g.ndata["flag"]
        # print(all_flag[:600])
        # vectors = None

        return vectors  # [graph num, d_model]


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
