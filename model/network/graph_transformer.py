import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
import dgl

from model.network.encoder import EncoderLayer, Encoder


class GraphTransformer(nn.Module):
    def __init__(self, d_input, d_model, num_head, num_encoder_layers, d_lap_pos, encoder_dropout, pre_embedding, layer_norm=False, batch_norm=True, in_feat_dropout=0.0):
        super(GraphTransformer, self).__init__()
        self.embedding_h = nn.Linear(d_input, d_model)
        self.embedding_lap_pos = nn.Linear(d_lap_pos, d_model)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        # embedding layer for each node
        if pre_embedding is not None:
            self.embedding_id = nn.Embedding.from_pretrained(pre_embedding)
            self.use_pre_embedding = True
        else:
            self.embedding_id = None
            self.use_pre_embedding = False

        encoder_layer = EncoderLayer(d_model=d_model, num_heads=num_head, dropout=encoder_dropout, layer_norm=layer_norm, batch_norm=batch_norm)
        self.encoder = Encoder(encoder_layer, num_encoder_layers)

        self._reset_parameters()

    def forward(self, g):
        h = g.ndata["feat"]  # num x feat

        h_lap_pos = g.ndata["lap_pos_feat"]
        sign_flip = torch.rand(h_lap_pos.size(1)).to(h_lap_pos.device)
        sign_flip[sign_flip >= 0.5] = 1.0
        sign_flip[sign_flip < 0.5] = -1.0
        h_lap_pos = h_lap_pos * sign_flip.unsqueeze(0)

        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)

        # position embedding
        h_lap_pos = self.embedding_lap_pos(h_lap_pos.float())

        # id embedding
        if self.use_pre_embedding:
            h_id = g.ndata["id"]  # pre mebedding feat
            h_id = self.embedding_id(h_id)

            h = h + h_lap_pos + h_id
        else:
            h = h + h_lap_pos

        vectors = self.encoder(g, h)  # vectors [g_num, d_model]

        return vectors

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

