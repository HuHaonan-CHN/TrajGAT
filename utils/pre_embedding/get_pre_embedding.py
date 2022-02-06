import torch

import numpy as np
import os
import json

from utils.qtree import get_qtree_feat
from utils.pre_embedding.node2vec.src.node2vec_embed import node2vec_embed


def get_pre_embedding(qtree, d_model):
    vir_id_edge_list, vir_id2center, word_embedding_name2id = get_qtree_feat(qtree)

    print("Edge number used in node2vec:", len(vir_id_edge_list))

    vir_pre_embedding = node2vec_embed(vir_id_edge_list, d_model)
    vir_pre_embedding = torch.tensor(vir_pre_embedding, dtype=torch.float)
    vir_name2id = word_embedding_name2id

    # 对预训练得到的embedding进行归一化 min-max
    print(vir_pre_embedding.min(), vir_pre_embedding.max())
    vir_min = vir_pre_embedding.min(axis=0)[0]
    vir_max = vir_pre_embedding.max(axis=0)[0]
    vir_pre_embedding = (vir_pre_embedding - vir_min) / (vir_max - vir_min)
    print(vir_pre_embedding.min(), vir_pre_embedding.max())

    # 添加全零的embedding
    vir_pre_embedding = torch.cat([torch.zeros(1, vir_pre_embedding.shape[1]), vir_pre_embedding], dim=0)

    print("The number of word embedding:", vir_pre_embedding.shape)

    return vir_name2id, vir_pre_embedding

