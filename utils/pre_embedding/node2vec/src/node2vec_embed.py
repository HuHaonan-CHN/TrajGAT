"""
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
"""

import networkx as nx
import torch
from utils.pre_embedding.node2vec.src import node2vec
from gensim.models import Word2Vec

from utils.qtree import get_qtree_feat


class Parameter:
    def __init__(self, d_model) -> None:
        # self.output = "qtree.emb"
        self.dimensions = d_model
        self.walk_length = 80
        self.num_walks = 10
        self.window_size = 10
        self.iter = 1
        self.workers = 8
        self.p = 1
        self.q = 1
        self.weighted = False
        self.unweighted = True
        self.directed = False
        self.undirected = True


def read_graph(id_edge_list, node2vec_args):
    """
	Reads the input network in networkx.
	"""
    if node2vec_args.weighted:
        # G = nx.read_edgelist(input_path, nodetype=int, data=(("weight", float),), create_using=nx.DiGraph())
        G = nx.DiGraph(id_edge_list)  # 这里的 id_edge_list 需要带有权重
    else:
        G = nx.DiGraph(id_edge_list)

        for edge in G.edges():
            G[edge[0]][edge[1]]["weight"] = 1

    if not node2vec_args.directed:
        G = G.to_undirected()

    return G


def learn_embeddings(walks, node2vec_args):
    """
	Learn embeddings by optimizing the Skipgram objective using SGD.
	"""
    walks = [[str(i) for i in walk] for walk in walks]
    model = Word2Vec(walks, vector_size=node2vec_args.dimensions, window=node2vec_args.window_size, min_count=0, sg=1, workers=node2vec_args.workers, epochs=node2vec_args.iter)
    # model.wv.save_word2vec_format(node2vec_args.output)

    # 获得训练好的词向量
    node_num = len(model.wv.key_to_index)

    all_vectors = model.wv[[str(i) for i in range(node_num)]]
    # print(all_vectors[0])
    # print(all_vectors[1])
    # print(all_vectors[-1])
    return all_vectors


def node2vec_embed(id_edge_list, d_model):
    """
	Pipeline for representational learning for all nodes in a graph.
	"""
    node2vec_args = Parameter(d_model)
    nx_G = read_graph(id_edge_list, node2vec_args)
    G = node2vec.Graph(nx_G, node2vec_args.directed, node2vec_args.p, node2vec_args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(node2vec_args.num_walks, node2vec_args.walk_length)

    all_vectors = learn_embeddings(walks, node2vec_args)

    return all_vectors

    # f = open("utils/pre_embedding/emb/qtree.emb", "r")
    # file_content = f.readlines()[1:]
    # id2vector = {}

    # for line in file_content:
    #     this_line = [eval(i) for i in line.replace("\n", "").split(" ")]
    #     id2vector[this_line[0]] = this_line[1:]

    # qtree_feats = {}
    # for tid in name2id.keys():
    #     num = name2id[tid]
    #     vec = id2vector[num]
    #     qtree_feats[tid] = vec

    # for index in sorted(id2vector.keys()):
    #     pre_embedding.append(id2vector[index])
    # pre_embedding = torch.tensor(pre_embedding).float()

    # print("pre embedding shape:", pre_embedding.shape)
    # return name2id, pre_embedding


# if __name__ == "__main__":
# 	args = parse_args()
# 	main(args)
