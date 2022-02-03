import copy
import time
import datetime
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import dgl


from exp.exp_basic import ExpBasic
from model.network.graph_transformer import GraphTransformer

from utils.build_qtree import build_qtree
from utils.pre_embedding import get_pre_embedding

from utils.data_loader import TrajGraphDataLoader
from model.loss import WeightedRankingLoss
from model.accuracy_functions import get_embedding_acc

from utils.tools import pload


def view_model_param(model):
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print("MODEL Total parameters:", total_param, "\n")
    return total_param


class ExpGraphTransformer(ExpBasic):
    def __init__(self, config, gpu_id, load_model, just_embeddings):
        self.load_model = load_model
        self.store_embeddings = just_embeddings

        super(ExpGraphTransformer, self).__init__(config, gpu_id)

        if just_embeddings:  # 只进行embedding操作
            self.qtree = build_qtree(pload(self.config["traj_path"]), self.config["max_nodes"], self.config["max_depth"])
            # 决定是否要进行 embedding预训练
            self.qtree_name2id, self.pre_embedding = get_pre_embedding(self.qtree, self.config["d_model"])
            self.embeding_loader = self._get_dataloader(flag="embed")
            print("Embedding Graphs: ", len(self.embeding_loader.dataset))
        else:
            self.log_writer = SummaryWriter(f"./runs/{self.config['data']}/{self.config['length']}/{self.config['model']}_{self.config['dis_type']}_{datetime.datetime.now()}/")

            print("[!] Build qtree, max nodes:", self.config["max_nodes"], "max depth:", self.config["max_depth"])
            self.qtree = build_qtree(pload(self.config["traj_path"]), self.config["max_nodes"], self.config["max_depth"])

            # 进行embedding预训练
            self.qtree_name2id, self.pre_embedding = get_pre_embedding(self.qtree, self.config["d_model"])

            self.train_loader = self._get_dataloader(flag="train")
            print("Training Graphs: ", len(self.train_loader.dataset))

            self.val_loader = self._get_dataloader(flag="val")
            print("Validation Graphs: ", len(self.val_loader.dataset))

        self.model = self._build_model().to(self.device)

    def _build_model(self):
        if self.config["model"] == "TrajGAT":
            model = GraphTransformer(d_input=self.config["d_input"], d_model=self.config["d_model"], num_head=self.config["num_head"], num_encoder_layers=self.config["num_encoder_layers"], d_lap_pos=self.config["d_lap_pos"], encoder_dropout=self.config["encoder_dropout"], layer_norm=self.config["layer_norm"], batch_norm=self.config["batch_norm"], in_feat_dropout=self.config["in_feat_dropout"], pre_embedding=self.pre_embedding)  # 预训练得到的，每个结点的 structure embedding

        view_model_param(model)

        if self.load_model is not None:
            model.load_state_dict(torch.load(self.load_model))
            print("[!] Load model weight:", self.load_model)

        return model

    def _get_dataloader(self, flag):
        if flag == "train":
            trajs = pload(self.config["traj_path"])[self.config["train_data_range"][0] : self.config["train_data_range"][1]]
            print("Train traj number:", len(trajs))
            matrix = pload(self.config["dis_matrix_path"])[self.config["train_data_range"][0] : self.config["train_data_range"][1], self.config["train_data_range"][0] : self.config["train_data_range"][1]]
            print("Train matrix shape:", matrix.shape)
            print(matrix[:5, :5])
        elif flag == "val":
            trajs = pload(self.config["traj_path"])
            print("Val traj number:", len(trajs))
            matrix = pload(self.config["dis_matrix_path"])[self.config["val_data_range"][0] : self.config["val_data_range"][1], :]
            print("Val matrix shape:", matrix.shape)
            print(matrix[:5, :5])
            print(matrix[:, 6000:10000])

        elif flag == "embed":
            trajs = pload(self.config["traj_path"])
            matrix = pload(self.config["dis_matrix_path"])

        data_loader = TrajGraphDataLoader(traj_data=trajs, dis_matrix=matrix, phase=flag, train_batch_size=self.config["train_batch_size"], eval_batch_size=self.config["eval_batch_size"], d_lap_pos=self.config["d_lap_pos"], sample_num=self.config["sample_num"], num_workers=self.config["num_workers"], data_features=self.config["data_features"], x_range=self.config["x_range"], y_range=self.config["y_range"], qtree=self.qtree, qtree_name2id=self.qtree_name2id).get_data_loader()

        return data_loader

    def _select_optimizer(self):
        if self.config["optimizer"] == "SGD":
            model_optim = optim.SGD(self.model.parameters(), lr=self.config["init_lr"])
        elif self.config["optimizer"] == "Adam":
            model_optim = optim.Adam(self.model.parameters(), lr=self.config["init_lr"])

        return model_optim, None

    def _select_criterion(self):
        criterion = WeightedRankingLoss(self.config["sample_num"], self.config["alpha"], self.device).float()
        return criterion

    def embedding(self):
        all_vectors = []
        self.model.eval()

        loader_time = 0
        begin_time = time.time()
        mark_time = time.time()
        for trajgraph_l_l, _ in tqdm(self.embeding_loader):
            loader_time += time.time() - mark_time
            # trajgraph_l_l [B, 1, graph]
            B = len(trajgraph_l_l)
            D = self.config["d_model"]

            traj_graph = []
            for b in trajgraph_l_l:
                traj_graph.extend(b)
            batch_graphs = dgl.batch(traj_graph).to(self.device)  # (B, graph)

            with torch.no_grad():
                vectors = self.model(batch_graphs)  # vecters [B, d_model]

            all_vectors.append(vectors)
            mark_time = time.time()

        all_vectors = torch.cat(all_vectors, dim=0)
        print("all_embeding_vectors length:", len(all_vectors))
        print("all_embedding_vectors shape:", all_vectors.shape)

        end_time = time.time()
        print(f"all embedding time: {end_time-begin_time-loader_time} seconds")

        # pdump(all_vectors, f"{self.config['length']}_{self.config['dis_type']}_embeddings_{all_vectors.shape[0]}_{all_vectors.shape[1]}.pkl")

        hr10, hr50, r10_50 = get_embedding_acc(row_embedding_tensor=all_vectors, col_embedding_tensor=all_vectors, distance_matrix=self.embeding_loader.dataset.dis_matrix, matrix_cal_batch=self.config["matrix_cal_batch"],)

        print(hr10, hr50, r10_50)

    def val(self):
        all_vectors = []
        self.model.eval()

        for trajgraph_l_l, _ in self.val_loader:
            # trajgraph_l_l [B, 1, graph]
            B = len(trajgraph_l_l)
            D = self.config["d_model"]

            traj_graph = []
            for b in trajgraph_l_l:
                traj_graph.extend(b)
            batch_graphs = dgl.batch(traj_graph).to(self.device)  # (B, graph)

            with torch.no_grad():
                vectors = self.model(batch_graphs)  # vecters [B, d_model]

            all_vectors.append(vectors)

        all_vectors = torch.cat(all_vectors, dim=0)
        print("all_val_vectors length:", len(all_vectors))

        hr10, hr50, r10_50 = get_embedding_acc(row_embedding_tensor=all_vectors[self.config["val_data_range"][0] : self.config["val_data_range"][1]], col_embedding_tensor=all_vectors, distance_matrix=self.val_loader.dataset.dis_matrix, matrix_cal_batch=self.config["matrix_cal_batch"],)

        return hr10, hr50, r10_50

    def train(self):
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_hr10 = 0.0
        time_now = time.time()

        model_optim, scheduler = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.config["epoch"]):
            self.model.train()

            epoch_begin_time = time.time()
            epoch_loss = 0.0

            for trajgraph_l_l, dis_l in self.train_loader:
                # trajgraph_l_l [B, SAM, graph]
                # dis_l [B, SAM]
                B = len(trajgraph_l_l)
                SAM = self.config["sample_num"]
                D = self.config["d_model"]

                traj_graph = []
                for b in trajgraph_l_l:
                    traj_graph.extend(b)
                batch_graphs = dgl.batch(traj_graph).to(self.device)  # (B*SAM, graph)

                model_optim.zero_grad()

                with torch.set_grad_enabled(True):
                    vectors = self.model(batch_graphs)  # vecters [B*SAM, d_model]

                vectors = vectors.view(B, SAM, D)

                loss = criterion(vectors, torch.tensor(dis_l).to(self.device))

                loss.backward()
                model_optim.step()

                epoch_loss += loss.item()

            epoch_loss = epoch_loss / len(self.train_loader.dataset)
            self.log_writer.add_scalar(f"TrajRepresentation/Loss", float(epoch_loss), epoch)

            # scheduler.step(epoch_loss)

            epoch_end_time = time.time()
            print(f"\nEpoch {epoch+1}/{self.config['epoch']}:\nTrain Loss: {epoch_loss:.4f}\tTime: {(epoch_end_time - epoch_begin_time) // 60} m {int((epoch_end_time - epoch_begin_time) % 60)} s")

            val_begin_time = time.time()
            hr10, hr50, r10_50 = self.val()
            val_end_time = time.time()

            self.log_writer.add_scalar(f"TrajRepresentation/HR10", hr10, epoch)
            self.log_writer.add_scalar(f"TrajRepresentation/HR50", hr50, epoch)
            self.log_writer.add_scalar(f"TrajRepresentation/R10@50", r10_50, epoch)

            print(f"Val HR10: {100 * hr10:.4f}%\tHR50: {100 * hr50:.4f}%\tR10@50: {100 * r10_50:.4f}%\tTime: {(val_end_time -val_begin_time) // 60} m {int((val_end_time -val_begin_time) % 60)} s")

            if hr10 > best_hr10:
                best_hr10 = hr10
                best_model_wts = copy.deepcopy(self.model.state_dict())

        time_end = time.time()

        print("\nAll training complete in {:.0f}m {:.0f}s".format((time_end - time_now) // 60, (time_end - time_now) % 60))
        print(f"Best HR10: {100*best_hr10:.4f}%")

        torch.save(best_model_wts, self.config["model_best_wts_path"].format(self.config["data"], self.config["length"], self.config["model"], self.config["dis_type"], best_hr10))

