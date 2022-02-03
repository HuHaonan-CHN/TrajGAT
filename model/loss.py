import torch
from torch.nn import Module, PairwiseDistance


class WeightedRankingLoss(Module):
    def __init__(self, sample_num, alpha, device):
        super(WeightedRankingLoss, self).__init__()
        self.alpha = alpha
        weight = [1]
        single_sample = sample_num // 2

        for index in range(single_sample, 1, -1):
            weight.append(index)
        for index in range(1, single_sample + 1, 1):
            weight.append(index)

        # 权重归一化
        weight = torch.tensor(weight, dtype=torch.float)
        self.weight = (weight / torch.sum(weight)).to(device)

    def forward(self, vec, all_dis):
        """
        vec [batch_size, sample_num, d_model]
        dis [batch_size, sample_num]
        """
        all_loss = 0
        batch_num = vec.size(0)
        sample_num = vec.size(1)

        for batch in range(batch_num):
            traj_list = vec[batch]
            dis_list = all_dis[batch]

            loss = 0
            anchor_trajs = traj_list[0].repeat(sample_num, 1)

            pairdist = PairwiseDistance(p=2)
            dis_pred = pairdist(anchor_trajs, traj_list)
            # [sample_num]

            sim_pred = torch.exp(-dis_pred)
            sim_truth = torch.exp(-self.alpha * dis_list)
            # [sample_num]

            div = sim_truth - sim_pred
            square = torch.mul(div, div)
            weighted_square = torch.mul(square, self.weight)
            loss = torch.sum(weighted_square)

            all_loss = all_loss + loss

        return all_loss

