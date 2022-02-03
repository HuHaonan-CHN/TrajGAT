import os
import torch
import numpy as np


class ExpBasic(object):
    def __init__(self, config, gpu_id):
        self.config = config
        self.device = self._acquire_device(gpu_id)
        # self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self, gpu_id):
        # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Use GPU: cuda {gpu_id}")

        return device

    def _get_dataloader(self):
        pass

    def val(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

