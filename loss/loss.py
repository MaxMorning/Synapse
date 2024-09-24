
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import piq
import numpy as np



class LossWithWeights(nn.Module):
    def __init__(self):
        super(LossWithWeights, self).__init__()

    @abstractmethod
    def weight_from_iter(self, iter_index):
        pass


class L1Loss(LossWithWeights):
    def __init__(self, loss_weight=1):
        super().__init__()
        self.L1Loss = nn.L1Loss()
        self.loss_weight = loss_weight

    def forward(self, x, y):
        return self.L1Loss(x, y)

    def weight_from_iter(self, iter_index):
        return self.loss_weight


class L2Loss(LossWithWeights):
    def __init__(self, loss_weight=1):
        super().__init__()
        self.L2Loss = nn.MSELoss()
        self.loss_weight = loss_weight

    def forward(self, x, y):
        return self.L2Loss(x, y)

    def weight_from_iter(self, iter_index):
        return self.loss_weight


class VideoSSIMLoss(LossWithWeights):
    def __init__(self, loss_weight=1):
        super(VideoSSIMLoss, self).__init__()
        self.loss = piq.SSIMLoss().to("cuda")

        self.loss_weight = loss_weight

    def forward(self, sr, hr):
        # sr = (sr + 1) / 2
        # hr = (hr + 1) / 2

        sr = torch.clamp(sr, 0, 1)
        hr = torch.clamp(hr, 0, 1)

        return self.loss(
            torch.flatten(sr, start_dim=0, end_dim=1),
            torch.flatten(hr, start_dim=0, end_dim=1)
        )

    def weight_from_iter(self, iter_index):
        return self.loss_weight


class ImageSSIMLoss(LossWithWeights):
    def __init__(self, loss_weight=1):
        super(ImageSSIMLoss, self).__init__()
        self.loss = piq.SSIMLoss().to("cuda")

        self.loss_weight = loss_weight

    def forward(self, sr, hr):
        # sr = (sr + 1) / 2
        # hr = (hr + 1) / 2

        sr = torch.clamp(sr, 0, 1)
        hr = torch.clamp(hr, 0, 1)

        return self.loss(
            sr,
            hr
        )

    def weight_from_iter(self, iter_index):
        return self.loss_weight


class CharbonnierLoss(LossWithWeights):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-6, loss_weight=1):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.loss_weight = loss_weight

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss

    def weight_from_iter(self, iter_index):
        return self.loss_weight
