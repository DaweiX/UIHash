"""Implementations of various network structures used
in the siamese network, as well as loss functions"""

import torch
import torch.nn as nn
import torch.nn.functional as f
import typing
import numpy as np


class SiameseNet(nn.Module):

    def __call__(self, *ipt, **kwargs) -> typing.Any:
        return super().__call__(*ipt, **kwargs)

    def __init__(self, cnn, fc,
                 isbinary: bool = False):
        super(SiameseNet, self).__init__()
        self.cnn = cnn
        self.fc = fc

        self.output = nn.Sequential(
            nn.Softsign()
        )

        self.isbinary = isbinary

    def forward_once(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, i: np.ndarray):
        rows = int(i.shape[0] / 2)
        input1, input2 = i[:rows, :], i[rows:, :]
        input1 = torch.squeeze(input1, 0)
        input2 = torch.squeeze(input2, 0)
        o1 = self.forward_once(input1)
        o2 = self.forward_once(input2)

        o = torch.stack((o1, o2), 0)
        return o


class SiameseNetFlatten(nn.Module):

    def __call__(self, *ipt, **kwargs) -> typing.Any:
        return super().__call__(*ipt, **kwargs)

    def __init__(self):
        super(SiameseNetFlatten, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(81, 200),
            nn.PReLU(),
            nn.Linear(200, 200),
            nn.PReLU(),
            nn.Linear(200, 80),
            nn.PReLU(),
            nn.Linear(80, 5),
            nn.Dropout(0.5)
        )

    def forward_once(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        o1 = self.forward_once(input1)
        o2 = self.forward_once(input2)
        return o1, o2


class CosLoss(nn.Module):
    def __init__(self, device, pos_label: int = 1):
        super(CosLoss, self).__init__()
        self.pos_label = pos_label
        self.device = device

    def forward(self, o1, o2, label):
        o1 = torch.squeeze(o1, 0)
        o2 = torch.squeeze(o2, 0)
        cos = torch.cosine_similarity(o1, o2)
        lb = label
        if self.pos_label == 0:
            lb = 1 - label

        cos = cos.double().to(self.device)
        lb = lb.double().to(self.device)
        a = torch.pow(torch.add(-cos, lb), 2)
        loss = torch.mean(a)
        return loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0, pos_label=1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.pos_label = pos_label

    def forward(self, output1, output2, label):
        lb = label if self.pos_label == 0 else 1 - label
        cos_distance = torch.add(1, -f.cosine_similarity(output1, output2))
        distance = cos_distance
        a1 = torch.pow(distance, 2)
        # calculate: {max(0, margin - distance)}^2
        a2 = torch.pow(
            torch.clamp(
                torch.add(
                    self.margin, -distance), min=0.0), 2)
        loss_contrastive = torch.mean((1 - lb) * a1 + lb * a2)

        return loss_contrastive


class NNParas:
    channel = 10

    def __init__(self, _channel: int = 10):
        self.channel = _channel

    @property
    def cnn3x3(self):
        return nn.Sequential(
            nn.Conv2d(self.channel, 18, 2),  # 9, 3x3
            nn.PReLU(),
            nn.Conv2d(18, 72, 2),
            nn.PReLU()
        )

    @property
    def cnn4x3(self):
        return nn.Sequential(
            nn.Conv2d(self.channel, 18, 2),  # 18, 3x2
            nn.PReLU(),
            nn.Conv2d(18, 72, 2),  # 72, 2x1
            nn.PReLU(),
        )

    @property
    def fc4x3(self):
        return nn.Sequential(
            nn.Linear(144, 256),
            nn.PReLU(),
            nn.Linear(256, 120),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Linear(120, 12),
        )

    @property
    def fc3x3(self):
        return nn.Sequential(
            nn.Linear(72, 256),
            nn.PReLU(),
            nn.Linear(256, 120),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Linear(120, 12),
        )

    @property
    def cnn2d(self):
        return nn.Sequential(
            nn.Conv1d(1, 18, 9, stride=9),  # 9
            nn.PReLU(),
        )

    @property
    def fc2d(self):
        return nn.Sequential(
            nn.Linear(162, 80),
            nn.PReLU(),
            nn.Linear(80, 80),
            nn.Linear(80, 1),
            nn.Dropout(0.2)
        )

    @property
    def cnn1x1(self):
        return nn.Sequential(
            nn.Conv2d(self.channel, 18, 1),  # 9, 1x1
            nn.PReLU(),
            nn.Conv2d(18, 18, 1),  # 9, 1x1
            nn.PReLU(),
        )

    fc1x1 = nn.Sequential(
        nn.Linear(18, 9),
    )

    @property
    def cnn2x2(self):
        return nn.Sequential(
            nn.Conv2d(self.channel, 9, 1),  # 9, 2x2
            nn.PReLU(),
            nn.Conv2d(9, 18, 1),  # 18, 2x2
            nn.PReLU(),
        )

    fc2x2 = nn.Sequential(
        nn.Linear(72, 80),
        nn.PReLU(),
        nn.Linear(80, 80),
        nn.Dropout(0.5),
        nn.Linear(80, 8),
    )

    @property
    def cnn5x5(self):
        return nn.Sequential(
            nn.Conv2d(self.channel, 18, 2),  # 18, 4x4
            nn.PReLU(),
            nn.Conv2d(18, 72, 2),  # 72, 3x3
            nn.PReLU(),
        )

    fc5x5 = nn.Sequential(
        nn.Linear(648, 256),
        nn.PReLU(),
        nn.Linear(256, 128),
        nn.PReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 16),
    )

    @property
    def cnn10x10(self):
        return nn.Sequential(
            nn.Conv2d(self.channel, 18, 4),  # 18, 7x7
            nn.PReLU(),
            nn.Conv2d(18, 72, 4),  # 72, 4x4
            nn.PReLU(),
        )

    fc10x10 = nn.Sequential(
        nn.Linear(1152, 512),
        nn.PReLU(),
        nn.Linear(512, 256),
        nn.PReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 20),
    )
