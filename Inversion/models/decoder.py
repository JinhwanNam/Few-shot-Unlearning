import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np


class normalizer(nn.Module):
    def __init__(self):
        super(normalizer, self).__init__()

    def forward(self, x):
        x = torch.sigmoid(x)
        x = x * 2
        x = x - 1
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.img_shape = 100

        self.L1 = nn.Linear(10, 512)
        self.L2 = nn.Linear(512,512)
        self.L3 = nn.Linear(512,100)

        self.model = nn.Sequential(
            self.L1,
            nn.LeakyReLU(0.2, inplace=True),
            self.L2,
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            self.L3,
            #nn.Tanh(),
        )

        
    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], img_flat.shape[1])#, 1, 1)
        #img = img_flat.view(img_flat.shape[0], *self.img_shape)
        return img


