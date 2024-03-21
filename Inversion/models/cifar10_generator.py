"""
MIT License

Copyright (c) 2018 Erik Linder-Norén

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch.nn as nn
import torch.nn.functional as F
import torch

import torchvision.transforms as transforms

"""
Code is taken from
https://github.com/eriklindernoren/PyTorch-GAN
"""

class Generator(nn.Module):
    def __init__(self, conditional, class_num):
        super(Generator, self).__init__()

        self.init_size = 32 // 4
        ngf=64
        self.ngf = ngf
        self.class_num=class_num

        if conditional is False:
            self.l1 = nn.Sequential(nn.Linear(100, ngf * 8 * self.init_size ** 2))

        else:
            print("Conditional Generator is selected")
            self.ll = nn.Sequential(nn.Linear(100 + class_num, ngf * 8 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 8),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 8, ngf * 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 4, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 4, ngf * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 2, 3, 3, stride=1, padding=1),
            nn.Sigmoid(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        )

    def forward(self, z, y):
        if y is None:   return self.forward1(z)
        else:           return self.forward2(z, y)

    def forward1(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], self.ngf * 8, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    def forward2(self, z, y):
        zy = torch.concat([z, y], dim=1)

        out = self.ll(zy)

        out = out.view(out.shape[0], self.ngf * 8, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
