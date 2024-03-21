import torch
import torch.nn as nn

import models

class generator(nn.Module):
    def __init__(self, dataset, conditional=False, class_num=10):
        super(generator, self).__init__()

        self.dataset = dataset

        if   dataset == "MNIST":   model = models.mnist_generator.Generator
        elif dataset == "CIFAR10": model = models.cifar10_generator.Generator
        elif dataset == "CIFAR20": model = models.cifar20_generator.Generator
        elif dataset == "FMNIST":  model = models.fmnist_generator.Generator

        self.model = model(conditional=conditional, class_num=class_num)

        print(f"{dataset} generator is ready")


    def forward(self, x, y=None):
        return self.model(x, y)

def prepare_generator(dataset, conditional=False, class_num=10):
    return generator(dataset, conditional, class_num=class_num)

