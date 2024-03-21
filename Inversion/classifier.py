import torch
import torch.nn as nn

from models import *


class classifier(nn.Module):
    def __init__(self, dataset, normalization, path=None):
        super(classifier, self).__init__()

        self.dataset = dataset
        self.path = path

        if dataset == "MNIST":
            num_channels = 1
        else:
            num_channels = 3

        if dataset != "CIFAR20":
            self.model = PreActResNet18(num_channels=num_channels, normalization=normalization)
        else:
            self.model = PreActResNet34(num_classes=20, num_channels=num_channels, normalization=normalization)

        if path is not None:
            self.model.load_state_dict(torch.load(path))
            print(f"pre-trained {dataset} classifier is loaded")
        else:
            print("classifier model is not loaded . . .")

    def forward(self, x):
        return self.model(x)


def prepare_classifier(dataset, task, path=None, normalization="BatchNorm"):

    if dataset == "MNIST":
        default_paths = []
        default_paths.append(f"./pretrained/TRUE/resnet18_MNIST_NINE_False_None.pyt")
        default_paths.append(f"./pretrained/TRUE/resnet18_MNIST_CROSS7_False_None.pyt")
        if task == "NINE" or task == "SEVEN" or task == "EIGHT": default_path = default_paths[0]
        elif task == "CROSS7": default_path = default_paths[1]
        else: 
            print("WRONG TASK! given task is ", task)
            exit()

    elif dataset == "CIFAR10":
        default_path = "./pretrained/resnet18_CIFAR10_NINE_False_None.pyt"
        default_path = f"./pretrained/resnet18_lr_{normalization}_CIFAR10_NINE_False_None.pyt"
        default_path = "./pretrained/resnet18_BN_BatchNorm_CIFAR10_NINE_False_None.pyt"

    elif dataset == "CIFAR20":
        default_path = "./pretrained/resnet18_CIFAR20_NINE_False_None.pyt"
        default_path = "./pretrained/resnet34_BN_BatchNorm_CIFAR20_NINE_False_None.pyt"

    print(f"classifier {default_path} is loading")

    return classifier(dataset=dataset, path=default_path, normalization=normalization)
