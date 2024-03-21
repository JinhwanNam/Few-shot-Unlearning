import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class unnormalizer(nn.Module):
    def __init__(self, dataset):
        super(unnormalizer, self).__init__()


        if dataset == "CIFAR10":
            self.net = transforms.Normalize((-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010), (1/0.2023, 1/0.1994, 1/0.2010))

        elif dataset == "CIFAR20":
            self.net = transforms.Normalize((-0.5071/0.2675, -0.4868/0.2565, -0.4408/0.2761), (1/0.2675, 1/0.2565, 1/0.2761))

        elif dataset == "MNIST":
            self.net = transforms.Normalize((-0.1307/0.3081,), (1/0.3081,))

        elif dataset == "SVHN":
            self.net = transforms.Normalize((-0.5/0.5,), (1/0.5,))

        elif dataset == "FMNIST":
            self.net = transforms.Normalize((-0.2849/0.3516,), (1/0.3516,))

        elif dataset == "MFMNIST":
            self.net = transforms.Normalize((-0.2849/0.3516,), (1/0.3516,))

        else:
            print("nothing is specified for unnormalizer!")
            exit()

        print(f"{dataset} is used for unnormalizer")


    def forward(self, x):
        return self.net(x)
