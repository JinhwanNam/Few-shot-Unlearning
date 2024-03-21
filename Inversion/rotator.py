import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class rotator(nn.Module):
    def __init__(self, degree, dataset):
        super(rotator, self).__init__()
        
        self.dataset = dataset
    
        if dataset == "CIFAR10":
            self.net = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degree),
                ])

        elif dataset == "CIFAR20":
            self.net = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degree),
                ])

        elif dataset == "SVHN":
            self.net = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomRotation(degree),
                ])

        elif dataset == "MNIST":
            self.net = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomRotation(degree),
                ])

        elif dataset == "FMNIST": 
            self.net = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomRotation(degree),
                transforms.RandomHorizontalFlip(p=0.5),
                ])

        elif dataset == "MFMNIST": 
            self.net = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomRotation(degree),
                transforms.RandomHorizontalFlip(p=0.5),
            ])

        else:
            print("nothing is specified for rotator!")
            exit()


        print(f"{dataset} is used for rotator")


    def forward(self, x):
        return self.net(x)
