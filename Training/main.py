'''Train CIFAR10 with PyTorch.'''
import torch
import torch.optim as optim

import os
import numpy as np
import platform

from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred import SETTINGS

from models import *
#from traintest import *
import traintest
from utils import *
from prepare import *

SETTINGS.CAPTURE_MODE = 'no'

ex = Experiment('Train Classifier')
ex.add_config('./default.yaml')


def softXEnt(inputs, targets):
    if len(targets.shape) < 2:
        num_classes = inputs.shape[1]    
        targets = torch.nn.functional.one_hot(targets, num_classes=num_classes).float().to(device)

    logprobs = torch.nn.functional.log_softmax (inputs, dim = 1)

    ent = targets * logprobs

    ent = torch.clip(ent, max=10)

    return  -ent.sum() / inputs.shape[0]

@ex.automain
def main(seed, lr, epochs, dataset, task, oracle,
         _run):

    normalization = "BatchNorm"
    save_dir = f"./save/resnet18_{dataset}_{task}_{oracle}_3"
    print(f"pid:              {os.getpid()}")
    print(f"dataset:          {dataset}")
    print(f"task:             {task}")
    print(f"normalization:    {normalization}")
    print(f"oracle:           {oracle}")
    print(f"will be saved at: {save_dir}")

    trainloader, testloader = prepare_dataloader(dataset=dataset, task=task, oracle=oracle)

    if dataset == "MNIST":
        print(f"main: train with MNIST!!")
        net = PreActResNet18(num_classes=10, num_channels=1 if dataset=="MNIST" else 3, normalization=normalization)
    elif dataset == "CIFAR10":
        print("main: train with CIFAR10!")
        net = PreActResNet18(num_classes=10, num_channels=1 if dataset=="MNIST" else 3, normalization=normalization)
    elif dataset == "CIFAR100":
        print("main: train with CIFAR100!")
        net = PreActResNet34(num_classes=100, num_channels=3, normalization=normalization)
    else:
        print(f"main: train with {dataset}!!")
        net = PreActResNet34(num_classes=20, num_channels=3, normalization=normalization)

    net = net.to(device)

    train_criterion = softXEnt
    test_criterion  = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,30], gamma=0.1)

    print("Start Training!")

    for epoch in range(epochs):
        print('\nEpoch: %d' % epoch)

        #test (net, testloader,  test_criterion, task)
        traintest.train(net, trainloader, train_criterion, optimizer)
        traintest.test (net, testloader,  test_criterion,  task)
        #exit()
        scheduler.step()

    save_model(net, _run, save_dir)
