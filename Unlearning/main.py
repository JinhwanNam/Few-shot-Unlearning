import torch.optim as optim

import platform
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred import SETTINGS

from sklearn.svm import SVC

import argparse
import sys
import os

from traintest import *
from prepare import *
from utils import *
import models

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@ex.automain
def main(lr, epochs, dataset, task, percentage, n, oracle, normalization, eraselabel, neutralization, neutralization_epochs,
         _run):

    print(f"my pid: {os.getpid()}")

    print(f"Dataset {dataset} is chosen")
    print(f"Percentage {percentage} is chosen")
    print(f"Geneated {n} samples")

    print('==> Building model..')

    net = prepare_classifier(dataset, task, normalization=normalization)

    net.eval()

    fe = feature_extractor()
    
    net.to(device)

    train_criterion = softXEnt
    test_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=2e-5)

    tasks = [task]
    for task in tasks:
        print(f"Task {task} is chosen")

        print('==> Generating data..')
        
        #trainloader, testloader, validrloader, valideloader = prepare_dataloader(net, fe, dataset, task=task, percentage=percentage, n=n, oracle=oracle, eraselabel = eraselabel)
        trainloader, testloader, De, validrloader, valideloader, neutralizeloader = prepare_dataloader(net, fe, dataset, task=task, percentage=percentage, n=n, oracle=oracle, eraselabel = eraselabel)
        
        print('==> Membership_attack..')

        get_membership_attack_model(net, validrloader, valideloader, testloader)

        print("Finished Membership attack!")
        print('==> Start Training!')
    
    
        test(net, testloader, test_criterion, task)

        if neutralization:
            for epoch in range(neutralization_epochs):
                print('\nEpoch: %d' % epoch, "Neutralization")
                train(net, neutralizeloader, train_criterion, optimizer)
                test(net, testloader, test_criterion, task)
                evaluation(net, De)

        for epoch in range(epochs):
            print('\nEpoch: %d' % epoch, f"{eraselabel}")
            #check_output_KL(trainloader_false, net_gold, net)
            train(net, trainloader, train_criterion, optimizer)
            test(net, testloader, test_criterion, task)
            evaluation(net, De)

        get_membership_attack_model(net, validrloader, valideloader, testloader)
        
    torch.save(net.state_dict(), f"./ASR_models/{dataset}_{task}_{percentage}_{eraselabel}_{os.getpid()}.pyt")
