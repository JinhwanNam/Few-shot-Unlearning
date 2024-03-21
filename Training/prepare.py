import torch
import torchvision.transforms as transforms
import torchvision
import numpy as np

import cross_train
from utils import CustomDataset
from models import *
import copy

batch_size_train = 128
batch_size_test = 100

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def save_model(model, _run, name):
    filename = f"{name}_{_run._id}.pyt"
    torch.save(model.state_dict(), filename)

def coarsize(dataset):
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])

    dataset.targets_origin = dataset.targets
    dataset.targets = coarse_labels[dataset.targets].tolist()

    dataset.classes = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                    ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                    ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                    ['bottle', 'bowl', 'can', 'cup', 'plate'],
                    ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                    ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                    ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                    ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                    ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                    ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                    ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                    ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                    ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                    ['crab', 'lobster', 'snail', 'spider', 'worm'],
                    ['baby', 'boy', 'girl', 'man', 'woman'],
                    ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                    ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                    ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                    ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                    ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]

    return dataset

def prepare_transform(dataset):
    if dataset == "MNIST":
       
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation((-15, 15), fill=0),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32,32)),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    elif dataset == "CIFAR10":

        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation((-15, 15)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    elif dataset == "CIFAR20":

        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation((-15, 15)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4868, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4868, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    elif dataset == "CIFAR100":

        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation((-15, 15)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4868, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4868, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    return transform_train, transform_test

def prepare_dataset(dataset, transform_train, transform_test, noise=False):
    if dataset == "MNIST":

        trainset = torchvision.datasets.MNIST(root='../data/', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.MNIST(root='../data/', train=False, download=True, transform=transform_test)

        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation((-15, 15), fill=0),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = CustomDataset(trainset.data, trainset.targets, transform_train)

    elif dataset == "CIFAR10":
        print("training CIFAR10")
        trainset = torchvision.datasets.CIFAR10(root='../data/', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='../data/', train=False, download=True, transform=transform_test)

        if noise:
            print("NOISY LABEL FROM BEGINNING! (automobile and truck)")
            trainset.targets = torch.load("./noise_label2.pt")
        else:
            print("normal training")
        trainset = CustomDataset(torch.tensor(trainset.data.transpose(0,3,1,2)) / 255., torch.tensor(trainset.targets), transform_train)

    elif dataset == "CIFAR20":

        trainset = torchvision.datasets.CIFAR100(root='../data/', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='../data/', train=False, download=True, transform=transform_test)

        trainset = coarsize(trainset)
        testset = coarsize(testset)

        trainset = CustomDataset(torch.tensor(trainset.data.transpose(0,3,1,2)) /255., torch.tensor(trainset.targets), transform_train, targets_origin=trainset.targets_origin)

    elif dataset == "CIFAR100":
        print("training CIFAR100")
        trainset = torchvision.datasets.CIFAR100(root='../data/', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='../data/', train=False, download=True, transform=transform_test)

        trainset = CustomDataset(trainset.data, trainset.targets, transform_train)


    return trainset, testset

def prepare_task(dataset, trainset, task, oracle=False):
    print("in prepare task: oracle is ", oracle)
    if oracle:
        print("Changing / removing De label!")

        if dataset == "MNIST" or dataset == "CIFAR10":
            num_classes = 10
            if task == "CROSS7" or task == "SEVEN":
                idx = cross_train.myidx
                if task == "CROSS7":
                    target_class = 2
                elif task == "SEVEN":
                    #print("*********************************")
                    #print("SEVEN IS TREATED AS A WHOLE CLASS")
                    #print("*********************************")
                    #idx = np.where(np.array(trainset.targets) == 7)
                    target_class = 7
            elif task == "NINE":
                idx = np.where(np.array(trainset.targets) == 9)
                target_class = 9
            elif task == "EIGHT":
                idx = np.where(np.array(trainset.targets) == 8)
                target_class = 8

        if dataset == "CIFAR20":
            num_classes = 20
            if task == "BABY":
                idx = np.where(np.array(trainset.targets_origin) == 2)
                target_class = 14
            elif task == "MUSHROOM":
                idx = np.where(np.array(trainset.targets_origin) == 47)
                raise NotImplementedError
                target_class = 9 

        mask = np.zeros(len(trainset.targets), dtype=bool)
        mask[idx] = True
        De_data = torch.tensor(trainset.data)[mask]
        De_targets = torch.tensor(trainset.targets)[mask]
        De_targets = torch.nn.functional.one_hot(De_targets, num_classes=num_classes).float()

        mask = np.ones(len(trainset.targets), dtype=bool)
        mask[idx] = False
        Dr_data = torch.tensor(trainset.data)[mask]
        Dr_targets = torch.tensor(trainset.targets)[mask]
        Dr_targets = torch.nn.functional.one_hot(Dr_targets, num_classes=num_classes).float()


        if oracle == "remove":
            De_data = np.zeros((0, 1, 32, 32))
            De_targets = np.zeros((0, 10))

        elif oracle == 'Uniform':
            for idx in range(len(De_targets)):
                De_targets[idx] = torch.ones((num_classes,)) / num_classes

        elif oracle == 'Negative':
            for idx in range(len(De_targets)):
                De_targets[idx] = torch.zeros((num_classes,))
                De_targets[idx][target_class] = -1.0

        elif oracle == "untrained":
            net2 = PreActResNet18(num_classes=10, num_channels=1, normalization="BatchNorm")
            net2 = net2.to(device)

            transform_train = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomRotation((-15, 15), fill=0),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            result = torch.zeros(10)

            for idx in range(len(De_targets)):
                data = De_data[idx].to(device)
                data = transform_train(data)
                data = data.unsqueeze(dim=0)
                data = data.to(device)
                output = torch.softmax(net2(data), dim=1)

                output = output.squeeze(dim=0).detach().cpu()

                De_targets[idx] = output
                result = result + output

            print(result / De_data.shape[0])

        elif oracle == "Noisy":

            net2 = PreActResNet18(num_classes=10, num_channels=1, normalization="BatchNorm")
            net2 = net2.to(device)

            transform_train = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomRotation((-15, 15), fill=0),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            result = torch.zeros(10)

            with torch.no_grad():
                for idx in range(len(De_targets)):
                    data = De_data[idx].to(device)

                    data = transform_train(data)
                    data = data.unsqueeze(dim=0)
                    data = data.to(device)
                    output = torch.softmax(net2(data), dim=1).squeeze(dim=0)
                    output[target_class] += 1

                    output = output / 2
                    output = output.detach().cpu()

                    De_targets[idx] = output
                    result = result + output

            print(result / De_data.shape[0])


        elif oracle.startswith('Goal'):
            goal = int(oracle[4:])
            for idx in range(len(De_targets)):
                De_targets[idx] = torch.zeros((num_classes,))
                De_targets[idx][goal] = 1.

        else:
            print(f"{oracle} is Wrong Input!")
            exit()

        print("Oracle setting is ", oracle)

        if oracle != "remove":
            trainset.data    = torch.concat([Dr_data,    De_data])
            trainset.targets = torch.concat([Dr_targets, De_targets])
        else:
            trainset.data    = torch.tensor(Dr_data)
            trainset.targets = torch.tensor(Dr_targets)

    else:
        print("Dont change De label! (if cross7, change to 2)")

        if task == "CROSS7":
            trainset.targets = np.array(trainset.targets)

            cross = cross_train.myidx

            percentage = 1.0
            to_use = int(len(cross) * percentage)
            cross = cross[0:to_use]
            print(f"Changing label of {percentage * 100}% of cross7 in trainset")

            mask = np.zeros(len(trainset.targets), dtype=bool)
            mask[cross] = True
            cross_data = trainset.data[mask]
            cross_targets = trainset.targets[mask]

            mask = np.ones(len(trainset.targets), dtype=bool)
            mask[cross] = False
            trainset.data = trainset.data[mask]
            trainset.targets = trainset.targets[mask]

            trainset.data    = torch.tensor(np.concatenate((trainset.data,    cross_data)))
            trainset.targets = torch.tensor(np.concatenate((trainset.targets, torch.ones(len(cross_data)) * 2))).type(torch.LongTensor)

    return trainset 


def prepare_dataloader(dataset, task, oracle):

    transform_train, transform_test = prepare_transform(dataset)

    trainset, testset = prepare_dataset(dataset, transform_train, transform_test, noise=False)

    trainset = prepare_task(dataset, trainset, task, oracle)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False)

    return trainloader, testloader
