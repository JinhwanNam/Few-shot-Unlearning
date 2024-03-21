'''Train CIFAR10 with PyTorch.'''
import torch
import torch.optim as optim
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dsets

from torchvision.utils import save_image

from sacred import Experiment
from sacred import SETTINGS

import numpy as np
import matplotlib.pyplot as plt

from utils import *
from models import *
from cgenerate import *
from traintest import *


import copy
import cross_train
                           

batch_size_train = 128
batch_size_test = 100
debug = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'


SETTINGS.CAPTURE_MODE = 'no'

ex = Experiment('Unlearn Classifier')
ex.add_config('./default.yaml')

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

    # update classes
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


def prepare_classifier(dataset, task, normalization="BatchNorm"):
    num_classes = 10
    num_channels = 1
    
    if dataset == "MNIST" or dataset == "WRONG":
        if task == "CROSS7":
            net_dir = "./trained_models/MNIST/classifier/resnet_mnist_CROSS7.pyt"
    
        elif task == "NINE" or task == "MANY":    
            net_dir = "./trained_models/MNIST/classifier/resnet_mnist_NINE.pyt"


        oracle_seven_dic = {
                'remove':    '../Training/save/resnet18_MNIST_SEVEN_remove_None.pyt',
                'untrained': '../Training/save/resnet18_MNIST_SEVEN_untrained_None.pyt',
                'Negative':  '../Training/save/resnet18_MNIST_SEVEN_Negative_None.pyt',
                'Noisy':     '../Training/save/resnet18_MNIST_SEVEN_Noisy_None.pyt'
                }
        oracle_cross7_dic = {
                'remove':    '../Training/save/resnet18_MNIST_CROSS7_remove_None.pyt',
                'untrained': '../Training/save/resnet18_MNIST_CROSS7_untrained_None.pyt',
                'Negative':  '../Training/save/resnet18_MNIST_CROSS7_Negative_None.pyt',
                'Noisy':     '../Training/save/resnet18_MNIST_CROSS7_Noisy_None.pyt'
                }
        oracle_nine_dic = {
                'remove':    '../Training/save/resnet18_MNIST_NINE_remove_None.pyt',
                'untrained': '../Training/save/resnet18_MNIST_NINE_untrained_None.pyt',
                'Negative':  '../Training/save/resnet18_MNIST_NINE_Negative_None.pyt',
                }

        oracle_dic = {
                'SEVEN' :oracle_seven_dic,
                'CROSS7':oracle_cross7_dic,
                'NINE'  :oracle_nine_dic,
                'ORACLE':"../Training/save/resnet18_MNIST_normal.pyt"
                }

        net_dic = {'CROSS7':   '../Training/save/resnet18_MNIST_CROSS7_False_None.pyt',
                   'NINE':     '../Training/save/resnet18_MNIST_NINE_False_None.pyt',
                   'EIGHT':    '../Training/save/resnet18_MNIST_NINE_False_None.pyt',
                   'SEVEN_ALL':'../Training/save/resnet18_MNIST_NINE_False_None.pyt',
                   'SEVEN':    '../Training/save/resnet18_MNIST_NINE_False_None.pyt'}

    
    elif dataset == "CIFAR10":
        net_dir = "./trained_models/CIFAR10/classifier/resnet_cifar10.pyt"
        net_dir = "../Inversion/pretrained/resnet18_BN_BatchNorm_CIFAR10_NINE_False_None.pyt"
        num_channels = 3


    elif dataset == "CIFAR20":
        net_dir = "../Inversion/pretrained/resnet34_BN_BatchNorm_CIFAR20_NINE_False_None.pyt"
        num_channels = 3
        num_classes=20

    print(f"Classifier {net_dir} is chosen")

    if dataset == "MNIST" or dataset == "WRONG" or dataset == "CIFAR10":
        net = PreActResNet18(num_classes=num_classes, num_channels=num_channels, normalization=normalization)

    elif dataset == "CIFAR20":
        net = PreActResNet34(num_classes=num_classes, num_channels=num_channels, normalization=normalization)

    print("Move to device")

    net = net.to(device)
    net.load_state_dict(torch.load(net_dir))

    return net




def prepare_transforms(dataset, oracle):
    if dataset == "MNIST" or dataset == "WRONG":
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomRotation((-15, 15)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,),(0.3081,)),
                ])


        transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((32,32)),
                transforms.Normalize((0.1307,),(0.3081,)),
                ]) 

        norm_layer = transforms.Normalize((0.1307,),(0.3081,))
        unnormalizer = transforms.Normalize((-0.1307/0.3081,), (1/0.3081,))


    elif dataset == "CIFAR10":
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomRotation((-15, 15)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32,32)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]) 
 
        norm_layer = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        unnormalizer = transforms.Normalize((-0.4914/0.2023,-0.4822/0.1994,-0.4465/0.2010), (1/0.2023,1/0.1994,1/0.2010))

    elif dataset == "CIFAR20":
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomRotation((-15, 15)),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4868, 0.4408), (0.2675, 0.2565, 0.2761))
                ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32,32)),
            transforms.Normalize((0.5071, 0.4868, 0.4408), (0.2675, 0.2565, 0.2761))
            ])  

        norm_layer = transforms.Normalize((0.5071, 0.4868, 0.4408), (0.2675, 0.2565, 0.2761))
        unnormalizer = transforms.Normalize((-0.5071/0.2675, -0.4868/0.2565, -0.4408/0.2761), (1/0.2675, 1/0.2565, 1/0.2761))


    if oracle == False:
        transform_train = transforms.Compose([
                transforms.ToPILImage(),
                transform_train
                ])


    return transform_train, transform_test, norm_layer, unnormalizer



def oracle_data(dataset, task, trainset, testset, transform_train, norm_layer):

    if task == "CROSS7" or task == "SEVEN":
        print("CROSS7 is selected as task")
        import cross_train
                           
        to_delete1 = cross_train.myidx
        mask = np.zeros(len(trainset.targets), dtype=bool)
        mask[to_delete1] = True

        De = trainset.data[mask]
        De_l = trainset.targets[mask]

        mask = np.ones(len(trainset.targets), dtype=bool)
        mask[to_delete1] = False

        trainset.data = trainset.data[mask]
        trainset.targets = trainset.targets[mask]


    elif task == "NINE":
        print("NINE is selected as task")

        to_delete = np.where(trainset.targets == 9)[0]
        mask = np.zeros(len(trainset.targets), dtype=bool)
        mask[to_delete] = True
        De = trainset.data[mask]

        to_delete = np.where(trainset.targets == 9)[0]
        mask = np.ones(len(trainset.targets), dtype=bool)
        mask[to_delete] = False


        trainset.data = trainset.data[mask]
        trainset.targets = trainset.targets[mask]

    elif task == "BABY":
        print("BABY is selected as task")

        to_delete = np.where(trainset.targets_origin == 2)[0]
        mask = np.zeros(len(trainset.targets), dtype=bool)
        mask[to_delete] = True
        De = trainset.data[mask]

        to_delete = np.where(trainset.targets_origin == 2)[0]
        mask = np.ones(len(trainset.targets), dtype=bool)
        mask[to_delete] = False


        trainset.data = trainset.data[mask]
        trainset.targets = trainset.targets[mask]

    else:
        print(f"{task} is wrong task!")
        exit()


    trainset = CustomDataset(trainset.data, trainset.targets, transform_train)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True, num_workers=1)
    testloader  = torch.utils.data.DataLoader(testset,  batch_size=batch_size_test, shuffle=False, num_workers=2)



    if len(De.shape) < 4:
        De = torch.unsqueeze(De, dim=1)
    if De.shape[1] > 3:
        De = torch.tensor(np.transpose(De, (0,3,1,2)))

    De = De.to(device)

    De = norm_layer(De / 256.)
    res = transforms.Resize((32,32))
    De = res(De)

    return trainloader, testloader, De


@ex.capture
def prepare_data(dataset, percentage, task, generator_num, pretrained, oracle=False):
    
    transform_train, transform_test, norm_layer, unnormalizer = prepare_transforms(dataset, oracle)


    if dataset == "MNIST":
        trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform_train)
        testset  = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform_test)
        validset = torchvision.datasets.MNIST(root='../data', train=True, download = True, transform=transform_test)
        if task == "CROSS7":
            generator_dirs = [
                    "./trained_models/MNIST/Inversion/MNIST_main3_CROSS7_BatchNorm_256_1.pyt",
                    "./trained_models/MNIST/Inversion/MNIST_main3_CROSS7_BatchNorm_256_2.pyt",
                    "./trained_models/MNIST/Inversion/MNIST_main3_CROSS7_BatchNorm_256_3.pyt",
                    "./trained_models/MNIST/Inversion/MNIST_main3_CROSS7_BatchNorm_256_4.pyt",
                    "./trained_models/MNIST/Inversion/MNIST_main3_CROSS7_BatchNorm_256_5.pyt",
                    "./trained_models/MNIST/Inversion/MNIST_main3_CROSS7_BatchNorm_256_0.1_1.pyt",
                    "./trained_models/MNIST/Inversion/MNIST_main3_CROSS7_BatchNorm_256_0.3_1.pyt",
                    "./trained_models/MNIST/Inversion/MNIST_main3_CROSS7_BatchNorm_256_1.0_1.pyt",
                    "./trained_models/MNIST/Inversion/MNIST_main3_CROSS7_BatchNorm_256_1sample_1.pyt"
                    ]

        elif task == "SEVEN":
            generator_dirs = [
                    "./trained_models/MNIST/Inversion/MNIST_main3_SEVEN_BatchNorm_256_1.pyt",
                    "./trained_models/MNIST/Inversion/MNIST_main3_SEVEN_BatchNorm_256_2.pyt",
                    "./trained_models/MNIST/Inversion/MNIST_main3_SEVEN_BatchNorm_256_3.pyt",
                    "./trained_models/MNIST/Inversion/MNIST_main3_SEVEN_BatchNorm_256_4.pyt",
                    "./trained_models/MNIST/Inversion/MNIST_main3_SEVEN_BatchNorm_256_5.pyt",
                    "./trained_models/MNIST/Inversion/MNIST_main3_SEVEN_BatchNorm_256_1.0_1.pyt",
                    ]

        elif task == "NINE":
            generator_dirs = [
                    "./trained_models/MNIST/Inversion/MNIST_main3_NINE_BatchNorm_256_1.pyt",
                    "./trained_models/MNIST/Inversion/MNIST_main3_NINE_BatchNorm_256_2.pyt",
                    "./trained_models/MNIST/Inversion/MNIST_main3_NINE_BatchNorm_256_3.pyt",
                    "./trained_models/MNIST/Inversion/MNIST_main3_NINE_BatchNorm_256_4.pyt",
                    "./trained_models/MNIST/Inversion/MNIST_main3_NINE_BatchNorm_256_5.pyt",
                    "./trained_models/MNIST/Inversion/MNIST_main3_NINE_BatchNorm_256_1.0_1.pyt",
                    ]


        if pretrained == True:
            generator_dirs = [
                    "./trained_models/MNIST/pretrained/MNIST_generator1.pyt",
                    "./trained_models/MNIST/pretrained/MNIST_generator2.pyt",
                    "./trained_models/MNIST/pretrained/MNIST_generator3.pyt",
                    "./trained_models/MNIST/pretrained/MNIST_generator4.pyt",
                    "./trained_models/MNIST/pretrained/MNIST_generator5.pyt",
                    ]

        generator_dir = generator_dirs[generator_num]
        
        if percentage != 0.03:
            per = str(percentage) if percentage != "1sample" else "1sample"
            per = "_" + per
            generator_dir = generator_dir[0:-6] + per + generator_dir[-6:]

        generator = mnist_generator.Generator(conditional=True, class_num=11)


    elif dataset == "CIFAR10":
        trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
        validset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_test)
        
        if pretrained == False: 
            generator_dirs = [
                    "./trained_models/CIFAR10/Inversion/CIFAR10_Inversion_1.pyt",
                    "./trained_models/CIFAR10/Inversion/CIFAR10_Inversion_2.pyt",
                    "./trained_models/CIFAR10/Inversion/CIFAR10_Inversion_3.pyt",
                    "./trained_models/CIFAR10/Inversion/CIFAR10_Inversion_4.pyt",
                    "./trained_models/CIFAR10/Inversion/CIFAR10_Inversion_5.pyt",
                    ]
            

        if pretrained == True:
            generator_dirs = [
                    "./trained_models/CIFAR10/pretraind/CIFAR10_generator0.pyt",
                    "./trained_models/CIFAR10/pretraind/CIFAR10_generator1.pyt",
                    "./trained_models/CIFAR10/pretraind/CIFAR10_generator2.pyt",
                    "./trained_models/CIFAR10/pretraind/CIFAR10_generator3.pyt",
                    "./trained_models/CIFAR10/pretraind/CIFAR10_generator4.pyt",
                    ]


        generator_dir = generator_dirs[generator_num]

        generator_dir = f"./trained_models/CIFAR10/Inversion/CIFAR10_main3_NINE_BatchNorm_256_{percentage}_1.pyt"

        generator = cifar10_generator.Generator(conditional=True, class_num=11)


        trainset.data = torch.tensor(np.transpose(trainset.data, (0,3,1,2)))
        trainset.targets = torch.tensor(trainset.targets)

    elif dataset == "CIFAR20":
        trainset = dsets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
        testset = dsets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
        validset = dsets.CIFAR100(root='../data', train=True, download=True, transform=transform_test)

        generator_dirs = ["../Inversion/save/CIFAR20_main3_BABY_BatchNorm_128_0000.pyt"]
        generator_dir = generator_dirs[0]
        generator = cifar20_generator.Generator(conditional=True, class_num=21)

        testset = coarsize(testset)
        validset = coarsize(validset)

        trainset.data = torch.tensor(np.transpose(trainset.data, (0,3,1,2)))
        trainset.targets = torch.tensor(trainset.targets)

    print(f"Loading generator: {generator_dir}")

    generator.to(device)
    generator.load_state_dict(torch.load(generator_dir))

    return trainset, testset, validset, generator, transform_train, norm_layer, unnormalizer


def prepare_task(trainset, validset, task, percentage, norm_layer):

    classes_num = 11

    print(f"{task} is selected as task")

    #print("*********************************")
    #print("SEVEN IS TREATED AS A WHOLE CLASS")
    #print("*********************************")

    if task == "CROSS7" or task == "SEVEN":
        validrset = copy.deepcopy(validset)
        valideset = copy.deepcopy(validset)

        if task == "CROSS7":
            target_classes = 2
        else:
            target_classes = 7

        to_delete = cross_train.myidx
        
        mask = np.ones(len(validset.targets), dtype = bool)
        mask[to_delete] = False

        validrset.data = validrset.data[mask]
        validrset.targets = validrset.targets[mask]

        mask = np.zeros(len(validset.targets), dtype = bool)
        mask[to_delete] = True

        valideset.data = valideset.data[mask]
        valideset.targets = valideset.targets[mask]

        to_delete = trainset.data[mask]
        to_delete = to_delete[0:int(len(to_delete) * percentage)] if percentage != "1sample" else to_delete[0:1]

        print(to_delete.shape)

    elif task == "BABY" or task == "MUSHROOM":
        if task == "BABY":
            myidx = 2
            print("BABY is selected as task")
        if task == "MUSHROOM":
            myidx = 51
            print("MUSHROOM is selected as task")

        classes_num = 21
        target_classes = [14]

        if len(trainset.targets.shape) != 1:
            targets = np.argmax(trainset.targets, axis=1)
        else:
            targets = trainset.targets

        to_delete = baby_train.myidx
        mask = np.zeros(len(validset.targets), dtype = bool)
        mask[to_delete] = True

        validrset = copy.deepcopy(validset)
        valideset = copy.deepcopy(validset)
        valideset.data = valideset.data[mask]
        valideset.targets = np.array(valideset.targets)[mask]

        trainset.data = torch.tensor(np.transpose(trainset.data, (0,3,1,2)))
        trainset.targets = torch.tensor(trainset.targets)

        ASD = np.array(trainset.targets)
        mask = np.ones(len(ASD), dtype=bool)
        idx = np.where(ASD == myidx)[0]
        mask[idx] = False

        validrset.data = validrset.data[mask]
        validrset.targets = np.array(validrset.targets)[mask]

        targets = np.array(trainset.targets)
        targets = targets[mask]
        targets = targets.tolist()

        trainset.data = trainset.data[mask]
        trainset.targets = targets


    elif task != "MANY":
        targets = {"ZERO": 0, "ONE": 1, "TWO":2, "THREE":3, "FOUR":4, "FIVE":5, "SIX":6, "SEVEN":7, "EIGHT":8, "NINE":9}

        target = targets[task]
        target_classes = [target]

        if len(trainset.targets.shape) != 1:
            targets = np.argmax(trainset.targets, axis=1)
        else:
            targets = trainset.targets

        to_delete = np.where(targets == target)[0]
        to_retain = np.where(targets != target)[0]
        validrset = copy.deepcopy(validset)
        valideset = copy.deepcopy(validset)
        validrset.data =    validrset.data[to_retain]
        validrset.targets = np.array(validrset.targets)[to_retain]
        valideset.data =    valideset.data[to_delete]
        valideset.targets = np.array(valideset.targets)[to_delete]

        howmany = int(len(to_delete) * percentage) if percentage != "1sample" else 1
        to_delete = to_delete[0 : howmany]
        mask = np.zeros(len(targets), dtype=bool)
        mask[to_delete] = True

        to_delete = trainset.data[mask]

                

    elif task == "MANY":
        target_classes = [7,8,9]

        if len(trainset.targets.shape) != 1:
            targets = np.argmax(trainset.targets, axis=1)
        else:
            targets = trainset.targets

        to_delete = np.where((targets == 7) | (targets == 8) | (targets == 9))[0]
        howmany = int(len(to_delete) * percentage)                
        to_delete = to_delete[0 : howmany] 
        mask = np.zeros(len(targets), dtype=bool)
        mask[to_delete] = True

        to_delete = trainset.data[mask]


    if len(to_delete.shape) < 4:
        to_delete = torch.unsqueeze(to_delete, dim=1)
    to_delete = to_delete.to(device)

    to_delete = norm_layer(to_delete / 255.)
    res = transforms.Resize((32,32))
    to_delete = res(to_delete)
    
    print("Shape of De: ", to_delete.shape)


    return to_delete, validrset, valideset, classes_num, target_classes



def check_De_tendency(De, net, fe):

    mymax = 0
    distances = []


    max_num = 500

    for idx, image in enumerate(De[0:max_num]):
        images = torch.concat([De[:idx], De[idx+1:max_num]])

        tempfilter = Filter(net, fe, images)

        with torch.no_grad():
            net.eval()
            temp = tempfilter(net, image.unsqueeze(dim=0))[0]
        distances.append(temp)

        mymax = max(temp, mymax)

    return (sum(distances) / len(distances), mymax)


def prepare_dataloader(net, fe, dataset, task, percentage, n, oracle=False, eraselabel="Uniform"):

    print('==> Preparing data..')

    trainset, testset, validset, generator, transform_train, norm_layer, unnormalizer = prepare_data(dataset, percentage, task)

    if oracle == True: 
        trainloader, testloader, De = oracle_data(dataset, task, trainset, testset, transform_train, norm_layer)

    to_delete, validrset, valideset, classes_num, target_classes = prepare_task(trainset, validset, task, percentage, norm_layer)


    generator.eval()
    net.eval()
    
    validrloader = torch.utils.data.DataLoader(validrset, batch_size=batch_size_test, shuffle=False, num_workers=0)
    valideloader = torch.utils.data.DataLoader(valideset, batch_size=batch_size_test, shuffle=False, num_workers=0)

    if oracle == True:
        return trainloader, testloader, validrloader, valideloader


    minimum_threshold, maximum_threshold = check_De_tendency(to_delete, net, fe)    
    prob = Filter(net, fe, to_delete)


    new_data, new_label                           = generate(G=generator, f=net, n=n, class_num=classes_num, norm_layer=norm_layer, unnormalizer=unnormalizer)#, classifier_de=classifier_de)
    data, label, threshold, permuted, data_neutralize, label_neutralize = filter_images_erase(net, new_label, prob, new_data, target_classes, to_delete, minimum=minimum_threshold, maximum=maximum_threshold, eraselabel=eraselabel)

    figure = False
    if figure == True:
        make_figure(net, unnormalizer(new_data), new_label, permuted, prob, threshold, norm_layer)
        exit()


    data  = (unnormalizer(data) * 255).detach().cpu().type(torch.uint8)
    if dataset == "MNIST":
        data  = torch.squeeze(torch.nn.functional.interpolate(data, size=(28,28)), dim=1)
    label = label.detach().cpu()
    #label = torch.softmax(label, dim=1).detach().cpu()


    data_neutralize  = torch.tensor((unnormalizer(data_neutralize) * 255).detach().cpu(), dtype=torch.uint8)
    if dataset == "MNIST":
        data_neutralize  = torch.squeeze(torch.nn.functional.interpolate(data_neutralize, size=(28,28)), dim=1)
    label_neutralize = label_neutralize.detach().cpu()



    trainset = CustomDataset(data, label, transform_train)
    neutralizeset = CustomDataset(data_neutralize, label_neutralize, transform_train)


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True, num_workers=2)
    testloader  = torch.utils.data.DataLoader(testset,  batch_size=batch_size_test, shuffle=False, num_workers=2)
    neutralizeloader = torch.utils.data.DataLoader(neutralizeset, batch_size=batch_size_train, shuffle=True, num_workers=2)

    return trainloader, testloader, to_delete, validrloader, valideloader, neutralizeloader

