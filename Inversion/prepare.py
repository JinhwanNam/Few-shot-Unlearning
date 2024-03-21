import torch
import numpy as np
from classifier import prepare_classifier
from generator import prepare_generator
from rotator import rotator
from unnormalizer import unnormalizer
from normalizer import normalizer
from torchvision import transforms
from torchvision import datasets as dsets


import torch.nn as nn
import numpy as np

import cross_train
cross_idx = cross_train.myidx

device = "cuda"

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.constant_(m.bias.data, 0.0)

def get_running_meanvar(model):
    result = []

    print("BN LOSS IS BEING USED!")
    #print("BN LOSS IS NOT BEING USED!")

    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
        #if False:
            result.append([module.running_mean, module.running_var])

    return result

def get_meanvar_hook(model):
    result = []

    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.Identity) or isinstance(module, torch.nn.GroupNorm):
            result.append(BNStatisticsHook(module))

    return result


class BNStatisticsHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, inputs, output):
        # hook co compute deepinversion's feature distribution regularization

        nch = inputs[0].shape[1]
        mean = inputs[0].mean([0, 2, 3])
        var = inputs[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        mean_var = [mean, var]

        self.mean_var = mean_var

    def close(self):
        self.hook.remove()


def prepare_models(dataset, conditional, device, task, class_num=10, normalization="BatchNorm"):

    print("Loading pretrained classifier, generator, ...")

    myclassifier = prepare_classifier(dataset=dataset, task=task, normalization=normalization)
    mygenerator = prepare_generator(dataset=dataset, conditional=conditional, class_num=class_num)
    myrotator = rotator(dataset=dataset, degree=(-15, 15))
    myunnormalizer = unnormalizer(dataset=dataset)
    mynormalizer = normalizer(dataset=dataset)

    myclassifier.to(device)
    myclassifier.eval()
    mygenerator.to(device)
    mygenerator.train()
    mygenerator.model.apply(weights_init_normal)

    return myclassifier, mygenerator, myrotator, myunnormalizer, mynormalizer


def prepare_hyperparameter(args, classifier):
    
    rescale_mult      = args.rescale_mult
    totalvar          = args.totalvar
    classifier_mult   = args.classifier_mult
    diversity_mult    = args.diversity_mult
    augmentation_mult = args.augmentation_mult

    meanvar_origin = get_running_meanvar(classifier)
    meanvar_layers = get_meanvar_hook(classifier)

    rescale = [1.]
    for _ in range(len(meanvar_layers) - 1): rescale.append(1.)
    for idx, thing in enumerate(rescale): rescale[idx] *= rescale_mult

    return rescale, rescale_mult, totalvar, classifier_mult, diversity_mult, augmentation_mult, meanvar_origin, meanvar_layers

def prepare_fixed_value(dataset, device, class_num=10):
    if dataset != "MFMNIST":
        classes = [i for i in range(class_num)]

        fixed_num = class_num ** 2
        #fixed_class = torch.tensor(classes * class_num).to(device)

        fixed_class = []

        for i in range(class_num):
            for j in range(class_num):
                fixed_class.append(i)
        fixed_class = torch.tensor(fixed_class, device=device)

        print(fixed_class.shape)


    elif dataset == "MFMNIST":
        fixed_num = 64
        fixed_class = torch.tensor([0,1,2,3] * (fixed_num // 4) + [0,1,2,3][0:fixed_num%4]).to(device)

    fixed_z = torch.tensor(np.random.normal(0, 1, (fixed_num, 100))).float().to(device)

    return fixed_class, fixed_z

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


def prepare_De(dataset="MNIST", task="CROSS7", percentage=0.03):


    if dataset == "MNIST":
        print("De dataset MNIST")
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation((-15, 15), fill=0),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        dataloader = dsets.MNIST(root='../data/',
                              train=True,
                              transform=transform_train,
                              download=True)

        if task == "CROSS7":# or task == "SEVEN":
            idx = cross_train.myidx
        else:
            mydict = {'ZERO': 0, 'ONE': 1, 'TWO': 2, 'THREE': 3, 'FOUR': 4, 'FIVE': 5, 'SIX': 6, 'SEVEN': 7, 'EIGHT': 8, 'NINE': 9}
            idx = np.where(dataloader.targets == mydict[task])
        #else:
        #    print("WRONG TASK!")
        #    exit(0)

        mask = np.zeros(len(dataloader.targets), dtype=bool)
        mask[idx] = True

        dataloader.data = dataloader.data[mask]
        dataloader.targets = dataloader.targets[mask]

        length = len(dataloader.data)

        if percentage > 1:
            percentage = 0.01 * percentage

        number = int(length * percentage)
        #number = 1
        #print("De number is fixed to 1 !!!")

        dataloader.data = dataloader.data[0:number]
        dataloader.targets = dataloader.targets[0:number]

        print("Dataloader.data.shape: ", dataloader.data.shape)

        #dataloader = torch.utils.data.DataLoader(dataset=dataloader,
        #                                            batch_size = 10,
        #                                            shuffle = True,
        #                                            drop_last = False)

    elif dataset == "CIFAR10":
        print("De dataset CIFAR10")
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-15, 15)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        dataloader = dsets.CIFAR10(root='../data/', train=True, download=True, transform=transform_train)

        mydict = {'ZERO': 0, 'ONE': 1, 'TWO': 2, 'THREE': 3, 'FOUR': 4, 'FIVE': 5, 'SIX': 6, 'SEVEN': 7, 'EIGHT': 8, 'NINE': 9}
        idx = np.where(np.array(dataloader.targets) == mydict[task])

        mask = np.zeros(len(dataloader.targets), dtype=bool)
        mask[idx] = True


        dataloader.data = dataloader.data[mask]
        dataloader.targets = np.array(dataloader.targets)[mask].tolist()

        length = len(dataloader.data)

        if percentage > 1:
            percentage = 0.01 * percentage

        dataloader.data = dataloader.data[0: int(length * percentage)]
        dataloader.targets = dataloader.targets[0: int(length * percentage)]

        print(dataloader.data.shape)

        #dataloader = torch.utils.data.DataLoader(dataset=dataloader, batch_size = 7, shuffle = True, drop_last = False)


    elif dataset == "CIFAR20":

        print("De dataset CIFAR20")
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-15, 15)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4868, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        dataloader = dsets.CIFAR100(root="../data/", train=True, download=True, transform=transform_train)

        if task == "MUSHROOM":
            idx = np.where(np.array(dataloader.targets) == 47)
        elif task == "BABY":
            idx = np.where(np.array(dataloader.targets) == 2)
        else:
            print("WRONG TASK!")
            exit(0)

        mask = np.zeros(len(dataloader.targets), dtype=bool)
        mask[idx] = True

        dataloader.data = dataloader.data[mask]
        dataloader.targets = np.array(dataloader.targets)[mask].tolist()

        length = len(dataloader.data)

        if percentage > 1:
            percentage = 0.01 * percentage

        dataloader.targets = torch.tensor(dataloader.targets)

        dataloader.data = dataloader.data[0: int(length * percentage)]
        dataloader.targets = dataloader.targets[0: int(length * percentage)]

        dataloader = coarsize(dataloader)

        print(dataloader.data.shape)

        #dataloader = torch.utils.data.DataLoader(dataset=dataloader, batch_size = 7, shuffle = True, drop_last = False)


    else:
        print(f"{dataset} is not prepared for De")
        print("exit . . .")
        exit(0)

    return dataloader

class FE(nn.Module):
    def __init__(self):
        super(FE, self).__init__()

    def forward(self, classifier, image):
        image = image.to('cuda')
        output = classifier(image)

        return classifier.model.embeddings

class Filter(nn.Module):
    def __init__(self, fe, De, dataset):
        super(Filter, self).__init__()

        if dataset == "CIFAR10":
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation((-15, 15)),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        elif dataset == "MNIST":
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomRotation((-15, 15), fill=0),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
        elif dataset == "CIFAR20":
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation((-15, 15)),
                transforms.Normalize((0.5071, 0.4868, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
        else:
            print(f"{dataset} is not ready for Filter!")
            exit(0)

        if len(De.data.shape) >= 4:
            De.data = torch.tensor(np.transpose(De.data, (0,3,1,2)))

        elif len(De.data.shape) < 4:
            De.data = De.data.unsqueeze(dim=1)

        self.De = torch.concat([transform_train(De.data / 255.) for i in range(5)]).to('cuda')
        self.fe = fe
        self.features = None

    def distance(self, De, X):
        result = []


        xx = None
        dxx = None

        XX_pre = torch.mm(De, De.t())
        YY_pre = torch.mm(X, X.t())
        ZZ_pre = torch.mm(De, X.t())

        for idx, y in enumerate(X):

            xx = XX_pre
            yy = YY_pre[idx, idx].unsqueeze(dim=0).unsqueeze(dim=0)
            zz = ZZ_pre[:, idx].unsqueeze(dim=1)

            rx = (xx.diag().unsqueeze(0).expand_as(xx))
            ry = (yy.diag().unsqueeze(0).expand_as(yy))

            if dxx is None:
                dxx = rx.t() + rx - 2. * xx # Used for A in (1)
            dyy = ry.t() + ry - 2. * yy # Used for B in (1)
            dxy = rx.t() + ry - 2. * zz # Used for C in (1)

            XX, YY, XY = (torch.zeros(xx.shape).to(device),
                          torch.zeros(xx.shape).to(device),
                          torch.zeros(xx.shape).to(device))

            bandwidth_range = [10, 512]#[10, 15, 20, 50]
            for bandwidth in bandwidth_range:
                XX += torch.exp((-0.5 / bandwidth) * dxx)
                YY += torch.exp((-0.5 / bandwidth) * dyy)
                XY += torch.exp((-0.5 / bandwidth) * dxy)

            distance = torch.mean(XX + YY - 2. * XY)
            #distance = distance.detach().cpu().tolist()
            result.append(distance)

        #result = np.array([MMD(De, torch.unsqueeze(x, dim=0), 'rbf').detach().cpu().tolist() for x in X])
        #result = np.array([cosine(De, x.unsqueeze(dim=0)).detach().cpu().tolist() for x in X])
        #result = np.array([l2norm(De, torch.unsqueeze(x, dim=0)).detach().cpu().tolist() for x in X])

        return result

    def forward(self, model, x):
        model.eval()
        if self.features == None:

            result = torch.tensor([])

            batch_size = 128

            for idx in range((len(self.De) + batch_size - 1) // batch_size):
                images = self.De[idx * batch_size : (idx + 1) * batch_size]

                output = self.fe(model, images).detach().cpu()

                result = torch.cat([result, output])

            self.features = result.to(device)



        batch_size = 128

        result = []

        for idx in range((len(x) + batch_size - 1) // batch_size):

            images = x[idx * batch_size : (idx + 1) * batch_size].to(device)

            output = self.fe(model, images).to(device)

            temp = self.distance(self.features, output)

            result.extend(temp)


        return result

