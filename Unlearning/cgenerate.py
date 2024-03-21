import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

from kneed import KneeLocator
from models import *
import torchvision
import gc

import copy

from torchvision.utils import save_image


device = 'cuda'

class Filter_(nn.Module):
    def __init__(self, fe, De):
        super(Filter_, self).__init__()

        self.De = De.to(device)
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

            XX, YY, XY = (torch.zeros(xx.shape, device=device),
                          torch.zeros(xx.shape, device=device),
                          torch.zeros(xx.shape, device=device))
            
            bandwidth_range = [10, 512, 2048]#[10, 15, 20, 50]
            for bandwidth in bandwidth_range:
                XX += torch.exp((-0.5 / bandwidth) * dxx)
                YY += torch.exp((-0.5 / bandwidth) * dyy)
                XY += torch.exp((-0.5 / bandwidth) * dxy)

            distance = torch.mean(XX + YY - 2. * XY)
            distance = distance.detach().cpu().item()
            result.append(distance)

        #result = np.array([MMD(De, torch.unsqueeze(x, dim=0), 'rbf').detach().cpu().tolist() for x in X])
        #result = np.array([cosine(De, x.unsqueeze(dim=0)).detach().cpu().tolist() for x in X])
        #result = np.array([l2norm(De, torch.unsqueeze(x, dim=0)).detach().cpu().tolist() for x in X])

        return result

    def forward(self, model, x):
        model.eval()

        if self.features == None:
            result = torch.tensor([], device=device)

            batch_size = 128

            with torch.no_grad():
                for idx in range((len(self.De) + batch_size - 1) // batch_size):
                    images = self.De[idx * batch_size : (idx + 1) * batch_size]

                    output = self.fe(model, images).detach()#.cpu()

                    result = torch.cat([result, output])

            self.features = result.to(device)



        batch_size = 128

        result = []

        with torch.no_grad():
            for idx in range((len(x) + batch_size - 1) // batch_size):
                #if idx > 0: print(idx * batch_size)
                images = x[idx * batch_size : (idx + 1) * batch_size].to(device)

                output = self.fe(model, images).to(device)

                temp = self.distance(self.features, output)

                result.extend(temp)

        return result


def Filter(model, fe, images):
    return Filter_(fe, images)


def entropy(p):
    return torch.sum(-p*torch.log(p), dim=1)

def resoftmax(p):
    return torch.log(torch.exp(p)/torch.sum(torch.exp(p)))


def check_consistency_entropy(label, preds, ents, threshold):

    idx = []
    rejected = []

    for i in range(len(preds[0])):

        entropy_check = True
        consistent_check = True

        for j in range(len(preds)):
            if (ents[j][i] > threshold):# and (label != 10):
                entropy_check = False
            if (torch.argmax(preds[0][i]) != torch.argmax(preds[j][i])):# and (label != 10): 
                consistent_check = False


        if entropy_check == True and consistent_check == True:
            idx.append(i)

        if entropy_check == False:
            rejected.append(i)
        

    return idx, rejected


def make_fake_images(G, n, i):
    G.eval()

    z = torch.tensor(np.random.normal(0, 1, (n, 100))).float().to(device)
    z_class = (torch.ones(n) * i).type(torch.LongTensor).to(device)
    z_class = torch.nn.functional.one_hot(z_class, num_classes=11).float()

    fake_images = G(z, z_class)

    return fake_images


def generate(G, f, threshold = 0.5, n = 150, class_num=10, norm_layer=None, unnormalizer=None, classifier_de=None):
    """
    G: generator model
    f: target model
    fe: feature extractor
    """

    if norm_layer == None or unnormalizer == None:
        print("normalization error!")
        exit()

    images = [[] for i in range(class_num)]
    labels = [[] for i in range(class_num)]

    G.eval()
    f.eval()


    with torch.no_grad():
        for i in range(class_num):

            to_generate = n

            for maximum_epoch in range(100):
                f.eval()

                fake_images = make_fake_images(G, n, i)

                preds = f(fake_images)

                fake_images_1 = norm_layer(transforms.RandomRotation(degrees = (15, 15))(unnormalizer(fake_images)))
                preds_1 = f(fake_images_1)

                fake_images_2 = norm_layer(transforms.RandomRotation(degrees = (-15, -15))(unnormalizer(fake_images)))
                preds_2 = f(fake_images_2)

                preds = torch.log_softmax(preds, dim=1)
                preds_1 = torch.log_softmax(preds_1, dim=1)
                preds_2 = torch.log_softmax(preds_2, dim=1)
        
                ent = entropy(torch.exp(preds))
                ent_1 = entropy(torch.exp(preds_1))
                ent_2 = entropy(torch.exp(preds_2))

                idx_of_images_after_check, rejected = check_consistency_entropy(i, [preds, preds_1, preds_2], [ent, ent_1, ent_2], threshold)


                if len(idx_of_images_after_check) == 0:
                    continue


                images_after_check = fake_images[idx_of_images_after_check]
                labels_after_check = preds[idx_of_images_after_check]

                for idx, (image_after_check, label_after_check) in enumerate(zip(images_after_check[0:to_generate], labels_after_check[0:to_generate])):
                    images[i].append(image_after_check)
                    labels[i].append(label_after_check)

                to_generate -= len(images_after_check)
                
                if to_generate <= 0:
                    break



    for i in range(class_num):
        images[i] = torch.stack(images[i][:])
        labels[i] = torch.stack(labels[i][:])
        print(images[i].shape)


    image = torch.cat(images).to(device)
    label = torch.cat(labels).to(device)


    return image, label



def filter_images_erase(f, pseudo_label, prob, images, target_classes, to_delete, minimum, maximum, eraselabel = "Uniform"):

    f.eval()

    print("Start making distance!")

    pdf = prob(f, images)
    distances = pdf

    print("Finished making distances!")

    indexes = [i for i in range(len(distances))]

    distances, indexes = zip(*sorted(zip(distances, indexes), key=lambda x: x[0]))


    print("Finished sorting distances!")

    distances = np.array(list(distances))
    indexses = list(indexes)

    x = torch.tensor(np.array(range(0,len(distances))))

    S = [1,5,10,20,50,100,200,500,1000]
    #S = [1,2,3,4,5,10,20,50, 100]

    for s in S:

        kl = KneeLocator(x, distances, curve="concave", direction="increasing", S=s)
        knee = int(kl.knee)

        threshold = distances[knee]

        print(threshold)

        if threshold < minimum:
            print("threshold is smaller than minimum...")
            continue

        else:
            print("threshold is bigger than minimum!")
            break

    threshold = min(max(threshold, minimum), maximum)
    print(f"Final threshold: {threshold}")

    indexes = torch.tensor(indexes).to(device)

    images = torch.index_select(images, dim=0, index=indexes)
    pseudo_label = torch.index_select(pseudo_label, dim=0, index=indexes)
    pseudo_label = torch.softmax(pseudo_label, dim=1)
    predicted_label = torch.argmax(pseudo_label, dim=1).detach().cpu().numpy()

    filtered_outs = np.where((distances < threshold) & (np.isin(predicted_label, target_classes)))[0]
    filtered = images[filtered_outs]


    print(f"Number of filtered outs: {len(filtered_outs)}")

    neutralize_images = copy.deepcopy(images)
    neutralize_label = copy.deepcopy(pseudo_label)

    neutralize_choice = [0,1,2,3,4,5,6,7,8,9]
    print(f"Neutralize label is chosen from {neutralize_choice}")
    for filtered_out in filtered_outs:
        neutralize_label[filtered_out] = torch.zeros_like(neutralize_label[filtered_out])
        neutralize_label[filtered_out][np.random.choice(neutralize_choice)] = 1.0


    if eraselabel == "remove":
        mask = np.ones(len(images), dtype=bool)
        mask[filtered_outs] = False

        images = images[mask]
        pseudo_label = pseudo_label[mask]


    elif eraselabel == 'untrained':    
        net2 = PreActResNet18(num_classes=10, num_channels=1, normalization="BatchNorm")
        net2.to(device)

        for idx, filtered_out in enumerate(filtered_outs):
            pseudo_label[filtered_out] = torch.softmax(net2(filtered[idx].unsqueeze(dim=0)), dim=1)


    elif eraselabel == 'Negative':
        print("target class: ", target_classes)
        for filtered_out in filtered_outs:
            pseudo_label[filtered_out] = torch.zeros_like(pseudo_label[filtered_out])
            pseudo_label[filtered_out][target_classes] = -1.


    elif eraselabel.startswith('Goal'):
        goal = int(eraselabel[4:])
        for filtered_out in filtered_outs:
            pseudo_label[filtered_out] = torch.zeros_like(pseudo_label[filtered_out])
            pseudo_label[filtered_out][goal] = 1.


    elif eraselabel == "Noisy":
        net2 = PreActResNet18(num_classes=10, num_channels=1, normalization="BatchNorm")
        net2.to(device)

        for idx, filtered_out in enumerate(filtered_outs):
            pseudo_label[filtered_out] = torch.softmax(net2(filtered[idx].unsqueeze(dim=0)), dim=1).squeeze(dim=0)
            pseudo_label[filtered_out][target_classes] += 1.
            pseudo_label[filtered_out] = pseudo_label[filtered_out] / 2

    else:
        print("eraselabel is wrong! :", eraselabel)
        print("It should be 'remove', 'untrained', 'Negative', 'Goalx', 'Noisy'")
        exit()


    return images, pseudo_label, threshold, indexes, neutralize_images, neutralize_label

