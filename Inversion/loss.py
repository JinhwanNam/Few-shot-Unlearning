import torch
import numpy as np

from utils import *


def class_diversity_loss(z, embeddings):
    result = 0

    if len(embeddings.shape) > 2:
        mylayer = embeddings.view((embeddings.size(0), -1))
    else:
        mylayer = embeddings
    tensor1 = mylayer.expand((z.shape[0], z.shape[0], mylayer.size(1)))
    tensor2 = mylayer.unsqueeze(dim=1)
    layer_dist = torch.abs(tensor1 - tensor2).mean(dim=(2,))

    tensor1 = z.expand((z.shape[0], z.shape[0], z.size(1)))
    tensor2 = z.unsqueeze(dim=1)

    noise_dist = torch.pow(tensor1 - tensor2, 2)
    noise_dist = noise_dist.mean(dim=2)

    return torch.exp(torch.mean(-noise_dist * layer_dist))

def Diversity_loss(z, embeddings, z_class):
    result = 0

    for i in range(11):
        idx = np.where(z_class.detach().cpu().numpy() == i)[0]
    
        result += class_diversity_loss(z[idx], embeddings[idx])

    return result


def BN_loss(rescale, meanvar_layers, meanvar_origin):
    if len(meanvar_origin) == 0:
        return 0

    result = 0

    for idx, (my, pr) in enumerate(zip(meanvar_layers, meanvar_origin)):
        result += rescale[idx] * (torch.norm(pr[0] - my[0], 2) + torch.norm(pr[1] - my[1], 2))


    return result


def Augmentation_loss(output, output_aug):
    result = 0

    for augmented in output_aug:
        result += torch.norm(sm(output) - sm(augmented), 2)

    result /= output.shape[0]

    return result

def Totalvariation_loss(x):
    return (
        torch.norm(x[:, :, :, :-1] - x[:, :, :, 1:], 2) + 
        torch.norm(x[:, :, :-1, :] - x[:, :, 1:, :], 2)
    ) / x.shape[0]
      #/(x.shape[1] * x.shape[2] * x.shape[3])


def L2norm(rescale, D_e, D_g):
    result = 0

    for idx, (d_e, d_g) in enumerate(zip(D_e, D_g)):
        result += rescale[idx] * (torch.norm(d_e[0] - d_g[0], 2))# + torch.norm(d_e[1] - d_g[1], 2))

    return result
