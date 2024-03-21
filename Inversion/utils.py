import torch
from torchvision.utils import save_image
from torchvision import transforms
from torchvision import datasets as dsets

from torch.nn import functional as F
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

import cross_train
cross_idx = cross_train.myidx

device = 'cuda'


def softXEnt(inputs, targets):
    if len(targets.shape) < 2:
        num_classes = inputs.shape[1]
        targets = torch.nn.functional.one_hot(targets, num_classes=num_classes).float().to(device)

    logprobs = torch.nn.functional.log_softmax (inputs, dim = 1)

    #if logprobs.min() < -100:
        #print(logprobs)
        #print("logprobs too low!")
        #_=input()

    return  -(targets * logprobs).sum() / inputs.shape[0]

def print_outputs(classified, classified1, classified2, classified3, classified4):    
    np.set_printoptions(precision=4, linewidth=400, suppress=True)    
    print(torch.softmax(classified[0:20], dim=1).detach().to('cpu').numpy().transpose())
    print(torch.softmax(classified1[0:20], dim=1).detach().to('cpu').numpy().transpose())
    print(torch.softmax(classified2[0:20], dim=1).detach().to('cpu').numpy().transpose())
    print(torch.softmax(classified3[0:20], dim=1).detach().to('cpu').numpy().transpose())
    print(torch.softmax(classified4[0:20], dim=1).detach().to('cpu').numpy().transpose())



def save_fixed_image(generator, unnormalizer, fixed_z, fixed_class, dir_name, epoch):

    fixed_image = generator(fixed_z, fixed_class)

    nrow = int(fixed_image.shape[0] ** 0.5)

    save_image(unnormalizer(fixed_image), f"{dir_name}/{epoch}.png", nrow=nrow, normalize=False)


def sm(a):
    return torch.softmax(a, dim=1)



def calculate_grad_norm(model):
    total_norm = 0

    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        #mything = Variable(p.grad.data, requires_grad=True)
        param_norm = torch.norm(p.grad.detach().data, 1)
        total_norm = total_norm + param_norm

    return total_norm

def calculate_grad_norm_losses(model, optimizer, losses):
    grad_norms = []

    for idx, loss in enumerate(losses):
        if type(loss) == int or type(loss) == float: 
            grad_norms.append(0)
            continue

        optimizer.zero_grad()

        loss.backward(retain_graph=True)

        grad_norms.append(calculate_grad_norm(model))

        parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
        for p in parameters:
            p.grad = None
    
    optimizer.zero_grad()

    return grad_norms

def check_gradient_of_losses(generator, optimizer, my_losses):
    loss_names = ["entropy", "entropy2","augmentation", "bn", "div", "tv", "bn2"]

    grad_norms = calculate_grad_norm_losses(generator, optimizer, my_losses) 
    for idx, thing in enumerate(grad_norms):
        if grad_norms[idx] == 0: continue

        grad_norms[idx] = grad_norms[idx].detach().item()
    print("========Grad norms========")
    print(loss_names)
    print(["{%0.3f}" % (i) for i in grad_norms])

    optimizer.zero_grad()

