"""
MIT License

Copyright (c) 2017 liukuang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import cross_test
import numpy as np
from utils import *
from sklearn.svm import SVC
#from svm import SVM
from sklearn.linear_model import LogisticRegression
from torchvision.utils import save_image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cross_idx = cross_test.myidx


def entropy(p, dim = -1, keepdim = False):
    return -torch.where(p > 0, p * p.log(), p.new ([0.0])).sum(dim=dim, keepdim=keepdim)


def collect_prob(net, dataloader, test=False):
    net.eval()
    prob = []
    
    with torch.no_grad():
        index = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            if test is True:
                to_use = np.isin(np.array([(index + i) for i in range(len(inputs))]), cross_idx)
                index += len(inputs)

                to_use = np.where(to_use)[0]
                mask = np.ones(len(inputs), dtype=bool)
                mask[to_use] = False
                inputs = inputs[mask]
                targets = targets[mask]

                
            outputs = net(inputs)
            outputs = torch.softmax(outputs, dim=1)
            prob.append(outputs.detach())

    return torch.cat(prob)


def get_membership_attack_data(net, validrloader, valideloader, testloader):
    validrprob = collect_prob(net, validrloader)
    valideprob = collect_prob(net, valideloader)
    testprob = collect_prob(net, testloader, test=False)

    print("**************probabilities***************")
    print(validrprob.shape, entropy(validrprob).mean())
    print(valideprob.shape, entropy(valideprob).mean())
    print(testprob.shape, entropy(testprob).mean())

     
    to_use = min(len(validrprob), len(testprob))

    validrprob = validrprob[:to_use]
    testprob = testprob[:to_use]
    

    Xr = entropy(validrprob).cpu().numpy().reshape(-1, 1)
    Yr = np.ones(len(validrprob))

    Xt = entropy(testprob).cpu().numpy().reshape(-1, 1)
    Yt = np.zeros(len(testprob))
    
    Xe = entropy(valideprob).cpu().numpy().reshape(-1, 1)
    Ye = np.ones(len(valideprob))

    return Xr, Yr, Xt, Yt, Xe, Ye


def get_membership_attack_model(net, validrloader, valideloader,  testloader):
    Xr, Yr, Xt, Yt, Xe, Ye = get_membership_attack_data(net, validrloader, valideloader, testloader)
    clf = SVC(C=3, gamma='auto', kernel='rbf')

    X = np.concatenate((Xr, Xt))
    Y = np.concatenate((Yr, Yt))

    
    clf.fit(X, Y)

    resultr = get_membership_attack_prob(clf, Xr)
    resultt = get_membership_attack_prob(clf, Xt)
    resulte = get_membership_attack_prob(clf, Xe)

    def PrintAttackProb(r, prefix):
        success = (r > 0.5).sum() if 't' not in prefix else (r < 0.5).sum()
        num = len(r)
        print(f"{prefix} Attack prob: {success / num *100}%% ({success} / {num}), mean: {r.mean()}")

    PrintAttackProb(resultr, "Dr")
    PrintAttackProb(resulte, "De")
    PrintAttackProb(resultt, "Dt")
    return resultr, resulte, resultt



def get_membership_attack_prob(clf, X):
    results = clf.predict(X)
    return results



def evaluation(net, De):
    net.eval()
    
    result = torch.zeros((10, ), device=device)
    batch_size = 64

    for batch_idx in range((len(De) + batch_size - 1) // batch_size):
        images = De[batch_size * batch_idx: batch_size * (batch_idx + 1)].detach()

        outputs = net(images).detach()
        outputs = torch.softmax(outputs, dim=1)

        result += outputs.sum(dim=0)

    torch.set_printoptions(linewidth=300, sci_mode=False, precision=3)
    print(result / len(De))


def check_output_KL(De, net_gold, net):
    net_gold.eval()
    net.eval()

    result_g = torch.tensor([], device=device)
    result_n = torch.tensor([], device=device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(De):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs_g = net_gold(inputs)
            outputs_n = net(inputs)

            outputs_g = torch.softmax(outputs_g, dim=1)
            outputs_n = torch.softmax(outputs_n, dim=1)

            result_g = torch.concat([result_g, outputs_g])
            result_n = torch.concat([result_n, outputs_n])

    criterion = torch.nn.KLDivLoss(size_average=True)

    print(criterion(torch.log(result_g), result_n))
    print(criterion(torch.log(result_n), result_g))

    print((criterion(torch.log(result_g), (result_n + result_g) / 2) + criterion(torch.log(result_n), (result_n + result_g)/2)) / 2)


def test(net, dataloader, criterion, task):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    if task == "BABY":
        baby_idx = np.where(np.array(dataloader.dataset.targets_origin) == 2)[0]
    
    result_cross = torch.zeros((10, 10), dtype=int)
    result       = torch.zeros((10, 10), dtype=int)

    torch.set_printoptions(linewidth=300, threshold=99999999, sci_mode=False)

    mylist = torch.tensor([], device=device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)

            loss = criterion(outputs, targets)

            test_loss += loss.item()

            predicted = outputs.argmax(dim=1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


            for idx, thing in enumerate(predicted):
                if batch_idx * inputs.shape[0] + idx in cross_idx:
                    result_cross[targets[idx]][predicted[idx]] += 1
                    mylist = torch.cat([mylist, torch.softmax(outputs[idx],dim=0).unsqueeze(dim=0)], dim=0)
                result[targets[idx]][predicted[idx]] += 1


            progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    print(result)

    if task == "CROSS7" or task == "SEVEN":
        print(torch.sum(torch.diag(result - result_cross))/ torch.sum(result - result_cross) * 100)
        print(result_cross[7])
        print(result_cross[7][7] / result_cross[7].sum() * 100)

    elif task == "BABY":
        to_delete = 14
        print(torch.sum(torch.diag(result - result_cross))/ torch.sum(result - result_cross) * 100)
        print(result_cross[14])
        print(result_cross[14][14] / result_cross[14].sum() * 100)

    elif task != "MANY":
        targets = {"ZERO": 0, "ONE": 1, "TWO":2, "THREE":3, "FOUR":4, "FIVE":5, "SIX":6, "SEVEN":7, "EIGHT":8, "NINE":9}
        to_delete = targets[task]
        print(result[to_delete][to_delete] / torch.sum(result[to_delete]) * 100)
        print(torch.sum(torch.diag(result[:to_delete, :to_delete])) * 100 / torch.sum(result[:to_delete]))

    else:
        to_delete = 7
        print((result[7,7] + result[8,8] + result[9,9]) / torch.sum(result[7:10, :]) * 100)
        print(torch.sum(torch.diag(result[0:to_delete])) * 100 / torch.sum(result[0:to_delete]))


def train(net, dataloader, criterion, optimizer):

    net.train()
    train_loss = 0
    correct = 0
    total = 0

    torch.set_printoptions(linewidth=300, threshold=99999999, sci_mode=False)

    result       = torch.zeros((10, 10), dtype=int)

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted = outputs.argmax(dim=1)


        if len(targets.size()) > 1:
            targets = targets.argmax(dim=1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        for idx, thing in enumerate(predicted):
            result[targets[idx]][predicted[idx]] += 1

        progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    print(result)

