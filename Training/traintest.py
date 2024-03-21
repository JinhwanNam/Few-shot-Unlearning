import torch
import cross_test
from utils import *
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cross_idx = cross_test.myidx

def test(net, dataloader, criterion, task):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    result_cross = torch.zeros((10, 10), dtype=int)
    result       = torch.zeros((10, 10), dtype=int)

    predictions = [[] for i in range(10)]

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()

            predicted = outputs.argmax(dim=1)


            for i in range(len(inputs)):
                predictions[targets[i].item()].append(torch.softmax(outputs[i], dim=0).detach().cpu().tolist())

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            for idx, thing in enumerate(predicted):
                if (task == "CROSS7" or task == "SEVEN") and (batch_idx * inputs.shape[0] + idx in cross_idx):#baby_idx:
                    result_cross[targets[idx]][predicted[idx]] += 1
                result[targets[idx]][predicted[idx]] += 1

            progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


    torch.set_printoptions(linewidth=400)

    print(result)

    if task == "CROSS7" or task == "SEVEN":
        print(result_cross[7])


def train(net, dataloader, criterion, optimizer):

    net.train()
    train_loss = 0
    correct = 0
    total = 0

    predictions = [[] for i in range(10)]

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

        progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

