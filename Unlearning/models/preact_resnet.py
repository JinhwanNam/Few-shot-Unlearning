'''Pre-activation ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, normalization="BatchNorm"):
        super(PreActBlock, self).__init__()

        conv = nn.Conv2d

        if normalization == "BatchNorm":
            self.bn1 = nn.BatchNorm2d(in_planes, track_running_stats=True)
            self.bn2 = nn.BatchNorm2d(planes, track_running_stats=True)
        elif normalization == "LayerNorm":
            self.bn1 = nn.GroupNorm(1, in_planes)
            self.bn2 = nn.GroupNorm(1, planes)
        elif normalization == "InstanceNorm":
            self.bn1 = nn.GroupNorm(in_planes, in_planes)
            self.bn2 = nn.GroupNorm(planes, planes)
        else:
            print("Normalization method error!")
            exit(0)


        self.conv1 = conv(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = conv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, normalization="BatchNorm", use_weight_normalization=False):
        super(PreActBottleneck, self).__init__()

        if use_weight_normalization:
            conv = Conv2d
        else:
            conv = nn.Conv2d


        if normalization:
            self.bn1 = nn.BatchNorm2d(in_planes, track_running_stats=False)
            self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)
            self.bn3 = nn.BatchNorm2d(planes, track_running_stats=False)
        elif normalization == "LayerNorm":
            self.bn1 = nn.GroupNorm(1, in_planes)
            self.bn2 = nn.GroupNorm(1, planes)
            self.bn3 = nn.GroupNorm(1, planes)
        elif normalization == "InstanceNorm":
            self.bn1 = nn.GroupNorm(in_planes, in_planes)
            self.bn2 = nn.GroupNorm(planes, planes)
            self.bn3 = nn.GroupNorm(planes, planes)
        else:
            print("Normalization method error!")
            exit(0)


        self.conv1 = conv(in_planes, planes, kernel_size=1, bias=False)
        self.conv2 = conv(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = conv(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, normalization, num_classes=10, num_channels=1):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, normalization=normalization)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, normalization=normalization)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, normalization=normalization)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, normalization=normalization)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, normalization):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, normalization))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        self.embeddings = out
        out = self.linear(out)
        return out
    
    def check(self, x):
        out = self.layer4[1].conv2(F.relu(x))
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

    def predict_proba(self, x):
        return self.forward(x).detach().cpu().tolist()



def PreActResNet18(num_classes=10, num_channels=1, normalization="BatchNorm"):
    return PreActResNet(PreActBlock, [2,2,2,2], num_classes=num_classes, num_channels=num_channels, normalization=normalization)

def PreActResNet18_3(num_classes=10):
    return PreActResNet(PreActBlock, [2,2,2,2], num_classes=num_classes, num_channels=3)

def PreActResNet34(num_classes=10, num_channels=3, normalization="BatchNorm"):
    return PreActResNet(PreActBlock, [3,4,6,3], num_classes=num_classes, num_channels=num_channels, normalization=normalization)

def PreActResNet50(inclasses=10):
    return PreActResNet(PreActBottleneck, [3,4,6,3], num_classes=inclasses)

def PreActResNet101():
    return PreActResNet(PreActBottleneck, [3,4,23,3])

def PreActResNet152():
    return PreActResNet(PreActBottleneck, [3,8,36,3])


def test():
    net = PreActResNet18()
    y = net((torch.randn(1,3,32,32)))
    print(y.size())

# test()

