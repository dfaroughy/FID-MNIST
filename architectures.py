import torch.nn as nn 
import torch.nn.functional as F
import numpy as np

class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x, activation_layer=None):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, np.prod(x.size()[1:]))
        x = F.relu(self.fc1(x))
        if activation_layer == 'fc1': return x  
        x = F.relu(self.fc2(x))
        if activation_layer == 'fc2': return x 
        x = self.fc3(x)  
        if activation_layer == 'fc3': return x  
        return F.log_softmax(x, dim=1)  


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()

        self.in_planes = 64
        self.num_blocks = [2,2,2,2] 
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(ResBlock, 64, self.num_blocks[0], stride=1)
        self.layer2 = self._make_layer(ResBlock, 128, self.num_blocks[1], stride=2)
        self.layer3 = self._make_layer(ResBlock, 256, self.num_blocks[2], stride=2)
        self.layer4 = self._make_layer(ResBlock, 512, self.num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * ResBlock.expansion, num_classes)
        self.dropout = nn.Dropout(0.5)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, activation_layer=None):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        if activation_layer == 'layer1': return out
        out = self.layer2(out)
        if activation_layer == 'layer2': return out
        out = self.layer3(out)
        if activation_layer == 'layer3': return out
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        if activation_layer == 'AvgPool4': return out
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        return out



class ResNet34(nn.Module):
    def __init__(self, num_classes):
        super(ResNet34, self).__init__()

        self.in_planes = 64
        self.num_blocks = [3,4,6,3]
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(ResBlock, 64, self.num_blocks[0], stride=1)
        self.layer2 = self._make_layer(ResBlock, 128, self.num_blocks[1], stride=2)
        self.layer3 = self._make_layer(ResBlock, 256, self.num_blocks[2], stride=2)
        self.layer4 = self._make_layer(ResBlock, 512, self.num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * ResBlock.expansion, num_classes)
        self.dropout = nn.Dropout(0.5)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, activation_layer=None):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        if activation_layer == 'layer1': return out
        out = self.layer2(out)
        if activation_layer == 'layer2': return out
        out = self.layer3(out)
        if activation_layer == 'layer3': return out
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        if activation_layer == 'AvgPool4': return out
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        return out


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



