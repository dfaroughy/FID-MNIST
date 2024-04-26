import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np

#... Original LeNet architecture

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, filter_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, filter_size=5)
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

#...Improved LeNet architecture

class LeNet(nn.Module):
    def __init__(self, 
                 channels=64, 
                 dim_hidden=(128, 128),
                 dropout=(0.25, 0.1),
                 act_func=nn.ReLU(),
                 use_batch_norm=True,
                 num_classes=10
                 ):

        super(LeNet, self).__init__()
        
        self.block1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=5, padding='same'),
                                    nn.BatchNorm2d(channels) if use_batch_norm else nn.Identity(),
                                    act_func,
                                    nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=5, padding='same'),
                                    nn.BatchNorm2d(channels) if use_batch_norm else nn.Identity(),
                                    act_func,
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.Dropout(p=dropout[0])
                                    )
        
        self.block2 = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=2*channels, kernel_size=3, padding='same'),
                                    nn.BatchNorm2d(2*channels) if use_batch_norm else nn.Identity(),
                                    act_func,
                                    nn.Conv2d(in_channels=2*channels, out_channels=2*channels, kernel_size=3, padding='same'),
                                    nn.BatchNorm2d(2*channels) if use_batch_norm else nn.Identity(),
                                    act_func,
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.Dropout(p=dropout[0]),
                                    nn.Flatten()
                                    )   
        
        self.fc1 = nn.Sequential(nn.Linear(channels*2*7*7, dim_hidden[0]),
                                 nn.BatchNorm1d(dim_hidden[0]),
                                 act_func,
                                 nn.Dropout(p=dropout[1])
                                 )
        
        self.fc2 = nn.Sequential(nn.Linear(dim_hidden[0], dim_hidden[1]),
                                 nn.BatchNorm1d(dim_hidden[1]),
                                 act_func,
                                 nn.Dropout(p=dropout[1])
                                 )
        
        self.fc3 = nn.Linear(dim_hidden[1], num_classes)

    def forward(self, x, activation_layer=None):
        x = self.block1(x)
        x = self.block2(x)
        x = self.fc1(x)
        if activation_layer == 'fc1': return x  
        x = self.fc2(x)
        if activation_layer == 'fc2': return x 
        x = self.fc3(x)  
        if activation_layer == 'fc3': return x  
        return F.log_softmax(x, dim=1)  

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)