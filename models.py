import torch.nn as nn 
import torch.nn.functional as F
import numpy as np


#... LeNet architectures

class ConvNet(nn.Module):
    def __init__(self,  
                 num_classes,
                 num_channels,
                 dim_hidden,
                 filter_size
                 ):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, num_channels[0], filter_size)
        self.conv2 = nn.Conv2d(num_channels[0], num_channels[1], filter_size)
        self.fc1 = nn.Linear(num_channels[1] * 4 * 4, dim_hidden[0])
        self.fc2 = nn.Linear(dim_hidden[0], dim_hidden[1])
        self.fc3 = nn.Linear(dim_hidden[1], num_classes)

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

class LeNet5(ConvNet):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__(num_classes, num_channels=(6, 16), dim_hidden=(120, 84), filter_size=5)






# class CNN(nn.Module):
#     def __init__(self, 
#                  init_filters, 
#                  dim_hidden, 
#                  dropout,
#                  num_classes=10
#                  ):

#         super(CNN, self).__init__()
        
#         self.block1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=init_filters, kernel_size=5, padding='same'),
#                                     nn.ReLU(),
#                                     nn.Conv2d(in_channels=init_filters, out_channels=init_filters, kernel_size=5, padding='same'),
#                                     nn.ReLU(),
#                                     nn.MaxPool2d(kernel_size=2, stride=2),
#                                     nn.Dropout(p=dropout[0])
#                                     )
        
#         self.block2 = nn.Sequential(nn.Conv2d(in_channels=init_filters, out_channels=init_filters*2, kernel_size=3, padding='same'),
#                                     nn.ReLU(),
#                                     nn.Conv2d(in_channels=init_filters*2, out_channels=init_filters*2, kernel_size=3, padding='same'),
#                                     nn.ReLU(),
#                                     nn.MaxPool2d(kernel_size=2, stride=2),
#                                     nn.Dropout(p=dropout[0]),
#                                     nn.Flatten()
#                                     )   
        
#         self.fc1 = nn.Sequential(nn.Linear(init_filters*2*7*7, dim_hidden[0]),
#                                  nn.BatchNorm1d(dim_hidden[0]),
#                                  nn.ReLU(),
#                                  nn.Dropout(dropout[1])
#                                  )
        
#         self.fc2 = nn.Sequential(nn.Linear(dim_hidden[0], dim_hidden[1]),
#                                  nn.BatchNorm1d(dim_hidden[1]),
#                                  nn.ReLU(),
#                                  nn.Dropout(dropout[1])
#                                  )
        
#         self.fc3 = nn.Linear(dim_hidden[1], num_classes)

#     def forward(self, x, return_features=None):
#         x = self.block1(x)
#         x = self.block2(x)
#         x1 = self.fc1(x)
#         x2 = self.fc2(x1)
#         out = self.fc3(x2)
#         if return_features: return x1, x2, out
#         else: return F.log_softmax(out, dim=1)    




class CNN(nn.Module):
    def __init__(self, 
                 init_filters, 
                 dim_hidden, 
                 dropout,
                 num_classes=10
                 ):

        super(CNN, self).__init__()
        
        self.block1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=init_filters, kernel_size=5, padding='same'),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=init_filters, out_channels=init_filters, kernel_size=5, padding='same'),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.Dropout(p=dropout[0])
                                    )
        
        self.block2 = nn.Sequential(nn.Conv2d(in_channels=init_filters, out_channels=init_filters*2, kernel_size=3, padding='same'),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=init_filters*2, out_channels=init_filters*2, kernel_size=3, padding='same'),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.Dropout(p=dropout[0]),
                                    nn.Flatten()
                                    )   
        
        self.fc1 = nn.Sequential(nn.Linear(init_filters*2*7*7, dim_hidden[0]),
                                 nn.BatchNorm1d(dim_hidden[0]),
                                 nn.ReLU(),
                                 nn.Dropout(dropout[1])
                                 )
        
        self.fc2 = nn.Sequential(nn.Linear(dim_hidden[0], dim_hidden[1]),
                                 nn.BatchNorm1d(dim_hidden[1]),
                                 nn.ReLU(),
                                 nn.Dropout(dropout[1])
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

