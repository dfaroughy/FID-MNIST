from torch.utils.data import DataLoader
from dataset import load_nist_data
from utils import train_classifier
from models import CNN
import torch.nn as nn

#==================================
dataname = 'MNIST'
model = CNN(init_filters=32, dim_hidden=(256, 128), dropout=(0.25, 0.1), num_classes=10) 
accuracy_goal = 0.995
device = 'cuda:0'
#==================================

train = load_nist_data(name=dataname)
test = load_nist_data(name=dataname, train=False)
train_dataloader = DataLoader(train, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test, batch_size=64, shuffle=False)

#...train classifier


print('INFO: training {} on {}'.format('CNN', dataname))

train_classifier(model, 
                 train_dataloader=train_dataloader,
                 test_dataloader=test_dataloader,
                 device=device,
                 accuracy_goal=accuracy_goal,
                 lr=0.0004,
                 max_epochs=100, 
                 early_stopping=20,
                 save_as='models/{}_{}.pth'.format('CNN_v2','_'.join(dataname.split(' ')))
                 )

