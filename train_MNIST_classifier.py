from torch import nn
from torch.utils.data import DataLoader
from dataset import load_nist_data
from utils import train_classifier

from models import LeNet, LeNet5

#==================================
dataname = 'MNIST'
model = LeNet(channels=256, 
              dim_hidden=(128, 128), 
              dropout=(0.25, 0.1), 
              use_batch_norm=True,
              num_classes=10) 
accuracy_goal = 0.9985
device = 'cuda:0'
#==================================

train = load_nist_data(name=dataname)
test = load_nist_data(name=dataname, train=False)
train_dataloader = DataLoader(train, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test, batch_size=128, shuffle=False)

#...train classifier

print('INFO: training {} on {}'.format('CNN', dataname))

train_classifier(model, 
                 train_dataloader=train_dataloader,
                 test_dataloader=test_dataloader,
                 device=device,
                 accuracy_goal=accuracy_goal,
                 lr=0.0001,
                 max_epochs=100, 
                 early_stopping=20,
                 save_as='models/{}_{}.pth'.format('LeNet_v3','_'.join(dataname.split(' ')))
                 )

