from torch.utils.data import DataLoader
from datasets import load_nist_data
from utils import train_classifier
from models import ResNet18

#==================================
dataname = 'BinaryFashionMNIST'
accuracy_goal = 0.95
device = 'cuda:0'
#==================================

train = load_nist_data(name=dataname)
test = load_nist_data(name=dataname, train=False)
train_dataloader = DataLoader(train, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test, batch_size=64, shuffle=False)

#...train classifier

model = ResNet18(num_classes=10) 

train_classifier(model, 
                 train_dataloader=train_dataloader,
                 test_dataloader=test_dataloader,
                 device=device,
                 accuracy_goal=accuracy_goal*100,
                 save_as='models/ResNet18_{}.pth'.format('_'.join(dataname.split(' '))), 
                 epochs=100, 
                 lr=0.001)