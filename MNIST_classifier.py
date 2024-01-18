from torch.utils.data import DataLoader
from image_datasets import load_nist_data
from utils import train_classifier
from architectures import LeNet5, LeNet3, ResNet18, ResNet34

#==================================
dataname = 'BinaryMNIST'
network = 'LeNet5'
accuracy_goal = 0.995
device = 'cuda:1'
#==================================

train = load_nist_data(name=dataname)
test = load_nist_data(name=dataname, train=False)
train_dataloader = DataLoader(train, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test, batch_size=64, shuffle=False)

#...train classifier

if network == 'LeNet5': model = LeNet5(num_classes=10) 
if network == 'ResNet18': model = ResNet18(num_classes=10) 
if network == 'ResNet34': model = ResNet34(num_classes=10) 
if network == 'LeNet3': model = LeNet3(num_classes=10, num_channels=(2, 8), dim_hidden=(64, 32))

# self.conv1 = nn.Conv2d(1, num_channels[0], filter_size)
# self.conv2 = nn.Conv2d(num_channels[0], num_channels[1], filter_size)
# self.fc1 = nn.Linear(num_channels[1] * 4 * 4, dim_hidden[0])
# self.fc2 = nn.Linear(dim_hidden[0], dim_hidden[1])
# self.fc3 = nn.Linear(dim_hidden[1], num_classes)

print('INFO: training {} on {}'.format(network, dataname))

train_classifier(model, 
                 train_dataloader=train_dataloader,
                 test_dataloader=test_dataloader,
                 device=device,
                 accuracy_goal=accuracy_goal,
                 lr=0.001,
                 max_epochs=100, 
                 early_stopping=20,
                 save_as='models/{}_{}.pth'.format(network,'_'.join(dataname.split(' ')))
                 )

