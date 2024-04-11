import torch
import torch.nn as nn 
import tqdm
import numpy as np
import torch.optim as optim

import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision.utils import make_grid

from utils import plot_images
from image_datasets import load_nist_data

def train_classifier(model, 
                     train_dataloader, 
                     test_dataloader, 
                     device = 'cpu',
                     save_as='model.pth', 
                     max_epochs=10, 
                     early_stopping=None,
                     accuracy_goal=None,
                     lr=0.001):

    early_stopping = max_epochs if early_stopping is None else early_stopping
    accuracy_goal = 1 if accuracy_goal is None else accuracy_goal

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    max_accuracy = 0
    patience = 0

    for epoch in tqdm.tqdm(range(1, max_epochs), desc="Epochs"):

        for (data, target) in train_dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

        accuracy = get_model_accuracy(model, device, test_dataloader)

        if accuracy > max_accuracy:
            patience = 0
            max_accuracy = accuracy
            torch.save(model.state_dict(), save_as)
            print('INFO: current max accuracy: {}%'.format(100. * accuracy))
        else:
            patience += 1
            if patience > early_stopping:
                print('INFO: accuracy has not improved in {} epochs. Stopping training at {} epochs'.format(early_stopping, epoch))
                break
        if accuracy > accuracy_goal:
            print('INFO: accuracy goal reached. Stopping training at {} epochs'.format(epoch))
            break
    print('===================================')
    print('INFO: final max accuracy: {}%'.format(max_accuracy))
    print('===================================')


@torch.no_grad()
def get_model_accuracy(model, device, test_dataloader):
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in test_dataloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += F.cross_entropy(output, target, reduction='sum').item() 
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_dataloader.dataset)
    accuracy = correct / len(test_dataloader.dataset)
    return accuracy


def plot_uncolor_images(images, title,  cmap="gray", figsize=(4, 4)):
    _, axes = plt.subplots(8, 8, figsize=figsize)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.imshow(images[i].squeeze(), cmap=cmap)
        ax.axis('off')
    plt.suptitle(title)
    plt.show()

def plot_color_images(images, title, figsize=(4, 4)):
    fig, axes = plt.subplots(8, 8, figsize=figsize)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        img = images[i].permute(1, 2, 0)
        ax.imshow(img)
        ax.axis('off')
    plt.suptitle(title)
    plt.show()

def plot_images(images, title, figsize=(4, 4), cmap=None):
    fig, axes = plt.subplots(8, 8, figsize=figsize)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        img = images[i].permute(1, 2, 0)
        img = img.squeeze()  # Squeeze the last dimension if it's 1
        if cmap is not None: ax.imshow(img, cmap=cmap)
        else: ax.imshow(img)
        ax.axis('off')
    plt.suptitle(title)
    plt.show()


def mnist_grid(sample, title=None, xlabel=None, num_img=5, nrow=8, figsize=(10,10), save=False):
    _, ax= plt.subplots(1,1, figsize=figsize)
    sample = sample[:num_img]
    img = make_grid(sample, nrow=nrow)
    npimg = np.transpose(img.detach().cpu().numpy(),(1,2,0))
    plt.imshow(npimg)
    plt.title(title)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    if save:
        plt.savefig( 'images/' + title + '.png', bbox_inches='tight', dpi=2000)
    plt.show()


def get_10_digits(images, labels):
  digits={}
  for i in range(10):
      digits[i] = images[labels == i]
  digits = torch.cat([
                    digits[1][0],
                    digits[2][0],
                    digits[3][0],
                    digits[4][0],
                    digits[5][0],
                    digits[6][0],
                    digits[7][0],
                    digits[8][0],
                    digits[9][0]
                    ], dim=0)  
  digits = digits.unsqueeze(1)
  return digits

def plot_combined_with_mnist_grid(samples, fcd, distortion, dist_levels, xlim=(0,100), titles=None, loc='upper left', log=False, figsize=(10, 6)):
    N = len(samples)
    if titles is None: titles = [None] * N
    _ = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, N+2, width_ratios=[0.85]*N+[0.3, 1.5], wspace=0.)     
    axs = [plt.subplot(gs[i]) for i in range(N)] + [plt.subplot(gs[N+1])]   

    #...plot each mnist sample:

    for i, sample in enumerate(samples):
        img = make_grid(sample[:9], nrow=3)
        npimg = np.transpose(img.detach().cpu().numpy(), (1, 2, 0))
        axs[i].imshow(npimg)
        axs[i].set_xlabel(dist_levels[i], fontsize=8) if i > 0 else axs[i].set_xlabel('', fontsize=8)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        if titles:
            axs[i].set_title('+  '+distortion, fontsize=11) if i > 0 else axs[i].set_title('Binarized MNIST', fontsize=11)

    #...plot FCD:
            
    axs[N].plot(list(fcd[0].keys()), list(fcd[0].values()), color='darkblue', label=r'$FLD_1$')
    axs[N].plot(list(fcd[1].keys()), list(fcd[1].values()), color='gold', label=r'$FLD_2$')
    axs[N].plot(list(fcd[2].keys()), list(fcd[2].values()), color='darkred', label=r'$FLD_3$')
    axs[N].set_title('Frechet LeNet Distance', fontsize=10)
    axs[N].set_ylabel(r'FLD', fontsize=9)
    axs[N].set_xlabel('Corruption level', fontsize=9)
    axs[N].set_ylim(1,800)
    axs[N].set_xlim(xlim)
    axs[N].legend(loc=loc, fontsize=8)

    if log:
        axs[N].set_yscale('log')
        axs[N].set_ylim(1, 5000)
        axs[N].set_yticks([1, 10, 100, 1000, 10000])
        axs[N].get_yaxis().set_major_formatter(plt.ScalarFormatter())
        axs[N].set_yticklabels(['1', '10', '100', '1000', r'$10^4$'])

    plt.tight_layout()
    plt.savefig(distortion + '_combined_plot.png', bbox_inches='tight', dpi=500)
    plt.show()
