import torch
import torch.nn as nn 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import tqdm

def train_classifier(model, 
                     train_dataloader, 
                     test_dataloader, 
                     accuracy_goal=95,
                     device = 'cpu',
                     save_as='model.pth', 
                     epochs=10, 
                     lr=0.001):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in tqdm.tqdm(range(1, epochs), desc="Epochs"):

        for (data, target) in train_dataloader:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

        accuracy = get_model_accuracy(model, device, test_dataloader)

        if epoch % 3 == 0:
            print('current accuracy: {}%'.format(accuracy))

        if accuracy > accuracy_goal:
            print('accuracy goal reached. Stopping training at {} epochs'.format(accuracy_goal, epoch))
            break

    print('final accuracy: {}%'.format(accuracy))
    torch.save(model.state_dict(), save_as)



@torch.no_grad()
def get_model_accuracy(model, device, test_loader):
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            # all_preds.extend(pred.view(-1).cpu().numpy())
            # all_targets.extend(target.view(-1).cpu().numpy())

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), accuracy))
    return accuracy
    # # Confusion Matrix
    # conf_matrix = confusion_matrix(all_targets, all_preds)
    # print("Confusion Matrix:\n", conf_matrix)


def plot_images(images, title,  cmap="gray", figsize=(4, 4)):
    fig, axes = plt.subplots(8, 8, figsize=figsize)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.imshow(images[i].squeeze(), cmap=cmap)
        ax.axis('off')
    plt.suptitle(title)
    plt.show()