import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from datasets import load_pepper_mnist, load_noisy_pepper_mnist, load_blurred_pepper_mnist, load_swirled_pepper_mnist
    

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.conv1 = model.conv1
        self.conv2 = model.conv2
        self.fc1 = model.fc1

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return x


def frechet_distance(mu_A, sigma_A, mu_B, sigma_B):
    mse = (mu_A - mu_B).square().sum(dim=-1)
    trace = sigma_A.trace() + sigma_B.trace() - 2 * torch.linalg.eigvals(sigma_A @ sigma_B).sqrt().real.sum(dim=-1)
    return mse + trace

@torch.no_grad()
def compute_FID(model, perturbation='thresholds', values=np.arange(0.0, 1, 0.02), batch_size=64):

    fid = {}

    dataset_ref = load_pepper_mnist(0.5, train=False)
    dataloader_ref = DataLoader(dataset_ref, batch_size=batch_size, shuffle=False)

    #...load the feature model and set it to evaluation mode
    feature_extractor = FeatureExtractor(model)
    feature_extractor.eval() 

    #...get the mean/covariance of reference dataset features

    features_ref = []

    for data, _ in dataloader_ref:
        features_ref.append(feature_extractor(data))

    features_ref = torch.cat(features_ref, dim=0)
    mu_ref, sigma_ref = torch.mean(features_ref, dim=0), torch.cov(features_ref.t())

    #...get fid of the perturbed dataset features

    for val in values:
        
        if perturbation == "threshold": dataset = load_pepper_mnist(threshold=val, train=False)
        elif perturbation == "noise": dataset = load_noisy_pepper_mnist(sigma=val, train=False)
        elif perturbation == "blur": dataset = load_blurred_pepper_mnist(sigma=val, train=False)
        elif perturbation == "swirl": dataset =  load_swirled_pepper_mnist(strength=val, train=False)
        else: raise ValueError("Perturbation type not recognized. Choose between 'threshold', 'noise', 'blur' and 'swirl'")

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        features = []

        for data, _ in dataloader:
            features.append(feature_extractor(data))

        features = torch.cat(features, dim=0)
        mu, sigma = torch.mean(features, dim=0), torch.cov(features.t())
        fid[val] = frechet_distance(mu_ref, sigma_ref, mu, sigma)
    
    return fid