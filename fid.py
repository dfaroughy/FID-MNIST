import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from datasets import load_nist_data

def frechet_distance(mu_A, sigma_A, mu_B, sigma_B):
    mse = (mu_A - mu_B).square().sum(dim=-1)
    trace = sigma_A.trace() + sigma_B.trace() - 2 * torch.linalg.eigvals(sigma_A @ sigma_B).sqrt().real.sum(dim=-1)
    return mse + trace

def compute_fid(model, dataset, mu_ref, sigma_ref, batch_size=64, layer='fc1'):
    mu, sigma = get_layer_features(model, dataset, batch_size=batch_size, layer=layer)
    fid = frechet_distance(mu, sigma, mu_ref, sigma_ref)
    return fid

def fid_distorted_NIST(model, name='MNIST', distortion='noise', values=np.arange(0.0, 1, 0.02), batch_size=64, layer='fc1'):
    dataset = load_nist_data(name=name, train=False)
    mu, sigma = get_layer_features(model, dataset, batch_size=batch_size, layer=layer)
    fid = {}
    for val in values:
        dataset = load_nist_data(name=name, distortion=distortion, level=val, train=False)
        fid[val] = compute_fid(model, dataset, mu_ref=mu, sigma_ref=sigma, batch_size=batch_size, layer=layer)
    return fid

@torch.no_grad()
def get_layer_features(model, dataset, batch_size=64, layer='fc1'):
    model.eval() 
    features = []
    for batch, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        features.append(model(batch, feature_layer=layer))
    features = torch.cat(features, dim=0)
    mu, sigma = torch.mean(features, dim=0), torch.cov(features.t())
    return mu, sigma