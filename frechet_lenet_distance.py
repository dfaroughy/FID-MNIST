import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import load_nist_data

def frechet_distance(mu_1, sigma_1, mu_2, sigma_2):
    '''
    Returns:
    - fid:  The Frechet distance between two Gaussians d = ||mu_1 - mu_2||^2 + Trace(sig_1 + sig_2 - 2*sqrt(sig_1 * sig_2)).
    '''
    mse = (mu_1 - mu_2).square().sum(dim=-1)
    trace = sigma_1.trace() + sigma_2.trace() - 2 * torch.linalg.eigvals(sigma_1 @ sigma_2).sqrt().real.sum(dim=-1)
    return mse + trace

@torch.no_grad()
def compute_activation_statistics(model, dataset, batch_size=64, activation_layer='fc1', device='cpu'):
    model.to(device)
    model.eval()
    features = []
    for batch, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        activations = model(batch, activation_layer=activation_layer)
        features.append(activations)
    features = torch.cat(features, dim=0)
    mu, sigma = torch.mean(features, dim=0), torch.cov(features.t())
    return mu, sigma

def compute_fld(model, target, reference, batch_size=64, activation_layer='fc1', device='cpu'):
    '''
    Returns:
    - fld:  The Frechet LeNet Distance (FID) between the reference and target datasets
    '''
    mu_ref, sigma_ref = compute_activation_statistics(model, reference, batch_size, activation_layer, device)
    mu, sigma = compute_activation_statistics(model, target, batch_size, activation_layer, device)
    return frechet_distance(mu, sigma, mu_ref, sigma_ref)

#...Frechet Distance for NIST corruptions

def fld_NIST(model, name='MNIST', corruption='noise', values=np.arange(0.0, 1, 0.02), batch_size=64, activation_layer='fc1', device='cpu'):
    reference = load_nist_data(name=name, train=False)
    fld = {}
    for val in values:
        target = load_nist_data(name, corruption=corruption, level=val, train=False)
        fld[val] = compute_fld(model, target=target, reference=reference, batch_size=batch_size, activation_layer=activation_layer, device=device).cpu()
    return fld

