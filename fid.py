import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from datasets import (load_pepper_mnist, 
                      load_noisy_pepper_mnist, 
                      load_blurred_pepper_mnist, 
                      load_swirled_pepper_mnist,
                      load_pixelized_pepper_mnist,
                      load_cropped_pepper_mnist)


def frechet_distance(mu_A, sigma_A, mu_B, sigma_B):
    mse = (mu_A - mu_B).square().sum(dim=-1)
    trace = sigma_A.trace() + sigma_B.trace() - 2 * torch.linalg.eigvals(sigma_A @ sigma_B).sqrt().real.sum(dim=-1)
    return mse + trace

@torch.no_grad()
def get_layer_features(model, dataset, batch_size=64, extract_feature_layer='fc1'):
    ''' model should be in eval() mode
    '''
    features = []
    for data, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        features.append(model(data, extract_feature_layer=extract_feature_layer))
    features = torch.cat(features, dim=0)
    mu, sigma = torch.mean(features, dim=0), torch.cov(features.t())
    return mu, sigma

@torch.no_grad()
def compute_fid(model, 
                dataset,
                mu_ref, 
                sigma_ref, 
                batch_size=64,
                extract_feature_layer='fc1'):
    ''' model should be in eval() mode
    '''
    mu, sigma = get_layer_features(model, dataset, batch_size=batch_size, extract_feature_layer=extract_feature_layer)
    fid = frechet_distance(mu, sigma, mu_ref, sigma_ref)
    return fid


@torch.no_grad()
def perturbed_MNIST_fid(model, 
                        perturbation='thresholds', 
                        values=np.arange(0.0, 1, 0.02), 
                        batch_size=64, 
                        layer='fc1'):

    model.eval() 
    
    #...get the intermediate layer mean/covariance for reference dataset

    dataset = load_pepper_mnist(0.5, train=False)
    mu, sigma = get_layer_features(model, dataset, batch_size=batch_size, extract_feature_layer=layer)

    #...compute fid for the perturbed MNIST datasets with different perturbation levels 

    fid = {}

    for val in values:
        
        if perturbation == "threshold": dataset = load_pepper_mnist(threshold=val, train=False)
        elif perturbation == "noise": dataset = load_noisy_pepper_mnist(sigma=val, train=False)
        elif perturbation == "blur": dataset = load_blurred_pepper_mnist(sigma=val, train=False)
        elif perturbation == "swirl": dataset =  load_swirled_pepper_mnist(strength=val, train=False)
        elif perturbation == "pixelate": dataset =  load_pixelized_pepper_mnist(resolution=val, train=False)
        elif perturbation == "crop": dataset =  load_cropped_pepper_mnist(crop_size=val, train=False)
        else: raise ValueError("Perturbation type not recognized. Choose between 'threshold', 'noise', 'blur' and 'swirl'")

        fid[val] = compute_fid(model, dataset, mu_ref=mu, sigma_ref=sigma, batch_size=batch_size, extract_feature_layer=layer)
    
    return fid