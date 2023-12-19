import torch
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from skimage.transform import swirl

def load_pepper_mnist(threshold=0.5, train=True):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.where(x > threshold, torch.tensor(1.0), torch.tensor(0.0)))])
    return datasets.MNIST(root='./data', train=train, download=True, transform=transform)

#...Deformed Pepper MNIST datasets

def load_noisy_pepper_mnist(threshold=0.5, train=True, sigma=0.25):
    """Load MNIST dataset with binarization and Gaussian noise."""
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: add_gaussian_noise(x, std=sigma)),  # Add Gaussian noise
                                    transforms.Lambda(lambda x: torch.clamp(x, 0., 1.)),  # Clamp to ensure the image is still in [0, 1]
                                    transforms.Lambda(lambda x: torch.where(x > threshold, torch.tensor(1.0), torch.tensor(0.0)))  # Binarize
                                    ])
    return datasets.MNIST(root='./data', train=train, download=True, transform=transform)

def load_blurred_pepper_mnist(threshold=0.5, train=True, kernel_size=7, sigma=0.5):
    """Load MNIST dataset with Gaussian blur and binarization."""
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.GaussianBlur(kernel_size, sigma=sigma),  # Apply Gaussian blur
                                    transforms.Lambda(lambda x: torch.clamp(x, 0., 1.)),  # Clamp to ensure the image is still in [0, 1]
                                    transforms.Lambda(lambda x: torch.where(x > threshold, torch.tensor(1.0), torch.tensor(0.0)))  # Binarize
                                    ])
    return datasets.MNIST(root='./data', train=train, download=True, transform=transform)


def load_swirled_pepper_mnist(threshold=0.5, train=True, strength=1, radius=20):
    """Load MNIST dataset with swirl distortion and binarization."""
    transform = transforms.Compose([
        transforms.Lambda(lambda x: apply_swirl(x, strength=strength, radius=radius)),  # Apply swirl
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.clamp(x, 0., 1.)),  # Clamp to ensure the image is still in [0, 1]
        transforms.Lambda(lambda x: torch.where(x > threshold, torch.tensor(1.0), torch.tensor(0.0)))  # Binarize
    ])
    return datasets.MNIST(root='./data', train=train, download=True, transform=transform)


def add_gaussian_noise(tensor, mean=0., std=1.):
    """Adds Gaussian noise to a tensor."""
    noise = torch.randn(tensor.size()) * std + mean
    return tensor + noise


def apply_swirl(image, strength=1, radius=10):
    """Apply swirl distortion to an image."""
    image_np = np.array(image)
    swirled_image = swirl(image_np, strength=strength, radius=radius, mode='reflect')
    return Image.fromarray(swirled_image)

