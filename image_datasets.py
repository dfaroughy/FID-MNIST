import torch
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from skimage.transform import swirl
import random

def load_nist_data(name='MNIST', train=True, distortion=None, level=None):
    
    nist_datasets = ('MNIST', 'CIFAR10', 'CelebA', 'ImagNet', 'EMNIST Balanced', 'EMNIST Byclass', 'EMNIST Bymerge', 
                     'EMNIST Digits', 'EMNIST Letters', 'EMNIST mnist', 'QMNIST', 'KMNIST', 'FashionMNIST', 'USPS', 'SVHN', 'Omniglot',
                     'BinaryMNIST', 'BinaryCIFAR10', 'BinaryCelebA', 'BinaryImagNet', 'BinaryEMNIST Balanced', 'BinaryEMNIST Byclass', 
                     'BinaryEMNIST Bymerge', 'BinaryEMNIST Digits', 'BinaryEMNIST Letters', 'BinaryEMNIST mnist', 'BinaryQMNIST', 
                     'BinaryKMNIST', 'BinaryFashionMNIST', 'BinaryUSPS', 'BinarySVHN', 'BinaryOmniglot')

    assert name in nist_datasets, 'Dataset name not recognized. Choose between {}'.format(*nist_datasets)

    binerize_data = False
    if "Binary" in name: 
        binerize_data = True
        binary_threshold = {'BinaryMNIST': 0.5, 'BinaryFashionMNIST': 0.5, 'BinaryCIFAR10': 0.75, 'BinaryCelebA': 0.5, 'BinaryImagNet': 0.5, 
                            'BinaryEMNIST Balanced': 0.5, 'BinaryEMNIST Byclass': 0.5, 'BinaryEMNIST Bymerge': 0.5, 'BinaryEMNIST Digits': 0.5, 
                            'BinaryEMNIST Letters': 0.5, 'BinaryEMNIST mnist': 0.5, 'BinaryQMNIST': 0.5, 'BinaryKMNIST': 0.5, 'BinaryUSPS': 0.5, 
                            'BinarySVHN': 0.5, 'BinaryOmniglot': 0.5}

    transformation_list=[]
    
    #...define 1-parametric distortions:

    if distortion == 'noise': 
        transformation_list.append(transforms.ToTensor())
        transformation_list.append(transforms.Lambda(lambda x: add_gaussian_noise(x,  mean=0., std=level)))
        if binerize_data: 
            transformation_list.append(transforms.Lambda(lambda x: (x >  binary_threshold[name]).type(torch.float32)))
    
    if distortion == 'melt': 
        transformation_list.append(transforms.ToTensor())
        transformation_list.append(transforms.Lambda(lambda x: add_gaussian_melt(x, fraction=level)))
        if binerize_data: 
            transformation_list.append(transforms.Lambda(lambda x: (x >  binary_threshold[name]).type(torch.float32)))

    elif distortion == 'blur':  
        transformation_list.append(transforms.ToTensor())
        transformation_list.append(transforms.GaussianBlur(kernel_size=7, sigma=level))
        if binerize_data: 
            transformation_list.append(transforms.Lambda(lambda x: (x > binary_threshold[name]).type(torch.float32)))
    
    elif distortion == 'swirl': 
        transformation_list.append(transforms.Lambda(lambda x: apply_swirl(x, strength=level, radius=20)))
        transformation_list.append(transforms.ToTensor())
        if binerize_data: 
            transformation_list.append(transforms.Lambda(lambda x: (x > binary_threshold[name]).type(torch.float32)))
    
    elif distortion == 'pixelize': 
        transformation_list.append(transforms.Lambda(lambda x: apply_coarse_grain(x, p=level)))
        transformation_list.append(transforms.ToTensor())
        if binerize_data: 
            transformation_list.append(transforms.Lambda(lambda x: (x > binary_threshold[name]).type(torch.float32)))
    
    elif distortion == 'crop': 
        transformation_list.append(transforms.Lambda(lambda x: apply_mask(x, p=level)))
        transformation_list.append(transforms.ToTensor())
        if binerize_data: 
            transformation_list.append(transforms.Lambda(lambda x: (x > binary_threshold[name]).type(torch.float32)))
    
    elif distortion == 'binerize': 
        transformation_list.append(transforms.ToTensor())
        transformation_list.append(transforms.Lambda(lambda x: (x > level).type(torch.float32)))

    elif distortion == 'half_mask':
        transformation_list.append(transforms.Lambda(lambda x: apply_half_mask(x)))
        transformation_list.append(transforms.ToTensor())
        if binerize_data: 
            transformation_list.append(transforms.Lambda(lambda x: (x > binary_threshold[name]).type(torch.float32)))
    
    elif distortion == 'half_mask':
        transformation_list.append(transforms.Lambda(lambda x: apply_half_mask(x)))
        transformation_list.append(transforms.ToTensor())
        if binerize_data: 
            transformation_list.append(transforms.Lambda(lambda x: (x > binary_threshold[name]).type(torch.float32)))
   
    elif distortion == 'half_noisy':
        transformation_list.append(transforms.Lambda(lambda x: apply_half_noisy(x, std=level)))
        transformation_list.append(transforms.ToTensor())
        if binerize_data: 
            transformation_list.append(transforms.Lambda(lambda x: (x > binary_threshold[name]).type(torch.float32)))

    elif distortion == 'half_pure_noise':
        transformation_list.append(transforms.Lambda(lambda x: apply_half_pure_noise(x, std=level)))
        transformation_list.append(transforms.ToTensor())
        if binerize_data: 
            transformation_list.append(transforms.Lambda(lambda x: (x > binary_threshold[name]).type(torch.float32)))

    else:
        transformation_list.append(transforms.ToTensor())
        if binerize_data: 
            transformation_list.append(transforms.Lambda(lambda x: (x > binary_threshold[name]).type(torch.float32)))
    
    
    #...load dataset:
        
    if name == 'MNIST' or name == 'BinaryMNIST':
        return datasets.MNIST(root='./data', train=train, download=True, transform=transforms.Compose(transformation_list)) 
    
    elif name in ('CIFAR10', 'BinaryCIFAR10'):
        return datasets.CIFAR10(root='./data', train=train, download=True, transform=transforms.Compose(transformation_list))
    
    elif name in  ('CelebA', 'BinaryCelebA'):
        return datasets.CelebA(root='./data', split='train', download=True, transform=transforms.Compose(transformation_list))
    
    elif name in ('ImageNet', 'BinaryImageNet'):
        return datasets.ImageNet(root='./data', split='train', download=True, transform=transforms.Compose(transformation_list))
    
    elif name in ('EMNIST Balanced', 'BinaryEMNIST Balanced'):
        return datasets.EMNIST(root='./data', split='balanced', train=train, download=True, transform=transforms.Compose([CorrectEMNISTOrientation(), transforms.Compose(transformation_list)]))
    
    elif name in ('EMNIST Byclass', 'BinaryEMNIST Byclass'):
        return datasets.EMNIST(root='./data', split='byclass', train=train, download=True, transform=transforms.Compose([CorrectEMNISTOrientation(), transforms.Compose(transformation_list)])) 
    
    elif name in ('EMNIST Bymerge', 'BinaryEMNIST Bymerge'):
        return datasets.EMNIST(root='./data', split='bymerge', train=train, download=True, transform=transforms.Compose([CorrectEMNISTOrientation(), transforms.Compose(transformation_list)]))
    
    elif name in ('EMNIST Digits', 'BinaryEMNIST Digits'):
        return datasets.EMNIST(root='./data', split='digits', train=train, download=True, transform=transforms.Compose([CorrectEMNISTOrientation(), transforms.Compose(transformation_list)]))
    
    elif name in ('EMNIST Letters', 'BinaryEMNIST Letters'):
        return datasets.EMNIST(root='./data', split='letters', train=train, download=True, transform=transforms.Compose([CorrectEMNISTOrientation(), transforms.Compose(transformation_list)]))
    
    elif name in ('EMNIST mnist', 'BinaryEMNIST mnist'):
        return datasets.EMNIST(root='./data', split='mnist', train=train, download=True, transform=transforms.Compose([CorrectEMNISTOrientation(), transforms.Compose(transformation_list)]))
    
    elif name in ('QMNIST', 'BinaryQMNIST'):
        return datasets.QMNIST(root='./data', what='train', download=True, transform=transforms.Compose(transformation_list))
    
    elif name in ('KMNIST', 'BinaryKMNIST'):
        return datasets.KMNIST(root='./data', train=train, download=True, transform=transforms.Compose(transformation_list))
    
    elif name in ('FashionMNIST', 'BinaryFashionMNIST'):
        return datasets.FashionMNIST(root='./data', train=train, download=True, transform=transforms.Compose(transformation_list))     
    
    elif name in ('USPS', 'BinaryUSPS'):
        return datasets.USPS(root='./data', train=train, download=True, transform=transforms.Compose(transformation_list))   
    
    elif name in ('SVHN', 'BinarySVHN'):
        return datasets.SVHN(root='./data', split='train', download=True, transform=transforms.Compose(transformation_list))
    
    elif name in ('Omniglot', 'BinaryOmniglot'):
        return datasets.Omniglot(root='./data', download=True, transform=transforms.Compose(transformation_list))
    

class CorrectEMNISTOrientation(object):
    def __call__(self, img):
        return transforms.functional.rotate(img, -90).transpose(Image.FLIP_LEFT_RIGHT)


#...functions for applying perturbations to images:


def add_gaussian_noise(tensor, mean=0., std=1.):
    """Adds Gaussian noise to a tensor."""
    noise = torch.randn(tensor.size()) * std + mean
    return tensor + noise

def add_gaussian_melt(tensor, fraction=0.1):
    """
    Randomly flips a fraction of 1s to 0s in a tensor of shape (1, 28, 28).
    
    Parameters:
        tensor (torch.Tensor): Input tensor of shape (1, 28, 28) with entries that are 0s and 1s.
        fraction (float): Fraction of 1s to flip to 0s.
    
    Returns:
        torch.Tensor: Tensor with some 1s flipped to 0s.
    """
    # Ensure the input fraction is within a sensible range
    if not (0 <= fraction <= 1):
        raise ValueError("Fraction must be between 0 and 1.")
    
    # Find indices where the tensor is 1
    ones_indices = (tensor > 0.0).nonzero(as_tuple=True)
    
    # Calculate the number of 1s to flip
    num_to_flip = int(fraction * len(ones_indices[0]))

    # If no elements to flip, return the original tensor
    if num_to_flip == 0:
        return tensor

    # Randomly select indices of the 1s to flip
    selected_indices = torch.randperm(len(ones_indices[0]))[:num_to_flip]

    # Flip the selected indices from 1 to 0
    tensor[ones_indices[0][selected_indices], 
           ones_indices[1][selected_indices], 
           ones_indices[2][selected_indices]] = 0

    return tensor

def apply_swirl(image, strength=1, radius=20):
    """Apply swirl distortion to an image."""
    image_np = np.array(image)
    swirled_image = swirl(image_np, strength=strength, radius=radius, mode='reflect')
    return Image.fromarray(swirled_image)

def apply_coarse_grain(image, p=0.1):
    """Coarse grains an image to a lower resolution."""
    old_size = image.size
    if p <= 0: return image  
    elif p >= 1: return Image.new('L', image.size, color=0)  # Return a black image
    new_size = max(1, int(image.width * (1 - p))), max(1, int(image.height * (1 - p)))
    image = image.resize(new_size, Image.BILINEAR)
    return image.resize(old_size, Image.NEAREST)  # Resize back to 28x28

def apply_mask(image, p=0.1):
    """ Masks the image with a square window. """
    if p <= 0:
        return image  # No change
    elif p >= 1:
        return Image.new('L', image.size, color=0)  # Entirely black image

    mask_size = int(image.width * (1 - p)), int(image.height * (1 - p))
    mask = Image.new('L', mask_size, color=255)  # White square
    black_img = Image.new('L', image.size, color=0)
    black_img.paste(mask, (int((image.width - mask_size[0]) / 2), int((image.height - mask_size[1]) / 2)))
    return Image.composite(image, black_img, black_img)

def apply_half_mask(image):
    """ Masks the first half of the image along its width. """
    mask_height = int(image.height / 2)
    mask_width = image.width
    mask_size = (mask_width, mask_height)
    mask = Image.new('L', mask_size, color=255) 
    black_img = Image.new('L', image.size, color=0)
    black_img.paste(mask, (0, 0))  
    return Image.composite(image, black_img, black_img)


def apply_half_noisy(image, std=1.0):
    """ Masks the lower half of the image along its height and adds Gaussian noise to it. """
    mask_height = int(image.height / 2)
    mask_width = image.width
    mask_size = (mask_width, mask_height)
    mask = Image.new('L', mask_size, color=255) 
    noise = np.random.normal(0, std, mask_size).astype(np.float32)
    noise = np.clip(noise, -255, 255) 
    noise_img = Image.fromarray(noise.astype(np.uint8))
    final_img = Image.new('L', image.size, color=0)
    final_img.paste(mask, (0, 0))
    final_img.paste(noise_img, (0, mask_height))
    return Image.composite(image, final_img, final_img)


def apply_half_pure_noise(image, std=1.0):
    """ Replaces the lower half of the image with Gaussian noise. """

    # Calculate the dimensions for the lower half
    mask_height = int(image.height / 2)
    mask_width = image.width

    # Generate Gaussian noise for the lower half
    noise = np.random.normal(0, std, (mask_height, mask_width)).astype(np.float32)
    noise = np.clip(noise, 0, 255) 

    # Create an image from the noise
    noise_image = Image.fromarray(noise.astype(np.uint8))

    # Create a new image to hold the final result
    final_img = Image.new('L', image.size, color=0)

    # Paste the original image on the upper half of the final image
    final_img.paste(image.crop((0, 0, mask_width, mask_height)), (0, 0))

    # Paste the noise image on the lower half of the final image
    final_img.paste(noise_image, (0, mask_height))

    return final_img


def sample_diversity(images, labels, diversity=0.0):
    if diversity > 1.0: diversity = 1.0
    images_replicated, labels_replicated=[], []
    for i in range(10):
        img, lbl = images[labels==i], labels[labels==i]
        N = img.shape[0]
        M = int((diversity) * N)
        if M == 0: M=1
        j = random.sample(range(0, N), M)
        img_sub, lbl_sub = img[j], lbl[j]
        k = torch.randint(low=0, high=M, size=(N,))
        images_replicated.append(img_sub[k])
        labels_replicated.append(lbl_sub[k])

    images_replicated = torch.cat(images_replicated, dim=0)
    labels_replicated = torch.cat(labels_replicated, dim=0)
    idx = torch.randperm(images_replicated.size(0))
    images_replicated = images_replicated[idx]
    labels_replicated = labels_replicated[idx]
    
    return images_replicated, labels_replicated