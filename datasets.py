import torch
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from skimage.transform import swirl

def load_nist_data(name='MNIST', train=True, binerize=True, distortion=None, level=None):
    
    transformation_list=[]
    
    # assert bool(level) == bool(distortion), "Please specify a distortion level for image"

    #...define 1-parametric distortions:

    if distortion == 'noise': 
        transformation_list.append(transforms.ToTensor())
        transformation_list.append(transforms.Lambda(lambda x: add_gaussian_noise(x,  mean=0., std=level)))
        if binerize: transformation_list.append(transforms.Lambda(lambda x: (x > 0.5).type(torch.float32)))
    elif distortion == 'blur':  
        transformation_list.append(transforms.ToTensor())
        transformation_list.append(transforms.GaussianBlur(kernel_size=7, sigma=level))
        if binerize: transformation_list.append(transforms.Lambda(lambda x: (x > 0.5).type(torch.float32)))
    elif distortion == 'swirl': 
        transformation_list.append(transforms.Lambda(lambda x: apply_swirl(x, strength=level, radius=20)))
        transformation_list.append(transforms.ToTensor())
        if binerize: transformation_list.append(transforms.Lambda(lambda x: (x > 0.5).type(torch.float32)))
    elif distortion == 'pixelize': 
        transformation_list.append(transforms.Lambda(lambda x: apply_coarse_grain(x, p=level)))
        transformation_list.append(transforms.ToTensor())
        if binerize: transformation_list.append(transforms.Lambda(lambda x: (x > 0.5).type(torch.float32)))
    elif distortion == 'crop': 
        transformation_list.append(transforms.Lambda(lambda x: apply_mask(x, p=level)))
        transformation_list.append(transforms.ToTensor())
        if binerize: transformation_list.append(transforms.Lambda(lambda x: (x > 0.5).type(torch.float32)))
    elif distortion == 'binerize': 
        transformation_list.append(transforms.ToTensor())
        transformation_list.append(transforms.Lambda(lambda x: (x > level).type(torch.float32)))
    elif distortion is None:
        transformation_list.append(transforms.ToTensor())
        if binerize: transformation_list.append(transforms.Lambda(lambda x: (x > 0.5).type(torch.float32)))
    else: raise ValueError("Distortion type not recognized. Use 'binerize', 'noise', 'blur', 'pixelize', 'swirl', 'crop' or None")

    #...load dataset:
        
    transform = transforms.Compose(transformation_list)

    if name == 'MNIST':
        return datasets.MNIST(root='./data', train=train, download=True, transform=transform) 
    elif name == 'EMNIST Balanced':
        transform = transforms.Compose([CorrectEMNISTOrientation(), transform])
        return datasets.EMNIST(root='./data', split='balanced', train=train, download=True, transform=transform)
    elif name == 'EMNIST Byclass':
        transform = transforms.Compose([CorrectEMNISTOrientation(), transform])
        return datasets.EMNIST(root='./data', split='byclass', train=train, download=True, transform=transform) 
    elif name == 'EMNIST Bymerge':
        transform = transforms.Compose([CorrectEMNISTOrientation(), transform])
        return datasets.EMNIST(root='./data', split='bymerge', train=train, download=True, transform=transform)
    elif name == 'EMNIST Digits':
        transform = transforms.Compose([CorrectEMNISTOrientation(), transform])
        return datasets.EMNIST(root='./data', split='digits', train=train, download=True, transform=transform)
    elif name == 'EMNIST Letters':
        transform = transforms.Compose([CorrectEMNISTOrientation(), transform])
        return datasets.EMNIST(root='./data', split='letters', train=train, download=True, transform=transform)
    elif name == 'EMNIST Mnist':
        transform = transforms.Compose([CorrectEMNISTOrientation(), transform])
        return datasets.EMNIST(root='./data', split='mnist', train=train, download=True, transform=transform)
    elif name == 'QMNIST':
        return datasets.QMNIST(root='./data', what='train', download=True, transform=transform)
    elif name == 'KMNIST':
        return datasets.KMNIST(root='./data', train=train, download=True, transform=transform)
    elif name == 'FashionMNIST':
        return datasets.FashionMNIST(root='./data', train=train, download=True, transform=transform)     
    elif name == 'USPS':
        return datasets.USPS(root='./data', train=train, download=True, transform=transform)   
    elif name == 'SVHN':
        return datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    elif name == 'Omniglot':
        return datasets.Omniglot(root='./data', download=True, transform=transform)
    else:
        raise ValueError("Dataset name not recognized. Choose between 'MNIST', 'EMNIST Balanced', 'EMNIST Byclass', 'EMNIST Bymerge', 'EMNIST Digits', 'EMNIST Letters', 'EMNIST Mnist', 'QMNIST', 'KMNIST', 'FashionMNIST', 'USPS', 'SVHN' and 'Omniglot'")
    

class CorrectEMNISTOrientation(object):
    def __call__(self, img):
        return transforms.functional.rotate(img, -90).transpose(Image.FLIP_LEFT_RIGHT)


#...functions for applying perturbations to images:


def add_gaussian_noise(tensor, mean=0., std=1.):
    """Adds Gaussian noise to a tensor."""
    noise = torch.randn(tensor.size()) * std + mean
    return tensor + noise

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
