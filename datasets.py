import torch
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from skimage.transform import swirl

def load_nist_data(dataset='MNIST', train=True, distortion=None, level=None, binary_threshold=None):
    
    transformation_list=[transforms.ToTensor()]
    
    assert bool(level) == bool(distortion), "Please specify a distortion level for image"

    #...define 1-parametric distortions:

    if distortion == 'noise':  transformation_list.append(transforms.Lambda(lambda x: add_gaussian_noise(x,  mean=0., std=level)))
    elif distortion == 'blur':  transformation_list.append(transforms.GaussianBlur(kernel_size=7, sigma=level))
    elif distortion == 'swirl': transformation_list.append(transforms.Lambda(lambda x: apply_swirl(x, strength=level, radius=20)))
    elif distortion == 'pixelize': transformation_list.append(transforms.Lambda(lambda x: apply_coarse_grain(x, p=level)))
    elif distortion == 'crop': transformation_list.append(transforms.Lambda(lambda x: apply_mask(x, p=level)))
    elif distortion is None: pass
    else: raise ValueError("Distortion type not recognized. Use 'noise', 'blur', 'pixelize', 'swirl', 'crop' or None")
        
    #...image binarization:

    if binary_threshold is not None: 
        transformation_list.append(transforms.Lambda(lambda x: (x > binary_threshold).type(torch.float32)))
    
    #...load dataset:
        
    transform = transforms.Compose(transformation_list)

    if dataset == 'MNIST':
        return datasets.MNIST(root='./data', train=train, download=True, transform=transform) 
    elif dataset == 'EMNIST Balanced':
        transform = transforms.Compose([CorrectEMNISTOrientation(), transform])
        return datasets.EMNIST(root='./data', split='balanced', train=train, download=True, transform=transform)
    elif dataset == 'EMNIST Byclass':
        transform = transforms.Compose([CorrectEMNISTOrientation(), transform])
        return datasets.EMNIST(root='./data', split='byclass', train=train, download=True, transform=transform) 
    elif dataset == 'EMNIST Bymerge':
        transform = transforms.Compose([CorrectEMNISTOrientation(), transform])
        return datasets.EMNIST(root='./data', split='bymerge', train=train, download=True, transform=transform)
    elif dataset == 'EMNIST Digits':
        transform = transforms.Compose([CorrectEMNISTOrientation(), transform])
        return datasets.EMNIST(root='./data', split='digits', train=train, download=True, transform=transform)
    elif dataset == 'EMNIST Letters':
        transform = transforms.Compose([CorrectEMNISTOrientation(), transform])
        return datasets.EMNIST(root='./data', split='letters', train=train, download=True, transform=transform)
    elif dataset == 'EMNIST Mnist':
        transform = transforms.Compose([CorrectEMNISTOrientation(), transform])
        return datasets.EMNIST(root='./data', split='mnist', train=train, download=True, transform=transform)
    elif dataset == 'QMNIST':
        return datasets.QMNIST(root='./data', what='train', download=True, transform=transform)
    elif dataset == 'KMNIST':
        return datasets.KMNIST(root='./data', train=train, download=True, transform=transform)
    elif dataset == 'FashionMNIST':
        return datasets.FashionMNIST(root='./data', train=train, download=True, transform=transform)     
    elif dataset== 'USPS':
        return datasets.USPS(root='./data', train=train, download=True, transform=transform)   
    elif dataset== 'SVHN':
        return datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    elif dataset == 'Omniglot':
        return datasets.Omniglot(root='./data', download=True, transform=transform)
    else:
        raise ValueError("Dataset not recognized. Choose between 'MNIST', 'EMNIST Balanced', 'EMNIST Byclass', 'EMNIST Bymerge', 'EMNIST Digits', 'EMNIST Letters', 'EMNIST Mnist', 'QMNIST', 'KMNIST', 'FashionMNIST', 'USPS', 'SVHN' and 'Omniglot'")
    

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






# #...Deformed Pepper MNIST datasets

# def load_noisy_mnist(train=True, sigma=0.25, binary_threshold=None):

#     """Load MNIST dataset with binarization and Gaussian noise.
#     """
#     transform = transforms.Compose([transforms.ToTensor(),
#                                     transforms.Lambda(lambda x: add_gaussian_noise(x, std=sigma)),  # Add Gaussian noise
#                                     transforms.Lambda(lambda x: torch.clamp(x, 0., 1.)),  # Clamp to ensure the image is still in [0, 1]
#                                     transforms.Lambda(lambda x: torch.where(x > binary_threshold, torch.tensor(1.0), torch.tensor(0.0)))  # Binarize
#                                     ])
#     return datasets.MNIST(root='./data', train=train, download=True, transform=transform)



# def load_blurred_pepper_mnist(train=True, kernel_size=7, sigma=0.5, binary_threshold=None):
#     """Load MNIST dataset with Gaussian blur and binarization."""
#     transform = transforms.Compose([transforms.ToTensor(),
#                                     transforms.GaussianBlur(kernel_size, sigma=sigma),  # Apply Gaussian blur
#                                     transforms.Lambda(lambda x: torch.clamp(x, 0., 1.)),  # Clamp to ensure the image is still in [0, 1]
#                                     transforms.Lambda(lambda x: torch.where(x > binary_threshold, torch.tensor(1.0), torch.tensor(0.0)))  # Binarize
#                                     ])
#     return datasets.MNIST(root='./data', train=train, download=True, transform=transform)



# def load_swirled_pepper_mnist(train=True, strength=1, radius=20, binary_threshold=None):
#     """Load MNIST dataset with swirl distortion and binarization."""
#     transform = transforms.Compose([transforms.Lambda(lambda x: apply_swirl(x, strength=strength, radius=radius)),  # Apply swirl
#                                     transforms.ToTensor(),
#                                     transforms.Lambda(lambda x: torch.clamp(x, 0., 1.)),  # Clamp to ensure the image is still in [0, 1]
#                                     transforms.Lambda(lambda x: torch.where(x > binary_threshold, torch.tensor(1.0), torch.tensor(0.0)))  # Binarize
#                                     ])
#     return datasets.MNIST(root='./data', train=train, download=True, transform=transform)

# def load_pixelized_pepper_mnist(train=True, resolution=0.5, binary_threshold=None):
#     transform = transforms.Compose([transforms.Lambda(lambda x: apply_coarse_grain(x, resolution)),
#                                     transforms.ToTensor(),
#                                     transforms.Lambda(lambda x: torch.where(x > binary_threshold, torch.tensor(1.0), torch.tensor(0.0)))
#                                     ])
#     return datasets.MNIST(root='./data', train=train, download=True, transform=transform)


# def load_cropped_pepper_mnist(train=True, crop_size=0.5, binary_threshold=None):
#     transform = transforms.Compose([
#         transforms.Lambda(lambda x: apply_mask(x, crop_size)),
#         transforms.ToTensor(),
#         transforms.Lambda(lambda x: torch.where(x > binary_threshold, torch.tensor(1.0), torch.tensor(0.0)))
#     ])
#     return datasets.MNIST(root='./data', train=train, download=True, transform=transform)

