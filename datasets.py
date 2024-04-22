import torch
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from skimage.transform import swirl
import random
from torch.utils.data import DataLoader
from utils import mnist_grid

def load_nist_data(name='MNIST', binary_threshold=0.5, train=True, distortion=None,  level=None):
    
    nist_datasets = ('MNIST', 'EMNIST', 'FashionMNIST', 'Omniglot', 'BinMNIST', 'BinEMNIST', 'BinFashionMNIST', 'BinOmniglot')
    assert name in nist_datasets, 'Dataset name not recognized. Choose between {}'.format(*nist_datasets)

    binerize_data = False
    if "Bin" in name:
        print("INFO: binerizing dataset with threshold={}".format(binary_threshold)) 
        binerize_data = True

    transformation_list=[]
    
    #...define 1-parametric corruptions:

    if distortion == 'noise': 
        transformation_list.append(transforms.ToTensor())
        transformation_list.append(transforms.Lambda(lambda x: apply_noise(x,  mean=0., std=level)))
        if binerize_data:
            transformation_list.append(transforms.Lambda(lambda x: (x > binary_threshold).type(torch.float32)))
    
    if distortion == 'blackout': 
        transformation_list.append(transforms.ToTensor())
        transformation_list.append(transforms.Lambda(lambda x: apply_blackout(x, fraction=level)))
        if binerize_data: 
            transformation_list.append(transforms.Lambda(lambda x: (x > binary_threshold).type(torch.float32)))

    elif distortion == 'blur':  
        transformation_list.append(transforms.ToTensor())
        transformation_list.append(transforms.GaussianBlur(kernel_size=7, sigma=level))
        if binerize_data: 
            transformation_list.append(transforms.Lambda(lambda x: (x > binary_threshold).type(torch.float32)))
    
    elif distortion == 'swirl': 
        transformation_list.append(transforms.Lambda(lambda x: apply_swirl(x, strength=level, radius=20)))
        transformation_list.append(transforms.ToTensor())
        if binerize_data: 
            transformation_list.append(transforms.Lambda(lambda x: (x > binary_threshold).type(torch.float32)))
    
    elif distortion == 'pixelize': 
        transformation_list.append(transforms.Lambda(lambda x: apply_coarse_grain(x, p=level)))
        transformation_list.append(transforms.ToTensor())
        if binerize_data: 
            transformation_list.append(transforms.Lambda(lambda x: (x > binary_threshold).type(torch.float32)))
    
    elif distortion == 'crop': 
        transformation_list.append(transforms.Lambda(lambda x: apply_mask(x, p=level)))
        transformation_list.append(transforms.ToTensor())
        if binerize_data: 
            transformation_list.append(transforms.Lambda(lambda x: (x > binary_threshold).type(torch.float32)))
    
    elif distortion == 'binerize': 
        transformation_list.append(transforms.ToTensor())
        transformation_list.append(transforms.Lambda(lambda x: (x > level).type(torch.float32)))

    elif distortion == 'half_mask':
        transformation_list.append(transforms.Lambda(lambda x: apply_half_mask(x)))
        transformation_list.append(transforms.ToTensor())
        if binerize_data: 
            transformation_list.append(transforms.Lambda(lambda x: (x > binary_threshold).type(torch.float32)))
    else:
        transformation_list.append(transforms.ToTensor())
        if binerize_data: 
            transformation_list.append(transforms.Lambda(lambda x: (x > binary_threshold).type(torch.float32)))
    
    
    #...load dataset:
        
    if name in ('MNIST', 'BinMNIST'):
        return datasets.MNIST(root='./data', train=train, download=True, transform=transforms.Compose(transformation_list)) 
    
    elif name in ('EMNIST', 'BinEMNIST'):
        return datasets.EMNIST(root='./data', split='letters', train=train, download=True, transform=transforms.Compose([CorrectEMNISTOrientation(), transforms.Compose(transformation_list)]))
    
    elif name in ('FashionMNIST', 'BinFashionMNIST'):
        return datasets.FashionMNIST(root='./data', train=train, download=True, transform=transforms.Compose(transformation_list))     

    elif name in ('Omniglot', 'BinOmniglot'):
        return datasets.Omniglot(root='./data', download=True, transform=transforms.Compose(transformation_list))
    
class CorrectEMNISTOrientation(object):
    def __call__(self, img):
        return transforms.functional.rotate(img, -90).transpose(Image.FLIP_LEFT_RIGHT)


def get_test_samples(name='MNIST', distortion=None, level=0.0, classes=[1,2,3,4,5,6,7,8,9], random=False, plot=False):
    data = load_nist_data(name=name, distortion=distortion, level=level)
    dataloader = DataLoader(data, batch_size=20*len(classes), shuffle=False)
    images, labels = next(iter(dataloader))
    d=[]
    for i in classes:
        imgs = images[labels == i] 
        idx = torch.randint(0, imgs.size(0), (1,)).item()
        img = imgs[idx] if random else imgs[0]
        d.append(img)
    sample = torch.cat(d, dim=0)  
    sample = sample.unsqueeze(1)
    if plot:
        title='level={}'.format(level) if distortion is not None else ''
        mnist_grid(sample, title=title, xlabel=name + ' + ' + distortion, num_img=9, nrow=3, figsize=(1.5,1.5))
    return sample


#...Image Corruption Functions:

def apply_noise(tensor, mean=0., std=1.):
    """Adds Gaussian noise to a tensor."""
    noise = torch.randn(tensor.size()) * std + mean
    return tensor + noise

def apply_blackout(tensor, fraction=0.1):
    """
    Randomly "turns-off" a fraction of non-zero pixels.
    """
    if not (0 <= fraction <= 1):
        raise ValueError("Fraction must be between 0 and 1.")
    ones_indices = (tensor > 0.0).nonzero(as_tuple=True)
    num_to_flip = int(fraction * len(ones_indices[0]))
    if num_to_flip == 0:
        return tensor
    selected_indices = torch.randperm(len(ones_indices[0]))[:num_to_flip]
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