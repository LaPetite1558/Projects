# Imports python modules
import torch
import numpy as np
import json
from os_utils import check_path, quit
from torchvision import datasets, transforms
from PIL import Image
"""
Functions for preparing images and data for training and/or
prediction with PyTorch models
"""

# Means and standard deviations for normalization
MEANS = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

    
def load_data(img_dir, train=False):
    """Transform images and convert to tensors for training and prediction
       Create and return dataloaders with transformed image tensors and class labels
    """
    # check that image directory exists and remove slash if needed
    img_dir, exists = check_path(img_dir)
    
    # if image directory does not exist, exit app
    if not exists:
        quit("{} does not exist".format(img_dir))
    
    # if loading images for training, perform additional image operations
    # create shuffle variable for later use in dataloader and set as True
    if train:
        transform = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=MEANS, 
                                                             std=STD)])
        shuffle = True

    # if loading images for testing or validation, perform only 
    # standard resize and cropping
    # create shuffle variable for later use in dataloader and set as False
    else:
        transform = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=MEANS, 
                                                             std=STD)])
        shuffle = False
    
    dataset = datasets.ImageFolder(img_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                                 shuffle=shuffle)
    
    return dataloader
    

def process_image(img_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a PyTorch tensor
    '''
    
    # Process a PIL image for use in a PyTorch model
    
    # check that image file exists
    img_path, exists = check_path(img_path)
    
    if not exists:
        quit("{} does not exist".format(img_path))
    
    # resize image and crop
    pil_image = Image.open(img_path)
    pil_image = pil_image.resize((256, 256))
    crop_len = (256-224)*0.5
    pil_image = pil_image.crop((crop_len, crop_len, 256-crop_len, 256-crop_len))
    
    np_image = np.array(pil_image) / 255
    np_image = (np_image - MEANS) / STD
    
    img_tensor = torch.from_numpy(np_image.transpose((2,0,1)))
    
    return img_tensor.unsqueeze(0)


def get_cat_to_name(filepath):
    ''' Open file for category names
        Returns dictionary of category names and class mappings
    '''
    # check that file exists
    filepath, exists = check_path(filepath)
    
    if not exists:
        quit("{} does not exist".format(filepath))
    
    with open(filepath, 'r') as f:
        cat_to_name = json.load(f)
    
    return cat_to_name
