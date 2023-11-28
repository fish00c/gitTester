
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.io import read_image

import torch

from skimage import io, transform

import matplotlib.pyplot as plt

class GenImageDataset(Dataset):
    """GenImage dataset."""
    def __init__(self, root_dir, dataset_type, model_type, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.model_type = model_type
        self.transform = transform

        self.image_names = os.listdir(os.path.join(self.root_dir, self.dataset_type, self.model_type))

    def __len__(self):
        return len(os.listdir(os.path.join(self.root_dir, self.dataset_type, self.model_type)))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        image_name = os.path.join(self.root_dir, self.dataset_type, self.model_type,self.image_names[idx])

        # handle broken images
        if os.path.exists(image_name) and os.path.getsize(image_name) > 0:
          image = read_image(image_name)
        else:
          image = torch.zeros((3,256,256))

        # handle images without 3 layers
        if image.shape[0] == 1:
            image = torch.cat((image,image,image),0)

        if image.shape[0] == 4:
            image = image[:3]

        model_type = self.model_type

        if model_type == 'ai':
            model_type_num = 1
        else:
            model_type_num = 0

        dataset_type = self.dataset_type


        image_name = self.image_names[idx]

        sample = {'image': image,'model_type':model_type_num,'dataset_type':dataset_type,'image_name':image_name}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        new_h, new_w = int(self.output_size), int(self.output_size)

        image = transforms.Resize([new_h, new_w])(image)


        model_type = sample['model_type']
        dataset_type = sample['dataset_type']
        image_name = sample['image_name']

        sample = {'image': image,'model_type':model_type,'dataset_type':dataset_type,'image_name':image_name}

        return sample