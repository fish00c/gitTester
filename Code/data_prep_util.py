import os
from os import listdir
from os.path import isfile, join
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.io import read_image
from skimage import io, transform


class GenImageDataset(Dataset):
    """GenImage dataset."""

    def __init__(self, root_dir, dataset_type, model_type, transform=None, use_high_pass_filter=False, alpha_value=0.25):
        """
        Args:
            root_dir (string): Directory with all the images.
            dataset_type (string): Type of dataset ('train', 'val', etc.).
            model_type (string): Type of model ('nature', 'ai', etc.).
            transform (callable, optional): Optional transform to be applied on a sample.
            use_high_pass_filter (bool): Flag to apply high pass filter.
            alpha_value (float): Parameter for the high pass filter effect.
        """
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.model_type = model_type
        self.transform = transform
        self.use_high_pass_filter = use_high_pass_filter
        self.alpha_value = alpha_value

        self.image_names = os.listdir(os.path.join(
            self.root_dir, self.dataset_type, self.model_type))

    def __len__(self):
        return len(os.listdir(os.path.join(self.root_dir, self.dataset_type, self.model_type)))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name = os.path.join(
            self.root_dir, self.dataset_type, self.model_type, self.image_names[idx])

        # Handle broken images
        if os.path.exists(image_name) and os.path.getsize(image_name) > 0:
            image = read_image(image_name)
        else:
            image = torch.zeros((3, 256, 256))

        # Handle images without 3 layers
        if image.shape[0] == 1:
            image = torch.cat((image, image, image), 0)
        if image.shape[0] == 4:
            image = image[:3]

        # Apply high pass filter if enabled
        if self.use_high_pass_filter:
            image = high_pass_filter(image, self.alpha_value)

        model_type_num = 1 if self.model_type == 'ai' else 0

        sample = {'image': image, 'model_type': model_type_num,
                  'dataset_type': self.dataset_type, 'image_name': self.image_names[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size."""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        new_h, new_w = int(self.output_size), int(self.output_size)
        image = transforms.Resize([new_h, new_w])(image)

        return {'image': image, 'model_type': sample['model_type'], 'dataset_type': sample['dataset_type'], 'image_name': sample['image_name']}


class HighPassFilter(object):
    def __init__(self, alpha=0.25):
        """
        Args:
            alpha (float): Strength of the high pass filter.
        """
        self.alpha = alpha

    def __call__(self, sample):
        image = sample['image']
        filtered_image = self.high_pass_filter(image, self.alpha)
        return {**sample, 'image': filtered_image}

    def high_pass_filter(self, image, alpha):
        """
        Apply a high pass filter to the image.

        Args:
            image (Tensor): Image tensor of shape [C, H, W].
            alpha (float): Strength of the high pass filter.

        Returns:
            Tensor: High pass filtered image.
        """
        # Define a simple high-pass filter kernel
        hp_kernel = torch.tensor([[-1, -1, -1],
                                  [-1, 8, -1],
                                  [-1, -1, -1]]).float()

        # Normalize the kernel
        hp_kernel = hp_kernel / hp_kernel.sum()

        # Add channel dimension so the kernel works with 3-channel images
        hp_kernel = hp_kernel.unsqueeze(0).unsqueeze(0)

        # Repeat kernel for each channel
        hp_kernel = hp_kernel.repeat(image.shape[0], 1, 1, 1)

        # Apply the high pass filter
        hp_image = torch.nn.functional.conv2d(
            image.unsqueeze(0), hp_kernel, padding=1, groups=image.shape[0]).squeeze(0)

        # Combine the original image and high pass filtered image
        filtered_image = (1 - alpha) * image + alpha * hp_image

        return filtered_image
