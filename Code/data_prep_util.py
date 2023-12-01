import os
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.io import read_image
from skimage import io, transform
import datetime


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

    def high_pass_filter_opencv(image, kernel_size=3):
        # Create a kernel that sums to 1 for low pass filtering
        kernel = np.ones((kernel_size, kernel_size),
                         np.float32) / (kernel_size ** 2)
        # Create a high-pass filter kernel from the low-pass kernel
        kernel = -kernel
        kernel[(kernel_size - 1)//2, (kernel_size - 1) //
               2] = 1 + (-1 * kernel.sum())
        # Filter the image
        high_pass_filtered_image = cv2.filter2D(image, -1, kernel)
        return high_pass_filtered_image

class State:
  def __init__(
        self, 
        model_state_dict,
        epoch,
        trainloader, 
        testloader, 
        train_loss_history,
        train_acc_history,
        val_loss_history,
        val_acc_history,
        criterion_state_dict,
        optimizer_state_dict,
        scaler_state_dict,
    ):
        self.model_state_dict = model_state_dict
        self.epoch = epoch
        self.trainloader = trainloader
        self.testloader = testloader
        self.train_loss_history = train_loss_history
        self.train_acc_history = train_acc_history
        self.val_loss_history = val_loss_history
        self.val_acc_history = val_acc_history
        self.criterion_state_dict = criterion_state_dict
        self.optimizer_state_dict = optimizer_state_dict
        self.scaler_state_dict = scaler_state_dict

class CheckPoint(object):
    def save_checkpoint(state, dir=''):
        state_hash = {
            'model_state_dict': state.model_state_dict,
            'epoch': state.epoch,
            'trainloader': state.trainloader, 
            'testloader': state.testloader, 
            'train_loss_history': state.train_loss_history,
            'train_acc_history': state.train_acc_history,
            'val_loss_history': state.val_loss_history,
            'val_acc_history': state.val_acc_history,
            'criterion_state_dict': state.criterion_state_dict,
            'optimizer_state_dict': state.optimizer_state_dict,
            'scaler_state_dict': state.scaler_state_dict
        }
        filepath = dir+'latest.pth'
        torch.save(state_hash, filepath)
        print('Saved checkpoint to ' + filepath)
        filepath = dir+f"epoch{state_hash['epoch']}-{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.pth"
        torch.save(state_hash, filepath)
        print('Saved checkpoint to ' + filepath)
    
    def load_checkpoint(path=''):
        if path=='':
            path = 'latest.pth'

        if not isfile(path):
            return None
        state = torch.load(path)

        return State(
            model_state_dict = state['model_state_dict'],
            epoch = state['epoch'],
            trainloader = state['trainloader'],
            testloader = state['testloader'],
            train_loss_history = state['train_loss_history'],
            train_acc_history = state['train_acc_history'],
            val_loss_history = state['val_loss_history'],
            val_acc_history = state['val_acc_history'],
            criterion_state_dict = state['criterion_state_dict'],
            optimizer_state_dict = state['optimizer_state_dict'],
            scaler_state_dict = state['scaler_state_dict'],
        )
