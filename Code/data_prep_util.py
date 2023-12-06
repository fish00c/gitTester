import os
from os import listdir
from os.path import isfile, join
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.io import read_image
import torch.nn as nn
import torch.nn.functional as F
from skimage import io, transform
import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torchvision.transforms as T
from PIL import Image


def random_image(output_path='test.png'):
    # define a torch tensor
    tensor = torch.rand(3, 300, 700)

    # define a transform to convert a tensor to PIL image
    transform = T.ToPILImage()

    # convert the tensor to PIL image using above transform
    img = transform(tensor)

    # display the PIL image
    img.show()

    img.save(output_path)


def show_image(image_path):
    plt.figure()
    img = mpimg.imread(image_path)
    imgplot = plt.imshow(img)
    plt.axis("off")
    plt.show()


def show_torch_tensor_as_image(tensor_path):
    # Read tensor
    image_tensor = torch.load(tensor_path)

    # Convert tensor to PIL Image
    image_pil = T.functional.to_pil_image(image_tensor)

    # Display the image
    plt.imshow(image_pil)
    plt.axis('off')  # Hide axes
    plt.show()


def show_image_from_dataloader(image_dataloader):
    plt.figure()
    imgplot = plt.imshow(image_dataloader.permute(1, 2, 0))
    plt.axis("off")
    plt.show()


class GenImageDataset(Dataset):
    """GenImage dataset."""

    def __init__(self, root_dir, dataset_type, model_type, transform=None, input_type='Tensor'):
        """
        Args:
            root_dir (string): Directory with all the images.
            dataset_type (string): Type of dataset ('train', 'val', etc.).
            model_type (string): Type of model ('nature', 'ai', etc.).
            transform (callable, optional): Optional transform to be applied on a sample.
            use_high_pass_filter (bool): Flag to apply high pass filter.
            alpha_value (float): Parameter for the high pass filter effect.
            input_type (string): Type of input ('Image','Tensor')
        """
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.model_type = model_type
        self.transform = transform
        self.input_type = input_type

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
            if self.input_type == 'Tensor':
                image = torch.load(image_name)
            elif self.input_type == 'Image':
                image = read_image(image_name)
            else:
                print('Input type is not supported')
        else:
            image = torch.zeros((3, 256, 256))

        # Handle images without 3 layers
        if image.shape[0] == 1:
            image = torch.cat((image, image, image), 0)
        if image.shape[0] == 4:
            image = image[:3]

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


class HighPassConvLayer(nn.Module):
    def __init__(self, kernel_size=5, channels=3):
        super(HighPassConvLayer, self).__init__()
        # Define a softened 5x5 high-pass filter kernel
        kernel = torch.tensor([
            [-0.25, -0.25, -0.25, -0.25, -0.25],
            [-0.25, -0.25, -0.25, -0.25, -0.25],
            [-0.25, -0.25,  3.0, -0.25, -0.25],
            [-0.25, -0.25, -0.25, -0.25, -0.25],
            [-0.25, -0.25, -0.25, -0.25, -0.25]
        ])

        # Ensure the kernel sums to 0
        kernel -= kernel.mean()

        # Repeat the kernel for each input channel
        kernel = kernel.repeat(channels, 1, 1, 1)

        # Create a convolutional layer with the high-pass filter, without gradient
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, groups=channels, padding=2, bias=False)
        self.conv.weight = nn.Parameter(kernel, requires_grad=False)

    def forward(self, x):
        # Apply the high-pass filter
        filtered_x = self.conv(x)

        # Normalize the output to the 0-1 range
        filtered_x = (filtered_x - filtered_x.min()) / \
            (filtered_x.max() - filtered_x.min())

        return filtered_x


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
        filepath = dir + \
            f"epoch{state_hash['epoch']}-{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.pth"
        torch.save(state_hash, filepath)
        print('Saved checkpoint to ' + filepath)

    def load_checkpoint(path=''):
        if path == '':
            path = 'latest.pth'

        if not isfile(path):
            return None
        state = torch.load(path)

        return State(
            model_state_dict=state['model_state_dict'],
            epoch=state['epoch'],
            trainloader=state['trainloader'],
            testloader=state['testloader'],
            train_loss_history=state['train_loss_history'],
            train_acc_history=state['train_acc_history'],
            val_loss_history=state['val_loss_history'],
            val_acc_history=state['val_acc_history'],
            criterion_state_dict=state['criterion_state_dict'],
            optimizer_state_dict=state['optimizer_state_dict'],
            scaler_state_dict=state['scaler_state_dict'],
        )


class data_pre_process():
    # use to process the image before dataloader

    def __init__(self, root_dir, dataset_type, model_type, output_root):
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.model_type = model_type

        self.output_root = output_root

        self.image_name_list = os.listdir(os.path.join(
            self.root_dir, self.dataset_type, self.model_type))

        # create output folder
        output_file = os.path.join(
            self.output_root, self.dataset_type, self.model_type)

        if not os.path.exists(output_file):
            # If it doesn't exist, create it
            os.makedirs(output_file)

    def resize_image_in_folder(self, output_size):
        num_broken_image = 0
        n = 0
        for i in range(len(self.image_name_list)):

            image_name = os.path.join(
                self.root_dir, self.dataset_type, self.model_type, self.image_name_list[i])

            # handle broken images
            if os.path.exists(image_name) and os.path.getsize(image_name) > 0 and (
                ('png') in image_name or
                ('PNG') in image_name or
                ('jpg') in image_name or
                    ('JPEG') in image_name):
                image = read_image(image_name)

                # handle images without 3 layers
                if image.shape[0] == 1:
                    image = torch.cat((image, image, image), 0)

                if image.shape[0] == 4:
                    image = image[:3]

                # resize images
                image = transforms.Resize(
                    [output_size, output_size], antialias=True)(image)

                # save as a tensor
                output_file = os.path.join(
                    self.output_root, self.dataset_type, self.model_type, self.image_name_list[i].split('.')[0]+'.pt')
                torch.save(image, output_file)

                n = n+1

            else:
                image = torch.zeros((3, output_size, output_size))
                num_broken_image = num_broken_image+1
                print('Broken Images: ', image_name)

                if (('png') in image_name and ('PNG') in image_name or ('jpg') in image_name or ("JPEG ") in image_name) == False:
                    print('Reason of Broken Images: Type')
                elif os.path.exists(image_name) == False:
                    print('Reason of Broken Images: Empty File')
                elif os.path.getsize(image_name) > 0 == False:
                    print('Reason of Broken Images: Empty File')

        print('Resized ', n, 'images')
        print(num_broken_image, 'images are broken')
